"""Shared scaffolding for GARCH-family conditional-variance models.

The :class:`GARCHBase` class implements the full variance-model
contract on a vanilla σ²-form GARCH(p, q) recursion (Bollerslev 1986).
Concrete singletons in :mod:`.garch`, :mod:`.igarch`, :mod:`.gjr_garch`,
etc. inherit from this base and override the per-family pieces:

* :meth:`_pack_x0` / :meth:`_unpack_raw` — the bound-enforcing
  reparameterisation.  The vanilla GARCH layout in this base is
  ``[raw_omega, raw_persistence, raw_weights, raw_residual_shape]``;
  IGARCH drops ``raw_persistence`` (pinned to 1), GJR adds the
  asymmetric-weights chunk plus the residual-law-dependent
  :math:`\\kappa = \\mathbb{E}[z^2 \\mathbf{1}\\{z<0\\}]` rescale.
* :meth:`_run_recursion` — wraps the family-specific kernel from
  :mod:`copulax._src.timeseries._recursions` and returns the per-step
  conditional variance plus a typed terminal carry.

The base provides:

* :meth:`fit` — Adam-driven negative-log-likelihood minimisation
  through :func:`copulax._src.optimize.projected_gradient`, JIT-
  compatible end-to-end.
* :meth:`conditional_variance`, :meth:`conditional_mean`,
  :meth:`residuals`, :meth:`standardised_residuals` — pure
  functions of ``(params, eps)`` reusing the same kernel as the
  fit.
* :meth:`stats` — analytic, parameter-only quantities:
  unconditional variance :math:`\\omega / (1 - \\sum \\alpha_i -
  \\sum \\beta_j)`, persistence, half-life :math:`\\log 1/2 / \\log
  \\text{persistence}`, stationarity flag.
* :meth:`forecast` — h-step analytical and simulation forecasts.
  The analytical path rolls the σ²-recursion forward replacing
  unknown future :math:`\\varepsilon^2` with their conditional
  expectation (which is the σ² from the same loop), so the
  forecast is exact under the GARCH specification.
* :meth:`rvs` — simulate paths from the fitted model with the
  full ``size`` / ``key`` / ``u`` / ``last_state`` contract used by
  the rest of the subpackage.
* :meth:`loglikelihood`, :meth:`aic`, :meth:`bic` — scalar
  diagnostics with the (``y=None`` → stored value, otherwise
  recompute) dispatch the rest of the family uses.

The variance-stage standard-error machinery (observed information,
Pagan-Newey two-stage sandwich) lives in a forthcoming ``_se.py``;
this base exposes the fit/forecast/residual surface only.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src._utils import _resolve_key
from copulax._src.optimize import projected_gradient
from copulax._src.timeseries._base import TerminalState, VarianceModel
from copulax._src.timeseries._init import (
    garch_pre_sample_state,
    init_garch_params,
)
from copulax._src.timeseries._recursions import run_garch
from copulax._src.timeseries._residuals._standardise import StandardisedResidual
from copulax._src.timeseries._stationarity import (
    garch_simplex,
    garch_unsimplex,
    positive_to_raw,
    raw_to_positive,
)


_VAR_FLOOR: float = 1e-12
_SIGMA_FLOOR: float = 1e-6


###############################################################################
# Per-family terminal state for σ²-form recursions
###############################################################################
class GARCHTerminalState(TerminalState):
    r"""Constant-size carry for σ²-form GARCH ``forecast(h)``.

    Stores the last ``p`` squared residuals and the last ``q``
    conditional variances — exactly what :func:`run_garch` consumes
    as initial state.  ``O(max(p, q))`` in size; serialises trivially.
    """
    eps_sq_lags: Array
    var_lags: Array


###############################################################################
# GARCHBase
###############################################################################
class GARCHBase(VarianceModel):
    r"""Vanilla GARCH(p, q) conditional-variance model (Bollerslev 1986).

    Recursion:

    .. math::

        \sigma^2_t = \omega
                   + \sum_{i=1}^p \alpha_i\, \varepsilon^2_{t-i}
                   + \sum_{j=1}^q \beta_j\, \sigma^2_{t-j},
        \qquad
        \omega > 0,\quad \alpha_i \ge 0,\quad \beta_j \ge 0,
        \quad \sum \alpha_i + \sum \beta_j < 1.

    Innovations follow the chosen standardised residual law
    :math:`z_t \sim f_z\,(\text{mean}=0, \mathrm{var}=1)` with
    :math:`\varepsilon_t = \sigma_t z_t`.  The likelihood is

    .. math::

        \ell = \sum_t \bigl[\log f_z(z_t) - \log \sigma_t \bigr],
        \qquad z_t = \varepsilon_t / \sigma_t.

    Concrete subclasses (``GARCH``, ``IGARCH``, ...) inherit this
    base and override only the reparameterisation-pack/unpack and
    (where the recursion shape differs) the kernel call.
    """

    # ---- static configuration -------------------------------------------
    p: int = eqx.field(static=True)
    q: int = eqx.field(static=True)
    residual_dist: Univariate = eqx.field(static=True)

    # ---- traced fitted parameters ---------------------------------------
    omega: Optional[Array] = None
    alpha: Optional[Array] = None
    beta: Optional[Array] = None
    residual_params: Optional[dict] = None

    # ---- per-fit terminal carry + diagnostics ---------------------------
    terminal_state: Optional[GARCHTerminalState] = None
    loglikelihood_: Optional[Array] = None
    aic_: Optional[Array] = None
    bic_: Optional[Array] = None
    n_train_: Optional[int] = None

    _supported_methods: ClassVar[frozenset] = frozenset(
        {"analytical", "backcast", "sample", "warm"}
    )

    def __init__(
        self,
        name: str = "GARCH",
        *,
        p: int = 0,
        q: int = 0,
        residual_dist: Univariate = None,
        omega: Optional[ArrayLike] = None,
        alpha: Optional[ArrayLike] = None,
        beta: Optional[ArrayLike] = None,
        residual_params: Optional[dict] = None,
        terminal_state: Optional[GARCHTerminalState] = None,
        loglikelihood_: Optional[ArrayLike] = None,
        aic_: Optional[ArrayLike] = None,
        bic_: Optional[ArrayLike] = None,
        n_train_: Optional[int] = None,
    ):
        super().__init__(name=name)
        self.p = int(p)
        self.q = int(q)
        if residual_dist is None:
            from copulax._src.univariate.normal import normal as _normal
            residual_dist = _normal
        self.residual_dist = residual_dist
        self.omega = (
            jnp.asarray(omega, dtype=float).reshape(())
            if omega is not None else None
        )
        self.alpha = (
            jnp.asarray(alpha, dtype=float).reshape(-1)
            if alpha is not None else None
        )
        self.beta = (
            jnp.asarray(beta, dtype=float).reshape(-1)
            if beta is not None else None
        )
        self.residual_params = residual_params
        self.terminal_state = terminal_state
        self.loglikelihood_ = (
            jnp.asarray(loglikelihood_, dtype=float).reshape(())
            if loglikelihood_ is not None else None
        )
        self.aic_ = (
            jnp.asarray(aic_, dtype=float).reshape(())
            if aic_ is not None else None
        )
        self.bic_ = (
            jnp.asarray(bic_, dtype=float).reshape(())
            if bic_ is not None else None
        )
        self.n_train_ = int(n_train_) if n_train_ is not None else None

    # ------------------------------------------------------------------
    # params property
    # ------------------------------------------------------------------
    @property
    def _stored_params(self) -> Optional[dict]:
        """Canonical parameter dict, or ``None`` for an unfitted instance.

        Schema:

        ``{
            "omega":     (),
            "alpha":     (p,),
            "beta":      (q,),
            "residual":  {<shape-only dict>},
        }``
        """
        if (
            self.omega is None or self.alpha is None or self.beta is None
            or self.residual_params is None
        ):
            return None
        return {
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "residual": dict(self.residual_params),
        }

    @property
    def n_params(self) -> int:
        r"""Number of free fitted parameters in the model."""
        wrapper = StandardisedResidual(self.residual_dist)
        return 1 + self.p + self.q + wrapper.n_shape_params

    def _wrapper(self) -> StandardisedResidual:
        return StandardisedResidual(self.residual_dist)

    # ------------------------------------------------------------------
    # ArmaGarch backend interface
    # ------------------------------------------------------------------
    # Each GARCH-family variant exposes a uniform interface used by the
    # ``ArmaGarch`` joint composite to run its variance section without
    # duplicating recursion / reparameterisation code.  Default
    # implementations here cover vanilla GARCH(p, q); subclasses
    # override for variants whose pack/unpack/recursion shape differs.

    def _ag_var_keys(self) -> tuple:
        r"""Names of the variance natural-parameter keys.

        Vanilla GARCH and IGARCH both have ``("omega", "alpha", "beta")``.
        """
        return ("omega", "alpha", "beta")

    def _ag_n_raw(self, wrapper: StandardisedResidual) -> int:
        r"""Number of entries the variance section of the joint
        unconstrained vector occupies."""
        # raw_omega + raw_persistence + raw_weights(p+q)
        return 1 + 1 + self.p + self.q

    def _ag_pack_x0(
        self,
        var_params: dict,
        wrapper: StandardisedResidual,
        residual_params: dict,
    ) -> Array:
        r"""Pack the variance natural params (no residual) into the
        flat raw section for the joint optimiser-state vector.
        """
        omega = jnp.asarray(var_params["omega"], dtype=float).reshape(())
        alpha = jnp.asarray(var_params["alpha"], dtype=float).reshape(-1)
        beta = jnp.asarray(var_params["beta"], dtype=float).reshape(-1)
        raw_omega = positive_to_raw(jnp.maximum(omega, _SIGMA_FLOOR))
        raw_persistence, raw_weights = garch_unsimplex(alpha, beta)
        return jnp.concatenate(
            [raw_omega.reshape((1,)), raw_persistence.reshape((1,)), raw_weights]
        )

    def _ag_unpack_raw(
        self,
        raw_section: Array,
        wrapper: StandardisedResidual,
        residual_params: dict,
    ) -> dict:
        r"""Inverse of :meth:`_ag_pack_x0` — natural variance params dict."""
        idx = 0
        raw_omega = raw_section[idx]
        idx += 1
        raw_persistence = raw_section[idx]
        idx += 1
        raw_weights = raw_section[idx : idx + self.p + self.q]
        omega = raw_to_positive(raw_omega)
        alpha, beta = garch_simplex(raw_persistence, raw_weights, p=self.p)
        return {"omega": omega, "alpha": alpha, "beta": beta}

    def _ag_initial_state(
        self,
        eps_proxy: Array,
        mode: str,
        backcast_length: Optional[int],
        residual_params: dict,
    ) -> tuple:
        r"""Pre-sample state for the variance recursion in the joint fit.

        Returns the family-specific carry tuple consumed by
        :meth:`_ag_run_recursion`.
        """
        return garch_pre_sample_state(
            eps_proxy, p=self.p, q=self.q,
            mode=mode, backcast_length=backcast_length,
        )

    def _ag_run_recursion(
        self,
        eps_seq: Array,
        var_params: dict,
        residual_params: dict,
        init_state: tuple,
    ) -> tuple[Array, tuple]:
        r"""Run the variance recursion on ``eps_seq``.

        Returns ``(var_seq, terminal_tuple)`` — both downstream of the
        ARMA innovation series in the joint composite.
        """
        omega = var_params["omega"]
        alpha = var_params["alpha"]
        beta = var_params["beta"]
        eps_sq_lags, var_lags = init_state
        var_seq, terminal = run_garch(
            eps=eps_seq, omega=omega, alpha=alpha, beta=beta,
            init_eps_sq_lags=eps_sq_lags, init_var_lags=var_lags,
        )
        return var_seq, terminal

    def _ag_cold_start(
        self,
        eps_proxy: Array,
        mode: str,
        backcast_length: Optional[int],
        wrapper: StandardisedResidual,
    ) -> dict:
        r"""Cold-start variance natural-params dict (no residual)."""
        seed = init_garch_params(
            eps_proxy, p=self.p, q=self.q, mode=mode,
            backcast_length=backcast_length,
        )
        return {
            "omega": seed["omega"],
            "alpha": seed["alpha"],
            "beta": seed["beta"],
        }

    def _ag_forecast_step(
        self,
        var_params: dict,
        residual_params: dict,
        terminal_state: tuple,
    ) -> tuple[Array, tuple]:
        r"""One analytical forecast step.

        Substitutes :math:`\mathbb{E}[\varepsilon^2_\tau] =
        \mathbb{E}[\sigma^2_\tau]` for unobserved future shocks —
        the textbook GARCH analytical-forecast trick.

        Returns ``(var_next, new_terminal_state)``.
        """
        omega = var_params["omega"]
        alpha = var_params["alpha"]
        beta = var_params["beta"]
        eps_sq_lags, var_lags = terminal_state
        ar_term = jnp.dot(alpha, eps_sq_lags) if self.p > 0 else 0.0
        ma_term = jnp.dot(beta, var_lags) if self.q > 0 else 0.0
        var_next = jnp.maximum(omega + ar_term + ma_term, _VAR_FLOOR)
        new_eps_sq_lags = (
            jnp.concatenate([var_next.reshape((1,)), eps_sq_lags[:-1]])
            if self.p > 0 else eps_sq_lags
        )
        new_var_lags = (
            jnp.concatenate([var_next.reshape((1,)), var_lags[:-1]])
            if self.q > 0 else var_lags
        )
        return var_next, (new_eps_sq_lags, new_var_lags)

    def _ag_rvs_step(
        self,
        var_params: dict,
        residual_params: dict,
        terminal_state: tuple,
        eps_t: Array,
    ) -> tuple[Array, tuple]:
        r"""One simulation step given the realised innovation ``eps_t``.

        Returns ``(var_t, new_terminal_state)``.
        """
        omega = var_params["omega"]
        alpha = var_params["alpha"]
        beta = var_params["beta"]
        eps_sq_lags, var_lags = terminal_state
        ar_term = jnp.dot(alpha, eps_sq_lags) if self.p > 0 else 0.0
        ma_term = jnp.dot(beta, var_lags) if self.q > 0 else 0.0
        var_t = jnp.maximum(omega + ar_term + ma_term, _VAR_FLOOR)
        new_eps_sq_lags = (
            jnp.concatenate(
                [(eps_t * eps_t).reshape((1,)), eps_sq_lags[:-1]]
            )
            if self.p > 0 else eps_sq_lags
        )
        new_var_lags = (
            jnp.concatenate([var_t.reshape((1,)), var_lags[:-1]])
            if self.q > 0 else var_lags
        )
        return var_t, (new_eps_sq_lags, new_var_lags)

    @staticmethod
    def _ag_supports_analytical_h_step() -> bool:
        r"""Whether the variant supports analytical h-step forecasts.

        ``True`` for σ²-form variants whose recursion is closed under
        the ``E[ε²]=σ²`` substitution (vanilla GARCH, IGARCH, GJR,
        QGARCH).  ``False`` for log-/σ-form variants where the
        substitution requires moments of ``z`` that lack closed form
        under non-Normal residuals (EGARCH, TGARCH).
        """
        return True

    def _ag_var_terminal_state_class(self) -> type:
        r"""TerminalState subclass appropriate for this variance variant.

        Used by the joint composite when reconstructing a typed
        terminal-state field from a flat carry tuple.
        """
        return GARCHTerminalState

    # ------------------------------------------------------------------
    # Reparameterisation pack / unpack
    # ------------------------------------------------------------------
    def _pack_x0(
        self,
        params_dict: dict,
        wrapper: StandardisedResidual,
    ) -> Array:
        r"""Pack a constrained ``params_dict`` into the unconstrained
        flat optimiser-state vector.

        Layout: ``[raw_omega (1,), raw_persistence (1,),
        raw_weights (p+q,), raw_residual_shape (n_shape_params,)]``.
        """
        omega = jnp.asarray(params_dict["omega"], dtype=float).reshape(())
        alpha = jnp.asarray(params_dict["alpha"], dtype=float).reshape(-1)
        beta = jnp.asarray(params_dict["beta"], dtype=float).reshape(-1)
        residual = params_dict.get("residual", {}) or {}

        raw_omega = positive_to_raw(jnp.maximum(omega, _SIGMA_FLOOR))
        raw_persistence, raw_weights = garch_unsimplex(alpha, beta)
        raw_residual = wrapper.shape_params_to_array(residual)
        return jnp.concatenate(
            [
                raw_omega.reshape((1,)),
                raw_persistence.reshape((1,)),
                raw_weights,
                raw_residual,
            ]
        )

    def _unpack_raw(
        self,
        raw: Array,
        wrapper: StandardisedResidual,
    ) -> tuple[Array, Array, Array, dict]:
        r"""Inverse of :meth:`_pack_x0`.

        Returns ``(omega, alpha, beta, residual_shape_dict)`` — every
        component already in its constrained / feasible form.
        """
        idx = 0
        raw_omega = raw[idx]
        idx += 1
        raw_persistence = raw[idx]
        idx += 1
        raw_weights = raw[idx : idx + self.p + self.q]
        idx += self.p + self.q
        raw_residual = raw[idx : idx + wrapper.n_shape_params]

        omega = raw_to_positive(raw_omega)
        alpha, beta = garch_simplex(raw_persistence, raw_weights, p=self.p)
        residual = wrapper.shape_params_from_array(raw_residual)
        return omega, alpha, beta, residual

    # ------------------------------------------------------------------
    # Recursion call — overridden by variants whose kernel differs
    # ------------------------------------------------------------------
    def _run_recursion(
        self,
        eps: Array,
        omega: Array,
        alpha: Array,
        beta: Array,
        init_eps_sq_lags: Array,
        init_var_lags: Array,
    ) -> tuple[Array, GARCHTerminalState]:
        """Run the σ²-recursion and wrap the terminal carry."""
        var_seq, terminal = run_garch(
            eps=eps, omega=omega, alpha=alpha, beta=beta,
            init_eps_sq_lags=init_eps_sq_lags, init_var_lags=init_var_lags,
        )
        terminal_state = GARCHTerminalState(
            eps_sq_lags=terminal[0], var_lags=terminal[1],
        )
        return var_seq, terminal_state

    # ------------------------------------------------------------------
    # Fit objective
    # ------------------------------------------------------------------
    def _make_objective(self, wrapper: StandardisedResidual):
        r"""Build a closure over the residual wrapper that
        :func:`projected_gradient` JIT-compiles cleanly.

        Returns:
            Callable ``(raw, eps, init_eps_sq_lags, init_var_lags) ->
            scalar`` — negative mean log-likelihood under the GARCH +
            standardised-residual specification:

            .. math::

                \\ell(\\theta; \\varepsilon)
                    = \\sum_t \\bigl[
                        \\log f_z(\\varepsilon_t / \\sigma_t)
                        - \\log \\sigma_t
                      \\bigr],
                \\quad
                \\sigma_t = \\sqrt{\\sigma^2_t}.

            Non-finite contributions are masked to a large penalty
            (matches ``Univariate._mle_objective``).
        """
        def objective(
            raw: Array,
            eps: Array,
            init_eps_sq_lags: Array,
            init_var_lags: Array,
        ) -> Array:
            omega, alpha, beta, residual_shape = self._unpack_raw(raw, wrapper)
            var_seq, _ = self._run_recursion(
                eps, omega, alpha, beta,
                init_eps_sq_lags, init_var_lags,
            )
            sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
            z = eps / sigma_seq
            logpdf = wrapper.logpdf(z, residual_shape) - jnp.log(sigma_seq)
            finite = jnp.isfinite(logpdf)
            safe_logpdf = jnp.where(finite, logpdf, 0.0)
            invalid_penalty = 1e6 * (~finite).mean()
            return -safe_logpdf.mean() + invalid_penalty

        return objective

    # ------------------------------------------------------------------
    # Cold-start parameter dict
    # ------------------------------------------------------------------
    def _build_cold_start(
        self,
        eps: Array,
        wrapper: StandardisedResidual,
        init: str,
        backcast_length: Optional[int],
    ) -> dict:
        r"""Cold-start parameter dict for the chosen ``init`` mode.

        For all three non-warm modes we lean on
        :func:`init_garch_params` for ``(omega, alpha, beta)`` and
        seed the residual half from the wrapper's
        :meth:`example_shape_params` (a moment-feasible default).
        """
        garch_seed = init_garch_params(
            eps, p=self.p, q=self.q, mode=init,
            backcast_length=backcast_length,
        )
        return {
            "omega": garch_seed["omega"],
            "alpha": garch_seed["alpha"],
            "beta": garch_seed["beta"],
            "residual": wrapper.example_shape_params(),
        }

    # ------------------------------------------------------------------
    # Public fit
    # ------------------------------------------------------------------
    def fit(
        self,
        eps: ArrayLike,
        *,
        init: str = "analytical",
        init_params: Optional[dict] = None,
        backcast_length: Optional[int] = None,
        maxiter: int = 200,
        lr: float = 0.05,
        name: Optional[str] = None,
    ) -> "GARCHBase":
        r"""Fit the GARCH(p, q) model to a mean-corrected innovation series.

        Orders ``(p, q)`` and the residual distribution are set at
        construction time — see :meth:`__init__`.  ``fit`` itself is
        data-only.

        Args:
            eps: shape ``(n,)`` — mean-corrected innovation series.
                Pass returns directly if you trust the data is
                already mean-zero, or pre-subtract the sample mean /
                the residuals from a fitted ARMA mean model.  For
                proper joint estimation use the forthcoming
                ``ArmaGarch`` composite.
            init: One of ``"analytical"`` (industry α=0.05, β=0.9
                prior, default), ``"backcast"``, ``"sample"``, or
                ``"warm"``.
            init_params: Warm-start parameter dict; required when
                ``init="warm"``.
            backcast_length: Window for the EWMA backcast under
                ``init="backcast"``.  ``None`` uses the full series.
            maxiter: Adam iterations.
            lr: Adam learning rate.
            name: Optional custom name for the fitted instance.

        Returns:
            A fitted instance with ``params``, ``terminal_state``,
            and the ``loglikelihood_`` / ``aic_`` / ``bic_`` /
            ``n_train_`` diagnostics populated.
        """
        self._check_method(init)
        wrapper = StandardisedResidual(self.residual_dist)
        eps_arr = self._validate_series(eps)
        n = int(eps_arr.shape[0])
        self._validate_backcast_length(backcast_length, n)

        if init == "warm":
            if init_params is None:
                raise ValueError(
                    "init='warm' requires init_params (a parameter dict "
                    "matching the schema returned by `model.params`)."
                )
            cold = dict(init_params)
            for key in ("omega", "alpha", "beta", "residual"):
                if key not in cold:
                    raise KeyError(
                        f"Warm-start init_params missing required key {key!r}."
                    )
        else:
            cold = self._build_cold_start(
                eps_arr, wrapper, init=init, backcast_length=backcast_length,
            )

        x0 = self._pack_x0(cold, wrapper)

        # Recursion's pre-sample state — independent of the parameter
        # init mode (uses a moment-based EWMA backcast or sample
        # variance).
        _state_mode = "sample" if init == "sample" else "backcast"
        init_eps_sq_lags, init_var_lags = garch_pre_sample_state(
            eps_arr, p=self.p, q=self.q,
            mode=_state_mode, backcast_length=backcast_length,
        )

        objective = self._make_objective(wrapper)
        res = projected_gradient(
            f=objective,
            x0=x0,
            projection_method="projection_box",
            projection_options={
                "lower": jnp.full((x0.shape[0], 1), -jnp.inf),
                "upper": jnp.full((x0.shape[0], 1), jnp.inf),
            },
            eps=eps_arr,
            init_eps_sq_lags=init_eps_sq_lags,
            init_var_lags=init_var_lags,
            lr=lr,
            maxiter=maxiter,
        )
        x_opt = res["x"]
        omega, alpha, beta, residual = self._unpack_raw(x_opt, wrapper)

        # Final pass at the optimum for terminal state.
        _, terminal = self._run_recursion(
            eps=eps_arr, omega=omega, alpha=alpha, beta=beta,
            init_eps_sq_lags=init_eps_sq_lags,
            init_var_lags=init_var_lags,
        )
        nll = objective(x_opt, eps_arr, init_eps_sq_lags, init_var_lags)
        loglike = -nll * n
        n_params_total = 1 + self.p + self.q + wrapper.n_shape_params
        aic = 2.0 * n_params_total - 2.0 * loglike
        bic = (
            n_params_total * jnp.log(jnp.asarray(n, dtype=float))
            - 2.0 * loglike
        )

        cls = type(self)
        if name is None:
            name = (
                f"Fitted{cls.__name__}({self.p},{self.q})"
                f"-{self.residual_dist.name}"
            )
        return cls(
            name=name,
            p=self.p,
            q=self.q,
            residual_dist=self.residual_dist,
            omega=omega,
            alpha=alpha,
            beta=beta,
            residual_params=residual,
            terminal_state=terminal,
            loglikelihood_=loglike,
            aic_=aic,
            bic_=bic,
            n_train_=n,
        )

    # ------------------------------------------------------------------
    # Conditional moments / residuals
    # ------------------------------------------------------------------
    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError(
                f"Model {self.name!r} is not fitted; call `.fit(eps)` first."
            )

    def _recursion_inputs(
        self,
        eps: ArrayLike,
        init: str,
        backcast_length: Optional[int],
    ) -> tuple[Array, Array, Array]:
        eps_arr = self._validate_series(eps)
        n = int(eps_arr.shape[0])
        self._validate_backcast_length(backcast_length, n)
        init_eps_sq_lags, init_var_lags = garch_pre_sample_state(
            eps_arr, p=self.p, q=self.q,
            mode=init, backcast_length=backcast_length,
        )
        return eps_arr, init_eps_sq_lags, init_var_lags

    def conditional_variance(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""One-step-ahead conditional variance trajectory ``σ²_t``."""
        self._require_fitted()
        eps_arr, init_eps_sq_lags, init_var_lags = self._recursion_inputs(
            eps, init, backcast_length,
        )
        var_seq, _ = self._run_recursion(
            eps_arr, self.omega, self.alpha, self.beta,
            init_eps_sq_lags=init_eps_sq_lags,
            init_var_lags=init_var_lags,
        )
        return var_seq

    def conditional_mean(self, eps: ArrayLike) -> Array:
        r"""Zero — variance models do not parameterise the mean.

        Returned as a length-``n`` array of zeros (rather than a
        scalar) so callers composing mean and variance trajectories
        do not need to broadcast manually.
        """
        eps_arr = self._validate_series(eps)
        return jnp.zeros_like(eps_arr)

    def residuals(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> tuple[Array, Array]:
        r"""Return ``(ε_t, z_t)`` where ``z_t = ε_t / σ_t``.

        Per plan §"Residuals API": variance models return both halves
        in one pass through the recursion so the user can run
        ARCH-LM tests on ``ε_t²`` *and* IID / Q-Q diagnostics on the
        standardised ``z_t`` without re-running the kernel.
        """
        self._require_fitted()
        eps_arr, init_eps_sq_lags, init_var_lags = self._recursion_inputs(
            eps, init, backcast_length,
        )
        var_seq, _ = self._run_recursion(
            eps_arr, self.omega, self.alpha, self.beta,
            init_eps_sq_lags=init_eps_sq_lags,
            init_var_lags=init_var_lags,
        )
        sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
        return eps_arr, eps_arr / sigma_seq

    def standardised_residuals(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Just the ``z_t = ε_t / σ_t`` half of :meth:`residuals`."""
        _, z = self.residuals(
            eps, init=init, backcast_length=backcast_length,
        )
        return z

    def terminal_state_from(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> GARCHTerminalState:
        r"""Build a terminal state from a (possibly new) series — used
        to roll :meth:`forecast` from a window other than the one
        the model was fit on."""
        self._require_fitted()
        eps_arr, init_eps_sq_lags, init_var_lags = self._recursion_inputs(
            eps, init, backcast_length,
        )
        _, terminal = self._run_recursion(
            eps_arr, self.omega, self.alpha, self.beta,
            init_eps_sq_lags=init_eps_sq_lags,
            init_var_lags=init_var_lags,
        )
        return terminal

    # ------------------------------------------------------------------
    # Forecast + sampling
    # ------------------------------------------------------------------
    def _analytical_forecast(
        self, h: int, state: GARCHTerminalState,
    ) -> Array:
        r"""Analytical h-step variance forecast.

        At step :math:`\\tau \\in \\{t+1, \\ldots, t+h\\}` future
        :math:`\\varepsilon^2_{\\tau-i}` for :math:`\\tau - i > t` is
        unobserved; we substitute :math:`\\mathbb{E}[\\varepsilon^2_{
        \\tau-i}] = \\mathbb{E}[\\sigma^2_{\\tau-i}]` (the same
        forecast value in the loop), giving the closed-form
        recursion that converges geometrically to the unconditional
        variance.

        Returns:
            shape ``(h,)`` array of ``E[σ²_{t+τ}]``.
        """
        var_path = []
        eps_sq_lags = state.eps_sq_lags  # (p,)
        var_lags = state.var_lags        # (q,)
        for _ in range(h):
            ar_term = jnp.dot(self.alpha, eps_sq_lags) if self.p > 0 else 0.0
            ma_term = jnp.dot(self.beta, var_lags) if self.q > 0 else 0.0
            var_t = self.omega + ar_term + ma_term
            var_t = jnp.maximum(var_t, _VAR_FLOOR)
            var_path.append(var_t)
            # Substitute E[eps²] = E[σ²] for unknown future steps.
            if self.p > 0:
                eps_sq_lags = jnp.concatenate(
                    [var_t.reshape((1,)), eps_sq_lags[:-1]]
                )
            if self.q > 0:
                var_lags = jnp.concatenate(
                    [var_t.reshape((1,)), var_lags[:-1]]
                )
        return jnp.stack(var_path)

    def forecast(
        self,
        h: int,
        *,
        method: str = "analytical",
        n_paths: int = 0,
        key: Optional[Array] = None,
        last_state: Optional[GARCHTerminalState] = None,
    ) -> dict:
        r"""``h``-step-ahead conditional moments.

        See :class:`copulax._src.timeseries._base.TimeSeriesModel.forecast`
        for the contract.  ``mean`` is identically zero for a pure
        variance model; the ``variance`` trajectory is the analytic
        h-step σ² forecast (or its Monte Carlo estimate under
        ``method='simulation'``).
        """
        self._require_fitted()
        h = int(h)
        if h <= 0:
            raise ValueError(f"forecast horizon h must be > 0; got {h}.")
        state = last_state if last_state is not None else self.terminal_state
        if state is None:
            raise ValueError(
                "No terminal state available; pass `last_state` explicitly "
                "or fit on a series first."
            )
        mean = jnp.zeros((h,), dtype=float)

        if method == "analytical":
            variance = self._analytical_forecast(h, state)
            return {"mean": mean, "variance": variance, "paths": None}

        elif method == "simulation":
            if n_paths <= 0:
                raise ValueError("method='simulation' requires n_paths > 0.")
            key = _resolve_key(key)
            paths = self.rvs(
                size=(int(n_paths), h), key=key, last_state=state,
            )
            mc_mean = jnp.mean(paths, axis=0)
            mc_var = jnp.var(paths, axis=0)
            return {"mean": mc_mean, "variance": mc_var, "paths": paths}

        else:
            raise ValueError(
                f"Unknown forecast method {method!r}; expected "
                "'analytical' or 'simulation'."
            )

    def rvs(
        self,
        size=None,
        *,
        key: Optional[Array] = None,
        u: Optional[ArrayLike] = None,
        last_state: Optional[GARCHTerminalState] = None,
    ) -> Array:
        r"""Simulate synthetic ``ε_t`` paths from the fitted model.

        Returns the innovation series (not a level series — variance
        models do not parameterise the mean).  Use the joint
        ``arma_garch`` composite when level paths are needed.
        """
        self._require_fitted()
        wrapper = self._wrapper()
        state = last_state if last_state is not None else self.terminal_state
        if state is None:
            raise ValueError(
                "No terminal state available; pass `last_state` explicitly "
                "or fit on a series first."
            )

        if u is not None:
            u_arr = jnp.asarray(u, dtype=float)
            shape = u_arr.shape
            if u_arr.ndim == 1:
                z = wrapper.ppf(u_arr, self.residual_params)
                return self._roll_path(z, state)
            elif u_arr.ndim == 2:
                z = wrapper.ppf(u_arr.reshape(-1), self.residual_params).reshape(shape)
                return jax.vmap(lambda zi: self._roll_path(zi, state))(z)
            else:
                raise ValueError(
                    f"u must have ndim 1 or 2; got ndim={u_arr.ndim}."
                )

        if size is None:
            raise ValueError(
                "rvs requires either `size` or `u` to determine output shape."
            )
        if isinstance(size, int):
            shape = (size,)
        else:
            shape = tuple(int(s) for s in size)
        key = _resolve_key(key)

        if len(shape) == 1:
            h = shape[0]
            z = wrapper.rvs(size=(h,), shape_params=self.residual_params, key=key)
            return self._roll_path(z, state)
        elif len(shape) == 2:
            n_paths, h = shape
            keys = jax.random.split(key, n_paths)
            z_batch = jax.vmap(
                lambda k: wrapper.rvs(size=(h,), shape_params=self.residual_params, key=k)
            )(keys)
            return jax.vmap(lambda z: self._roll_path(z, state))(z_batch)
        else:
            raise ValueError(
                f"size must be 1- or 2-dimensional; got {shape}."
            )

    def _roll_path(self, z: Array, state: GARCHTerminalState) -> Array:
        r"""Roll a single path of standardised innovations through the
        σ²-recursion to produce ``ε_t = σ_t z_t``.
        """
        omega = self.omega
        alpha = self.alpha
        beta = self.beta

        def step(carry, z_t):
            eps_sq_lags, var_lags = carry
            ar_term = jnp.dot(alpha, eps_sq_lags) if self.p > 0 else 0.0
            ma_term = jnp.dot(beta, var_lags) if self.q > 0 else 0.0
            var_t = omega + ar_term + ma_term
            var_t = jnp.maximum(var_t, _VAR_FLOOR)
            sigma_t = jnp.sqrt(var_t)
            eps_t = sigma_t * z_t
            new_eps_sq = (
                jnp.concatenate([(eps_t * eps_t).reshape((1,)), eps_sq_lags[:-1]])
                if self.p > 0 else eps_sq_lags
            )
            new_var = (
                jnp.concatenate([var_t.reshape((1,)), var_lags[:-1]])
                if self.q > 0 else var_lags
            )
            return (new_eps_sq, new_var), eps_t

        init_carry = (state.eps_sq_lags, state.var_lags)
        _, eps_seq = jax.lax.scan(step, init_carry, z)
        return eps_seq

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        r"""Analytic, parameter-only diagnostics.

        Returns:
            ``{
                "unconditional_variance":  ω / (1 - persistence),
                "persistence":             Σα + Σβ,
                "half_life":               log(0.5) / log(persistence),
                "is_stationary":           persistence < 1,
            }``
        """
        self._require_fitted()
        persistence = jnp.sum(self.alpha) + jnp.sum(self.beta)
        is_stat = persistence < 1.0
        denom = jnp.where(persistence < 1.0, 1.0 - persistence, _VAR_FLOOR)
        unconditional_variance = jnp.where(
            is_stat, self.omega / denom, jnp.inf
        )
        # Half-life: number of steps for an σ²-shock to decay to half
        # its size under the AR(1)-on-σ² approximation.
        log_pers = jnp.log(jnp.maximum(persistence, _VAR_FLOOR))
        half_life = jnp.where(
            jnp.logical_and(is_stat, persistence > 0.0),
            jnp.log(0.5) / log_pers,
            jnp.inf,
        )
        return {
            "unconditional_variance": unconditional_variance,
            "persistence": persistence,
            "half_life": half_life,
            "is_stationary": is_stat,
        }

    # ------------------------------------------------------------------
    # Loglikelihood / AIC / BIC
    # ------------------------------------------------------------------
    def _log_likelihood_on_series(
        self,
        eps: ArrayLike,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        self._require_fitted()
        wrapper = self._wrapper()
        eps_arr, init_eps_sq_lags, init_var_lags = self._recursion_inputs(
            eps, init, backcast_length,
        )
        var_seq, _ = self._run_recursion(
            eps_arr, self.omega, self.alpha, self.beta,
            init_eps_sq_lags=init_eps_sq_lags,
            init_var_lags=init_var_lags,
        )
        sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
        z = eps_arr / sigma_seq
        logpdf = wrapper.logpdf(z, self.residual_params) - jnp.log(sigma_seq)
        return jnp.sum(logpdf)

    def loglikelihood(
        self,
        eps: Optional[ArrayLike] = None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        if eps is None:
            self._require_fitted()
            return self.loglikelihood_
        return self._log_likelihood_on_series(
            eps, init=init, backcast_length=backcast_length,
        )

    def aic(
        self,
        eps: Optional[ArrayLike] = None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        if eps is None:
            self._require_fitted()
            return self.aic_
        ll = self._log_likelihood_on_series(eps, init=init, backcast_length=backcast_length)
        return 2.0 * self.n_params - 2.0 * ll

    def bic(
        self,
        eps: Optional[ArrayLike] = None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        if eps is None:
            self._require_fitted()
            return self.bic_
        eps_arr = self._validate_series(eps)
        ll = self._log_likelihood_on_series(eps_arr, init=init, backcast_length=backcast_length)
        n = jnp.asarray(int(eps_arr.shape[0]), dtype=float)
        return self.n_params * jnp.log(n) - 2.0 * ll

    @property
    def residual_distribution(self) -> Univariate:
        r"""The fitted standardised residual distribution as a regular
        :class:`Univariate` instance, ready for direct ``sample`` /
        ``logpdf`` / ``cdf`` calls without going through the wrapper.
        """
        self._require_fitted()
        return self._wrapper().to_distribution(
            self.residual_params,
            name=f"{self.residual_dist.name}-stdresid-{self.name}",
        )

    # ------------------------------------------------------------------
    # Diagnostics — route through ``_diagnostics`` on standardised
    # residuals ``z_t = ε_t / σ_t``
    # ------------------------------------------------------------------
    def acf(
        self,
        eps: ArrayLike,
        lags: int = 20,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Sample ACF of the standardised residuals."""
        from copulax._src.timeseries._diagnostics import acf as _acf
        z = self.standardised_residuals(
            eps, init=init, backcast_length=backcast_length,
        )
        return _acf(z, lags)

    def pacf(
        self,
        eps: ArrayLike,
        lags: int = 20,
        method: str = "yule_walker",
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Sample PACF of the standardised residuals."""
        from copulax._src.timeseries._diagnostics import pacf as _pacf
        z = self.standardised_residuals(
            eps, init=init, backcast_length=backcast_length,
        )
        return _pacf(z, lags, method=method)

    def ljung_box(
        self,
        eps: ArrayLike,
        lags: int = 10,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> tuple[Array, Array]:
        r"""Ljung-Box Q-test on the standardised residuals.

        H0: ``z_t`` are white noise — passing confirms the IID
        assumption of the residual law on the fitted variance series.
        """
        from copulax._src.timeseries._diagnostics import ljung_box as _lb
        z = self.standardised_residuals(
            eps, init=init, backcast_length=backcast_length,
        )
        return _lb(z, lags)

    def arch_lm(
        self,
        eps: ArrayLike,
        lags: int = 5,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> tuple[Array, Array]:
        r"""Engle's ARCH-LM test on the standardised residuals.

        H0: no remaining ARCH effect.  Passing means the variance
        model captured all the heteroskedasticity; failing
        motivates a richer variance specification (higher orders,
        an asymmetric variant, or a different residual law).
        """
        from copulax._src.timeseries._diagnostics import arch_lm as _alm
        z = self.standardised_residuals(
            eps, init=init, backcast_length=backcast_length,
        )
        return _alm(z, lags)

    def plot_acf(
        self,
        eps: ArrayLike,
        lags: int = 20,
        alpha: float = 0.05,
        ax=None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ):
        r"""ACF stem plot for the standardised residuals."""
        from copulax._src.timeseries._diagnostics import plot_acf as _plot_acf
        z = self.standardised_residuals(
            eps, init=init, backcast_length=backcast_length,
        )
        return _plot_acf(z, lags=lags, alpha=alpha, ax=ax)

    def plot_pacf(
        self,
        eps: ArrayLike,
        lags: int = 20,
        method: str = "yule_walker",
        alpha: float = 0.05,
        ax=None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ):
        r"""PACF stem plot for the standardised residuals."""
        from copulax._src.timeseries._diagnostics import plot_pacf as _plot_pacf
        z = self.standardised_residuals(
            eps, init=init, backcast_length=backcast_length,
        )
        return _plot_pacf(z, lags=lags, method=method, alpha=alpha, ax=ax)

    # ------------------------------------------------------------------
    # Model-fit plots
    # ------------------------------------------------------------------
    def plot_timeseries(
        self,
        eps: ArrayLike,
        m: int = 5,
        alpha: tuple = (0.05, 0.95),
        show_rolling: bool = True,
        ax=None,
    ):
        r"""Time-series chart with VaR bands derived from the fitted
        residual law's quantiles times :math:`\sigma_t`."""
        from copulax._src.timeseries._plotting import plot_timeseries_variance
        return plot_timeseries_variance(
            self, eps, m=m, alpha=alpha, show_rolling=show_rolling, ax=ax,
        )

    def plot_scatter(
        self,
        eps: ArrayLike,
        m: int = 5,
        axes=None,
    ) -> tuple:
        r"""Two-panel diagnostic: model :math:`\sigma_t` vs rolling
        :math:`\sigma`, plus a Q-Q plot of standardised residuals.
        Returns ``(ax_sigma, ax_qq)``."""
        from copulax._src.timeseries._plotting import plot_scatter_variance
        return plot_scatter_variance(self, eps, m=m, axes=axes)
