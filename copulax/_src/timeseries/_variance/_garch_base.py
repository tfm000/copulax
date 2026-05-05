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
from copulax._src.timeseries._diagnostics import (
    acf as _diag_acf,
    arch_lm as _diag_arch_lm,
    ljung_box as _diag_ljung_box,
    pacf as _diag_pacf,
)
from copulax._src.timeseries._init import (
    garch_pre_sample_state,
    init_garch_params,
)
from copulax._src.timeseries._recursions import run_garch
from copulax._src.timeseries._residuals._standardise import StandardisedResidual
from copulax._src.timeseries._se import (
    compute_param_cov,
    flat_to_params,
    params_to_flat,
)
from copulax._src.timeseries._stationarity import (
    garch_simplex,
    garch_unsimplex,
    positive_to_raw,
    raw_to_positive,
)
from copulax._src.timeseries._summary import (
    ParamSection,
    build_diagnostic_rows,
    display_residual_name,
    format_summary,
    iter_param_rows,
    residual_section,
)
from copulax._src.timeseries._unit_root import adf as _diag_adf, kpss as _diag_kpss


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

    # ---- residual distribution (traced PyTree) -------------------------
    # Pre-fit: user template singleton; post-fit: the fitted
    # standardised (mean=0, var=1) residual distribution — see
    # the analogous field in ``ARMABase`` for the full contract.
    residual_dist: Optional[Univariate] = None

    # ---- traced fitted parameters ---------------------------------------
    omega: Optional[Array] = None
    alpha: Optional[Array] = None
    beta: Optional[Array] = None
    residual_params: Optional[dict] = None

    # ---- per-fit terminal carry + sample size --------------------------
    terminal_state: Optional[GARCHTerminalState] = None
    n_train_: Optional[int] = None

    # ---- post-fit standard errors (observed-Hessian / "classic") -------
    cov_matrix_: Optional[Array] = None
    standard_errors_: Optional[dict] = None

    # ---- post-fit residual diagnostics (cached default-arg results) ----
    # Single canonical bundle of every fit-time scalar / array / test
    # result on the standardised training residuals.  Schema:
    #
    # * ``"loglikelihood"``, ``"aic"``, ``"bic"`` — model-fit scalars.
    # * ``"acf"``, ``"pacf"`` — shape ``(21,)`` autocorrelation arrays
    #   at the default ``lags=20`` (PACF uses ``method="yule_walker"``).
    # * ``"ljung_box"``, ``"ljung_box_sq"``, ``"arch_lm"``, ``"adf"``,
    #   ``"kpss"`` — standardised hypothesis-test result dicts.
    #
    # ``eps=None`` defaults across :meth:`loglikelihood`, :meth:`aic`,
    # :meth:`bic`, :meth:`acf`, :meth:`pacf`, :meth:`ljung_box`,
    # :meth:`arch_lm`, :meth:`adf_residuals`, :meth:`kpss_residuals`,
    # :meth:`plot_acf`, :meth:`plot_pacf` all read from this dict.
    residual_diagnostics_: Optional[dict] = None

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
        n_train_: Optional[int] = None,
        cov_matrix_: Optional[ArrayLike] = None,
        standard_errors_: Optional[dict] = None,
        residual_diagnostics_: Optional[dict] = None,
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
        self.n_train_ = int(n_train_) if n_train_ is not None else None
        self.cov_matrix_ = (
            jnp.asarray(cov_matrix_, dtype=float)
            if cov_matrix_ is not None else None
        )
        self.standard_errors_ = (
            dict(standard_errors_) if standard_errors_ is not None else None
        )
        self.residual_diagnostics_ = (
            dict(residual_diagnostics_)
            if residual_diagnostics_ is not None else None
        )

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
    # Natural-parameter NLL closures + SE machinery
    # ------------------------------------------------------------------
    def _natural_objective_closures(
        self,
        wrapper: StandardisedResidual,
        params_dict: dict,
        eps_arr: Array,
        init_state: tuple,
    ):
        r"""Per-observation / sum negative-log-likelihood closures over
        the **flat natural-parameter vector** at the constrained MLE.

        Drives the observed-Hessian SE pipeline.  Variants whose NLL
        structure deviates from vanilla GARCH (e.g. :class:`GARCH_M`,
        which adds a variance-in-mean term) override this method.
        Variants whose only difference is the variance recursion
        (:class:`GJR_GARCH`, :class:`EGARCH`, :class:`TGARCH`,
        :class:`QGARCH`, :class:`IGARCH`) inherit it unchanged — the
        recursion dispatches via :meth:`_ag_run_recursion`.

        Args:
            wrapper: Standardised-residual wrapper.
            params_dict: Constrained-natural-params dict at the MLE.
            eps_arr: shape ``(n,)`` — input innovation series.
            init_state: Pre-sample carry tuple matching whatever
                shape the variant's recursion expects (computed at
                fit time and passed through).

        Returns:
            ``(nll_total, per_obs_nll, schema)``.
        """
        _, schema = params_to_flat(params_dict)
        var_keys = self._ag_var_keys()

        def per_obs_nll(flat: Array) -> Array:
            params = flat_to_params(flat, schema)
            residual_ = params.get("residual", {}) or {}
            var_dict = {k: params[k] for k in var_keys if k in params}
            var_seq, _ = self._ag_run_recursion(
                eps_seq=eps_arr,
                var_params=var_dict,
                residual_params=residual_,
                init_state=init_state,
            )
            sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
            z = eps_arr / sigma_seq
            logpdf = wrapper.logpdf(z, residual_) - jnp.log(sigma_seq)
            return -jnp.where(jnp.isfinite(logpdf), logpdf, 0.0)

        def nll_total(flat: Array) -> Array:
            return jnp.sum(per_obs_nll(flat))

        return nll_total, per_obs_nll, schema

    def _compute_se(
        self,
        params_dict: dict,
        wrapper: StandardisedResidual,
        eps_arr: Array,
        init_state: tuple,
        n_obs: int,
    ) -> tuple[Array, Array, dict]:
        r"""Observed-Hessian asymptotic covariance + standard errors.

        Uses ``cov_type="classic"`` — inverse observed Hessian, no
        Bollerslev-Wooldridge sandwich.  Standalone variance fits
        assume the residual law is correctly specified; the QMLE
        sandwich is unnecessary for this case (if you need it, fit
        via :class:`copulax.timeseries.ArmaGarch` whose default is
        the robust sandwich).
        """
        nll_total, per_obs_nll, schema = self._natural_objective_closures(
            wrapper, params_dict, eps_arr, init_state,
        )
        params_flat, _ = params_to_flat(params_dict)
        cov = compute_param_cov(
            nll_total=nll_total,
            per_obs_nll=per_obs_nll,
            params_flat=params_flat,
            n_obs=n_obs,
            cov_type="classic",
        )
        se_flat = jnp.sqrt(jnp.maximum(jnp.diag(cov), 0.0))
        se_dict = flat_to_params(se_flat, schema)
        return cov, se_flat, se_dict

    def _post_fit_se_and_diagnostics(
        self,
        params_dict: dict,
        wrapper: StandardisedResidual,
        eps_arr: Array,
        init_state: tuple,
        z_train: Array,
        *,
        loglikelihood: Array,
        aic: Array,
        bic: Array,
    ) -> tuple[Array, dict, dict]:
        r"""Helper used by every variant's ``fit()`` to compute the
        post-fit SE / CI / diagnostics state in one call.

        Returns ``(cov_matrix, std_err_dict, residual_diagnostics_dict)``
        — feed these directly into the fitted-instance constructor.
        The diagnostics bundle includes the fit-time
        ``loglikelihood`` / ``aic`` / ``bic`` scalars passed in by
        the caller alongside the autocorrelation arrays and the
        five hypothesis-test result dicts.
        """
        cov, _, se_dict = self._compute_se(
            params_dict=params_dict,
            wrapper=wrapper, eps_arr=eps_arr,
            init_state=init_state, n_obs=int(eps_arr.shape[0]),
        )
        diagnostics = self._compute_residual_diagnostics(
            z_train, loglikelihood=loglikelihood, aic=aic, bic=bic,
        )
        return cov, se_dict, diagnostics

    def _recompute_se(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> tuple[Array, Array, dict]:
        r"""Recompute SEs against an alternate ``eps`` series.

        Used by :meth:`standard_errors` / :meth:`cov_matrix` when an
        explicit series is supplied.
        """
        wrapper = self._wrapper()
        eps_arr = self._validate_series(eps)
        n_obs = int(eps_arr.shape[0])
        init_state = self._ag_initial_state(
            eps_proxy=eps_arr,
            mode=("sample" if init == "sample" else "backcast"),
            backcast_length=backcast_length,
            residual_params=self.residual_params or {},
        )
        return self._compute_se(
            params_dict=self.params,
            wrapper=wrapper, eps_arr=eps_arr,
            init_state=init_state, n_obs=n_obs,
        )

    # ------------------------------------------------------------------
    # Residual-diagnostic caching
    # ------------------------------------------------------------------
    def _compute_residual_diagnostics(
        self,
        z: Array,
        *,
        loglikelihood: Array,
        aic: Array,
        bic: Array,
    ) -> dict:
        r"""Build the canonical fit-time diagnostics bundle used by
        every cached default-arg accessor and by ``summary()``.

        Combines:

        * the model-fit scalars ``loglikelihood`` / ``aic`` / ``bic``;
        * the standardised-residual autocorrelation arrays
          ``acf`` / ``pacf`` at the convenience-wrapper defaults
          (``lags=20``, PACF ``method="yule_walker"``);
        * the five hypothesis-test result dicts on the standardised
          residuals: ``ljung_box`` / ``ljung_box_sq`` / ``arch_lm``
          / ``adf`` / ``kpss``.

        The dof correction on the squared-residual Ljung-Box matches
        the convenience wrapper's default (``lags - p - q``, floored
        at 1) since the fitted GARCH parameters bias the χ² reference
        for ``z²`` autocorrelation (Box-Jenkins-Reinsel §8.2.2).
        Plain Ljung-Box on ``z`` keeps ``dof=lags`` because the
        variance recursion does not induce serial autocorrelation in
        ``z`` itself under correct specification.
        """
        return {
            "loglikelihood": loglikelihood,
            "aic":           aic,
            "bic":           bic,
            "acf":           _diag_acf(z, 20),
            "pacf":          _diag_pacf(z, 20, method="yule_walker"),
            "ljung_box":     _diag_ljung_box(z, 10),
            "ljung_box_sq":  _diag_ljung_box(
                z * z, 10, dof=max(10 - self.p - self.q, 1),
            ),
            "arch_lm":       _diag_arch_lm(z, 5),
            "adf":           _diag_adf(z, regression="c"),
            "kpss":          _diag_kpss(z, regression="c"),
        }

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
            ``n_train_``, ``cov_matrix_`` / ``standard_errors_``,
            and the consolidated ``residual_diagnostics_`` bundle
            (which holds ``loglikelihood`` / ``aic`` / ``bic`` /
            ``acf`` / ``pacf`` plus the five hypothesis-test
            result dicts) populated.
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
        var_seq, terminal = self._run_recursion(
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

        # Standardised training-window residuals for the cached
        # diagnostic suite + observed-Hessian SEs at the MLE.
        sigma_train = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
        z_train = eps_arr / sigma_train
        params_dict = {
            "omega": omega, "alpha": alpha, "beta": beta,
            "residual": residual,
        }
        cov, se_dict, diagnostics = self._post_fit_se_and_diagnostics(
            params_dict=params_dict,
            wrapper=wrapper, eps_arr=eps_arr,
            init_state=(init_eps_sq_lags, init_var_lags),
            z_train=z_train,
            loglikelihood=loglike, aic=aic, bic=bic,
        )

        # Promote the unfitted template to the fitted standardised
        # distribution so ``fit.residual_dist`` is the canonical
        # accessor.
        fitted_residual_dist = wrapper.to_distribution(
            residual,
            name=f"{self.residual_dist.name}-stdresid",
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
            residual_dist=fitted_residual_dist,
            omega=omega,
            alpha=alpha,
            beta=beta,
            residual_params=residual,
            terminal_state=terminal,
            cov_matrix_=cov,
            standard_errors_=se_dict,
            residual_diagnostics_=diagnostics,
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
    ) -> dict:
        r"""Mean-corrected innovations and standardised residuals.

        Returns ``{"residuals": ε_t, "standardised_residuals": z_t}``
        where :math:`z_t = \varepsilon_t / \sigma_t`.  Variance
        models are fit on a mean-corrected series, so
        ``"residuals"`` here is the input ``eps`` (returned for
        symmetry with ARMA / ArmaGarch which compute innovations
        from a raw level series).

        The dict return shape is uniform across ARMA / GARCH /
        ArmaGarch so user code does not need per-family branching.

        Requires ``eps``; the model does not retain the training
        series — pass it explicitly.
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
        return {
            "residuals": eps_arr,
            "standardised_residuals": eps_arr / sigma_seq,
        }

    def standardised_residuals(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Just the ``z_t = ε_t / σ_t`` half of :meth:`residuals`."""
        return self.residuals(
            eps, init=init, backcast_length=backcast_length,
        )["standardised_residuals"]

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
        r"""Log-likelihood of the fitted model.

        With ``eps=None`` (default) returns the cached fit-time
        value ``residual_diagnostics_["loglikelihood"]``; pass
        ``eps`` to recompute on a held-out series.
        """
        if eps is None:
            self._require_fitted()
            return self.residual_diagnostics_["loglikelihood"]
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
        r"""Akaike Information Criterion.

        Returns ``residual_diagnostics_["aic"]`` when ``eps`` is
        omitted; recomputes against ``eps`` otherwise.
        """
        if eps is None:
            self._require_fitted()
            return self.residual_diagnostics_["aic"]
        ll = self._log_likelihood_on_series(eps, init=init, backcast_length=backcast_length)
        return 2.0 * self.n_params - 2.0 * ll

    def bic(
        self,
        eps: Optional[ArrayLike] = None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Bayesian Information Criterion.

        Returns ``residual_diagnostics_["bic"]`` when ``eps`` is
        omitted; recomputes against ``eps`` otherwise.
        """
        if eps is None:
            self._require_fitted()
            return self.residual_diagnostics_["bic"]
        eps_arr = self._validate_series(eps)
        ll = self._log_likelihood_on_series(eps_arr, init=init, backcast_length=backcast_length)
        n = jnp.asarray(int(eps_arr.shape[0]), dtype=float)
        return self.n_params * jnp.log(n) - 2.0 * ll

    # ------------------------------------------------------------------
    # Diagnostics — route through ``_diagnostics`` on standardised
    # residuals ``z_t = ε_t / σ_t``
    # ------------------------------------------------------------------
    def acf(
        self,
        eps: Optional[ArrayLike] = None,
        lags: int = 20,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Sample ACF of the standardised residuals.

        With ``eps=None`` (default) AND ``lags`` at its default,
        returns the cached value from
        ``residual_diagnostics_["acf"]`` populated at fit time.
        Pass ``eps`` to recompute against an alternate series;
        non-default ``lags`` requires ``eps`` explicitly.
        """
        if eps is None:
            self._require_fitted()
            if lags == 20 and self.residual_diagnostics_ is not None:
                return self.residual_diagnostics_["acf"]
            raise ValueError(
                "eps is required when overriding the default kwargs of "
                "acf() — only the default-arg result is cached."
            )
        z = self.standardised_residuals(
            eps, init=init, backcast_length=backcast_length,
        )
        return _diag_acf(z, lags)

    def pacf(
        self,
        eps: Optional[ArrayLike] = None,
        lags: int = 20,
        method: str = "yule_walker",
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Sample PACF of the standardised residuals.

        With ``eps=None`` (default) AND ``lags`` / ``method`` at
        their defaults, returns the cached value from
        ``residual_diagnostics_["pacf"]`` populated at fit time.
        Pass ``eps`` to recompute against an alternate series;
        non-default kwargs require ``eps`` explicitly.
        """
        if eps is None:
            self._require_fitted()
            if (
                lags == 20 and method == "yule_walker"
                and self.residual_diagnostics_ is not None
            ):
                return self.residual_diagnostics_["pacf"]
            raise ValueError(
                "eps is required when overriding the default kwargs of "
                "pacf() — only the default-arg result is cached."
            )
        z = self.standardised_residuals(
            eps, init=init, backcast_length=backcast_length,
        )
        return _diag_pacf(z, lags, method=method)

    def ljung_box(
        self,
        eps: Optional[ArrayLike] = None,
        lags: int = 10,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
        on: str = "residuals",
        dof_correction: bool = True,
    ) -> dict:
        r"""Ljung-Box Q-test on the standardised residuals.

        ``on="residuals"`` (default) tests the standardised residual
        series :math:`z_t = \varepsilon_t / \sigma_t` for white noise
        — H0 confirms the IID assumption of the residual law.  Under
        a correctly-specified variance model the GARCH parameters do
        not generate autocorrelation in :math:`z_t`, so the
        asymptotic dof remains ``lags`` even when ``dof_correction``
        is ``True``.

        ``on="squared_residuals"`` tests :math:`z^2_t` for remaining
        ARCH effects.  Under H0 (no remaining ARCH), with
        ``dof_correction=True`` the asymptotic
        :math:`\chi^2(lags - p - q)` accounts for the fitted
        :math:`\alpha_i` / :math:`\beta_j` coefficients
        (Bollerslev 1986; Box-Jenkins-Reinsel §8.2.2).

        With ``eps=None`` and every other kwarg at its default,
        returns the cached default-arg value from
        ``residual_diagnostics_["ljung_box"]`` (or
        ``["ljung_box_sq"]`` for ``on="squared_residuals"``).

        Returns the standardised result dict from
        :func:`copulax.timeseries.ljung_box` —
        ``{"statistic", "p_value", "used_lag", "n_obs", "dof"}``.
        """
        if on not in ("residuals", "squared_residuals"):
            raise ValueError(
                f"on must be 'residuals' or 'squared_residuals'; got {on!r}."
            )
        if eps is None:
            self._require_fitted()
            if (
                lags == 10 and dof_correction is True
                and self.residual_diagnostics_ is not None
            ):
                key = (
                    "ljung_box" if on == "residuals" else "ljung_box_sq"
                )
                return self.residual_diagnostics_[key]
            raise ValueError(
                "eps is required when overriding the default kwargs of "
                "ljung_box() — only the default-arg result is cached."
            )
        z = self.standardised_residuals(
            eps, init=init, backcast_length=backcast_length,
        )
        series = z if on == "residuals" else z * z
        if dof_correction and on == "squared_residuals":
            dof = lags - self.p - self.q
        else:
            dof = lags
        return _diag_ljung_box(series, lags, dof=dof)

    def arch_lm(
        self,
        eps: Optional[ArrayLike] = None,
        lags: int = 5,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> dict:
        r"""Engle's ARCH-LM test on the standardised residuals.

        H0: no remaining ARCH effect.  Passing means the variance
        model captured all the heteroskedasticity; failing
        motivates a richer variance specification (higher orders,
        an asymmetric variant, or a different residual law).

        Cached default-arg behaviour: ``eps=None`` returns
        ``residual_diagnostics_["arch_lm"]``.

        Returns the standardised result dict from
        :func:`copulax.timeseries.arch_lm` —
        ``{"statistic", "p_value", "used_lag", "n_obs", "dof"}``.
        """
        if eps is None:
            self._require_fitted()
            if lags == 5 and self.residual_diagnostics_ is not None:
                return self.residual_diagnostics_["arch_lm"]
            raise ValueError(
                "eps is required when overriding the default kwargs of "
                "arch_lm() — only the default-arg result is cached."
            )
        z = self.standardised_residuals(
            eps, init=init, backcast_length=backcast_length,
        )
        return _diag_arch_lm(z, lags)

    def adf_residuals(
        self,
        eps: Optional[ArrayLike] = None,
        *,
        regression: str = "c",
        lags: Optional[int] = None,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> dict:
        r"""Augmented Dickey-Fuller test on the standardised residuals.

        Sanity check that the variance model has captured the
        heteroskedasticity in the input — for a healthy fit, ADF
        on standardised residuals should reject the unit-root null.

        Cached default-arg behaviour mirrors :meth:`ljung_box`.
        """
        if eps is None:
            self._require_fitted()
            if (
                regression == "c" and lags is None
                and self.residual_diagnostics_ is not None
            ):
                return self.residual_diagnostics_["adf"]
            raise ValueError(
                "eps is required when overriding the default kwargs of "
                "adf_residuals() — only the default-arg result is cached."
            )
        z = self.standardised_residuals(
            eps, init=init, backcast_length=backcast_length,
        )
        return _diag_adf(z, regression=regression, lags=lags)

    def kpss_residuals(
        self,
        eps: Optional[ArrayLike] = None,
        *,
        regression: str = "c",
        lags: Optional[int] = None,
        lags_choice: str = "short",
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> dict:
        r"""KPSS stationarity test on the standardised residuals.

        Complements :meth:`adf_residuals` — for a healthy fit, KPSS
        should fail to reject the stationarity null.

        Cached default-arg behaviour mirrors :meth:`ljung_box`.
        """
        if eps is None:
            self._require_fitted()
            if (
                regression == "c" and lags is None
                and lags_choice == "short"
                and self.residual_diagnostics_ is not None
            ):
                return self.residual_diagnostics_["kpss"]
            raise ValueError(
                "eps is required when overriding the default kwargs of "
                "kpss_residuals() — only the default-arg result is cached."
            )
        z = self.standardised_residuals(
            eps, init=init, backcast_length=backcast_length,
        )
        return _diag_kpss(
            z, regression=regression, lags=lags, lags_choice=lags_choice,
        )

    # ------------------------------------------------------------------
    # Standard errors / confidence intervals / summary
    # ------------------------------------------------------------------
    def cov_matrix(
        self,
        eps: Optional[ArrayLike] = None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Asymptotic covariance matrix of the natural-parameter MLE.

        Inverse observed Hessian (``cov_type="classic"``) — assumes
        the residual law is correctly specified.  Use
        :class:`copulax.timeseries.ArmaGarch` for the
        Bollerslev-Wooldridge sandwich form on a joint MLE.

        With ``eps=None`` returns the cached ``cov_matrix_``.
        """
        self._require_fitted()
        if eps is None:
            return self.cov_matrix_
        return self._recompute_se(
            eps, init=init, backcast_length=backcast_length,
        )[0]

    def standard_errors(
        self,
        eps: Optional[ArrayLike] = None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> dict:
        r"""Asymptotic standard errors structured to mirror :attr:`params`."""
        self._require_fitted()
        if eps is None:
            return self.standard_errors_
        return self._recompute_se(
            eps, init=init, backcast_length=backcast_length,
        )[2]

    def confidence_intervals(self, alpha: float = 0.05) -> dict:
        r"""``(lower, upper)`` confidence intervals at significance
        ``alpha`` for every fitted parameter, structured to mirror
        :attr:`params`."""
        self._require_fitted()
        if self.standard_errors_ is None or self.params is None:
            raise ValueError(
                "Confidence intervals require both `standard_errors_` "
                "and `params` to be populated."
            )
        from jax.scipy.stats import norm
        z_crit = float(norm.ppf(1.0 - alpha / 2.0))
        cis: dict = {}
        for key, val in self.params.items():
            if key == "residual":
                cis[key] = {}
                for sub_key, sub_val in val.items():
                    sub_se = self.standard_errors_[key][sub_key]
                    sub_arr = jnp.asarray(sub_val, dtype=float)
                    sub_se_arr = jnp.asarray(sub_se, dtype=float)
                    cis[key][sub_key] = (
                        sub_arr - z_crit * sub_se_arr,
                        sub_arr + z_crit * sub_se_arr,
                    )
            else:
                est = jnp.asarray(val, dtype=float)
                se = jnp.asarray(self.standard_errors_[key], dtype=float)
                cis[key] = (est - z_crit * se, est + z_crit * se)
        return cis

    def summary(self) -> str:
        r"""Render a printable parameter / diagnostics table.

        Sections (variance equation + residual distribution +
        residual diagnostics) are separated by inline-labelled
        dashed lines; empty sections (e.g. residual distribution
        under ``normal``-law fits, where there are no free shape
        parameters) are silently suppressed.
        """
        self._require_fitted()
        if (
            self.standard_errors_ is None
            or self.residual_diagnostics_ is None
        ):
            raise ValueError(
                "summary() requires `standard_errors_` and "
                "`residual_diagnostics_` to be populated.  Refit the "
                "model on a recent CopulAX version."
            )
        var_keys = [k for k in self.params if k != "residual"]
        var_section = ParamSection(
            label=self._variance_section_label(),
            rows=iter_param_rows(
                {k: self.params[k] for k in var_keys},
                {k: self.standard_errors_[k] for k in var_keys},
                vector_keys=(
                    "alpha", "beta", "gamma",
                    "alpha_pos", "alpha_neg",
                ),
            ),
        )
        res_section = residual_section(
            self.params["residual"],
            self.standard_errors_["residual"],
            dist_name=display_residual_name(self.residual_dist.name),
        )
        return format_summary(
            header=self._summary_header(),
            param_sections=[var_section, res_section],
            diagnostic_rows=build_diagnostic_rows(self.residual_diagnostics_),
            loglikelihood=float(self.residual_diagnostics_["loglikelihood"]),
            aic=float(self.residual_diagnostics_["aic"]),
            bic=float(self.residual_diagnostics_["bic"]),
            n_train=int(self.n_train_),
        )

    def _summary_header(self) -> str:
        r"""Top-line of :meth:`summary` — class-name driven, so every
        variant gets the right label out of the box."""
        return (
            f"{type(self).__name__}({self.p}, {self.q}) — "
            f"{display_residual_name(self.residual_dist.name)} residuals"
        )

    def _variance_section_label(self) -> str:
        r"""Label embedded in the variance-equation section's
        separator line."""
        return (
            f"Variance equation — {type(self).__name__}({self.p}, {self.q})"
        )

    def plot_acf(
        self,
        eps: Optional[ArrayLike] = None,
        lags: int = 20,
        alpha: float = 0.05,
        ax=None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ):
        r"""ACF stem plot for the standardised residuals.

        With ``eps=None`` (default) AND ``lags`` at its default,
        renders the plot from the cached
        ``residual_diagnostics_["acf"]`` array (no recursion).
        Pass ``eps`` to recompute against an alternate series;
        non-default ``lags`` requires ``eps`` explicitly.
        """
        from copulax._src.timeseries._diagnostics import (
            plot_acf as _plot_acf,
            plot_acf_from_corr as _plot_acf_from_corr,
        )
        if eps is None:
            self._require_fitted()
            if lags == 20 and self.residual_diagnostics_ is not None:
                return _plot_acf_from_corr(
                    self.residual_diagnostics_["acf"],
                    n_obs=int(self.n_train_),
                    alpha=alpha, ax=ax,
                )
            raise ValueError(
                "eps is required when overriding the default kwargs of "
                "plot_acf() — only the default-arg result is cached."
            )
        z = self.standardised_residuals(
            eps, init=init, backcast_length=backcast_length,
        )
        return _plot_acf(z, lags=lags, alpha=alpha, ax=ax)

    def plot_pacf(
        self,
        eps: Optional[ArrayLike] = None,
        lags: int = 20,
        method: str = "yule_walker",
        alpha: float = 0.05,
        ax=None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ):
        r"""PACF stem plot for the standardised residuals.

        With ``eps=None`` (default) AND ``lags`` / ``method`` at
        their defaults, renders the plot from the cached
        ``residual_diagnostics_["pacf"]`` array.  Pass ``eps`` to
        recompute against an alternate series; non-default kwargs
        require ``eps`` explicitly.
        """
        from copulax._src.timeseries._diagnostics import (
            plot_pacf as _plot_pacf,
            plot_pacf_from_corr as _plot_pacf_from_corr,
        )
        if eps is None:
            self._require_fitted()
            if (
                lags == 20 and method == "yule_walker"
                and self.residual_diagnostics_ is not None
            ):
                return _plot_pacf_from_corr(
                    self.residual_diagnostics_["pacf"],
                    n_obs=int(self.n_train_),
                    alpha=alpha, ax=ax,
                )
            raise ValueError(
                "eps is required when overriding the default kwargs of "
                "plot_pacf() — only the default-arg result is cached."
            )
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

    # ------------------------------------------------------------------
    # Deserialisation
    # ------------------------------------------------------------------
    @classmethod
    def _deserialise(
        cls,
        metadata: dict,
        arrays: dict,
        residual_dist,
        name: Optional[str] = None,
    ) -> "GARCHBase":
        r"""Reconstruct a fitted variance-model instance from saved
        state.  The default mapping handles vanilla GARCH / IGARCH
        (param keys ``omega``, ``alpha``, ``beta``); subclasses
        whose ``params`` dict introduces extra keys (e.g. ``gamma``
        for GJR/EGARCH, ``alpha_neg`` for TGARCH, ``psi`` for
        QGARCH, ``mu``/``lambda_m`` for GARCH-M) override only the
        ``_deserialise_extra_kwargs`` hook to thread their
        variant-specific keys into the constructor.
        """
        from copulax._src.timeseries._se import flat_to_params

        kwargs: dict = {
            "p": int(metadata["p"]),
            "q": int(metadata["q"]),
            "residual_dist": residual_dist,
        }
        if name is not None:
            kwargs["name"] = name

        if "params_schema" in metadata:
            schema = [(k, tuple(s)) for k, s in metadata["params_schema"]]
            params = flat_to_params(arrays["params_flat"], schema)
            kwargs["omega"] = params.get("omega")
            kwargs["alpha"] = params.get("alpha")
            kwargs["beta"] = params.get("beta")
            if "residual" in params:
                kwargs["residual_params"] = params["residual"]
                # Promote template → fitted instance so the
                # round-tripped ``residual_dist.params`` matches the
                # original fit (and ``residual_distribution`` is
                # gone, the field is the only accessor).
                kwargs["residual_dist"] = StandardisedResidual(
                    residual_dist,
                ).to_distribution(
                    params["residual"],
                    name=f"{residual_dist.name}-stdresid",
                )
            kwargs.update(cls._deserialise_extra_kwargs(params))

        if "ts_n_leaves" in metadata:
            ts_class_name = metadata["ts_class"]
            ts_class = _lookup_terminal_state_class(ts_class_name)
            n_leaves = int(metadata["ts_n_leaves"])
            leaves = [arrays[f"ts_{i}"] for i in range(n_leaves)]
            kwargs["terminal_state"] = ts_class(*leaves)

        if "diag_n_train_" in arrays:
            kwargs["n_train_"] = arrays["diag_n_train_"]

        if "cov_matrix_" in arrays:
            kwargs["cov_matrix_"] = arrays["cov_matrix_"]
        if "se_schema" in metadata and "se_flat" in arrays:
            se_schema = [(k, tuple(s)) for k, s in metadata["se_schema"]]
            kwargs["standard_errors_"] = flat_to_params(
                arrays["se_flat"], se_schema,
            )

        from copulax._src.timeseries._base import (
            _deserialise_residual_diagnostics,
        )
        diagnostics = _deserialise_residual_diagnostics(arrays, metadata)
        if diagnostics is not None:
            kwargs["residual_diagnostics_"] = diagnostics

        return cls(**kwargs)

    @classmethod
    def _deserialise_extra_kwargs(cls, params: dict) -> dict:
        r"""Hook for variant-specific kwargs at deserialisation.

        Vanilla GARCH / IGARCH have no extras; subclasses override
        to map ``params`` dict keys to additional ``__init__``
        kwargs (e.g. GJR maps ``params["gamma"]`` → ``gamma=``).
        """
        return {}


def _lookup_terminal_state_class(name: str) -> type:
    r"""Look up a TerminalState subclass by name across the
    timeseries subpackage.  Used by the deserialisation path.
    """
    from copulax._src.timeseries._mean._arma_base import ARMATerminalState
    from copulax._src.timeseries._variance._garch_base import GARCHTerminalState
    from copulax._src.timeseries._variance.egarch import EGARCHTerminalState
    from copulax._src.timeseries._variance.gjr_garch import GJRTerminalState
    from copulax._src.timeseries._variance.qgarch import QGARCHTerminalState
    from copulax._src.timeseries._variance.tgarch import TGARCHTerminalState
    from copulax._src.timeseries._joint.arma_garch import ArmaGarchTerminalState

    table = {
        "ARMATerminalState": ARMATerminalState,
        "GARCHTerminalState": GARCHTerminalState,
        "EGARCHTerminalState": EGARCHTerminalState,
        "GJRTerminalState": GJRTerminalState,
        "QGARCHTerminalState": QGARCHTerminalState,
        "TGARCHTerminalState": TGARCHTerminalState,
        "ArmaGarchTerminalState": ArmaGarchTerminalState,
    }
    if name not in table:
        raise ValueError(
            f"Unknown TerminalState class {name!r}.  Add a new entry to "
            "_lookup_terminal_state_class in "
            "copulax/_src/timeseries/_variance/_garch_base.py."
        )
    return table[name]
