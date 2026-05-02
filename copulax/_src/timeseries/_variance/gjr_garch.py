r"""GJR-GARCH(p, q) — asymmetric / leverage GARCH (Glosten-Jagannathan-Runkle 1993).

Adds an asymmetric ``γ_i`` term to the σ²-recursion that activates
only when the lagged innovation was negative — capturing the
empirical regularity that negative shocks raise volatility more than
positive shocks of the same magnitude:

.. math::

    \sigma^2_t = \omega
               + \sum_{i=1}^p \alpha_i\, \varepsilon^2_{t-i}
               + \sum_{i=1}^p \gamma_i\, \varepsilon^2_{t-i}\,
                              \mathbf{1}\{\varepsilon_{t-i} < 0\}
               + \sum_{j=1}^q \beta_j\, \sigma^2_{t-j}.

The persistence condition is residual-law dependent:

.. math::

    \sum_i \alpha_i + \kappa \sum_i \gamma_i + \sum_j \beta_j < 1,
    \qquad
    \kappa = \mathbb{E}[z^2 \mathbf{1}\{z<0\}].

Under symmetric residuals :math:`\kappa = 1/2` and the textbook
``Σα + Σγ/2 + Σβ < 1`` recovers; under skewed residuals
:math:`\kappa \neq 1/2` and substituting one for the other
*understates* persistence, which biases the fit precisely where
GJR-GARCH is most useful.  We therefore compute :math:`\kappa` via
quadrature inside the fit objective at every step (autograd
through :math:`\kappa` flows into the residual's shape parameters
automatically) — see
:meth:`copulax._src.timeseries._residuals._standardise.StandardisedResidual.expected_z2_negative`.

Reference:
    Glosten, L.R., Jagannathan, R. & Runkle, D.E. (1993).  *On the
    relation between the expected value and the volatility of the
    nominal excess return on stocks*.  Journal of Finance, 48(5),
    1779-1801.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src.optimize import projected_gradient
from copulax._src.timeseries._base import TerminalState
from copulax._src.timeseries._init import (
    garch_pre_sample_state,
    init_garch_params,
)
from copulax._src.timeseries._recursions import run_gjr_garch
from copulax._src.timeseries._residuals._standardise import StandardisedResidual
from copulax._src.timeseries._stationarity import (
    gjr_simplex,
    gjr_unsimplex,
    positive_to_raw,
    raw_to_positive,
)
from copulax._src.timeseries._variance._garch_base import GARCHBase


_VAR_FLOOR: float = 1e-12
_SIGMA_FLOOR: float = 1e-6


###############################################################################
# Per-family terminal state — adds a "negative-shock" lag buffer
###############################################################################
class GJRTerminalState(TerminalState):
    r"""Constant-size carry for GJR-GARCH ``forecast(h)``.

    Stores the last ``p`` squared residuals, the last ``p``
    *negative-only* squared residuals (``ε² · 1{ε<0}``), and the
    last ``q`` conditional variances — exactly what
    :func:`run_gjr_garch` consumes as initial state.
    """
    eps_sq_lags: Array
    neg_eps_sq_lags: Array
    var_lags: Array


###############################################################################
# GJR_GARCH
###############################################################################
class GJR_GARCH(GARCHBase):
    r"""GJR-GARCH(p, q) — asymmetric leverage σ²-recursion.

    Construct with the desired orders and residual law:

    .. code-block:: python

        from copulax.timeseries import GJR_GARCH
        from copulax.univariate import skewed_t
        fit = GJR_GARCH(p=1, q=1, residual_dist=skewed_t).fit(eps)

    Inherits the :meth:`forecast` / :meth:`residuals` /
    :meth:`stats` etc. surface from :class:`GARCHBase` (with
    overrides where the recursion shape differs).
    """

    # Add the asymmetric coefficients as an additional traced field.
    gamma: Optional[Array] = None

    # Override the terminal-state field so the type-checker / equinox
    # round-trip uses the GJR shape.
    terminal_state: Optional[GJRTerminalState] = None

    def __init__(
        self,
        p: int = 0,
        q: int = 0,
        *,
        residual_dist: Optional[Univariate] = None,
        name: str = "GJR-GARCH",
        omega=None,
        alpha=None,
        gamma=None,
        beta=None,
        residual_params=None,
        terminal_state: Optional[GJRTerminalState] = None,
        loglikelihood_=None,
        aic_=None,
        bic_=None,
        n_train_: Optional[int] = None,
    ):
        super().__init__(
            name=name,
            p=p,
            q=q,
            residual_dist=residual_dist,
            omega=omega,
            alpha=alpha,
            beta=beta,
            residual_params=residual_params,
            terminal_state=terminal_state,
            loglikelihood_=loglikelihood_,
            aic_=aic_,
            bic_=bic_,
            n_train_=n_train_,
        )
        self.gamma = (
            jnp.asarray(gamma, dtype=float).reshape(-1)
            if gamma is not None else None
        )

    # ------------------------------------------------------------------
    # params property
    # ------------------------------------------------------------------
    @property
    def _stored_params(self) -> Optional[dict]:
        r"""Canonical parameter dict, or ``None`` for an unfitted instance.

        Schema:

        ``{
            "omega":     (),
            "alpha":     (p,),
            "gamma":     (p,),
            "beta":      (q,),
            "residual":  {<shape-only dict>},
        }``
        """
        if (
            self.omega is None or self.alpha is None or self.beta is None
            or self.gamma is None or self.residual_params is None
        ):
            return None
        return {
            "omega": self.omega,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "beta": self.beta,
            "residual": dict(self.residual_params),
        }

    @property
    def n_params(self) -> int:
        r"""Number of free fitted parameters: ω + α (p) + γ (p) + β (q)
        + residual shape."""
        wrapper = StandardisedResidual(self.residual_dist)
        return 1 + 2 * self.p + self.q + wrapper.n_shape_params

    # ------------------------------------------------------------------
    # Reparameterisation pack / unpack
    # ------------------------------------------------------------------
    def _pack_x0_gjr(
        self, params_dict: dict, wrapper: StandardisedResidual,
    ) -> Array:
        r"""Pack a GJR ``params_dict`` into the unconstrained flat
        optimiser-state vector.

        Layout: ``[raw_omega (1,), raw_persistence (1,),
        raw_weights (2p+q,), raw_residual_shape (n_shape_params,)]``.
        """
        omega = jnp.asarray(params_dict["omega"], dtype=float).reshape(())
        alpha = jnp.asarray(params_dict["alpha"], dtype=float).reshape(-1)
        gamma = jnp.asarray(params_dict["gamma"], dtype=float).reshape(-1)
        beta = jnp.asarray(params_dict["beta"], dtype=float).reshape(-1)
        residual = params_dict.get("residual", {}) or {}

        raw_omega = positive_to_raw(jnp.maximum(omega, _SIGMA_FLOOR))
        # κ at the residual's example shape so the unsimplex is
        # consistent with the simplex's interpretation of the chunks.
        kappa = wrapper.expected_z2_negative(residual)
        raw_persistence, raw_weights = gjr_unsimplex(alpha, gamma, beta, kappa)
        raw_residual = wrapper.shape_params_to_array(residual)
        return jnp.concatenate(
            [
                raw_omega.reshape((1,)),
                raw_persistence.reshape((1,)),
                raw_weights,
                raw_residual,
            ]
        )

    def _unpack_raw_gjr(
        self, raw: Array, wrapper: StandardisedResidual,
    ) -> tuple[Array, Array, Array, Array, dict]:
        r"""Inverse of :meth:`_pack_x0_gjr`.

        Returns ``(omega, alpha, gamma, beta, residual_shape_dict)``.
        """
        idx = 0
        raw_omega = raw[idx]
        idx += 1
        raw_persistence = raw[idx]
        idx += 1
        raw_weights = raw[idx : idx + 2 * self.p + self.q]
        idx += 2 * self.p + self.q
        raw_residual = raw[idx : idx + wrapper.n_shape_params]

        residual = wrapper.shape_params_from_array(raw_residual)
        kappa = wrapper.expected_z2_negative(residual)

        omega = raw_to_positive(raw_omega)
        alpha, gamma, beta = gjr_simplex(
            raw_persistence, raw_weights, p=self.p, q=self.q, kappa=kappa,
        )
        return omega, alpha, gamma, beta, residual

    # ------------------------------------------------------------------
    # Recursion + initial state
    # ------------------------------------------------------------------
    def _initial_state_gjr(
        self,
        eps: Array,
        mode: str,
        backcast_length: Optional[int],
    ) -> tuple[Array, Array, Array]:
        r"""Three-buffer pre-sample state for the GJR recursion."""
        eps_sq_lags, var_lags = garch_pre_sample_state(
            eps, p=self.p, q=self.q,
            mode=mode, backcast_length=backcast_length,
        )
        # Pre-sample neg-eps² lags: half of the eps² anchor (assuming
        # symmetry of the leading window — exact for symmetric
        # innovations and a small bias for skewed ones, which the
        # recursion absorbs in its first ``p`` steps anyway).
        neg_eps_sq_lags = 0.5 * eps_sq_lags
        return eps_sq_lags, neg_eps_sq_lags, var_lags

    def _run_recursion_gjr(
        self,
        eps: Array,
        omega: Array,
        alpha: Array,
        gamma: Array,
        beta: Array,
        init_state: tuple[Array, Array, Array],
    ) -> tuple[Array, GJRTerminalState]:
        eps_sq, neg_eps_sq, var_lags = init_state
        var_seq, terminal = run_gjr_garch(
            eps=eps, omega=omega, alpha=alpha, gamma=gamma, beta=beta,
            init_eps_sq_lags=eps_sq,
            init_neg_eps_sq_lags=neg_eps_sq,
            init_var_lags=var_lags,
        )
        return var_seq, GJRTerminalState(
            eps_sq_lags=terminal[0],
            neg_eps_sq_lags=terminal[1],
            var_lags=terminal[2],
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def _build_cold_start(
        self,
        eps: Array,
        wrapper: StandardisedResidual,
        init: str,
        backcast_length: Optional[int],
    ) -> dict:
        r"""Cold-start params for GJR-GARCH.

        Reuse the vanilla GARCH cold-start for ``(omega, alpha, beta)``
        and seed γ at zero (the symmetric / GARCH-equivalent starting
        point).
        """
        base = super()._build_cold_start(
            eps, wrapper, init=init, backcast_length=backcast_length,
        )
        base["gamma"] = jnp.zeros((self.p,), dtype=float)
        return base

    def _make_objective_gjr(self, wrapper: StandardisedResidual):
        def objective(
            raw: Array,
            eps: Array,
            init_eps_sq_lags: Array,
            init_neg_eps_sq_lags: Array,
            init_var_lags: Array,
        ) -> Array:
            omega, alpha, gamma, beta, residual_shape = self._unpack_raw_gjr(
                raw, wrapper,
            )
            init_state = (init_eps_sq_lags, init_neg_eps_sq_lags, init_var_lags)
            var_seq, _ = self._run_recursion_gjr(
                eps, omega, alpha, gamma, beta, init_state,
            )
            sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
            z = eps / sigma_seq
            logpdf = wrapper.logpdf(z, residual_shape) - jnp.log(sigma_seq)
            finite = jnp.isfinite(logpdf)
            safe_logpdf = jnp.where(finite, logpdf, 0.0)
            invalid_penalty = 1e6 * (~finite).mean()
            return -safe_logpdf.mean() + invalid_penalty
        return objective

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
    ) -> "GJR_GARCH":
        r"""Fit GJR-GARCH(p, q) to a mean-corrected innovation series.

        Identical contract to :meth:`GARCHBase.fit`; see that method
        for argument documentation.
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
            for key in ("omega", "alpha", "gamma", "beta", "residual"):
                if key not in cold:
                    raise KeyError(
                        f"Warm-start init_params missing required key {key!r}."
                    )
        else:
            cold = self._build_cold_start(
                eps_arr, wrapper, init=init, backcast_length=backcast_length,
            )

        x0 = self._pack_x0_gjr(cold, wrapper)

        _state_mode = "sample" if init == "sample" else "backcast"
        init_eps_sq_lags, init_neg_eps_sq_lags, init_var_lags = (
            self._initial_state_gjr(
                eps_arr, mode=_state_mode, backcast_length=backcast_length,
            )
        )

        objective = self._make_objective_gjr(wrapper)
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
            init_neg_eps_sq_lags=init_neg_eps_sq_lags,
            init_var_lags=init_var_lags,
            lr=lr,
            maxiter=maxiter,
        )
        x_opt = res["x"]
        omega, alpha, gamma, beta, residual = self._unpack_raw_gjr(x_opt, wrapper)

        # Final pass for terminal state.
        _, terminal = self._run_recursion_gjr(
            eps_arr, omega, alpha, gamma, beta,
            init_state=(
                init_eps_sq_lags, init_neg_eps_sq_lags, init_var_lags,
            ),
        )
        nll = objective(
            x_opt, eps_arr,
            init_eps_sq_lags, init_neg_eps_sq_lags, init_var_lags,
        )
        loglike = -nll * n
        n_params_total = 1 + 2 * self.p + self.q + wrapper.n_shape_params
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
            gamma=gamma,
            beta=beta,
            residual_params=residual,
            terminal_state=terminal,
            loglikelihood_=loglike,
            aic_=aic,
            bic_=bic,
            n_train_=n,
        )

    # ------------------------------------------------------------------
    # Conditional variance / residuals — override to use GJR kernel
    # ------------------------------------------------------------------
    def _gjr_recursion_inputs(
        self,
        eps: ArrayLike,
        init: str,
        backcast_length: Optional[int],
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        eps_arr = self._validate_series(eps)
        n = int(eps_arr.shape[0])
        self._validate_backcast_length(backcast_length, n)
        init_state = self._initial_state_gjr(
            eps_arr, mode=init, backcast_length=backcast_length,
        )
        return eps_arr, init_state

    def conditional_variance(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        self._require_fitted()
        eps_arr, init_state = self._gjr_recursion_inputs(
            eps, init, backcast_length,
        )
        var_seq, _ = self._run_recursion_gjr(
            eps_arr, self.omega, self.alpha, self.gamma, self.beta, init_state,
        )
        return var_seq

    def residuals(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> tuple[Array, Array]:
        self._require_fitted()
        eps_arr, init_state = self._gjr_recursion_inputs(
            eps, init, backcast_length,
        )
        var_seq, _ = self._run_recursion_gjr(
            eps_arr, self.omega, self.alpha, self.gamma, self.beta, init_state,
        )
        sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
        return eps_arr, eps_arr / sigma_seq

    def terminal_state_from(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> GJRTerminalState:
        self._require_fitted()
        eps_arr, init_state = self._gjr_recursion_inputs(
            eps, init, backcast_length,
        )
        _, terminal = self._run_recursion_gjr(
            eps_arr, self.omega, self.alpha, self.gamma, self.beta, init_state,
        )
        return terminal

    # ------------------------------------------------------------------
    # Loglikelihood / aic / bic — override to use GJR kernel
    # ------------------------------------------------------------------
    def _log_likelihood_on_series(
        self,
        eps: ArrayLike,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        self._require_fitted()
        wrapper = self._wrapper()
        eps_arr, init_state = self._gjr_recursion_inputs(
            eps, init, backcast_length,
        )
        var_seq, _ = self._run_recursion_gjr(
            eps_arr, self.omega, self.alpha, self.gamma, self.beta, init_state,
        )
        sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
        z = eps_arr / sigma_seq
        logpdf = wrapper.logpdf(z, self.residual_params) - jnp.log(sigma_seq)
        return jnp.sum(logpdf)

    # ------------------------------------------------------------------
    # Forecast — analytical recursion now incorporates γ·E[ε²·1{ε<0}]
    # ------------------------------------------------------------------
    def _analytical_forecast(self, h: int, state: GJRTerminalState) -> Array:
        r"""Analytical h-step variance forecast for GJR-GARCH.

        For unobserved future shocks, substitute :math:`\mathbb{E}[
        \varepsilon^2_{\tau-i}] = \mathbb{E}[\sigma^2_{\tau-i}]` and
        :math:`\mathbb{E}[\varepsilon^2_{\tau-i} \mathbf{1}\{
        \varepsilon_{\tau-i}<0\}] = \kappa \, \mathbb{E}[\sigma^2_{
        \tau-i}]`, where :math:`\kappa` is the truncated second
        moment of the standardised residual.
        """
        wrapper = self._wrapper()
        kappa = wrapper.expected_z2_negative(self.residual_params)
        var_path = []
        eps_sq_lags = state.eps_sq_lags
        neg_eps_sq_lags = state.neg_eps_sq_lags
        var_lags = state.var_lags
        for _ in range(h):
            ar_term = (
                jnp.dot(self.alpha, eps_sq_lags) if self.p > 0 else 0.0
            )
            asymm_term = (
                jnp.dot(self.gamma, neg_eps_sq_lags) if self.p > 0 else 0.0
            )
            ma_term = jnp.dot(self.beta, var_lags) if self.q > 0 else 0.0
            var_t = self.omega + ar_term + asymm_term + ma_term
            var_t = jnp.maximum(var_t, _VAR_FLOOR)
            var_path.append(var_t)
            if self.p > 0:
                eps_sq_lags = jnp.concatenate(
                    [var_t.reshape((1,)), eps_sq_lags[:-1]]
                )
                neg_eps_sq_lags = jnp.concatenate(
                    [(kappa * var_t).reshape((1,)), neg_eps_sq_lags[:-1]]
                )
            if self.q > 0:
                var_lags = jnp.concatenate(
                    [var_t.reshape((1,)), var_lags[:-1]]
                )
        return jnp.stack(var_path)

    # ------------------------------------------------------------------
    # rvs roll-path — override to track negative-shock buffer
    # ------------------------------------------------------------------
    def _roll_path(self, z: Array, state: GJRTerminalState) -> Array:
        omega = self.omega
        alpha = self.alpha
        gamma = self.gamma
        beta = self.beta
        import jax

        def step(carry, z_t):
            eps_sq_lags, neg_eps_sq_lags, var_lags = carry
            ar_term = jnp.dot(alpha, eps_sq_lags) if self.p > 0 else 0.0
            asymm_term = jnp.dot(gamma, neg_eps_sq_lags) if self.p > 0 else 0.0
            ma_term = jnp.dot(beta, var_lags) if self.q > 0 else 0.0
            var_t = omega + ar_term + asymm_term + ma_term
            var_t = jnp.maximum(var_t, _VAR_FLOOR)
            sigma_t = jnp.sqrt(var_t)
            eps_t = sigma_t * z_t
            eps_sq_t = eps_t * eps_t
            neg_eps_sq_t = jnp.where(eps_t < 0.0, eps_sq_t, 0.0)
            new_eps_sq = (
                jnp.concatenate([eps_sq_t.reshape((1,)), eps_sq_lags[:-1]])
                if self.p > 0 else eps_sq_lags
            )
            new_neg_eps_sq = (
                jnp.concatenate(
                    [neg_eps_sq_t.reshape((1,)), neg_eps_sq_lags[:-1]]
                )
                if self.p > 0 else neg_eps_sq_lags
            )
            new_var = (
                jnp.concatenate([var_t.reshape((1,)), var_lags[:-1]])
                if self.q > 0 else var_lags
            )
            return (new_eps_sq, new_neg_eps_sq, new_var), eps_t

        init_carry = (
            state.eps_sq_lags, state.neg_eps_sq_lags, state.var_lags,
        )
        _, eps_seq = jax.lax.scan(step, init_carry, z)
        return eps_seq

    # ------------------------------------------------------------------
    # Stats — persistence is κ-weighted
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        r"""Analytic, parameter-only diagnostics for GJR-GARCH.

        Persistence:
        :math:`\sum \alpha + \kappa \sum \gamma + \sum \beta`,
        with :math:`\kappa = \mathbb{E}[z^2 \mathbf{1}\{z<0\}]` from
        the fitted residual law.
        """
        self._require_fitted()
        wrapper = self._wrapper()
        kappa = wrapper.expected_z2_negative(self.residual_params)
        persistence = (
            jnp.sum(self.alpha)
            + kappa * jnp.sum(self.gamma)
            + jnp.sum(self.beta)
        )
        is_stat = persistence < 1.0
        denom = jnp.where(is_stat, 1.0 - persistence, _VAR_FLOOR)
        unconditional_variance = jnp.where(
            is_stat, self.omega / denom, jnp.inf,
        )
        log_pers = jnp.log(jnp.maximum(persistence, _VAR_FLOOR))
        half_life = jnp.where(
            jnp.logical_and(is_stat, persistence > 0.0),
            jnp.log(0.5) / log_pers,
            jnp.inf,
        )
        return {
            "unconditional_variance": unconditional_variance,
            "persistence": persistence,
            "kappa": kappa,
            "half_life": half_life,
            "is_stationary": is_stat,
        }

    # ------------------------------------------------------------------
    # ArmaGarch backend — GJR-specific overrides
    # ------------------------------------------------------------------
    def _ag_var_keys(self) -> tuple:
        return ("omega", "alpha", "gamma", "beta")

    def _ag_n_raw(self, wrapper: StandardisedResidual) -> int:
        # raw_omega + raw_persistence + raw_weights(2p + q)
        return 1 + 1 + 2 * self.p + self.q

    def _ag_pack_x0(
        self,
        var_params: dict,
        wrapper: StandardisedResidual,
        residual_params: dict,
    ) -> Array:
        omega = jnp.asarray(var_params["omega"], dtype=float).reshape(())
        alpha = jnp.asarray(var_params["alpha"], dtype=float).reshape(-1)
        gamma = jnp.asarray(var_params["gamma"], dtype=float).reshape(-1)
        beta = jnp.asarray(var_params["beta"], dtype=float).reshape(-1)
        raw_omega = positive_to_raw(jnp.maximum(omega, _SIGMA_FLOOR))
        kappa = wrapper.expected_z2_negative(residual_params)
        raw_persistence, raw_weights = gjr_unsimplex(alpha, gamma, beta, kappa)
        return jnp.concatenate(
            [raw_omega.reshape((1,)), raw_persistence.reshape((1,)), raw_weights]
        )

    def _ag_unpack_raw(
        self,
        raw_section: Array,
        wrapper: StandardisedResidual,
        residual_params: dict,
    ) -> dict:
        idx = 0
        raw_omega = raw_section[idx]
        idx += 1
        raw_persistence = raw_section[idx]
        idx += 1
        raw_weights = raw_section[idx : idx + 2 * self.p + self.q]
        kappa = wrapper.expected_z2_negative(residual_params)
        omega = raw_to_positive(raw_omega)
        alpha, gamma, beta = gjr_simplex(
            raw_persistence, raw_weights, p=self.p, q=self.q, kappa=kappa,
        )
        return {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta}

    def _ag_initial_state(
        self,
        eps_proxy: Array,
        mode: str,
        backcast_length: Optional[int],
        residual_params: dict,
    ) -> tuple:
        return self._initial_state_gjr(
            eps_proxy, mode=mode, backcast_length=backcast_length,
        )

    def _ag_run_recursion(
        self,
        eps_seq: Array,
        var_params: dict,
        residual_params: dict,
        init_state: tuple,
    ) -> tuple[Array, tuple]:
        omega = var_params["omega"]
        alpha = var_params["alpha"]
        gamma = var_params["gamma"]
        beta = var_params["beta"]
        eps_sq, neg_eps_sq, var_lags = init_state
        var_seq, terminal_state = run_gjr_garch(
            eps=eps_seq, omega=omega, alpha=alpha, gamma=gamma, beta=beta,
            init_eps_sq_lags=eps_sq,
            init_neg_eps_sq_lags=neg_eps_sq,
            init_var_lags=var_lags,
        )
        # Return as flat tuple matching the carry layout above.
        return var_seq, (terminal_state[0], terminal_state[1], terminal_state[2])

    def _ag_cold_start(
        self,
        eps_proxy: Array,
        mode: str,
        backcast_length: Optional[int],
        wrapper: StandardisedResidual,
    ) -> dict:
        # Reuse vanilla GARCH cold-start, then seed γ = 0.
        base = init_garch_params(
            eps_proxy, p=self.p, q=self.q, mode=mode,
            backcast_length=backcast_length,
        )
        return {
            "omega": base["omega"],
            "alpha": base["alpha"],
            "gamma": jnp.zeros((self.p,), dtype=float),
            "beta": base["beta"],
        }

    def _ag_forecast_step(
        self,
        var_params: dict,
        residual_params: dict,
        terminal_state: tuple,
    ) -> tuple[Array, tuple]:
        r"""Substitutes :math:`\mathbb{E}[\varepsilon^2_{\tau-i}
        \mathbf{1}\{\varepsilon_{\tau-i} < 0\}] = \kappa \cdot
        \mathbb{E}[\sigma^2_{\tau-i}]` for unobserved future."""
        wrapper = self._wrapper()
        kappa = wrapper.expected_z2_negative(residual_params)
        omega = var_params["omega"]
        alpha = var_params["alpha"]
        gamma = var_params["gamma"]
        beta = var_params["beta"]
        eps_sq_lags, neg_eps_sq_lags, var_lags = terminal_state
        ar_term = jnp.dot(alpha, eps_sq_lags) if self.p > 0 else 0.0
        asymm_term = jnp.dot(gamma, neg_eps_sq_lags) if self.p > 0 else 0.0
        ma_term = jnp.dot(beta, var_lags) if self.q > 0 else 0.0
        var_next = jnp.maximum(omega + ar_term + asymm_term + ma_term, _VAR_FLOOR)
        new_eps_sq = (
            jnp.concatenate([var_next.reshape((1,)), eps_sq_lags[:-1]])
            if self.p > 0 else eps_sq_lags
        )
        new_neg_eps_sq = (
            jnp.concatenate(
                [(kappa * var_next).reshape((1,)), neg_eps_sq_lags[:-1]]
            )
            if self.p > 0 else neg_eps_sq_lags
        )
        new_var_lags = (
            jnp.concatenate([var_next.reshape((1,)), var_lags[:-1]])
            if self.q > 0 else var_lags
        )
        return var_next, (new_eps_sq, new_neg_eps_sq, new_var_lags)

    def _ag_rvs_step(
        self,
        var_params: dict,
        residual_params: dict,
        terminal_state: tuple,
        eps_t: Array,
    ) -> tuple[Array, tuple]:
        omega = var_params["omega"]
        alpha = var_params["alpha"]
        gamma = var_params["gamma"]
        beta = var_params["beta"]
        eps_sq_lags, neg_eps_sq_lags, var_lags = terminal_state
        ar_term = jnp.dot(alpha, eps_sq_lags) if self.p > 0 else 0.0
        asymm_term = jnp.dot(gamma, neg_eps_sq_lags) if self.p > 0 else 0.0
        ma_term = jnp.dot(beta, var_lags) if self.q > 0 else 0.0
        var_t = jnp.maximum(omega + ar_term + asymm_term + ma_term, _VAR_FLOOR)
        eps_t_sq = eps_t * eps_t
        neg_eps_t_sq = jnp.where(eps_t < 0.0, eps_t_sq, 0.0)
        new_eps_sq = (
            jnp.concatenate([eps_t_sq.reshape((1,)), eps_sq_lags[:-1]])
            if self.p > 0 else eps_sq_lags
        )
        new_neg_eps_sq = (
            jnp.concatenate([neg_eps_t_sq.reshape((1,)), neg_eps_sq_lags[:-1]])
            if self.p > 0 else neg_eps_sq_lags
        )
        new_var_lags = (
            jnp.concatenate([var_t.reshape((1,)), var_lags[:-1]])
            if self.q > 0 else var_lags
        )
        return var_t, (new_eps_sq, new_neg_eps_sq, new_var_lags)

    def _ag_var_terminal_state_class(self) -> type:
        return GJRTerminalState
