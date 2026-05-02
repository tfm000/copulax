r"""EGARCH(p, q) — exponential GARCH (Nelson 1991).

The recursion is on log-variance, eliminating positivity constraints
on the parameters:

.. math::

    \log \sigma^2_t = \omega
                   + \sum_{i=1}^p \alpha_i\, (|z_{t-i}| - \mathbb{E}|z|)
                   + \sum_{i=1}^p \gamma_i\, z_{t-i}
                   + \sum_{j=1}^q \beta_j\, \log \sigma^2_{t-j},

where :math:`z_t = \varepsilon_t / \sigma_t` is the standardised
residual under the chosen unit-variance residual law.  Centring the
:math:`|z_{t-i}|` term by :math:`\mathbb{E}|z|` ensures the
unconditional log-variance equals
:math:`\omega / (1 - \sum \beta_j)`.

The :math:`\gamma_i z_{t-i}` term captures sign-dependent asymmetry
(the "leverage" effect — negative shocks raise volatility more than
equally-sized positive ones); :math:`\alpha_i` captures
shock-magnitude effects.

Stationarity reduces to the AR-polynomial-roots-outside-unit-circle
condition on :math:`\beta`; we enforce it via the same reflection-
coefficient reparameterisation used by the AR / MA mean models.
:math:`\omega`, :math:`\alpha`, and :math:`\gamma` are
unconstrained.

Reference:
    Nelson, D.B. (1991).  *Conditional heteroskedasticity in asset
    returns: a new approach*.  Econometrica, 59(2), 347-370.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src.optimize import projected_gradient
from copulax._src.timeseries._base import TerminalState
from copulax._src.timeseries._init import garch_pre_sample_state
from copulax._src.timeseries._recursions import run_egarch
from copulax._src.timeseries._residuals._standardise import StandardisedResidual
from copulax._src.timeseries._stationarity import (
    ar_to_raw,
    raw_to_ar,
)
from copulax._src.timeseries._variance._garch_base import GARCHBase


_VAR_FLOOR: float = 1e-12
_SIGMA_FLOOR: float = 1e-6


###############################################################################
# Per-family terminal state
###############################################################################
class EGARCHTerminalState(TerminalState):
    r"""Constant-size carry for EGARCH ``forecast(h)``.

    Stores the last ``p`` standardised residuals
    :math:`(z_{n}, \ldots, z_{n-p+1})` and the last ``q``
    log-variances :math:`(\log \sigma^2_n, \ldots, \log \sigma^2_{n-q+1})`
    — exactly what :func:`run_egarch` consumes as initial state.
    """
    z_lags: Array
    log_var_lags: Array


###############################################################################
# EGARCH
###############################################################################
class EGARCH(GARCHBase):
    r"""EGARCH(p, q) log-variance recursion (Nelson 1991).

    Construct with the desired orders and residual law:

    .. code-block:: python

        from copulax.timeseries import EGARCH
        from copulax.univariate import normal
        fit = EGARCH(p=1, q=1, residual_dist=normal).fit(eps)

    Inherits :meth:`residuals` / :meth:`stats` etc. from
    :class:`GARCHBase` (with overrides where the recursion shape
    differs).
    """

    gamma: Optional[Array] = None
    terminal_state: Optional[EGARCHTerminalState] = None

    def __init__(
        self,
        p: int = 0,
        q: int = 0,
        *,
        residual_dist: Optional[Univariate] = None,
        name: str = "EGARCH",
        omega=None,
        alpha=None,
        gamma=None,
        beta=None,
        residual_params=None,
        terminal_state: Optional[EGARCHTerminalState] = None,
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
        r"""Number of free fitted parameters: ω + α (p) + γ (p) + β (q) + residual."""
        wrapper = StandardisedResidual(self.residual_dist)
        return 1 + 2 * self.p + self.q + wrapper.n_shape_params

    # ------------------------------------------------------------------
    # Reparameterisation pack / unpack
    # ------------------------------------------------------------------
    def _pack_x0_egarch(
        self, params_dict: dict, wrapper: StandardisedResidual,
    ) -> Array:
        r"""Pack EGARCH params to flat raw vector.

        Layout: ``[omega (1,), alpha (p,), gamma (p,), raw_beta (q,),
        raw_residual_shape (n_shape,)]`` — ``omega``, ``alpha``,
        ``gamma`` are unconstrained (no positivity needed under log-
        variance form); ``beta`` uses the reflection-coefficient
        reparam to enforce stationarity.
        """
        omega = jnp.asarray(params_dict["omega"], dtype=float).reshape(())
        alpha = jnp.asarray(params_dict["alpha"], dtype=float).reshape(-1)
        gamma = jnp.asarray(params_dict["gamma"], dtype=float).reshape(-1)
        beta = jnp.asarray(params_dict["beta"], dtype=float).reshape(-1)
        residual = params_dict.get("residual", {}) or {}

        raw_beta = ar_to_raw(beta) if self.q > 0 else jnp.zeros((0,), dtype=float)
        raw_residual = wrapper.shape_params_to_array(residual)
        return jnp.concatenate(
            [omega.reshape((1,)), alpha, gamma, raw_beta, raw_residual]
        )

    def _unpack_raw_egarch(
        self, raw: Array, wrapper: StandardisedResidual,
    ) -> tuple[Array, Array, Array, Array, dict]:
        r"""Inverse of :meth:`_pack_x0_egarch`.

        Returns ``(omega, alpha, gamma, beta, residual_shape_dict)``.
        """
        idx = 0
        omega = raw[idx]
        idx += 1
        alpha = raw[idx : idx + self.p]
        idx += self.p
        gamma = raw[idx : idx + self.p]
        idx += self.p
        raw_beta = raw[idx : idx + self.q]
        idx += self.q
        raw_residual = raw[idx : idx + wrapper.n_shape_params]

        beta = raw_to_ar(raw_beta) if self.q > 0 else jnp.zeros((0,), dtype=float)
        residual = wrapper.shape_params_from_array(raw_residual)
        return omega, alpha, gamma, beta, residual

    # ------------------------------------------------------------------
    # Recursion + initial state
    # ------------------------------------------------------------------
    def _initial_state_egarch(
        self,
        eps: Array,
        mode: str,
        backcast_length: Optional[int],
    ) -> tuple[Array, Array]:
        r"""Pre-sample state for EGARCH: zero ``z`` lags + ``log
        var_anchor`` repeated for ``log σ²`` lags.

        We reuse the σ²-form backcast from
        :func:`garch_pre_sample_state` and convert to log-space.  For
        the standardised-residual lag buffer we use zeros (the
        recursion is mean-zero in ``z`` under correct
        specification).
        """
        eps_sq_lags, var_lags = garch_pre_sample_state(
            eps, p=self.p, q=self.q,
            mode=mode, backcast_length=backcast_length,
        )
        # log of the variance anchor — both var_lags entries are
        # equal (set to the same anchor in the helper).
        var_anchor = jnp.maximum(var_lags[0] if self.q > 0 else eps_sq_lags[0], _VAR_FLOOR)
        log_var_lags = jnp.full((self.q,), jnp.log(var_anchor))
        z_lags = jnp.zeros((self.p,), dtype=float)
        return z_lags, log_var_lags

    def _run_recursion_egarch(
        self,
        eps: Array,
        omega: Array,
        alpha: Array,
        gamma: Array,
        beta: Array,
        expected_abs_z: Array,
        init_state: tuple[Array, Array],
    ) -> tuple[Array, EGARCHTerminalState]:
        r"""Run EGARCH and return ``(σ²_seq, terminal_state)``.

        :func:`run_egarch` produces a log-variance sequence; we
        exponentiate to surface σ² in the standard interface.
        """
        z_lags, log_var_lags = init_state
        log_var_seq, terminal = run_egarch(
            eps=eps, omega=omega, alpha=alpha, gamma=gamma, beta=beta,
            expected_abs_z=expected_abs_z,
            init_z_lags=z_lags, init_log_var_lags=log_var_lags,
        )
        var_seq = jnp.exp(log_var_seq)
        return var_seq, EGARCHTerminalState(
            z_lags=terminal[0], log_var_lags=terminal[1],
        )

    # ------------------------------------------------------------------
    # Cold-start
    # ------------------------------------------------------------------
    def _build_cold_start(
        self,
        eps: Array,
        wrapper: StandardisedResidual,
        init: str,
        backcast_length: Optional[int],
    ) -> dict:
        r"""Cold-start params for EGARCH.

        ω is set to ``log(var(eps)) · (1 − 0.95)`` so the implied
        unconditional log-variance equals the sample log-variance
        under a default β=0.95 prior.  α and γ start at zero
        (symmetric / no-shock-effect baseline); β at 0.95.
        """
        sample_log_var = jnp.log(jnp.maximum(jnp.var(eps), _VAR_FLOOR))
        alpha = jnp.zeros((self.p,), dtype=float)
        gamma = jnp.zeros((self.p,), dtype=float)
        if self.q > 0:
            beta = jnp.full((self.q,), 0.95 / max(self.q, 1), dtype=float)
            persistence = jnp.sum(beta)
            omega = sample_log_var * (1.0 - persistence)
        else:
            beta = jnp.zeros((0,), dtype=float)
            omega = sample_log_var
        return {
            "omega": omega,
            "alpha": alpha,
            "gamma": gamma,
            "beta": beta,
            "residual": wrapper.example_shape_params(),
        }

    def _make_objective_egarch(self, wrapper: StandardisedResidual):
        def objective(
            raw: Array,
            eps: Array,
            init_z_lags: Array,
            init_log_var_lags: Array,
        ) -> Array:
            omega, alpha, gamma, beta, residual_shape = self._unpack_raw_egarch(
                raw, wrapper,
            )
            expected_abs_z = wrapper.expected_abs_z(residual_shape)
            var_seq, _ = self._run_recursion_egarch(
                eps, omega, alpha, gamma, beta, expected_abs_z,
                init_state=(init_z_lags, init_log_var_lags),
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
    ) -> "EGARCH":
        r"""Fit EGARCH(p, q) to a mean-corrected innovation series.

        Identical contract to :meth:`GARCHBase.fit`.
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

        x0 = self._pack_x0_egarch(cold, wrapper)
        _state_mode = "sample" if init == "sample" else "backcast"
        init_z_lags, init_log_var_lags = self._initial_state_egarch(
            eps_arr, mode=_state_mode, backcast_length=backcast_length,
        )

        objective = self._make_objective_egarch(wrapper)
        res = projected_gradient(
            f=objective,
            x0=x0,
            projection_method="projection_box",
            projection_options={
                "lower": jnp.full((x0.shape[0], 1), -jnp.inf),
                "upper": jnp.full((x0.shape[0], 1), jnp.inf),
            },
            eps=eps_arr,
            init_z_lags=init_z_lags,
            init_log_var_lags=init_log_var_lags,
            lr=lr,
            maxiter=maxiter,
        )
        x_opt = res["x"]
        omega, alpha, gamma, beta, residual = self._unpack_raw_egarch(x_opt, wrapper)

        expected_abs_z = wrapper.expected_abs_z(residual)
        _, terminal = self._run_recursion_egarch(
            eps_arr, omega, alpha, gamma, beta, expected_abs_z,
            init_state=(init_z_lags, init_log_var_lags),
        )
        nll = objective(x_opt, eps_arr, init_z_lags, init_log_var_lags)
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
    # Conditional variance / residuals
    # ------------------------------------------------------------------
    def _egarch_recursion_inputs(
        self,
        eps: ArrayLike,
        init: str,
        backcast_length: Optional[int],
    ) -> tuple[Array, tuple[Array, Array]]:
        eps_arr = self._validate_series(eps)
        n = int(eps_arr.shape[0])
        self._validate_backcast_length(backcast_length, n)
        init_state = self._initial_state_egarch(
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
        wrapper = self._wrapper()
        eps_arr, init_state = self._egarch_recursion_inputs(
            eps, init, backcast_length,
        )
        expected_abs_z = wrapper.expected_abs_z(self.residual_params)
        var_seq, _ = self._run_recursion_egarch(
            eps_arr, self.omega, self.alpha, self.gamma, self.beta,
            expected_abs_z, init_state,
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
        wrapper = self._wrapper()
        eps_arr, init_state = self._egarch_recursion_inputs(
            eps, init, backcast_length,
        )
        expected_abs_z = wrapper.expected_abs_z(self.residual_params)
        var_seq, _ = self._run_recursion_egarch(
            eps_arr, self.omega, self.alpha, self.gamma, self.beta,
            expected_abs_z, init_state,
        )
        sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
        return eps_arr, eps_arr / sigma_seq

    def terminal_state_from(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> EGARCHTerminalState:
        self._require_fitted()
        wrapper = self._wrapper()
        eps_arr, init_state = self._egarch_recursion_inputs(
            eps, init, backcast_length,
        )
        expected_abs_z = wrapper.expected_abs_z(self.residual_params)
        _, terminal = self._run_recursion_egarch(
            eps_arr, self.omega, self.alpha, self.gamma, self.beta,
            expected_abs_z, init_state,
        )
        return terminal

    # ------------------------------------------------------------------
    # Loglikelihood / aic / bic
    # ------------------------------------------------------------------
    def _log_likelihood_on_series(
        self,
        eps: ArrayLike,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        self._require_fitted()
        wrapper = self._wrapper()
        eps_arr, init_state = self._egarch_recursion_inputs(
            eps, init, backcast_length,
        )
        expected_abs_z = wrapper.expected_abs_z(self.residual_params)
        var_seq, _ = self._run_recursion_egarch(
            eps_arr, self.omega, self.alpha, self.gamma, self.beta,
            expected_abs_z, init_state,
        )
        sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
        z = eps_arr / sigma_seq
        logpdf = wrapper.logpdf(z, self.residual_params) - jnp.log(sigma_seq)
        return jnp.sum(logpdf)

    # ------------------------------------------------------------------
    # Forecast — h=1 closed form; h>=2 deferred to simulation
    # ------------------------------------------------------------------
    def forecast(
        self,
        h: int,
        *,
        method: str = "analytical",
        n_paths: int = 0,
        key: Optional[Array] = None,
        last_state: Optional[EGARCHTerminalState] = None,
    ) -> dict:
        r"""``h``-step-ahead conditional moments for EGARCH.

        Note:
            ``method="analytical"`` is supported only at ``h = 1``.
            For ``h >= 2`` the analytical forecast requires
            ``E[exp(α(|z| - E|z|) + γz)]``, which has a closed form
            only under normal residuals (Nelson 1991, eqn 2.16) and
            is not implemented in v1.  Use ``method="simulation"``
            for any horizon.

        Raises:
            ValueError: When ``method == "analytical"`` and ``h >= 2``,
                regardless of residual law.
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
            if h == 1:
                wrapper = self._wrapper()
                expected_abs_z = wrapper.expected_abs_z(self.residual_params)
                ar_term = (
                    jnp.dot(self.alpha, jnp.abs(state.z_lags) - expected_abs_z)
                    if self.p > 0 else 0.0
                )
                asymm_term = (
                    jnp.dot(self.gamma, state.z_lags)
                    if self.p > 0 else 0.0
                )
                ma_term = (
                    jnp.dot(self.beta, state.log_var_lags)
                    if self.q > 0 else 0.0
                )
                log_var_t1 = self.omega + ar_term + asymm_term + ma_term
                variance = jnp.exp(log_var_t1).reshape((1,))
                return {"mean": mean, "variance": variance, "paths": None}
            raise ValueError(
                "EGARCH analytical forecast for h>=2 requires the closed-form "
                "MGF of the residual law, which is implemented only under "
                "normal residuals (Nelson 1991, eqn 2.16) and is not yet "
                "provided in copulax.timeseries.  Use method='simulation' "
                "for h>=2."
            )

        elif method == "simulation":
            if n_paths <= 0:
                raise ValueError("method='simulation' requires n_paths > 0.")
            from copulax._src._utils import _resolve_key
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

    # ------------------------------------------------------------------
    # rvs roll-path
    # ------------------------------------------------------------------
    def _roll_path(self, z: Array, state: EGARCHTerminalState) -> Array:
        wrapper = self._wrapper()
        expected_abs_z = wrapper.expected_abs_z(self.residual_params)
        omega = self.omega
        alpha = self.alpha
        gamma = self.gamma
        beta = self.beta

        def step(carry, z_t):
            z_lags, log_var_lags = carry
            ar_term = (
                jnp.dot(alpha, jnp.abs(z_lags) - expected_abs_z)
                if self.p > 0 else 0.0
            )
            asymm_term = jnp.dot(gamma, z_lags) if self.p > 0 else 0.0
            ma_term = jnp.dot(beta, log_var_lags) if self.q > 0 else 0.0
            log_var_t = omega + ar_term + asymm_term + ma_term
            sigma_t = jnp.exp(0.5 * log_var_t)
            sigma_t = jnp.maximum(sigma_t, _SIGMA_FLOOR)
            eps_t = sigma_t * z_t
            new_z_lags = (
                jnp.concatenate([z_t.reshape((1,)), z_lags[:-1]])
                if self.p > 0 else z_lags
            )
            new_log_var_lags = (
                jnp.concatenate([log_var_t.reshape((1,)), log_var_lags[:-1]])
                if self.q > 0 else log_var_lags
            )
            return (new_z_lags, new_log_var_lags), eps_t

        init_carry = (state.z_lags, state.log_var_lags)
        _, eps_seq = jax.lax.scan(step, init_carry, z)
        return eps_seq

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        r"""Analytic, parameter-only diagnostics for EGARCH.

        Persistence under the log-variance recursion is
        :math:`\sum \beta_j` (the :math:`\alpha`, :math:`\gamma`
        terms are mean-zero contributions).  Stationarity requires
        the AR polynomial in the lag operator on
        :math:`\log \sigma^2` to have all roots outside the unit
        circle — for ``q = 1`` this reduces to :math:`|\beta| < 1`.
        """
        self._require_fitted()
        from copulax._src.timeseries._stationarity import ar_is_stationary
        persistence = jnp.sum(self.beta) if self.q > 0 else jnp.asarray(0.0)
        is_stat = (
            ar_is_stationary(self.beta) if self.q > 0 else jnp.asarray(True)
        )
        # Unconditional log-variance for q=1 simplification
        denom = jnp.where(jnp.abs(1.0 - persistence) < 1e-12, 1e-12, 1.0 - persistence)
        unconditional_log_variance = jnp.where(
            is_stat, self.omega / denom, jnp.inf,
        )
        unconditional_variance = jnp.where(
            is_stat, jnp.exp(unconditional_log_variance), jnp.inf,
        )
        log_pers = jnp.log(jnp.maximum(jnp.abs(persistence), _VAR_FLOOR))
        half_life = jnp.where(
            jnp.logical_and(is_stat, jnp.abs(persistence) > 0.0),
            jnp.log(0.5) / log_pers,
            jnp.inf,
        )
        return {
            "unconditional_variance": unconditional_variance,
            "unconditional_log_variance": unconditional_log_variance,
            "persistence": persistence,
            "half_life": half_life,
            "is_stationary": is_stat,
        }

    # ------------------------------------------------------------------
    # ArmaGarch backend — EGARCH-specific overrides
    # ------------------------------------------------------------------
    def _ag_var_keys(self) -> tuple:
        return ("omega", "alpha", "gamma", "beta")

    def _ag_n_raw(self, wrapper: StandardisedResidual) -> int:
        # omega(1) + alpha(p) + gamma(p) + raw_beta(q) — log-form has
        # no positivity / persistence simplex.
        return 1 + self.p + self.p + self.q

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
        raw_beta = (
            ar_to_raw(beta) if self.q > 0 else jnp.zeros((0,), dtype=float)
        )
        return jnp.concatenate([omega.reshape((1,)), alpha, gamma, raw_beta])

    def _ag_unpack_raw(
        self,
        raw_section: Array,
        wrapper: StandardisedResidual,
        residual_params: dict,
    ) -> dict:
        idx = 0
        omega = raw_section[idx]
        idx += 1
        alpha = raw_section[idx : idx + self.p]
        idx += self.p
        gamma = raw_section[idx : idx + self.p]
        idx += self.p
        raw_beta = raw_section[idx : idx + self.q]
        beta = (
            raw_to_ar(raw_beta) if self.q > 0
            else jnp.zeros((0,), dtype=float)
        )
        return {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta}

    def _ag_initial_state(
        self,
        eps_proxy: Array,
        mode: str,
        backcast_length: Optional[int],
        residual_params: dict,
    ) -> tuple:
        return self._initial_state_egarch(
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
        z_lags, log_var_lags = init_state
        expected_abs_z = self._wrapper().expected_abs_z(residual_params)
        log_var_seq, terminal = run_egarch(
            eps=eps_seq, omega=omega, alpha=alpha, gamma=gamma, beta=beta,
            expected_abs_z=expected_abs_z,
            init_z_lags=z_lags, init_log_var_lags=log_var_lags,
        )
        var_seq = jnp.exp(log_var_seq)
        return var_seq, (terminal[0], terminal[1])

    def _ag_cold_start(
        self,
        eps_proxy: Array,
        mode: str,
        backcast_length: Optional[int],
        wrapper: StandardisedResidual,
    ) -> dict:
        base = self._build_cold_start(
            eps_proxy, wrapper, init=mode, backcast_length=backcast_length,
        )
        return {k: v for k, v in base.items() if k != "residual"}

    def _ag_forecast_step(
        self,
        var_params: dict,
        residual_params: dict,
        terminal_state: tuple,
    ) -> tuple[Array, tuple]:
        r"""1-step closed-form log-variance forecast.

        At ``h ≥ 2`` the recursion requires the MGF
        :math:`\mathbb{E}[\exp(\alpha(|z| - \mathbb{E}|z|) +
        \gamma z)]`, which has no closed form for non-Normal
        residuals (Nelson 1991, eqn 2.16).  ArmaGarch detects this
        via :meth:`_ag_supports_analytical_h_step` and routes
        ``h ≥ 2`` to simulation.
        """
        wrapper = self._wrapper()
        expected_abs_z = wrapper.expected_abs_z(residual_params)
        omega = var_params["omega"]
        alpha = var_params["alpha"]
        gamma = var_params["gamma"]
        beta = var_params["beta"]
        z_lags, log_var_lags = terminal_state
        ar_term = (
            jnp.dot(alpha, jnp.abs(z_lags) - expected_abs_z)
            if self.p > 0 else 0.0
        )
        asymm_term = jnp.dot(gamma, z_lags) if self.p > 0 else 0.0
        ma_term = jnp.dot(beta, log_var_lags) if self.q > 0 else 0.0
        log_var_next = omega + ar_term + asymm_term + ma_term
        var_next = jnp.exp(log_var_next)
        # New z_lag is 0 (E[z_{t+1}] = 0); new log_var_lag is the
        # 1-step value.  These updates are correct only at h=1.
        new_z_lags = (
            jnp.concatenate([jnp.zeros((1,), dtype=float), z_lags[:-1]])
            if self.p > 0 else z_lags
        )
        new_log_var_lags = (
            jnp.concatenate([log_var_next.reshape((1,)), log_var_lags[:-1]])
            if self.q > 0 else log_var_lags
        )
        return var_next, (new_z_lags, new_log_var_lags)

    def _ag_rvs_step(
        self,
        var_params: dict,
        residual_params: dict,
        terminal_state: tuple,
        eps_t: Array,
    ) -> tuple[Array, tuple]:
        wrapper = self._wrapper()
        expected_abs_z = wrapper.expected_abs_z(residual_params)
        omega = var_params["omega"]
        alpha = var_params["alpha"]
        gamma = var_params["gamma"]
        beta = var_params["beta"]
        z_lags, log_var_lags = terminal_state
        ar_term = (
            jnp.dot(alpha, jnp.abs(z_lags) - expected_abs_z)
            if self.p > 0 else 0.0
        )
        asymm_term = jnp.dot(gamma, z_lags) if self.p > 0 else 0.0
        ma_term = jnp.dot(beta, log_var_lags) if self.q > 0 else 0.0
        log_var_t = omega + ar_term + asymm_term + ma_term
        sigma_t = jnp.maximum(jnp.exp(0.5 * log_var_t), _SIGMA_FLOOR)
        var_t = sigma_t ** 2
        # New z = eps_t / σ_t; new log_var = log_var_t.
        z_t = eps_t / sigma_t
        new_z_lags = (
            jnp.concatenate([z_t.reshape((1,)), z_lags[:-1]])
            if self.p > 0 else z_lags
        )
        new_log_var_lags = (
            jnp.concatenate([log_var_t.reshape((1,)), log_var_lags[:-1]])
            if self.q > 0 else log_var_lags
        )
        return var_t, (new_z_lags, new_log_var_lags)

    @staticmethod
    def _ag_supports_analytical_h_step() -> bool:
        return False  # Only h=1 closed form; h≥2 needs MGF.

    def _ag_var_terminal_state_class(self) -> type:
        return EGARCHTerminalState
