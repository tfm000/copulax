r"""GARCH-M(p, q) — GARCH-in-mean (Engle, Lilien & Robins 1987).

Conditional variance enters the mean equation, capturing the
standard "risk premium" effect — higher conditional volatility
implies a higher conditional expected return:

.. math::

    y_t       &= \mu + \lambda_m\, \sigma^2_t + \varepsilon_t,\\
    \sigma^2_t &= \omega
                + \sum_{i=1}^p \alpha_i\, \varepsilon^2_{t-i}
                + \sum_{j=1}^q \beta_j\, \sigma^2_{t-j}.

The σ²-recursion is identical to vanilla GARCH; only the mean
equation differs.  Note that the input series is the **level
series** :math:`y_t`, not the pre-mean-corrected
:math:`\varepsilon_t` — innovation residuals are derived inside
the recursion as :math:`\varepsilon_t = y_t - \mu_t` after
σ² is computed from the carry.

Parameters: ``mu`` (intercept), ``lambda_m`` (variance-in-mean
coefficient), and the GARCH parameters ``omega``, ``alpha``,
``beta`` plus the residual-law shape parameters.

Stationarity / positivity: identical to vanilla GARCH on the
σ²-recursion side; ``μ`` and ``λ_m`` are unconstrained.

Reference:
    Engle, R., Lilien, D., & Robins, R. (1987).  *Estimating Time
    Varying Risk Premia in the Term Structure: The ARCH-M Model*.
    Econometrica, 55(2), 391-407.
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
from copulax._src.timeseries._init import garch_pre_sample_state
from copulax._src.timeseries._recursions import run_garch_m
from copulax._src.timeseries._residuals._standardise import StandardisedResidual
from copulax._src.timeseries._stationarity import (
    garch_simplex,
    garch_unsimplex,
    positive_to_raw,
    raw_to_positive,
)
from copulax._src.timeseries._variance._garch_base import (
    GARCHBase,
    GARCHTerminalState,
)


_VAR_FLOOR: float = 1e-12
_SIGMA_FLOOR: float = 1e-6


class GARCH_M(GARCHBase):
    r"""GARCH-in-mean(p, q).

    Construct with the desired orders and residual law:

    .. code-block:: python

        from copulax.timeseries import GARCH_M
        from copulax.univariate import normal
        fit = GARCH_M(p=1, q=1, residual_dist=normal).fit(y)

    Note:
        Unlike vanilla GARCH / IGARCH / GJR / EGARCH / TGARCH /
        QGARCH (which expect mean-corrected innovations as input),
        :meth:`fit` and the conditional-moment methods take the
        **level series** :math:`y_t` directly.  The mean
        :math:`\mu_t = \mu + \lambda_m \sigma^2_t` is fit jointly
        with the variance recursion.

    Inherits :meth:`forecast` / :meth:`stats` etc. from
    :class:`GARCHBase` (with overrides where the recursion shape
    differs).
    """

    mu: Optional[Array] = None
    lambda_m: Optional[Array] = None

    def __init__(
        self,
        p: int = 0,
        q: int = 0,
        *,
        residual_dist: Optional[Univariate] = None,
        name: str = "GARCH-M",
        mu=None,
        lambda_m=None,
        omega=None,
        alpha=None,
        beta=None,
        residual_params=None,
        terminal_state: Optional[GARCHTerminalState] = None,
        n_train_: Optional[int] = None,
        cov_matrix_=None,
        standard_errors_=None,
        residual_diagnostics_=None,
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
            n_train_=n_train_,
            cov_matrix_=cov_matrix_,
            standard_errors_=standard_errors_,
            residual_diagnostics_=residual_diagnostics_,
        )
        self.mu = (
            jnp.asarray(mu, dtype=float).reshape(())
            if mu is not None else None
        )
        self.lambda_m = (
            jnp.asarray(lambda_m, dtype=float).reshape(())
            if lambda_m is not None else None
        )

    @property
    def _stored_params(self) -> Optional[dict]:
        r"""Canonical params dict.

        ``{
            "mu":         (),
            "lambda_m":   (),
            "omega":      (),
            "alpha":      (p,),
            "beta":       (q,),
            "residual":   {<shape-only dict>},
        }``
        """
        if (
            self.mu is None or self.lambda_m is None
            or self.omega is None or self.alpha is None or self.beta is None
            or self.residual_params is None
        ):
            return None
        return {
            "mu": self.mu,
            "lambda_m": self.lambda_m,
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "residual": dict(self.residual_params),
        }

    @property
    def n_params(self) -> int:
        r"""Number of free fitted parameters: ω + α + β + μ + λ_m + residual."""
        wrapper = StandardisedResidual(self.residual_dist)
        return 1 + self.p + self.q + 2 + wrapper.n_shape_params

    # ------------------------------------------------------------------
    # Reparameterisation pack / unpack
    # ------------------------------------------------------------------
    def _pack_x0_garchm(
        self, params_dict: dict, wrapper: StandardisedResidual,
    ) -> Array:
        r"""Layout: ``[mu (1,), lambda_m (1,), raw_omega (1,),
        raw_persistence (1,), raw_weights (p+q,), raw_residual (n_shape,)]``.
        """
        mu = jnp.asarray(params_dict["mu"], dtype=float).reshape(())
        lambda_m = jnp.asarray(params_dict["lambda_m"], dtype=float).reshape(())
        omega = jnp.asarray(params_dict["omega"], dtype=float).reshape(())
        alpha = jnp.asarray(params_dict["alpha"], dtype=float).reshape(-1)
        beta = jnp.asarray(params_dict["beta"], dtype=float).reshape(-1)
        residual = params_dict.get("residual", {}) or {}

        raw_omega = positive_to_raw(jnp.maximum(omega, _SIGMA_FLOOR))
        raw_persistence, raw_weights = garch_unsimplex(alpha, beta)
        raw_residual = wrapper.shape_params_to_array(residual)
        return jnp.concatenate(
            [
                mu.reshape((1,)),
                lambda_m.reshape((1,)),
                raw_omega.reshape((1,)),
                raw_persistence.reshape((1,)),
                raw_weights,
                raw_residual,
            ]
        )

    def _unpack_raw_garchm(
        self, raw: Array, wrapper: StandardisedResidual,
    ) -> tuple[Array, Array, Array, Array, Array, dict]:
        r"""Returns ``(mu, lambda_m, omega, alpha, beta, residual_shape_dict)``."""
        idx = 0
        mu = raw[idx]
        idx += 1
        lambda_m = raw[idx]
        idx += 1
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
        return mu, lambda_m, omega, alpha, beta, residual

    # ------------------------------------------------------------------
    # Recursion + initial state
    # ------------------------------------------------------------------
    def _initial_state_garchm(
        self,
        y: Array,
        mode: str,
        backcast_length: Optional[int],
    ) -> tuple[Array, Array]:
        r"""GARCH-M shares the σ²-recursion's pre-sample state with
        vanilla GARCH (last p ε² + last q σ²).  Use the variance of
        the level series as the anchor — this is a slight bias
        relative to using the variance of the innovations
        :math:`\varepsilon_t = y_t - \mu_t` (which the fit objective
        actually consumes), but the bias washes out within the first
        ``max(p, q)`` recursion steps.
        """
        return garch_pre_sample_state(
            y, p=self.p, q=self.q,
            mode=mode, backcast_length=backcast_length,
        )

    def _run_recursion_garchm(
        self,
        y: Array,
        mu: Array,
        lambda_m: Array,
        omega: Array,
        alpha: Array,
        beta: Array,
        init_state: tuple[Array, Array],
    ) -> tuple[Array, Array, Array, GARCHTerminalState]:
        r"""Run GARCH-M; returns ``(mu_seq, eps_seq, var_seq, terminal_state)``."""
        eps_sq_lags, var_lags = init_state
        mu_seq, eps_seq, var_seq, terminal = run_garch_m(
            y=y, mu=mu, lambda_m=lambda_m,
            omega=omega, alpha=alpha, beta=beta,
            init_eps_sq_lags=eps_sq_lags,
            init_var_lags=var_lags,
        )
        return mu_seq, eps_seq, var_seq, GARCHTerminalState(
            eps_sq_lags=terminal[0], var_lags=terminal[1],
        )

    # ------------------------------------------------------------------
    # Cold-start
    # ------------------------------------------------------------------
    def _build_cold_start(
        self,
        y: Array,
        wrapper: StandardisedResidual,
        init: str,
        backcast_length: Optional[int],
    ) -> dict:
        r"""Cold-start: ``μ = mean(y)``, ``λ_m = 0`` (no risk premium
        prior), GARCH part as in vanilla."""
        base = super()._build_cold_start(
            y, wrapper, init=init, backcast_length=backcast_length,
        )
        base["mu"] = jnp.mean(y).reshape(())
        base["lambda_m"] = jnp.asarray(0.0, dtype=float)
        return base

    def _make_objective_garchm(self, wrapper: StandardisedResidual):
        def objective(
            raw: Array,
            y: Array,
            init_eps_sq_lags: Array,
            init_var_lags: Array,
        ) -> Array:
            mu, lambda_m, omega, alpha, beta, residual_shape = self._unpack_raw_garchm(
                raw, wrapper,
            )
            init_state = (init_eps_sq_lags, init_var_lags)
            _, eps_seq, var_seq, _ = self._run_recursion_garchm(
                y, mu, lambda_m, omega, alpha, beta, init_state,
            )
            sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
            z = eps_seq / sigma_seq
            logpdf = wrapper.logpdf(z, residual_shape) - jnp.log(sigma_seq)
            finite = jnp.isfinite(logpdf)
            safe_logpdf = jnp.where(finite, logpdf, 0.0)
            invalid_penalty = 1e6 * (~finite).mean()
            return -safe_logpdf.mean() + invalid_penalty
        return objective

    # ------------------------------------------------------------------
    # GARCH-M-specific natural-parameter NLL closures
    # ------------------------------------------------------------------
    def _natural_objective_closures(
        self,
        wrapper: StandardisedResidual,
        params_dict: dict,
        eps_arr: Array,
        init_state: tuple,
    ):
        r"""GARCH-M overrides the base because the standalone NLL has
        an in-mean term ``μ + λ·σ²`` so the innovations are computed
        *inside* the recursion (not pre-extracted as in the other
        variants).  The flat-parameter vector therefore needs ``mu``
        and ``lambda_m`` alongside the variance keys; the
        per-observation log-density depends on the recursion's eps
        output, not directly on ``eps_arr`` (here the input is
        actually the level series ``y``).
        """
        from copulax._src.timeseries._se import params_to_flat, flat_to_params
        _, schema = params_to_flat(params_dict)

        def per_obs_nll(flat: Array) -> Array:
            params = flat_to_params(flat, schema)
            mu = params["mu"]
            lambda_m = params["lambda_m"]
            omega = params["omega"]
            alpha = params["alpha"]
            beta = params["beta"]
            residual_ = params.get("residual", {}) or {}
            _, eps_seq, var_seq, _ = self._run_recursion_garchm(
                eps_arr, mu, lambda_m, omega, alpha, beta,
                init_state=init_state,
            )
            sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
            z = eps_seq / sigma_seq
            logpdf = wrapper.logpdf(z, residual_) - jnp.log(sigma_seq)
            return -jnp.where(jnp.isfinite(logpdf), logpdf, 0.0)

        def nll_total(flat: Array) -> Array:
            return jnp.sum(per_obs_nll(flat))

        return nll_total, per_obs_nll, schema

    def fit(
        self,
        y: ArrayLike,
        *,
        init: str = "analytical",
        init_params: Optional[dict] = None,
        backcast_length: Optional[int] = None,
        maxiter: int = 200,
        lr: float = 0.05,
        name: Optional[str] = None,
    ) -> "GARCH_M":
        r"""Fit GARCH-M(p, q) to a level return series ``y``."""
        self._check_method(init)
        wrapper = StandardisedResidual(self.residual_dist)
        y_arr = self._validate_series(y)
        n = int(y_arr.shape[0])
        self._validate_backcast_length(backcast_length, n)

        if init == "warm":
            if init_params is None:
                raise ValueError(
                    "init='warm' requires init_params (a parameter dict "
                    "matching the schema returned by `model.params`)."
                )
            cold = dict(init_params)
            for key in (
                "mu", "lambda_m", "omega", "alpha", "beta", "residual",
            ):
                if key not in cold:
                    raise KeyError(
                        f"Warm-start init_params missing required key {key!r}."
                    )
        else:
            cold = self._build_cold_start(
                y_arr, wrapper, init=init, backcast_length=backcast_length,
            )

        x0 = self._pack_x0_garchm(cold, wrapper)
        _state_mode = "sample" if init == "sample" else "backcast"
        init_eps_sq_lags, init_var_lags = self._initial_state_garchm(
            y_arr, mode=_state_mode, backcast_length=backcast_length,
        )

        objective = self._make_objective_garchm(wrapper)
        res = projected_gradient(
            f=objective,
            x0=x0,
            projection_method="projection_box",
            projection_options={
                "lower": jnp.full((x0.shape[0], 1), -jnp.inf),
                "upper": jnp.full((x0.shape[0], 1), jnp.inf),
            },
            y=y_arr,
            init_eps_sq_lags=init_eps_sq_lags,
            init_var_lags=init_var_lags,
            lr=lr,
            maxiter=maxiter,
        )
        x_opt = res["x"]
        mu, lambda_m, omega, alpha, beta, residual = self._unpack_raw_garchm(
            x_opt, wrapper,
        )

        _, eps_seq, var_seq, terminal = self._run_recursion_garchm(
            y_arr, mu, lambda_m, omega, alpha, beta,
            init_state=(init_eps_sq_lags, init_var_lags),
        )
        nll = objective(x_opt, y_arr, init_eps_sq_lags, init_var_lags)
        loglike = -nll * n
        n_params_total = 1 + self.p + self.q + 2 + wrapper.n_shape_params
        aic = 2.0 * n_params_total - 2.0 * loglike
        bic = (
            n_params_total * jnp.log(jnp.asarray(n, dtype=float))
            - 2.0 * loglike
        )

        sigma_train = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
        z_train = eps_seq / sigma_train
        params_dict = {
            "mu": mu, "lambda_m": lambda_m,
            "omega": omega, "alpha": alpha, "beta": beta,
            "residual": residual,
        }
        # GARCH-M's _natural_objective_closures override consumes the
        # level series y_arr, not eps; pass it through under the
        # ``eps_arr`` arg name (matching the base helper signature).
        cov, se_dict, diagnostics = self._post_fit_se_and_diagnostics(
            params_dict=params_dict,
            wrapper=wrapper, eps_arr=y_arr,
            init_state=(init_eps_sq_lags, init_var_lags),
            z_train=z_train,
            loglikelihood=loglike, aic=aic, bic=bic,
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
            mu=mu,
            lambda_m=lambda_m,
            omega=omega,
            alpha=alpha,
            beta=beta,
            residual_params=residual,
            terminal_state=terminal,
            n_train_=n,
            cov_matrix_=cov,
            standard_errors_=se_dict,
            residual_diagnostics_=diagnostics,
        )

    # ------------------------------------------------------------------
    # Conditional moments / residuals
    # ------------------------------------------------------------------
    def _garchm_recursion_inputs(
        self,
        y: ArrayLike,
        init: str,
        backcast_length: Optional[int],
    ) -> tuple[Array, tuple[Array, Array]]:
        y_arr = self._validate_series(y)
        n = int(y_arr.shape[0])
        self._validate_backcast_length(backcast_length, n)
        init_state = self._initial_state_garchm(
            y_arr, mode=init, backcast_length=backcast_length,
        )
        return y_arr, init_state

    def conditional_mean(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""``μ_t = μ + λ_m σ²_t`` over ``y``."""
        self._require_fitted()
        y_arr, init_state = self._garchm_recursion_inputs(
            y, init, backcast_length,
        )
        mu_seq, _, _, _ = self._run_recursion_garchm(
            y_arr, self.mu, self.lambda_m,
            self.omega, self.alpha, self.beta, init_state,
        )
        return mu_seq

    def conditional_variance(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        self._require_fitted()
        y_arr, init_state = self._garchm_recursion_inputs(
            y, init, backcast_length,
        )
        _, _, var_seq, _ = self._run_recursion_garchm(
            y_arr, self.mu, self.lambda_m,
            self.omega, self.alpha, self.beta, init_state,
        )
        return var_seq

    def residuals(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> dict:
        r"""Innovations and standardised residuals.

        Returns ``{"residuals": ε_t, "standardised_residuals": z_t}``
        where :math:`\varepsilon_t = y_t - \mu_t` and
        :math:`z_t = \varepsilon_t / \sigma_t`.  Uniform return
        shape with ARMA / GARCH / ArmaGarch.
        """
        self._require_fitted()
        y_arr, init_state = self._garchm_recursion_inputs(
            y, init, backcast_length,
        )
        _, eps_seq, var_seq, _ = self._run_recursion_garchm(
            y_arr, self.mu, self.lambda_m,
            self.omega, self.alpha, self.beta, init_state,
        )
        sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
        return {
            "residuals": eps_seq,
            "standardised_residuals": eps_seq / sigma_seq,
        }

    def terminal_state_from(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> GARCHTerminalState:
        self._require_fitted()
        y_arr, init_state = self._garchm_recursion_inputs(
            y, init, backcast_length,
        )
        _, _, _, terminal = self._run_recursion_garchm(
            y_arr, self.mu, self.lambda_m,
            self.omega, self.alpha, self.beta, init_state,
        )
        return terminal

    # ------------------------------------------------------------------
    # Loglikelihood
    # ------------------------------------------------------------------
    def _log_likelihood_on_series(
        self,
        y: ArrayLike,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        self._require_fitted()
        wrapper = self._wrapper()
        y_arr, init_state = self._garchm_recursion_inputs(
            y, init, backcast_length,
        )
        _, eps_seq, var_seq, _ = self._run_recursion_garchm(
            y_arr, self.mu, self.lambda_m,
            self.omega, self.alpha, self.beta, init_state,
        )
        sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
        z = eps_seq / sigma_seq
        logpdf = wrapper.logpdf(z, self.residual_params) - jnp.log(sigma_seq)
        return jnp.sum(logpdf)

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------
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

        Returns:
            ``{"mean": (h,) E[y], "variance": (h,) E[σ²], "paths": optional}``.

            Mean is :math:`\mathbb{E}[y_{t+\tau}] = \mu + \lambda_m
            \cdot \mathbb{E}[\sigma^2_{t+\tau}]` — non-zero unlike
            pure variance models.  Variance is the GARCH-style
            σ² forecast; future :math:`\varepsilon^2` are replaced
            by :math:`\mathbb{E}[\sigma^2]` per the standard
            stationarity argument.
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

        if method == "analytical":
            variance = self._analytical_forecast(h, state)
            mean = self.mu + self.lambda_m * variance
            return {"mean": mean, "variance": variance, "paths": None}

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
    # rvs roll-path — simulate y (level), not eps
    # ------------------------------------------------------------------
    def _roll_path(self, z: Array, state: GARCHTerminalState) -> Array:
        omega = self.omega
        alpha = self.alpha
        beta = self.beta
        mu = self.mu
        lambda_m = self.lambda_m

        def step(carry, z_t):
            eps_sq_lags, var_lags = carry
            ar_term = jnp.dot(alpha, eps_sq_lags) if self.p > 0 else 0.0
            ma_term = jnp.dot(beta, var_lags) if self.q > 0 else 0.0
            var_t = omega + ar_term + ma_term
            var_t = jnp.maximum(var_t, _VAR_FLOOR)
            mu_t = mu + lambda_m * var_t
            sigma_t = jnp.sqrt(var_t)
            eps_t = sigma_t * z_t
            y_t = mu_t + eps_t
            new_eps_sq = (
                jnp.concatenate([(eps_t * eps_t).reshape((1,)), eps_sq_lags[:-1]])
                if self.p > 0 else eps_sq_lags
            )
            new_var = (
                jnp.concatenate([var_t.reshape((1,)), var_lags[:-1]])
                if self.q > 0 else var_lags
            )
            return (new_eps_sq, new_var), y_t

        init_carry = (state.eps_sq_lags, state.var_lags)
        _, y_seq = jax.lax.scan(step, init_carry, z)
        return y_seq

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        r"""Analytic, parameter-only diagnostics for GARCH-M.

        Augments the vanilla GARCH stats with
        ``unconditional_mean = μ + λ_m · unconditional_variance``,
        the long-run risk-premium-implied mean.
        """
        self._require_fitted()
        base = super().stats()
        unconditional_mean = self.mu + self.lambda_m * base["unconditional_variance"]
        return {
            **base,
            "unconditional_mean": unconditional_mean,
        }

    @classmethod
    def _deserialise_extra_kwargs(cls, params: dict) -> dict:
        return {
            "mu": params.get("mu"),
            "lambda_m": params.get("lambda_m"),
        }
