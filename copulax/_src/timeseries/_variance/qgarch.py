r"""QGARCH(1, q) — quadratic ARCH (Sentana 1995).

Adds a linear-in-ε asymmetry term to the GARCH(1, q) recursion:

.. math::

    \sigma^2_t = \omega
               + \alpha\, \varepsilon^2_{t-1}
               + \psi\, \varepsilon_{t-1}
               + \sum_{j=1}^q \beta_j\, \sigma^2_{t-j}.

The :math:`\psi` term picks up sign-dependent asymmetry while
leaving the unconditional variance unchanged
(:math:`\mathbb{E}[\psi \varepsilon] = 0` under any zero-mean
residual law), so QGARCH is *not* an asymmetric persistence model
in the same sense as GJR-GARCH or EGARCH — it's a richer
asymmetric *level* model.

Positivity of :math:`\sigma^2_t` requires :math:`\omega \ge
\psi^2 / (4 \alpha)` (the discriminant of the quadratic
:math:`\omega + \alpha \varepsilon^2 + \psi \varepsilon` in
:math:`\varepsilon`).  We enforce this by parameterising

.. math::

    \omega = \psi^2 / (4 \alpha) + \mathrm{softplus}(\mathrm{raw}_\omega),

so positivity is structural; no projection needed.

Stationarity is the standard :math:`\alpha + \sum \beta_j < 1`
(unaffected by :math:`\psi` since :math:`\mathbb{E}[\psi
\varepsilon] = 0`); enforced via the same
:func:`copulax._src.timeseries._stationarity.garch_simplex`
reparameterisation as vanilla GARCH.

**v1 restricts to p = 1.**  Per Sentana (1995) and the plan
§"Stationarity / positivity / sign constraints",
:math:`\sigma^2_t > 0` for :math:`p \ge 2` is a *matrix*
positive-semidefiniteness condition on an augmented
:math:`(p+1) \times (p+1)` matrix — implementing a Cholesky-style
reparameterisation of that matrix is non-trivial and rarely
needed in practice (most empirical QGARCH papers use ``(1, 1)``).
The constructor raises :class:`ValueError` on ``p >= 2``.

**Identifiability note:** :math:`\psi` and the residual-law skew
parameter are weakly co-identified: under symmetric residuals
:math:`\psi` carries the asymmetry alone; under skewed residuals
both can carry it.  Tests assert sensible parameter recovery on
simulated data with both ``ψ ≠ 0, skew = 0`` and ``ψ = 0,
skew ≠ 0`` configurations.

Reference:
    Sentana, E. (1995).  *Quadratic ARCH Models*.  Review of
    Economic Studies, 62(4), 639-661.
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
from copulax._src.timeseries._init import (
    garch_pre_sample_state,
    init_garch_params,
)
from copulax._src.timeseries._recursions import run_qgarch
from copulax._src.timeseries._residuals._standardise import StandardisedResidual
from copulax._src.timeseries._stationarity import (
    garch_simplex,
    garch_unsimplex,
    positive_to_raw,
    raw_to_positive,
)
from copulax._src.timeseries._variance._garch_base import GARCHBase


_VAR_FLOOR: float = 1e-12
_SIGMA_FLOOR: float = 1e-6


class QGARCHTerminalState(TerminalState):
    r"""Constant-size carry for QGARCH ``forecast(h)``.

    Stores ``ε_{t-1}`` (a single scalar — p=1 only), the lagged
    squared innovation ``ε²_{t-1}``, and the last ``q`` conditional
    variances.
    """
    eps_lags: Array       # shape (1,)
    eps_sq_lags: Array    # shape (1,)
    var_lags: Array       # shape (q,)


class QGARCH(GARCHBase):
    r"""QGARCH(1, q) quadratic-asymmetry σ²-recursion (Sentana 1995).

    Construct with ``p = 1`` and the desired ``q`` and residual law:

    .. code-block:: python

        from copulax.timeseries import QGARCH
        from copulax.univariate import skewed_t
        fit = QGARCH(p=1, q=1, residual_dist=skewed_t).fit(eps)

    Raises:
        ValueError: When ``p`` is not exactly 1 — see module
            docstring for the matrix-PSD reasoning.
    """

    psi: Optional[Array] = None
    terminal_state: Optional[QGARCHTerminalState] = None

    def __init__(
        self,
        p: int = 0,
        q: int = 0,
        *,
        residual_dist: Optional[Univariate] = None,
        name: str = "QGARCH",
        omega=None,
        alpha=None,
        psi=None,
        beta=None,
        residual_params=None,
        terminal_state: Optional[QGARCHTerminalState] = None,
        n_train_: Optional[int] = None,
        cov_matrix_=None,
        standard_errors_=None,
        residual_diagnostics_=None,
    ):
        if int(p) != 1:
            raise ValueError(
                f"QGARCH requires p=1; got p={int(p)}.  "
                "p>=2 needs a Cholesky-style matrix-PSD reparam "
                "(Sentana 1995) and is deferred to a future release."
            )
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
        self.psi = (
            jnp.asarray(psi, dtype=float).reshape(-1)
            if psi is not None else None
        )

    @property
    def _stored_params(self) -> Optional[dict]:
        r"""Canonical params dict.

        ``{
            "omega":     (),
            "alpha":     (1,),
            "psi":       (1,),
            "beta":      (q,),
            "residual":  {<shape-only dict>},
        }``
        """
        if (
            self.omega is None or self.alpha is None or self.psi is None
            or self.beta is None or self.residual_params is None
        ):
            return None
        return {
            "omega": self.omega,
            "alpha": self.alpha,
            "psi": self.psi,
            "beta": self.beta,
            "residual": dict(self.residual_params),
        }

    @property
    def n_params(self) -> int:
        wrapper = StandardisedResidual(self.residual_dist)
        return 1 + 1 + 1 + self.q + wrapper.n_shape_params

    # ------------------------------------------------------------------
    # Reparameterisation pack / unpack
    # ------------------------------------------------------------------
    def _pack_x0_qgarch(
        self, params_dict: dict, wrapper: StandardisedResidual,
    ) -> Array:
        r"""Layout: ``[raw_omega_minus (1,), raw_persistence (1,),
        raw_weights (1+q,), psi (1,), raw_residual_shape (n_shape,)]``,
        where ``raw_omega_minus`` parameterises ``ω' = ω -
        ψ²/(4α)`` via softplus.
        """
        omega = jnp.asarray(params_dict["omega"], dtype=float).reshape(())
        alpha = jnp.asarray(params_dict["alpha"], dtype=float).reshape(-1)
        psi = jnp.asarray(params_dict["psi"], dtype=float).reshape(-1)
        beta = jnp.asarray(params_dict["beta"], dtype=float).reshape(-1)
        residual = params_dict.get("residual", {}) or {}

        psi_sq_over_4alpha = (psi[0] ** 2) / (4.0 * jnp.maximum(alpha[0], _SIGMA_FLOOR))
        omega_minus = jnp.maximum(omega - psi_sq_over_4alpha, _SIGMA_FLOOR)
        raw_omega_minus = positive_to_raw(omega_minus)

        raw_persistence, raw_weights = garch_unsimplex(alpha, beta)
        raw_residual = wrapper.shape_params_to_array(residual)
        return jnp.concatenate(
            [
                raw_omega_minus.reshape((1,)),
                raw_persistence.reshape((1,)),
                raw_weights,
                psi,
                raw_residual,
            ]
        )

    def _unpack_raw_qgarch(
        self, raw: Array, wrapper: StandardisedResidual,
    ) -> tuple[Array, Array, Array, Array, dict]:
        r"""Returns ``(omega, alpha, psi, beta, residual_shape_dict)``.

        ``ω = ψ²/(4α) + softplus(raw_omega_minus)`` enforces
        positivity of σ²_t structurally.
        """
        idx = 0
        raw_omega_minus = raw[idx]
        idx += 1
        raw_persistence = raw[idx]
        idx += 1
        raw_weights = raw[idx : idx + 1 + self.q]
        idx += 1 + self.q
        psi = raw[idx : idx + 1]
        idx += 1
        raw_residual = raw[idx : idx + wrapper.n_shape_params]

        residual = wrapper.shape_params_from_array(raw_residual)
        alpha, beta = garch_simplex(raw_persistence, raw_weights, p=1)
        alpha_safe = jnp.maximum(alpha[0], _SIGMA_FLOOR)
        omega_minus = raw_to_positive(raw_omega_minus)
        omega = omega_minus + (psi[0] ** 2) / (4.0 * alpha_safe)
        return omega, alpha, psi, beta, residual

    # ------------------------------------------------------------------
    # Recursion + initial state
    # ------------------------------------------------------------------
    def _initial_state_qgarch(
        self,
        eps: Array,
        mode: str,
        backcast_length: Optional[int],
    ) -> tuple[Array, Array, Array]:
        eps_sq_lags, var_lags = garch_pre_sample_state(
            eps, p=1, q=self.q,
            mode=mode, backcast_length=backcast_length,
        )
        # ε_{t-1} pre-sample lag: zero (the leading-window sample mean
        # of mean-corrected innovations is zero by construction).
        eps_lags = jnp.zeros((1,), dtype=float)
        return eps_lags, eps_sq_lags, var_lags

    def _run_recursion_qgarch(
        self,
        eps: Array,
        omega: Array,
        alpha: Array,
        psi: Array,
        beta: Array,
        init_state: tuple[Array, Array, Array],
    ) -> tuple[Array, QGARCHTerminalState]:
        eps_lags, eps_sq_lags, var_lags = init_state
        var_seq, terminal = run_qgarch(
            eps=eps, omega=omega, alpha=alpha, psi=psi, beta=beta,
            init_eps_lags=eps_lags,
            init_eps_sq_lags=eps_sq_lags,
            init_var_lags=var_lags,
        )
        return var_seq, QGARCHTerminalState(
            eps_lags=terminal[0],
            eps_sq_lags=terminal[1],
            var_lags=terminal[2],
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
        r"""Cold-start: vanilla GARCH(1, q) starting point with ``ψ = 0``."""
        base = super()._build_cold_start(
            eps, wrapper, init=init, backcast_length=backcast_length,
        )
        base["psi"] = jnp.zeros((1,), dtype=float)
        return base

    def _make_objective_qgarch(self, wrapper: StandardisedResidual):
        def objective(
            raw: Array,
            eps: Array,
            init_eps_lags: Array,
            init_eps_sq_lags: Array,
            init_var_lags: Array,
        ) -> Array:
            omega, alpha, psi, beta, residual_shape = self._unpack_raw_qgarch(
                raw, wrapper,
            )
            init_state = (init_eps_lags, init_eps_sq_lags, init_var_lags)
            var_seq, _ = self._run_recursion_qgarch(
                eps, omega, alpha, psi, beta, init_state,
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
    ) -> "QGARCH":
        r"""Fit QGARCH(1, q) to a mean-corrected innovation series."""
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
            for key in ("omega", "alpha", "psi", "beta", "residual"):
                if key not in cold:
                    raise KeyError(
                        f"Warm-start init_params missing required key {key!r}."
                    )
        else:
            cold = self._build_cold_start(
                eps_arr, wrapper, init=init, backcast_length=backcast_length,
            )

        x0 = self._pack_x0_qgarch(cold, wrapper)
        _state_mode = "sample" if init == "sample" else "backcast"
        init_eps, init_eps_sq, init_var = self._initial_state_qgarch(
            eps_arr, mode=_state_mode, backcast_length=backcast_length,
        )

        objective = self._make_objective_qgarch(wrapper)
        res = projected_gradient(
            f=objective,
            x0=x0,
            projection_method="projection_box",
            projection_options={
                "lower": jnp.full((x0.shape[0], 1), -jnp.inf),
                "upper": jnp.full((x0.shape[0], 1), jnp.inf),
            },
            eps=eps_arr,
            init_eps_lags=init_eps,
            init_eps_sq_lags=init_eps_sq,
            init_var_lags=init_var,
            lr=lr,
            maxiter=maxiter,
        )
        x_opt = res["x"]
        omega, alpha, psi, beta, residual = self._unpack_raw_qgarch(
            x_opt, wrapper,
        )

        var_seq, terminal = self._run_recursion_qgarch(
            eps_arr, omega, alpha, psi, beta,
            init_state=(init_eps, init_eps_sq, init_var),
        )
        nll = objective(x_opt, eps_arr, init_eps, init_eps_sq, init_var)
        loglike = -nll * n
        n_params_total = 1 + 1 + 1 + self.q + wrapper.n_shape_params
        aic = 2.0 * n_params_total - 2.0 * loglike
        bic = (
            n_params_total * jnp.log(jnp.asarray(n, dtype=float))
            - 2.0 * loglike
        )

        sigma_train = jnp.sqrt(jnp.maximum(var_seq, 1e-12))
        z_train = eps_arr / sigma_train
        params_dict = {
            "omega": omega, "alpha": alpha, "psi": psi, "beta": beta,
            "residual": residual,
        }
        cov, se_dict, diagnostics = self._post_fit_se_and_diagnostics(
            params_dict=params_dict,
            wrapper=wrapper, eps_arr=eps_arr,
            init_state=(init_eps, init_eps_sq, init_var),
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
            omega=omega,
            alpha=alpha,
            psi=psi,
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
    def _qgarch_recursion_inputs(
        self,
        eps: ArrayLike,
        init: str,
        backcast_length: Optional[int],
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        eps_arr = self._validate_series(eps)
        n = int(eps_arr.shape[0])
        self._validate_backcast_length(backcast_length, n)
        init_state = self._initial_state_qgarch(
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
        eps_arr, init_state = self._qgarch_recursion_inputs(
            eps, init, backcast_length,
        )
        var_seq, _ = self._run_recursion_qgarch(
            eps_arr, self.omega, self.alpha, self.psi, self.beta,
            init_state,
        )
        return var_seq

    def residuals(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> dict:
        self._require_fitted()
        eps_arr, init_state = self._qgarch_recursion_inputs(
            eps, init, backcast_length,
        )
        var_seq, _ = self._run_recursion_qgarch(
            eps_arr, self.omega, self.alpha, self.psi, self.beta,
            init_state,
        )
        sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
        return {
            "residuals": eps_arr,
            "standardised_residuals": eps_arr / sigma_seq,
        }

    def terminal_state_from(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> QGARCHTerminalState:
        self._require_fitted()
        eps_arr, init_state = self._qgarch_recursion_inputs(
            eps, init, backcast_length,
        )
        _, terminal = self._run_recursion_qgarch(
            eps_arr, self.omega, self.alpha, self.psi, self.beta,
            init_state,
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
        eps_arr, init_state = self._qgarch_recursion_inputs(
            eps, init, backcast_length,
        )
        var_seq, _ = self._run_recursion_qgarch(
            eps_arr, self.omega, self.alpha, self.psi, self.beta,
            init_state,
        )
        sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
        z = eps_arr / sigma_seq
        logpdf = wrapper.logpdf(z, self.residual_params) - jnp.log(sigma_seq)
        return jnp.sum(logpdf)

    # ------------------------------------------------------------------
    # Forecast — analytical recursion: E[ψ·ε_{τ-1}] = 0 for unobserved future
    # ------------------------------------------------------------------
    def _analytical_forecast(self, h: int, state: QGARCHTerminalState) -> Array:
        var_path = []
        eps_lags = state.eps_lags
        eps_sq_lags = state.eps_sq_lags
        var_lags = state.var_lags
        for step_idx in range(h):
            ar_term = self.alpha[0] * eps_sq_lags[0]
            psi_term = self.psi[0] * eps_lags[0]
            ma_term = jnp.dot(self.beta, var_lags) if self.q > 0 else 0.0
            var_t = self.omega + ar_term + psi_term + ma_term
            var_t = jnp.maximum(var_t, _VAR_FLOOR)
            var_path.append(var_t)
            # Substitute E[ε_τ] = 0, E[ε²_τ] = E[σ²_τ] for unobserved
            # future shocks (Sentana 1995, eqn 4.4).
            eps_lags = jnp.zeros((1,), dtype=float)
            eps_sq_lags = var_t.reshape((1,))
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
        last_state: Optional[QGARCHTerminalState] = None,
    ) -> dict:
        r"""``h``-step-ahead conditional moments.

        QGARCH supports analytical h-step forecasting at every
        horizon: under stationarity the expected ``ψ·ε`` term
        vanishes for unobserved future shocks
        (:math:`\mathbb{E}[\varepsilon_\tau] = 0`), so the σ²
        recursion collapses to vanilla-GARCH form for the forecast.
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
    def _roll_path(self, z: Array, state: QGARCHTerminalState) -> Array:
        omega = self.omega
        alpha = self.alpha
        psi = self.psi
        beta = self.beta

        def step(carry, z_t):
            eps_lags, eps_sq_lags, var_lags = carry
            ar_term = alpha[0] * eps_sq_lags[0]
            psi_term = psi[0] * eps_lags[0]
            ma_term = jnp.dot(beta, var_lags) if self.q > 0 else 0.0
            var_t = omega + ar_term + psi_term + ma_term
            var_t = jnp.maximum(var_t, _VAR_FLOOR)
            sigma_t = jnp.sqrt(var_t)
            eps_t = sigma_t * z_t
            new_eps_lags = eps_t.reshape((1,))
            new_eps_sq_lags = (eps_t * eps_t).reshape((1,))
            new_var_lags = (
                jnp.concatenate([var_t.reshape((1,)), var_lags[:-1]])
                if self.q > 0 else var_lags
            )
            return (new_eps_lags, new_eps_sq_lags, new_var_lags), eps_t

        init_carry = (state.eps_lags, state.eps_sq_lags, state.var_lags)
        _, eps_seq = jax.lax.scan(step, init_carry, z)
        return eps_seq

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        r"""Analytic, parameter-only diagnostics for QGARCH(1, q).

        Persistence: :math:`\alpha + \sum \beta_j` (the :math:`\psi`
        term is mean-zero and does not contribute).  Unconditional
        variance: :math:`\omega / (1 - \text{persistence})` — also
        unaffected by :math:`\psi`.
        """
        self._require_fitted()
        persistence = self.alpha[0] + jnp.sum(self.beta)
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
            "half_life": half_life,
            "is_stationary": is_stat,
        }

    # ------------------------------------------------------------------
    # ArmaGarch backend — QGARCH-specific overrides
    # ------------------------------------------------------------------
    def _ag_var_keys(self) -> tuple:
        return ("omega", "alpha", "psi", "beta")

    def _ag_n_raw(self, wrapper: StandardisedResidual) -> int:
        # raw_omega_minus(1) + raw_persistence(1) + raw_weights(1+q)
        # + psi(1)
        return 1 + 1 + (1 + self.q) + 1

    def _ag_pack_x0(
        self,
        var_params: dict,
        wrapper: StandardisedResidual,
        residual_params: dict,
    ) -> Array:
        omega = jnp.asarray(var_params["omega"], dtype=float).reshape(())
        alpha = jnp.asarray(var_params["alpha"], dtype=float).reshape(-1)
        psi = jnp.asarray(var_params["psi"], dtype=float).reshape(-1)
        beta = jnp.asarray(var_params["beta"], dtype=float).reshape(-1)
        psi_sq_over_4alpha = (
            (psi[0] ** 2) / (4.0 * jnp.maximum(alpha[0], _SIGMA_FLOOR))
        )
        omega_minus = jnp.maximum(omega - psi_sq_over_4alpha, _SIGMA_FLOOR)
        raw_omega_minus = positive_to_raw(omega_minus)
        raw_persistence, raw_weights = garch_unsimplex(alpha, beta)
        return jnp.concatenate(
            [
                raw_omega_minus.reshape((1,)),
                raw_persistence.reshape((1,)),
                raw_weights,
                psi,
            ]
        )

    def _ag_unpack_raw(
        self,
        raw_section: Array,
        wrapper: StandardisedResidual,
        residual_params: dict,
    ) -> dict:
        idx = 0
        raw_omega_minus = raw_section[idx]
        idx += 1
        raw_persistence = raw_section[idx]
        idx += 1
        raw_weights = raw_section[idx : idx + 1 + self.q]
        idx += 1 + self.q
        psi = raw_section[idx : idx + 1]
        alpha, beta = garch_simplex(raw_persistence, raw_weights, p=1)
        alpha_safe = jnp.maximum(alpha[0], _SIGMA_FLOOR)
        omega_minus = raw_to_positive(raw_omega_minus)
        omega = omega_minus + (psi[0] ** 2) / (4.0 * alpha_safe)
        return {"omega": omega, "alpha": alpha, "psi": psi, "beta": beta}

    def _ag_initial_state(
        self,
        eps_proxy: Array,
        mode: str,
        backcast_length: Optional[int],
        residual_params: dict,
    ) -> tuple:
        return self._initial_state_qgarch(
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
        psi = var_params["psi"]
        beta = var_params["beta"]
        var_seq, terminal = run_qgarch(
            eps=eps_seq, omega=omega, alpha=alpha, psi=psi, beta=beta,
            init_eps_lags=init_state[0],
            init_eps_sq_lags=init_state[1],
            init_var_lags=init_state[2],
        )
        return var_seq, (terminal[0], terminal[1], terminal[2])

    def _ag_cold_start(
        self,
        eps_proxy: Array,
        mode: str,
        backcast_length: Optional[int],
        wrapper: StandardisedResidual,
    ) -> dict:
        base = init_garch_params(
            eps_proxy, p=self.p, q=self.q, mode=mode,
            backcast_length=backcast_length,
        )
        return {
            "omega": base["omega"],
            "alpha": base["alpha"],
            "psi": jnp.zeros((1,), dtype=float),
            "beta": base["beta"],
        }

    def _ag_forecast_step(
        self,
        var_params: dict,
        residual_params: dict,
        terminal_state: tuple,
    ) -> tuple[Array, tuple]:
        r"""Analytical h-step forecast.  Future ``ε`` has expectation
        zero (so the ``ψ·ε`` term drops out) and ``E[ε²] = E[σ²]``."""
        omega = var_params["omega"]
        alpha = var_params["alpha"]
        psi = var_params["psi"]
        beta = var_params["beta"]
        eps_lags, eps_sq_lags, var_lags = terminal_state
        ar_term = alpha[0] * eps_sq_lags[0]
        psi_term = psi[0] * eps_lags[0]
        ma_term = jnp.dot(beta, var_lags) if self.q > 0 else 0.0
        var_next = jnp.maximum(omega + ar_term + psi_term + ma_term, _VAR_FLOOR)
        # Future ε has E[ε]=0; future ε² has expectation σ²_next.
        new_eps_lags = jnp.zeros((1,), dtype=float)
        new_eps_sq_lags = var_next.reshape((1,))
        new_var_lags = (
            jnp.concatenate([var_next.reshape((1,)), var_lags[:-1]])
            if self.q > 0 else var_lags
        )
        return var_next, (new_eps_lags, new_eps_sq_lags, new_var_lags)

    def _ag_rvs_step(
        self,
        var_params: dict,
        residual_params: dict,
        terminal_state: tuple,
        eps_t: Array,
    ) -> tuple[Array, tuple]:
        omega = var_params["omega"]
        alpha = var_params["alpha"]
        psi = var_params["psi"]
        beta = var_params["beta"]
        eps_lags, eps_sq_lags, var_lags = terminal_state
        ar_term = alpha[0] * eps_sq_lags[0]
        psi_term = psi[0] * eps_lags[0]
        ma_term = jnp.dot(beta, var_lags) if self.q > 0 else 0.0
        var_t = jnp.maximum(omega + ar_term + psi_term + ma_term, _VAR_FLOOR)
        new_eps_lags = eps_t.reshape((1,))
        new_eps_sq_lags = (eps_t * eps_t).reshape((1,))
        new_var_lags = (
            jnp.concatenate([var_t.reshape((1,)), var_lags[:-1]])
            if self.q > 0 else var_lags
        )
        return var_t, (new_eps_lags, new_eps_sq_lags, new_var_lags)

    def _ag_var_terminal_state_class(self) -> type:
        return QGARCHTerminalState

    @classmethod
    def _deserialise_extra_kwargs(cls, params: dict) -> dict:
        return {"psi": params.get("psi")}
