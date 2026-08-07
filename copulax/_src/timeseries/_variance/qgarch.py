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

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src.timeseries._base import TerminalState
from copulax._src.timeseries._init import (
    garch_pre_sample_state,
    garch_presample_warmup,
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

    eps_lags: Array  # shape (1,)
    eps_sq_lags: Array  # shape (1,)
    var_lags: Array  # shape (q,)


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

    References
    ----------
    .. [1] Sentana, E. (1995). *Quadratic ARCH Models*. Review of
       Economic Studies, 62(4), 639-661 (the linear-in-:math:`\varepsilon`
       :math:`\psi\, \varepsilon_{t-1}` asymmetry term; positivity via the
       quadratic discriminant :math:`\omega \geq \psi^2 / (4\alpha)`; the
       :math:`p \geq 2` augmented-matrix PSD condition, restricted to
       :math:`p = 1` in v1).  CRAN ``qgarch`` is a *different* model and is
       not a valid oracle.
    """

    psi: Array | None = None
    terminal_state: QGARCHTerminalState | None = None

    def __init__(
        self,
        p: int = 0,
        q: int = 0,
        *,
        residual_dist: Univariate | None = None,
        name: str = "QGARCH",
        omega: ArrayLike | None = None,
        alpha: ArrayLike | None = None,
        psi: ArrayLike | None = None,
        beta: ArrayLike | None = None,
        residual_params: dict | None = None,
        terminal_state: QGARCHTerminalState | None = None,
        n_train_: int | None = None,
        cov_matrix_: ArrayLike | None = None,
        standard_errors_: dict | None = None,
        residual_diagnostics_: dict | None = None,
        converged: ArrayLike | None = None,
        grad_norm: ArrayLike | None = None,
        n_iterations: ArrayLike | None = None,
        nan_encountered: ArrayLike | None = None,
        n_finite_candidates: ArrayLike | None = None,
        best_candidate: ArrayLike | None = None,
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
            converged=converged,
            grad_norm=grad_norm,
            n_iterations=n_iterations,
            nan_encountered=nan_encountered,
            n_finite_candidates=n_finite_candidates,
            best_candidate=best_candidate,
        )
        self.psi = (
            jnp.asarray(psi, dtype=float).reshape(-1) if psi is not None else None
        )

    @property
    def _stored_params(self) -> dict | None:
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
            self.omega is None
            or self.alpha is None
            or self.psi is None
            or self.beta is None
            or self.residual_params is None
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
        wrapper = StandardisedResidual(cast("Univariate", self.residual_dist))
        return 1 + 1 + 1 + self.q + wrapper.n_shape_params

    # ------------------------------------------------------------------
    # Reparameterisation pack / unpack
    # ------------------------------------------------------------------
    def _pack_x0_qgarch(
        self,
        params_dict: dict,
        wrapper: StandardisedResidual,
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
        self,
        raw: Array,
        wrapper: StandardisedResidual,
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
        backcast_length: int | None,
    ) -> tuple[Array, Array, Array]:
        eps_sq_lags, var_lags = garch_pre_sample_state(
            eps,
            p=1,
            q=self.q,
            mode=mode,
            backcast_length=backcast_length,
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
        n_warmup: int = 0,
        warmup_var: ArrayLike = 0.0,
    ) -> tuple[Array, QGARCHTerminalState]:
        eps_lags, eps_sq_lags, var_lags = init_state
        var_seq, terminal = run_qgarch(
            eps=eps,
            omega=omega,
            alpha=alpha,
            psi=psi,
            beta=beta,
            init_eps_lags=eps_lags,
            init_eps_sq_lags=eps_sq_lags,
            init_var_lags=var_lags,
            n_warmup=n_warmup,
            warmup_var=warmup_var,
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
        backcast_length: int | None,
    ) -> dict:
        r"""Cold-start: vanilla GARCH(1, q) starting point with ``ψ = 0``."""
        base = super()._build_cold_start(
            eps,
            wrapper,
            init=init,
            backcast_length=backcast_length,
        )
        base["psi"] = jnp.zeros((1,), dtype=float)
        return base

    def _make_objective_qgarch(
        self,
        wrapper: StandardisedResidual,
    ) -> Callable[[Array, Array, Array, Array, Array], Array]:
        def objective(
            raw: Array,
            eps: Array,
            init_eps_lags: Array,
            init_eps_sq_lags: Array,
            init_var_lags: Array,
        ) -> Array:
            omega, alpha, psi, beta, residual_shape = self._unpack_raw_qgarch(
                raw,
                wrapper,
            )
            init_state = (init_eps_lags, init_eps_sq_lags, init_var_lags)
            var_seq, _ = self._run_recursion_qgarch(
                eps,
                omega,
                alpha,
                psi,
                beta,
                init_state,
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
        init_params: dict | None = None,
        n_starts: int = 1,
        backcast_length: int | None = None,
        maxiter: int = 200,
        lr: float = 0.05,
        name: str | None = None,
    ) -> QGARCH:
        r"""Fit QGARCH(1, q) to a mean-corrected innovation series.

        Note:
            If you intend to jit wrap this function, ensure that
            ``n_starts`` is a static argument.

        Args:
            eps: shape ``(n,)`` — mean-corrected innovation series.
            init: One of ``"analytical"``, ``"backcast"``, ``"sample"``
                or ``"warm"``.
            init_params: Warm-start parameter dict; required when
                ``init="warm"``.
            n_starts: Number of optimiser starts.  The default ``1`` fits
                from the single ``init`` seed.  Values ``> 1`` run a
                multi-start fit that additionally seeds from the other
                cold-start init modes and returns the best
                finite-likelihood result; the count is capped at the
                number of available candidates.  Ignored when
                ``init="warm"`` (a warm start is always a single
                explicit-parameter start).
            backcast_length: Window for the EWMA backcast under
                ``init="backcast"``.  ``None`` uses the full series.
            maxiter: Adam iterations.
            lr: Adam learning rate.
            name: Optional custom name for the fitted instance.

        Returns:
            A fitted ``QGARCH`` instance.
        """
        self._check_method(init)
        n_starts = self._validate_n_starts(n_starts)
        wrapper = StandardisedResidual(cast("Univariate", self.residual_dist))
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
            starts = [self._pack_x0_qgarch(cold, wrapper)]
        else:
            starts = self._cold_start_x0_batch(
                eps_arr,
                wrapper,
                backcast_length=backcast_length,
                init=init,
                n_starts=n_starts,
                pack=self._pack_x0_qgarch,
            )

        # The pre-sample state is keyed on the CALLER's chosen init mode and
        # shared across every candidate, so all candidates are scored on the
        # identical likelihood surface and only the start point varies.
        _state_mode = "sample" if init == "sample" else "backcast"
        init_eps, init_eps_sq, init_var = self._initial_state_qgarch(
            eps_arr,
            mode=_state_mode,
            backcast_length=backcast_length,
        )

        objective = self._make_objective_qgarch(wrapper)
        # HARD-04: vmap the candidate starts through the best-iterate
        # projected_gradient and keep the finite-likelihood argmax.
        res, candidate_stats = self._multi_start_fit(
            objective,
            starts,
            {
                "eps": eps_arr,
                "init_eps_lags": init_eps,
                "init_eps_sq_lags": init_eps_sq,
                "init_var_lags": init_var,
            },
            lr=lr,
            maxiter=maxiter,
        )
        x_opt = res["x"]
        omega, alpha, psi, beta, residual = self._unpack_raw_qgarch(
            x_opt,
            wrapper,
        )

        # D-09: convergence status from the solver result, including the
        # real multi-start candidate aggregates.
        status = self._compute_convergence_status(
            res,
            objective,
            x_opt,
            (eps_arr, init_eps, init_eps_sq, init_var),
            maxiter,
            candidate_stats=candidate_stats,
        )
        # D-10: fire the convergence / data-scale warnings host-side.
        self._deliver_fit_warnings(status, jnp.var(eps_arr))

        var_seq, terminal = self._run_recursion_qgarch(
            eps_arr,
            omega,
            alpha,
            psi,
            beta,
            init_state=(init_eps, init_eps_sq, init_var),
        )
        sigma_train = jnp.sqrt(jnp.maximum(var_seq, 1e-12))
        z_train = eps_arr / sigma_train

        # WR-05: raw NaN-propagating log-likelihood sum at the fitted
        # params (degenerate fit -> NaN, not the penalised -2e9 objective).
        loglike = self._raw_ll_sum(
            wrapper,
            z_train,
            jnp.log(sigma_train),
            residual,
        )
        n_params_total = 1 + 1 + 1 + self.q + wrapper.n_shape_params
        aic = 2.0 * n_params_total - 2.0 * loglike
        bic = n_params_total * jnp.log(jnp.asarray(n, dtype=float)) - 2.0 * loglike

        params_dict = {
            "omega": omega,
            "alpha": alpha,
            "psi": psi,
            "beta": beta,
            "residual": residual,
        }
        cov, se_dict, diagnostics = self._post_fit_se_and_diagnostics(
            params_dict=params_dict,
            wrapper=wrapper,
            eps_arr=eps_arr,
            init_state=(init_eps, init_eps_sq, init_var),
            z_train=z_train,
            loglikelihood=loglike,
            aic=aic,
            bic=bic,
        )

        fitted = self._build_fitted_instance(
            params_dict,
            wrapper=wrapper,
            terminal_state=terminal,
            n_train=n,
            cov_matrix=cov,
            standard_errors=se_dict,
            residual_diagnostics=diagnostics,
            name=name,
            status=status,
        )
        return cast("QGARCH", fitted)

    # ------------------------------------------------------------------
    # Conditional moments / residuals
    # ------------------------------------------------------------------
    def _qgarch_recursion_inputs(
        self,
        eps: ArrayLike,
        init: str,
        backcast_length: int | None,
    ) -> tuple[Array, tuple[Array, Array, Array], int, Array]:
        eps_arr = self._validate_series(eps)
        n = int(eps_arr.shape[0])
        self._validate_backcast_length(backcast_length, n)
        init_state = self._initial_state_qgarch(
            eps_arr,
            mode=init,
            backcast_length=backcast_length,
        )
        n_warmup, warmup_var = garch_presample_warmup(
            eps_arr,
            p=self.p,
            q=self.q,
            mode=init,
        )
        return eps_arr, init_state, n_warmup, warmup_var

    def conditional_variance(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: int | None = None,
    ) -> Array:
        self._require_fitted()
        eps_arr, init_state, n_warmup, warmup_var = self._qgarch_recursion_inputs(
            eps, init, backcast_length
        )
        var_seq, _ = self._run_recursion_qgarch(
            eps_arr,
            cast("Array", self.omega),
            cast("Array", self.alpha),
            cast("Array", self.psi),
            cast("Array", self.beta),
            init_state,
            n_warmup=n_warmup,
            warmup_var=warmup_var,
        )
        return var_seq

    def residuals(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: int | None = None,
    ) -> dict:
        self._require_fitted()
        eps_arr, init_state, n_warmup, warmup_var = self._qgarch_recursion_inputs(
            eps, init, backcast_length
        )
        var_seq, _ = self._run_recursion_qgarch(
            eps_arr,
            cast("Array", self.omega),
            cast("Array", self.alpha),
            cast("Array", self.psi),
            cast("Array", self.beta),
            init_state,
            n_warmup=n_warmup,
            warmup_var=warmup_var,
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
        backcast_length: int | None = None,
    ) -> QGARCHTerminalState:
        self._require_fitted()
        eps_arr, init_state, n_warmup, warmup_var = self._qgarch_recursion_inputs(
            eps, init, backcast_length
        )
        _, terminal = self._run_recursion_qgarch(
            eps_arr,
            cast("Array", self.omega),
            cast("Array", self.alpha),
            cast("Array", self.psi),
            cast("Array", self.beta),
            init_state,
            n_warmup=n_warmup,
            warmup_var=warmup_var,
        )
        return terminal

    # ------------------------------------------------------------------
    # Loglikelihood / aic / bic
    # ------------------------------------------------------------------
    def _log_likelihood_on_series(
        self,
        eps: ArrayLike,
        init: str = "backcast",
        backcast_length: int | None = None,
    ) -> Array:
        self._require_fitted()
        wrapper = self._wrapper()
        eps_arr, init_state, n_warmup, warmup_var = self._qgarch_recursion_inputs(
            eps, init, backcast_length
        )
        var_seq, _ = self._run_recursion_qgarch(
            eps_arr,
            cast("Array", self.omega),
            cast("Array", self.alpha),
            cast("Array", self.psi),
            cast("Array", self.beta),
            init_state,
            n_warmup=n_warmup,
            warmup_var=warmup_var,
        )
        sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
        z = eps_arr / sigma_seq
        logpdf = wrapper.logpdf(z, cast("dict", self.residual_params)) - jnp.log(
            sigma_seq
        )
        return jnp.sum(logpdf)

    # ------------------------------------------------------------------
    # Forecast — analytical recursion: E[ψ·ε_{τ-1}] = 0 for unobserved future
    # ------------------------------------------------------------------
    def _analytical_forecast(self, h: int, state: TerminalState) -> Array:
        vstate = cast("QGARCHTerminalState", state)
        var_path = []
        eps_lags = vstate.eps_lags
        eps_sq_lags = vstate.eps_sq_lags
        var_lags = vstate.var_lags
        for _ in range(h):
            ar_term = cast("Array", self.alpha)[0] * eps_sq_lags[0]
            psi_term = cast("Array", self.psi)[0] * eps_lags[0]
            ma_term = jnp.dot(cast("Array", self.beta), var_lags) if self.q > 0 else 0.0
            var_t = cast("Array", self.omega) + ar_term + psi_term + ma_term
            var_t = jnp.maximum(var_t, _VAR_FLOOR)
            var_path.append(var_t)
            # Substitute E[ε_τ] = 0, E[ε²_τ] = E[σ²_τ] for unobserved
            # future shocks (Sentana 1995, eqn 4.4).
            eps_lags = jnp.zeros((1,), dtype=float)
            eps_sq_lags = var_t.reshape((1,))
            if self.q > 0:
                var_lags = jnp.concatenate([var_t.reshape((1,)), var_lags[:-1]])
        return jnp.stack(var_path)

    def forecast(
        self,
        h: int,
        *,
        method: str = "analytical",
        n_paths: int = 0,
        key: Array | None = None,
        u: ArrayLike | None = None,
        last_state: TerminalState | None = None,
    ) -> dict:
        r"""``h``-step-ahead conditional moments.

        QGARCH supports analytical h-step forecasting at every
        horizon: under stationarity the expected ``ψ·ε`` term
        vanishes for unobserved future shocks
        (:math:`\mathbb{E}[\varepsilon_\tau] = 0`), so the σ²
        recursion collapses to vanilla-GARCH form for the forecast.

        Note:
            If you intend to jit wrap this function, ensure that
            ``h`` and ``n_paths`` are static arguments.

        Args:
            h: Forecast horizon (number of steps ahead), ``> 0``.
            method: ``'analytical'`` or ``'simulation'``.
            n_paths: Number of Monte Carlo paths for
                ``method='simulation'`` when ``u`` is not supplied.
            key: JAX random key for internal simulation sampling
                (ignored when ``u`` is supplied).
            u: Optional pre-drawn uniform ``(0, 1)`` samples for
                ``method='simulation'``.  When provided, the uniforms
                are forwarded through the identical ppf path as
                :py:meth:`rvs` (``self.rvs(u=u, last_state=state)``),
                giving full parity between ``forecast(u=U)`` and
                ``rvs(u=U)``.  ``u`` may be 1D (``(h,)``) or 2D
                (``(n_paths, h)``).
            last_state: Terminal state to forecast from.  Defaults to
                the fitted model's ``terminal_state``.
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
            from copulax._src._utils import _resolve_key

            if u is not None:
                # Forward pre-drawn uniforms through the identical ppf
                # path as rvs(u=) — full parity.
                paths = self.rvs(u=u, last_state=state)
            else:
                if n_paths <= 0:
                    raise ValueError(
                        "method='simulation' requires n_paths > 0 (or "
                        "pre-drawn uniforms via u=)."
                    )
                key = _resolve_key(key)
                paths = self.rvs(
                    size=(int(n_paths), h),
                    key=key,
                    last_state=state,
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
    def _roll_path(self, z: Array, state: TerminalState) -> Array:
        vstate = cast("QGARCHTerminalState", state)
        omega = cast("Array", self.omega)
        alpha = cast("Array", self.alpha)
        psi = cast("Array", self.psi)
        beta = cast("Array", self.beta)

        def step(
            carry: tuple[Array, Array, Array],
            z_t: Array,
        ) -> tuple[tuple[Array, Array, Array], Array]:
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
                if self.q > 0
                else var_lags
            )
            return (new_eps_lags, new_eps_sq_lags, new_var_lags), eps_t

        init_carry = (vstate.eps_lags, vstate.eps_sq_lags, vstate.var_lags)
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
        persistence = cast("Array", self.alpha)[0] + jnp.sum(cast("Array", self.beta))
        is_stat = persistence < 1.0
        denom = jnp.where(is_stat, 1.0 - persistence, _VAR_FLOOR)
        unconditional_variance = jnp.where(
            is_stat,
            cast("Array", self.omega) / denom,
            jnp.inf,
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
        psi_sq_over_4alpha = (psi[0] ** 2) / (4.0 * jnp.maximum(alpha[0], _SIGMA_FLOOR))
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
        backcast_length: int | None,
        residual_params: dict,
    ) -> tuple:
        return self._initial_state_qgarch(
            eps_proxy,
            mode=mode,
            backcast_length=backcast_length,
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
            eps=eps_seq,
            omega=omega,
            alpha=alpha,
            psi=psi,
            beta=beta,
            init_eps_lags=init_state[0],
            init_eps_sq_lags=init_state[1],
            init_var_lags=init_state[2],
        )
        return var_seq, (terminal[0], terminal[1], terminal[2])

    def _ag_cold_start(
        self,
        eps_proxy: Array,
        mode: str,
        backcast_length: int | None,
        wrapper: StandardisedResidual,
    ) -> dict:
        base = init_garch_params(
            eps_proxy,
            p=self.p,
            q=self.q,
            mode=mode,
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
            if self.q > 0
            else var_lags
        )
        return var_next, (new_eps_lags, new_eps_sq_lags, new_var_lags)

    def _ag_rvs_step(
        self,
        var_params: dict,
        residual_params: dict,
        terminal_state: tuple,
        z_t: Array,
    ) -> tuple[Array, Array, tuple]:
        omega = var_params["omega"]
        alpha = var_params["alpha"]
        psi = var_params["psi"]
        beta = var_params["beta"]
        eps_lags, eps_sq_lags, var_lags = terminal_state
        ar_term = alpha[0] * eps_sq_lags[0]
        psi_term = psi[0] * eps_lags[0]
        ma_term = jnp.dot(beta, var_lags) if self.q > 0 else 0.0
        var_t = jnp.maximum(omega + ar_term + psi_term + ma_term, _VAR_FLOOR)
        sigma_t = jnp.sqrt(var_t)
        eps_t = sigma_t * z_t
        new_eps_lags = eps_t.reshape((1,))
        new_eps_sq_lags = (eps_t * eps_t).reshape((1,))
        new_var_lags = (
            jnp.concatenate([var_t.reshape((1,)), var_lags[:-1]])
            if self.q > 0
            else var_lags
        )
        return var_t, eps_t, (new_eps_lags, new_eps_sq_lags, new_var_lags)

    def _ag_var_terminal_state_class(self) -> type:
        return QGARCHTerminalState

    @classmethod
    def _deserialise_extra_kwargs(cls, params: dict) -> dict:
        return {"psi": params.get("psi")}
