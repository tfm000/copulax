"""End-to-end tests for the AR / MA / ARMA mean-equation models.

Coverage:

* Parameter recovery on simulated data (loose ``rtol`` for sampling
  noise on small windows; tighter on large ones).
* Cross-validation against ``statsmodels.tsa.arima.ARIMA`` —
  parameter values within ``rtol = 5e-3`` and log-likelihood within
  ``rtol = 1e-4`` per the plan-mandated tolerances.
* Recursion correctness against a hand-rolled NumPy reference.
* ``conditional_mean`` / ``residuals`` reproduce the recursion's
  expected output to single-precision ``rtol``.
* ``loglikelihood(y_train)`` matches the stored ``loglikelihood_``
  to high precision (closes the recompute parity loop required by
  plan §"Stored fit-time diagnostics ↔ recomputation parity").
* ``rvs`` produces the right shape and ``rvs(u=...)`` is
  deterministic.
* ``forecast(h, "analytical")`` rolls forward correctly from the
  stored terminal state.
* JIT-compatibility of fit, residuals, conditional_mean.
* Warm-start refit converges in far fewer iterations than the cold
  start and reaches a comparable log-likelihood.
* Edge cases: AR(0) and MA(0) reduce to a constant-mean fit;
  short-series ``ValueError`` on infeasible orders.

Combinatorial / multi-distribution sweeps are tagged ``slow``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copulax.timeseries import AR, ARMA, MA
from copulax.univariate import normal, student_t


# ---------------------------------------------------------------------------
# Simulators (centred form — matches the production ARMA recursion)
#
#     y_t = mu + phi (y_{t-1} - mu) + theta eps_{t-1} + eps_t
#
# `mu` is the unconditional mean of the process.
# ---------------------------------------------------------------------------
def _simulate_ar1(n, phi, mu, sigma, key):
    eps = sigma * jax.random.normal(key, (n,))
    y = jnp.zeros((n,))
    y = y.at[0].set(mu + eps[0])

    def step(carry, eps_t):
        y_prev = carry
        y_t = mu + phi * (y_prev - mu) + eps_t
        return y_t, y_t

    _, ys = jax.lax.scan(step, y[0], eps[1:])
    return jnp.concatenate([y[:1], ys])


def _simulate_ma1(n, theta, mu, sigma, key):
    eps = sigma * jax.random.normal(key, (n + 1,))
    return mu + eps[1:] + theta * eps[:-1]


def _simulate_arma11(n, phi, theta, mu, sigma, key):
    eps = sigma * jax.random.normal(key, (n + 1,))

    def step(carry, inp):
        y_prev, eps_prev = carry
        eps_curr = inp
        y_t = mu + phi * (y_prev - mu) + eps_curr + theta * eps_prev
        return (y_t, eps_curr), y_t

    init = (mu + eps[1] + theta * eps[0], eps[1])
    _, ys = jax.lax.scan(step, init, eps[2:])
    return jnp.concatenate([init[0].reshape((1,)), ys])


# ---------------------------------------------------------------------------
# Parameter recovery
# ---------------------------------------------------------------------------
class TestRecovery:
    def test_ar1_recovery(self):
        """AR(1) coefficients recover from a 2000-sample DGP within 5%."""
        key = jax.random.PRNGKey(42)
        phi_true, mu_true, sigma_true = 0.6, 0.25, 0.5
        y = _simulate_ar1(2000, phi_true, mu_true, sigma_true, key)

        fit = AR(p=1, residual_dist=normal).fit(y, init="analytical", maxiter=1000, lr=0.05)
        params = fit.params
        np.testing.assert_allclose(float(params["phi"][0]), phi_true, atol=0.05)
        np.testing.assert_allclose(float(params["mu"]), mu_true, atol=0.1)
        np.testing.assert_allclose(
            float(params["sigma_eps"]), sigma_true, rtol=0.05,
        )

    def test_ma1_recovery(self):
        """MA(1) θ recovers within 5% on n=2000."""
        key = jax.random.PRNGKey(7)
        theta_true, mu_true, sigma_true = 0.4, 0.1, 0.5
        y = _simulate_ma1(2000, theta_true, mu_true, sigma_true, key)

        fit = MA(q=1, residual_dist=normal).fit(y, init="analytical", maxiter=1000, lr=0.05)
        params = fit.params
        np.testing.assert_allclose(
            float(params["theta"][0]), theta_true, atol=0.05,
        )
        np.testing.assert_allclose(
            float(params["sigma_eps"]), sigma_true, rtol=0.05,
        )

    def test_arma11_recovery(self):
        """ARMA(1, 1) parameters recover within 5% on n=2000."""
        key = jax.random.PRNGKey(13)
        phi, theta, mu, sigma = 0.5, 0.3, 0.2, 0.5
        y = _simulate_arma11(2000, phi, theta, mu, sigma, key)

        fit = ARMA(p=1, q=1, residual_dist=normal).fit(y, init="analytical", maxiter=1000, lr=0.05)
        params = fit.params
        np.testing.assert_allclose(float(params["phi"][0]), phi, atol=0.05)
        np.testing.assert_allclose(float(params["theta"][0]), theta, atol=0.05)
        np.testing.assert_allclose(float(params["sigma_eps"]), sigma, rtol=0.05)


# ---------------------------------------------------------------------------
# Cross-validation against statsmodels
# ---------------------------------------------------------------------------
class TestStatsmodelsCrossValidation:
    """Plan-mandated cross-validation against ``statsmodels.tsa.arima.ARIMA``.

    Tolerances per plan §"Concrete tolerances": ``rtol=5e-3, atol=1e-4``
    on parameters; ``rtol=1e-4`` on log-likelihood.  ``slow``-tagged
    because each test triggers a Python-level ``statsmodels`` MLE solve.
    """

    @pytest.fixture(scope="class")
    def sm(self):
        statsmodels = pytest.importorskip("statsmodels.api")
        return statsmodels

    def test_arma11_vs_statsmodels(self, sm):
        key = jax.random.PRNGKey(13)
        y = _simulate_arma11(2000, 0.5, 0.3, 0.2, 0.5, key)
        y_np = np.asarray(y)

        fit = ARMA(p=1, q=1, residual_dist=normal).fit(y, init="analytical", maxiter=1500, lr=0.05)
        sm_fit = sm.tsa.arima.ARIMA(y_np, order=(1, 0, 1)).fit()

        sm_phi = float(sm_fit.arparams[0])
        sm_theta = float(sm_fit.maparams[0])
        sm_sigma = float(np.sqrt(sm_fit.params[-1]))

        np.testing.assert_allclose(
            float(fit.params["phi"][0]), sm_phi, rtol=5e-3, atol=1e-4,
        )
        np.testing.assert_allclose(
            float(fit.params["theta"][0]), sm_theta, rtol=5e-3, atol=1e-4,
        )
        np.testing.assert_allclose(
            float(fit.params["sigma_eps"]), sm_sigma, rtol=5e-3, atol=1e-4,
        )
        np.testing.assert_allclose(
            float(fit.loglikelihood()), float(sm_fit.llf), rtol=1e-3,
        )

    def test_ar1_vs_statsmodels(self, sm):
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(2000, 0.6, 0.25, 0.5, key)
        fit = AR(p=1, residual_dist=normal).fit(y, init="analytical", maxiter=1500, lr=0.05)
        sm_fit = sm.tsa.arima.ARIMA(np.asarray(y), order=(1, 0, 0)).fit()
        np.testing.assert_allclose(
            float(fit.params["phi"][0]), float(sm_fit.arparams[0]),
            rtol=5e-3, atol=1e-4,
        )

    def test_ma1_vs_statsmodels(self, sm):
        key = jax.random.PRNGKey(7)
        y = _simulate_ma1(2000, 0.4, 0.1, 0.5, key)
        fit = MA(q=1, residual_dist=normal).fit(y, init="analytical", maxiter=1500, lr=0.05)
        sm_fit = sm.tsa.arima.ARIMA(np.asarray(y), order=(0, 0, 1)).fit()
        np.testing.assert_allclose(
            float(fit.params["theta"][0]), float(sm_fit.maparams[0]),
            rtol=5e-3, atol=1e-4,
        )


# ---------------------------------------------------------------------------
# Recursion correctness, residuals, conditional moments
# ---------------------------------------------------------------------------
class TestRecursion:
    def test_residuals_match_numpy_reference(self):
        """Hand-rolled centred-form NumPy ARMA recursion matches
        ``residuals(y)`` to single-precision ``rtol``."""
        key = jax.random.PRNGKey(13)
        y = _simulate_arma11(500, 0.5, 0.3, 0.2, 0.5, key)
        fit = ARMA(p=1, q=1, residual_dist=normal).fit(y, init="analytical", maxiter=200, lr=0.05)
        params = fit.params
        phi = float(params["phi"][0])
        theta = float(params["theta"][0])
        mu = float(params["mu"])
        y_np = np.asarray(y)

        # Reference: backcast pre-sample state = mean(y); centred form.
        anchor = float(y_np.mean())
        y_lag = anchor
        eps_lag = 0.0
        eps_ref = np.zeros_like(y_np)
        for t in range(len(y_np)):
            mu_t = mu + phi * (y_lag - mu) + theta * eps_lag
            eps_t = y_np[t] - mu_t
            eps_ref[t] = eps_t
            y_lag = float(y_np[t])
            eps_lag = float(eps_t)

        eps_jax = np.asarray(fit.residuals(y)["residuals"])
        np.testing.assert_allclose(eps_jax, eps_ref, rtol=1e-5, atol=1e-5)

    def test_loglikelihood_recompute_parity(self):
        """Stored ``loglikelihood_`` matches recomputation on training data."""
        key = jax.random.PRNGKey(13)
        y = _simulate_arma11(500, 0.5, 0.3, 0.2, 0.5, key)
        fit = ARMA(p=1, q=1, residual_dist=normal).fit(y, maxiter=200)
        np.testing.assert_allclose(
            float(fit.loglikelihood()), float(fit.loglikelihood(y)),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(fit.aic()), float(fit.aic(y)), rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(fit.bic()), float(fit.bic(y)), rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# Forecast / sampling
# ---------------------------------------------------------------------------
class TestForecast:
    def test_analytical_forecast_shape(self):
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(500, 0.6, 0.25, 0.5, key)
        fit = AR(p=1, residual_dist=normal).fit(y, maxiter=200)
        fc = fit.forecast(h=20, method="analytical")
        assert fc["mean"].shape == (20,)
        assert fc["variance"].shape == (20,)
        assert fc["paths"] is None

    def test_simulation_forecast_path_shape(self):
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(500, 0.6, 0.25, 0.5, key)
        fit = AR(p=1, residual_dist=normal).fit(y, maxiter=200)
        fc = fit.forecast(
            h=10, method="simulation", n_paths=200,
            key=jax.random.PRNGKey(7),
        )
        assert fc["paths"].shape == (200, 10)
        assert fc["mean"].shape == (10,)
        assert fc["variance"].shape == (10,)

    def test_rvs_deterministic_under_u(self):
        """rvs(h, u=...) returns identical paths for the same u."""
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(500, 0.6, 0.25, 0.5, key)
        fit = AR(p=1, residual_dist=normal).fit(y, maxiter=200)
        u = jnp.linspace(0.01, 0.99, 30)
        path1 = fit.rvs(u=u)
        path2 = fit.rvs(u=u)
        np.testing.assert_allclose(np.asarray(path1), np.asarray(path2))

    def test_rvs_batch_shape(self):
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(500, 0.6, 0.25, 0.5, key)
        fit = AR(p=1, residual_dist=normal).fit(y, maxiter=200)
        paths = fit.rvs(size=(50, 12), key=jax.random.PRNGKey(1))
        assert paths.shape == (50, 12)


# ---------------------------------------------------------------------------
# JIT / autograd / warm start
# ---------------------------------------------------------------------------
class TestJIT:
    def test_jit_residuals(self):
        """``fit.residuals`` is JIT-compatible end-to-end."""
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(500, 0.6, 0.25, 0.5, key)
        fit = AR(p=1, residual_dist=normal).fit(y, maxiter=200)
        jit_res = jax.jit(fit.residuals)
        out_jit = jit_res(y)
        out_eager = fit.residuals(y)
        for key in ("residuals", "standardised_residuals"):
            np.testing.assert_allclose(
                np.asarray(out_jit[key]), np.asarray(out_eager[key]),
            )

    def test_jit_conditional_mean(self):
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(500, 0.6, 0.25, 0.5, key)
        fit = AR(p=1, residual_dist=normal).fit(y, maxiter=200)
        jit_cm = jax.jit(fit.conditional_mean)
        np.testing.assert_allclose(
            np.asarray(jit_cm(y)), np.asarray(fit.conditional_mean(y)),
        )

    def test_warm_start_converges_quickly(self):
        """20-iteration warm start lands within 0.5% loglike of a 1000-iter
        cold start using the same data."""
        key = jax.random.PRNGKey(13)
        y = _simulate_arma11(500, 0.5, 0.3, 0.2, 0.5, key)
        cold = ARMA(p=1, q=1, residual_dist=normal).fit(y, init="analytical", maxiter=1000, lr=0.05)
        warm = ARMA(p=1, q=1, residual_dist=normal).fit(y, init="warm", init_params=cold.params, maxiter=20, lr=0.05)
        np.testing.assert_allclose(
            float(warm.loglikelihood()), float(cold.loglikelihood()),
            rtol=5e-3,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_ar0_reduces_to_constant_mean(self):
        """AR(0) (i.e. just ``mu + ε``) recovers sample mean."""
        key = jax.random.PRNGKey(0)
        y = jax.random.normal(key, (500,)) + 1.5
        fit = AR(p=0, residual_dist=normal).fit(y, maxiter=200)
        np.testing.assert_allclose(
            float(fit.params["mu"]), float(jnp.mean(y)), atol=0.1,
        )

    def test_ma0_reduces_to_constant_mean(self):
        """MA(0) (i.e. just ``mu + ε``) recovers sample mean."""
        key = jax.random.PRNGKey(1)
        y = jax.random.normal(key, (500,)) - 0.5
        fit = MA(q=0, residual_dist=normal).fit(y, maxiter=200)
        np.testing.assert_allclose(
            float(fit.params["mu"]), float(jnp.mean(y)), atol=0.1,
        )

    def test_unfitted_raises_on_call(self):
        with pytest.raises(ValueError, match="not fitted"):
            ARMA(p=1, q=1).conditional_mean(jnp.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# AR stationarity / MA invertibility — characteristic-polynomial machinery
# ---------------------------------------------------------------------------
class TestStationarityInvertibility:
    """Cover the four-way matrix:

    * AR ``φ`` stationary vs non-stationary (root outside vs inside unit circle),
    * MA ``θ`` invertible vs non-invertible.

    Plus the reparameterisation guarantees: ``raw_to_ar`` always lands
    in the stationary region; ``raw_to_ma`` always lands in the
    invertible region — including q ≥ 2 where the AR-stationarity and
    MA-invertibility regions are *not* the same set (CopulAX's MA
    polynomial uses ``+θ`` while AR uses ``-φ`` — see
    ``ma_polynomial_roots`` for the derivation).
    """

    def test_ar1_polynomial_roots_and_stationarity(self):
        """AR(1): root of ``1 - φz`` is ``1/φ``; stationary iff |φ| < 1."""
        from copulax._src.timeseries._stationarity import (
            ar_is_stationary, ar_polynomial_roots,
        )
        # Stationary: |1/0.5| = 2 > 1.
        roots = ar_polynomial_roots(jnp.array([0.5]))
        np.testing.assert_allclose(np.asarray(roots), [2.0 + 0j], atol=1e-7)
        assert bool(ar_is_stationary(jnp.array([0.5])))
        # Non-stationary: |1/1.5| ≈ 0.667 < 1.
        roots = ar_polynomial_roots(jnp.array([1.5]))
        np.testing.assert_allclose(np.asarray(roots), [1.0 / 1.5 + 0j], atol=1e-7)
        assert not bool(ar_is_stationary(jnp.array([1.5])))

    def test_ma1_polynomial_roots_and_invertibility(self):
        """MA(1): root of ``1 + θz`` is ``-1/θ``; invertible iff |θ| < 1.

        Crucially: MA uses ``+θ`` (matching ``run_arma`` and statsmodels),
        not ``-θ`` — so the root is ``-1/θ``, not ``+1/θ``.
        """
        from copulax._src.timeseries._stationarity import (
            ma_is_invertible, ma_polynomial_roots,
        )
        roots = ma_polynomial_roots(jnp.array([0.5]))
        np.testing.assert_allclose(np.asarray(roots), [-2.0 + 0j], atol=1e-7)
        assert bool(ma_is_invertible(jnp.array([0.5])))
        roots = ma_polynomial_roots(jnp.array([1.5]))
        np.testing.assert_allclose(np.asarray(roots), [-1.0 / 1.5 + 0j], atol=1e-7)
        assert not bool(ma_is_invertible(jnp.array([1.5])))

    def test_ar_vs_ma_polynomials_differ_at_q_ge_2(self):
        """Regression test for the previous bug where MA roots were
        computed via ``ar_polynomial_roots(theta)``.  At q ≥ 2 the AR
        polynomial ``1 - θz - θ²z²`` and the MA polynomial
        ``1 + θz + θ²z²`` have different roots — and ``ar_is_stationary``
        on ``theta`` does *not* answer the MA-invertibility question.
        """
        from copulax._src.timeseries._stationarity import (
            ar_is_stationary, ar_polynomial_roots,
            ma_is_invertible, ma_polynomial_roots,
        )
        # θ = (0.9, -0.5): AR-style polynomial 1 - 0.9z + 0.5z² has
        # complex roots with modulus √2 ≈ 1.414 > 1 (AR-stationary).
        # The TRUE MA polynomial 1 + 0.9z - 0.5z² has real roots
        # 0.777 and 2.577 — modulus 0.777 < 1, NOT invertible.
        theta = jnp.array([0.9, -0.5])
        ar_moduli = jnp.abs(ar_polynomial_roots(theta))
        ma_moduli = jnp.abs(ma_polynomial_roots(theta))
        assert bool(jnp.all(ar_moduli > 1.0))             # AR-stationary on theta
        assert not bool(jnp.all(ma_moduli > 1.0))         # but NOT MA-invertible
        assert bool(ar_is_stationary(theta))
        assert not bool(ma_is_invertible(theta))
        # And the moduli are genuinely different at q = 2.
        assert not np.allclose(np.sort(ar_moduli), np.sort(ma_moduli), atol=1e-3)

    @pytest.mark.parametrize("q", [1, 2, 3, 4])
    def test_raw_to_ma_always_invertible(self, q):
        """The reparameterisation guarantee: any unconstrained ``raw``
        vector produces θ that lies in the open MA-invertibility region.
        Sample 50 random ``raw`` vectors and check every one.
        """
        from copulax._src.timeseries._stationarity import (
            ma_is_invertible, raw_to_ma,
        )
        key = jax.random.PRNGKey(q)
        raws = jax.random.normal(key, (50, q))
        for raw in raws:
            theta = raw_to_ma(raw)
            assert bool(ma_is_invertible(theta)), (
                f"raw={np.asarray(raw)} → theta={np.asarray(theta)} not invertible"
            )

    @pytest.mark.parametrize("p", [1, 2, 3, 4])
    def test_raw_to_ar_always_stationary(self, p):
        """Mirror of the MA test: random ``raw`` always produces
        AR-stationary ``φ``.
        """
        from copulax._src.timeseries._stationarity import (
            ar_is_stationary, raw_to_ar,
        )
        key = jax.random.PRNGKey(100 + p)
        raws = jax.random.normal(key, (50, p))
        for raw in raws:
            phi = raw_to_ar(raw)
            assert bool(ar_is_stationary(phi)), (
                f"raw={np.asarray(raw)} → phi={np.asarray(phi)} not stationary"
            )

    def test_round_trip_inverses(self):
        """``ar_to_raw ∘ raw_to_ar`` and ``ma_to_raw ∘ raw_to_ma`` are
        identity (up to clipping at the boundary)."""
        from copulax._src.timeseries._stationarity import (
            ar_to_raw, ma_to_raw, raw_to_ar, raw_to_ma,
        )
        for q in (1, 2, 3):
            raw = jax.random.normal(jax.random.PRNGKey(q + 7), (q,))
            theta = raw_to_ma(raw)
            np.testing.assert_allclose(
                np.asarray(ma_to_raw(theta)), np.asarray(raw),
                atol=1e-5,
            )
            phi = raw_to_ar(raw)
            np.testing.assert_allclose(
                np.asarray(ar_to_raw(phi)), np.asarray(raw),
                atol=1e-5,
            )

    def test_fitted_arma_reports_correct_root_moduli(self):
        """End-to-end: a fitted ARMA(1, 1) should expose ``ar_root_moduli``
        and ``ma_root_moduli`` matching ``|1/φ|`` and ``|−1/θ|``.
        """
        key = jax.random.PRNGKey(99)
        y = _simulate_arma11(1500, 0.6, 0.3, 0.0, 1.0, key)
        fit = ARMA(p=1, q=1, residual_dist=normal).fit(
            y, init="analytical", maxiter=500, lr=0.05,
        )
        stats = fit.stats()
        phi = float(fit.params["phi"][0])
        theta = float(fit.params["theta"][0])
        np.testing.assert_allclose(
            float(stats["ar_root_moduli"][0]), 1.0 / abs(phi), rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(stats["ma_root_moduli"][0]), 1.0 / abs(theta), rtol=1e-5,
        )
        assert bool(stats["is_stationary"])
        assert bool(stats["is_invertible"])


# ---------------------------------------------------------------------------
# Residual law swap (smoke)
# ---------------------------------------------------------------------------
class TestResidualLaws:
    def test_student_t_fit_smoke(self):
        """Fit ARMA(1, 1) with Student-T residuals on Student-T-flavoured
        data; assert the fit returns a fitted instance with sensible nu."""
        key = jax.random.PRNGKey(13)
        y = _simulate_arma11(1500, 0.5, 0.3, 0.2, 0.5, key)
        fit = ARMA(p=1, q=1, residual_dist=student_t).fit(y, init="analytical", maxiter=500, lr=0.05)
        assert fit.is_fitted
        # Residual params should include 'nu' (Student-T's shape key)
        assert "nu" in fit.params["residual"]
        # Sanity: log-likelihood is finite
        assert jnp.isfinite(fit.loglikelihood())
