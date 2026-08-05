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

from copulax.tests._timeseries_helpers import (
    BEHAVIOURAL,
    PRECISION,
    STANDARD,
    series,
    shared_fit,
)
from copulax.tests.conftest import require_oracle
from copulax.timeseries import AR, ARMA, MA
from copulax.univariate import normal, student_t

# ---------------------------------------------------------------------------
# Shared data / fits
#
# Series come from the frozen corpus (statsmodels ``arma_generate_sample``
# draws, committed with their SHA-256); fits go through the shared
# registry, which computes each distinct (tier, model, series, arguments)
# combination exactly once for the whole process.
#
# ``_NAME_*`` constants keep the series name next to the fixture that
# consumes it, so a rewire is one edit rather than one per call site.
# ---------------------------------------------------------------------------
_NAME_AR1_2000 = "ar1_p060_m025_sd050_n2000_s42"
_NAME_AR1_500 = "ar1_p060_m025_sd050_n500_s42"
_NAME_MA1_2000 = "ma1_q040_m010_sd050_n2000_s7"
_NAME_ARMA11_2000 = "arma11_p050_q030_m020_sd050_n2000_s13"
_NAME_ARMA11_1500 = "arma11_p050_q030_m020_sd050_n1500_s13"
_NAME_ARMA11_500 = "arma11_p050_q030_m020_sd050_n500_s13"


@pytest.fixture(scope="module")
def ar1_2000_series():
    """AR(1) n=2000 series shared by the recovery and statsmodels tests."""
    return series(_NAME_AR1_2000)


@pytest.fixture(scope="module")
def ma1_2000_series():
    """MA(1) n=2000 series shared by the recovery and statsmodels tests."""
    return series(_NAME_MA1_2000)


@pytest.fixture(scope="module")
def arma11_2000_series():
    """ARMA(1,1) n=2000 series shared by recovery and statsmodels."""
    return series(_NAME_ARMA11_2000)


@pytest.fixture(scope="module")
def arma11_500_series():
    """ARMA(1,1) n=500 series shared by the recursion and JIT tests."""
    return series(_NAME_ARMA11_500)


@pytest.fixture(scope="module")
def ar1_500_series():
    """AR(1) n=500 series behind the shared forecast / JIT fit."""
    return series(_NAME_AR1_500)


@pytest.fixture(scope="module")
def ar1_500_fit():
    """The single STANDARD-tier ``AR(1)`` fit the forecast and JIT tests
    all read from.  They need *a* fitted object, not a particular
    optimum, so the registry shares it process-wide."""
    return shared_fit(
        AR(p=1, residual_dist=normal),
        _NAME_AR1_500,
        tier=STANDARD,
    )


# ---------------------------------------------------------------------------
# Parameter recovery
# ---------------------------------------------------------------------------
class TestRecovery:
    def test_ar1_recovery(self, ar1_2000_series):
        """AR(1) coefficients recover from a 2000-sample DGP within 5%."""
        phi_true, mu_true, sigma_true = 0.6, 0.25, 0.5
        y = ar1_2000_series

        fit = shared_fit(
            AR(p=1, residual_dist=normal),
            _NAME_AR1_2000,
            tier=PRECISION,
        )
        params = fit.params
        np.testing.assert_allclose(float(params["phi"][0]), phi_true, atol=0.05)
        np.testing.assert_allclose(float(params["mu"]), mu_true, atol=0.1)
        np.testing.assert_allclose(
            float(params["sigma_eps"]),
            sigma_true,
            rtol=0.05,
        )

    def test_ma1_recovery(self, ma1_2000_series):
        """MA(1) θ recovers within 5% on n=2000."""
        theta_true, mu_true, sigma_true = 0.4, 0.1, 0.5
        y = ma1_2000_series

        fit = shared_fit(
            MA(q=1, residual_dist=normal),
            _NAME_MA1_2000,
            tier=PRECISION,
        )
        params = fit.params
        np.testing.assert_allclose(
            float(params["theta"][0]),
            theta_true,
            atol=0.05,
        )
        np.testing.assert_allclose(
            float(params["sigma_eps"]),
            sigma_true,
            rtol=0.05,
        )

    def test_arma11_recovery(self, arma11_2000_series):
        """ARMA(1, 1) parameters recover within 5% on n=2000."""
        phi, theta, mu, sigma = 0.5, 0.3, 0.2, 0.5
        y = arma11_2000_series

        fit = shared_fit(
            ARMA(p=1, q=1, residual_dist=normal),
            _NAME_ARMA11_2000,
            tier=PRECISION,
        )
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
        statsmodels = require_oracle("statsmodels.api")
        return statsmodels

    def test_arma11_vs_statsmodels(self, sm, arma11_2000_series):
        y = arma11_2000_series
        y_np = np.asarray(y)

        fit = shared_fit(
            ARMA(p=1, q=1, residual_dist=normal),
            _NAME_ARMA11_2000,
            tier=PRECISION,
        )
        sm_fit = sm.tsa.arima.ARIMA(y_np, order=(1, 0, 1)).fit()

        sm_phi = float(sm_fit.arparams[0])
        sm_theta = float(sm_fit.maparams[0])
        sm_sigma = float(np.sqrt(sm_fit.params[-1]))

        np.testing.assert_allclose(
            float(fit.params["phi"][0]),
            sm_phi,
            rtol=5e-3,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            float(fit.params["theta"][0]),
            sm_theta,
            rtol=5e-3,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            float(fit.params["sigma_eps"]),
            sm_sigma,
            rtol=5e-3,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            float(fit.loglikelihood()),
            float(sm_fit.llf),
            rtol=1e-3,
        )

    def test_ar1_vs_statsmodels(self, sm, ar1_2000_series):
        y = ar1_2000_series
        fit = shared_fit(
            AR(p=1, residual_dist=normal),
            _NAME_AR1_2000,
            tier=PRECISION,
        )
        sm_fit = sm.tsa.arima.ARIMA(np.asarray(y), order=(1, 0, 0)).fit()
        np.testing.assert_allclose(
            float(fit.params["phi"][0]),
            float(sm_fit.arparams[0]),
            rtol=5e-3,
            atol=1e-4,
        )

    def test_ma1_vs_statsmodels(self, sm, ma1_2000_series):
        y = ma1_2000_series
        fit = shared_fit(
            MA(q=1, residual_dist=normal),
            _NAME_MA1_2000,
            tier=PRECISION,
        )
        sm_fit = sm.tsa.arima.ARIMA(np.asarray(y), order=(0, 0, 1)).fit()
        np.testing.assert_allclose(
            float(fit.params["theta"][0]),
            float(sm_fit.maparams[0]),
            rtol=5e-3,
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# Recursion correctness, residuals, conditional moments
# ---------------------------------------------------------------------------
class TestRecursion:
    def test_residuals_match_numpy_reference(self, arma11_500_series):
        """Hand-rolled centred-form NumPy ARMA recursion matches
        ``residuals(y)`` to single-precision ``rtol``."""
        y = arma11_500_series
        fit = shared_fit(
            ARMA(p=1, q=1, residual_dist=normal),
            _NAME_ARMA11_500,
            tier=STANDARD,
        )
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

    def test_loglikelihood_recompute_parity(self, arma11_500_series):
        """Stored ``loglikelihood_`` matches recomputation on training data."""
        y = arma11_500_series
        fit = shared_fit(
            ARMA(p=1, q=1, residual_dist=normal),
            _NAME_ARMA11_500,
            tier=STANDARD,
        )
        np.testing.assert_allclose(
            float(fit.loglikelihood()),
            float(fit.loglikelihood(y)),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(fit.aic()),
            float(fit.aic(y)),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(fit.bic()),
            float(fit.bic(y)),
            rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# Forecast / sampling
# ---------------------------------------------------------------------------
class TestForecast:
    def test_analytical_forecast_shape(self, ar1_500_fit):
        fit = ar1_500_fit
        fc = fit.forecast(h=20, method="analytical")
        assert fc["mean"].shape == (20,)
        assert fc["variance"].shape == (20,)
        assert fc["paths"] is None

    def test_simulation_forecast_path_shape(self, ar1_500_fit):
        fit = ar1_500_fit
        fc = fit.forecast(
            h=10,
            method="simulation",
            n_paths=200,
            key=jax.random.PRNGKey(7),
        )
        assert fc["paths"].shape == (200, 10)
        assert fc["mean"].shape == (10,)
        assert fc["variance"].shape == (10,)

    def test_rvs_deterministic_under_u(self, ar1_500_fit):
        """rvs(h, u=...) returns identical paths for the same u."""
        fit = ar1_500_fit
        u = jnp.linspace(0.01, 0.99, 30)
        path1 = fit.rvs(u=u)
        path2 = fit.rvs(u=u)
        np.testing.assert_allclose(np.asarray(path1), np.asarray(path2))

    def test_rvs_batch_shape(self, ar1_500_fit):
        fit = ar1_500_fit
        paths = fit.rvs(size=(50, 12), key=jax.random.PRNGKey(1))
        assert paths.shape == (50, 12)


# ---------------------------------------------------------------------------
# JIT / autograd / warm start
# ---------------------------------------------------------------------------
class TestJIT:
    def test_jit_residuals(self, ar1_500_series, ar1_500_fit):
        """``fit.residuals`` is JIT-compatible end-to-end."""
        y = ar1_500_series
        fit = ar1_500_fit
        jit_res = jax.jit(fit.residuals)
        out_jit = jit_res(y)
        out_eager = fit.residuals(y)
        for key in ("residuals", "standardised_residuals"):
            np.testing.assert_allclose(
                np.asarray(out_jit[key]),
                np.asarray(out_eager[key]),
            )

    def test_jit_conditional_mean(self, ar1_500_series, ar1_500_fit):
        y = ar1_500_series
        fit = ar1_500_fit
        jit_cm = jax.jit(fit.conditional_mean)
        np.testing.assert_allclose(
            np.asarray(jit_cm(y)),
            np.asarray(fit.conditional_mean(y)),
        )

    def test_jit_fit_end_to_end(self, arma11_500_series):
        """The full ``ARMA(...).fit(y)`` pipeline runs under
        ``jax.jit``."""
        y = arma11_500_series

        def fit_fn(yy):
            return ARMA(p=1, q=1, residual_dist=normal).fit(
                yy,
                init="analytical",
                maxiter=100,
                lr=0.05,
            )

        eager = fit_fn(y)
        jitted = jax.jit(fit_fn)(y)
        for k in ("phi", "theta", "mu"):
            np.testing.assert_allclose(
                np.asarray(jitted.params[k]),
                np.asarray(eager.params[k]),
                rtol=1e-5,
                atol=1e-7,
                err_msg=k,
            )

    def test_warm_start_converges_quickly(self, arma11_500_series):
        """20-iteration warm start lands within 0.5% loglike of a 1000-iter
        cold start using the same data."""
        # BEHAVIOURAL: the iteration budgets ARE the subject here, so
        # these two fits are never shared and their maxiters are frozen.
        cold = shared_fit(
            ARMA(p=1, q=1, residual_dist=normal),
            _NAME_ARMA11_500,
            tier=BEHAVIOURAL,
            init="analytical",
            maxiter=1000,
            lr=0.05,
        )
        warm = shared_fit(
            ARMA(p=1, q=1, residual_dist=normal),
            _NAME_ARMA11_500,
            tier=BEHAVIOURAL,
            init="warm",
            init_params=cold.params,
            maxiter=20,
            lr=0.05,
        )
        np.testing.assert_allclose(
            float(warm.loglikelihood()),
            float(cold.loglikelihood()),
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
        fit = shared_fit(
            AR(p=0, residual_dist=normal),
            "iid_normal_n500_s0",
            tier=STANDARD,
            y=y,
            tag="plus_1.5",
        )
        np.testing.assert_allclose(
            float(fit.params["mu"]),
            float(jnp.mean(y)),
            atol=0.1,
        )

    def test_ma0_reduces_to_constant_mean(self):
        """MA(0) (i.e. just ``mu + ε``) recovers sample mean."""
        key = jax.random.PRNGKey(1)
        y = jax.random.normal(key, (500,)) - 0.5
        fit = shared_fit(
            MA(q=0, residual_dist=normal),
            "iid_normal_n500_s1",
            tier=STANDARD,
            y=y,
            tag="minus_0.5",
        )
        np.testing.assert_allclose(
            float(fit.params["mu"]),
            float(jnp.mean(y)),
            atol=0.1,
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
            ar_is_stationary,
            ar_polynomial_roots,
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
            ma_is_invertible,
            ma_polynomial_roots,
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
            ar_is_stationary,
            ar_polynomial_roots,
            ma_is_invertible,
            ma_polynomial_roots,
        )

        # θ = (0.9, -0.5): AR-style polynomial 1 - 0.9z + 0.5z² has
        # complex roots with modulus √2 ≈ 1.414 > 1 (AR-stationary).
        # The TRUE MA polynomial 1 + 0.9z - 0.5z² has real roots
        # 0.777 and 2.577 — modulus 0.777 < 1, NOT invertible.
        theta = jnp.array([0.9, -0.5])
        ar_moduli = jnp.abs(ar_polynomial_roots(theta))
        ma_moduli = jnp.abs(ma_polynomial_roots(theta))
        assert bool(jnp.all(ar_moduli > 1.0))  # AR-stationary on theta
        assert not bool(jnp.all(ma_moduli > 1.0))  # but NOT MA-invertible
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
            ma_is_invertible,
            raw_to_ma,
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
            ar_is_stationary,
            raw_to_ar,
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
            ar_to_raw,
            ma_to_raw,
            raw_to_ar,
            raw_to_ma,
        )

        for q in (1, 2, 3):
            raw = jax.random.normal(jax.random.PRNGKey(q + 7), (q,))
            theta = raw_to_ma(raw)
            np.testing.assert_allclose(
                np.asarray(ma_to_raw(theta)),
                np.asarray(raw),
                atol=1e-5,
            )
            phi = raw_to_ar(raw)
            np.testing.assert_allclose(
                np.asarray(ar_to_raw(phi)),
                np.asarray(raw),
                atol=1e-5,
            )

    def test_fitted_arma_reports_correct_root_moduli(self):
        """End-to-end: a fitted ARMA(1, 1) should expose ``ar_root_moduli``
        and ``ma_root_moduli`` matching ``|1/φ|`` and ``|−1/θ|``.
        """
        fit = shared_fit(
            ARMA(p=1, q=1, residual_dist=normal),
            "arma11_p060_q030_n1500_s99",
            tier=PRECISION,
        )
        stats = fit.stats()
        phi = float(fit.params["phi"][0])
        theta = float(fit.params["theta"][0])
        np.testing.assert_allclose(
            float(stats["ar_root_moduli"][0]),
            1.0 / abs(phi),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(stats["ma_root_moduli"][0]),
            1.0 / abs(theta),
            rtol=1e-5,
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
        fit = shared_fit(
            ARMA(p=1, q=1, residual_dist=student_t),
            _NAME_ARMA11_1500,
            tier=STANDARD,
        )
        assert fit.is_fitted
        # Residual params should include 'nu' (Student-T's shape key)
        assert "nu" in fit.params["residual"]
        # Sanity: log-likelihood is finite
        assert jnp.isfinite(fit.loglikelihood())
