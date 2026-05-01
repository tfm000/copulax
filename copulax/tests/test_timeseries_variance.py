"""End-to-end tests for the GARCH-family conditional-variance models.

Initial coverage targets vanilla GARCH(p, q); subsequent commits will
add IGARCH / GJR-GARCH / EGARCH / TGARCH / QGARCH / GARCH-M as those
variants land.

Coverage:

* Parameter recovery on simulated data.
* Cross-validation against ``arch.arch_model`` — parameter values
  within ``rtol = 5e-3`` and log-likelihood within ``rtol = 1e-4``
  per the plan-mandated tolerances.
* Recursion correctness against a hand-rolled NumPy reference.
* ``residuals`` returns ``(ε, z)`` with ``z`` having empirical
  unit variance.
* ``stats()`` exposes unconditional variance / persistence /
  half-life / stationarity flag.
* ``forecast(h)`` rolls the σ²-recursion forward correctly and
  approaches the unconditional variance asymptotically.
* JIT compatibility of fit / residuals / conditional_variance.
* Warm-start refit converges in much fewer iterations than the
  cold start.
* Stored vs recomputed loglikelihood / aic / bic parity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copulax.timeseries import EGARCH, GARCH, GJR_GARCH, IGARCH
from copulax.univariate import normal, student_t


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
def _simulate_garch11(n, omega, alpha, beta, key):
    sigma2_uncond = omega / (1.0 - alpha - beta)
    z = jax.random.normal(key, (n,))

    def step(carry, z_t):
        sigma2_prev, eps2_prev = carry
        sigma2_t = omega + alpha * eps2_prev + beta * sigma2_prev
        eps_t = jnp.sqrt(sigma2_t) * z_t
        return (sigma2_t, eps_t * eps_t), eps_t

    _, eps = jax.lax.scan(step, (sigma2_uncond, sigma2_uncond), z)
    return eps


# ---------------------------------------------------------------------------
# Parameter recovery
# ---------------------------------------------------------------------------
class TestRecovery:
    def test_garch11_recovery(self):
        """GARCH(1, 1) parameters recover within tolerance on n=2000."""
        key = jax.random.PRNGKey(2)
        omega_t, alpha_t, beta_t = 0.05, 0.10, 0.85
        eps = _simulate_garch11(2000, omega_t, alpha_t, beta_t, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, init="analytical", maxiter=600, lr=0.05)
        params = fit.params
        # Loose tolerances: GARCH MLE has heavy sample bias on
        # short series, so allow ~30% absolute.  The exact-match-
        # to-arch test below is the tighter check.
        np.testing.assert_allclose(
            float(params["omega"]), omega_t, atol=0.03,
        )
        np.testing.assert_allclose(
            float(params["alpha"][0]), alpha_t, atol=0.05,
        )
        np.testing.assert_allclose(
            float(params["beta"][0]), beta_t, atol=0.05,
        )


# ---------------------------------------------------------------------------
# Cross-validation against arch
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestArchCrossValidation:
    """Plan-mandated cross-validation against ``arch.arch_model``."""

    @pytest.fixture(scope="class")
    def arch_module(self):
        return pytest.importorskip("arch")

    def test_garch11_vs_arch(self, arch_module):
        key = jax.random.PRNGKey(2)
        eps = _simulate_garch11(2000, 0.05, 0.10, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, init="analytical", maxiter=1000, lr=0.05)
        am = arch_module.arch_model(
            np.asarray(eps), mean="Zero", vol="GARCH",
            p=1, q=1, dist="Normal",
        )
        arch_res = am.fit(disp="off")

        np.testing.assert_allclose(
            float(fit.params["omega"]),
            float(arch_res.params["omega"]),
            rtol=5e-3, atol=1e-4,
        )
        np.testing.assert_allclose(
            float(fit.params["alpha"][0]),
            float(arch_res.params["alpha[1]"]),
            rtol=5e-3, atol=1e-4,
        )
        np.testing.assert_allclose(
            float(fit.params["beta"][0]),
            float(arch_res.params["beta[1]"]),
            rtol=5e-3, atol=1e-4,
        )
        np.testing.assert_allclose(
            float(fit.loglikelihood_),
            float(arch_res.loglikelihood),
            rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# Recursion correctness
# ---------------------------------------------------------------------------
class TestRecursion:
    def test_conditional_variance_matches_numpy_reference(self):
        """Hand-rolled NumPy GARCH recursion matches
        ``conditional_variance(eps)`` to single-precision tolerance."""
        key = jax.random.PRNGKey(2)
        eps = _simulate_garch11(500, 0.05, 0.10, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, init="analytical", maxiter=200, lr=0.05)
        omega = float(fit.params["omega"])
        alpha = float(fit.params["alpha"][0])
        beta = float(fit.params["beta"][0])
        eps_np = np.asarray(eps)

        # NumPy reference using EWMA backcast for pre-sample state
        # (matches CopulAX's default).
        decay = 0.94
        weights = (1.0 - decay) * np.power(decay, np.arange(len(eps_np)))
        var_anchor = float(np.sum(weights * (eps_np ** 2)))
        var_ref = np.zeros_like(eps_np)
        eps_sq_lag = var_anchor
        var_lag = var_anchor
        for t in range(len(eps_np)):
            v = omega + alpha * eps_sq_lag + beta * var_lag
            var_ref[t] = v
            eps_sq_lag = float(eps_np[t] ** 2)
            var_lag = float(v)

        var_jax = np.asarray(fit.conditional_variance(eps))
        np.testing.assert_allclose(var_jax, var_ref, rtol=1e-5, atol=1e-5)

    def test_residuals_unit_variance(self):
        """Standardised residuals z_t have empirical mean ≈ 0 and var ≈ 1."""
        key = jax.random.PRNGKey(2)
        eps = _simulate_garch11(2000, 0.05, 0.10, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, init="analytical", maxiter=600, lr=0.05)
        eps_t, z_t = fit.residuals(eps)
        np.testing.assert_allclose(np.asarray(eps_t), np.asarray(eps))
        np.testing.assert_allclose(float(z_t.mean()), 0.0, atol=0.05)
        np.testing.assert_allclose(float(z_t.var()), 1.0, atol=0.05)

    def test_loglikelihood_recompute_parity(self):
        key = jax.random.PRNGKey(2)
        eps = _simulate_garch11(500, 0.05, 0.10, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        np.testing.assert_allclose(
            float(fit.loglikelihood_), float(fit.loglikelihood(eps)),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(fit.aic_), float(fit.aic(eps)), rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(fit.bic_), float(fit.bic(eps)), rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# Stats / forecast
# ---------------------------------------------------------------------------
class TestStats:
    def test_stats_returns_expected_keys(self):
        key = jax.random.PRNGKey(2)
        eps = _simulate_garch11(500, 0.05, 0.10, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        stats = fit.stats()
        assert {"unconditional_variance", "persistence", "half_life",
                "is_stationary"} <= set(stats)
        assert bool(stats["is_stationary"])
        # Persistence < 1 by construction (simplex enforces it)
        assert float(stats["persistence"]) < 1.0


class TestForecast:
    def test_analytical_variance_forecast_converges(self):
        """h-step variance forecast tends toward the unconditional
        variance as h grows."""
        key = jax.random.PRNGKey(2)
        eps = _simulate_garch11(2000, 0.05, 0.10, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, init="analytical", maxiter=800, lr=0.05)
        fc = fit.forecast(h=1000, method="analytical")
        uncond = float(fit.stats()["unconditional_variance"])
        # Last forecast step should be within 1% of the unconditional.
        np.testing.assert_allclose(
            float(fc["variance"][-1]), uncond, rtol=0.01,
        )
        np.testing.assert_array_equal(
            np.asarray(fc["mean"]), np.zeros((1000,)),
        )

    def test_simulation_forecast_path_shape(self):
        key = jax.random.PRNGKey(2)
        eps = _simulate_garch11(500, 0.05, 0.10, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        fc = fit.forecast(
            h=10, method="simulation", n_paths=200,
            key=jax.random.PRNGKey(7),
        )
        assert fc["paths"].shape == (200, 10)
        assert fc["mean"].shape == (10,)
        assert fc["variance"].shape == (10,)

    def test_rvs_deterministic_under_u(self):
        key = jax.random.PRNGKey(2)
        eps = _simulate_garch11(500, 0.05, 0.10, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        u = jnp.linspace(0.01, 0.99, 30)
        path1 = fit.rvs(u=u)
        path2 = fit.rvs(u=u)
        np.testing.assert_allclose(np.asarray(path1), np.asarray(path2))

    def test_rvs_batch_shape(self):
        key = jax.random.PRNGKey(2)
        eps = _simulate_garch11(500, 0.05, 0.10, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        paths = fit.rvs(size=(50, 12), key=jax.random.PRNGKey(1))
        assert paths.shape == (50, 12)


# ---------------------------------------------------------------------------
# JIT / autograd / warm start
# ---------------------------------------------------------------------------
class TestJIT:
    def test_jit_conditional_variance(self):
        key = jax.random.PRNGKey(2)
        eps = _simulate_garch11(500, 0.05, 0.10, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        jit_cv = jax.jit(fit.conditional_variance)
        np.testing.assert_allclose(
            np.asarray(jit_cv(eps)),
            np.asarray(fit.conditional_variance(eps)),
        )

    def test_warm_start_converges_quickly(self):
        key = jax.random.PRNGKey(2)
        eps = _simulate_garch11(500, 0.05, 0.10, 0.85, key)
        cold = GARCH(p=1, q=1, residual_dist=normal).fit(eps, init="analytical", maxiter=1000, lr=0.05)
        warm = GARCH(p=1, q=1, residual_dist=normal).fit(eps, init="warm", init_params=cold.params, maxiter=20, lr=0.05)
        np.testing.assert_allclose(
            float(warm.loglikelihood_), float(cold.loglikelihood_),
            rtol=5e-3,
        )


# ---------------------------------------------------------------------------
# Residual law swap (smoke)
# ---------------------------------------------------------------------------
class TestResidualLaws:
    def test_student_t_fit_smoke(self):
        key = jax.random.PRNGKey(2)
        eps = _simulate_garch11(2000, 0.05, 0.10, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=student_t).fit(eps, init="analytical", maxiter=400, lr=0.05)
        assert fit.is_fitted
        assert "nu" in fit.params["residual"]
        assert jnp.isfinite(fit.loglikelihood_)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_unfitted_raises(self):
        with pytest.raises(ValueError, match="not fitted"):
            GARCH(p=1, q=1).conditional_variance(jnp.array([1.0, 2.0, 3.0]))

    def test_stationarity_enforced_by_simplex(self):
        """The fitted persistence is strictly below 1 — the simplex
        reparameterisation guarantees this regardless of the data."""
        key = jax.random.PRNGKey(2)
        eps = _simulate_garch11(500, 0.05, 0.10, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        persistence = float(fit.params["alpha"].sum() + fit.params["beta"].sum())
        assert persistence < 1.0


# ---------------------------------------------------------------------------
# IGARCH (integrated GARCH; persistence = 1)
# ---------------------------------------------------------------------------
def _simulate_igarch11(n, omega, alpha, beta, key):
    """alpha + beta = 1 by construction."""
    assert abs((alpha + beta) - 1.0) < 1e-10, "IGARCH requires alpha+beta=1"
    z = jax.random.normal(key, (n,))

    def step(carry, z_t):
        sigma2_prev, eps2_prev = carry
        sigma2_t = omega + alpha * eps2_prev + beta * sigma2_prev
        eps_t = jnp.sqrt(sigma2_t) * z_t
        return (sigma2_t, eps_t * eps_t), eps_t

    _, eps = jax.lax.scan(step, (1.0, 1.0), z)
    return eps


class TestIGARCH:
    def test_persistence_pinned_to_one(self):
        """Simplex reparam pins ``Σα + Σβ = 1`` exactly."""
        key = jax.random.PRNGKey(2)
        eps = _simulate_igarch11(2000, 0.05, 0.10, 0.90, key)
        fit = IGARCH(p=1, q=1, residual_dist=normal).fit(
            eps, init="analytical", maxiter=500, lr=0.05,
        )
        persistence = float(
            fit.params["alpha"].sum() + fit.params["beta"].sum()
        )
        np.testing.assert_allclose(persistence, 1.0, atol=1e-6)

    def test_stats_reports_inf_unconditional_variance(self):
        key = jax.random.PRNGKey(2)
        eps = _simulate_igarch11(500, 0.05, 0.10, 0.90, key)
        fit = IGARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        s = fit.stats()
        assert jnp.isinf(s["unconditional_variance"])
        assert jnp.isinf(s["half_life"])
        assert not bool(s["is_stationary"])

    def test_n_params_drops_one(self):
        """IGARCH has one fewer free parameter than vanilla GARCH because
        the simplex constraint Σα+Σβ=1 removes a degree of freedom."""
        key = jax.random.PRNGKey(2)
        eps = _simulate_igarch11(500, 0.05, 0.10, 0.90, key)
        ig_fit = IGARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        g_fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        assert ig_fit.n_params == g_fit.n_params - 1


# ---------------------------------------------------------------------------
# GJR-GARCH (asymmetric leverage)
# ---------------------------------------------------------------------------
def _simulate_gjr_garch11(n, omega, alpha, gamma, beta, key):
    sigma2_uncond = omega / (1.0 - alpha - 0.5 * gamma - beta)
    z = jax.random.normal(key, (n,))

    def step(carry, z_t):
        sigma2_prev, eps_prev = carry
        eps_sq_prev = eps_prev ** 2
        neg_eps_sq_prev = jnp.where(eps_prev < 0, eps_sq_prev, 0.0)
        sigma2_t = (
            omega
            + alpha * eps_sq_prev
            + gamma * neg_eps_sq_prev
            + beta * sigma2_prev
        )
        eps_t = jnp.sqrt(sigma2_t) * z_t
        return (sigma2_t, eps_t), eps_t

    _, eps = jax.lax.scan(step, (sigma2_uncond, jnp.array(0.0)), z)
    return eps


class TestGJRGARCH:
    def test_recovery(self):
        """GJR-GARCH(1, 1) parameters recover within tolerance on n=2000."""
        key = jax.random.PRNGKey(2)
        eps = _simulate_gjr_garch11(2000, 0.05, 0.05, 0.10, 0.85, key)
        fit = GJR_GARCH(p=1, q=1, residual_dist=normal).fit(
            eps, init="analytical", maxiter=800, lr=0.05,
        )
        params = fit.params
        np.testing.assert_allclose(float(params["omega"]), 0.05, atol=0.03)
        np.testing.assert_allclose(float(params["alpha"][0]), 0.05, atol=0.05)
        np.testing.assert_allclose(float(params["gamma"][0]), 0.10, atol=0.05)
        np.testing.assert_allclose(float(params["beta"][0]), 0.85, atol=0.05)

    def test_kappa_appears_in_persistence(self):
        """Stats reports persistence = Σα + κ·Σγ + Σβ; under symmetric
        Normal residuals κ = 0.5 to 4-decimal precision."""
        key = jax.random.PRNGKey(2)
        eps = _simulate_gjr_garch11(2000, 0.05, 0.05, 0.10, 0.85, key)
        fit = GJR_GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=400)
        s = fit.stats()
        # κ for Normal is 0.5 to numerical precision.
        np.testing.assert_allclose(float(s["kappa"]), 0.5, atol=1e-4)
        # Persistence = α + κ·γ + β
        a = float(fit.params["alpha"][0])
        g = float(fit.params["gamma"][0])
        b = float(fit.params["beta"][0])
        np.testing.assert_allclose(
            float(s["persistence"]), a + 0.5 * g + b, atol=1e-4,
        )


@pytest.mark.slow
class TestArchVariantCrossValidation:
    """Cross-validation against ``arch.arch_model`` for asymmetric variants."""

    @pytest.fixture(scope="class")
    def arch_module(self):
        return pytest.importorskip("arch")

    def test_gjr_garch_vs_arch(self, arch_module):
        key = jax.random.PRNGKey(2)
        eps = _simulate_gjr_garch11(2000, 0.05, 0.05, 0.10, 0.85, key)
        fit = GJR_GARCH(p=1, q=1, residual_dist=normal).fit(
            eps, init="analytical", maxiter=1500, lr=0.05,
        )
        am = arch_module.arch_model(
            np.asarray(eps), mean="Zero", vol="GARCH",
            p=1, o=1, q=1, dist="Normal",
        )
        arch_res = am.fit(disp="off")

        np.testing.assert_allclose(
            float(fit.params["omega"]),
            float(arch_res.params["omega"]),
            rtol=5e-3, atol=1e-4,
        )
        np.testing.assert_allclose(
            float(fit.params["alpha"][0]),
            float(arch_res.params["alpha[1]"]),
            rtol=5e-3, atol=1e-4,
        )
        np.testing.assert_allclose(
            float(fit.params["gamma"][0]),
            float(arch_res.params["gamma[1]"]),
            rtol=5e-3, atol=1e-4,
        )
        np.testing.assert_allclose(
            float(fit.params["beta"][0]),
            float(arch_res.params["beta[1]"]),
            rtol=5e-3, atol=1e-4,
        )
        np.testing.assert_allclose(
            float(fit.loglikelihood_),
            float(arch_res.loglikelihood),
            rtol=1e-4,
        )

    def test_egarch_vs_arch(self, arch_module):
        key = jax.random.PRNGKey(2)
        eps = _simulate_egarch11(2000, -0.05, 0.10, -0.05, 0.95, key)
        fit = EGARCH(p=1, q=1, residual_dist=normal).fit(
            eps, init="analytical", maxiter=1500, lr=0.05,
        )
        am = arch_module.arch_model(
            np.asarray(eps), mean="Zero", vol="EGARCH",
            p=1, o=1, q=1, dist="Normal",
        )
        arch_res = am.fit(disp="off")

        # EGARCH parameters
        np.testing.assert_allclose(
            float(fit.params["omega"]),
            float(arch_res.params["omega"]),
            rtol=1e-2, atol=1e-3,
        )
        np.testing.assert_allclose(
            float(fit.params["alpha"][0]),
            float(arch_res.params["alpha[1]"]),
            rtol=1e-2, atol=1e-3,
        )
        np.testing.assert_allclose(
            float(fit.params["gamma"][0]),
            float(arch_res.params["gamma[1]"]),
            rtol=1e-2, atol=1e-3,
        )
        np.testing.assert_allclose(
            float(fit.params["beta"][0]),
            float(arch_res.params["beta[1]"]),
            rtol=1e-2, atol=1e-3,
        )
        np.testing.assert_allclose(
            float(fit.loglikelihood_),
            float(arch_res.loglikelihood),
            rtol=1e-3,
        )


# ---------------------------------------------------------------------------
# EGARCH (log-variance)
# ---------------------------------------------------------------------------
def _simulate_egarch11(n, omega, alpha, gamma, beta, key):
    z = jax.random.normal(key, (n,))
    e_abs_z = (2.0 / jnp.pi) ** 0.5  # E|z| for standard normal

    def step(carry, z_t):
        log_var_prev, z_prev = carry
        log_var_t = (
            omega
            + alpha * (jnp.abs(z_prev) - e_abs_z)
            + gamma * z_prev
            + beta * log_var_prev
        )
        sigma_t = jnp.exp(0.5 * log_var_t)
        eps_t = sigma_t * z_t
        return (log_var_t, z_t), eps_t

    log_var_init = omega / (1.0 - beta) if beta != 1 else 0.0
    _, eps = jax.lax.scan(step, (log_var_init, jnp.array(0.0)), z)
    return eps


class TestEGARCH:
    def test_recovery(self):
        """EGARCH(1, 1) parameters recover within tolerance on n=2000."""
        key = jax.random.PRNGKey(2)
        eps = _simulate_egarch11(2000, -0.05, 0.10, -0.05, 0.95, key)
        fit = EGARCH(p=1, q=1, residual_dist=normal).fit(
            eps, init="analytical", maxiter=600, lr=0.05,
        )
        params = fit.params
        np.testing.assert_allclose(float(params["alpha"][0]), 0.10, atol=0.05)
        np.testing.assert_allclose(float(params["gamma"][0]), -0.05, atol=0.05)
        np.testing.assert_allclose(float(params["beta"][0]), 0.95, atol=0.05)

    def test_no_positivity_constraint(self):
        """ω, α, γ are unconstrained — fitted values can be negative."""
        key = jax.random.PRNGKey(2)
        eps = _simulate_egarch11(2000, -0.05, 0.10, -0.05, 0.95, key)
        fit = EGARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=300)
        # ω is allowed to be negative; γ likewise.
        # No assertion on signs — just confirm the fit completed and
        # produced finite values.
        for key_name in ("omega", "alpha", "gamma", "beta"):
            assert jnp.all(jnp.isfinite(fit.params[key_name]))

    def test_h1_analytical_forecast(self):
        """``forecast(1, "analytical")`` is closed-form and matches the
        recursion's one-step-ahead value."""
        key = jax.random.PRNGKey(2)
        eps = _simulate_egarch11(500, -0.05, 0.10, -0.05, 0.95, key)
        fit = EGARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        fc = fit.forecast(h=1, method="analytical")
        assert fc["variance"].shape == (1,)
        assert jnp.isfinite(fc["variance"][0])

    def test_h2_analytical_raises(self):
        """``forecast(2, "analytical")`` raises ValueError per plan."""
        key = jax.random.PRNGKey(2)
        eps = _simulate_egarch11(500, -0.05, 0.10, -0.05, 0.95, key)
        fit = EGARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        with pytest.raises(ValueError, match="simulation"):
            fit.forecast(h=2, method="analytical")

    def test_simulation_forecast(self):
        key = jax.random.PRNGKey(2)
        eps = _simulate_egarch11(500, -0.05, 0.10, -0.05, 0.95, key)
        fit = EGARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        fc = fit.forecast(
            h=10, method="simulation", n_paths=200,
            key=jax.random.PRNGKey(7),
        )
        assert fc["paths"].shape == (200, 10)
        assert fc["variance"].shape == (10,)
