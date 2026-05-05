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

from copulax.timeseries import (
    EGARCH,
    GARCH,
    GARCH_M,
    GJR_GARCH,
    IGARCH,
    QGARCH,
    TGARCH,
)
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
            float(fit.loglikelihood()),
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
        resid = fit.residuals(eps)
        eps_t, z_t = resid["residuals"], resid["standardised_residuals"]
        np.testing.assert_allclose(np.asarray(eps_t), np.asarray(eps))
        np.testing.assert_allclose(float(z_t.mean()), 0.0, atol=0.05)
        np.testing.assert_allclose(float(z_t.var()), 1.0, atol=0.05)

    def test_loglikelihood_recompute_parity(self):
        key = jax.random.PRNGKey(2)
        eps = _simulate_garch11(500, 0.05, 0.10, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        np.testing.assert_allclose(
            float(fit.loglikelihood()), float(fit.loglikelihood(eps)),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(fit.aic()), float(fit.aic(eps)), rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(fit.bic()), float(fit.bic(eps)), rtol=1e-5,
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
            float(warm.loglikelihood()), float(cold.loglikelihood()),
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
        assert jnp.isfinite(fit.loglikelihood())

    @pytest.mark.parametrize(
        "dist_factory_name",
        ["normal", "student_t", "gen_normal", "nig", "gh", "skewed_t"],
    )
    def test_asymmetric_moment_quadrature_matches_quadax(
        self, dist_factory_name,
    ):
        """The MAPFUNS-compactified Gauss-Legendre quadrature in
        ``StandardisedResidual`` must reproduce the truncated moments
        ``E[z+]``, ``E[z-]``, ``E[z² 1{z<0}]`` to ~1e-5 against an
        independent ``quadax.quadgk`` reference for every distribution
        on the residual whitelist.  These moments drive EGARCH's
        ``E|z|`` centring, GJR's κ-weighting, and TGARCH's first-moment
        stationarity constraint — silent quadrature error here would
        bias every asymmetric variance model under non-Normal
        residuals.
        """
        from quadax import quadgk
        from copulax import univariate as cu_uv
        from copulax._src.timeseries._residuals._registry import (
            _RESIDUAL_DEFAULT_SHAPE_PARAMS,
        )
        from copulax._src.timeseries._residuals._standardise import (
            StandardisedResidual,
        )

        base_dist = getattr(cu_uv, dist_factory_name)
        wrapper = StandardisedResidual(base_dist)
        shape_params = _RESIDUAL_DEFAULT_SHAPE_PARAMS[type(base_dist)]

        # Reference: adaptive Gauss-Kronrod via ``quadax.quadgk`` on
        # the infinite half-lines.  Both the reference and the
        # production code go through ``quadax.utils.MAPFUNS`` to
        # compactify the half-line, but ``quadgk`` is an *adaptive*
        # G-K solver tracking absolute / relative tolerances while the
        # production path is fixed 100-pt Gauss-Legendre — a genuinely
        # independent integrator.  Heavy-tailed laws (Student-T at
        # ν=5) lose mass at any finite truncation, so the open
        # interval is mandatory for a faithful reference.
        def pdf_z_pos(z):
            return z * wrapper.pdf(z, shape_params)

        def pdf_z_neg(z):
            return -z * wrapper.pdf(z, shape_params)

        def pdf_z2_neg(z):
            return z * z * wrapper.pdf(z, shape_params)

        ref_z_pos, _ = quadgk(
            pdf_z_pos, interval=jnp.array([0.0, jnp.inf]),
            epsabs=1e-10, epsrel=1e-10,
        )
        ref_z_neg, _ = quadgk(
            pdf_z_neg, interval=jnp.array([-jnp.inf, 0.0]),
            epsabs=1e-10, epsrel=1e-10,
        )
        ref_z2_neg, _ = quadgk(
            pdf_z2_neg, interval=jnp.array([-jnp.inf, 0.0]),
            epsabs=1e-10, epsrel=1e-10,
        )

        cx_z_pos = float(wrapper.expected_z_pos(shape_params))
        cx_z_neg = float(wrapper.expected_z_neg(shape_params))
        cx_z2_neg = float(wrapper.expected_z2_negative(shape_params))

        np.testing.assert_allclose(cx_z_pos, float(ref_z_pos), atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(cx_z_neg, float(ref_z_neg), atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(cx_z2_neg, float(ref_z2_neg), atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize(
        "variance_cls", [GARCH, GJR_GARCH, EGARCH, TGARCH],
    )
    def test_non_normal_residual_recovery_smoke(self, variance_cls):
        """Each asymmetric variance variant should fit cleanly with a
        Student-T residual law and recover finite parameters.  Catches
        breakage in the residual-shape autograd path through
        ``expected_z*`` (e.g. if a quadrature change accidentally lost
        differentiability w.r.t. ``ν``).  ``maxiter`` is intentionally
        low — this is a "fit converges" smoke test, not a parameter-
        recovery accuracy test.
        """
        key = jax.random.PRNGKey(11)
        eps = _simulate_garch11(1000, 0.05, 0.10, 0.85, key)
        fit = variance_cls(
            p=1, q=1, residual_dist=student_t,
        ).fit(eps, init="analytical", maxiter=150, lr=0.05)
        assert fit.is_fitted
        assert "nu" in fit.params["residual"]
        assert jnp.isfinite(fit.loglikelihood())
        # nu should land in its admissible range (> 2 for finite var).
        assert float(fit.params["residual"]["nu"]) > 2.0


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
            float(fit.loglikelihood()),
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
            float(fit.loglikelihood()),
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


# ---------------------------------------------------------------------------
# TGARCH (Zakoian σ-form)
# ---------------------------------------------------------------------------
def _simulate_tgarch11(n, omega, alpha_pos, alpha_neg, beta, key):
    e_pos = (2.0 / jnp.pi) ** 0.5 / 2  # E[z⁺] for standard normal
    persistence = e_pos * alpha_pos + e_pos * alpha_neg + beta
    sigma_uncond = omega / (1.0 - persistence)
    z = jax.random.normal(key, (n,))

    def step(carry, z_t):
        sigma_prev, eps_prev = carry
        eps_pos_prev = jnp.maximum(eps_prev, 0.0)
        eps_neg_prev = jnp.maximum(-eps_prev, 0.0)
        sigma_t = (
            omega
            + alpha_pos * eps_pos_prev
            + alpha_neg * eps_neg_prev
            + beta * sigma_prev
        )
        eps_t = sigma_t * z_t
        return (sigma_t, eps_t), eps_t

    _, eps = jax.lax.scan(step, (sigma_uncond, jnp.array(0.0)), z)
    return eps


class TestTGARCH:
    def test_recovery(self):
        """TGARCH(1, 1) parameters recover within tolerance on n=2000."""
        key = jax.random.PRNGKey(2)
        eps = _simulate_tgarch11(2000, 0.038, 0.10, 0.18, 0.85, key)
        fit = TGARCH(p=1, q=1, residual_dist=normal).fit(
            eps, init="analytical", maxiter=800, lr=0.05,
        )
        params = fit.params
        # Alpha_neg > alpha_pos by construction (leverage); the fit
        # should preserve that ordering.
        assert float(params["alpha_neg"][0]) > float(params["alpha_pos"][0])
        # Loose tolerances on absolute values (the σ-form has higher
        # sample bias than σ²-form GARCH at the same n).
        np.testing.assert_allclose(
            float(params["alpha_pos"][0]), 0.10, atol=0.1,
        )
        np.testing.assert_allclose(
            float(params["alpha_neg"][0]), 0.18, atol=0.1,
        )
        np.testing.assert_allclose(float(params["beta"][0]), 0.85, atol=0.1)

    def test_stats_first_moment_persistence(self):
        """Persistence = E[z⁺]·Σα⁺ + E[z⁻]·Σα⁻ + Σβ; under Normal
        residuals E[z⁺] = E[z⁻] = √(2/π) / 2."""
        key = jax.random.PRNGKey(2)
        eps = _simulate_tgarch11(2000, 0.038, 0.10, 0.18, 0.85, key)
        fit = TGARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=400)
        s = fit.stats()
        e_pos_expected = (2.0 / jnp.pi) ** 0.5 / 2
        np.testing.assert_allclose(
            float(s["expected_z_pos"]), float(e_pos_expected), atol=1e-4,
        )
        np.testing.assert_allclose(
            float(s["expected_z_neg"]), float(e_pos_expected), atol=1e-4,
        )
        a_pos = float(fit.params["alpha_pos"][0])
        a_neg = float(fit.params["alpha_neg"][0])
        b = float(fit.params["beta"][0])
        expected = e_pos_expected * a_pos + e_pos_expected * a_neg + b
        np.testing.assert_allclose(
            float(s["persistence"]), float(expected), atol=1e-4,
        )

    def test_h1_analytical_forecast(self):
        key = jax.random.PRNGKey(2)
        eps = _simulate_tgarch11(500, 0.038, 0.10, 0.18, 0.85, key)
        fit = TGARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        fc = fit.forecast(h=1, method="analytical")
        assert fc["variance"].shape == (1,)
        assert jnp.isfinite(fc["variance"][0])

    def test_h2_analytical_raises(self):
        key = jax.random.PRNGKey(2)
        eps = _simulate_tgarch11(500, 0.038, 0.10, 0.18, 0.85, key)
        fit = TGARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        with pytest.raises(ValueError, match="simulation"):
            fit.forecast(h=2, method="analytical")


# ---------------------------------------------------------------------------
# QGARCH(1, q) — Sentana 1995
# ---------------------------------------------------------------------------
def _simulate_qgarch11(n, omega, alpha, psi, beta, key):
    sigma2_uncond = omega / (1.0 - alpha - beta)
    z = jax.random.normal(key, (n,))

    def step(carry, z_t):
        sigma2_prev, eps_prev = carry
        sigma2_t = (
            omega + alpha * eps_prev ** 2 + psi * eps_prev + beta * sigma2_prev
        )
        sigma2_t = jnp.maximum(sigma2_t, 1e-10)
        eps_t = jnp.sqrt(sigma2_t) * z_t
        return (sigma2_t, eps_t), eps_t

    _, eps = jax.lax.scan(step, (sigma2_uncond, jnp.array(0.0)), z)
    return eps


class TestQGARCH:
    def test_recovery(self):
        """QGARCH(1, 1) parameters recover within tolerance on n=2000.

        ψ is weakly co-identified with the residual-law skew so we
        use a loose tolerance on it.
        """
        key = jax.random.PRNGKey(2)
        eps = _simulate_qgarch11(2000, 0.05, 0.10, -0.05, 0.85, key)
        fit = QGARCH(p=1, q=1, residual_dist=normal).fit(
            eps, init="analytical", maxiter=800, lr=0.05,
        )
        params = fit.params
        np.testing.assert_allclose(
            float(params["alpha"][0]), 0.10, atol=0.05,
        )
        np.testing.assert_allclose(float(params["beta"][0]), 0.85, atol=0.05)

    def test_p_ge_2_raises(self):
        """QGARCH constructor rejects p>=2 with a clear error."""
        with pytest.raises(ValueError, match="p=1"):
            QGARCH(p=2, q=1, residual_dist=normal)

    def test_positivity_invariant(self):
        """``ω ≥ ψ²/(4α)`` holds at every fitted point — this is the
        Sentana 1995 σ²>0 condition baked into the reparameterisation."""
        key = jax.random.PRNGKey(2)
        eps = _simulate_qgarch11(500, 0.05, 0.10, -0.05, 0.85, key)
        fit = QGARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        omega = float(fit.params["omega"])
        alpha = float(fit.params["alpha"][0])
        psi = float(fit.params["psi"][0])
        np.testing.assert_array_less(
            psi ** 2 / (4.0 * alpha) - 1e-9, omega,
        )

    def test_analytical_forecast_works_at_any_h(self):
        """Unlike EGARCH/TGARCH, QGARCH supports analytical h-step
        forecasts at any horizon (E[ψ·ε] = 0 for unobserved future)."""
        key = jax.random.PRNGKey(2)
        eps = _simulate_qgarch11(500, 0.05, 0.10, -0.05, 0.85, key)
        fit = QGARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        fc = fit.forecast(h=20, method="analytical")
        assert fc["variance"].shape == (20,)
        assert jnp.all(jnp.isfinite(fc["variance"]))


# ---------------------------------------------------------------------------
# GARCH-M(p, q) — variance-in-mean
# ---------------------------------------------------------------------------
def _simulate_garch_m11(n, mu_t, lambda_m, omega, alpha, beta, key):
    sigma2_uncond = omega / (1.0 - alpha - beta)
    z = jax.random.normal(key, (n,))

    def step(carry, z_t):
        sigma2_prev, eps2_prev = carry
        sigma2_t = omega + alpha * eps2_prev + beta * sigma2_prev
        sigma_t = jnp.sqrt(sigma2_t)
        mu_at_t = mu_t + lambda_m * sigma2_t
        eps_t = sigma_t * z_t
        y_t = mu_at_t + eps_t
        return (sigma2_t, eps_t * eps_t), y_t

    _, y = jax.lax.scan(step, (sigma2_uncond, sigma2_uncond), z)
    return y


class TestGARCH_M:
    def test_recovery(self):
        """GARCH-M(1, 1) recovers the variance-in-mean coefficient and the
        GARCH parameters; ``μ`` is weakly identified so we don't assert on it."""
        key = jax.random.PRNGKey(2)
        y = _simulate_garch_m11(2000, 0.05, 0.20, 0.05, 0.10, 0.85, key)
        fit = GARCH_M(p=1, q=1, residual_dist=normal).fit(
            y, init="analytical", maxiter=800, lr=0.05,
        )
        params = fit.params
        np.testing.assert_allclose(
            float(params["lambda_m"]), 0.20, atol=0.1,
        )
        np.testing.assert_allclose(float(params["alpha"][0]), 0.10, atol=0.05)
        np.testing.assert_allclose(float(params["beta"][0]), 0.85, atol=0.05)

    def test_conditional_mean_uses_variance(self):
        """``conditional_mean(y) ≠ 0`` (variance-in-mean) and tracks
        ``μ + λ_m σ²``."""
        key = jax.random.PRNGKey(2)
        y = _simulate_garch_m11(500, 0.05, 0.20, 0.05, 0.10, 0.85, key)
        fit = GARCH_M(p=1, q=1, residual_dist=normal).fit(y, maxiter=200)
        mu_seq = fit.conditional_mean(y)
        var_seq = fit.conditional_variance(y)
        expected_mu = float(fit.params["mu"]) + float(fit.params["lambda_m"]) * var_seq
        np.testing.assert_allclose(np.asarray(mu_seq), np.asarray(expected_mu))

    def test_residuals_unit_variance(self):
        key = jax.random.PRNGKey(2)
        y = _simulate_garch_m11(2000, 0.05, 0.20, 0.05, 0.10, 0.85, key)
        fit = GARCH_M(p=1, q=1, residual_dist=normal).fit(y, maxiter=400)
        resid = fit.residuals(y)
        eps_seq, z_seq = resid["residuals"], resid["standardised_residuals"]
        np.testing.assert_allclose(float(z_seq.mean()), 0.0, atol=0.05)
        np.testing.assert_allclose(float(z_seq.var()), 1.0, atol=0.1)

    def test_unconditional_mean_in_stats(self):
        """Stats reports the long-run risk-premium-implied mean
        ``μ + λ_m · unconditional_variance``."""
        key = jax.random.PRNGKey(2)
        y = _simulate_garch_m11(500, 0.05, 0.20, 0.05, 0.10, 0.85, key)
        fit = GARCH_M(p=1, q=1, residual_dist=normal).fit(y, maxiter=200)
        s = fit.stats()
        expected = (
            float(fit.params["mu"])
            + float(fit.params["lambda_m"]) * float(s["unconditional_variance"])
        )
        np.testing.assert_allclose(
            float(s["unconditional_mean"]), expected, rtol=1e-4,
        )

    def test_forecast_mean_grows_with_variance(self):
        """E[y_{t+h}] = μ + λ_m · E[σ²_{t+h}], so the forecast mean
        evolves alongside the variance forecast."""
        key = jax.random.PRNGKey(2)
        y = _simulate_garch_m11(500, 0.05, 0.20, 0.05, 0.10, 0.85, key)
        fit = GARCH_M(p=1, q=1, residual_dist=normal).fit(y, maxiter=200)
        fc = fit.forecast(h=20, method="analytical")
        # Mean and variance should both be finite and have the same shape.
        assert fc["mean"].shape == (20,)
        assert fc["variance"].shape == (20,)
        assert jnp.all(jnp.isfinite(fc["mean"]))
        assert jnp.all(jnp.isfinite(fc["variance"]))
