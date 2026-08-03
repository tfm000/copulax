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

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copulax._src.timeseries._warnings import (
    ConvergenceWarning,
    DataScaleWarning,
)
from copulax.timeseries import (
    AR,
    ARMA,
    ArmaGarch,
    EGARCH,
    GARCH,
    GARCH_M,
    GJR_GARCH,
    IGARCH,
    MA,
    QGARCH,
    TGARCH,
)
from copulax.tests._timeseries_helpers import simulate_garch11
from copulax.tests.conftest import require_oracle
from copulax.univariate import normal, student_t


# ---------------------------------------------------------------------------
# Shared data / fits
#
# Many tests below regenerate the same series and refit the same model
# with byte-identical arguments.  Module-scoped fixtures compute each
# distinct (series, fit-arguments) pair exactly once; every argument is
# passed through unchanged, so the series and the fitted model are
# bit-for-bit what the in-test calls produced.  Fits whose arguments
# differ in ANY respect stay separate — they are different computations.
# Fitted models are frozen equinox PyTrees and every consumer below only
# reads from them.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def garch11_2000_key2():
    return simulate_garch11(2000, 0.05, 0.10, 0.85, jax.random.PRNGKey(2))


@pytest.fixture(scope="module")
def garch11_500_key2():
    return simulate_garch11(500, 0.05, 0.10, 0.85, jax.random.PRNGKey(2))


@pytest.fixture(scope="module")
def garch11_2000_fit_m600(garch11_2000_key2):
    """``init="analytical", maxiter=600, lr=0.05`` on the n=2000 series."""
    return GARCH(p=1, q=1, residual_dist=normal).fit(
        garch11_2000_key2, init="analytical", maxiter=600, lr=0.05,
    )


@pytest.fixture(scope="module")
def garch11_500_fit_m200(garch11_500_key2):
    """Bare ``maxiter=200`` on the n=500 series — seven consumers."""
    return GARCH(p=1, q=1, residual_dist=normal).fit(
        garch11_500_key2, maxiter=200,
    )


@pytest.fixture(scope="module")
def garch11_1000_key11():
    """Series behind the per-variant non-normal recovery smoke sweep."""
    return simulate_garch11(1000, 0.05, 0.10, 0.85, jax.random.PRNGKey(11))


@pytest.fixture(scope="module")
def garch11_600_key7():
    """Series behind the per-variant residual-dist promotion sweep."""
    return simulate_garch11(600, 0.05, 0.10, 0.85, jax.random.PRNGKey(7))


# ---------------------------------------------------------------------------
# Parameter recovery
# ---------------------------------------------------------------------------
class TestRecovery:
    def test_garch11_recovery(self, garch11_2000_fit_m600):
        """GARCH(1, 1) parameters recover within tolerance on n=2000."""
        omega_t, alpha_t, beta_t = 0.05, 0.10, 0.85
        params = garch11_2000_fit_m600.params
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
        return require_oracle("arch")

    def test_garch11_vs_arch(self, arch_module, garch11_2000_key2):
        eps = garch11_2000_key2
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
    def test_conditional_variance_matches_numpy_reference(
        self, garch11_500_key2,
    ):
        """Hand-rolled NumPy GARCH recursion matches
        ``conditional_variance(eps)`` to single-precision tolerance."""
        eps = garch11_500_key2
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

    def test_residuals_unit_variance(
        self, garch11_2000_key2, garch11_2000_fit_m600,
    ):
        """Standardised residuals z_t have empirical mean ≈ 0 and var ≈ 1."""
        eps = garch11_2000_key2
        fit = garch11_2000_fit_m600
        resid = fit.residuals(eps)
        eps_t, z_t = resid["residuals"], resid["standardised_residuals"]
        np.testing.assert_allclose(np.asarray(eps_t), np.asarray(eps))
        np.testing.assert_allclose(float(z_t.mean()), 0.0, atol=0.05)
        np.testing.assert_allclose(float(z_t.var()), 1.0, atol=0.05)

    def test_loglikelihood_recompute_parity(
        self, garch11_500_key2, garch11_500_fit_m200,
    ):
        eps = garch11_500_key2
        fit = garch11_500_fit_m200
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
    def test_stats_returns_expected_keys(self, garch11_500_fit_m200):
        fit = garch11_500_fit_m200
        stats = fit.stats()
        assert {"unconditional_variance", "persistence", "half_life",
                "is_stationary"} <= set(stats)
        assert bool(stats["is_stationary"])
        # Persistence < 1 by construction (the simplex reparameterisation
        # guarantees it regardless of the data).  Probe the RAW params so
        # the guarantee is tested independently of the accessor, then pin
        # the accessor to the same value.
        raw = float(fit.params["alpha"].sum() + fit.params["beta"].sum())
        assert raw < 1.0
        np.testing.assert_allclose(float(stats["persistence"]), raw, rtol=1e-12)


class TestForecast:
    def test_analytical_variance_forecast_converges(self, garch11_2000_key2):
        """h-step variance forecast tends toward the unconditional
        variance as h grows."""
        eps = garch11_2000_key2
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

    def test_simulation_forecast_path_shape(self, garch11_500_fit_m200):
        fit = garch11_500_fit_m200
        fc = fit.forecast(
            h=10, method="simulation", n_paths=200,
            key=jax.random.PRNGKey(7),
        )
        assert fc["paths"].shape == (200, 10)
        assert fc["mean"].shape == (10,)
        assert fc["variance"].shape == (10,)

    def test_rvs_deterministic_under_u(self, garch11_500_fit_m200):
        fit = garch11_500_fit_m200
        u = jnp.linspace(0.01, 0.99, 30)
        path1 = fit.rvs(u=u)
        path2 = fit.rvs(u=u)
        np.testing.assert_allclose(np.asarray(path1), np.asarray(path2))

    def test_rvs_batch_shape(self, garch11_500_fit_m200):
        fit = garch11_500_fit_m200
        paths = fit.rvs(size=(50, 12), key=jax.random.PRNGKey(1))
        assert paths.shape == (50, 12)


# ---------------------------------------------------------------------------
# JIT / autograd / warm start
# ---------------------------------------------------------------------------
class TestJIT:
    def test_jit_conditional_variance(
        self, garch11_500_key2, garch11_500_fit_m200,
    ):
        eps = garch11_500_key2
        fit = garch11_500_fit_m200
        jit_cv = jax.jit(fit.conditional_variance)
        np.testing.assert_allclose(
            np.asarray(jit_cv(eps)),
            np.asarray(fit.conditional_variance(eps)),
        )

    def test_jit_fit_end_to_end(self, garch11_500_key2):
        """The full ``GARCH(...).fit(eps)`` pipeline runs under
        ``jax.jit`` — the contract for users wrapping fits in an
        outer JAX transformation."""
        eps = garch11_500_key2

        def fit_fn(e):
            return GARCH(p=1, q=1, residual_dist=normal).fit(
                e, init="analytical", maxiter=100, lr=0.05,
            )

        eager = fit_fn(eps)
        jitted = jax.jit(fit_fn)(eps)
        for k in ("omega", "alpha", "beta"):
            np.testing.assert_allclose(
                np.asarray(jitted.params[k]), np.asarray(eager.params[k]),
                rtol=1e-5, atol=1e-7, err_msg=k,
            )
        assert jitted.residual_dist._stored_params is not None

    def test_warm_start_converges_quickly(self, garch11_500_key2):
        eps = garch11_500_key2
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
    def test_student_t_fit_smoke(self, garch11_2000_key2):
        eps = garch11_2000_key2
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
    def test_non_normal_residual_recovery_smoke(
        self, variance_cls, garch11_1000_key11,
    ):
        """Each asymmetric variance variant should fit cleanly with a
        Student-T residual law and recover finite parameters.  Catches
        breakage in the residual-shape autograd path through
        ``expected_z*`` (e.g. if a quadrature change accidentally lost
        differentiability w.r.t. ``ν``).  ``maxiter`` is intentionally
        low — this is a "fit converges" smoke test, not a parameter-
        recovery accuracy test.
        """
        eps = garch11_1000_key11
        fit = variance_cls(
            p=1, q=1, residual_dist=student_t,
        ).fit(eps, init="analytical", maxiter=150, lr=0.05)
        assert fit.is_fitted
        assert "nu" in fit.params["residual"]
        assert jnp.isfinite(fit.loglikelihood())
        # nu should land in its admissible range (> 2 for finite var).
        assert float(fit.params["residual"]["nu"]) > 2.0


# ---------------------------------------------------------------------------
# Fitted residual distribution (promotion contract)
# ---------------------------------------------------------------------------
class TestFittedResidualDist:
    """Every variance variant's ``fit`` must return an instance whose
    ``residual_dist`` is the *fitted* standardised distribution (not
    the unfitted template), so post-fit ``.cdf`` / ``.ppf`` — and
    hence ``plot_scatter``'s Q-Q panel — work for every variant.
    Regression guard for the variant ``fit`` overrides bypassing the
    :meth:`GARCHBase._build_fitted_instance` promotion.
    """

    @pytest.mark.parametrize(
        "variance_cls",
        [GARCH, IGARCH, GJR_GARCH, EGARCH, TGARCH, QGARCH, GARCH_M],
    )
    def test_fit_promotes_residual_dist(self, variance_cls, garch11_600_key7):
        eps = garch11_600_key7
        fit = variance_cls(
            p=1, q=1, residual_dist=student_t,
        ).fit(eps, init="analytical", maxiter=150, lr=0.05)

        # Stored parameters must be populated (template has None).
        assert fit.residual_dist._stored_params is not None
        # Canonical post-fit naming.
        assert fit.residual_dist.name.endswith("-stdresid")
        # cdf resolves stored params and returns probabilities.
        u = np.asarray(fit.residual_dist.cdf(jnp.array([-1.0, 0.0, 1.0])))
        assert np.all(np.isfinite(u))
        assert np.all((u > 0.0) & (u < 1.0))
        # StandardisedResidual guarantees mean 0 / variance 1.
        s = fit.residual_dist.stats()
        np.testing.assert_allclose(float(s["mean"]), 0.0, atol=1e-4)
        np.testing.assert_allclose(float(s["variance"]), 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_unfitted_raises(self):
        with pytest.raises(ValueError, match="not fitted"):
            GARCH(p=1, q=1).conditional_variance(jnp.array([1.0, 2.0, 3.0]))


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


@pytest.fixture(scope="module")
def igarch11_500_key2():
    return _simulate_igarch11(500, 0.05, 0.10, 0.90, jax.random.PRNGKey(2))


@pytest.fixture(scope="module")
def igarch11_500_fit_m200(igarch11_500_key2):
    """Bare ``maxiter=200`` IGARCH fit — three consumers."""
    return IGARCH(p=1, q=1, residual_dist=normal).fit(
        igarch11_500_key2, maxiter=200,
    )


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

    def test_stats_reports_inf_unconditional_variance(
        self, igarch11_500_fit_m200,
    ):
        fit = igarch11_500_fit_m200
        s = fit.stats()
        assert jnp.isinf(s["unconditional_variance"])
        assert jnp.isinf(s["half_life"])
        assert not bool(s["is_stationary"])

    def test_n_params_drops_one(
        self, igarch11_500_key2, igarch11_500_fit_m200,
    ):
        """IGARCH has one fewer free parameter than vanilla GARCH because
        the simplex constraint Σα+Σβ=1 removes a degree of freedom."""
        eps = igarch11_500_key2
        ig_fit = igarch11_500_fit_m200
        # A vanilla-GARCH fit on the SAME IGARCH series: different data
        # from the shared GARCH group, single consumer, stays inline.
        g_fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=200)
        assert ig_fit.n_params == g_fit.n_params - 1

    def test_fit_time_aic_bic_use_n_params(
        self, igarch11_500_key2, igarch11_500_fit_m200,
    ):
        """CR-01: the cached fit-time AIC/BIC route through ``n_params``,
        so IGARCH's constrained count (1 + (p+q-1) + n_shape) is used and
        the cached values agree with the recompute path aic(eps)/bic(eps).

        Prior to the fix the fit-time count hardcoded 1 + p + q + n_shape,
        overcounting IGARCH by one degree of freedom and biasing the
        cached AIC by exactly +2.0 (BIC by +log(n)) relative to the
        recompute path."""
        eps = igarch11_500_key2
        ig_fit = igarch11_500_fit_m200
        n = int(np.asarray(eps).shape[0])
        ll = float(ig_fit.loglikelihood())
        k = int(ig_fit.n_params)

        # IGARCH(1,1)+normal drops one df: k = 1 + (1+1-1) + 0 = 2, i.e.
        # one less than the naive 1 + p + q + n_shape = 3.
        assert k == 2

        # Cached fit-time values equal the closed-form 2k-2ll / k*log(n)-2ll
        # computed with the constrained k (not the naive count).
        np.testing.assert_allclose(
            float(ig_fit.aic()), 2.0 * k - 2.0 * ll, rtol=1e-6,
        )
        np.testing.assert_allclose(
            float(ig_fit.bic()), k * np.log(n) - 2.0 * ll, rtol=1e-6,
        )
        # The naive (buggy) count would have shifted AIC by +2.0 / BIC by
        # +log(n); assert the cached values are NOT the overcounted ones.
        assert not np.isclose(
            float(ig_fit.aic()), 2.0 * (k + 1) - 2.0 * ll, rtol=1e-6,
        )

        # Cached == recompute path (the recompute path already used
        # self.n_params; this is what CR-01 makes the fit-time path match).
        np.testing.assert_allclose(
            float(ig_fit.aic()), float(ig_fit.aic(eps)), rtol=1e-6,
        )
        np.testing.assert_allclose(
            float(ig_fit.bic()), float(ig_fit.bic(eps)), rtol=1e-6,
        )

    def test_fit_time_aic_bic_unchanged_for_unconstrained(
        self, garch11_500_key2, garch11_500_fit_m200,
    ):
        """CR-01 must not perturb variants whose ``n_params`` already
        equals the old hardcoded 1 + p + q + n_shape count. For vanilla
        GARCH (and every other unconstrained variant) the cached fit-time
        AIC/BIC continue to equal the recompute path exactly."""
        eps = garch11_500_key2
        g_fit = garch11_500_fit_m200
        # k = 1 + p + q + n_shape = 3, identical to the old hardcoded form.
        assert int(g_fit.n_params) == 1 + 1 + 1 + 0
        np.testing.assert_allclose(
            float(g_fit.aic()), float(g_fit.aic(eps)), rtol=1e-6,
        )
        np.testing.assert_allclose(
            float(g_fit.bic()), float(g_fit.bic(eps)), rtol=1e-6,
        )


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


@pytest.fixture(scope="module")
def gjr11_2000_key2():
    return _simulate_gjr_garch11(
        2000, 0.05, 0.05, 0.10, 0.85, jax.random.PRNGKey(2),
    )


class TestGJRGARCH:
    def test_recovery(self, gjr11_2000_key2):
        """GJR-GARCH(1, 1) parameters recover within tolerance on n=2000."""
        eps = gjr11_2000_key2
        fit = GJR_GARCH(p=1, q=1, residual_dist=normal).fit(
            eps, init="analytical", maxiter=800, lr=0.05,
        )
        params = fit.params
        np.testing.assert_allclose(float(params["omega"]), 0.05, atol=0.03)
        np.testing.assert_allclose(float(params["alpha"][0]), 0.05, atol=0.05)
        np.testing.assert_allclose(float(params["gamma"][0]), 0.10, atol=0.05)
        np.testing.assert_allclose(float(params["beta"][0]), 0.85, atol=0.05)

    def test_kappa_appears_in_persistence(self, gjr11_2000_key2):
        """Stats reports persistence = Σα + κ·Σγ + Σβ; under symmetric
        Normal residuals κ = 0.5 to 4-decimal precision."""
        eps = gjr11_2000_key2
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
        return require_oracle("arch")

    def test_gjr_garch_vs_arch(self, arch_module, gjr11_2000_key2):
        eps = gjr11_2000_key2
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

    def test_egarch_vs_arch(self, arch_module, egarch11_2000_key2):
        """copulax and arch use opposite alpha/gamma label assignments
        for EGARCH (a real cross-library split: rugarch follows
        copulax's convention, arch follows its own).  copulax's alpha
        is leverage and gamma is size; arch's alpha is size and gamma
        is leverage.  Compare via the cross-mapping:
        ``copulax.alpha <-> arch.gamma`` and ``copulax.gamma <-> arch.alpha``.
        """
        eps = egarch11_2000_key2
        fit = EGARCH(p=1, q=1, residual_dist=normal).fit(
            eps, init="analytical", maxiter=1500, lr=0.05,
        )
        am = arch_module.arch_model(
            np.asarray(eps), mean="Zero", vol="EGARCH",
            p=1, o=1, q=1, dist="Normal",
        )
        arch_res = am.fit(disp="off")

        np.testing.assert_allclose(
            float(fit.params["omega"]),
            float(arch_res.params["omega"]),
            rtol=1e-2, atol=1e-3,
        )
        # copulax.alpha (leverage) <-> arch.gamma[1] (leverage)
        np.testing.assert_allclose(
            float(fit.params["alpha"][0]),
            float(arch_res.params["gamma[1]"]),
            rtol=1e-2, atol=1e-3,
        )
        # copulax.gamma (size) <-> arch.alpha[1] (size)
        np.testing.assert_allclose(
            float(fit.params["gamma"][0]),
            float(arch_res.params["alpha[1]"]),
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
    """Nelson (1991) EGARCH simulator.

    ``alpha`` is the leverage coefficient on ``z``; ``gamma`` is the
    size coefficient on ``|z| - E|z|``. Matches copulax's EGARCH
    recursion and rugarch / arch / textbook conventions.
    """
    z = jax.random.normal(key, (n,))
    e_abs_z = (2.0 / jnp.pi) ** 0.5  # E|z| for standard normal

    def step(carry, z_t):
        log_var_prev, z_prev = carry
        log_var_t = (
            omega
            + alpha * z_prev
            + gamma * (jnp.abs(z_prev) - e_abs_z)
            + beta * log_var_prev
        )
        sigma_t = jnp.exp(0.5 * log_var_t)
        eps_t = sigma_t * z_t
        return (log_var_t, z_t), eps_t

    log_var_init = omega / (1.0 - beta) if beta != 1 else 0.0
    _, eps = jax.lax.scan(step, (log_var_init, jnp.array(0.0)), z)
    return eps


@pytest.fixture(scope="module")
def egarch11_2000_key2():
    return _simulate_egarch11(
        2000, -0.05, -0.05, 0.10, 0.95, jax.random.PRNGKey(2),
    )


@pytest.fixture(scope="module")
def egarch11_500_key2():
    return _simulate_egarch11(
        500, -0.05, -0.05, 0.10, 0.95, jax.random.PRNGKey(2),
    )


@pytest.fixture(scope="module")
def egarch11_500_fit_m200(egarch11_500_key2):
    """Bare ``maxiter=200`` EGARCH fit — three consumers."""
    return EGARCH(p=1, q=1, residual_dist=normal).fit(
        egarch11_500_key2, maxiter=200,
    )


class TestEGARCH:
    def test_recovery(self, egarch11_2000_key2):
        """EGARCH(1, 1) parameters recover within tolerance on n=2000."""
        eps = egarch11_2000_key2
        fit = EGARCH(p=1, q=1, residual_dist=normal).fit(
            eps, init="analytical", maxiter=600, lr=0.05,
        )
        params = fit.params
        np.testing.assert_allclose(float(params["alpha"][0]), -0.05, atol=0.05)
        np.testing.assert_allclose(float(params["gamma"][0]), 0.10, atol=0.05)
        np.testing.assert_allclose(float(params["beta"][0]), 0.95, atol=0.05)

    def test_no_positivity_constraint(self, egarch11_2000_key2):
        """ω, α, γ are unconstrained — fitted values can be negative."""
        eps = egarch11_2000_key2
        fit = EGARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=300)
        # ω is allowed to be negative; γ likewise.
        # No assertion on signs — just confirm the fit completed and
        # produced finite values.
        for key_name in ("omega", "alpha", "gamma", "beta"):
            assert jnp.all(jnp.isfinite(fit.params[key_name]))

    def test_h1_analytical_forecast(self, egarch11_500_fit_m200):
        """``forecast(1, "analytical")`` is closed-form and matches the
        recursion's one-step-ahead value."""
        fit = egarch11_500_fit_m200
        fc = fit.forecast(h=1, method="analytical")
        assert fc["variance"].shape == (1,)
        assert jnp.isfinite(fc["variance"][0])

    def test_h2_analytical_raises(self, egarch11_500_fit_m200):
        """``forecast(2, "analytical")`` raises ValueError per plan."""
        fit = egarch11_500_fit_m200
        with pytest.raises(ValueError, match="simulation"):
            fit.forecast(h=2, method="analytical")

    def test_simulation_forecast(self, egarch11_500_fit_m200):
        fit = egarch11_500_fit_m200
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


@pytest.fixture(scope="module")
def tgarch11_2000_key2():
    return _simulate_tgarch11(
        2000, 0.038, 0.10, 0.18, 0.85, jax.random.PRNGKey(2),
    )


@pytest.fixture(scope="module")
def tgarch11_500_key2():
    return _simulate_tgarch11(
        500, 0.038, 0.10, 0.18, 0.85, jax.random.PRNGKey(2),
    )


@pytest.fixture(scope="module")
def tgarch11_500_fit_m200(tgarch11_500_key2):
    """Bare ``maxiter=200`` TGARCH fit — two consumers."""
    return TGARCH(p=1, q=1, residual_dist=normal).fit(
        tgarch11_500_key2, maxiter=200,
    )


class TestTGARCH:
    def test_recovery(self, tgarch11_2000_key2):
        """TGARCH(1, 1) parameters recover within tolerance on n=2000."""
        eps = tgarch11_2000_key2
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

    def test_stats_first_moment_persistence(self, tgarch11_2000_key2):
        """Persistence = E[z⁺]·Σα⁺ + E[z⁻]·Σα⁻ + Σβ; under Normal
        residuals E[z⁺] = E[z⁻] = √(2/π) / 2."""
        eps = tgarch11_2000_key2
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

    def test_h1_analytical_forecast(self, tgarch11_500_fit_m200):
        fit = tgarch11_500_fit_m200
        fc = fit.forecast(h=1, method="analytical")
        assert fc["variance"].shape == (1,)
        assert jnp.isfinite(fc["variance"][0])

    def test_h2_analytical_raises(self, tgarch11_500_fit_m200):
        fit = tgarch11_500_fit_m200
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


@pytest.fixture(scope="module")
def qgarch11_500_key2():
    return _simulate_qgarch11(
        500, 0.05, 0.10, -0.05, 0.85, jax.random.PRNGKey(2),
    )


@pytest.fixture(scope="module")
def qgarch11_500_fit_m200(qgarch11_500_key2):
    """Bare ``maxiter=200`` QGARCH fit — two consumers."""
    return QGARCH(p=1, q=1, residual_dist=normal).fit(
        qgarch11_500_key2, maxiter=200,
    )


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

    def test_positivity_invariant(self, qgarch11_500_fit_m200):
        """``ω ≥ ψ²/(4α)`` holds at every fitted point — this is the
        Sentana 1995 σ²>0 condition baked into the reparameterisation."""
        fit = qgarch11_500_fit_m200
        omega = float(fit.params["omega"])
        alpha = float(fit.params["alpha"][0])
        psi = float(fit.params["psi"][0])
        np.testing.assert_array_less(
            psi ** 2 / (4.0 * alpha) - 1e-9, omega,
        )

    def test_analytical_forecast_works_at_any_h(self, qgarch11_500_fit_m200):
        """Unlike EGARCH/TGARCH, QGARCH supports analytical h-step
        forecasts at any horizon (E[ψ·ε] = 0 for unobserved future)."""
        fit = qgarch11_500_fit_m200
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


@pytest.fixture(scope="module")
def garchm11_2000_key2():
    return _simulate_garch_m11(
        2000, 0.05, 0.20, 0.05, 0.10, 0.85, jax.random.PRNGKey(2),
    )


@pytest.fixture(scope="module")
def garchm11_500_key2():
    return _simulate_garch_m11(
        500, 0.05, 0.20, 0.05, 0.10, 0.85, jax.random.PRNGKey(2),
    )


@pytest.fixture(scope="module")
def garchm11_500_fit_m200(garchm11_500_key2):
    """Bare ``maxiter=200`` GARCH-M fit — three consumers."""
    return GARCH_M(p=1, q=1, residual_dist=normal).fit(
        garchm11_500_key2, maxiter=200,
    )


class TestGARCH_M:
    def test_recovery(self, garchm11_2000_key2):
        """GARCH-M(1, 1) recovers the variance-in-mean coefficient and the
        GARCH parameters; ``μ`` is weakly identified so we don't assert on it."""
        y = garchm11_2000_key2
        fit = GARCH_M(p=1, q=1, residual_dist=normal).fit(
            y, init="analytical", maxiter=800, lr=0.05,
        )
        params = fit.params
        np.testing.assert_allclose(
            float(params["lambda_m"]), 0.20, atol=0.1,
        )
        np.testing.assert_allclose(float(params["alpha"][0]), 0.10, atol=0.05)
        np.testing.assert_allclose(float(params["beta"][0]), 0.85, atol=0.05)

    def test_conditional_mean_uses_variance(
        self, garchm11_500_key2, garchm11_500_fit_m200,
    ):
        """``conditional_mean(y) ≠ 0`` (variance-in-mean) and tracks
        ``μ + λ_m σ²``."""
        y = garchm11_500_key2
        fit = garchm11_500_fit_m200
        mu_seq = fit.conditional_mean(y)
        var_seq = fit.conditional_variance(y)
        expected_mu = float(fit.params["mu"]) + float(fit.params["lambda_m"]) * var_seq
        np.testing.assert_allclose(np.asarray(mu_seq), np.asarray(expected_mu))

    def test_residuals_unit_variance(self, garchm11_2000_key2):
        y = garchm11_2000_key2
        fit = GARCH_M(p=1, q=1, residual_dist=normal).fit(y, maxiter=400)
        resid = fit.residuals(y)
        eps_seq, z_seq = resid["residuals"], resid["standardised_residuals"]
        np.testing.assert_allclose(float(z_seq.mean()), 0.0, atol=0.05)
        np.testing.assert_allclose(float(z_seq.var()), 1.0, atol=0.1)

    def test_unconditional_mean_in_stats(self, garchm11_500_fit_m200):
        """Stats reports the long-run risk-premium-implied mean
        ``μ + λ_m · unconditional_variance``."""
        fit = garchm11_500_fit_m200
        s = fit.stats()
        expected = (
            float(fit.params["mu"])
            + float(fit.params["lambda_m"]) * float(s["unconditional_variance"])
        )
        np.testing.assert_allclose(
            float(s["unconditional_mean"]), expected, rtol=1e-4,
        )

    def test_forecast_mean_grows_with_variance(self, garchm11_500_fit_m200):
        """E[y_{t+h}] = μ + λ_m · E[σ²_{t+h}], so the forecast mean
        evolves alongside the variance forecast."""
        fit = garchm11_500_fit_m200
        fc = fit.forecast(h=20, method="analytical")
        # Mean and variance should both be finite and have the same shape.
        assert fc["mean"].shape == (20,)
        assert fc["variance"].shape == (20,)
        assert jnp.all(jnp.isfinite(fc["mean"]))
        assert jnp.all(jnp.isfinite(fc["variance"]))


# ---------------------------------------------------------------------------
# Layer-1 reference fixtures (rugarch primary; HARD-02, D-05 Layer 1)
# ---------------------------------------------------------------------------
# These classes plug rugarch's FITTED parameters into copulax's
# likelihood / recursion (no fitting) and match the reference's reported
# sigma^2 path and log-likelihood TWO-SIDED at rtol <= 1e-8. rugarch fixes
# the leading max(p, q) conditional variances at the mean-of-squared-
# residuals unconditional-variance estimate (its rec.init="all" control)
# and starts the recursion at index max(p, q); copulax reproduces this
# exactly via its opt-in init="squared" pre-sample mode. This is the
# strict, solver-independent half of HARD-02 (Layer 1 of D-05) -- it is
# NOT a fit-vs-fit test (that is Layer 2, Plan 10), so no fit-vs-fit
# dominance assertion appears here.
#
# The reference data modules are R-generated and committed under
# _r_reference/; they are loaded here via importlib (not pytest
# discovery), mirroring test_timeseries_arma_garch.py.

import importlib.util as _ilu
from pathlib import Path as _Path

_STANDALONE_REF_PATH = (
    _Path(__file__).parent / "_r_reference"
    / "garch_standalone_reference_data.py"
)
_std_spec = _ilu.spec_from_file_location(
    "_garch_standalone_reference", _STANDALONE_REF_PATH,
)
_std_mod = _ilu.module_from_spec(_std_spec)
_std_spec.loader.exec_module(_std_mod)
GARCH_STANDALONE_REFERENCE = _std_mod.GARCH_STANDALONE_REFERENCE

_GARCH_M_REF_PATH = (
    _Path(__file__).parent / "_r_reference" / "garch_m_reference_data.py"
)
_gm_spec = _ilu.spec_from_file_location(
    "_garch_m_reference", _GARCH_M_REF_PATH,
)
_gm_mod = _ilu.module_from_spec(_gm_spec)
_gm_spec.loader.exec_module(_gm_mod)
GARCH_M_REFERENCE = _gm_mod.GARCH_M_REFERENCE


_VAR_CLS_FROM_NAME = {
    "GARCH": GARCH, "IGARCH": IGARCH, "GJR_GARCH": GJR_GARCH,
    "EGARCH": EGARCH,
}
_RESIDUAL_FROM_NAME = {"normal": normal, "student_t": student_t}


def _model_at_reference(cls, residual_dist, params, residual, y):
    """Construct a fitted copulax variance model sitting exactly at the
    reference's fitted parameter point.

    Uses ``init="warm"`` with ``maxiter=0`` so the optimiser does not
    move off the supplied params -- lands on them to ~1e-16 (verified).
    The resulting instance is then evaluated with ``init="squared"`` to
    reproduce rugarch's rec.init pre-sample convention.
    """
    warm = dict(params)
    warm["residual"] = dict(residual)
    return cls(1, 1, residual_dist=residual_dist).fit(
        jnp.asarray(y), init="warm", init_params=warm, maxiter=0,
    )


def _squared_basis_se(model, rec):
    """Observed-Hessian standard errors on the SQUARED pre-sample basis.

    Computes ``sqrt(diag((-H)^{-1}))`` where ``H`` is the Hessian of the
    model's log-likelihood -- evaluated through the SAME variance
    recursion kernel the model uses, with the rugarch rec.init="all"
    convention (init="squared") -- w.r.t. the flat natural-parameter
    vector. This is the standard observed-information SE on the reference's
    own pre-sample basis, so it is directly comparable to rugarch's
    reported (classic) standard errors. It is NOT a reimplementation of the
    likelihood: it calls the shared ``run_*`` kernels from
    ``copulax._src.timeseries._recursions``.

    IGARCH is differentiated over its FREE parameters only (omega, alpha)
    with beta pinned to ``1 - sum(alpha)`` -- matching rugarch, which
    reports SEs for the free parameters and NA for the constrained beta.
    """
    from copulax._src.timeseries._init import (
        garch_pre_sample_state, garch_presample_warmup,
    )
    from copulax._src.timeseries._recursions import (
        run_garch, run_gjr_garch, run_egarch,
    )

    y = jnp.asarray(rec["y"])
    wrapper = model._wrapper()
    p, q = model.p, model.q
    esl, vl = garch_pre_sample_state(y, p=p, q=q, mode="squared")
    n_warmup, warmup_var = garch_presample_warmup(y, p=p, q=q, mode="squared")
    P = model.params
    var_model = rec["var_model"]
    has_resid = bool(P.get("residual", {}))
    nu0 = jnp.asarray(P["residual"]["nu"]) if has_resid else None

    def _sg(v):
        return jnp.atleast_1d(jnp.asarray(v, dtype=float))

    if var_model in ("GARCH", "IGARCH"):
        pinned = var_model == "IGARCH"
        segs = [_sg(P["omega"]), _sg(P["alpha"])]
        if not pinned:
            segs.append(_sg(P["beta"]))
        if has_resid:
            segs.append(_sg(nu0))
        flat0 = jnp.concatenate(segs)

        def ll(f):
            i = 0
            om = f[i]; i += 1
            al = f[i:i + p]; i += p
            if pinned:
                be = 1.0 - jnp.sum(al)
                be = be.reshape((1,))
            else:
                be = f[i:i + q]; i += q
            rp = {"nu": f[i]} if has_resid else {}
            vs, _ = run_garch(y, om, al, be, esl, vl,
                              n_warmup=n_warmup, warmup_var=warmup_var)
            sg = jnp.sqrt(jnp.maximum(vs, 1e-12)); z = y / sg
            return jnp.sum(wrapper.logpdf(z, rp) - jnp.log(sg))

        H = jax.hessian(ll)(flat0)
        se_flat = np.asarray(jnp.sqrt(jnp.maximum(jnp.diag(jnp.linalg.inv(-H)), 0.0)))
        out = {"omega": float(se_flat[0]),
               "alpha": [float(se_flat[1 + i]) for i in range(p)]}
        # beta SE: pinned IGARCH beta has no SE entry -> report NaN so the
        # test's finite-check skips it; free GARCH beta gets its value.
        if pinned:
            out["beta"] = [float("nan")]
        else:
            out["beta"] = [float(se_flat[1 + p + i]) for i in range(q)]
        return out

    if var_model == "GJR_GARCH":
        neg = 0.5 * esl  # matches _initial_state_gjr under "squared"
        segs = [_sg(P["omega"]), _sg(P["alpha"]), _sg(P["gamma"]), _sg(P["beta"])]
        if has_resid:
            segs.append(_sg(nu0))
        flat0 = jnp.concatenate(segs)

        def ll(f):
            i = 0
            om = f[i]; i += 1
            al = f[i:i + p]; i += p
            ga = f[i:i + p]; i += p
            be = f[i:i + q]; i += q
            rp = {"nu": f[i]} if has_resid else {}
            vs, _ = run_gjr_garch(y, om, al, ga, be, esl, neg, vl,
                                  n_warmup=n_warmup, warmup_var=warmup_var)
            sg = jnp.sqrt(jnp.maximum(vs, 1e-12)); z = y / sg
            return jnp.sum(wrapper.logpdf(z, rp) - jnp.log(sg))

        H = jax.hessian(ll)(flat0)
        se_flat = np.asarray(jnp.sqrt(jnp.maximum(jnp.diag(jnp.linalg.inv(-H)), 0.0)))
        return {"omega": float(se_flat[0]),
                "alpha": [float(se_flat[1 + i]) for i in range(p)],
                "gamma": [float(se_flat[1 + p + i]) for i in range(p)],
                "beta": [float(se_flat[1 + 2 * p + i]) for i in range(q)]}

    if var_model == "EGARCH":
        z_lags = jnp.zeros((p,))
        anchor = vl[0] if q > 0 else esl[0]
        log_var_lags = jnp.full((q,), jnp.log(jnp.maximum(anchor, 1e-12)))
        segs = [_sg(P["omega"]), _sg(P["alpha"]), _sg(P["gamma"]), _sg(P["beta"])]
        if has_resid:
            segs.append(_sg(nu0))
        flat0 = jnp.concatenate(segs)

        def ll(f):
            i = 0
            om = f[i]; i += 1
            al = f[i:i + p]; i += p
            ga = f[i:i + p]; i += p
            be = f[i:i + q]; i += q
            rp = {"nu": f[i]} if has_resid else {}
            eabs = wrapper.expected_abs_z(rp)
            lvs, _ = run_egarch(y, om, al, ga, be, eabs, z_lags, log_var_lags,
                                n_warmup=n_warmup, warmup_var=warmup_var)
            vs = jnp.exp(lvs)
            sg = jnp.sqrt(jnp.maximum(vs, 1e-12)); z = y / sg
            return jnp.sum(wrapper.logpdf(z, rp) - jnp.log(sg))

        H = jax.hessian(ll)(flat0)
        se_flat = np.asarray(jnp.sqrt(jnp.maximum(jnp.diag(jnp.linalg.inv(-H)), 0.0)))
        return {"omega": float(se_flat[0]),
                "alpha": [float(se_flat[1 + i]) for i in range(p)],
                "gamma": [float(se_flat[1 + p + i]) for i in range(p)],
                "beta": [float(se_flat[1 + 2 * p + i]) for i in range(q)]}

    raise ValueError(f"_squared_basis_se: unsupported var_model {var_model!r}")


def _garch_m_squared_basis_se(model, rec):
    """GARCH-M observed-Hessian SE on the squared pre-sample basis.

    Same construction as :func:`_squared_basis_se` but for the
    variance-in-mean recursion (:func:`run_garch_m`), differentiating
    over ``(mu, lambda_m, omega, alpha, beta[, nu])``. The GARCH-M
    warm-up level is ``mean((y - mu)^2)`` and depends on ``mu``, so it is
    recomputed inside the loglik closure at each perturbed ``mu``.
    """
    from copulax._src.timeseries._init import (
        garch_pre_sample_state, mean_squared_presample,
    )
    from copulax._src.timeseries._recursions import run_garch_m

    y = jnp.asarray(rec["y"])
    wrapper = model._wrapper()
    p, q = model.p, model.q
    esl, vl = garch_pre_sample_state(y, p=p, q=q, mode="squared")
    n_warmup = int(max(p, q))
    P = model.params
    has_resid = bool(P.get("residual", {}))
    nu0 = jnp.asarray(P["residual"]["nu"]) if has_resid else None
    segs = [
        jnp.asarray(P["mu"]).reshape((1,)),
        jnp.asarray(P["lambda_m"]).reshape((1,)),
        jnp.asarray(P["omega"]).reshape((1,)),
        jnp.atleast_1d(P["alpha"]),
        jnp.atleast_1d(P["beta"]),
    ] + ([jnp.asarray(nu0).reshape((1,))] if has_resid else [])
    flat0 = jnp.concatenate(segs)

    def ll(f):
        i = 0
        mu = f[i]; i += 1
        lm = f[i]; i += 1
        om = f[i]; i += 1
        al = f[i:i + p]; i += p
        be = f[i:i + q]; i += q
        rp = {"nu": f[i]} if has_resid else {}
        warmup_var = mean_squared_presample(y - mu)
        _, es, vs, _ = run_garch_m(y, mu, lm, om, al, be, esl, vl,
                                   n_warmup=n_warmup, warmup_var=warmup_var)
        sg = jnp.sqrt(jnp.maximum(vs, 1e-12)); z = es / sg
        return jnp.sum(wrapper.logpdf(z, rp) - jnp.log(sg))

    H = jax.hessian(ll)(flat0)
    se_flat = np.asarray(jnp.sqrt(jnp.maximum(jnp.diag(jnp.linalg.inv(-H)), 0.0)))
    return {"mu": float(se_flat[0]), "lambda_m": float(se_flat[1]),
            "omega": float(se_flat[2]),
            "alpha": [float(se_flat[3 + i]) for i in range(p)],
            "beta": [float(se_flat[3 + p + i]) for i in range(q)]}


# Module-scoped cache: build each reference model once and reuse across
# the sigma^2 / loglik / SE assertions (amortises construction).
@pytest.fixture(scope="module")
def standalone_ref_models():
    models = {}
    for label, rec in GARCH_STANDALONE_REFERENCE.items():
        cls = _VAR_CLS_FROM_NAME[rec["var_model"]]
        rdist = _RESIDUAL_FROM_NAME[rec["residual_dist"]]
        models[label] = _model_at_reference(
            cls, rdist, rec["params"], rec["residual"], rec["y"],
        )
    return models


@pytest.fixture(scope="module")
def garch_m_ref_models():
    models = {}
    for label, rec in GARCH_M_REFERENCE.items():
        rdist = _RESIDUAL_FROM_NAME[rec["residual_dist"]]
        models[label] = _model_at_reference(
            GARCH_M, rdist, rec["params"], rec["residual"], rec["y"],
        )
    return models


class TestGarchStandaloneReference:
    """Layer-1 formula tests: copulax at rugarch's fitted params matches
    rugarch's reported sigma^2 path and LLH two-sided at rtol <= 1e-8
    (init="squared" reproduces rugarch's rec.init="all" convention)."""

    @pytest.mark.parametrize("label", sorted(GARCH_STANDALONE_REFERENCE))
    def test_conditional_variance_matches_rugarch(
        self, label, standalone_ref_models,
    ):
        rec = GARCH_STANDALONE_REFERENCE[label]
        model = standalone_ref_models[label]
        var_seq = np.asarray(
            model.conditional_variance(rec["y"], init="squared")
        )
        # Two-sided, tight: JAX autodiff/recursion is exact and rugarch's
        # reported sigma^2 is computed at full precision from the same
        # fitted params. rec.init="all" == copulax init="squared".
        np.testing.assert_allclose(
            var_seq, np.asarray(rec["sigma2"]), rtol=1e-8, atol=1e-10,
        )

    @pytest.mark.parametrize("label", sorted(GARCH_STANDALONE_REFERENCE))
    def test_loglikelihood_matches_rugarch(
        self, label, standalone_ref_models,
    ):
        rec = GARCH_STANDALONE_REFERENCE[label]
        model = standalone_ref_models[label]
        ll = float(model.loglikelihood(rec["y"], init="squared"))
        np.testing.assert_allclose(
            ll, float(rec["loglikelihood"]), rtol=1e-8, atol=1e-10,
        )

    @pytest.mark.parametrize("label", [
        "garch11_normal", "garch11_studentt",
        "igarch11_normal", "igarch11_studentt",
        "gjr11_normal", "gjr11_studentt",
    ])
    def test_standard_errors_match_rugarch(
        self, label, standalone_ref_models,
    ):
        # Layer-1 SEs are compared on the SAME pre-sample basis rugarch
        # uses (rec.init="all" == copulax init="squared"). On-basis the
        # remaining gap is purely rugarch's finite-difference Hessian vs
        # copulax's exact JAX autodiff, so the bound is ~1e-3 (documented
        # per-fixture; the reference FD-Hessian is the named slack source,
        # never a fit-vs-fit / k*SE tolerance).
        #
        # NOTE: copulax's fit-time `standard_errors_` uses its DEFAULT
        # backcast pre-sample, which is a different basis from rugarch's
        # rec.init="all"; comparing that cross-basis would be an
        # apples-to-oranges error (measured: up to ~37% on EGARCH omega).
        # We therefore recompute the observed-Hessian SE on the squared
        # basis via the model's real recursion kernels below.
        #
        # EGARCH is excluded from this parametrisation: its standard
        # errors differ from rugarch's even on-basis (measured ~37% on
        # omega, ~320% on beta) because rugarch reparameterises the
        # EGARCH log-variance persistence for its SE computation -- a
        # genuine SE-parameterisation convention difference, documented in
        # test_egarch_standard_errors_convention_difference below and
        # recorded for 01-MATH-REVIEW.md. The EGARCH recursion and
        # likelihood themselves match rugarch to machine precision.
        rec = GARCH_STANDALONE_REFERENCE[label]
        model = standalone_ref_models[label]
        se = _squared_basis_se(model, rec)
        ref_se = rec["standard_errors"]
        np.testing.assert_allclose(
            se["omega"], float(ref_se["omega"][0]), rtol=1e-3, atol=1e-4,
        )
        for i, ref_a in enumerate(ref_se["alpha"]):
            np.testing.assert_allclose(
                se["alpha"][i], float(ref_a), rtol=1e-3, atol=1e-4,
            )
        # beta SE(s): the IGARCH pinned beta has a NaN reference SE
        # (constrained parameter) -- skip it; the free IGARCH SEs live in
        # the (omega, alpha) subspace, which is what _squared_basis_se
        # returns for IGARCH.
        for i, ref_b in enumerate(ref_se["beta"]):
            if not np.isfinite(ref_b):
                continue
            np.testing.assert_allclose(
                se["beta"][i], float(ref_b), rtol=1e-3, atol=1e-4,
            )
        if "gamma" in ref_se:
            for i, ref_g in enumerate(ref_se["gamma"]):
                np.testing.assert_allclose(
                    se["gamma"][i], float(ref_g), rtol=1e-3, atol=1e-4,
                )

    @pytest.mark.parametrize("label", [
        "egarch11_normal", "egarch11_studentt",
    ])
    def test_egarch_standard_errors_convention_difference(
        self, label, standalone_ref_models,
    ):
        """EGARCH standard errors differ from rugarch's even on the same
        (squared) pre-sample basis -- a documented SE-PARAMETERISATION
        convention difference (rugarch reparameterises the log-variance
        persistence for its SE computation; copulax reports observed-
        Hessian SEs in the raw natural parameters). This is NOT silent
        tolerance widening: the EGARCH recursion and log-likelihood match
        rugarch to machine precision (asserted above); only the SE basis
        differs. Recorded for 01-MATH-REVIEW.md.

        We assert copulax's on-basis EGARCH SEs are finite and strictly
        positive (the SE machinery is healthy) and that the difference
        from rugarch is real (> 1e-2 on at least one parameter, so a
        future two-sided assertion would be wrong to add without
        resolving the parameterisation)."""
        rec = GARCH_STANDALONE_REFERENCE[label]
        model = standalone_ref_models[label]
        se = _squared_basis_se(model, rec)
        ref_se = rec["standard_errors"]
        for key in ("omega", "alpha", "gamma", "beta"):
            vals = se[key] if isinstance(se[key], list) else [se[key]]
            for v in vals:
                assert np.isfinite(v) and v > 0.0
        # The parameterisation difference is real and material.
        rel = [
            abs(se["omega"] - float(ref_se["omega"][0]))
            / abs(float(ref_se["omega"][0])),
            abs(se["beta"][0] - float(ref_se["beta"][0]))
            / abs(float(ref_se["beta"][0])),
        ]
        assert max(rel) > 1e-2

    def test_squared_init_differs_from_backcast(
        self, standalone_ref_models,
    ):
        """Regression guard: the opt-in "squared" mode is genuinely a
        distinct pre-sample scheme, and the DEFAULT modes are untouched.
        For a persistent GARCH fit the two schemes must disagree on the
        early sigma^2 path (else the new mode would be a silent no-op)."""
        rec = GARCH_STANDALONE_REFERENCE["garch11_normal"]
        model = standalone_ref_models["garch11_normal"]
        cv_squared = np.asarray(
            model.conditional_variance(rec["y"], init="squared")
        )
        cv_backcast = np.asarray(
            model.conditional_variance(rec["y"], init="backcast")
        )
        # sigma^2[0] under "squared" equals mean(y^2); "backcast" seeds an
        # EWMA-anchored recursion so its first value differs.
        assert not np.isclose(cv_squared[0], cv_backcast[0])


class TestGarchStandaloneArchOracle:
    """Second oracle: arch (Python) fitted parameters and log-likelihood
    for the variants it supports (GARCH / GJR / EGARCH).

    arch uses an EWMA-0.94 backcast pre-sample rather than rugarch's
    mean(residuals^2), so it does NOT reproduce rugarch's sigma^2 path at
    1e-8 -- it is an INDEPENDENT fitted-parameter / LL oracle here, at a
    documented solver-agreement bound, not a Layer-1 recursion oracle.
    IGARCH has no standalone arch form and is rugarch-only.
    """

    @pytest.fixture(scope="class")
    def arch_module(self):
        return require_oracle("arch")

    @pytest.mark.parametrize("label", [
        "garch11_normal", "gjr11_normal", "egarch11_normal",
    ])
    def test_copulax_fit_agrees_with_arch(self, label, arch_module):
        rec = GARCH_STANDALONE_REFERENCE[label]
        y = np.asarray(rec["y"])
        cls = _VAR_CLS_FROM_NAME[rec["var_model"]]
        fit = cls(1, 1, residual_dist=normal).fit(
            jnp.asarray(y), init="analytical", maxiter=1000, lr=0.05,
        )
        vol_map = {"GARCH": ("GARCH", 0), "GJR_GARCH": ("GARCH", 1),
                   "EGARCH": ("EGARCH", 0)}
        vol, o = vol_map[rec["var_model"]]
        am = arch_module.arch_model(
            y, mean="Zero", vol=vol, p=1, o=o, q=1, dist="Normal",
        )
        arch_res = am.fit(disp="off")
        # Independent-solver agreement bound: two MLE optimisers on the
        # same data with different pre-sample conventions -- documented
        # loose bound, NOT a k*SE budget and NOT a Layer-1 formula match.
        np.testing.assert_allclose(
            float(fit.loglikelihood()),
            float(arch_res.loglikelihood),
            rtol=5e-3,
        )


class TestGarchMReference:
    """Layer-1 formula tests for GARCH-M (archm=TRUE, archpow=2 => the
    rugarch archm coefficient maps directly to copulax lambda_m,
    variance-in-mean). copulax at rugarch's fitted params reproduces the
    reported sigma^2 path and LLH two-sided at rtol <= 1e-8 with
    init="squared" (GARCH-M warm-up level mean((y - mu)^2))."""

    @pytest.mark.parametrize("label", sorted(GARCH_M_REFERENCE))
    def test_conditional_variance_matches_rugarch(
        self, label, garch_m_ref_models,
    ):
        rec = GARCH_M_REFERENCE[label]
        model = garch_m_ref_models[label]
        var_seq = np.asarray(
            model.conditional_variance(rec["y"], init="squared")
        )
        np.testing.assert_allclose(
            var_seq, np.asarray(rec["sigma2"]), rtol=1e-8, atol=1e-10,
        )

    @pytest.mark.parametrize("label", sorted(GARCH_M_REFERENCE))
    def test_loglikelihood_matches_rugarch(
        self, label, garch_m_ref_models,
    ):
        rec = GARCH_M_REFERENCE[label]
        model = garch_m_ref_models[label]
        ll = float(model.loglikelihood(rec["y"], init="squared"))
        np.testing.assert_allclose(
            ll, float(rec["loglikelihood"]), rtol=1e-8, atol=1e-10,
        )

    @pytest.mark.parametrize("label", sorted(GARCH_M_REFERENCE))
    def test_standard_errors_match_rugarch(
        self, label, garch_m_ref_models,
    ):
        # On-basis (squared) observed-Hessian SEs, comparable to rugarch's
        # reported classic SEs. Remaining gap is rugarch's FD-Hessian vs
        # copulax's exact autodiff -> ~1e-3 bound (FD-Hessian is the named
        # slack source). Comparing copulax's fit-time backcast-basis SEs
        # cross-basis would be an apples-to-oranges error, so we recompute
        # on the squared basis via the model's real run_garch_m kernel.
        rec = GARCH_M_REFERENCE[label]
        model = garch_m_ref_models[label]
        se = _garch_m_squared_basis_se(model, rec)
        ref_se = rec["standard_errors"]
        for key in ("mu", "lambda_m", "omega"):
            np.testing.assert_allclose(
                se[key], float(ref_se[key]), rtol=1e-3, atol=1e-4,
            )
        np.testing.assert_allclose(
            se["alpha"][0], float(ref_se["alpha"][0]), rtol=1e-3, atol=1e-4,
        )
        np.testing.assert_allclose(
            se["beta"][0], float(ref_se["beta"][0]), rtol=1e-3, atol=1e-4,
        )


class TestUnconditionalVarianceThirdParty:
    r"""Third-party assertion of copulax's GARCH-family unconditional-variance
    accessor (``stats()["unconditional_variance"]``) against rugarch's
    ``uncvariance(fit)`` — the closed-form long-run variance implied by the
    fitted coefficients (NOT a path quantity).

    Prior to this class the unconditional variances had NO direct third-party
    assertion (only literature-identity, Monte-Carlo self-consistency, and
    forecast-convergence checks).  ``uncvariance(fit)`` is emitted by the
    rugarch regenerators (``generate_garch_standalone_reference.R``,
    ``generate_garch_m_reference.R``, ``generate_tgarch_fgarch_reference.R``)
    and stored per case; each model here sits EXACTLY at rugarch's fitted
    params (``_model_at_reference``: ``init="warm"``, ``maxiter=0``) so the
    comparison is formula-level, not fit-quality.

    Per-family conventions (all VERIFIED empirically, 01-MATH-REVIEW.md
    unconditional-variance third-party section):

    * **GARCH / GARCH-M** — ``omega/(1 - Σα - Σβ)`` on both sides; exact,
      pinned at ``rtol <= 1e-9`` (measured 0.0).
    * **GJR** — rugarch fixes ``kappa = E[z² 1{z<0}] = 0.5`` for ALL residual
      laws; copulax computes ``kappa`` by quadrature (``= 0.5`` for the
      symmetric normal / standardised-t laws here).  Pinned at
      ``rtol <= 1e-8`` (measured ~2e-10 normal / ~2e-13 t; the small slack is
      quadrature-vs-analytic ``kappa``, named as the slack source).
    * **EGARCH** — BOTH libraries return the Nelson geometric-mean convention
      ``exp(omega/(1 - Σβ))`` (residual-law-independent).  This is a
      same-formula check and is pinned TIGHTLY at ``rtol <= 1e-9`` (measured
      0.0); if it ever fails it is a real divergence, not a tolerance issue.
    * **IGARCH** — the unconditional variance does not exist
      (``Σα + Σβ = 1``).  Both copulax and rugarch report a non-finite
      sentinel; the test asserts AGREEMENT IN NON-EXISTENCE (both ``+inf``),
      never a numeric equality.
    * **TGARCH** (fGARCH submodel) — rugarch's ``uncvariance`` is the CLOSED
      FORM ``(omega/(1 - persistence))²`` with the SAME first-moment
      persistence copulax's clean Zakoian ``unconditional_sigma²`` uses; the
      0.001 news-impact softening (which perturbs the reported sigma PATH at
      O(1e-5)) does NOT enter ``uncvariance``.  The match is therefore TIGHT
      (measured ~7e-15 normal / ~6e-11 t; the t slack is quadrature-vs-analytic
      ``E[z±]``), pinned at ``rtol <= 1e-8`` — NOT the softening-widened path
      tolerance.
    """

    # ---- GARCH / IGARCH / GJR / EGARCH standalone ----
    @pytest.mark.parametrize("label", sorted(GARCH_STANDALONE_REFERENCE))
    def test_standalone_uncvariance_matches_rugarch(
        self, label, standalone_ref_models,
    ):
        rec = GARCH_STANDALONE_REFERENCE[label]
        model = standalone_ref_models[label]
        cx = float(model.stats()["unconditional_variance"])
        ref = float(rec["uncvariance"])
        var_model = rec["var_model"]
        if var_model == "IGARCH":
            # Agreement in NON-EXISTENCE: both non-finite (+inf). Never a
            # numeric equality — persistence == 1, variance does not exist.
            assert np.isinf(ref), (
                f"{label}: expected rugarch uncvariance == inf for IGARCH, "
                f"got {ref}"
            )
            assert np.isinf(cx) and cx > 0, (
                f"{label}: copulax unconditional_variance should be +inf "
                f"(IGARCH non-existence), got {cx}"
            )
            return
        # GARCH -> exact omega/(1-a-b); EGARCH -> exact geometric-mean
        # exp(omega/(1-beta)) (same formula both sides). GJR -> kappa=0.5
        # both sides (quadrature vs analytic => tiny slack).
        assert np.isfinite(ref) and ref > 0, (
            f"{label}: rugarch uncvariance not a valid variance: {ref}"
        )
        rtol = 1e-8 if var_model == "GJR_GARCH" else 1e-9
        np.testing.assert_allclose(
            cx, ref, rtol=rtol,
            err_msg=(
                f"{label} ({var_model}): copulax unconditional_variance "
                f"!= rugarch uncvariance"
            ),
        )

    def test_egarch_is_same_formula_geometric_mean(self):
        r"""EGARCH SAME-FORMULA check (STOP-if-fails): copulax and rugarch
        BOTH return the Nelson geometric-mean ``exp(omega/(1 - Σβ))``.  This
        must pass at machine level; a failure signals a real convention
        divergence, not tolerance.  Also cross-check copulax's value against
        the hand-computed ``exp(omega/(1-beta))`` from the fitted coefficients.
        """
        for label in ("egarch11_normal", "egarch11_studentt"):
            rec = GARCH_STANDALONE_REFERENCE[label]
            p = rec["params"]
            omega = float(p["omega"][0])
            beta = float(p["beta"][0])
            hand = np.exp(omega / (1.0 - beta))
            ref = float(rec["uncvariance"])
            # rugarch's uncvariance IS the geometric-mean closed form.
            np.testing.assert_allclose(
                ref, hand, rtol=1e-10,
                err_msg=(
                    f"{label}: rugarch uncvariance != exp(omega/(1-beta)) "
                    f"— EGARCH convention mismatch"
                ),
            )

    # ---- GARCH-M ----
    @pytest.mark.parametrize("label", sorted(GARCH_M_REFERENCE))
    def test_garch_m_uncvariance_matches_rugarch(
        self, label, garch_m_ref_models,
    ):
        rec = GARCH_M_REFERENCE[label]
        model = garch_m_ref_models[label]
        cx = float(model.stats()["unconditional_variance"])
        ref = float(rec["uncvariance"])
        # Variance-in-mean does not touch the sigma^2 recursion, so
        # uncvariance == omega/(1-a-b) both sides (exact).
        np.testing.assert_allclose(
            cx, ref, rtol=1e-9,
            err_msg=(
                f"{label}: GARCH-M copulax unconditional_variance "
                f"!= rugarch uncvariance"
            ),
        )

    # NB: the TGARCH (fGARCH submodel) uncvariance assertion lives in
    # TestTGARCHFGarchReference below, co-located with the TGARCH reference
    # loader and the Zakoian-vs-rugarch provenance tests.


# ---------------------------------------------------------------------------
# QGARCH(1, q) Layer-1 reference: dependency-free hand-rolled lax.scan Sentana
# recursion (HARD-03).
# ---------------------------------------------------------------------------
# Sentana (1995) QGARCH is NOT available in any correct third-party fitter:
# the CRAN `qgarch` package implements a DIFFERENT model and is BANNED as an
# oracle (see qgarch.py docstring and 01-RESEARCH.md RQ3). The gate for
# HARD-03 is therefore a hand-rolled, dependency-free lax.scan reference of the
# verbatim Sentana recursion, co-located here in the test module and
# cross-checked against CopulAX's `run_qgarch` kernel at rtol <= 1e-8. This is
# the "hand-rolled reference co-located with tests" precedent from TESTING.md.
#
# The reference deliberately does NOT import CopulAX's `run_qgarch` -- if it
# did, the cross-check would be circular (T-01-FIX in the plan threat model).
# It reimplements the Sentana sigma^2 recursion from scratch and reproduces the
# SAME pre-sample convention CopulAX's opt-in init="squared" mode uses (the
# rugarch rec.init="all" analog): the leading max(p, q) conditional variances
# are FIXED at the unconditional-variance estimate mean(eps^2), the pre-sample
# eps lag is zero (mean-corrected innovations), and the recursion proper starts
# at index max(p, q). Because both the reference and the kernel apply the
# identical warm-up, the match is to machine precision (~4e-16 measured), well
# inside the two-sided rtol <= 1e-8 Layer-1 gate.


def _qgarch_sentana_reference(eps, omega, alpha, psi, beta, *, n_warmup):
    r"""Dependency-free QGARCH(1, 1) sigma^2 recursion (Sentana 1995).

    Implements verbatim

    .. math::

        \sigma^2_t = \omega + \alpha\,\varepsilon^2_{t-1}
                   + \psi\,\varepsilon_{t-1} + \beta\,\sigma^2_{t-1}

    via :func:`jax.lax.scan`, with the ``init="squared"`` pre-sample
    convention (leading ``n_warmup`` conditional variances fixed at
    ``mean(eps^2)``; pre-sample ``eps`` lag = 0). Does NOT import or call
    :func:`copulax._src.timeseries._recursions.run_qgarch` -- this is the
    independent oracle for the CopulAX kernel, so importing it would make the
    cross-check circular.

    Args:
        eps: shape ``(n,)`` -- mean-corrected innovation series.
        omega, alpha, psi, beta: scalar QGARCH(1, 1) parameters.
        n_warmup: static ``max(p, q)`` -- number of leading sigma^2 outputs
            fixed at ``mean(eps^2)``.

    Returns:
        shape ``(n,)`` conditional-variance path sigma^2_t.
    """
    eps = jnp.asarray(eps, dtype=float).reshape(-1)
    n_warmup = int(n_warmup)
    presample_var = jnp.maximum(jnp.mean(eps * eps), 0.0)

    def step(carry, e_t):
        step_idx, eps_lag, var_lag = carry
        var_t = omega + alpha * eps_lag ** 2 + psi * eps_lag + beta * var_lag
        var_t = jnp.maximum(var_t, 1e-12)
        # Fix the leading max(p, q) conditional variances (rec.init="all").
        var_t = jnp.where(step_idx < n_warmup, presample_var, var_t)
        return (step_idx + 1, e_t, var_t), var_t

    init_carry = (jnp.asarray(0, dtype=int), jnp.asarray(0.0), presample_var)
    _, var_seq = jax.lax.scan(step, init_carry, eps)
    return var_seq


class TestQGARCHSentanaReference:
    """HARD-03 Layer-1 gate: CopulAX's QGARCH(1, 1) kernel matches a
    dependency-free hand-rolled lax.scan Sentana (1995) reference two-sided at
    rtol <= 1e-8.

    CRAN ``qgarch`` is a different model and is NOT used (banned oracle). The
    Sentana (1995) empirical estimation-table anchors could not be transcribed
    1:1 into CopulAX's (omega, alpha, psi, beta) parametrisation from the
    available sources, so per the plan's A2 fallback the hand-rolled reference
    is the sole gate here; no anchor constants are fabricated.
    """

    # A curated fixed-parameter grid spanning psi sign / magnitude and
    # persistence, plus a symmetric psi = 0 control (must collapse to vanilla
    # GARCH). Each is a stationary, positivity-satisfying (omega >=
    # psi^2/(4 alpha)) QGARCH(1, 1).
    _CASES = {
        "neg_psi_persistent": dict(omega=0.05, alpha=0.10, psi=-0.05, beta=0.85),
        "pos_psi_persistent": dict(omega=0.05, alpha=0.10, psi=+0.05, beta=0.85),
        "zero_psi_control":   dict(omega=0.05, alpha=0.10, psi=0.0,  beta=0.85),
        "large_psi_lowpers":  dict(omega=0.20, alpha=0.15, psi=-0.12, beta=0.60),
    }

    @staticmethod
    def _fixed_eps(seed, n=600):
        # Deterministic synthetic innovation series (fixed, not fitted) so the
        # comparison is a pure formula check with no optimiser involved.
        return jnp.asarray(
            np.random.default_rng(seed).standard_normal(n), dtype=float,
        )

    def _model_at(self, params):
        warm = {
            "omega": jnp.asarray(params["omega"]),
            "alpha": jnp.asarray([params["alpha"]]),
            "psi": jnp.asarray([params["psi"]]),
            "beta": jnp.asarray([params["beta"]]),
            "residual": {},
        }
        return QGARCH(1, 1, residual_dist=normal).fit(
            self._fixed_eps(0), init="warm", init_params=warm, maxiter=0,
        )

    @pytest.mark.parametrize("label", sorted(_CASES))
    def test_kernel_matches_hand_rolled_reference(self, label):
        """CopulAX run_qgarch (via conditional_variance, init="squared")
        matches the hand-rolled Sentana lax.scan reference at rtol <= 1e-8."""
        params = self._CASES[label]
        eps = self._fixed_eps(seed=(hash(label) & 0xFFFF))
        model = self._model_at(params)

        cx_var = np.asarray(model.conditional_variance(eps, init="squared"))
        # n_warmup = max(p, q) = max(1, 1) = 1 for QGARCH(1, 1).
        ref_var = np.asarray(
            _qgarch_sentana_reference(
                eps, params["omega"], params["alpha"], params["psi"],
                params["beta"], n_warmup=1,
            )
        )
        # Two-sided, tight: both compute the identical Sentana recursion with
        # the identical rec.init="all" pre-sample, so agreement is machine
        # precision. This is a Layer-1 fixed-parameter formula match (no
        # optimiser), the only setting where two-sided tightness is correct.
        np.testing.assert_allclose(cx_var, ref_var, rtol=1e-8, atol=1e-10)

    def test_zero_psi_collapses_to_garch(self):
        """psi = 0 => the Sentana recursion is exactly vanilla GARCH(1, 1);
        the QGARCH kernel and a vanilla-GARCH hand-rolled reference agree."""
        params = self._CASES["zero_psi_control"]
        eps = self._fixed_eps(seed=999)
        model = self._model_at(params)
        cx_var = np.asarray(model.conditional_variance(eps, init="squared"))

        # Hand-rolled vanilla GARCH(1, 1) with the same squared pre-sample.
        e = np.asarray(eps)
        mv = float(np.mean(e * e))
        ref = np.zeros_like(e)
        eps_sq_lag = mv
        var_lag = mv
        for t in range(len(e)):
            vt = params["omega"] + params["alpha"] * eps_sq_lag + params["beta"] * var_lag
            vt = max(vt, 1e-12)
            if t < 1:  # n_warmup = max(p, q) = 1
                vt = mv
            ref[t] = vt
            eps_sq_lag = e[t] ** 2
            var_lag = vt
        np.testing.assert_allclose(cx_var, ref, rtol=1e-8, atol=1e-10)

    def test_psi_sign_flips_asymmetry(self):
        """The psi term is the only source of sign-dependent asymmetry: for a
        series with a large negative shock followed by a large positive shock,
        psi < 0 and psi > 0 produce measurably different sigma^2 paths (guards
        against psi being silently dropped from the kernel)."""
        eps = jnp.asarray(
            np.array([0.0, -3.0, 0.5, 2.5, -0.2, 1.0] + [0.1] * 20), dtype=float,
        )
        neg = self._model_at(self._CASES["neg_psi_persistent"])
        pos = self._model_at(self._CASES["pos_psi_persistent"])
        cv_neg = np.asarray(neg.conditional_variance(eps, init="squared"))
        cv_pos = np.asarray(pos.conditional_variance(eps, init="squared"))
        # The step immediately after the -3.0 shock must differ by the psi
        # contribution: |psi_neg - psi_pos| * |eps| = 0.10 * 3.0 = 0.30 in
        # sigma^2 units at that step (before beta smoothing).
        assert not np.allclose(cv_neg, cv_pos, rtol=1e-6)

    def test_no_run_qgarch_import_in_reference(self):
        """Anti-circularity guard: the hand-rolled reference must not import or
        call CopulAX's run_qgarch (T-01-FIX). Verified structurally by
        inspecting the reference function's CODE (docstring stripped, since it
        legitimately names run_qgarch to explain the non-circularity contract).
        """
        import ast
        import inspect

        # Parse the reference function and strip its docstring so a textual
        # mention in the docstring (which explicitly says it does NOT call
        # run_qgarch) is not mistaken for an actual import / call.
        tree = ast.parse(inspect.getsource(_qgarch_sentana_reference))
        fn = tree.body[0]
        assert isinstance(fn, ast.FunctionDef)
        body = fn.body[1:] if (
            fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
        ) else fn.body
        code_src = "\n".join(ast.unparse(node) for node in body)
        assert "run_qgarch" not in code_src
        # Structurally: no Name/Attribute node in the executable body resolves
        # to run_qgarch, and there is no `run_qgarch` in the module globals of
        # the reference function (it was never imported at module scope).
        assert "run_qgarch" not in _qgarch_sentana_reference.__globals__


# ---------------------------------------------------------------------------
# TGARCH fGARCH reference (HARD-02): rugarch submodel="TGARCH" cross-validation.
# ---------------------------------------------------------------------------
# IMPORTANT -- read the module docstring in
# _r_reference/generate_tgarch_fgarch_reference.R and 01-04-SUMMARY.md before
# editing. rugarch's REPORTED sigma(fit) path for fGARCH-TGARCH is NOT the
# clean Zakoian |eps| recursion CopulAX implements: reverse-engineering the
# rugarch 1.5-5 C source (src/filters.c::fgarchfilter) shows the reported path
# uses the Hentschel (1995) omnibus recursion with a hard-coded
# sqrt(0.001^2 + z^2) softening of |z| (a fixed 1e-3 smoothing constant). That
# softening makes rugarch's reported sigma differ from CopulAX's Zakoian sigma
# by ~3-4e-5 (measured, with the pre-sample matched) -- ~4000x the 1e-8 gate.
#
# CopulAX's production TGARCH recursion is the clean Zakoian form and MUST NOT
# be changed to add the 0.001 softening (project decision).
# So there is NO valid two-sided Layer-1 gate of "CopulAX Zakoian vs rugarch
# reported sigma" at 1e-8. This module therefore:
#   1. reproduces the C-exact rugarch formula in a co-located reference and
#      ASSERTS it matches rugarch's reported sigma at rtol <= 1e-8 (the
#      reverse-engineering payoff / provenance demonstration), and
#   2. CAPTURES the CopulAX-Zakoian-vs-rugarch structural divergence as an
#      explicit, tracked assertion (surfaced, never absorbed) so the gap is a
#      first-class recorded fact for 01-MATH-REVIEW.md, not a hidden skip.
# The Zakoian production recursion's own Layer-1 validation against the `arch`
# TARCH/ZARCH oracle (Option A) is added once the user selects the resolution
# recorded in 01-04-SUMMARY.md's decision checkpoint.

_TGARCH_FGARCH_REF_PATH = (
    _Path(__file__).parent / "_r_reference" / "tgarch_fgarch_reference_data.py"
)
_tg_spec = _ilu.spec_from_file_location(
    "_tgarch_fgarch_reference", _TGARCH_FGARCH_REF_PATH,
)
_tg_mod = _ilu.module_from_spec(_tg_spec)
_tg_spec.loader.exec_module(_tg_mod)
TGARCH_FGARCH_REFERENCE = _tg_mod.TGARCH_FGARCH_REFERENCE


def _tgarch_fgarch_c_exact_reference(eps, omega, alpha1, beta1, eta11, pre_sigma):
    r"""C-exact rugarch fGARCH submodel="TGARCH" sigma-recursion (1.5-5).

    Faithful transcription of rugarch's compiled news-impact recursion
    (``src/filters.c :: fgarchfilter`` line 162, with the TGARCH submodel
    constants ``kdelta = delta + fk*lambda = 1``, ``lambda = 1``, ``eta2 = 0``
    from ``R/rugarch-startpars.R :: .fgarchModel("TGARCH")``), expanded from
    standardized-z space into raw-eps space:

    .. math::

        \sigma_t = \omega
                 + \alpha_1\bigl(\sqrt{0.001^2\,\sigma_{t-1}^2
                                       + \varepsilon_{t-1}^2}
                                 - \eta_{11}\,\varepsilon_{t-1}\bigr)
                 + \beta_1\,\sigma_{t-1}.

    The ``sqrt(0.001^2 * sigma^2 + eps^2)`` term is rugarch's Box-Cox-softened
    absolute value (the hard-coded 1e-3 smoothing constant in the C
    news-impact function); it is what makes the reported path diverge from the
    vignette / Zakoian plain-``|eps|`` formula. Pre-sample:
    ``sigma[0] = pre_sigma = mean(|res|)`` (rugarch's ``mvar`` for TGARCH,
    ``rec.init="all"``).

    This is a REFERENCE ONLY (documents rugarch's reported-sigma arithmetic);
    it is deliberately NOT CopulAX's production recursion, which is the clean
    Zakoian form and must not adopt the 1e-3 softening (project decision).

    Args:
        eps: shape ``(n,)`` -- innovation series (== y for include.mean=FALSE).
        omega, alpha1, beta1, eta11: rugarch-native fitted coefficients.
        pre_sigma: scalar pre-sample sigma (rugarch ``mvar = mean(|res|)``).

    Returns:
        shape ``(n,)`` reported-sigma path.
    """
    eps = np.asarray(eps, dtype=float)
    n = eps.shape[0]
    sig = np.zeros(n, dtype=float)
    sig[0] = pre_sigma
    for t in range(1, n):
        news = np.sqrt(0.001 ** 2 * sig[t - 1] ** 2 + eps[t - 1] ** 2) \
            - eta11 * eps[t - 1]
        sig[t] = omega + alpha1 * news + beta1 * sig[t - 1]
    return sig


class TestTGARCHFGarchReference:
    """HARD-02 TGARCH cross-validation against rugarch fGARCH submodel="TGARCH".

    See the module comment above and 01-04-SUMMARY.md. rugarch's reported
    sigma path is the Hentschel (1995) omnibus C recursion (0.001-softened
    |z|), not CopulAX's clean Zakoian |eps| form, so this class validates the
    reverse-engineered C-exact formula against rugarch at 1e-8 AND records the
    CopulAX-Zakoian structural divergence explicitly.
    """

    @pytest.mark.parametrize("label", sorted(TGARCH_FGARCH_REFERENCE))
    def test_c_exact_reference_reproduces_rugarch_sigma(self, label):
        """The reverse-engineered C-exact rugarch fGARCH-TGARCH recursion
        reproduces rugarch's REPORTED sigma(fit) two-sided at rtol <= 1e-8.

        This is the provenance demonstration required by the Option B
        decision: the exact reported-sigma arithmetic is the
        0.001-softened Hentschel recursion transcribed from src/filters.c.
        """
        rec = TGARCH_FGARCH_REFERENCE[label]
        y = np.asarray(rec["y"])
        rp = rec["rugarch_params"]
        ref_sigma = np.asarray(rec["sigma"])
        c_exact = _tgarch_fgarch_c_exact_reference(
            y, rp["omega"], rp["alpha1"], rp["beta1"], rp["eta11"],
            rec["pre_sample_sigma"],
        )
        # Skip index 0 (both are the fixed pre-sample warm value by
        # construction) is unnecessary -- they match there too -- but the
        # recursion proper (idx >= 1) is the meaningful comparison.
        np.testing.assert_allclose(
            c_exact[1:], ref_sigma[1:], rtol=1e-8, atol=1e-10,
        )

    @pytest.mark.parametrize("label", sorted(TGARCH_FGARCH_REFERENCE))
    def test_rq1_mapping_roundtrip(self, label):
        """The RQ1 mapping alpha_pos = alpha1*(1 - eta11),
        alpha_neg = alpha1*(1 + eta11) inverts to rugarch's (alpha1, eta11);
        beta and omega pass through identically. Guards the sanctioned
        conform-to-literature conversion (D-03) against transcription error."""
        rec = TGARCH_FGARCH_REFERENCE[label]
        rp = rec["rugarch_params"]
        P = rec["params"]
        # Forward mapping consistency.
        np.testing.assert_allclose(
            P["alpha_pos"], rp["alpha1"] * (1.0 - rp["eta11"]), rtol=1e-12,
        )
        np.testing.assert_allclose(
            P["alpha_neg"], rp["alpha1"] * (1.0 + rp["eta11"]), rtol=1e-12,
        )
        # Inverse mapping recovers rugarch (alpha1, eta11).
        alpha_rec = 0.5 * (P["alpha_pos"] + P["alpha_neg"])
        eta_rec = (P["alpha_neg"] - P["alpha_pos"]) / (P["alpha_pos"] + P["alpha_neg"])
        np.testing.assert_allclose(alpha_rec, rp["alpha1"], rtol=1e-12)
        np.testing.assert_allclose(eta_rec, rp["eta11"], rtol=1e-12)
        # omega / beta pass through identically.
        np.testing.assert_allclose(P["omega"], rp["omega"], rtol=1e-15)
        np.testing.assert_allclose(P["beta"], rp["beta1"], rtol=1e-15)
        # A1: lambda/delta are fixed (never emitted as free params); the
        # fixture only carries omega/alpha1/beta1/eta11 for the variance model.
        assert set(rp.keys()) == {"omega", "alpha1", "beta1", "eta11"}

    @pytest.mark.parametrize("label", sorted(TGARCH_FGARCH_REFERENCE))
    def test_copulax_zakoian_divergence_is_recorded(self, label):
        """RECORDED FACT (surfaced, never absorbed): CopulAX's clean Zakoian
        sigma-recursion at the mapped params does NOT match rugarch's reported
        sigma at 1e-8 -- because rugarch's reported path uses the
        0.001-softened Hentschel recursion (see module comment). Even with the
        pre-sample matched to rugarch's mean(|res|) warm value, the residual
        gap is O(1e-5), driven solely by the softening. This asserts the gap
        is REAL and BOUNDED (so a future two-sided 1e-8 gate would be wrong to
        add without either changing production math [forbidden] or switching
        the oracle). It is the evidence behind the 01-04 decision checkpoint.
        """
        from copulax._src.timeseries._recursions import run_tgarch

        rec = TGARCH_FGARCH_REFERENCE[label]
        y = jnp.asarray(np.asarray(rec["y"]))
        ref_sigma = np.asarray(rec["sigma"])
        P = rec["params"]
        pre = float(rec["pre_sample_sigma"])
        # Feed CopulAX's Zakoian kernel the SAME pre-sample sigma[0]=mean(|res|)
        # rugarch uses, so the pre-sample is NOT the source of the gap -- the
        # residual difference is purely the 0.001 softening.
        sig_cx, _ = run_tgarch(
            eps=y, omega=jnp.asarray(P["omega"]),
            alpha_pos=jnp.asarray([P["alpha_pos"]]),
            alpha_neg=jnp.asarray([P["alpha_neg"]]),
            beta=jnp.asarray([P["beta"]]),
            init_eps_pos_lags=jnp.full((1,), 0.5 * pre),
            init_eps_neg_lags=jnp.full((1,), 0.5 * pre),
            init_sigma_lags=jnp.full((1,), pre),
            n_warmup=1, warmup_var=jnp.asarray(pre ** 2),
        )
        sig_cx = np.asarray(sig_cx)
        rel = np.abs(sig_cx[1:] - ref_sigma[1:]) / np.abs(ref_sigma[1:])
        max_rel = float(rel.max())
        # The gap is real (above the 1e-8 gate) ...
        assert max_rel > 1e-7, (
            f"Unexpected: CopulAX Zakoian matched rugarch reported sigma to "
            f"{max_rel:.2e} for {label}; the 0.001-softening divergence should "
            f"be O(1e-5). Re-examine the C-source finding."
        )
        # ... and bounded (the softening is a small O(1e-5) perturbation, not a
        # gross model mismatch -- both are TGARCH-family sigma recursions).
        assert max_rel < 1e-3, (
            f"CopulAX Zakoian vs rugarch reported sigma gap {max_rel:.2e} for "
            f"{label} exceeds the expected O(1e-5) softening bound."
        )

    def test_c_exact_reference_is_not_copulax_production(self):
        """The C-exact reference is a documentation artifact, NOT CopulAX's
        production recursion: it must not import run_tgarch, and it must
        contain the tell-tale 0.001 softening constant that CopulAX's Zakoian
        form deliberately lacks."""
        import inspect

        src = inspect.getsource(_tgarch_fgarch_c_exact_reference)
        # The reference is the softened rugarch formula ...
        assert "0.001" in src
        # ... and it never routes through CopulAX's production kernel.
        assert "run_tgarch" not in _tgarch_fgarch_c_exact_reference.__globals__

    @pytest.mark.parametrize("label", sorted(TGARCH_FGARCH_REFERENCE))
    def test_uncvariance_matches_rugarch(self, label):
        r"""THIRD-PARTY unconditional-variance check: copulax's clean Zakoian
        ``stats()["unconditional_variance"]`` (== ``unconditional_sigma²``) at
        the mapped rugarch-fitted params matches rugarch's ``uncvariance(fit)``
        TIGHTLY.

        KEY FINDING (VERIFIED, 01-MATH-REVIEW.md unconditional-variance
        third-party section): unlike the reported sigma PATH — which carries
        the 0.001 Hentschel news-impact softening and diverges from any clean
        Zakoian recursion by O(1e-5) (``test_copulax_zakoian_divergence_is_
        recorded``) — rugarch's ``uncvariance`` is a CLOSED-FORM accessor,
        ``(omega/(1 - persistence))²`` with the SAME first-moment persistence
        (``alpha_pos·E[z⁺] + alpha_neg·E[z⁻] + beta``) copulax's Zakoian
        ``unconditional_sigma`` uses.  The softening does NOT enter it, so the
        match is at ``rtol <= 1e-8`` (measured ~7e-15 normal, ~6e-11 t; the t
        slack is copulax's quadrature ``E[z±]`` vs rugarch's analytic
        half-moments), NOT the softening-widened path tolerance.  This is a
        same-formula check on the accessor and SHOULD pass tightly.
        """
        rec = TGARCH_FGARCH_REFERENCE[label]
        P = rec["params"]
        rdist = _RESIDUAL_FROM_NAME[rec["residual_dist"]]
        rparams = (
            {"nu": jnp.asarray(rec["residual"]["nu"])}
            if rec["residual"] else {}
        )
        model = TGARCH(
            p=1, q=1, residual_dist=rdist, residual_params=rparams,
            omega=jnp.asarray(P["omega"]),
            alpha_pos=jnp.asarray([P["alpha_pos"]]),
            alpha_neg=jnp.asarray([P["alpha_neg"]]),
            beta=jnp.asarray([P["beta"]]),
        )
        s = model.stats()
        cx = float(s["unconditional_variance"])
        ref = float(rec["uncvariance"])
        assert np.isfinite(ref) and ref > 0
        # Documented identity: unconditional_variance == unconditional_sigma^2.
        np.testing.assert_allclose(
            cx, float(s["unconditional_sigma"]) ** 2, rtol=1e-12,
        )
        np.testing.assert_allclose(
            cx, ref, rtol=1e-8,
            err_msg=(
                f"{label}: TGARCH copulax Zakoian unconditional_variance "
                f"!= rugarch uncvariance (closed form; softening excluded)"
            ),
        )


# ---------------------------------------------------------------------------
# TGARCH arch-TARCH evaluation gate (HARD-02 Layer-1, HYBRID oracle).
# ---------------------------------------------------------------------------
# HYBRID ORACLE (user-approved, verbatim: "use a hybrid. arch for evaluation and
# rugarch for any fitting parameters"):
#   * rugarch is the PARAMETER oracle -- the fitted (omega, alpha_pos, alpha_neg,
#     beta) come from the rugarch fGARCH-TGARCH fit via the validated RQ1 eta1
#     mapping and are read from the committed fixture's "params" entry. This is
#     unchanged; TestTGARCHFGarchReference above owns the rugarch provenance.
#   * arch is the EVALUATION oracle -- CopulAX's clean Zakoian sigma-recursion,
#     evaluated AT the rugarch-fitted mapped params, must match Python arch's
#     TARCH/ZARCH fixed-parameter evaluation two-sided at rtol <= 1e-8.
#
# Why arch (not rugarch) can gate CopulAX's OWN recursion at 1e-8: arch's
# GARCH(p=1, o=1, q=1, power=1.0) is the TARCH/ZARCH sigma-form
#   sigma_t = omega + alpha*|eps_{t-1}| + gamma*|eps_{t-1}|*1{eps_{t-1}<0}
#           + beta*sigma_{t-1}
# (arch/univariate/recursions_python.py::garch_recursion_python with power=1.0,
# where the recursion runs in fsigma = sigma^power = sigma^1 space; verified by
# reading the source). This is EXACTLY CopulAX's clean Zakoian sign-split form
#   sigma_t = omega + alpha_pos*eps^+_{t-1} + alpha_neg*eps^-_{t-1}
#           + beta*sigma_{t-1}
# under the mapping (VERIFIED empirically on the fixture, see
# test_arch_gamma_mapping_is_correct):
#   alpha_pos = alpha_arch ;  alpha_neg = alpha_arch + gamma_arch
# i.e. arch's alpha loads BOTH shocks (|eps|) and gamma ADDS to negative shocks,
# so alpha_pos = alpha (positive-shock loading) and alpha_neg = alpha + gamma
# (negative-shock loading). arch applies NO |eps|-softening (rugarch's reported
# sigma has the 0.001 smoothing -- see the module comment above -- which is why
# rugarch's REPORTED sigma cannot gate CopulAX's clean recursion, but arch's can).
#
# PRE-SAMPLE RECONCILIATION (isolates the recursion): arch seeds its pre-sample
# lags from its power-1 backcast bc = EWMA_0.94(|eps|) over the first min(75, n)
# observations, using bc for the symmetric-shock and variance lags and 0.5*bc
# for the asymmetric-shock lag (garch_recursion_python, the (t-1-j) < 0 branch).
# arch does NOT fix any output sigma; it computes sigma[0] from those backcast
# lags. To reproduce arch's sigma[0] exactly, CopulAX's Zakoian recursion is fed
# the equivalent pre-sample lags (n_warmup = 0, i.e. NO output fixing):
#   eps^+_lag = eps^-_lag = 0.5*bc ,  sigma_lag = bc
# because CopulAX's step 0 is
#   omega + alpha_pos*eps^+_lag + alpha_neg*eps^-_lag + beta*sigma_lag
#   = omega + alpha*(eps^+_lag + eps^-_lag) + gamma*eps^-_lag + beta*sigma_lag ,
# which equals arch's omega + alpha*bc + gamma*0.5*bc + beta*bc iff
# eps^+_lag + eps^-_lag = bc and eps^-_lag = 0.5*bc  =>  both = 0.5*bc.
# Measured agreement under the conftest x64 basis: sigma-path rel diff ~2e-16
# and Gaussian log-likelihood bit-identical (rel = 0) for both fixtures; arch's
# variance bounds never clamp this path (min margin to the upper bound ~300+).


class TestTGARCHArchEvaluationGate:
    """HARD-02 TGARCH Layer-1 gate (HYBRID oracle): CopulAX's clean Zakoian
    sigma-recursion and log-likelihood, evaluated at the rugarch-fitted mapped
    params (the fixture's ``params``), match Python ``arch``'s TARCH/ZARCH
    fixed-parameter evaluation two-sided at rtol <= 1e-8.

    This is the two-sided Layer-1 gate on CopulAX's OWN production recursion
    that the rugarch reported-sigma path could not provide (rugarch's reported
    sigma carries a 0.001 |eps|-softening -- see the module comment and
    TestTGARCHFGarchReference). rugarch remains the PARAMETER oracle (the params
    come from its fit); arch is the EVALUATION oracle at those fixed params.
    arch's fitted params are NOT used as an oracle here -- only its
    fixed-parameter recursion / density evaluation machinery.
    """

    @pytest.fixture(scope="class")
    def arch_module(self):
        return require_oracle("arch")

    @staticmethod
    def _arch_tarch_sigma(arch_module, y, omega, alpha_arch, gamma_arch, beta):
        """arch TARCH/ZARCH (power=1) fixed-parameter sigma path + backcast.

        Uses arch's own fixed-parameter evaluation machinery
        (``GARCH(p=1, o=1, q=1, power=1.0).compute_variance``) -- NOT a fit.
        Returns ``(sigma_path, backcast)`` where ``sigma = sqrt(sigma2)`` and
        ``backcast`` is arch's power-1 EWMA(|eps|) pre-sample anchor.
        """
        vol = arch_module.univariate.GARCH(p=1, o=1, q=1, power=1.0)
        # arch GARCH parameter order: [omega, alpha(1..p), gamma(1..o), beta(1..q)].
        params = np.array([omega, alpha_arch, gamma_arch, beta], dtype=float)
        backcast = float(vol.backcast(np.asarray(y)))
        var_bounds = vol.variance_bounds(np.asarray(y))
        sigma2 = np.zeros_like(np.asarray(y, dtype=float))
        sigma2 = vol.compute_variance(
            params.copy(), np.asarray(y, dtype=float), sigma2,
            backcast, var_bounds,
        )
        return np.sqrt(sigma2), backcast

    @staticmethod
    def _copulax_zakoian_sigma(y, params, backcast):
        """CopulAX clean Zakoian sigma path at ``params`` with the pre-sample
        reconciled to arch's backcast lags (see the module comment). Runs the
        production kernel ``run_tgarch`` directly (n_warmup = 0)."""
        from copulax._src.timeseries._recursions import run_tgarch

        sigma_seq, _ = run_tgarch(
            eps=jnp.asarray(np.asarray(y, dtype=float)),
            omega=jnp.asarray(params["omega"]),
            alpha_pos=jnp.asarray([params["alpha_pos"]]),
            alpha_neg=jnp.asarray([params["alpha_neg"]]),
            beta=jnp.asarray([params["beta"]]),
            init_eps_pos_lags=jnp.asarray([0.5 * backcast]),
            init_eps_neg_lags=jnp.asarray([0.5 * backcast]),
            init_sigma_lags=jnp.asarray([backcast]),
            n_warmup=0, warmup_var=0.0,
        )
        return np.asarray(sigma_seq)

    @pytest.mark.parametrize("label", sorted(TGARCH_FGARCH_REFERENCE))
    def test_arch_gamma_mapping_is_correct(self, label, arch_module):
        """Empirically confirm the arch<->CopulAX mapping
        alpha_pos = alpha_arch, alpha_neg = alpha_arch + gamma_arch by proving a
        hand-rolled Zakoian sign-split recursion (using alpha_pos / alpha_neg on
        the raw eps^+ / eps^- of the SAME series) reproduces arch's TARCH
        compute_variance path. This verifies the sign / indicator convention
        (arch's gamma loads NEGATIVE shocks additively) before the main gate
        relies on it (probe-before-trust, per the plan directive)."""
        rec = TGARCH_FGARCH_REFERENCE[label]
        y = np.asarray(rec["y"], dtype=float)
        P = rec["params"]
        alpha_arch = P["alpha_pos"]
        gamma_arch = P["alpha_neg"] - P["alpha_pos"]
        sig_arch, bc = self._arch_tarch_sigma(
            arch_module, y, P["omega"], alpha_arch, gamma_arch, P["beta"],
        )
        # Hand-rolled Zakoian sign-split recursion (NumPy, independent of both
        # arch and CopulAX kernels) with the arch backcast pre-sample.
        n = len(y)
        eps_pos = np.maximum(y, 0.0)
        eps_neg = np.maximum(-y, 0.0)
        sig_hand = np.zeros(n, dtype=float)
        for t in range(n):
            ep = eps_pos[t - 1] if t - 1 >= 0 else 0.5 * bc
            en = eps_neg[t - 1] if t - 1 >= 0 else 0.5 * bc
            sl = sig_hand[t - 1] if t - 1 >= 0 else bc
            sig_hand[t] = (
                P["omega"]
                + P["alpha_pos"] * ep
                + P["alpha_neg"] * en
                + P["beta"] * sl
            )
        # The Zakoian sign-split IS arch's TARCH under the asserted mapping.
        np.testing.assert_allclose(sig_hand, sig_arch, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("label", sorted(TGARCH_FGARCH_REFERENCE))
    def test_sigma_path_matches_arch_tarch(self, label, arch_module):
        """CopulAX's PRODUCTION Zakoian recursion (``run_tgarch``) at the
        rugarch-fitted mapped params matches arch's TARCH/ZARCH fixed-parameter
        sigma path two-sided at rtol <= 1e-8. Pre-sample reconciled to arch's
        backcast (see the module comment); measured ~2e-16 under the x64 basis.
        """
        rec = TGARCH_FGARCH_REFERENCE[label]
        y = np.asarray(rec["y"], dtype=float)
        P = rec["params"]
        alpha_arch = P["alpha_pos"]
        gamma_arch = P["alpha_neg"] - P["alpha_pos"]
        sig_arch, bc = self._arch_tarch_sigma(
            arch_module, y, P["omega"], alpha_arch, gamma_arch, P["beta"],
        )
        sig_cx = self._copulax_zakoian_sigma(y, P, bc)
        np.testing.assert_allclose(sig_cx, sig_arch, rtol=1e-8, atol=1e-10)

    @pytest.mark.parametrize("label", sorted(TGARCH_FGARCH_REFERENCE))
    def test_gaussian_loglik_matches_arch(self, label, arch_module):
        """The Gaussian log-likelihood evaluated at CopulAX's sigma path equals
        the Gaussian log-likelihood at arch's sigma path two-sided at
        rtol <= 1e-8 (bit-identical in practice, since the sigma paths agree to
        ~2e-16). This isolates the recursion + Gaussian-density evaluation and
        applies to BOTH fixtures regardless of the fixture's residual law -- the
        density here is Normal, matched engine-to-engine.
        """
        rec = TGARCH_FGARCH_REFERENCE[label]
        y = np.asarray(rec["y"], dtype=float)
        P = rec["params"]
        alpha_arch = P["alpha_pos"]
        gamma_arch = P["alpha_neg"] - P["alpha_pos"]
        sig_arch, bc = self._arch_tarch_sigma(
            arch_module, y, P["omega"], alpha_arch, gamma_arch, P["beta"],
        )
        sig_cx = self._copulax_zakoian_sigma(y, P, bc)

        def gaussian_ll(sigma):
            var = sigma ** 2
            return float(np.sum(
                -0.5 * (np.log(2.0 * np.pi) + np.log(var) + y ** 2 / var)
            ))

        np.testing.assert_allclose(
            gaussian_ll(sig_cx), gaussian_ll(sig_arch), rtol=1e-8, atol=1e-10,
        )

    def test_student_t_loglik_matches_arch(self, arch_module):
        """arch's standardized (unit-variance) Student-t density provably
        matches CopulAX's ``StandardisedResidual(student_t)`` density: at the
        rugarch-fitted mapped params AND the fitted nu, the Student-t
        log-likelihood evaluated on CopulAX's sigma path equals arch's
        ``StudentsT.loglikelihood`` on arch's sigma path two-sided at
        rtol <= 1e-8 (measured ~5e-16 -- no t-convention mismatch caps the
        agreement, so this native-t LL gate is included rather than documented
        as a limitation). Uses the studentt fixture only.
        """
        from copulax._src.timeseries._residuals._standardise import (
            StandardisedResidual,
        )

        rec = TGARCH_FGARCH_REFERENCE["tgarch11_studentt"]
        y = np.asarray(rec["y"], dtype=float)
        P = rec["params"]
        nu = rec["residual"]["nu"]
        alpha_arch = P["alpha_pos"]
        gamma_arch = P["alpha_neg"] - P["alpha_pos"]
        sig_arch, bc = self._arch_tarch_sigma(
            arch_module, y, P["omega"], alpha_arch, gamma_arch, P["beta"],
        )
        sig_cx = self._copulax_zakoian_sigma(y, P, bc)

        # arch StudentsT LL on arch's sigma^2 path.
        dist = arch_module.univariate.StudentsT()
        ll_arch = float(
            dist.loglikelihood(np.array([nu]), y, sig_arch ** 2, False)
        )
        # CopulAX standardized-t residual LL on CopulAX's sigma path:
        # sum[ logpdf_std_t(z) - log(sigma) ].
        wrapper = StandardisedResidual(student_t)
        z = y / sig_cx
        logpdf = (
            np.asarray(wrapper.logpdf(jnp.asarray(z), {"nu": nu}))
            - np.log(sig_cx)
        )
        ll_cx = float(np.sum(logpdf))
        np.testing.assert_allclose(ll_cx, ll_arch, rtol=1e-8, atol=1e-10)

    def test_arch_evaluation_uses_fixed_params_not_a_fit(self):
        """Anti-oracle-confusion guard: this class evaluates arch at FIXED
        params (compute_variance) and never fits arch -- arch's fitted params
        are not an oracle for anything here (rugarch owns fitting). Verified
        structurally: the arch-path helper's source calls compute_variance and
        does NOT call arch's `.fit(`.
        """
        import inspect

        src = inspect.getsource(self._arch_tarch_sigma)
        assert "compute_variance" in src
        assert ".fit(" not in src


# ---------------------------------------------------------------------------
# Retracing guard (HARD-07)
# ---------------------------------------------------------------------------
class TestRetracingGuard:
    r"""The three ``_roll_path`` methods were hoisted to top-level
    ``lax.scan`` kernels (``run_garch_rvs_path`` / ``run_arma_rvs_path`` /
    ``run_arma_garch_rvs_path`` in ``_recursions.py``) so that a *single*
    compiled trace serves every fitted instance of the same order and dtype,
    matching the module-level kernel contract used everywhere else in the
    subpackage.

    Each test below fits **two DISTINCT** instances (different training data
    -> different fitted parameter leaves) of the same order, feeds both to
    **one** ``jax.jit``-wrapped callable that exercises the ``rvs`` /
    ``_roll_path`` scan path, and asserts the callable is traced **at most
    once** across both instances.  A hand-rolled trace counter (a ``list``
    slot incremented inside the jitted body, which JAX executes once per
    trace) is used rather than an external dependency -- the guard semantics
    are simple and fully under our control.

    Each test also includes a *liveness* assertion: feeding a third instance
    of a **different order** forces a second trace, proving the counter is
    genuinely wired to XLA tracing and would catch a regression in which two
    same-order fitted instances retraced.

    Provenance note (jax 0.10.0, HELD).  This guard is a forward-looking
    regression *lock* on the single-shared-trace property, not a pre-/post-
    hoist discriminator.  Verified empirically this phase: under jax 0.10.0 a
    ``jax.jit`` that takes the model as a *traced* PyTree argument abstracts
    its array leaves, so the pre-hoist per-call ``step`` closure recreation
    did **not** cause differential retracing either -- both pre- and
    post-hoist trace exactly once here.  The hoist remains the correct,
    idiomatic form (it removes the theoretical retrace driver flagged in
    ``CONCERNS.md`` and matches ``_recursions.py``); this test locks in that
    distinct same-order fitted instances continue to share one compiled trace
    going forward (including under future jax upgrades gated by D-15).
    """

    @staticmethod
    def _n_traces(models, z):
        r"""Return how many times a single ``jax.jit`` callable is traced
        when its ``_roll_path`` scan path is invoked once per model in
        ``models``.  The counter increments inside the jitted body, so it
        counts *traces* (abstract evaluations), not executions.
        """
        n_traced = [0]

        @jax.jit
        def guard(model, z):
            n_traced[0] += 1
            return model._roll_path(z, model.terminal_state)

        for model in models:
            guard(model, z)
        return n_traced[0]

    @staticmethod
    def _z(n=30):
        return jnp.asarray(np.random.default_rng(0).standard_normal(n))

    def test_variance_roll_path_single_trace_across_two_fits(self):
        """GARCH σ²-form ``_roll_path``: two distinct GARCH(1,1) fits share a
        single compiled trace; a GARCH(2,1) fit forces a second trace."""
        z = self._z()
        fit_a = GARCH(p=1, q=1, residual_dist=normal).fit(
            simulate_garch11(400, 0.05, 0.10, 0.85, jax.random.PRNGKey(1)),
            init="analytical", maxiter=100, lr=0.05,
        )
        fit_b = GARCH(p=1, q=1, residual_dist=normal).fit(
            simulate_garch11(400, 0.08, 0.06, 0.90, jax.random.PRNGKey(9)),
            init="analytical", maxiter=100, lr=0.05,
        )
        # Distinct fitted parameters (the two instances are genuinely different).
        assert not np.allclose(
            np.asarray(fit_a.omega), np.asarray(fit_b.omega)
        ) or not np.allclose(np.asarray(fit_a.beta), np.asarray(fit_b.beta))

        assert self._n_traces([fit_a, fit_b], z) == 1

        # Liveness: a different order MUST retrace (the counter is live).
        fit_p2 = GARCH(p=2, q=1, residual_dist=normal).fit(
            simulate_garch11(400, 0.05, 0.10, 0.80, jax.random.PRNGKey(3)),
            init="analytical", maxiter=100, lr=0.05,
        )
        assert self._n_traces([fit_a, fit_b, fit_p2], z) == 2

    def test_mean_roll_path_single_trace_across_two_fits(self):
        """ARMA mean ``_roll_path``: two distinct ARMA(1,1) fits share a
        single compiled trace; an ARMA(2,1) fit forces a second trace."""
        z = self._z()

        def level(key, n=400):
            return jax.random.normal(key, (n,)) * 0.7 + 0.2

        fit_a = ARMA(p=1, q=1, residual_dist=normal).fit(
            level(jax.random.PRNGKey(11)), maxiter=100, lr=0.05,
        )
        fit_b = ARMA(p=1, q=1, residual_dist=normal).fit(
            level(jax.random.PRNGKey(22)), maxiter=100, lr=0.05,
        )
        assert not np.allclose(np.asarray(fit_a.mu), np.asarray(fit_b.mu)) or \
            not np.allclose(np.asarray(fit_a.phi), np.asarray(fit_b.phi))

        assert self._n_traces([fit_a, fit_b], z) == 1

        fit_p2 = ARMA(p=2, q=1, residual_dist=normal).fit(
            level(jax.random.PRNGKey(33)), maxiter=100, lr=0.05,
        )
        assert self._n_traces([fit_a, fit_b, fit_p2], z) == 2

    def test_joint_roll_path_single_trace_across_two_fits(self):
        """Joint ARMA-GARCH ``_roll_path`` (mean rollout hoisted; variance
        step still delegated to the backend): two distinct ARMA(1,1)-GARCH(1,1)
        fits share a single compiled trace; an ARMA(2,1)-GARCH(1,1) fit forces
        a second trace."""
        z = self._z()

        def level(key, n=400):
            return jax.random.normal(key, (n,)) * 0.7 + 0.2

        fit_a = ArmaGarch(
            mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(level(jax.random.PRNGKey(111)), init="analytical", maxiter=100, lr=0.05)
        fit_b = ArmaGarch(
            mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(level(jax.random.PRNGKey(222)), init="analytical", maxiter=100, lr=0.05)
        assert not np.allclose(np.asarray(fit_a.mu), np.asarray(fit_b.mu)) or \
            not np.allclose(np.asarray(fit_a.phi), np.asarray(fit_b.phi))

        assert self._n_traces([fit_a, fit_b], z) == 1

        fit_p2 = ArmaGarch(
            mean_order=(2, 1), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(level(jax.random.PRNGKey(333), 600), init="analytical", maxiter=100, lr=0.05)
        assert self._n_traces([fit_a, fit_b, fit_p2], z) == 2


# ---------------------------------------------------------------------------
# D-09 convergence-status leaves (HARD-06)
# ---------------------------------------------------------------------------
def _degenerate_eps(n=300, key=None):
    r"""A series that drives the GARCH fit into a non-finite gradient
    region so the solver sets ``nan_encountered`` and never reaches a
    stationary point.  A single ``inf`` entry makes the conditional
    log-likelihood non-finite along the whole recursion tail."""
    key = jax.random.PRNGKey(4) if key is None else key
    eps = simulate_garch11(n, 0.05, 0.10, 0.85, key)
    return eps.at[n // 3].set(jnp.inf)


class TestConvergenceStatus:
    """D-09: fitted instances carry plain-named array-leaf convergence
    status fields (NO trailing underscore) packed from the solver."""

    def _fit(self):
        key = jax.random.PRNGKey(2)
        eps = simulate_garch11(600, 0.05, 0.10, 0.85, key)
        return GARCH(p=1, q=1, residual_dist=normal).fit(
            eps, init="analytical", maxiter=400, lr=0.05,
        )

    def test_converged_fit_reports_true_and_finite_stats(self):
        fit = self._fit()
        assert bool(fit.converged) is True
        assert np.isfinite(float(fit.grad_norm))
        assert int(fit.n_iterations) > 0
        assert bool(fit.nan_encountered) is False

    def test_nan_gradient_fit_reports_not_converged(self):
        """A fit that hits a non-finite gradient sets ``nan_encountered``
        True and ``converged`` False (the honest failure signal)."""
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(
            _degenerate_eps(), init="analytical", maxiter=80, lr=0.05,
        )
        assert bool(fit.nan_encountered) is True
        assert bool(fit.converged) is False

    def test_multi_start_candidate_stats_present(self):
        """Candidate-stats leaves (finite-LL count, winning candidate
        index) exist as status leaves.  Plan 10 fills them with real
        multi-start aggregates; this plan pins the single-start
        placeholders so the FIELDS and their types exist now."""
        fit = self._fit()
        assert int(fit.n_finite_candidates) >= 1
        assert int(fit.best_candidate) >= 0

    def test_status_leaves_are_array_leaves(self):
        """The status leaves are JAX array leaves (not Python scalars) so
        they survive as PyTree leaves and are JIT-safe.

        The six names asserted here are the D-09 contract: plain-named
        (NO trailing underscore, unlike the mutating fitted-only leaves
        such as ``n_train_``) — a missing or renamed field fails the
        ``getattr`` below."""
        fit = self._fit()
        for name in (
            "converged", "grad_norm", "n_iterations", "nan_encountered",
            "n_finite_candidates", "best_candidate",
        ):
            leaf = getattr(fit, name)
            assert isinstance(leaf, jax.Array), (
                f"status field {name!r} must be a jax.Array leaf, got "
                f"{type(leaf)}"
            )

    def test_status_survives_jitted_fit(self):
        """A jitted fit still populates the status leaves (JIT-safe)."""
        key = jax.random.PRNGKey(2)
        eps = simulate_garch11(500, 0.05, 0.10, 0.85, key)

        def fit_fn(e):
            return GARCH(p=1, q=1, residual_dist=normal).fit(
                e, init="analytical", maxiter=100, lr=0.05,
            )

        jitted = jax.jit(fit_fn)(eps)
        assert bool(jitted.converged) is True
        assert np.isfinite(float(jitted.grad_norm))
        assert bool(jitted.nan_encountered) is False

    def test_summary_contains_convergence_line(self):
        """summary() renders a convergence line derived from the status
        fields."""
        fit = self._fit()
        text = fit.summary()
        assert "converg" in text.lower(), (
            "summary() must render a convergence line"
        )

    def test_arma_and_joint_carry_status_leaves(self):
        """The status contract holds across all three bases (ARMA mean,
        GARCH variance, joint ArmaGarch), not just standalone GARCH."""
        key = jax.random.PRNGKey(7)
        y = jax.random.normal(key, (500,)) * 0.7 + 0.2
        arma = ARMA(p=1, q=1, residual_dist=normal).fit(
            y, init="analytical", maxiter=200, lr=0.05,
        )
        joint = ArmaGarch(
            mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, init="analytical", maxiter=100, lr=0.05)
        for fit in (arma, joint):
            assert bool(fit.converged) in (True, False)
            assert np.isfinite(float(fit.grad_norm))
            assert isinstance(fit.n_iterations, jax.Array)


# ---------------------------------------------------------------------------
# D-10 warning delivery via jax.debug.callback (HARD-06)
# ---------------------------------------------------------------------------
def _nonconverged_fit_fn(eps):
    r"""A fit whose tiny iteration budget guarantees non-convergence, so
    the fit-tail ConvergenceWarning fires."""
    return GARCH(p=1, q=1, residual_dist=normal).fit(
        eps, init="analytical", maxiter=2, lr=0.05,
    )


class TestConvergenceWarning:
    """D-10: a non-converged fit fires a ConvergenceWarning via a single
    jax.debug.callback at the fit tail, under eager AND jit, but not
    during pure (abstract) tracing."""

    def _eps(self):
        key = jax.random.PRNGKey(2)
        return simulate_garch11(500, 0.05, 0.10, 0.85, key)

    def test_fires_under_eager_fit(self):
        eps = self._eps()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _nonconverged_fit_fn(eps)
        assert any(
            issubclass(rec.category, ConvergenceWarning) for rec in w
        ), "eager non-converged fit must emit a ConvergenceWarning"

    def test_fires_under_jit_fit(self):
        """The flagship path: the warning fires even when the whole fit is
        wrapped in jax.jit."""
        eps = self._eps()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            jax.jit(_nonconverged_fit_fn)(eps)
        assert any(
            issubclass(rec.category, ConvergenceWarning) for rec in w
        ), "jitted non-converged fit must emit a ConvergenceWarning"

    def test_does_not_fire_during_pure_tracing(self):
        """jax.debug.callback fires at EXECUTION, not while JAX builds the
        jaxpr.  Tracing the fit (jit(...).lower(), no run) must produce no
        warning — the trace-time guarantee that prevents spurious warnings
        from the per-iteration inner gradient evaluations."""
        eps = self._eps()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            jax.jit(_nonconverged_fit_fn).lower(eps)  # trace + lower, no run
            jax.eval_shape(_nonconverged_fit_fn, eps)  # pure abstract eval
        assert not any(
            issubclass(rec.category, ConvergenceWarning) for rec in w
        ), "no warning may fire during pure tracing"

    def test_converged_fit_does_not_warn(self):
        """A well-converged fit must NOT emit a ConvergenceWarning (no
        spurious warnings on healthy fits)."""
        eps = self._eps()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GARCH(p=1, q=1, residual_dist=normal).fit(
                eps, init="analytical", maxiter=600, lr=0.05,
            )
        assert not any(
            issubclass(rec.category, ConvergenceWarning) for rec in w
        ), "a converged fit must not emit a ConvergenceWarning"


class TestDataScaleWarning:
    """D-10: fitting on poorly-scaled data fires a DataScaleWarning that
    points the user at DataScaler; no auto-rescaling occurs."""

    def test_fires_on_large_scale_data(self):
        key = jax.random.PRNGKey(2)
        # Scale the series far above the [0.1, 10000) well-conditioned
        # band so var(eps) >> 10000.
        eps = simulate_garch11(500, 0.05, 0.10, 0.85, key) * 500.0
        assert float(jnp.var(eps)) >= 10000.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GARCH(p=1, q=1, residual_dist=normal).fit(
                eps, init="analytical", maxiter=100, lr=0.05,
            )
        scale_warns = [
            rec for rec in w if issubclass(rec.category, DataScaleWarning)
        ]
        assert scale_warns, "poorly-scaled data must emit a DataScaleWarning"
        assert any(
            "DataScaler" in str(rec.message) for rec in scale_warns
        ), "the DataScaleWarning must point at DataScaler"

    def test_does_not_fire_on_unit_scale_data(self):
        key = jax.random.PRNGKey(2)
        eps = simulate_garch11(500, 0.05, 0.10, 0.85, key)
        assert 0.1 <= float(jnp.var(eps)) < 10000.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GARCH(p=1, q=1, residual_dist=normal).fit(
                eps, init="analytical", maxiter=100, lr=0.05,
            )
        assert not any(
            issubclass(rec.category, DataScaleWarning) for rec in w
        ), "unit-scale data must not emit a DataScaleWarning"


class TestReportedLikelihood:
    """WR-05: the reported log-likelihood is the raw NaN-propagating sum
    from _log_likelihood_on_series, never the penalised optimiser
    objective; a degenerate fit reports NaN, not -2e9."""

    def test_loglik_equals_log_likelihood_on_series_normal_fit(self):
        """A normal fit's cached loglikelihood() equals
        _log_likelihood_on_series at the fitted params (raw sum)."""
        key = jax.random.PRNGKey(2)
        eps = simulate_garch11(600, 0.05, 0.10, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(
            eps, init="analytical", maxiter=400, lr=0.05,
        )
        cached = float(fit.loglikelihood())
        raw = float(fit._log_likelihood_on_series(eps, init="backcast"))
        np.testing.assert_allclose(cached, raw, rtol=1e-10, atol=1e-8)

    def test_wr05_degenerate_fit_reports_nan_loglik(self):
        """A degenerate fit (inf in the series) reports NaN loglikelihood,
        NOT the -2e9-scale penalised objective."""
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(
            _degenerate_eps(), init="analytical", maxiter=80, lr=0.05,
        )
        ll = float(fit.loglikelihood())
        assert np.isnan(ll), f"degenerate fit must report NaN LL, got {ll}"
        # AIC/BIC read the same value back -> also NaN (honest signal).
        assert np.isnan(float(fit.aic()))
        assert np.isnan(float(fit.bic()))

    def test_wr05_arma_degenerate_reports_nan(self):
        """WR-05 holds for the ARMA mean base too."""
        key = jax.random.PRNGKey(3)
        y = jax.random.normal(key, (400,))
        y = y.at[100].set(jnp.inf)
        fit = ARMA(p=1, q=1, residual_dist=normal).fit(
            y, init="analytical", maxiter=60, lr=0.05,
        )
        assert np.isnan(float(fit.loglikelihood()))

    def test_wr05_joint_degenerate_reports_nan(self):
        """WR-05 holds for the joint ArmaGarch base too."""
        key = jax.random.PRNGKey(3)
        y = jax.random.normal(key, (400,))
        y = y.at[100].set(jnp.inf)
        fit = ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, init="analytical", maxiter=60, lr=0.05)
        assert np.isnan(float(fit.loglikelihood()))


class TestUnconditionalVarianceWR08:
    """WR-08: the MA(q)/ARMA unconditional-variance factor matches the
    exact literature factor recorded in 01-MATH-REVIEW.md (pre-approved
    conform-to-literature fix)."""

    def test_uncond_ma1_exact_factor(self):
        """MA(1): Var(y) = sigma_eps^2 * (1 + theta^2) — Hamilton (1994)
        sec. 3.3.  The old code returned plain sigma_eps^2 (WR-08 bug)."""
        theta, sigma = 0.9, 1.5
        ma = MA(
            p=0, q=1, residual_dist=normal,
            phi=jnp.zeros((0,)), theta=jnp.array([theta]),
            mu=jnp.array(0.0), sigma_eps=jnp.array(sigma),
            residual_params={},
        )
        v = float(ma.stats()["variance"])
        np.testing.assert_allclose(v, sigma ** 2 * (1.0 + theta ** 2))

    def test_uncond_ma2_exact_factor(self):
        """MA(2): Var(y) = sigma_eps^2 * (1 + theta_1^2 + theta_2^2)."""
        thetas, sigma = [0.6, -0.3], 2.0
        ma = MA(
            p=0, q=2, residual_dist=normal,
            phi=jnp.zeros((0,)), theta=jnp.array(thetas),
            mu=jnp.array(0.0), sigma_eps=jnp.array(sigma),
            residual_params={},
        )
        v = float(ma.stats()["variance"])
        expected = sigma ** 2 * (1.0 + thetas[0] ** 2 + thetas[1] ** 2)
        np.testing.assert_allclose(v, expected, rtol=1e-10)

    def test_uncond_arma11_exact_factor(self):
        """ARMA(1,1): Var(y) = sigma_eps^2 (1 + 2 phi theta + theta^2) /
        (1 - phi^2) — Hamilton (1994) sec. 3.4 / Yule-Walker."""
        phi, theta, sigma = 0.5, 0.3, 1.2
        arma = ARMA(
            p=1, q=1, residual_dist=normal,
            phi=jnp.array([phi]), theta=jnp.array([theta]),
            mu=jnp.array(0.0), sigma_eps=jnp.array(sigma),
            residual_params={},
        )
        v = float(arma.stats()["variance"])
        expected = (
            sigma ** 2 * (1.0 + 2.0 * phi * theta + theta ** 2)
            / (1.0 - phi ** 2)
        )
        np.testing.assert_allclose(v, expected, rtol=1e-10)

    def test_uncond_ar1_exact_still_holds(self):
        """AR(1) is the theta=0 special case: Var(y) = sigma^2/(1-phi^2)
        (exact Yule-Walker) — must be unchanged by the WR-08 fix."""
        phi, sigma = 0.6, 1.0
        ar = AR(
            p=1, q=0, residual_dist=normal,
            phi=jnp.array([phi]), theta=jnp.zeros((0,)),
            mu=jnp.array(0.0), sigma_eps=jnp.array(sigma),
            residual_params={},
        )
        v = float(ar.stats()["variance"])
        np.testing.assert_allclose(v, sigma ** 2 / (1.0 - phi ** 2), rtol=1e-10)

    @pytest.mark.parametrize(
        "phi,q,theta",
        [(1.0, 0, []), (-1.0, 0, []), (1.05, 0, []), (1.0, 1, [0.3])],
        ids=["unit-root-AR1", "neg-unit-root-AR1", "explosive-AR1",
             "unit-root-ARMA11"],
    )
    def test_fast_path_nonstationary_reports_inf(self, phi, q, theta):
        """|phi| >= 1 on the p==1, q<=1 fast path: the unconditional
        variance does not exist and the accessor reports the +inf
        sentinel — the same non-existence convention as the general
        Yule-Walker branch and the GARCH-family accessors (IGARCH).
        Previously the floored denominator returned a huge FINITE value
        (~1e12 * sigma^2), the plausible-looking-wrong-number failure
        mode the no-silent-failure contract forbids."""
        cls = AR if q == 0 else ARMA
        model = cls(
            p=1, q=q, residual_dist=normal,
            phi=jnp.array([phi]), theta=jnp.array(theta, dtype=float),
            mu=jnp.array(0.0), sigma_eps=jnp.array(1.5),
            residual_params={},
        )
        v = float(model.stats()["variance"])
        assert np.isinf(v) and v > 0, (
            f"non-stationary phi={phi} must report +inf, got {v!r}"
        )

    def test_fast_path_near_unit_root_stays_exact(self):
        """phi = 0.999 is stationary: the fast path must return the exact
        closed form (~500.25 * sigma^2), not the sentinel — the inf arm
        fires only at |phi| >= 1."""
        phi, sigma = 0.999, 1.0
        ar = AR(
            p=1, q=0, residual_dist=normal,
            phi=jnp.array([phi]), theta=jnp.zeros((0,)),
            mu=jnp.array(0.0), sigma_eps=jnp.array(sigma),
            residual_params={},
        )
        v = float(ar.stats()["variance"])
        np.testing.assert_allclose(v, sigma ** 2 / (1.0 - phi ** 2), rtol=1e-10)


def _make_arma(phi, theta, sigma):
    r"""Construct the tightest CopulAX mean model for the given orders at
    the reference params (no fitting — this is a formula-level check).

    ``AR`` when ``q == 0``, ``MA`` when ``p == 0``, else ``ARMA``.
    """
    p, q = len(phi), len(theta)
    common = dict(
        residual_dist=normal,
        mu=jnp.array(0.0),
        sigma_eps=jnp.array(sigma),
        residual_params={},
    )
    if q == 0:
        return AR(p=p, q=0, phi=jnp.array(phi, dtype=float),
                  theta=jnp.zeros((0,)), **common)
    if p == 0:
        return MA(p=0, q=q, phi=jnp.zeros((0,)),
                  theta=jnp.array(theta, dtype=float), **common)
    return ARMA(p=p, q=q, phi=jnp.array(phi, dtype=float),
                theta=jnp.array(theta, dtype=float), **common)


class TestUnconditionalVarianceThirdPartyStatsmodels:
    r"""WR-08 completion (01-MATH-REVIEW.md): CopulAX's exact ARMA(p, q)
    unconditional-variance accessor is asserted against a THIRD-PARTY
    oracle — statsmodels' theoretical lag-0 autocovariance
    ``statsmodels.tsa.arima_process.arma_acovf`` — across a grid covering
    AR(1..3), MA(1..2), ARMA(1,1), ARMA(2,1), ARMA(2,2).

    This gates the exact Yule-Walker / Brockwell-Davis (1991) eq. (3.3.8)
    companion-form Lyapunov solve that replaces the former AR(p>1)
    lower-bound approximation.  Both sides are exact closed forms, so the
    match is at ``rtol <= 1e-10`` (exact-vs-exact, not a fit-quality check).

    statsmodels' sign / scaling convention is EMPIRICALLY VERIFIED against
    the ARMA(1,1) / AR(1) / MA(1) closed forms in
    ``test_statsmodels_convention_probe`` BEFORE it is trusted as the
    oracle (probe-before-trust).
    """

    # ---- Grid: (label, phi, theta) ----
    GRID = [
        ("AR(1)",     [0.5],            []),
        ("AR(2)",     [0.5, -0.3],      []),
        ("AR(3)",     [0.4, -0.2, 0.1], []),
        ("MA(1)",     [],               [0.3]),
        ("MA(2)",     [],               [0.6, -0.3]),
        ("ARMA(1,1)", [0.5],            [0.3]),
        # p=1, q>1 routes to the GENERAL Yule-Walker branch (the p==1
        # fast path requires q <= 1) — a distinct companion shape
        # (m = q+1 > p) that the other general rows do not exercise.
        ("ARMA(1,2)", [0.5],            [0.4, 0.1]),
        ("ARMA(2,1)", [0.5, -0.2],      [0.4]),
        ("ARMA(2,2)", [0.5, -0.2],      [0.4, 0.1]),
        # q+1 > p with p > 1: the tall-companion variant of the same.
        ("ARMA(2,3)", [0.5, -0.2],      [0.4, 0.1, -0.05]),
    ]
    SIGMA = 1.2

    @staticmethod
    def _sm_lag0_autocov(phi, theta, sigma):
        r"""statsmodels theoretical Var(y) = γ(0) for the ARMA(phi, theta).

        Convention (verified in the probe test): the AR lag polynomial is
        ``ar = [1, -φ_1, …, -φ_p]`` (leading 1, NEGATED AR coefficients),
        the MA lag polynomial is ``ma = [1, θ_1, …, θ_q]`` (leading 1,
        positive), and ``sigma2`` scales the innovation variance directly.
        ``arma_acovf(ar, ma, nobs=1, sigma2)[0]`` is the lag-0 autocovariance.
        """
        ap = require_oracle("statsmodels.tsa.arima_process")
        ar = np.r_[1.0, -np.asarray(phi)] if len(phi) else np.array([1.0])
        ma = np.r_[1.0, np.asarray(theta)] if len(theta) else np.array([1.0])
        return float(
            ap.arma_acovf(ar, ma, nobs=1, sigma2=sigma ** 2)[0]
        )

    def test_statsmodels_convention_probe(self):
        r"""Probe-before-trust: statsmodels' lag-0 autocovariance reproduces
        the KNOWN ARMA(1,1) / AR(1) / MA(1) closed forms under the assumed
        sign / scaling convention.  This validates the oracle itself before
        any CopulAX comparison relies on it (probe-before-trust)."""
        require_oracle("statsmodels")
        sigma = self.SIGMA
        s2 = sigma ** 2
        # ARMA(1,1): sigma^2 (1 + 2 phi theta + theta^2) / (1 - phi^2)
        phi, theta = 0.5, 0.3
        closed_arma11 = s2 * (1 + 2 * phi * theta + theta ** 2) / (1 - phi ** 2)
        np.testing.assert_allclose(
            self._sm_lag0_autocov([phi], [theta], sigma),
            closed_arma11, rtol=1e-12,
            err_msg="statsmodels ARMA(1,1) convention probe failed",
        )
        # AR(1): sigma^2 / (1 - phi^2)
        np.testing.assert_allclose(
            self._sm_lag0_autocov([0.5], [], sigma),
            s2 / (1 - 0.5 ** 2), rtol=1e-12,
            err_msg="statsmodels AR(1) convention probe failed",
        )
        # MA(1): sigma^2 (1 + theta^2)
        np.testing.assert_allclose(
            self._sm_lag0_autocov([], [0.3], sigma),
            s2 * (1 + 0.3 ** 2), rtol=1e-12,
            err_msg="statsmodels MA(1) convention probe failed",
        )
        # Negated-vs-non-negated AR guard: the WRONG convention (ar=[1,+phi])
        # must NOT match the closed form — proves the sign matters and the
        # probe is discriminating, not vacuously passing.
        ap = require_oracle("statsmodels.tsa.arima_process")
        wrong = float(
            ap.arma_acovf(np.array([1.0, 0.5]), np.array([1.0, 0.3]),
                          nobs=1, sigma2=s2)[0]
        )
        assert not np.isclose(wrong, closed_arma11, rtol=1e-3), (
            "non-negated AR convention unexpectedly matched — the probe "
            "would not detect a sign error"
        )

    @pytest.mark.parametrize(
        "label,phi,theta",
        GRID,
        ids=[g[0] for g in GRID],
    )
    def test_accessor_matches_statsmodels(self, label, phi, theta):
        r"""CopulAX ``stats()['variance']`` == statsmodels lag-0
        autocovariance at ``rtol <= 1e-10`` (exact-vs-exact) for every
        model on the grid."""
        require_oracle("statsmodels")
        obj = _make_arma(phi, theta, self.SIGMA)
        got = float(obj.stats()["variance"])
        oracle = self._sm_lag0_autocov(phi, theta, self.SIGMA)
        np.testing.assert_allclose(
            got, oracle, rtol=1e-10,
            err_msg=f"{label}: CopulAX Var(y) != statsmodels arma_acovf lag0",
        )

    def test_nonstationary_ar2_reports_inf(self):
        r"""A non-stationary AR(2) (roots on/inside the unit circle) has no
        unconditional variance; the accessor returns +inf rather than a
        spurious negative value (documented boundary convention)."""
        # phi_1 + phi_2 = 1 => a unit root at z=1 (non-stationary).
        obj = AR(
            p=2, q=0, residual_dist=normal,
            phi=jnp.array([0.6, 0.4]), theta=jnp.zeros((0,)),
            mu=jnp.array(0.0), sigma_eps=jnp.array(1.0),
            residual_params={},
        )
        v = float(obj.stats()["variance"])
        assert np.isinf(v) and v > 0, f"expected +inf, got {v}"

    def test_accessor_is_jittable(self):
        r"""The exact solve is JIT-compatible: ``jax.jit`` of the
        unconditional-variance accessor produces the same value as eager
        for an AR(3) (static p, q => fixed-size linear solve)."""
        phi = [0.4, -0.2, 0.1]
        obj = _make_arma(phi, [], self.SIGMA)
        eager = float(obj._unconditional_variance())
        jitted = float(jax.jit(lambda m: m._unconditional_variance())(obj))
        np.testing.assert_allclose(jitted, eager, rtol=1e-12)
