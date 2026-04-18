"""Rigorous tests for all 11 CopulAX univariate distributions.

Cross-validates logpdf, CDF, stats, and fitting against scipy equivalents.
Verifies PDF integration, inverse consistency, and parameter recovery.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from copulax.univariate import (
    normal, student_t, gamma, lognormal, uniform,
    ig, gen_normal, gig, gh, skewed_t, asym_gen_normal, wald,
)
from copulax.tests_new.conftest import (
    get_scipy_dist, gen_test_points, assert_scipy_logpdf_match,
    assert_scipy_cdf_match, assert_pdf_integrates_to_one,
    assert_inverse_consistency, assert_stats_match_scipy,
    no_nans, is_finite,
)


# ---------------------------------------------------------------------------
# Distribution configurations for parametrized tests
# ---------------------------------------------------------------------------

# Distributions with parameters chosen to exercise non-trivial code paths
# and expose known bugs.

ALL_DISTS = [normal, student_t, gamma, lognormal, uniform,
             ig, gen_normal, gig, gh, skewed_t, asym_gen_normal, wald]

# Configs: (dist, params) tuples with carefully chosen parameters
DIST_CONFIGS = [
    (normal, {"mu": 2.0, "sigma": 1.5}),
    (student_t, {"nu": 5.0, "mu": 1.0, "sigma": 2.0}),
    (gamma, {"alpha": 3.0, "beta": 2.0}),
    (lognormal, {"mu": 0.5, "sigma": 0.8}),
    (uniform, {"a": -1.0, "b": 3.0}),
    (ig, {"alpha": 4.0, "beta": 2.0}),
    (gen_normal, {"mu": 0.5, "alpha": 1.5, "beta": 2.0}),
    (gig, {"lambda": 1.0, "chi": 2.0, "psi": 3.0}),
    (gh, {"lambda": 1.0, "chi": 2.0, "psi": 3.0,
          "mu": 0.5, "sigma": 1.0, "gamma": 0.0}),
    (skewed_t, {"nu": 6.0, "mu": 0.0, "sigma": 1.0, "gamma": 0.5}),
    (asym_gen_normal, {"zeta": 0.0, "alpha": 1.0, "kappa": -0.5}),
    (wald, {"mu": 1.0, "lamb": 2.0}),
]

# Subset with scipy equivalents
SCIPY_CONFIGS = [(d, p) for d, p in DIST_CONFIGS
                 if d.name in ("Normal", "Student-T", "Gamma", "LogNormal",
                               "Uniform", "IG", "Gen-Normal", "GIG", "GH",
                               "Wald")]

DIST_IDS = [f"{d.name}" for d, _ in DIST_CONFIGS]
SCIPY_IDS = [f"{d.name}" for d, _ in SCIPY_CONFIGS]


# ---------------------------------------------------------------------------
# Cross-validation against scipy
# ---------------------------------------------------------------------------

class TestLogpdfAgainstScipy:
    """Verify logpdf matches scipy for all distributions with equivalents."""

    @pytest.mark.parametrize("dist,params", SCIPY_CONFIGS, ids=SCIPY_IDS)
    def test_logpdf_matches_scipy(self, dist, params):
        x = gen_test_points(dist, params, n=50)
        assert_scipy_logpdf_match(dist, params, x, rtol=1e-6)


class TestCdfAgainstScipy:
    """Verify CDF matches scipy for all distributions with equivalents."""

    @pytest.mark.parametrize("dist,params", SCIPY_CONFIGS, ids=SCIPY_IDS)
    def test_cdf_matches_scipy(self, dist, params):
        x = gen_test_points(dist, params, n=50)
        assert_scipy_cdf_match(dist, params, x, rtol=1e-5)


# ---------------------------------------------------------------------------
# PDF integration (catches normalizing constant bugs)
# ---------------------------------------------------------------------------

class TestPdfIntegratesToOne:
    """Verify PDF integrates to 1 over the support."""

    @pytest.mark.parametrize("dist,params", DIST_CONFIGS, ids=DIST_IDS)
    def test_pdf_integrates_to_one(self, dist, params):
        # Skip distributions where quadrature is unreliable
        if dist.name == "Asym-Gen-Normal":
            pytest.skip("Asym-Gen-Normal has complex support for quadrature")
        assert_pdf_integrates_to_one(dist, params, rtol=1e-2)


# ---------------------------------------------------------------------------
# CDF-PPF inverse consistency
# ---------------------------------------------------------------------------

class TestInverseConsistency:
    """Verify CDF(PPF(q)) ≈ q for all distributions."""

    @pytest.mark.parametrize("dist,params", DIST_CONFIGS, ids=DIST_IDS)
    def test_cdf_ppf_roundtrip(self, dist, params):
        if dist.name == "Asym-Gen-Normal":
            pytest.skip("Asym-Gen-Normal PPF not reliable for roundtrip test")
        assert_inverse_consistency(dist, params, rtol=1e-2, n_points=15)


# ---------------------------------------------------------------------------
# Stats vs theory (catches moment formula bugs)
# ---------------------------------------------------------------------------

class TestStatsAgainstTheory:
    """Verify stats() mean and variance match scipy's analytical values."""

    @pytest.mark.parametrize("dist,params", SCIPY_CONFIGS, ids=SCIPY_IDS)
    def test_stats_match_scipy(self, dist, params):
        assert_stats_match_scipy(dist, params, rtol=1e-4)


class TestStatsAgainstSampling:
    """Verify stats() moments match large-sample empirical moments.

    Catches FINDING-03-07: GIG sampler missing sqrt(chi/psi) scaling.
    If the sampler is wrong, sample mean won't match analytical mean.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("dist,params", DIST_CONFIGS, ids=DIST_IDS)
    def test_sample_moments_match_stats(self, dist, params):
        if dist.name in ("Uniform", "Asym-Gen-Normal"):
            pytest.skip("Trivial or complex support")

        key = jax.random.PRNGKey(123)
        n = 100_000
        samples = np.array(dist.rvs(size=n, params=params, key=key))

        # Remove any NaN/inf samples
        samples = samples[np.isfinite(samples)]
        if len(samples) < n // 2:
            pytest.xfail(f"{dist.name}: >50% non-finite samples")

        stats = dist.stats(params=params)
        analytical_mean = float(stats["mean"])
        analytical_var = float(stats["variance"])

        if np.isfinite(analytical_mean):
            sample_mean = np.mean(samples)
            np.testing.assert_allclose(
                sample_mean, analytical_mean, rtol=0.05,
                err_msg=f"{dist.name} sample mean doesn't match stats()"
            )

        if np.isfinite(analytical_var) and analytical_var > 0:
            sample_var = np.var(samples)
            np.testing.assert_allclose(
                sample_var, analytical_var, rtol=0.15,
                err_msg=f"{dist.name} sample variance doesn't match stats()"
            )


# ---------------------------------------------------------------------------
# Parameter recovery via fitting
# ---------------------------------------------------------------------------

class TestParameterRecovery:
    """Verify fit() recovers known parameters from synthetic data."""

    @pytest.mark.parametrize("dist,params", [
        (normal, {"mu": 2.0, "sigma": 1.5}),
        (gamma, {"alpha": 3.0, "beta": 2.0}),
        (lognormal, {"mu": 0.5, "sigma": 0.8}),
        (uniform, {"a": -1.0, "b": 3.0}),
        (wald, {"mu": 2.0, "lamb": 5.0}),
    ], ids=["Normal", "Gamma", "LogNormal", "Uniform", "Wald"])
    def test_simple_parameter_recovery(self, dist, params):
        """Simple distributions: fit should recover params from 5000 samples."""
        sp = get_scipy_dist(dist, params)
        np.random.seed(42)
        data = sp.rvs(size=5000)

        fitted = dist.fit(x=jnp.array(data))
        fitted_params = fitted.params

        for key in params:
            np.testing.assert_allclose(
                float(fitted_params[key]), float(params[key]),
                rtol=0.15, atol=0.1,
                err_msg=f"{dist.name} param '{key}' not recovered"
            )

    def test_student_t_recovery(self):
        """Student-T parameter recovery with non-trivial sigma."""
        np.random.seed(42)
        data = scipy.stats.t.rvs(df=8, loc=1.0, scale=2.0, size=5000)
        fitted = student_t.fit(x=jnp.array(data))
        p = fitted.params
        nu_val = float(p["nu"])
        assert nu_val > 1.5, f"nu should be > 1.5, got {nu_val}"
        np.testing.assert_allclose(float(p["mu"]), 1.0, atol=0.5)
        np.testing.assert_allclose(float(p["sigma"]), 2.0, rtol=0.5)


# ---------------------------------------------------------------------------
# Sampling tail coverage
# ---------------------------------------------------------------------------

class TestSamplingTailCoverage:
    """Verify inverse-transform sampling reaches deep tail quantiles."""

    def test_normal_tail_coverage(self):
        """Normal samples should reach beyond the 0.5th/99.5th percentiles."""
        key = jax.random.PRNGKey(42)
        params = {"mu": 0.0, "sigma": 1.0}
        samples = np.array(normal.rvs(size=10000, params=params, key=key))
        q01 = scipy.stats.norm.ppf(0.005)
        q99 = scipy.stats.norm.ppf(0.995)
        assert np.any(samples < q01) or np.any(samples > q99), (
            "Tail sampling appears truncated — check eps in _rvs.py"
        )


# ---------------------------------------------------------------------------
# Gradient correctness
# ---------------------------------------------------------------------------

class TestGradientCorrectness:
    """Verify jax.grad(logpdf) matches finite differences."""

    @pytest.mark.parametrize("dist,params", [
        (normal, {"mu": 0.0, "sigma": 1.0}),
        (gamma, {"alpha": 3.0, "beta": 2.0}),
        (student_t, {"nu": 5.0, "mu": 0.0, "sigma": 1.0}),
        (lognormal, {"mu": 0.0, "sigma": 1.0}),
        (gen_normal, {"mu": 0.0, "alpha": 1.5, "beta": 2.0}),
    ], ids=["Normal", "Gamma", "StudentT", "LogNormal", "GenNormal"])
    def test_logpdf_gradient_vs_finite_diff(self, dist, params):
        """d/dx logpdf(x) via jax.grad vs central finite differences."""
        x = gen_test_points(dist, params, n=10)
        h = 1e-5

        for i in range(min(5, len(x))):
            xi = float(x[i])

            def logpdf_scalar(xval):
                return dist.logpdf(x=xval, params=params).flatten()[0]

            analytic = float(jax.grad(logpdf_scalar)(jnp.array(xi)))
            numerical = (float(logpdf_scalar(jnp.array(xi + h)))
                         - float(logpdf_scalar(jnp.array(xi - h)))) / (2 * h)

            if np.isfinite(analytic) and np.isfinite(numerical) and abs(numerical) > 1e-8:
                np.testing.assert_allclose(
                    analytic, numerical, rtol=1e-2,
                    err_msg=f"{dist.name} gradient mismatch at x={xi}"
                )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and convention checks."""

    def test_logpdf_outside_support_is_neg_inf(self):
        """logpdf should return -inf outside the support."""
        # Uniform
        params = {"a": 0.0, "b": 1.0}
        x = jnp.array([-1.0, 2.0])
        vals = np.array(uniform.logpdf(x=x, params=params)).flatten()
        assert np.all(vals == -np.inf), f"logpdf outside support != -inf, got {vals}"

        # Gamma (support [0, inf))
        params_g = {"alpha": 2.0, "beta": 1.0}
        x_neg = jnp.array([-1.0])
        val_g = float(gamma.logpdf(x=x_neg, params=params_g).flatten()[0])
        assert val_g == -np.inf or val_g < -1e10, f"Gamma logpdf(-1) = {val_g}"

    def test_cdf_boundary_values(self):
        """CDF should be 0 below support and 1 above support."""
        params = {"a": 0.0, "b": 1.0}
        below = float(uniform.cdf(x=jnp.array(-1.0), params=params))
        above = float(uniform.cdf(x=jnp.array(2.0), params=params))
        assert below == 0.0, f"CDF below support = {below}, expected 0"
        assert above == 1.0, f"CDF above support = {above}, expected 1"

    def test_gen_normal_kurtosis_convention(self):
        """GenNormal(beta=2) is Normal; excess kurtosis should be ~0."""
        params = {"mu": 0.0, "alpha": 1.0, "beta": 2.0}
        stats = gen_normal.stats(params=params)
        kurt = float(stats["kurtosis"])
        np.testing.assert_allclose(
            kurt, 0.0, atol=0.1,
            err_msg="GenNormal(beta=2) kurtosis should be ~0 (excess convention)"
        )

    def test_sampling_shape_correctness(self):
        """rvs() returns correct shapes for various size arguments."""
        params = normal.example_params()
        key = jax.random.PRNGKey(0)

        s1 = normal.rvs(size=10, params=params, key=key)
        assert s1.shape == (10,), f"Expected (10,), got {s1.shape}"

        s0 = normal.rvs(size=0, params=params, key=key)
        assert s0.shape == (0,), f"Expected (0,), got {s0.shape}"

    def test_support_correctness(self):
        """Support bounds are correct for each distribution."""
        # Normal: (-inf, inf)
        s = np.array(normal._support({"mu": 0.0, "sigma": 1.0})).flatten()
        assert s[0] == -np.inf and s[1] == np.inf

        # Gamma: (0, inf)
        s = np.array(gamma._support({"alpha": 1.0, "beta": 1.0})).flatten()
        assert s[0] == 0.0 and s[1] == np.inf

        # Uniform: (a, b)
        s = np.array(uniform._support({"a": 1.0, "b": 3.0})).flatten()
        assert s[0] == 1.0 and s[1] == 3.0

    def test_logpdf_pdf_consistency(self):
        """exp(logpdf(x)) == pdf(x) for all distributions."""
        for dist, params in DIST_CONFIGS:
            x = gen_test_points(dist, params, n=10)
            logp = np.array(dist.logpdf(x=x, params=params))
            p = np.array(dist.pdf(x=x, params=params))
            mask = np.isfinite(logp) & (p > 0)
            if mask.sum() == 0:
                continue
            np.testing.assert_allclose(
                np.exp(logp[mask]), p[mask], rtol=1e-5,
                err_msg=f"{dist.name}: exp(logpdf) != pdf"
            )

    def test_jit_compilability(self):
        """All distributions are JIT-compatible."""
        for dist, params in DIST_CONFIGS:
            x = gen_test_points(dist, params, n=5)
            f = jax.jit(lambda x_: dist.logpdf(x=x_, params=params))
            result = f(x)
            assert no_nans(result), f"{dist.name} JIT logpdf has NaNs"


# ---------------------------------------------------------------------------
# logcdf consistency
# ---------------------------------------------------------------------------

class TestLogCdf:
    """Verify logcdf matches log(cdf) for distributions with scipy equivalents."""

    @pytest.mark.parametrize("dist,params", SCIPY_CONFIGS, ids=SCIPY_IDS)
    def test_logcdf_matches_log_cdf(self, dist, params):
        """logcdf(x) should equal log(cdf(x)) for interior points."""
        x = gen_test_points(dist, params, n=30)
        logcdf_val = np.array(dist.logcdf(x=x, params=params))
        cdf_val = np.array(dist.cdf(x=x, params=params))

        mask = (cdf_val > 1e-15) & np.isfinite(logcdf_val)
        np.testing.assert_allclose(
            logcdf_val[mask], np.log(cdf_val[mask]), rtol=1e-5,
            err_msg=f"{dist.name}: logcdf != log(cdf)"
        )


# ---------------------------------------------------------------------------
# Asym-Gen-Normal numerical validation (skipped by generic scaffolding)
# ---------------------------------------------------------------------------

class TestAsymGenNormalValidation:
    """Parameter-aware tests for Asym-Gen-Normal.

    The generic TestPdfIntegratesToOne and TestInverseConsistency skip this
    distribution because its support depends on parameters. These tests
    use parameter-aware domains to fill the gap.
    """

    def test_logpdf_finite_on_interior(self):
        """logpdf should be finite at interior points of support."""
        params = {"zeta": 0.0, "alpha": 1.0, "kappa": -0.5}
        x = jnp.linspace(-3.0, 3.0, 50)
        lp = np.array(asym_gen_normal.logpdf(x=x, params=params))
        # At least most interior points should be finite
        n_finite = np.sum(np.isfinite(lp))
        assert n_finite >= 40, f"Only {n_finite}/50 finite logpdf values"

    def test_pdf_positive_on_interior(self):
        """pdf should be > 0 for points within support."""
        params = {"zeta": 0.0, "alpha": 1.0, "kappa": -0.5}
        x = jnp.linspace(-2.0, 2.0, 30)
        p = np.array(asym_gen_normal.pdf(x=x, params=params))
        mask = np.isfinite(p)
        assert np.all(p[mask] >= 0), "pdf has negative values"
        assert np.sum(p[mask] > 0) >= 20, "Too few positive pdf values"

    def test_cdf_monotone(self):
        """cdf should be non-decreasing."""
        params = {"zeta": 0.0, "alpha": 1.0, "kappa": -0.5}
        x = jnp.linspace(-4.0, 4.0, 100)
        c = np.array(asym_gen_normal.cdf(x=x, params=params))
        mask = np.isfinite(c)
        c_valid = c[mask]
        diffs = np.diff(c_valid)
        assert np.all(diffs >= -1e-10), "cdf is not monotone"

    def test_cdf_boundary_values(self):
        """cdf should approach 0 at left and 1 at right of support."""
        params = {"zeta": 0.0, "alpha": 1.0, "kappa": -0.5}
        c_left = float(asym_gen_normal.cdf(x=jnp.array(-10.0), params=params))
        c_right = float(asym_gen_normal.cdf(x=jnp.array(10.0), params=params))
        assert c_left < 0.01, f"cdf(-10) = {c_left}, expected near 0"
        assert c_right > 0.99, f"cdf(10) = {c_right}, expected near 1"

    def test_sampling_shape(self):
        """rvs should produce the right shape."""
        params = {"zeta": 0.0, "alpha": 1.0, "kappa": -0.5}
        key = jax.random.PRNGKey(42)
        samples = asym_gen_normal.rvs(size=100, params=params, key=key)
        assert samples.shape == (100,)
        assert np.all(np.isfinite(np.array(samples)))
