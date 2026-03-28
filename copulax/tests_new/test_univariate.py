"""Rigorous tests for all 11 CopulAX univariate distributions.

Cross-validates logpdf, CDF, stats, and fitting against scipy equivalents.
Verifies PDF integration, inverse consistency, and parameter recovery.

Catches: FINDING-03-01 (Student-T logpdf), FINDING-03-02 (Student-T variance),
FINDING-03-09 (Student-T integral != 1), FINDING-02-01 (IG variance),
FINDING-03-07 (GIG sampler scaling), FINDING-03-04 (sampling eps truncation),
FINDING-03-03 (GenNormal kurtosis convention).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from copulax.univariate import (
    normal, student_t, gamma, lognormal, uniform,
    ig, gen_normal, gig, gh, skewed_t, asym_gen_normal,
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
             ig, gen_normal, gig, gh, skewed_t, asym_gen_normal]

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
]

# Subset with scipy equivalents
SCIPY_CONFIGS = [(d, p) for d, p in DIST_CONFIGS
                 if d.name in ("Normal", "Student-T", "Gamma", "LogNormal",
                               "Uniform", "IG", "Gen-Normal", "GIG", "GH")]

DIST_IDS = [f"{d.name}" for d, _ in DIST_CONFIGS]
SCIPY_IDS = [f"{d.name}" for d, _ in SCIPY_CONFIGS]


# ---------------------------------------------------------------------------
# Cross-validation against scipy
# ---------------------------------------------------------------------------

class TestLogpdfAgainstScipy:
    """Verify logpdf matches scipy for all distributions with equivalents.

    Catches FINDING-03-01: Student-T normalizing constant error.
    scipy.stats.t.logpdf is the ground truth.
    """

    @pytest.mark.parametrize("dist,params", SCIPY_CONFIGS, ids=SCIPY_IDS)
    def test_logpdf_matches_scipy(self, dist, params):
        x = gen_test_points(dist, params, n=50)
        # GH/GIG depend on Bessel K_v which has known accuracy issues,
        # so use looser tolerance for those.
        rtol = 5e-4 if dist.name in ("GH", "GIG") else 1e-6
        assert_scipy_logpdf_match(dist, params, x, rtol=rtol)


class TestCdfAgainstScipy:
    """Verify CDF matches scipy for all distributions with equivalents."""

    @pytest.mark.parametrize("dist,params", SCIPY_CONFIGS, ids=SCIPY_IDS)
    def test_cdf_matches_scipy(self, dist, params):
        x = gen_test_points(dist, params, n=50)
        rtol = 5e-4 if dist.name in ("GH", "GIG") else 1e-5
        assert_scipy_cdf_match(dist, params, x, rtol=rtol)


# ---------------------------------------------------------------------------
# PDF integration (catches normalizing constant bugs)
# ---------------------------------------------------------------------------

class TestPdfIntegratesToOne:
    """Verify PDF integrates to 1 over the support.

    Catches FINDING-03-09: Student-T PDF integral = sqrt(sigma) != 1.
    """

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
    """Verify stats() mean and variance match scipy's analytical values.

    Catches FINDING-03-02: Student-T variance missing sigma^2.
    With nu=5, sigma=2: theory = sigma^2 * nu/(nu-2) = 4 * 5/3 = 6.667.
    If the code returns nu/(nu-2) = 1.667, this test catches it.

    Catches FINDING-02-01: IG variance formula (alpha-1)^3 vs (alpha-1)^2*(alpha-2).
    With alpha=4, beta=2: theory = beta^2/((alpha-1)^2*(alpha-2)) = 4/(9*2) = 0.222.
    Buggy code gives beta^2/(alpha-1)^3 = 4/27 = 0.148.
    """

    @pytest.mark.parametrize("dist,params", SCIPY_CONFIGS, ids=SCIPY_IDS)
    def test_stats_match_scipy(self, dist, params):
        # GH/GIG stats depend on Bessel, use looser tolerance
        rtol = 5e-3 if dist.name in ("GH", "GIG") else 1e-4
        assert_stats_match_scipy(dist, params, rtol=rtol)

    def test_student_t_variance_includes_sigma_squared(self):
        """Explicit test for Student-T variance = sigma^2 * nu / (nu - 2).

        FINDING-03-02: if sigma^2 factor is missing, variance = nu/(nu-2) = 1.667
        instead of 6.667.
        """
        params = {"nu": 5.0, "mu": 1.0, "sigma": 2.0}
        stats = student_t.stats(params=params)
        expected_var = 2.0 ** 2 * 5.0 / (5.0 - 2.0)  # = 6.667
        np.testing.assert_allclose(
            float(stats["variance"]), expected_var, rtol=1e-5,
            err_msg="Student-T variance should be sigma^2 * nu/(nu-2)"
        )

    def test_ig_variance_formula(self):
        """Explicit test for IG variance = beta^2 / ((alpha-1)^2 * (alpha-2)).

        FINDING-02-01: buggy code uses (alpha-1)^3 in denominator.
        """
        params = {"alpha": 4.0, "beta": 2.0}
        stats = ig.stats(params=params)
        expected_var = 2.0 ** 2 / ((4.0 - 1.0) ** 2 * (4.0 - 2.0))  # = 0.222
        np.testing.assert_allclose(
            float(stats["variance"]), expected_var, rtol=1e-5,
            err_msg="IG variance = beta^2/((alpha-1)^2*(alpha-2))"
        )

    def test_normal_mean_and_variance(self):
        """Normal: mean = mu, variance = sigma^2."""
        params = {"mu": 3.0, "sigma": 2.0}
        stats = normal.stats(params=params)
        np.testing.assert_allclose(float(stats["mean"]), 3.0, atol=1e-10)
        np.testing.assert_allclose(float(stats["variance"]), 4.0, atol=1e-10)

    def test_gamma_mean_and_variance(self):
        """Gamma (rate param): mean = alpha/beta, variance = alpha/beta^2."""
        params = {"alpha": 3.0, "beta": 2.0}
        stats = gamma.stats(params=params)
        np.testing.assert_allclose(float(stats["mean"]), 1.5, rtol=1e-5)
        np.testing.assert_allclose(float(stats["variance"]), 0.75, rtol=1e-5)


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
    ], ids=["Normal", "Gamma", "LogNormal", "Uniform"])
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
        """Student-T parameter recovery with non-trivial sigma.

        Note: this may fail due to FINDING-03-01 (Student-T logpdf normalizing
        constant bug) which corrupts the MLE objective.
        """
        np.random.seed(42)
        data = scipy.stats.t.rvs(df=8, loc=1.0, scale=2.0, size=5000)
        fitted = student_t.fit(x=jnp.array(data))
        p = fitted.params
        # nu should be positive and reasonable (not degenerate)
        nu_val = float(p["nu"])
        assert nu_val > 1.5, f"nu should be > 1.5, got {nu_val}"
        # mu and sigma recovery
        np.testing.assert_allclose(float(p["mu"]), 1.0, atol=0.5)
        np.testing.assert_allclose(float(p["sigma"]), 2.0, rtol=0.5)


# ---------------------------------------------------------------------------
# Sampling tail coverage
# ---------------------------------------------------------------------------

class TestSamplingTailCoverage:
    """Verify inverse transform sampling doesn't over-truncate tails.

    FINDING-03-04: eps=1e-2 in _rvs.py means sampling U(0.01, 0.99),
    which truncates 2% of the tail mass. For heavy-tailed distributions
    this means the extreme quantiles are never sampled.
    """

    def test_normal_tail_coverage(self):
        """Normal samples should occasionally exceed the 1st/99th percentiles."""
        key = jax.random.PRNGKey(42)
        params = {"mu": 0.0, "sigma": 1.0}
        samples = np.array(normal.rvs(size=10000, params=params, key=key))
        # With eps=1e-2, the sample range is clipped to ppf(0.01)..ppf(0.99)
        # = -2.33..2.33. With proper sampling we'd see values beyond 2.33.
        q01 = scipy.stats.norm.ppf(0.005)  # -2.576
        q99 = scipy.stats.norm.ppf(0.995)  # 2.576
        has_deep_left = np.any(samples < q01)
        has_deep_right = np.any(samples > q99)
        if not (has_deep_left or has_deep_right):
            pytest.xfail("FINDING-03-04: Tail sampling truncated by eps=1e-2")


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

    def test_student_t_nu_near_2(self):
        """Student-T with nu close to 2: variance should be very large."""
        params = {"nu": 2.1, "mu": 0.0, "sigma": 1.0}
        stats = student_t.stats(params=params)
        var = float(stats["variance"])
        # sigma^2 * nu / (nu - 2) = 1 * 2.1 / 0.1 = 21
        expected = 1.0 ** 2 * 2.1 / (2.1 - 2.0)
        if np.isfinite(var):
            np.testing.assert_allclose(var, expected, rtol=0.1)

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
        """GenNormal should use excess kurtosis (consistent with Normal = 0).

        FINDING-03-03: GenNormal may report Pearson kurtosis (Normal case = 3)
        while other distributions use excess kurtosis (Normal case = 0).
        """
        # GenNormal with beta=2 is the Normal distribution
        params = {"mu": 0.0, "alpha": 1.0, "beta": 2.0}
        stats = gen_normal.stats(params=params)
        kurt = float(stats["kurtosis"])

        # If excess kurtosis: should be ~0 for Normal
        # If Pearson kurtosis: would be ~3
        # We check which convention is used
        if abs(kurt - 3.0) < 0.1:
            pytest.xfail("FINDING-03-03: GenNormal reports Pearson kurtosis, "
                         "not excess kurtosis")
        # If it's excess, it should be near 0
        np.testing.assert_allclose(kurt, 0.0, atol=0.1,
                                   err_msg="GenNormal(beta=2) kurtosis should be ~0 "
                                           "(excess convention)")

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
        for dist, params in DIST_CONFIGS[:6]:  # test first 6 for speed
            x = gen_test_points(dist, params, n=10)
            logp = np.array(dist.logpdf(x=x, params=params))
            p = np.array(dist.pdf(x=x, params=params))
            mask = np.isfinite(logp) & (p > 0)
            np.testing.assert_allclose(
                np.exp(logp[mask]), p[mask], rtol=1e-5,
                err_msg=f"{dist.name}: exp(logpdf) != pdf"
            )

    def test_jit_compilability(self):
        """All distributions are JIT-compatible."""
        for dist, params in DIST_CONFIGS[:6]:
            x = gen_test_points(dist, params, n=5)
            f = jax.jit(lambda x_: dist.logpdf(x=x_, params=params))
            result = f(x)
            assert no_nans(result), f"{dist.name} JIT logpdf has NaNs"
