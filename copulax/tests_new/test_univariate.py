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
    ig, gen_normal, gig, gh, skewed_t, asym_gen_normal, wald, nig,
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
             ig, gen_normal, gig, gh, skewed_t, asym_gen_normal, wald, nig]

# Configs: (dist, params) tuples with carefully chosen parameters
DIST_CONFIGS = [
    (normal, {"mu": 2.0, "sigma": 1.5}),
    (student_t, {"nu": 5.0, "mu": 1.0, "sigma": 2.0}),
    (gamma, {"alpha": 3.0, "beta": 2.0}),
    (lognormal, {"mu": 0.5, "sigma": 0.8}),
    (uniform, {"a": -1.0, "b": 3.0}),
    (ig, {"alpha": 4.0, "beta": 2.0}),
    (gen_normal, {"mu": 0.5, "alpha": 1.5, "beta": 2.0}),
    (gig, {"lamb": 1.0, "chi": 2.0, "psi": 3.0}),
    (gh, {"lamb": 1.0, "chi": 2.0, "psi": 3.0,
          "mu": 0.5, "sigma": 1.0, "gamma": 0.0}),
    (skewed_t, {"nu": 6.0, "mu": 0.0, "sigma": 1.0, "gamma": 0.5}),
    (asym_gen_normal, {"zeta": 0.0, "alpha": 1.0, "kappa": -0.5}),
    (wald, {"mu": 1.0, "lamb": 2.0}),
    (nig, {"mu": 0.0, "alpha": 2.5, "beta": 1.0, "delta": 1.0}),
]

# Subset with scipy equivalents
SCIPY_CONFIGS = [(d, p) for d, p in DIST_CONFIGS
                 if d.name in ("Normal", "Student-T", "Gamma", "LogNormal",
                               "Uniform", "IG", "Gen-Normal", "GIG", "GH",
                               "Wald", "NIG")]

DIST_IDS = [f"{d.name}" for d, _ in DIST_CONFIGS]
SCIPY_IDS = [f"{d.name}" for d, _ in SCIPY_CONFIGS]


# Asym-Gen-Normal flips which side of the support is bounded based on
# sign(kappa). DIST_CONFIGS already contains the kappa<0 (finite-lower)
# variant; we add a kappa>0 (finite-upper) variant so both branches of the
# support formula get exercised by the boundary-behaviour tests.
ASYM_GEN_NORMAL_POS_KAPPA = (
    asym_gen_normal,
    {"zeta": 0.0, "alpha": 1.0, "kappa": 0.5},
)

DIST_CONFIGS_FINITE_LOWER = [
    cfg for cfg in DIST_CONFIGS
    if cfg[0].name in ("Gamma", "LogNormal", "IG", "GIG", "Wald",
                       "Uniform", "Asym-Gen-Normal")
]  # Asym-Gen-Normal here is the kappa=-0.5 variant from DIST_CONFIGS.

DIST_CONFIGS_FINITE_UPPER = [
    cfg for cfg in DIST_CONFIGS if cfg[0].name == "Uniform"
] + [ASYM_GEN_NORMAL_POS_KAPPA]

# CDF saturation must cover both kappa polarities of Asym-Gen-Normal.
DIST_CONFIGS_WITH_AGN_BOTH = DIST_CONFIGS + [ASYM_GEN_NORMAL_POS_KAPPA]

FINITE_LOWER_IDS = [d.name for d, _ in DIST_CONFIGS_FINITE_LOWER]
FINITE_UPPER_IDS = [
    d.name if d.name != "Asym-Gen-Normal" else "Asym-Gen-Normal-PosKappa"
    for d, _ in DIST_CONFIGS_FINITE_UPPER
]
SATURATION_IDS = DIST_IDS + ["Asym-Gen-Normal-PosKappa"]


# Default `method=` kwarg passed to `fit()` by the JIT-contract test. Keyed by
# `dist.name`. Distributions whose `fit()` has no `method` kwarg are absent.
FIT_JIT_METHODS = {
    "Student-T": "ldmle",
    "Gen-Normal": "mle",
    "Asym-Gen-Normal": "mle",
    "GH": "em",
    "NIG": "em",
    "Skewed-T": "em",
}


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

    @pytest.mark.parametrize("dist,params,expect_inf_mean,expect_inf_var", [
        (skewed_t, {"nu": 3.0, "mu": 0.0, "sigma": 1.0, "gamma": 0.4}, False, True),
        (skewed_t, {"nu": 1.5, "mu": 0.0, "sigma": 1.0, "gamma": 0.4}, True,  True),
        (ig, {"alpha": 1.5, "beta": 1.0}, False, True),
        (ig, {"alpha": 0.5, "beta": 1.0}, True,  True),
    ], ids=["skewedT-nu3", "skewedT-nu1.5", "ig-alpha1.5", "ig-alpha0.5"])
    def test_divergent_moments_are_inf(
        self, dist, params, expect_inf_mean, expect_inf_var
    ):
        """Divergent moments must return +inf, not fabricated finite values or NaN."""
        s = dist.stats(params=params)
        m, v = float(s["mean"]), float(s["variance"])
        if expect_inf_mean:
            assert m == float("inf"), f"{dist.name}: mean should be +inf, got {m}"
        else:
            assert np.isfinite(m), f"{dist.name}: mean should be finite, got {m}"
        if expect_inf_var:
            assert v == float("inf"), f"{dist.name}: variance should be +inf, got {v}"
        else:
            assert np.isfinite(v), f"{dist.name}: variance should be finite, got {v}"


class TestStatsAgainstSampling:
    """Verify stats() moments match large-sample empirical moments.

    Catches FINDING-03-07: GIG sampler missing sqrt(chi/psi) scaling.
    If the sampler is wrong, sample mean won't match analytical mean.
    """

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

    @pytest.mark.parametrize("dist, params", [
        (skewed_t, {"nu": 6.0, "mu": 0.0, "sigma": 1.0, "gamma": 0.5}),
        (gh, {"lamb": -0.5, "chi": 2.0, "psi": 1.5,
              "mu": 0.0, "sigma": 1.0, "gamma": 0.4}),
    ], ids=["skewed_t", "gh"])
    def test_ldmle_within_5pct_of_oracle(self, dist, params):
        """LDMLE log-likelihood must land within 5% of oracle on n=2000 samples.

        Locks in the 1D feasibility reparam: without it, the optimiser rolls
        into a spurious basin via the silent sqrt(abs(sigma_sq)) repair and the
        LL gap blows past 5%.
        """
        key = jax.random.PRNGKey(42)
        samples = dist.rvs(size=2000, params=params, key=key)
        ll_true = float(jnp.sum(dist.logpdf(samples, params=params)))
        fitted = dist.fit(samples, method="ldmle")
        fp = fitted.params
        ll_fit = float(jnp.sum(dist.logpdf(samples, params=fp)))
        # ll_true is negative; ll_true * 1.05 is 5% more negative.
        assert ll_fit > ll_true * 1.05, (
            f"{dist.name}: LDMLE LL ({ll_fit:.1f}) too far from oracle "
            f"({ll_true:.1f})"
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

    @pytest.mark.parametrize("method", ["em", "mle", "mom"])
    def test_nig_recovery(self, method):
        """NIG parameter recovery via Karlis (2002) EM, 3-parameter MLE, and MoM."""
        true = {"mu": 0.0, "alpha": 2.5, "beta": 1.0, "delta": 1.0}
        sp = scipy.stats.norminvgauss(
            a=true["alpha"] * true["delta"], b=true["beta"] * true["delta"],
            loc=true["mu"], scale=true["delta"],
        )
        rng = np.random.default_rng(2026_04_18)
        x = jnp.asarray(sp.rvs(size=5000, random_state=rng))

        maxiter = 500 if method == "mle" else 200
        fitted = nig.fit(x, method=method, maxiter=maxiter).params
        # MoM is a moment-matching initialiser, not a full MLE — allow wider tol.
        rtol = 0.25 if method == "mom" else 0.1
        atol = 0.15 if method == "mom" else 0.05
        for k, v in true.items():
            np.testing.assert_allclose(
                float(fitted[k]), v, rtol=rtol, atol=atol,
                err_msg=f"NIG[{method}] param '{k}' not recovered"
            )

    def test_nig_em_and_mle_agree(self):
        """EM and MLE target the same likelihood optimum, so they must agree."""
        rng = np.random.default_rng(2026_04_18)
        sp = scipy.stats.norminvgauss(a=2.5, b=1.0, loc=0.0, scale=1.0)
        x = jnp.asarray(sp.rvs(size=5000, random_state=rng))
        em = nig.fit(x, method="em", maxiter=500).params
        mle = nig.fit(x, method="mle", maxiter=500).params
        for k in ("mu", "alpha", "beta", "delta"):
            np.testing.assert_allclose(
                float(em[k]), float(mle[k]), rtol=0.02, atol=0.005,
                err_msg=f"NIG EM/MLE disagree on '{k}'"
            )

    def test_nig_beta_score_identity_at_mle(self):
        """Karlis (2002) Lemma: at the MLE, ``μ = x̄ − δβ/γ`` exactly."""
        rng = np.random.default_rng(2026_04_18)
        sp = scipy.stats.norminvgauss(a=2.5, b=1.0, loc=0.0, scale=1.0)
        x = jnp.asarray(sp.rvs(size=5000, random_state=rng))
        p = nig.fit(x, method="mle", maxiter=500).params
        alpha, beta, delta, mu = (float(p[k]) for k in ("alpha", "beta", "delta", "mu"))
        gamma = np.sqrt(alpha ** 2 - beta ** 2)
        np.testing.assert_allclose(
            mu, float(x.mean()) - delta * beta / gamma,
            rtol=1e-10, atol=1e-10,
        )

    def test_nig_mom_fallback_on_near_normal_data(self):
        """MoM falls back to the symmetric-NIG branch when ``3·kurt − 5·skew² ≤ 0``."""
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.normal(loc=2.0, scale=1.5, size=3000))
        p = nig.fit(x, method="mom").params
        alpha, beta, delta, mu = (float(p[k]) for k in ("alpha", "beta", "delta", "mu"))
        assert alpha > 0.0 and delta > 0.0 and abs(beta) < alpha
        np.testing.assert_allclose(beta, 0.0, atol=1e-8)
        np.testing.assert_allclose(mu, float(x.mean()), atol=1e-6)

    def test_nig_unknown_method_raises(self):
        x = jnp.asarray(np.random.default_rng(0).normal(size=100))
        with pytest.raises(ValueError, match="not supported"):
            nig.fit(x, method="BOGUS")


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

    @pytest.mark.parametrize("dist,params", DIST_CONFIGS_FINITE_LOWER,
                             ids=FINITE_LOWER_IDS)
    def test_logpdf_below_finite_lower_is_neg_inf(self, dist, params):
        """logpdf must return -inf strictly below the finite lower support bound."""
        lower = float(np.array(dist._support(params)).flatten()[0])
        x = jnp.array([lower - 1e-6, lower - 1.0])
        vals = np.array(dist.logpdf(x=x, params=params)).flatten()
        assert np.all(np.isneginf(vals)), \
            f"{dist.name}: logpdf below lower support != -inf, got {vals}"

    @pytest.mark.parametrize("dist,params", DIST_CONFIGS_FINITE_UPPER,
                             ids=FINITE_UPPER_IDS)
    def test_logpdf_above_finite_upper_is_neg_inf(self, dist, params):
        """logpdf must return -inf strictly above the finite upper support bound."""
        upper = float(np.array(dist._support(params)).flatten()[1])
        x = jnp.array([upper + 1e-6, upper + 1.0])
        vals = np.array(dist.logpdf(x=x, params=params)).flatten()
        assert np.all(np.isneginf(vals)), \
            f"{dist.name}: logpdf above upper support != -inf, got {vals}"

    @pytest.mark.parametrize("dist,params", DIST_CONFIGS_WITH_AGN_BOTH,
                             ids=SATURATION_IDS)
    def test_cdf_far_left_tail_is_zero(self, dist, params):
        """CDF must saturate to 0 in the far left tail.

        Exact 0 below a finite lower bound (base class enforces via jnp.where);
        <= 1e-6 in the deep tail of an infinite-lower distribution.
        """
        lower = float(np.array(dist._support(params)).flatten()[0])
        if np.isfinite(lower):
            x = jnp.array([lower - 1e-6, lower - 1.0])
            tol = 0.0
        else:
            x = jnp.array([-1e6, -1e8])
            tol = 1e-6
        vals = np.array(dist.cdf(x=x, params=params)).flatten()
        assert np.all(vals <= tol), \
            f"{dist.name}: CDF far-left = {vals}, expected <= {tol}"

    @pytest.mark.parametrize("dist,params", DIST_CONFIGS_WITH_AGN_BOTH,
                             ids=SATURATION_IDS)
    def test_cdf_far_right_tail_is_one(self, dist, params):
        """CDF must saturate to 1 in the far right tail.

        Exact 1 above a finite upper bound (base class enforces via jnp.where);
        >= 1 - 1e-6 in the deep tail of an infinite-upper distribution.
        """
        upper = float(np.array(dist._support(params)).flatten()[1])
        if np.isfinite(upper):
            x = jnp.array([upper + 1e-6, upper + 1.0])
            tol = 0.0
        else:
            x = jnp.array([1e6, 1e8])
            tol = 1e-6
        vals = np.array(dist.cdf(x=x, params=params)).flatten()
        assert np.all(vals >= 1.0 - tol), \
            f"{dist.name}: CDF far-right = {vals}, expected >= {1.0 - tol}"

    @pytest.mark.parametrize("dist,params", [
        (gh, {"lamb": 1.0, "chi": 2.0, "psi": 3.0,
              "mu": 0.5, "sigma": 1.0, "gamma": 0.0}),
        (gig, {"lamb": 1.0, "chi": 2.0, "psi": 3.0}),
        (nig, {"mu": 0.0, "alpha": 2.5, "beta": 1.0, "delta": 1.0}),
        (skewed_t, {"nu": 6.0, "mu": 0.0, "sigma": 1.0, "gamma": 0.5}),
        # Heavy-tailed parameterisations that stress the quadrature.
        (skewed_t, {"nu": 3.0, "mu": 0.0, "sigma": 1.0, "gamma": 2.0}),
        (skewed_t, {"nu": 1.5, "mu": 0.0, "sigma": 1.0, "gamma": 0.0}),
    ], ids=["GH", "GIG", "NIG", "SkewedT",
            "SkewedT-heavy-skew", "SkewedT-var-undefined"])
    def test_cdf_piecewise_matches_adaptive_reference(self, dist, params):
        """Public CDF must match a tight-tolerance adaptive-quadgk reference.

        GL32 piecewise has no built-in error estimate, so this is the
        explicit empirical safety net required by the 'no unverified
        data paths' principle: call the public CDF on an in-bulk grid
        and compare against a vmap'd per-xi adaptive quadgk reference
        with tight tolerances. Any systematic GL32 inaccuracy on the
        target distributions would surface here.
        """
        from quadax import quadgk
        stats = dist.stats(params=params)
        mean_m = jnp.asarray(stats["mean"])
        var_m = jnp.asarray(stats["variance"])

        # Fall back to the distribution's scale parameter for grid sizing
        # when the variance moment diverges (skewed-t at nu <= 4 etc.).
        # Use jnp.where so the branch is traceable under JIT.
        scale_param = jnp.asarray(
            params.get("sigma", params.get("delta", 1.0))
        )
        mu_param = jnp.asarray(params.get("mu", 0.0))
        centre_j = jnp.where(jnp.isfinite(mean_m), mean_m, mu_param)
        std_j = jnp.where(
            jnp.isfinite(var_m), jnp.sqrt(var_m), 10.0 * scale_param
        )
        centre = float(np.array(centre_j))
        std = float(np.array(std_j))
        lower_f = float(np.array(dist._support(params)).flatten()[0])
        upper_f = float(np.array(dist._support(params)).flatten()[1])
        lo = max(centre - 10 * std, lower_f + 1e-3 * abs(std)) \
            if np.isfinite(lower_f) else centre - 10 * std
        hi = min(centre + 10 * std, upper_f - 1e-3 * abs(std)) \
            if np.isfinite(upper_f) else centre + 10 * std
        x = jnp.linspace(lo, hi, 20)

        cdf_batched = np.array(dist.cdf(x=x, params=params))

        # Reference: vmap adaptive quadgk per xi with tight tolerances.
        # One shared adaptive scan across all lanes under vmap — much
        # faster than a Python loop that would JIT-recompile per call.
        lower_s, _ = dist._support_bounds(params)
        params_array = dist._params_to_array(params)

        def _scalar_ref(xi):
            val, _ = quadgk(
                dist._pdf_for_cdf,
                interval=jnp.array([lower_s, xi]),
                args=params_array,
                epsabs=1e-13, epsrel=1e-11,
            )
            return val

        cdf_ref = np.array(jax.vmap(_scalar_ref)(x))

        max_abs = float(np.max(np.abs(cdf_batched - cdf_ref)))
        assert max_abs < 1e-5, (
            f"{dist.name}: piecewise vs adaptive reference disagree, "
            f"max |diff| = {max_abs:.3e}"
        )

    @pytest.mark.parametrize("dist,params", DIST_CONFIGS_FINITE_LOWER,
                             ids=FINITE_LOWER_IDS)
    def test_cdf_out_of_support_isolation(self, dist, params):
        """Mixed in- and out-of-support x must not produce NaNs.

        An out-of-support x could in principle drive the PDF to NaN at
        a quadrature node and poison the neighbouring CDFs through the
        prefix sum. This test confirms that in-support entries are
        unaffected by neighbouring out-of-support queries.
        """
        lower = float(np.array(dist._support(params)).flatten()[0])
        upper = float(np.array(dist._support(params)).flatten()[1])
        # Build a mix: two out-of-support below, three in-support, and
        # one out-of-support above if the upper bound is finite.
        in_support = list(np.linspace(
            lower + 1e-3 * (1 + abs(lower)),
            (upper - 1e-3 * (1 + abs(upper))) if np.isfinite(upper) else lower + 5.0,
            3,
        ))
        below = [lower - 2.0, lower - 1e-6]
        above = [upper + 1e-6, upper + 2.0] if np.isfinite(upper) else []
        x = jnp.array(below + in_support + above)
        cdf = np.array(dist.cdf(x=x, params=params)).flatten()
        assert not np.any(np.isnan(cdf)), \
            f"{dist.name}: NaN in CDF, got {cdf}"
        # Below-lower entries are exactly 0; above-upper entries are 1.
        for i in range(len(below)):
            assert cdf[i] == 0.0, \
                f"{dist.name}: cdf({x[i]}) expected 0, got {cdf[i]}"
        for j, i in enumerate(range(len(below) + len(in_support), len(x))):
            assert cdf[i] == 1.0, \
                f"{dist.name}: cdf({x[i]}) expected 1, got {cdf[i]}"
        # In-support entries are monotone and in [0, 1].
        in_idx = slice(len(below), len(below) + len(in_support))
        in_vals = cdf[in_idx]
        assert np.all((0.0 <= in_vals) & (in_vals <= 1.0)), \
            f"{dist.name}: in-support CDFs out of [0,1]: {in_vals}"
        assert np.all(np.diff(in_vals) >= -1e-6), \
            f"{dist.name}: in-support CDFs non-monotone: {in_vals}"

    @pytest.mark.parametrize("dist,params", DIST_CONFIGS_WITH_AGN_BOTH,
                             ids=SATURATION_IDS)
    def test_cdf_handles_inf_input(self, dist, params):
        """cdf([+inf]) == 1 and cdf([-inf]) == 0 for every distribution.

        Saturating infinities must be handled cleanly: F(+inf) = 1 and
        F(-inf) = 0 regardless of whether the support itself is
        infinite. No NaNs should leak into the result.
        """
        lower = float(np.array(dist._support(params)).flatten()[0])
        upper = float(np.array(dist._support(params)).flatten()[1])
        # Always include both infinities; if support is finite on one
        # side, it's still a valid query and should return the bound.
        x = jnp.array([-jnp.inf, jnp.inf])
        cdf = np.array(dist.cdf(x=x, params=params)).flatten()
        assert not np.any(np.isnan(cdf)), \
            f"{dist.name}: NaN in CDF at +/-inf, got {cdf}"
        assert cdf[0] == 0.0, \
            f"{dist.name}: cdf(-inf) expected 0.0, got {cdf[0]}"
        assert cdf[1] == 1.0, \
            f"{dist.name}: cdf(+inf) expected 1.0, got {cdf[1]}"

    @pytest.mark.parametrize("dist,params", [
        # Heavy-tailed parameterisations stress the far-tail quadrature
        # beyond what DIST_CONFIGS exercises. skewed-T at low nu has
        # polynomial tails where standard fixed-grid Gauss-Legendre can
        # under-integrate if the integration strategy is naive.
        (gh, {"lamb": -0.5, "chi": 0.1, "psi": 0.1,
              "mu": 0.0, "sigma": 1.0, "gamma": 0.0}),
        (skewed_t, {"nu": 3.0, "mu": 0.0, "sigma": 1.0, "gamma": 2.0}),
        (skewed_t, {"nu": 1.5, "mu": 0.0, "sigma": 1.0, "gamma": 0.0}),
    ], ids=["GH-heavy-tail", "SkewedT-nu3-gamma2", "SkewedT-nu1.5"])
    def test_cdf_heavy_tail_extreme_x_monotonic(self, dist, params):
        """Heavy-tailed distributions must saturate to 1 and stay monotone.

        For skewed-T at ``nu <= 3`` the polynomial tail decays slowly,
        so even at ``x = 1e6`` there is non-negligible PDF mass beyond
        any fixed breakpoint grid. The CDF must still saturate to
        ``>= 1 - 1e-5`` at ``x = 1e8`` and be monotone across the
        stress grid.
        """
        x = jnp.array([5.0, 100.0, 1000.0, 1e6, 1e8])
        cdf = np.array(dist.cdf(x=x, params=params)).flatten()
        assert not np.any(np.isnan(cdf)), \
            f"{dist.name}: NaN in CDF at heavy-tail extreme x, got {cdf}"
        diffs = np.diff(cdf)
        worst = float(np.min(diffs))
        assert worst >= -1e-6, (
            f"{dist.name}: CDF non-monotone, worst diff = {worst:.3e}, "
            f"values = {cdf}"
        )
        assert cdf[-1] >= 1.0 - 1e-5, (
            f"{dist.name}: CDF(1e8) = {cdf[-1]} did not saturate "
            f"(tolerance 1e-5)"
        )

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

    @pytest.mark.parametrize("dist,params", DIST_CONFIGS, ids=DIST_IDS)
    def test_fit_is_jittable(self, dist, params):
        """Every dist.fit() must be JIT-compatible.

        CopulAX is a JAX-first library. A fit method that silently falls
        out of JIT still produces correct results but runs 10–100× slower —
        a regression users only notice via wall-clock time. This test makes
        any such regression a failing test instead.
        """
        key = jax.random.PRNGKey(0)
        x = dist.rvs(size=(200,), params=params, key=key)
        method = FIT_JIT_METHODS.get(dist.name)
        if method is not None:
            fit_jit = jax.jit(dist.fit, static_argnames=("method",))
            fitted = fit_jit(x, method=method)
        else:
            fit_jit = jax.jit(dist.fit)
            fitted = fit_jit(x)
        assert isinstance(fitted, type(dist)), (
            f"{dist.name}.fit() under JIT did not return "
            f"a {type(dist).__name__} instance (got {type(fitted).__name__})"
        )


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


# ---------------------------------------------------------------------------
# Skewed-T third-party cross-validation via scipy.stats.genhyperbolic limit
# ---------------------------------------------------------------------------
#
# The McNeil (2005) skewed-t is not directly implemented by any Python
# package. It is however the ψ → 0 limit of the generalised hyperbolic
# distribution with λ = -ν/2, χ = ν. We exploit this to cross-validate
# against scipy.stats.genhyperbolic at ψ = 1e-12, where the GH formula
# numerically coincides with the skewed-t to O(ψ) error. Empirically
# max|logpdf_scipy - logpdf_copulax| ≈ 1e-12 at ψ = 1e-12, far below the
# rtol=1e-9 threshold used here.
#
# Reference mapping (from the GH → scipy mapping in conftest.py with
# χ = ν, ψ = ε, λ = -ν/2):
#     scipy.stats.genhyperbolic(
#         p     = -ν/2,
#         a     = sqrt(ν·ε + ν·γ² / σ²),
#         b     = γ·sqrt(ν) / σ,
#         loc   = μ,
#         scale = σ·sqrt(ν),
#     )

PSI_EPS = 1e-12  # limiting value for the ψ → 0 approximation


def _skewed_t_to_scipy_genhyperbolic(params, psi_eps=PSI_EPS):
    """Map CopulAX skewed-t params to scipy.stats.genhyperbolic at ψ=ε."""
    nu = float(params["nu"])
    mu = float(params["mu"])
    sigma = float(params["sigma"])
    gamma = float(params["gamma"])
    chi = nu
    p = -nu / 2.0
    delta = sigma * np.sqrt(chi)
    a = np.sqrt(chi * psi_eps + chi * gamma ** 2 / sigma ** 2)
    b = gamma * np.sqrt(chi) / sigma
    return scipy.stats.genhyperbolic(p=p, a=a, b=b, loc=mu, scale=delta)


SKEWED_T_CONFIGS = [
    ({"nu": 4.5, "mu": 0.0, "sigma": 1.0, "gamma": 1.0}, "example_params"),
    ({"nu": 10.0, "mu": -2.0, "sigma": 0.5, "gamma": -2.0}, "neg_skew_heavy"),
    ({"nu": 3.0, "mu": 1.0, "sigma": 2.0, "gamma": 3.0}, "low_nu_large_skew"),
]


class TestSkewedTAgainstScipyLimit:
    """Cross-validate skewed-T against scipy.stats.genhyperbolic at ψ → 0.

    Replaces the v1.0.1 golden-fixture regression test by comparing
    against an independent third-party implementation rather than a
    self-reference. See module header comment for the mapping.
    """

    @pytest.mark.parametrize(
        "params,tag",
        SKEWED_T_CONFIGS,
        ids=[t for _, t in SKEWED_T_CONFIGS],
    )
    def test_logpdf_matches_scipy_limit(self, params, tag):
        sp = _skewed_t_to_scipy_genhyperbolic(params)
        # Test points spanning ±5σ around μ.
        mu, sigma = float(params["mu"]), float(params["sigma"])
        x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 50)

        cx = np.asarray(skewed_t.logpdf(jnp.array(x), params))
        sp_vals = sp.logpdf(x)

        mask = np.isfinite(cx) & np.isfinite(sp_vals)
        assert mask.sum() >= 40, f"Too few finite points: {mask.sum()}/50"
        np.testing.assert_allclose(
            cx[mask], sp_vals[mask], rtol=1e-9, atol=1e-12,
            err_msg=f"skewed_t logpdf mismatch vs scipy GH-limit at {tag}",
        )

    @pytest.mark.parametrize(
        "params,tag",
        SKEWED_T_CONFIGS,
        ids=[t for _, t in SKEWED_T_CONFIGS],
    )
    def test_pdf_matches_scipy_limit(self, params, tag):
        sp = _skewed_t_to_scipy_genhyperbolic(params)
        mu, sigma = float(params["mu"]), float(params["sigma"])
        x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 50)

        cx = np.asarray(skewed_t.pdf(jnp.array(x), params))
        sp_vals = sp.pdf(x)

        mask = np.isfinite(cx) & np.isfinite(sp_vals) & (sp_vals > 0)
        assert mask.sum() >= 40, f"Too few finite points: {mask.sum()}/50"
        np.testing.assert_allclose(
            cx[mask], sp_vals[mask], rtol=1e-9, atol=1e-12,
            err_msg=f"skewed_t pdf mismatch vs scipy GH-limit at {tag}",
        )


# ===================================================================
# Skewed-T at γ = 0 should equal Student-T exactly
# ===================================================================

class TestSkewedTGammaZeroMatchesStudentT:
    r"""Covers the skewed-t log-PDF's removable singularity at ``γ = 0``.

    Mathematically, ``skewed_t.logpdf(x; ν, μ, σ, γ=0) ==
    student_t.logpdf(x; ν, μ, σ)`` exactly — the skewed-t is a proper
    generalisation.  The old dual-branch dispatcher enforced this via a
    runtime ``jnp.where(γ == 0, student_t_branch, skewed_branch)``; the
    new implementation handles the limit *inside* the single skewed
    formula via :py:func:`log_kv_plus_s_log_r`.  These tests pin the
    equivalence at γ = 0 exactly and verify the limit is approached
    smoothly as γ → 0 from either side.
    """

    @pytest.mark.parametrize("nu", [3.0, 5.0, 10.0, 30.0])
    def test_logpdf_at_gamma_zero_matches_student_t(self, nu):
        r"""``skewed_t.logpdf(x; γ=0)`` must equal ``student_t.logpdf(x)``
        to float64 eps on a wide grid.

        Tolerance (``rtol=1e-10, atol=1e-12``) reflects the fact that
        the new implementation uses an analytical limit, not an
        approximation — the only error is float64 rounding of the
        underlying arithmetic, which differs marginally between the
        skewed-t and student-t evaluation paths (at high ν the two
        formulas accumulate rounding differently through distinct
        ``lgamma`` calls).  Empirically the disagreement is ≤ 3e-12
        at ν = 30 on a 50-point grid; the old dual-branch dispatcher
        literally returned the student-t formula at γ = 0 and
        therefore had bit-exact agreement, but at the cost of the
        dual-branch performance overhead this refactor removes.
        """
        x = jnp.linspace(-6.0, 6.0, 50)
        sk = np.asarray(skewed_t.logpdf(
            x, skewed_t._params_dict(nu=nu, mu=0.0, sigma=1.0, gamma=0.0),
        ))
        ref = np.asarray(student_t.logpdf(
            x, student_t._params_dict(nu=nu, mu=0.0, sigma=1.0),
        ))
        np.testing.assert_allclose(
            sk, ref, rtol=1e-10, atol=1e-12,
            err_msg=(
                f"skewed_t(γ=0) != student_t at ν={nu}: "
                f"max |diff| = {np.max(np.abs(sk - ref)):.3e}"
            ),
        )

    @pytest.mark.parametrize("gamma", [1e-15, 1e-10, 1e-6, 1e-3])
    def test_logpdf_continuous_approach_to_gamma_zero(self, gamma):
        r"""``skewed_t.logpdf`` is continuous in ``γ`` across ``γ = 0``.

        Small-γ calls previously produced corrupted values (error of
        order ``1e+2`` at ``γ = 1e-30`` under the old ``stability=1e-30``
        additive).  With the refactor, small γ converges smoothly to
        the student-t limit at rate ``O(γ^{2})`` — documented here so
        that any regression re-introducing a nonzero ``stability``
        additive on the ``(ν+Q)·R`` log or a γ floor above ``1e-6``
        would fail this test.
        """
        nu = 5.0
        x = jnp.linspace(-4.0, 4.0, 20)
        ref = np.asarray(student_t.logpdf(
            x, student_t._params_dict(nu=nu, mu=0.0, sigma=1.0),
        ))
        sk = np.asarray(skewed_t.logpdf(
            x, skewed_t._params_dict(nu=nu, mu=0.0, sigma=1.0, gamma=gamma),
        ))
        # Expected deviation is bounded above by O(γ): at γ = 1e-3, the
        # logpdf deviates from the student-t limit by < 1e-2.  Keep the
        # tolerance generous enough to catch regressions (which produce
        # +∞ or >1e+1 errors) while not over-constraining the physical
        # γ-dependence.
        max_err = np.max(np.abs(sk - ref))
        assert np.all(np.isfinite(sk)), (
            f"skewed_t returned non-finite at γ={gamma}"
        )
        # Loose bound: linear-in-γ tolerance up to 1e-3 (worst case).
        tol = max(10.0 * gamma, 1e-12)
        assert max_err <= tol, (
            f"skewed_t(γ={gamma}) deviates from student_t by {max_err:.3e}, "
            f"expected ≤ {tol:.3e}"
        )

    @pytest.mark.parametrize("nu", [3.0, 5.0, 10.0])
    def test_gradients_finite_at_gamma_zero(self, nu):
        r"""``jax.grad`` through ``skewed_t.logpdf`` at ``γ = 0`` returns
        finite values for both ``∂/∂x`` and ``∂/∂γ``.

        The old code computed ``log_kv(s, 0) + (s/2) log((ν+Q)·R + stability)``
        as two separate logs that each diverge as ``γ → 0``; their
        subtraction propagated ``NaN`` through autograd even though the
        forward ``jnp.where`` masked the forward value.  The refactor
        computes the combination via ``log_kv_plus_s_log_r``, which
        floors ``r`` internally and keeps ``∂/∂γ`` finite.
        """
        x = jnp.asarray(0.5, dtype=float)

        def lp_at_gamma(g):
            return skewed_t.logpdf(
                x, skewed_t._params_dict(nu=nu, mu=0.0, sigma=1.0, gamma=g),
            ).sum()

        def lp_at_x(xi):
            return skewed_t.logpdf(
                xi, skewed_t._params_dict(nu=nu, mu=0.0, sigma=1.0, gamma=0.0),
            ).sum()

        d_dg = float(jax.grad(lp_at_gamma)(jnp.asarray(0.0)))
        d_dx = float(jax.grad(lp_at_x)(x))
        assert np.isfinite(d_dg), f"∂/∂γ at γ=0, ν={nu}: {d_dg}"
        assert np.isfinite(d_dx), f"∂/∂x at γ=0, ν={nu}: {d_dx}"


# ---------------------------------------------------------------------------
# Method-string casing regression (v2 lowercase convention)
# ---------------------------------------------------------------------------

class TestUnivariateMethodStringCasingIsLowercase:
    r"""Lock in the v2 lowercase method-string contract across the
    univariate layer.  Prior versions accepted ``'EM'``/``'MLE'``/etc.;
    callers that haven't migrated should fail loudly rather than silently
    running the wrong code path (or worse, passing the old string to a
    renamed-to-lowercase dispatcher and having it fall through).
    """

    @pytest.fixture
    def x(self):
        np.random.seed(11)
        return jnp.array(np.random.normal(size=200))

    @pytest.mark.parametrize(
        "dist, bad_method",
        [
            (gh, "EM"),
            (gh, "MLE"),
            (gh, "LDMLE"),
            (skewed_t, "EM"),
            (skewed_t, "MLE"),
            (skewed_t, "LDMLE"),
            (student_t, "MLE"),
            (student_t, "LDMLE"),
            (gen_normal, "MLE"),
            (gen_normal, "MOM"),
            (asym_gen_normal, "MLE"),
            (asym_gen_normal, "MOM"),
            (nig, "EM"),
            (nig, "MLE"),
            (nig, "MoM"),
        ],
    )
    def test_uppercase_method_string_rejected(self, dist, bad_method, x):
        r"""Uppercase / mixed-case method strings must raise ``ValueError``
        from the base-class ``_check_method`` gate.
        """
        with pytest.raises(ValueError):
            dist.fit(x, method=bad_method)


# ---------------------------------------------------------------------------
# _supported_methods introspection contract (all 13 univariate dists)
# ---------------------------------------------------------------------------


class TestUnivariateSupportedMethodsDeclared:
    r"""Every univariate distribution declares a non-empty
    ``_supported_methods`` frozenset, every method in that set is
    callable via ``dist.fit(x, method=m)`` (where applicable), and
    unknown methods are rejected by the base-class ``_check_method``
    gate on :class:`Distribution`.
    """

    ALL_DISTS = [
        normal, uniform, gamma, lognormal, ig, wald, gig,
        student_t, gh, skewed_t, nig, gen_normal, asym_gen_normal,
    ]
    POSITIVE_DISTS = {gamma, lognormal, ig, wald, gig}

    def _x(self, dist):
        np.random.seed(11)
        x = np.random.standard_normal(200)
        if dist in self.POSITIVE_DISTS:
            x = np.abs(x) + 0.1
        return jnp.asarray(x)

    @pytest.mark.parametrize("dist", ALL_DISTS, ids=[d.name for d in ALL_DISTS])
    def test_supported_methods_nonempty(self, dist):
        assert len(dist._supported_methods) >= 1, (
            f"{dist.name} declares empty _supported_methods"
        )

    @pytest.mark.parametrize("dist", ALL_DISTS, ids=[d.name for d in ALL_DISTS])
    def test_check_method_rejects_unknown(self, dist):
        with pytest.raises(ValueError, match="not supported"):
            dist._check_method("definitely_not_a_method")

    # Only dispatching dists have a ``method`` kwarg on ``fit``; the
    # single-method ones (Normal, Uniform, LogNormal, Wald, Gamma, IG,
    # GIG) fit one-shot without a method kwarg, matching the multivariate
    # convention (mvt_normal et al).  Their ``_supported_methods`` set
    # is documentation-only.
    DISPATCHING_DISTS = [student_t, gh, skewed_t, nig, gen_normal, asym_gen_normal]

    @pytest.mark.parametrize(
        "dist", DISPATCHING_DISTS, ids=[d.name for d in DISPATCHING_DISTS],
    )
    def test_every_supported_method_is_callable(self, dist):
        import inspect
        x = self._x(dist)
        # Only pass ``maxiter`` when the dist's fit signature declares it
        # (GenNormal uses Brent internally and has no maxiter kwarg).
        accepts_maxiter = "maxiter" in inspect.signature(dist.fit).parameters
        for m in sorted(dist._supported_methods):
            kw = {"maxiter": 5} if accepts_maxiter else {}
            fitted = dist.fit(x, method=m, **kw)
            assert isinstance(fitted, type(dist)), (
                f"{dist.name}.fit(method={m!r}) returned "
                f"{type(fitted).__name__} instead of {type(dist).__name__}"
            )
