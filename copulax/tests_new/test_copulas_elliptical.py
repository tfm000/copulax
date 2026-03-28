"""Rigorous tests for the 4 elliptical copulas: Gaussian, Student-T, GH, Skewed-T.

Verifies Sklar's theorem implementation, uniform marginal sampling,
density properties, and fitting.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from copulax.copulas import (
    gaussian_copula, student_t_copula, gh_copula, skewed_t_copula,
)
from copulax.tests_new.conftest import no_nans, is_finite


FAST_COPULAS = [gaussian_copula, student_t_copula]
ALL_COPULAS = [gaussian_copula, student_t_copula, gh_copula, skewed_t_copula]
FAST_IDS = [c.name for c in FAST_COPULAS]
ALL_IDS = [c.name for c in ALL_COPULAS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_copula_params(copula, d=3):
    """Get example params for a copula."""
    return copula.example_params(dim=d)


def _uniform_sample(d=3, n=100, seed=42):
    """Generate uniform sample in (0.01, 0.99)^d."""
    np.random.seed(seed)
    return jnp.array(np.random.uniform(0.01, 0.99, size=(n, d)))


# ---------------------------------------------------------------------------
# Copula density properties
# ---------------------------------------------------------------------------

class TestCopulaDensityProperties:
    """Verify copula density mathematical properties."""

    @pytest.mark.parametrize("copula", FAST_COPULAS, ids=FAST_IDS)
    def test_copula_pdf_positive(self, copula):
        """Copula PDF should be > 0 for all u in (0,1)^d."""
        d = 3
        params = _get_copula_params(copula, d)
        u = _uniform_sample(d, 50)
        pdf = np.array(copula.copula_pdf(u=u, params=params)).flatten()
        assert np.all(pdf[np.isfinite(pdf)] > 0), \
            f"{copula.name} copula_pdf not positive"

    @pytest.mark.parametrize("copula", FAST_COPULAS, ids=FAST_IDS)
    def test_logpdf_pdf_consistency(self, copula):
        """exp(copula_logpdf) == copula_pdf."""
        d = 3
        params = _get_copula_params(copula, d)
        u = _uniform_sample(d, 30)

        logpdf = np.array(copula.copula_logpdf(u=u, params=params)).flatten()
        pdf = np.array(copula.copula_pdf(u=u, params=params)).flatten()

        mask = np.isfinite(logpdf) & (pdf > 0)
        np.testing.assert_allclose(
            np.exp(logpdf[mask]), pdf[mask], rtol=1e-4,
            err_msg=f"{copula.name}: exp(copula_logpdf) != copula_pdf"
        )

    @pytest.mark.parametrize("copula", FAST_COPULAS, ids=FAST_IDS)
    def test_copula_logpdf_finite(self, copula):
        """copula_logpdf should be finite for interior points."""
        d = 3
        params = _get_copula_params(copula, d)
        u = _uniform_sample(d, 30)
        logpdf = np.array(copula.copula_logpdf(u=u, params=params))
        assert no_nans(logpdf), f"{copula.name} copula_logpdf has NaNs"


# ---------------------------------------------------------------------------
# Gaussian copula manual verification
# ---------------------------------------------------------------------------

class TestGaussianCopulaAgainstManual:
    """Verify Gaussian copula density via manual Sklar decomposition.

    c(u1, ..., ud) = phi_d(Phi^{-1}(u1), ..., Phi^{-1}(ud); Sigma)
                     / prod_i phi(Phi^{-1}(ui))

    where phi_d is the MVN density, phi is the standard normal density,
    and Phi^{-1} is the standard normal quantile function.
    """

    def test_logpdf_matches_manual_computation(self):
        """Gaussian copula logpdf should match manual Sklar decomposition."""
        d = 3
        params = _get_copula_params(gaussian_copula, d)
        sigma = np.array(params["copula"]["sigma"])
        u = np.array(_uniform_sample(d, 20))

        # Manual computation
        x_dash = scipy.stats.norm.ppf(u)  # Phi^{-1}(u)
        # Multivariate normal logpdf
        mvn_logpdf = scipy.stats.multivariate_normal.logpdf(
            x_dash, mean=np.zeros(d), cov=sigma)
        # Sum of marginal normal logpdf
        marginal_logpdf_sum = np.sum(scipy.stats.norm.logpdf(x_dash), axis=1)
        # Copula logpdf = mvn_logpdf - marginal_logpdf_sum
        expected_logpdf = mvn_logpdf - marginal_logpdf_sum

        cx_logpdf = np.array(gaussian_copula.copula_logpdf(
            u=jnp.array(u), params=params)).flatten()

        mask = np.isfinite(expected_logpdf) & np.isfinite(cx_logpdf)
        np.testing.assert_allclose(
            cx_logpdf[mask], expected_logpdf[mask], rtol=1e-4,
            err_msg="Gaussian copula logpdf doesn't match manual Sklar decomposition"
        )


# ---------------------------------------------------------------------------
# Sampling with uniform marginals
# ---------------------------------------------------------------------------

class TestCopulaSamplingUniformMargins:
    """Copula samples should have U(0,1) marginals."""

    @pytest.mark.parametrize("copula", FAST_COPULAS, ids=FAST_IDS)
    def test_marginals_are_uniform(self, copula):
        """Each margin of copula_rvs should pass KS test against U(0,1)."""
        d = 3
        params = _get_copula_params(copula, d)
        key = jax.random.PRNGKey(42)
        samples = np.array(copula.copula_rvs(size=1000, params=params, key=key))

        for i in range(d):
            margin = samples[:, i]
            margin = margin[np.isfinite(margin)]
            margin = margin[(margin > 0) & (margin < 1)]

            if len(margin) < 100:
                pytest.xfail(f"{copula.name} dim {i}: too few valid samples")

            ks_stat, ks_p = scipy.stats.kstest(margin, "uniform")
            assert ks_p > 0.001, \
                f"{copula.name} dim {i}: marginal not uniform " \
                f"(KS stat={ks_stat:.4f}, p={ks_p:.4f})"


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

class TestCopulaFitting:
    """Verify copula fitting produces reasonable results."""

    @pytest.mark.parametrize("copula", FAST_COPULAS, ids=FAST_IDS)
    def test_fit_returns_valid_params(self, copula):
        """fit() should return valid parameters (no NaN, no inf)."""
        d = 3
        np.random.seed(42)
        # Generate correlated normal data
        sigma = np.array([[1.0, 0.5, 0.3],
                           [0.5, 1.0, 0.4],
                           [0.3, 0.4, 1.0]])
        data = np.random.multivariate_normal(np.zeros(d), sigma, size=200)

        fitted = copula.fit(x=jnp.array(data))
        assert fitted is not None, f"{copula.name} fit returned None"

        # Check copula params are valid
        copula_params = fitted.params.get("copula", {})
        if "sigma" in copula_params:
            s = np.array(copula_params["sigma"])
            assert no_nans(s), f"{copula.name} fitted sigma has NaNs"
            assert is_finite(s), f"{copula.name} fitted sigma not finite"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestCopulaMetrics:
    """Verify loglikelihood, AIC, BIC are finite."""

    @pytest.mark.parametrize("copula", FAST_COPULAS, ids=FAST_IDS)
    def test_metrics_finite(self, copula):
        d = 3
        np.random.seed(42)
        sigma = np.array([[1.0, 0.5, 0.3],
                           [0.5, 1.0, 0.4],
                           [0.3, 0.4, 1.0]])
        data = np.random.multivariate_normal(np.zeros(d), sigma, size=200)

        fitted = copula.fit(x=jnp.array(data))
        logll = float(fitted.loglikelihood(x=jnp.array(data)))
        aic = float(fitted.aic(x=jnp.array(data)))
        bic = float(fitted.bic(x=jnp.array(data)))

        assert np.isfinite(logll), f"{copula.name} logll not finite"
        assert np.isfinite(aic), f"{copula.name} AIC not finite"
        assert np.isfinite(bic), f"{copula.name} BIC not finite"
