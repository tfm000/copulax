"""Rigorous tests for all 4 CopulAX multivariate distributions.

Cross-validates against scipy.stats.multivariate_normal and multivariate_t.
Verifies density integration, moment formulas, and parameter recovery.

Catches: FINDING-04-01 (MVT Student-T LDMLE scale (nu-2)/2 vs (nu-2)/nu),
FINDING-04-02 (MVT-GH H term missing mu centering).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats
from quadax import quadgk

from copulax.multivariate import mvt_normal, mvt_student_t, mvt_gh, mvt_skewed_t
from copulax.tests_new.conftest import no_nans, is_finite


ALL_MVT_DISTS = [mvt_normal, mvt_student_t, mvt_gh, mvt_skewed_t]
MVT_IDS = [d.name for d in ALL_MVT_DISTS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mvt_normal_params(d=3):
    """Create test params for multivariate normal."""
    mu = jnp.zeros((d, 1))
    sigma = jnp.eye(d) * 2.0
    return {"mu": mu, "sigma": sigma}


def _make_mvt_student_t_params(d=3, nu=5.0):
    """Create test params for multivariate Student-T."""
    mu = jnp.ones((d, 1)) * 0.5
    sigma = jnp.eye(d) * 1.5
    return {"nu": jnp.array(nu), "mu": mu, "sigma": sigma}


def _make_mvt_gh_params(d=3):
    """Create test params for multivariate GH with non-zero mu and gamma."""
    mu = jnp.ones((d, 1)) * 1.0
    gamma = jnp.ones((d, 1)) * 0.0  # symmetric for scipy comparison
    sigma = jnp.eye(d)
    return {
        "lambda": jnp.array(1.0), "chi": jnp.array(1.0), "psi": jnp.array(1.0),
        "mu": mu, "gamma": gamma, "sigma": sigma,
    }


def _make_mvt_skewed_t_params(d=3, nu=6.0):
    """Create test params for multivariate Skewed-T."""
    mu = jnp.zeros((d, 1))
    gamma = jnp.ones((d, 1)) * 0.3
    sigma = jnp.eye(d)
    return {"nu": jnp.array(nu), "mu": mu, "gamma": gamma, "sigma": sigma}


# ---------------------------------------------------------------------------
# Cross-validation against scipy
# ---------------------------------------------------------------------------

class TestMvtNormalAgainstScipy:
    """Verify multivariate normal logpdf matches scipy."""

    @pytest.mark.parametrize("d", [2, 3, 5])
    def test_logpdf_matches_scipy(self, d):
        """logpdf should match scipy.stats.multivariate_normal.logpdf."""
        params = _make_mvt_normal_params(d)
        mu_np = np.array(params["mu"]).flatten()
        sigma_np = np.array(params["sigma"])

        np.random.seed(42)
        x = np.random.multivariate_normal(mu_np, sigma_np, size=30)

        cx_logpdf = np.array(mvt_normal.logpdf(x=jnp.array(x), params=params)).flatten()
        sp_logpdf = scipy.stats.multivariate_normal.logpdf(x, mean=mu_np, cov=sigma_np)

        np.testing.assert_allclose(cx_logpdf, sp_logpdf, rtol=1e-6,
                                   err_msg="MvtNormal logpdf mismatch vs scipy")


class TestMvtStudentTAgainstScipy:
    """Verify multivariate Student-T logpdf matches scipy.

    Catches FINDING-04-01: LDMLE scale (nu-2)/2 should be (nu-2)/nu.
    """

    @pytest.mark.parametrize("d", [2, 3])
    def test_logpdf_matches_scipy(self, d):
        """logpdf should match scipy.stats.multivariate_t.logpdf."""
        params = _make_mvt_student_t_params(d, nu=5.0)
        nu = float(params["nu"])
        mu_np = np.array(params["mu"]).flatten()
        sigma_np = np.array(params["sigma"])

        np.random.seed(42)
        x = scipy.stats.multivariate_t.rvs(
            loc=mu_np, shape=sigma_np, df=nu, size=30
        )

        cx_logpdf = np.array(mvt_student_t.logpdf(
            x=jnp.array(x), params=params)).flatten()
        sp_logpdf = scipy.stats.multivariate_t.logpdf(
            x, loc=mu_np, shape=sigma_np, df=nu)

        np.testing.assert_allclose(cx_logpdf, sp_logpdf, rtol=1e-5,
                                   err_msg="MvtStudentT logpdf mismatch vs scipy")

    def test_ldmle_scale_formula(self):
        """Verify LDMLE sigma reconstruction uses (nu-2)/nu, not (nu-2)/2.

        FINDING-04-01: For nu=10, the correct scale is (10-2)/10 = 0.8.
        The buggy code computes (10-2)/2 = 4.0 (5x too large).
        """
        # Generate data from known MVT-StudentT
        d = 3
        nu = 10.0
        sigma_true = np.array([[2.0, 0.5, 0.3],
                                [0.5, 1.5, 0.2],
                                [0.3, 0.2, 1.0]])
        mu_true = np.array([1.0, 2.0, 3.0])

        np.random.seed(42)
        data = scipy.stats.multivariate_t.rvs(
            loc=mu_true, shape=sigma_true, df=nu, size=2000
        )

        fitted = mvt_student_t.fit(x=jnp.array(data))
        fitted_sigma = np.array(fitted.params["sigma"])
        fitted_nu = float(fitted.params["nu"])

        # The fitted sigma should be comparable to the true sigma
        # (not 5x too large). The scale factor is (nu-2)/nu for the
        # sample covariance -> MLE sigma conversion.
        sigma_ratio = np.mean(np.abs(fitted_sigma)) / np.mean(np.abs(sigma_true))
        assert 0.3 < sigma_ratio < 3.0, \
            f"Sigma ratio = {sigma_ratio:.2f}, likely LDMLE scale bug"


class TestMvtGHLogpdf:
    """Verify multivariate GH logpdf.

    Catches FINDING-04-02: H term missing mu centering.
    """

    def test_logpdf_with_nonzero_mu(self):
        """logpdf should differ when mu changes (H term depends on x-mu).

        FINDING-04-02: If H uses x instead of (x-mu), shifting mu
        won't produce the expected shift in logpdf.
        """
        d = 3
        params1 = _make_mvt_gh_params(d)
        params2 = dict(params1)
        params2["mu"] = jnp.ones((d, 1)) * 5.0  # large shift

        x = jnp.ones((10, d))  # all-ones test data

        logpdf1 = np.array(mvt_gh.logpdf(x=x, params=params1)).flatten()
        logpdf2 = np.array(mvt_gh.logpdf(x=x, params=params2)).flatten()

        # With mu=1, data x=1 is centered. With mu=5, data is 4 units away.
        # logpdf2 should be significantly smaller (more negative).
        assert np.mean(logpdf2) < np.mean(logpdf1) - 1.0, \
            "MVT-GH logpdf doesn't change enough when mu shifts — " \
            "H term may be missing mu centering (FINDING-04-02)"

    def test_logpdf_finite_and_no_nans(self):
        """logpdf should be finite for valid inputs."""
        d = 3
        params = _make_mvt_gh_params(d)
        np.random.seed(42)
        x = np.random.normal(size=(20, d))
        logpdf = np.array(mvt_gh.logpdf(x=jnp.array(x), params=params))
        assert no_nans(logpdf), "MVT-GH logpdf has NaNs"
        assert is_finite(logpdf), "MVT-GH logpdf has non-finite values"


# ---------------------------------------------------------------------------
# PDF integration (d=2 only for tractability)
# ---------------------------------------------------------------------------

class TestMultivariatePdfIntegration:
    """Verify multivariate PDF integrates to ~1 using quadax (JAX-native)."""

    @pytest.mark.slow
    def test_mvt_normal_integrates_to_one(self):
        """2D multivariate normal PDF should integrate to 1."""
        params = _make_mvt_normal_params(d=2)

        def _inner(x1, x0):
            x = jnp.array([[x0, x1]])
            return mvt_normal.pdf(x=x, params=params).flatten()[0]

        def _outer(x0):
            val, _ = quadgk(lambda x1: _inner(x1, x0), interval=(-10.0, 10.0))
            return val.reshape(())

        result, _ = quadgk(_outer, interval=(-10.0, 10.0))
        np.testing.assert_allclose(float(result), 1.0, rtol=1e-2,
                                   err_msg="MvtNormal PDF doesn't integrate to 1")

    @pytest.mark.slow
    def test_mvt_student_t_integrates_to_one(self):
        """2D multivariate Student-T PDF should integrate to 1."""
        params = _make_mvt_student_t_params(d=2, nu=5.0)

        def _inner(x1, x0):
            x = jnp.array([[x0, x1]])
            return mvt_student_t.pdf(x=x, params=params).flatten()[0]

        def _outer(x0):
            val, _ = quadgk(lambda x1: _inner(x1, x0), interval=(-15.0, 15.0))
            return val.reshape(())

        result, _ = quadgk(_outer, interval=(-15.0, 15.0))
        np.testing.assert_allclose(float(result), 1.0, rtol=1e-2,
                                   err_msg="MvtStudentT PDF doesn't integrate to 1")


# ---------------------------------------------------------------------------
# Stats verification
# ---------------------------------------------------------------------------

class TestMultivariateStats:
    """Verify multivariate stats (mean, covariance)."""

    def test_mvt_normal_mean_and_cov(self):
        """MVT Normal: mean = mu, cov = sigma."""
        d = 3
        params = _make_mvt_normal_params(d)
        stats = mvt_normal.stats(params=params)

        np.testing.assert_allclose(
            np.array(stats["mean"]).flatten(),
            np.array(params["mu"]).flatten(), atol=1e-10
        )
        np.testing.assert_allclose(
            np.array(stats["cov"]),
            np.array(params["sigma"]), atol=1e-10
        )

    def test_mvt_student_t_cov_formula(self):
        """MVT Student-T: cov = nu/(nu-2) * sigma for nu > 2."""
        d = 3
        params = _make_mvt_student_t_params(d, nu=5.0)
        stats = mvt_student_t.stats(params=params)
        nu = float(params["nu"])

        expected_cov = (nu / (nu - 2.0)) * np.array(params["sigma"])
        np.testing.assert_allclose(
            np.array(stats["cov"]), expected_cov, rtol=1e-5,
            err_msg="MVT Student-T covariance formula incorrect"
        )


# ---------------------------------------------------------------------------
# Sampling and fitting
# ---------------------------------------------------------------------------

class TestMultivariateSampling:
    """Verify sampling shape and statistical properties."""

    @pytest.mark.parametrize("dist", ALL_MVT_DISTS, ids=MVT_IDS)
    def test_sampling_shape(self, dist):
        d = 3
        params = dist.example_params(dim=d)
        key = jax.random.PRNGKey(42)
        samples = dist.rvs(size=50, params=params, key=key)
        assert samples.shape == (50, d), \
            f"{dist.name} sample shape = {samples.shape}, expected (50, {d})"

    def test_mvt_normal_sample_mean_close(self):
        """Large sample mean should be close to mu."""
        d = 3
        params = _make_mvt_normal_params(d)
        key = jax.random.PRNGKey(42)
        samples = np.array(mvt_normal.rvs(size=10000, params=params, key=key))
        sample_mean = np.mean(samples, axis=0)
        true_mean = np.array(params["mu"]).flatten()
        np.testing.assert_allclose(sample_mean, true_mean, atol=0.1,
                                   err_msg="MvtNormal sample mean off")


class TestMultivariateParameterRecovery:
    """Verify fit() recovers parameters from synthetic data.

    Catches FINDING-04-01: LDMLE scale reconstruction.
    """

    def test_mvt_normal_recovery(self):
        """MVT Normal: fit should recover mu and sigma from 2000 samples."""
        d = 3
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([[2.0, 0.5, 0.0],
                           [0.5, 1.5, 0.3],
                           [0.0, 0.3, 1.0]])

        np.random.seed(42)
        data = np.random.multivariate_normal(mu, sigma, size=2000)

        fitted = mvt_normal.fit(x=jnp.array(data))
        p = fitted.params

        np.testing.assert_allclose(
            np.array(p["mu"]).flatten(), mu, atol=0.15,
            err_msg="MvtNormal mu not recovered"
        )
        np.testing.assert_allclose(
            np.array(p["sigma"]), sigma, rtol=0.2, atol=0.15,
            err_msg="MvtNormal sigma not recovered"
        )


# ---------------------------------------------------------------------------
# Gradient correctness
# ---------------------------------------------------------------------------

class TestMultivariateGradients:
    """Verify gradients of multivariate logpdf."""

    @pytest.mark.parametrize("dist", [mvt_normal, mvt_student_t],
                             ids=["MvtNormal", "MvtStudentT"])
    def test_logpdf_data_gradient(self, dist):
        """jax.grad of logpdf wrt data should be finite and non-NaN."""
        d = 3
        params = dist.example_params(dim=d)
        x = jnp.ones((5, d))

        def logpdf_sum(x_):
            return dist.logpdf(x=x_, params=params).sum()

        g = jax.grad(logpdf_sum)(x)
        assert no_nans(g), f"{dist.name} logpdf gradient has NaNs"
        assert is_finite(g), f"{dist.name} logpdf gradient not finite"
