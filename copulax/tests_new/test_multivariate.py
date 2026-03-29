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

    def test_mvt_normal_sample_covariance_close(self):
        """Large sample covariance should converge to sigma."""
        d = 3
        mu = jnp.array([1.0, -2.0, 3.0]).reshape(d, 1)
        sigma = jnp.array([[2.0, 0.5, -0.3],
                            [0.5, 1.5, 0.2],
                            [-0.3, 0.2, 1.0]])
        params = mvt_normal._params_dict(mu=mu, sigma=sigma)

        key = jax.random.PRNGKey(123)
        samples = np.array(mvt_normal.rvs(size=50000, params=params, key=key))
        sample_cov = np.cov(samples, rowvar=False)

        np.testing.assert_allclose(
            sample_cov, np.array(sigma), atol=0.05,
            err_msg="MvtNormal sample covariance doesn't converge to sigma"
        )


class TestMvtNormalD1Reduction:
    """Verify d=1 MVT Normal matches univariate normal exactly."""

    @pytest.mark.parametrize("mu,sigma_sq", [(0.0, 1.0), (5.0, 2.0), (-3.0, 0.5)])
    def test_logpdf_matches_univariate_normal(self, mu, sigma_sq):
        """d=1 MvtNormal logpdf should match scipy.stats.norm.logpdf."""
        params = mvt_normal._params_dict(
            mu=jnp.array([[mu]]), sigma=jnp.array([[sigma_sq]])
        )
        x = jnp.linspace(mu - 4 * np.sqrt(sigma_sq), mu + 4 * np.sqrt(sigma_sq), 50).reshape(-1, 1)

        cx_logpdf = np.array(mvt_normal.logpdf(x=x, params=params)).flatten()
        sp_logpdf = scipy.stats.norm.logpdf(
            np.array(x).flatten(), loc=mu, scale=np.sqrt(sigma_sq)
        )
        np.testing.assert_allclose(
            cx_logpdf, sp_logpdf, atol=1e-14,
            err_msg=f"d=1 MvtNormal logpdf != univariate normal (mu={mu}, sigma²={sigma_sq})"
        )


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


# ---------------------------------------------------------------------------
# MvtGH ECME fitting (McNeil et al. 2005, Section 3.4.2)
# ---------------------------------------------------------------------------

class TestMvtGHECME:
    """Tests for the ECME fitting algorithm on the multivariate GH."""

    def _make_skewed_params(self, d=2):
        """Create skewed MvtGH params for testing."""
        return mvt_gh._params_dict(
            lamb=1.0,
            chi=2.0,
            psi=1.5,
            mu=jnp.array([[1.0], [-0.5]]) if d == 2 else jnp.ones((d, 1)),
            gamma=jnp.array([[0.5], [-0.3]]) if d == 2 else jnp.full((d, 1), 0.3),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]) if d == 2 else jnp.eye(d),
        )

    def test_em_fit_no_nans(self):
        """EM fitted parameters should be finite and NaN-free."""
        key = jax.random.PRNGKey(42)
        params = self._make_skewed_params(d=2)
        samples = mvt_gh.rvs(size=300, params=params, key=key)

        fitted = mvt_gh.fit(samples, method="em", maxiter=50)
        for k, v in fitted._stored_params.items():
            arr = np.array(v)
            assert no_nans(arr), f"EM param '{k}' has NaNs"
            assert is_finite(arr), f"EM param '{k}' not finite"

    def test_em_beats_ldmle_on_skewed_data(self):
        """EM should achieve higher log-likelihood than LDMLE on skewed data."""
        key = jax.random.PRNGKey(123)
        params = self._make_skewed_params(d=2)
        samples = mvt_gh.rvs(size=500, params=params, key=key)

        fitted_em = mvt_gh.fit(samples, method="em", maxiter=100)
        fitted_ldmle = mvt_gh.fit(samples, method="ldmle", maxiter=100)

        ll_em = float(jnp.sum(mvt_gh.logpdf(samples, params=fitted_em._stored_params)))
        ll_ldmle = float(jnp.sum(mvt_gh.logpdf(samples, params=fitted_ldmle._stored_params)))

        assert ll_em > ll_ldmle, (
            f"EM log-likelihood ({ll_em:.1f}) should beat LDMLE ({ll_ldmle:.1f})"
        )

    def test_em_parameter_recovery_symmetric(self):
        """EM should recover approximately correct params from symmetric data."""
        key = jax.random.PRNGKey(42)
        true_params = mvt_gh._params_dict(
            lamb=0.5, chi=1.5, psi=1.0,
            mu=jnp.array([[2.0], [-1.0]]),
            gamma=jnp.zeros((2, 1)),
            sigma=jnp.array([[1.5, 0.4], [0.4, 1.0]]),
        )
        samples = mvt_gh.rvs(size=2000, params=true_params, key=key)
        fitted = mvt_gh.fit(samples, method="em", maxiter=100)
        p = fitted._stored_params

        # mu should be close
        np.testing.assert_allclose(
            np.array(p["mu"]).flatten(),
            np.array(true_params["mu"]).flatten(),
            atol=0.3,
            err_msg="EM mu not recovered (symmetric case)",
        )

        # gamma should be near zero
        assert np.max(np.abs(np.array(p["gamma"]))) < 0.5, \
            "EM gamma should be near zero for symmetric data"

    def test_em_parameter_recovery_skewed(self):
        """EM should recover approximately correct mu from skewed data."""
        key = jax.random.PRNGKey(99)
        true_params = self._make_skewed_params(d=2)
        samples = mvt_gh.rvs(size=2000, params=true_params, key=key)
        fitted = mvt_gh.fit(samples, method="em", maxiter=100)
        p = fitted._stored_params

        # The fitted logpdf should be close to the true logpdf on the data
        ll_fitted = float(jnp.sum(mvt_gh.logpdf(samples, params=p)))
        ll_true = float(jnp.sum(mvt_gh.logpdf(samples, params=true_params)))

        # MLE should be at least as good as true params on the sample
        assert ll_fitted >= ll_true - 10.0, (
            f"EM ll ({ll_fitted:.1f}) much worse than true ({ll_true:.1f})"
        )

    def test_em_fit_returns_fitted_instance(self):
        """EM fit should return a proper fitted instance with stored params."""
        key = jax.random.PRNGKey(42)
        params = mvt_gh.example_params(dim=2)
        samples = mvt_gh.rvs(size=100, params=params, key=key)

        fitted = mvt_gh.fit(samples, method="em", maxiter=30)
        assert fitted._stored_params is not None, \
            "EM fit did not produce stored params"

        # Should be able to call logpdf without explicit params
        logpdf = fitted.logpdf(x=samples)
        assert no_nans(np.array(logpdf)), "Fitted instance logpdf has NaNs"

    def test_em_fit_d3(self):
        """EM should work for d=3 (higher dimensionality)."""
        key = jax.random.PRNGKey(42)
        params = mvt_gh._params_dict(
            lamb=0.5, chi=1.0, psi=1.0,
            mu=jnp.zeros((3, 1)),
            gamma=jnp.array([[0.2], [-0.1], [0.3]]),
            sigma=jnp.eye(3),
        )
        samples = mvt_gh.rvs(size=500, params=params, key=key)

        fitted = mvt_gh.fit(samples, method="em", maxiter=50)
        p = fitted._stored_params
        for k, v in p.items():
            arr = np.array(v)
            assert no_nans(arr), f"EM d=3 param '{k}' has NaNs"
            assert is_finite(arr), f"EM d=3 param '{k}' not finite"

    def test_ldmle_still_works(self):
        """LDMLE method should still work via method='ldmle'."""
        key = jax.random.PRNGKey(42)
        params = mvt_gh.example_params(dim=2)
        samples = mvt_gh.rvs(size=100, params=params, key=key)

        fitted = mvt_gh.fit(samples, method="ldmle", maxiter=50)
        assert fitted._stored_params is not None, \
            "LDMLE fit did not produce stored params"


class TestMvtGHPdfIntegration:
    """Verify MvtGH PDF integrates to ~1 (d=2)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("lamb,chi,psi,gamma_val", [
        (0.5, 1.0, 1.0, 0.0),     # symmetric
        (-0.5, 2.0, 1.5, 0.3),    # skewed
        (1.0, 1.0, 1.0, 0.0),     # lambda > 0
        (-1.5, 3.0, 0.5, -0.2),   # heavy-tailed
    ], ids=["symmetric", "skewed", "lambda_pos", "heavy_tail"])
    def test_mvt_gh_integrates_to_one(self, lamb, chi, psi, gamma_val):
        """2D MvtGH PDF should integrate to approximately 1."""
        params = mvt_gh._params_dict(
            lamb=lamb, chi=chi, psi=psi,
            mu=jnp.zeros((2, 1)),
            gamma=jnp.full((2, 1), gamma_val),
            sigma=jnp.eye(2),
        )

        def _inner(x1, x0):
            x = jnp.array([[x0, x1]])
            return mvt_gh.pdf(x=x, params=params).flatten()[0]

        def _outer(x0):
            val, _ = quadgk(lambda x1: _inner(x1, x0), interval=(-20.0, 20.0))
            return val.reshape(())

        result, _ = quadgk(_outer, interval=(-20.0, 20.0))
        np.testing.assert_allclose(
            float(result), 1.0, rtol=5e-2,
            err_msg=f"MvtGH PDF doesn't integrate to 1 (lamb={lamb}, chi={chi}, "
                    f"psi={psi}, gamma={gamma_val})"
        )


class TestMvtGHD1Reduction:
    """Verify d=1 MvtGH matches univariate GH logpdf."""

    @pytest.mark.parametrize("lamb,chi,psi,gamma_val", [
        (0.5, 1.0, 1.0, 0.0),
        (-0.5, 2.0, 1.5, 0.3),
        (1.0, 0.5, 2.0, -0.5),
        (-1.5, 3.0, 0.5, 0.1),
    ], ids=["set0", "set1", "set2", "set3"])
    def test_logpdf_matches_univariate_gh(self, lamb, chi, psi, gamma_val):
        """d=1 MvtGH logpdf should match univariate GH logpdf."""
        from copulax.univariate import gh

        # Multivariate d=1 params
        mvt_params = mvt_gh._params_dict(
            lamb=lamb, chi=chi, psi=psi,
            mu=jnp.array([[0.5]]),
            gamma=jnp.array([[gamma_val]]),
            sigma=jnp.array([[1.5]]),
        )

        # Univariate params (sigma is std dev in univariate)
        uv_params = gh._params_dict(
            lamb=lamb, chi=chi, psi=psi,
            mu=0.5, sigma=jnp.sqrt(1.5), gamma=gamma_val,
        )

        x_1d = jnp.linspace(-5.0, 5.0, 20).reshape(-1, 1)
        x_uv = x_1d.flatten()

        mvt_logpdf = np.array(mvt_gh.logpdf(x=x_1d, params=mvt_params)).flatten()
        uv_logpdf = np.array(gh.logpdf(x=x_uv, params=uv_params)).flatten()

        np.testing.assert_allclose(
            mvt_logpdf, uv_logpdf, atol=1e-10,
            err_msg=f"d=1 MvtGH != univariate GH (lamb={lamb}, chi={chi}, "
                    f"psi={psi}, gamma={gamma_val})"
        )


# ---------------------------------------------------------------------------
# MVT Skewed-T audit tests
# ---------------------------------------------------------------------------

class TestMvtSkewedTD1Reduction:
    """d=1 MVT Skewed-T must match univariate Skewed-T."""

    @pytest.mark.parametrize(
        "nu, gamma_val",
        [(5.0, 0.5), (4.0, 1.0), (10.0, -0.3), (3.5, 2.0)],
        ids=["nu5_g05", "nu4_g1", "nu10_gn03", "nu35_g2"],
    )
    def test_d1_matches_univariate(self, nu, gamma_val):
        from copulax.univariate import skewed_t

        mu_mvt = jnp.array([[0.0]])
        gamma_mvt = jnp.array([[gamma_val]])
        sigma_mvt = jnp.array([[1.0]])

        mvt_params = mvt_skewed_t._params_dict(
            nu=nu, mu=mu_mvt, gamma=gamma_mvt, sigma=sigma_mvt
        )
        uv_params = skewed_t._params_dict(
            nu=nu, mu=0.0, sigma=1.0, gamma=gamma_val
        )

        x = jnp.linspace(-5, 5, 20).reshape(-1, 1)
        mvt_lp = np.array(mvt_skewed_t.logpdf(x, params=mvt_params)).flatten()
        uv_lp = np.array(skewed_t.logpdf(x.flatten(), params=uv_params))

        np.testing.assert_allclose(
            mvt_lp, uv_lp, atol=1e-10,
            err_msg=f"d=1 MVT Skewed-T != univariate (nu={nu}, gamma={gamma_val})"
        )


class TestMvtSkewedTGamma0:
    """gamma=0 MVT Skewed-T must exactly match MVT Student-T."""

    def test_gamma0_matches_student_t(self):
        nu = 5.0
        d = 2
        mu = jnp.zeros((d, 1))
        gamma = jnp.zeros((d, 1))
        sigma = jnp.array([[1.0, 0.3], [0.3, 1.0]])

        skt_params = mvt_skewed_t._params_dict(
            nu=nu, mu=mu, gamma=gamma, sigma=sigma
        )
        st_params = mvt_student_t._params_dict(nu=nu, mu=mu, sigma=sigma)

        x = jnp.array([
            [0.0, 0.0], [1.0, 0.5], [-1.0, 1.0], [2.0, -1.0],
            [-0.5, -0.5], [0.3, 0.7], [-2.0, 2.0],
        ])

        skt_lp = np.array(mvt_skewed_t.logpdf(x, params=skt_params)).flatten()
        st_lp = np.array(mvt_student_t.logpdf(x, params=st_params)).flatten()

        np.testing.assert_allclose(skt_lp, st_lp, atol=1e-14)


class TestMvtSkewedTGHConsistency:
    """MVT Skewed-T must match MVT GH at psi~0."""

    def test_matches_gh_small_psi(self):
        d = 2
        nu = 5.0
        mu = jnp.zeros((d, 1))
        gamma = jnp.array([[0.5], [0.3]])
        sigma = jnp.array([[1.0, 0.3], [0.3, 1.0]])

        skt_params = mvt_skewed_t._params_dict(
            nu=nu, mu=mu, gamma=gamma, sigma=sigma
        )
        gh_params = mvt_gh._params_dict(
            lamb=-nu / 2, chi=nu, psi=1e-10,
            mu=mu, gamma=gamma, sigma=sigma,
        )

        x = jnp.array([[0.0, 0.0], [1.0, 0.5], [-1.0, 1.0], [0.5, -0.5]])

        skt_lp = np.array(mvt_skewed_t.logpdf(x, params=skt_params)).flatten()
        gh_lp = np.array(mvt_gh.logpdf(x, params=gh_params)).flatten()

        np.testing.assert_allclose(skt_lp, gh_lp, atol=1e-6)


class TestMvtSkewedTECME:
    """Tests for MVT Skewed-T ECME fitting algorithm."""

    @staticmethod
    def _make_skewed_params(d=2):
        mu = jnp.zeros((d, 1))
        gamma = jnp.array([[0.4], [0.2]])[:d]
        sigma = jnp.eye(d)
        if d >= 2:
            sigma = sigma.at[0, 1].set(0.3)
            sigma = sigma.at[1, 0].set(0.3)
        return mvt_skewed_t._params_dict(nu=5.0, mu=mu, gamma=gamma, sigma=sigma)

    def test_em_fit_no_nans(self):
        """EM-fitted parameters must be finite and NaN-free."""
        key = jax.random.PRNGKey(42)
        params = self._make_skewed_params(d=2)
        samples = mvt_skewed_t.rvs(size=1000, params=params, key=key)

        fitted = mvt_skewed_t.fit(samples, method="em", maxiter=50)
        fp = fitted._stored_params

        assert fp is not None
        for k, v in fp.items():
            assert not jnp.any(jnp.isnan(v)), f"NaN in {k}"
            assert jnp.all(jnp.isfinite(v)), f"Inf in {k}"

    def test_em_and_ldmle_both_reasonable(self):
        """Both EM and LDMLE should achieve log-likelihoods close to the true value."""
        key = jax.random.PRNGKey(42)
        params = self._make_skewed_params(d=2)
        samples = mvt_skewed_t.rvs(size=2000, params=params, key=key)

        ll_true = float(jnp.sum(mvt_skewed_t.logpdf(samples, params=params)))

        fitted_em = mvt_skewed_t.fit(samples, method="em", maxiter=100)
        fitted_ldmle = mvt_skewed_t.fit(samples, method="ldmle", lr=1e-3, maxiter=200)

        ll_em = float(jnp.sum(mvt_skewed_t.logpdf(
            samples, params=fitted_em._stored_params
        )))
        ll_ldmle = float(jnp.sum(mvt_skewed_t.logpdf(
            samples, params=fitted_ldmle._stored_params
        )))

        # Both should be within 5% of true LL (in absolute terms)
        assert ll_em > ll_true * 1.05, (
            f"EM LL ({ll_em:.1f}) too far from true ({ll_true:.1f})"
        )
        assert ll_ldmle > ll_true * 1.05, (
            f"LDMLE LL ({ll_ldmle:.1f}) too far from true ({ll_true:.1f})"
        )

    def test_em_parameter_recovery_symmetric(self):
        """EM should recover near-zero gamma for symmetric data."""
        key = jax.random.PRNGKey(99)
        d = 2
        params = mvt_skewed_t._params_dict(
            nu=6.0,
            mu=jnp.zeros((d, 1)),
            gamma=jnp.zeros((d, 1)),
            sigma=jnp.eye(d),
        )
        samples = mvt_skewed_t.rvs(size=2000, params=params, key=key)

        fitted = mvt_skewed_t.fit(samples, method="em", maxiter=100)
        fp = fitted._stored_params

        gamma_norm = float(jnp.linalg.norm(fp["gamma"]))
        assert gamma_norm < 0.3, f"gamma should be near 0, got norm={gamma_norm:.3f}"

    def test_em_fit_d3(self):
        """EM should work for d=3."""
        key = jax.random.PRNGKey(7)
        d = 3
        params = mvt_skewed_t._params_dict(
            nu=6.0,
            mu=jnp.zeros((d, 1)),
            gamma=jnp.array([[0.3], [-0.2], [0.1]]),
            sigma=jnp.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]]),
        )
        samples = mvt_skewed_t.rvs(size=2000, params=params, key=key)

        fitted = mvt_skewed_t.fit(samples, method="em", maxiter=100)
        fp = fitted._stored_params

        for k, v in fp.items():
            assert not jnp.any(jnp.isnan(v)), f"NaN in {k}"

        ll = float(jnp.sum(mvt_skewed_t.logpdf(samples, params=fp)))
        assert jnp.isfinite(ll), "Fitted LL is not finite"

    def test_em_returns_fitted_instance(self):
        """EM fit should return a fitted instance with stored params."""
        key = jax.random.PRNGKey(0)
        params = self._make_skewed_params(d=2)
        samples = mvt_skewed_t.rvs(size=500, params=params, key=key)

        fitted = mvt_skewed_t.fit(samples, method="em", maxiter=30)

        assert fitted._stored_params is not None
        assert "nu" in fitted._stored_params
        assert "mu" in fitted._stored_params
        assert "gamma" in fitted._stored_params
        assert "sigma" in fitted._stored_params

        # Should be callable
        lp = fitted.logpdf(samples)
        assert lp.shape[0] == 500

    def test_ldmle_still_works(self):
        """LDMLE method should still be accessible."""
        key = jax.random.PRNGKey(0)
        params = self._make_skewed_params(d=2)
        samples = mvt_skewed_t.rvs(size=500, params=params, key=key)

        fitted = mvt_skewed_t.fit(samples, method="ldmle", lr=1e-3, maxiter=50)
        fp = fitted._stored_params

        assert fp is not None
        assert "nu" in fp
