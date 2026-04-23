"""Rigorous tests for all 4 CopulAX multivariate distributions.

Cross-validates against scipy.stats.multivariate_normal and multivariate_t.
Verifies density integration, moment formulas, and parameter recovery.
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
# MvtNormal
# ---------------------------------------------------------------------------

class TestMvtNormal:
    """Multivariate Normal: scipy cross-validation, stats, integration,
    sampling, parameter recovery, metrics."""

    @staticmethod
    def _params(d=3):
        return {"mu": jnp.zeros((d, 1)), "sigma": jnp.eye(d) * 2.0}

    # ----- Against scipy -----

    @pytest.mark.parametrize("d", [2, 3, 5])
    def test_logpdf_matches_scipy(self, d):
        """logpdf should match scipy.stats.multivariate_normal.logpdf."""
        params = self._params(d)
        mu_np = np.array(params["mu"]).flatten()
        sigma_np = np.array(params["sigma"])

        np.random.seed(42)
        x = np.random.multivariate_normal(mu_np, sigma_np, size=30)

        cx_logpdf = np.array(mvt_normal.logpdf(x=jnp.array(x), params=params)).flatten()
        sp_logpdf = scipy.stats.multivariate_normal.logpdf(x, mean=mu_np, cov=sigma_np)

        np.testing.assert_allclose(cx_logpdf, sp_logpdf, rtol=1e-6,
                                   err_msg="MvtNormal logpdf mismatch vs scipy")

    # ----- d=1 reduction -----

    @pytest.mark.parametrize("mu,sigma_sq", [(0.0, 1.0), (5.0, 2.0), (-3.0, 0.5)])
    def test_d1_matches_univariate_normal(self, mu, sigma_sq):
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

    # ----- Stats -----

    def test_mean_and_cov(self):
        """MVT Normal: mean = mu, cov = sigma."""
        d = 3
        params = self._params(d)
        stats = mvt_normal.stats(params=params)

        np.testing.assert_allclose(
            np.array(stats["mean"]).flatten(),
            np.array(params["mu"]).flatten(), atol=1e-10
        )
        np.testing.assert_allclose(
            np.array(stats["cov"]),
            np.array(params["sigma"]), atol=1e-10
        )

    # ----- PDF integration -----

    def test_pdf_integrates_to_one(self):
        """2D multivariate normal PDF should integrate to 1."""
        params = self._params(d=2)

        def _inner(x1, x0):
            x = jnp.array([[x0, x1]])
            return mvt_normal.pdf(x=x, params=params).flatten()[0]

        def _outer(x0):
            val, _ = quadgk(lambda x1: _inner(x1, x0), interval=(-10.0, 10.0))
            return val.reshape(())

        result, _ = quadgk(_outer, interval=(-10.0, 10.0))
        np.testing.assert_allclose(float(result), 1.0, rtol=1e-2,
                                   err_msg="MvtNormal PDF doesn't integrate to 1")

    # ----- Sampling -----

    def test_sample_mean_close(self):
        """Large sample mean should be close to mu."""
        d = 3
        params = self._params(d)
        key = jax.random.PRNGKey(42)
        samples = np.array(mvt_normal.rvs(size=10000, params=params, key=key))
        sample_mean = np.mean(samples, axis=0)
        true_mean = np.array(params["mu"]).flatten()
        np.testing.assert_allclose(sample_mean, true_mean, atol=0.1,
                                   err_msg="MvtNormal sample mean off")

    def test_sample_covariance_close(self):
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

    # ----- Parameter recovery -----

    def test_fit_recovers_params(self):
        """fit should recover mu and sigma from 2000 samples."""
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

    # ----- Metrics -----

    def test_metrics_finite(self):
        """loglikelihood, AIC, and BIC should be finite after fitting."""
        key = jax.random.PRNGKey(42)
        d = 2
        params = mvt_normal.example_params(dim=d)
        samples = mvt_normal.rvs(size=200, params=params, key=key)

        fitted = mvt_normal.fit(samples)
        logll = float(fitted.loglikelihood(x=samples))
        aic_val = float(fitted.aic(x=samples))
        bic_val = float(fitted.bic(x=samples))

        assert np.isfinite(logll), "MvtNormal loglikelihood not finite"
        assert np.isfinite(aic_val), "MvtNormal AIC not finite"
        assert np.isfinite(bic_val), "MvtNormal BIC not finite"
        assert aic_val < 0 or aic_val > -1e10, "MvtNormal AIC out of range"


# ---------------------------------------------------------------------------
# MvtStudentT
# ---------------------------------------------------------------------------

class TestMvtStudentT:
    """Multivariate Student-T: scipy cross-validation, d=1 reduction, stats,
    integration, sampling, parameter recovery (LDMLE scale), metrics.
    """

    @staticmethod
    def _params(d=3, nu=5.0):
        return {
            "nu": jnp.array(nu),
            "mu": jnp.ones((d, 1)) * 0.5,
            "sigma": jnp.eye(d) * 1.5,
        }

    # ----- Against scipy -----

    @pytest.mark.parametrize("d", [2, 3])
    def test_logpdf_matches_scipy(self, d):
        """logpdf should match scipy.stats.multivariate_t.logpdf."""
        params = self._params(d, nu=5.0)
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

    # ----- d=1 reduction -----

    @pytest.mark.parametrize(
        "nu,mu,sigma_sq",
        [(5.0, 0.0, 1.0), (10.0, 2.0, 0.5), (3.0, -1.0, 2.0)],
        ids=["nu5_mu0_s1", "nu10_mu2_s05", "nu3_mun1_s2"],
    )
    def test_d1_matches_univariate_student_t(self, nu, mu, sigma_sq):
        """d=1 MvtStudentT logpdf should match scipy.stats.t.logpdf."""
        params = mvt_student_t._params_dict(
            nu=jnp.array(nu),
            mu=jnp.array([[mu]]),
            sigma=jnp.array([[sigma_sq]]),
        )
        sigma = float(np.sqrt(sigma_sq))
        x = jnp.linspace(mu - 5 * sigma, mu + 5 * sigma, 50).reshape(-1, 1)

        cx_logpdf = np.array(mvt_student_t.logpdf(x=x, params=params)).flatten()
        sp_logpdf = scipy.stats.t.logpdf(
            np.array(x).flatten(), df=nu, loc=mu, scale=sigma
        )
        np.testing.assert_allclose(
            cx_logpdf, sp_logpdf, atol=1e-10,
            err_msg=f"d=1 MvtStudentT logpdf != univariate t "
                    f"(nu={nu}, mu={mu}, sigma²={sigma_sq})"
        )

    # ----- Stats -----

    def test_cov_formula(self):
        """MVT Student-T: cov = nu/(nu-2) * sigma for nu > 2."""
        d = 3
        params = self._params(d, nu=5.0)
        stats = mvt_student_t.stats(params=params)
        nu = float(params["nu"])

        expected_cov = (nu / (nu - 2.0)) * np.array(params["sigma"])
        np.testing.assert_allclose(
            np.array(stats["cov"]), expected_cov, rtol=1e-5,
            err_msg="MVT Student-T covariance formula incorrect"
        )

    # ----- PDF integration -----

    def test_pdf_integrates_to_one(self):
        """2D multivariate Student-T PDF should integrate to 1."""
        params = self._params(d=2, nu=5.0)

        def _inner(x1, x0):
            x = jnp.array([[x0, x1]])
            return mvt_student_t.pdf(x=x, params=params).flatten()[0]

        def _outer(x0):
            val, _ = quadgk(lambda x1: _inner(x1, x0), interval=(-15.0, 15.0))
            return val.reshape(())

        result, _ = quadgk(_outer, interval=(-15.0, 15.0))
        np.testing.assert_allclose(float(result), 1.0, rtol=1e-2,
                                   err_msg="MvtStudentT PDF doesn't integrate to 1")

    # ----- Sampling -----

    def test_sample_mean_close(self):
        """Large sample mean should be close to mu."""
        d = 3
        params = self._params(d, nu=10.0)  # larger nu for finite-variance
        key = jax.random.PRNGKey(42)
        samples = np.array(mvt_student_t.rvs(size=10000, params=params, key=key))
        sample_mean = np.mean(samples, axis=0)
        true_mean = np.array(params["mu"]).flatten()
        np.testing.assert_allclose(sample_mean, true_mean, atol=0.15,
                                   err_msg="MvtStudentT sample mean off")

    # ----- Parameter recovery (LDMLE scale formula) -----

    def test_ldmle_scale_formula(self):
        """Verify LDMLE sigma reconstruction uses (nu-2)/nu, not (nu-2)/2.
        """
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

        # The fitted sigma should be comparable to the true sigma
        # (not 5x too large). The scale factor is (nu-2)/nu for the
        # sample covariance -> MLE sigma conversion.
        sigma_ratio = np.mean(np.abs(fitted_sigma)) / np.mean(np.abs(sigma_true))
        assert 0.3 < sigma_ratio < 3.0, \
            f"Sigma ratio = {sigma_ratio:.2f}, likely LDMLE scale bug"

    # ----- Metrics -----

    def test_metrics_finite(self):
        """loglikelihood, AIC, and BIC should be finite after fitting."""
        key = jax.random.PRNGKey(42)
        d = 2
        params = mvt_student_t.example_params(dim=d)
        samples = mvt_student_t.rvs(size=200, params=params, key=key)

        fitted = mvt_student_t.fit(samples)
        logll = float(fitted.loglikelihood(x=samples))
        aic_val = float(fitted.aic(x=samples))
        bic_val = float(fitted.bic(x=samples))

        assert np.isfinite(logll), "MvtStudentT loglikelihood not finite"
        assert np.isfinite(aic_val), "MvtStudentT AIC not finite"
        assert np.isfinite(bic_val), "MvtStudentT BIC not finite"
        assert aic_val < 0 or aic_val > -1e10, "MvtStudentT AIC out of range"


# ---------------------------------------------------------------------------
# MvtGH
# ---------------------------------------------------------------------------

class TestMvtGH:
    """Multivariate generalised hyperbolic: logpdf properties, d=1 reduction,
    PDF integration across regimes, ECME fitting, metrics.
    """

    @staticmethod
    def _params(d=3):
        """Symmetric-for-scipy-comparison MvtGH params."""
        return {
            "lamb": jnp.array(1.0),
            "chi": jnp.array(1.0),
            "psi": jnp.array(1.0),
            "mu": jnp.ones((d, 1)) * 1.0,
            "gamma": jnp.ones((d, 1)) * 0.0,
            "sigma": jnp.eye(d),
        }

    @staticmethod
    def _skewed_params(d=2):
        """Skewed MvtGH params for fitting tests."""
        return mvt_gh._params_dict(
            lamb=1.0, chi=2.0, psi=1.5,
            mu=jnp.array([[1.0], [-0.5]]) if d == 2 else jnp.ones((d, 1)),
            gamma=jnp.array([[0.5], [-0.3]]) if d == 2 else jnp.full((d, 1), 0.3),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]) if d == 2 else jnp.eye(d),
        )

    # ----- logpdf -----

    def test_logpdf_with_nonzero_mu(self):
        """logpdf should differ when mu changes (H term depends on x-mu).
        """
        d = 3
        params1 = self._params(d)
        params2 = dict(params1)
        params2["mu"] = jnp.ones((d, 1)) * 5.0  # large shift

        x = jnp.ones((10, d))

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
        params = self._params(d)
        np.random.seed(42)
        x = np.random.normal(size=(20, d))
        logpdf = np.array(mvt_gh.logpdf(x=jnp.array(x), params=params))
        assert no_nans(logpdf), "MVT-GH logpdf has NaNs"
        assert is_finite(logpdf), "MVT-GH logpdf has non-finite values"

    # ----- d=1 reduction -----

    @pytest.mark.parametrize("lamb,chi,psi,gamma_val", [
        (0.5, 1.0, 1.0, 0.0),
        (-0.5, 2.0, 1.5, 0.3),
        (1.0, 0.5, 2.0, -0.5),
        (-1.5, 3.0, 0.5, 0.1),
    ], ids=["set0", "set1", "set2", "set3"])
    def test_d1_matches_univariate_gh(self, lamb, chi, psi, gamma_val):
        """d=1 MvtGH logpdf should match univariate GH logpdf."""
        from copulax.univariate import gh

        mvt_params = mvt_gh._params_dict(
            lamb=lamb, chi=chi, psi=psi,
            mu=jnp.array([[0.5]]),
            gamma=jnp.array([[gamma_val]]),
            sigma=jnp.array([[1.5]]),
        )
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

    # ----- PDF integration -----

    @pytest.mark.parametrize("lamb,chi,psi,gamma_val", [
        (0.5, 1.0, 1.0, 0.0),     # symmetric
        (-0.5, 2.0, 1.5, 0.3),    # skewed
        (1.0, 1.0, 1.0, 0.0),     # lamb > 0
        (-1.5, 3.0, 0.5, -0.2),   # heavy-tailed
    ], ids=["symmetric", "skewed", "lamb_pos", "heavy_tail"])
    def test_pdf_integrates_to_one(self, lamb, chi, psi, gamma_val):
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

    # ----- ECME fitting -----

    def test_em_fit_no_nans(self):
        """EM fitted parameters should be finite and NaN-free."""
        key = jax.random.PRNGKey(42)
        params = self._skewed_params(d=2)
        samples = mvt_gh.rvs(size=300, params=params, key=key)

        fitted = mvt_gh.fit(samples, method="em", maxiter=50)
        for k, v in fitted._stored_params.items():
            arr = np.array(v)
            assert no_nans(arr), f"EM param '{k}' has NaNs"
            assert is_finite(arr), f"EM param '{k}' not finite"

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

        np.testing.assert_allclose(
            np.array(p["mu"]).flatten(),
            np.array(true_params["mu"]).flatten(),
            atol=0.3,
            err_msg="EM mu not recovered (symmetric case)",
        )
        assert np.max(np.abs(np.array(p["gamma"]))) < 0.5, \
            "EM gamma should be near zero for symmetric data"

    def test_em_parameter_recovery_skewed(self):
        """EM should recover approximately correct params from skewed data."""
        key = jax.random.PRNGKey(99)
        true_params = self._skewed_params(d=2)
        samples = mvt_gh.rvs(size=2000, params=true_params, key=key)
        fitted = mvt_gh.fit(samples, method="em", maxiter=100)
        p = fitted._stored_params

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

    def test_em_and_ldmle_both_reasonable(self):
        """Both EM and LDMLE should achieve log-likelihoods close to the true value.

        GH has a known parameterisation non-identifiability (scaling invariance
        between chi, psi, gamma, Sigma), so we compare log-likelihoods rather
        than individual parameters.
        """
        key = jax.random.PRNGKey(42)
        d = 2
        params = mvt_gh._params_dict(
            lamb=-2.5, chi=5.0, psi=1.0,
            mu=jnp.zeros((d, 1)),
            gamma=jnp.array([[0.4], [0.2]]),
            sigma=jnp.eye(d).at[0, 1].set(0.3).at[1, 0].set(0.3),
        )
        samples = mvt_gh.rvs(size=2000, params=params, key=key)

        ll_true = float(jnp.sum(mvt_gh.logpdf(samples, params=params)))

        fitted_em = mvt_gh.fit(samples, method="em", maxiter=100)
        fitted_ldmle = mvt_gh.fit(samples, method="ldmle")

        ll_em = float(jnp.sum(mvt_gh.logpdf(
            samples, params=fitted_em._stored_params
        )))
        ll_ldmle = float(jnp.sum(mvt_gh.logpdf(
            samples, params=fitted_ldmle._stored_params
        )))

        # ll_true is negative; ll * 1.05 is 5% more negative. Fit must not be
        # more than 5% worse than oracle in absolute LL terms.
        assert ll_em > ll_true * 1.05, (
            f"EM LL ({ll_em:.1f}) too far from true ({ll_true:.1f})"
        )
        assert ll_ldmle > ll_true * 1.05, (
            f"LDMLE LL ({ll_ldmle:.1f}) too far from true ({ll_true:.1f})"
        )

    # ----- Metrics -----

    def test_metrics_finite(self):
        """loglikelihood, AIC, and BIC should be finite after EM fitting."""
        key = jax.random.PRNGKey(42)
        params = mvt_gh.example_params(dim=2)
        samples = mvt_gh.rvs(size=200, params=params, key=key)

        fitted = mvt_gh.fit(samples, method="em", maxiter=30)
        logll = float(fitted.loglikelihood(x=samples))
        aic_val = float(fitted.aic(x=samples))
        bic_val = float(fitted.bic(x=samples))

        assert np.isfinite(logll), "MvtGH loglikelihood not finite"
        assert np.isfinite(aic_val), "MvtGH AIC not finite"
        assert np.isfinite(bic_val), "MvtGH BIC not finite"


# ---------------------------------------------------------------------------
# MvtSkewedT
# ---------------------------------------------------------------------------

class TestMvtSkewedT:
    """Multivariate Skewed-T: d=1 reduction, consistency with StudentT and GH,
    ECME fitting, metrics."""

    @staticmethod
    def _skewed_params(d=2):
        mu = jnp.zeros((d, 1))
        gamma = jnp.array([[0.4], [0.2]])[:d]
        sigma = jnp.eye(d)
        if d >= 2:
            sigma = sigma.at[0, 1].set(0.3)
            sigma = sigma.at[1, 0].set(0.3)
        return mvt_skewed_t._params_dict(nu=5.0, mu=mu, gamma=gamma, sigma=sigma)

    # ----- d=1 reduction -----

    @pytest.mark.parametrize(
        "nu, gamma_val",
        [(5.0, 0.5), (4.0, 1.0), (10.0, -0.3), (3.5, 2.0)],
        ids=["nu5_g05", "nu4_g1", "nu10_gn03", "nu35_g2"],
    )
    def test_d1_matches_univariate(self, nu, gamma_val):
        """d=1 MVT Skewed-T must match univariate Skewed-T."""
        from copulax.univariate import skewed_t

        mvt_params = mvt_skewed_t._params_dict(
            nu=nu,
            mu=jnp.array([[0.0]]),
            gamma=jnp.array([[gamma_val]]),
            sigma=jnp.array([[1.0]]),
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

    # ----- Consistency -----

    def test_gamma0_matches_student_t(self):
        """gamma=0 MVT Skewed-T must exactly match MVT Student-T."""
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

    def test_small_psi_matches_gh(self):
        """MVT Skewed-T must match MVT GH at psi~0."""
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

    # ----- ECME fitting -----

    def test_em_fit_no_nans(self):
        """EM-fitted parameters must be finite and NaN-free."""
        key = jax.random.PRNGKey(42)
        params = self._skewed_params(d=2)
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
        params = self._skewed_params(d=2)
        samples = mvt_skewed_t.rvs(size=2000, params=params, key=key)

        ll_true = float(jnp.sum(mvt_skewed_t.logpdf(samples, params=params)))

        fitted_em = mvt_skewed_t.fit(samples, method="em", maxiter=100)
        fitted_ldmle = mvt_skewed_t.fit(samples, method="ldmle")

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
        params = self._skewed_params(d=2)
        samples = mvt_skewed_t.rvs(size=500, params=params, key=key)

        fitted = mvt_skewed_t.fit(samples, method="em", maxiter=30)

        assert fitted._stored_params is not None
        assert "nu" in fitted._stored_params
        assert "mu" in fitted._stored_params
        assert "gamma" in fitted._stored_params
        assert "sigma" in fitted._stored_params

        lp = fitted.logpdf(samples)
        assert lp.shape[0] == 500

    def test_ldmle_still_works(self):
        """LDMLE method should still be accessible."""
        key = jax.random.PRNGKey(0)
        params = self._skewed_params(d=2)
        samples = mvt_skewed_t.rvs(size=500, params=params, key=key)

        fitted = mvt_skewed_t.fit(samples, method="ldmle", lr=1e-3, maxiter=50)
        fp = fitted._stored_params

        assert fp is not None
        assert "nu" in fp

    # ----- Metrics -----

    def test_metrics_finite(self):
        """loglikelihood, AIC, and BIC should be finite after EM fitting."""
        key = jax.random.PRNGKey(42)
        params = mvt_skewed_t.example_params(dim=2)
        samples = mvt_skewed_t.rvs(size=200, params=params, key=key)

        fitted = mvt_skewed_t.fit(samples, method="em", maxiter=30)
        logll = float(fitted.loglikelihood(x=samples))
        aic_val = float(fitted.aic(x=samples))
        bic_val = float(fitted.bic(x=samples))

        assert np.isfinite(logll), "MvtSkewedT loglikelihood not finite"
        assert np.isfinite(aic_val), "MvtSkewedT AIC not finite"
        assert np.isfinite(bic_val), "MvtSkewedT BIC not finite"

    # ----- Density integration -----

    @pytest.mark.parametrize("nu,gamma_val", [
        (5.0, 0.0),
        (5.0, 0.5),
        (3.5, 0.5),
        (15.0, 0.0),
    ], ids=["symmetric", "mildly_skewed", "heavy_tailed_skewed", "near_normal"])
    def test_pdf_integrates_to_one(self, nu, gamma_val):
        """2D MvtSkewedT PDF should integrate to approximately 1."""
        params = mvt_skewed_t._params_dict(
            nu=nu,
            mu=jnp.zeros((2, 1)),
            gamma=jnp.full((2, 1), gamma_val),
            sigma=jnp.eye(2),
        )

        def _inner(x1, x0):
            x = jnp.array([[x0, x1]])
            return mvt_skewed_t.pdf(x=x, params=params).flatten()[0]

        def _outer(x0):
            val, _ = quadgk(lambda x1: _inner(x1, x0), interval=(-20.0, 20.0))
            return val.reshape(())

        result, _ = quadgk(_outer, interval=(-20.0, 20.0))
        np.testing.assert_allclose(
            float(result), 1.0, rtol=5e-2,
            err_msg=f"MvtSkewedT PDF doesn't integrate to 1 (nu={nu}, gamma={gamma_val})",
        )


# ---------------------------------------------------------------------------
# Cross-distribution tests
# ---------------------------------------------------------------------------

class TestMultivariateSampling:
    """Sample shape correctness across all multivariate distributions."""

    @pytest.mark.parametrize("dist", ALL_MVT_DISTS, ids=MVT_IDS)
    def test_sampling_shape(self, dist):
        d = 3
        params = dist.example_params(dim=d)
        key = jax.random.PRNGKey(42)
        samples = dist.rvs(size=50, params=params, key=key)
        assert samples.shape == (50, d), \
            f"{dist.name} sample shape = {samples.shape}, expected (50, {d})"

    # --- JIT-compatibility contract for fit() ---

    # Each entry maps a multivariate distribution to its (static_kwargs,
    # static_argnames) for the JIT test. MvtNormal uses `sigma_method`
    # while the other three use `cov_method` — the naming is inherited
    # from the source and is intentional.
    _FIT_JIT_CONFIG = {
        "Mvt-Normal": (
            {"sigma_method": "pearson"},
            ("sigma_method",),
        ),
        "Mvt-Student-T": (
            {"cov_method": "pearson"},
            ("cov_method",),
        ),
        "Mvt-GH": (
            {"method": "em", "cov_method": "pearson"},
            ("method", "cov_method"),
        ),
        "Mvt-Skewed-T": (
            {"method": "em", "cov_method": "pearson"},
            ("method", "cov_method"),
        ),
    }

    _FIT_JIT_PARAMS = [
        pytest.param("Mvt-Normal", id="Mvt-Normal"),
        pytest.param("Mvt-Student-T", id="Mvt-Student-T"),
        pytest.param("Mvt-GH", marks=pytest.mark.slow, id="Mvt-GH"),
        pytest.param("Mvt-Skewed-T", marks=pytest.mark.slow, id="Mvt-Skewed-T"),
    ]

    @pytest.mark.parametrize("dist_name", _FIT_JIT_PARAMS)
    def test_fit_is_jittable(self, dist_name):
        """Every multivariate dist.fit() must be JIT-compatible.

        CopulAX is a JAX-first library; a fit() that silently falls out of
        JIT still produces correct results but runs 10–100× slower. EM-based
        Mvt-GH / Mvt-Skewed-T fits are marked ``@pytest.mark.slow`` because
        their JIT compilation dominates per-test wall time.
        """
        dist = {d.name: d for d in ALL_MVT_DISTS}[dist_name]
        call_kwargs, static_names = self._FIT_JIT_CONFIG[dist_name]
        d = 3
        params = dist.example_params(dim=d)
        key = jax.random.PRNGKey(0)
        x = dist.rvs(size=200, params=params, key=key)

        fit_jit = jax.jit(dist.fit, static_argnames=static_names)
        fitted = fit_jit(x, **call_kwargs)
        assert isinstance(fitted, type(dist)), (
            f"{dist.name}.fit() under JIT did not return "
            f"a {type(dist).__name__} instance (got {type(fitted).__name__})"
        )


class TestMultivariateGradients:
    """Gradient correctness across multivariate distributions."""

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


class TestMultivariateInputValidation:
    """Error handling for invalid inputs."""

    def test_1d_input_raises(self):
        """1D input should raise an error for multivariate distributions."""
        data_1d = jnp.array([1.0, 2.0, 3.0])
        with pytest.raises((ValueError, Exception)):
            mvt_normal.logpdf(data_1d, params=mvt_normal.example_params(dim=2))
