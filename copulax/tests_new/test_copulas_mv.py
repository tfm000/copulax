"""Rigorous tests for the 4 elliptical copulas: Gaussian, Student-T, GH, Skewed-T.

Verifies Sklar's theorem implementation, uniform marginal sampling,
density properties, and fitting.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats
from quadax import quadgk

from copulax.copulas import (
    gaussian_copula, student_t_copula, gh_copula, skewed_t_copula,
)
from copulax.univariate import student_t
from copulax.tests_new.conftest import no_nans, is_finite


FAST_COPULAS = [gaussian_copula, student_t_copula]
FAST_IDS = [c.name for c in FAST_COPULAS]

# GH/SkewedT rows carry `slow` so `-m "not slow"` still runs the fast pair.
ALL_COPULAS_PARAMS = [
    pytest.param(gaussian_copula, id=gaussian_copula.name),
    pytest.param(student_t_copula, id=student_t_copula.name),
    pytest.param(gh_copula, id=gh_copula.name, marks=pytest.mark.slow),
    pytest.param(skewed_t_copula, id=skewed_t_copula.name, marks=pytest.mark.slow),
]


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

    @pytest.mark.parametrize("copula", ALL_COPULAS_PARAMS)
    def test_copula_pdf_positive(self, copula):
        """Copula PDF should be > 0 for all u in (0,1)^d."""
        d = 3
        params = _get_copula_params(copula, d)
        u = _uniform_sample(d, 50)
        pdf = np.array(copula.copula_pdf(u=u, params=params)).flatten()
        assert np.all(pdf[np.isfinite(pdf)] > 0), \
            f"{copula.name} copula_pdf not positive"

    @pytest.mark.parametrize("copula", ALL_COPULAS_PARAMS)
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

    @pytest.mark.parametrize("copula", ALL_COPULAS_PARAMS)
    def test_copula_logpdf_finite(self, copula):
        """copula_logpdf should be finite for interior points."""
        d = 3
        params = _get_copula_params(copula, d)
        u = _uniform_sample(d, 30)
        logpdf = np.array(copula.copula_logpdf(u=u, params=params))
        assert no_nans(logpdf), f"{copula.name} copula_logpdf has NaNs"

    @pytest.mark.parametrize("copula", ALL_COPULAS_PARAMS)
    def test_copula_pdf_integrates_to_one(self, copula):
        """Copula density must integrate to 1 on (eps, 1-eps)^2.

        Foundational contract: any copula density on [0,1]^d integrates to 1
        by definition. Carves a small margin at the corners because density
        can diverge on the boundary for non-trivial copulas.
        """
        d = 2
        params = _get_copula_params(copula, d)
        eps = 1e-6

        def _inner(u1, u0):
            u = jnp.array([[u0, u1]])
            return copula.copula_pdf(u=u, params=params).flatten()[0]

        def _outer(u0):
            val, _ = quadgk(lambda u1: _inner(u1, u0), interval=(eps, 1.0 - eps))
            return val.reshape(())

        result, _ = quadgk(_outer, interval=(eps, 1.0 - eps))
        np.testing.assert_allclose(
            float(result), 1.0, rtol=5e-2,
            err_msg=f"{copula.name} copula_pdf integrates to {float(result)}, not 1.0",
        )


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
            cx_logpdf[mask], expected_logpdf[mask], rtol=1e-4, atol=1e-12,
            err_msg="Gaussian copula logpdf doesn't match manual Sklar decomposition"
        )


# ---------------------------------------------------------------------------
# Student-t copula manual Sklar verification
# ---------------------------------------------------------------------------
#
# Replaces the v1.0.1 golden-fixture regression by verifying against an
# independent manual construction using scipy.stats.multivariate_t and
# scipy.stats.t. The Sklar decomposition for a Student-t copula is
#
#     c(u; ν, Σ) = f_d(t_ν⁻¹(u); ν, Σ) / Π_i f_1(t_ν⁻¹(u_i); ν)
#
# where f_d is the d-dimensional Student-t density and t_ν⁻¹ is the
# univariate Student-t quantile function (standard, scale=1). Σ is a
# correlation matrix so the marginals have unit scale.


def _build_student_t_copula_params(d: int, nu: float, sigma: np.ndarray) -> dict:
    """Construct a student_t_copula params dict with given (d, nu, sigma)."""
    return {
        "marginals": tuple(
            (student_t, {"nu": jnp.asarray(nu),
                         "mu": jnp.asarray(0.0),
                         "sigma": jnp.asarray(1.0)})
            for _ in range(d)
        ),
        "copula": {
            "nu": jnp.asarray(nu),
            "mu": jnp.asarray(np.zeros((d, 1))),
            "sigma": jnp.asarray(sigma),
        },
    }


class TestStudentTCopulaAgainstManual:
    """Verify Student-t copula density via manual Sklar decomposition.

    Two cases: identity correlation (matches the v1.0.1 example_params
    golden fixture) and a non-trivial correlation matrix (exercises
    the copula's dependence structure).
    """

    def test_logpdf_example_params_identity_sigma(self):
        """Matches v1.0.1 golden coverage: example_params has identity sigma."""
        d = 3
        params = _get_copula_params(student_t_copula, d)
        nu = float(params["copula"]["nu"])
        sigma = np.array(params["copula"]["sigma"])
        u = np.array(_uniform_sample(d, 20))

        x_dash = scipy.stats.t.ppf(u, df=nu)
        mv_t_logpdf = scipy.stats.multivariate_t(
            loc=np.zeros(d), shape=sigma, df=nu
        ).logpdf(x_dash)
        marginal_sum = np.sum(scipy.stats.t.logpdf(x_dash, df=nu), axis=1)
        expected = mv_t_logpdf - marginal_sum

        # brent=True routes through machine-epsilon Brent PPF, matching
        # scipy's internal inversion so the Sklar identity can be
        # checked at atol=1e-10 independent of cubic-spline tolerance.
        cx = np.array(student_t_copula.copula_logpdf(
            u=jnp.array(u), params=params, brent=True)).flatten()

        mask = np.isfinite(expected) & np.isfinite(cx)
        np.testing.assert_allclose(
            cx[mask], expected[mask], rtol=1e-4, atol=1e-10,
            err_msg="Student-t copula logpdf != manual Sklar (identity sigma)"
        )

    def test_logpdf_nontrivial_correlation(self):
        """Non-trivial correlation exercises the copula's dependence structure."""
        d = 3
        nu = 4.0
        sigma = np.array([[1.0, 0.5, 0.3],
                          [0.5, 1.0, 0.4],
                          [0.3, 0.4, 1.0]])
        params = _build_student_t_copula_params(d, nu, sigma)
        u = np.array(_uniform_sample(d, 20))

        x_dash = scipy.stats.t.ppf(u, df=nu)
        mv_t_logpdf = scipy.stats.multivariate_t(
            loc=np.zeros(d), shape=sigma, df=nu
        ).logpdf(x_dash)
        marginal_sum = np.sum(scipy.stats.t.logpdf(x_dash, df=nu), axis=1)
        expected = mv_t_logpdf - marginal_sum

        cx = np.array(student_t_copula.copula_logpdf(
            u=jnp.array(u), params=params, brent=True)).flatten()

        mask = np.isfinite(expected) & np.isfinite(cx)
        np.testing.assert_allclose(
            cx[mask], expected[mask], rtol=1e-4, atol=1e-10,
            err_msg="Student-t copula logpdf != manual Sklar (non-trivial sigma)"
        )

    def test_pdf_nontrivial_correlation(self):
        """pdf (not just logpdf) should also match."""
        d = 3
        nu = 4.0
        sigma = np.array([[1.0, 0.5, 0.3],
                          [0.5, 1.0, 0.4],
                          [0.3, 0.4, 1.0]])
        params = _build_student_t_copula_params(d, nu, sigma)
        u = np.array(_uniform_sample(d, 20))

        x_dash = scipy.stats.t.ppf(u, df=nu)
        mv_t_pdf = scipy.stats.multivariate_t(
            loc=np.zeros(d), shape=sigma, df=nu
        ).pdf(x_dash)
        marginal_prod = np.prod(scipy.stats.t.pdf(x_dash, df=nu), axis=1)
        expected = mv_t_pdf / marginal_prod

        cx = np.array(student_t_copula.copula_pdf(
            u=jnp.array(u), params=params)).flatten()

        mask = np.isfinite(expected) & np.isfinite(cx) & (expected > 0)
        np.testing.assert_allclose(
            cx[mask], expected[mask], rtol=1e-4, atol=1e-10,
            err_msg="Student-t copula pdf != manual Sklar (non-trivial sigma)"
        )


# ---------------------------------------------------------------------------
# Sampling with uniform marginals
# ---------------------------------------------------------------------------

class TestCopulaSamplingUniformMargins:
    """Copula samples should have U(0,1) marginals."""

    @pytest.mark.parametrize("copula", ALL_COPULAS_PARAMS)
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

    @pytest.mark.parametrize("copula", ALL_COPULAS_PARAMS)
    def test_fit_returns_valid_params(self, copula):
        """fit() should return valid parameters (no NaN, no inf)."""
        d = 3
        np.random.seed(42)
        # Generate correlated normal data
        sigma = np.array([[1.0, 0.5, 0.3],
                           [0.5, 1.0, 0.4],
                           [0.3, 0.4, 1.0]])
        data = np.random.multivariate_normal(np.zeros(d), sigma, size=200)

        # maxiter=30 bounds EM iteration budget for GH/SkewedT; Gaussian and
        # StudentT forward kwargs to fit_copula and converge well within 30.
        # Mirrors test_multivariate.py precedent for analogous GH/SkewedT fits.
        fitted = copula.fit(x=jnp.array(data), maxiter=30)
        assert fitted is not None, f"{copula.name} fit returned None"

        # Check copula params are valid
        copula_params = fitted.params.get("copula", {})
        if "sigma" in copula_params:
            s = np.array(copula_params["sigma"])
            assert no_nans(s), f"{copula.name} fitted sigma has NaNs"
            assert is_finite(s), f"{copula.name} fitted sigma not finite"

    @pytest.mark.parametrize("copula", ALL_COPULAS_PARAMS)
    def test_fit_is_jittable(self, copula):
        """Every copula fit_copula() (Stage 2, copula-only) must be JIT-compatible.

        The top-level ``copula.fit(x)`` is intentionally not JIT-able — it
        dispatches over a Python-level tuple of distribution objects during
        marginal fitting (via ``univariate_fitter``), which is not a tracing
        operation. The performance-critical path is ``fit_copula(u)`` on
        pre-computed pseudo-observations, and that is what this contract
        protects. A regression that silently pushes ``fit_copula`` out of
        JIT still returns correct results but runs 10–100× slower.
        """
        d = 3
        u = _uniform_sample(d=d, n=200, seed=7)
        fit_copula_jit = jax.jit(
            copula.fit_copula,
            static_argnames=("method", "corr_method", "brent"),
        )
        copula_params = fit_copula_jit(
            u, method="fc_mle", corr_method="rm_pp_kendall",
        )
        # `fit_copula` returns a params dict of the form {"copula": {...}}
        # or the copula-params dict directly, depending on the subclass.
        inner = copula_params.get("copula", copula_params)
        assert "sigma" in inner, (
            f"{copula.name}.fit_copula() did not return a 'sigma' entry; got "
            f"keys={list(inner.keys())}"
        )
        s = np.array(inner["sigma"])
        assert no_nans(s), f"{copula.name} JIT fit_copula sigma has NaNs"
        assert is_finite(s), f"{copula.name} JIT fit_copula sigma not finite"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestCopulaMetrics:
    """Verify loglikelihood, AIC, BIC are finite."""

    @pytest.mark.parametrize("copula", ALL_COPULAS_PARAMS)
    def test_metrics_finite(self, copula):
        d = 3
        np.random.seed(42)
        sigma = np.array([[1.0, 0.5, 0.3],
                           [0.5, 1.0, 0.4],
                           [0.3, 0.4, 1.0]])
        data = np.random.multivariate_normal(np.zeros(d), sigma, size=200)

        fitted = copula.fit(x=jnp.array(data), maxiter=30)
        logll = float(fitted.loglikelihood(x=jnp.array(data)))
        aic = float(fitted.aic(x=jnp.array(data)))
        bic = float(fitted.bic(x=jnp.array(data)))

        assert np.isfinite(logll), f"{copula.name} logll not finite"
        assert np.isfinite(aic), f"{copula.name} AIC not finite"
        assert np.isfinite(bic), f"{copula.name} BIC not finite"


# ---------------------------------------------------------------------------
# fit_copula(method=...) matrix — student_t_copula only
# ---------------------------------------------------------------------------

class TestStudentTCopulaFitMethods:
    """Verify fit_copula method dispatch for Student-T copula.

    Student-T copula only supports ``method='fc_mle'``. The mean-variance
    fitting methods (mle / ecme*) are only implemented for Skewed-T and
    GH copulas; verify ``fc_mle`` works and unsupported methods raise
    ``ValueError`` (no longer ``NotImplementedError`` — validation now
    happens at the dispatcher's ``_supported_methods`` gate).
    """

    @pytest.fixture
    def pseudo_obs(self):
        """Generate pseudo-observations from correlated normal data."""
        np.random.seed(42)
        sigma = np.array([[1.0, 0.5, 0.3],
                           [0.5, 1.0, 0.4],
                           [0.3, 0.4, 1.0]])
        data = np.random.multivariate_normal(np.zeros(3), sigma, size=200)
        # Convert to pseudo-observations via empirical CDF
        from scipy.stats import rankdata
        n = data.shape[0]
        u = np.column_stack([rankdata(data[:, j]) / (n + 1) for j in range(3)])
        return jnp.array(u)

    def test_fc_mle_produces_valid_params(self, pseudo_obs):
        """method='fc_mle' (default) returns finite params with valid nu."""
        result = student_t_copula.fit_copula(pseudo_obs, method="fc_mle")
        copula_params = result["copula"]
        assert "nu" in copula_params
        nu = float(copula_params["nu"])
        assert np.isfinite(nu), f"nu not finite: {nu}"
        assert nu > 2.0, f"nu should be > 2 for valid variance: {nu}"
        sigma = np.array(copula_params["sigma"])
        assert no_nans(sigma), "fitted sigma has NaNs"
        assert is_finite(sigma), "fitted sigma not finite"

    @pytest.mark.parametrize(
        "method",
        ["mle", "ecme", "ecme_double_gamma", "ecme_outer_gamma"],
    )
    def test_unsupported_methods_raise(self, pseudo_obs, method):
        """MV-only methods should raise ValueError for Student-T."""
        with pytest.raises(ValueError, match="not supported"):
            student_t_copula.fit_copula(pseudo_obs, method=method)


# ---------------------------------------------------------------------------
# fit_copula method/kwarg validation contract
# ---------------------------------------------------------------------------

class TestCopulaFitMethodValidation:
    """Verify the dispatcher's _supported_methods + _METHOD_KWARGS contract.

    Replaces the prior NotImplementedError-on-runtime behaviour with
    pre-dispatch ValueError so that:
      (1) the user gets a clear error message naming the supported set;
      (2) inapplicable kwargs are rejected instead of silently dropped;
      (3) the dispatcher is JIT-safe with method as a static argname.
    """

    @pytest.fixture
    def u(self):
        return _uniform_sample(d=3, n=200, seed=11)

    # ---- (1) unknown method ----

    @pytest.mark.parametrize("copula", [gaussian_copula, student_t_copula])
    def test_unknown_method_raises_with_supported_list(self, copula, u):
        with pytest.raises(ValueError, match="not supported"):
            copula.fit_copula(u, method="defin1tely_n0t_a_method")

        # The error message should name the actual supported set.
        try:
            copula.fit_copula(u, method="bogus")
        except ValueError as exc:
            msg = str(exc)
            for m in copula._supported_methods:
                assert repr(m) in msg or m in msg, (
                    f"{copula.name}: error message must list supported method "
                    f"{m!r}. Got: {msg!r}"
                )

    # ---- (2) inapplicable kwarg ----

    @pytest.mark.parametrize("copula", [gaussian_copula, student_t_copula])
    @pytest.mark.parametrize(
        "bad_kwargs",
        [
            {"tol": 1e-4},
            {"patience": 3},
            {"brent": True},
            {"nodes": 50},
            {"banana": "split"},
        ],
    )
    def test_fc_mle_rejects_inapplicable_kwargs(self, copula, u, bad_kwargs):
        with pytest.raises(ValueError, match="does not accept kwargs"):
            copula.fit_copula(u, method="fc_mle", **bad_kwargs)

    @pytest.mark.parametrize("copula", [gaussian_copula, student_t_copula])
    def test_fc_mle_accepts_lr_and_maxiter(self, copula, u):
        # These are the two kwargs in _METHOD_KWARGS["fc_mle"] — must
        # not raise.
        out = copula.fit_copula(u, method="fc_mle", lr=5e-3, maxiter=10)
        assert "copula" in out and "sigma" in out["copula"]

    # ---- (3) JIT-safety for every supported method ----

    @pytest.mark.parametrize("copula", [gaussian_copula, student_t_copula])
    def test_jit_compiles_for_every_supported_method(self, copula, u):
        for method in sorted(copula._supported_methods):
            fit_jit = jax.jit(
                copula.fit_copula,
                static_argnames=("method", "corr_method"),
            )
            params = fit_jit(u, method=method, corr_method="rm_pp_kendall")
            inner = params.get("copula", params)
            assert "sigma" in inner, (
                f"{copula.name}.fit_copula(method={method!r}) under JIT "
                f"returned unexpected keys {list(inner.keys())}"
            )
            sigma = np.asarray(inner["sigma"])
            assert no_nans(sigma), (
                f"{copula.name}/{method}: JIT fit produced NaN sigma"
            )
            assert is_finite(sigma), (
                f"{copula.name}/{method}: JIT fit produced non-finite sigma"
            )

    # ---- (4) supported_methods ground truth ----

    def test_elliptical_supported_methods(self):
        """Elliptical copulas only support fc_mle (no γ to fit)."""
        assert gaussian_copula._supported_methods == frozenset({"fc_mle"})
        assert student_t_copula._supported_methods == frozenset({"fc_mle"})

    def test_mean_variance_supported_methods(self):
        """MV copulas additionally support the four γ-aware methods."""
        expected = frozenset(
            {"fc_mle", "mle", "ecme", "ecme_double_gamma", "ecme_outer_gamma"}
        )
        assert gh_copula._supported_methods == expected
        assert skewed_t_copula._supported_methods == expected


# ---------------------------------------------------------------------------
# fit_marginals, fit_copula, get_u independently (gaussian + student_t)
# ---------------------------------------------------------------------------

class TestCopulaComponentMethods:
    """Test fit_marginals, fit_copula, and get_u independently."""

    @pytest.fixture
    def correlated_data(self):
        np.random.seed(42)
        sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
        return jnp.array(np.random.multivariate_normal(np.zeros(2), sigma, size=300))

    @pytest.mark.parametrize("copula", FAST_COPULAS, ids=FAST_IDS)
    def test_fit_marginals_produces_marginal_params(self, copula, correlated_data):
        """fit_marginals should produce marginal parameters for each dimension."""
        result = copula.fit_marginals(correlated_data)
        assert isinstance(result, dict), f"{copula.name}: fit_marginals should return dict"
        marginals = result.get("marginals", None)
        assert marginals is not None, f"{copula.name}: no marginals in result"
        assert len(marginals) == 2, f"{copula.name}: expected 2 marginals"

    @pytest.mark.parametrize("copula", FAST_COPULAS, ids=FAST_IDS)
    def test_get_u_returns_uniform_values(self, copula, correlated_data):
        """get_u should produce values in (0, 1) after fitting marginals."""
        fitted = copula.fit(x=correlated_data)
        u = np.array(fitted.get_u(x=correlated_data))
        assert u.shape == (300, 2), f"get_u shape: {u.shape}"
        assert np.all(u > 0) and np.all(u < 1), \
            f"{copula.name}: get_u values outside (0,1)"

    @pytest.mark.parametrize("copula", FAST_COPULAS, ids=FAST_IDS)
    def test_fit_copula_from_pseudo_obs(self, copula, correlated_data):
        """fit_copula on pseudo-observations should return copula params."""
        from scipy.stats import rankdata
        n = correlated_data.shape[0]
        data_np = np.array(correlated_data)
        u = jnp.array(np.column_stack(
            [rankdata(data_np[:, j]) / (n + 1) for j in range(2)]
        ))
        result = copula.fit_copula(u)
        assert "copula" in result
        sigma = np.array(result["copula"]["sigma"])
        assert no_nans(sigma), f"{copula.name} fit_copula sigma has NaNs"
        assert is_finite(sigma), f"{copula.name} fit_copula sigma not finite"
