"""Tests for Archimedean copula distributions.

Covers:
    - Structural properties (type, methods, params)
    - Boundary conditions and Fréchet bounds
    - Concordance ordering and Kendall tau recovery
    - Generator roundtrip consistency
    - Copula density properties + JIT + gradients (consolidated)
    - RVS marginal uniformity
    - Metrics (AIC, BIC, loglikelihood)
"""

import pytest
import jax
import jax.numpy as jnp
from jax import jit, grad
import numpy as np

from copulax.copulas import (
    clayton_copula,
    frank_copula,
    gumbel_copula,
    joe_copula,
    amh_copula,
    independence_copula,
)
from copulax._src.copulas._archimedean import ArchimedeanCopula
from copulax.tests.helpers import no_nans, is_finite, is_positive, gradients


# ──────────────────────────────────────────────────────────────────────
# Constants & helpers
# ──────────────────────────────────────────────────────────────────────
NUM_SAMPLES = 200
NUM_ASSETS_3D = 3
NUM_ASSETS_2D = 2
EPS = 1e-3

DISTS_3D = [
    pytest.param(clayton_copula, id="Clayton"),
    pytest.param(frank_copula, id="Frank"),
    pytest.param(gumbel_copula, id="Gumbel"),
    pytest.param(joe_copula, id="Joe"),
    pytest.param(independence_copula, id="Independence"),
]
DISTS_2D = [pytest.param(amh_copula, id="AMH")]
ALL_DISTS = DISTS_3D + DISTS_2D

# Copulas that have a theta parameter (excludes independence copula)
THETA_DISTS_3D = [
    pytest.param(clayton_copula, id="Clayton"),
    pytest.param(frank_copula, id="Frank"),
    pytest.param(gumbel_copula, id="Gumbel"),
    pytest.param(joe_copula, id="Joe"),
]
THETA_DISTS = THETA_DISTS_3D + DISTS_2D


def _get_dim(dist):
    return NUM_ASSETS_2D if dist is amh_copula else NUM_ASSETS_3D


def _uniform_data(dim, n=NUM_SAMPLES, seed=42):
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, shape=(n, dim), minval=EPS, maxval=1 - EPS)


def _get_params(dist):
    return dist.example_params(dim=_get_dim(dist))


# ──────────────────────────────────────────────────────────────────────
# Structural tests
# ──────────────────────────────────────────────────────────────────────
class TestStructure:
    """Structural tests for Archimedean copula objects."""

    @pytest.mark.parametrize("dist", ALL_DISTS)
    def test_is_archimedean(self, dist):
        assert isinstance(dist, ArchimedeanCopula)

    @pytest.mark.parametrize("dist", ALL_DISTS)
    def test_metadata(self, dist):
        assert dist.dist_type == "copula"
        assert dist.dtype == "continuous"
        assert isinstance(dist.name, str) and len(dist.name) > 0
        assert dist.name == str(dist)

    @pytest.mark.parametrize("dist", THETA_DISTS)
    def test_example_params(self, dist):
        dim = _get_dim(dist)
        params = dist.example_params(dim=dim)
        assert "marginals" in params and "copula" in params
        assert "theta" in params["copula"]
        assert len(params["marginals"]) == dim

    def test_independence_example_params(self):
        params = independence_copula.example_params(dim=3)
        assert "marginals" in params and "copula" in params
        assert params["copula"] == {}
        assert len(params["marginals"]) == 3

    @pytest.mark.parametrize("dist", ALL_DISTS)
    def test_methods_exist(self, dist):
        required = [
            "generator",
            "generator_inv",
            "copula_cdf",
            "copula_logpdf",
            "copula_pdf",
            "copula_rvs",
            "copula_sample",
            "logpdf",
            "pdf",
            "rvs",
            "sample",
            "fit_marginals",
            "fit_copula",
            "fit",
            "aic",
            "bic",
            "loglikelihood",
            "support",
            "example_params",
        ]
        for method in required:
            assert hasattr(dist, method), f"{dist} missing: {method}"

    @pytest.mark.parametrize("dist", ALL_DISTS)
    def test_jittable(self, dist):
        from copulax.tests.helpers import jittable

        jittable(dist)

    def test_amh_dimension_restriction(self):
        with pytest.raises(ValueError, match="d=2"):
            amh_copula.example_params(dim=3)


# ──────────────────────────────────────────────────────────────────────
# Generator tests
# ──────────────────────────────────────────────────────────────────────
class TestGenerator:
    """Tests for generator/inverse generator properties."""

    @pytest.mark.parametrize("dist", THETA_DISTS)
    def test_roundtrip(self, dist):
        """ψ(φ(t)) = t for t in (0, 1)."""
        theta = _get_params(dist)["copula"]["theta"]
        ts = jnp.linspace(0.05, 0.95, 20)
        for t in ts:
            phi_t = dist.generator(t, theta)
            recovered = dist.generator_inv(phi_t, theta)
            assert jnp.allclose(
                recovered, t, atol=1e-5
            ), f"{dist} ψ(φ({t})) = {recovered}"

    def test_independence_roundtrip(self):
        """ψ(φ(t)) = t for independence copula (theta ignored)."""
        theta = 1.0  # dummy, ignored by independence copula
        ts = jnp.linspace(0.05, 0.95, 20)
        for t in ts:
            phi_t = independence_copula.generator(t, theta)
            recovered = independence_copula.generator_inv(phi_t, theta)
            assert jnp.allclose(recovered, t, atol=1e-5)

    @pytest.mark.parametrize("dist", THETA_DISTS)
    def test_boundary(self, dist):
        """φ(1) = 0."""
        theta = _get_params(dist)["copula"]["theta"]
        assert jnp.allclose(dist.generator(1.0, theta), 0.0, atol=1e-6)

    def test_independence_boundary(self):
        """φ(1) = 0 for independence copula."""
        assert jnp.allclose(independence_copula.generator(1.0, 1.0), 0.0, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────
# Boundary conditions & Fréchet bounds
# ──────────────────────────────────────────────────────────────────────
class TestBoundaryAndBounds:
    """Boundary conditions and Fréchet bounds."""

    @pytest.mark.parametrize("dist", ALL_DISTS)
    def test_boundary_zero(self, dist):
        """C(u) ≈ 0 when any u_i ≈ 0."""
        dim = _get_dim(dist)
        params = _get_params(dist)
        for j in range(dim):
            u = jnp.full((5, dim), 0.5)
            u = u.at[:, j].set(EPS * 1e-3)
            cdf = dist.copula_cdf(u, params)
            assert jnp.all(cdf < 0.05)

    @pytest.mark.parametrize("dist", DISTS_3D)
    def test_boundary_one_reduces(self, dist):
        """Setting u_i = 1 in 3D ≈ 2D copula."""
        params = dist.example_params(dim=3)
        key = jax.random.PRNGKey(0)
        u_2d = jax.random.uniform(key, shape=(10, 2), minval=0.1, maxval=0.9)
        u_3d = jnp.concatenate([u_2d, jnp.ones((10, 1))], axis=1)
        cdf_3d = dist.copula_cdf(u_3d, params)
        params_2d = dist.example_params(dim=2)
        cdf_2d = dist.copula_cdf(u_2d, params_2d)
        assert jnp.allclose(cdf_3d, cdf_2d, atol=1e-3)

    @pytest.mark.parametrize("dist", ALL_DISTS)
    def test_frechet_upper(self, dist):
        """C(u) ≤ min(u_i)."""
        dim = _get_dim(dist)
        params = _get_params(dist)
        u = _uniform_data(dim, n=50)
        cdf = dist.copula_cdf(u, params)
        upper = jnp.min(u, axis=1, keepdims=True)
        assert jnp.all(cdf <= upper + 1e-4)

    @pytest.mark.parametrize("dist", ALL_DISTS)
    def test_frechet_lower(self, dist):
        """C(u) ≥ max(Σu_i - (d-1), 0)."""
        dim = _get_dim(dist)
        params = _get_params(dist)
        u = _uniform_data(dim, n=50)
        cdf = dist.copula_cdf(u, params)
        lower = jnp.maximum(jnp.sum(u, axis=1, keepdims=True) - (dim - 1), 0.0)
        assert jnp.all(cdf >= lower - 1e-4)

    @pytest.mark.parametrize("dist", ALL_DISTS)
    def test_cdf_in_unit_interval(self, dist):
        dim = _get_dim(dist)
        params = _get_params(dist)
        u = _uniform_data(dim, n=50)
        cdf = dist.copula_cdf(u, params)
        assert jnp.all(cdf >= -1e-6) and jnp.all(cdf <= 1 + 1e-6)


# ──────────────────────────────────────────────────────────────────────
# Concordance ordering & Kendall tau
# ──────────────────────────────────────────────────────────────────────
class TestDependence:
    """Concordance ordering and Kendall tau recovery."""

    @pytest.mark.parametrize("dist", [clayton_copula, gumbel_copula, joe_copula])
    def test_concordance_ordering(self, dist):
        """Higher θ → higher C(u)."""
        u = _uniform_data(3, n=20, seed=99)
        params_lo = dist.example_params(dim=3)
        params_hi = {
            "marginals": params_lo["marginals"],
            "copula": {"theta": params_lo["copula"]["theta"] * 2.0},
        }
        cdf_lo = dist.copula_cdf(u, params_lo)
        cdf_hi = dist.copula_cdf(u, params_hi)
        assert jnp.all(cdf_hi >= cdf_lo - 1e-4)

    @pytest.mark.parametrize("dist", THETA_DISTS)
    def test_kendall_tau_recovery(self, dist):
        """Fit recovers θ approximately."""
        dim = _get_dim(dist)
        params = _get_params(dist)
        rvs = dist.copula_rvs(size=500, params=params)
        fitted = dist.fit_copula(rvs)
        fitted_theta = float(fitted["copula"]["theta"])
        true_theta = float(params["copula"]["theta"])
        assert np.isfinite(fitted_theta)
        if abs(true_theta) > 0.1:
            assert (
                np.sign(fitted_theta) == np.sign(true_theta) or abs(fitted_theta) < 0.5
            )

    def test_independence_fit_copula(self):
        """Independence copula fit returns empty copula dict."""
        u = _uniform_data(3, n=50)
        fitted = independence_copula.fit_copula(u)
        assert fitted == {"copula": {}}


# ──────────────────────────────────────────────────────────────────────
# Copula density: properties + JIT + gradients (consolidated)
# ──────────────────────────────────────────────────────────────────────
class TestCopulaDensity:
    """Copula density properties, JIT, and gradients — consolidated.

    Each method (copula_cdf, copula_logpdf, copula_pdf) is tested once
    for properties, JIT, and gradients rather than in separate tests.
    """

    @pytest.mark.parametrize("dist", ALL_DISTS)
    def test_copula_cdf(self, dist):
        dim = _get_dim(dist)
        params = _get_params(dist)
        u = _uniform_data(dim, n=5)

        result = jit(dist.copula_cdf)(u, params)
        assert no_nans(result)

        func = lambda u, params: dist.copula_cdf(u, params=params).sum()
        u_grad = grad(func, argnums=0)(u, params)
        assert no_nans(u_grad) and is_finite(u_grad)

    @pytest.mark.parametrize("dist", ALL_DISTS)
    def test_copula_logpdf(self, dist):
        dim = _get_dim(dist)
        params = _get_params(dist)
        u = _uniform_data(dim, n=5)

        result = jit(dist.copula_logpdf)(u, params)
        assert no_nans(result) and is_finite(result)
        gradients(dist.copula_logpdf, f"{dist} copula_logpdf", u, params)

    @pytest.mark.parametrize("dist", ALL_DISTS)
    def test_copula_pdf(self, dist):
        dim = _get_dim(dist)
        params = _get_params(dist)
        u = _uniform_data(dim, n=5)

        result = jit(dist.copula_pdf)(u, params)
        assert is_positive(result)
        assert no_nans(result) and is_finite(result)
        gradients(dist.copula_pdf, f"{dist} copula_pdf", u, params)

    @pytest.mark.parametrize("dist", ALL_DISTS)
    def test_pdf_logpdf_consistency(self, dist):
        """exp(logpdf) should equal pdf."""
        dim = _get_dim(dist)
        params = _get_params(dist)
        u = _uniform_data(dim, n=10)
        logpdf = dist.copula_logpdf(u, params)
        pdf = dist.copula_pdf(u, params)
        assert jnp.allclose(jnp.exp(logpdf), pdf, rtol=1e-4)


# ──────────────────────────────────────────────────────────────────────
# RVS
# ──────────────────────────────────────────────────────────────────────
class TestRVS:
    """Random variate sampling tests."""

    @pytest.mark.parametrize("dist", ALL_DISTS)
    def test_marginal_uniformity(self, dist):
        """Each marginal of copula_rvs ≈ U(0,1)."""
        dim = _get_dim(dist)
        params = _get_params(dist)
        n = 1000
        rvs = dist.copula_rvs(size=n, params=params)

        assert rvs.shape == (n, dim)
        assert jnp.all(rvs > 0) and jnp.all(rvs < 1)
        assert no_nans(rvs) and is_finite(rvs)

        for j in range(dim):
            sorted_u = jnp.sort(rvs[:, j])
            ecdf = jnp.arange(1, n + 1) / n
            ks_stat = float(jnp.max(jnp.abs(sorted_u - ecdf)))
            assert ks_stat < 0.15, f"{dist} marginal {j} KS = {ks_stat:.4f}"

    @pytest.mark.parametrize("dist", ALL_DISTS)
    @pytest.mark.parametrize("size", [0, 1, 5, 50])
    def test_rvs_sizes(self, dist, size):
        dim = _get_dim(dist)
        params = _get_params(dist)
        rvs = dist.copula_rvs(size=size, params=params)
        assert rvs.shape == (size, dim)

    @pytest.mark.parametrize("dist", ALL_DISTS)
    def test_jit_copula_rvs(self, dist):
        dim = _get_dim(dist)
        params = _get_params(dist)
        result = jit(dist.copula_rvs, static_argnums=0)(10, params)
        assert result.shape == (10, dim) and no_nans(result)

    @pytest.mark.parametrize("dist", THETA_DISTS)
    def test_jit_fit_copula(self, dist):
        dim = _get_dim(dist)
        u = _uniform_data(dim, n=50)
        fitted = jit(dist.fit_copula)(u)
        assert "copula" in fitted and "theta" in fitted["copula"]
        assert jnp.isfinite(fitted["copula"]["theta"])

    def test_jit_fit_copula_independence(self):
        u = _uniform_data(3, n=50)
        fitted = jit(independence_copula.fit_copula)(u)
        assert fitted == {"copula": {}}


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────
class TestMetrics:
    """Tests for loglikelihood, AIC, BIC."""

    @pytest.mark.parametrize("dist", ALL_DISTS)
    @pytest.mark.parametrize("metric", ["loglikelihood", "aic", "bic"])
    def test_metric_finite(self, dist, metric):
        dim = _get_dim(dist)
        x = np.random.normal(size=(50, dim))
        params = dist.example_params(dim=dim)
        result = getattr(dist, metric)(x, params)
        assert np.isfinite(float(result)), f"{dist} {metric} not finite"


# ──────────────────────────────────────────────────────────────────────
# Independence copula specific tests
# ──────────────────────────────────────────────────────────────────────
class TestIndependenceCopula:
    """Tests specific to the independence copula."""

    def test_copula_logpdf_is_zero(self):
        """Independence copula log-density is always 0."""
        u = _uniform_data(3, n=20)
        params = independence_copula.example_params(dim=3)
        logpdf = independence_copula.copula_logpdf(u, params)
        assert jnp.allclose(logpdf, 0.0, atol=1e-8)

    def test_copula_pdf_is_one(self):
        """Independence copula density is always 1."""
        u = _uniform_data(3, n=20)
        params = independence_copula.example_params(dim=3)
        pdf = independence_copula.copula_pdf(u, params)
        assert jnp.allclose(pdf, 1.0, atol=1e-6)

    def test_copula_cdf_is_product(self):
        """Independence copula CDF = product of marginals."""
        u = _uniform_data(3, n=20)
        params = independence_copula.example_params(dim=3)
        cdf = independence_copula.copula_cdf(u, params)
        expected = jnp.prod(u, axis=1, keepdims=True)
        assert jnp.allclose(cdf, expected, atol=1e-6)

    def test_rvs_independent_margins(self):
        """RVS margins are uncorrelated (Kendall tau ≈ 0)."""
        params = independence_copula.example_params(dim=3)
        rvs = independence_copula.copula_rvs(size=1000, params=params)

        # Check pairwise Kendall's tau is near zero
        from copulax._src.multivariate._shape import corr

        tau_matrix = corr(rvs, method="kendall")
        off_diag = tau_matrix - jnp.eye(3)
        assert jnp.all(
            jnp.abs(off_diag) < 0.1
        ), f"Independence copula RVS has unexpected correlation: {tau_matrix}"

    def test_any_dimension(self):
        """Independence copula works for any dimension."""
        for dim in [2, 3, 5, 10]:
            params = independence_copula.example_params(dim=dim)
            u = _uniform_data(dim, n=5)
            cdf = independence_copula.copula_cdf(u, params)
            assert cdf.shape == (5, 1)
            assert jnp.allclose(cdf, jnp.prod(u, axis=1, keepdims=True), atol=1e-6)

    def test_aic_bic_zero_params(self):
        """AIC and BIC with k=0 free parameters."""
        params = independence_copula.example_params(dim=3)
        x = np.random.normal(size=(50, 3))
        aic = independence_copula.aic(x, params)
        bic = independence_copula.bic(x, params)
        ll = independence_copula.loglikelihood(x, params)
        assert np.isfinite(float(aic))
        assert np.isfinite(float(bic))
        # With k=0: AIC = BIC = -2·loglikelihood
        assert jnp.allclose(aic, -2.0 * ll, atol=1e-5)
        assert jnp.allclose(bic, -2.0 * ll, atol=1e-5)
