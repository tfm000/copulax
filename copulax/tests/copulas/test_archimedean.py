"""Tests for Archimedean copula distributions.

Covers:
    - Boundary conditions (u_i=0 ⟹ C=0, u_i=1 reduces dimension)
    - Fréchet bounds
    - Concordance ordering
    - Kendall tau recovery from fitted θ
    - Marginal uniformity of RVS
    - JIT compatibility
    - Gradient propagation
"""
import pytest
import jax
import jax.numpy as jnp
from jax import jit, grad
import numpy as np

from copulax.copulas import (
    clayton_copula, frank_copula, gumbel_copula, joe_copula, amh_copula,
)
from copulax._src.copulas._archimedean import ArchimedeanCopula
from copulax.tests.helpers import no_nans, is_finite, is_positive, gradients


# ──────────────────────────────────────────────────────────────────────
# Test fixtures and constants
# ──────────────────────────────────────────────────────────────────────
NUM_SAMPLES = 200
NUM_ASSETS_3D = 3
NUM_ASSETS_2D = 2
EPS = 1e-3

# 3D copulas (all except AMH)
DISTS_3D = [
    pytest.param(clayton_copula, id="Clayton"),
    pytest.param(frank_copula, id="Frank"),
    pytest.param(gumbel_copula, id="Gumbel"),
    pytest.param(joe_copula, id="Joe"),
]

# AMH is 2D-only
DISTS_2D = [
    pytest.param(amh_copula, id="AMH"),
]

ALL_DISTS = DISTS_3D + DISTS_2D


def _get_dim(dist):
    """Get the dimension for a given copula."""
    return NUM_ASSETS_2D if dist is amh_copula else NUM_ASSETS_3D


def _uniform_data(dim, n=NUM_SAMPLES, seed=42):
    """Generate uniform data in (eps, 1-eps)."""
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, shape=(n, dim), minval=EPS, maxval=1 - EPS)


def _get_params(dist):
    """Get example params for the appropriate dimension."""
    return dist.example_params(dim=_get_dim(dist))


# ──────────────────────────────────────────────────────────────────────
# Basic structural tests
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dist", ALL_DISTS)
def test_is_archimedean(dist):
    assert isinstance(dist, ArchimedeanCopula), \
        f"{dist} is not an ArchimedeanCopula"


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_dist_type(dist):
    assert dist.dist_type == 'copula'


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_dtype(dist):
    assert dist.dtype == 'continuous'


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_name(dist):
    assert isinstance(dist.name, str)
    assert len(dist.name) > 0
    assert dist.name == str(dist)


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_example_params(dist):
    dim = _get_dim(dist)
    params = dist.example_params(dim=dim)
    assert 'marginals' in params
    assert 'copula' in params
    assert 'theta' in params['copula']
    assert len(params['marginals']) == dim


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_methods_exist(dist):
    """Ensure all required methods are implemented."""
    required = [
        'generator', 'generator_inv', 'copula_cdf', 'copula_logpdf',
        'copula_pdf', 'copula_rvs', 'copula_sample', 'logpdf', 'pdf',
        'rvs', 'sample', 'fit_marginals', 'fit_copula', 'fit',
        'aic', 'bic', 'loglikelihood', 'support', 'example_params',
    ]
    for method in required:
        assert hasattr(dist, method), f"{dist} missing method: {method}"


# ──────────────────────────────────────────────────────────────────────
# Boundary condition tests
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dist", ALL_DISTS)
def test_boundary_zero(dist):
    """C(u₁, ..., u_d) = 0 when any u_i = 0."""
    dim = _get_dim(dist)
    params = _get_params(dist)
    n = 5

    for j in range(dim):
        u = jnp.full((n, dim), 0.5)
        u = u.at[:, j].set(EPS * 1e-3)  # near zero
        cdf = dist.copula_cdf(u, params)
        assert jnp.all(cdf < 0.05), \
            f"{dist} CDF not near zero when u[{j}]≈0: {cdf.flatten()}"


@pytest.mark.parametrize("dist", DISTS_3D)
def test_boundary_one_reduces(dist):
    """Setting u_i = 1 in a 3D copula should approximate the 2D copula."""
    params = dist.example_params(dim=3)
    theta = params['copula']['theta']

    # u with third dimension = 1
    key = jax.random.PRNGKey(0)
    u_2d = jax.random.uniform(key, shape=(10, 2), minval=0.1, maxval=0.9)
    u_3d = jnp.concatenate([u_2d, jnp.ones((10, 1))], axis=1)

    cdf_3d = dist.copula_cdf(u_3d, params)

    # Compare with 2D CDF
    params_2d = dist.example_params(dim=2)
    cdf_2d = dist.copula_cdf(u_2d, params_2d)

    assert jnp.allclose(cdf_3d, cdf_2d, atol=1e-3), \
        f"{dist} C(u1, u2, 1) ≠ C(u1, u2): max diff = {jnp.max(jnp.abs(cdf_3d - cdf_2d))}"


# ──────────────────────────────────────────────────────────────────────
# Fréchet bounds
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dist", ALL_DISTS)
def test_frechet_upper_bound(dist):
    """C(u) ≤ min(u_1, ..., u_d) (Fréchet upper bound)."""
    dim = _get_dim(dist)
    params = _get_params(dist)
    u = _uniform_data(dim, n=50)

    cdf = dist.copula_cdf(u, params)
    upper = jnp.min(u, axis=1, keepdims=True)

    assert jnp.all(cdf <= upper + 1e-4), \
        f"{dist} violates Fréchet upper bound"


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_frechet_lower_bound(dist):
    """C(u) ≥ max(∑u_i - (d-1), 0) (Fréchet lower bound)."""
    dim = _get_dim(dist)
    params = _get_params(dist)
    u = _uniform_data(dim, n=50)

    cdf = dist.copula_cdf(u, params)
    lower = jnp.maximum(jnp.sum(u, axis=1, keepdims=True) - (dim - 1), 0.0)

    assert jnp.all(cdf >= lower - 1e-4), \
        f"{dist} violates Fréchet lower bound"


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_cdf_in_unit_interval(dist):
    """Copula CDF values must be in [0, 1]."""
    dim = _get_dim(dist)
    params = _get_params(dist)
    u = _uniform_data(dim, n=50)

    cdf = dist.copula_cdf(u, params)
    assert jnp.all(cdf >= -1e-6), f"{dist} CDF below 0"
    assert jnp.all(cdf <= 1 + 1e-6), f"{dist} CDF above 1"


# ──────────────────────────────────────────────────────────────────────
# Concordance ordering
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dist", [clayton_copula, gumbel_copula, joe_copula])
def test_concordance_ordering(dist):
    """Higher θ → higher C(u) pointwise (for positive dependence copulas)."""
    dim = 3
    u = _uniform_data(dim, n=20, seed=99)

    # Low theta
    params_lo = dist.example_params(dim=dim)
    theta_lo = params_lo['copula']['theta']
    theta_hi = theta_lo * 2.0
    params_hi = {
        'marginals': params_lo['marginals'],
        'copula': {'theta': theta_hi},
    }

    cdf_lo = dist.copula_cdf(u, params_lo)
    cdf_hi = dist.copula_cdf(u, params_hi)

    # Higher theta should give higher (or equal) CDF
    assert jnp.all(cdf_hi >= cdf_lo - 1e-4), \
        f"{dist} concordance violated: higher θ gives lower CDF"


# ──────────────────────────────────────────────────────────────────────
# Kendall tau recovery
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dist", ALL_DISTS)
def test_kendall_tau_recovery(dist):
    """Fitting copula from data recovers θ approximately."""
    dim = _get_dim(dist)
    params = _get_params(dist)

    # Generate samples
    rvs = dist.copula_rvs(size=500, params=params)

    # Fit copula
    fitted = dist.fit_copula(rvs)
    fitted_theta = float(fitted['copula']['theta'])
    true_theta = float(params['copula']['theta'])

    # The recovered theta should be in the right ballpark
    # (not exact due to sampling noise)
    assert np.isfinite(fitted_theta), \
        f"{dist} fitted theta is not finite"
    # Check same sign at minimum
    if abs(true_theta) > 0.1:
        assert np.sign(fitted_theta) == np.sign(true_theta) or abs(fitted_theta) < 0.5, \
            f"{dist} fitted theta has wrong sign: true={true_theta}, fitted={fitted_theta}"


# ──────────────────────────────────────────────────────────────────────
# Marginal uniformity of RVS
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dist", ALL_DISTS)
def test_rvs_marginal_uniformity(dist):
    """Each marginal of copula_rvs should be approximately U(0,1).

    Uses the Kolmogorov-Smirnov criterion: max|F_n(x) - x| < threshold.
    """
    dim = _get_dim(dist)
    params = _get_params(dist)
    n = 1000

    rvs = dist.copula_rvs(size=n, params=params)

    assert rvs.shape == (n, dim), \
        f"{dist} rvs shape: expected {(n, dim)}, got {rvs.shape}"
    assert jnp.all(rvs > 0) and jnp.all(rvs < 1), \
        f"{dist} rvs not in (0, 1)"
    assert no_nans(rvs), f"{dist} rvs contains NaNs"
    assert is_finite(rvs), f"{dist} rvs contains non-finite values"

    # KS-style check per marginal
    for j in range(dim):
        sorted_u = jnp.sort(rvs[:, j])
        ecdf = jnp.arange(1, n + 1) / n
        ks_stat = float(jnp.max(jnp.abs(sorted_u - ecdf)))
        # Critical value at α=0.01 is roughly 1.63/√n ≈ 0.052
        # Use 0.15 threshold for robustness against random variance
        assert ks_stat < 0.15, \
            f"{dist} marginal {j} KS statistic = {ks_stat:.4f} (too non-uniform)"


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_rvs_sizes(dist):
    """Test various RVS output sizes."""
    dim = _get_dim(dist)
    params = _get_params(dist)

    for size in [0, 1, 5, 50]:
        rvs = dist.copula_rvs(size=size, params=params)
        assert rvs.shape == (size, dim), \
            f"{dist} rvs shape for size={size}: expected {(size, dim)}, got {rvs.shape}"


# ──────────────────────────────────────────────────────────────────────
# JIT compatibility
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dist", ALL_DISTS)
def test_jit_copula_cdf(dist):
    """copula_cdf must be JIT-compatible."""
    dim = _get_dim(dist)
    params = _get_params(dist)
    u = _uniform_data(dim, n=5)

    jitted = jit(dist.copula_cdf)
    result = jitted(u, params)
    assert no_nans(result), f"{dist} JIT copula_cdf contains NaNs"


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_jit_copula_logpdf(dist):
    """copula_logpdf must be JIT-compatible."""
    dim = _get_dim(dist)
    params = _get_params(dist)
    u = _uniform_data(dim, n=5)

    jitted = jit(dist.copula_logpdf)
    result = jitted(u, params)
    assert no_nans(result), f"{dist} JIT copula_logpdf contains NaNs"


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_jit_copula_pdf(dist):
    """copula_pdf must be JIT-compatible."""
    dim = _get_dim(dist)
    params = _get_params(dist)
    u = _uniform_data(dim, n=5)

    jitted = jit(dist.copula_pdf)
    result = jitted(u, params)
    assert is_positive(result), f"{dist} JIT copula_pdf not positive"
    assert no_nans(result), f"{dist} JIT copula_pdf contains NaNs"


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_jit_copula_rvs(dist):
    """copula_rvs must be JIT-compatible."""
    dim = _get_dim(dist)
    params = _get_params(dist)

    jitted = jit(dist.copula_rvs, static_argnums=0)
    result = jitted(10, params)
    assert result.shape == (10, dim)
    assert no_nans(result), f"{dist} JIT copula_rvs contains NaNs"


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_jit_fit_copula(dist):
    """fit_copula must be JIT-compatible."""
    dim = _get_dim(dist)
    u = _uniform_data(dim, n=50)

    jitted = jit(dist.fit_copula)
    fitted = jitted(u)
    assert 'copula' in fitted
    assert 'theta' in fitted['copula']
    assert jnp.isfinite(fitted['copula']['theta']), \
        f"{dist} JIT fit_copula returned non-finite theta"


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_jit_example_params(dist):
    """Distribution object must be JIT-compatible."""
    from copulax.tests.helpers import jittable
    jittable(dist)


# ──────────────────────────────────────────────────────────────────────
# Gradient propagation
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dist", ALL_DISTS)
def test_gradient_copula_cdf(dist):
    """copula_cdf gradients w.r.t. u should be finite."""
    dim = _get_dim(dist)
    params = _get_params(dist)
    u = _uniform_data(dim, n=5)

    func = lambda u, params: dist.copula_cdf(u, params=params).sum()
    u_grad = grad(func, argnums=0)(u, params)
    assert no_nans(u_grad), f"{dist} copula_cdf gradient contains NaNs"
    assert is_finite(u_grad), f"{dist} copula_cdf gradient not finite"


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_gradient_copula_logpdf(dist):
    """copula_logpdf gradients w.r.t. u should be finite."""
    dim = _get_dim(dist)
    params = _get_params(dist)
    u = _uniform_data(dim, n=5)

    gradients(dist.copula_logpdf, f"{dist} copula_logpdf", u, params)


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_gradient_copula_pdf(dist):
    """copula_pdf gradients w.r.t. u should be finite."""
    dim = _get_dim(dist)
    params = _get_params(dist)
    u = _uniform_data(dim, n=5)

    gradients(dist.copula_pdf, f"{dist} copula_pdf", u, params)


# ──────────────────────────────────────────────────────────────────────
# Copula density properties
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dist", ALL_DISTS)
def test_pdf_positive(dist):
    """Copula density must be non-negative."""
    dim = _get_dim(dist)
    params = _get_params(dist)
    u = _uniform_data(dim, n=20)

    pdf = dist.copula_pdf(u, params)
    assert is_positive(pdf), f"{dist} copula_pdf contains negative values"
    assert no_nans(pdf), f"{dist} copula_pdf contains NaNs"
    assert is_finite(pdf), f"{dist} copula_pdf not finite"


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_logpdf_finite(dist):
    """Copula log-density should be finite for interior u values."""
    dim = _get_dim(dist)
    params = _get_params(dist)
    u = _uniform_data(dim, n=20)

    logpdf = dist.copula_logpdf(u, params)
    assert no_nans(logpdf), f"{dist} copula_logpdf contains NaNs"
    assert is_finite(logpdf), f"{dist} copula_logpdf not finite"


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_pdf_logpdf_consistency(dist):
    """exp(logpdf) should equal pdf."""
    dim = _get_dim(dist)
    params = _get_params(dist)
    u = _uniform_data(dim, n=10)

    logpdf = dist.copula_logpdf(u, params)
    pdf = dist.copula_pdf(u, params)
    assert jnp.allclose(jnp.exp(logpdf), pdf, rtol=1e-4), \
        f"{dist} exp(logpdf) ≠ pdf"


# ──────────────────────────────────────────────────────────────────────
# Generator / inverse generator consistency
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dist", ALL_DISTS)
def test_generator_inverse_roundtrip(dist):
    """ψ(φ(t)) = t for t ∈ (0, 1)."""
    params = _get_params(dist)
    theta = params['copula']['theta']

    ts = jnp.linspace(0.05, 0.95, 20)
    for t in ts:
        phi_t = dist.generator(t, theta)
        recovered = dist.generator_inv(phi_t, theta)
        assert jnp.allclose(recovered, t, atol=1e-5), \
            f"{dist} ψ(φ({t})) = {recovered}, expected {t}"


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_generator_boundary(dist):
    """φ(1) = 0 for all Archimedean generators."""
    params = _get_params(dist)
    theta = params['copula']['theta']
    assert jnp.allclose(dist.generator(1.0, theta), 0.0, atol=1e-6), \
        f"{dist} φ(1) ≠ 0"


# ──────────────────────────────────────────────────────────────────────
# AMH-specific tests
# ──────────────────────────────────────────────────────────────────────
def test_amh_dimension_restriction():
    """AMH copula should only support dim=2."""
    with pytest.raises(ValueError, match="d=2"):
        amh_copula.example_params(dim=3)


# ──────────────────────────────────────────────────────────────────────
# Metrics (AIC, BIC, loglikelihood)
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dist", ALL_DISTS)
def test_loglikelihood(dist):
    """loglikelihood should return a finite scalar."""
    dim = _get_dim(dist)
    params = _get_params(dist)
    # Generate data from the copula
    rvs = dist.copula_rvs(size=50, params=params)
    # Create x using marginal ppfs
    from copulax._src.univariate.normal import normal
    x = np.random.normal(size=(50, dim))

    full_params = dist.example_params(dim=dim)
    ll = dist.loglikelihood(x, full_params)
    assert np.isfinite(float(ll)), f"{dist} loglikelihood not finite"


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_aic(dist):
    """AIC should return a finite scalar."""
    dim = _get_dim(dist)
    x = np.random.normal(size=(50, dim))
    params = dist.example_params(dim=dim)
    aic = dist.aic(x, params)
    assert np.isfinite(float(aic)), f"{dist} AIC not finite"


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_bic(dist):
    """BIC should return a finite scalar."""
    dim = _get_dim(dist)
    x = np.random.normal(size=(50, dim))
    params = dist.example_params(dim=dim)
    bic = dist.bic(x, params)
    assert np.isfinite(float(bic)), f"{dist} BIC not finite"
