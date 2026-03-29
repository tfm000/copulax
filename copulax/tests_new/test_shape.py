"""Rigorous tests for copulax/_src/multivariate/_shape.py.

Cross-validates correlation and covariance methods against scipy/numpy,
tests edge cases, round-trip consistency, and error handling.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy import stats as sp_stats

from copulax.multivariate import corr, cov, random_correlation, random_covariance
from copulax._src.multivariate._shape import _corr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def correlated_data():
    """Correlated multivariate normal data (n=200, d=4)."""
    np.random.seed(12345)
    R = np.array([
        [1.0, 0.7, -0.3, 0.5],
        [0.7, 1.0, 0.2, -0.4],
        [-0.3, 0.2, 1.0, 0.6],
        [0.5, -0.4, 0.6, 1.0],
    ])
    sigma = np.diag([2.0, 1.5, 3.0, 0.5])
    C = sigma @ R @ sigma
    return np.random.multivariate_normal(np.zeros(4), C, size=200)


@pytest.fixture(scope="module")
def uncorrelated_data():
    """Uncorrelated standard normal data (n=200, d=3)."""
    np.random.seed(99999)
    return np.random.normal(size=(200, 3))


CORRELATION_METHODS = [
    "pearson", "spearman", "kendall", "pp_kendall",
    "rm_pearson", "rm_spearman", "rm_kendall", "rm_pp_kendall",
    "laloux_pearson", "laloux_spearman", "laloux_kendall", "laloux_pp_kendall",
]


# ---------------------------------------------------------------------------
# Scipy cross-validation
# ---------------------------------------------------------------------------

class TestCorrVsScipy:
    """Cross-validate correlation methods against scipy/numpy."""

    def test_pearson_vs_numpy(self, correlated_data):
        copulax_R = np.array(corr(correlated_data, method="pearson"))
        numpy_R = np.corrcoef(correlated_data, rowvar=False)
        np.testing.assert_allclose(copulax_R, numpy_R, atol=1e-14)

    def test_spearman_vs_scipy(self, correlated_data):
        copulax_R = np.array(corr(correlated_data, method="spearman"))
        scipy_R = sp_stats.spearmanr(correlated_data).statistic
        np.testing.assert_allclose(copulax_R, scipy_R, atol=1e-14)

    def test_kendall_vs_scipy(self, correlated_data):
        copulax_R = np.array(corr(correlated_data, method="kendall"))
        d = correlated_data.shape[1]
        scipy_R = np.eye(d)
        for i in range(d):
            for j in range(i + 1, d):
                tau, _ = sp_stats.kendalltau(
                    correlated_data[:, i], correlated_data[:, j]
                )
                scipy_R[i, j] = tau
                scipy_R[j, i] = tau
        np.testing.assert_allclose(copulax_R, scipy_R, atol=1e-14)


class TestCovVsScipy:
    """Cross-validate covariance against numpy."""

    def test_pearson_cov_vs_numpy(self, correlated_data):
        """Pearson covariance should match numpy.cov (ddof=1)."""
        copulax_C = np.array(cov(correlated_data, method="pearson"))
        numpy_C = np.cov(correlated_data, rowvar=False, ddof=1)
        np.testing.assert_allclose(copulax_C, numpy_C, atol=1e-12)

    def test_cov_diagonal_is_sample_variance(self, correlated_data):
        """Diagonal of Pearson covariance should equal sample variances (ddof=1)."""
        copulax_C = np.array(cov(correlated_data, method="pearson"))
        expected_vars = np.var(correlated_data, axis=0, ddof=1)
        np.testing.assert_allclose(np.diag(copulax_C), expected_vars, atol=1e-12)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test that invalid inputs produce clear errors."""

    def test_invalid_method_raises_valueerror(self, correlated_data):
        with pytest.raises(ValueError, match="Unknown correlation method"):
            corr(correlated_data, method="nonexistent")

    def test_invalid_method_in_cov_raises(self, correlated_data):
        """cov() delegates to corr(), so invalid method should also raise."""
        with pytest.raises(ValueError, match="Unknown correlation method"):
            cov(correlated_data, method="bogus_method")


# ---------------------------------------------------------------------------
# Round-trip and conversion consistency
# ---------------------------------------------------------------------------

class TestRoundTrips:
    """Verify cov <-> corr conversion consistency."""

    def test_cov_to_corr_to_cov(self, correlated_data):
        """cov -> _corr_from_cov -> _cov_from_vars should round-trip."""
        C = jnp.array(np.cov(correlated_data, rowvar=False))
        R = _corr._corr_from_cov(C)
        vars_orig = jnp.diag(C)
        C_reconstructed = _corr._cov_from_vars(vars_orig, R)
        np.testing.assert_allclose(
            np.array(C_reconstructed), np.array(C), atol=1e-12
        )

    def test_corr_from_cov_unit_diagonal(self, correlated_data):
        """_corr_from_cov should produce unit diagonal."""
        C = jnp.array(np.cov(correlated_data, rowvar=False))
        R = _corr._corr_from_cov(C)
        np.testing.assert_allclose(
            np.array(jnp.diag(R)), np.ones(C.shape[0]), atol=1e-14
        )

    def test_corr_from_cov_preserves_psd(self):
        """_corr_from_cov on a PSD matrix should produce a PSD result."""
        key = random.PRNGKey(7)
        W = random.normal(key, (5, 5))
        C = W @ W.T + 0.1 * jnp.eye(5)
        R = _corr._corr_from_cov(C)
        eigvals = np.array(jnp.linalg.eigvalsh(R))
        assert np.all(eigvals >= -1e-12), f"min eigenvalue: {eigvals.min()}"


# ---------------------------------------------------------------------------
# random_correlation and random_covariance
# ---------------------------------------------------------------------------

class TestRandomMatrices:
    """Test random matrix generation."""

    @pytest.mark.parametrize("n", [2, 3, 5, 10, 50])
    def test_random_correlation_properties(self, n):
        R = np.array(random_correlation(n, key=random.PRNGKey(n)))
        assert R.shape == (n, n)
        np.testing.assert_allclose(R, R.T, atol=1e-12, err_msg="not symmetric")
        np.testing.assert_allclose(
            np.diag(R), np.ones(n), atol=1e-12, err_msg="diagonal != 1"
        )
        eigvals = np.linalg.eigvalsh(R)
        assert eigvals.min() >= -1e-10, f"not PSD: min eig = {eigvals.min()}"
        assert np.all(np.abs(R) <= 1 + 1e-10), "entries out of [-1, 1]"

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_random_covariance_diagonal_matches_input(self, n):
        """Diagonal of random_covariance should equal the input variances."""
        input_vars = jnp.array(np.random.uniform(0.5, 5.0, size=(n,)))
        rcov = np.array(random_covariance(input_vars, key=random.PRNGKey(n)))
        np.testing.assert_allclose(
            np.diag(rcov), np.array(input_vars), atol=1e-12
        )

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_random_covariance_is_psd(self, n):
        input_vars = jnp.array(np.random.uniform(0.5, 5.0, size=(n,)))
        rcov = np.array(random_covariance(input_vars, key=random.PRNGKey(n)))
        eigvals = np.linalg.eigvalsh(rcov)
        assert eigvals.min() > -1e-10, f"not PSD: min eig = {eigvals.min()}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: minimal dimensions, rank-deficient, etc."""

    @pytest.mark.parametrize("method", CORRELATION_METHODS)
    def test_d2_minimal_dimension(self, method):
        """All methods should work for d=2."""
        x = np.array(random.normal(random.PRNGKey(0), (50, 2)))
        R = np.array(corr(x, method=method))
        assert R.shape == (2, 2)
        np.testing.assert_allclose(R, R.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(R), [1.0, 1.0], atol=1e-10)
        assert np.all(np.abs(R) <= 1 + 1e-10)

    @pytest.mark.parametrize("method", ["pearson", "rm_pearson", "laloux_pearson"])
    def test_n_less_than_d(self, method):
        """Should handle rank-deficient data (n < d) without crashing."""
        x = np.array(random.normal(random.PRNGKey(42), (5, 10)))
        R = np.array(corr(x, method=method))
        assert R.shape == (10, 10)
        np.testing.assert_allclose(R, R.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(R), np.ones(10), atol=1e-10)
        assert np.all(np.abs(R) <= 1 + 1e-10)

    def test_uncorrelated_data_off_diagonals_near_zero(self, uncorrelated_data):
        """Pearson of uncorrelated data should have off-diagonals near 0."""
        R = np.array(corr(uncorrelated_data, method="pearson"))
        off_diag = R[~np.eye(R.shape[0], dtype=bool)]
        assert np.max(np.abs(off_diag)) < 0.25, (
            f"Off-diag too large for uncorrelated data: {np.max(np.abs(off_diag)):.4f}"
        )

    @pytest.mark.parametrize("method", CORRELATION_METHODS)
    def test_corr_output_shape(self, correlated_data, method):
        """Output shape should be (d, d)."""
        R = corr(correlated_data, method=method)
        d = correlated_data.shape[1]
        assert R.shape == (d, d)

    @pytest.mark.parametrize("method", CORRELATION_METHODS)
    def test_cov_output_shape(self, correlated_data, method):
        """Output shape should be (d, d)."""
        C = cov(correlated_data, method=method)
        d = correlated_data.shape[1]
        assert C.shape == (d, d)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

class TestJIT:
    """Verify JIT compilation works for all public functions."""

    @pytest.mark.parametrize("method", CORRELATION_METHODS)
    def test_corr_jit(self, correlated_data, method):
        jit_corr = jax.jit(corr, static_argnames=("method",))
        R = np.array(jit_corr(correlated_data, method=method))
        R_eager = np.array(corr(correlated_data, method=method))
        np.testing.assert_allclose(R, R_eager, atol=1e-14)

    @pytest.mark.parametrize("method", CORRELATION_METHODS)
    def test_cov_jit(self, correlated_data, method):
        jit_cov = jax.jit(cov, static_argnames=("method",))
        C = np.array(jit_cov(correlated_data, method=method))
        C_eager = np.array(cov(correlated_data, method=method))
        np.testing.assert_allclose(C, C_eager, atol=1e-14)

    def test_random_correlation_jit(self):
        jit_rc = jax.jit(random_correlation, static_argnames=("size",))
        R = np.array(jit_rc(5, key=random.PRNGKey(0)))
        R_eager = np.array(random_correlation(5, key=random.PRNGKey(0)))
        np.testing.assert_allclose(R, R_eager, atol=1e-14)
