"""Tests for correlation and shape matrices."""
import pytest
import jax.numpy as jnp
import numpy as np
from jax import jit

from copulax.multivariate import corr, cov


# Helper functions for testing matrix properties
def is_square(matrix):
    """Check if a matrix is square."""
    return matrix.shape[0] == matrix.shape[1]


def is_symmetric(matrix):
    """Check if a matrix is symmetric."""
    return jnp.allclose(matrix, matrix.T)


def is_positive_semi_definite(matrix):
    """Check if a matrix is positive semi-definite."""
    eigenvalues = jnp.linalg.eigvalsh(matrix)
    return jnp.all(eigenvalues >= 0)


def is_positive_definite(matrix):
    """Check if a matrix is positive definite."""
    eigenvalues = jnp.linalg.eigvalsh(matrix)
    return jnp.all(eigenvalues > 0)


def has_ones_on_diagonal(matrix, tol=1e-10):
    """Check if a matrix has ones on the diagonal."""
    diag = jnp.diag(matrix)
    return jnp.allclose(diag, jnp.ones_like(diag), atol=tol)


def is_bounded(matrix, tol=1e-5):
    """Check if a matrix is bounded."""
    return jnp.all(jnp.abs(matrix) <= 1 + tol)


# Test methods
CORRELATION_METHODS = [
    "pearson", "spearman", "kendall", "pp_kendall", 
    "rm_pearson", "rm_spearman", "rm_kendall", "rm_pp_kendall",
    "laloux_pearson", "laloux_spearman", "laloux_kendall", "laloux_pp_kendall"
]


# Tests for correlation matrices
@pytest.mark.parametrize("method", CORRELATION_METHODS)
def test_corr_on_correlated_data(correlated_sample, method):
    correlation = corr(correlated_sample, method=method)

    # Check properties
    assert is_square(correlation), f"{method} correlation matrix is not square"
    assert is_symmetric(correlation), f"{method} correlation matrix is not symmetric"
    assert has_ones_on_diagonal(correlation), f"{method} correlation matrix does not have ones on diagonal"
    assert is_positive_semi_definite(correlation), f"{method} correlation matrix is not positive semi-definite"
    assert is_bounded(correlation), f"{method} correlation matrix is not bounded"

    # Checking works with jit
    jit(corr, static_argnames=("method",))(correlated_sample, method)

    
@pytest.mark.parametrize("method", CORRELATION_METHODS)
def test_corr_on_uncorrelated_data(uncorrelated_sample, method):
    correlation = corr(uncorrelated_sample, method=method)
    # Check properties
    assert is_square(correlation), f"{method} correlation matrix is not square with uncorrelated data"
    assert is_symmetric(correlation), f"{method} correlation matrix is not symmetric with uncorrelated data"
    assert has_ones_on_diagonal(correlation), f"{method} correlation matrix does not have ones on diagonal with uncorrelated data"
    assert is_positive_semi_definite(correlation), f"{method} correlation matrix is not positive semi-definite with uncorrelated data"
    assert is_bounded(correlation), f"{method} correlation matrix is not bounded with uncorrelated data"

    # Checking works with jit
    jit(corr, static_argnames=("method",))(uncorrelated_sample, method)


# Tests for covariance matrices
@pytest.mark.parametrize("method", CORRELATION_METHODS)
def test_cov_on_correlated_data(correlated_sample, method):
    covariance = cov(correlated_sample, method=method)
    # Check properties
    assert is_square(covariance), f"{method} covariance matrix is not square"
    assert is_symmetric(covariance), f"{method} covariance matrix is not symmetric"
    assert is_positive_definite(covariance), f"{method} covariance matrix is not positive definite"

    # Checking works with jit
    jit(cov, static_argnames=("method",))(correlated_sample, method)
    
@pytest.mark.parametrize("method", CORRELATION_METHODS)
def test_cov_on_uncorrelated_data(uncorrelated_sample, method):
    covariance = cov(uncorrelated_sample, method=method)
    # Check properties
    assert is_square(covariance), f"{method} covariance matrix is not square with uncorrelated data"
    assert is_symmetric(covariance), f"{method} covariance matrix is not symmetric with uncorrelated data"
    assert is_positive_definite(covariance), f"{method} covariance matrix is not positive definite with uncorrelated data"

    # Checking works with jit
    jit(cov, static_argnames=("method",))(uncorrelated_sample, method)


sizes = tuple((2, 3, 5))

@pytest.mark.parametrize("n", sizes)
def test_random_correlation(n):
    """Test random correlation matrix generation."""
    random_corr = corr.random_correlation(n)
    
    # Check properties
    assert is_square(random_corr), "Random correlation matrix is not square"
    assert random_corr.shape == (n, n), "Random correlation matrix has incorrect shape"
    assert is_symmetric(random_corr), "Random correlation matrix is not symmetric"
    assert has_ones_on_diagonal(random_corr), "Random correlation matrix does not have ones on diagonal"
    assert is_positive_definite(random_corr), "Random correlation matrix is not positive definite"
    assert is_bounded(random_corr), "Random correlation matrix is not bounded"

    # Checking works with jit
    jit(corr.random_correlation, static_argnames=("size",))(n)


@pytest.mark.parametrize("n", sizes)
def test_random_covariance(n):
    """Test random covariance matrix generation."""
    random_vars = np.random.uniform(size=(n, 1))
    random_cov = cov.random_covariance(random_vars)
    
    # Check properties
    assert is_square(random_cov), "Random covariance matrix is not square"
    assert random_cov.shape == (n, n), "Random covariance matrix has incorrect shape"
    assert is_symmetric(random_cov), "Random covariance matrix is not symmetric"
    assert is_positive_definite(random_cov), "Random covariance matrix is not positive definite"

    # Checking works with jit
    jit(cov.random_covariance)(random_vars)


# # Edge case tests
# def test_corr_small_sample(correlated_small_sample):
#     """Test correlation matrices with small samples."""
#     for method in CORRELATION_METHODS:
#         correlation = corr(correlated_small_sample, method=method)
#         assert is_square(correlation), f"{method} correlation matrix is not square with small sample"
#         assert is_symmetric(correlation), f"{method} correlation matrix is not symmetric with small sample"
#         assert has_ones_on_diagonal(correlation), f"{method} correlation matrix does not have ones on diagonal with small sample"
#         assert is_positive_definite(correlation), f"{method} correlation matrix is not positive definite with small sample"


# def test_cov_small_sample(correlated_small_sample):
#     """Test covariance matrices with small samples."""
#     for method in CORRELATION_METHODS:
#         covariance = cov(correlated_small_sample, method=method)
#         assert is_square(covariance), f"{method} covariance matrix is not square with small sample"
#         assert is_symmetric(covariance), f"{method} covariance matrix is not symmetric with small sample"
#         assert is_positive_semi_definite(covariance), f"{method} covariance matrix is not positive semi-definite with small sample"