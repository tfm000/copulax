"""Tests for correlation and shape matrices."""
import pytest
import jax.numpy as jnp
import numpy as np

from copulax._src.multivariate._shape import corr, cov


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

    
@pytest.mark.parametrize("method", CORRELATION_METHODS)
def test_corr_on_uncorrelated_data(uncorrelated_sample, method):
    correlation = corr(uncorrelated_sample, method=method)
    # Check properties
    assert is_square(correlation), f"{method} correlation matrix is not square with uncorrelated data"
    assert is_symmetric(correlation), f"{method} correlation matrix is not symmetric with uncorrelated data"
    assert has_ones_on_diagonal(correlation), f"{method} correlation matrix does not have ones on diagonal with uncorrelated data"
    assert is_positive_semi_definite(correlation), f"{method} correlation matrix is not positive semi-definite with uncorrelated data"


# Tests for covariance matrices
@pytest.mark.parametrize("method", CORRELATION_METHODS)
def test_cov_on_correlated_data(correlated_sample, method):
    covariance = cov(correlated_sample, method=method)
    # Check properties
    assert is_square(covariance), f"{method} covariance matrix is not square"
    assert is_symmetric(covariance), f"{method} covariance matrix is not symmetric"
    assert is_positive_definite(covariance), f"{method} covariance matrix is not positive definite"
    
@pytest.mark.parametrize("method", CORRELATION_METHODS)
def test_cov_on_uncorrelated_data(uncorrelated_sample, method):
    covariance = cov(uncorrelated_sample, method=method)
    # Check properties
    assert is_square(covariance), f"{method} covariance matrix is not square with uncorrelated data"
    assert is_symmetric(covariance), f"{method} covariance matrix is not symmetric with uncorrelated data"
    assert is_positive_definite(covariance), f"{method} covariance matrix is not positive definite with uncorrelated data"


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