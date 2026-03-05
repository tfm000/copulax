"""Shared test fixtures for copulax test suite."""

import pytest
import numpy as np

NUM_SAMPLES: int = 100
NUM_ASSETS: int = 3


def gen_uncorrelated_data(num_samples: int, num_assets: int):
    """Generate uncorrelated multivariate normal data."""
    return np.random.normal(size=(num_samples, num_assets))


def gen_correlated_data(num_assets: int, num_samples: int):
    """Generate correlated multivariate normal data."""
    random_uniform = np.random.uniform(-1, 1, size=(num_assets, num_assets))
    lower_triangular = np.tril(random_uniform)
    correlation = lower_triangular + lower_triangular.T
    np.fill_diagonal(correlation, 1.0)

    sigma_diag = np.diag(np.random.uniform(1, 4, num_assets))
    covariance_matrix = sigma_diag @ correlation @ sigma_diag

    return np.random.multivariate_normal(
        np.zeros(num_assets), covariance_matrix, num_samples
    )
