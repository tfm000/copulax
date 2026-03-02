import pytest
import numpy as np

from copulax.copulas import gaussian_copula, student_t_copula, gh_copula


NUM_SAMPLES: int = 100
NUM_ASSETS: int = 3


np.random.seed(0)

def gen_uncorrelated_data(num_samples: int, num_assets: int):
    return np.random.normal(size=(num_samples, num_assets))


def gen_correlated_data(num_assets: int, num_samples: int):
    # generating a random correlation matrix
    random_uniform: np.ndarray = np.random.uniform(-1, 1, size=(num_assets, num_assets))
    lower_triangular: np.ndarray = np.tril(random_uniform)
    correlation: np.ndarray = lower_triangular + lower_triangular.T
    np.fill_diagonal(correlation, 1.0)

    # converting to covariance matrix
    sigma_diag: np.ndarray = np.diag(np.random.uniform(1, 4, num_assets))
    covariance_matrix: np.ndarray = sigma_diag @ correlation @ sigma_diag

    # generating correlated data
    return np.random.multivariate_normal(np.zeros(num_assets), covariance_matrix, num_samples)


@pytest.fixture(scope='package', autouse=True)
def u_sample():
    """Generate a sample of uniform marginals."""
    eps = 1e-2
    u_raw = np.random.uniform(size=(NUM_SAMPLES, NUM_ASSETS))
    return np.clip(u_raw, eps, 1 - eps)


@pytest.fixture(scope='package', autouse=True)
def datasets():
    return {'uncorrelated_sample': gen_uncorrelated_data(NUM_SAMPLES, NUM_ASSETS),
            'correlated_sample': gen_correlated_data(NUM_ASSETS, NUM_SAMPLES), 
            
            'too_large_dim_sample': gen_uncorrelated_data(NUM_SAMPLES, 100),
            }