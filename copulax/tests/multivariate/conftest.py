import pytest
import numpy as np


NUM_SAMPLES: int = 100
NUM_ASSETS: int = 5


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
    covariance_matrix: np.ndarray = correlation @ sigma_diag @ correlation.T

    # generating correlated data
    return np.random.multivariate_normal(np.zeros(num_assets), covariance_matrix, num_samples)


@pytest.fixture(scope='package', autouse=True)
def uncorrelated_sample():
    return gen_uncorrelated_data(NUM_SAMPLES, NUM_ASSETS)


@pytest.fixture(scope='package', autouse=True)
def correlated_sample():
    return gen_correlated_data(NUM_ASSETS, NUM_SAMPLES)


@pytest.fixture(scope='package', autouse=True)
def uncorrelated_small_sample():
    return gen_uncorrelated_data(NUM_ASSETS, NUM_ASSETS)


@pytest.fixture(scope='package', autouse=True)
def correlated_small_sample():
    return gen_correlated_data(NUM_ASSETS+1, NUM_ASSETS)

@pytest.fixture(scope='package', autouse=True)
def datasets(uncorrelated_sample, correlated_sample, uncorrelated_small_sample, correlated_small_sample):
    return {'uncorrelated_sample': gen_uncorrelated_data(NUM_SAMPLES, 2),
            'correlated_sample': gen_correlated_data(2, NUM_SAMPLES), 
            
            # 'too_small_dim_sample': gen_uncorrelated_data(NUM_SAMPLES, 1),
            'too_large_dim_sample': gen_uncorrelated_data(NUM_SAMPLES, 100),
            }


