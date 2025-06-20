import pytest
import numpy as np

from copulax.univariate import *


NUM_SAMPLES: int = 100
np.random.seed(0)


@pytest.fixture(scope='package', autouse=True)
def continuous_data():
    # creating some skewed, strictly positive data
    normal = np.random.normal(0, 1, NUM_SAMPLES)
    uniform = np.random.uniform(0, 1, NUM_SAMPLES)
    sample = normal * uniform
    sample[sample > 0] = sample[sample > 0] * 3 # adding skew
    sample = sample + 3 # shift the data to the right
    return sample[sample > 0]


@pytest.fixture(scope='package', autouse=True)
def discrete_data():
    # creating some discrete data
    sample = np.random.randint(0, 10, NUM_SAMPLES)
    return sample


@pytest.fixture(scope='package', autouse=True)
def continuous_dists():
    return {gamma, gh, gig, ig, lognormal, normal, skewed_t, student_t, uniform}


@pytest.fixture(scope='package', autouse=True)
def non_inverse_transform_dists():
    return {gamma, gh, gig, ig, lognormal, normal, skewed_t, student_t, uniform}


@pytest.fixture(scope='package', autouse=True)
def inverse_transform_dists():
    return {}


@pytest.fixture(scope='package', autouse=True)
def uniform_data():
    eps: float = 1e-6
    return np.random.uniform(eps, 1-eps, 10)