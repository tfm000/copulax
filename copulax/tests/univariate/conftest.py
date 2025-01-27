import pytest
import numpy as np

from copulax.univariate import *


NUM_SAMPLES: int = 100


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
def continuous_dists():
    return {'gamma': gamma, 'gh': gh, 'gig': gig, 'ig': ig, 'lognormal': lognormal, 'normal': normal, 'skewed_t': skewed_t, 'student_t': student_t, 'uniform': uniform}


@pytest.fixture(scope='package', autouse=True)
def non_inverse_transform_dists():
    return {'gh': gh, 'gig': gig, 'lognormal': lognormal, 'normal': normal, 'student_t': student_t, 'uniform': uniform}


@pytest.fixture(scope='package', autouse=True)
def inverse_transform_dists():
    return {'skewed_t': skewed_t,}


@pytest.fixture(scope='package', autouse=True)
def continuous_uniform_data():
    eps: float = 1e-6
    return np.random.uniform(eps, 1-eps, 10)