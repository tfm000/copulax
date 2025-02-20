import pytest
import numpy as np

# from copulax.univariate import *

from copulax._src.univariate.uniform_new import uniform
from copulax._src.univariate.gh_new import gh
from copulax._src.univariate.gig_new import gig
from copulax._src.univariate.gamma_new import gamma
from copulax._src.univariate.lognormal_new import lognormal
from copulax._src.univariate.normal_new import normal
from copulax._src.univariate.skewed_t_new import skewed_t
from copulax._src.univariate.student_t_new import student_t
from copulax._src.univariate.ig_new import ig


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
    return {gamma, gh, gig, ig, lognormal, normal, skewed_t, student_t, uniform}


@pytest.fixture(scope='package', autouse=True)
def non_inverse_transform_dists():
    return {gamma, gh, gig, ig, lognormal, normal, skewed_t, student_t, uniform}


@pytest.fixture(scope='package', autouse=True)
def inverse_transform_dists():
    return {}


@pytest.fixture(scope='package', autouse=True)
def continuous_uniform_data():
    eps: float = 1e-6
    return np.random.uniform(eps, 1-eps, 10)