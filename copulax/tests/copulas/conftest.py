import pytest
import numpy as np

from copulax.copulas import gaussian_copula, student_t_copula, gh_copula
from copulax.tests.conftest import (
    gen_uncorrelated_data,
    gen_correlated_data,
    NUM_SAMPLES,
    NUM_ASSETS,
)


np.random.seed(0)


@pytest.fixture(scope="package", autouse=True)
def u_sample():
    """Generate a sample of uniform marginals."""
    eps = 1e-2
    u_raw = np.random.uniform(size=(NUM_SAMPLES, NUM_ASSETS))
    return np.clip(u_raw, eps, 1 - eps)


@pytest.fixture(scope="package", autouse=True)
def datasets():
    return {
        "uncorrelated_sample": gen_uncorrelated_data(NUM_SAMPLES, NUM_ASSETS),
        "correlated_sample": gen_correlated_data(NUM_ASSETS, NUM_SAMPLES),
        "too_large_dim_sample": gen_uncorrelated_data(NUM_SAMPLES, 100),
    }
