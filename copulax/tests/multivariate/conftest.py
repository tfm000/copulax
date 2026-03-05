import pytest
import numpy as np

from copulax.tests.conftest import (
    gen_uncorrelated_data,
    gen_correlated_data,
    NUM_SAMPLES,
    NUM_ASSETS,
)


np.random.seed(0)


@pytest.fixture(scope="package", autouse=True)
def uncorrelated_sample():
    return gen_uncorrelated_data(NUM_SAMPLES, NUM_ASSETS)


@pytest.fixture(scope="package", autouse=True)
def correlated_sample():
    return gen_correlated_data(NUM_ASSETS, NUM_SAMPLES)


@pytest.fixture(scope="package", autouse=True)
def uncorrelated_small_sample():
    return gen_uncorrelated_data(NUM_ASSETS, NUM_ASSETS)


@pytest.fixture(scope="package", autouse=True)
def correlated_small_sample():
    return gen_correlated_data(NUM_ASSETS + 1, NUM_ASSETS)


@pytest.fixture(scope="package", autouse=True)
def datasets(
    uncorrelated_sample,
    correlated_sample,
    uncorrelated_small_sample,
    correlated_small_sample,
):
    return {
        "uncorrelated_sample": gen_uncorrelated_data(NUM_SAMPLES, NUM_ASSETS),
        "correlated_sample": gen_correlated_data(NUM_ASSETS, NUM_SAMPLES),
        # 'too_small_dim_sample': gen_uncorrelated_data(NUM_SAMPLES, 1),
        "too_large_dim_sample": gen_uncorrelated_data(NUM_SAMPLES, 100),
    }
