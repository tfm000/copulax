"""Tests for multivariate probability distributions."""
import pytest
import jax.numpy as jnp
from jax import grad, jit
import numpy as np

from copulax._src.typing import Scalar
from copulax.multivariate import mvt_normal, mvt_student_t, mvt_gh


# Helper functions for testing multivariate distributions
def is_correct_shape(x, output):
    """Check if the shape of the output array is correct."""
    expected_shape: tuple = (x.shape[0], 1)
    return output.shape == expected_shape


def is_positive(output):
    """Check if the output array is positive."""
    return np.all(output >= 0)


def no_nans(output):
    """Check if the output array has no NaNs."""
    return not np.any(np.isnan(output))


def is_finite(output):
    """Check if the output array contains only finite values."""
    return np.all(np.isfinite(output))


def gradients(func, s, data):
    """Calculate the gradients of the output."""
    new_func = lambda x: func(x).sum()
    grad_output = grad(new_func)(data)
    assert no_nans(grad_output), f"{s} gradient contains NaNs"
    assert is_finite(grad_output), f"{s} gradient contains non-finite values"


@jit
def jitable(dist, data):
    return dist.pdf(data)


# Test combinations
DISTRIBUTIONS = [mvt_normal, mvt_student_t, mvt_gh]
ERROR_CASES = ['too_large_dim_sample',]
DATASETS = ['uncorrelated_sample', 'correlated_sample', 
            *ERROR_CASES]
COMBINATIONS = [(dist, dataset) for dist in DISTRIBUTIONS for dataset in DATASETS]


# Tests for multivariate distributions
@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_all_methods_implemented(dist):
    methods: tuple[str] = (
        'support', 'logpdf', 'pdf', 'rvs', 'sample', 'fit', 'stats', 
        'loglikelihood', 'aic', 'bic', 'dtype', 'dist_type', 
        'name'
    )
    for method in methods:
        assert hasattr(dist, method), f"{dist} missing method {method}"


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_name(dist):
    assert isinstance(dist.name, str), f"{dist} name is not a string"
    assert dist.name != "", f"{dist} name is an empty string"


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_dist_object_jitable(dist, datasets):
    data = datasets['uncorrelated_sample']
    jitable(dist, data)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_dtype(dist):
    assert isinstance(dist.dtype, str), f"{dist.name} dtype is not a string"
    assert dist.dtype != "", f"{dist.name} dtype is an empty string"
    assert dist.dtype in ('continuous', 'discrete'), f"dtype is not 'continuous' or 'discrete' for {dist.name}"


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_dist_type(dist):
    assert isinstance(dist.dist_type, str), f"{dist.name} dist_type is not a string"
    assert dist.dist_type != "", f"{dist.name} dist_type is an empty string"
    assert dist.dist_type == 'multivariate', f"dist_type is not 'multivariate' for {dist.name}"


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_support(dist):
    support = dist.support()
    # Checking properties
    assert isinstance(support, jnp.ndarray), f"{dist.name} support is not a JAX array"
    assert support.ndim == 2, f"{dist.name} support is not a 2D array"
    assert np.all(support[:, 0] < support[:, 1]), f"{dist} support bounds are not in order for {dist.name}"
    # Check jit
    jitted_support = jit(dist.support)()


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_logpdf(dist, dataset, datasets):
    data = datasets[dataset]

    # Check error cases - dim mismatch between data and params
    if dataset in ERROR_CASES:
        with pytest.raises(TypeError):
            output = dist.logpdf(data)
        return
    
    # Check non-error cases
    output = dist.logpdf(data)
    # Check properties
    assert is_correct_shape(data, output), f"{dist.name} logpdf has incorrect shape."
    assert no_nans(output), f"{dist.name} logpdf contains NaNs."
    # Check jit
    jitted_pdf = jit(dist.logpdf)(data)
    # Check gradients
    gradients(dist.logpdf, f"{dist.name} logpdf", data)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_pdf(dist, dataset, datasets):
    data = datasets[dataset]

    # Check error cases - dim mismatch between data and params
    if dataset in ERROR_CASES:
        with pytest.raises(TypeError):
            output = dist.pdf(data)
        return
    
    # Check non-error cases
    output = dist.pdf(data)
    # Check properties
    assert is_correct_shape(data, output), f"{dist.name} pdf has incorrect shape."
    assert is_positive(output), f"{dist.name} pdf contains negative values."
    assert no_nans(output), f"{dist.name} pdf contains NaNs."
    assert is_finite(output), f"{dist.name} pdf contains non-finite values."
    # Check jit
    jitted_pdf = jit(dist.pdf)(data)
    # Check gradients
    gradients(dist.pdf, f"{dist.name} pdf", data)


SIZES = [0, 1, 2, 11]
SIZE_COMBINATIONS = [(dist, size) for dist in DISTRIBUTIONS for size in SIZES]
@pytest.mark.parametrize("dist, size", SIZE_COMBINATIONS)
def test_rvs(dist, size):
    output = dist.rvs(size=size)
    # Check properties
    expected_shape = (size, 2)
    assert output.shape == expected_shape, f"{dist} rvs has incorrect shape. Expected {expected_shape}, got {output.shape}"
    assert no_nans(output), f"{dist} rvs contains NaNs for size {size}"
    assert is_finite(output), f"{dist} rvs contains infinite values for size {size}"
    # Check jit
    jitted_rvs = jit(dist.rvs, static_argnums=0)(size=size)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_fit(dist, dataset, datasets):
    if dataset in ERROR_CASES:
        return
    data = datasets[dataset]
    params = dist.fit(data)
    # Check properties
    assert isinstance(params, dict), f"{dist.name} fit does not return a dictionary"
    assert all(isinstance(v, jnp.ndarray) for v in params.values()), f"{dist.name} fit parameters are not all arrays."
    assert all(no_nans(v) for v in params.values()), f"{dist.name} fit parameters contain NaNs."
    assert all(is_finite(v) for v in params.values()), f"{dist.name} fit parameters contain infinite values."
    # Check jit
    jitted_fit = jit(dist.fit)(data)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def stats(dist):
    stats = dist.stats()
    # Check properties
    assert isinstance(stats, dict), f"{dist.name} stats is not a dictionary"
    assert len(stats) > 0, f"{dist.name} stats is an empty dictionary"
    assert all(isinstance(v, jnp.ndarray) for v in stats.values()), f"{dist.name} stats values are not arrays"
    # Check jit
    jitted_stats = jit(dist.stats)()


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_loglikelihood(dist, dataset, datasets):
    data = datasets[dataset]

    # Check error cases - dim mismatch between data and params
    if dataset in ERROR_CASES:
        with pytest.raises(TypeError):
            output = dist.loglikelihood(data)
        return
    
    # Check non-error cases
    output = dist.loglikelihood(data)
    # Check properties
    assert isinstance(output, jnp.ndarray) and output.shape == (), f"{dist.name} loglikelihood is non-scalar."
    assert is_finite(output), f"{dist.name} loglikelihood is infinite."
    assert no_nans(output), f"{dist.name} loglikelihood is NaN."
    # Check jit
    jitted_loglikelihood = jit(dist.loglikelihood)(data)
    # Check gradients
    gradients(dist.loglikelihood, f"{dist.name} loglikelihood", data)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_aic(dist, dataset, datasets):
    data = datasets[dataset]

    # Check error cases - dim mismatch between data and params
    if dataset in ERROR_CASES:
        with pytest.raises(TypeError):
            output = dist.aic(data)
        return
    
    # Check non-error cases
    output = dist.aic(data)
    # Check properties
    assert isinstance(output, jnp.ndarray) and output.shape == (), f"{dist.name} aic is non-scalar."
    assert is_finite(output), f"{dist.name} aic is infinite."
    assert no_nans(output), f"{dist.name} aic is NaN."
    # Check jit
    jitted_aic = jit(dist.aic)(data)
    # Check gradients
    gradients(dist.aic, f"{dist.name} aic on {dataset}", data)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_bic(dist, dataset, datasets):
    data = datasets[dataset]

    # Check error cases - dim mismatch between data and params
    if dataset in ERROR_CASES:
        with pytest.raises(TypeError):
            output = dist.bic(data)
        return
    
    # Check non-error cases
    output = dist.bic(data)
    # Check properties
    assert isinstance(output, jnp.ndarray) and output.shape == (), f"{dist.name} bic is non-scalar."
    assert is_finite(output), f"{dist.name} bic is infinite."
    assert no_nans(output), f"{dist.name} bic is NaN."
    # Check jit
    jitted_bic = jit(dist.bic)(data)
    # Check gradients
    gradients(dist.bic, f"{dist.name} bic", data)