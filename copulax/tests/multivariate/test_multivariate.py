"""Tests for multivariate probability distributions."""
import pytest
import jax.numpy as jnp
from jax import grad, jit
import numpy as np

from copulax._src._distributions import Multivariate
from copulax.tests.helpers import *
from copulax._src.typing import Scalar
from copulax.multivariate import mvt_normal, mvt_student_t, mvt_gh
from copulax.tests.multivariate.conftest import NUM_ASSETS

# Test combinations
DISTRIBUTIONS = tuple((mvt_normal, mvt_student_t, mvt_gh))
ERROR_CASES = tuple(('too_large_dim_sample',))
DATASETS = tuple(('uncorrelated_sample', 'correlated_sample', *ERROR_CASES))
COMBINATIONS = tuple((dist, dataset) for dist in DISTRIBUTIONS for dataset in DATASETS)


# Helper functions for testing univariate distributions
def check_params(params, s):
    assert isinstance(params, dict), f"{s} params is not a dict"
    assert len(params) > 0, f"{s} is empty"
    assert all(isinstance(k, str) for k in params.keys()), f"{s} params keys are not strings"
    assert all(isinstance(v, jnp.ndarray) for v in params.values()), f"{s} params values are not arrays"
    assert all((v.ndim == 0 and v.shape == () and v.size == 1) or (v.ndim == 2 and v.shape == (v.size, 1) and v.size > 1) or (v.ndim == 2 and v.shape == (int(v.size ** 0.5), int(v.size ** 0.5)) and v.size > 1) for v in params.values()), f"{s} params values are not scalars, 1D-vectors or 2D-square matrices"
    assert any(jnp.any(jnp.isnan(v)) for v in params.values()) == False, f"{s} params values are NaN"
    assert all(jnp.all(jnp.isfinite(v)) for v in params.values()) == True, f"{s} params values are not finite"


# Tests for multivariate distributions
@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_all_objects(dist):
    assert isinstance(dist, Multivariate), f"{dist} is not a Multivariate object"


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_all_methods_implemented(dist):
    methods: set[str] = {
        'support', 'logpdf', 'pdf', 'rvs', 'sample', 'fit', 'stats', 
        'loglikelihood', 'aic', 'bic', 'dtype', 'dist_type', 'name', 
        'example_params',
    }
    # testing desired methods are implemented
    for method in methods:
        assert hasattr(dist, method), f"{dist} missing method {method}"

    # testing no additional methods are implemented
    pytree_methods: set[str] = {'tree_flatten', 'tree_unflatten'}
    extra_methods = set(dist.__dict__.keys()) - methods - pytree_methods
    extra_methods = {m for m in extra_methods if not m.startswith('_')}
    assert not extra_methods, f"{dist} has extra methods: {extra_methods}"
    

@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_name(dist):
    assert isinstance(dist.name, str), f"{dist} name is not a string"
    assert dist.name != "", f"{dist} name is an empty string"
    assert dist.name == str(dist), f"{dist} name does not match its string representation"


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_dtype(dist):
    assert isinstance(dist.dtype, str), f"{dist} dtype is not a string"
    assert dist.dtype != "", f"{dist} dtype is an empty string"
    assert dist.dtype in ('continuous', 'discrete'), f"dtype is not 'continuous' or 'discrete' for {dist}"


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_dist_type(dist):
    assert isinstance(dist.dist_type, str), f"{dist} dist_type is not a string"
    assert dist.dist_type != "", f"{dist} dist_type is an empty string"
    assert dist.dist_type == 'multivariate', f"dist_type is not 'multivariate' for {dist}"


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_example_params(dist):
    params = dist.example_params()

    # checking properties
    s = f"{dist} example_params"
    check_params(params, s)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_dist_object_jitable(dist):
    jittable(dist)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_support(dist):
    params: dict = dist.example_params()
    support = dist.support(params)
    
    # Checking properties
    assert isinstance(support, jnp.ndarray), f"{dist} support is not a JAX array"
    assert support.ndim == 2, f"{dist} support is not a 2D array"
    assert np.all(support[:, 0] < support[:, 1]), f"{dist} support bounds are not in order."
    assert no_nans(support), f"{dist} support contains NaNs."
    
    # Check jit
    jitted_support = jit(dist.support)(params)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_logpdf(dist, dataset, datasets):
    data = datasets[dataset]
    params = dist.example_params()

    if dataset in ERROR_CASES:
        # Check error cases - dim mismatch between data and params
        with pytest.raises(TypeError):
            output = dist.logpdf(data, params=params)
    else:
        # Check non-error cases
        output = dist.logpdf(data, params=params)
        # Check properties
        correct_mvt_shape(data, output, dist, 'logpdf')
        assert no_nans(output), f"{dist} logpdf contains NaNs."
        # Check jit
        jitted_pdf = jit(dist.logpdf)(data, params=params)
        # Check gradients
        gradients(dist.logpdf, f"{dist} logpdf", data, params=params)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_pdf(dist, dataset, datasets):
    data = datasets[dataset]
    params = dist.example_params()

    if dataset in ERROR_CASES:
        # Check error cases - dim mismatch between data and params
        with pytest.raises(TypeError):
            output = dist.pdf(data, params=params)
    else:
        # Check non-error cases
        output = dist.pdf(data, params=params)
        # Check properties
        correct_mvt_shape(data, output, dist, 'pdf')
        assert is_positive(output), f"{dist} pdf contains negative values."
        assert no_nans(output), f"{dist} pdf contains NaNs."
        assert is_finite(output), f"{dist} pdf contains non-finite values."
        # Check jit
        jitted_pdf = jit(dist.pdf)(data, params=params)
        # Check gradients
        gradients(dist.pdf, f"{dist} pdf", data, params)


SIZES = (0, 1, 2, 11)
SIZE_COMBINATIONS = tuple((dist, size) for dist in DISTRIBUTIONS for size in SIZES)
@pytest.mark.parametrize("dist, size", SIZE_COMBINATIONS)
def test_rvs(dist, size):
    params: dict = dist.example_params()
    output = dist.rvs(size=size, params=params)
    # Check properties
    expected_shape = (size, NUM_ASSETS)
    assert output.shape == expected_shape, f"{dist} rvs has incorrect shape. Expected {expected_shape}, got {output.shape}"
    assert no_nans(output), f"{dist} rvs contains NaNs for size {size}"
    assert is_finite(output), f"{dist} rvs contains infinite values for size {size}"
    # Check jit
    jitted_rvs = jit(dist.rvs, static_argnums=0)(size=size, params=params)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_fit(dist, dataset, datasets):
    if dataset not in ERROR_CASES:
        data = datasets[dataset]
        params = dist.fit(data)
        # Check properties
        check_params(params, f"{dist} fit")
        # Check jit
        jitted_fit = jit(dist.fit)(data)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def stats(dist):
    params: dict = dist.example_params()
    stats = dist.stats(params=params)
    # Check properties
    assert isinstance(stats, dict), f"{dist} stats outputted wrong type"
    assert len(stats) > 0, f"{dist} stats is empty"
    assert all(isinstance(k, str) for k in stats.keys()), f"{dist} stats keys are not strings"
    assert all(isinstance(v, jnp.ndarray) for v in stats.values()), f"{dist} stats values are not arrays"
    assert all((v.ndim == 0 and v.shape == () and v.size == 1) or (v.ndim == 1 and v.shape == (v.size, 1) and v.size > 0) or (v.ndim == 2 and v.shape == (int(v.size ** 0.5), int(v.size ** 0.5)) and v.size > 0) for v in stats.values()), f"{dist} stats values are not scalars, 1D-vectors or 2D-square matrices"
    # Check jit
    jitted_stats = jit(dist.stats)(params=params)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_loglikelihood(dist, dataset, datasets):
    data = datasets[dataset]
    params = dist.example_params()

    if dataset in ERROR_CASES:
        # Check error cases - dim mismatch between data and params
        with pytest.raises(TypeError):
            output = dist.loglikelihood(data, params=params)
    else:
        # Check non-error cases
        output = dist.loglikelihood(data, params=params)

        # Check properties
        check_metric_output(dist, output, 'loglikelihood')

        # Check jit
        jitted_loglikelihood = jit(dist.loglikelihood)(data, params=params)

        # Check gradients
        gradients(dist.loglikelihood, f"{dist} loglikelihood", data, params)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_aic(dist, dataset, datasets):
    data = datasets[dataset]
    params = dist.example_params()

    if dataset in ERROR_CASES:
        # Check error cases - dim mismatch between data and params
        with pytest.raises(TypeError):
            output = dist.aic(data, params=params)
    else:
        # Check non-error cases
        output = dist.aic(data, params=params)

        # Check properties
        check_metric_output(dist, output, 'aic')

        # Check jit
        jitted_aic = jit(dist.aic)(data, params=params)

        # Check gradients
        gradients(dist.aic, f"{dist} aic on {dataset}", data, params)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_bic(dist, dataset, datasets):
    data = datasets[dataset]
    params = dist.example_params()

    if dataset in ERROR_CASES:
        # Check error cases - dim mismatch between data and params
        with pytest.raises(TypeError):
            output = dist.bic(data, params=params)
    else:
        # Check non-error cases
        output = dist.bic(data, params=params)

        # Check properties
        check_metric_output(dist, output, 'bic')

        # Check jit
        jitted_bic = jit(dist.bic)(data, params=params)
        
        # Check gradients
        gradients(dist.bic, f"{dist} bic", data, params)