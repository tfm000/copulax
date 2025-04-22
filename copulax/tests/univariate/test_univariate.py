"""Tests for univariate probability distributions."""
import numpy as np
from jax import jit, grad
import inspect
import pytest
from jax import numpy as jnp

from copulax._src._distributions import Univariate
from copulax.univariate import distributions
from copulax.univariate.distributions import _all_dists, skewed_t
from copulax._src.typing import Scalar
from copulax.tests.helpers import *


# Helper functions for testing univariate distributions
def get_data(dist, continuous_data, discrete_data):
    """Get data for testing."""
    if dist.dtype == 'continuous':
        return continuous_data
    elif dist.dtype == 'discrete':
        return discrete_data
    else:
        raise ValueError(f"Unknown distribution type: {dist.dtype}")
    

def check_params(params, s):
    assert isinstance(params, dict), f"{s} params is not a dict"
    assert len(params) > 0, f"{s} is empty"
    assert all(isinstance(k, str) for k in params.keys()), f"{s} params keys are not strings"
    assert all(isinstance(v, Scalar) for v in params.values()), f"{s} params values are not scalars"
    assert all(v.ndim == 0 and v.shape == () and v.size == 1 for v in params.values()), f"{s} params values are not scalars"
    assert any(jnp.isnan(v) for v in params.values()) == False, f"{s} params values are NaN"
    assert any(jnp.isfinite(v) for v in params.values()) == True, f"{s} params values are not finite"

def check_metric_output(dist, output, metric_name):
    assert isinstance(output, Scalar) and output.shape == () and output.size == 1, f"{dist} {metric_name} is non-scalar."
    assert no_nans(output), f"{dist} {metric_name} contains NaNs."
    # assert is_finite(output), f"{dist} {metric_name} contains non-finite values."


@jit
def jittable(dist):
    return dist.example_params()


# Test combinations
DISTRIBUTIONS = tuple(value for value in distributions.values() if isinstance(value, Univariate)) # note has to be a tuple otherwise pytest parametrize will remove elements as it scans along, causing test skipping
IVTF_DISTRIBUTIONS = tuple()
UNNORMALISED_DISTRIBUTIONS = tuple((skewed_t.name,))
rvs_sizes = (
        ((), 1),
        ((1, ), 1),
        ((1, 1), 1),
        ((3, ), 3),
        ((3, 1), 3),
        ((3, 2), 6),
        (0, 0),
        (1, 1),
        (3, 3),
        )
RVS_COMBINATIONS = tuple((dist, size) for dist in DISTRIBUTIONS for size in rvs_sizes)
IVTF_RVS_COMBINATIONS = tuple((dist, size) for dist in IVTF_DISTRIBUTIONS for size in rvs_sizes)

# Tests

@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_all_objects(dist):
    assert isinstance(dist, Univariate), f"{dist} is not a Univariate object."


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_all_methods_implemented(dist):
    methods: set[str] = {
        'support', 'logpdf', 'pdf', 'logcdf', 'cdf', 'ppf', 'inverse_cdf', 
        'rvs', 'sample', 'fit', 'stats', 'loglikelihood', 'aic', 'bic', 'dtype', 
        'dist_type', 'name', 'example_params'
        }
    
    for method_name in methods:
        assert hasattr(dist, method_name), f"{dist} missing {method_name} method."

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
    assert dist.dist_type == 'univariate', f"dist_type is not 'univariate' for {dist}"


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_example_params(dist):
    params = dist.example_params()

    # testing properties
    s = f'{dist} example_params'
    check_params(params, s)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_dist_object_jittable(dist):
    jittable(dist)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_support(dist):
    params: dict = dist.example_params()
    support = dist.support(params)
    
    # testing properties
    assert isinstance(support, jnp.ndarray), f"{dist} support is not a JAX array"
    assert support.size == 2, f"{dist} does not contain 2 elements"
    assert support.ndim == 1 and support.shape == (2,), f"{dist} support is not a flattened array"
    assert np.all(support[0] < support[1]), f"{dist} support bounds are not in order."
    assert no_nans(support), f"{dist} support contains NaNs."
    
    # testing jit works
    jitted_support = jit(dist.support)(params)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_logpdf(dist, continuous_data, discrete_data):
    data: jnp.ndarray = get_data(dist, continuous_data, discrete_data)
    params: dict = dist.example_params()

    output = dist.logpdf(data, params)
    
    # testing properties
    correct_uvt_shape(data, output, dist, "logpdf")
    assert no_nans(output), f"{dist.name} logpdf contains NaNs."
    
    # testing jit works
    jitted_pdf = jit(dist.logpdf)(data, params)
    
    # testing gradients
    gradients(dist.logpdf, f"{dist.name} logpdf", data, params, params_error=dist.name not in UNNORMALISED_DISTRIBUTIONS)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_pdf(dist, continuous_data, discrete_data):
    data: jnp.ndarray = get_data(dist, continuous_data, discrete_data)
    params: dict = dist.example_params()

    output = dist.pdf(data, params)

    # Check properties
    correct_uvt_shape(data, output, dist, "pdf")
    assert no_nans(output), f"{dist} pdf contains NaNs."
    assert is_positive(output), f"{dist} pdf contains negative values."
    assert is_finite(output), f"{dist} pdf contains non-finite values."

    # testing jit works
    jitted_pdf = jit(dist.pdf)(data, params)
    
    # testing gradients
    gradients(dist.pdf, f"{dist.name} pdf", data, params, params_error=dist.name not in UNNORMALISED_DISTRIBUTIONS)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_logcdf(dist, continuous_data, discrete_data):
    data: jnp.ndarray = get_data(dist, continuous_data, discrete_data)
    params: dict = dist.example_params()

    output = dist.logcdf(data, params)
    
    # testing properties
    correct_uvt_shape(data, output, dist, "logcdf")
    assert no_nans(output), f"{dist} logcdf contains NaNs."
    
    # testing jit works
    jitted_logcdf = jit(dist.logcdf)(data, params)
    
    # testing gradients
    gradients(dist.logcdf, f"{dist.name} logcdf", data, params, params_error=dist.name not in UNNORMALISED_DISTRIBUTIONS)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_cdf(dist, continuous_data, discrete_data):
    data: jnp.ndarray = get_data(dist, continuous_data, discrete_data)
    params: dict = dist.example_params()

    output = dist.cdf(data, params)

    # Check properties
    correct_uvt_shape(data, output, dist, "cdf")
    assert no_nans(output), f"{dist} cdf contains NaNs."
    assert is_positive(output), f"{dist} cdf contains negative values."
    assert is_finite(output), f"{dist} cdf contains non-finite values."
    assert np.all(0 <= output) and np.all(output <= 1), f"cdf not in [0, 1] range for {dist}"
    
    # testing jit works
    jitted_cdf = jit(dist.cdf)(data, params)
    
    # testing gradients
    gradients(dist.cdf, f"{dist.name} cdf", data, params, params_error=dist.name not in UNNORMALISED_DISTRIBUTIONS)


def _ppf(dist, uniform_data, cubic):
    """Test the ppf function."""
    data: jnp.ndarray = uniform_data
    params: dict = dist.example_params()
    support = dist.support(params)
    s: str = " with cubic spline approximation" if cubic else ""

    output = dist.ppf(data, params, cubic=cubic, maxiter=5)
    
    # testing properties
    correct_uvt_shape(data, output, dist, "ppf")
    assert no_nans(output), f"{dist} ppf contains NaNs{s}."
    assert is_finite(output), f"{dist} ppf contains non-finite values{s}."
    assert np.all(output >= support[0]) and np.all(output <= support[1]), f"ppf not in support range for {dist}{s}."

    # testing jit works
    jitted_ppf_func = jit(dist.ppf, static_argnames='cubic')
    jitted_ppf = jitted_ppf_func(data, params, cubic=cubic)

    # testing gradients
    gradients(jitted_ppf_func, f"{dist.name} ppf", data, params, cubic=cubic)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_ppf(dist, uniform_data):
    _ppf(dist, uniform_data, cubic=False)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_ppf_cubic(dist, uniform_data):
    _ppf(dist, uniform_data, cubic=True)


def _rvs(dist, tup):
    """Test the rvs function."""
    params: dict = dist.example_params()
    size, num = tup

    output = dist.rvs(size, params)

    # testing properties
    if isinstance(size, int):
        shape = (size, )
    else:
        shape = size
    assert isinstance(output, jnp.ndarray), f"{dist} rvs is not a JAX array."
    assert no_nans(output), f"{dist} rvs contains NaNs."
    assert is_finite(output), f"{dist} rvs contains non-finite values."
    assert output.size == num, f"{dist} rvs size mismatch."
    assert output.shape == shape, f"{dist} rvs shape mismatch."

    # testing jit works
    jitted_rvs = jit(dist.rvs, static_argnames='size')(size=size, params=params)


@pytest.mark.parametrize("dist, size", RVS_COMBINATIONS)
def test_rvs(dist, size):
    _rvs(dist, size)


@pytest.mark.local_only
@pytest.mark.parametrize("dist, size", IVTF_RVS_COMBINATIONS)
def test_rvs_ivtf(dist, size):
    _rvs(dist, size)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_fit(dist, continuous_data, discrete_data):
    data: jnp.ndarray = get_data(dist, continuous_data, discrete_data)

    fitted_params = dist.fit(data)

    # testing properties
    check_params(fitted_params, f"{dist} fitted")
    assert set(fitted_params.keys()) == set(dist.example_params().keys()), f"{dist} fitted params and example_params mismatch."
    
    # testing jit works
    fit_args: list = inspect.getfullargspec(dist.fit).args
    if 'method' in fit_args:
        jit_fit = jit(dist.fit, static_argnames='method')(continuous_data)
    else:
        jit_fit = jit(dist.fit)(continuous_data)
    check_params(fitted_params, f"{dist} jit fitted")


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_stats(dist):
    stats = dist.stats(dist.example_params())

    # testing properties    
    assert isinstance(stats, dict), f"{dist} stats outputted wrong type"
    assert len(stats) > 0, f"{dist} stats is empty"
    assert all(isinstance(k, str) for k in stats.keys()), f"{dist} stats keys are not strings"
    assert all(isinstance(v, Scalar) for v in stats.values()), f"{dist} stats values are not scalars"
    assert all(v.ndim == 0 and v.shape == () and v.size == 1 for v in stats.values()), f"{dist} stats values are not scalars"


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_loglikelihood(dist, continuous_data, discrete_data):
    data: jnp.ndarray = get_data(dist, continuous_data, discrete_data)
    params: dict = dist.example_params()

    output = dist.loglikelihood(data, params)

    # testing properties
    check_metric_output(dist, output, "loglikelihood")

    # testing jit works
    jitted_loglikelihood = jit(dist.loglikelihood)(data, params)

    # testing gradients
    gradients(dist.loglikelihood, f"{dist.name} loglikelihood", data, params)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_aic(dist, continuous_data, discrete_data):
    data: jnp.ndarray = get_data(dist, continuous_data, discrete_data)
    params: dict = dist.example_params()

    output = dist.aic(data, params)

    # testing properties
    check_metric_output(dist, output, "aic")

    # testing jit works
    jitted_aic = jit(dist.aic)(data, params)

    # testing gradients
    gradients(dist.aic, f"{dist.name} aic", data, params)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_bic(dist, continuous_data, discrete_data):
    data: jnp.ndarray = get_data(dist, continuous_data, discrete_data)
    params: dict = dist.example_params()

    output = dist.bic(data, params)

    # testing properties
    check_metric_output(dist, output, "bic")

    # testing jit works
    jitted_bic = jit(dist.bic)(data, params)

    # testing gradients
    gradients(dist.bic, f"{dist.name} bic", data, params)
