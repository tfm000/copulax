"""Tests for copula distributions."""
import pytest
import jax.numpy as jnp
from jax import grad, jit
import numpy as np

from copulax._src._distributions import Univariate, Multivariate
from copulax._src.copulas._distributions import Copula
from copulax.tests.helpers import *
from copulax.tests.copulas.conftest import NUM_ASSETS
from copulax._src.typing import Scalar
from copulax.copulas import gaussian_copula, student_t_copula, skewed_t_copula, gh_copula


# Test combinations 
DISTRIBUTIONS = tuple((gaussian_copula, student_t_copula, 
                       skewed_t_copula, gh_copula))
ERROR_CASES = tuple(('too_large_dim_sample',))
DATASETS = tuple(('uncorrelated_sample', 'correlated_sample'))
COMBINATIONS = tuple((dist, dataset) for dist in DISTRIBUTIONS for dataset in DATASETS)


# def gradients(func, params: dict, s, data):
#     """Calculate the gradients of the output."""
#     new_func = lambda x: func(x, params=params).sum()
#     grad_output = grad(new_func)(data)
#     assert no_nans(grad_output), f"{s} gradient contains NaNs"
#     assert is_finite(grad_output), f"{s} gradient contains non-finite values"


def _check_marginals(dist, dim, fitted_marginals):
    # Checking properties
    assert isinstance(fitted_marginals, dict), f"{dist} fitted_marginals is not a dictionary."
    assert len(fitted_marginals) == 1, f"{dist} fitted_marginals does not have length 1."
    assert 'marginals' in fitted_marginals, f"{dist} fitted_marginals does not have 'marginals' key."
    marginals: tuple = fitted_marginals['marginals']
    assert isinstance(marginals, tuple), f"{dist} fitted_marginals subdict is not a tuple."
    fitted_num_dims = len(marginals)
    assert fitted_num_dims == dim, f"{dist} fitted_marginals has not fitted the correct number of dimensions. Expected {dim}, got {fitted_num_dims}."
    for marginal_tup in marginals:
        assert isinstance(marginal_tup, tuple), f"{dist} underlying fitted_marginals are not in tuple form."
        assert len(marginal_tup) == 2, f"{dist} underlying fitted_marginals tuples do not have length 2."
        marginal_dist, marginal_params = marginal_tup
        assert isinstance(marginal_dist, Univariate), f"{dist} underlying marginals distribution is not a Univariate object."
        assert isinstance(marginal_params, dict), f"{dist} underlying marginals parameters are not in dictionary form."
        check_uvt_params(marginal_params, f"{dist} underlying marginals parameters {marginal_params} are not valid.")


def _check_copula_params(dist, fitted_copula):
    # Checking properties
    assert isinstance(fitted_copula, dict), f"{dist} fitted_copula is not a dictionary."
    assert len(fitted_copula) == 1, f"{dist} fitted_copula does not have length 1."
    assert 'copula' in fitted_copula, f"{dist} fitted_copula does not have 'copula' key."
    copula_params = fitted_copula['copula']
    check_mvt_params(copula_params, f"{dist} copula_params")


# Tests for copula distributions
@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_all_objects(dist):
    assert isinstance(dist, Copula), f"{dist} is not a Copula object"
    assert isinstance(dist._mvt, Multivariate), f"{dist} mvt object is not a Multivariate object"
    assert isinstance(dist._uvt, Univariate), f"{dist} uvt object is not a Univariate object"


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_all_methods_implemented(dist):
    methods: set[str] = {
        'dtype', 'dist_type', 'name', 'support', 'get_u', 'get_x_dash', 
        'copula_logpdf', 'copula_pdf', 'logpdf', 'pdf', 'copula_rvs', 
        'copula_sample', 'rvs', 'sample', 'fit_marginals', 'fit_copula', 
        'fit', 'aic', 'bic', 'loglikelihood', 'stats', 'example_params',
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
    assert dist.dist_type == 'copula', f"dist_type is not 'copula' for {dist}"


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_example_params(dist):
    params = dist.example_params(dim=NUM_ASSETS)

    # checking properties
    s = f"{dist} example_params"
    assert len(params) == 2 and 'marginals' in params and 'copula' in params, f"{s} does not have 'marginals' and 'copula' keys"
    
    # Check copula parameters
    copula_params = params.copy()
    copula_params.pop('marginals')
    _check_copula_params(dist, copula_params)

    # Check marginals parameters
    marginal_params = params.copy()
    marginal_params.pop('copula')
    _check_marginals(dist, NUM_ASSETS, marginal_params)


    for marginal, params in params['marginals']:
        assert isinstance(marginal, Univariate), f"{s} marginal {marginal} is not a Univariate object"
        check_uvt_params(params, f"{s} marginal {marginal} does not have valid params: {params}")


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_dist_object_jitable(dist):
    jittable(dist)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_support(dist):
    params = dist.example_params()
    support = dist.support(params)

    # Checking properties
    assert isinstance(support, jnp.ndarray), f"{dist} support is not a JAX array"
    assert two_dim(support), f"{dist} support is not a 2D array"
    assert np.all(support[:, 0] < support[:, 1]), f"{dist} support bounds are not in order."
    assert no_nans(support), f"{dist} support contains NaNs."

    # Check jit
    jitted_support = jit(dist.support)(params)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_get_u(dist, dataset, datasets):
    params = dist.example_params()
    sample = datasets[dataset]

    if dataset in ERROR_CASES:
        # Check error cases - dim mismatch between data and params
        with pytest.raises(TypeError):
            u = dist.get_u(sample, params)
    else:
        # Check non-error cases
        u = dist.get_u(sample, params)
        # Checking properties
        assert isinstance(u, jnp.ndarray), f"{dist} u is not a JAX array"
        assert two_dim(u), f"{dist} u is not a 2D array"
        assert np.all(u >= 0) and np.all(u <= 1), f"{dist} u is not in [0, 1]"
        assert no_nans(u), f"{dist} u contains NaNs."

        # Check jit
        jitted_u = jit(dist.get_u)(sample, params)

        # Check gradients
        gradients(dist.get_u, f"{dist} u", sample, params)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_get_x_dash(dist, u_sample):
    params = dist.example_params()
    x_dash = dist.get_x_dash(u_sample, params)
    
    # Checking properties
    assert isinstance(x_dash, jnp.ndarray), f"{dist} x_dash is not a JAX array"
    assert two_dim(x_dash), f"{dist} x_dash is not a 2D array"
    assert no_nans(x_dash), f"{dist} x_dash contains NaNs."

    # Check jit
    jitted_x_dash = jit(dist.get_x_dash)(u_sample, params)

    # Check gradients
    gradients(dist.get_x_dash, f"{dist} x_dash", u_sample, params)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_copula_logpdf(dist, u_sample):
    params = dist.example_params()
    logpdf = dist.copula_logpdf(u_sample, params)

    # Checking properties
    correct_mvt_shape(u_sample, logpdf, dist, 'copula_logpdf')
    assert no_nans(logpdf), f"{dist} copula_logpdf contains NaNs."

    # Check jit
    jitted_logpdf = jit(dist.copula_logpdf)(u_sample, params)

    # Check gradients
    gradients(dist.copula_logpdf, f"{dist} copula_logpdf", u_sample, params)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_copula_pdf(dist, u_sample):
    params = dist.example_params()
    pdf = dist.copula_pdf(u_sample, params)

    # Checking properties
    correct_mvt_shape(u_sample, pdf, dist, 'copula_pdf')
    assert is_positive(pdf), f"{dist} copula_pdf is not positive."
    assert no_nans(pdf), f"{dist} copula_pdf contains NaNs."
    assert is_finite(pdf), f"{dist} copula_pdf contains non-finite values."

    # Check jit
    jitted_pdf = jit(dist.copula_pdf)(u_sample, params)

    # Check gradients
    gradients(dist.copula_pdf, f"{dist} copula_pdf", u_sample, params)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_logpdf(dist, dataset, datasets):
    params = dist.example_params()
    sample = datasets[dataset]

    if dataset in ERROR_CASES:
        # Check error cases - dim mismatch between data and params
        with pytest.raises(TypeError):
            logpdf = dist.logpdf(sample, params)

    else:
        logpdf = dist.logpdf(sample, params)

        # Checking properties
        correct_mvt_shape(sample, logpdf, dist, 'logpdf')
        assert no_nans(logpdf), f"{dist} logpdf contains NaNs."

        # Check jit
        jitted_logpdf = jit(dist.logpdf)(sample, params)

        # Check gradients
        gradients(dist.logpdf, f"{dist} logpdf", sample, params)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_pdf(dist, dataset, datasets):
    params = dataset['params']
    sample = datasets[dataset]

    if dataset in ERROR_CASES:
        # Check error cases - dim mismatch between data and params
        with pytest.raises(TypeError):
            pdf = dist.pdf(sample, params)
    else:
        # Check non-error cases
        pdf = dist.pdf(sample, params)

        # Checking properties
        correct_mvt_shape(sample, pdf, dist, 'pdf')
        assert is_positive(pdf), f"{dist} pdf contains negative values."
        assert no_nans(pdf), f"{dist} pdf contains NaNs."
        assert is_finite(pdf), f"{dist} pdf contains non-finite values."

        # Check jit
        jitted_pdf = jit(dist.pdf)(sample, params)

        # Check gradients
        gradients(dist.pdf, f"{dist} pdf", sample, params)


SIZES = (0, 1, 2, 11)
SIZE_COMBINATIONS = tuple((dist, size) for dist in DISTRIBUTIONS for size in SIZES)
@pytest.mark.parametrize("dist, size", SIZE_COMBINATIONS)
def test_copula_rvs(dist, size):
    params = dist.example_params()
    rvs = dist.copula_rvs(size=size, params=params)

    # Checking properties
    assert two_dim(rvs), f"{dist} copula_rvs is not a 2D array"
    expected_shape = (size, NUM_ASSETS)
    assert rvs.shape == expected_shape, f"{dist} copula_rvs has incorrect shape. Expected {expected_shape}, got {rvs.shape}"
    assert no_nans(rvs), f"{dist} copula_rvs contains NaNs for size {size}"
    assert is_finite(rvs), f"{dist} copula_rvs contains infinite values for size {size}"
    
    # Check jit
    jitted_rvs = jit(dist.copula_rvs, static_argnums=0)(size=size, params=params)


@pytest.mark.parametrize("dist, size", SIZE_COMBINATIONS)
def test_rvs(dist, size):
    params = dist.example_params()
    rvs = dist.rvs(size=size, params=params)

    # Checking properties
    assert two_dim(rvs), f"{dist} rvs is not a 2D array"
    expected_shape = (size, NUM_ASSETS)
    assert rvs.shape == expected_shape, f"{dist} rvs has incorrect shape. Expected {expected_shape}, got {rvs.shape}"
    assert no_nans(rvs), f"{dist} rvs contains NaNs for size {size}"
    assert is_finite(rvs), f"{dist} rvs contains infinite values for size {size}"

    # Check jit
    jitted_rvs = jit(dist.rvs, static_argnums=0)(size=size, params=params)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_fit_marginals(dist, dataset, datasets):
    sample = datasets[dataset]
    fitted_marginals = dist.fit_marginals(sample)

    # Checking properties
    _check_marginals(dist, sample.shape[1], fitted_marginals)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_fit_copula(dist, u_sample):
    fitted_copula = dist.fit_copula(u_sample)
    # Checking properties
    _check_copula_params(dist, fitted_copula)
    # Check jit
    jitted_fitted_copula = jit(dist.fit_copula)(u_sample)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_fit(dist, dataset, datasets):
    sample = datasets[dataset]
    fitted_joint = dist.fit(sample)
    
    # Checking properties
    assert isinstance(fitted_joint, dict), f"{dist} fit does not return a dictionary"
    assert len(fitted_joint) == 2, f"{dist} fit does not have length 2."
    assert 'marginals' in fitted_joint and 'copula' in fitted_joint, f"{dist} fit does not have 'marginals' and 'copula' keys."
    fitted_marginals = fitted_joint.copy()
    fitted_marginals.pop('copula')
    _check_marginals(dist, sample.shape[1], fitted_marginals)
    fitted_copula = fitted_joint.copy()
    fitted_copula.pop('marginals')
    _check_copula_params(dist, fitted_copula)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_aic(dist, dataset, datasets):
    params = dist.example_params()
    sample = datasets[dataset]

    if dataset in ERROR_CASES:
        # Check error cases - dim mismatch between data and params
        with pytest.raises(TypeError):
            output = dist.aic(sample, params)
    else:
        output = dist.aic(sample, params)

        # Checking properties
        check_metric_output(dist, output, 'aic')

        # Check jit
        jitted_aic = jit(dist.aic)(sample, params)

        # Check gradients
        gradients(dist.aic, f"{dist} aic", sample, params)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_bic(dist, dataset, datasets):
    params = dist.example_params()
    sample = datasets[dataset]

    if dataset in ERROR_CASES:
        # Check error cases - dim mismatch between data and params
        with pytest.raises(TypeError):
            output = dist.bic(sample, params)
    else:
        output = dist.bic(sample, params)

        # Checking properties
        check_metric_output(dist, output, 'bic')

        # Check jit
        jitted_bic = jit(dist.bic)(sample, params)

        # Check gradients
        gradients(dist.bic, f"{dist} bic", sample, params)


@pytest.mark.parametrize("dist, dataset", COMBINATIONS)
def test_loglikelihood(dist, dataset, datasets):
    params = dist.example_params()
    sample = datasets[dataset]

    if dataset in ERROR_CASES:
        # Check error cases - dim mismatch between data and params
        with pytest.raises(TypeError):
            output = dist.loglikelihood(sample, params)
    else:
        output = dist.loglikelihood(sample, params)

        # Checking properties
        check_metric_output(dist, output, 'loglikelihood')

        # Check jit
        jitted_loglikelihood = jit(dist.loglikelihood)(sample, params)

        # Check gradients
        gradients(dist.loglikelihood, f"{dist} loglikelihood", sample, params)
