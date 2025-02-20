"""Tests for univariate probability distributions."""
import numpy as np
from jax import jit, grad
import inspect
import pytest
from jax import numpy as jnp

from copulax._src.univariate._distributions import Univariate


def test_all_methods_implemented(continuous_dists):
    for dist in continuous_dists:
        for func_name in ('support', 'logpdf', 'pdf', 'logcdf', 'cdf', 'ppf', 'inverse_cdf', 'rvs', 'sample', 'fit', 'stats', 'loglikelihood', 'aic', 'bic', 'dtype', 'dist_type', 'name'):
            assert hasattr(dist, func_name), f"{dist} missing {func_name} method."


@jit
def jit_dist(dist, data):
    return dist.pdf(data)


def test_dist_object_jitable(continuous_data, continuous_dists):
    """Tests univariate distribution objects are jitable."""
    for dist in continuous_dists:
        assert isinstance(dist, Univariate), f"{dist} is not a subclass of Univariate"
        jit_dist(dist, continuous_data)



def test_name(continuous_dists):
    for dist in continuous_dists:
        assert isinstance(dist.name, str), f"name is not a string for {dist}"
        assert dist.name != "", f"name is empty for {dist}"


def test_dtype(continuous_dists):
    for dist in continuous_dists:
        assert isinstance(dist.dtype, str), f"dtype is not a string for {dist.name}"
        assert dist.dtype != "", f"dtype is empty for {dist.name}"
        assert dist.dtype in ('continuous', 'discrete'), f"dtype is not 'continuous' or 'discrete' for {dist.name}"


def test_dist_type(continuous_dists):
    for dist in continuous_dists:
        assert isinstance(dist.dist_type, str), f"dist_type is not a string for {dist.name}"
        assert dist.dist_type != "", f"dist_type is empty for {dist.name}"
        assert dist.dist_type  == 'univariate', f"dist_type is not 'univariate'for {dist.name}"


def test_support(continuous_dists):
    for dist in continuous_dists:
        name = dist.name

        # testing properties
        a, b = dist.support()
        assert isinstance(a, jnp.ndarray) and isinstance(b, jnp.ndarray), f"support is not a tuple of jnp.ndarrays for {name}"
        assert a.shape == () and b.shape == (), f"support is not a tuple of scalars for {name}"
        # assert isinstance(a, float) and isinstance(b, float), f"support is not a tuple of floats for {name}"
        assert a < b, f"support bounds are not in order for {name}"

        # testing jit works
        jit_support = jit(dist.support)
        a, b = jit_support()
        

def test_logpdf(continuous_data, continuous_dists):
    for dist in continuous_dists:
        name = dist.name

        # testing properties
        logpdf_vals = dist.logpdf(continuous_data)
        assert logpdf_vals.size == continuous_data.size, f"logpdf size mismatch for {name}"
        assert logpdf_vals.shape == continuous_data.shape, f"logpdf shape mismatch for {name}"
        assert np.all(np.isnan(logpdf_vals) == False), f"logpdf contains NaNs for {name}"

        # testing jit works
        jit_logpdf = jit(dist.logpdf)(continuous_data)

        # testing gradient works
        func = lambda x: np.sum(dist.logpdf(x))
        grad_logpdf = grad(func)(continuous_data)
        assert np.all(np.isnan(grad_logpdf) == False), f"gradient of logpdf contains NaNs for {name}"


def test_pdf(continuous_data, continuous_dists):
    for dist in continuous_dists:
        name = dist.name

        # testing properties
        pdf_vals = dist.pdf(continuous_data)
        assert pdf_vals.size == continuous_data.size, f"pdf size mismatch for {name}"
        assert pdf_vals.shape == continuous_data.shape, f"pdf shape mismatch for {name}"
        assert np.all(np.isfinite(pdf_vals)), f"pdf not finite for {name}"
        assert np.all(pdf_vals >= 0), f"pdf not positive for {name}"
        assert np.all(np.isnan(pdf_vals) == False), f"pdf contains NaNs for {name}"

        # testing jit works
        jit_pdf = jit(dist.pdf)(continuous_data)

        # testing gradient works
        func = lambda x: np.sum(dist.pdf(x))
        grad_pdf = grad(func)(continuous_data)
        assert np.all(np.isnan(grad_pdf) == False), f"gradient of pdf contains NaNs for {name}"
        assert np.all(np.isinf(grad_pdf) == False), f"gradient of pdf contains infs for {name}"


def test_logcdf(continuous_data, continuous_dists):
    for dist in continuous_dists:
        name = dist.name

        # testing properties
        logcdf_vals = dist.logcdf(continuous_data)
        assert logcdf_vals.size == continuous_data.size, f"logcdf size mismatch for {name}"
        assert logcdf_vals.shape == continuous_data.shape, f"logcdf shape mismatch for {name}"
        assert np.all(np.isnan(logcdf_vals) == False), f"logcdf contains NaNs for {name}"

        # testing jit works
        jit_logcdf = jit(dist.logcdf)(continuous_data)

        # testing gradient works
        func = lambda x: np.sum(dist.logcdf(x))
        grad_logcdf = grad(func)(continuous_data)
        assert np.all(np.isnan(grad_logcdf) == False), f"gradient of logcdf contains NaNs for {name}"


def test_cdf(continuous_data, continuous_dists):
    for dist in continuous_dists:
        name = dist.name
        
        # testing properties
        cdf_vals = dist.cdf(continuous_data)
        assert cdf_vals.size == continuous_data.size, f"cdf size mismatch for {name}"
        assert cdf_vals.shape == continuous_data.shape, f"cdf shape mismatch for {name}"
        assert np.all(np.isfinite(cdf_vals)), f"cdf not finite for {name}"
        assert np.all(0 <= cdf_vals) and np.all(cdf_vals <= 1), f"cdf not in [0, 1] range for {name}"
        assert np.all(np.isnan(cdf_vals) == False), f"cdf contains NaNs for {name}"

        # testing jit works
        jit_cdf = jit(dist.cdf)(continuous_data)

        # testing gradient works
        func = lambda x: np.sum(dist.cdf(x))
        grad_cdf = grad(func)(continuous_data)
        assert np.all(np.isnan(grad_cdf) == False), f"gradient of cdf contains NaNs for {name}"


def test_ppf(continuous_uniform_data, continuous_dists):
    for dist in continuous_dists:
        name = dist.name
        
        # testing properties
        ppf_vals = dist.ppf(continuous_uniform_data)
        assert ppf_vals.size == continuous_uniform_data.size, f"ppf size mismatch for {name}"
        assert ppf_vals.shape == continuous_uniform_data.shape, f"ppf shape mismatch for {name}"
        assert np.all(np.isnan(ppf_vals) == False), f"ppf contains NaNs for {name}"
        assert np.all(ppf_vals >= dist.support()[0]) & np.all(ppf_vals <= dist.support()[1]), f"ppf lies outside support for {name}"

        # testing jit works
        jit_ppf = jit(dist.ppf)(continuous_uniform_data)

        # testing gradient works
        func = lambda x: np.sum(dist.ppf(x))
        grad_ppf = grad(func)(continuous_uniform_data)
        assert np.all(np.isnan(grad_ppf) == False), f"gradient of ppf contains NaNs for {name}"


def _rvs(dists):
    gen = (
        ((), 1),
        ((1, ), 1),
        ((1, 1), 1),
        ((3, ), 3),
        ((3, 1), 3),
        ((3, 2), 6),
        )
    for dist in dists:
        name = dist.name
        for gen_shape, gen_size in gen:
            # testing properties
            sample = dist.rvs(gen_shape)
            assert sample.size == gen_size, f"rvs size mismatch for {name}"
            assert sample.shape == gen_shape, f"rvs shape mismatch for {name}"
            assert np.all(sample >= dist.support()[0]) & np.all(sample <= dist.support()[1]), f"rvs lies outside support for {name}"
            assert np.all(np.isnan(sample) == False), f"rvs contains NaNs for {name}"

            # testing jit works
            jit_rvs = jit(dist.rvs, static_argnames='shape')(gen_shape)


def test_rvs(non_inverse_transform_dists):
    _rvs(non_inverse_transform_dists)


@pytest.mark.local_only
def test_inverse_transform_rvs(inverse_transform_dists):
    _rvs(inverse_transform_dists)
    

def test_fit(continuous_data, continuous_dists):
    for dist in continuous_dists:
        name = dist.name
        
        # testing properties
        params: dict = dist.fit(continuous_data)
        assert isinstance(params, dict), f"fit outputted wrong type for {name}"
        params_array: np.ndarray = np.array(list(params.values()))
        assert np.all(np.isfinite(params_array)), f"fit produced infinite parameters for {name}"
        assert np.all(np.isnan(params_array) == False), f"fit produced nan parameters for {name}"

        # testing jit works
        fit_args: list = inspect.getfullargspec(dist.fit).args
        if 'method' in fit_args:
            jit_fit = jit(dist.fit, static_argnames='method')(continuous_data)
        else:
            jit_fit = jit(dist.fit)(continuous_data)


def test_stats(continuous_dists):
    for dist in continuous_dists:
        name = dist.name
        
        # testing properties
        stats = dist.stats()
        assert isinstance(stats, dict), f"stats outputted wrong type for {name}"


def test_loglikelihood(continuous_data, continuous_dists):
    for dist in continuous_dists:
        name = dist.name
        
        # testing properties
        loglike = dist.loglikelihood(continuous_data)
        # assert np.isfinite(loglike), f"loglikelihood produced infinite value for {name}"
        assert np.isnan(loglike) == False, f"loglikelihood produced nan value for {name}"

        # testing jit works
        jit_loglike = jit(dist.loglikelihood)(continuous_data)

        # testing gradient works
        func = lambda x: dist.loglikelihood(x)
        grad_loglike = grad(func)(continuous_data)
        assert np.all(np.isnan(grad_loglike) == False), f"gradient of loglikelihood contains NaNs for {name}"


def test_aic(continuous_data, continuous_dists):
    for dist in continuous_dists:
        name = dist.name
        
        # testing properties
        aic = dist.aic(continuous_data)
        # assert np.isfinite(aic), f"aic produced infinite value for {name}"
        assert np.isnan(aic) == False, f"aic produced nan value for {name}"

        # testing jit works
        jit_aic = jit(dist.aic)(continuous_data)

        # testing gradient works
        func = lambda x: dist.aic(x)
        grad_aic = grad(func)(continuous_data)
        assert np.all(np.isnan(grad_aic) == False), f"gradient of aic contains NaNs for {name}"


def test_bic(continuous_data, continuous_dists):
    for dist in continuous_dists:
        name = dist.name
        
        # testing properties
        bic = dist.bic(continuous_data)
        # assert np.isfinite(bic), f"bic produced infinite value for {name}"
        assert np.isnan(bic) == False, f"bic produced nan value for {name}"

        # testing jit works
        jit_bic = jit(dist.bic)(continuous_data)

        # testing gradient works
        func = lambda x: dist.bic(x)
        grad_bic = grad(func)(continuous_data)
        assert np.all(np.isnan(grad_bic) == False), f"gradient of bic contains NaNs for {name}"
