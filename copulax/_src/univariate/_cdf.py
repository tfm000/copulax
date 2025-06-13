"""Implements a jit-able, jax-differentiable version of numerical univariate cdf integration."""
from jax import numpy as jnp
from jax import grad, lax
from typing import Callable
from quadax import quadgk, quadcc


from copulax._src.univariate._utils import _univariate_input


METHOD: Callable = quadgk


def _cdf_single_x(pdf_func: Callable, lower_bound: float, xi: float, params_array) -> float:
    cdf_vals, info = METHOD(fun=pdf_func, interval=[lower_bound, xi], args=params_array, )
    return cdf_vals.reshape(())


def _cdf(dist, x: jnp.ndarray, params: dict) -> jnp.ndarray:
    # adding right bound to the cdf integral
    x, xshape = _univariate_input(x)
    xsize = x.size
    lower_bound, upper_bound = dist.support(params)
    x = jnp.append(x, upper_bound.reshape((1, 1)), axis=0)

    params_array: jnp.ndarray = dist._params_to_array(params)

    def _iter(carry, xi):
        cdf_i = _cdf_single_x(dist._pdf_for_cdf, lower_bound, xi, params_array)
        return carry, cdf_i
    
    _, cdf_raw_ = lax.scan(_iter, None, x.flatten())

    # ensuring the cdf is scaled to be between 0 and 1
    cdf_raw = lax.dynamic_slice_in_dim(cdf_raw_, 0, xsize, axis=0)
    scale = lax.dynamic_slice_in_dim(cdf_raw_, xsize, 1, axis=0)
    cdf_adj = cdf_raw / scale
    cdf_adj = jnp.where(cdf_adj> 1.0, 1.0, cdf_adj)
    cdf_adj = jnp.where(cdf_adj < 0.0, 0.0, cdf_adj)

    return cdf_adj.reshape(xshape)


def _cdf_fwd(dist, cdf_func: Callable, x: jnp.ndarray, params: dict):
    x, xshape = _univariate_input(x)

    def cdf_single(xi, params):
        return cdf_func(xi, params).reshape(())

    def iter(carry, xi):
        params_grad_i = grad(cdf_single, argnums=1)(xi, params)
        return carry, params_grad_i

    _, param_grads = lax.scan(iter, None, x.flatten())
    pdf_values = dist.pdf(x=x, params=params).reshape(xshape)
    return cdf_func(x=x, params=params).reshape(xshape), (pdf_values, param_grads)


def cdf_bwd(res, g):
    xshape = res[0].shape
    g = g.reshape(xshape)
    x_grad = res[0] * g
    param_grads: dict = {key: jnp.mean(jnp.nan_to_num(val, 0.0) * g) 
                         for key, val in res[1].items()}  # average parameter gradients over x
    return x_grad, param_grads
