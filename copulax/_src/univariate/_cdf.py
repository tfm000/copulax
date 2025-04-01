"""Implements a jit-able, jax-differentiable version of numerical univariate cdf integration."""
from jax import numpy as jnp
from jax import grad, lax
from typing import Callable
from quadax import quadgk, quadcc


from copulax._src.univariate._utils import _univariate_input


def _cdf_single_x(pdf_func: Callable, lower_bound: float, xi: float, params_array) -> float:
    cdf_vals, info = quadgk(fun=pdf_func, interval=[lower_bound, xi], args=params_array, )#max_ninter=200, order=61)
    return cdf_vals.reshape(())


def _cdf(dist, x: jnp.ndarray, params: dict) -> jnp.ndarray:
    x, xshape = _univariate_input(x)
    params_array: jnp.ndarray = dist._params_to_array(params)
    lower_bound = dist.support(params)[0]
    # sorted_index = jnp.argsort(x, axis=0)
    # reverse_index = jnp.argsort(sorted_index, axis=0)

    def _iter(carry, xi):
        cdf_i = _cdf_single_x(dist._pdf_for_cdf, lower_bound, xi, params_array)
        return carry, cdf_i
    
    _, cdf_raw = lax.scan(_iter, None, x.flatten())

    cdf_adj = jnp.where(cdf_raw > 1.0, 1.0, cdf_raw)
    cdf_adj = jnp.where(cdf_adj < 0.0, 0.0, cdf_adj)

    return cdf_adj.reshape(xshape)


def _cdf_fwd(dist, cdf_func: Callable, x: jnp.ndarray, params: dict):
    x, xshape = _univariate_input(x)
    params_array: jnp.ndarray = dist._params_to_array(params)

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
    param_grads: dict = {key: jnp.mean(val * g) for key, val in res[1].items()}  # average parameter gradients over x
    return x_grad, param_grads
