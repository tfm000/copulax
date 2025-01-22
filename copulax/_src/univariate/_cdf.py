"""Implements a jit-able, jax-differentiable version of numerical univariate cdf integration."""
from jax import numpy as jnp
from jax import grad, lax
from typing import Callable
from quadax import quadgk, quadcc


from copulax._src.univariate._utils import _univariate_input


def _cdf_single_x(pdf_func: Callable, lower_bound: float, xi: float, params: dict) -> float:
    cdf_vals, info = quadcc(fun=pdf_func, interval=[lower_bound, xi], args=params.values(), )#max_ninter=200, order=61)
    return cdf_vals.reshape(())


def _cdf(pdf_func: Callable, lower_bound: float, x: jnp.ndarray, params: dict) -> jnp.ndarray:
    x, xshape = _univariate_input(x)
    # sorted_index = jnp.argsort(x, axis=0)
    # reverse_index = jnp.argsort(sorted_index, axis=0)

    def _iter(carry, xi):
        cdf_i = _cdf_single_x(pdf_func, lower_bound, xi, params)
        return carry, cdf_i
    
    _, cdf_raw = lax.scan(_iter, None, x.flatten())

    cdf_adj = jnp.where(cdf_raw > 1.0, 1.0, cdf_raw)
    cdf_adj = jnp.where(cdf_adj < 0.0, 0.0, cdf_adj)

    return cdf_adj.reshape(xshape)


def _cdf_fwd(pdf_func, cdf_func: Callable, x: jnp.ndarray, params: dict):
    x, xshape = _univariate_input(x)

    def cdf_single(xi, params):
        return cdf_func(xi, **params).reshape(())

    def iter(carry, xi):
        params_grad_i = grad(cdf_single, argnums=1)(xi, params)
        return carry, params_grad_i

    _, param_grads = lax.scan(iter, None, x.flatten())
    pdf_values = pdf_func(x, **params).reshape(xshape)
    return cdf_func(x, **params).reshape(xshape), (pdf_values, param_grads)


def cdf_bwd(res, g):
    xshape = res[0].shape
    g = g.reshape(xshape)
    x_grad = res[0] * g
    param_grads = tuple([jnp.mean(res_i * g) for res_i in res[1:]])  # average parameter gradients over x
    return x_grad, *param_grads
