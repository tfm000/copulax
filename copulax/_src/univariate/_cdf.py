"""Implements a jit-able, jax-differentiable version of numerical univariate cdf integration."""

from jax import numpy as jnp
from jax import grad, vmap, value_and_grad
from typing import Callable
from quadax import quadgk, quadcc


from copulax._src.univariate._utils import _univariate_input


METHOD: Callable = quadgk


def _cdf_single_x(
    pdf_func: Callable, lower_bound: float, xi: float, params_array
) -> float:
    """Compute the CDF at a single point by numerical integration of the PDF."""
    cdf_vals, info = METHOD(fun=pdf_func, interval=(lower_bound, xi), args=params_array)
    return cdf_vals.reshape(())


def _cdf(dist, x: jnp.ndarray, params: dict) -> jnp.ndarray:
    """Compute the CDF by numerically integrating the PDF from the lower support bound.

    Assumes the PDF is analytically normalised (integrates to 1). Use
    ``_cdf_normalised`` instead when the PDF may not integrate exactly to 1
    (e.g. due to numerical issues in the log-density).
    """
    x, xshape = _univariate_input(x)
    lower_bound, upper_bound = dist.support(params)
    params_array: jnp.ndarray = dist._params_to_array(params)

    # vectorize CDF computation across all x values
    _cdf_vec = vmap(
        lambda xi: _cdf_single_x(dist._pdf_for_cdf, lower_bound, xi, params_array)
    )
    cdf_raw = _cdf_vec(x.flatten())
    cdf_adj = jnp.clip(cdf_raw, 0.0, 1.0)

    return cdf_adj.reshape(xshape)


def _cdf_normalised(dist, x: jnp.ndarray, params: dict) -> jnp.ndarray:
    """Compute the CDF with explicit normalisation by the total PDF integral.

    Divides the raw integral by the full-support integral so that the CDF
    reaches exactly 1.  This is necessary when the PDF implementation has
    known numerical inaccuracies (e.g. Bessel-function underflow) that
    prevent the density from integrating to 1.
    """
    x, xshape = _univariate_input(x)
    lower_bound, upper_bound = dist.support(params)
    params_array: jnp.ndarray = dist._params_to_array(params)

    # compute normalising constant (full-support integral) once
    scale = _cdf_single_x(dist._pdf_for_cdf, lower_bound, upper_bound, params_array)

    # vectorize CDF computation across all x values
    _cdf_vec = vmap(
        lambda xi: _cdf_single_x(dist._pdf_for_cdf, lower_bound, xi, params_array)
    )
    cdf_raw = _cdf_vec(x.flatten())

    # scale to [0, 1]
    cdf_adj = cdf_raw / scale
    cdf_adj = jnp.clip(cdf_adj, 0.0, 1.0)

    return cdf_adj.reshape(xshape)


def _cdf_fwd(dist, cdf_func: Callable, x: jnp.ndarray, params: dict):
    """Forward pass for the custom CDF VJP: returns CDF values and residuals for backward."""
    x, xshape = _univariate_input(x)

    def cdf_single(xi, params):
        return cdf_func(xi, params).reshape(())

    # vmap value_and_grad to parallelize across x values
    _val_and_grad = value_and_grad(cdf_single, argnums=1)
    _val_and_grad_vec = vmap(lambda xi: _val_and_grad(xi, params))

    cdf_values, param_grads = _val_and_grad_vec(x.flatten())
    pdf_values = dist.pdf(x=x, params=params).reshape(xshape)
    return cdf_values.reshape(xshape), (pdf_values, param_grads)


def cdf_bwd(res, g):
    """Backward pass for the custom CDF VJP: computes gradients w.r.t. x and params."""
    xshape = res[0].shape
    g = g.reshape(xshape)
    x_grad = res[0] * g
    param_grads: dict = {
        key: jnp.sum(jnp.nan_to_num(val, 0.0) * g) for key, val in res[1].items()
    }  # sum parameter gradients over x
    return x_grad, param_grads
