"""File containing code for numerical approximations of univariate distribution 
functions."""
from jax._src.typing import ArrayLike, Array
import jax
import jax.numpy as jnp
from typing import Callable

from copulax._src.optimize import projected_gradient
from copulax._src.univariate._utils import _univariate_input

def _ppf(cdf_func: Callable, bounds: tuple, q: ArrayLike, params: dict, x0: float) -> Array:
    q, qshape = _univariate_input(q)
    x0 = jnp.asarray(x0, dtype=float).reshape((1,))

    @jax.jit
    def _ppf_func_single(xi: float, qi: float):
        return jnp.abs((cdf_func(xi, **params) - qi)).reshape(())
    
    min_val, max_val = jnp.asarray(bounds, dtype=float).reshape((2,1))
    SCALE = 0.5
    q0_small = jnp.max(jnp.array([x0 * (1-SCALE), min_val]))
    q0_large = jnp.min(jnp.array([x0 * (1+SCALE), max_val]))
    def _iter(carry, qi):
        q0 = jnp.where(qi <= 0.5, q0_small, q0_large)
        res = projected_gradient(
            f=_ppf_func_single, x0=q0, lr=0.1, 
            projection_method='projection_box', 
            projection_options={'hyperparams': bounds}, qi=qi)
        return carry, res['x']

    _, x = jax.lax.scan(_iter, None, q)

    x = jnp.where(jnp.logical_and(x >= bounds[0], x <= bounds[1]), x, jnp.nan)
    x = jnp.where(q == 0, bounds[0], x)
    x = jnp.where(q == 1, bounds[1], x)
    return x.reshape(qshape)
    
