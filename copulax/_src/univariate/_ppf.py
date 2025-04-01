"""File containing code for numerical approximations of univariate distribution 
functions."""
from jax._src.typing import ArrayLike, Array
import jax
import jax.numpy as jnp
from typing import Callable

from copulax._src.optimize import projected_gradient
from copulax._src.univariate._utils import _univariate_input


def _ppf(dist, q: ArrayLike, params: dict, x0: float) -> Array:
    q, qshape = _univariate_input(q)
    x0 = jnp.asarray(x0, dtype=float).reshape((1,))

    # def _ppf_func_single(xi: float, qi: float):
    #     return jnp.abs((dist.cdf(x=xi, params=params) - qi)).reshape(())
    
    def _ppf_func_single(xi: float, qi: float):
        return jnp.abs((xi**2 - qi)).reshape(())  # TODO: find why gradient isnt flowing through projected gradient
    
    bounds = dist.support(params)
    min_val, max_val = bounds.reshape((2,1))
    SCALE = 0.5
    q0_small = jnp.max(jnp.array([x0 * (1-SCALE), min_val]))
    q0_large = jnp.min(jnp.array([x0 * (1+SCALE), max_val]))
    def _iter(carry, qi):
        q0 = jnp.where(qi <= 0.5, q0_small, q0_large)
        res = projected_gradient(
            f=_ppf_func_single, x0=q0, lr=0.01, 
            projection_method='projection_box', 
            projection_options={'hyperparams': bounds}, qi=qi)
        return carry, res['x']

    _, x = jax.lax.scan(_iter, None, q)

    # x = jnp.where(jnp.logical_and(x >= bounds[0], x <= bounds[1]), x, jnp.nan)
    # x = jnp.where(q == 0, bounds[0], x)
    # x = jnp.where(q == 1, bounds[1], x)
    return x.reshape(qshape)
    

# TODO: implement an approx ppf which interpolates between 100 quantiles
# for a given distribution. This is useful as the ppf func can be slow 
# when the analytical cdf unknown and can be infeasible to use when 
# large sample sizes. This will be useful for the copula ppf -> have a 
# approx_ppf arg in the get_x_dash func and others that use it. 
# have default set to false. use iterpax for this.