"""File containing code for numerical approximations of univariate distribution 
functions."""
from jax._src.typing import ArrayLike, Array
import jax.numpy as jnp
from typing import Callable

from copulax._src.univariate._utils import _univariate_input


def _cdf_approx(key: ArrayLike, rvs_func: Callable, x: ArrayLike, params: tuple,
                num_points: int = 500) -> Array:
    """Approximate the cdf of a distribution by evaluating it at a set of 
    points. Uses a linear interpolation to reduce the required number of 
    similated random variables.

    Args:
        key: PRNGKey for random number generation.
        rvs_func: Function which simulates univariate random variables.
        x: arraylike, value(s) at which to evaluate the cdf. This must be a
        one-dimensional array.
        params: Parameters of the distribution.
        num_points: Number of simulated random points at which to evaluate the 
        cdf.

    Returns:
        Array of cdf values.
    """
    x: jnp.ndarray = _univariate_input(x)

    # performing linear interpolation 
    xp: jnp.ndarray = rvs_func(*params, key=key, shape=(num_points,)) 
    xp = xp.sort(axis=0)
    fp = jnp.where(xp <= xp.reshape((num_points, 1)), 1.0, 0.0).mean(axis=1)
    return jnp.interp(x=x, xp=xp, fp=fp, left=0.0, right=1.0)


def _ppf_approx(key: ArrayLike, rvs_func: Callable, q: ArrayLike, params: tuple,
                num_points: int = 500) -> Array:
    """Approximate the percent point function (inverse of cdf) of a 
    distribution by evaluating it at a set of points. Uses a linear 
    interpolation to reduce the required number of simulated random variables.
    
    Args:
        key: PRNGKey for random number generation.
        rvs_func: Function which simulates univariate random variables.
        q: arraylike, value(s) at which to evaluate the ppf.
        params: Parameters of the distribution.
        num_points: Number of simulated random points at which to evaluate the 
        cdf.
    """
    q: jnp.ndarray = _univariate_input(q)

    # performing linear interpolation 
    fp: jnp.ndarray = rvs_func(key, (num_points,), *params) 
    fp = fp.sort(axis=0)
    xp = jnp.where(fp <= fp.reshape((num_points, 1)), 1.0, 0.0).mean(axis=1)
    return jnp.interp(x=q, xp=xp, fp=fp)
