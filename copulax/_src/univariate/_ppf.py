"""File containing code for numerical PPF (percent point function) computation."""
from jax._src.typing import ArrayLike, Array
from jax import lax
import jax.numpy as jnp
import math
from interpax import Interpolator1D

from copulax._src.optimize import brent
from copulax._src.typing import Scalar


def _ppf_func_single(xi: float, qi: float, dist, params):
    return (dist.cdf(x=xi, params=params) - qi).reshape(())


def _get_bound_maxiter(dtype) -> int:
    """Compute the number of iterations needed to find finite bounds,
    based on the dtype's representable range and resolution."""
    try:
        info = jnp.finfo(dtype)
        # log10 of max representable value gives the width
        width = int(math.log10(info.max))
        # number of significant decimal digits gives the resolution
        resolution_power = int(-math.log10(info.resolution))
    except ValueError:
        # integer types
        try:
            info = jnp.iinfo(dtype)
            width = int(math.log10(max(abs(info.max), 1)))
            resolution_power = 0
        except Exception:
            width = 38
            resolution_power = 6
    return width + resolution_power


def _ppf_optimizer(dist, q: ArrayLike, params: dict, bounds: tuple, maxiter: int) -> Array:
    factor: int = 10
    bound_maxiter: int = _get_bound_maxiter(q.dtype)

    def _left_iter(carry, _):
         left, right, qi, last_val = carry
         left, right = lax.cond(last_val > 0, lambda: (left * factor, left), lambda: (left, right))
         next_val = _ppf_func_single(xi=left, qi=qi, dist=dist, params=params)
         return (left, right, qi, next_val), _
    
    def _right_iter(carry, _):
         left, right, qi, last_val = carry
         left, right = lax.cond(last_val < 0, lambda: (right, right * factor), lambda: (left, right))
         next_val = _ppf_func_single(xi=right, qi=qi, dist=dist, params=params)
         return (left, right, qi, next_val), _
    
    def _iter(carry, qi):
        eps = 1e-5

        # getting non-infinite left bound
        left, right = bounds
        non_inf_left = jnp.min(jnp.array([-factor, right]))
        non_inf_left_val = _ppf_func_single(non_inf_left, qi, dist, params) 
        left_res = lax.cond(jnp.isinf(left), lambda: lax.scan(_left_iter, (non_inf_left, right, qi, non_inf_left_val), length=bound_maxiter)[0], lambda: (left + eps, right, qi, non_inf_left_val))
        left, right = left_res[0], left_res[1]

        # getting non-infinite right bound
        non_inf_right = jnp.max(jnp.array([factor, left]))
        non_inf_right_val = _ppf_func_single(non_inf_right, qi, dist, params) 
        left_res = lax.cond(jnp.isinf(right), lambda: lax.scan(_right_iter, (left, non_inf_right, qi, non_inf_right_val), length=bound_maxiter)[0], lambda: (left, right - eps, qi, non_inf_right_val))
        left, right = left_res[0], left_res[1]

        # solving for root within bounds via bisection
        new_qi = brent(method='bisection', g=_ppf_func_single, bounds=jnp.array([left, right]), maxiter=maxiter, qi=qi, dist=dist, params=params)
        return carry, new_qi
    
    _, x = lax.scan(_iter, None, q)
    return x.flatten()


def _cubic_ppf(dist, q: ArrayLike, params: dict, 
                bounds: tuple, num_points: int,  maxiter: int) -> Array:
    # ensuring q has at least 3 elements
    if q.size < 3:
        # more efficient to use ppf_optimiser
        return dist.ppf(q=q, params=params, cubic=False, maxiter=maxiter)

    # clip q to (eps, 1-eps) to avoid numerical issues near bounds
    eps: float = 1e-5
    q_clipped: jnp.ndarray = jnp.clip(q, eps, 1-eps)

    # getting interpolation bounds
    q_range: jnp.ndarray = jnp.array([jnp.min(q_clipped), jnp.max(q_clipped)]
                                     ).reshape((2, 1))
    x_min, x_max = _ppf_optimizer(dist=dist, q=q_range, params=params, 
                                  bounds=bounds, maxiter=maxiter)

    # getting interpolation points
    x_range: jnp.ndarray = jnp.linspace(x_min, x_max, num_points).flatten()  
    cdf_values: jnp.ndarray = dist.cdf(x=x_range, params=params)

    # interpolating
    interpolator = Interpolator1D(x=cdf_values, f=x_range, method='monotonic', extrap=True)
    interpolated_clipped_values: jnp.ndarray = interpolator(q_clipped)
    interpolated_values: jnp.ndarray = jnp.where(q <= eps, bounds[0], interpolated_clipped_values)
    interpolated_values = jnp.where(q >= 1-eps, bounds[1], interpolated_values)
    return interpolated_values


def _ppf(dist, q: ArrayLike, params: dict, cubic: bool, 
         num_points: int, maxiter: int) -> Array:
    q = q.flatten()
    bounds: tuple = dist._support(params)
    if cubic:
        x: jnp.ndarray = _cubic_ppf(dist=dist, q=q, params=params, 
                                     bounds=bounds, num_points=num_points, 
                                     maxiter=maxiter)
    else:
        x: jnp.ndarray = _ppf_optimizer(dist=dist, q=q, params=params, 
                                        bounds=bounds, maxiter=maxiter)
        
    x = jnp.where(jnp.logical_and(x >= bounds[0], x <= bounds[1]), x, jnp.nan)
    x = jnp.where(q == 0, bounds[0], x)
    return jnp.where(q == 1, bounds[1], x)
