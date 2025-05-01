"""File containing code for numerical approximations of univariate distribution 
functions."""
from jax._src.typing import ArrayLike, Array
from jax import jit, lax
import jax.numpy as jnp
from typing import Callable
from interpax import Interpolator1D

from copulax._src.optimize import projected_gradient
from copulax._src.typing import Scalar


# def _ppf_optimizer(dist, q: ArrayLike, params: dict, x0: Scalar, 
#                    bounds: jnp.ndarray, lr: float, maxiter: int) -> Array:
    
#     def _ppf_func_single(xi: float, qi: float):
#         return jnp.abs((dist.cdf(x=xi, params=params) - qi)).reshape(())
    
#     min_val, max_val = bounds.reshape((2,1))
#     SCALE = 0.5
#     x0_small = jnp.max(jnp.array([x0 * (1-SCALE), min_val]))
#     x0_large = jnp.min(jnp.array([x0 * (1+SCALE), max_val]))
#     def _iter(carry, qi):
#         x0 = jnp.where(qi <= 0.5, x0_small, x0_large).reshape((1,))
#         res = projected_gradient(
#             f=_ppf_func_single, x0=x0, lr=lr, maxiter=maxiter, 
#             projection_method='projection_box', 
#             projection_options={'hyperparams': bounds}, qi=qi)
#         return carry, res['x']

#     _, x = jax.lax.scan(_iter, None, q)
#     return x.flatten()

@jit
def _ppf_func_single(xi: float, qi: float, dist, params):
        return (dist.cdf(x=xi, params=params) - qi).reshape(())


@jit
def _ppf_func_single_abs(xi: float, qi: float, dist, params):
        return jnp.abs(_ppf_func_single(xi, qi, dist, params))


def _ppf_optimizer(dist, q: ArrayLike, params: dict, x0: Scalar, 
                   bounds: tuple, lr: float, maxiter: int) -> Array:
    # getting bound_maxiter
    factor: int = 10
    if q.dtype == jnp.int32 or q.dtype == int or q.dtype == jnp.uint32:
         width: int = 9
         resolution_power: int = 0
    elif q.dtype == jnp.int64 or q.dtype == jnp.uint or q.dtype == jnp.uint64:
         width: int = 19
         resolution_power: int = 0
    elif q.dtype == jnp.int4:
            width: int = 1
            resolution_power: int = 0
    elif q.dtype == jnp.int8 or q.dtype == jnp.uint8:
         width: int = 2
         resolution_power: int = 0
    elif q.dtype == jnp.int16 or q.dtype == jnp.uint16:
         width: int = 4
         resolution_power: int = 0
    elif q.dtype == jnp.float16:
         width: int = 4
         resolution_power: int = 3
    elif q.dtype == jnp.float32:
         width: int = 38
         resolution_power: int = 6
    else:
        # float64
        width: int = 308
        resolution_power: int = 15
    bound_maxiter: int = width + resolution_power

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
        # getting non-infinite left bound
        left, right = bounds
        non_inf_left = jnp.min(jnp.array([-factor, right]))
        non_inf_left_val = _ppf_func_single(non_inf_left, qi, dist, params) 
        left_res = lax.cond(jnp.isinf(left), lambda: lax.scan(_left_iter, (non_inf_left, right, qi, non_inf_left_val), length=bound_maxiter)[0], lambda: (left, right, qi, non_inf_left_val))
        left, right = left_res[0], left_res[1]

        # getting non-infinite right bound
        non_inf_right = jnp.max(jnp.array([factor, left]))
        non_inf_right_val = _ppf_func_single(non_inf_right, qi, dist, params) 
        left_res = lax.cond(jnp.isinf(right), lambda: lax.scan(_right_iter, (left, non_inf_right, qi, non_inf_right_val), length=bound_maxiter)[0], lambda: (left, right, qi, non_inf_right_val))
        left, right = left_res[0], left_res[1]
        x0 = (left + right).reshape((1,)) / 2

        # solving for root within bounds
        res = projected_gradient(
            f=_ppf_func_single_abs, x0=x0, lr=lr, maxiter=maxiter, 
            projection_method='projection_box', 
            projection_options={'hyperparams': jnp.array([left, right]).flatten()}, qi=qi, dist=dist, params=params)
        return carry, res['x']

    _, x = lax.scan(_iter, None, q)
    return x.flatten()


def _cubic_ppf(dist, q: ArrayLike, params: dict, x0: Scalar, 
                bounds: tuple, num_points: int,  lr: float, 
                maxiter: int) -> Array:
    # ensuring q has at least 3 elements
    if q.size < 3:
        # more efficient to use ppf_optimiser
        return _ppf(dist=dist, q=q, params=params, x0=x0, cubic=False, 
                    lr=lr, maxiter=maxiter, num_points=num_points)

    # clip q to (eps, 1-eps) to avoid numerical issues near bounds
    eps: float = 1e-5
    q_clipped: jnp.ndarray = jnp.clip(q, eps, 1-eps)

    # getting interpolation bounds
    q_range: jnp.ndarray = jnp.array([jnp.min(q_clipped), jnp.max(q_clipped)]
                                     ).reshape((2, 1))
    x_min, x_max = _ppf_optimizer(dist=dist, q=q_range, params=params, x0=x0, 
                                  bounds=bounds, lr=lr, maxiter=maxiter)

    # getting interpolation points
    x_range: jnp.ndarray = jnp.linspace(x_min, x_max, num_points).flatten()  
    cdf_values: jnp.ndarray = dist.cdf(x=x_range, params=params)

    # interpolating
    interpolator = Interpolator1D(x=cdf_values, f=x_range, method='monotonic', extrap=True)
    interpolated_clipped_values: jnp.ndarray = interpolator(q_clipped)
    interpolated_values: jnp.ndarray = jnp.where(q <= eps, bounds[0], interpolated_clipped_values)
    interpolated_values = jnp.where(q >= 1-eps, bounds[1], interpolated_values)
    return interpolated_values
    


def _ppf(dist, x0: Scalar, q: ArrayLike, params: dict, cubic: bool, 
         num_points: int, lr: float, maxiter: int) -> Array:
    q = q.flatten()
    x0 = jnp.asarray(x0, dtype=float).reshape((1,))
    bounds: tuple = dist._support(params)
    if cubic:
        x: jnp.ndarray = _cubic_ppf(dist=dist, q=q, params=params, x0=x0, 
                                     bounds=bounds, num_points=num_points, 
                                     lr=lr, maxiter=maxiter)
    else:
        x: jnp.ndarray = _ppf_optimizer(dist=dist, q=q, params=params, x0=x0, 
                                        bounds=bounds, lr=lr, maxiter=maxiter)
        
    x = jnp.where(jnp.logical_and(x >= bounds[0], x <= bounds[1]), x, jnp.nan)
    x = jnp.where(q == 0, bounds[0], x)
    return jnp.where(q == 1, bounds[1], x)
