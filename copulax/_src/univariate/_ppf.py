"""File containing code for numerical PPF (percent point function) computation."""

from jax import Array
from jax.typing import ArrayLike
from jax import lax, vmap
import jax.numpy as jnp
import math
from interpax import Interpolator1D

from copulax._src.optimize import brent
from copulax._src.typing import Scalar


def _ppf_func_single(xi: float, qi: float, dist, params):
    """Residual function for root-finding: CDF(xi) - qi."""
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


def _ppf_optimizer(
    dist, q: ArrayLike, params: dict, bounds: Array, maxiter: int
) -> Array:
    """Compute PPF values via Brent bisection with automatic bound expansion.

    Performance notes:
    - For infinite supports, bracketing bounds are resolved once using the
      min/max requested quantiles (instead of once per quantile).
    - Root solves are vmapped across quantiles.
    """
    factor: int = 10
    bound_maxiter: int = _get_bound_maxiter(q.dtype)
    eps: float = 1e-5
    # We clip only for solving roots; exact q=0/1 are handled by caller.
    q_solve = jnp.clip(q, eps, 1 - eps)
    q_min = jnp.min(q_solve)
    q_max = jnp.max(q_solve)

    def _resolve_left(bounds_):
        left, right = bounds_
        non_inf_left = jnp.min(jnp.array([-factor, right]))
        non_inf_left_val = lax.stop_gradient(
            _ppf_func_single(non_inf_left, q_min, dist, params)
        )

        def _left_iter(carry, _):
            l, gl = carry
            l = lax.cond(gl > 0, lambda: l * factor, lambda: l)
            gl = lax.stop_gradient(_ppf_func_single(l, q_min, dist, params))
            return (l, gl), _

        left_res = lax.scan(
            _left_iter,
            (non_inf_left, non_inf_left_val),
            xs=None,
            length=bound_maxiter,
        )[0]
        return jnp.array([left_res[0], right])

    def _resolve_right(bounds_):
        left, right = bounds_
        non_inf_right = jnp.max(jnp.array([factor, left]))
        non_inf_right_val = lax.stop_gradient(
            _ppf_func_single(non_inf_right, q_max, dist, params)
        )

        def _right_iter(carry, _):
            r, gr = carry
            r = lax.cond(gr < 0, lambda: r * factor, lambda: r)
            gr = lax.stop_gradient(_ppf_func_single(r, q_max, dist, params))
            return (r, gr), _

        right_res = lax.scan(
            _right_iter,
            (non_inf_right, non_inf_right_val),
            xs=None,
            length=bound_maxiter,
        )[0]
        return jnp.array([left, right_res[0]])

    # Resolve global bracketing bounds once for this q batch.
    bounds = jnp.asarray(bounds, dtype=float).flatten()
    bounds = lax.cond(
        jnp.isinf(bounds[0]),
        _resolve_left,
        lambda b: jnp.array([b[0] + eps, b[1]]),
        bounds,
    )
    bounds = lax.cond(
        jnp.isinf(bounds[1]),
        _resolve_right,
        lambda b: jnp.array([b[0], b[1] - eps]),
        bounds,
    )

    def _solve_qi(qi):
        return brent(
            method="bisection",
            g=_ppf_func_single,
            bounds=bounds,
            maxiter=maxiter,
            qi=qi,
            dist=dist,
            params=params,
        )

    return vmap(_solve_qi)(q_solve).flatten()


def _cubic_ppf(
    dist, q: ArrayLike, params: dict, bounds: Array, num_points: int, maxiter: int
) -> Array:
    """Compute PPF values via cubic monotonic interpolation of the inverse CDF."""
    # ensuring q has at least 3 elements
    if q.size < 3:
        # more efficient to use ppf_optimiser
        return dist.ppf(q=q, params=params, cubic=False, maxiter=maxiter)

    # clip q to (eps, 1-eps) to avoid numerical issues near bounds
    eps: float = 1e-5
    q_clipped: jnp.ndarray = jnp.clip(q, eps, 1 - eps)

    # getting interpolation bounds
    q_range: jnp.ndarray = jnp.array([jnp.min(q_clipped), jnp.max(q_clipped)]).reshape(
        (2, 1)
    )
    x_min, x_max = _ppf_optimizer(
        dist=dist, q=q_range, params=params, bounds=bounds, maxiter=maxiter
    )

    # getting interpolation points
    x_range: jnp.ndarray = jnp.linspace(x_min, x_max, num_points).flatten()
    cdf_values: jnp.ndarray = dist.cdf(x=x_range, params=params)

    # interpolating
    interpolator = Interpolator1D(
        x=cdf_values, f=x_range, method="monotonic", extrap=True
    )
    interpolated_clipped_values: jnp.ndarray = interpolator(q_clipped)
    interpolated_values: jnp.ndarray = jnp.where(
        q <= eps, bounds[0], interpolated_clipped_values
    )
    interpolated_values = jnp.where(q >= 1 - eps, bounds[1], interpolated_values)
    return interpolated_values


def _ppf(
    dist, q: ArrayLike, params: dict, cubic: bool, num_points: int, maxiter: int
) -> Array:
    """Unified PPF dispatcher: chooses cubic interpolation or direct optimization."""
    q = q.flatten()
    bounds: Array = dist._support(params)
    if cubic:
        x: jnp.ndarray = _cubic_ppf(
            dist=dist,
            q=q,
            params=params,
            bounds=bounds,
            num_points=num_points,
            maxiter=maxiter,
        )
    else:
        x: jnp.ndarray = _ppf_optimizer(
            dist=dist, q=q, params=params, bounds=bounds, maxiter=maxiter
        )

    x = jnp.where(jnp.logical_and(x >= bounds[0], x <= bounds[1]), x, jnp.nan)
    x = jnp.where(q == 0, bounds[0], x)
    return jnp.where(q == 1, bounds[1], x)
