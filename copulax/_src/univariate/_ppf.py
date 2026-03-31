"""File containing code for numerical PPF (percent point function) computation.

The PPF (inverse CDF) is equipped with a custom VJP rule via the
implicit function theorem (IFT).  If :math:`F(x; \\theta) = q`, then:

.. math::

    \\frac{\\partial x}{\\partial q}
        = \\frac{1}{f(x; \\theta)},
    \\qquad
    \\frac{\\partial x}{\\partial \\theta}
        = -\\frac{\\partial F / \\partial \\theta}{f(x; \\theta)},

where :math:`f` is the PDF.  This gives exact, efficient gradients
regardless of how the forward solve is performed (Brent bisection
or cubic interpolation).
"""

from functools import partial
import jax
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
        width = int(math.log10(info.max))
        resolution_power = int(-math.log10(info.resolution))
    except ValueError:
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
            g=_ppf_func_single,
            bounds=bounds,
            maxiter=maxiter,
            qi=qi,
            dist=dist,
            params=params,
        )

    return vmap(_solve_qi)(q_solve).flatten()


###############################################################################
# Shared IFT backward pass
###############################################################################

def _ift_ppf_bwd(dist, res, g):
    """Shared IFT backward pass for the PPF.

    Given ``F(x; θ) = q`` and upstream cotangent ``g = ∂L/∂x``:

    - ``∂L/∂q = g / f(x; θ)``
    - ``∂L/∂θ = VJP of CDF w.r.t. θ with cotangent −g/f(x; θ)``
    - ``∂L/∂bounds = 0``
    """
    x, q, params = res
    x_flat = x.flatten()
    q_flat = q.flatten()
    g_flat = g.flatten()

    # PDF at solution points (IFT denominator)
    pdf_x = dist.pdf(x_flat, params=params).flatten()
    safe_pdf = jnp.maximum(jnp.abs(pdf_x), 1e-30)

    # Zero out gradient at boundaries / NaN
    at_boundary = (q_flat == 0) | (q_flat == 1) | jnp.isnan(x_flat)
    g_over_pdf = jnp.where(at_boundary, 0.0, g_flat / safe_pdf)

    # ∂L/∂q = g / f(x)
    q_bar = g_over_pdf.reshape(q.shape)

    # ∂L/∂params via VJP of CDF w.r.t. params
    def _cdf_of_params(p):
        return dist.cdf(x=lax.stop_gradient(x_flat), params=p).flatten()

    _, vjp_fn = jax.vjp(_cdf_of_params, params)
    (params_bar,) = vjp_fn(-g_over_pdf)

    # ∂L/∂bounds = 0
    bounds_bar = jnp.zeros(2)

    return q_bar, params_bar, bounds_bar


###############################################################################
# Brent PPF with IFT custom VJP
###############################################################################

def _ppf_brent_solve(dist, q, params, bounds):
    """Raw PPF forward solve via Brent root-finding (no custom gradient)."""
    q = jnp.asarray(q)
    q_flat = q.flatten()
    x = _ppf_optimizer(
        dist=dist, q=q_flat, params=params, bounds=bounds, maxiter=20,
    )
    x = jnp.where(jnp.logical_and(x >= bounds[0], x <= bounds[1]), x, jnp.nan)
    x = jnp.where(q_flat == 0, bounds[0], x)
    return jnp.where(q_flat == 1, bounds[1], x)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _ppf_brent(dist, q, params, bounds):
    """PPF via Brent with custom VJP (IFT)."""
    return _ppf_brent_solve(dist, q, params, bounds)


def _ppf_brent_fwd(dist, q, params, bounds):
    x = _ppf_brent_solve(dist, q, params, bounds)
    return x, (x, q, params)


def _ppf_brent_bwd(dist, res, g):
    return _ift_ppf_bwd(dist, res, g)


_ppf_brent.defvjp(_ppf_brent_fwd, _ppf_brent_bwd)


###############################################################################
# Cubic spline PPF with IFT custom VJP
###############################################################################

def _cubic_ppf_solve(
    dist, q, params, bounds, num_points, maxiter
) -> Array:
    """Raw cubic spline PPF forward solve (no custom gradient).

    Builds a monotonic cubic spline of the inverse CDF on a grid
    of ``num_points`` CDF evaluations, then interpolates at the
    requested quantiles.
    """
    eps: float = 1e-5
    q_clipped = jnp.clip(q, eps, 1 - eps)

    q_range = jnp.array([jnp.min(q_clipped), jnp.max(q_clipped)]).reshape(
        (2, 1)
    )
    x_min, x_max = _ppf_optimizer(
        dist=dist, q=q_range, params=params, bounds=bounds, maxiter=maxiter
    )

    x_range = jnp.linspace(x_min, x_max, num_points).flatten()
    cdf_values = dist.cdf(x=x_range, params=params)

    interpolator = Interpolator1D(
        x=cdf_values, f=x_range, method="monotonic", extrap=True
    )
    x = interpolator(q_clipped)
    x = jnp.where(q <= eps, bounds[0], x)
    x = jnp.where(q >= 1 - eps, bounds[1], x)
    return x


def _make_ppf_cubic(num_points: int, maxiter: int):
    """Create a cubic PPF function with IFT custom VJP.

    ``num_points`` and ``maxiter`` are captured in the closure so the
    returned function has the same ``(dist, q, params, bounds)``
    signature as ``_ppf_brent``.
    """

    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def _ppf_cubic(dist, q, params, bounds):
        return _cubic_ppf_solve(
            dist, q, params, bounds, num_points, maxiter
        )

    def _ppf_cubic_fwd(dist, q, params, bounds):
        x = _cubic_ppf_solve(
            dist, q, params, bounds, num_points, maxiter
        )
        return x, (x, q, params)

    def _ppf_cubic_bwd(dist, res, g):
        return _ift_ppf_bwd(dist, res, g)

    _ppf_cubic.defvjp(_ppf_cubic_fwd, _ppf_cubic_bwd)
    return _ppf_cubic


###############################################################################
# Unified dispatcher
###############################################################################

def _ppf(
    dist, q: ArrayLike, params: dict, cubic: bool, num_points: int, maxiter: int
) -> Array:
    """Unified PPF dispatcher: chooses cubic interpolation or direct
    Brent root-finding.  Both paths have IFT custom VJP rules for
    exact, efficient gradients.
    """
    q_arr = jnp.asarray(q).flatten()
    bounds: Array = dist._support(params)
    if cubic:
        if q_arr.size < 3:
            return _ppf_brent(dist, q_arr, params, bounds)
        ppf_cubic_fn = _make_ppf_cubic(num_points, maxiter)
        return ppf_cubic_fn(dist, q_arr, params, bounds)
    else:
        return _ppf_brent(dist, q_arr, params, bounds)
