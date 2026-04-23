"""Numerical PPF (percent point function) computation.

The PPF is the inverse of the CDF: given a quantile :math:`q \\in [0, 1]`,
find :math:`x` such that :math:`F(x; \\theta) = q`.  This module provides
two solvers, both wrapped with an implicit-function-theorem (IFT) custom
VJP rule:

.. math::

    \\frac{\\partial x}{\\partial q}
        = \\frac{1}{f(x; \\theta)},
    \\qquad
    \\frac{\\partial x}{\\partial \\theta}
        = -\\frac{\\partial F / \\partial \\theta}{f(x; \\theta)},

where :math:`f` is the PDF.  The gradient path is identical regardless
of which forward solver is used, so ``brent`` selects purely the
forward algorithm and not the differentiation strategy.

Both solvers operate in **t-space** — the same quadax-transformed
coordinate system used by :py:mod:`_cdf`.  The full support (possibly
infinite) is mapped to ``t \\in [-1, 1]`` via ``quadax.utils.MAPFUNS``,
and both solvers bracket the solve inside ``[-1 + _T_EPS, 1 - _T_EPS]``.
This eliminates the bound-expansion :py:func:`lax.scan` that the prior
implementation required for infinite supports.

- **Default path — Chebyshev cubic** (:py:func:`_cubic_ppf_solve`,
  ``brent=False``): evaluates the CDF on a Chebyshev-Lobatto grid of
  ``nodes`` points in t-space, then builds a monotonic inverse-CDF
  spline and interpolates at the requested quantiles.  Cosine knot
  spacing is dense near ``t = \\pm 1`` — exactly where the x-space
  map has steep slope — so deep-tail accuracy does not collapse the
  way a uniform t-grid does.  Uses the fast batched
  :py:func:`_piecewise_cdf_tspace` for numerical-CDF distributions
  and the direct ``dist.cdf`` call for closed-form ones.  Falls back
  to Brent when ``q.size < 3`` (too few queries to amortise the
  spline build).

- **Opt-in path — per-quantile Brent** (:py:func:`_ppf_brent_solve`,
  ``brent=True``): Brent root-finding on the residual
  ``CDF(MAPFUNS(t)) - q`` with the static bracket
  ``[-1 + _T_EPS, 1 - _T_EPS]``, vmapped across quantiles.
  Machine-epsilon accurate; used as the gold-standard reference by
  the test suite and available to users who need tight precision.
"""

from functools import lru_cache, partial
import jax
from jax import Array
from jax.typing import ArrayLike
from jax import lax, vmap
import jax.numpy as jnp
from interpax import Interpolator1D
from quadax.utils import MAPFUNS

from copulax._src.optimize import brent
from copulax._src.typing import Scalar
from copulax._src.univariate._cdf import _piecewise_cdf_tspace, _T_EPS


# Boundary clip on quantile queries.  q values closer than _EPS to 0
# or 1 short-circuit to the support endpoints (avoiding t-space solves
# near the open boundaries where the transform is steep).
_EPS = 1e-5


###############################################################################
# Shared utilities
###############################################################################

def _support_bitmask(lower: Scalar, upper: Scalar) -> Array:
    """Return the quadax MAPFUNS/MAPFUNS_INV dispatch index for a support.

    0: finite/finite.  1: (-inf, b).  2: (a, +inf).  3: (-inf, +inf).
    """
    return (jnp.isinf(lower).astype(jnp.int32)
            + 2 * jnp.isinf(upper).astype(jnp.int32))


def _tspace_to_x(t: Scalar, bitmask: Scalar, lower: Scalar, upper: Scalar) -> Scalar:
    """Map a t-space value back to x-space via the quadax forward transform."""
    x_val, _w = lax.switch(bitmask, MAPFUNS, t, lower, upper)
    return x_val


def _apply_edge_cases(q: Array, x: Array, lower: Scalar, upper: Scalar) -> Array:
    """Enforce PPF boundary semantics on the output.

    - ``q == 0``  -> ``lower`` support bound.
    - ``q == 1``  -> ``upper`` support bound.
    - ``q < 0``, ``q > 1``, or ``isnan(q)`` -> NaN.
    """
    x = jnp.where(q == 0, lower, x)
    x = jnp.where(q == 1, upper, x)
    invalid = (q < 0) | (q > 1) | jnp.isnan(q)
    return jnp.where(invalid, jnp.nan, x)


###############################################################################
# Shared IFT backward pass
###############################################################################

def _ift_ppf_bwd(dist, res, g):
    """Shared IFT backward pass for the PPF.

    Given ``F(x; theta) = q`` and upstream cotangent ``g = dL/dx``:

    - ``dL/dq = g / f(x; theta)``
    - ``dL/dtheta = VJP of CDF w.r.t. theta with cotangent -g/f(x; theta)``
    - ``dL/dbounds = 0``
    """
    x, q, params = res
    x_flat = x.flatten()
    q_flat = q.flatten()
    g_flat = g.flatten()

    # PDF at solution points (IFT denominator).
    pdf_x = dist.pdf(x_flat, params=params).flatten()
    safe_pdf = jnp.maximum(jnp.abs(pdf_x), 1e-30)

    # Zero out gradient at boundaries / NaN quantiles.
    at_boundary = (q_flat == 0) | (q_flat == 1) | jnp.isnan(x_flat)
    g_over_pdf = jnp.where(at_boundary, 0.0, g_flat / safe_pdf)

    # dL/dq = g / f(x).
    q_bar = g_over_pdf.reshape(q.shape)

    # dL/dparams via VJP of CDF w.r.t. params.
    def _cdf_of_params(p):
        return dist.cdf(x=lax.stop_gradient(x_flat), params=p).flatten()

    _, vjp_fn = jax.vjp(_cdf_of_params, params)
    (params_bar,) = vjp_fn(-g_over_pdf)

    # dL/dbounds = 0.
    bounds_bar = jnp.zeros(2)

    return q_bar, params_bar, bounds_bar


###############################################################################
# Brent PPF in t-space with IFT custom VJP
###############################################################################

def _ppf_brent_solve(dist, q, params, bounds, maxiter: int) -> Array:
    r"""Per-quantile Brent root-finding of ``CDF(MAPFUNS(t)) - q`` in t-space.

    The full support ``(lower, upper)`` is mapped to ``t \in [-1, 1]``
    via ``quadax.utils.MAPFUNS``; Brent brackets the solve inside
    ``[-1 + _T_EPS, 1 - _T_EPS]``.  No bound-expansion ``lax.scan`` is
    needed because the t-bracket covers the entire support by
    construction.
    """
    q = jnp.asarray(q)
    q_flat = q.flatten()
    q_solve = jnp.clip(q_flat, _EPS, 1.0 - _EPS)

    lower, upper = bounds[0], bounds[1]
    bitmask = _support_bitmask(lower, upper)

    def _residual(t, qi):
        x_val = _tspace_to_x(t, bitmask, lower, upper)
        return (dist.cdf(x=x_val, params=params) - qi).reshape(())

    t_bounds = jnp.array([-1.0 + _T_EPS, 1.0 - _T_EPS])

    def _solve_qi(qi):
        t_star = brent(g=_residual, bounds=t_bounds, maxiter=maxiter, qi=qi)
        return _tspace_to_x(t_star, bitmask, lower, upper)

    x = vmap(_solve_qi)(q_solve).flatten()
    return _apply_edge_cases(q_flat, x, lower, upper)


@lru_cache(maxsize=None)
def _make_ppf_brent(maxiter: int):
    """Create a Brent PPF function with IFT custom VJP.

    ``maxiter`` is captured in the closure so the returned function has
    the signature ``(dist, q, params, bounds)``.  The factory is
    memoised so JAX's trace cache keys on the same wrapper object
    across calls with matching ``maxiter`` -- important for iterative
    fits that would otherwise retrace on every iteration.
    """

    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def _ppf_brent(dist, q, params, bounds):
        return _ppf_brent_solve(dist, q, params, bounds, maxiter)

    def _ppf_brent_fwd(dist, q, params, bounds):
        x = _ppf_brent_solve(dist, q, params, bounds, maxiter)
        return x, (x, q, params)

    def _ppf_brent_bwd(dist, res, g):
        return _ift_ppf_bwd(dist, res, g)

    _ppf_brent.defvjp(_ppf_brent_fwd, _ppf_brent_bwd)
    return _ppf_brent


###############################################################################
# Chebyshev cubic PPF with IFT custom VJP
###############################################################################

def _cubic_ppf_solve(dist, q, params, bounds, nodes: int) -> Array:
    r"""Chebyshev cubic-spline PPF in t-space.

    Builds a Chebyshev-Lobatto grid of ``nodes`` points in t-space,
    maps to x-space via ``MAPFUNS``, populates the CDF on that grid
    (using the fast piecewise routine for numerical-CDF distributions
    and the direct ``dist.cdf`` call for closed-form ones), and
    constructs a monotonic inverse-CDF spline.  Query quantiles are
    interpolated on that spline.

    Cosine knot spacing :math:`t_i = \cos(\pi \, (N-1-i) / (N-1))` is
    dense near ``t = +/-1`` -- exactly where the x(t) map is steep --
    so deep-tail accuracy scales properly with ``nodes``.  No Brent
    bracket discovery, no cold/warm distinction, no adaptive
    refinement.
    """
    q = jnp.asarray(q).flatten()
    q_clipped = jnp.clip(q, _EPS, 1.0 - _EPS)

    lower, upper = bounds[0], bounds[1]
    bps = jnp.asarray(dist._cdf_breakpoints(params)).flatten()
    bitmask = _support_bitmask(lower, upper)

    # Chebyshev-Lobatto nodes on (-(1-_T_EPS), (1-_T_EPS)), ascending.
    i = jnp.arange(nodes)
    t_grid = jnp.cos(jnp.pi * (nodes - 1 - i) / (nodes - 1)) * (1.0 - _T_EPS)

    # Map t-grid to x-grid (monotonic because MAPFUNS is monotonic in t).
    x_grid = vmap(lambda t: _tspace_to_x(t, bitmask, lower, upper))(t_grid)

    # CDF-on-grid dispatch: numerical-CDF distributions use the fast
    # piecewise t-space routine; closed-form ones call dist.cdf
    # directly.  Resolved at trace time on the subclass override check.
    if "_pdf_for_cdf" in type(dist).__dict__:
        params_array = dist._params_to_array(params)
        cdf_grid = _piecewise_cdf_tspace(
            dist, x_grid, bps, lower, upper, params_array
        )
    else:
        cdf_grid = dist.cdf(x=x_grid, params=params).flatten()

    # Defensive clipping + cummax: the underlying integrals are
    # non-negative so cdf_grid is monotonic by construction, but
    # quadrature noise or closed-form rounding can introduce tiny
    # reversals that would break the monotonic spline.
    cdf_grid = jax.lax.cummax(jnp.clip(cdf_grid, 0.0, 1.0))

    interp = Interpolator1D(
        x=cdf_grid, f=x_grid, method="monotonic", extrap=True
    )
    x = interp(q_clipped)
    return _apply_edge_cases(q, x, lower, upper)


@lru_cache(maxsize=None)
def _make_ppf_cubic(nodes: int):
    """Create a cubic PPF function with IFT custom VJP.

    ``nodes`` is captured in the closure so the returned function has
    the signature ``(dist, q, params, bounds)``.  The factory is
    memoised so JAX's trace cache keys on the same wrapper object
    across calls with matching ``nodes`` -- important for iterative
    fits that would otherwise retrace on every iteration.
    """

    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def _ppf_cubic(dist, q, params, bounds):
        return _cubic_ppf_solve(dist, q, params, bounds, nodes)

    def _ppf_cubic_fwd(dist, q, params, bounds):
        x = _cubic_ppf_solve(dist, q, params, bounds, nodes)
        return x, (x, q, params)

    def _ppf_cubic_bwd(dist, res, g):
        return _ift_ppf_bwd(dist, res, g)

    _ppf_cubic.defvjp(_ppf_cubic_fwd, _ppf_cubic_bwd)
    return _ppf_cubic


###############################################################################
# Unified dispatcher
###############################################################################

def _ppf(
    dist,
    q: ArrayLike,
    params: dict,
    brent: bool,
    nodes: int,
    maxiter: int,
) -> Array:
    """Dispatch to the Chebyshev-cubic or Brent PPF solver.

    When ``brent=False`` (default) the Chebyshev cubic spline is used
    — unless the query batch is too small to amortise the spline build
    (``q.size < 3``), in which case the dispatcher falls back to Brent
    automatically.  When ``brent=True`` the per-quantile t-space Brent
    solver runs unconditionally.  Both paths carry the same IFT custom
    VJP rule, so downstream gradient behaviour is identical.
    """
    q_arr = jnp.asarray(q).flatten()
    bounds: Array = dist._support(params)
    if (not brent) and q_arr.size >= 3:
        return _make_ppf_cubic(nodes)(dist, q_arr, params, bounds)
    return _make_ppf_brent(maxiter)(dist, q_arr, params, bounds)
