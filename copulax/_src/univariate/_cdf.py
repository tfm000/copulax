"""JIT-compatible, JAX-differentiable numerical CDF integration.

Design summary (see ``.claude/plans/`` for the full rationale):

* Per-xi CDF is computed via a **shorter-tail switch**: when ``xi`` is
  right of the bulk (above the topmost breakpoint), integrate the upper
  tail ``1 - int_{xi}^{upper} pdf`` instead of the lower tail
  ``int_{lower}^{xi} pdf``. This keeps the integrand concentrated at the
  finite endpoint of the integration interval, where quadax's infinite-
  interval algebraic transforms put their dense quadrature nodes.
  Without this, a query at ``xi = 1e8`` with a standard-normal-scale
  mode maps the bulk to ``t ~ -1 + 2e-8`` in quadax's ``[-1, 1]`` domain
  where no Gauss-Kronrod node sits, the quadrature misses the bulk
  entirely, and the CDF collapses to 0.

* All ``K`` breakpoints returned by ``dist._cdf_breakpoints(params)``
  are passed as forced subdivision points to ``quadgk``. Quadax auto-
  collapses out-of-domain breakpoints to zero-length segments
  (``utils.py:110-113``), so a single call handles any K regardless of
  which breakpoints happen to lie in the active integration interval.

* For array inputs, the public ``_cdf`` uses a **sorted piecewise**
  path: one adaptive tail integral at the leftmost sorted x, then
  fixed 32-point Gauss-Legendre segment integrals between consecutive
  sorted x values, then prefix-sum. Query cost drops from O(N) adaptive
  scans to O(1) adaptive scan + O(N) fixed-rule evaluation. User x
  values are clamped to the support before the computation to keep
  NaNs from out-of-support PDF evaluations from polluting neighbours
  via the prefix sum; the base class's ``_enforce_support_on_cdf``
  restores exact 0/1 at originally out-of-support positions.

* The custom VJP (``_cdf_fwd``/``cdf_bwd``) uses ``F'(xi) = pdf(xi)``
  and does NOT differentiate through the adaptive loop. Gradient path
  cost is unchanged from prior revisions.
"""

import numpy as np
import jax
from jax import numpy as jnp
from jax import lax, vmap, value_and_grad
from typing import Callable
from quadax import quadgk


from copulax._src.univariate._utils import _univariate_input


# Tight tolerances for CDF quadrature. The quadax default of
# sqrt(eps) ~ 1.5e-8 is too loose to get tail probabilities below 1e-8
# correct; 1e-12 absolute / 1e-10 relative hits machine precision for
# the distributions in this library.
_EPSABS = 1e-12
_EPSREL = 1e-10


# Precomputed Gauss-Legendre nodes and weights on [-1, 1], stored as
# JAX arrays at module load so they are baked into JIT traces as
# constants rather than recomputed per call.
#
# GL16 (31st-order accurate): used by the PPF cubic-spline builder in
# _cdf_grid_piecewise, which feeds a dense monotonic grid where the
# peak is guaranteed to sit at or outside the leftmost grid point, so
# every segment is monotonic by construction and GL16 is sufficient.
#
# GL32 (63rd-order accurate): used by the public _cdf for arbitrary
# user grids. Its denser node set resolves interior peaks cleanly for
# skewed distributions where the breakpoint (mean) may not coincide
# with the mode.
_GL16_NODES_NP, _GL16_WEIGHTS_NP = np.polynomial.legendre.leggauss(16)
_GL16_NODES = jnp.asarray(_GL16_NODES_NP)
_GL16_WEIGHTS = jnp.asarray(_GL16_WEIGHTS_NP)

_GL32_NODES_NP, _GL32_WEIGHTS_NP = np.polynomial.legendre.leggauss(32)
_GL32_NODES = jnp.asarray(_GL32_NODES_NP)
_GL32_WEIGHTS = jnp.asarray(_GL32_WEIGHTS_NP)


def _cdf_single_x(
    pdf_func: Callable,
    lower: float,
    upper: float,
    bps: jnp.ndarray,
    xi: float,
    params_array: jnp.ndarray,
) -> float:
    r"""Compute the CDF at a single scalar ``xi`` via the shorter-tail
    switch with all K breakpoints passed as adaptive subdivisions.

    The integration form is chosen per-xi:

    * ``xi <= bps[-1]``: integrate ``[lower, xi]`` (lower-tail form).
    * ``xi >  bps[-1]``: integrate ``[xi, upper]`` and return
      ``1 - integral`` (upper-tail form).

    In both forms the integrand is concentrated at the finite endpoint
    ``xi``, where quadax's algebraic infinite-interval transform places
    dense quadrature nodes. All K breakpoints are passed as interior
    subdivisions; quadax collapses those outside ``[start, end]`` to
    zero-length segments (``utils.py:110-113``).

    Boundary handling: ``xi <= lower`` returns 0 and ``xi >= upper``
    returns 1 exactly (F(lower) = 0, F(upper) = 1 by definition).
    This also avoids a zero-length integration interval, which
    quadax's adaptive path returns as NaN rather than 0.
    """
    at_lower = (xi <= lower) & jnp.isfinite(lower)
    at_upper = (xi >= upper) & jnp.isfinite(upper)
    # For the quadrature, keep xi strictly inside the support. A tiny
    # nudge suffices; the PDF at the boundary is 0 for all distributions
    # in this library, so a nonzero-length sliver to the boundary
    # integrates to ~0 and gives the correct F(boundary) = 0 or 1.
    eps_nudge = 1e-30
    xi_safe = jnp.where(
        at_lower, lower + eps_nudge,
        jnp.where(at_upper, upper - eps_nudge, xi),
    )

    use_upper = xi_safe > bps[-1]
    start = jnp.where(use_upper, xi_safe, lower)
    end = jnp.where(use_upper, upper, xi_safe)
    interval = jnp.concatenate([jnp.array([start]), bps, jnp.array([end])])
    result, _info = quadgk(
        pdf_func,
        interval=interval,
        args=params_array,
        epsabs=_EPSABS,
        epsrel=_EPSREL,
    )
    cdf_quad = jnp.where(use_upper, 1.0 - result, result)
    cdf = jnp.where(at_lower, 0.0, jnp.where(at_upper, 1.0, cdf_quad))
    return cdf.reshape(())


def _piecewise_cdf(
    dist,
    x_work: jnp.ndarray,
    bps: jnp.ndarray,
    lower: float,
    upper: float,
    params_array: jnp.ndarray,
) -> jnp.ndarray:
    """Sorted-grid piecewise CDF on ``x_work`` (already support-clamped).

    Inner routine used by both ``_cdf`` (public array path) and
    ``_cdf_normalised``. Performs: augment with breakpoints, argsort,
    one adaptive tail integral via ``_cdf_single_x``, batched GL32
    segments, prefix-sum, inverse-permute, drop augmented entries.

    Returns an array of shape ``x_work.shape`` with raw (pre-clip,
    pre-support-enforcement) CDF values.
    """
    n = x_work.shape[0]
    k = bps.shape[0]

    # Step 3: augment with breakpoints.
    x_aug = jnp.concatenate([x_work, bps])

    # Step 4: sort (stable) and record inverse permutation.
    order = jnp.argsort(x_aug, stable=True)
    inv_order = jnp.argsort(order, stable=True)
    x_sorted = x_aug[order]

    # Step 5: tail CDF at the leftmost sorted point via shorter-tail
    # scalar adaptive call.
    tail = _cdf_single_x(
        dist._pdf_for_cdf, lower, upper, bps, x_sorted[0], params_array
    )

    # Step 6: GL32 segment integrals, batched.
    a = x_sorted[:-1]
    b = x_sorted[1:]
    half = (b - a) / 2.0  # (N+K-1,)
    mid = (a + b) / 2.0  # (N+K-1,)
    # (N+K-1, 32) grid of evaluation points.
    eval_points = mid[:, None] + half[:, None] * _GL32_NODES[None, :]
    pdf_at = vmap(
        vmap(lambda xi: dist._pdf_for_cdf(xi, params_array))
    )(eval_points)
    # Weighted sum per segment, scaled by half-width. All weights and
    # PDF values are non-negative, so segment integrals are >= 0.
    segment_integrals = (pdf_at * _GL32_WEIGHTS[None, :]).sum(axis=1) * half

    # Step 7: prefix-sum. F(x_sorted[0]) = tail; F(x_sorted[k]) = tail
    # + cumsum(segment_integrals)[k-1]. Monotonic by construction.
    cdf_sorted = jnp.concatenate(
        [jnp.array([tail]), tail + jnp.cumsum(segment_integrals)]
    )

    # Step 8: inverse-permute back to x_aug order.
    cdf_aug = cdf_sorted[inv_order]

    # Step 9: drop the K breakpoint-augmented entries (they are at the
    # end of x_aug, i.e., user entries 0..n-1 first, then K breakpoints).
    return cdf_aug[:n]


def _cdf(dist, x: jnp.ndarray, params: dict) -> jnp.ndarray:
    """Compute the CDF via sorted piecewise integration.

    Assumes the PDF is analytically normalised (integrates to 1). Use
    ``_cdf_normalised`` when the PDF may have a non-unity integral.

    For array inputs, this path costs one adaptive tail integral plus
    N fixed-order GL32 segment evaluations, regardless of N. For N=1
    it degenerates to one adaptive tail + 1 GL32 segment.
    """
    x_in, xshape = _univariate_input(x)
    x_flat = x_in.flatten()
    n = x_flat.shape[0]

    lower, upper = dist._support_bounds(params)
    bps = jnp.asarray(dist._cdf_breakpoints(params)).flatten()
    params_array: jnp.ndarray = dist._params_to_array(params)

    # Step 2: clamp to support. No-op on doubly-infinite supports;
    # prevents PDF-at-out-of-support NaNs from contaminating neighbours
    # through the GL32 prefix sum.
    x_work = jnp.clip(x_flat, lower, upper)

    def piecewise_branch(_):
        return _piecewise_cdf(dist, x_work, bps, lower, upper, params_array)

    def trivial_branch(_):
        # All user values lie on a single side of the support. CDF is
        # uniformly 0 (all below) or 1 (all above). The piecewise path
        # would produce the same answer via clamp + enforce, but the
        # lax.cond skips the scan entirely.
        return jnp.where(x_flat < lower, 0.0, 1.0)

    all_below = jnp.all(x_flat < lower)
    all_above = jnp.all(x_flat > upper)
    cdf_raw = lax.cond(
        all_below | all_above, trivial_branch, piecewise_branch, operand=None
    )

    # Step 11: final clip as outermost guard.
    cdf_clipped = jnp.clip(cdf_raw, 0.0, 1.0)
    return cdf_clipped.reshape(xshape)


def _cdf_grid_piecewise(
    dist, x_grid: jnp.ndarray, params: dict
) -> jnp.ndarray:
    r"""Compute the CDF at a sorted grid via piecewise GL16.

    Used by the cubic-spline PPF builder, which passes a dense sorted
    grid spanning the distribution's bulk. The tail integral
    ``F(x_grid[0])`` is computed via the shorter-tail-switched scalar
    adaptive routine, so the same far-tail protections apply here.
    GL16 (31st-order accurate) is adequate on the dense PPF grid
    because each segment is narrow and strictly monotonic (the PPF
    grid is left-of-mode by construction at the leftmost point).
    """
    x_grid = jnp.asarray(x_grid).flatten()

    lower, upper = dist._support_bounds(params)
    bps = jnp.asarray(dist._cdf_breakpoints(params)).flatten()
    params_array: jnp.ndarray = dist._params_to_array(params)

    # Tail integral to the leftmost grid point via the shorter-tail-
    # switched scalar adaptive call (one quadgk scan regardless of
    # grid size).
    tail = _cdf_single_x(
        dist._pdf_for_cdf, lower, upper, bps, x_grid[0], params_array
    )

    # Batched GL16 segment integrals.
    a = x_grid[:-1]
    b = x_grid[1:]
    half = (b - a) / 2.0
    mid = (a + b) / 2.0
    eval_points = mid[:, None] + half[:, None] * _GL16_NODES[None, :]
    pdf_at = vmap(
        vmap(lambda xi: dist._pdf_for_cdf(xi, params_array))
    )(eval_points)
    segment_integrals = (pdf_at * _GL16_WEIGHTS[None, :]).sum(axis=1) * half

    cdf_values = jnp.concatenate(
        [jnp.array([tail]), tail + jnp.cumsum(segment_integrals)]
    )
    return jnp.clip(cdf_values, 0.0, 1.0)


def _cdf_normalised(dist, x: jnp.ndarray, params: dict) -> jnp.ndarray:
    """Compute the CDF with explicit normalisation.

    Divides the piecewise CDF by the full-support integral so the CDF
    reaches exactly 1. Used by distributions whose PDF implementations
    have known numerical inaccuracies that prevent it from integrating
    to 1 (e.g. log-Bessel underflow in the GH family tail).

    The normalising integral ``Z = int_{lower}^{upper} pdf`` is
    computed once with all breakpoints as subdivisions so quadax
    splits at the bulk instead of missing it.
    """
    x_in, xshape = _univariate_input(x)
    x_flat = x_in.flatten()

    lower, upper = dist._support_bounds(params)
    bps = jnp.asarray(dist._cdf_breakpoints(params)).flatten()
    params_array: jnp.ndarray = dist._params_to_array(params)

    # Normalising constant with all breakpoints forced as subdivisions.
    z_interval = jnp.concatenate(
        [jnp.array([lower]), bps, jnp.array([upper])]
    )
    z, _info = quadgk(
        dist._pdf_for_cdf,
        interval=z_interval,
        args=params_array,
        epsabs=_EPSABS,
        epsrel=_EPSREL,
    )

    x_work = jnp.clip(x_flat, lower, upper)

    def piecewise_branch(_):
        return _piecewise_cdf(dist, x_work, bps, lower, upper, params_array)

    def trivial_branch(_):
        # When all user values are on one side, CDF is 0 or z. Dividing
        # by z below yields 0 or 1.
        return jnp.where(x_flat < lower, 0.0, z)

    all_below = jnp.all(x_flat < lower)
    all_above = jnp.all(x_flat > upper)
    cdf_raw = lax.cond(
        all_below | all_above, trivial_branch, piecewise_branch, operand=None
    )

    cdf_clipped = jnp.clip(cdf_raw / z, 0.0, 1.0)
    return cdf_clipped.reshape(xshape)


def _cdf_fwd(dist, cdf_func: Callable, x: jnp.ndarray, params: dict):
    """Forward pass for the custom CDF VJP.

    ``cdf_func`` is accepted for backwards-compatibility with existing
    call sites but is not used here: we call ``_cdf_single_x`` directly
    to avoid running the piecewise augment/sort machinery on scalar
    inputs (which would add wasted overhead when vmapped per-xi).
    """
    x_in, xshape = _univariate_input(x)

    lower, upper = dist._support_bounds(params)
    bps = jnp.asarray(dist._cdf_breakpoints(params)).flatten()
    params_array: jnp.ndarray = dist._params_to_array(params)

    def cdf_single_of_params(xi, params):
        # Rebuild params_array inside this closure so the autodiff
        # chain flows through to the original params dict entries.
        pa = dist._params_to_array(params)
        return _cdf_single_x(
            dist._pdf_for_cdf, lower, upper, bps, xi, pa
        )

    _val_and_grad = value_and_grad(cdf_single_of_params, argnums=1)
    _val_and_grad_vec = vmap(lambda xi: _val_and_grad(xi, params))

    cdf_values, param_grads = _val_and_grad_vec(x_in.flatten())
    pdf_values = dist.pdf(x=x, params=params).reshape(xshape)
    return cdf_values.reshape(xshape), (pdf_values, param_grads)


def cdf_bwd(res, g):
    """Backward pass for the custom CDF VJP.

    Uses ``F'(xi) = pdf(xi)`` for the x gradient, avoiding
    differentiation through the adaptive loop. Parameter gradients
    come from the per-xi value_and_grad computed in the forward pass.
    """
    xshape = res[0].shape
    g = g.reshape(xshape)
    x_grad = res[0] * g
    param_grads: dict = {
        key: jnp.sum(jnp.nan_to_num(val, 0.0) * g) for key, val in res[1].items()
    }  # sum parameter gradients over x
    return x_grad, param_grads
