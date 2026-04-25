"""JIT-compatible, JAX-differentiable numerical CDF integration.

Design summary:

* **Piecewise GL32 on a bounded domain**: the public ``_cdf`` maps
  the entire support to ``[-1, 1]`` via quadax's ``map_interval``,
  transforming the integrand ``pdf(x)`` into
  ``pdf(x(t)) * (dx/dt) * sgn``. All user query points and breakpoints
  are mapped through the matching inverse transform. Integration
  happens piecewise on segments of the bounded domain using fixed
  32-point Gauss-Legendre. The transform concentrates the integrand's
  mass inside ``[-1, 1]`` for every support type and makes
  ``xi = +/-inf`` land cleanly at ``t = +/-1``, so the public CDF
  runs as a single forward path regardless of how extreme the query
  points are.

* **Per-xi scalar path** (``_cdf_single_x``): used by the custom VJP
  forward rule. Adapts the shorter-tail switch and passes all
  breakpoints to ``quadgk`` as subdivisions, so gradient cost stays
  O(N adaptive scans) under vmap.

* **Custom VJP**: ``_cdf_fwd`` calls ``_cdf_single_x`` directly
  per-xi under vmap; ``cdf_bwd`` uses ``F'(xi) = pdf(xi)``. The
  gradient path bypasses the piecewise machinery to keep the
  backward-pass cost independent of batch size.
"""

import numpy as np
import jax
from jax import numpy as jnp
from jax import lax, vmap, value_and_grad
from typing import Callable
from quadax import quadgk
from quadax.utils import MAPFUNS, MAPFUNS_INV


from copulax._src.univariate._utils import _univariate_input


# Tight tolerances for the scalar adaptive path (used by the custom VJP).
# The quadax default of sqrt(eps) ~ 1.5e-8 is too loose for CDF tail
# probabilities below 1e-8; 1e-12 abs / 1e-10 rel hits machine precision
# for every distribution in this library.
_EPSABS = 1e-12
_EPSREL = 1e-10

# Tiny offset from the open boundaries of [-1, 1]. The quadax infinite-
# interval maps have 1/(1-t)^2-style divergences at t = +/-1, so we
# stay strictly inside. The value must be representable in the working
# dtype and large enough that (1 - t)^2 doesn't underflow; float32 has
# eps ~ 1.19e-7 so anything smaller than ~1e-7 collapses to the
# boundary exactly. 1e-6 gives a safety margin while truncating at an
# x-value far beyond where the PDF has any mass (for the _map_ainf
# transform with a=0 and t=0.999999, x ~ 2e6, where every PDF in the
# library has underflowed to 0).
_T_EPS = 1e-6


# Precomputed Gauss-Legendre nodes and weights on [-1, 1], stored as
# JAX arrays at module load so they are baked into JIT traces as
# constants rather than recomputed per call.
#
# GL16 (31st-order): used by the PPF cubic-spline builder in
# ``_cdf_grid_piecewise``, which feeds a dense monotonic grid where
# each segment is narrow and GL16 is sufficient.
#
# GL32 (63rd-order): used by the public ``_cdf`` piecewise path in
# t-space. Its denser node set resolves the transformed integrand's
# endpoint behaviour cleanly for every support type.
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
    r"""Scalar CDF at ``xi`` via shorter-tail switch + K-breakpoint subdivision.

    Used by the custom VJP forward rule to compute per-xi values +
    parameter gradients. The public array CDF path uses the t-space
    piecewise routine below instead.

    The integration form is chosen per xi:

    * ``xi <= bps[-1]``: integrate ``[lower, xi]`` (lower-tail form).
    * ``xi >  bps[-1]``: integrate ``[xi, upper]`` and return
      ``1 - integral`` (upper-tail form).

    All K breakpoints are passed as interior subdivisions; quadax
    auto-collapses those outside ``[start, end]`` to zero-length
    segments. Boundary cases ``xi <= lower`` and ``xi >= upper``
    short-circuit to 0 / 1 respectively, also avoiding the zero-length
    interval that quadgk otherwise returns as NaN.
    """
    at_lower = (xi <= lower) & jnp.isfinite(lower)
    at_upper = (xi >= upper) & jnp.isfinite(upper)
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


def _piecewise_cdf_tspace(
    dist,
    x_flat: jnp.ndarray,
    bps: jnp.ndarray,
    lower: float,
    upper: float,
    params_array: jnp.ndarray,
) -> jnp.ndarray:
    r"""Piecewise GL32 CDF in quadax-transformed t-space.

    Steps:
      1. Compute the bitmask / scale / sign for the support ``[lower, upper]``
         (same logic as quadax's ``map_interval``).
      2. Map user x and breakpoints to t in ``[-1, 1]`` via ``MAPFUNS_INV``.
         ``xi = +/-inf`` lands cleanly at ``t = +/-1``.
      3. Augment t_aug with left/right sentinel endpoints at
         ``+/-(1 - _T_EPS)`` so the prefix sum anchors at F=0 at the
         support's lower bound.
      4. Sort, compute per-segment GL32 integrals of the transformed
         integrand ``pdf(x(t)) * (dx/dt)``, prefix-sum.
      5. Inverse-permute, slice out the user entries.

    Returns an array of shape ``x_flat.shape`` with raw CDF values
    (pre-clip, pre-support-enforcement).
    """
    n = x_flat.shape[0]

    # Step 1: reproduce quadax's mapping setup directly (no reliance on
    # the _MappedFunction class; we only need the scalar a, b, bitmask
    # so we can forward-map x via MAPFUNS_INV and compute the
    # transformed integrand ourselves).
    a = lower
    b = upper
    # sgn is always +1 here because support bounds are ordered lower <= upper.
    bitmask = jnp.isinf(a).astype(jnp.int32) + 2 * jnp.isinf(b).astype(jnp.int32)

    # Step 2: forward-map x and bps to t-space. lax.switch dispatches
    # on the traced bitmask; each branch is a MAPFUNS_INV function
    # taking (x_values, a, b) and returning t_values.
    def _forward_map(x_values):
        return lax.switch(bitmask, MAPFUNS_INV, x_values, a, b)

    t_user = _forward_map(x_flat)
    t_bps = _forward_map(bps)

    # MAPFUNS_INV can produce NaN at infinite endpoints (division by
    # zero in _map_ainf_inv etc.). Replace those with the correct
    # boundary value explicitly.
    t_user = jnp.where(jnp.isinf(x_flat) & (x_flat > 0), 1.0, t_user)
    t_user = jnp.where(jnp.isinf(x_flat) & (x_flat < 0), -1.0, t_user)
    t_bps = jnp.where(jnp.isinf(bps) & (bps > 0), 1.0, t_bps)
    t_bps = jnp.where(jnp.isinf(bps) & (bps < 0), -1.0, t_bps)

    # Clip away from +/-1 to avoid the sec^2 divergence at the
    # boundary. Still captures all of the distribution's mass because
    # the transformed integrand vanishes at +/-1 for any convergent PDF.
    t_user = jnp.clip(t_user, -1.0 + _T_EPS, 1.0 - _T_EPS)
    t_bps = jnp.clip(t_bps, -1.0 + _T_EPS, 1.0 - _T_EPS)

    # Step 3: augment with sentinel endpoints so the prefix sum has
    # F = 0 at the left boundary.
    t_left = jnp.asarray(-1.0 + _T_EPS)
    t_right = jnp.asarray(1.0 - _T_EPS)
    t_aug = jnp.concatenate([
        t_user,                  # (n,)  — positions [0..n)
        t_bps,                   # (mk,) — positions [n..n+mk)
        t_left[None],            # (1,)  — position  n+mk
        t_right[None],           # (1,)  — position  n+mk+1
    ])

    # Step 4: sort, compute segment integrals via GL32.
    order = jnp.argsort(t_aug, stable=True)
    inv_order = jnp.argsort(order, stable=True)
    t_sorted = t_aug[order]

    seg_a = t_sorted[:-1]
    seg_b = t_sorted[1:]
    half = (seg_b - seg_a) / 2.0           # (L-1,)
    mid = (seg_a + seg_b) / 2.0            # (L-1,)
    eval_t = mid[:, None] + half[:, None] * _GL32_NODES[None, :]  # (L-1, 32)
    eval_t = jnp.clip(eval_t, -1.0 + _T_EPS, 1.0 - _T_EPS)

    # Transformed integrand: pdf(x(t)) * (dx/dt). Use MAPFUNS directly
    # (the forward transform returns both x and w = dx/dt).
    def _integrand(t_val):
        x_val, w = lax.switch(bitmask, MAPFUNS, t_val, a, b)
        pdf_val = dist._pdf_for_cdf(x_val, params_array)
        return pdf_val * w

    integrand_vals = vmap(vmap(_integrand))(eval_t)  # (L-1, 32)
    segment_integrals = (
        integrand_vals * _GL32_WEIGHTS[None, :]
    ).sum(axis=1) * half                    # (L-1,)

    # Step 4 cont'd: prefix sum. F at the left-most sorted t equals 0
    # (that t is the left sentinel t_left, i.e. corresponds to the
    # distribution's lower support bound).
    cdf_sorted = jnp.concatenate([
        jnp.array([0.0]),
        jnp.cumsum(segment_integrals),
    ])                                       # (L,)

    # Step 5: inverse-permute and slice out user entries.
    cdf_aug = cdf_sorted[inv_order]
    return cdf_aug[:n]


def _cdf(dist, x: jnp.ndarray, params: dict) -> jnp.ndarray:
    r"""Public CDF via piecewise GL32 in t-space.

    Single code path — no gated fallback, no scalar vmap branch.
    Handles ``xi`` at arbitrary extremes (including ``+/-inf``) by
    construction because the quadax transform maps the full support
    to the bounded ``[-1, 1]``. Assumes the PDF is analytically
    normalised (integrates to 1).
    """
    x_in, xshape = _univariate_input(x)
    x_flat = x_in.flatten()

    lower, upper = dist._support_bounds(params)
    bps = jnp.asarray(dist._cdf_breakpoints(params)).flatten()
    params_array: jnp.ndarray = dist._params_to_array(params)

    cdf_raw = _piecewise_cdf_tspace(dist, x_flat, bps, lower, upper, params_array)
    return jnp.clip(cdf_raw, 0.0, 1.0).reshape(xshape)


def _cdf_grid_piecewise(
    dist, x_grid: jnp.ndarray, params: dict
) -> jnp.ndarray:
    r"""Compute the CDF at a dense sorted grid via piecewise GL16.

    Used by the cubic-spline PPF builder, which passes a dense sorted
    grid spanning the distribution's bulk. The grid's density
    guarantees each segment is narrow enough that 16-point
    Gauss-Legendre resolves the PDF decay within each segment to
    float32 machine precision.
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


def _cdf_fwd(dist, cdf_func: Callable, x: jnp.ndarray, params: dict):
    """Forward pass for the custom CDF VJP.

    Calls ``_cdf_single_x`` directly per-xi under vmap to compute both
    the forward CDF value and the parameter gradients via
    ``value_and_grad``. Bypasses the t-space piecewise ``_cdf`` entirely
    because gradient computation doesn't benefit from the piecewise
    speedup (each xi still needs its own adaptive call for the
    parameter-derivative tracing).

    The ``cdf_func`` argument is accepted for backwards compatibility
    with existing call sites but is not used — the scalar routine is
    invoked directly.
    """
    x_in, xshape = _univariate_input(x)

    lower, upper = dist._support_bounds(params)
    bps = jnp.asarray(dist._cdf_breakpoints(params)).flatten()

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
