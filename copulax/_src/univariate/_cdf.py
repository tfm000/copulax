"""Implements a jit-able, jax-differentiable version of numerical univariate cdf integration."""

import numpy as np
from jax import numpy as jnp
from jax import grad, vmap, value_and_grad
from typing import Callable
from quadax import quadgk, quadcc


from copulax._src.univariate._utils import _univariate_input


METHOD: Callable = quadgk


# Precomputed 16-point Gauss-Legendre nodes and weights on [-1, 1].
# Stored as JAX arrays at module load so they are baked into JIT
# traces as constants rather than being recomputed per call.
_GL16_NODES_NP, _GL16_WEIGHTS_NP = np.polynomial.legendre.leggauss(16)
_GL16_NODES = jnp.asarray(_GL16_NODES_NP)
_GL16_WEIGHTS = jnp.asarray(_GL16_WEIGHTS_NP)


def _cdf_single_x(
    pdf_func: Callable, lower_bound: float, xi: float, params_array
) -> float:
    """Compute the CDF at a single point by numerical integration of the PDF."""
    cdf_vals, info = METHOD(fun=pdf_func, interval=(lower_bound, xi), args=params_array)
    return cdf_vals.reshape(())


def _cdf(dist, x: jnp.ndarray, params: dict) -> jnp.ndarray:
    """Compute the CDF by numerically integrating the PDF from the lower support bound.

    Assumes the PDF is analytically normalised (integrates to 1). Use
    ``_cdf_normalised`` instead when the PDF may not integrate exactly to 1
    (e.g. due to numerical issues in the log-density).
    """
    x, xshape = _univariate_input(x)
    lower_bound, upper_bound = dist.support(params)
    params_array: jnp.ndarray = dist._params_to_array(params)

    # vectorize CDF computation across all x values
    _cdf_vec = vmap(
        lambda xi: _cdf_single_x(dist._pdf_for_cdf, lower_bound, xi, params_array)
    )
    cdf_raw = _cdf_vec(x.flatten())
    cdf_adj = jnp.clip(cdf_raw, 0.0, 1.0)

    return cdf_adj.reshape(xshape)


def _cdf_grid_piecewise(
    dist, x_grid: jnp.ndarray, params: dict
) -> jnp.ndarray:
    r"""Compute the CDF at a sorted grid of x values via piecewise
    Gauss-Legendre quadrature.

    Equivalent in semantics to ``vmap(dist.cdf)(x_grid)`` for the
    common case of a sorted grid, but much faster when the grid is
    dense (e.g. the 100-point grid built by the cubic-spline PPF).

    The standard CDF integrator runs an independent adaptive
    ``quadgk`` call from the lower support bound to each query point,
    so a 100-point grid costs ~100 independent adaptive quadratures.
    This function instead computes the tail integral once,
    ``F(x_grid[0]) = ∫_{lower}^{x_grid[0]} f(t) dt``, then accumulates
    short segment integrals
    ``F(x_grid[k]) = F(x_grid[k-1]) + ∫_{x_grid[k-1]}^{x_grid[k]} f(t) dt``
    via a fixed 16-point Gauss-Legendre rule.  All segment evaluations
    are dispatched as a single batched PDF call so that the underlying
    Bessel/special-function work fully vectorises across both segments
    and quadrature nodes.

    The 16-point rule integrates polynomials up to degree 31 exactly;
    on the smooth log-Bessel-based PDFs used by the GH/Skewed-T
    families, each segment is well within the regime where this is
    accurate to several digits past the noise floor of the underlying
    PDF.

    This function is intentionally **not** wired into the public
    :py:meth:`~copulax._src._distributions.Univariate.cdf` interface,
    which must continue to handle arbitrary unsorted ``x``.  It is
    consumed only by the cubic-spline PPF builder, which always passes
    a sorted ``linspace`` grid.

    Args:
        dist: Distribution object exposing ``support(params)`` and
            ``_pdf_for_cdf(x, *params_tuple)``.
        x_grid: Sorted 1D array of x values at which to evaluate the
            CDF.  Must contain at least one element.
        params: Parameter dictionary for ``dist``.

    Returns:
        1D array of CDF values at each grid point, clipped to ``[0, 1]``.
    """
    x_grid = jnp.asarray(x_grid).flatten()
    lower_bound, _ = dist.support(params)
    params_array: jnp.ndarray = dist._params_to_array(params)

    # Tail integral [lower_bound, x_grid[0]] via adaptive quadgk.
    # This is one call regardless of grid size, so the per-point
    # adaptive cost only matters for the long tail (where it is
    # genuinely needed).
    tail = _cdf_single_x(
        dist._pdf_for_cdf, lower_bound, x_grid[0], params_array
    )

    # Short segment integrals via fixed-order Gauss-Legendre.
    a = x_grid[:-1]
    b = x_grid[1:]
    half = (b - a) / 2.0  # (N-1,)
    mid = (a + b) / 2.0  # (N-1,)

    # Evaluation points: shape (N-1, n_quad).  Each row is a single
    # segment with the GL nodes affinely mapped from [-1, 1] to [a, b].
    eval_points = mid[:, None] + half[:, None] * _GL16_NODES[None, :]

    # Batched PDF evaluation.  ``_pdf_for_cdf`` is the same hook used
    # by ``quadgk``; it accepts a single positional ``params_array``.
    pdf_at = vmap(
        vmap(lambda xi: dist._pdf_for_cdf(xi, params_array))
    )(eval_points)

    # Weighted sum per segment, scaled by half-width:
    #   ∫_{a}^{b} f(t) dt ≈ half * Σ_k w_k f(mid + half * node_k)
    segment_integrals = (pdf_at * _GL16_WEIGHTS[None, :]).sum(axis=1) * half

    cdf_values = jnp.concatenate(
        [jnp.array([tail]), tail + jnp.cumsum(segment_integrals)]
    )
    return jnp.clip(cdf_values, 0.0, 1.0)


def _cdf_normalised(dist, x: jnp.ndarray, params: dict) -> jnp.ndarray:
    """Compute the CDF with explicit normalisation by the total PDF integral.

    Divides the raw integral by the full-support integral so that the CDF
    reaches exactly 1.  This is necessary when the PDF implementation has
    known numerical inaccuracies (e.g. Bessel-function underflow) that
    prevent the density from integrating to 1.
    """
    x, xshape = _univariate_input(x)
    lower_bound, upper_bound = dist.support(params)
    params_array: jnp.ndarray = dist._params_to_array(params)

    # compute normalising constant (full-support integral) once
    scale = _cdf_single_x(dist._pdf_for_cdf, lower_bound, upper_bound, params_array)

    # vectorize CDF computation across all x values
    _cdf_vec = vmap(
        lambda xi: _cdf_single_x(dist._pdf_for_cdf, lower_bound, xi, params_array)
    )
    cdf_raw = _cdf_vec(x.flatten())

    # scale to [0, 1]
    cdf_adj = cdf_raw / scale
    cdf_adj = jnp.clip(cdf_adj, 0.0, 1.0)

    return cdf_adj.reshape(xshape)


def _cdf_fwd(dist, cdf_func: Callable, x: jnp.ndarray, params: dict):
    """Forward pass for the custom CDF VJP: returns CDF values and residuals for backward."""
    x, xshape = _univariate_input(x)

    def cdf_single(xi, params):
        return cdf_func(xi, params).reshape(())

    # vmap value_and_grad to parallelize across x values
    _val_and_grad = value_and_grad(cdf_single, argnums=1)
    _val_and_grad_vec = vmap(lambda xi: _val_and_grad(xi, params))

    cdf_values, param_grads = _val_and_grad_vec(x.flatten())
    pdf_values = dist.pdf(x=x, params=params).reshape(xshape)
    return cdf_values.reshape(xshape), (pdf_values, param_grads)


def cdf_bwd(res, g):
    """Backward pass for the custom CDF VJP: computes gradients w.r.t. x and params."""
    xshape = res[0].shape
    g = g.reshape(xshape)
    x_grad = res[0] * g
    param_grads: dict = {
        key: jnp.sum(jnp.nan_to_num(val, 0.0) * g) for key, val in res[1].items()
    }  # sum parameter gradients over x
    return x_grad, param_grads
