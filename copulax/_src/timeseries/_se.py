r"""Standard-error machinery for the time-series subpackage.

Asymptotic-covariance computation in the **natural parameter space**
at the constrained MLE — matching the convention in ``arch``
(Sheppard) and ``statsmodels.tsa.arima.ARIMA``.  Following ``arch``,
the optimiser may use any reparameterisation it likes (softmax
simplex for stationarity, softplus for positivity, reflection
coefficients for AR/MA invertibility); the covariance pipeline
discards all of that and computes the Hessian / scores directly on
the constrained natural parameters.

Three cov-type formulas, mirroring ``arch``:

* ``"robust"`` — Bollerslev-Wooldridge sandwich (the default; robust
  to misspecification of the residual law)::

        V = J⁻¹ · S · J⁻¹ / n

* ``"classic"`` — observed information / inverse Hessian (correct
  under correct specification)::

        V = J⁻¹ / n

* ``"opg"`` — outer product of gradients / BHHH (asymptotically
  equivalent to ``classic`` under correct specification but uses
  scores only — no Hessian)::

        V = (Sᵀ S)⁻¹ / n  ≡ S⁻¹ / n

Notation:

* :math:`\mathrm{params\_flat}` — flat natural-parameter vector
  (e.g. for vanilla GARCH(1,1):
  :math:`(\omega, \alpha_1, \beta_1, \text{residual shape}\dots)`).
* :math:`\ell_t(\theta)` — per-observation log-likelihood.
* :math:`J = -(1/n) \sum_t \partial^2 \ell_t / \partial \theta
  \partial \theta^\top` — observed information per observation
  (matches ``arch``'s ``hess = approx_hess(...) / nobs``).
* :math:`S = \mathrm{Cov}(s_t)` — sample covariance of per-obs
  scores (matches ``arch``'s ``np.cov(scores.T)``, including
  mean-subtraction with ``ddof=1``).
* :math:`V` — asymptotic covariance of the parameter estimate.

References:

* ``arch.univariate.base.compute_param_cov`` (lines 885-932) —
  the canonical formula we mirror.
* Bollerslev, T. & Wooldridge, J. (1992). *Quasi-maximum
  likelihood estimation and inference in dynamic models with
  time-varying covariances*.  Econometric Reviews, 11(2), 143-172.

The plan's Pagan-Newey two-stage sandwich for the separable
``ARMA → GARCH-on-residuals`` workflow is a separate computation
on top of this module's natural-space ``J``, ``S`` blocks; that
machinery is deferred to a future commit per plan §"Standard
errors".
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array


_VALID_COV_TYPES = frozenset({"robust", "classic", "opg"})


def per_obs_score(
    per_obs_nll: Callable[[Array], Array],
    params_flat: Array,
) -> Array:
    r"""Per-observation score matrix
    :math:`s_t = \partial \ell_t / \partial \theta`.

    ``arch`` computes this via ``approx_fprime(..., individual=True)``;
    we use :func:`jax.jacrev` for analytical correctness — equivalent
    asymptotically and exact in finite samples.

    Args:
        per_obs_nll: Closure
            ``params_flat -> (n,) per-observation negative
            log-likelihoods``.
        params_flat: Flat natural-parameter vector at the MLE.

    Returns:
        ``(n, k)`` matrix; row ``t`` is :math:`s_t`.
    """
    return jax.jacrev(per_obs_nll)(params_flat)


def per_obs_information(
    nll_total: Callable[[Array], Array],
    params_flat: Array,
    n_obs: int,
) -> Array:
    r"""Per-observation observed information matrix
    :math:`J = -(1/n) \sum_t \partial^2 \ell_t / \partial \theta \partial \theta^\top`.

    Computed as :math:`(1/n) \cdot \mathrm{hess}(\sum_t -\ell_t)` —
    matches ``arch.univariate.base.compute_param_cov`` line 904
    (``hess = approx_hess(self._loglikelihood, ...) / nobs``).

    Args:
        nll_total: Closure
            ``params_flat -> sum_{t=1..n} -ell_t`` (the **sum**
            negative log-likelihood, not the mean).
        params_flat: Flat natural-parameter vector at the MLE.
        n_obs: Number of observations ``n``.

    Returns:
        ``(k, k)`` per-observation observed information matrix.
    """
    H_total = jax.hessian(nll_total)(params_flat)
    return H_total / n_obs


def score_covariance(scores: Array) -> Array:
    r"""Sample covariance of the per-obs scores,
    :math:`S = \mathrm{Cov}(s_t)`.

    Mirrors ``arch``'s ``np.cov(scores.T)``: mean-subtraction with
    Bessel's correction (``ddof=1``).  This is the standard
    finite-sample estimator of the score covariance under correct
    specification.

    Args:
        scores: ``(n, k)`` per-observation score matrix.

    Returns:
        ``(k, k)`` score covariance matrix.
    """
    n = scores.shape[0]
    mean_score = jnp.mean(scores, axis=0, keepdims=True)
    demeaned = scores - mean_score
    return (demeaned.T @ demeaned) / (n - 1)


def compute_param_cov(
    nll_total: Callable[[Array], Array],
    per_obs_nll: Callable[[Array], Array],
    params_flat: Array,
    n_obs: int,
    cov_type: str = "robust",
) -> Array:
    r"""Asymptotic covariance of the natural-parameter MLE.

    Implements the three formulas from ``arch.univariate.base
    .compute_param_cov``:

    .. math::

        V_{\mathrm{robust}}  &= J^{-1}\, S\, J^{-1} / n
                                & \text{(Bollerslev-Wooldridge sandwich)},\\
        V_{\mathrm{classic}} &= J^{-1} / n
                                & \text{(observed information)},\\
        V_{\mathrm{opg}}     &= S^{-1} / n
                                & \text{(outer product of gradients / BHHH)}.

    All three are computed in natural parameter space at the
    interior MLE — :math:`J` is full rank there, so plain
    :func:`jnp.linalg.solve` suffices (no pseudo-inverse needed).

    Args:
        nll_total: Closure ``params_flat -> sum_t -ell_t``.
        per_obs_nll: Closure
            ``params_flat -> (n,) per-obs negative log-likelihoods``.
        params_flat: Flat natural-parameter vector at the MLE.
        n_obs: Number of observations.
        cov_type: One of ``"robust"`` (default, BW sandwich),
            ``"classic"`` (observed information), or ``"opg"``
            (outer product of gradients).

    Returns:
        ``(k, k)`` asymptotic covariance matrix.

    Raises:
        ValueError: When ``cov_type`` is not one of the supported
            strings.
    """
    if cov_type not in _VALID_COV_TYPES:
        raise ValueError(
            f"cov_type must be one of {sorted(_VALID_COV_TYPES)}; "
            f"got {cov_type!r}."
        )

    k = params_flat.shape[0]
    eye_k = jnp.eye(k, dtype=params_flat.dtype)

    if cov_type == "opg":
        scores = per_obs_score(per_obs_nll, params_flat)
        S = score_covariance(scores)
        return jnp.linalg.solve(S, eye_k) / n_obs

    # Both "classic" and "robust" need the inverse Hessian.
    J = per_obs_information(nll_total, params_flat, n_obs)
    inv_J = jnp.linalg.solve(J, eye_k)

    if cov_type == "classic":
        return inv_J / n_obs

    # cov_type == "robust" — Bollerslev-Wooldridge sandwich.
    scores = per_obs_score(per_obs_nll, params_flat)
    S = score_covariance(scores)
    return inv_J @ S @ inv_J / n_obs


###############################################################################
# Param-dict ↔ flat-vector helpers
###############################################################################
def _flatten_dict(d: dict, parent_key: str = "") -> list[tuple[str, Array]]:
    r"""Recursively flatten a nested params dict to a list of
    ``(qualified_key, leaf_array)`` tuples in deterministic
    sorted-key order.
    """
    items: list[tuple[str, Array]] = []
    for k in sorted(d.keys()):
        v = d[k]
        full_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            sub_items = _flatten_dict(v, full_key)
            if sub_items:
                items.extend(sub_items)
            else:
                # Preserve empty sub-dicts in the schema so the
                # round-trip ``params_to_flat`` → ``flat_to_params``
                # produces the same top-level keys as the input.
                items.append((full_key + ".__empty__", jnp.zeros((0,), dtype=float)))
        else:
            items.append((full_key, jnp.asarray(v, dtype=float)))
    return items


def params_to_flat(
    params: dict,
) -> tuple[Array, list[tuple[str, tuple[int, ...]]]]:
    r"""Flatten a constrained-params dict to a single vector with a
    recoverable schema.

    Empty sub-dicts (e.g. ``residual={}`` for the Normal residual
    law, which has zero shape parameters) are preserved via a
    sentinel schema entry so the round-trip with
    :func:`flat_to_params` reproduces the original top-level keys.
    """
    items = _flatten_dict(params)
    schema: list[tuple[str, tuple[int, ...]]] = []
    pieces: list[Array] = []
    for key, leaf in items:
        leaf = jnp.asarray(leaf, dtype=float)
        pieces.append(leaf.flatten())
        schema.append((key, leaf.shape))
    flat = jnp.concatenate(pieces) if pieces else jnp.zeros((0,), dtype=float)
    return flat, schema


def flat_to_params(
    flat: Array, schema: list[tuple[str, tuple[int, ...]]],
) -> dict:
    r"""Inverse of :func:`params_to_flat`: rebuild a nested params
    dict from a flat vector and the schema returned by the
    forward pass.
    """
    out: dict = {}
    idx = 0
    for key, shape in schema:
        size = (
            int(jnp.prod(jnp.asarray(shape, dtype=int))) if shape else 1
        )
        chunk = flat[idx : idx + size].reshape(shape)
        idx += size
        parts = key.split(".")
        # Sentinel for empty sub-dict: ensure the parent dict exists
        # but don't add the sentinel key itself.
        if parts[-1] == "__empty__":
            node = out
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    node[part] = {}
                node = node[part]
            continue
        node = out
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = chunk
    return out


__all__ = [
    "compute_param_cov",
    "per_obs_information",
    "per_obs_score",
    "score_covariance",
    "params_to_flat",
    "flat_to_params",
]
