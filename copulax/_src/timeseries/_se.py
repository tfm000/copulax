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

The Pagan-Newey two-stage sandwich for the separable
``ARMA → GARCH-on-residuals`` workflow is implemented as
:func:`pagan_newey_cov` below.  Given closures for both stages'
negative log-likelihoods, it builds the cross-stage Hessian
:math:`J_{21}` via JAX autodiff and corrects the GARCH
covariance for the noise contributed by the ARMA estimate.
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


###############################################################################
# Pagan-Newey two-stage sandwich
###############################################################################
def pagan_newey_cov(
    nll1_total: Callable[[Array], Array],
    per_obs_nll1: Callable[[Array], Array],
    nll2_total_joint: Callable[[Array, Array], Array],
    per_obs_nll2_joint: Callable[[Array, Array], Array],
    params1_flat: Array,
    params2_flat: Array,
    n_obs: int,
) -> Array:
    r"""Pagan-Newey (1988) two-stage covariance sandwich.

    For a separable two-stage MLE:

    1. Stage 1: :math:`\hat\theta_1 = \mathrm{argmin}_\theta\,
       (-\ell_1(\theta; y))`.
    2. Stage 2: :math:`\hat\theta_2 = \mathrm{argmin}_\theta\,
       (-\ell_2(\theta; \hat\theta_1, y))` — the second stage
       likelihood treats :math:`\hat\theta_1` as fixed but its
       output (e.g. the ARMA residual series) implicitly depends
       on :math:`\theta_1`.

    The naive plug-in covariance :math:`J_{22}^{-1} S_{22} J_{22}^{-1}`
    is biased: it ignores the fact that :math:`\hat\theta_1` is itself
    a noisy estimate.  Pagan & Newey's correction (Newey 1984,
    Pagan 1986, Newey & McFadden 1994 §6.2) replaces the per-obs
    score :math:`s_{2,t}` in the sandwich with the **adjusted**
    score

    .. math::

        u_t = s_{2,t} - J_{21}\, J_{11}^{-1}\, s_{1,t},

    where :math:`J_{ij} = (1/n) \sum_t \partial^2(-\ell_2)
    /\partial\theta_i\,\partial\theta_j^\top` and
    :math:`s_{i,t} = \partial(-\ell_{i,t})/\partial\theta_i`.
    The corrected covariance is then

    .. math::

        V_2 = J_{22}^{-1}\,
              \mathrm{Cov}(u_t)\,
              J_{22}^{-\top} \big/ n.

    For an ARMA → GARCH-on-residuals workflow, :math:`J_{21}`
    captures how the GARCH likelihood moves when the ARMA params
    move (through the residual series :math:`\varepsilon_t =
    y_t - \mu_t(\theta_1)`).  When :math:`J_{21} \to 0`
    (independent stages) the formula reduces to the naive plug-in.

    Args:
        nll1_total: Closure ``params1_flat -> sum_t -ell_{1,t}``.
        per_obs_nll1: Closure
            ``params1_flat -> (n,) per-obs negative log-likelihoods``
            for stage 1.
        nll2_total_joint: Closure
            ``(params1_flat, params2_flat) -> sum_t -ell_{2,t}`` —
            stage-2 NLL written as a function of *both* parameter
            vectors.  The dependence on ``params1_flat`` flows
            through the stage-2 inputs (residuals computed from
            stage-1 params).
        per_obs_nll2_joint: Closure
            ``(params1_flat, params2_flat) -> (n,) per-obs
            stage-2 negative log-likelihoods``.
        params1_flat: Stage-1 MLE parameter vector (e.g. the ARMA
            natural parameters at the optimum).
        params2_flat: Stage-2 MLE parameter vector (e.g. the GARCH
            natural parameters at the optimum, fit on the
            stage-1 residuals).
        n_obs: Number of observations.

    Returns:
        ``(k_2, k_2)`` Pagan-Newey corrected covariance of the
        stage-2 estimator.

    References:
        * Newey, W. K. (1984). *A Method of Moments Interpretation
          of Sequential Estimators*. Economics Letters 14(2-3),
          201-206.
        * Pagan, A. (1986). *Two Stage and Related Estimators and
          Their Applications*. Review of Economic Studies 53(4),
          517-538.
        * Newey, W. K., & McFadden, D. (1994). *Large Sample
          Estimation and Hypothesis Testing*. Handbook of
          Econometrics IV, Ch. 36, §6.2.
    """
    # ---- Stage-1 information J11 -----------------------------------
    H11_total = jax.hessian(nll1_total)(params1_flat)
    J11 = H11_total / n_obs
    k1 = params1_flat.shape[0]
    eye_k1 = jnp.eye(k1, dtype=params1_flat.dtype)
    inv_J11 = jnp.linalg.solve(J11, eye_k1)

    # ---- Stage-2 own information J22 -------------------------------
    H22_total = jax.hessian(
        lambda p2: nll2_total_joint(params1_flat, p2)
    )(params2_flat)
    J22 = H22_total / n_obs
    k2 = params2_flat.shape[0]
    eye_k2 = jnp.eye(k2, dtype=params2_flat.dtype)
    inv_J22 = jnp.linalg.solve(J22, eye_k2)

    # ---- Cross-stage Hessian J21 -----------------------------------
    # J21 = (1/n) ∂² (sum -ell_2) / ∂θ_2 ∂θ_1^T,
    # built as the Jacobian-w.r.t.-θ_1 of the gradient-w.r.t.-θ_2.
    grad_nll2_wrt_p2 = lambda p1: jax.grad(
        lambda p2: nll2_total_joint(p1, p2)
    )(params2_flat)
    J21 = jax.jacfwd(grad_nll2_wrt_p2)(params1_flat) / n_obs

    # ---- Per-observation scores ------------------------------------
    s1 = jax.jacrev(per_obs_nll1)(params1_flat)  # (n, k1)
    s2 = jax.jacrev(
        lambda p2: per_obs_nll2_joint(params1_flat, p2)
    )(params2_flat)  # (n, k2)

    # ---- Adjusted scores u_t = s2_t - J21 J11^{-1} s1_t -------------
    # J11^{-1} @ s1.T -> (k1, n); J21 @ that -> (k2, n);
    # transpose -> (n, k2).
    correction = (J21 @ inv_J11 @ s1.T).T
    u = s2 - correction

    # ---- Sample covariance of u_t (Bessel correction) --------------
    n = u.shape[0]
    u_demeaned = u - u.mean(axis=0, keepdims=True)
    Sigma = (u_demeaned.T @ u_demeaned) / (n - 1)

    # ---- Corrected sandwich V_2 = J22^{-1} Sigma J22^{-T} / n ------
    return inv_J22 @ Sigma @ inv_J22.T / n_obs


__all__ = [
    "compute_param_cov",
    "pagan_newey_cov",
    "per_obs_information",
    "per_obs_score",
    "score_covariance",
    "params_to_flat",
    "flat_to_params",
]
