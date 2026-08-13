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

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

_VALID_COV_TYPES = frozenset({"robust", "classic", "opg"})


# Condition-number ceiling for the covariance / OLS linear solves.
#
# A square system ``A x = b`` loses roughly ``log10(cond(A))`` decimal
# digits of accuracy.  In float64 (``eps ≈ 2.2e-16``, ``1/eps ≈ 4.5e15``)
# a matrix with ``cond(A) ≳ 1/eps`` is numerically singular: the solve
# retains no significant digits and any finite result it returns is
# meaningless.  We set the ceiling one order of magnitude below the
# reciprocal machine epsilon so a solve that has lost all but ~1 digit is
# already treated as degenerate.  This mirrors the ``rcond = eps * max(M, N)``
# default that LAPACK / NumPy use to declare rank deficiency in ``lstsq``.
#
# Rationale for float64: every CopulAX numerical path runs with
# ``jax_enable_x64`` for the log-likelihood / Hessian arithmetic, so the
# float64 reciprocal-eps ceiling is the operative one.  Evaluated from
# ``jnp.finfo`` at import so the constant tracks the active default dtype
# rather than hard-coding a literal.
_COND_THRESHOLD: float = 0.1 / float(jnp.finfo(jnp.result_type(float)).eps)


def safe_solve(A: Array, rhs: Array) -> tuple[Array, Array]:
    r"""Condition-number-guarded linear solve ``A x = rhs``.

    Computes :func:`jnp.linalg.solve` for well-conditioned ``A`` and
    surfaces a diagnostic for degenerate ``A`` instead of silently
    returning a finite-but-meaningless result.

    The guard checks :func:`jnp.linalg.cond` against
    :data:`_COND_THRESHOLD` (``≈ 0.1/eps``).  When ``A`` is
    ill-conditioned the solution is replaced element-wise with ``NaN``
    and the boolean ``ill_conditioned`` flag is set.  ``NaN`` is the
    honest signal here — for a covariance / standard-error solve a
    pseudo-inverse (``lstsq``) would return a finite number that
    silently drops the unidentified null-space directions of a
    rank-deficient Hessian, i.e. exactly the "finite, plausible-looking
    but wrong" failure the project forbids.  A ``NaN`` propagates
    through the downstream ``sqrt(maximum(diag(cov), 0))`` reduction
    (``maximum(NaN, 0)`` is ``NaN``) so a degenerate fit surfaces
    ``NaN`` standard errors rather than a clamped, finite ``0``.

    Fully JIT- and autograd-compatible: the branch is a value-level
    :func:`jnp.where` on a traced predicate with static output shapes —
    no Python control flow over traced values (Pitfall 5), and the
    idiom matches the ``_safe_div`` guard in
    :mod:`copulax._src.optimize`.

    Args:
        A: ``(k, k)`` square matrix (a Hessian / observed-information /
            score-covariance / Gram matrix at the MLE).
        rhs: ``(k,)`` or ``(k, m)`` right-hand side.

    Returns:
        ``(x, ill_conditioned)`` where ``x`` has the shape of
        ``jnp.linalg.solve(A, rhs)`` (``NaN``-filled when
        ill-conditioned) and ``ill_conditioned`` is a boolean scalar
        array.
    """
    x = jnp.linalg.solve(A, rhs)
    ill_conditioned = jnp.linalg.cond(A) > _COND_THRESHOLD
    x = jnp.where(ill_conditioned, jnp.full_like(x, jnp.nan), x)
    return x, ill_conditioned


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
    interior MLE.  Under correct specification :math:`J` (and the
    score covariance :math:`S`) is positive-definite there, so a plain
    linear solve suffices — no pseudo-inverse is needed on the happy
    path.  That positive-definiteness can nonetheless fail in practice
    at a boundary solution, on a near-flat likelihood, or when the
    series is too short for the model order; to keep those degenerate
    cases from returning a silent finite-but-meaningless standard error
    the solves are routed through :func:`safe_solve`, which surfaces
    ``NaN`` (propagating through ``sqrt(diag(cov))``) when the matrix is
    ill-conditioned.  On a well-conditioned :math:`J` / :math:`S` the
    guard is a no-op and the result is numerically identical to
    :func:`jnp.linalg.solve`.

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
            f"cov_type must be one of {sorted(_VALID_COV_TYPES)}; got {cov_type!r}."
        )

    k = params_flat.shape[0]
    eye_k = jnp.eye(k, dtype=params_flat.dtype)

    if cov_type == "opg":
        # Outer product of gradients / BHHH — Berndt, Hall, Hall &
        # Hausman (1974).  Guarded solve: a singular score covariance
        # (e.g. a flat score direction) surfaces NaN, not a silent SE.
        scores = per_obs_score(per_obs_nll, params_flat)
        S = score_covariance(scores)
        inv_S, _ill_S = safe_solve(S, eye_k)
        return inv_S / n_obs

    # Both "classic" and "robust" need the inverse Hessian.  Guarded
    # solve: a rank-deficient / near-singular J surfaces NaN downstream.
    J = per_obs_information(nll_total, params_flat, n_obs)
    inv_J, _ill_J = safe_solve(J, eye_k)

    if cov_type == "classic":
        # Observed information / inverse Hessian — Hamilton (1994),
        # sec. 5.8.
        return inv_J / n_obs

    # cov_type == "robust" — Bollerslev-Wooldridge sandwich
    # (Bollerslev & Wooldridge 1992); J^{-1} S J^{-1}.
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
    flat: Array,
    schema: list[tuple[str, tuple[int, ...]]],
) -> dict:
    r"""Inverse of :func:`params_to_flat`: rebuild a nested params
    dict from a flat vector and the schema returned by the
    forward pass.
    """
    out: dict = {}
    idx = 0
    for key, shape in schema:
        # Static Python arithmetic on the schema's shape tuples —
        # ``jnp`` ops here would be staged (and hence break ``int()``)
        # under an enclosing ``jax.jit`` trace.  ``math.prod(())`` is
        # 1, covering the scalar-leaf case.
        size = math.prod(shape)
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
    # Guarded solve: a degenerate stage-1 Hessian (boundary ARMA fit,
    # near-flat mean likelihood) surfaces NaN rather than a silent SE.
    H11_total = jax.hessian(nll1_total)(params1_flat)
    J11 = H11_total / n_obs
    k1 = params1_flat.shape[0]
    eye_k1 = jnp.eye(k1, dtype=params1_flat.dtype)
    inv_J11, _ill_J11 = safe_solve(J11, eye_k1)

    # ---- Stage-2 own information J22 -------------------------------
    # Guarded solve: a degenerate stage-2 Hessian surfaces NaN
    # downstream through sqrt(diag(V_2)).
    H22_total = jax.hessian(lambda p2: nll2_total_joint(params1_flat, p2))(params2_flat)
    J22 = H22_total / n_obs
    k2 = params2_flat.shape[0]
    eye_k2 = jnp.eye(k2, dtype=params2_flat.dtype)
    inv_J22, _ill_J22 = safe_solve(J22, eye_k2)

    # ---- Cross-stage Hessian J21 -----------------------------------
    # J21 = (1/n) ∂² (sum -ell_2) / ∂θ_2 ∂θ_1^T,
    # built as the Jacobian-w.r.t.-θ_1 of the gradient-w.r.t.-θ_2.
    def grad_nll2_wrt_p2(p1: Array) -> Array:
        return jax.grad(lambda p2: nll2_total_joint(p1, p2))(params2_flat)

    J21 = jax.jacfwd(grad_nll2_wrt_p2)(params1_flat) / n_obs

    # ---- Per-observation scores ------------------------------------
    s1 = jax.jacrev(per_obs_nll1)(params1_flat)  # (n, k1)
    s2 = jax.jacrev(lambda p2: per_obs_nll2_joint(params1_flat, p2))(
        params2_flat
    )  # (n, k2)

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
    "flat_to_params",
    "pagan_newey_cov",
    "params_to_flat",
    "per_obs_information",
    "per_obs_score",
    "safe_solve",
    "score_covariance",
]
