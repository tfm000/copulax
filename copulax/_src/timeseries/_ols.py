r"""Plain OLS helper used by the diagnostics and stationarity machinery.

JAX exposes ``jnp.linalg.lstsq`` and ``jax.scipy.linalg.lstsq``, but
both return only ``(beta, sum_sq_residuals, rank, singular_values)``.
None of CopulAX's call sites use that surface — we always need the
residual *vector* (for downstream Q² / R² / chained regressions) and
sometimes the t-statistics on individual coefficients.  This module
exposes those as a single :class:`OLSResult` PyTree so the three
internal call sites — Engle's ARCH-LM auxiliary regression
(``copulax._src.timeseries._diagnostics.arch_lm``), the ADF
Dickey-Fuller t-stat, and the KPSS deterministic-component regression
(both in ``copulax._src.timeseries._unit_root``) — share one
numerically audited path rather than re-deriving SEs / residuals
each time.

Pure JAX, JIT- and autograd-compatible end-to-end.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from copulax._src.timeseries._se import safe_solve


class OLSResult(NamedTuple):
    r"""Output of :func:`ols_fit` — every leaf is a JAX array.

    Attributes:
        beta: shape ``(k,)`` — OLS estimates.
        fitted: shape ``(n,)`` — :math:`X \hat\beta`.
        residuals: shape ``(n,)`` — :math:`y - X \hat\beta`.
        sigma2: scalar — :math:`\hat\sigma^2 = \mathrm{RSS} / (n - k)`,
            unbiased under homoskedastic errors.
        standard_errors: shape ``(k,)`` —
            :math:`\hat\sigma \sqrt{\mathrm{diag}((X^T X)^{-1})}`.
        t_stats: shape ``(k,)`` —
            :math:`\hat\beta_j / \mathrm{SE}(\hat\beta_j)`.
        r_squared: scalar — :math:`1 - \mathrm{RSS} / \mathrm{TSS}`
            against the sample mean of ``y``.
        adj_r_squared: scalar — Theil (1961) degrees-of-freedom-
            corrected coefficient of determination,
            :math:`1 - (1 - R^2) \cdot \frac{n - 1}{n - k}`.  ``k``
            counts every column of ``X`` (including the intercept if
            the caller put one in), matching the convention used by
            ``statsmodels``' ``OLSResults.rsquared_adj``.
        ill_conditioned: boolean scalar — ``True`` when the Gram matrix
            :math:`X^T X` is numerically singular (condition number
            above :data:`copulax._src.timeseries._se._COND_THRESHOLD`).
            When set, ``beta`` / ``standard_errors`` / ``t_stats`` are
            ``NaN`` (the honest diagnostic) rather than a finite but
            meaningless value.  Callers that only need
            ``residuals`` / ``r_squared`` (e.g. the ARCH-LM statistic)
            can inspect this flag to distinguish a degenerate auxiliary
            regression from a well-posed one.
    """

    beta: Array
    fitted: Array
    residuals: Array
    sigma2: Array
    standard_errors: Array
    t_stats: Array
    r_squared: Array
    adj_r_squared: Array
    ill_conditioned: Array


def ols_fit(X: ArrayLike, y: ArrayLike) -> OLSResult:
    r"""Solve :math:`\min_\beta \|y - X\beta\|_2^2` and return the full
    inferential bundle.

    Uses the symmetric normal-equations solve
    :math:`X^T X \hat\beta = X^T y` (cheaper than SVD for the small
    ``k`` regimes inside CopulAX — ``k ≤ 2 + lags`` for ADF,
    ``k ≤ 2`` for KPSS, ``k = 1 + lags`` for ARCH-LM).  Both the
    coefficient solve and the standard-error / t-stat path go through
    :func:`copulax._src.timeseries._se.safe_solve`, the shared
    condition-number guard used by the covariance machinery — the SE
    path solves ``X^T X \cdot M = I`` rather than forming an explicit
    ``jnp.linalg.inv`` so the two paths share one numerically audited
    routine.  If ``X^T X`` is numerically singular (collinear columns,
    or too few observations relative to ``k`` — e.g. a series with
    constant regions feeding the ARCH-LM auxiliary regression) both
    solves return ``NaN`` and the :attr:`OLSResult.ill_conditioned`
    flag is set, rather than a finite but meaningless coefficient / SE.

    ``jnp.maximum(n - k, 1)`` floors the residual degrees of freedom so
    ``sigma²`` / ``r_squared`` stay finite on degenerate just-identified
    inputs (``sigma²`` has no statistical interpretation in that regime;
    finiteness keeps gradients flowing through callers that use ARCH-LM
    as a soft penalty during optimisation).

    Args:
        X: shape ``(n, k)`` — design matrix.
        y: shape ``(n,)`` — target.

    Returns:
        :class:`OLSResult` PyTree with the eight inferential fields
        (including the ``ill_conditioned`` diagnostic).
    """
    X_arr = jnp.asarray(X, dtype=float)
    y_arr = jnp.asarray(y, dtype=float).reshape(-1)
    XtX = X_arr.T @ X_arr
    Xty = X_arr.T @ y_arr
    n, k = X_arr.shape
    # Guarded coefficient solve — a singular Gram matrix surfaces NaN
    # (with the ill_conditioned flag) instead of a silent finite beta.
    beta, ill_beta = safe_solve(XtX, Xty)
    fitted = X_arr @ beta
    residuals = y_arr - fitted
    df_resid = jnp.maximum(n - k, 1)
    rss = jnp.sum(residuals ** 2)
    sigma2 = rss / df_resid
    # SE path: guarded solve of X^T X · M = I (consistent with _se.py)
    # in place of an explicit jnp.linalg.inv.
    eye_k = jnp.eye(k, dtype=XtX.dtype)
    XtX_inv, ill_inv = safe_solve(XtX, eye_k)
    standard_errors = jnp.sqrt(sigma2 * jnp.diag(XtX_inv))
    t_stats = beta / standard_errors
    tss = jnp.sum((y_arr - jnp.mean(y_arr)) ** 2)
    r_squared = 1.0 - rss / jnp.maximum(tss, 1e-30)
    df_total = jnp.maximum(n - 1, 1)
    adj_r_squared = 1.0 - (1.0 - r_squared) * (df_total / df_resid)
    ill_conditioned = jnp.logical_or(ill_beta, ill_inv)
    return OLSResult(
        beta=beta,
        fitted=fitted,
        residuals=residuals,
        sigma2=sigma2,
        standard_errors=standard_errors,
        t_stats=t_stats,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        ill_conditioned=ill_conditioned,
    )


__all__ = ["OLSResult", "ols_fit"]
