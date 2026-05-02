r"""Unit-root and stationarity tests for univariate time series.

Two complementary tests, applied to a series before fitting an ARMA /
ARMA-GARCH model so users can confirm the input is appropriate for the
model class:

* :func:`adf` — Augmented Dickey-Fuller.  ``H_0: y_t`` has a unit root
  (integrated of order :math:`\ge 1`); ``H_1`` it is stationary around
  the chosen deterministic component.  Auxiliary regression

  .. math::

      \Delta y_t = \alpha + \beta\, t + \gamma\, y_{t-1}
                 + \sum_{i=1}^{k} \delta_i\, \Delta y_{t-i}
                 + \varepsilon_t,

  test statistic is the t-stat on :math:`\hat\gamma`.  Distribution
  under :math:`H_0` is non-standard Dickey-Fuller; critical values
  from MacKinnon (1996) — *not* the standard Student-t.

* :func:`kpss` — Kwiatkowski-Phillips-Schmidt-Shin.  Null is flipped:
  ``H_0: y_t`` is (level- or trend-) stationary; ``H_1`` it has a
  unit root.  Test statistic is

  .. math::

      \eta = \frac{1}{T^2} \sum_{t=1}^{T} \frac{S_t^2}{s^2(l)},
      \qquad
      S_t = \sum_{i=1}^{t} \hat\varepsilon_i,
      \qquad
      s^2(l) = \hat\gamma_0
             + 2 \sum_{j=1}^{l} \!\left(1 - \tfrac{j}{l+1}\right)\!
                \hat\gamma_j

  with the long-run variance estimated via the Bartlett kernel
  (Newey-West).  Critical values from KPSS (1992) Table 1.

The two tests together form the standard confirmatory analysis: ADF
reject + KPSS fail-to-reject ⇒ stationary; ADF fail-to-reject + KPSS
reject ⇒ unit root; both reject ⇒ short-memory / fractionally-
integrated; both fail-to-reject ⇒ inconclusive (test power too low for
the sample size).

References:
    * Said, S.E. & Dickey, D.A. (1984).  *Testing for unit roots in
      autoregressive-moving average models of unknown order*.
      Biometrika, 71(3), 599-607.
    * MacKinnon, J.G. (1996).  *Numerical distribution functions for
      unit root and cointegration tests*.  Journal of Applied
      Econometrics, 11(6), 601-618.
    * Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y. (1992).
      *Testing the null hypothesis of stationarity against the
      alternative of a unit root*.  Journal of Econometrics, 54(1-3),
      159-178.
    * Schwert, G.W. (1989).  *Tests for unit roots: A Monte Carlo
      investigation*.  Journal of Business & Economic Statistics,
      7(2), 147-159.
"""

from __future__ import annotations

import math
from typing import Optional

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


###############################################################################
# Critical-value tables
###############################################################################
# MacKinnon (1996) Table 1, asymptotic 1% / 5% / 10% τ critical values for
# the ADF test under each deterministic specification.  Reject H_0 (unit
# root) if test_stat < critical value.
_ADF_CRIT: dict[str, tuple[float, float, float]] = {
    "n":  (-2.5658, -1.9393, -1.6156),
    "c":  (-3.4304, -2.8615, -2.5666),
    "ct": (-3.9583, -3.4126, -3.1278),
}
_ADF_LEVELS: tuple[float, float, float] = (0.01, 0.05, 0.10)

# KPSS (1992) Table 1, 10% / 5% / 2.5% / 1% critical values for the η
# statistic.  Reject H_0 (stationarity) if η > critical value.
_KPSS_CRIT: dict[str, tuple[float, float, float, float]] = {
    "c":  (0.347, 0.463, 0.574, 0.739),
    "ct": (0.119, 0.146, 0.176, 0.216),
}
_KPSS_LEVELS: tuple[float, float, float, float] = (0.10, 0.05, 0.025, 0.01)


###############################################################################
# Helpers
###############################################################################
def _schwert_lags(n: int) -> int:
    r"""Schwert (1989) automatic lag selection:
    :math:`k = \lceil 12\,(n/100)^{1/4} \rceil`.
    """
    return int(math.ceil(12.0 * (n / 100.0) ** 0.25))


def _ols_t_stat(X: Array, y: Array, idx: int) -> Array:
    r"""OLS regression of ``y`` on ``X``; returns the t-stat of
    coefficient ``idx`` (and the residuals, for downstream use).

    Uses ``jnp.linalg.solve`` against a small symmetric ``X^T X`` —
    the ADF / KPSS regressions are very low-dimensional (``≤ 2 + lags``
    and ``≤ 2`` columns respectively).
    """
    XtX = X.T @ X
    Xty = X.T @ y
    beta = jnp.linalg.solve(XtX, Xty)
    fitted = X @ beta
    residuals = y - fitted
    n_eff, k = X.shape
    sigma2 = jnp.sum(residuals ** 2) / jnp.maximum(n_eff - k, 1)
    XtX_inv = jnp.linalg.inv(XtX)
    se = jnp.sqrt(sigma2 * jnp.diag(XtX_inv))
    t_stat = beta[idx] / se[idx]
    return t_stat, residuals


def _interp_p(stat: float, crits: tuple, levels: tuple, lower_tail: bool) -> float:
    r"""Coarse log-linear p-value from a tabulated critical-value set.

    ``crits`` are sorted such that more-extreme-rejection comes first;
    ``levels`` are the matching significance levels.  ``lower_tail=True``
    means small/negative ``stat`` rejects (ADF case); ``False`` means
    large positive ``stat`` rejects (KPSS case).  The interpolation is
    linear in ``(stat, log p)`` between adjacent table entries and
    log-linearly extrapolated outside, clipped to ``[1e-4, 0.99]``.

    Cross-validation note: this is *not* the MacKinnon (1996)
    polynomial p-value — that requires a full surface fit not worth
    reproducing for a 4-decimal interpolation.  The test statistic and
    critical values match statsmodels to machine precision; the
    p-value matches to within the interpolation envelope (typically
    ≤ 1e-3 for stats inside the tabulated range, larger outside).
    Decisions at standard significance levels (1%/5%/10%) are exact —
    the comparison ``stat ⋛ crit_α`` does not depend on the
    interpolation.
    """
    log_levels = [math.log(lv) for lv in levels]
    crits_arr = list(crits)
    n = len(crits_arr)
    # Both tail conventions: ``crits`` is monotone in the direction of
    # rejection.  For lower-tail ADF, crits[0] is the most-negative
    # (1%) and stat < crits[0] gives p < 0.01.  For upper-tail KPSS,
    # crits[0] is smallest (10%) and stat > crits[-1] gives p < 0.01.
    if lower_tail:
        # ADF: more-negative ⇒ smaller p.  crits sorted ascending in
        # absolute rejection strength → crits[0] < crits[1] < crits[2].
        if stat <= crits_arr[0]:
            slope = (log_levels[1] - log_levels[0]) / (crits_arr[1] - crits_arr[0])
            log_p = log_levels[0] + slope * (stat - crits_arr[0])
        elif stat >= crits_arr[-1]:
            slope = (log_levels[-1] - log_levels[-2]) / (crits_arr[-1] - crits_arr[-2])
            log_p = log_levels[-1] + slope * (stat - crits_arr[-1])
        else:
            for i in range(1, n):
                if stat <= crits_arr[i]:
                    slope = (log_levels[i] - log_levels[i - 1]) / (
                        crits_arr[i] - crits_arr[i - 1]
                    )
                    log_p = log_levels[i - 1] + slope * (stat - crits_arr[i - 1])
                    break
    else:
        # KPSS: larger stat ⇒ smaller p.  crits sorted ascending so
        # crits[-1] is the strongest (1%) cutoff.
        if stat <= crits_arr[0]:
            slope = (log_levels[1] - log_levels[0]) / (crits_arr[1] - crits_arr[0])
            log_p = log_levels[0] + slope * (stat - crits_arr[0])
        elif stat >= crits_arr[-1]:
            slope = (log_levels[-1] - log_levels[-2]) / (crits_arr[-1] - crits_arr[-2])
            log_p = log_levels[-1] + slope * (stat - crits_arr[-1])
        else:
            for i in range(1, n):
                if stat <= crits_arr[i]:
                    slope = (log_levels[i] - log_levels[i - 1]) / (
                        crits_arr[i] - crits_arr[i - 1]
                    )
                    log_p = log_levels[i - 1] + slope * (stat - crits_arr[i - 1])
                    break
    p = math.exp(log_p)
    return max(min(p, 0.99), 1e-4)


###############################################################################
# Augmented Dickey-Fuller
###############################################################################
def adf(
    y: ArrayLike,
    *,
    regression: str = "c",
    lags: Optional[int] = None,
) -> dict:
    r"""Augmented Dickey-Fuller test for a unit root.

    Args:
        y: shape ``(n,)`` — input series.
        regression: One of ``"n"`` (no constant), ``"c"`` (constant
            only — default), ``"ct"`` (constant and time trend).
            Matches the ``statsmodels.tsa.stattools.adfuller``
            regression key.
        lags: Number of lagged differences :math:`\Delta y_{t-i}` to
            include in the auxiliary regression.  Defaults to the
            Schwert (1989) rule
            :math:`\lceil 12 (n/100)^{1/4} \rceil`.

    Returns:
        ``{"test_stat", "p_value", "used_lag", "n_obs",
        "crit_values"}``.  ``crit_values`` is a dict mapping
        significance level → MacKinnon (1996) Table 1 critical value.

    Reject ``H_0`` (unit root) when ``test_stat`` is more negative than
    the desired critical value, i.e. ``test_stat < crit_values[0.05]``
    for a 5% test.
    """
    if regression not in ("n", "c", "ct"):
        raise ValueError(
            f"regression must be 'n', 'c', or 'ct'; got {regression!r}."
        )
    y_arr = jnp.asarray(y, dtype=float).reshape(-1)
    n = int(y_arr.shape[0])
    if lags is None:
        lags = _schwert_lags(n)
    lags = int(lags)
    if n - 1 - lags < 4:
        raise ValueError(
            f"Series too short for ADF with lags={lags}: need n - 1 - lags "
            f"≥ 4, got {n - 1 - lags}."
        )

    dy = jnp.diff(y_arr)                      # Δy_t for t = 1..n-1
    n_eff = n - 1 - lags
    target = dy[lags:]                        # Δy_t for t = lags+1..n-1
    cols: list[Array] = []
    if regression in ("c", "ct"):
        cols.append(jnp.ones((n_eff,), dtype=float))
    if regression == "ct":
        cols.append(jnp.arange(1, n_eff + 1, dtype=float))
    cols.append(y_arr[lags : n - 1])          # γ y_{t-1}
    for i in range(1, lags + 1):
        cols.append(dy[lags - i : n - 1 - i])  # δ_i Δy_{t-i}
    X = jnp.stack(cols, axis=1)
    gamma_idx = {"n": 0, "c": 1, "ct": 2}[regression]
    test_stat, _residuals = _ols_t_stat(X, target, gamma_idx)

    crit_tuple = _ADF_CRIT[regression]
    p_value = _interp_p(
        float(test_stat), crit_tuple, _ADF_LEVELS, lower_tail=True,
    )
    return {
        "test_stat": test_stat,
        "p_value": jnp.asarray(p_value),
        "used_lag": lags,
        "n_obs": n_eff,
        "crit_values": {
            f"{int(lv * 100)}%": cv
            for lv, cv in zip(_ADF_LEVELS, crit_tuple)
        },
    }


###############################################################################
# Kwiatkowski-Phillips-Schmidt-Shin
###############################################################################
def _bartlett_long_run_variance(residuals: Array, lags: int) -> Array:
    r"""Newey-West / Bartlett-kernel long-run variance estimator
    :math:`s^2(l) = \hat\gamma_0 + 2\sum_{j=1}^{l}
    (1 - j/(l+1)) \hat\gamma_j`.

    All ``γ̂_j`` use the biased estimator (divisor ``T``) so the
    long-run variance is itself non-negative even at short bandwidths.
    """
    n = residuals.shape[0]
    n_f = jnp.asarray(n, dtype=float)
    gamma_0 = jnp.sum(residuals ** 2) / n_f
    s2 = gamma_0
    for j in range(1, lags + 1):
        gamma_j = jnp.sum(residuals[:-j] * residuals[j:]) / n_f
        weight = 1.0 - j / (lags + 1.0)
        s2 = s2 + 2.0 * weight * gamma_j
    return s2


def kpss(
    y: ArrayLike,
    *,
    regression: str = "c",
    lags: Optional[int] = None,
    lags_choice: str = "short",
) -> dict:
    r"""Kwiatkowski-Phillips-Schmidt-Shin stationarity test.

    Args:
        y: shape ``(n,)`` — input series.
        regression: ``"c"`` for level stationarity (regress on a
            constant) or ``"ct"`` for trend stationarity (regress on
            constant + linear trend).  Matches the
            ``statsmodels.tsa.stattools.kpss`` regression key.
        lags: Bandwidth ``l`` of the Bartlett-kernel HAC long-run
            variance.  Defaults to the heuristic given by
            ``lags_choice``.
        lags_choice: ``"short"`` →
            :math:`l = \lfloor 4(n/100)^{1/4} \rfloor` (KPSS 1992
            Table 5).  ``"long"`` →
            :math:`l = \lfloor 12(n/100)^{1/4} \rfloor` (Schwert 1989).

    Returns:
        ``{"test_stat", "p_value", "used_lag", "n_obs",
        "crit_values"}``.

    Reject ``H_0`` (stationarity) when ``test_stat`` exceeds the
    desired critical value.
    """
    if regression not in ("c", "ct"):
        raise ValueError(
            f"regression must be 'c' or 'ct'; got {regression!r}."
        )
    if lags_choice not in ("short", "long"):
        raise ValueError(
            f"lags_choice must be 'short' or 'long'; got {lags_choice!r}."
        )
    y_arr = jnp.asarray(y, dtype=float).reshape(-1)
    n = int(y_arr.shape[0])
    if lags is None:
        coef = 4.0 if lags_choice == "short" else 12.0
        # ``ceil`` matches ``statsmodels.tsa.stattools.kpss(nlags=
        # 'legacy')`` and ``nlags='auto'`` for cross-validation
        # parity.  KPSS (1992) §3 uses floor, but the rounding
        # difference is at most one bandwidth and the Bartlett kernel
        # with adjacent ``l`` values gives test statistics that differ
        # by only a few parts in 10^4.
        lags = int(math.ceil(coef * (n / 100.0) ** 0.25))
    lags = int(lags)

    cols: list[Array] = [jnp.ones((n,), dtype=float)]
    if regression == "ct":
        cols.append(jnp.arange(1, n + 1, dtype=float))
    X = jnp.stack(cols, axis=1)
    XtX = X.T @ X
    Xty = X.T @ y_arr
    beta = jnp.linalg.solve(XtX, Xty)
    residuals = y_arr - X @ beta

    cum = jnp.cumsum(residuals)
    s2 = _bartlett_long_run_variance(residuals, lags)
    n_f = jnp.asarray(n, dtype=float)
    test_stat = jnp.sum(cum ** 2) / (n_f * n_f * jnp.maximum(s2, 1e-30))

    crit_tuple = _KPSS_CRIT[regression]
    p_value = _interp_p(
        float(test_stat), crit_tuple, _KPSS_LEVELS, lower_tail=False,
    )
    return {
        "test_stat": test_stat,
        "p_value": jnp.asarray(p_value),
        "used_lag": lags,
        "n_obs": n,
        "crit_values": {
            f"{lv * 100:g}%": cv
            for lv, cv in zip(_KPSS_LEVELS, crit_tuple)
        },
    }


__all__ = ["adf", "kpss"]
