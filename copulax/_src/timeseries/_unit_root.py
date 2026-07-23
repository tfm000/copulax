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

Both functions are end-to-end ``jax.jit``-compatible with
``static_argnames=("regression", "lags", "lags_choice")`` (``adf``
takes the first two; ``kpss`` takes all three).  The result-dict
schema is uniform with the serial-correlation diagnostics in
:mod:`copulax._src.timeseries._diagnostics`:

  ``{"statistic", "p_value", "used_lag", "n_obs", "crit_values"}``

Every leaf is a JAX array — ``statistic`` and ``p_value`` are scalar
floats; ``used_lag`` and ``n_obs`` are scalar ``int32``;
``crit_values`` is a 1-D float array indexed positionally against
:data:`ADF_CRIT_LEVELS` / :data:`KPSS_CRIT_LEVELS`.

ADF and KPSS take different p-value paths.  ADF p-values come from
the JAX port of MacKinnon's (1994) response-surface polynomial in
:mod:`copulax._src.timeseries._mackinnon` — continuous, smooth, and
matched bit-for-bit to ``statsmodels.tsa.adfvalues.mackinnonp``.
``crit_values`` reports the asymptotic ``[1%, 5%, 10%]`` MacKinnon
(2010) critical values for ``c`` / ``ct`` (and the unchanged 1996
values for ``n``).  KPSS still has only the four published
Kwiatkowski-Phillips-Schmidt-Shin (1992) Table 1 knots ``(0.10,
0.05, 0.025, 0.01)``; pending authoritative extreme-percentile
values, it keeps the older piecewise-linear-in-(η, log p)
extrapolation with a numerical ``[1e-4, 0.99]`` clip.  Replacing
the KPSS path with simulated 0.1% / 99.9% η values is tracked as
follow-up work.

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

from copulax._src.timeseries._mackinnon import (
    mackinnon_asymptotic_crit,
    mackinnonp_jit,
)
from copulax._src.timeseries._ols import ols_fit


###############################################################################
# Critical-value tables (ADF)
###############################################################################
# MacKinnon (2010 for c/ct, 1996 for n) asymptotic τ critical values at the
# standard 1% / 5% / 10% significance levels.  Reject H_0 (unit root) if
# test_stat is more negative than the cutoff at the desired level.  Values
# are sourced from ``copulax._src.timeseries._mackinnon`` (which vendors
# them verbatim from statsmodels) — we just precompute the per-regression
# arrays here so module load time pays the conversion once and ``adf()``
# returns a static reference rather than recomputing per call.
ADF_CRIT_LEVELS: tuple[float, float, float] = (0.01, 0.05, 0.10)
_ADF_CRIT_N: Array = mackinnon_asymptotic_crit("n")
_ADF_CRIT_C: Array = mackinnon_asymptotic_crit("c")
_ADF_CRIT_CT: Array = mackinnon_asymptotic_crit("ct")
_ADF_CRIT: dict[str, Array] = {
    "n":  _ADF_CRIT_N,
    "c":  _ADF_CRIT_C,
    "ct": _ADF_CRIT_CT,
}

# KPSS (1992) Table 1, 10% / 5% / 2.5% / 1% critical values for the η
# statistic.  Reject H_0 (stationarity) if η > critical value at the
# desired level.
KPSS_CRIT_LEVELS: tuple[float, float, float, float] = (0.10, 0.05, 0.025, 0.01)
_KPSS_CRIT_C: Array = jnp.asarray(
    (0.347, 0.463, 0.574, 0.739), dtype=float,
)
_KPSS_CRIT_CT: Array = jnp.asarray(
    (0.119, 0.146, 0.176, 0.216), dtype=float,
)
_KPSS_CRIT: dict[str, Array] = {
    "c":  _KPSS_CRIT_C,
    "ct": _KPSS_CRIT_CT,
}
_KPSS_LOG_LEVELS: Array = jnp.log(
    jnp.asarray(KPSS_CRIT_LEVELS, dtype=float)
)


###############################################################################
# Helpers
###############################################################################
def _schwert_lags(n: int) -> int:
    r"""Schwert (1989) automatic lag selection:
    :math:`k = \lceil 12\,(n/100)^{1/4} \rceil`.

    ``n`` is a Python ``int`` (sourced from ``shape[0]`` at trace time),
    so the result is a Python ``int`` consumed as a static-loop bound
    inside the compiled graph.
    """
    return int(math.ceil(12.0 * (n / 100.0) ** 0.25))


def _interp_p_jit(
    stat: Array,
    crits: Array,
    log_levels: Array,
) -> Array:
    r"""Coarse log-linear p-value from a tabulated critical-value set,
    JIT- and autograd-compatible.

    ``crits`` is monotone increasing.  ``log_levels`` are the matching
    log-significance values, monotone in the direction of rejection
    strength (so the same formula serves both the ADF lower-tail
    convention — ``log_levels`` increasing — and the KPSS upper-tail
    convention — ``log_levels`` decreasing).  Inside the tabulated
    range the result is piecewise-linear-in-``(stat, log p)``; outside
    we extend the outermost segment slope.  ``p`` is clipped to
    ``[1e-4, 0.99]``.

    Numerically equivalent (bit-exact at the table knots and along
    every segment) to the previous Python implementation; the only
    change is that all branching is via ``jnp.where`` rather than
    Python ``if`` / ``for`` over traced scalars, so the function is
    safe to embed inside a ``jax.jit`` trace.

    Cross-validation note: this is *not* the MacKinnon (1996)
    polynomial p-value — that requires a full surface fit not worth
    reproducing for a 4-decimal interpolation.  The test statistic
    and critical values match statsmodels to machine precision; the
    p-value matches to within the interpolation envelope (typically
    ≤ 1e-3 for stats inside the tabulated range, larger outside).
    Decisions at standard significance levels (1%/5%/10%) are exact —
    the comparison ``stat ⋛ crit_α`` does not depend on the
    interpolation.
    """
    interior = jnp.interp(stat, crits, log_levels)
    s_lo = (log_levels[1] - log_levels[0]) / (crits[1] - crits[0])
    s_hi = (log_levels[-1] - log_levels[-2]) / (crits[-1] - crits[-2])
    log_p = jnp.where(
        stat < crits[0],
        log_levels[0] + s_lo * (stat - crits[0]),
        jnp.where(
            stat > crits[-1],
            log_levels[-1] + s_hi * (stat - crits[-1]),
            interior,
        ),
    )
    return jnp.clip(jnp.exp(log_p), 1e-4, 0.99)


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

    Hypotheses:
        * :math:`H_0`: ``y`` has a unit root (non-stationary).
        * :math:`H_1` adapts to ``regression``:

          - ``"n"`` → ``y`` is stationary with zero mean.
          - ``"c"`` → ``y`` is stationary around a constant.
          - ``"ct"`` → ``y`` is stationary around a linear trend.

    Args:
        y: shape ``(n,)`` — input series.
        regression: One of ``"n"`` (no constant), ``"c"`` (constant
            only — default), ``"ct"`` (constant and time trend).
            Matches the ``statsmodels.tsa.stattools.adfuller``
            regression key.  Treat as a static argument under
            ``jax.jit`` (``static_argnames=("regression", "lags")``).
        lags: Number of lagged differences :math:`\Delta y_{t-i}` to
            include in the auxiliary regression.  Defaults to the
            Schwert (1989) rule
            :math:`\lceil 12 (n/100)^{1/4} \rceil`.  Static under JIT.

    Returns:
        ``{"statistic", "p_value", "used_lag", "n_obs", "crit_values"}``
        — every entry is a JAX array.  ``statistic`` is the
        Dickey-Fuller t-stat on :math:`\hat\gamma` (scalar float);
        ``p_value`` is a scalar float from MacKinnon's (1994)
        response-surface polynomial (see
        :func:`copulax._src.timeseries._mackinnon.mackinnonp_jit`),
        clamped at the polynomial's calibration boundaries to ``0.0``
        below ``tau_min`` and ``1.0`` above ``tau_max`` per regression
        (matches ``statsmodels.tsa.adfvalues.mackinnonp`` exactly);
        ``used_lag`` and ``n_obs`` are scalar ``int32``;
        ``crit_values`` is shape ``(3,)`` aligned positionally with
        :data:`ADF_CRIT_LEVELS` ``= (0.01, 0.05, 0.10)`` — the
        MacKinnon (2010) asymptotic critical values for ``c`` / ``ct``
        and the MacKinnon (1996) values for ``n``.  The dict is a
        pure-JAX pytree and round-trips through ``jax.jit``.

    Reject ``H_0`` (unit root) when ``statistic`` is more negative
    than the desired critical value, e.g.
    ``statistic < crit_values[1]`` (the 5% cutoff) for a 5% test.
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
    ols = ols_fit(X, target)
    test_stat = ols.t_stats[gamma_idx]

    crit_values = _ADF_CRIT[regression]
    # Continuous p-value from the JAX port of MacKinnon's (1994)
    # response-surface polynomial — no extrapolation, no asymmetric
    # numerical clip; saturation at ``[0.0, 1.0]`` is hardcoded by
    # MacKinnon's published tau_min / tau_max boundaries (see
    # :mod:`copulax._src.timeseries._mackinnon`).
    p_value = mackinnonp_jit(test_stat, regression)
    return {
        "statistic":   test_stat,
        "p_value":     p_value,
        "used_lag":    jnp.asarray(lags, dtype=jnp.int32),
        "n_obs":       jnp.asarray(n_eff, dtype=jnp.int32),
        "crit_values": crit_values,
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

    Hypotheses:
        * :math:`H_0` adapts to ``regression``:

          - ``"c"`` → ``y`` is level-stationary.
          - ``"ct"`` → ``y`` is trend-stationary.

        * :math:`H_1`: ``y`` has a unit root (non-stationary).

    Args:
        y: shape ``(n,)`` — input series.
        regression: ``"c"`` for level stationarity (regress on a
            constant) or ``"ct"`` for trend stationarity (regress on
            constant + linear trend).  Matches the
            ``statsmodels.tsa.stattools.kpss`` regression key.  Static
            under JIT (``static_argnames=("regression", "lags",
            "lags_choice")``).
        lags: Bandwidth ``l`` of the Bartlett-kernel HAC long-run
            variance.  Defaults to the heuristic given by
            ``lags_choice``.  Static under JIT.
        lags_choice: ``"short"`` →
            :math:`l = \lfloor 4(n/100)^{1/4} \rfloor` (KPSS 1992
            Table 5).  ``"long"`` →
            :math:`l = \lfloor 12(n/100)^{1/4} \rfloor` (Schwert 1989).
            Static under JIT.

    Returns:
        ``{"statistic", "p_value", "used_lag", "n_obs", "crit_values"}``
        — every entry is a JAX array.  ``statistic`` is the η
        statistic (scalar float); ``p_value`` is a scalar float
        clipped to ``[1e-4, 0.99]``; ``used_lag`` and ``n_obs`` are
        scalar ``int32``; ``crit_values`` is shape ``(4,)`` aligned
        positionally with :data:`KPSS_CRIT_LEVELS` (i.e.
        ``crit_values[0]`` is the 10% cutoff, ``[1]`` the 5%, ``[2]``
        the 2.5%, ``[3]`` the 1%).  The dict is a pure-JAX pytree and
        round-trips through ``jax.jit``.

    Reject ``H_0`` (stationarity) when ``statistic`` exceeds the
    desired critical value, e.g. ``statistic > crit_values[1]`` (the
    5% cutoff) for a 5% test.
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
    residuals = ols_fit(X, y_arr).residuals

    cum = jnp.cumsum(residuals)
    s2 = _bartlett_long_run_variance(residuals, lags)
    n_f = jnp.asarray(n, dtype=float)
    test_stat = jnp.sum(cum ** 2) / (n_f * n_f * jnp.maximum(s2, 1e-30))

    crit_values = _KPSS_CRIT[regression]
    p_value = _interp_p_jit(test_stat, crit_values, _KPSS_LOG_LEVELS)
    return {
        "statistic":   test_stat,
        "p_value":     p_value,
        "used_lag":    jnp.asarray(lags, dtype=jnp.int32),
        "n_obs":       jnp.asarray(n, dtype=jnp.int32),
        "crit_values": crit_values,
    }


__all__ = ["adf", "kpss", "ADF_CRIT_LEVELS", "KPSS_CRIT_LEVELS"]
