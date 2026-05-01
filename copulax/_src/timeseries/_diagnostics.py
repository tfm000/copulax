r"""Serial-correlation and ARCH-effect diagnostics.

Four standalone tests on a (univariate) time series, all pure JAX
and JIT- / autograd-compatible:

* :func:`acf` — sample autocorrelation function via the biased
  ACVF estimator (which guarantees a PSD Toeplitz matrix; matches
  the textbook convention used by ``statsmodels.tsa.stattools``).
* :func:`pacf` — partial autocorrelation function via the
  Levinson-Durbin recursion (Brockwell-Davis 1991 §3.4).  v1
  supports ``method='yule_walker'`` only.
* :func:`ljung_box` — Ljung-Box / Box-Pierce-Ljung Q-test for
  serial correlation up to lag ``h``; ``Q ~ χ²(h)`` under H0.
* :func:`arch_lm` — Engle's (1982) Lagrange-multiplier test for
  ARCH effects.  Per plan §"Diagnostics" the regressand is the
  *centred* squared residuals (matches Engle's finite-sample form
  and ``statsmodels.stats.diagnostic.het_arch``).

All four return raw JAX scalars / arrays.  The matplotlib-based
``plot_acf`` / ``plot_pacf`` helpers (also in this module) drop
out of JAX traces — they're plain Python and consume the JAX
output as numpy arrays for rendering.

Cross-validation against ``statsmodels.tsa.stattools`` to
``rtol = 1e-6, atol = 1e-8`` (these are exact linear algebra
operations, no optimisation involved); see
``tests/test_timeseries_diagnostics.py``.

References:

* Brockwell, P.J. & Davis, R.A. (1991). *Time Series: Theory and
  Methods* (2nd ed.).  Springer.  ACVF + Levinson-Durbin recursion.
* Ljung, G.M. & Box, G.E.P. (1978).  *On a measure of lack of fit
  in time series models*.  Biometrika, 65(2), 297-303.
* Engle, R.F. (1982).  *Autoregressive conditional
  heteroscedasticity with estimates of the variance of United
  Kingdom inflation*.  Econometrica, 50(4), 987-1007.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.stats import chi2
from jax.typing import ArrayLike

from copulax._src.timeseries._init import acvf


_VALID_PACF_METHODS = frozenset({"yule_walker"})


###############################################################################
# Autocorrelation
###############################################################################
def acf(y: ArrayLike, lags: int) -> Array:
    r"""Sample autocorrelation function at lags :math:`0, 1, \ldots, \mathrm{lags}`.

    .. math::

        \rho(k) = \hat\gamma(k) / \hat\gamma(0),

    where :math:`\hat\gamma(k)` is the **biased** sample
    autocovariance from :func:`acvf`.  The biased estimator
    guarantees a PSD ACVF Toeplitz matrix (Brockwell & Davis 1991,
    Proposition 1.6.1) which matters when downstream consumers
    feed the ACF into Yule-Walker / Levinson-Durbin solves.

    Args:
        y: shape ``(n,)`` — input series.
        lags: maximum lag (static Python ``int``).

    Returns:
        shape ``(lags + 1,)`` array; entry ``0`` is identically
        ``1.0`` by definition.
    """
    gamma = acvf(y, lags)
    gamma0 = jnp.maximum(gamma[0], 1e-30)
    return gamma / gamma0


###############################################################################
# Partial autocorrelation
###############################################################################
def pacf(
    y: ArrayLike,
    lags: int,
    method: str = "yule_walker",
) -> Array:
    r"""Sample partial autocorrelation function at lags
    :math:`0, 1, \ldots, \mathrm{lags}`.

    Computed via the Levinson-Durbin recursion (Brockwell & Davis
    1991, eqn 3.4.11):

    .. math::

        \phi_{n+1, n+1} &= \bigl(\rho(n+1) -
                              \textstyle\sum_{j=1}^n \phi_{n,j}\,
                              \rho(n+1-j)\bigr) / v_n,\\
        \phi_{n+1, j}   &= \phi_{n, j} - \phi_{n+1, n+1}\,
                              \phi_{n, n+1-j},
                           \quad j = 1, \ldots, n,\\
        v_{n+1}         &= v_n \,(1 - \phi_{n+1, n+1}^2),

    with :math:`\phi_{0, *} = 0` and :math:`v_0 = 1`.  The PACF at
    lag :math:`k` is :math:`\phi_{k, k}`.

    Cross-validation note: matches
    ``statsmodels.tsa.stattools.pacf(y, method='ywm')`` (or
    equivalently ``'ldbiased'``) to machine precision — the
    biased-ACVF Yule-Walker variant.  ``statsmodels``' default
    ``method='yw'`` uses the *unbiased* ACVF
    (denominator :math:`n - k` rather than :math:`n`); we use
    biased to keep the Toeplitz matrix PSD (Brockwell-Davis
    Proposition 1.6.1) and to share the ACVF implementation with
    :mod:`copulax._src.timeseries._init`.

    Args:
        y: shape ``(n,)`` — input series.
        lags: maximum lag.
        method: ``"yule_walker"`` (default; the only method
            supported in v1 — OLS-style regression PACFs would need
            padded design matrices to JIT cleanly and are deferred).

    Returns:
        shape ``(lags + 1,)`` array; entry ``0`` is ``1.0``.

    Raises:
        ValueError: When ``method`` is not on the supported list.
    """
    if method not in _VALID_PACF_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_VALID_PACF_METHODS)}; "
            f"got {method!r}.  Plan §\"Diagnostics\" defers OLS-style "
            "PACFs to a future commit."
        )
    lags = int(lags)
    rho = acf(y, lags)  # (lags + 1,)
    pacf_arr = jnp.zeros((lags + 1,), dtype=float)
    pacf_arr = pacf_arr.at[0].set(1.0)
    if lags == 0:
        return pacf_arr

    # phi at the previous order, padded to length ``lags`` with
    # trailing zeros so the JAX shape is fixed across iterations.
    phi_prev = jnp.zeros((lags,), dtype=float)
    v_prev = jnp.asarray(1.0, dtype=float)

    for k in range(1, lags + 1):
        if k == 1:
            phi_kk = rho[1]
        else:
            # Inner sum: Σ_{j=1..k-1} φ_{k-1, j} ρ(k - j)
            #   = dot(phi_prev[:k-1], rho[1:k][::-1])
            inner = jnp.dot(phi_prev[: k - 1], rho[1:k][::-1])
            phi_kk = (rho[k] - inner) / jnp.maximum(v_prev, 1e-30)
        pacf_arr = pacf_arr.at[k].set(phi_kk)
        # Update φ-vector for the next iteration.
        if k > 1:
            new_phi_head = phi_prev[: k - 1] - phi_kk * phi_prev[: k - 1][::-1]
            phi_prev = phi_prev.at[: k - 1].set(new_phi_head)
        phi_prev = phi_prev.at[k - 1].set(phi_kk)
        v_prev = v_prev * (1.0 - phi_kk ** 2)

    return pacf_arr


###############################################################################
# Ljung-Box
###############################################################################
def ljung_box(
    y: ArrayLike,
    lags: int,
) -> tuple[Array, Array]:
    r"""Ljung-Box Q-statistic and chi-square p-value.

    .. math::

        Q = n(n + 2) \sum_{k=1}^{h}
                \frac{\hat\rho(k)^2}{n - k},
        \qquad
        Q \xrightarrow{H_0} \chi^2(h),

    where :math:`H_0` is "no serial autocorrelation up to lag
    :math:`h`".  Reject :math:`H_0` for large :math:`Q` (small
    p-value).

    Args:
        y: shape ``(n,)`` — input series.
        lags: number of autocorrelation lags ``h`` to test.

    Returns:
        Tuple ``(Q, p_value)`` of JAX scalars.
    """
    y_arr = jnp.asarray(y, dtype=float).reshape(-1)
    n = jnp.asarray(int(y_arr.shape[0]), dtype=float)
    lags = int(lags)
    rho_lagged = acf(y_arr, lags)[1:]  # ρ(1..h), shape (h,)
    k_idx = jnp.arange(1, lags + 1, dtype=float)
    Q = n * (n + 2.0) * jnp.sum(rho_lagged ** 2 / (n - k_idx))
    p_value = chi2.sf(Q, df=lags)
    return Q, p_value


###############################################################################
# Engle's ARCH-LM test
###############################################################################
def arch_lm(
    eps: ArrayLike,
    lags: int,
) -> tuple[Array, Array]:
    r"""Engle's (1982) Lagrange-multiplier test for ARCH effects.

    Regress the **centred** squared residuals
    :math:`\varepsilon^2_t - \overline{\varepsilon^2}` on a
    constant and ``lags`` of their own lagged values; the test
    statistic is

    .. math::

        \mathrm{LM} = n_{\mathrm{eff}} \cdot R^2,
        \qquad
        \mathrm{LM} \xrightarrow{H_0} \chi^2(\mathrm{lags}),

    with :math:`H_0` = "no ARCH effect" / "the regressand is
    serially uncorrelated".  Per plan §"Diagnostics" the centring
    matches Engle's original derivation and the
    ``statsmodels.stats.diagnostic.het_arch`` convention.

    Args:
        eps: shape ``(n,)`` — innovation series (or standardised
            residuals from a fitted variance model).
        lags: number of lagged squared residuals in the auxiliary
            regression.

    Returns:
        Tuple ``(LM, p_value)`` of JAX scalars.
    """
    eps_arr = jnp.asarray(eps, dtype=float).reshape(-1)
    n = int(eps_arr.shape[0])
    lags = int(lags)
    eps_sq = eps_arr * eps_arr
    eps_sq_centred = eps_sq - jnp.mean(eps_sq)

    # Auxiliary regression on (1, eps²_{t-1}, ..., eps²_{t-lags}).
    n_eff = n - lags
    # Design matrix: shape (n_eff, lags + 1).
    intercept = jnp.ones((n_eff, 1), dtype=float)
    # Build the lagged columns as eps²[lags - k : n - k] for k = 1..lags.
    lag_cols = jnp.stack(
        [eps_sq[lags - k : n - k] for k in range(1, lags + 1)],
        axis=1,
    )
    X = jnp.concatenate([intercept, lag_cols], axis=1)
    y_reg = eps_sq_centred[lags:]

    # OLS: β = (XᵀX)⁻¹ Xᵀy via solve for numerical stability.
    XtX = X.T @ X
    Xty = X.T @ y_reg
    beta = jnp.linalg.solve(XtX, Xty)
    fitted = X @ beta
    residuals = y_reg - fitted
    ss_res = jnp.sum(residuals ** 2)
    ss_tot = jnp.sum((y_reg - jnp.mean(y_reg)) ** 2)
    r_squared = 1.0 - ss_res / jnp.maximum(ss_tot, 1e-30)

    LM = jnp.asarray(n_eff, dtype=float) * r_squared
    p_value = chi2.sf(LM, df=lags)
    return LM, p_value


###############################################################################
# Plot helpers (matplotlib; not JAX-traced)
###############################################################################
def plot_acf(
    y: ArrayLike,
    lags: int = 20,
    alpha: float = 0.05,
    ax=None,
):
    r"""Stem plot of the ACF up to ``lags`` with a Bartlett-IID
    confidence band.

    The band is :math:`\pm z(\alpha/2) / \sqrt{n}` — the
    asymptotic standard error of :math:`\rho(k)` under the IID
    null hypothesis (Brockwell & Davis 1991, Theorem 7.2.2).
    Lags whose stem extends beyond the band are evidence against
    IID at significance ``alpha``.

    Mirrors the look of ``statsmodels.graphics.tsaplots.plot_acf``.

    Args:
        y: shape ``(n,)`` — input series.
        lags: Maximum lag.
        alpha: Significance level for the confidence band.
        ax: Optional matplotlib axes to draw onto.

    Returns:
        The matplotlib axes used for the plot.
    """
    import matplotlib.pyplot as plt
    from jax.scipy.stats import norm
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    rho = np.asarray(acf(y, lags))  # (lags + 1,)
    n = int(np.asarray(y).shape[-1])
    z = float(norm.ppf(1.0 - alpha / 2.0))
    band = z / np.sqrt(n)
    lag_idx = np.arange(lags + 1)
    ax.stem(lag_idx, rho, basefmt=" ")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.fill_between(
        lag_idx, -band, band, alpha=0.2, color="C0",
        label=f"{int((1 - alpha) * 100)}% band",
    )
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    ax.set_title("Autocorrelation")
    ax.set_xlim(-0.5, lags + 0.5)
    ax.legend(loc="upper right")
    return ax


def plot_pacf(
    y: ArrayLike,
    lags: int = 20,
    method: str = "yule_walker",
    alpha: float = 0.05,
    ax=None,
):
    r"""Stem plot of the PACF up to ``lags`` with a Bartlett-IID
    confidence band.

    Same conventions as :func:`plot_acf`; the band is
    :math:`\pm z(\alpha/2) / \sqrt{n}`.
    """
    import matplotlib.pyplot as plt
    from jax.scipy.stats import norm
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    pi = np.asarray(pacf(y, lags, method=method))  # (lags + 1,)
    n = int(np.asarray(y).shape[-1])
    z = float(norm.ppf(1.0 - alpha / 2.0))
    band = z / np.sqrt(n)
    lag_idx = np.arange(lags + 1)
    ax.stem(lag_idx, pi, basefmt=" ")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.fill_between(
        lag_idx, -band, band, alpha=0.2, color="C0",
        label=f"{int((1 - alpha) * 100)}% band",
    )
    ax.set_xlabel("lag")
    ax.set_ylabel("PACF")
    ax.set_title("Partial autocorrelation")
    ax.set_xlim(-0.5, lags + 0.5)
    ax.legend(loc="upper right")
    return ax


__all__ = [
    "acf", "pacf", "ljung_box", "arch_lm",
    "plot_acf", "plot_pacf",
]
