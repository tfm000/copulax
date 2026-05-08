r"""Serial-correlation and ARCH-effect diagnostics.

Four standalone tests on a (univariate) time series, all pure
JAX and JIT- / autograd-compatible:

* :func:`acf` — sample autocorrelation function via the biased
  ACVF estimator (which guarantees a PSD Toeplitz matrix; matches
  the textbook convention used by ``statsmodels.tsa.stattools``).
  Returns a 1-D JAX array.
* :func:`pacf` — partial autocorrelation function via the
  Levinson-Durbin recursion (Brockwell-Davis 1991 §3.4).  v1
  supports ``method='yule_walker'`` only.  Returns a 1-D JAX
  array.
* :func:`ljung_box` — Ljung-Box / Box-Pierce-Ljung Q-test for
  serial correlation up to lag ``h``; ``Q ~ χ²(h)`` under H0.
  Returns a self-describing result dict.
* :func:`arch_lm` — Engle's (1982) Lagrange-multiplier test for
  ARCH effects.  Per plan §"Diagnostics" the regressand is the
  *centred* squared residuals (matches Engle's finite-sample form
  and ``statsmodels.stats.diagnostic.het_arch``).  Returns a
  self-describing result dict.

The two hypothesis-test functions (``ljung_box``, ``arch_lm``)
share a common dict schema with ``adf`` / ``kpss`` in
:mod:`copulax._src.timeseries._unit_root`:

  ``{"statistic", "p_value", "used_lag", "n_obs", "dof"}``

— every leaf is a JAX array.  ``statistic`` and ``p_value`` are
scalar floats; ``used_lag``, ``n_obs``, ``dof`` are scalar
``int32``.  Hypothesis statements (H0 / H1) are documented in
each function's docstring rather than the dict, so the result is
a pure-JAX pytree and round-trips through ``jax.jit``
(``static_argnames=("lags",)``) without wrapping; it also
composes cleanly under ``jax.vmap`` / ``eqx.filter_jit`` because
no leaves are Python scalars.  The matplotlib-based ``plot_acf``
/ ``plot_pacf`` helpers (also in this module) drop out of JAX
traces — they're plain Python and consume the JAX output as
numpy arrays for rendering.

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
from copulax._src.timeseries._ols import ols_fit


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
    *,
    dof: Optional[int] = None,
) -> dict:
    r"""Ljung-Box Q-statistic and chi-square p-value.

    .. math::

        Q = n(n + 2) \sum_{k=1}^{h}
                \frac{\hat\rho(k)^2}{n - k},
        \qquad
        Q \xrightarrow{H_0} \chi^2(\mathrm{dof}),

    Reject :math:`H_0` for large :math:`Q` (small p-value).

    Hypotheses:
        * :math:`H_0`: no serial autocorrelation up to lag ``h``.
        * :math:`H_1`: serial autocorrelation present at one or
          more lags :math:`1, \ldots, h`.

    Args:
        y: shape ``(n,)`` — input series.
        lags: number of autocorrelation lags ``h`` to test.
        dof: degrees of freedom for the asymptotic
            :math:`\chi^2` reference.  Defaults to ``lags`` (the
            primitive form on raw series).  Pass ``lags - p - q``
            when applying the test to ARMA(p, q) residuals so the
            null distribution accounts for fitted parameters
            (Box-Jenkins-Reinsel §8.2.2).

    Returns:
        ``{"statistic", "p_value", "used_lag", "n_obs", "dof"}``.
        ``statistic`` is the Q-statistic and ``p_value`` the
        upper-tail :math:`\chi^2(\mathrm{dof})` probability (both
        scalar floats).  ``used_lag``, ``n_obs``, ``dof`` are
        scalar ``int32`` arrays.  The dict is a pure-JAX pytree
        and round-trips through ``jax.jit`` directly with
        ``static_argnames=("lags",)``.
    """
    y_arr = jnp.asarray(y, dtype=float).reshape(-1)
    n_obs = int(y_arr.shape[0])
    n = jnp.asarray(n_obs, dtype=float)
    lags = int(lags)
    rho_lagged = acf(y_arr, lags)[1:]  # ρ(1..h), shape (h,)
    k_idx = jnp.arange(1, lags + 1, dtype=float)
    Q = n * (n + 2.0) * jnp.sum(rho_lagged ** 2 / (n - k_idx))
    df = lags if dof is None else max(int(dof), 1)
    p_value = chi2.sf(Q, df=df)
    return {
        "statistic": Q,
        "p_value":   p_value,
        "used_lag":  jnp.asarray(lags,  dtype=jnp.int32),
        "n_obs":     jnp.asarray(n_obs, dtype=jnp.int32),
        "dof":       jnp.asarray(df,    dtype=jnp.int32),
    }


###############################################################################
# Engle's ARCH-LM test
###############################################################################
def arch_lm(
    eps: ArrayLike,
    lags: int,
) -> dict:
    r"""Engle's (1982) Lagrange-multiplier test for ARCH effects.

    Regress the **centred** squared residuals
    :math:`\varepsilon^2_t - \overline{\varepsilon^2}` on a
    constant and ``lags`` of their own lagged values; the test
    statistic is

    .. math::

        \mathrm{LM} = n_{\mathrm{eff}} \cdot R^2,
        \qquad
        \mathrm{LM} \xrightarrow{H_0} \chi^2(\mathrm{lags}).

    Per plan §"Diagnostics" the centring matches Engle's original
    derivation and the ``statsmodels.stats.diagnostic.het_arch``
    convention.

    Hypotheses:
        * :math:`H_0`: no ARCH effect — the squared residuals are
          serially uncorrelated.
        * :math:`H_1`: ARCH effect present — conditional
          heteroskedasticity.

    Args:
        eps: shape ``(n,)`` — innovation series (or standardised
            residuals from a fitted variance model).
        lags: number of lagged squared residuals in the auxiliary
            regression.

    Returns:
        ``{"statistic", "p_value", "used_lag", "n_obs", "dof"}``.
        ``statistic`` is the LM-statistic and ``p_value`` the
        upper-tail :math:`\chi^2(\mathrm{lags})` probability (both
        scalar floats).  ``used_lag`` is the ``lags`` argument;
        ``n_obs`` is the effective regression sample size
        :math:`n - \mathrm{lags}`; ``dof`` equals ``lags`` —
        all three are scalar ``int32`` arrays.  The dict is a
        pure-JAX pytree and round-trips through ``jax.jit``
        directly with ``static_argnames=("lags",)``.
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

    # Auxiliary OLS via the shared :func:`ols_fit` helper — its
    # ``r_squared`` field consumes the centred regressand exactly as
    # Engle (1982) prescribes.
    r_squared = ols_fit(X, y_reg).r_squared

    LM = jnp.asarray(n_eff, dtype=float) * r_squared
    p_value = chi2.sf(LM, df=lags)
    return {
        "statistic": LM,
        "p_value":   p_value,
        "used_lag":  jnp.asarray(lags,  dtype=jnp.int32),
        "n_obs":     jnp.asarray(n_eff, dtype=jnp.int32),
        "dof":       jnp.asarray(lags,  dtype=jnp.int32),
    }


###############################################################################
# Plot helpers (matplotlib; not JAX-traced)
###############################################################################
def _plot_corr_stem(
    corr: ArrayLike,
    n_obs: int,
    alpha: float,
    ax,
    *,
    ylabel: str,
    default_title: str,
    title: Optional[str],
):
    r"""Shared stem-plotting kernel for ACF / PACF visuals.

    Takes a precomputed correlation array and the sample size used
    to construct it; draws the stem plot with the Bartlett
    :math:`\pm z(\alpha/2) / \sqrt{n}` confidence band.  Factored
    out so :func:`plot_acf` / :func:`plot_pacf` (which compute the
    correlation from a series) and :func:`plot_acf_from_corr` /
    :func:`plot_pacf_from_corr` (which consume cached values from
    a fitted timeseries model) share one pixel-identical
    rendering path.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from jax.scipy.stats import norm
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    corr_np = np.asarray(corr)
    lags = corr_np.shape[-1] - 1
    z = float(norm.ppf(1.0 - alpha / 2.0))
    band = z / np.sqrt(int(n_obs))
    lag_idx = np.arange(lags + 1)
    ax.stem(lag_idx, corr_np, basefmt=" ")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.fill_between(
        lag_idx, -band, band, alpha=0.2, color="C0",
        label=f"{int((1 - alpha) * 100)}% band",
    )
    ax.set_xlabel("lag")
    ax.set_ylabel(ylabel)
    ax.set_title(default_title if title is None else title)
    ax.set_xlim(-0.5, lags + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right")
    return ax


def plot_acf(
    y: ArrayLike,
    lags: int = 20,
    alpha: float = 0.05,
    ax=None,
    title: Optional[str] = None,
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
        title: Optional axes title.  Defaults to ``"Autocorrelation"``.

    Returns:
        The matplotlib axes used for the plot.
    """
    return _plot_corr_stem(
        corr=np.asarray(acf(y, lags)),
        n_obs=int(np.asarray(y).shape[-1]),
        alpha=alpha, ax=ax,
        ylabel="ACF",
        default_title="Autocorrelation",
        title=title,
    )


def plot_acf_from_corr(
    corr: ArrayLike,
    n_obs: int,
    alpha: float = 0.05,
    ax=None,
    title: Optional[str] = None,
):
    r"""Stem plot of an ACF that has already been computed.

    Used by fitted timeseries models to render
    :meth:`plot_acf(y=None)` from the cached
    ``residual_diagnostics_["acf"]`` array without recomputing the
    correlation series.  ``n_obs`` is the sample size the
    correlation was estimated on (used to size the
    :math:`\pm z(\alpha/2) / \sqrt{n}` Bartlett band).

    Args:
        corr: shape ``(lags + 1,)`` — precomputed ACF values
            including ``ρ(0) = 1``.
        n_obs: sample size used to estimate ``corr``.
        alpha: Significance level for the confidence band.
        ax: Optional matplotlib axes to draw onto.
        title: Optional axes title.  Defaults to ``"Autocorrelation"``.

    Returns:
        The matplotlib axes used for the plot.
    """
    return _plot_corr_stem(
        corr=corr, n_obs=n_obs, alpha=alpha, ax=ax,
        ylabel="ACF",
        default_title="Autocorrelation",
        title=title,
    )


def plot_pacf(
    y: ArrayLike,
    lags: int = 20,
    method: str = "yule_walker",
    alpha: float = 0.05,
    ax=None,
    title: Optional[str] = None,
):
    r"""Stem plot of the PACF up to ``lags`` with a Bartlett-IID
    confidence band.

    Same conventions as :func:`plot_acf`; the band is
    :math:`\pm z(\alpha/2) / \sqrt{n}`.

    Args:
        y: shape ``(n,)`` — input series.
        lags: Maximum lag.
        method: PACF estimator (see :func:`pacf`).
        alpha: Significance level for the confidence band.
        ax: Optional matplotlib axes to draw onto.
        title: Optional axes title.  Defaults to
            ``"Partial autocorrelation"``.

    Returns:
        The matplotlib axes used for the plot.
    """
    return _plot_corr_stem(
        corr=np.asarray(pacf(y, lags, method=method)),
        n_obs=int(np.asarray(y).shape[-1]),
        alpha=alpha, ax=ax,
        ylabel="PACF",
        default_title="Partial autocorrelation",
        title=title,
    )


def plot_pacf_from_corr(
    corr: ArrayLike,
    n_obs: int,
    alpha: float = 0.05,
    ax=None,
    title: Optional[str] = None,
):
    r"""Stem plot of a PACF that has already been computed.

    Counterpart to :func:`plot_acf_from_corr` for the partial
    autocorrelation array; used by fitted timeseries models to
    render :meth:`plot_pacf(y=None)` from
    ``residual_diagnostics_["pacf"]`` without recomputing.

    Args:
        corr: shape ``(lags + 1,)`` — precomputed PACF values
            including ``π(0) = 1``.
        n_obs: sample size used to estimate ``corr``.
        alpha: Significance level for the confidence band.
        ax: Optional matplotlib axes to draw onto.
        title: Optional axes title.  Defaults to
            ``"Partial autocorrelation"``.

    Returns:
        The matplotlib axes used for the plot.
    """
    return _plot_corr_stem(
        corr=corr, n_obs=n_obs, alpha=alpha, ax=ax,
        ylabel="PACF",
        default_title="Partial autocorrelation",
        title=title,
    )


__all__ = [
    "acf", "pacf", "ljung_box", "arch_lm",
    "plot_acf", "plot_pacf",
    "plot_acf_from_corr", "plot_pacf_from_corr",
]
