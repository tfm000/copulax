r"""Matplotlib plotting helpers for fitted time-series models.

Three families of plots, mirroring plan §"Plotting":

* **Mean models** (``AR``, ``MA``, ``ARMA``):
  - :func:`plot_timeseries_mean` — actual returns ``y_t`` overlaid
    with one-step-ahead conditional mean ``μ_t``.
  - :func:`plot_scatter_mean` — scatter of ``y_t`` vs ``μ_t`` with
    a ``y = x`` reference line; visualises forecast bias and
    dispersion.

* **Variance models** (GARCH-family — vanilla, IGARCH, GJR,
  EGARCH, TGARCH, QGARCH):
  - :func:`plot_timeseries_variance` — ``ε_t`` with VaR bands at
    ``residual.ppf(α_low)·σ_t`` and ``residual.ppf(α_high)·σ_t``
    (so the bands correctly reflect the *fat tails* of the chosen
    residual law); optional rolling-``m`` empirical band overlay.
  - :func:`plot_scatter_variance` — two-panel: model σ_t vs
    rolling-``m`` empirical σ scatter + Q-Q plot of standardised
    residuals against the fitted residual distribution.

* **Joint composite** (``ArmaGarch``):
  - :func:`plot_timeseries_joint` — top panel mean overlay, bottom
    panel variance bands.
  - :func:`plot_scatter_joint` — three-panel: mean scatter, σ
    scatter, residual Q-Q.

All helpers accept an optional ``ax``/``axes`` kwarg for figure
composition and return the matplotlib axes for further
customisation.  All three plot-scatter helpers return a tuple of
axes — even the single-panel mean case returns ``(ax,)`` — so
calling code can index uniformly across families
(``axes = fit.plot_scatter(y); axes[0]...``).
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike


def _rolling_std(eps: ArrayLike, m: int) -> np.ndarray:
    r"""Centred ``m``-period rolling sample standard deviation.

    Returns an ``(n,)`` numpy array; the first ``(m-1)//2`` and the
    last ``m//2`` entries are ``np.nan`` (boundary effect).  Used
    by :func:`plot_timeseries_variance` and
    :func:`plot_scatter_variance` for the empirical-σ overlay.
    """
    eps_np = np.asarray(eps).ravel()
    n = len(eps_np)
    out = np.full(n, np.nan)
    half = m // 2
    for t in range(half, n - half):
        window = eps_np[t - half : t + half + 1]
        out[t] = float(np.std(window, ddof=1)) if len(window) >= 2 else np.nan
    return out


def _qq_data(
    z_t: ArrayLike,
    residual_distribution,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Theoretical and empirical quantiles for a Q-Q plot.

    Returns ``(theoretical, empirical)`` numpy arrays — both sorted
    so a ``y = x`` line indicates a perfect fit.  The theoretical
    quantiles come from
    ``residual_distribution.ppf((rank - 0.5) / n)``; this is the
    Tukey position-of-data convention used by
    ``scipy.stats.probplot``.
    """
    z = np.asarray(z_t).ravel()
    n = len(z)
    sorted_z = np.sort(z)
    plotting_positions = (np.arange(1, n + 1) - 0.5) / n
    theoretical = np.asarray(
        residual_distribution.ppf(jnp.asarray(plotting_positions))
    )
    return theoretical, sorted_z


# ---------------------------------------------------------------------------
# Mean-model plots
# ---------------------------------------------------------------------------
def plot_timeseries_mean(
    fit,
    y: ArrayLike,
    h: int = 0,
    ax=None,
):
    r"""Time-series chart for a fitted mean model.

    Plots actual returns ``y_t`` overlaid with the one-step-ahead
    conditional mean ``μ_t = fit.conditional_mean(y)``; if ``h > 0``
    extends with an ``h``-step analytical forecast at the right
    edge.

    Args:
        fit: Fitted mean model (``AR``, ``MA``, or ``ARMA``).
        y: shape ``(n,)`` — series to plot.
        h: Forecast extension length (0 = no extension).
        ax: Optional matplotlib axes.

    Returns:
        The matplotlib axes used.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    y_np = np.asarray(y).ravel()
    mu_np = np.asarray(fit.conditional_mean(y))
    n = len(y_np)
    t = np.arange(n)
    ax.plot(t, y_np, color="C0", lw=0.8, label="y", alpha=0.75)
    ax.plot(t, mu_np, color="C3", lw=1.2, label=r"$\mu_t$")
    if int(h) > 0:
        fc = fit.forecast(h=int(h), method="analytical")
        forecast_mean = np.asarray(fc["mean"])
        t_fc = np.arange(n, n + int(h))
        ax.plot(t_fc, forecast_mean, color="C3", lw=1.2, ls="--",
                label=f"{int(h)}-step forecast")
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title(
        f"{type(fit).__name__}({fit.p},{fit.q}) — {fit.residual_dist.name}"
    )
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.3)
    return ax


def plot_scatter_mean(fit, y: ArrayLike, ax=None) -> tuple:
    r"""Scatter of actual ``y_t`` vs forecast ``μ_t`` with ``y = x``
    reference.

    Returns a single-element tuple ``(ax,)`` so calling code can
    index uniformly across mean / variance / joint plot families.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    y_np = np.asarray(y).ravel()
    mu_np = np.asarray(fit.conditional_mean(y))
    ax.scatter(mu_np, y_np, s=8, alpha=0.5, color="C0")
    lo = float(min(np.min(y_np), np.min(mu_np)))
    hi = float(max(np.max(y_np), np.max(mu_np)))
    ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label=r"$y=x$")
    ax.set_xlabel(r"$\mu_t$ (forecast)")
    ax.set_ylabel(r"$y_t$ (actual)")
    ax.set_title(
        f"{type(fit).__name__}({fit.p},{fit.q}) — actual vs forecast"
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    return (ax,)


# ---------------------------------------------------------------------------
# Variance-model plots
# ---------------------------------------------------------------------------
def plot_timeseries_variance(
    fit,
    eps: ArrayLike,
    m: int = 5,
    alpha: tuple[float, float] = (0.05, 0.95),
    show_rolling: bool = True,
    ax=None,
):
    r"""Time-series chart for a fitted variance model.

    Plots ``ε_t`` with VaR bands at the lower and upper quantiles of
    the fitted residual distribution scaled by ``σ_t``:

    .. math::

        \mathrm{upper}(t) &=
            f_z^{-1}(\alpha_{\mathrm{high}}) \cdot \sigma_t,\\
        \mathrm{lower}(t) &=
            f_z^{-1}(\alpha_{\mathrm{low}}) \cdot \sigma_t,

    where :math:`f_z^{-1}` is the standardised residual law's
    inverse CDF (:meth:`Univariate.ppf`).  Bands therefore widen
    correctly under heavy-tailed residual choices (e.g. Student-T
    at low :math:`\nu`) — exactly as expected.

    With ``show_rolling=True`` overlays an empirical band built from
    a centred ``m``-period rolling sample-std of ``ε_t`` multiplied
    by the same ppf quantiles — a robust visual fit-quality check.

    Args:
        fit: Fitted variance model.
        eps: shape ``(n,)`` — mean-corrected innovation series.
        m: Rolling window for the empirical-σ overlay.
        alpha: ``(low, high)`` tail quantiles (default 5/95).
        show_rolling: Whether to overlay the empirical band.
        ax: Optional matplotlib axes.

    Returns:
        The matplotlib axes used.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    eps_np = np.asarray(eps).ravel()
    var_np = np.asarray(fit.conditional_variance(eps))
    sigma_np = np.sqrt(np.maximum(var_np, 1e-12))
    rd = fit.residual_distribution
    q_lo = float(rd.ppf(jnp.asarray(alpha[0])))
    q_hi = float(rd.ppf(jnp.asarray(1.0 - alpha[1]) if alpha[1] < 1.0
                        else jnp.asarray(alpha[1])))
    band_lo = q_lo * sigma_np
    band_hi = q_hi * sigma_np
    n = len(eps_np)
    t = np.arange(n)
    ax.plot(t, eps_np, color="C0", lw=0.6, alpha=0.6, label=r"$\varepsilon_t$")
    ax.fill_between(
        t, band_lo, band_hi, alpha=0.2, color="C3",
        label=(
            f"{int(alpha[0]*100)}/{int(alpha[1]*100)}% VaR band "
            f"({rd.name})"
        ),
    )
    if show_rolling:
        roll_std = _rolling_std(eps_np, m)
        ax.plot(t, q_lo * roll_std, color="C2", lw=1.0, ls="--",
                label=f"{m}-period rolling band")
        ax.plot(t, q_hi * roll_std, color="C2", lw=1.0, ls="--")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\varepsilon_t$")
    ax.set_title(
        f"{type(fit).__name__}({fit.p},{fit.q}) — {fit.residual_dist.name}"
    )
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.3)
    return ax


def plot_scatter_variance(
    fit,
    eps: ArrayLike,
    m: int = 5,
    axes=None,
) -> tuple:
    r"""Two-panel diagnostic: σ scatter + Q-Q plot.

    * Panel 1: model-forecast :math:`\sigma_t` vs empirical
      ``m``-period rolling sample-:math:`\sigma`, with ``y = x``
      reference.
    * Panel 2: Q-Q plot of standardised residuals
      :math:`z_t = \varepsilon_t / \sigma_t` against the fitted
      residual distribution's theoretical quantiles.

    Returns a 2-tuple ``(ax_sigma, ax_qq)``.
    """
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_sigma, ax_qq = axes

    eps_np = np.asarray(eps).ravel()
    var_np = np.asarray(fit.conditional_variance(eps))
    sigma_np = np.sqrt(np.maximum(var_np, 1e-12))
    roll_std = _rolling_std(eps_np, m)

    # Panel 1: σ scatter.
    valid = ~np.isnan(roll_std)
    ax_sigma.scatter(sigma_np[valid], roll_std[valid], s=8, alpha=0.5, color="C0")
    lo = float(min(np.nanmin(sigma_np), np.nanmin(roll_std[valid])))
    hi = float(max(np.nanmax(sigma_np), np.nanmax(roll_std[valid])))
    ax_sigma.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label=r"$y=x$")
    ax_sigma.set_xlabel(r"model $\sigma_t$")
    ax_sigma.set_ylabel(f"rolling-{m} empirical $\\sigma$")
    ax_sigma.set_title(r"forecast vs empirical $\sigma$")
    ax_sigma.legend(loc="best")
    ax_sigma.grid(alpha=0.3)

    # Panel 2: Q-Q plot.
    _, z_t = fit.residuals(eps)
    theoretical, empirical = _qq_data(z_t, fit.residual_distribution)
    ax_qq.scatter(theoretical, empirical, s=8, alpha=0.5, color="C0")
    qmin = float(min(theoretical[0], empirical[0]))
    qmax = float(max(theoretical[-1], empirical[-1]))
    ax_qq.plot([qmin, qmax], [qmin, qmax], color="black", lw=1, ls="--", label=r"$y=x$")
    ax_qq.set_xlabel(f"theoretical ({fit.residual_dist.name})")
    ax_qq.set_ylabel(r"empirical $z_t$")
    ax_qq.set_title("standardised-residual Q-Q")
    ax_qq.legend(loc="best")
    ax_qq.grid(alpha=0.3)

    return (ax_sigma, ax_qq)


# ---------------------------------------------------------------------------
# Joint-composite plots
# ---------------------------------------------------------------------------
def plot_timeseries_joint(
    fit,
    y: ArrayLike,
    h: int = 0,
    m: int = 5,
    show_rolling: bool = True,
    alpha: tuple[float, float] = (0.05, 0.95),
    axes=None,
) -> tuple:
    r"""Two-panel time-series chart for an :class:`ArmaGarch` fit.

    * Top panel: actual ``y_t`` with conditional-mean overlay
      ``μ_t`` (and optional ``h``-step extension).
    * Bottom panel: innovation series ``ε_t = y_t − μ_t`` with VaR
      bands as in :func:`plot_timeseries_variance`.

    Returns ``(ax_mean, ax_vol)``.
    """
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_mean, ax_vol = axes

    # Top: y + μ_t.
    y_np = np.asarray(y).ravel()
    mu_np = np.asarray(fit.conditional_mean(y))
    n = len(y_np)
    t = np.arange(n)
    ax_mean.plot(t, y_np, color="C0", lw=0.8, alpha=0.75, label="y")
    ax_mean.plot(t, mu_np, color="C3", lw=1.2, label=r"$\mu_t$")
    if int(h) > 0:
        fc = fit.forecast(h=int(h), method="analytical")
        ax_mean.plot(np.arange(n, n + int(h)),
                     np.asarray(fc["mean"]),
                     color="C3", lw=1.2, ls="--",
                     label=f"{int(h)}-step forecast")
    ax_mean.set_ylabel("y")
    ax_mean.set_title(
        f"{type(fit).__name__}({fit.p},{fit.q})×"
        f"{fit.var_model.__name__}({fit.p_var},{fit.q_var}) — "
        f"{fit.residual_dist.name}"
    )
    ax_mean.legend(loc="best", frameon=True)
    ax_mean.grid(alpha=0.3)

    # Bottom: ε with VaR bands.
    resid = fit.residuals(y)
    eps_np = np.asarray(resid["mean_residuals"])
    var_np = np.asarray(fit.conditional_variance(y))
    sigma_np = np.sqrt(np.maximum(var_np, 1e-12))
    rd = fit.residual_distribution
    q_lo = float(rd.ppf(jnp.asarray(alpha[0])))
    q_hi = float(rd.ppf(jnp.asarray(alpha[1])))
    band_lo = q_lo * sigma_np
    band_hi = q_hi * sigma_np
    ax_vol.plot(t, eps_np, color="C0", lw=0.6, alpha=0.6,
                label=r"$\varepsilon_t$")
    ax_vol.fill_between(
        t, band_lo, band_hi, alpha=0.2, color="C3",
        label=f"{int(alpha[0]*100)}/{int(alpha[1]*100)}% VaR ({rd.name})",
    )
    if show_rolling:
        roll_std = _rolling_std(eps_np, m)
        ax_vol.plot(t, q_lo * roll_std, color="C2", lw=1.0, ls="--",
                    label=f"{m}-period rolling band")
        ax_vol.plot(t, q_hi * roll_std, color="C2", lw=1.0, ls="--")
    ax_vol.set_xlabel("t")
    ax_vol.set_ylabel(r"$\varepsilon_t$")
    ax_vol.legend(loc="best", frameon=True)
    ax_vol.grid(alpha=0.3)

    return (ax_mean, ax_vol)


def plot_scatter_joint(
    fit,
    y: ArrayLike,
    m: int = 5,
    axes=None,
) -> tuple:
    r"""Three-panel diagnostic for an :class:`ArmaGarch` fit.

    * Panel 1: actual ``y`` vs forecast ``μ_t`` (mean fit).
    * Panel 2: model ``σ_t`` vs rolling-``m`` empirical ``σ``
      (variance fit).
    * Panel 3: Q-Q plot of standardised residuals against the
      fitted residual distribution.

    Returns ``(ax_mean, ax_vol, ax_qq)``.
    """
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_mean, ax_vol, ax_qq = axes

    y_np = np.asarray(y).ravel()
    mu_np = np.asarray(fit.conditional_mean(y))
    var_np = np.asarray(fit.conditional_variance(y))
    sigma_np = np.sqrt(np.maximum(var_np, 1e-12))
    resid = fit.residuals(y)

    # Mean panel.
    ax_mean.scatter(mu_np, y_np, s=8, alpha=0.5, color="C0")
    lo = float(min(np.min(y_np), np.min(mu_np)))
    hi = float(max(np.max(y_np), np.max(mu_np)))
    ax_mean.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label=r"$y=x$")
    ax_mean.set_xlabel(r"$\mu_t$ (forecast)")
    ax_mean.set_ylabel(r"$y_t$")
    ax_mean.set_title("actual vs forecast (mean)")
    ax_mean.legend(loc="best")
    ax_mean.grid(alpha=0.3)

    # σ panel.
    eps_np = np.asarray(resid["mean_residuals"])
    roll_std = _rolling_std(eps_np, m)
    valid = ~np.isnan(roll_std)
    ax_vol.scatter(sigma_np[valid], roll_std[valid], s=8, alpha=0.5, color="C0")
    lo = float(min(np.nanmin(sigma_np), np.nanmin(roll_std[valid])))
    hi = float(max(np.nanmax(sigma_np), np.nanmax(roll_std[valid])))
    ax_vol.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label=r"$y=x$")
    ax_vol.set_xlabel(r"model $\sigma_t$")
    ax_vol.set_ylabel(f"rolling-{m} empirical $\\sigma$")
    ax_vol.set_title(r"forecast vs empirical $\sigma$")
    ax_vol.legend(loc="best")
    ax_vol.grid(alpha=0.3)

    # Q-Q panel.
    z_t = resid["standardised_residuals"]
    theoretical, empirical = _qq_data(z_t, fit.residual_distribution)
    ax_qq.scatter(theoretical, empirical, s=8, alpha=0.5, color="C0")
    qmin = float(min(theoretical[0], empirical[0]))
    qmax = float(max(theoretical[-1], empirical[-1]))
    ax_qq.plot([qmin, qmax], [qmin, qmax], color="black", lw=1, ls="--", label=r"$y=x$")
    ax_qq.set_xlabel(f"theoretical ({fit.residual_dist.name})")
    ax_qq.set_ylabel(r"empirical $z_t$")
    ax_qq.set_title("standardised-residual Q-Q")
    ax_qq.legend(loc="best")
    ax_qq.grid(alpha=0.3)

    return (ax_mean, ax_vol, ax_qq)


__all__ = [
    "plot_timeseries_mean",
    "plot_scatter_mean",
    "plot_timeseries_variance",
    "plot_scatter_variance",
    "plot_timeseries_joint",
    "plot_scatter_joint",
]
