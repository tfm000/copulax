"""Plot smoke + structural tests for fitted time-series models.

Per plan §"Plotting":

* every plot method runs without exception on a fitted instance,
  with and without the optional ``ax`` / ``axes`` kwarg.
* ``plot_scatter`` always returns a tuple of axes — even the
  single-panel mean-model case returns ``(ax,)`` — so calling code
  can index uniformly.
* axis-correctness sanity checks (number of lines / collections).
* save-to-buffer round-trip succeeds on every CI platform.
"""

from __future__ import annotations

import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from copulax.timeseries import (
    ARMA,
    ArmaGarch,
    EGARCH,
    GARCH,
    GJR_GARCH,
    MA,
)
from copulax.tests._timeseries_helpers import STANDARD, series, shared_fit
from copulax.tests.conftest import SERIES_AR1_P050_N500_S42
from copulax.univariate import gh, normal, student_t


# ---------------------------------------------------------------------------
# Shared fits
#
# Every series is loaded from the frozen corpus and every fit goes
# through the shared registry at the STANDARD tier: plotting only ever
# needs *a* fitted object, never a particular optimum, so these fits are
# shared with every other STANDARD consumer of the same series across the
# whole time-series test family.  Plotting reads from the fitted model
# without mutating it.
#
# The AR(1)-Normal and GARCH(1,1)-Normal fits this file used to declare
# are consumed by a second module too, so they now live in
# ``copulax/tests/conftest.py`` as
# ``ar1_p060_n500_s42_normal_fit_standard`` and
# ``garch11_n500_s2_normal_fit_standard``, alongside their series.  The
# joint ArmaGarch fit below has no second consumer and stays here.
# ---------------------------------------------------------------------------
@pytest.fixture
def armagarch_normal_fit(ar1_p050_n500_s42):
    """Joint ArmaGarch fit shared by the two composite plot tests.

    Single-module: no other file fits ``ArmaGarch(1, 0) x GARCH(1, 1)``
    on this series.

    Returns
    -------
    copulax.timeseries.ArmaGarch
        The fitted composite — the registry's shared instance.
    """
    return shared_fit(
        ArmaGarch(
            mean_order=(1, 0),
            var_model=GARCH,
            var_order=(1, 1),
            residual_dist=normal,
        ),
        SERIES_AR1_P050_N500_S42,
        tier=STANDARD,
    )


# ---------------------------------------------------------------------------
# Mean-model plots
# ---------------------------------------------------------------------------
class TestMeanPlots:
    def test_plot_timeseries_renders(
        self,
        ar1_p060_n500_s42,
        ar1_p060_n500_s42_normal_fit_standard,
    ):
        y = ar1_p060_n500_s42
        fit = ar1_p060_n500_s42_normal_fit_standard
        fig, ax = plt.subplots()
        out = fit.plot_timeseries(y, ax=ax)
        # Returned axes is the same one we passed in.
        assert out is ax
        # At least 2 lines (y + μ_t).
        assert len(ax.lines) >= 2
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        assert buf.tell() > 0
        plt.close(fig)

    def test_plot_timeseries_with_forecast_extension(
        self,
        ar1_p060_n500_s42,
        ar1_p060_n500_s42_normal_fit_standard,
    ):
        y = ar1_p060_n500_s42
        fit = ar1_p060_n500_s42_normal_fit_standard
        fig, ax = plt.subplots()
        out = fit.plot_timeseries(y, h=15, ax=ax)
        assert out is ax
        # Adds a third line (the dashed forecast).
        assert len(ax.lines) >= 3
        plt.close(fig)

    def test_plot_scatter_returns_tuple(
        self,
        ar1_p060_n500_s42,
        ar1_p060_n500_s42_normal_fit_standard,
    ):
        y = ar1_p060_n500_s42
        fit = ar1_p060_n500_s42_normal_fit_standard
        axes = fit.plot_scatter(y)
        assert isinstance(axes, tuple)
        assert len(axes) == 1
        ax = axes[0]
        # Scatter + y=x line.
        assert len(ax.collections) >= 1
        assert len(ax.lines) >= 1
        plt.close(ax.figure)

    def test_arma_plot_methods(self, ar1_p050_n500_s42):
        """ARMA(1, 1) inherits the same plot surface."""
        y = ar1_p050_n500_s42
        fit = shared_fit(
            ARMA(p=1, q=1, residual_dist=normal),
            SERIES_AR1_P050_N500_S42,
            tier=STANDARD,
        )
        ax = fit.plot_timeseries(y)
        plt.close(ax.figure)
        axes = fit.plot_scatter(y)
        assert isinstance(axes, tuple) and len(axes) == 1
        plt.close(axes[0].figure)

    def test_ma_plot_methods(self):
        name = "ar1_p040_n500_s13"
        y = series(name)
        fit = shared_fit(MA(q=1, residual_dist=normal), name, tier=STANDARD)
        ax = fit.plot_timeseries(y)
        plt.close(ax.figure)


# ---------------------------------------------------------------------------
# Variance-model plots
# ---------------------------------------------------------------------------
class TestVariancePlots:
    def test_plot_timeseries_with_var_bands(
        self,
        garch11_n500_s2,
        garch11_n500_s2_normal_fit_standard,
    ):
        eps = garch11_n500_s2
        fit = garch11_n500_s2_normal_fit_standard
        fig, ax = plt.subplots()
        out = fit.plot_timeseries(eps, m=20, ax=ax)
        assert out is ax
        # Eps line + rolling lo + rolling hi (3 lines).
        assert len(ax.lines) >= 3
        # Filled band: at least one PolyCollection.
        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_plot_timeseries_no_rolling(
        self,
        garch11_n500_s2,
        garch11_n500_s2_normal_fit_standard,
    ):
        eps = garch11_n500_s2
        fit = garch11_n500_s2_normal_fit_standard
        fig, ax = plt.subplots()
        fit.plot_timeseries(eps, m=20, show_rolling=False, ax=ax)
        # Without rolling: only ε line.
        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_plot_scatter_returns_two_panels(
        self,
        garch11_n500_s2,
        garch11_n500_s2_normal_fit_standard,
    ):
        eps = garch11_n500_s2
        fit = garch11_n500_s2_normal_fit_standard
        axes = fit.plot_scatter(eps, m=20)
        assert isinstance(axes, tuple)
        assert len(axes) == 2
        # Both panels have a scatter + y=x line.
        for ax in axes:
            assert len(ax.collections) >= 1
            assert len(ax.lines) >= 1
        plt.close(axes[0].figure)

    def test_save_to_buffer_round_trip(
        self,
        garch11_n500_s2,
        garch11_n500_s2_normal_fit_standard,
    ):
        eps = garch11_n500_s2
        fit = garch11_n500_s2_normal_fit_standard
        ax = fit.plot_timeseries(eps)
        buf = io.BytesIO()
        ax.figure.savefig(buf, format="png")
        assert buf.tell() > 0
        plt.close(ax.figure)


# ---------------------------------------------------------------------------
# Joint composite plots
# ---------------------------------------------------------------------------
class TestArmaGarchPlots:
    def test_plot_timeseries_returns_two_panels(
        self,
        ar1_p050_n500_s42,
        armagarch_normal_fit,
    ):
        axes = armagarch_normal_fit.plot_timeseries(
            ar1_p050_n500_s42,
            h=10,
            m=20,
        )
        assert isinstance(axes, tuple)
        assert len(axes) == 2
        plt.close(axes[0].figure)

    def test_plot_scatter_returns_three_panels(
        self,
        ar1_p050_n500_s42,
        armagarch_normal_fit,
    ):
        axes = armagarch_normal_fit.plot_scatter(ar1_p050_n500_s42, m=20)
        assert isinstance(axes, tuple)
        assert len(axes) == 3
        plt.close(axes[0].figure)

    def test_works_with_gjr_variance(self, ar1_p050_n500_s42):
        """Plotting reads the residual distribution off the fitted
        composite — works for any variance variant."""
        y = ar1_p050_n500_s42
        fit = shared_fit(
            ArmaGarch(
                mean_order=(1, 0),
                var_model=GJR_GARCH,
                var_order=(1, 1),
                residual_dist=normal,
            ),
            SERIES_AR1_P050_N500_S42,
            tier=STANDARD,
        )
        axes = fit.plot_timeseries(y, m=20)
        assert len(axes) == 2
        plt.close(axes[0].figure)


# ---------------------------------------------------------------------------
# Non-normal residual Q-Q path (plot_scatter with heavy-tailed residuals)
# ---------------------------------------------------------------------------
class TestNonNormalResidualScatter:
    r"""Smoke coverage for the ``residual_dist.ppf`` Q-Q path in
    ``plot_scatter`` under *non-normal* residual laws.

    The Q-Q panel of both :func:`plot_scatter_variance` (panel 2) and
    :func:`plot_scatter_joint` (panel 3) computes its theoretical
    quantiles via ``fit.residual_dist.ppf(...)`` (see
    ``copulax._src.timeseries._plotting._qq_data``).  Every other
    plotting test in this module fits with ``residual_dist=normal``,
    which routes through the closed-form Gaussian inverse CDF and
    leaves the heavy-tailed ``ppf`` branch — the one users actually
    reach when modelling fat-tailed financial returns with
    ``student_t`` / ``gh`` residuals — untested.  These smoke tests
    fit a variance model and the joint composite with ``student_t``
    and ``gh`` residuals and assert the scatter call returns the
    documented axes tuple with a finite Q-Q panel; they would fail if
    the non-normal ``ppf`` path raised or produced non-finite
    quantiles.

    Smoke level only: no exact pixel / numeric assertions, small ``n``
    for speed, and figures closed after each test to avoid resource
    warnings.  ``gh`` fits are genuinely slow (heavy-tailed EM /
    optimiser cost) so the two ``gh`` cases carry ``@pytest.mark.slow``
    while the ``student_t`` cases stay in the default ``-m "not slow"``
    suite.  No plotting numerical logic is changed and the British
    ``standardised_residuals`` key is untouched (that rename is Phase 3
    API-01).
    """

    @staticmethod
    def _qq_panel_is_finite(ax) -> bool:
        r"""True iff the Q-Q panel drew at least one scatter collection
        whose (theoretical, empirical) offsets — the theoretical column
        is the ``residual_dist.ppf`` output — are all finite."""
        assert len(ax.collections) >= 1
        offsets = np.asarray(ax.collections[0].get_offsets())
        return bool(offsets.size) and bool(np.all(np.isfinite(offsets)))

    def test_variance_scatter_student_t_qq_path(self):
        r"""``plot_scatter_variance`` Q-Q panel exercises
        ``student_t.ppf`` without error and returns 2 finite panels."""
        name = "garch11_n250_s2"
        eps = series(name)
        fit = shared_fit(
            GARCH(p=1, q=1, residual_dist=student_t),
            name,
            tier=STANDARD,
        )
        assert fit.residual_dist.name.startswith("Student-T")
        axes = fit.plot_scatter(eps, m=20)
        assert isinstance(axes, tuple)
        assert len(axes) == 2
        for ax in axes:
            assert len(ax.collections) >= 1
            assert len(ax.lines) >= 1
        # Panel 2 is the residual Q-Q — its theoretical quantiles come
        # from the non-normal student_t ppf.
        assert self._qq_panel_is_finite(axes[1])
        plt.close(axes[0].figure)

    def test_joint_scatter_student_t_qq_path(self):
        r"""``plot_scatter_joint`` Q-Q panel exercises ``student_t.ppf``
        without error and returns 3 finite panels."""
        name = "ar1_p050_n250_s13"
        y = series(name)
        fit = shared_fit(
            ArmaGarch(
                mean_order=(1, 0),
                var_model=GARCH,
                var_order=(1, 1),
                residual_dist=student_t,
            ),
            name,
            tier=STANDARD,
        )
        assert fit.residual_dist.name.startswith("Student-T")
        axes = fit.plot_scatter(y, m=20)
        assert isinstance(axes, tuple)
        assert len(axes) == 3
        # Panel 3 is the residual Q-Q — theoretical quantiles from the
        # non-normal student_t ppf.
        assert self._qq_panel_is_finite(axes[2])
        plt.close(axes[0].figure)

    @pytest.mark.slow
    def test_variance_scatter_gh_qq_path(self):
        r"""``plot_scatter_variance`` Q-Q panel exercises ``gh.ppf``
        (numerically-inverted heavy-tailed CDF) without error."""
        name = "garch11_n250_s2"
        eps = series(name)
        fit = shared_fit(
            GARCH(p=1, q=1, residual_dist=gh),
            name,
            tier=STANDARD,
        )
        assert fit.residual_dist.name.startswith("GH")
        axes = fit.plot_scatter(eps, m=20)
        assert isinstance(axes, tuple)
        assert len(axes) == 2
        for ax in axes:
            assert len(ax.collections) >= 1
            assert len(ax.lines) >= 1
        assert self._qq_panel_is_finite(axes[1])
        plt.close(axes[0].figure)

    @pytest.mark.slow
    def test_joint_scatter_gh_qq_path(self):
        r"""``plot_scatter_joint`` Q-Q panel exercises ``gh.ppf``
        without error and returns 3 finite panels."""
        name = "ar1_p050_n250_s13"
        y = series(name)
        fit = shared_fit(
            ArmaGarch(
                mean_order=(1, 0),
                var_model=GARCH,
                var_order=(1, 1),
                residual_dist=gh,
            ),
            name,
            tier=STANDARD,
        )
        assert fit.residual_dist.name.startswith("GH")
        axes = fit.plot_scatter(y, m=20)
        assert isinstance(axes, tuple)
        assert len(axes) == 3
        assert self._qq_panel_is_finite(axes[2])
        plt.close(axes[0].figure)
