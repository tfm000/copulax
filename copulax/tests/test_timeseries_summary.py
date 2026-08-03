"""Standard-errors / confidence-intervals / summary tests.

Covers the inferential surface added to standalone mean-equation
(``AR`` / ``MA`` / ``ARMA``) and conditional-variance (``GARCH``,
``IGARCH``, ``GJR_GARCH``, ``EGARCH``, ``TGARCH``, ``QGARCH``,
``GARCH_M``) models, plus the refactored sectioned-format summary
on ``ArmaGarch``.

Coverage:

* Shape / positivity / finiteness of ``standard_errors_`` for every
  model class with both Normal and Student-T residuals where
  applicable.
* Confidence-interval correctness (``lo < est < hi`` per parameter).
* Cached-vs-recompute round-trip on every diagnostic accessor
  (``ljung_box`` / ``arch_lm`` / ``adf_residuals`` /
  ``kpss_residuals``); error path when non-default kwargs are
  supplied without ``y``/``eps``.
* ``summary()`` rendering: header label, section labels, expected
  param row labels, diagnostic block, significance-code legend,
  ``✓`` / ``✗`` glyph polarity.
* Third-party cross-validation: AR(1) vs ``statsmodels.tsa.arima.ARIMA``;
  GARCH(1,1) vs ``arch.univariate``.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copulax.tests._timeseries_helpers import (
    PRECISION,
    STANDARD,
    series,
    shared_fit,
)
from copulax.tests.conftest import (
    SERIES_GARCH11_N1500_S42,
    SERIES_GARCH11_N2000_S2,
    require_oracle,
)
from copulax.timeseries import (
    AR,
    ARMA,
    ArmaGarch,
    EGARCH,
    GARCH,
    GARCH_M,
    GJR_GARCH,
    IGARCH,
    MA,
    QGARCH,
    TGARCH,
)
from copulax.univariate import normal, student_t


# ---------------------------------------------------------------------------
# Frozen series used by this module
#
# Seed-variant collapses (the consuming tests are SE-shape, CI-bracket,
# caching and rendering checks — none needs an independent draw):
#   garch11_n2000_s{2,9,14,15,16,17,40,41,42,43,44,61} -> garch11_n2000_s2
#   garch11_n1500_s7                                   -> garch11_n1500_s42
#   ar1_p050_n2000_s8                                  -> ar1_p050_n2000_s0
#   arma11_p050_qm030_n2000_s{3,4,5,60}                -> ..._n2000_s3
#   arma11_p050_qm030_n1500_s{6,11}                    -> ..._n1500_s6
#   ar1garch11_p050_n1500_s{13,18}                     -> ..._n1500_s13
# The two arma11 n=2000 collapses additionally retire the pair of
# near-cancelling ARMA(1,1) realizations (seeds 5 and 60) flagged in the
# 01-15 audit, replacing them with the better-conditioned seed-3 draw.
#
# Only the series this module ALONE consumes are named here.  The two it
# shares with another module — ``garch11_n2000_s2`` (with variance) and
# ``garch11_n1500_s42`` (with diagnostics) — are fixtures in
# ``copulax/tests/conftest.py``, as are the STANDARD GARCH(1,1)-Student-T
# fit on ``garch11_n2000_s2`` and the ``arch`` oracle.
# ---------------------------------------------------------------------------
_NAME_AR1_N2000 = "ar1_p050_n2000_s0"
_NAME_AR1_N3000 = "ar1_p060_n3000_s20"
_NAME_AR3_N2000 = "ar3_n2000_s1"
_NAME_MA1_N2000 = "ma1_q040_n2000_s2"
_NAME_MA1_N1500 = "ma1_q040_n1500_s10"
_NAME_ARMA11_N2000 = "arma11_p050_qm030_n2000_s3"
_NAME_ARMA11_N1500 = "arma11_p050_qm030_n1500_s6"
_NAME_ARMA11_N800 = "arma11_p050_qm030_n800_s101"
_NAME_GARCH11_N3000 = "garch11_n3000_s50"
_NAME_GARCH11_N800 = "garch11_n800_s102"
_NAME_AG_N1500 = "ar1garch11_p050_n1500_s13"
_NAME_AG_N800 = "ar1garch11_p050_n800_s103"


# ---------------------------------------------------------------------------
# Snapshot of the ArmaGarch(1,0) × GARCH(1,1) summary on the frozen
# ``ar1garch11_p050_n1500_s13`` series.  Regenerate by running:
#
#     .venv/bin/python -c "
#     from copulax.tests._timeseries_helpers import STANDARD, shared_fit
#     from copulax.timeseries import ArmaGarch, GARCH
#     from copulax.univariate import normal
#     print(shared_fit(ArmaGarch(mean_order=(1, 0), var_model=GARCH,
#         var_order=(1, 1), residual_dist=normal),
#         'ar1garch11_p050_n1500_s13', tier=STANDARD).summary())"
#
# and pasting the result below verbatim, after manually verifying the
# new output is *correct* (not just different).  The snapshot guards
# against silent format drift; it is *not* a guarantee that the
# numerical values are right — that's what the cross-validation tests
# above are for.  The comparison is structural, not byte-exact: the
# text skeleton (headers, labels, separators, significance marks,
# decisions) must match exactly, while printed numbers are compared as
# floats at a magnitude-sanity tolerance, because optimizer arithmetic
# drifts in the trailing digits across jax versions and platforms.
# ---------------------------------------------------------------------------
_ARMAGARCH_SUMMARY_SNAPSHOT = """\
ArmaGarch(1,0) × GARCH(1,1) — Normal residuals
==============================================================================
param            estimate          CI             std err       z    P>|z|
---- Mean equation — ARMA(1, 0) ----------------------------------------------
phi[1]             0.4664  [+0.4197, +0.5130]      0.0238   19.61   0.0000 ***
mu                -0.0115  [-0.0899, +0.0669]      0.0400   -0.29   0.7742
---- Variance equation — GARCH(1, 1) -----------------------------------------
omega              0.0501  [+0.0122, +0.0880]      0.0193    2.59   0.0096 **
alpha[1]           0.1134  [+0.0599, +0.1669]      0.0273    4.16   0.0000 ***
beta[1]            0.8312  [+0.7451, +0.9174]      0.0440   18.91   0.0000 ***
---- Residual diagnostics ----------------------------------------------------
test                               statistic    p-value decision (α=0.05)
ljung_box(z, lags=10)                   3.91     0.9175 fail to reject H0 ✓
ljung_box(z², lags=10)                  9.78     0.2810 fail to reject H0 ✓
arch_lm(z, lags=5)                      3.99     0.5514 fail to reject H0 ✓
adf(z, regression="c")                 -7.63     0.0000 reject H0 ✓
kpss(z, regression="c")                 0.03     0.6781 fail to reject H0 ✓
------------------------------------------------------------------------------
Signif. codes:  ***  p<0.001    **  p<0.01    *  p<0.05    .  p<0.1
------------------------------------------------------------------------------
loglikelihood: -1988.4907  AIC: 3986.9814  BIC: 4013.5476  n_train: 1500
convergence: converged  (grad_norm: 8.85e-06, iters: 300)
=============================================================================="""


# ---------------------------------------------------------------------------
# Shape / positivity / finiteness — mean models
# ---------------------------------------------------------------------------
class TestMeanModelStandardErrors:
    """Every standalone mean-model fit populates ``cov_matrix_``,
    ``standard_errors_``, and ``residual_diagnostics_`` with positive,
    finite entries that mirror ``params``' nested shape.
    """

    def _assert_se_dict_shape(self, fit):
        assert fit.cov_matrix_ is not None
        assert fit.standard_errors_ is not None
        # Top-level keys mirror params.
        assert set(fit.standard_errors_.keys()) == set(fit.params.keys())
        for key, val in fit.params.items():
            se = fit.standard_errors_[key]
            if key == "residual":
                assert isinstance(se, dict)
                for sub_key, sub_val in val.items():
                    np.testing.assert_array_equal(
                        np.asarray(se[sub_key]).shape,
                        np.asarray(sub_val).shape,
                    )
                    assert float(jnp.all(jnp.asarray(se[sub_key]) >= 0.0))
                    assert float(jnp.all(jnp.isfinite(jnp.asarray(se[sub_key]))))
            else:
                np.testing.assert_array_equal(
                    np.asarray(se).shape, np.asarray(val).shape,
                )
                assert float(jnp.all(jnp.asarray(se) >= 0.0))
                assert float(jnp.all(jnp.isfinite(jnp.asarray(se))))

    def test_ar1_normal(self):
        fit = shared_fit(
            AR(p=1, residual_dist=normal), _NAME_AR1_N2000, tier=STANDARD,
        )
        self._assert_se_dict_shape(fit)

    def test_ar3_normal(self):
        fit = shared_fit(
            AR(p=3, residual_dist=normal), _NAME_AR3_N2000, tier=STANDARD,
        )
        self._assert_se_dict_shape(fit)
        assert fit.standard_errors_["phi"].shape == (3,)

    def test_ma1_normal(self):
        fit = shared_fit(
            MA(q=1, residual_dist=normal), _NAME_MA1_N2000, tier=STANDARD,
        )
        self._assert_se_dict_shape(fit)

    def test_arma11_normal(self):
        fit = shared_fit(
            ARMA(p=1, q=1, residual_dist=normal), _NAME_ARMA11_N2000,
            tier=STANDARD,
        )
        self._assert_se_dict_shape(fit)

    def test_arma11_student_t(self):
        """Non-Gaussian residual coverage."""
        fit = shared_fit(
            ARMA(p=1, q=1, residual_dist=student_t), _NAME_ARMA11_N2000,
            tier=STANDARD,
        )
        self._assert_se_dict_shape(fit)
        assert "nu" in fit.standard_errors_["residual"]


# ---------------------------------------------------------------------------
# Shape / positivity / finiteness — variance models
# ---------------------------------------------------------------------------
class TestVarianceModelStandardErrors:
    """Every standalone variance-model fit populates SE machinery
    with positive, finite entries; variant-specific keys are
    present (``gamma`` for GJR/EGARCH, ``alpha_pos``/``alpha_neg``
    for TGARCH, ``psi`` for QGARCH, ``mu``/``lambda_m`` for
    GARCH-M).
    """

    def _assert_finite_positive(self, fit, expected_keys):
        assert fit.standard_errors_ is not None
        actual = set(fit.standard_errors_.keys()) - {"residual"}
        assert actual == set(expected_keys)
        for key in expected_keys:
            se = jnp.atleast_1d(jnp.asarray(fit.standard_errors_[key]))
            assert float(jnp.all(se >= 0.0))
            assert float(jnp.all(jnp.isfinite(se)))

    def test_garch11_normal(self):
        fit = shared_fit(
            GARCH(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N2000_S2,
            tier=STANDARD,
        )
        self._assert_finite_positive(fit, {"omega", "alpha", "beta"})

    def test_garch11_student_t(self, garch11_n2000_s2_student_t_fit_standard):
        fit = garch11_n2000_s2_student_t_fit_standard
        self._assert_finite_positive(fit, {"omega", "alpha", "beta"})
        assert "nu" in fit.standard_errors_["residual"]

    def test_igarch11_normal(self):
        fit = shared_fit(
            IGARCH(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N2000_S2,
            tier=STANDARD,
        )
        self._assert_finite_positive(fit, {"omega", "alpha", "beta"})

    def test_gjr_garch11_normal(self):
        fit = shared_fit(
            GJR_GARCH(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N2000_S2,
            tier=STANDARD,
        )
        self._assert_finite_positive(fit, {"omega", "alpha", "gamma", "beta"})

    def test_egarch11_normal(self):
        fit = shared_fit(
            EGARCH(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N2000_S2,
            tier=STANDARD,
        )
        self._assert_finite_positive(fit, {"omega", "alpha", "gamma", "beta"})

    def test_tgarch11_normal(self):
        fit = shared_fit(
            TGARCH(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N2000_S2,
            tier=STANDARD,
        )
        self._assert_finite_positive(
            fit, {"omega", "alpha_pos", "alpha_neg", "beta"},
        )

    def test_qgarch11_normal(self):
        fit = shared_fit(
            QGARCH(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N2000_S2,
            tier=STANDARD,
        )
        self._assert_finite_positive(fit, {"omega", "alpha", "psi", "beta"})

    def test_garch_m11_normal(self):
        # 0.02 in-mean intercept on the frozen GARCH(1,1) residuals.
        fit = shared_fit(
            GARCH_M(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N2000_S2,
            tier=STANDARD, transform=lambda eps: 0.02 + eps,
            tag="plus_0.02",
        )
        self._assert_finite_positive(
            fit, {"mu", "lambda_m", "omega", "alpha", "beta"},
        )


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------
class TestConfidenceIntervals:
    """``confidence_intervals(alpha=0.05)`` produces ``(lo, hi)``
    tuples that bracket each estimate."""

    def _assert_ci_brackets_estimate(self, fit):
        ci = fit.confidence_intervals(alpha=0.05)
        for key, val in fit.params.items():
            if key == "residual":
                for sub_key, sub_val in val.items():
                    lo, hi = ci[key][sub_key]
                    est = jnp.atleast_1d(jnp.asarray(sub_val))
                    lo_arr = jnp.atleast_1d(jnp.asarray(lo))
                    hi_arr = jnp.atleast_1d(jnp.asarray(hi))
                    assert float(jnp.all(lo_arr <= est))
                    assert float(jnp.all(est <= hi_arr))
            else:
                lo, hi = ci[key]
                est = jnp.atleast_1d(jnp.asarray(val))
                lo_arr = jnp.atleast_1d(jnp.asarray(lo))
                hi_arr = jnp.atleast_1d(jnp.asarray(hi))
                assert float(jnp.all(lo_arr <= est))
                assert float(jnp.all(est <= hi_arr))

    def test_arma11_ci_brackets_estimate(self):
        fit = shared_fit(
            ARMA(p=1, q=1, residual_dist=normal), _NAME_ARMA11_N2000,
            tier=STANDARD,
        )
        self._assert_ci_brackets_estimate(fit)

    def test_garch11_ci_brackets_estimate(self):
        fit = shared_fit(
            GARCH(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N2000_S2,
            tier=STANDARD,
        )
        self._assert_ci_brackets_estimate(fit)


# ---------------------------------------------------------------------------
# Cached-vs-recompute on residual-diagnostic methods
# ---------------------------------------------------------------------------
class TestResidualDiagnosticsCaching:
    """Cached default-arg fallback returns the stored dict; non-default
    kwargs without an explicit y/eps raise ``ValueError``."""

    @pytest.fixture(scope="class")
    def arma_fit(self):
        """The single ARMA fit every test in this class reads from.

        Fitted models are frozen equinox PyTrees and every accessor
        below is read-only, so one instance serves all consumers.
        """
        return shared_fit(
            ARMA(p=1, q=1, residual_dist=normal), _NAME_ARMA11_N1500,
            tier=STANDARD,
        ), series(_NAME_ARMA11_N1500)

    def _fit_garch(self):
        return shared_fit(
            GARCH(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N1500_S42,
            tier=STANDARD,
        ), series(SERIES_GARCH11_N1500_S42)

    def test_arma_cached_dicts(self, arma_fit):
        fit, _ = arma_fit
        rd = fit.residual_diagnostics_
        # Consolidated bundle: model-fit scalars, autocorrelation
        # arrays, and the five hypothesis-test result dicts share
        # one canonical home.
        assert set(rd.keys()) == {
            "loglikelihood", "aic", "bic", "acf", "pacf",
            "ljung_box", "ljung_box_sq", "arch_lm", "adf", "kpss",
        }
        # Scalars are finite.
        for key in ("loglikelihood", "aic", "bic"):
            assert float(jnp.isfinite(jnp.asarray(rd[key])))
        # Autocorrelation arrays are length lags+1 and start at 1.0.
        for key in ("acf", "pacf"):
            arr = jnp.asarray(rd[key])
            assert arr.shape == (21,)
            np.testing.assert_allclose(float(arr[0]), 1.0, atol=1e-6)
        # Hypothesis-test dicts have the standardised schema.
        for key in ("ljung_box", "ljung_box_sq", "arch_lm", "adf", "kpss"):
            entry = rd[key]
            assert "statistic" in entry and "p_value" in entry
            assert float(jnp.isfinite(jnp.asarray(entry["statistic"])))
            assert float(jnp.isfinite(jnp.asarray(entry["p_value"])))

    def test_arma_cached_fallback(self, arma_fit):
        fit, _ = arma_fit
        assert fit.ljung_box() is fit.residual_diagnostics_["ljung_box"]
        assert fit.arch_lm() is fit.residual_diagnostics_["arch_lm"]
        assert fit.adf_residuals() is fit.residual_diagnostics_["adf"]
        assert fit.kpss_residuals() is fit.residual_diagnostics_["kpss"]

    def test_arma_non_default_kwargs_require_y(self, arma_fit):
        fit, _ = arma_fit
        with pytest.raises(ValueError, match="y is required"):
            fit.ljung_box(lags=20)
        with pytest.raises(ValueError, match="y is required"):
            fit.arch_lm(lags=10)
        with pytest.raises(ValueError, match="y is required"):
            fit.adf_residuals(regression="ct")
        with pytest.raises(ValueError, match="y is required"):
            fit.kpss_residuals(regression="ct")

    def test_arma_recompute_with_y(self, arma_fit):
        fit, y = arma_fit
        recomp = fit.ljung_box(y)
        cached = fit.residual_diagnostics_["ljung_box"]
        np.testing.assert_allclose(
            float(recomp["statistic"]), float(cached["statistic"]), rtol=1e-6,
        )

    def test_garch_cached_fallback(self):
        fit, _ = self._fit_garch()
        assert fit.ljung_box() is fit.residual_diagnostics_["ljung_box"]
        assert fit.arch_lm() is fit.residual_diagnostics_["arch_lm"]
        # Cached "squared_residuals" path.
        assert (
            fit.ljung_box(on="squared_residuals")
            is fit.residual_diagnostics_["ljung_box_sq"]
        )
        assert fit.adf_residuals() is fit.residual_diagnostics_["adf"]
        assert fit.kpss_residuals() is fit.residual_diagnostics_["kpss"]


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------
class TestSummaryRenders:
    """``summary()`` produces a ``str`` with the expected sections,
    significance-code legend, and diagnostic glyphs."""

    @pytest.fixture(scope="class")
    def ar1_fit(self):
        """The single AR(1) fit the three rendering tests read from;
        ``summary()`` is a pure render over a frozen fitted model."""
        return shared_fit(
            AR(p=1, residual_dist=normal), _NAME_AR1_N2000, tier=STANDARD,
        )

    def _fit_garch(self):
        return shared_fit(
            GARCH(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N2000_S2,
            tier=STANDARD,
        )

    @pytest.fixture(scope="class")
    def armagarch_fit(self):
        """The single joint fit behind both the glyph test and the
        summary snapshot — identical fit object, identical render."""
        return shared_fit(
            ArmaGarch(
                mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
                residual_dist=normal,
            ),
            _NAME_AG_N1500, tier=STANDARD,
        )

    def test_ar_summary_renders(self, ar1_fit):
        out = ar1_fit.summary()
        assert isinstance(out, str)
        assert "AR(1)" in out
        assert "Mean equation — AR(1)" in out
        assert "Residual diagnostics" in out
        assert "estimate" in out and "std err" in out and "P>|z|" in out
        assert "phi[1]" in out
        assert "sigma_eps" in out
        assert "loglikelihood" in out and "AIC" in out
        # All five diagnostic test labels must appear.
        for label in (
            "ljung_box(z, lags=10)",
            "ljung_box(z²",
            "arch_lm(z, lags=5)",
            'adf(z, regression="c")',
            'kpss(z, regression="c")',
        ):
            assert label in out, f"missing diagnostic label {label!r}"
        assert "Signif. codes:" in out

    def test_ma_summary_renders(self):
        fit = shared_fit(
            MA(q=1, residual_dist=normal), _NAME_MA1_N1500, tier=STANDARD,
        )
        out = fit.summary()
        assert "MA(1)" in out
        assert "Mean equation — MA(1)" in out
        assert "theta[1]" in out
        # phi rows must NOT be present (p=0).
        assert "phi[" not in out

    def test_arma_summary_renders(self):
        fit = shared_fit(
            ARMA(p=1, q=1, residual_dist=normal), _NAME_ARMA11_N1500,
            tier=STANDARD,
        )
        out = fit.summary()
        assert "ARMA(1, 1)" in out
        assert "phi[1]" in out and "theta[1]" in out

    def test_garch_summary_renders(self):
        out = self._fit_garch().summary()
        assert "GARCH(1, 1)" in out
        assert "Variance equation — GARCH(1, 1)" in out
        assert "omega" in out and "alpha[1]" in out and "beta[1]" in out

    def test_igarch_summary_renders(self):
        fit = shared_fit(
            IGARCH(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N2000_S2,
            tier=STANDARD,
        )
        out = fit.summary()
        assert "IGARCH(1, 1)" in out
        assert "Variance equation — IGARCH(1, 1)" in out
        assert "omega" in out and "alpha[1]" in out and "beta[1]" in out

    def test_egarch_summary_has_gamma(self):
        fit = shared_fit(
            EGARCH(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N2000_S2,
            tier=STANDARD,
        )
        out = fit.summary()
        assert "EGARCH(1, 1)" in out
        # EGARCH leverage parameter — distinct from GJR's gamma but
        # uses the same key name in CopulAX.
        assert "gamma[1]" in out

    def test_gjr_garch_summary_has_gamma(self):
        fit = shared_fit(
            GJR_GARCH(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N2000_S2,
            tier=STANDARD,
        )
        out = fit.summary()
        assert "GJR_GARCH(1, 1)" in out
        assert "gamma[1]" in out

    def test_tgarch_summary_has_alpha_pos_neg(self):
        fit = shared_fit(
            TGARCH(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N2000_S2,
            tier=STANDARD,
        )
        out = fit.summary()
        assert "TGARCH(1, 1)" in out
        assert "alpha_pos[1]" in out and "alpha_neg[1]" in out

    def test_qgarch_summary_has_psi(self):
        fit = shared_fit(
            QGARCH(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N2000_S2,
            tier=STANDARD,
        )
        out = fit.summary()
        assert "QGARCH(1, 1)" in out
        assert "psi" in out

    def test_garch_m_summary_has_mu_and_lambda(self):
        fit = shared_fit(
            GARCH_M(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N2000_S2,
            tier=STANDARD, transform=lambda eps: 0.02 + eps,
            tag="plus_0.02",
        )
        out = fit.summary()
        assert "GARCH_M(1, 1)" in out
        assert "mu" in out and "lambda_m" in out

    def test_section_separators(self):
        """ArmaGarch with skewed-T residuals exercises all three param
        sections + diagnostics."""
        from copulax.univariate import skewed_t
        fit = shared_fit(
            ArmaGarch(
                mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
                residual_dist=skewed_t,
            ),
            _NAME_AG_N1500, tier=STANDARD,
        )
        out = fit.summary()
        # Four inline-labelled separators in this fit.
        assert "---- Mean equation —" in out
        assert "---- Variance equation —" in out
        assert "---- Residual distribution —" in out
        assert "---- Residual diagnostics ----" in out

    def test_section_separator_residual_distribution_suppressed_for_normal(
        self, ar1_fit,
    ):
        """Normal residual law has no free shape params — section is
        silently suppressed."""
        out = ar1_fit.summary()
        assert "---- Residual distribution —" not in out
        # But the other separators still appear.
        assert "---- Mean equation —" in out
        assert "---- Residual diagnostics ----" in out

    def test_significance_codes_emitted(self, ar1_fit):
        """Well-determined model produces ``***`` codes on the strong
        coefficients and the legend appears exactly once."""
        out = ar1_fit.summary()
        assert "***" in out  # phi[1] should be highly significant
        assert out.count("Signif. codes:") == 1

    def test_diagnostic_decisions_glyphs(self, armagarch_fit):
        """A well-specified fit produces all-✓ diagnostics."""
        out = armagarch_fit.summary()
        # All five diagnostic rows should end with ✓ for a healthy fit.
        diag_lines = [
            line for line in out.splitlines()
            if any(line.startswith(prefix) for prefix in (
                "ljung_box", "arch_lm(", "adf(", "kpss(",
            ))
        ]
        assert len(diag_lines) == 5
        for line in diag_lines:
            assert line.rstrip().endswith("✓"), (
                f"expected ✓ for healthy fit, got: {line!r}"
            )

    _SNAPSHOT_NUM_RE = re.compile(r"-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\d+")

    def test_armagarch_summary_snapshot(self, armagarch_fit):
        """Locks the rendered ArmaGarch summary against the reference
        snapshot structurally: every line's text skeleton (headers,
        parameter labels, separators, significance marks, decision
        strings) must match EXACTLY, while the printed numbers are
        compared as floats at a 2% magnitude-sanity tolerance.  This
        keeps the test valid across jax versions and platforms whose
        optimizer arithmetic drifts in the trailing printed digits;
        numeric CORRECTNESS is owned by the cross-validation tests.

        The convergence line's grad_norm is noise-scale at the optimum
        (its relative value is environment luck), so it is asserted
        against the convergence criterion (< 1e-3) instead of the
        snapshot value.

        If this fails after an intentional format change: regenerate
        the snapshot via the helper at the top of the file and verify
        the new output is correct *visually* before accepting the diff.
        """
        out = armagarch_fit.summary()
        out_lines = out.splitlines()
        exp_lines = _ARMAGARCH_SUMMARY_SNAPSHOT.splitlines()
        assert len(out_lines) == len(exp_lines), (
            f"summary line count diverged from snapshot: "
            f"{len(out_lines)} != {len(exp_lines)}.  "
            "Regenerate via the comment at the top of the test file."
        )
        for i, (got, exp) in enumerate(zip(out_lines, exp_lines)):
            got_skel = self._SNAPSHOT_NUM_RE.sub("<n>", got)
            exp_skel = self._SNAPSHOT_NUM_RE.sub("<n>", exp)
            assert got_skel == exp_skel, (
                f"summary line {i} skeleton diverged from snapshot:\n"
                f"  got: {got!r}\n  exp: {exp!r}\n"
                "Regenerate via the comment at the top of the test file."
            )
            got_nums = [float(x) for x in self._SNAPSHOT_NUM_RE.findall(got)]
            exp_nums = [float(x) for x in self._SNAPSHOT_NUM_RE.findall(exp)]
            if got.startswith("convergence:"):
                # nums = [grad_norm, iters]: grad_norm vs the criterion,
                # the remainder vs the snapshot.
                assert got_nums[0] < 1e-3, (
                    f"convergence line grad_norm {got_nums[0]!r} exceeds "
                    f"the < 1e-3 criterion: {got!r}"
                )
                np.testing.assert_allclose(
                    got_nums[1:], exp_nums[1:], rtol=0.02,
                    err_msg=f"summary line {i} numbers diverged: {got!r}",
                )
            else:
                np.testing.assert_allclose(
                    got_nums, exp_nums, rtol=0.02, atol=1e-3,
                    err_msg=(
                        f"summary line {i} numbers diverged beyond the "
                        f"2% sanity tolerance:\n  got: {got!r}\n  exp: {exp!r}"
                    ),
                )


# ---------------------------------------------------------------------------
# Unfitted summary error path
# ---------------------------------------------------------------------------
class TestSummaryErrors:
    @pytest.mark.parametrize("cls", [AR, MA, ARMA, GARCH])
    def test_unfitted_summary_raises(self, cls):
        # Provide minimum required orders.
        if cls is AR:
            inst = cls(p=1, residual_dist=normal)
        elif cls is MA:
            inst = cls(q=1, residual_dist=normal)
        else:
            inst = cls(p=1, q=1, residual_dist=normal)
        with pytest.raises(ValueError, match="not fitted"):
            inst.summary()


# ---------------------------------------------------------------------------
# Cross-validation against statsmodels / arch
# ---------------------------------------------------------------------------
class TestStatsmodelsCrossValidation:
    """Numerical validation against ``statsmodels.tsa.arima.ARIMA``.

    Important caveat: ``statsmodels.tsa.arima.ARIMA`` uses the
    **exact** likelihood via a Kalman-filter ``state-space`` representation,
    whereas CopulAX uses the **conditional** likelihood with a
    backcast-anchored pre-sample state.  These two MLE objectives have
    different optima — at finite ``n`` the parameter estimates
    typically differ in the 4th decimal (rel-diff ~3e-4) and the SEs
    differ by a few percent (rel-diff ~5e-2).  Both are "correct" SEs
    for their respective likelihoods — the tolerances below reflect
    the genuine model-formulation gap, not optimisation noise.
    Empirically CopulAX's conditional log-likelihood is HIGHER than
    statsmodels' (under the conditional likelihood, which is what
    CopulAX optimises), confirming the optimisation finds its own
    objective's minimum.
    """

    @pytest.fixture(scope="class")
    def smt(self):
        return require_oracle("statsmodels.tsa.arima.model")

    def test_ar1_se_vs_statsmodels(self, smt):
        y = series(_NAME_AR1_N3000)
        cx = shared_fit(
            AR(p=1, residual_dist=normal), _NAME_AR1_N3000, tier=PRECISION,
        )
        sm = smt.ARIMA(np.asarray(y), order=(1, 0, 0)).fit()

        # Parameter recovery — exact-MLE vs conditional-MLE differ in
        # the ~3rd-4th decimal at n=3000.  ``rtol=5e-3`` is the
        # observed envelope.
        cx_phi = float(cx.params["phi"][0])
        sm_phi = float(sm.params[1])
        np.testing.assert_allclose(cx_phi, sm_phi, rtol=5e-3, atol=1e-3)

        # SE agreement: the exact-vs-conditional likelihood difference
        # produces a slightly different Hessian curvature so the two
        # SEs differ by ~4-5%.  Both are genuine SEs for their own
        # likelihood.
        cx_se_phi = float(cx.standard_errors_["phi"][0])
        sm_se_phi = float(sm.bse[1])
        np.testing.assert_allclose(cx_se_phi, sm_se_phi, rtol=8e-2, atol=2e-3)


class TestADvsFDSelfConsistency:
    """AD-Hessian (production) vs finite-difference Hessian (in-test).

    This is the strongest correctness check for the SE pipeline because
    it makes no third-party comparison — it verifies CopulAX's
    ``jax.hessian``-based covariance against an independent
    finite-difference Hessian computed in the test, on the *same*
    negative-log-likelihood function.  Catches AD-pipeline bugs (wrong
    sign in score, missed term, bad pack/unpack) that cross-validation
    against statsmodels / arch wouldn't expose because those libraries
    use different likelihoods (exact vs conditional) and different
    parameter conventions.

    Marked slow because the finite-difference Hessian on 5+ parameters
    requires O(k²) extra fits.  Run before merging changes that touch
    ``_natural_objective_closures`` or ``_compute_se``.
    """

    @pytest.mark.slow
    def test_arma11_normal_ad_vs_fd_hessian(self):
        from copulax._src.timeseries._se import params_to_flat
        y = series(_NAME_ARMA11_N2000)
        cx = shared_fit(
            ARMA(p=1, q=1, residual_dist=normal), _NAME_ARMA11_N2000,
            tier=PRECISION,
        )

        # Build the natural-NLL closure CopulAX uses for SEs, then
        # finite-difference its Hessian outside JAX.
        wrapper = cx._wrapper()
        from copulax._src.timeseries._init import arma_pre_sample_state
        init_y_lags, init_eps_lags = arma_pre_sample_state(
            jnp.asarray(y), cx.p, cx.q, mode="backcast",
            backcast_length=None,
        )
        nll_total, _, schema = cx._natural_objective_closures(
            wrapper, cx.params, jnp.asarray(y), init_y_lags, init_eps_lags,
        )
        params_flat, _ = params_to_flat(cx.params)
        k = params_flat.shape[0]

        # Symmetric finite-difference Hessian: O(k²) NLL evaluations.
        # h chosen as sqrt(eps_mach) ≈ 1.5e-8 scaled by parameter
        # magnitude; standard textbook recipe.
        h_scale = float(np.sqrt(np.finfo(np.float32).eps))
        h_vec = h_scale * np.maximum(np.abs(np.asarray(params_flat)), 1.0)
        H_fd = np.zeros((k, k), dtype=float)
        f0 = float(nll_total(params_flat))
        for i in range(k):
            for j in range(i, k):
                ei = np.zeros(k); ei[i] = h_vec[i]
                ej = np.zeros(k); ej[j] = h_vec[j]
                f_pp = float(nll_total(params_flat + ei + ej))
                f_pm = float(nll_total(params_flat + ei - ej))
                f_mp = float(nll_total(params_flat - ei + ej))
                f_mm = float(nll_total(params_flat - ei - ej))
                H_fd[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h_vec[i] * h_vec[j])
                H_fd[j, i] = H_fd[i, j]

        # AD Hessian — what production uses.
        H_ad = np.asarray(jax.hessian(nll_total)(params_flat))

        # Compare SEs derived from the two Hessians rather than per-cell
        # Hessian agreement: off-diagonal cells with magnitude near
        # zero are dominated by FD noise (∝ √eps_f32 ≈ 1e-4) and would
        # cause spurious failures, but the SEs are scaled to parameter
        # magnitudes so the comparison is robust.
        n_obs = int(y.shape[0])
        eye_k = np.eye(k)
        cov_ad = np.linalg.solve(H_ad / n_obs, eye_k) / n_obs
        cov_fd = np.linalg.solve(H_fd / n_obs, eye_k) / n_obs
        se_ad = np.sqrt(np.maximum(np.diag(cov_ad), 0.0))
        se_fd = np.sqrt(np.maximum(np.diag(cov_fd), 0.0))
        np.testing.assert_allclose(se_ad, se_fd, rtol=5e-2, atol=1e-3)

    @pytest.mark.slow
    def test_garch11_student_t_ad_vs_fd_hessian(self):
        """Same self-consistency check for a non-Gaussian residual law,
        which is where third-party validation isn't available."""
        from copulax._src.timeseries._se import params_to_flat
        eps = series(SERIES_GARCH11_N2000_S2)
        cx = shared_fit(
            GARCH(p=1, q=1, residual_dist=student_t), SERIES_GARCH11_N2000_S2,
            tier=PRECISION,
        )

        wrapper = cx._wrapper()
        init_state = cx._ag_initial_state(
            eps_proxy=jnp.asarray(eps), mode="backcast",
            backcast_length=None,
            residual_params=cx.residual_params,
        )
        nll_total, _, schema = cx._natural_objective_closures(
            wrapper, cx.params, jnp.asarray(eps), init_state,
        )
        params_flat, _ = params_to_flat(cx.params)
        k = params_flat.shape[0]
        h_scale = float(np.sqrt(np.finfo(np.float32).eps))
        h_vec = h_scale * np.maximum(np.abs(np.asarray(params_flat)), 1.0)
        H_fd = np.zeros((k, k), dtype=float)
        for i in range(k):
            for j in range(i, k):
                ei = np.zeros(k); ei[i] = h_vec[i]
                ej = np.zeros(k); ej[j] = h_vec[j]
                f_pp = float(nll_total(params_flat + ei + ej))
                f_pm = float(nll_total(params_flat + ei - ej))
                f_mp = float(nll_total(params_flat - ei + ej))
                f_mm = float(nll_total(params_flat - ei - ej))
                H_fd[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h_vec[i] * h_vec[j])
                H_fd[j, i] = H_fd[i, j]

        H_ad = np.asarray(jax.hessian(nll_total)(params_flat))
        # SE-based agreement (per the rationale in the ARMA test above):
        # the SEs scale with parameter magnitudes so the comparison is
        # robust to FD noise on off-diagonal Hessian cells.
        n_obs = int(eps.shape[0])
        eye_k = np.eye(k)
        cov_ad = np.linalg.solve(H_ad / n_obs, eye_k) / n_obs
        cov_fd = np.linalg.solve(H_fd / n_obs, eye_k) / n_obs
        se_ad = np.sqrt(np.maximum(np.diag(cov_ad), 0.0))
        se_fd = np.sqrt(np.maximum(np.diag(cov_fd), 0.0))
        np.testing.assert_allclose(se_ad, se_fd, rtol=8e-2, atol=1e-3)


class TestArchCrossValidation:
    """Numerical validation of vanilla GARCH(1,1) against the
    industry-standard ``arch`` library (Sheppard).

    Compared apples-to-apples by forcing ``arch`` to use
    ``cov_type='classic'`` (inverse observed Hessian — what CopulAX
    standalone fits use).  Both libraries optimise the same
    conditional likelihood so parameter recovery agrees to
    ``rtol≈1e-4`` (well-converged optimisers) and SEs agree to
    ``rtol≈1e-3`` (the AD-vs-FD-Hessian floor).  ``arch``'s default
    ``cov_type='robust'`` (Bollerslev-Wooldridge sandwich) would
    differ by ~5% in finite samples — that's a real estimator
    difference, not numerical noise, and is verified separately by
    the ``ArmaGarch`` test suite.
    """

    def test_garch11_se_vs_arch(self, arch_module):
        eps = series(_NAME_GARCH11_N3000)
        cx = shared_fit(
            GARCH(p=1, q=1, residual_dist=normal), _NAME_GARCH11_N3000,
            tier=PRECISION,
        )

        am = arch_module.arch_model(
            np.asarray(eps), mean="Zero", vol="GARCH", p=1, q=1, dist="Normal",
        )
        sm = am.fit(
            disp="off", show_warning=False,
            cov_type="classic",  # match CopulAX's cov_type
            options={"ftol": 1e-12},
        )

        # Parameter recovery — well-converged on n=3000 simulated
        # data, both libraries find essentially the same MLE.
        cx_omega = float(cx.params["omega"])
        cx_alpha = float(cx.params["alpha"][0])
        cx_beta = float(cx.params["beta"][0])
        sm_omega = float(sm.params["omega"])
        sm_alpha = float(sm.params["alpha[1]"])
        sm_beta = float(sm.params["beta[1]"])
        np.testing.assert_allclose(cx_omega, sm_omega, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(cx_alpha, sm_alpha, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(cx_beta, sm_beta, rtol=1e-3, atol=1e-5)

        # SE agreement — AD Hessian (CopulAX) vs FD Hessian (arch)
        # at the same MLE.  Tight tolerance is the genuine
        # AD-vs-FD-Hessian floor (~1e-3 to 1e-4).
        cx_se = np.array([
            float(cx.standard_errors_["omega"]),
            float(cx.standard_errors_["alpha"][0]),
            float(cx.standard_errors_["beta"][0]),
        ])
        sm_se = np.array([
            float(sm.std_err["omega"]),
            float(sm.std_err["alpha[1]"]),
            float(sm.std_err["beta[1]"]),
        ])
        np.testing.assert_allclose(cx_se, sm_se, rtol=2e-3, atol=1e-5)


# ---------------------------------------------------------------------------
# residual_dist promotion + uniform residuals() return shape
# ---------------------------------------------------------------------------
class TestResidualDistAndShape:
    """Pin the post-fit ``residual_dist`` contract (Fix 1) and the
    uniform ``residuals()`` return shape (Fix 3) across all three
    base families.  Guards against any future revert to passing the
    unfitted template into the fitted-instance constructor or to
    family-specific tuple / bare-array return shapes.
    """

    @pytest.fixture(scope="class")
    def arma_fit(self):
        """Shared ARMA fit — the three tests below all read the same
        frozen fitted model and its training series."""
        return shared_fit(
            ARMA(p=1, q=1, residual_dist=student_t), _NAME_ARMA11_N800,
            tier=STANDARD,
        ), series(_NAME_ARMA11_N800)

    @pytest.fixture(scope="class")
    def garch_fit(self):
        """Shared GARCH fit for the same three tests."""
        return shared_fit(
            GARCH(p=1, q=1, residual_dist=student_t), _NAME_GARCH11_N800,
            tier=STANDARD,
        ), series(_NAME_GARCH11_N800)

    @pytest.fixture(scope="class")
    def armagarch_fit(self):
        """Shared joint fit for the same three tests."""
        return (
            shared_fit(
                ArmaGarch(
                    mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
                    residual_dist=student_t,
                ),
                _NAME_AG_N800, tier=STANDARD,
            ),
            series(_NAME_AG_N800),
        )

    def test_residual_dist_is_fitted_post_fit(
        self, arma_fit, garch_fit, armagarch_fit,
    ):
        """``fit.residual_dist.params`` is non-empty + matches the
        wrapper-rebuilt full param dict."""
        from copulax._src.timeseries._residuals._standardise import (
            StandardisedResidual,
        )
        for fit, _ in (arma_fit, garch_fit, armagarch_fit):
            # Non-empty params on the field.
            assert fit.residual_dist.params is not None
            assert "nu" in fit.residual_dist.params  # student_t shape
            # Round-trip identity: rebuild from residual_params via
            # the wrapper and compare to the field.
            wrapper = StandardisedResidual(fit.residual_dist)
            rebuilt = wrapper.to_distribution(fit.residual_params)
            np.testing.assert_allclose(
                float(fit.residual_dist.params["nu"]),
                float(rebuilt.params["nu"]),
                rtol=1e-6,
            )
            # No legacy property.
            assert not hasattr(fit, "residual_distribution")

    def test_residual_dist_standardised_contract(
        self, arma_fit, garch_fit, armagarch_fit,
    ):
        """Samples drawn from ``fit.residual_dist`` honour the
        (mean ≈ 0, var ≈ 1) standardised contract."""
        for fit, _ in (arma_fit, garch_fit, armagarch_fit):
            samples = fit.residual_dist.sample(
                size=(2000,), key=jax.random.PRNGKey(7),
            )
            assert samples.shape == (2000,)
            assert abs(float(samples.mean())) < 0.15
            assert abs(float(samples.var()) - 1.0) < 0.25

    def test_residuals_uniform_dict_shape(
        self, arma_fit, garch_fit, armagarch_fit,
    ):
        """``.residuals(y)`` returns the same dict schema across
        ARMA / GARCH / ArmaGarch."""
        for fit, ser in (arma_fit, garch_fit, armagarch_fit):
            r = fit.residuals(ser)
            assert isinstance(r, dict)
            assert set(r.keys()) == {"residuals", "standardised_residuals"}
            assert r["residuals"].shape == r["standardised_residuals"].shape
            # standardised residual var ≈ 1 (loose tolerance — short
            # series + finite-iteration optimisation).
            z = r["standardised_residuals"]
            assert abs(float(z.var()) - 1.0) < 0.4
