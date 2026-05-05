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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
# Snapshot of the ArmaGarch(1,0) × GARCH(1,1) summary on a fixed-seed
# simulation.  Regenerate by running:
#
#     .venv/bin/python -c "
#     <import the simulator + ArmaGarch from this file>; print(fit.summary())"
#
# and pasting the result below verbatim, after manually verifying the
# new output is *correct* (not just different).  The snapshot guards
# against silent format drift; it is *not* a guarantee that the
# numerical values are right — that's what the cross-validation tests
# above are for.
# ---------------------------------------------------------------------------
_ARMAGARCH_SUMMARY_SNAPSHOT = """\
ArmaGarch(1,0) × GARCH(1,1) — Normal residuals
==============================================================================
param            estimate          CI             std err       z    P>|z|
---- Mean equation — ARMA(1, 0) ----------------------------------------------
phi[1]             0.4791  [+0.4306, +0.5275]      0.0247   19.38   0.0000 ***
c                 -0.0241  [-0.0729, +0.0247]      0.0249   -0.97   0.3332
---- Variance equation — GARCH(1, 1) -----------------------------------------
omega              0.0719  [+0.0176, +0.1262]      0.0277    2.59   0.0095 **
alpha[1]           0.1175  [+0.0673, +0.1676]      0.0256    4.59   0.0000 ***
beta[1]            0.8201  [+0.7313, +0.9089]      0.0453   18.11   0.0000 ***
---- Residual diagnostics ----------------------------------------------------
test                               statistic    p-value decision (α=0.05)
ljung_box(z, lags=10)                   4.95     0.8385 fail to reject H0 ✓
ljung_box(z², lags=10)                  7.12     0.5236 fail to reject H0 ✓
arch_lm(z, lags=5)                      4.82     0.4380 fail to reject H0 ✓
adf(z, regression="c")                 -6.73     0.0001 reject H0 ✓
kpss(z, regression="c")                 0.18     0.2723 fail to reject H0 ✓
------------------------------------------------------------------------------
Signif. codes:  ***  p<0.001    **  p<0.01    *  p<0.05    .  p<0.1
------------------------------------------------------------------------------
loglikelihood: -2175.6836  AIC: 4361.3673  BIC: 4387.9334  n_train: 1500
=============================================================================="""


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------
def _sim_ar1(n, phi, key, sigma=1.0):
    eps = sigma * jax.random.normal(key, (n,))

    def step(carry, e):
        return phi * carry + e, phi * carry + e

    _, ys = jax.lax.scan(step, jnp.array(0.0), eps)
    return ys


def _sim_arp(n, phi, key, sigma=1.0):
    """Simulate AR(p) with general phi vector via lag-window scan."""
    p = len(phi)
    phi_arr = jnp.asarray(phi, dtype=float)
    eps = sigma * jax.random.normal(key, (n,))

    def step(carry, e):
        # carry holds the last p observations, most recent first.
        mu = jnp.dot(phi_arr, carry)
        new_y = mu + e
        new_carry = jnp.concatenate([new_y.reshape((1,)), carry[:-1]])
        return new_carry, new_y

    init_carry = jnp.zeros((p,), dtype=float)
    _, ys = jax.lax.scan(step, init_carry, eps)
    return ys


def _sim_ma1(n, theta, key, sigma=1.0):
    eps = sigma * jax.random.normal(key, (n,))

    def step(carry, e):
        eps_lag = carry
        y_t = theta * eps_lag + e
        return e, y_t

    _, ys = jax.lax.scan(step, jnp.array(0.0), eps)
    return ys


def _sim_arma11(n, phi, theta, key, sigma=1.0):
    eps = sigma * jax.random.normal(key, (n,))

    def step(carry, e):
        y_lag, eps_lag = carry
        y_t = phi * y_lag + theta * eps_lag + e
        return (y_t, e), y_t

    _, ys = jax.lax.scan(step, (jnp.array(0.0), jnp.array(0.0)), eps)
    return ys


def _sim_garch11(n, omega, alpha, beta, key):
    sigma2_uncond = omega / (1.0 - alpha - beta)
    z = jax.random.normal(key, (n,))

    def step(carry, z_t):
        sigma2_prev, eps2_prev = carry
        sigma2_t = omega + alpha * eps2_prev + beta * sigma2_prev
        eps_t = jnp.sqrt(sigma2_t) * z_t
        return (sigma2_t, eps_t * eps_t), eps_t

    _, eps = jax.lax.scan(step, (sigma2_uncond, sigma2_uncond), z)
    return eps


def _sim_ar1_garch11(n, phi, omega, alpha, beta, key):
    sigma2_uncond = omega / (1.0 - alpha - beta)
    z = jax.random.normal(key, (n,))

    def step(carry, z_t):
        y_prev, sigma2_prev, eps2_prev = carry
        sigma2_t = omega + alpha * eps2_prev + beta * sigma2_prev
        eps_t = jnp.sqrt(sigma2_t) * z_t
        y_t = phi * y_prev + eps_t
        return (y_t, sigma2_t, eps_t * eps_t), y_t

    _, y = jax.lax.scan(step, (0.0, sigma2_uncond, sigma2_uncond), z)
    return y


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
        key = jax.random.PRNGKey(0)
        y = _sim_ar1(2000, 0.5, key)
        fit = AR(p=1, residual_dist=normal).fit(y, maxiter=300)
        self._assert_se_dict_shape(fit)

    def test_ar3_normal(self):
        key = jax.random.PRNGKey(1)
        y = _sim_arp(2000, [0.4, -0.2, 0.1], key)
        fit = AR(p=3, residual_dist=normal).fit(y, maxiter=300)
        self._assert_se_dict_shape(fit)
        assert fit.standard_errors_["phi"].shape == (3,)

    def test_ma1_normal(self):
        key = jax.random.PRNGKey(2)
        y = _sim_ma1(2000, 0.4, key)
        fit = MA(q=1, residual_dist=normal).fit(y, maxiter=300)
        self._assert_se_dict_shape(fit)

    def test_arma11_normal(self):
        key = jax.random.PRNGKey(3)
        y = _sim_arma11(2000, 0.5, -0.3, key)
        fit = ARMA(p=1, q=1, residual_dist=normal).fit(y, maxiter=300)
        self._assert_se_dict_shape(fit)

    def test_arma11_student_t(self):
        """Non-Gaussian residual coverage."""
        key = jax.random.PRNGKey(4)
        y = _sim_arma11(2000, 0.5, -0.3, key)
        fit = ARMA(p=1, q=1, residual_dist=student_t).fit(y, maxiter=300)
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

    @pytest.fixture(scope="class")
    def garch11_eps(self):
        return _sim_garch11(2000, 0.05, 0.10, 0.85, jax.random.PRNGKey(42))

    def _assert_finite_positive(self, fit, expected_keys):
        assert fit.standard_errors_ is not None
        actual = set(fit.standard_errors_.keys()) - {"residual"}
        assert actual == set(expected_keys)
        for key in expected_keys:
            se = jnp.atleast_1d(jnp.asarray(fit.standard_errors_[key]))
            assert float(jnp.all(se >= 0.0))
            assert float(jnp.all(jnp.isfinite(se)))

    def test_garch11_normal(self, garch11_eps):
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(
            garch11_eps, maxiter=300,
        )
        self._assert_finite_positive(fit, {"omega", "alpha", "beta"})

    def test_garch11_student_t(self, garch11_eps):
        fit = GARCH(p=1, q=1, residual_dist=student_t).fit(
            garch11_eps, maxiter=300,
        )
        self._assert_finite_positive(fit, {"omega", "alpha", "beta"})
        assert "nu" in fit.standard_errors_["residual"]

    def test_igarch11_normal(self, garch11_eps):
        fit = IGARCH(p=1, q=1, residual_dist=normal).fit(
            garch11_eps, maxiter=300,
        )
        self._assert_finite_positive(fit, {"omega", "alpha", "beta"})

    def test_gjr_garch11_normal(self, garch11_eps):
        fit = GJR_GARCH(p=1, q=1, residual_dist=normal).fit(
            garch11_eps, maxiter=300,
        )
        self._assert_finite_positive(fit, {"omega", "alpha", "gamma", "beta"})

    def test_egarch11_normal(self, garch11_eps):
        fit = EGARCH(p=1, q=1, residual_dist=normal).fit(
            garch11_eps, maxiter=300,
        )
        self._assert_finite_positive(fit, {"omega", "alpha", "gamma", "beta"})

    def test_tgarch11_normal(self, garch11_eps):
        fit = TGARCH(p=1, q=1, residual_dist=normal).fit(
            garch11_eps, maxiter=300,
        )
        self._assert_finite_positive(
            fit, {"omega", "alpha_pos", "alpha_neg", "beta"},
        )

    def test_qgarch11_normal(self, garch11_eps):
        fit = QGARCH(p=1, q=1, residual_dist=normal).fit(
            garch11_eps, maxiter=300,
        )
        self._assert_finite_positive(fit, {"omega", "alpha", "psi", "beta"})

    def test_garch_m11_normal(self):
        key = jax.random.PRNGKey(43)
        eps = _sim_garch11(2000, 0.05, 0.10, 0.85, key)
        y = 0.02 + eps  # add an in-mean intercept
        fit = GARCH_M(p=1, q=1, residual_dist=normal).fit(y, maxiter=300)
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
        key = jax.random.PRNGKey(5)
        y = _sim_arma11(2000, 0.5, -0.3, key)
        fit = ARMA(p=1, q=1, residual_dist=normal).fit(y, maxiter=300)
        self._assert_ci_brackets_estimate(fit)

    def test_garch11_ci_brackets_estimate(self):
        key = jax.random.PRNGKey(44)
        eps = _sim_garch11(2000, 0.05, 0.10, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=300)
        self._assert_ci_brackets_estimate(fit)


# ---------------------------------------------------------------------------
# Cached-vs-recompute on residual-diagnostic methods
# ---------------------------------------------------------------------------
class TestResidualDiagnosticsCaching:
    """Cached default-arg fallback returns the stored dict; non-default
    kwargs without an explicit y/eps raise ``ValueError``."""

    def _fit_arma(self):
        key = jax.random.PRNGKey(6)
        y = _sim_arma11(1500, 0.5, -0.3, key)
        fit = ARMA(p=1, q=1, residual_dist=normal).fit(y, maxiter=300)
        return fit, y

    def _fit_garch(self):
        key = jax.random.PRNGKey(7)
        eps = _sim_garch11(1500, 0.05, 0.10, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=300)
        return fit, eps

    def test_arma_cached_dicts(self):
        fit, _ = self._fit_arma()
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

    def test_arma_cached_fallback(self):
        fit, _ = self._fit_arma()
        assert fit.ljung_box() is fit.residual_diagnostics_["ljung_box"]
        assert fit.arch_lm() is fit.residual_diagnostics_["arch_lm"]
        assert fit.adf_residuals() is fit.residual_diagnostics_["adf"]
        assert fit.kpss_residuals() is fit.residual_diagnostics_["kpss"]

    def test_arma_non_default_kwargs_require_y(self):
        fit, _ = self._fit_arma()
        with pytest.raises(ValueError, match="y is required"):
            fit.ljung_box(lags=20)
        with pytest.raises(ValueError, match="y is required"):
            fit.arch_lm(lags=10)
        with pytest.raises(ValueError, match="y is required"):
            fit.adf_residuals(regression="ct")
        with pytest.raises(ValueError, match="y is required"):
            fit.kpss_residuals(regression="ct")

    def test_arma_recompute_with_y(self):
        fit, y = self._fit_arma()
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

    def _fit_ar(self, p=1):
        key = jax.random.PRNGKey(8)
        if p == 1:
            y = _sim_ar1(2000, 0.5, key)
        else:
            y = _sim_arp(2000, [0.4, -0.2, 0.1][:p], key)
        return AR(p=p, residual_dist=normal).fit(y, maxiter=300)

    def _fit_garch(self):
        key = jax.random.PRNGKey(9)
        eps = _sim_garch11(2000, 0.05, 0.10, 0.85, key)
        return GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=300)

    def _fit_armagarch(self):
        key = jax.random.PRNGKey(13)
        y = _sim_ar1_garch11(1500, 0.5, 0.05, 0.10, 0.85, key)
        return ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, maxiter=400)

    def test_ar_summary_renders(self):
        out = self._fit_ar(p=1).summary()
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
        key = jax.random.PRNGKey(10)
        y = _sim_ma1(1500, 0.4, key)
        fit = MA(q=1, residual_dist=normal).fit(y, maxiter=300)
        out = fit.summary()
        assert "MA(1)" in out
        assert "Mean equation — MA(1)" in out
        assert "theta[1]" in out
        # phi rows must NOT be present (p=0).
        assert "phi[" not in out

    def test_arma_summary_renders(self):
        key = jax.random.PRNGKey(11)
        y = _sim_arma11(1500, 0.5, -0.3, key)
        fit = ARMA(p=1, q=1, residual_dist=normal).fit(y, maxiter=300)
        out = fit.summary()
        assert "ARMA(1, 1)" in out
        assert "phi[1]" in out and "theta[1]" in out

    def test_garch_summary_renders(self):
        out = self._fit_garch().summary()
        assert "GARCH(1, 1)" in out
        assert "Variance equation — GARCH(1, 1)" in out
        assert "omega" in out and "alpha[1]" in out and "beta[1]" in out

    def test_igarch_summary_renders(self):
        key = jax.random.PRNGKey(40)
        eps = _sim_garch11(2000, 0.05, 0.10, 0.85, key)
        fit = IGARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=300)
        out = fit.summary()
        assert "IGARCH(1, 1)" in out
        assert "Variance equation — IGARCH(1, 1)" in out
        assert "omega" in out and "alpha[1]" in out and "beta[1]" in out

    def test_egarch_summary_has_gamma(self):
        key = jax.random.PRNGKey(41)
        eps = _sim_garch11(2000, 0.05, 0.10, 0.85, key)
        fit = EGARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=300)
        out = fit.summary()
        assert "EGARCH(1, 1)" in out
        # EGARCH leverage parameter — distinct from GJR's gamma but
        # uses the same key name in CopulAX.
        assert "gamma[1]" in out

    def test_gjr_garch_summary_has_gamma(self):
        key = jax.random.PRNGKey(14)
        eps = _sim_garch11(2000, 0.05, 0.10, 0.85, key)
        fit = GJR_GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=300)
        out = fit.summary()
        assert "GJR_GARCH(1, 1)" in out
        assert "gamma[1]" in out

    def test_tgarch_summary_has_alpha_pos_neg(self):
        key = jax.random.PRNGKey(15)
        eps = _sim_garch11(2000, 0.05, 0.10, 0.85, key)
        fit = TGARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=300)
        out = fit.summary()
        assert "TGARCH(1, 1)" in out
        assert "alpha_pos[1]" in out and "alpha_neg[1]" in out

    def test_qgarch_summary_has_psi(self):
        key = jax.random.PRNGKey(16)
        eps = _sim_garch11(2000, 0.05, 0.10, 0.85, key)
        fit = QGARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=300)
        out = fit.summary()
        assert "QGARCH(1, 1)" in out
        assert "psi" in out

    def test_garch_m_summary_has_mu_and_lambda(self):
        key = jax.random.PRNGKey(17)
        eps = _sim_garch11(2000, 0.05, 0.10, 0.85, key)
        y = 0.02 + eps
        fit = GARCH_M(p=1, q=1, residual_dist=normal).fit(y, maxiter=300)
        out = fit.summary()
        assert "GARCH_M(1, 1)" in out
        assert "mu" in out and "lambda_m" in out

    def test_section_separators(self):
        """ArmaGarch with skewed-T residuals exercises all three param
        sections + diagnostics."""
        key = jax.random.PRNGKey(18)
        y = _sim_ar1_garch11(1500, 0.5, 0.05, 0.10, 0.85, key)
        from copulax.univariate import skewed_t
        fit = ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=skewed_t,
        ).fit(y, maxiter=400)
        out = fit.summary()
        # Four inline-labelled separators in this fit.
        assert "---- Mean equation —" in out
        assert "---- Variance equation —" in out
        assert "---- Residual distribution —" in out
        assert "---- Residual diagnostics ----" in out

    def test_section_separator_residual_distribution_suppressed_for_normal(self):
        """Normal residual law has no free shape params — section is
        silently suppressed."""
        out = self._fit_ar(p=1).summary()
        assert "---- Residual distribution —" not in out
        # But the other separators still appear.
        assert "---- Mean equation —" in out
        assert "---- Residual diagnostics ----" in out

    def test_significance_codes_emitted(self):
        """Well-determined model produces ``***`` codes on the strong
        coefficients and the legend appears exactly once."""
        out = self._fit_ar(p=1).summary()
        assert "***" in out  # phi[1] should be highly significant
        assert out.count("Signif. codes:") == 1

    def test_diagnostic_decisions_glyphs(self):
        """A well-specified fit produces all-✓ diagnostics."""
        out = self._fit_armagarch().summary()
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

    def test_armagarch_summary_snapshot(self):
        """Locks the full rendered output of a deterministic ArmaGarch
        fit to a byte-snapshot.  Catches silent format drift that the
        substring-only checks above would miss.

        If this test fails after an intentional format change:
        regenerate the snapshot via the helper at the top of the
        file and verify the new output is correct *visually* before
        accepting the diff.
        """
        out = self._fit_armagarch().summary()
        expected = _ARMAGARCH_SUMMARY_SNAPSHOT
        assert out == expected, (
            "ArmaGarch summary diverged from snapshot.  "
            "Regenerate via the comment at the top of the test file."
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
        return pytest.importorskip("statsmodels.tsa.arima.model")

    def test_ar1_se_vs_statsmodels(self, smt):
        key = jax.random.PRNGKey(20)
        y = _sim_ar1(3000, 0.6, key, sigma=1.0)
        cx = AR(p=1, residual_dist=normal).fit(y, maxiter=400)
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
        key = jax.random.PRNGKey(60)
        y = _sim_arma11(2000, 0.5, -0.3, key)
        cx = ARMA(p=1, q=1, residual_dist=normal).fit(y, maxiter=500)

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
        key = jax.random.PRNGKey(61)
        eps = _sim_garch11(2000, 0.05, 0.10, 0.85, key)
        cx = GARCH(p=1, q=1, residual_dist=student_t).fit(eps, maxiter=500)

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

    @pytest.fixture(scope="class")
    def arch_mod(self):
        return pytest.importorskip("arch")

    def test_garch11_se_vs_arch(self, arch_mod):
        key = jax.random.PRNGKey(50)
        eps = _sim_garch11(3000, 0.05, 0.10, 0.85, key)
        cx = GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=800)

        am = arch_mod.arch_model(
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

    def _arma_fit(self):
        key = jax.random.PRNGKey(101)
        y = _sim_arma11(800, 0.5, -0.3, key)
        return ARMA(p=1, q=1, residual_dist=student_t).fit(y, maxiter=200), y

    def _garch_fit(self):
        key = jax.random.PRNGKey(102)
        eps = _sim_garch11(800, 0.05, 0.10, 0.85, key)
        return GARCH(p=1, q=1, residual_dist=student_t).fit(eps, maxiter=200), eps

    def _armagarch_fit(self):
        key = jax.random.PRNGKey(103)
        y = _sim_ar1_garch11(800, 0.5, 0.05, 0.10, 0.85, key)
        return (
            ArmaGarch(
                mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
                residual_dist=student_t,
            ).fit(y, maxiter=200),
            y,
        )

    def test_residual_dist_is_fitted_post_fit(self):
        """``fit.residual_dist.params`` is non-empty + matches the
        wrapper-rebuilt full param dict."""
        from copulax._src.timeseries._residuals._standardise import (
            StandardisedResidual,
        )
        for fit, _ in (
            self._arma_fit(),
            self._garch_fit(),
            self._armagarch_fit(),
        ):
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

    def test_residual_dist_standardised_contract(self):
        """Samples drawn from ``fit.residual_dist`` honour the
        (mean ≈ 0, var ≈ 1) standardised contract."""
        for fit, _ in (
            self._arma_fit(),
            self._garch_fit(),
            self._armagarch_fit(),
        ):
            samples = fit.residual_dist.sample(
                size=(2000,), key=jax.random.PRNGKey(7),
            )
            assert samples.shape == (2000,)
            assert abs(float(samples.mean())) < 0.15
            assert abs(float(samples.var()) - 1.0) < 0.25

    def test_residuals_uniform_dict_shape(self):
        """``.residuals(y)`` returns the same dict schema across
        ARMA / GARCH / ArmaGarch."""
        for fit, ser in (
            self._arma_fit(),
            self._garch_fit(),
            self._armagarch_fit(),
        ):
            r = fit.residuals(ser)
            assert isinstance(r, dict)
            assert set(r.keys()) == {"residuals", "standardised_residuals"}
            assert r["residuals"].shape == r["standardised_residuals"].shape
            # standardised residual var ≈ 1 (loose tolerance — short
            # series + finite-iteration optimisation).
            z = r["standardised_residuals"]
            assert abs(float(z.var()) - 1.0) < 0.4
