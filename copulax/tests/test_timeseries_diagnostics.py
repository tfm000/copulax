"""Diagnostics tests — ACF, PACF, Ljung-Box, ARCH-LM.

Coverage:

* Cross-validation against ``statsmodels.tsa.stattools`` and
  ``statsmodels.stats.diagnostic`` to ``rtol=1e-6, atol=1e-8`` per
  plan §"Diagnostics".  PACF compared against
  ``method='ldbiased'`` / ``method='ywm'`` (the biased-ACVF
  variant we use; statsmodels' ``method='yw'`` default is the
  *unbiased* variant and produces values that differ from us in
  the third decimal — a documented convention difference, not a
  bug).
* Smoke / shape invariants — return types, shapes, value ranges.
* Convenience methods on fitted models — ``ARMA``, ``GARCH``, and
  ``ArmaGarch`` route the diagnostics through their standardised
  residuals.
* Plot helpers — ``plot_acf`` / ``plot_pacf`` render without
  raising and return a matplotlib axes.
* JIT-compatibility of the four numerical functions.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copulax.tests._timeseries_helpers import (
    STANDARD,
    series,
    shared_fit,
)
from copulax.tests.conftest import SERIES_GARCH11_N1000_S42, require_oracle
from copulax.timeseries import (
    ARMA,
    GARCH,
    ArmaGarch,
    acf,
    adf,
    arch_lm,
    kpss,
    ljung_box,
    pacf,
    plot_acf,
    plot_pacf,
)
from copulax.univariate import normal

# ---------------------------------------------------------------------------
# Frozen series used by this module
#
# Every series is loaded from the committed corpus.  The two AR-injected
# series this file used to build with a Python ``.at[t].set(...)`` loop
# over a GARCH residual series are now frozen AR(1)-GARCH(1, 1) draws
# from rugarch (``armaOrder=c(1,0)``, ``ar1=0.3``, ``mu=0``), which is
# exactly the DGP the loop stood in for.
#
# Seed-variant collapse: the AR(1) phi=0.5 n=500 sites previously drew
# seeds 0, 13, 42 and 123.  None of the consuming tests needs an
# independent draw — they are shape, JIT, plot-smoke, p-value-range and
# unit-root checks — so they all share ``ar1_p050_n500_s42``.
#
# Only the series this module ALONE consumes are named here.  The four
# it shares with another module — ``ar1_p050_n500_s42`` and
# ``ar1_p060_n500_s42`` (with plotting), ``garch11_n1000_s42`` (with
# variance) and ``garch11_n1500_s42`` (with summary) — are fixtures in
# ``copulax/tests/conftest.py``, together with the AR(1)-Normal fit on
# ``ar1_p060_n500_s42`` that this module and plotting both request.
# ---------------------------------------------------------------------------
_AR1_P060_N1000 = "ar1_p060_n1000_s42"
_AR1_P060_N800 = "ar1_p060_n800_s7"
_AR1GARCH11_N1500_S42 = "ar1garch11_p030_n1500_s42"
_AR1GARCH11_N1500_S7 = "ar1garch11_p030_n1500_s7"


# ---------------------------------------------------------------------------
# Cross-validation against statsmodels
# ---------------------------------------------------------------------------
class TestStatsmodelsCrossValidation:
    @pytest.fixture(scope="class")
    def sm_stattools(self):
        return require_oracle("statsmodels.tsa.stattools")

    @pytest.fixture(scope="class")
    def sm_diag(self):
        return require_oracle("statsmodels.stats.diagnostic")

    def test_acf_vs_statsmodels(self, sm_stattools):
        y = series(_AR1_P060_N1000)
        cx = np.asarray(acf(y, 20))
        sm = sm_stattools.acf(np.asarray(y), nlags=20, fft=False)
        np.testing.assert_allclose(cx, sm, rtol=1e-5, atol=1e-7)

    def test_pacf_vs_statsmodels_ywm(self, sm_stattools):
        """We use biased-ACVF Yule-Walker; statsmodels exposes the
        same as ``method='ywm'`` or ``method='ldbiased'``.
        ``method='yw'`` (default) is the unbiased variant and
        differs from us by ~1e-3 — that's a documented convention
        difference per the module docstring, not a bug."""
        y = series(_AR1_P060_N1000)
        cx = np.asarray(pacf(y, 20, method="yule_walker"))
        sm = sm_stattools.pacf(np.asarray(y), nlags=20, method="ywm")
        np.testing.assert_allclose(cx, sm, rtol=1e-5, atol=1e-7)

    def test_ljung_box_vs_statsmodels(self, sm_diag):
        y = series(_AR1_P060_N1000)
        out = ljung_box(y, 10)
        sm = sm_diag.acorr_ljungbox(np.asarray(y), lags=[10], return_df=True)
        np.testing.assert_allclose(
            float(out["statistic"]),
            float(sm["lb_stat"].iloc[0]),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(out["p_value"]),
            float(sm["lb_pvalue"].iloc[0]),
            rtol=1e-3,  # very small p-values amplify rel diff
            atol=1e-100,
        )

    def test_arch_lm_vs_statsmodels(self, sm_diag, garch11_n1500_s42):
        eps = garch11_n1500_s42
        out = arch_lm(eps, 5)
        sm = sm_diag.het_arch(np.asarray(eps), nlags=5)
        # sm = (LM, p_LM, F, p_F)
        np.testing.assert_allclose(float(out["statistic"]), float(sm[0]), rtol=1e-5)
        np.testing.assert_allclose(
            float(out["p_value"]),
            float(sm[1]),
            rtol=1e-3,
            atol=1e-100,
        )


# ---------------------------------------------------------------------------
# Shape / smoke / value invariants
# ---------------------------------------------------------------------------
class TestShapes:
    def test_acf_pacf_shapes(self, ar1_p050_n500_s42):
        y = ar1_p050_n500_s42
        rho = acf(y, 15)
        pi = pacf(y, 15)
        assert rho.shape == (16,)
        assert pi.shape == (16,)
        # Lag 0 is identically 1 by definition.
        np.testing.assert_allclose(float(rho[0]), 1.0, atol=1e-12)
        np.testing.assert_allclose(float(pi[0]), 1.0, atol=1e-12)

    def test_pacf_lag_one_equals_acf_lag_one(self, ar1_p050_n500_s42):
        """By construction PACF(1) = ρ(1) under the biased-ACVF
        Yule-Walker variant."""
        y = ar1_p050_n500_s42
        rho = acf(y, 5)
        pi = pacf(y, 5)
        np.testing.assert_allclose(float(pi[1]), float(rho[1]), atol=1e-12)

    def test_ljung_box_returns_dict(self, ar1_p050_n500_s42):
        y = ar1_p050_n500_s42
        out = ljung_box(y, 10)
        assert set(out) == {
            "statistic",
            "p_value",
            "used_lag",
            "n_obs",
            "dof",
        }
        # Every leaf is a JAX array — statistic / p_value as scalar
        # float, used_lag / n_obs / dof as scalar int32.
        assert out["statistic"].shape == ()
        assert out["p_value"].shape == ()
        for sk in ("used_lag", "n_obs", "dof"):
            assert out[sk].shape == ()
            assert out[sk].dtype == jnp.int32
        # Q ≥ 0 and 0 ≤ p ≤ 1.
        assert float(out["statistic"]) >= 0.0
        assert 0.0 <= float(out["p_value"]) <= 1.0
        assert int(out["used_lag"]) == 10
        assert int(out["dof"]) == 10
        assert int(out["n_obs"]) == 500

    def test_arch_lm_returns_dict(self, garch11_n1000_s42):
        eps = garch11_n1000_s42
        out = arch_lm(eps, 5)
        assert set(out) == {
            "statistic",
            "p_value",
            "used_lag",
            "n_obs",
            "dof",
        }
        assert out["statistic"].shape == ()
        assert out["p_value"].shape == ()
        for sk in ("used_lag", "n_obs", "dof"):
            assert out[sk].shape == ()
            assert out[sk].dtype == jnp.int32
        assert float(out["statistic"]) >= 0.0
        assert 0.0 <= float(out["p_value"]) <= 1.0
        assert int(out["used_lag"]) == 5
        assert int(out["dof"]) == 5
        # arch_lm trims ``lags`` observations off the front of the
        # auxiliary regression: n_eff = n - lags.
        assert int(out["n_obs"]) == 1000 - 5


# ---------------------------------------------------------------------------
# Power: the tests reject under their respective alternative hypotheses.
# ---------------------------------------------------------------------------
class TestPower:
    def test_ljung_box_rejects_ar1(self):
        """Strong AR(1) autocorrelation should produce p ≈ 0."""
        y = series(_AR1_P060_N1000)
        assert float(ljung_box(y, 10)["p_value"]) < 1e-10

    def test_ljung_box_does_not_reject_iid(self):
        """IID Normal noise should have p > 0.05 (most of the time)."""
        key = jax.random.PRNGKey(42)
        eps = jax.random.normal(key, (1000,))
        assert float(ljung_box(eps, 10)["p_value"]) > 0.05

    def test_arch_lm_rejects_garch(self, garch11_n1500_s42):
        """GARCH(1, 1) data has strong ARCH effects; expect p ≈ 0."""
        eps = garch11_n1500_s42
        assert float(arch_lm(eps, 5)["p_value"]) < 1e-5

    def test_arch_lm_does_not_reject_iid(self):
        key = jax.random.PRNGKey(42)
        eps = jax.random.normal(key, (1000,))
        assert float(arch_lm(eps, 5)["p_value"]) > 0.05


# ---------------------------------------------------------------------------
# Convenience methods on fitted models
# ---------------------------------------------------------------------------
class TestModelDiagnosticMethods:
    def test_arma_diagnostics(
        self,
        ar1_p060_n500_s42,
        ar1_p060_n500_s42_normal_fit_standard,
    ):
        """ARMA fit exposes ``acf`` / ``pacf`` / ``ljung_box`` /
        ``arch_lm`` routed through standardised residuals."""
        y = ar1_p060_n500_s42
        fit = ar1_p060_n500_s42_normal_fit_standard
        rho = fit.acf(y, lags=10)
        pi = fit.pacf(y, lags=10)
        lb = fit.ljung_box(y, lags=10)
        al = fit.arch_lm(y, lags=5)
        assert rho.shape == (11,)
        assert pi.shape == (11,)
        assert jnp.isfinite(lb["statistic"]) and jnp.isfinite(lb["p_value"])
        assert jnp.isfinite(al["statistic"]) and jnp.isfinite(al["p_value"])

    def test_garch_diagnostics(self, garch11_n1000_s42):
        eps = garch11_n1000_s42
        fit = shared_fit(
            GARCH(p=1, q=1, residual_dist=normal),
            SERIES_GARCH11_N1000_S42,
            tier=STANDARD,
        )
        rho = fit.acf(eps, lags=10)
        pi = fit.pacf(eps, lags=10)
        lb = fit.ljung_box(eps, lags=10)
        al = fit.arch_lm(eps, lags=5)
        assert rho.shape == (11,)
        assert pi.shape == (11,)
        assert jnp.isfinite(lb["statistic"]) and jnp.isfinite(lb["p_value"])
        # After a successful GARCH fit, the standardised residuals
        # should look much closer to IID — ARCH-LM p-value much
        # higher than on the raw eps.
        raw = arch_lm(eps, lags=5)
        assert float(al["p_value"]) > float(raw["p_value"])

    def test_arma_garch_diagnostics(self):
        # AR(1)-GARCH(1,1) with phi=0.3: exactly the DGP the joint
        # composite is specified for.
        y = series(_AR1GARCH11_N1500_S42)
        fit = shared_fit(
            ArmaGarch(
                mean_order=(1, 0),
                var_model=GARCH,
                var_order=(1, 1),
                residual_dist=normal,
            ),
            _AR1GARCH11_N1500_S42,
            tier=STANDARD,
        )
        lb = fit.ljung_box(y, lags=10)
        al = fit.arch_lm(y, lags=5)
        # After joint ARMA-GARCH fit on AR(1)-GARCH(1, 1) data, both
        # tests should fail to reject (well-specified model).
        assert float(lb["p_value"]) > 0.01
        assert float(al["p_value"]) > 0.01

    def test_ljung_box_dof_correction_shifts_p_value(self):
        """``dof_correction=True`` (default) replaces ``df=lags`` with
        ``df=lags - p - q``.  Same Q statistic, smaller df shifts the
        chi-square reference left so a given Q is further into the
        upper tail and the p-value drops — making the test slightly
        less lenient on residual autocorrelation, which is the whole
        point of correcting for fitted parameters
        (Box-Jenkins-Reinsel §8.2.2).
        """
        y = series(_AR1_P060_N800)
        fit = shared_fit(
            ARMA(p=1, q=1, residual_dist=normal),
            _AR1_P060_N800,
            tier=STANDARD,
        )
        corr = fit.ljung_box(y, lags=10, dof_correction=True)
        raw = fit.ljung_box(y, lags=10, dof_correction=False)
        np.testing.assert_allclose(
            float(corr["statistic"]),
            float(raw["statistic"]),
            rtol=1e-12,
        )
        assert float(corr["p_value"]) < float(raw["p_value"])
        # The dof correction surfaces in the dict itself.
        assert int(corr["dof"]) == 10 - 1 - 1  # lags - p - q
        assert int(raw["dof"]) == 10
        # Joint composite exposes the same kwarg + an ``on=`` selector.
        ar_y = series(_AR1GARCH11_N1500_S7)
        joint = shared_fit(
            ArmaGarch(
                mean_order=(1, 0),
                var_model=GARCH,
                var_order=(1, 1),
                residual_dist=normal,
            ),
            _AR1GARCH11_N1500_S7,
            tier=STANDARD,
        )
        z_out = joint.ljung_box(ar_y, lags=10, on="residuals")
        z2_out = joint.ljung_box(ar_y, lags=10, on="squared_residuals")
        # The two ``on=`` paths consume different series so the Q
        # statistics differ in general.
        assert not np.isclose(
            float(z_out["statistic"]),
            float(z2_out["statistic"]),
            rtol=1e-3,
        )
        with pytest.raises(ValueError, match="on"):
            joint.ljung_box(ar_y, lags=10, on="invalid")


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------
class TestJIT:
    def test_acf_jit(self, ar1_p050_n500_s42):
        y = ar1_p050_n500_s42
        out_eager = acf(y, 10)
        out_jit = jax.jit(acf, static_argnames=("lags",))(y, lags=10)
        np.testing.assert_allclose(np.asarray(out_eager), np.asarray(out_jit))

    def test_ljung_box_jit(self, ar1_p050_n500_s42):
        """``ljung_box`` is directly JIT-compatible — the result dict
        is a pure-JAX pytree (every leaf is a JAX array; ``lags`` is a
        static arg, so the integer leaves are baked into the trace)."""
        y = ar1_p050_n500_s42
        eager = ljung_box(y, 10)
        jit_lb = jax.jit(ljung_box, static_argnames=("lags",))
        jit_out = jit_lb(y, lags=10)
        np.testing.assert_allclose(
            float(eager["statistic"]),
            float(jit_out["statistic"]),
        )
        np.testing.assert_allclose(
            float(eager["p_value"]),
            float(jit_out["p_value"]),
        )
        assert int(jit_out["used_lag"]) == int(eager["used_lag"])
        assert int(jit_out["dof"]) == int(eager["dof"])
        assert int(jit_out["n_obs"]) == int(eager["n_obs"])

    def test_arch_lm_jit(self, garch11_n1000_s42):
        """``arch_lm`` is directly JIT-compatible by the same argument
        — the dict is a pure-JAX pytree."""
        eps = garch11_n1000_s42
        eager = arch_lm(eps, 5)
        jit_al = jax.jit(arch_lm, static_argnames=("lags",))
        jit_out = jit_al(eps, lags=5)
        np.testing.assert_allclose(
            float(eager["statistic"]),
            float(jit_out["statistic"]),
        )
        np.testing.assert_allclose(
            float(eager["p_value"]),
            float(jit_out["p_value"]),
        )

    def test_adf_jit(self, ar1_p050_n500_s42):
        """``adf`` is directly JIT-compatible with ``regression`` and
        ``lags`` as static argnames.  The result dict is a pure-JAX
        pytree — ``crit_values`` is a ``(3,)`` array, every other leaf
        is a 0-d array."""
        y = ar1_p050_n500_s42
        eager = adf(y, regression="c", lags=12)
        jit_adf = jax.jit(adf, static_argnames=("regression", "lags"))
        jit_out = jit_adf(y, regression="c", lags=12)
        np.testing.assert_allclose(
            float(eager["statistic"]),
            float(jit_out["statistic"]),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            float(eager["p_value"]),
            float(jit_out["p_value"]),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(eager["crit_values"]),
            np.asarray(jit_out["crit_values"]),
        )
        assert int(jit_out["used_lag"]) == int(eager["used_lag"])
        assert int(jit_out["n_obs"]) == int(eager["n_obs"])

    def test_kpss_jit(self, ar1_p050_n500_s42):
        """``kpss`` is directly JIT-compatible with ``regression``,
        ``lags``, and ``lags_choice`` as static argnames.  The result
        dict is a pure-JAX pytree — ``crit_values`` is a ``(4,)``
        array."""
        y = ar1_p050_n500_s42
        eager = kpss(y, regression="c", lags_choice="long")
        jit_kpss = jax.jit(
            kpss,
            static_argnames=("regression", "lags", "lags_choice"),
        )
        jit_out = jit_kpss(y, regression="c", lags_choice="long")
        np.testing.assert_allclose(
            float(eager["statistic"]),
            float(jit_out["statistic"]),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            float(eager["p_value"]),
            float(jit_out["p_value"]),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(eager["crit_values"]),
            np.asarray(jit_out["crit_values"]),
        )
        assert int(jit_out["used_lag"]) == int(eager["used_lag"])
        assert int(jit_out["n_obs"]) == int(eager["n_obs"])


# ---------------------------------------------------------------------------
# OLS helper
# ---------------------------------------------------------------------------
class TestOLSHelper:
    """Direct contract tests on :func:`copulax._src.timeseries._ols.ols_fit`.

    The three callers (``arch_lm``, ``adf``, ``kpss``) test it
    indirectly via cross-validation against statsmodels, but a tiny
    self-contained sweep also pins the OLS arithmetic itself so a
    regression in the helper would be diagnosable in isolation.
    """

    @pytest.fixture(scope="class")
    def synthetic(self):
        """Linear DGP with i.i.d. Gaussian noise on a 3-column design
        (intercept, two regressors).  Sample size large enough to keep
        OLS estimates near the true coefficients to ~1%."""
        from copulax._src.timeseries._ols import ols_fit

        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        n = 2000
        x1 = jax.random.normal(k1, (n,))
        x2 = jax.random.normal(k2, (n,))
        eps = 0.5 * jax.random.normal(k3, (n,))
        beta_true = jnp.array([1.0, 2.5, -0.7])
        X = jnp.stack([jnp.ones((n,)), x1, x2], axis=1)
        y = X @ beta_true + eps
        return ols_fit, X, y, beta_true

    def test_beta_recovers_truth(self, synthetic):
        ols_fit, X, y, beta_true = synthetic
        out = ols_fit(X, y)
        np.testing.assert_allclose(
            np.asarray(out.beta),
            np.asarray(beta_true),
            rtol=0,
            atol=0.05,
        )

    def test_residuals_orthogonal_to_columns(self, synthetic):
        """OLS first-order condition: ``Xᵀ(y - Xβ̂) = 0``."""
        ols_fit, X, y, _ = synthetic
        out = ols_fit(X, y)
        gram_proj = X.T @ out.residuals
        np.testing.assert_allclose(
            np.asarray(gram_proj),
            0.0,
            atol=1e-9,
        )

    def test_t_stats_match_manual_se(self, synthetic):
        """t = β̂ / SE matches the canonical homoskedastic formula
        ``β̂ / sqrt(σ̂² · diag((XᵀX)⁻¹))`` to machine precision."""
        ols_fit, X, y, _ = synthetic
        out = ols_fit(X, y)
        XtX_inv = np.linalg.inv(np.asarray(X.T @ X))
        se_manual = np.sqrt(np.asarray(out.sigma2) * np.diag(XtX_inv))
        np.testing.assert_allclose(
            np.asarray(out.standard_errors),
            se_manual,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(out.t_stats),
            np.asarray(out.beta) / se_manual,
            rtol=1e-10,
        )

    def test_r_squared_matches_explicit(self, synthetic):
        """R² agrees with the textbook ``1 - RSS/TSS`` and the
        Theil-adjusted R² agrees with ``1 - (1 - R²) · (n-1)/(n-k)``,
        each to machine precision against the sample mean of ``y``."""
        ols_fit, X, y, _ = synthetic
        out = ols_fit(X, y)
        rss = float(jnp.sum(out.residuals**2))
        tss = float(jnp.sum((y - jnp.mean(y)) ** 2))
        np.testing.assert_allclose(
            float(out.r_squared),
            1.0 - rss / tss,
            rtol=1e-10,
        )
        n, k = X.shape
        adj_explicit = 1.0 - (1.0 - float(out.r_squared)) * (n - 1) / (n - k)
        np.testing.assert_allclose(
            float(out.adj_r_squared),
            adj_explicit,
            rtol=1e-10,
        )
        # Adjusted R² is below ordinary R² (DOF correction shrinks the
        # uncorrected score) but only by a hair on a 2000 × 3 problem.
        assert float(out.adj_r_squared) < float(out.r_squared)
        # And with a healthy DGP the regression should explain most
        # of the variance — sanity check the value is plausible.
        assert 0.9 < float(out.r_squared) < 1.0
        assert 0.9 < float(out.adj_r_squared) < 1.0

    def test_jit_round_trip(self, synthetic):
        """``ols_fit`` is pure JAX and traces under ``jax.jit``.

        ``rtol=1e-10`` covers XLA reordering differences in the inner
        ``XᵀX``/``Xᵀy`` reductions at ``n=2000``; the mathematical
        contract is unchanged."""
        ols_fit, X, y, _ = synthetic
        eager = ols_fit(X, y)
        jit_out = jax.jit(ols_fit)(X, y)
        for field in eager._fields:
            np.testing.assert_allclose(
                np.asarray(getattr(jit_out, field)),
                np.asarray(getattr(eager, field)),
                rtol=1e-10,
                atol=1e-12,
            )


# ---------------------------------------------------------------------------
# _interp_p_jit bit-exactness sweep
# ---------------------------------------------------------------------------
def _interp_p_python_reference(stat, crits, log_levels):
    """Pure-Python reproduction of the previous ``_interp_p`` behaviour
    (with the decorative ``lower_tail`` flag dropped — both branches
    of the original computed identical formulas).  Used only as the
    bit-exactness oracle for ``_interp_p_jit``.
    """
    import math

    n = len(crits)
    if stat <= crits[0]:
        slope = (log_levels[1] - log_levels[0]) / (crits[1] - crits[0])
        log_p = log_levels[0] + slope * (stat - crits[0])
    elif stat >= crits[-1]:
        slope = (log_levels[-1] - log_levels[-2]) / (crits[-1] - crits[-2])
        log_p = log_levels[-1] + slope * (stat - crits[-1])
    else:
        for i in range(1, n):
            if stat <= crits[i]:
                slope = (log_levels[i] - log_levels[i - 1]) / (crits[i] - crits[i - 1])
                log_p = log_levels[i - 1] + slope * (stat - crits[i - 1])
                break
    p = math.exp(log_p)
    return max(min(p, 0.99), 1e-4)


class TestInterpP:
    """Parametric bit-exactness sweep against the previous Python
    implementation of ``_interp_p_jit`` — the helper still drives the
    KPSS p-value path, so any departure from the documented log-linear
    extrapolation formula would silently shift every KPSS result
    downstream.  ADF no longer uses this helper (it interpolates
    directly between five tabulated knots with ``jnp.interp``); see
    :class:`TestADFPValueClamp` for the ADF contract.
    """

    @pytest.fixture(scope="class")
    def kpss_setup(self):
        from copulax._src.timeseries._unit_root import (
            _KPSS_CRIT_C,
            _KPSS_LOG_LEVELS,
            KPSS_CRIT_LEVELS,
            _interp_p_jit,
        )

        crits_py = tuple(float(x) for x in _KPSS_CRIT_C.tolist())
        levels_py = tuple(math.log(lv) for lv in KPSS_CRIT_LEVELS)
        return _interp_p_jit, _KPSS_CRIT_C, _KPSS_LOG_LEVELS, crits_py, levels_py

    @pytest.mark.parametrize(
        "stat",
        # Knots, midpoints, far tails.  KPSS crits for "c": (0.347,
        # 0.463, 0.574, 0.739).
        [-1.0, 0.0, 0.2, 0.347, 0.4, 0.463, 0.52, 0.574, 0.65, 0.739, 1.0, 2.0, 5.0],
    )
    def test_kpss_interp_bit_exact(self, kpss_setup, stat):
        jit_fn, crits_arr, log_levels_arr, crits_py, levels_py = kpss_setup
        ref = _interp_p_python_reference(stat, crits_py, levels_py)
        got = float(jit_fn(jnp.asarray(stat, dtype=float), crits_arr, log_levels_arr))
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)

    def test_interp_p_jit_is_jittable(self, kpss_setup):
        """The helper itself must trace under jax.jit — that's the
        whole point of the rewrite."""
        jit_fn, crits_arr, log_levels_arr, *_ = kpss_setup
        compiled = jax.jit(jit_fn)
        out = compiled(jnp.asarray(0.5), crits_arr, log_levels_arr)
        assert out.shape == ()
        assert 1e-4 <= float(out) <= 0.99


# ---------------------------------------------------------------------------
# JAX MacKinnon polynomial: cross-validation against statsmodels
# ---------------------------------------------------------------------------
class TestMacKinnonp:
    """Pin the JAX port of ``statsmodels.tsa.adfvalues.mackinnonp`` to
    machine precision against the original at every τ value the
    polynomial supports.  Catches paste / coefficient errors in
    :mod:`copulax._src.timeseries._mackinnon`.

    statsmodels is a dev-only import (the production runtime never
    pulls it).
    """

    @pytest.fixture(scope="class")
    def sm_p(self):
        sm = require_oracle("statsmodels.tsa.adfvalues")
        return sm.mackinnonp

    @pytest.mark.parametrize("regression", ["n", "c", "ct"])
    @pytest.mark.parametrize(
        "stat",
        # Sweep covering both polynomial branches (small-p / large-p),
        # exact knot percentiles, and the saturation boundaries.
        [
            -15.0,
            -10.0,
            -8.0,
            -6.0,
            -5.0,
            -4.5,
            -4.0,
            -3.5,
            -3.0,
            -2.89,
            -2.62,
            -2.5,
            -2.0,
            -1.61,
            -1.5,
            -1.04,
            -1.0,
            -0.5,
            0.0,
            0.5,
            0.7,
            1.0,
            2.0,
            2.5,
            2.74,
            3.0,
            5.0,
        ],
    )
    def test_jax_matches_statsmodels(self, sm_p, regression, stat):
        """Bit-equivalence (to ~1e-12) between our JAX port and the
        reference ``statsmodels.mackinnonp`` across the full polynomial
        support, both saturation tails, and both branch-split regions."""
        from copulax._src.timeseries._mackinnon import mackinnonp_jit

        ref = float(sm_p(stat, regression=regression, N=1))
        got = float(mackinnonp_jit(jnp.asarray(stat, dtype=float), regression))
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("regression", ["n", "c", "ct"])
    def test_saturation_clip(self, sm_p, regression):
        """Above ``tau_max`` and below ``tau_min`` (per regression) the
        p-value must clip to ``1.0`` and ``0.0`` respectively, matching
        statsmodels' early-return path."""
        from copulax._src.timeseries._mackinnon import (
            _TAU_MAX_N1,
            _TAU_MIN_N1,
            mackinnonp_jit,
        )

        tau_max = _TAU_MAX_N1[regression]
        tau_min = _TAU_MIN_N1[regression]
        # Above the upper saturation cutoff (skip 'n' where it's +∞).
        if jnp.isfinite(jnp.asarray(tau_max)):
            stat_above = tau_max + 1.0
            assert float(mackinnonp_jit(stat_above, regression)) == 1.0
            # statsmodels also returns 1.0:
            assert float(sm_p(stat_above, regression=regression, N=1)) == 1.0
        # Below the lower saturation cutoff.
        stat_below = tau_min - 1.0
        assert float(mackinnonp_jit(stat_below, regression)) == 0.0
        assert float(sm_p(stat_below, regression=regression, N=1)) == 0.0

    def test_jit_round_trip(self):
        """``mackinnonp_jit`` must trace under ``jax.jit`` with
        ``regression`` as a static argname."""
        from copulax._src.timeseries._mackinnon import mackinnonp_jit

        compiled = jax.jit(mackinnonp_jit, static_argnames=("regression",))
        eager = float(mackinnonp_jit(jnp.asarray(-3.0), "c"))
        jit_out = float(compiled(jnp.asarray(-3.0), regression="c"))
        np.testing.assert_allclose(jit_out, eager, rtol=1e-12)

    def test_asymptotic_crit_matches_statsmodels(self):
        """The vendored ``tau_2010s`` asymptotic critical values must
        match statsmodels' ``mackinnoncrit(N=1, regression=reg, nobs=inf)``
        bit-for-bit (it's the same data, sliced from the same response
        surface)."""
        sm_crit = require_oracle("statsmodels.tsa.adfvalues").mackinnoncrit
        import numpy as _np

        from copulax._src.timeseries._mackinnon import (
            mackinnon_asymptotic_crit,
        )

        for reg in ("n", "c", "ct"):
            ours = _np.asarray(mackinnon_asymptotic_crit(reg))
            theirs = sm_crit(N=1, regression=reg, nobs=float("inf"))
            np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-14)


# ---------------------------------------------------------------------------
# ADF p-value contract (uses mackinnonp polynomial)
# ---------------------------------------------------------------------------
class TestADFPValueContract:
    """End-to-end checks on ``adf()`` once the polynomial path is
    wired in: p-values stay in ``[0, 1]``, monotone in τ, and reject /
    fail-to-reject correctly on synthetic stationary / random-walk
    data."""

    def test_p_value_in_unit_interval(self, ar1_p050_n500_s42):
        y = ar1_p050_n500_s42
        out = adf(y, regression="c", lags=12)
        p = float(out["p_value"])
        assert 0.0 <= p <= 1.0

    def test_p_value_monotone_in_stat(self):
        """As τ increases (less evidence against H0), p-value must be
        non-decreasing across the polynomial's calibrated range."""
        from copulax._src.timeseries._mackinnon import mackinnonp_jit

        sweep = jnp.linspace(-15.0, 2.5, 71)
        for reg in ("n", "c", "ct"):
            ps = jax.vmap(lambda s, r=reg: mackinnonp_jit(s, r))(sweep)
            diffs = jnp.diff(ps)
            assert float(jnp.min(diffs)) >= -1e-12, (
                f"p-value not monotone for regression={reg!r}"
            )


# ---------------------------------------------------------------------------
# Plot smoke
# ---------------------------------------------------------------------------
class TestPlots:
    def test_plot_acf_smoke(self, ar1_p050_n500_s42):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y = ar1_p050_n500_s42
        fig, ax = plt.subplots()
        out_ax = plot_acf(y, lags=15, ax=ax)
        assert out_ax is ax
        plt.close(fig)

    def test_plot_pacf_smoke(self, ar1_p050_n500_s42):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y = ar1_p050_n500_s42
        fig, ax = plt.subplots()
        out_ax = plot_pacf(y, lags=15, ax=ax)
        assert out_ax is ax
        plt.close(fig)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_invalid_pacf_method_raises(self):
        y = jnp.array([1.0, 2.0, 1.5, 0.5])
        with pytest.raises(ValueError, match="method"):
            pacf(y, lags=2, method="ols")


# ---------------------------------------------------------------------------
# Unit-root and stationarity tests (ADF + KPSS)
# ---------------------------------------------------------------------------
class TestUnitRoot:
    """ADF and KPSS, the two complementary unit-root tests.

    Coverage:
    * Cross-validation against ``statsmodels.tsa.stattools`` for both
      tests under all supported regression flavours.
    * Power: ADF rejects on a stationary AR(1); KPSS rejects on a
      random walk.  Both should fail to reject in the matched
      situation (KPSS on AR(1), ADF on random walk).
    * Edge cases: invalid ``regression`` arg raises ``ValueError``.
    """

    @pytest.fixture(scope="class")
    def smt(self):
        return require_oracle("statsmodels.tsa.stattools")

    @pytest.fixture(scope="class")
    def random_walk(self):
        key = jax.random.PRNGKey(1)
        return jnp.cumsum(jax.random.normal(key, (500,)))

    @pytest.mark.parametrize("regression", ["n", "c", "ct"])
    def test_adf_test_stat_matches_statsmodels(
        self,
        smt,
        ar1_p050_n500_s42,
        regression,
    ):
        r_cx = adf(ar1_p050_n500_s42, regression=regression, lags=12)
        r_sm = smt.adfuller(
            np.asarray(ar1_p050_n500_s42),
            regression=regression,
            autolag=None,
            maxlag=12,
        )
        np.testing.assert_allclose(
            float(r_cx["statistic"]),
            r_sm[0],
            rtol=1e-5,
        )
        # Critical values are from MacKinnon (1996) Table 1
        # (asymptotic).  ``statsmodels`` adds a finite-sample
        # polynomial correction (MacKinnon 1996 §3), so values agree
        # within ~0.02 — the comparison ``stat ⋛ crit`` at standard
        # levels is unaffected by this difference for any reasonable
        # sample size.
        # crit_values is a (3,) array aligned with ADF_CRIT_LEVELS =
        # (0.01, 0.05, 0.10), so index 0 is the 1% cutoff.
        np.testing.assert_allclose(
            float(r_cx["crit_values"][0]),
            r_sm[4]["1%"],
            atol=0.05,
        )

    @pytest.mark.parametrize("regression", ["c", "ct"])
    def test_kpss_test_stat_matches_statsmodels(
        self,
        smt,
        ar1_p050_n500_s42,
        regression,
    ):
        r_cx = kpss(ar1_p050_n500_s42, regression=regression, lags_choice="long")
        # Suppress statsmodels' InterpolationWarning when the stat is
        # outside the tabulated range — we don't compare p-values, so
        # the warning is irrelevant.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_sm = smt.kpss(
                np.asarray(ar1_p050_n500_s42),
                regression=regression,
                nlags="legacy",
            )
        np.testing.assert_allclose(
            float(r_cx["statistic"]),
            r_sm[0],
            rtol=1e-5,
        )

    def test_adf_rejects_stationary_ar1(self, ar1_p050_n500_s42):
        r = adf(ar1_p050_n500_s42, regression="c", lags=12)
        # AR(1) with phi=0.5 has a stationary root at z=2 — ADF should
        # decisively reject the unit-root null.  ADF_CRIT_LEVELS =
        # (0.01, 0.05, 0.10) ⇒ index 1 is the 5% cutoff.
        assert float(r["statistic"]) < float(r["crit_values"][1])
        assert float(r["p_value"]) < 0.05

    def test_adf_fails_to_reject_random_walk(self, random_walk):
        r = adf(random_walk, regression="c", lags=12)
        assert float(r["statistic"]) > float(r["crit_values"][1])
        assert float(r["p_value"]) > 0.05

    def test_kpss_rejects_random_walk(self, random_walk):
        r = kpss(random_walk, regression="c", lags_choice="long")
        # Random walk has a unit root — KPSS should reject the
        # stationarity null.  KPSS_CRIT_LEVELS = (0.10, 0.05, 0.025,
        # 0.01) ⇒ index 1 is the 5% cutoff.
        assert float(r["statistic"]) > float(r["crit_values"][1])
        assert float(r["p_value"]) < 0.05

    def test_kpss_fails_to_reject_stationary_ar1(self, ar1_p050_n500_s42):
        r = kpss(ar1_p050_n500_s42, regression="c", lags_choice="long")
        assert float(r["statistic"]) < float(r["crit_values"][1])
        assert float(r["p_value"]) > 0.05

    def test_unit_root_dict_schema(self, ar1_p050_n500_s42):
        """ADF / KPSS share the standardised hypothesis-test contract:
        ``statistic``, ``p_value``, ``used_lag``, ``n_obs``,
        ``crit_values``.  H0 / H1 statements live in the docstrings
        (the dicts stay pure-JAX so they round-trip through
        ``jax.jit``)."""
        common = {
            "statistic",
            "p_value",
            "used_lag",
            "n_obs",
            "crit_values",
        }
        for reg in ("n", "c", "ct"):
            r = adf(ar1_p050_n500_s42, regression=reg, lags=12)
            assert common <= set(r)
            # All scalar leaves are JAX arrays of the documented dtypes;
            # crit_values is a (3,) float array aligned with
            # ADF_CRIT_LEVELS = (0.01, 0.05, 0.10).
            assert r["statistic"].shape == ()
            assert r["p_value"].shape == ()
            assert r["used_lag"].dtype == jnp.int32
            assert r["n_obs"].dtype == jnp.int32
            assert r["crit_values"].shape == (3,)
        for reg in ("c", "ct"):
            r = kpss(ar1_p050_n500_s42, regression=reg, lags_choice="long")
            assert common <= set(r)
            assert r["statistic"].shape == ()
            assert r["p_value"].shape == ()
            assert r["used_lag"].dtype == jnp.int32
            assert r["n_obs"].dtype == jnp.int32
            assert r["crit_values"].shape == (4,)

    def test_invalid_regression_raises(self):
        y = jnp.arange(50, dtype=float)
        with pytest.raises(ValueError, match="regression"):
            adf(y, regression="foo")
        with pytest.raises(ValueError, match="regression"):
            kpss(y, regression="n")  # "n" only valid for ADF

    def test_kpss_docstring_documents_knot_limitation(self):
        """The public ``kpss`` API docstring must disclose the four-knot
        p-value interpolation limitation and the exactness of
        standard-level decisions.

        This is a user-honesty guard (no silent failures): a caller
        relying on an extreme KPSS ``p_value`` must be told from the
        docstring — the surface they actually read via ``help(kpss)`` —
        that the tail is extrapolated from only the four KPSS (1992)
        Table 1 knots.  Structural, not numerical: it pins the
        documentation, not the interpolation arithmetic (that contract
        lives in :class:`TestInterpP`)."""
        doc = kpss.__doc__
        assert doc is not None
        lowered = doc.lower()
        # (a) four-knot interpolation limit is named.
        assert "knot" in lowered
        assert "1992" in doc
        # (b) standard-level decisions are exact.
        assert "exact" in lowered
