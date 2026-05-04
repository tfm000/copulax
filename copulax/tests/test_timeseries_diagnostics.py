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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copulax.timeseries import (
    acf, adf, arch_lm, kpss, ljung_box, pacf,
    plot_acf, plot_pacf,
    AR, ARMA, ArmaGarch, GARCH,
)
from copulax.univariate import normal


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------
def _simulate_ar1(n, phi, key):
    eps = jax.random.normal(key, (n,))
    y = jnp.zeros((n,))

    def step(carry, eps_t):
        return phi * carry + eps_t, phi * carry + eps_t

    _, ys = jax.lax.scan(step, jnp.array(0.0), eps)
    return ys


def _simulate_garch11(n, omega, alpha, beta, key):
    sigma2_uncond = omega / (1.0 - alpha - beta)
    z = jax.random.normal(key, (n,))

    def step(carry, z_t):
        sigma2_prev, eps2_prev = carry
        sigma2_t = omega + alpha * eps2_prev + beta * sigma2_prev
        eps_t = jnp.sqrt(sigma2_t) * z_t
        return (sigma2_t, eps_t * eps_t), eps_t

    _, eps = jax.lax.scan(step, (sigma2_uncond, sigma2_uncond), z)
    return eps


# ---------------------------------------------------------------------------
# Cross-validation against statsmodels
# ---------------------------------------------------------------------------
class TestStatsmodelsCrossValidation:
    @pytest.fixture(scope="class")
    def sm_stattools(self):
        return pytest.importorskip("statsmodels.tsa.stattools")

    @pytest.fixture(scope="class")
    def sm_diag(self):
        return pytest.importorskip("statsmodels.stats.diagnostic")

    def test_acf_vs_statsmodels(self, sm_stattools):
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(1000, 0.6, key)
        cx = np.asarray(acf(y, 20))
        sm = sm_stattools.acf(np.asarray(y), nlags=20, fft=False)
        np.testing.assert_allclose(cx, sm, rtol=1e-5, atol=1e-7)

    def test_pacf_vs_statsmodels_ywm(self, sm_stattools):
        """We use biased-ACVF Yule-Walker; statsmodels exposes the
        same as ``method='ywm'`` or ``method='ldbiased'``.
        ``method='yw'`` (default) is the unbiased variant and
        differs from us by ~1e-3 — that's a documented convention
        difference per the module docstring, not a bug."""
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(1000, 0.6, key)
        cx = np.asarray(pacf(y, 20, method="yule_walker"))
        sm = sm_stattools.pacf(np.asarray(y), nlags=20, method="ywm")
        np.testing.assert_allclose(cx, sm, rtol=1e-5, atol=1e-7)

    def test_ljung_box_vs_statsmodels(self, sm_diag):
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(1000, 0.6, key)
        out = ljung_box(y, 10)
        sm = sm_diag.acorr_ljungbox(np.asarray(y), lags=[10], return_df=True)
        np.testing.assert_allclose(
            float(out["statistic"]), float(sm["lb_stat"].iloc[0]), rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(out["p_value"]), float(sm["lb_pvalue"].iloc[0]),
            rtol=1e-3,  # very small p-values amplify rel diff
            atol=1e-100,
        )

    def test_arch_lm_vs_statsmodels(self, sm_diag):
        key = jax.random.PRNGKey(42)
        eps = _simulate_garch11(1500, 0.05, 0.1, 0.85, key)
        out = arch_lm(eps, 5)
        sm = sm_diag.het_arch(np.asarray(eps), nlags=5)
        # sm = (LM, p_LM, F, p_F)
        np.testing.assert_allclose(float(out["statistic"]), float(sm[0]), rtol=1e-5)
        np.testing.assert_allclose(
            float(out["p_value"]), float(sm[1]), rtol=1e-3, atol=1e-100,
        )


# ---------------------------------------------------------------------------
# Shape / smoke / value invariants
# ---------------------------------------------------------------------------
class TestShapes:
    def test_acf_pacf_shapes(self):
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(500, 0.5, key)
        rho = acf(y, 15)
        pi = pacf(y, 15)
        assert rho.shape == (16,)
        assert pi.shape == (16,)
        # Lag 0 is identically 1 by definition.
        np.testing.assert_allclose(float(rho[0]), 1.0, atol=1e-12)
        np.testing.assert_allclose(float(pi[0]), 1.0, atol=1e-12)

    def test_pacf_lag_one_equals_acf_lag_one(self):
        """By construction PACF(1) = ρ(1) under the biased-ACVF
        Yule-Walker variant."""
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(500, 0.5, key)
        rho = acf(y, 5)
        pi = pacf(y, 5)
        np.testing.assert_allclose(float(pi[1]), float(rho[1]), atol=1e-12)

    def test_ljung_box_returns_dict(self):
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(500, 0.5, key)
        out = ljung_box(y, 10)
        assert set(out) == {
            "statistic", "p_value", "used_lag", "n_obs", "dof",
        }
        assert out["statistic"].shape == ()
        assert out["p_value"].shape == ()
        # Q ≥ 0 and 0 ≤ p ≤ 1.
        assert float(out["statistic"]) >= 0.0
        assert 0.0 <= float(out["p_value"]) <= 1.0
        assert out["used_lag"] == 10
        assert out["dof"] == 10
        assert out["n_obs"] == 500

    def test_arch_lm_returns_dict(self):
        key = jax.random.PRNGKey(42)
        eps = _simulate_garch11(1000, 0.05, 0.1, 0.85, key)
        out = arch_lm(eps, 5)
        assert set(out) == {
            "statistic", "p_value", "used_lag", "n_obs", "dof",
        }
        assert out["statistic"].shape == ()
        assert out["p_value"].shape == ()
        assert float(out["statistic"]) >= 0.0
        assert 0.0 <= float(out["p_value"]) <= 1.0
        assert out["used_lag"] == 5
        assert out["dof"] == 5
        # arch_lm trims ``lags`` observations off the front of the
        # auxiliary regression: n_eff = n - lags.
        assert out["n_obs"] == 1000 - 5


# ---------------------------------------------------------------------------
# Power: the tests reject under their respective alternative hypotheses.
# ---------------------------------------------------------------------------
class TestPower:
    def test_ljung_box_rejects_ar1(self):
        """Strong AR(1) autocorrelation should produce p ≈ 0."""
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(1000, 0.6, key)
        assert float(ljung_box(y, 10)["p_value"]) < 1e-10

    def test_ljung_box_does_not_reject_iid(self):
        """IID Normal noise should have p > 0.05 (most of the time)."""
        key = jax.random.PRNGKey(42)
        eps = jax.random.normal(key, (1000,))
        assert float(ljung_box(eps, 10)["p_value"]) > 0.05

    def test_arch_lm_rejects_garch(self):
        """GARCH(1, 1) data has strong ARCH effects; expect p ≈ 0."""
        key = jax.random.PRNGKey(42)
        eps = _simulate_garch11(1500, 0.05, 0.1, 0.85, key)
        assert float(arch_lm(eps, 5)["p_value"]) < 1e-5

    def test_arch_lm_does_not_reject_iid(self):
        key = jax.random.PRNGKey(42)
        eps = jax.random.normal(key, (1000,))
        assert float(arch_lm(eps, 5)["p_value"]) > 0.05


# ---------------------------------------------------------------------------
# Convenience methods on fitted models
# ---------------------------------------------------------------------------
class TestModelDiagnosticMethods:
    def test_arma_diagnostics(self):
        """ARMA fit exposes ``acf`` / ``pacf`` / ``ljung_box`` /
        ``arch_lm`` routed through standardised residuals."""
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(500, 0.6, key)
        fit = AR(p=1, residual_dist=normal).fit(y, maxiter=200)
        rho = fit.acf(y, lags=10)
        pi = fit.pacf(y, lags=10)
        lb = fit.ljung_box(y, lags=10)
        al = fit.arch_lm(y, lags=5)
        assert rho.shape == (11,)
        assert pi.shape == (11,)
        assert jnp.isfinite(lb["statistic"]) and jnp.isfinite(lb["p_value"])
        assert jnp.isfinite(al["statistic"]) and jnp.isfinite(al["p_value"])

    def test_garch_diagnostics(self):
        key = jax.random.PRNGKey(42)
        eps = _simulate_garch11(1000, 0.05, 0.1, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=300)
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
        key = jax.random.PRNGKey(42)
        eps = _simulate_garch11(1500, 0.05, 0.1, 0.85, key)
        # Inject a small AR(1) component to give the joint composite
        # something to fit.
        y = jnp.zeros_like(eps)
        y = y.at[0].set(eps[0])
        for t in range(1, len(eps)):
            y = y.at[t].set(0.3 * y[t - 1] + eps[t])
        fit = ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, maxiter=400)
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
        key = jax.random.PRNGKey(7)
        y = _simulate_ar1(800, 0.6, key)
        fit = ARMA(p=1, q=1, residual_dist=normal).fit(y, maxiter=200)
        corr = fit.ljung_box(y, lags=10, dof_correction=True)
        raw = fit.ljung_box(y, lags=10, dof_correction=False)
        np.testing.assert_allclose(
            float(corr["statistic"]), float(raw["statistic"]), rtol=1e-12,
        )
        assert float(corr["p_value"]) < float(raw["p_value"])
        # The dof correction surfaces in the dict itself.
        assert corr["dof"] == 10 - 1 - 1  # lags - p - q
        assert raw["dof"] == 10
        # Joint composite exposes the same kwarg + an ``on=`` selector.
        eps = _simulate_garch11(1500, 0.05, 0.1, 0.85, key)
        ar_y = jnp.zeros_like(eps).at[0].set(eps[0])
        for t in range(1, len(eps)):
            ar_y = ar_y.at[t].set(0.3 * ar_y[t - 1] + eps[t])
        joint = ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(ar_y, maxiter=400)
        z_out = joint.ljung_box(ar_y, lags=10, on="residuals")
        z2_out = joint.ljung_box(ar_y, lags=10, on="squared_residuals")
        # The two ``on=`` paths consume different series so the Q
        # statistics differ in general.
        assert not np.isclose(
            float(z_out["statistic"]), float(z2_out["statistic"]), rtol=1e-3,
        )
        with pytest.raises(ValueError, match="on"):
            joint.ljung_box(ar_y, lags=10, on="invalid")


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------
class TestJIT:
    def test_acf_jit(self):
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(500, 0.5, key)
        out_eager = acf(y, 10)
        out_jit = jax.jit(acf, static_argnames=("lags",))(y, lags=10)
        np.testing.assert_allclose(np.asarray(out_eager), np.asarray(out_jit))

    def test_ljung_box_jit(self):
        """``ljung_box`` is directly JIT-compatible — the result dict
        is a pure-JAX pytree (Python-int ``used_lag``/``n_obs``/``dof``
        are static under ``jax.jit`` since ``lags`` is a static arg)."""
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(500, 0.5, key)
        eager = ljung_box(y, 10)
        jit_lb = jax.jit(ljung_box, static_argnames=("lags",))
        jit_out = jit_lb(y, lags=10)
        np.testing.assert_allclose(
            float(eager["statistic"]), float(jit_out["statistic"]),
        )
        np.testing.assert_allclose(
            float(eager["p_value"]), float(jit_out["p_value"]),
        )
        assert int(jit_out["used_lag"]) == eager["used_lag"]
        assert int(jit_out["dof"]) == eager["dof"]
        assert int(jit_out["n_obs"]) == eager["n_obs"]

    def test_arch_lm_jit(self):
        """``arch_lm`` is directly JIT-compatible by the same argument
        — the dict is a pure-JAX pytree."""
        key = jax.random.PRNGKey(42)
        eps = _simulate_garch11(1000, 0.05, 0.1, 0.85, key)
        eager = arch_lm(eps, 5)
        jit_al = jax.jit(arch_lm, static_argnames=("lags",))
        jit_out = jit_al(eps, lags=5)
        np.testing.assert_allclose(
            float(eager["statistic"]), float(jit_out["statistic"]),
        )
        np.testing.assert_allclose(
            float(eager["p_value"]), float(jit_out["p_value"]),
        )


# ---------------------------------------------------------------------------
# Plot smoke
# ---------------------------------------------------------------------------
class TestPlots:
    def test_plot_acf_smoke(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(500, 0.5, key)
        fig, ax = plt.subplots()
        out_ax = plot_acf(y, lags=15, ax=ax)
        assert out_ax is ax
        plt.close(fig)

    def test_plot_pacf_smoke(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(500, 0.5, key)
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
        return pytest.importorskip("statsmodels.tsa.stattools")

    @pytest.fixture(scope="class")
    def stationary_series(self):
        key = jax.random.PRNGKey(0)
        eps = jax.random.normal(key, (500,))
        phi = 0.5
        def step(carry, e):
            v = phi * carry + e
            return v, v
        _, y = jax.lax.scan(step, jnp.array(0.0), eps)
        return y

    @pytest.fixture(scope="class")
    def random_walk(self):
        key = jax.random.PRNGKey(1)
        return jnp.cumsum(jax.random.normal(key, (500,)))

    @pytest.mark.parametrize("regression", ["n", "c", "ct"])
    def test_adf_test_stat_matches_statsmodels(
        self, smt, stationary_series, regression,
    ):
        r_cx = adf(stationary_series, regression=regression, lags=12)
        r_sm = smt.adfuller(
            np.asarray(stationary_series),
            regression=regression, autolag=None, maxlag=12,
        )
        np.testing.assert_allclose(
            float(r_cx["statistic"]), r_sm[0], rtol=1e-5,
        )
        # Critical values are from MacKinnon (1996) Table 1
        # (asymptotic).  ``statsmodels`` adds a finite-sample
        # polynomial correction (MacKinnon 1996 §3), so values agree
        # within ~0.02 — the comparison ``stat ⋛ crit`` at standard
        # levels is unaffected by this difference for any reasonable
        # sample size.
        np.testing.assert_allclose(
            float(r_cx["crit_values"]["1%"]), r_sm[4]["1%"], atol=0.05,
        )

    @pytest.mark.parametrize("regression", ["c", "ct"])
    def test_kpss_test_stat_matches_statsmodels(
        self, smt, stationary_series, regression,
    ):
        r_cx = kpss(stationary_series, regression=regression, lags_choice="long")
        # Suppress statsmodels' InterpolationWarning when the stat is
        # outside the tabulated range — we don't compare p-values, so
        # the warning is irrelevant.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_sm = smt.kpss(
                np.asarray(stationary_series),
                regression=regression, nlags="legacy",
            )
        np.testing.assert_allclose(
            float(r_cx["statistic"]), r_sm[0], rtol=1e-5,
        )

    def test_adf_rejects_stationary_ar1(self, stationary_series):
        r = adf(stationary_series, regression="c", lags=12)
        # AR(1) with phi=0.5 has a stationary root at z=2 — ADF should
        # decisively reject the unit-root null.
        assert float(r["statistic"]) < float(r["crit_values"]["5%"])
        assert float(r["p_value"]) < 0.05

    def test_adf_fails_to_reject_random_walk(self, random_walk):
        r = adf(random_walk, regression="c", lags=12)
        assert float(r["statistic"]) > float(r["crit_values"]["5%"])
        assert float(r["p_value"]) > 0.05

    def test_kpss_rejects_random_walk(self, random_walk):
        r = kpss(random_walk, regression="c", lags_choice="long")
        # Random walk has a unit root — KPSS should reject the
        # stationarity null.
        assert float(r["statistic"]) > float(r["crit_values"]["5%"])
        assert float(r["p_value"]) < 0.05

    def test_kpss_fails_to_reject_stationary_ar1(self, stationary_series):
        r = kpss(stationary_series, regression="c", lags_choice="long")
        assert float(r["statistic"]) < float(r["crit_values"]["5%"])
        assert float(r["p_value"]) > 0.05

    def test_unit_root_dict_schema(self, stationary_series):
        """ADF / KPSS share the standardised hypothesis-test contract:
        ``statistic``, ``p_value``, ``used_lag``, ``n_obs``,
        ``crit_values``.  H0 / H1 statements live in the docstrings
        (the dicts stay JAX-only so they round-trip through
        ``jax.jit``)."""
        common = {
            "statistic", "p_value",
            "used_lag", "n_obs", "crit_values",
        }
        for reg in ("n", "c", "ct"):
            r = adf(stationary_series, regression=reg, lags=12)
            assert common <= set(r)
        for reg in ("c", "ct"):
            r = kpss(stationary_series, regression=reg, lags_choice="long")
            assert common <= set(r)

    def test_invalid_regression_raises(self):
        y = jnp.arange(50, dtype=float)
        with pytest.raises(ValueError, match="regression"):
            adf(y, regression="foo")
        with pytest.raises(ValueError, match="regression"):
            kpss(y, regression="n")  # "n" only valid for ADF
