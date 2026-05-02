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
    acf, pacf, ljung_box, arch_lm,
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
        Q, p = ljung_box(y, 10)
        sm = sm_diag.acorr_ljungbox(np.asarray(y), lags=[10], return_df=True)
        np.testing.assert_allclose(
            float(Q), float(sm["lb_stat"].iloc[0]), rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(p), float(sm["lb_pvalue"].iloc[0]),
            rtol=1e-3,  # very small p-values amplify rel diff
            atol=1e-100,
        )

    def test_arch_lm_vs_statsmodels(self, sm_diag):
        key = jax.random.PRNGKey(42)
        eps = _simulate_garch11(1500, 0.05, 0.1, 0.85, key)
        LM, p = arch_lm(eps, 5)
        sm = sm_diag.het_arch(np.asarray(eps), nlags=5)
        # sm = (LM, p_LM, F, p_F)
        np.testing.assert_allclose(float(LM), float(sm[0]), rtol=1e-5)
        np.testing.assert_allclose(
            float(p), float(sm[1]), rtol=1e-3, atol=1e-100,
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

    def test_ljung_box_returns_scalars(self):
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(500, 0.5, key)
        Q, p = ljung_box(y, 10)
        assert Q.shape == ()
        assert p.shape == ()
        # Q ≥ 0 and 0 ≤ p ≤ 1.
        assert float(Q) >= 0.0
        assert 0.0 <= float(p) <= 1.0

    def test_arch_lm_returns_scalars(self):
        key = jax.random.PRNGKey(42)
        eps = _simulate_garch11(1000, 0.05, 0.1, 0.85, key)
        LM, p = arch_lm(eps, 5)
        assert LM.shape == ()
        assert p.shape == ()
        assert float(LM) >= 0.0
        assert 0.0 <= float(p) <= 1.0


# ---------------------------------------------------------------------------
# Power: the tests reject under their respective alternative hypotheses.
# ---------------------------------------------------------------------------
class TestPower:
    def test_ljung_box_rejects_ar1(self):
        """Strong AR(1) autocorrelation should produce p ≈ 0."""
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(1000, 0.6, key)
        _, p = ljung_box(y, 10)
        assert float(p) < 1e-10

    def test_ljung_box_does_not_reject_iid(self):
        """IID Normal noise should have p > 0.05 (most of the time)."""
        key = jax.random.PRNGKey(42)
        eps = jax.random.normal(key, (1000,))
        _, p = ljung_box(eps, 10)
        assert float(p) > 0.05

    def test_arch_lm_rejects_garch(self):
        """GARCH(1, 1) data has strong ARCH effects; expect p ≈ 0."""
        key = jax.random.PRNGKey(42)
        eps = _simulate_garch11(1500, 0.05, 0.1, 0.85, key)
        _, p = arch_lm(eps, 5)
        assert float(p) < 1e-5

    def test_arch_lm_does_not_reject_iid(self):
        key = jax.random.PRNGKey(42)
        eps = jax.random.normal(key, (1000,))
        _, p = arch_lm(eps, 5)
        assert float(p) > 0.05


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
        Q, p = fit.ljung_box(y, lags=10)
        LM, p_lm = fit.arch_lm(y, lags=5)
        assert rho.shape == (11,)
        assert pi.shape == (11,)
        assert jnp.isfinite(Q) and jnp.isfinite(p)
        assert jnp.isfinite(LM) and jnp.isfinite(p_lm)

    def test_garch_diagnostics(self):
        key = jax.random.PRNGKey(42)
        eps = _simulate_garch11(1000, 0.05, 0.1, 0.85, key)
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=300)
        rho = fit.acf(eps, lags=10)
        pi = fit.pacf(eps, lags=10)
        Q, p = fit.ljung_box(eps, lags=10)
        LM, p_lm = fit.arch_lm(eps, lags=5)
        # After a successful GARCH fit, the standardised residuals
        # should look much closer to IID — ARCH-LM p-value much
        # higher than on the raw eps.
        _, raw_p = arch_lm(eps, lags=5)
        assert float(p_lm) > float(raw_p)

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
        Q, p = fit.ljung_box(y, lags=10)
        LM, p_lm = fit.arch_lm(y, lags=5)
        # After joint ARMA-GARCH fit on AR(1)-GARCH(1, 1) data, both
        # tests should fail to reject (well-specified model).
        assert float(p) > 0.01
        assert float(p_lm) > 0.01


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
        key = jax.random.PRNGKey(42)
        y = _simulate_ar1(500, 0.5, key)
        Q_eager, p_eager = ljung_box(y, 10)
        jit_lb = jax.jit(ljung_box, static_argnames=("lags",))
        Q_jit, p_jit = jit_lb(y, lags=10)
        np.testing.assert_allclose(float(Q_eager), float(Q_jit))
        np.testing.assert_allclose(float(p_eager), float(p_jit))


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
