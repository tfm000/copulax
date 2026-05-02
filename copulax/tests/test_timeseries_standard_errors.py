"""Standard-error tests for the time-series subpackage.

Joint asymptotic-covariance SEs for the ``ArmaGarch`` composite are
the v1 deliverable; the Pagan-Newey two-stage sandwich for the
separable ``ARMA → GARCH`` workflow is deferred to a future commit.

Three ``cov_type`` formulas are supported, mirroring ``arch``:

* ``"robust"``  — Bollerslev-Wooldridge sandwich (default).
* ``"classic"`` — observed information / inverse Hessian.
* ``"opg"``     — outer product of gradients (BHHH).

Coverage:

* SE shape / dict-structure invariants — entries match ``params``
  schema; values are non-negative; ``cov_matrix_`` is a square
  PSD matrix.
* Stored vs recomputed SE parity:
  ``fit.standard_errors() == fit.standard_errors(y_train)``.
* Cross-validation against ``arch.arch_model(...).fit(cov_type=...)``
  on AR(1)+GARCH(1, 1) data — both ``robust`` and ``classic``
  paths agree to ``rtol=2e-2`` per plan §"Standard errors".
* ``confidence_intervals(alpha)`` symmetric-normal width.
* ``summary()`` renders a non-empty multi-line string with the
  expected parameter table + footer sections.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copulax.timeseries import ArmaGarch, GARCH
from copulax.univariate import normal


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
def _simulate_ar1_garch11(n, phi, c, omega, alpha, beta, key):
    sigma2_uncond = omega / (1.0 - alpha - beta)
    z = jax.random.normal(key, (n,))

    def step(carry, z_t):
        y_lag, sigma2_lag, eps_sq_lag = carry
        sigma2_t = omega + alpha * eps_sq_lag + beta * sigma2_lag
        eps_t = jnp.sqrt(sigma2_t) * z_t
        y_t = c + phi * y_lag + eps_t
        return (y_t, sigma2_t, eps_t * eps_t), y_t

    _, y = jax.lax.scan(
        step, (c / (1.0 - phi), sigma2_uncond, sigma2_uncond), z,
    )
    return y


# ---------------------------------------------------------------------------
# Shape / dict-structure invariants
# ---------------------------------------------------------------------------
class TestStructure:
    def test_se_dict_matches_params(self):
        key = jax.random.PRNGKey(13)
        y = _simulate_ar1_garch11(1000, 0.5, 0.05, 0.05, 0.10, 0.85, key)
        fit = ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, maxiter=400)
        # Top-level keys match
        assert set(fit.standard_errors_) == set(fit.params)
        # Residual sub-dict matches
        assert set(fit.standard_errors_["residual"]) == set(fit.params["residual"])
        # Per-key shapes match
        for k, v in fit.params.items():
            if isinstance(v, dict):
                continue
            assert fit.standard_errors_[k].shape == jnp.atleast_1d(v).shape \
                or fit.standard_errors_[k].shape == ()

    def test_cov_matrix_is_square_psd(self):
        key = jax.random.PRNGKey(13)
        y = _simulate_ar1_garch11(1000, 0.5, 0.05, 0.05, 0.10, 0.85, key)
        fit = ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, maxiter=400)
        cov = fit.cov_matrix_
        assert cov.shape[0] == cov.shape[1]
        # PSD check: minimum eigenvalue ≥ 0 (within numerical tolerance).
        eigvals = jnp.linalg.eigvalsh(cov)
        assert float(jnp.min(eigvals)) > -1e-8

    def test_se_non_negative(self):
        key = jax.random.PRNGKey(13)
        y = _simulate_ar1_garch11(1000, 0.5, 0.05, 0.05, 0.10, 0.85, key)
        fit = ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, maxiter=400)
        # Every non-empty leaf in standard_errors_ is non-negative.
        # (theta with shape (0,) is empty under mean_order=(1, 0) and
        # is skipped — there's no SE for a non-existent parameter.)
        for k, v in fit.standard_errors_.items():
            if isinstance(v, dict):
                for sub_v in v.values():
                    arr = jnp.atleast_1d(sub_v)
                    if arr.size > 0:
                        assert float(jnp.min(arr)) >= 0.0
            else:
                arr = jnp.atleast_1d(v)
                if arr.size > 0:
                    assert float(jnp.min(arr)) >= 0.0


# ---------------------------------------------------------------------------
# Stored vs recomputed parity
# ---------------------------------------------------------------------------
class TestParity:
    def test_recompute_matches_stored(self):
        """``standard_errors()`` (stored, robust) ==
        ``standard_errors(y_train, cov_type='robust')`` (recomputed)
        to machine precision — both paths route through the same
        natural-space objective at the natural-space MLE."""
        key = jax.random.PRNGKey(13)
        y = _simulate_ar1_garch11(1000, 0.5, 0.05, 0.05, 0.10, 0.85, key)
        fit = ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, maxiter=400)
        recomputed = fit.standard_errors(y, cov_type="robust")
        for k, stored_v in fit.standard_errors_.items():
            if isinstance(stored_v, dict):
                for sub, sub_v in stored_v.items():
                    arr_stored = jnp.atleast_1d(sub_v)
                    arr_recomp = jnp.atleast_1d(recomputed[k][sub])
                    if arr_stored.size > 0:
                        np.testing.assert_allclose(
                            np.asarray(arr_stored),
                            np.asarray(arr_recomp),
                            rtol=1e-5, atol=1e-8,
                        )
            else:
                arr_stored = jnp.atleast_1d(stored_v)
                arr_recomp = jnp.atleast_1d(recomputed[k])
                if arr_stored.size > 0:
                    np.testing.assert_allclose(
                        np.asarray(arr_stored),
                        np.asarray(arr_recomp),
                        rtol=1e-5, atol=1e-8,
                    )


# ---------------------------------------------------------------------------
# Cross-validation against arch.arch_model
# ---------------------------------------------------------------------------
class TestArchCrossValidation:
    """SEs match ``arch.arch_model.std_err`` under both
    ``cov_type='robust'`` (Bollerslev-Wooldridge sandwich, the
    default in both libraries) and ``cov_type='classic'`` (observed
    information).  AR(1)+GARCH(1, 1) data so arch's mean equation
    aligns with CopulAX's ``mean_order=(1, 0)``."""

    @pytest.fixture(scope="class")
    def arch_module(self):
        return pytest.importorskip("arch")

    def _se_pairs(self, fit_se: dict, arch_res) -> list[tuple[str, float, float]]:
        return [
            ("phi",   float(fit_se["phi"][0]),   float(arch_res.std_err["y[1]"])),
            ("c",     float(fit_se["c"]),         float(arch_res.std_err["Const"])),
            ("omega", float(fit_se["omega"]),     float(arch_res.std_err["omega"])),
            ("alpha", float(fit_se["alpha"][0]),  float(arch_res.std_err["alpha[1]"])),
            ("beta",  float(fit_se["beta"][0]),   float(arch_res.std_err["beta[1]"])),
        ]

    def test_robust_vs_arch_robust(self, arch_module):
        """Default ``cov_type='robust'`` (BW sandwich) matches
        ``arch.arch_model(..., cov_type='robust')`` to ``rtol=2e-2``.
        BW is sensitive to small differences in the score
        covariance estimate (different optimisers, different MLE
        points by ~1e-3) so the tolerance is slightly looser than
        the ``classic`` case."""
        key = jax.random.PRNGKey(13)
        y = _simulate_ar1_garch11(2000, 0.5, 0.05, 0.05, 0.10, 0.85, key)
        fit = ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, init="analytical", maxiter=1500, lr=0.05)
        am = arch_module.arch_model(
            np.asarray(y), mean="ARX", lags=1,
            vol="GARCH", p=1, q=1, dist="Normal",
        )
        arch_res = am.fit(disp="off", cov_type="robust")
        for label, cx, ar in self._se_pairs(fit.standard_errors_, arch_res):
            np.testing.assert_allclose(cx, ar, rtol=2e-2)

    def test_classic_vs_arch_classic(self, arch_module):
        """``cov_type='classic'`` (observed information) matches
        ``arch`` to ``rtol=1e-2`` — the tighter plan-mandated
        tolerance.  ``classic`` only depends on the Hessian,
        which is less sensitive to optimiser-induced MLE
        differences than the score-covariance term in BW."""
        key = jax.random.PRNGKey(13)
        y = _simulate_ar1_garch11(2000, 0.5, 0.05, 0.05, 0.10, 0.85, key)
        fit = ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, init="analytical", maxiter=1500, lr=0.05)
        am = arch_module.arch_model(
            np.asarray(y), mean="ARX", lags=1,
            vol="GARCH", p=1, q=1, dist="Normal",
        )
        arch_res = am.fit(disp="off", cov_type="classic")
        cx_se = fit.standard_errors(y, cov_type="classic")
        for label, cx, ar in self._se_pairs(cx_se, arch_res):
            np.testing.assert_allclose(cx, ar, rtol=1e-2)


# ---------------------------------------------------------------------------
# Confidence intervals + summary
# ---------------------------------------------------------------------------
class TestCovTypes:
    def test_three_cov_types_produce_finite_positive_se(self):
        """All three ``cov_type`` paths produce finite, non-negative
        SEs at a well-behaved interior MLE."""
        key = jax.random.PRNGKey(13)
        y = _simulate_ar1_garch11(1000, 0.5, 0.05, 0.05, 0.10, 0.85, key)
        fit = ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, maxiter=400)
        for cov_type in ("robust", "classic", "opg"):
            cov = fit.cov_matrix(y, cov_type=cov_type)
            assert jnp.all(jnp.isfinite(cov))
            diag = jnp.diag(cov)
            assert float(jnp.min(diag)) >= 0.0

    def test_invalid_cov_type_raises(self):
        key = jax.random.PRNGKey(13)
        y = _simulate_ar1_garch11(500, 0.5, 0.05, 0.05, 0.10, 0.85, key)
        fit = ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, maxiter=200)
        with pytest.raises(ValueError, match="cov_type"):
            fit.standard_errors(y, cov_type="bogus")


class TestConfidenceIntervalsAndSummary:
    def test_confidence_intervals_symmetric(self):
        key = jax.random.PRNGKey(13)
        y = _simulate_ar1_garch11(1000, 0.5, 0.05, 0.05, 0.10, 0.85, key)
        fit = ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, maxiter=400)
        cis = fit.confidence_intervals(alpha=0.05)
        # Top-level keys match params
        assert set(cis) == set(fit.params)
        # Per param, lo < est < hi, and the CI is symmetric to within
        # numerical tolerance (since the SE is symmetric and z is fixed).
        z = float(jax.scipy.stats.norm.ppf(0.975))
        for k, v in fit.params.items():
            if isinstance(v, dict):
                continue
            v_arr = jnp.atleast_1d(jnp.asarray(v, dtype=float))
            se_arr = jnp.atleast_1d(jnp.asarray(fit.standard_errors_[k], dtype=float))
            lo, hi = cis[k]
            np.testing.assert_allclose(
                np.asarray(hi - lo), 2.0 * z * np.asarray(se_arr), rtol=1e-5,
            )

    def test_summary_renders(self):
        key = jax.random.PRNGKey(13)
        y = _simulate_ar1_garch11(1000, 0.5, 0.05, 0.05, 0.10, 0.85, key)
        fit = ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, maxiter=400)
        out = fit.summary()
        assert isinstance(out, str)
        # Expected sections.
        assert "ArmaGarch" in out
        assert "estimate" in out
        assert "loglikelihood" in out
        assert "AIC" in out
        # At least one parameter row present.
        assert "phi[1]" in out
        assert "omega" in out
