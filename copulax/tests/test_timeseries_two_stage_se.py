"""Pagan-Newey two-stage standard-error tests.

The two-stage workflow fits ARMA on the level series, then a
GARCH-family variance model on ``arma_fit.residuals(y)``.  The
naive variance-stage covariance ``J_22^{-1} S_22 J_22^{-1}``
ignores the noise from the first-stage estimate;
:func:`pagan_newey_cov` corrects this via the cross-stage Hessian
:math:`J_{21}` and the first-stage score :math:`s_1`.

These tests verify:

* Output shape matches the variance-stage flat parameter vector.
* The corrected SE dict matches the variance fit's ``params``
  schema.
* Specifying an ARMA(0, 0) (no first-stage parameters except
  intercept / scale) reproduces the naive plug-in covariance.
* The correction is symmetric / positive-semidefinite.
* The corrected covariance is asymptotically close to the joint
  ArmaGarch sandwich on the same data — they target the same
  asymptotic variance under correct specification.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copulax.timeseries import (
    ARMA,
    ArmaGarch,
    GARCH,
    GJR_GARCH,
    two_stage_cov,
    two_stage_standard_errors,
)
from copulax._src.timeseries._two_stage_se import _build_two_stage_closures
from copulax.univariate import normal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def y_series():
    key = jax.random.PRNGKey(13)
    return jax.random.normal(key, (500,)) * 0.1


@pytest.fixture(scope="module")
def arma_fit(y_series):
    return ARMA(p=1, q=1, residual_dist=normal).fit(y_series, maxiter=120)


@pytest.fixture(scope="module")
def garch_fit(arma_fit, y_series):
    eps = arma_fit.residuals(y_series)
    return GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=120)


# ---------------------------------------------------------------------------
# Shape / schema invariants
# ---------------------------------------------------------------------------
class TestShape:
    def test_cov_is_square_with_garch_n_params(
        self, arma_fit, garch_fit, y_series,
    ):
        cov = two_stage_cov(arma_fit, garch_fit, y_series)
        # GARCH(1,1) + Normal residual = 3 natural params (omega, alpha, beta).
        assert cov.shape == (3, 3)

    def test_cov_is_symmetric(self, arma_fit, garch_fit, y_series):
        cov = np.asarray(two_stage_cov(arma_fit, garch_fit, y_series))
        np.testing.assert_allclose(cov, cov.T, rtol=1e-8, atol=1e-12)

    def test_cov_diagonal_is_nonneg(self, arma_fit, garch_fit, y_series):
        cov = np.asarray(two_stage_cov(arma_fit, garch_fit, y_series))
        assert np.all(np.diag(cov) >= -1e-12)

    def test_se_dict_matches_garch_param_schema(
        self, arma_fit, garch_fit, y_series,
    ):
        se = two_stage_standard_errors(arma_fit, garch_fit, y_series)
        # Same top-level keys as the GARCH params dict.
        assert set(se.keys()) == set(garch_fit.params.keys())
        for key in ("omega", "alpha", "beta"):
            assert key in se
        # Shapes match.
        np.testing.assert_array_equal(
            jnp.shape(se["omega"]), jnp.shape(garch_fit.params["omega"]),
        )
        np.testing.assert_array_equal(
            jnp.shape(se["alpha"]), jnp.shape(garch_fit.params["alpha"]),
        )
        np.testing.assert_array_equal(
            jnp.shape(se["beta"]), jnp.shape(garch_fit.params["beta"]),
        )


# ---------------------------------------------------------------------------
# Correctness — formula identities
# ---------------------------------------------------------------------------
class TestFormula:
    def test_zero_cross_hessian_reduces_to_naive(
        self, arma_fit, garch_fit, y_series,
    ):
        """When :math:`J_{21}\\equiv 0`, the PN correction collapses
        to :math:`J_{22}^{-1} \\mathrm{Cov}(s_2)\\, J_{22}^{-1} / n`
        — exactly the naive plug-in.  We patch the joint NLL so it
        is identically constant in :math:`\\theta_1`, forcing
        :math:`J_{21}=0`, and verify the result equals the
        independently-computed naive sandwich on the variance
        stage."""
        from copulax._src.timeseries._se import (
            compute_param_cov,
            pagan_newey_cov,
            params_to_flat,
        )

        y_arr = arma_fit._validate_series(y_series)
        n = int(y_arr.shape[0])

        (
            nll1_total,
            per_obs_nll1,
            nll2_joint,
            per_obs_nll2_joint,
            _schemas,
            (p1_flat, p2_flat),
        ) = _build_two_stage_closures(
            arma_fit, garch_fit, y_arr,
            arma_init="backcast", arma_backcast_length=None,
            var_init="backcast", var_backcast_length=None,
        )

        # Patch stage-2 closures so they ignore ``p1_flat`` entirely.
        def nll2_zero_cross(p1, p2):
            return nll2_joint(p1_flat, p2)

        def per_obs_zero_cross(p1, p2):
            return per_obs_nll2_joint(p1_flat, p2)

        pn_cov = pagan_newey_cov(
            nll1_total=nll1_total, per_obs_nll1=per_obs_nll1,
            nll2_total_joint=nll2_zero_cross,
            per_obs_nll2_joint=per_obs_zero_cross,
            params1_flat=p1_flat, params2_flat=p2_flat, n_obs=n,
        )
        # Naive sandwich on the GARCH stage alone.
        naive_cov = compute_param_cov(
            nll_total=lambda p2: nll2_joint(p1_flat, p2),
            per_obs_nll=lambda p2: per_obs_nll2_joint(p1_flat, p2),
            params_flat=p2_flat, n_obs=n, cov_type="robust",
        )
        np.testing.assert_allclose(
            np.asarray(pn_cov), np.asarray(naive_cov),
            rtol=1e-6, atol=1e-12,
        )

    def test_nontrivial_cross_hessian_changes_cov(
        self, arma_fit, garch_fit, y_series,
    ):
        """With non-zero ARMA dynamics, PN cov differs from the
        naive plug-in.  This guards against a silent regression
        where :math:`J_{21}` evaluates to zero by construction."""
        from copulax._src.timeseries._se import (
            compute_param_cov,
            params_to_flat,
        )
        from copulax._src.timeseries._two_stage_se import (
            _build_two_stage_closures,
        )

        y_arr = arma_fit._validate_series(y_series)
        n = int(y_arr.shape[0])

        (
            _,
            _,
            nll2_joint,
            per_obs_nll2_joint,
            _schemas,
            (p1_flat, p2_flat),
        ) = _build_two_stage_closures(
            arma_fit, garch_fit, y_arr,
            arma_init="backcast", arma_backcast_length=None,
            var_init="backcast", var_backcast_length=None,
        )
        pn_cov = np.asarray(
            two_stage_cov(arma_fit, garch_fit, y_series)
        )
        naive_cov = np.asarray(compute_param_cov(
            nll_total=lambda p2: nll2_joint(p1_flat, p2),
            per_obs_nll=lambda p2: per_obs_nll2_joint(p1_flat, p2),
            params_flat=p2_flat, n_obs=n, cov_type="robust",
        ))
        # Diagonal should differ — meaning the cross-stage adjustment
        # had a measurable effect.
        assert not np.allclose(
            np.diag(pn_cov), np.diag(naive_cov), rtol=0.0, atol=1e-10,
        )


# ---------------------------------------------------------------------------
# API ergonomics
# ---------------------------------------------------------------------------
class TestAPI:
    def test_unfitted_arma_raises(self, garch_fit, y_series):
        unfitted = ARMA(p=1, q=1, residual_dist=normal)
        with pytest.raises(ValueError, match="arma_fit must be"):
            two_stage_cov(unfitted, garch_fit, y_series)

    def test_unfitted_var_raises(self, arma_fit, y_series):
        unfitted = GARCH(p=1, q=1, residual_dist=normal)
        with pytest.raises(ValueError, match="var_fit must be"):
            two_stage_cov(arma_fit, unfitted, y_series)

    def test_works_with_gjr_garch(self, arma_fit, y_series):
        eps = arma_fit.residuals(y_series)
        gjr = GJR_GARCH(p=1, q=1, residual_dist=normal).fit(
            eps, maxiter=80,
        )
        cov = np.asarray(two_stage_cov(arma_fit, gjr, y_series))
        # GJR(1,1) + Normal = omega + alpha + gamma + beta = 4 params.
        assert cov.shape == (4, 4)
        np.testing.assert_allclose(cov, cov.T, rtol=1e-8, atol=1e-12)


# ---------------------------------------------------------------------------
# Asymptotic agreement with joint MLE (loose tolerance)
# ---------------------------------------------------------------------------
class TestAsymptoticAgreement:
    def test_pn_se_within_factor_of_joint_se(self):
        """On a long series, PN SEs and joint MLE SEs should be in
        the same ballpark (asymptotically equivalent under correct
        specification).  Tolerance is loose because finite-sample
        noise + different optimisation paths cause real divergence
        — we just guard against orders-of-magnitude bugs."""
        key = jax.random.PRNGKey(0)
        # Simulate from a known ARMA(1)+GARCH(1,1) DGP.
        n = 4000
        phi_t, omega_t, alpha_t, beta_t = 0.3, 0.05, 0.1, 0.85
        z = jax.random.normal(key, (n,))

        def step(carry, z_t):
            y_prev, sigma2_prev, eps2_prev = carry
            sigma2_t = omega_t + alpha_t * eps2_prev + beta_t * sigma2_prev
            eps_t = jnp.sqrt(sigma2_t) * z_t
            y_t = phi_t * y_prev + eps_t
            return (y_t, sigma2_t, eps_t * eps_t), y_t

        sigma2_uncond = omega_t / (1.0 - alpha_t - beta_t)
        _, y_sim = jax.lax.scan(
            step, (0.0, sigma2_uncond, sigma2_uncond), z,
        )

        arma = ARMA(p=1, q=0, residual_dist=normal).fit(y_sim, maxiter=200)
        eps = arma.residuals(y_sim)
        garch = GARCH(p=1, q=1, residual_dist=normal).fit(eps, maxiter=300)
        joint = ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y_sim, maxiter=400)

        pn_se = two_stage_standard_errors(arma, garch, y_sim)
        joint_se = joint.standard_errors_

        for key in ("omega", "alpha", "beta"):
            pn_v = float(jnp.asarray(pn_se[key]).reshape(()))
            joint_v = float(jnp.asarray(joint_se[key]).reshape(()))
            ratio = pn_v / joint_v
            # PN and joint should be within an order of magnitude.
            assert 0.2 < ratio < 5.0, (
                f"{key}: PN SE {pn_v:.4g} vs joint SE {joint_v:.4g} "
                f"(ratio {ratio:.2f}) — should be similar magnitude."
            )
