"""Rigorous tests for the 6 Archimedean copulas.

Verifies generator identities, tau-to-theta round-trips, Frechet bounds,
marginal uniformity, and density properties.

Catches FINDING-06-01: Frank tau-to-theta returns 0 for most tau values.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from copulax.copulas import (
    clayton_copula, frank_copula, gumbel_copula,
    joe_copula, amh_copula, independence_copula,
)
from copulax.tests_new.conftest import no_nans


# Copulas that support d >= 3
COPULAS_3D = [clayton_copula, frank_copula, gumbel_copula, joe_copula]
# AMH is limited to d=2
COPULAS_2D = [amh_copula]
ALL_ARCH_COPULAS = COPULAS_3D + COPULAS_2D + [independence_copula]

COPULAS_3D_IDS = [c.name for c in COPULAS_3D]
ALL_IDS = [c.name for c in ALL_ARCH_COPULAS]


def _get_arch_params(copula, d=3, theta=None):
    """Get example params for an Archimedean copula."""
    if copula.name == "AMH-Copula":
        d = 2
    if copula.name == "Independence-Copula":
        return copula.example_params(dim=d)
    params = copula.example_params(dim=d)
    if theta is not None:
        params["copula"]["theta"] = jnp.array(float(theta))
    return params


# ---------------------------------------------------------------------------
# Generator properties
# ---------------------------------------------------------------------------

class TestGeneratorProperties:
    """Verify Archimedean generator mathematical properties."""

    @pytest.mark.parametrize("copula", COPULAS_3D, ids=COPULAS_3D_IDS)
    def test_generator_at_one_is_zero(self, copula):
        """phi(1) == 0 for all Archimedean generators."""
        params = _get_arch_params(copula)
        theta = float(params["copula"]["theta"])
        val = float(copula.generator(jnp.array(1.0), jnp.array(theta)))
        np.testing.assert_allclose(val, 0.0, atol=1e-6,
                                   err_msg=f"{copula.name}: phi(1) != 0")

    @pytest.mark.parametrize("copula", COPULAS_3D, ids=COPULAS_3D_IDS)
    def test_generator_inverse_roundtrip(self, copula):
        """phi_inv(phi(t)) == t for t in (0, 1)."""
        params = _get_arch_params(copula)
        theta = jnp.array(float(params["copula"]["theta"]))
        t_vals = jnp.linspace(0.05, 0.95, 20)

        for t in t_vals:
            s = copula.generator(t, theta)
            t_recovered = copula.generator_inv(s, theta)
            np.testing.assert_allclose(
                float(t_recovered), float(t), rtol=1e-5, atol=1e-6,
                err_msg=f"{copula.name}: phi_inv(phi({float(t):.2f})) != {float(t):.2f}"
            )

    @pytest.mark.parametrize("copula", COPULAS_3D, ids=COPULAS_3D_IDS)
    def test_generator_is_decreasing(self, copula):
        """phi(t) should be decreasing: phi(t1) > phi(t2) for t1 < t2."""
        params = _get_arch_params(copula)
        theta = jnp.array(float(params["copula"]["theta"]))
        t_vals = jnp.linspace(0.05, 0.95, 20)
        phi_vals = np.array([float(copula.generator(t, theta)) for t in t_vals])
        diffs = np.diff(phi_vals)
        assert np.all(diffs <= 1e-6), \
            f"{copula.name}: generator not decreasing"


# ---------------------------------------------------------------------------
# Tau-to-theta inversion
# ---------------------------------------------------------------------------

class TestTauToTheta:
    """Verify Kendall's tau -> theta -> tau round-trip.

    Catches FINDING-06-01: Frank _tau_to_theta returns 0 for most tau values.
    """

    def test_clayton_tau_roundtrip(self):
        """Clayton: tau = theta/(theta+2), so theta = 2*tau/(1-tau)."""
        for tau in [0.1, 0.3, 0.5, 0.7, 0.9]:
            expected_theta = 2 * tau / (1 - tau)
            recovered_theta = float(clayton_copula._tau_to_theta(jnp.array(tau)))
            np.testing.assert_allclose(
                recovered_theta, expected_theta, rtol=1e-2,
                err_msg=f"Clayton tau={tau}: theta mismatch"
            )

    def test_gumbel_tau_roundtrip(self):
        """Gumbel: tau = 1 - 1/theta, so theta = 1/(1-tau)."""
        for tau in [0.1, 0.3, 0.5, 0.7, 0.9]:
            expected_theta = 1.0 / (1.0 - tau)
            recovered_theta = float(gumbel_copula._tau_to_theta(jnp.array(tau)))
            np.testing.assert_allclose(
                recovered_theta, expected_theta, rtol=1e-2,
                err_msg=f"Gumbel tau={tau}: theta mismatch"
            )

    @pytest.mark.parametrize("tau", [0.1, 0.3, 0.5, 0.7])
    def test_frank_tau_roundtrip(self, tau):
        """Frank: tau = 1 - 4/theta*(1 - D1(theta)).

        FINDING-06-01: If _tau_to_theta returns 0 (or near-zero),
        then tau_recovered will not match the input tau.
        """
        theta = float(frank_copula._tau_to_theta(jnp.array(tau)))

        # theta must be non-zero for Frank copula
        assert abs(theta) > 0.01, \
            f"Frank _tau_to_theta({tau}) returned theta={theta} (near zero!)"

        # Round-trip: compute tau from theta
        D1 = float(frank_copula._debye1(jnp.array(theta)))
        tau_recovered = 1.0 - 4.0 / theta * (1.0 - D1)

        np.testing.assert_allclose(
            tau_recovered, tau, rtol=1e-2,
            err_msg=f"Frank tau round-trip failed: tau={tau}, theta={theta}, "
                    f"tau_recovered={tau_recovered}"
        )

    @pytest.mark.parametrize("copula", [joe_copula, amh_copula],
                             ids=["Joe", "AMH"])
    def test_other_copula_tau_roundtrip(self, copula):
        """Generic tau -> theta -> tau round-trip for Joe and AMH."""
        # Use moderate tau values
        for tau in [0.1, 0.3]:
            theta = float(copula._tau_to_theta(jnp.array(tau)))
            assert abs(theta) > 1e-3, \
                f"{copula.name} _tau_to_theta({tau}) returned theta={theta}"
            # Since there's no tau_kendall method, we verify theta is
            # in the valid range and non-trivial
            if copula.name == "Joe-Copula":
                assert theta >= 1.0, f"Joe theta must be >= 1, got {theta}"
            elif copula.name == "AMH-Copula":
                assert -1.0 <= theta < 1.0, f"AMH theta must be in [-1,1), got {theta}"


# ---------------------------------------------------------------------------
# Copula CDF properties
# ---------------------------------------------------------------------------

class TestCopulaCdf:
    """Verify copula CDF properties."""

    @pytest.mark.parametrize("copula", COPULAS_3D, ids=COPULAS_3D_IDS)
    def test_cdf_in_unit_interval(self, copula):
        """C(u) in [0, 1] for all u in (0,1)^d."""
        d = 3
        params = _get_arch_params(copula, d)
        np.random.seed(42)
        u = jnp.array(np.random.uniform(0.01, 0.99, (30, d)))
        cdf = np.array(copula.copula_cdf(u=u, params=params)).flatten()
        assert np.all(cdf >= -1e-6), f"{copula.name} CDF < 0"
        assert np.all(cdf <= 1 + 1e-6), f"{copula.name} CDF > 1"

    @pytest.mark.parametrize("copula", COPULAS_3D, ids=COPULAS_3D_IDS)
    def test_frechet_upper_bound(self, copula):
        """C(u) <= min(u_i) (Frechet upper bound)."""
        d = 3
        params = _get_arch_params(copula, d)
        np.random.seed(42)
        u = jnp.array(np.random.uniform(0.1, 0.9, (30, d)))
        cdf = np.array(copula.copula_cdf(u=u, params=params)).flatten()
        upper_bound = np.min(np.array(u), axis=1)
        assert np.all(cdf <= upper_bound + 1e-4), \
            f"{copula.name} CDF exceeds Frechet upper bound"

    @pytest.mark.parametrize("copula", COPULAS_3D, ids=COPULAS_3D_IDS)
    def test_boundary_any_zero(self, copula):
        """C(u) ≈ 0 when any u_i ≈ 0."""
        d = 3
        params = _get_arch_params(copula, d)
        u = jnp.array([[1e-6, 0.5, 0.5]])
        cdf = float(copula.copula_cdf(u=u, params=params).flatten()[0])
        assert cdf < 0.01, f"{copula.name} CDF should be ~0 when u_i~0, got {cdf}"


# ---------------------------------------------------------------------------
# Copula density
# ---------------------------------------------------------------------------

class TestCopulaDensity:
    """Verify copula density properties."""

    @pytest.mark.parametrize("copula", COPULAS_3D, ids=COPULAS_3D_IDS)
    def test_copula_pdf_positive(self, copula):
        """copula_pdf > 0 on the interior of the unit cube."""
        d = 3
        params = _get_arch_params(copula, d)
        np.random.seed(42)
        u = jnp.array(np.random.uniform(0.1, 0.9, (20, d)))
        pdf = np.array(copula.copula_pdf(u=u, params=params)).flatten()
        mask = np.isfinite(pdf)
        assert np.all(pdf[mask] > 0), f"{copula.name} copula_pdf not positive"

    @pytest.mark.parametrize("copula", COPULAS_3D, ids=COPULAS_3D_IDS)
    def test_logpdf_pdf_consistency(self, copula):
        """exp(copula_logpdf) == copula_pdf."""
        d = 3
        params = _get_arch_params(copula, d)
        np.random.seed(42)
        u = jnp.array(np.random.uniform(0.1, 0.9, (20, d)))
        logpdf = np.array(copula.copula_logpdf(u=u, params=params)).flatten()
        pdf = np.array(copula.copula_pdf(u=u, params=params)).flatten()
        mask = np.isfinite(logpdf) & (pdf > 0)
        np.testing.assert_allclose(
            np.exp(logpdf[mask]), pdf[mask], rtol=1e-4,
            err_msg=f"{copula.name}: exp(logpdf) != pdf"
        )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestCopulaSampling:
    """Verify copula sampling properties."""

    @pytest.mark.parametrize("copula", COPULAS_3D, ids=COPULAS_3D_IDS)
    def test_marginal_uniformity(self, copula):
        """Each margin of copula_rvs should be approximately U(0,1)."""
        d = 3
        params = _get_arch_params(copula, d)
        key = jax.random.PRNGKey(42)
        samples = np.array(copula.copula_rvs(size=500, params=params, key=key))

        for i in range(d):
            margin = samples[:, i]
            margin = margin[np.isfinite(margin) & (margin > 0) & (margin < 1)]
            if len(margin) < 50:
                continue
            ks_stat, ks_p = scipy.stats.kstest(margin, "uniform")
            assert ks_stat < 0.15, \
                f"{copula.name} dim {i}: KS stat = {ks_stat:.3f} (not uniform)"

    @pytest.mark.parametrize("copula", COPULAS_3D, ids=COPULAS_3D_IDS)
    def test_kendall_tau_positive(self, copula):
        """Copula samples with positive theta should show positive dependence."""
        d = 3
        params = _get_arch_params(copula, d)
        theta = float(params["copula"]["theta"])
        key = jax.random.PRNGKey(42)
        samples = np.array(copula.copula_rvs(size=500, params=params, key=key))

        # Compute empirical Kendall's tau between first two dimensions
        valid = samples[np.all(np.isfinite(samples), axis=1)]
        if len(valid) < 50:
            pytest.skip("Too few valid samples")

        empirical_tau, _ = scipy.stats.kendalltau(valid[:, 0], valid[:, 1])

        # For positive theta, empirical tau should also be positive
        if theta > 0.5:
            assert empirical_tau > 0, \
                f"{copula.name}: theta={theta} but empirical tau={empirical_tau:.3f}"


# ---------------------------------------------------------------------------
# Independence copula
# ---------------------------------------------------------------------------

class TestIndependenceCopula:
    """Verify independence copula special properties."""

    def test_cdf_is_product(self):
        """C(u) = prod(u_i) for independence copula."""
        d = 3
        params = independence_copula.example_params(dim=d)
        np.random.seed(42)
        u = jnp.array(np.random.uniform(0.1, 0.9, (20, d)))
        cdf = np.array(independence_copula.copula_cdf(u=u, params=params)).flatten()
        expected = np.prod(np.array(u), axis=1)
        np.testing.assert_allclose(cdf, expected, rtol=1e-5,
                                   err_msg="Independence CDF != product")

    def test_pdf_is_one(self):
        """copula_pdf == 1 for independence copula."""
        d = 3
        params = independence_copula.example_params(dim=d)
        np.random.seed(42)
        u = jnp.array(np.random.uniform(0.1, 0.9, (20, d)))
        pdf = np.array(independence_copula.copula_pdf(u=u, params=params)).flatten()
        np.testing.assert_allclose(pdf, 1.0, rtol=1e-5,
                                   err_msg="Independence PDF != 1")

    def test_logpdf_is_zero(self):
        """copula_logpdf == 0 for independence copula."""
        d = 3
        params = independence_copula.example_params(dim=d)
        np.random.seed(42)
        u = jnp.array(np.random.uniform(0.1, 0.9, (20, d)))
        logpdf = np.array(independence_copula.copula_logpdf(u=u, params=params)).flatten()
        np.testing.assert_allclose(logpdf, 0.0, atol=1e-5,
                                   err_msg="Independence logpdf != 0")

    def test_samples_independent(self):
        """Independence copula samples should show near-zero Kendall's tau."""
        d = 3
        params = independence_copula.example_params(dim=d)
        key = jax.random.PRNGKey(42)
        samples = np.array(independence_copula.copula_rvs(
            size=500, params=params, key=key))
        valid = samples[np.all(np.isfinite(samples), axis=1)]
        if len(valid) < 50:
            pytest.skip("Too few valid samples")
        tau, _ = scipy.stats.kendalltau(valid[:, 0], valid[:, 1])
        assert abs(tau) < 0.15, \
            f"Independence copula tau = {tau}, expected ~0"
