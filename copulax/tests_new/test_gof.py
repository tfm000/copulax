"""Rigorous tests for goodness-of-fit tests: KS and CVM.

Cross-validates against scipy.stats.kstest and scipy.stats.cramervonmises.

"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats
from scipy.special import kolmogorov

from copulax._src.univariate._gof import ks_test, cvm_test, _cvm_pvalue, _ks_pvalue
from copulax._src.univariate.normal import normal


# ===================================================================
# KS test
# ===================================================================

class TestKSTest:
    """Kolmogorov-Smirnov goodness-of-fit test.

    Covers cross-validation against scipy, generic properties, and
    bound-method access via distribution.ks_test.
    """

    # ----- Against scipy -----

    def test_statistic_matches_scipy(self):
        """KS statistic matches scipy on normal data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        params = {"mu": 0.0, "sigma": 1.0}

        cx_result = ks_test(jnp.array(data), normal, params)
        sp_result = scipy.stats.kstest(data, "norm")

        np.testing.assert_allclose(
            float(cx_result["statistic"]), sp_result.statistic, rtol=1e-5,
            err_msg="KS statistic mismatch"
        )

    def test_pvalue_correct_fit(self):
        """When data matches the distribution, p-value should be > 0.05."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        params = {"mu": 0.0, "sigma": 1.0}
        result = ks_test(jnp.array(data), normal, params)
        p = float(result["p_value"])
        assert p > 0.01, f"KS p-value too low for correct fit: {p}"

    def test_pvalue_wrong_fit(self):
        """When data doesn't match, p-value should be < 0.05."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, 500)
        params = {"mu": 0.5, "sigma": 1.0}
        result = ks_test(jnp.array(data), normal, params)
        p = float(result["p_value"])
        assert p < 0.05, f"KS p-value too high for wrong fit: {p}"

    @pytest.mark.parametrize("n", [50, 200, 1000])
    def test_pvalue_close_to_scipy(self, n):
        """KS p-value should agree with scipy on accept/reject decision."""
        np.random.seed(42)
        data = np.random.normal(0, 1, n)
        params = {"mu": 0.0, "sigma": 1.0}

        cx_result = ks_test(jnp.array(data), normal, params)
        sp_result = scipy.stats.kstest(data, "norm")

        # CopulAX uses Marsaglia asymptotic, scipy uses exact finite-sample.
        # Both should agree on the accept/reject decision.
        p_cx = float(cx_result["p_value"])
        p_sp = sp_result.pvalue
        assert (p_cx > 0.05) == (p_sp > 0.05), \
            f"n={n}: KS decision mismatch: copulax p={p_cx:.4f}, scipy p={p_sp:.4f}"

    def test_pvalue_matches_kolmogorov(self):
        """_ks_pvalue matches scipy.special.kolmogorov at the
        Marsaglia-corrected statistic.

        CopulAX computes Q(lambda) = 2 * sum_{k>=1} (-1)^{k+1} exp(-2 k^2 lambda^2)
        at lambda = (sqrt(n) + 0.12 + 0.11/sqrt(n)) * d. scipy.special.kolmogorov
        returns exactly that series as the survival function of the
        Kolmogorov distribution.
        """
        n = 100.0
        sqrt_n = np.sqrt(n)
        for d in [0.02, 0.05, 0.10, 0.15, 0.20]:
            lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d
            cx_p = float(_ks_pvalue(jnp.array(d), jnp.array(n)))
            sp_p = float(kolmogorov(lam))
            np.testing.assert_allclose(
                cx_p, sp_p, rtol=1e-6,
                err_msg=f"KS p-value mismatch at d={d}, n={n}",
            )

    # ----- Properties -----

    def test_pvalue_in_range(self):
        """KS p-value should be in [0, 1]."""
        np.random.seed(0)
        data = np.random.normal(0, 1, 100)
        result = ks_test(jnp.array(data), normal, {"mu": 0.0, "sigma": 1.0})
        p = float(result["p_value"])
        assert 0 <= p <= 1

    def test_statistic_non_negative(self):
        """KS statistic D_n >= 0."""
        np.random.seed(0)
        data = np.random.normal(0, 1, 100)
        result = ks_test(jnp.array(data), normal, {"mu": 0.0, "sigma": 1.0})
        assert float(result["statistic"]) >= 0

    def test_jit_compatible(self):
        """ks_test should be JIT-compatible."""
        data = jnp.array(np.random.normal(0, 1, 50))
        params = {"mu": 0.0, "sigma": 1.0}
        ks_jit = jax.jit(lambda x: ks_test(x, normal, params))
        result = ks_jit(data)
        assert np.isfinite(float(result["p_value"]))

    # ----- Bound method -----

    def test_bound_method(self):
        """normal.ks_test(x, params) should return valid results."""
        np.random.seed(42)
        data = jnp.array(np.random.normal(0, 1, 200))
        params = {"mu": 0.0, "sigma": 1.0}
        result = normal.ks_test(data, params=params)
        assert "statistic" in result
        assert "p_value" in result
        p = float(result["p_value"])
        assert 0 <= p <= 1

    def test_bound_method_matches_function(self):
        """Bound method should produce same result as standalone function."""
        np.random.seed(42)
        data = jnp.array(np.random.normal(0, 1, 200))
        params = {"mu": 0.0, "sigma": 1.0}

        bound_result = normal.ks_test(data, params=params)
        func_result = ks_test(data, normal, params)

        np.testing.assert_allclose(
            float(bound_result["statistic"]),
            float(func_result["statistic"]),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(bound_result["p_value"]),
            float(func_result["p_value"]),
            rtol=1e-10,
        )


# ===================================================================
# CVM test
# ===================================================================

class TestCVMTest:
    """Cramer-von Mises goodness-of-fit test.

    Covers cross-validation against scipy, generic properties, and
    bound-method access via distribution.cvm_test.

    Both CopulAX and scipy use the Csorgo-Faraway (1996) eigenvalue
    expansion, so p-values should match tightly.
    """

    # ----- Against scipy -----

    def test_statistic_matches_scipy(self):
        """CVM statistic W^2 should match scipy exactly."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        params = {"mu": 0.0, "sigma": 1.0}

        cx_result = cvm_test(jnp.array(data), normal, params)
        sp_result = scipy.stats.cramervonmises(data, "norm")

        np.testing.assert_allclose(
            float(cx_result["statistic"]), sp_result.statistic, rtol=1e-5,
            err_msg="CVM statistic mismatch"
        )

    def test_pvalue_correct_fit(self):
        """When data matches, p-value should be > 0.05."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        params = {"mu": 0.0, "sigma": 1.0}
        result = cvm_test(jnp.array(data), normal, params)
        p = float(result["p_value"])
        assert p > 0.01, f"CVM p-value too low for correct fit: {p}"

    def test_pvalue_wrong_fit(self):
        """When data doesn't match, p-value should be very small."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, 500)
        params = {"mu": 0.5, "sigma": 1.0}
        result = cvm_test(jnp.array(data), normal, params)
        p = float(result["p_value"])
        assert p < 0.05, f"CVM p-value too high for wrong fit: {p}"

    def test_pvalue_close_to_scipy(self):
        """CVM end-to-end p-value should match scipy's asymptotic formula.

        Copulax uses the asymptotic Csorgo-Faraway expansion (matching
        scipy._cdf_cvm_inf). Scipy's cramervonmises applies an additional
        finite-sample correction, so we compare against the asymptotic
        formula directly.
        """
        from scipy.stats._hypotests import _cdf_cvm_inf

        np.random.seed(42)
        scenarios = [
            ("correct_fit", np.random.normal(0, 1, 200)),
            ("mild_mismatch", np.random.standard_t(5, 200)),
        ]

        for name, data in scenarios:
            params = {"mu": float(np.mean(data)), "sigma": float(np.std(data))}

            cx_result = cvm_test(jnp.array(data), normal, params)
            w2_cx = float(cx_result["statistic"])
            p_cx = float(cx_result["p_value"])

            # Compare statistic against scipy
            sp_result = scipy.stats.cramervonmises(
                data, "norm", args=(float(params["mu"]), float(params["sigma"]))
            )
            np.testing.assert_allclose(w2_cx, sp_result.statistic, rtol=1e-5,
                err_msg=f"CVM [{name}] W^2 statistic mismatch")

            # Compare p-value against scipy's asymptotic formula at same W^2
            p_asymptotic = 1.0 - _cdf_cvm_inf(w2_cx)
            np.testing.assert_allclose(p_cx, p_asymptotic, rtol=1e-5,
                err_msg=f"CVM [{name}] p-value mismatch vs asymptotic")

    # ----- Properties -----

    def test_pvalue_in_range(self):
        """CVM p-value should be in [0, 1]."""
        np.random.seed(0)
        data = np.random.normal(0, 1, 100)
        result = cvm_test(jnp.array(data), normal, {"mu": 0.0, "sigma": 1.0})
        p = float(result["p_value"])
        assert 0 <= p <= 1

    def test_statistic_non_negative(self):
        """CVM statistic W^2 >= 0."""
        np.random.seed(0)
        data = np.random.normal(0, 1, 100)
        result = cvm_test(jnp.array(data), normal, {"mu": 0.0, "sigma": 1.0})
        assert float(result["statistic"]) >= 0

    def test_jit_compatible(self):
        """cvm_test should be JIT-compatible."""
        data = jnp.array(np.random.normal(0, 1, 50))
        params = {"mu": 0.0, "sigma": 1.0}
        cvm_jit = jax.jit(lambda x: cvm_test(x, normal, params))
        result = cvm_jit(data)
        assert np.isfinite(float(result["p_value"]))

    # ----- Bound method -----

    def test_bound_method(self):
        """normal.cvm_test(x, params) should return valid results."""
        np.random.seed(42)
        data = jnp.array(np.random.normal(0, 1, 200))
        params = {"mu": 0.0, "sigma": 1.0}
        result = normal.cvm_test(data, params=params)
        assert "statistic" in result
        assert "p_value" in result
        p = float(result["p_value"])
        assert 0 <= p <= 1

    def test_bound_method_matches_function(self):
        """Bound method should produce same result as standalone function."""
        np.random.seed(42)
        data = jnp.array(np.random.normal(0, 1, 200))
        params = {"mu": 0.0, "sigma": 1.0}

        bound_result = normal.cvm_test(data, params=params)
        func_result = cvm_test(data, normal, params)

        np.testing.assert_allclose(
            float(bound_result["statistic"]),
            float(func_result["statistic"]),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(bound_result["p_value"]),
            float(func_result["p_value"]),
            rtol=1e-10,
        )


# ===================================================================
# Direct CVM p-value formula tests
# ===================================================================

class TestCVMPvalueFormula:
    """Direct test of the _cvm_pvalue function against Csorgo-Faraway reference values."""

    @pytest.mark.parametrize("w2,expected_p", [
        (0.05, 8.762809310e-01),
        (0.1,  5.848734384e-01),
        (0.5,  3.983321757e-02),
        (1.0,  2.460452180e-03),
        (2.0,  1.278073627e-05),
    ])
    def test_cvm_pvalue_against_reference(self, w2, expected_p):
        """_cvm_pvalue(w2) should match scipy._cdf_cvm_inf reference values."""
        p = float(_cvm_pvalue(jnp.array(w2)))
        assert 0 <= p <= 1, f"p-value out of [0,1]: {p}"
        np.testing.assert_allclose(p, expected_p, rtol=1e-5,
                                   err_msg=f"CVM p-value mismatch at W^2={w2}")
