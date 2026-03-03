"""Tests for the Kolmogorov-Smirnov and Cramér-von Mises goodness-of-fit tests."""

import numpy as np
import pytest
from jax import jit

from copulax.univariate import ks_test, cvm_test, normal, uniform, student_t


np.random.seed(42)
NORMAL_DATA = np.random.normal(0, 1, 200)
UNIFORM_DATA = np.random.uniform(0, 1, 200)


# ── helpers ──────────────────────────────────────────────────────────────────
def _check_result(result):
    """Validate the shape of a GoF result dict."""
    assert isinstance(result, dict)
    assert "statistic" in result and "p_value" in result
    stat = float(result["statistic"])
    pval = float(result["p_value"])
    assert 0.0 <= stat, "Statistic must be non-negative"
    assert 0.0 <= pval <= 1.0, f"p-value out of range: {pval}"


# ── KS test ──────────────────────────────────────────────────────────────────
class TestKSTest:
    def test_should_not_reject_correct_dist(self):
        """Normal data fitted with normal should not be rejected."""
        params = normal.fit(NORMAL_DATA)
        result = ks_test(NORMAL_DATA, normal, params)
        _check_result(result)
        assert float(result["p_value"]) > 0.05

    def test_should_reject_wrong_dist(self):
        """Uniform data tested against a standard normal should be rejected."""
        params = normal.fit(UNIFORM_DATA)
        result = ks_test(UNIFORM_DATA, normal, params)
        _check_result(result)
        assert float(result["p_value"]) < 0.05

    def test_jit_compilable(self):
        """ks_test must be jit-compilable."""
        params = normal.fit(NORMAL_DATA)
        jitted = jit(ks_test)
        result = jitted(NORMAL_DATA, normal, params)
        _check_result(result)

    def test_as_method(self):
        """ks_test accessible as a distribution method."""
        params = normal.fit(NORMAL_DATA)
        result = normal.ks_test(NORMAL_DATA, params)
        _check_result(result)
        assert float(result["p_value"]) > 0.05


# ── CvM test ─────────────────────────────────────────────────────────────────
class TestCvMTest:
    def test_should_not_reject_correct_dist(self):
        """Normal data fitted with normal should not be rejected."""
        params = normal.fit(NORMAL_DATA)
        result = cvm_test(NORMAL_DATA, normal, params)
        _check_result(result)
        assert float(result["p_value"]) > 0.05

    def test_should_reject_wrong_dist(self):
        """Uniform data tested against a standard normal should be rejected."""
        # use more samples to make the rejection unambiguous at float32
        np.random.seed(0)
        unif = np.random.uniform(0, 1, 500)
        params = normal.fit(unif)
        result = cvm_test(unif, normal, params)
        _check_result(result)
        assert float(result["p_value"]) < 0.05

    def test_jit_compilable(self):
        """cvm_test must be jit-compilable."""
        params = normal.fit(NORMAL_DATA)
        jitted = jit(cvm_test)
        result = jitted(NORMAL_DATA, normal, params)
        _check_result(result)

    def test_as_method(self):
        """cvm_test accessible as a distribution method."""
        params = normal.fit(NORMAL_DATA)
        result = normal.cvm_test(NORMAL_DATA, params)
        _check_result(result)
        assert float(result["p_value"]) > 0.05


# ── Uniform distribution GoF ─────────────────────────────────────────────────
class TestUniformGoF:
    def test_ks_uniform_correct(self):
        """Uniform data fitted with uniform should not be rejected."""
        params = uniform.fit(UNIFORM_DATA)
        result = ks_test(UNIFORM_DATA, uniform, params)
        _check_result(result)
        assert float(result["p_value"]) > 0.05

    def test_cvm_uniform_correct(self):
        params = uniform.fit(UNIFORM_DATA)
        result = cvm_test(UNIFORM_DATA, uniform, params)
        _check_result(result)
        assert float(result["p_value"]) > 0.05
