"""Tests for copulax.stats (skew, kurtosis).

Compares against scipy.stats for correctness and verifies JIT + gradient support.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats as sp_stats

from copulax.stats import skew, kurtosis


# Shared test data: asymmetric, symmetric, heavy-tailed, constant-ish
_DATA = {
    "right_skewed": np.array([1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 15.0]),
    "left_skewed": np.array([15.0, 10.0, 5.0, 3.0, 2.5, 2.0, 1.5, 1.0]),
    "symmetric": np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
    "normal_sample": np.random.default_rng(0).standard_normal(500),
}


class TestSkew:
    """Tests for copulax.stats.skew."""

    @pytest.mark.parametrize("key", list(_DATA.keys()))
    @pytest.mark.parametrize("bias", [True, False])
    def test_matches_scipy(self, key, bias):
        x = _DATA[key]
        expected = sp_stats.skew(x, bias=bias)
        result = float(skew(x, bias=bias))
        assert np.allclose(
            result, expected, atol=1e-6
        ), f"skew({key}, bias={bias}): {result} vs scipy {expected}"

    def test_symmetric_near_zero(self):
        result = float(skew(_DATA["symmetric"]))
        assert abs(result) < 1e-10

    def test_jit(self):
        x = jnp.array(_DATA["right_skewed"])
        result = jax.jit(skew)(x)
        assert jnp.isfinite(result)

    def test_gradient(self):
        x = jnp.array(_DATA["right_skewed"])
        g = jax.grad(lambda x: skew(x).sum())(x)
        assert jnp.all(jnp.isfinite(g))


class TestKurtosis:
    """Tests for copulax.stats.kurtosis."""

    @pytest.mark.parametrize("key", list(_DATA.keys()))
    @pytest.mark.parametrize("fisher", [True, False])
    @pytest.mark.parametrize("bias", [True, False])
    def test_matches_scipy(self, key, fisher, bias):
        x = _DATA[key]
        expected = sp_stats.kurtosis(x, fisher=fisher, bias=bias)
        result = float(kurtosis(x, fisher=fisher, bias=bias))
        assert np.allclose(
            result, expected, atol=1e-5
        ), f"kurtosis({key}, fisher={fisher}, bias={bias}): {result} vs scipy {expected}"

    def test_normal_excess_near_zero(self):
        """Normal sample should have excess kurtosis ≈ 0."""
        result = float(kurtosis(_DATA["normal_sample"], fisher=True, bias=True))
        assert abs(result) < 0.5  # expected ~0 for n=500

    def test_pearson_normal_near_three(self):
        result = float(kurtosis(_DATA["normal_sample"], fisher=False, bias=True))
        assert abs(result - 3.0) < 0.5

    def test_jit(self):
        x = jnp.array(_DATA["right_skewed"])
        result = jax.jit(kurtosis)(x)
        assert jnp.isfinite(result)

    def test_gradient(self):
        x = jnp.array(_DATA["right_skewed"])
        g = jax.grad(lambda x: kurtosis(x).sum())(x)
        assert jnp.all(jnp.isfinite(g))
