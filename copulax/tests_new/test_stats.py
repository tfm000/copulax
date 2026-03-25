"""Rigorous tests for copulax._src.stats: skew and kurtosis.

Cross-validates against scipy.stats with tight tolerances.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from copulax._src.stats import skew, kurtosis


class TestSkew:
    """Tests for sample skewness."""

    @pytest.mark.parametrize("bias", [True, False])
    def test_matches_scipy(self, bias):
        """skew(x, bias) matches scipy.stats.skew(x, bias=bias)."""
        np.random.seed(42)
        for data in [np.random.normal(0, 1, 500),
                     np.random.exponential(1, 500),
                     np.random.uniform(-1, 1, 500)]:
            cx = float(skew(jnp.array(data), bias=bias))
            sp = float(scipy.stats.skew(data, bias=bias))
            np.testing.assert_allclose(cx, sp, atol=1e-8,
                                       err_msg=f"Skew mismatch (bias={bias})")

    def test_symmetric_near_zero(self):
        """Symmetric distribution should have skewness near 0."""
        np.random.seed(0)
        data = np.random.normal(0, 1, 10000)
        s = float(skew(jnp.array(data)))
        assert abs(s) < 0.1, f"Symmetric data skewness = {s}"

    def test_known_positive_skew(self):
        """Exponential distribution has positive skewness (theoretical = 2)."""
        np.random.seed(0)
        data = np.random.exponential(1, 50000)
        s = float(skew(jnp.array(data)))
        np.testing.assert_allclose(s, 2.0, rtol=0.15,
                                   err_msg="Exponential skewness should be ~2")

    def test_jit_and_grad(self):
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 10.0])
        f = jax.jit(lambda x: skew(x))
        result = f(data)
        assert np.isfinite(float(result))
        # Gradient should be computable
        g = jax.grad(lambda x: skew(x).sum())(data)
        assert np.all(np.isfinite(np.array(g)))


class TestKurtosis:
    """Tests for sample kurtosis."""

    @pytest.mark.parametrize("fisher", [True, False])
    @pytest.mark.parametrize("bias", [True, False])
    def test_matches_scipy(self, fisher, bias):
        """kurtosis(x, fisher, bias) matches scipy.stats.kurtosis."""
        np.random.seed(42)
        for data in [np.random.normal(0, 1, 500),
                     np.random.exponential(1, 500),
                     np.random.uniform(-1, 1, 500)]:
            cx = float(kurtosis(jnp.array(data), fisher=fisher, bias=bias))
            sp = float(scipy.stats.kurtosis(data, fisher=fisher, bias=bias))
            np.testing.assert_allclose(cx, sp, atol=1e-8,
                                       err_msg=f"Kurtosis mismatch "
                                               f"(fisher={fisher}, bias={bias})")

    def test_normal_excess_near_zero(self):
        """Normal distribution excess kurtosis = 0."""
        np.random.seed(0)
        data = np.random.normal(0, 1, 50000)
        k = float(kurtosis(jnp.array(data), fisher=True))
        np.testing.assert_allclose(k, 0.0, atol=0.1,
                                   err_msg="Normal excess kurtosis should be ~0")

    def test_normal_pearson_near_three(self):
        """Normal distribution Pearson kurtosis = 3."""
        np.random.seed(0)
        data = np.random.normal(0, 1, 50000)
        k = float(kurtosis(jnp.array(data), fisher=False))
        np.testing.assert_allclose(k, 3.0, atol=0.1,
                                   err_msg="Normal Pearson kurtosis should be ~3")

    def test_uniform_excess_kurtosis(self):
        """Uniform distribution excess kurtosis = -6/5."""
        np.random.seed(0)
        data = np.random.uniform(0, 1, 100000)
        k = float(kurtosis(jnp.array(data), fisher=True))
        np.testing.assert_allclose(k, -6 / 5, atol=0.05,
                                   err_msg="Uniform excess kurtosis should be -6/5")

    def test_jit_and_grad(self):
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 10.0])
        f = jax.jit(lambda x: kurtosis(x))
        result = f(data)
        assert np.isfinite(float(result))
        g = jax.grad(lambda x: kurtosis(x).sum())(data)
        assert np.all(np.isfinite(np.array(g)))
