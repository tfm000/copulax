"""Tests for input validation — malformed inputs should produce clear errors.

Covers:
    - 1D inputs to multivariate distributions
    - 3D+ inputs to multivariate distributions
    - Mismatched dimensions between data and params
    - AMH dimension restrictions
"""

import pytest
import jax.numpy as jnp

from copulax.multivariate import mvt_normal, mvt_student_t, mvt_gh, mvt_skewed_t
from copulax.copulas import amh_copula

MVT_DISTS = [mvt_normal, mvt_student_t, mvt_gh, mvt_skewed_t]


class TestMultivariateInputValidation:
    """Multivariate distributions should reject invalid input shapes."""

    @pytest.mark.parametrize("dist", MVT_DISTS)
    @pytest.mark.parametrize("method", ["logpdf", "pdf"])
    def test_1d_input_raises(self, dist, method):
        """1D array should raise for multivariate logpdf/pdf."""
        params = dist.example_params()
        x_1d = jnp.array([1.0, 2.0, 3.0])
        with pytest.raises((ValueError, TypeError, IndexError)):
            getattr(dist, method)(x_1d, params=params)

    @pytest.mark.parametrize("dist", MVT_DISTS)
    @pytest.mark.parametrize("method", ["logpdf", "pdf"])
    def test_3d_input_raises(self, dist, method):
        """3D array should raise for multivariate logpdf/pdf."""
        params = dist.example_params()
        x_3d = jnp.ones((2, 3, 3))
        with pytest.raises((ValueError, TypeError)):
            getattr(dist, method)(x_3d, params=params)

    @pytest.mark.parametrize("dist", MVT_DISTS)
    @pytest.mark.parametrize("method", ["logpdf", "pdf"])
    def test_dim_mismatch_raises(self, dist, method):
        """Mismatched dimensions between data and params should raise."""
        params = dist.example_params()
        x_wrong_dim = jnp.ones((10, 100))
        with pytest.raises((ValueError, TypeError)):
            getattr(dist, method)(x_wrong_dim, params=params)


class TestAMHValidation:
    """AMH copula should reject dim > 2."""

    @pytest.mark.parametrize("dim", [3, 4])
    def test_high_dim_raises(self, dim):
        with pytest.raises(ValueError, match="d=2"):
            amh_copula.example_params(dim=dim)
