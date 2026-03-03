"""Tests for input validation — malformed inputs should produce clear errors.

Covers:
    - 1D inputs to multivariate distributions
    - 3D+ inputs to multivariate distributions
    - Mismatched dimensions between data and params
"""
import pytest
import jax.numpy as jnp
import numpy as np

from copulax.multivariate import mvt_normal, mvt_student_t, mvt_gh, mvt_skewed_t
from copulax.copulas import (
    gaussian_copula, student_t_copula,
    clayton_copula, frank_copula, gumbel_copula, joe_copula, amh_copula,
)

MVT_DISTS = [mvt_normal, mvt_student_t, mvt_gh, mvt_skewed_t]
COPULA_DISTS = [gaussian_copula, student_t_copula]
ARCH_DISTS = [clayton_copula, frank_copula, gumbel_copula, joe_copula]


# ──────────────────────────────────────────────────────────────────────
# Multivariate: 1D input
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dist", MVT_DISTS)
def test_mvt_1d_input_logpdf(dist):
    """1D array should raise an error for multivariate logpdf."""
    params = dist.example_params()
    x_1d = jnp.array([1.0, 2.0, 3.0])  # shape (3,) not (n, d)
    with pytest.raises((ValueError, TypeError, IndexError)):
        dist.logpdf(x_1d, params=params)


@pytest.mark.parametrize("dist", MVT_DISTS)
def test_mvt_1d_input_pdf(dist):
    """1D array should raise an error for multivariate pdf."""
    params = dist.example_params()
    x_1d = jnp.array([1.0, 2.0, 3.0])
    with pytest.raises((ValueError, TypeError, IndexError)):
        dist.pdf(x_1d, params=params)


# ──────────────────────────────────────────────────────────────────────
# Multivariate: 3D+ input
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dist", MVT_DISTS)
def test_mvt_3d_input_logpdf(dist):
    """3D array should raise an error or reshape for multivariate logpdf."""
    params = dist.example_params()
    x_3d = jnp.ones((2, 3, 3))
    with pytest.raises((ValueError, TypeError)):
        dist.logpdf(x_3d, params=params)


@pytest.mark.parametrize("dist", MVT_DISTS)
def test_mvt_3d_input_pdf(dist):
    """3D array should raise an error or reshape for multivariate pdf."""
    params = dist.example_params()
    x_3d = jnp.ones((2, 3, 3))
    with pytest.raises((ValueError, TypeError)):
        dist.pdf(x_3d, params=params)


# ──────────────────────────────────────────────────────────────────────
# Multivariate: Dimension mismatch
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dist", MVT_DISTS)
def test_mvt_dim_mismatch_logpdf(dist):
    """Mismatched dimensions between data and params should raise."""
    params = dist.example_params()  # default dim=3
    x_wrong_dim = jnp.ones((10, 100))  # d=100 ≠ 3
    with pytest.raises((ValueError, TypeError)):
        dist.logpdf(x_wrong_dim, params=params)


@pytest.mark.parametrize("dist", MVT_DISTS)
def test_mvt_dim_mismatch_pdf(dist):
    """Mismatched dimensions between data and params should raise."""
    params = dist.example_params()
    x_wrong_dim = jnp.ones((10, 100))
    with pytest.raises((ValueError, TypeError)):
        dist.pdf(x_wrong_dim, params=params)


# ──────────────────────────────────────────────────────────────────────
# AMH specific: dimension restriction
# ──────────────────────────────────────────────────────────────────────
def test_amh_dim_3_raises():
    """AMH copula should raise ValueError for dim > 2."""
    with pytest.raises(ValueError, match="d=2"):
        amh_copula.example_params(dim=3)


def test_amh_dim_4_raises():
    """AMH copula should raise ValueError for dim > 2."""
    with pytest.raises(ValueError, match="d=2"):
        amh_copula.example_params(dim=4)
