"""Golden regression tests — verify numerical outputs match stored fixtures.

These tests load pre-computed fixtures from .npz files and compare
current distribution outputs against them using jnp.allclose(rtol=1e-5).

Run `python -m copulax.tests.golden.generate_golden` to regenerate fixtures.

These are marked with @pytest.mark.golden so they can be excluded from
regular CI runs: pytest -m "not golden"
"""

import os
import pytest
import numpy as np
import jax.numpy as jnp

from copulax.univariate import (
    gamma,
    gh,
    gig,
    ig,
    lognormal,
    normal,
    skewed_t,
    student_t,
    uniform,
)
from copulax.multivariate import mvt_normal, mvt_student_t, mvt_gh, mvt_skewed_t
from copulax.copulas import (
    gaussian_copula,
    student_t_copula,
    clayton_copula,
    frank_copula,
    gumbel_copula,
    joe_copula,
    amh_copula,
)

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
RTOL = 1e-5
ATOL = 1e-6
COPULA_ATOL = 1e-5


def _load_npz(name):
    path = os.path.join(GOLDEN_DIR, f"{name}.npz")
    if not os.path.exists(path):
        pytest.skip(f"Golden fixture {path} not found. Run generate_golden.py first.")
    return np.load(path, allow_pickle=True)


# ──────────────────────────────────────────────────────────────────────
# Univariate
# ──────────────────────────────────────────────────────────────────────
UVT_DISTS = [
    ("gamma", gamma),
    ("gh", gh),
    ("gig", gig),
    ("ig", ig),
    ("lognormal", lognormal),
    ("normal", normal),
    ("skewed_t", skewed_t),
    ("student_t", student_t),
    ("uniform", uniform),
]


@pytest.mark.golden
@pytest.mark.parametrize("name, dist", UVT_DISTS)
@pytest.mark.parametrize("method", ["logpdf", "pdf"])
def test_univariate(name, dist, method):
    data = _load_npz("univariate")
    x = jnp.asarray(data[f"{name}/x"])
    expected = data[f"{name}/{method}"]
    actual = getattr(dist, method)(x, dist.example_params())
    assert np.allclose(
        actual, expected, rtol=RTOL, atol=ATOL
    ), f"{name} {method} regression: max diff = {np.max(np.abs(np.asarray(actual) - expected))}"


# ──────────────────────────────────────────────────────────────────────
# Multivariate
# ──────────────────────────────────────────────────────────────────────
MVT_DISTS = [
    ("mvt_normal", mvt_normal),
    ("mvt_student_t", mvt_student_t),
    ("mvt_gh", mvt_gh),
    ("mvt_skewed_t", mvt_skewed_t),
]


@pytest.mark.golden
@pytest.mark.parametrize("name, dist", MVT_DISTS)
@pytest.mark.parametrize("method", ["logpdf", "pdf"])
def test_multivariate(name, dist, method):
    data = _load_npz("multivariate")
    x = jnp.asarray(data[f"{name}/x"])
    expected = data[f"{name}/{method}"]
    actual = getattr(dist, method)(x, params=dist.example_params())
    assert np.allclose(
        actual, expected, rtol=RTOL, atol=ATOL
    ), f"{name} {method} regression: max diff = {np.max(np.abs(np.asarray(actual) - expected))}"


# ──────────────────────────────────────────────────────────────────────
# Elliptical copulas
# ──────────────────────────────────────────────────────────────────────
COPULA_DISTS = [
    ("gaussian_copula", gaussian_copula),
    ("student_t_copula", student_t_copula),
]


@pytest.mark.golden
@pytest.mark.parametrize("name, dist", COPULA_DISTS)
@pytest.mark.parametrize("method", ["copula_logpdf", "copula_pdf"])
def test_copula(name, dist, method):
    data = _load_npz("copulas")
    expected = data[f"{name}/{method}"]
    u_input = jnp.asarray(data[f"{name}/u"])
    actual = getattr(dist, method)(u_input, dist.example_params())
    assert np.allclose(
        actual, expected, rtol=RTOL, atol=COPULA_ATOL
    ), f"{name} {method} regression: max diff = {np.max(np.abs(np.asarray(actual) - expected))}"


# ──────────────────────────────────────────────────────────────────────
# Archimedean copulas
# ──────────────────────────────────────────────────────────────────────
ARCH_DISTS = [
    ("clayton_copula", clayton_copula),
    ("frank_copula", frank_copula),
    ("gumbel_copula", gumbel_copula),
    ("joe_copula", joe_copula),
    ("amh_copula", amh_copula),
]


@pytest.mark.golden
@pytest.mark.parametrize("name, dist", ARCH_DISTS)
@pytest.mark.parametrize("method", ["copula_cdf", "copula_logpdf", "copula_pdf"])
def test_archimedean(name, dist, method):
    data = _load_npz("archimedean")
    u_input = jnp.asarray(data[f"{name}/u"])
    expected = data[f"{name}/{method}"]
    dim = u_input.shape[1]
    params = dist.example_params(dim=dim)
    actual = getattr(dist, method)(u_input, params)
    assert np.allclose(
        actual, expected, rtol=RTOL, atol=ATOL
    ), f"{name} {method} regression: max diff = {np.max(np.abs(np.asarray(actual) - expected))}"
