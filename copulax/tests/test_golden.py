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
# Copulas involve compositions of CDFs/PPFs across multiple distributions,
# so minor numerical drift (e.g. betainc changes) can accumulate.
COPULA_ATOL = 1e-5


def _load_npz(name):
    """Load a golden fixture .npz file and return as nested dict."""
    path = os.path.join(GOLDEN_DIR, f"{name}.npz")
    if not os.path.exists(path):
        pytest.skip(f"Golden fixture {path} not found. Run generate_golden.py first.")
    data = np.load(path, allow_pickle=True)
    return data


def _get_array(data, key):
    """Get an array from the npz data by slash-separated key."""
    return data[key]


# ──────────────────────────────────────────────────────────────────────
# Univariate golden regression tests
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
def test_univariate_logpdf(name, dist):
    data = _load_npz("univariate")
    x = jnp.asarray(_get_array(data, f"{name}/x"))
    expected = _get_array(data, f"{name}/logpdf")
    params = dist.example_params()
    actual = dist.logpdf(x, params)
    assert np.allclose(
        actual, expected, rtol=RTOL, atol=ATOL
    ), f"{name} logpdf regression: max diff = {np.max(np.abs(np.asarray(actual) - expected))}"


@pytest.mark.golden
@pytest.mark.parametrize("name, dist", UVT_DISTS)
def test_univariate_pdf(name, dist):
    data = _load_npz("univariate")
    x = jnp.asarray(_get_array(data, f"{name}/x"))
    expected = _get_array(data, f"{name}/pdf")
    params = dist.example_params()
    actual = dist.pdf(x, params)
    assert np.allclose(
        actual, expected, rtol=RTOL, atol=ATOL
    ), f"{name} pdf regression: max diff = {np.max(np.abs(np.asarray(actual) - expected))}"


# ──────────────────────────────────────────────────────────────────────
# Multivariate golden regression tests
# ──────────────────────────────────────────────────────────────────────
MVT_DISTS = [
    ("mvt_normal", mvt_normal),
    ("mvt_student_t", mvt_student_t),
    ("mvt_gh", mvt_gh),
    ("mvt_skewed_t", mvt_skewed_t),
]


@pytest.mark.golden
@pytest.mark.parametrize("name, dist", MVT_DISTS)
def test_multivariate_logpdf(name, dist):
    data = _load_npz("multivariate")
    x = jnp.asarray(_get_array(data, f"{name}/x"))
    expected = _get_array(data, f"{name}/logpdf")
    params = dist.example_params()
    actual = dist.logpdf(x, params=params)
    assert np.allclose(
        actual, expected, rtol=RTOL, atol=ATOL
    ), f"{name} logpdf regression: max diff = {np.max(np.abs(np.asarray(actual) - expected))}"


@pytest.mark.golden
@pytest.mark.parametrize("name, dist", MVT_DISTS)
def test_multivariate_pdf(name, dist):
    data = _load_npz("multivariate")
    x = jnp.asarray(_get_array(data, f"{name}/x"))
    expected = _get_array(data, f"{name}/pdf")
    params = dist.example_params()
    actual = dist.pdf(x, params=params)
    assert np.allclose(
        actual, expected, rtol=RTOL, atol=ATOL
    ), f"{name} pdf regression: max diff = {np.max(np.abs(np.asarray(actual) - expected))}"


# ──────────────────────────────────────────────────────────────────────
# Copula golden regression tests (elliptical)
# ──────────────────────────────────────────────────────────────────────
COPULA_DISTS = [
    ("gaussian_copula", gaussian_copula),
    ("student_t_copula", student_t_copula),
]


@pytest.mark.golden
@pytest.mark.parametrize("name, dist", COPULA_DISTS)
def test_copula_logpdf(name, dist):
    data = _load_npz("copulas")
    expected = _get_array(data, f"{name}/copula_logpdf")
    params = dist.example_params()
    u_input = jnp.asarray(_get_array(data, f"{name}/u"))
    actual = dist.copula_logpdf(u_input, params)
    assert np.allclose(
        actual, expected, rtol=RTOL, atol=COPULA_ATOL
    ), f"{name} copula_logpdf regression: max diff = {np.max(np.abs(np.asarray(actual) - expected))}"


@pytest.mark.golden
@pytest.mark.parametrize("name, dist", COPULA_DISTS)
def test_copula_pdf(name, dist):
    data = _load_npz("copulas")
    expected = _get_array(data, f"{name}/copula_pdf")
    params = dist.example_params()
    u_input = jnp.asarray(_get_array(data, f"{name}/u"))
    actual = dist.copula_pdf(u_input, params)
    assert np.allclose(
        actual, expected, rtol=RTOL, atol=COPULA_ATOL
    ), f"{name} copula_pdf regression: max diff = {np.max(np.abs(np.asarray(actual) - expected))}"


# ──────────────────────────────────────────────────────────────────────
# Archimedean copula golden regression tests
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
def test_archimedean_copula_cdf(name, dist):
    data = _load_npz("archimedean")
    u_input = jnp.asarray(_get_array(data, f"{name}/u"))
    expected = _get_array(data, f"{name}/copula_cdf")
    dim = u_input.shape[1]
    params = dist.example_params(dim=dim)
    actual = dist.copula_cdf(u_input, params)
    assert np.allclose(
        actual, expected, rtol=RTOL, atol=ATOL
    ), f"{name} copula_cdf regression: max diff = {np.max(np.abs(np.asarray(actual) - expected))}"


@pytest.mark.golden
@pytest.mark.parametrize("name, dist", ARCH_DISTS)
def test_archimedean_copula_logpdf(name, dist):
    data = _load_npz("archimedean")
    u_input = jnp.asarray(_get_array(data, f"{name}/u"))
    expected = _get_array(data, f"{name}/copula_logpdf")
    dim = u_input.shape[1]
    params = dist.example_params(dim=dim)
    actual = dist.copula_logpdf(u_input, params)
    assert np.allclose(
        actual, expected, rtol=RTOL, atol=ATOL
    ), f"{name} copula_logpdf regression: max diff = {np.max(np.abs(np.asarray(actual) - expected))}"


@pytest.mark.golden
@pytest.mark.parametrize("name, dist", ARCH_DISTS)
def test_archimedean_copula_pdf(name, dist):
    data = _load_npz("archimedean")
    u_input = jnp.asarray(_get_array(data, f"{name}/u"))
    expected = _get_array(data, f"{name}/copula_pdf")
    dim = u_input.shape[1]
    params = dist.example_params(dim=dim)
    actual = dist.copula_pdf(u_input, params)
    assert np.allclose(
        actual, expected, rtol=RTOL, atol=ATOL
    ), f"{name} copula_pdf regression: max diff = {np.max(np.abs(np.asarray(actual) - expected))}"
