"""Tests for Distribution __eq__ and __hash__ behaviour.

Verifies that:
- Equality is based on type + stored parameters, not name.
- Distributions are not hashable (contain JAX arrays).
- Edge cases (None params, mixed types, non-Distribution comparands)
  behave correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copulax.univariate import normal, student_t, uniform
from copulax._src.univariate.normal import Normal
from copulax._src.univariate.student_t import StudentT
from copulax.multivariate import mvt_normal
from copulax._src.multivariate.mvt_normal import MvtNormal
from copulax.copulas import gaussian_copula, clayton_copula


@pytest.fixture(autouse=True, scope="module")
def _enable_x64():
    jax.config.update("jax_enable_x64", True)
    yield


# ---------------------------------------------------------------------------
# Univariate equality
# ---------------------------------------------------------------------------

class TestUnivariateEquality:
    """Equality for univariate distributions."""

    def test_unparameterized_same_class_equal(self):
        """Two unparameterized singletons of the same class are equal."""
        a = Normal(name="a")
        b = Normal(name="b")
        assert a == b

    def test_fitted_same_params_equal(self):
        """Two fitted instances with identical params are equal."""
        a = Normal(name="x", mu=1.0, sigma=2.0)
        b = Normal(name="y", mu=1.0, sigma=2.0)
        assert a == b

    def test_fitted_different_params_not_equal(self):
        """Two fitted instances with different params are not equal."""
        a = Normal(name="x", mu=1.0, sigma=2.0)
        b = Normal(name="x", mu=1.0, sigma=3.0)
        assert a != b

    def test_same_params_different_names_equal(self):
        """Name is cosmetic — same type and params means equal."""
        a = Normal(name="model_v1", mu=5.0, sigma=1.0)
        b = Normal(name="model_v2", mu=5.0, sigma=1.0)
        assert a == b

    def test_different_classes_not_equal(self):
        """Different distribution types are never equal."""
        a = Normal(name="a", mu=0.0, sigma=1.0)
        # StudentT also has mu and sigma but is a different type
        b = StudentT(name="a", nu=10.0, mu=0.0, sigma=1.0)
        assert a != b

    def test_unparameterized_vs_parameterized_not_equal(self):
        """An unfitted template != a fitted instance of the same class."""
        a = Normal(name="Normal")
        b = Normal(name="Normal", mu=0.0, sigma=1.0)
        assert a != b

    def test_symmetry(self):
        """a == b implies b == a."""
        a = Normal(name="x", mu=1.0, sigma=2.0)
        b = Normal(name="y", mu=1.0, sigma=2.0)
        assert (a == b) == (b == a)

    def test_non_distribution_returns_not_implemented(self):
        """Comparing with a non-Distribution returns NotImplemented."""
        a = Normal(name="a", mu=0.0, sigma=1.0)
        result = a.__eq__("not a distribution")
        assert result is NotImplemented

    def test_non_distribution_inequality(self):
        """!= against a non-Distribution should not raise."""
        a = Normal(name="a", mu=0.0, sigma=1.0)
        assert a != 42
        assert a != "Normal"
        assert a != None


# ---------------------------------------------------------------------------
# Multivariate equality
# ---------------------------------------------------------------------------

class TestMultivariateEquality:
    """Equality for multivariate distributions."""

    def test_same_params_equal(self):
        mu = jnp.array([[1.0], [2.0]])
        sigma = jnp.eye(2)
        a = MvtNormal(name="a", mu=mu, sigma=sigma)
        b = MvtNormal(name="b", mu=mu, sigma=sigma)
        assert a == b

    def test_different_params_not_equal(self):
        mu = jnp.array([[1.0], [2.0]])
        a = MvtNormal(name="a", mu=mu, sigma=jnp.eye(2))
        b = MvtNormal(name="a", mu=mu, sigma=2.0 * jnp.eye(2))
        assert a != b

    def test_unparameterized_equal(self):
        a = MvtNormal(name="a")
        b = MvtNormal(name="b")
        assert a == b


# ---------------------------------------------------------------------------
# Copula equality
# ---------------------------------------------------------------------------

class TestCopulaEquality:
    """Equality for copula distributions."""

    def test_same_copula_params_equal(self):
        """Copulas with same marginals and copula params are equal."""
        params = gaussian_copula.example_params(dim=3)
        a = gaussian_copula._fitted_instance(params, name="a")
        b = gaussian_copula._fitted_instance(params, name="b")
        assert a == b

    def test_different_copula_params_not_equal(self):
        """Copulas with different copula params are not equal."""
        params_a = gaussian_copula.example_params(dim=3)
        params_b = gaussian_copula.example_params(dim=3)
        params_b["copula"]["sigma"] = 2.0 * params_b["copula"]["sigma"]
        a = gaussian_copula._fitted_instance(params_a, name="a")
        b = gaussian_copula._fitted_instance(params_b, name="b")
        assert a != b

    def test_different_marginals_not_equal(self):
        """Copulas with different marginal params are not equal."""
        params_a = gaussian_copula.example_params(dim=2)
        params_b = gaussian_copula.example_params(dim=2)
        # Modify one marginal's mu
        dist, mparams = params_b["marginals"][0]
        modified = {k: v for k, v in mparams.items()}
        modified["mu"] = mparams["mu"] + 10.0
        params_b["marginals"] = ((dist, modified),) + params_b["marginals"][1:]
        a = gaussian_copula._fitted_instance(params_a, name="a")
        b = gaussian_copula._fitted_instance(params_b, name="b")
        assert a != b

    def test_archimedean_same_params_equal(self):
        """Archimedean copulas with same params are equal."""
        params = clayton_copula.example_params(dim=3)
        a = clayton_copula._fitted_instance(params, name="a")
        b = clayton_copula._fitted_instance(params, name="b")
        assert a == b


# ---------------------------------------------------------------------------
# Hashing (identity-based, not value-based)
# ---------------------------------------------------------------------------

class TestHashing:
    """Hash is object-identity based (required by JAX/equinox for tracing).

    Two equal distributions may have different hashes — this is valid
    because hash equality is a necessary but not sufficient condition
    for value equality.
    """

    def test_hash_is_identity_based(self):
        """Two equal instances have different hashes (identity-based)."""
        a = Normal(name="x", mu=1.0, sigma=2.0)
        b = Normal(name="y", mu=1.0, sigma=2.0)
        assert a == b
        # Different objects → different id → different hash
        assert hash(a) != hash(b)

    def test_same_object_same_hash(self):
        """Same object always has the same hash."""
        a = Normal(name="x", mu=1.0, sigma=2.0)
        assert hash(a) == hash(a)

    def test_hash_no_longer_name_based(self):
        """Hash is NOT based on name (old behaviour)."""
        a = Normal(name="same_name", mu=1.0, sigma=2.0)
        b = Normal(name="same_name", mu=1.0, sigma=2.0)
        # Same name but different objects → different hash
        assert hash(a) != hash(b)
