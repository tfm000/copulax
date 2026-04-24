"""Rigorous tests for ``copulax._src._utils.get_random_key``.

The function is JIT-safe via ``jax.pure_callback``: every call inside a
JIT-compiled function produces a fresh seed at runtime instead of baking
the ``os.urandom`` result at trace time. These tests lock that contract
in place so a future regression cannot silently reintroduce trace-time
seed pollution.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from copulax import get_random_key
from copulax.univariate import normal as _normal


class TestGetRandomKey:
    """Behavioural tests for the public PRNG-key utility."""

    def test_returns_valid_prng_key(self):
        """``get_random_key()`` returns a JAX PRNG key usable downstream."""
        key = get_random_key()
        assert isinstance(key, jax.Array)
        # downstream consumer should not raise
        _ = jr.normal(key, (3,))

    def test_successive_calls_outside_jit_differ(self):
        """Two unwrapped calls return distinct keys (entropy sanity check)."""
        k1 = get_random_key()
        k2 = get_random_key()
        s1 = jr.normal(k1, (5,))
        s2 = jr.normal(k2, (5,))
        assert not jnp.allclose(s1, s2)

    @pytest.mark.parametrize("size", [1, 4, 7, 16])
    def test_accepts_custom_bytestring_size(self, size):
        """Valid positive ``bytestring_size`` values produce usable keys."""
        key = get_random_key(bytestring_size=size)
        # any valid PRNG key must be consumable by jax.random.normal
        _ = jr.normal(key, (2,))

    def test_rejects_non_integer_bytestring_size(self):
        """Float / string inputs raise ``TypeError`` from the size guard."""
        with pytest.raises(TypeError):
            get_random_key(bytestring_size=3.5)
        with pytest.raises(TypeError):
            get_random_key(bytestring_size="seven")

    @pytest.mark.parametrize("size", [0, -1, -7])
    def test_rejects_non_positive_bytestring_size(self, size):
        """Zero / negative ``bytestring_size`` raises ``ValueError``."""
        with pytest.raises(ValueError):
            get_random_key(bytestring_size=size)

    def test_successive_calls_inside_jit_differ(self):
        """JIT-safety contract: fresh seed per call inside ``@jax.jit``.

        Regression guard. Without the ``jax.pure_callback`` wrapper this
        test fails: ``os.urandom`` would be evaluated at trace time and
        every successive call to the same compiled function would reuse
        the baked-in seed.
        """
        @jax.jit
        def sample():
            return jr.normal(get_random_key(), (3,))

        s1 = sample()
        s2 = sample()
        s3 = sample()
        assert not jnp.allclose(s1, s2)
        assert not jnp.allclose(s2, s3)
        assert not jnp.allclose(s1, s3)

    def test_rvs_with_default_key_inside_jit_differs(self):
        """End-to-end JIT-safety via the ``rvs`` ``key=None`` path.

        Every distribution's ``rvs`` falls back to ``_resolve_key(None)``
        which now routes through the JIT-safe ``get_random_key``. This
        test exercises the actual library entry point, not just the
        utility in isolation.
        """
        @jax.jit
        def sample():
            return _normal.rvs(
                size=(5,),
                params={"mu": jnp.array(0.0), "sigma": jnp.array(1.0)},
                key=None,
            )

        s1 = sample()
        s2 = sample()
        assert not jnp.allclose(s1, s2)

    def test_vmap_limitation_is_documented(self):
        """``jax.pure_callback`` is hoisted out of ``vmap``.

        This encodes the documented limitation as a regression guard: if
        a future JAX release makes ``pure_callback`` vmap-aware, this
        test should be updated to assert distinctness rather than silently
        masking the new behaviour.
        """
        keys = jax.vmap(lambda _: get_random_key())(jnp.arange(5))
        samples = jax.vmap(lambda k: jr.normal(k, ()))(keys)
        # all five samples come from the same baked-in key, so they're equal
        assert jnp.allclose(samples, samples[0])
