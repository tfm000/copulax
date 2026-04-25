"""Tests for ``copulax.get_random_key`` and the JIT-safety contract."""

import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from copulax import get_random_key
from copulax.univariate import normal as _normal


class TestGetRandomKey:
    """Tests for the public PRNG-key utility."""

    def test_returns_valid_prng_key(self):
        """Output is a JAX PRNG key usable by ``jax.random.*``."""
        key = get_random_key()
        assert isinstance(key, jax.Array)
        _ = jr.normal(key, (3,))

    def test_successive_calls_outside_jit_differ(self):
        """Two unwrapped calls return distinct keys."""
        s1 = jr.normal(get_random_key(), (5,))
        s2 = jr.normal(get_random_key(), (5,))
        assert not jnp.allclose(s1, s2)

    @pytest.mark.parametrize("size", [1, 4, 7, 16])
    def test_accepts_custom_bytestring_size(self, size):
        """Positive ``bytestring_size`` values produce usable keys."""
        _ = jr.normal(get_random_key(bytestring_size=size), (2,))

    def test_rejects_non_integer_bytestring_size(self):
        """Non-integer ``bytestring_size`` raises ``TypeError``."""
        with pytest.raises(TypeError):
            get_random_key(bytestring_size=3.5)
        with pytest.raises(TypeError):
            get_random_key(bytestring_size="seven")

    @pytest.mark.parametrize("size", [0, -1, -7])
    def test_rejects_non_positive_bytestring_size(self, size):
        """Non-positive ``bytestring_size`` raises ``ValueError``."""
        with pytest.raises(ValueError):
            get_random_key(bytestring_size=size)

    def test_successive_calls_inside_jit_differ(self):
        """Successive calls inside ``@jax.jit`` return distinct keys."""
        @jax.jit
        def sample():
            return jr.normal(get_random_key(), (3,))

        s1, s2, s3 = sample(), sample(), sample()
        assert not jnp.allclose(s1, s2)
        assert not jnp.allclose(s2, s3)
        assert not jnp.allclose(s1, s3)

    def test_rvs_with_default_key_inside_jit_differs(self):
        """``dist.rvs(key=None)`` produces fresh samples per JIT call."""
        @jax.jit
        def sample():
            return _normal.rvs(
                size=(5,),
                params={"mu": jnp.array(0.0), "sigma": jnp.array(1.0)},
                key=None,
            )

        assert not jnp.allclose(sample(), sample())

    def test_vmap_limitation_is_documented(self):
        """``pure_callback`` is hoisted out of ``vmap`` (returns
        identical keys across the batch)."""
        keys = jax.vmap(lambda _: get_random_key())(jnp.arange(5))
        samples = jax.vmap(lambda k: jr.normal(k, ()))(keys)
        assert jnp.allclose(samples, samples[0])

    def test_works_with_x64_disabled(self):
        """Regression: ``get_random_key`` must work in JAX's default
        x64=off mode. The conftest forces x64 on for the rest of the
        suite, so this case is exercised in a fresh subprocess."""
        script = textwrap.dedent(
            """
            import jax
            assert not jax.config.jax_enable_x64
            from copulax import get_random_key
            from jax.random import split
            k = get_random_key()
            k, sk = split(k)

            @jax.jit
            def f():
                return get_random_key()

            k1, k2 = f(), f()
            assert (k1 != k2).any()
            print('OK')
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"x64=off subprocess failed:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OK" in result.stdout
