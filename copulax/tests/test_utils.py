import jax
import pytest

from copulax import get_random_key, get_local_random_key


class TestRandomKey:
    """Tests for random key generation utilities."""

    @staticmethod
    def _check_valid_key(key, s: str) -> None:
        assert isinstance(key, jax.Array), f"{s} key is not a valid JAX PRNGKey."

    def test_get_local_random_key(self):
        local_key = get_local_random_key()
        self._check_valid_key(local_key, "Local")

    def test_get_random_key(self):
        wrapper_key = get_random_key()
        self._check_valid_key(wrapper_key, "Wrapper")
