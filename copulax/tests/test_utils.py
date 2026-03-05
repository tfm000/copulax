import jax
import pytest

from copulax import get_random_key, get_local_random_key


def _check_valid_key(key, s: str) -> None:
    """Checks if the key is a valid JAX PRNGKey."""
    assert isinstance(key, jax.Array), f"{s} key is not a valid JAX PRNGKey."


def test_get_local_random_key():
    # testing local method
    local_key = get_local_random_key()
    _check_valid_key(local_key, "Local")


def test_wrapper_get_local_random_key():
    # testing wrapper local method
    wrapper_local_key = get_random_key()
    _check_valid_key(wrapper_local_key, "Wrapper Local")
