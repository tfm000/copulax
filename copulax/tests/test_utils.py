from jax._src.prng import PRNGKeyArray
import pytest

from copulax import get_random_key, get_api_random_key, get_local_random_key

def _check_valid_key(key, s: str) -> None:
    """Checks if the key is a valid JAX PRNGKey."""
    assert isinstance(key, PRNGKeyArray), f"{s} key is not a valid JAX PRNGKey."


def test_get_api_random_key():
    # testing api method
    try:
        api_key = get_api_random_key()
        _check_valid_key(api_key, "API")
    except ConnectionError as e:
        if str(e) == "Unable to connect to random.org after max = 10 attempts.":
            pytest.skip("Skipping test due to connection error with random.org API.")
        else:
            pytest.fail(f"Failed to get API key: {e}")


def test_wrapper_get_api_random_key():
    try:
        wrapper_api_key = get_random_key(local=False)
        _check_valid_key(wrapper_api_key, "Wrapper API")
    except ConnectionError as e:
        if str(e) == "Unable to connect to random.org after max = 10 attempts.":
            pytest.skip("Skipping test due to connection error with random.org API.")
        else:
            pytest.fail(f"Failed to get API key: {e}")


def test_get_local_random_key():
    # testing local method
    local_key = get_local_random_key()
    _check_valid_key(local_key, "Local")


def test_wrapper_get_local_random_key():
    # testing wrapper local method
    wrapper_local_key = get_random_key(local=True)
    _check_valid_key(wrapper_local_key, "Wrapper Local")
