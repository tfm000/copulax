# from jax._src.prng import PRNGKeyArray

# from copulax import get_random_key, get_api_random_key, get_local_random_key

# def _check_valid_key(key, s: str) -> None:
#     """Checks if the key is a valid JAX PRNGKey."""
#     assert isinstance(key, PRNGKeyArray), f"{s} key is not a valid JAX PRNGKey."

# def test_get_random_key():
#     """Tests the get_random_key function."""
#     # testing api method
#     api_key = get_api_random_key()
#     _check_valid_key(api_key, "API")
#     wrapper_api_key = get_random_key(local=False)
#     _check_valid_key(wrapper_api_key, "Wrapper API")

#     # testing local method
#     local_key = get_local_random_key()
#     _check_valid_key(local_key, "Local")
#     wrapper_local_key = get_random_key(local=True)
#     _check_valid_key(wrapper_local_key, "Wrapper Local")