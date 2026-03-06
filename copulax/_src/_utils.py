"""Contains utility functions for the copulax package."""
from jax import random
import jax.numpy as jnp
import os
import sys


def _type_check_pos_int(value: int, name: str) -> None:
    """Check if the value is an integer."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


###############################################################################
# Random Key Generation
###############################################################################
def get_local_random_key(bytestring_size: int = 7) -> random.key:
    """Returns a random key for use in JAX functions.

    This function uses a hardware based approach to generate a random 
    seed. This involves generating random bytes using os.urandom and 
    converting the byte string to an integer. This method is usually 
    faster than the API method and can be used if reliance on a third 
    party is not desired. Quality of randomness is dependent upon the OS 
    implementation.

    Args:
        bytestring_size (int, optional): The length of the byte string 
            to generate using os.urandom. If the resultant integer is 
            outside the bounds of int64, the integer is truncated to fit 
            within the bounds as to be compatible with jax. 
            The default is 7.

    Note:
        Not jitable.
        No gradients.
    
    Returns:
        A random key for use in JAX functions.
    """
    _type_check_pos_int(bytestring_size, "bytestring_size")
    # generating raw seed int from random bytes
    byte_str: str = os.urandom(bytestring_size)
    seed: int = int.from_bytes(bytes=byte_str, byteorder=sys.byteorder, 
                               signed=True)

    # ensuring the seed is within the bounds of int64
    int64_bounds = jnp.iinfo(jnp.int64)
    if not (int64_bounds.min <= seed <= int64_bounds.max):
        # generated integer is outside the bounds of int64, hence truncating
        seed_str: str = str(seed)
        relevant_bound = int64_bounds.max if seed > 0 else int64_bounds.min
        truncated_length = len(str(relevant_bound)) - 1
        seed = int(seed_str[:truncated_length])
    return random.key(seed)


def get_random_key(bytestring_size: int = 7) -> random.key:
    """Returns a random key for use in JAX functions.

    Uses a hardware based approach to generate a random seed via 
    os.urandom and converts the byte string to an integer. Quality of 
    randomness is dependent upon the OS implementation.

    See Also:
        get_local_random_key

    Note:
        Not jitable.
        No gradients.

    Args:
        bytestring_size (int, optional): The length of the byte string 
            to generate using os.urandom. If the resultant integer is 
            outside the bounds of int64, the integer is truncated to fit 
            within the bounds as to be compatible with jax. 
            The default is 7.
    
    Returns:
        A random key for use in JAX functions.
    """
    return get_local_random_key(bytestring_size=bytestring_size)


def _resolve_key(key):
    """Resolve a random key, generating one lazily if None."""
    if key is None:
        return get_local_random_key()
    return key