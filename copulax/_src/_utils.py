"""Contains utility functions for the copulax package."""
from jax import random
import jax.numpy as jnp
import requests
import os
import sys
import logging


def _type_check_pos_int(value: int, name: str) -> None:
    """Check if the value is an integer."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


###############################################################################
# Random Key Generation
###############################################################################
def get_api_random_key(max_attempts: int = 10) -> random.key:
    """Returns a random key for use in JAX functions.

    This function connects to random.org to obtain a true random seed 
    using atmospheric noise. This involves making a GET request to the 
    random.org API. This can be slow and if being called many times in a 
    loop, may not be desirable. It also involves using a third party 
    source for randomness, which may not be desirable for all 
    applications.

    Note:
        https://www.random.org/
        Not jitable.
        No gradients.

    Args:
        max_attempts (int, optional): The maximum number of attempts to 
            connect to random.org to obtain a true random seed. If a 
            connection cannot be established after this threshold, an 
            error is raised.

    Returns:
        A random key for use in JAX functions.
    """
    _type_check_pos_int(max_attempts, "max_attempts")
    # attempt to connect to random.org to obtain a true random seed
    url: str = "https://www.random.org/integers/?num=1&min=-1000000000&max=1000000000&col=1&base=10&format=plain&rnd=new"
    attempts: int = 0
    seed = None
    while attempts < max_attempts:
        response = requests.get(url)
        if response.status_code == 200:
            seed: int = int(response.text)
            break
        attempts += 1
    
    if seed is None:
        raise ConnectionError(f"Unable to connect to random.org after max = {max_attempts} attempts.")
    return random.key(seed)


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


def get_random_key(local: bool = True, bytestring_size: int = 7, 
                   max_attempts: int = 10) -> random.key:
    """Returns a random key for use in JAX functions.

    Wrapper for the different copulAX for generating a random key. These are:
        1. get_local_random_key; Use a hardware based approach to 
        generate a random seed. This is the default method and involves 
        generating random bytes using os.urandom and converting the byte 
        string to an integer. This method is usually faster than the API 
        method and can be used if reliance on a third party is not 
        desired. Quality of randomness is dependent upon the OS 
        implementation.
        2. get_api_random_key; An API based approach which connects to 
        random.org to obtain a true random seed using atmospheric noise. 
        This involves making a GET request to the random.org API. This 
        can be slow and if being called many times in a loop, may not be 
        desirable. It also involves using a third party source for 
        randomness, which may not be desirable for all applications.

    See Also:
        get_local_random_key, get_api_random_key

    Note:
        https://www.random.org/
        Not jitable.
        No gradients.

    Args:
        local (bool, optional): If True, use the local method to 
            generate a random key. If False, use the API method to 
            generate a random key.
        bytestring_size (int, optional): Only used if local is True. The 
            length of the byte string to generate using os.urandom. If 
            the resultant integer is outside the bounds of int64, the 
            integer is truncated to fit within the bounds as to be 
            compatible with jax. The default is 7.
        max_attempts (int, optional): Only used if local is False. The 
            maximum number of attempts to connect to random.org to 
            obtain a true random seed. If a connection cannot be 
            established after this threshold, the local generation 
            method is used. The default is 10.
    
    Returns:
        A random key for use in JAX functions.
    """
    if local:
        # use the local method to generate a random key
        return get_local_random_key(bytestring_size=bytestring_size)
    
    # use the API method to generate a random key
    try:
        return get_api_random_key(max_attempts=max_attempts)
    except ConnectionError as e:
        msg: str = f"{str(e)}\nUsing local method instead."
        logging.warning(msg)
        return get_local_random_key(bytestring_size=bytestring_size)


DEFAULT_RANDOM_KEY = get_local_random_key()