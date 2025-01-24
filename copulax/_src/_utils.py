"""Contains utility functions for the copulax package."""
from jax import random
import requests
import os
import sys


def get_random_key(max_attempts: int = 0) -> random.key:
    """Returns a random key for use in JAX functions.

    Two different generation methods are used to obtain a random key: 
    1. Connect to random.org to obtain a true random seed using atmospheric 
    noise. This involves making a GET request to the random.org API. This can 
    be slow and if being called many times in a loop, may not be desirable.
    2. Use a hardware based approach to generate a random seed. This is the 
    default method and involves generating random bytes using os.urandom and 
    converting the byte string to an integer. This method is faster and can be 
    used if the first method is not desired.

    Args:
    max_attempts : int, optional
        The maximum number of attempts to connect to random.org to obtain a 
        true random seed. If a connection cannot be established after this 
        threshold, a non-API method is used. Set max_attempts=0 to use the 
        non-API method (os.urandom). The default is 0.
    
    Returns:
        A random key for use in JAX functions.
    """

    # attempt to connect to random.org to obtain a true random seed
    attempts: int = 0
    while attempts < max_attempts:
        response = requests.get("https://www.random.org/integers/?num=1&min=-1000000000&max=1000000000&col=1&base=10&format=plain&rnd=new")
        if response.status_code == 200:
            seed: int = int(response.text)
            break
        attempts += 1
    
    # if unsuccessful use a hardware approach for the seed
    if max_attempts <= attempts:
        # generating random bytes
        byte_str: str = os.urandom(8)

        # convert the byte string to an integer
        raw_int: int = int.from_bytes(bytes=byte_str, byteorder=sys.byteorder, signed=True)
        seed_str: str = str(raw_int)
        if len(seed_str) > 10:
            rand_len: int = int(seed_str.replace('0', '')[-1])
            rand_len = rand_len + 1 if raw_int < 0 else rand_len
            seed_str = seed_str[:rand_len]
        seed: int = int(seed_str)

    return random.key(seed)


DEFAULT_RANDOM_KEY = get_random_key(0)