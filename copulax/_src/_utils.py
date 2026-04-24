"""Contains utility functions for the copulax package."""
import os
import sys

import jax
from jax import random
import jax.numpy as jnp


def _type_check_pos_int(value: int, name: str) -> None:
    """Check if the value is an integer."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


###############################################################################
# Random Key Generation
###############################################################################
def _host_random_seed(bytestring_size: int) -> int:
    """Host-side fresh seed: ``os.urandom`` + int64-bounds clamp.

    Pulled out of :func:`get_random_key` so it can be wrapped in
    :func:`jax.pure_callback` and remain JIT-safe.
    """
    byte_str: bytes = os.urandom(bytestring_size)
    seed: int = int.from_bytes(bytes=byte_str, byteorder=sys.byteorder,
                               signed=True)

    int64_bounds = jnp.iinfo(jnp.int64)
    if not (int64_bounds.min <= seed <= int64_bounds.max):
        range_size: int = int(int64_bounds.max) - int(int64_bounds.min) + 1
        seed = int(int64_bounds.min) + (seed - int(int64_bounds.min)) % range_size
    return seed


def get_random_key(bytestring_size: int = 7) -> random.key:
    """Returns a fresh JAX PRNG key seeded from ``os.urandom``.

    A hardware-backed seed is drawn from ``os.urandom``, normalised to fit
    inside the int64 range, and used to construct a ``jax.random.key``.
    Quality of randomness is dependent upon the OS implementation.

    The hardware draw is wrapped in :func:`jax.pure_callback`, so successive
    invocations inside a single ``@jax.jit``-compiled function each receive
    a distinct seed at runtime (instead of the seed being baked in at trace
    time, which would happen if ``os.urandom`` were called directly).

    Args:
        bytestring_size (int, optional): The length of the byte string to
            generate via ``os.urandom``. If the resulting integer falls
            outside the int64 range it is reduced modulo the range so that
            the JAX seed remains valid. The default is 7.

    Note:
        - JIT-safe: every call inside a JIT-compiled function produces a
          fresh seed via the host callback.
        - **Not vmap-safe**: ``jax.pure_callback`` is hoisted out of
          ``vmap``, so a vmap'd call returns identical keys across the
          batch. For per-leaf entropy, pass an explicit ``key`` argument
          and split it via :func:`jax.random.split`.
        - No autograd: gradients are not defined.

    Returns:
        A fresh JAX PRNG key.
    """
    _type_check_pos_int(bytestring_size, "bytestring_size")
    seed = jax.pure_callback(
        lambda: _host_random_seed(bytestring_size),
        jax.ShapeDtypeStruct((), jnp.int64),
    )
    return random.key(seed)


def _resolve_key(key):
    """Resolve a random key, generating one lazily if None."""
    if key is None:
        return get_random_key()
    return key
