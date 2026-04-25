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
def _seed_dtype():
    """Return the JAX-canonical signed int dtype for the current
    ``jax_enable_x64`` setting (``jnp.int64`` when on, ``jnp.int32`` when
    off — JAX's default)."""
    return jnp.int64 if jax.config.jax_enable_x64 else jnp.int32


def _host_random_seed(bytestring_size: int) -> int:
    """Host-side fresh seed: ``os.urandom`` clamped (mod) to the
    JAX-canonical signed int range so the value survives transfer
    through ``jax.pure_callback`` regardless of ``jax_enable_x64``."""
    byte_str: bytes = os.urandom(bytestring_size)
    seed: int = int.from_bytes(bytes=byte_str, byteorder=sys.byteorder,
                               signed=True)

    bounds = jnp.iinfo(_seed_dtype())
    if not (bounds.min <= seed <= bounds.max):
        range_size: int = int(bounds.max) - int(bounds.min) + 1
        seed = int(bounds.min) + (seed - int(bounds.min)) % range_size
    return seed


def get_random_key(bytestring_size: int = 7) -> random.key:
    """Returns a fresh JAX PRNG key seeded from ``os.urandom``.

    The hardware draw is wrapped in :func:`jax.pure_callback`, so each
    call inside an ``@jax.jit``-compiled function receives a distinct
    seed at runtime. Quality of randomness depends on the OS, and on
    the ``jax_enable_x64`` setting — when x64 is disabled (JAX's
    default), the seed is clamped to the int32 range; enable x64 for
    the full int64 entropy budget.

    Args:
        bytestring_size (int, optional): Length of the byte string from
            ``os.urandom``. Out-of-range integers are reduced modulo
            the active int range (int64 with x64, int32 without).
            Default ``7``.

    Note:
        Not ``vmap``-safe: ``pure_callback`` is hoisted out of ``vmap``,
        so a vmap'd call returns identical keys across the batch.  Pass
        an explicit ``key`` and ``jax.random.split`` it for per-leaf
        entropy.  No autograd.

    Returns:
        A fresh JAX PRNG key.
    """
    _type_check_pos_int(bytestring_size, "bytestring_size")
    seed = jax.pure_callback(
        lambda: _host_random_seed(bytestring_size),
        jax.ShapeDtypeStruct((), _seed_dtype()),
    )
    return random.key(seed)


def _resolve_key(key):
    """Resolve a random key, generating one lazily if None."""
    if key is None:
        return get_random_key()
    return key
