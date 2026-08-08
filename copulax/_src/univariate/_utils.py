"""File containing utility functions for univariate distributions."""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike, DTypeLike


def _univariate_input(
    x: ArrayLike, dtype: DTypeLike = float
) -> tuple[Array, tuple[int, ...]]:
    """Ensures all input arrays are of the same dtype and (n, 1) shape."""
    x_arr: Array = jnp.asarray(x, dtype=dtype)
    xshape = x_arr.shape
    return x_arr.reshape((x_arr.size, 1)), xshape
