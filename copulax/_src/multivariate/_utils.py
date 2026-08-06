"""File containing utility functions for multivariate distributions."""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike, DTypeLike


def _multivariate_input(
    x: ArrayLike, dtype: DTypeLike = float
) -> tuple[Array, tuple[int, int], int, int]:
    """Ensures all input arrays are of the same dtype and (n, d) shape."""
    x_arr: Array = jnp.asarray(x, dtype=dtype)
    xshape = x_arr.shape
    n, d = xshape
    return x_arr.reshape((n, d)), (n, 1), n, d
