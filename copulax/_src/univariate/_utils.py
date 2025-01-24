"""File containing utility functions for univariate distributions."""
import jax.numpy as jnp
from jax._src.typing import ArrayLike


def _univariate_input(x: ArrayLike, dtype=float) -> tuple[jnp.ndarray, tuple[int]]:
    """Ensures all input arrays are of the same dtype and (n, 1) shape."""
    x_arr: jnp.array = jnp.asarray(x, dtype=dtype)
    xshape = x_arr.shape
    return x_arr.reshape((x_arr.size, 1)), xshape
