"""File containing utility functions for multivariate distributions."""
import jax.numpy as jnp
from jax._src.typing import ArrayLike


def _multivariate_input(x: ArrayLike, dtype=float) -> tuple[jnp.ndarray, tuple[int], int]:
    """Ensures all input arrays are of the same dtype and (n, d) shape."""
    x_arr: jnp.array = jnp.asarray(x, dtype=dtype)
    xshape = x_arr.shape
    n, d = xshape
    return x_arr.reshape((n, d)), (n, 1), n, d