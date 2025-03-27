from typing import Callable
from jax._src.typing import Array
import jax.numpy as jnp
from jax import lax, random


def inverse_transform_sampling(ppf_func: Callable, shape: tuple, params: dict, key: Array) -> Array:
    """Generate random samples using the inverse transform sampling method.

    Args:
        ppf_func: The percent point function of the distribution.
        shape: The shape of the output array.
        params: The parameters of the distribution.

    Returns:
        Array: The generated random samples.
    """
    # num_samples = jnp.asarray(shape).prod()
    eps: float = 1e-2
    u: jnp.ndarray = random.uniform(key=key, shape=shape, minval=eps, maxval=1-eps)
    return ppf_func(q=u, **params).reshape(shape)


def mean_variance_sampling(key: Array, W: jnp.ndarray, shape: tuple, mu: float, sigma: float, gamma: float) -> jnp.ndarray:
    Z: jnp.ndarray = random.normal(key=key, shape=shape)
    m: jnp.ndarray = mu + W * gamma
    s: jnp.ndarray = lax.sqrt(W) * sigma * Z
    s = lax.mul(lax.sqrt(W) * sigma, Z)
    X: jnp.ndarray = m + s
    return X.reshape(shape)