from typing import Callable
from jax._src.typing import Array
import jax
import jax.numpy as jnp
from typing import Callable

from copulax._src.univariate import uniform


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
    u = uniform.rvs(shape=shape, key=key, a=eps, b=1-eps)
    return ppf_func(q=u, **params).reshape(shape)