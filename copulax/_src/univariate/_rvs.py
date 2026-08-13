from collections.abc import Callable, Sequence
from typing import cast

from jax import Array, lax, random

from copulax._src.typing import Scalar

# ``jax.random`` canonicalises a bare integer size at runtime, but its
# ``Shape`` alias only spells the sequence form, so the scalar half of the
# library's documented ``tuple | Scalar`` size contract needs a cast at
# every jax sampling call below.


def inverse_transform_sampling(
    ppf_func: Callable, shape: tuple | Scalar, params: dict, key: Array
) -> Array:
    """Generate random samples using the inverse transform sampling method.

    Args:
        ppf_func: The percent point function of the distribution.
        shape: The shape of the output array.
        params: The parameters of the distribution.

    Returns:
        Array: The generated random samples.
    """
    eps: float = 1e-5
    u: Array = random.uniform(
        key=key, shape=cast(Sequence[int], shape), minval=eps, maxval=1 - eps
    )
    return ppf_func(q=u, params=params).reshape(shape)


def mean_variance_sampling(
    key: Array,
    W: Array,
    shape: tuple | Scalar,
    mu: Array,
    sigma: Array,
    gamma: Array,
) -> Array:
    """Generate samples from a mean-variance normal mixture:
    X = mu + W*gamma + sqrt(W)*sigma*Z."""
    Z: Array = random.normal(key=key, shape=cast(Sequence[int], shape))
    m: Array = mu + W * gamma
    s: Array = lax.sqrt(W) * sigma * Z
    X: Array = m + s
    return X.reshape(shape)
