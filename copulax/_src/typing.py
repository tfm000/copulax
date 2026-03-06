"""Custom type aliases for copulAX."""

from typing import Union
from typing import TypeAlias
from jax.typing import ArrayLike

Scalar = Union[float, int, ArrayLike]
