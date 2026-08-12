"""Custom type aliases for copulAX.

Annotation convention
---------------------

``Scalar``
    Public inputs that are mathematically rank-0 — distribution shape,
    location, scale and skewness parameters, copula dependence
    parameters, and similar single-valued quantities.

``ArrayLike``
    Public inputs that carry rank — sample data, evaluation points,
    quantiles, residual series, coefficient vectors and matrices.

``Array``
    Internal values, locals and return types. Everything copulAX
    produces is a concrete :class:`jax.Array`, whatever was passed in.

``Scalar`` and ``ArrayLike`` describe the same set of accepted types, so
the choice between them carries no runtime or type-checking
consequence — it documents the expected dimensionality of an argument.
"""

from jax.typing import ArrayLike

Scalar = float | int | ArrayLike
