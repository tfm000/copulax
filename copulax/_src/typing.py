"""Custom type aliases for copulAX.

Annotation convention
---------------------

The alias records *where* a value sits relative to the public API
boundary, not its dimensionality alone.

Arguments, as you pass them in:

``Scalar``
    A rank-0 or ``1 x 1`` quantity — distribution shape, location,
    scale and skewness parameters, copula dependence parameters, and
    similar single-valued inputs.

``ArrayLike``
    A rank-carrying or broadcasting quantity — sample data, evaluation
    points, quantiles, residual series, coefficient vectors, matrices.

Everything past that boundary:

``Array``
    Return types, locals, stored parameters, and internal signatures
    that only ever receive values copulAX has already coerced with
    ``jnp.asarray``. An internal argument stays ``Array`` even when it
    holds a single number — ``Scalar`` marks an un-coerced input, not
    dimensionality on its own.

``Scalar`` and ``ArrayLike`` accept the same set of types, so choosing
between them is documentation only. ``Array`` is narrower than both: it
rules out the plain Python and NumPy forms a caller may supply.
"""

from jax.typing import ArrayLike

Scalar = float | int | ArrayLike
