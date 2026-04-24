"""File containing the copulAX implementation of the continuous uniform
distribution."""

import jax.numpy as jnp
from jax import lax, random
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import _resolve_key
from copulax._src.typing import Scalar


class Uniform(Univariate):
    r"""The continuous uniform distribution.

    The continuous uniform distribution is defined as:

    .. math::

        f(x|a, b) = \frac{1}{b - a}

    where :math:`a` is the lower bound of the distribution and :math:`b` is the
    upper bound.
    """

    a: Array = None
    b: Array = None

    def __init__(self, name="Uniform", *, a=None, b=None):
        """Initialize the Uniform distribution.

        Args:
            name: Display name for the distribution.
            a: Lower bound of the interval.
            b: Upper bound of the interval.
        """
        super().__init__(name)
        self.a = jnp.asarray(a, dtype=float).reshape(()) if a is not None else None
        self.b = jnp.asarray(b, dtype=float).reshape(()) if b is not None else None

    @property
    def _stored_params(self):
        """Return stored parameters if all are set, else None."""
        if self.a is None or self.b is None:
            return None
        return {"a": self.a, "b": self.b}

    @classmethod
    def _params_dict(cls, a: Scalar, b: Scalar) -> dict:
        """Create a parameter dictionary from lower bound ``a`` and upper bound ``b``."""
        d: dict = {"a": a, "b": b}
        return cls._args_transform(d)

    @classmethod
    def _params_to_tuple(cls, params: dict) -> tuple:
        """Extract (a, b) from the parameter dictionary."""
        params = cls._args_transform(params)
        return params["a"], params["b"]

    def example_params(self, *args, **kwargs) -> dict:
        r"""Example parameters for the uniform distribution.

        This is a two parameter family, with the uniform being defined by
        its lower and upper bounds.
        """
        return self._params_dict(a=0.0, b=1.0)

    @classmethod
    def _support(cls, params: dict) -> Array:
        """Return the support ``[a, b]`` from the given parameters."""
        a, b = cls._params_to_tuple(params)
        return jnp.array([a, b])

    def logpdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the log probability density function.

        Returns ``log(1 / (b - a))`` inside the support, ``-inf`` outside.
        """
        params = self._resolve_params(params)
        x, xshape = _univariate_input(x)
        a, b = self._params_to_tuple(params)

        log_pdf: jnp.ndarray = -jnp.log(lax.sub(b, a))
        log_pdf = jnp.where(jnp.logical_and(x >= a, x <= b), log_pdf, -jnp.inf)
        return log_pdf.reshape(xshape)

    def logcdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the log cumulative distribution function."""
        params = self._resolve_params(params)
        return jnp.log(self.cdf(x=x, params=params))

    def cdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the cumulative distribution function."""
        params = self._resolve_params(params)
        x, xshape = _univariate_input(x)
        a, b = self._params_to_tuple(params)

        cdf: jnp.ndarray = (x - a) / (b - a)
        return self._enforce_support_on_cdf(
            x=x, cdf=cdf.reshape(xshape), params=params
        )

    # ppf
    def _ppf(self, q: ArrayLike, params: dict, *args, **kwargs) -> Array:
        """Compute the percent-point function (inverse CDF) via linear interpolation."""
        q, qshape = _univariate_input(q)
        a, b = self._params_to_tuple(params)

        ppf_values: jnp.ndarray = lax.add(a, lax.mul(q, lax.sub(b, a)))
        ppf_values = jnp.where(jnp.logical_and(q >= 0, q <= 1), ppf_values, jnp.nan)
        return ppf_values.reshape(qshape)

    # sampling
    def rvs(self, size: tuple | Scalar, params: dict = None, key=None) -> Array:
        """Generate random variates from the uniform distribution.

        Args:
            size: Shape of the output array.
            params: Distribution parameters. Uses stored parameters if None.
            key: JAX PRNG key. A default key is used if None.

        Returns:
            Array of random samples in ``[a, b)``.
        """
        params = self._resolve_params(params)
        key = _resolve_key(key)
        a, b = self._params_to_tuple(params)
        return random.uniform(key=key, shape=size, minval=a, maxval=b)

    # stats
    def stats(self, params: dict = None) -> dict:
        """Compute distribution statistics (mean, median, variance, std, skewness, kurtosis)."""
        params = self._resolve_params(params)
        a, b = self._params_to_tuple(params)

        mean: Scalar = (a + b) / 2
        variance: Scalar = lax.pow(b - a, 2) / 12
        std: Scalar = jnp.sqrt(variance)
        return self._scalar_transform(
            {
                "mean": mean,
                "median": mean,
                "variance": variance,
                "std": std,
                "skewness": 0.0,
                "kurtosis": -6 / 5,
            }
        )

    # fitting
    _supported_methods = frozenset({"mle"})

    def fit(self, x: ArrayLike, *args, name: str = None, **kwargs):
        r"""Fit the distribution to data via **closed-form** MLE:
        ``â = min(x)``, ``b̂ = max(x)``.

        The closed-form estimator takes no tuning parameters.

        Args:
            x: Input data to fit.
            name: Optional custom name for the fitted instance.

        Returns:
            Uniform: A fitted ``Uniform`` instance.
        """
        x: jnp.ndarray = _univariate_input(x)[0]
        a: Scalar = jnp.min(x)
        b: Scalar = jnp.max(x)
        return self._fitted_instance(self._params_dict(a=a, b=b), name=name)


uniform = Uniform("Uniform")
