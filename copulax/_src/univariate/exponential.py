"""File containing the copulAX implementation of the exponential distribution."""

import jax.numpy as jnp
from jax import random
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import _resolve_key

class Exponential(Univariate):
    r"""The exponential distribution is a continuous distribution that 
    describes the time between events in a Poisson process.

    The exponential distribution is defined as:

    .. math::

        f(x|\lambda) = \lambda e^{-\lambda x}

    where :math:`\lambda` is the rate parameter of the distribution.

    https://en.wikipedia.org/wiki/Exponential_distribution
    """

    lamb: Array = None

    def __init__(self, name="Exponential", *, lamb=None):
        """Initialize the Exponential distribution.

        Args:
            name: Display name for the distribution.
            lamb: Rate parameter (lambda). If provided, stored on the instance.
        """
        super().__init__(name)
        self.lamb = jnp.asarray(lamb, dtype=float).reshape(()) if lamb is not None else None

    @property
    def _stored_params(self):
        """Return stored parameters if all are set, else None."""
        if self.lamb is None:
            return None
        return self._params_dict(self.lamb)

    @classmethod
    def _params_dict(cls, lamb: Scalar) -> dict:
        """Create a parameter dictionary from the rate value."""
        d: dict = {"lamb": lamb}
        return cls._args_transform(d)

    def _params_to_tuple(self, params: dict) -> tuple:
        """Convert a parameter dictionary to a tuple of parameters."""
        return (params["lamb"],)

    def example_params(self, *args, **kwargs) -> dict:
        r"""Return example parameters for the distribution.
        
        This is a single parameter family defined by the rate parameter 
        lambda.
        """
        return self._params_dict(lamb=1.0)

    @classmethod
    def _support(cls, *args, **kwargs) -> Array:
        r"""Return the support of the distribution."""
        return jnp.array([0.0, jnp.inf])

    def logpdf(self, x: ArrayLike, params: dict = None) -> Array:
        r"""Compute the log probability density function.

        Args:
            x: Input values at which to evaluate the log-PDF.
            params: Distribution parameters. Uses stored parameters if None.

        Returns:
            Log-PDF values with the same shape as ``x``.
        """
        params = self._resolve_params(params)
        x, xshape = _univariate_input(x)
        lamb = self._params_to_tuple(params)[0]
        
        logpdf: jnp.ndarray = jnp.log(lamb) - lamb * x
        return self._enforce_support_on_logpdf(
            x=x, logpdf=logpdf.reshape(xshape), params=params
        )

    def cdf(self, x: ArrayLike, params: dict = None) -> Array:
        r"""Compute the cumulative distribution function.

        Args:
            x: Input values at which to evaluate the CDF.
            params: Distribution parameters. Uses stored parameters if None.

        Returns:
            CDF values with the same shape as ``x``.
        """
        params = self._resolve_params(params)
        x, xshape = _univariate_input(x)
        lamb = self._params_to_tuple(params)[0]
        
        cdf: jnp.ndarray = 1 - jnp.exp(-lamb * x)
        return self._enforce_support_on_cdf(
            x=x, cdf=cdf.reshape(xshape), params=params
        )

    # ppf
    def _ppf(self, q: ArrayLike, params: dict, *args, **kwargs) -> Array:
        r"""Compute the percent point function (inverse CDF).

        Args:
            q: Input quantiles at which to evaluate the PPF.
            params: Distribution parameters. Uses stored parameters if None.

        Returns:
            PPF values with the same shape as ``q``.
        """
        params = self._resolve_params(params)
        q, qshape = _univariate_input(q)
        lamb = self._params_to_tuple(params)[0]
        
        ppf: jnp.ndarray = -jnp.log1p(-q) / lamb
        return ppf.reshape(qshape)

    # sampling
    def rvs(self, size: tuple | Scalar, params: dict = None, key=None) -> Array:
        r"""Generate random variates from the exponential distribution 
        via inverse transform sampling.

        Args:
            size: Shape of the output array.
            params: Distribution parameters. Uses stored parameters if None.
            key: JAX PRNG key. A default key is used if None.

        Returns:
            Array of random samples.
        """
        params = self._resolve_params(params)
        key = _resolve_key(key)
        lamb = self._params_to_tuple(params)[0]
        uniform_samples = random.uniform(key=key, shape=size)
        return self.ppf(uniform_samples, params=params)

    # stats
    def stats(self, params: dict = None) -> dict:
        params = self._resolve_params(params)
        lamb = self._params_to_tuple(params)[0]

        mean = 1 / lamb
        median = jnp.log(2) / lamb
        mode = 0.0
        variance = 1 / (lamb ** 2)
        skewness = 2.0
        kurtosis = 6.0
        return self._scalar_transform(
            {
                "mean": mean,
                "median": median,
                "mode": mode,
                "variance": variance,
                "skewness": skewness,
                "kurtosis": kurtosis,
            }
        )

    # fitting
    _supported_methods = frozenset({"mle"})

    def fit(self, x: ArrayLike, *args, name: str = None, **kwargs):
        r"""Fit the distribution to data using maximum likelihood estimation.

        Args:
            x: Input data to fit.
            name: Optional name for the fitted distribution instance.

        Returns:
            A new Exponential instance with fitted parameters.
        """
        x: jnp.ndarray = _univariate_input(x)[0]
        x_positive = jnp.where(x >= 0, x, jnp.nan)
        lamb_hat = 1 / jnp.mean(x_positive)  # MLE for lambda is 1/mean
        return self._fitted_instance(self._params_dict(lamb=lamb_hat), name=name)
        

exponential = Exponential("Exponential")