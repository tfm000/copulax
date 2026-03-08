"""File containing the copulAX implementation of the normal distribution."""

import jax.numpy as jnp
from jax import lax, random
from jax import Array
from jax.typing import ArrayLike
from jax.scipy import special

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import _resolve_key


class Normal(Univariate):
    r"""The normal / Gaussian distribution is a continuous 'bell shaped' 2
    parameter family.

    The normal distribution is defined as:

    .. math::

        f(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)

    where :math:`\mu` is the mean and :math:`\sigma` the standard deviation
    of the data.

    https://en.wikipedia.org/wiki/Normal_distribution
    """

    mu: Array = None
    sigma: Array = None

    def __init__(self, name="Normal", *, mu=None, sigma=None):
        """Initialize the Normal distribution.

        Args:
            name: Display name for the distribution.
            mu: Location parameter (mean). If provided, stored on the instance.
            sigma: Scale parameter (standard deviation). If provided, stored on the instance.
        """
        super().__init__(name)
        self.mu = jnp.asarray(mu, dtype=float).reshape(()) if mu is not None else None
        self.sigma = (
            jnp.asarray(sigma, dtype=float).reshape(()) if sigma is not None else None
        )

    @property
    def _stored_params(self):
        """Return stored parameters if all are set, else None."""
        if self.mu is None or self.sigma is None:
            return None
        return {"mu": self.mu, "sigma": self.sigma}

    @classmethod
    def _params_dict(cls, mu: Scalar, sigma: Scalar) -> dict:
        """Create a parameter dictionary from mu and sigma values."""
        d: dict = {"mu": mu, "sigma": sigma}
        return cls._args_transform(d)

    def _params_to_tuple(self, params: dict):
        """Extract (mu, sigma) from the parameter dictionary."""
        params = self._args_transform(params)
        return params["mu"], params["sigma"]

    def example_params(self, *args, **kwargs):
        r"""Example parameters for the normal distribution.

        This is a two parameter family, with the normal / gaussian being
        defined by its mean and standard deviation.
        """
        return self._params_dict(mu=0.0, sigma=1.0)

    @classmethod
    def _support(cls, *args, **kwargs) -> Array:
        """Return the support ``[-inf, inf]``."""
        return jnp.array([-jnp.inf, jnp.inf])

    def logpdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the log probability density function.

        Args:
            x: Input values at which to evaluate the log-PDF.
            params: Distribution parameters. Uses stored parameters if None.

        Returns:
            Log-PDF values with the same shape as ``x``.
        """
        params = self._resolve_params(params)
        x, xshape = _univariate_input(x)
        mu, sigma = self._params_to_tuple(params)

        const: jnp.ndarray = -0.5 * jnp.log(2 * jnp.pi)
        c: jnp.ndarray = lax.sub(const, jnp.log(sigma))
        e: jnp.ndarray = lax.div(lax.pow(lax.sub(x, mu), 2), 2 * lax.pow(sigma, 2))
        logpdf: jnp.ndarray = lax.sub(c, e)
        return self._enforce_support_on_logpdf(
            x=x, logpdf=logpdf.reshape(xshape), params=params
        )

    def logcdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the log cumulative distribution function.

        Args:
            x: Input values at which to evaluate the log-CDF.
            params: Distribution parameters. Uses stored parameters if None.

        Returns:
            Log-CDF values with the same shape as ``x``.
        """
        params = self._resolve_params(params)
        x, xshape = _univariate_input(x)
        mu, sigma = self._params_to_tuple(params)

        z: jnp.ndarray = lax.div(lax.sub(x, mu), sigma)
        logcdf: jnp.ndarray = special.log_ndtr(z)
        return logcdf.reshape(xshape)

    def cdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the cumulative distribution function.

        Args:
            x: Input values at which to evaluate the CDF.
            params: Distribution parameters. Uses stored parameters if None.

        Returns:
            CDF values with the same shape as ``x``.
        """
        params = self._resolve_params(params)
        x, xshape = _univariate_input(x)
        mu, sigma = self._params_to_tuple(params)

        z: jnp.ndarray = lax.div(lax.sub(x, mu), sigma)
        cdf: jnp.ndarray = special.ndtr(z)
        return self._enforce_support_on_cdf(
            x=x, cdf=cdf.reshape(xshape), params=params
        )

    # ppf
    def _ppf(self, q: ArrayLike, params: dict, *args, **kwargs) -> Array:
        """Compute the percent-point function (inverse CDF) via ``ndtri``."""
        mu, sigma = self._params_to_tuple(params)
        z: jnp.array = jnp.asarray(special.ndtri(q), dtype=float)
        return lax.add(mu, lax.mul(sigma, z))

    # sampling
    def rvs(self, size: tuple | Scalar, params: dict = None, key=None) -> Array:
        """Generate random variates from the normal distribution.

        Args:
            size: Shape of the output array.
            params: Distribution parameters. Uses stored parameters if None.
            key: JAX PRNG key. A default key is used if None.

        Returns:
            Array of random samples.
        """
        params = self._resolve_params(params)
        key = _resolve_key(key)
        mu, sigma = self._params_to_tuple(params)
        return random.normal(key=key, shape=size) * sigma + mu

    # stats
    def stats(self, params: dict = None) -> dict:
        """Compute distribution statistics (mean, median, mode, variance, std, skewness, kurtosis)."""
        params = self._resolve_params(params)
        mu, sigma = self._params_to_tuple(params)
        return self._scalar_transform(
            {
                "mean": mu,
                "median": mu,
                "mode": mu,
                "variance": lax.pow(sigma, 2),
                "std": sigma,
                "skewness": 0.0,
                "kurtosis": 0.0,
            }
        )

    # fitting
    def fit(self, x: ArrayLike, *args, **kwargs):
        """Fit the distribution to data using closed-form MLE.

        Args:
            x: Input data to fit.

        Returns:
            A new fitted Normal instance.
        """
        x: jnp.ndarray = _univariate_input(x)[0]
        mu: jnp.ndarray = x.mean()
        sigma: jnp.ndarray = x.std()
        return self._fitted_instance(self._params_dict(mu=mu, sigma=sigma))


normal = Normal("Normal")
