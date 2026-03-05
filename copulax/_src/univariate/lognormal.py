"""File containing the copulAX implementation of the lognormal distribution."""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import _resolve_key
from copulax._src.univariate.normal import normal


class LogNormal(Univariate):
    r"""The log-normal distribution of X is one where Y = log(X) is normally
    distributed. It is a continuous 2 parameter family of distributions.

    https://en.wikipedia.org/wiki/Log-normal_distribution
    """

    mu: Array = None
    sigma: Array = None

    def __init__(self, name="LogNormal", *, mu=None, sigma=None):
        """Initialize the LogNormal distribution.

        Args:
            name: Display name for the distribution.
            mu: Mean of the underlying normal distribution.
            sigma: Standard deviation of the underlying normal distribution.
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

    def _params_to_tuple(self, params: dict):
        """Extract (mu, sigma) from the parameter dictionary."""
        return normal._params_to_tuple(params)

    def example_params(self, *args, **kwargs):
        r"""Example parameters for the log-normal distribution.

        This is a two parameter family, with the log-normal being
        defined by the mean and standard deviation of its transformed
        distribution Y = log(X).
        """
        return normal.example_params()

    @classmethod
    def _support(cls, *args, **kwargs) -> Array:
        """Return the support ``[0, inf)``."""
        return jnp.array([0.0, jnp.inf])

    def logpdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the log-PDF by transforming to the underlying normal."""
        params = self._resolve_params(params)
        x, xshape = _univariate_input(x)
        x = x.reshape(xshape)
        return normal.logpdf(x=jnp.log(x), params=params) - jnp.log(x)

    def logcdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the log-CDF by transforming to the underlying normal."""
        params = self._resolve_params(params)
        return normal.logcdf(x=jnp.log(x), params=params)

    def cdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the CDF by transforming to the underlying normal."""
        params = self._resolve_params(params)
        return normal.cdf(x=jnp.log(x), params=params)

    # ppf
    def _ppf(self, q: ArrayLike, params: dict, *args, **kwargs) -> Array:
        """Compute the PPF as ``exp(normal_ppf(q))``."""
        return jnp.exp(normal._ppf(q=q, params=params, *args, **kwargs))

    # sampling
    def rvs(
        self, size: tuple | Scalar, params: dict = None, key: Array = None
    ) -> Array:
        """Generate random variates as ``exp(normal_rvs)``."""
        params = self._resolve_params(params)
        key = _resolve_key(key)
        return jnp.exp(normal.rvs(size=size, key=key, params=params))

    # stats
    def stats(self, params: dict = None) -> dict:
        """Compute distribution statistics (mean, median, mode, variance, std, skewness, kurtosis)."""
        params = self._resolve_params(params)
        mu, sigma = self._params_to_tuple(params)

        mean: float = jnp.exp(mu + jnp.pow(sigma, 2) / 2)
        median: float = jnp.exp(mu)
        mode: float = jnp.exp(mu - jnp.pow(sigma, 2))
        variance: float = (jnp.exp(jnp.pow(sigma, 2)) - 1) * jnp.exp(
            2 * mu + jnp.pow(sigma, 2)
        )
        std: float = jnp.sqrt(variance)
        skewness: float = (jnp.exp(jnp.pow(sigma, 2)) + 2) * jnp.sqrt(
            jnp.exp(jnp.pow(sigma, 2)) - 1
        )
        kurtosis: float = (
            jnp.exp(4 * jnp.pow(sigma, 2))
            + 2 * jnp.exp(3 * jnp.pow(sigma, 2))
            + 3 * jnp.exp(2 * jnp.pow(sigma, 2))
            - 6
        )

        return self._scalar_transform(
            {
                "mean": mean,
                "median": median,
                "mode": mode,
                "variance": variance,
                "std": std,
                "skewness": skewness,
                "kurtosis": kurtosis,
            }
        )

    # fitting
    def fit(self, x: ArrayLike, *args, **kwargs):
        """Fit by applying the normal MLE to ``log(x)``.

        Args:
            x: Input data to fit (must be positive).

        Returns:
            A new fitted LogNormal instance.
        """
        fitted_normal = normal.fit(jnp.log(x))
        return self._fitted_instance(fitted_normal.params)


lognormal = LogNormal("LogNormal")
