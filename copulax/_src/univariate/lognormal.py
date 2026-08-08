"""File containing the copulAX implementation of the lognormal distribution."""

from typing import Any, cast

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src._utils import _resolve_key
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src.univariate.normal import normal


class LogNormal(Univariate):
    r"""The log-normal distribution on :math:`(0, \infty)` describes a
    positive variate :math:`X` whose logarithm :math:`Y = \log X` is
    normally distributed with mean :math:`\mu` and standard deviation
    :math:`\sigma`. Two-parameter continuous family.

    The PDF is

    .. math::

        f(x | \mu, \sigma) =
            \frac{1}{x \sigma \sqrt{2\pi}}
            \exp\!\left(-\frac{(\log x - \mu)^2}{2 \sigma^2}\right),
        \qquad x > 0

    where :math:`\mu \in \mathbb{R}` and :math:`\sigma > 0` are the mean
    and standard deviation **of the underlying normal** :math:`\log X`
    (not of :math:`X` itself; the mean of :math:`X` is
    :math:`\exp(\mu + \sigma^2 / 2)`).

    https://en.wikipedia.org/wiki/Log-normal_distribution
    """

    mu: Array | None = None
    sigma: Array | None = None

    def __init__(
        self,
        name: str = "LogNormal",
        *,
        mu: ArrayLike | None = None,
        sigma: ArrayLike | None = None,
    ) -> None:
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
    def _stored_params(self) -> dict | None:
        """Return stored parameters if all are set, else None."""
        if self.mu is None or self.sigma is None:
            return None
        return {"mu": self.mu, "sigma": self.sigma}

    def _params_to_tuple(self, params: dict) -> tuple:
        """Extract (mu, sigma) from the parameter dictionary."""
        return normal._params_to_tuple(params)

    def example_params(self, *args: Any, **kwargs: Any) -> dict:
        return normal.example_params()

    @classmethod
    def _support(cls, *args: Any, **kwargs: Any) -> Array:
        """Return the support ``[0, inf)``."""
        return jnp.array([0.0, jnp.inf])

    def logpdf(self, x: ArrayLike, params: dict | None = None) -> Array:
        """Compute the log-PDF by transforming to the underlying normal."""
        params = self._resolve_params(params)
        x, xshape = _univariate_input(x)
        x = x.reshape(xshape)
        logpdf = normal.logpdf(x=jnp.log(x), params=params) - jnp.log(x)
        return self._enforce_support_on_logpdf(x=x, logpdf=logpdf, params=params)

    def logcdf(self, x: ArrayLike, params: dict | None = None) -> Array:
        """Compute the log-CDF by transforming to the underlying normal."""
        params = self._resolve_params(params)
        return normal.logcdf(x=jnp.log(x), params=params)

    def cdf(self, x: ArrayLike, params: dict | None = None) -> Array:
        """Compute the CDF by transforming to the underlying normal."""
        params = self._resolve_params(params)
        cdf = normal.cdf(x=jnp.log(x), params=params)
        return self._enforce_support_on_cdf(x=x, cdf=cdf, params=params)

    # ppf
    def _ppf(self, q: ArrayLike, params: dict, *args: Any, **kwargs: Any) -> Array:
        """Compute the PPF as ``exp(normal_ppf(q))``."""
        return jnp.exp(normal._ppf(q, params, *args, **kwargs))

    # sampling
    def rvs(
        self, size: tuple | Scalar, params: dict | None = None, key: Array | None = None
    ) -> Array:
        """Generate random variates as ``exp(normal_rvs)``."""
        params = self._resolve_params(params)
        key = _resolve_key(key)
        return jnp.exp(normal.rvs(size=size, key=key, params=params))

    # stats
    def stats(self, params: dict | None = None) -> dict:
        """Compute distribution statistics (mean, median, mode, variance,
        std, skewness, kurtosis)."""
        params = self._resolve_params(params)
        mu, sigma = self._params_to_tuple(params)

        mean: Array = jnp.exp(mu + jnp.pow(sigma, 2) / 2)
        median: Array = jnp.exp(mu)
        mode: Array = jnp.exp(mu - jnp.pow(sigma, 2))
        variance: Array = (jnp.exp(jnp.pow(sigma, 2)) - 1) * jnp.exp(
            2 * mu + jnp.pow(sigma, 2)
        )
        std: Array = jnp.sqrt(variance)
        skewness: Array = (jnp.exp(jnp.pow(sigma, 2)) + 2) * jnp.sqrt(
            jnp.exp(jnp.pow(sigma, 2)) - 1
        )
        kurtosis: Array = (
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
    _supported_methods = frozenset({"mle"})

    def fit(
        self, x: ArrayLike, *args: Any, name: str | None = None, **kwargs: Any
    ) -> "LogNormal":
        r"""Fit by applying the normal **closed-form** MLE to ``log(x)``.

        Delegates to :meth:`Normal.fit` on the log-transformed data,
        which has no tuning parameters.

        Args:
            x: Input data to fit (must be positive).
            name: Optional custom name for the fitted instance.

        Returns:
            LogNormal: A fitted ``LogNormal`` instance.
        """
        fitted_normal = normal.fit(jnp.log(x))
        # ``params`` is ``dict | None`` because an UNfitted distribution
        # stores nothing; ``fit`` always populates it, so this instance's
        # is a dict.  ``cast`` is a runtime identity, unlike an ``assert``.
        return self._fitted_instance(cast("dict", fitted_normal.params), name=name)


lognormal = LogNormal("LogNormal")
