"""File containing the copulAX implementation of the Wald/Inverse Gaussian distribution."""

import jax.numpy as jnp
from jax import random
from jax import Array
from jax.typing import ArrayLike
from jax.scipy import special

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import _resolve_key
from copulax._src.univariate.normal import normal


class Wald(Univariate):
    r"""The Wald distribution, also known as the Inverse Gaussian distribution,
    is a continuous 2 parameter family.

    The Wald distribution is defined as:

    .. math::

        f(x|\mu, \lambda) = \sqrt{\frac{\lambda}{2\pi x^3}} \exp\left(-\frac{\lambda(x-\mu)^2}{2\mu^2 x}\right)

    where :math:`\mu` is the mean and :math:`\lambda` the shape parameter of the distribution.

    https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
    """

    mu: Array = None
    lamb: Array = None

    def __init__(self, name="Wald", *, mu=None, lamb=None):
        """Initialize the Wald distribution.

        Args:
            name: Display name for the distribution.
            mu: Location parameter (mean). If provided, stored on the instance.
            lamb: Shape parameter. If provided, stored on the instance.
        """
        super().__init__(name=name)
        self.mu = jnp.asarray(mu, dtype=float).reshape(()) if mu is not None else None
        self.lamb = jnp.asarray(lamb, dtype=float).reshape(()) if lamb is not None else None

    @property
    def _stored_params(self):
        """Return stored parameters if all are set, else None."""
        if self.mu is None or self.lamb is None:
            return None
        return self._params_dict(self.mu, self.lamb)

    @classmethod
    def _params_dict(cls, mu: Scalar, lamb: Scalar) -> dict:
        """Create a parameter dictionary from mu and lamb values."""
        d: dict = {"mu": mu, "lamb": lamb}
        return cls._args_transform(d)

    def _params_to_tuple(self, params: dict) -> tuple:
        """Extract (mu, lamb) from the parameter dictionary."""
        params = self._args_transform(params)
        return params["mu"], params["lamb"]

    def example_params(self, *args, **kwargs) -> dict:
        """Return example parameters for the Wald / Inverse Gaussian distribution."""
        return self._params_dict(mu=1.0, lamb=1.0)

    @classmethod
    def _support(cls, *args, **kwargs) -> Array:
        """Return the support ``[0, inf]`` of the Wald distribution."""
        return jnp.array([0.0, jnp.inf])


    def logpdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the log probability density function of the Wald distribution.

        Args:
            x: Input values at which to evaluate the log-PDF.
            params: Dictionary containing the parameters of the distribution.
                Uses stored parameters if None.

        Returns:
            Log-PDF values with the same shape as ``x``.
        """
        params = self._resolve_params(params)
        x, xshape = _univariate_input(x)
        mu, lamb = self._params_to_tuple(params)

        diff = x - mu
        log_exponent = -0.5 * lamb * diff ** 2 / (mu**2 * x)
        log_prefactor = 0.5 * jnp.log(lamb) - 1.5 * jnp.log(x) - 0.5 * jnp.log(2 * jnp.pi)

        log_pdf = log_prefactor + log_exponent
        return self._enforce_support_on_logpdf(
            x=x, logpdf=log_pdf.reshape(xshape), params=params
        )

    def cdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the cumulative distribution function of the Wald / Inverse Gaussian distribution.

        Args:
            x: Input values at which to evaluate the CDF.
            params: Dictionary containing the parameters of the distribution.
                Uses stored parameters if None.

        Returns:
            CDF values with the same shape as ``x``.
        """
        params = self._resolve_params(params)
        x, xshape = _univariate_input(x)
        mu, lamb = self._params_to_tuple(params)

        sqrt_lamb_over_x = jnp.sqrt(lamb / x)
        x_over_mu = x / mu

        z1 = sqrt_lamb_over_x * (x_over_mu - 1)
        z2 = -sqrt_lamb_over_x * (x_over_mu + 1)

        # Stable form: exp(2λ/μ)·Φ(z2) = exp(2λ/μ + log_ndtr(z2)).
        # z2 < 0 for x > 0, so log_ndtr(z2) → -∞ as 2λ/μ → ∞, keeping the sum bounded.
        mirror_term = jnp.exp(2 * lamb / mu + special.log_ndtr(z2))
        cdf = special.ndtr(z1) + mirror_term
        return self._enforce_support_on_cdf(x=x, cdf=cdf.reshape(xshape), params=params)

    # sampling
    def rvs(self, size: tuple | Scalar, params: dict = None, key: Array = None) -> Array:
        """Generate random variates from the Wald distribution via Michael-Schucany-Haas."""
        params = self._resolve_params(params)
        key = _resolve_key(key)
        mu, lamb = self._params_to_tuple(params)
        key1, key2 = random.split(key)

        # Step 1: Sample y = z^2, z ~ N(0, 1)
        z = normal.rvs(size=size, params={"mu": 0.0, "sigma": 1.0}, key=key1)
        y = z ** 2

        # Step 2: Compute the smaller root of the MSH quadratic
        x_candidate = mu + (mu**2 * y) / (2 * lamb) \
            - (mu / (2 * lamb)) * jnp.sqrt(4 * mu * lamb * y + mu**2 * y**2)

        # Step 3: Accept x_candidate with prob mu/(mu + x_candidate), else mu^2/x_candidate
        u = random.uniform(key2, shape=z.shape)
        return jnp.where(u <= mu / (mu + x_candidate), x_candidate, (mu**2) / x_candidate)

    # stats
    def stats(self, params: dict = None) -> dict:
        """Compute the mean and variance of the Wald distribution given its parameters.

        Args:
            params: Dictionary containing the parameters of the distribution.
                Uses stored parameters if None.
        """
        params = self._resolve_params(params)
        mu, lamb = self._params_to_tuple(params)

        mean = mu
        mode = mu * (jnp.sqrt(1 + (9 * mu**2) / (4 * lamb**2)) - (3 * mu) / (2 * lamb))
        variance = (mu**3) / lamb
        skewness = 3 * jnp.sqrt(mu / lamb)
        kurtosis = 15 * mu / lamb

        return self._scalar_transform({"mean": mean, "variance": variance, "mode": mode, "skewness": skewness, "kurtosis": kurtosis})
    
    # fitting
    def fit(self, x: ArrayLike, *args, name: str = None, **kwargs) -> dict:
        """Fit the parameters of the Wald distribution to data using closed-form MLE.
        
        Args:
            x: Input data to fit.
            name: Optional custom name for the fitted distribution instance.

        Returns:
            A new fitted Wald instance with parameters estimated from the data.
        """
        x = _univariate_input(x)[0]
        mean_x = x.mean()
        inv_mean_x = (1 / x).mean()

        mu: jnp.ndarray = mean_x
        lamb: jnp.ndarray = 1 / (inv_mean_x - (1 / mean_x))
    
        return self._fitted_instance(self._params_dict(mu=mu, lamb=lamb), name=name)
    

wald = Wald("Wald")
        