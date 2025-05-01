"""File containing the copulAX implementation of the normal distribution."""
import jax.numpy as jnp
from jax import lax, random
from jax._src.typing import ArrayLike, Array
from jax.scipy import special

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY


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
    @classmethod
    def _params_dict(cls, mu: Scalar, sigma: Scalar) -> dict:
        d: dict = {"mu": mu, "sigma": sigma}
        return cls._args_transform(d)
    
    def _params_to_tuple(self, params: dict):
        params = self._args_transform(params)
        return params["mu"], params["sigma"]

    def example_params(self, *args, **kwargs):
        r"""Example parameters for the normal distribution.
        
        This is a two parameter family, with the normal / gaussian being 
        defined by its mean and standard deviation.
        """
        return self._params_dict(mu=0.0, sigma=1.0)
    
    def _support(self, *args, **kwargs) -> tuple:
        return -jnp.inf, jnp.inf
    
    def logpdf(self, x: ArrayLike, params: dict) -> Array:
        x, xshape = _univariate_input(x)
        mu, sigma = self._params_to_tuple(params)

        const: jnp.ndarray = -0.5 * jnp.log(2 * jnp.pi)
        c: jnp.ndarray = lax.sub(const, jnp.log(sigma)) 
        e: jnp.ndarray = lax.div(lax.pow(lax.sub(x, mu), 2), 2 * lax.pow(sigma, 2))
        logpdf: jnp.ndarray = lax.sub(c, e)
        return logpdf.reshape(xshape)
    
    def logcdf(self, x: ArrayLike, params: dict) -> Array:
        x, xshape = _univariate_input(x)
        mu, sigma = self._params_to_tuple(params)

        z: jnp.ndarray = lax.div(lax.sub(x, mu), sigma)
        logcdf: jnp.ndarray = special.log_ndtr(z)
        return logcdf.reshape(xshape)
    
    def cdf(self, x: ArrayLike, params: dict) -> Array:
        x, xshape = _univariate_input(x)
        mu, sigma = self._params_to_tuple(params)

        z: jnp.ndarray = lax.div(lax.sub(x, mu), sigma)
        cdf: jnp.ndarray = special.ndtr(z)
        return cdf.reshape(xshape)
    
    # ppf
    def _get_x0(self, params: dict):
        return self._args_transform(params)["mu"]

    def _ppf(self, q: ArrayLike, params: dict, *args, **kwargs) -> Array:
        mu, sigma = self._params_to_tuple(params)
        z: jnp.array = jnp.asarray(special.ndtri(q), dtype=float)
        return lax.add(mu, lax.mul(sigma, z))
    
    # sampling
    def rvs(self, size: tuple | Scalar, params: dict, 
            key=DEFAULT_RANDOM_KEY) -> Array:
        mu, sigma = self._params_to_tuple(params)
        return random.normal(key=key, shape=size) * sigma + mu
    
    # stats
    def stats(self, params: dict) -> dict:
        mu, sigma = self._params_to_tuple(params)
        return self._scalar_transform({
            'mean': mu,
            'median': mu,
            'mode': mu,
            'variance': lax.pow(sigma, 2),
            'std': sigma,
            'skewness': 0.0,
            'kurtosis': 0.0,
        })
    
    # fitting
    def fit(self, x: ArrayLike, *args, **kwargs) -> dict:
        x: jnp.ndarray = _univariate_input(x)[0]
        mu: jnp.ndarray = x.mean()
        sigma: jnp.ndarray = x.std()
        return self._params_dict(mu=mu, sigma=sigma)
    

normal = Normal("Normal")