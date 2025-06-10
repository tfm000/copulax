"""File containing the copulAX implementation of the lognormal distribution."""
import jax.numpy as jnp
from jax._src.typing import ArrayLike, Array

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.univariate.normal import normal


class LogNormal(Univariate):
    r"""The log-normal distribution of X is one where Y = log(X) is normally 
    distributed. It is a continuous 2 parameter family of distributions.
    
    https://en.wikipedia.org/wiki/Log-normal_distribution
    """
    def _params_to_tuple(self, params: dict):
        return normal._params_to_tuple(params)

    def example_params(self, *args, **kwargs):
        r"""Example parameters for the log-normal distribution.
        
        This is a two parameter family, with the log-normal being 
        defined by the mean and standard deviation of its transformed 
        distribution Y = log(X).
        """
        return normal.example_params()
    
    @classmethod
    def _support(cls, *args, **kwargs) -> tuple:
        return 0.0, jnp.inf
    
    def logpdf(self, x: ArrayLike, params: dict) -> Array:
        x, xshape = _univariate_input(x)
        x = x.reshape(xshape)
        return normal.logpdf(x=jnp.log(x), params=params) - jnp.log(x)
    
    def logcdf(self, x: ArrayLike, params: dict) -> Array:
        return normal.logcdf(x=jnp.log(x), params=params)
    
    def cdf(self, x: ArrayLike, params: dict) -> Array:
        return normal.cdf(x=jnp.log(x), params=params)

    # ppf    
    # def _get_x0(self, params: dict):
    #     return normal._get_x0(params=params)
    
    def _ppf(self, q: ArrayLike, params: dict, *args, **kwargs) -> Array:
        return jnp.exp(normal._ppf(q=q, params=params, *args, **kwargs))
    
    # sampling
    def rvs(self, size: tuple | Scalar, params: dict, key: Array = DEFAULT_RANDOM_KEY) -> Array:
        return jnp.exp(normal.rvs(size=size, key=key, params=params))
    
    # stats
    def stats(self, params: dict) -> dict:
        mu, sigma = self._params_to_tuple(params)

        mean: float = jnp.exp(mu + jnp.pow(sigma, 2) / 2)
        median: float = jnp.exp(mu)
        mode: float = jnp.exp(mu - jnp.pow(sigma, 2))
        variance: float = (jnp.exp(jnp.pow(sigma, 2)) - 1) * jnp.exp(2 * mu + jnp.pow(sigma, 2))
        std: float = jnp.sqrt(variance)
        skewness: float = (jnp.exp(jnp.pow(sigma, 2)) + 2) * jnp.sqrt(jnp.exp(jnp.pow(sigma, 2)) - 1)
        kurtosis: float = jnp.exp(4 * jnp.pow(sigma, 2)) + 2 * jnp.exp(3 * jnp.pow(sigma, 2)) + 3 * jnp.exp(2 * jnp.pow(sigma, 2)) - 6

        return self._scalar_transform({
            'mean': mean, 
            'median': median, 
            'mode': mode, 
            'variance': variance, 
            'std': std,
            'skewness': skewness, 
            'kurtosis': kurtosis})
    
    # fitting
    def fit(self, x: ArrayLike, *args, **kwargs) -> dict:
        return normal.fit(jnp.log(x))
    

lognormal = LogNormal("LogNormal")