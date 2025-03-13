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
    def support(*args, **kwargs):
        return jnp.array(0.0), jnp.array(jnp.inf)
    
    def logpdf(self, x: ArrayLike, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        x, xshape = _univariate_input(x)
        x = x.reshape(xshape)
        return normal.logpdf(x=jnp.log(x), mu=mu, sigma=sigma) - jnp.log(x)
    
    def pdf(self, x: ArrayLike, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return super().pdf(x=x, mu=mu, sigma=sigma)
    
    def logcdf(self, x: ArrayLike, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return normal.logcdf(x=jnp.log(x), mu=mu, sigma=sigma)
    
    def cdf(self, x: ArrayLike, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return normal.cdf(x=jnp.log(x), mu=mu, sigma=sigma)
    
    def ppf(self, q: ArrayLike, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return jnp.exp(normal.ppf(q=q, mu=mu, sigma=sigma))
    
    def inverse_cdf(self, q: ArrayLike, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return super().inverse_cdf(q=q, mu=mu, sigma=sigma)
    
    # sampling
    def rvs(self, size: tuple | Scalar = (), key: Array = DEFAULT_RANDOM_KEY, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return jnp.exp(normal.rvs(size=size, key=key, mu=mu, sigma=sigma))
    
    def sample(self, size: tuple | Scalar = (), key: Array = DEFAULT_RANDOM_KEY, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return super().sample(size=size, key=key, mu=mu, sigma=sigma)
    
    # stats
    def stats(self, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> dict:
        mu, sigma = self._args_transform(mu, sigma)

        mean: float = jnp.exp(mu + jnp.pow(sigma, 2) / 2)
        median: float = jnp.exp(mu)
        mode: float = jnp.exp(mu - jnp.pow(sigma, 2))
        variance: float = (jnp.exp(jnp.pow(sigma, 2)) - 1) * jnp.exp(2 * mu + jnp.pow(sigma, 2))
        skewness: float = (jnp.exp(jnp.pow(sigma, 2)) + 2) * jnp.sqrt(jnp.exp(jnp.pow(sigma, 2)) - 1)
        kurtosis: float = jnp.exp(4 * jnp.pow(sigma, 2)) + 2 * jnp.exp(3 * jnp.pow(sigma, 2)) + 3 * jnp.exp(2 * jnp.pow(sigma, 2)) - 6

        return {'mean': mean, 'median': median, 'mode': mode, 'variance': variance, 'skewness': skewness, 'kurtosis': kurtosis}
    
    # metrics
    def loglikelihood(self, x, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Scalar:
        return super().loglikelihood(x, mu=mu, sigma=sigma)
    
    def aic(self, x, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Scalar:
        return super().aic(x, mu=mu, sigma=sigma)
    
    def bic(self, x, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Scalar:
        return super().bic(x, mu=mu, sigma=sigma)
    
    # fitting
    def fit(self, x: ArrayLike, *args, **kwargs) -> dict:
        return normal.fit(jnp.log(x), *args, **kwargs)
    

lognormal = LogNormal("LogNormal")