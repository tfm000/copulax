"""File containing the copulAX implementation of the normal distribution."""
import jax.numpy as jnp
from jax import lax, random
from jax._src.typing import ArrayLike, Array
from jax.scipy import special

from copulax._src.univariate._distributions import Univariate
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
    def _params_dict(self, mu: Scalar, sigma: Scalar) -> dict:
        mu, sigma = self._args_transform(mu, sigma)
        return {"mu": mu, "sigma": sigma}
    
    def support(self, *args, **kwargs) -> tuple[Scalar, Scalar]:
        return jnp.array(-jnp.inf), jnp.array(jnp.inf)
    
    def logpdf(self, x: ArrayLike, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        x, xshape = _univariate_input(x)
        mu, sigma = self._args_transform(mu, sigma)

        const: jnp.ndarray = -0.5 * jnp.log(2 * jnp.pi)
        c: jnp.ndarray = lax.sub(const, jnp.log(sigma)) 
        e: jnp.ndarray = lax.div(lax.pow(lax.sub(x, mu), 2), 2 * lax.pow(sigma, 2))
        logpdf: jnp.ndarray = lax.sub(c, e)
        return logpdf.reshape(xshape)
    
    def pdf(self, x: ArrayLike, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return super().pdf(x=x, mu=mu, sigma=sigma)
    
    def logcdf(self, x: ArrayLike, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        x, xshape = _univariate_input(x)
        mu, sigma = self._args_transform(mu, sigma)

        z: jnp.ndarray = lax.div(lax.sub(x, mu), sigma)
        logcdf: jnp.ndarray = special.log_ndtr(z)
        return logcdf.reshape(xshape)
    
    def cdf(self, x: ArrayLike, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        x, xshape = _univariate_input(x)
        mu, sigma = self._args_transform(mu, sigma)

        z: jnp.ndarray = lax.div(lax.sub(x, mu), sigma)
        cdf: jnp.ndarray = special.ndtr(z)
        return cdf.reshape(xshape)
    
    def ppf(self, q: ArrayLike, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        q, qshape = _univariate_input(q)
        mu, sigma = self._args_transform(mu, sigma)

        z: jnp.array = jnp.asarray(special.ndtri(q), dtype=float)
        x: jnp.ndarray = lax.add(mu, lax.mul(sigma, z))
        return x.reshape(qshape)
    
    def inverse_cdf(self, q: ArrayLike, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return super().inverse_cdf(q=q, mu=mu, sigma=sigma)
    
    # sampling
    def rvs(self, shape = (), key=DEFAULT_RANDOM_KEY, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        mu, sigma = self._args_transform(mu, sigma)
        return random.normal(key=key, shape=shape) * sigma + mu
    
    def sample(self, shape = (), key=DEFAULT_RANDOM_KEY, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return super().sample(shape=shape, key=key, mu=mu, sigma=sigma)
    
    # stats
    def stats(self, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> dict:
        mu, sigma = self._args_transform(mu, sigma)
        return {
            'mean': mu,
            'median': mu,
            'mode': mu,
            'variance': lax.pow(sigma, 2),
            'skewness': 0.0,
            'kurtosis': 0.0,
        }
    
    # metrics
    def loglikelihood(self, x, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Scalar:
        return super().loglikelihood(x, mu=mu, sigma=sigma)
    
    def aic(self, x, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Scalar:
        return super().aic(x, mu=mu, sigma=sigma)
    
    def bic(self, x, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Scalar:
        return super().bic(x, mu=mu, sigma=sigma)
    
    # fitting
    def fit(self, x: ArrayLike, *args, **kwargs) -> dict:
        x: jnp.ndarray = _univariate_input(x)[0]
        mu: jnp.ndarray = x.mean()
        sigma: jnp.ndarray = x.std()
        return self._params_dict(mu=mu, sigma=sigma)
    

normal = Normal("Normal")