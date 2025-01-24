"""File containing the copulAX implementation of the multivariate normal distribution."""
import jax.numpy as jnp
from jax import lax, random, jit
from jax._src.typing import ArrayLike, Array

from copulax._src.multivariate._utils import _multivariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.multivariate._shape import cov


def mvt_normal_args_check(mu: ArrayLike, sigma: ArrayLike) -> tuple:
    mu, sigma = jnp.asarray(mu, dtype=float), jnp.asarray(sigma, dtype=float)
    n: int = mu.size
    return mu, sigma.reshape((n, n))


def mvt_normal_params_dict(mu: ArrayLike, sigma: ArrayLike) -> dict:
    mu, sigma = mvt_normal_args_check(mu=mu, sigma=sigma)
    return {'mu': mu, 'sigma': sigma}


@jit
def _single_qi(carry: tuple, xi: jnp.ndarray) -> jnp.ndarray:
    mu, sigma = carry
    return carry, lax.sub(xi, mu).T @ jnp.linalg.inv(sigma) @ lax.sub(xi, mu)


def logpdf(x: ArrayLike, mu: ArrayLike=jnp.zeros((2, 1)), sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
    r"""Log-probability density function of the multivariate normal distribution.
    
    The multivariate normal pdf is defined as:

    .. math::

        f(x|\mu, \Sigma) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)

    Args:
        x: arraylike, value(s) at which to evaluate the log-pdf.
        mu: Mean vector of the multivariate normal distribution.
        sigma: Covariance matrix of the multivariate normal distribution.

    Returns:
        array of log-pdf values.
    """
    x, yshape, n, d = _multivariate_input(x)
    mu, sigma = mvt_normal_args_check(mu=mu, sigma=sigma)

    const: jnp.ndarray = -0.5 * (d * jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(sigma)[1])

    Q: jnp.ndarray = lax.scan(f=_single_qi, init=(mu.flatten(), sigma),  xs=x)[1]

    logpdf: jnp.ndarray = -0.5 * Q + const
    return logpdf.reshape(yshape)


def pdf(x: ArrayLike, mu: ArrayLike=jnp.zeros((2, 1)), sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
    r"""Probability density function of the multivariate normal distribution.
    
    The multivariate normal pdf is defined as:

    .. math::

        f(x|\mu, \Sigma) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)

    Args:
        x: arraylike, value(s) at which to evaluate the pdf.
        mu: Mean vector of the multivariate normal distribution.
        sigma: Covariance matrix of the multivariate normal distribution.

    Returns:
        array of pdf values.
    """
    return jnp.exp(logpdf(x=x, mu=mu, sigma=sigma))


def rvs(size: int = 1, key:ArrayLike=DEFAULT_RANDOM_KEY, mu: ArrayLike=jnp.zeros((2, 1)), sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
    r"""Generate random samples from the multivariate normal distribution.
    
    Args:
        size: Number of samples to generate.
        key: Key used to generate the samples.
        mu: Mean vector of the multivariate normal distribution.
        sigma: Covariance matrix of the multivariate normal distribution.

    Returns:
        array of random samples.
    """
    mu, sigma = mvt_normal_args_check(mu=mu, sigma=sigma)
    return random.multivariate_normal(key=key, mean=mu.flatten(), cov=sigma, shape=(size,))


def fit(x: ArrayLike, method: str) -> dict:
    r"""Fit the parameters of a multivariate normal distribution to the data.
    
    Args:
        x: arraylike, data to fit the distribution to.
        method: str, method to use for fitting the covariance / shape matrix, sigma.
        See copulax.multivariate.cov for available methods.

    Returns:
        dict containing the fitted parameters.
    """
    x, _, n, d = _multivariate_input(x)
    mu: jnp.ndarray = jnp.mean(x, axis=0).reshape((d, 1))
    sigma: jnp.ndarray = cov(x=x, method=method)
    return mvt_normal_params_dict(mu=mu, sigma=sigma)


def stats(mu: ArrayLike, sigma: ArrayLike) -> dict:
    r"""Compute the mean and covariance matrix of the multivariate normal distribution.
    
    Args:
        mu: Mean vector of the multivariate normal distribution.
        sigma: Covariance matrix of the multivariate normal distribution.

    Returns:
        dict containing the mean and covariance matrix of the distribution.
    """
    mu, sigma = mvt_normal_args_check(mu=mu, sigma=sigma)
    return {'mean': mu, 'cov': sigma}

