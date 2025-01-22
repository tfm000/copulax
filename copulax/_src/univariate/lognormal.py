"""File containing the copulAX implementation of the lognormal distribution."""
import jax.numpy as jnp
from jax._src.typing import ArrayLike, Array

from copulax._src.univariate._utils import _univariate_input, DEFAULT_RANDOM_KEY
from copulax._src.univariate import normal


def support(*args) -> tuple[float, float]:
    r"""The support of the distribution is the subset of x for which the pdf 
    is non-zero. 
    
    Returns:
        (float, float): Tuple containing the support of the distribution.
    """
    return 0.0, jnp.inf


def logpdf(x: ArrayLike, mu: float = 0.0, sigma: float = 1.0) -> Array:
    r"""Log-probability density function of the lognormal distribution.
    
    The lognormal pdf is defined as:

    .. math::

        f(x|\mu, \sigma) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\log(x) - \mu)^2}{2\sigma^2}\right)

    Args:
        x: arraylike, value(s) at which to evaluate the log-pdf.
        mu: Mean/location of the transformed log(x) normal distribution.
        sigma: Standard deviation of the transformed log(x) normal distribution.

    Returns:
        array of log-pdf values.
    """
    x, xshape = _univariate_input(x)
    x = x.reshape(xshape)
    return normal.logpdf(x=jnp.log(x), mu=mu, sigma=sigma) - jnp.log(x)


def pdf(x: ArrayLike, mu: float = 0.0, sigma: float = 1.0) -> Array:
    r"""Probability density function of the lognormal distribution.
    
    The lognormal pdf is defined as:

    .. math::

        f(x|\mu, \sigma) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\log(x) - \mu)^2}{2\sigma^2}\right)

    Args:
        x: arraylike, value(s) at which to evaluate the pdf.
        mu: Mean/location of the transformed log(x) normal distribution.
        sigma: Standard deviation of the transformed log(x) normal distribution.

    Returns:
        array of pdf values.
    """
    return jnp.exp(logpdf(x=x, mu=mu, sigma=sigma))


def cdf(x: ArrayLike, mu: float = 0.0, sigma: float = 1.0) -> Array:
    r"""Cumulative distribution function of the lognormal distribution.
    
    The lognormal pdf is defined as:

    .. math::

        f(x|\mu, \sigma) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\log(x) - \mu)^2}{2\sigma^2}\right)

    Args:
        x: arraylike, value(s) at which to evaluate the cdf.
        mu: Mean/location of the transformed log(x) normal distribution.
        sigma: Standard deviation of the transformed log(x) normal distribution.

    Returns:
        array of cdf values.
    """
    return normal.cdf(x=jnp.log(x), mu=mu, sigma=sigma)


def logcdf(x: ArrayLike, mu: float = 0.0, sigma: float = 1.0) -> Array:
    r"""Log-cumulative distribution function of the lognormal distribution.
    
    .. math::

        f(x|\mu, \sigma) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\log(x) - \mu)^2}{2\sigma^2}\right)

    Args:
        x: arraylike, value(s) at which to evaluate the log-cdf.
        mu: Mean/location of the transformed log(x) normal distribution.
        sigma: Standard deviation of the transformed log(x) normal distribution.

    Returns:
        array of log-cdf values.
    """
    return normal.logcdf(x=jnp.log(x), mu=mu, sigma=sigma)


def ppf(q: ArrayLike, mu: float = 0.0, sigma: float = 1.0) -> Array:
    r"""Percent point function (inverse cdf) of the lognormal distribution.
    
    The lognormal pdf is defined as:

    .. math::

        f(x|\mu, \sigma) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\log(x) - \mu)^2}{2\sigma^2}\right)

    Args:
        q: arraylike, value(s) at which to evaluate the ppf.
        mu: Mean/location of the transformed log(x) normal distribution.
        sigma: Standard deviation of the transformed log(x) normal distribution.

    Returns:
        array of ppf values.
    """
    return jnp.exp(normal.ppf(q=q, mu=mu, sigma=sigma))


def rvs(shape: tuple = (1, ), key: Array = DEFAULT_RANDOM_KEY, mu: float = 0.0, sigma: float = 1.0,) -> Array:
    r"""Random variates sampled from the lognormal distribution.
    
    The lognormal pdf is defined as:

    .. math::

        f(x|\mu, \sigma) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\log(x) - \mu)^2}{2\sigma^2}\right)

    Args:
        shape: The shape of the random number array to generate.
        key: Key for random number generation.
        mu: Mean/location of the transformed log(x) normal distribution.
        sigma: Standard deviation of the transformed log(x) normal distribution.
        

    Returns:
        array of random variates.
    """
    return jnp.exp(normal.rvs(key=key, mu=mu, sigma=sigma, shape=shape))


def fit(x: ArrayLike) -> dict:
    r"""Fit the parameters of the lognormal distribution to the data.
    
    Args:
        x: arraylike, data to fit.

    Returns:
        dictionary of fitted parameters.
    """
    return normal.fit(jnp.log(x))


def stats(mu: float = 0.0, sigma: float = 1.0) -> dict:
    r"""Distribution statistics for the lognormal distribution. Returns the 
    mean, median, mode, variance, skewness and (excess) kurtosis of the 
    distribution.

    Args:
        mu: Mean/location of the transformed log(x) normal distribution.
        sigma: Standard deviation of the transformed log(x) normal distribution.
    
    """
    mu, sigma = normal.normal_args_check(mu, sigma)

    mean: float = jnp.exp(mu + jnp.pow(sigma, 2) / 2)
    median: float = jnp.exp(mu)
    mode: float = jnp.exp(mu - jnp.pow(sigma, 2))
    variance: float = (jnp.exp(jnp.pow(sigma, 2)) - 1) * jnp.exp(2 * mu + jnp.pow(sigma, 2))
    skewness: float = (jnp.exp(jnp.pow(sigma, 2)) + 2) * jnp.sqrt(jnp.exp(jnp.pow(sigma, 2)) - 1)
    kurtosis: float = jnp.exp(4 * jnp.pow(sigma, 2)) + 2 * jnp.exp(3 * jnp.pow(sigma, 2)) + 3 * jnp.exp(2 * jnp.pow(sigma, 2)) - 6

    return {'mean': mean, 'median': median, 'mode': mode, 'variance': variance, 'skewness': skewness, 'kurtosis': kurtosis}
