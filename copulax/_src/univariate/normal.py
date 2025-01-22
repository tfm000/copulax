"""File containing the copulAX implementation of the normal distribution."""
import jax.numpy as jnp
from jax import lax, random
from jax._src.typing import ArrayLike, Array
from jax.scipy import special

from copulax._src.univariate._utils import _univariate_input, DEFAULT_RANDOM_KEY


def normal_args_check(mu: float | ArrayLike, sigma: float | ArrayLike) -> tuple:
    return jnp.asarray(mu, dtype=float), jnp.asarray(sigma, dtype=float)


def normal_params_dict(mu: float, sigma: float) -> dict:
    mu, sigma = normal_args_check(mu, sigma)
    return {'mu': mu, 'sigma': sigma}


def support(*args) -> tuple[float, float]:
    r"""The support of the distribution is the subset of x for which the pdf 
    is non-zero. 
    
    Returns:
        (float, float): Tuple containing the support of the distribution.
    """
    return -jnp.inf, jnp.inf


def logpdf(x: ArrayLike, mu: float = 0.0, sigma: float = 1.0) -> Array:
    r"""Log-probability density function of the normal distribution.
    
    The normal pdf is defined as:

    .. math::

        f(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)

    Args:
        x: arraylike, value(s) at which to evaluate the pdf.
        mu: Mean/location of the normal distribution.
        sigma: Standard deviation of the normal distribution.

    Returns:
        array of log-pdf values.
    """
    x, xshape = _univariate_input(x)
    mu, sigma = normal_args_check(mu, sigma)

    const: jnp.ndarray = -0.5 * jnp.log(2 * jnp.pi)
    c: jnp.ndarray = lax.sub(const, jnp.log(sigma)) 
    e: jnp.ndarray = lax.div(lax.pow(lax.sub(x, mu), 2), 2 * lax.pow(sigma, 2))
    logpdf: jnp.ndarray = lax.sub(c, e)
    return logpdf.reshape(xshape)


def pdf(x: ArrayLike, mu: float=0.0, sigma: float=1.0) -> Array:
    r"""Probability density function of the normal distribution.
    
    The normal pdf is defined as:

    .. math::

        f(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)

    Args:
        x: arraylike, value(s) at which to evaluate the pdf.
        mu: Mean/location of the normal distribution.
        sigma: Standard deviation of the normal distribution.

    Returns:
        array of pdf values.
    """
    return jnp.exp(logpdf(x, mu, sigma))


def logcdf(x: ArrayLike, mu: float=0.0, sigma: float=1.0) -> Array:
    r"""Log-cumulative distribution function of the normal distribution.
    
    The normal pdf is defined as:

    .. math::

        f(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)

    Args:
        x: arraylike, value(s) at which to evaluate the cdf.
        mu: Mean/location of the normal distribution.
        sigma: Standard deviation of the normal distribution.

    Returns:
        array of log-cdf values.
    """
    x, xshape = _univariate_input(x)
    mu, sigma = normal_args_check(mu, sigma)

    z: jnp.ndarray = lax.div(lax.sub(x, mu), sigma)
    logcdf: jnp.ndarray = special.log_ndtr(z)
    return logcdf.reshape(xshape)


def cdf(x: ArrayLike, mu: float=0.0, sigma: float=1.0) -> Array:
    r"""Cumulative distribution function of the normal distribution.
    
    The normal pdf is defined as:

    .. math::

        F(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)

    Args:
        x: arraylike, value(s) at which to evaluate the cdf.
        mu: Mean/location of the normal distribution.
        sigma: Standard deviation of the normal distribution.

    Returns:
        array of cdf values.
    """
    x, xshape = _univariate_input(x)
    mu, sigma = normal_args_check(mu, sigma)

    z: jnp.ndarray = lax.div(lax.sub(x, mu), sigma)
    cdf: jnp.ndarray = special.ndtr(z)
    return cdf.reshape(xshape)


def ppf(q: ArrayLike, mu: float=0.0, sigma: float=1.0) -> Array:
    r"""Percent point function (inverse of cdf) of the normal distribution.
    
    The normal pdf is defined as:

    .. math::

        F(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)

    Args:
        q: arraylike, value(s) at which to evaluate the ppf.
        mu: Mean/location of the normal distribution.
        sigma: Standard deviation of the normal distribution.

    Returns:
        array of ppf values.
    """
    q, qshape = _univariate_input(q)
    mu, sigma = normal_args_check(mu, sigma)

    z: jnp.array = jnp.asarray(special.ndtri(q), dtype=float)
    x: jnp.ndarray = lax.add(mu, lax.mul(sigma, z))
    return x.reshape(qshape)


def rvs(shape: tuple = (1,), key: ArrayLike = DEFAULT_RANDOM_KEY, mu: float = 0.0, sigma: float = 1.0) -> Array:
    r"""Generate random samples from the normal distribution.
    
    The normal pdf is defined as:

    .. math::

        f(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)

    Args:
        shape: The shape of the random number array to generate.
        key: Key for random number generation.
        mu: Mean/location of the normal distribution.
        sigma: Standard deviation of the normal distribution.

    Returns:
        array of random variates.
    """
    mu, sigma = normal_args_check(mu, sigma)
    z: jnp.array = random.normal(key, shape)
    return lax.add(lax.mul(z, sigma), mu)


# def cdf_approx(key: ArrayLike, x: ArrayLike, mu=0.0, sigma=1.0, num_points: int = 100,) -> Array:
#     r"""Approximate the cdf of a normal distribution by evaluating it at a set of 
#     points. Uses a linear interpolation to reduce the required number of 
#     similated random variables.

#     Args:
#         key: PRNGKey for random number generation.
#         x: arraylike, value(s) at which to evaluate the cdf. This must be a
#         one-dimensional array.
#         mu: Mean/location of the normal distribution.
#         sigma: Standard deviation of the normal distribution.
#         num_points: Number of simulated random points at which to evaluate the 
#         cdf.
        
#     Returns:
#         Array of cdf values.
#     """
#     return _cdf_approx(key=key, rvs_func=rvs, x=x, num_points=num_points, params=(mu, sigma))


# def ppf_approx(key: ArrayLike, q: ArrayLike, mu=0.0, sigma=1.0, num_points: int = 100) -> Array:
#     r"""Approximate the percent point function (inverse of cdf) of a normal 
#     distribution by evaluating it at a set of points. Uses a linear 
#     interpolation to reduce the required number of simulated random variables.
    
#     Args:
#         key: PRNGKey for random number generation.
#         q: arraylike, value(s) at which to evaluate the ppf.
#         num_points: Number of simulated random points at which to evaluate the 
#         pdf.
#         mu: Mean/location of the normal distribution.
#         sigma: Standard deviation of the normal distribution.
#     """
#     return _ppf_approx(key=key, rvs_func=rvs, q=q, num_points=num_points, params=(mu, sigma))


def fit(x: ArrayLike) -> dict:
    r"""Fit the parameters of a normal distribution to the data, using maximum 
    likelihood estimation.
    
    Args:
        x: arraylike, data to fit the distribution to.

    Returns:
        dictionary of fitted parameters.
    """
    x: jnp.ndarray = _univariate_input(x)[0]
    mu: jnp.ndarray = x.mean()
    sigma: jnp.ndarray = x.std()
    return normal_params_dict(mu=mu, sigma=sigma)


def stats(mu: float = 0.0, sigma: float = 1.0) -> dict:
    r"""Distribution statistics for the normal distribution. Returns the mean,
    median, mode, variance, skewness and (excess) kurtosis of the distribution
    
    Args:
        mu: Mean/location of the normal distribution.
        sigma: Standard deviation of the normal distribution.

    Returns:
        Dictionary of statistics.
    """
    mu, sigma = normal_args_check(mu, sigma)
    return {
        'mean': mu,
        'median': mu,
        'mode': mu,
        'variance': lax.pow(sigma, 2),
        'skewness': 0.0,
        'kurtosis': 0.0,
    }