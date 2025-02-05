"""File containing the copulAX implementation of the uniform distribution."""
import jax.numpy as jnp
from jax import lax, random
from jax._src.typing import ArrayLike, Array

from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.univariate._metrics import (_loglikelihood, _aic, _bic)


def uniform_args_check(a: float | ArrayLike, b: float | ArrayLike) -> tuple:
    return jnp.asarray(a, dtype=float), jnp.asarray(b, dtype=float)


def uniform_params_dict(a: float, b: float) -> dict:
    a, b = uniform_args_check(a=a, b=b)
    return {'a': a, 'b': b}


def support(a: float = 0.0, b: float = 1.0, *args, **kwargs) -> tuple[float, float]:
    r"""The support of the distribution is the subset of x for which the pdf 
    is non-zero. 
    
    Args:
        a: Lower bound of the uniform distribution.
        b: Upper bound of the uniform distribution.

    Returns:
        (float, float): Tuple containing the support of the distribution.
    """
    return float(a), float(b)


def logpdf(x: ArrayLike, a: float = 0.0, b: float = 1.0) -> Array:
    r"""Log-probability density function of the uniform distribution.
    
    The uniform pdf is defined as:

    .. math::

        f(x|a, b) = \frac{1}{b - a} \mathbb{1}_{[a, b]}(x)

    Args:
        x: arraylike, value(s) at which to evaluate the pdf.
        a: Lower bound of the uniform distribution.
        b: Upper bound of the uniform distribution.

    Returns:
        array of log-pdf values.
    """
    x, xshape = _univariate_input(x)
    a, b = uniform_args_check(a=a, b=b)

    log_pdf: jnp.ndarray = -jnp.log(lax.sub(b, a))
    log_pdf = jnp.where(jnp.logical_and(x >= a, x <= b), log_pdf, -jnp.inf)
    return log_pdf.reshape(xshape)


def pdf(x: ArrayLike, a=0.0, b=1.0) -> Array:
    r"""Probability density function of the uniform distribution.
    
    The uniform pdf is defined as:

    .. math::

        f(x|a, b) = \frac{1}{b - a} \mathbb{1}_{[a, b]}(x)

    Args:
        x: arraylike, value(s) at which to evaluate the pdf.
        a: Lower bound of the uniform distribution.
        b: Upper bound of the uniform distribution.

    Returns:
        array of pdf values.
    """
    return jnp.exp(logpdf(x=x, a=a, b=b))


def logcdf(x: ArrayLike, a=0.0, b=1.0) -> Array:
    r"""Log-cumulative distribution function of the uniform distribution.
    
    The uniform cdf is defined as:

    .. math::

        F(x|a, b) = \frac{x - a}{b - a} \mathbb{1}_{[a, b]}(x)

    Args:
        x: arraylike, value(s) at which to evaluate the cdf.
        a: Lower bound of the uniform distribution.
        b: Upper bound of the uniform distribution.

    Returns:
        array of log-cdf values.
    """
    x, xshape = _univariate_input(x)
    a, b = uniform_args_check(a=a, b=b)

    log_cdf: jnp.ndarray = jnp.log(lax.sub(x, a)) - jnp.log(lax.sub(b, a))
    log_cdf = jnp.where(jnp.logical_and(x >= a, x <= b), log_cdf, -jnp.inf)
    return log_cdf.reshape(xshape)


def cdf(x: ArrayLike, a=0.0, b=1.0) -> Array:
    r"""Cumulative distribution function of the uniform distribution.
    
    The uniform cdf is defined as:

    .. math::

        F(x|a, b) = \frac{x - a}{b - a} \mathbb{1}_{[a, b]}(x)

    Args:
        x: arraylike, value(s) at which to evaluate the cdf.
        a: Lower bound of the uniform distribution.
        b: Upper bound of the uniform distribution.

    Returns:
        array of cdf values.
    """
    return jnp.exp(logcdf(x=x, a=a, b=b))


def ppf(q: ArrayLike, a=0.0, b=1.0) -> Array:
    r"""Percent point function (inverse cdf) of the uniform distribution.
    
    The uniform cdf is defined as:

    .. math::

        F(x|a, b) = \frac{x - a}{b - a} \mathbb{1}_{[a, b]}(x)

    Args:
        q: arraylike, value(s) at which to evaluate the ppf.
        a: Lower bound of the uniform distribution.
        b: Upper bound of the uniform distribution.

    Returns:
        array of ppf values.
    """
    q, qshape = _univariate_input(q)
    a, b = uniform_args_check(a=a, b=b)
    
    ppf_values: jnp.ndarray = lax.add(a, lax.mul(q, lax.sub(b, a)))
    ppf_values = jnp.where(jnp.logical_and(q >= 0, q <= 1), ppf_values, jnp.nan)
    return ppf_values.reshape(qshape)


def rvs(shape=(1,), key=DEFAULT_RANDOM_KEY, a=0.0, b=1.0) -> Array:
    r"""Generate random samples from the uniform distribution.
    
    Args:
        shape: The shape of the random number array to generate.
        key: Key for random number generation.
        a: Lower bound of the uniform distribution.
        b: Upper bound of the uniform distribution.

    Returns:
        array of random samples.
    """
    a, b = uniform_args_check(a=a, b=b)
    return random.uniform(key=key, shape=shape, minval=a, maxval=b)


# def cdf_approx(key, x: ArrayLike, a=0.0, b=1.0, num_points: int = 100) -> Array:
#     r"""Approximate the cdf of the uniform distribution using numerical integration.
    
#     Args:
#         key: PRNGKey, key for random number generation.
#         x: arraylike, value(s) at which to evaluate the cdf. This must be a 
#         one-dimensional array.
#         a: Lower bound of the uniform distribution.
#         b: Upper bound of the uniform distribution.
#         num_points: Number of simulated random points at which to evaluate the 
#         cdf.

#     Returns:
#         Array of cdf values.
#     """
#     x: jnp.ndarray = _univariate_input(x)
#     a, b = uniform_args_check(a, b)

#     return _cdf_approx(key=key, rvs_func=rvs, x=x, num_points=num_points, params=(a, b))


# def ppf_approx(key, q: ArrayLike, a=0.0, b=1.0, num_points: int = 100) -> Array:
#     r"""Approximate the ppf of the uniform distribution using numerical integration.
    
#     Args:
#         key: PRNGKey, key for random number generation.
#         q: arraylike, value(s) at which to evaluate the ppf. This must be a 
#         one-dimensional array.
#         a: Lower bound of the uniform distribution.
#         b: Upper bound of the uniform distribution.
#         num_points: Number of simulated random points at which to evaluate the 
#         ppf.

#     Returns:
#         Array of ppf values.
#     """
#     q: jnp.ndarray = _univariate_input(q)
#     a, b = uniform_args_check(a, b)

#     return _ppf_approx(key=key, rvs_func=rvs, q=q, num_points=num_points, params=(a, b))


def fit(x: ArrayLike, *args, **kwargs) -> dict:
    r"""Fit the parameters of a uniform distribution to the data, using maximum 
    likelihood estimation.
    
    Args:
        x: arraylike, data to fit the distribution to.

    Returns:
        dictionary of fitted parameters.
    """
    x, _ = _univariate_input(x)
    a: float = jnp.min(x)
    b: float = jnp.max(x)
    return uniform_params_dict(a=a, b=b)


def stats(a: float = 0.0, b: float = 1.0) -> dict:
    r"""Distribution statistics for the continuous uniform distribution. 
    Returns the mean, median, variance, skewness and (excess) kurtosis of the
    distribution.

    Args:
        a: Lower bound of the uniform distribution.
        b: Upper bound of the uniform distribution.

    Returns:
        dictionary of statistics.
    """
    mean = (a + b) / 2
    variance = lax.pow(b - a, 2) / 12
    return {'mean': mean, 'median': mean, 'variance': variance, 'skewness': 0.0, 'kurtosis': -6 / 5}


def loglikelihood(x: ArrayLike, a: float = 0.0, b: float = 1.0) -> float:
    r"""Log-likelihood of the uniform distribution.
    
    Args:
        x: arraylike, value(s) at which to evaluate the log-likelihood.
        a: Lower bound of the uniform distribution.
        b: Upper bound of the uniform distribution.

    Returns:
        float, log-likelihood of the distribution.
    """
    return _loglikelihood(logpdf_func=logpdf, x=x, 
                          params=uniform_params_dict(a=a, b=b))


def aic(x: ArrayLike, a: float = 0.0, b: float = 1.0) -> float:
    r"""Akaike Information Criterion (AIC) of the uniform distribution.

    Args:
        x: arraylike, data to fit the distribution to.
        a: Lower bound of the uniform distribution.
        b: Upper bound of the uniform distribution.

    Returns:
        float, AIC of the distribution.
    """
    return _aic(logpdf_func=logpdf, params=uniform_params_dict(a=a, b=b), x=x)


def bic(x: ArrayLike, a: float = 0.0, b: float = 1.0) -> float:
    r"""Bayesian Information Criterion (BIC) of the uniform distribution.

    Args:
        x: arraylike, data to fit the distribution to.
        a: Lower bound of the uniform distribution.
        b: Upper bound of the uniform distribution.

    Returns:
        float, BIC of the distribution.
    """
    return _bic(logpdf_func=logpdf, params=uniform_params_dict(a=a, b=b), x=x)
