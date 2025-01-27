"""File containing the copulAX implementation of the Gamma distribution.
Uses the rate parameterization of the Gamma distribution.
"""
import jax.numpy as jnp
from jax import lax, random, scipy
from jax._src.typing import ArrayLike, Array

from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.optimize import projected_gradient
from copulax._src.univariate import lognormal


def gamma_args_check(alpha: float | ArrayLike, beta: float | ArrayLike) -> None:
    alpha: jnp.ndarray = jnp.asarray(alpha, dtype=float)
    beta: jnp.ndarray = jnp.asarray(beta, dtype=float)
    return alpha, beta


def gamma_params_dict(alpha: float | ArrayLike, beta: float | ArrayLike) -> dict:
    alpha, beta = gamma_args_check(alpha, beta)
    return {"alpha": alpha, "beta": beta}


def support(*args, **kwargs) -> tuple[float, float]:
    r"""The support of the distribution is the subset of x for which the pdf 
    is non-zero. 
    
    Returns:
        (float, float): Tuple containing the support of the distribution.
    """
    return 0.0, jnp.inf


def logpdf(x: ArrayLike, alpha: float = 1.0, beta: float = 1.0) -> Array:
    r"""The log of the probability density function (pdf) of the Gamma 
    distribution.

    Note:
        copulAX uses the rate parameterization of the Gamma distribution.
        https://en.wikipedia.org/wiki/Gamma_distribution
    
    Args:
        x (ArrayLike): The input at which to evaluate the log-pdf.
        alpha: The shape parameter of the Gamma distribution.
        beta: The rate parameter of the Gamma distribution.
    
    Returns:
        array of cdf values.
    """
    x, xshape = _univariate_input(x)
    alpha, beta = gamma_args_check(alpha=alpha, beta=beta)
    
    logpdf: jnp.ndarray = (alpha * jnp.log(beta) - lax.lgamma(alpha) + (alpha - 1) * jnp.log(x) - beta * x)
    return logpdf.reshape(xshape)


def pdf(x: ArrayLike, alpha: float = 1.0, beta: float = 1.0) -> Array:
    r"""The probability density function (pdf) of the Gamma distribution.

    Note:
        copulAX uses the rate parameterization of the Gamma distribution.
        https://en.wikipedia.org/wiki/Gamma_distribution
    
    Args:
        x (ArrayLike): The input at which to evaluate the pdf.
        alpha: The shape parameter of the Gamma distribution.
        beta: The rate parameter of the Gamma distribution.
    
    Returns:
        array of cdf values.
    """
    return jnp.exp(logpdf(x=x, alpha=alpha, beta=beta))


def cdf(x: ArrayLike, alpha: float = 1.0, beta: float = 1.0) -> Array:
    r"""The cumulative distribution function (cdf) of the Gamma distribution.
    
    Note:
        copulAX uses the rate parameterization of the Gamma distribution.
        https://en.wikipedia.org/wiki/Gamma_distribution

    Args:
        x (ArrayLike): The input at which to evaluate the cdf.
        alpha: The shape parameter of the Gamma distribution.
        beta: The rate parameter of the Gamma distribution.

    Returns:
        array of cdf values.
    """
    x, xshape = _univariate_input(x)
    alpha, beta = gamma_args_check(alpha=alpha, beta=beta)
    cdf: jnp.ndarray = scipy.special.gammainc(a=alpha, x=beta*x)
    return cdf.reshape(xshape)


def logcdf(x: ArrayLike, alpha: float = 1.0, beta: float = 1.0) -> Array:
    r"""The log of the cumulative distribution function (cdf) of the Gamma 
    distribution.
    
    Note:
        copulAX uses the rate parameterization of the Gamma distribution.
        https://en.wikipedia.org/wiki/Gamma_distribution

    Args:
        x (ArrayLike): The input at which to evaluate the log-cdf.
        alpha: The shape parameter of the Gamma distribution.
        beta: The rate parameter of the Gamma distribution.

    Returns:
        array of log-cdf values.
    """
    return jnp.log(cdf(x=x, alpha=alpha, beta=beta))


def ppf(q: ArrayLike, alpha: float = 1.0, beta: float = 1.0) -> Array:
    r"""The percent point function (ppf) of the Gamma distribution.
    
    Note:
        copulAX uses the rate parameterization of the Gamma distribution.
        https://en.wikipedia.org/wiki/Gamma_distribution

    Args:
        q (ArrayLike): The input at which to evaluate the ppf.
        alpha: The shape parameter of the Gamma distribution.
        beta: The rate parameter of the Gamma distribution.

    Returns:
        array of ppf values.
    """
    q, qshape = _univariate_input(q)
    alpha, beta = gamma_args_check(alpha=alpha, beta=beta)
    ppf: jnp.ndarray = scipy.special.gammaincinv(a=alpha, q=q) / beta
    return ppf.reshape(qshape)


def rvs(shape: tuple = (1,), key: Array = DEFAULT_RANDOM_KEY, alpha: float = 1.0, beta: float = 1.0) -> Array:
    r"""Generate random variates of the Gamma distribution.
    
    Note:
        copulAX uses the rate parameterization of the Gamma distribution.
        https://en.wikipedia.org/wiki/Gamma_distribution

        If you intend to jit wrap this functino, ensure that 'shape' is a 
        static argument.

    Args:
        shape: The output shape of the random variates.
        key: An array key for JAX's random number generator.
        alpha: The shape parameter of the Gamma distribution.
        beta: The rate parameter of the Gamma distribution.

    Returns:
        array of random variates.
    """
    alpha, beta = gamma_args_check(alpha=alpha, beta=beta)

    unscales_rvs: jnp.ndarray = random.gamma(key, shape=shape, a=alpha)
    return unscales_rvs / beta


def _mle_objective(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    alpha, beta = params
    return -jnp.sum(logpdf(x=x, alpha=alpha, beta=beta))


def _fit_mle(x: ArrayLike) -> tuple[dict, float]:
    beta0: float = lognormal.rvs(shape=())
    alpha0: float = x.mean() * beta0
    params0: jnp.ndarray = jnp.array([alpha0, beta0])

    res = projected_gradient(f=_mle_objective, x0=params0, 
                             projection_method='projection_non_negative', x=x)
    alpha, beta = res['x']
    return gamma_params_dict(alpha=alpha, beta=beta)#, res['fun']


def fit(x: ArrayLike) -> dict:
    r"""Fit the parameters of the Gamma distribution to the data using maximum
    likelihood estimation.

    Note:
        copulAX uses the rate parameterization of the Gamma distribution.
        https://en.wikipedia.org/wiki/Gamma_distribution
    
    Args:
        x (ArrayLike): The data to fit the Gamma distribution to.
    
    Returns:
        dict: A dictionary containing the shape and rate parameters of the Gamma 
        distribution.
    """
    x: jnp.ndarray = _univariate_input(x)[0]
    return _fit_mle(x)


def stats(alpha: float = 1.0, beta: float = 1.0) -> dict[str, float]:
    r"""The mean, mode, variance, skewness and (excess) kurtosis of the Gamma 
    distribution.
    
    Note:
        copulAX uses the rate parameterization of the Gamma distribution.
        https://en.wikipedia.org/wiki/Gamma_distribution

    Args:
        alpha: The shape parameter of the Gamma distribution.
        beta: The rate parameter of the Gamma distribution.

    Returns:
        dict: A dictionary containing the first four moments of the Gamma 
        distribution.
    """
    alpha, beta = gamma_args_check(alpha=alpha, beta=beta)
    mean: float = alpha / beta
    mode: float = jnp.where(alpha >= 1.0, (alpha - 1) / beta, 0.0)
    variance: float = alpha / (beta ** 2)
    skewness: float = 2.0 / jnp.sqrt(alpha)
    kurtosis: float = 6.0 / alpha
    return {"mean": mean, "mode": mode, "variance": variance, "skewness": skewness, "kurtosis": kurtosis}


