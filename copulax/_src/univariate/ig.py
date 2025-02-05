"""File containing the copulAX implementation of the Inverse Gaussian distribution."""
import jax.numpy as jnp
from jax import lax, random, scipy
from jax._src.typing import ArrayLike, Array
from tensorflow_probability.substrates import jax as tfp

from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.optimize import projected_gradient
from copulax._src.univariate import gamma
from copulax._src.univariate._metrics import (_loglikelihood, _aic, _bic, _mle_objective as __mle_objective)


def ig_args_check(alpha: float | ArrayLike, beta: float | ArrayLike) -> None:
    alpha: jnp.ndarray = jnp.asarray(alpha, dtype=float)
    beta: jnp.ndarray = jnp.asarray(beta, dtype=float)
    return alpha, beta


def ig_params_dict(alpha: float | ArrayLike, beta: float | ArrayLike) -> dict:
    alpha, beta = ig_args_check(alpha, beta)
    return {"alpha": alpha, "beta": beta}


def support(*args, **kwargs) -> tuple[float, float]:
    r"""The support of the distribution is the subset of x for which the pdf 
    is non-zero. 
    
    Returns:
        (float, float): Tuple containing the support of the distribution.
    """
    return 0.0, jnp.inf


def logpdf(x: ArrayLike, alpha: float = 1.0, beta: float = 1.0) -> Array:
    r"""The log of the probability density function (pdf) of the Inverse 
    Gaussian (Ig) distribution.
    
    Args:
        x (ArrayLike): The input at which to evaluate the log-pdf.
        alpha: The shape parameter of the Inverse Gaussian distribution.
        beta: The rate parameter of the Inverse Gaussian distribution.
    
    Returns:
        array of log-pdf values.
    """
    x, xshape = _univariate_input(x)
    alpha, beta = ig_args_check(alpha=alpha, beta=beta)
    
    logpdf: jnp.ndarray = alpha * jnp.log(beta) - lax.lgamma(alpha) - (alpha + 1) * jnp.log(x) - (beta / x)
    return logpdf.reshape(xshape)


def pdf(x: ArrayLike, alpha: float = 1.0, beta: float = 1.0) -> Array:
    r"""The probability density function (pdf) of the Inverse Gaussian (Ig) 
    distribution.
    
    Args:
        x (ArrayLike): The input at which to evaluate the pdf.
        alpha: The shape parameter of the Inverse Gaussian distribution.
        beta: The rate parameter of the Inverse Gaussian distribution.
    
    Returns:
        array of pdf values.
    """
    return jnp.exp(logpdf(x=x, alpha=alpha, beta=beta))


def cdf(x: ArrayLike, alpha: float = 1.0, beta: float = 1.0) -> Array:
    r"""The cumulative distribution function (cdf) of the Inverse Gaussian (Ig) 
    distribution.
    
    Args:
        x (ArrayLike): The input at which to evaluate the cdf.
        alpha: The shape parameter of the Inverse Gaussian distribution.
        beta: The rate parameter of the Inverse Gaussian distribution.
    
    Returns:
        array of cdf values.
    """
    x, xshape = _univariate_input(x)
    alpha, beta = ig_args_check(alpha=alpha, beta=beta)
    cdf: jnp.ndarray = scipy.special.gammaincc(a=alpha, x=(beta / x))
    return cdf.reshape(xshape)


def logcdf(x: ArrayLike, alpha: float = 1.0, beta: float = 1.0) -> Array:
    r"""The log of the cumulative distribution function (cdf) of the Inverse 
    Gaussian (Ig) distribution.
    
    Args:
        x (ArrayLike): The input at which to evaluate the log-cdf.
        alpha: The shape parameter of the Inverse Gaussian distribution.
        beta: The rate parameter of the Inverse Gaussian distribution.
    
    Returns:
        array of log-cdf values.
    """
    return jnp.log(cdf(x=x, alpha=alpha, beta=beta))


def ppf(q: ArrayLike, alpha: float = 1.0, beta: float = 1.0) -> Array:
    r"""The percent point function (ppf) of the Inverse Gaussian (Ig) 
    distribution.
    
    Args:
        q (ArrayLike): The input at which to evaluate the ppf.
        alpha: The shape parameter of the Inverse Gaussian distribution.
        beta: The rate parameter of the Inverse Gaussian distribution.
    
    Returns:
        array of ppf values.
    """
    q, qshape = _univariate_input(q)
    alpha, beta = ig_args_check(alpha=alpha, beta=beta)
    # ppf: jnp.ndarray = beta / scipy.special.gammainccinv(a=alpha, q=q)
    ppf: jnp.ndarray = beta / tfp.math.igammacinv(a=alpha, p=q)
    return ppf.reshape(qshape)


def rvs(shape: tuple = (1,), key: Array=DEFAULT_RANDOM_KEY, alpha: float = 1.0, beta: float = 1.0) -> Array:
    r"""Generate random variates from the Inverse Gaussian (Ig) distribution.
    
    Args:
        shape (tuple): The desired shape of the output.
        key (Array): The PRNG key.
        alpha: The shape parameter of the Inverse Gaussian distribution.
        beta: The rate parameter of the Inverse Gaussian distribution.
    
    Returns:
        array of random variates.
    """
    alpha, beta = ig_args_check(alpha=alpha, beta=beta)
    return 1 / gamma.rvs(shape=shape, key=key, alpha=alpha, beta=beta)


def _mle_objective(params: jnp.ndarray, x: jnp.ndarray) -> Array:
    alpha, beta = params
    return __mle_objective(
        logpdf_func=logpdf, x=x, 
        params=ig_params_dict(alpha=alpha, beta=beta))


def _fit_mle(x: ArrayLike) -> tuple[dict, float]:
    key1, key2 = random.split(DEFAULT_RANDOM_KEY)
    params0: jnp.ndarray = jnp.array([gamma.rvs(shape=(), key=key1), 
                                      gamma.rvs(shape=(), key=key2)])
    
    res = projected_gradient(f=_mle_objective, x0=params0, 
                             projection_method='projection_non_negative', x=x)
    
    alpha, beta = res["x"]
    return ig_params_dict(alpha=alpha, beta=beta)#, res["fun"]


def fit(x: ArrayLike, *args, **kwargs) -> dict:
    r"""Estimate the parameters of the Inverse Gaussian (Ig) distribution using 
    maximum likelihood estimation.
    
    Args:
        x (ArrayLike): The data from which to estimate the parameters.
    
    Returns:
        dict: A dictionary containing the estimated parameters.
    """
    x: jnp.ndarray = _univariate_input(x)[0]
    return _fit_mle(x=x)


def stats(alpha: float = 1.0, beta: float = 1.0) -> dict[str, float]:
    r"""Calculate the mean, mode, variance, skewness and (excess) kurtosis of
    the Inverse Gaussian (Ig) distribution.
    
    Args:
        alpha: The shape parameter of the Inverse Gaussian distribution.
        beta: The rate parameter of the Inverse Gaussian distribution.
    
    Returns:
        dict: A dictionary containing the mean and variance of the distribution.
    """
    alpha, beta = ig_args_check(alpha=alpha, beta=beta)

    mean: float = jnp.where(alpha > 1.0, beta / (alpha - 1), jnp.nan)
    mode: float = beta / (alpha + 1)
    variance: float = jnp.where(alpha > 2.0, lax.pow(beta, 2) / (lax.pow(alpha - 1, 2) * (alpha - 1)), jnp.nan)
    skewness: float = jnp.where(alpha > 3.0, 4 * jnp.sqrt(alpha - 2) / (alpha - 3), jnp.nan)
    kurtosis: float = jnp.where(alpha > 4.0, 6 * (5*alpha - 11) / ((alpha - 3) * (alpha - 4)), jnp.nan)
    return {"mean": mean, "mode": mode, "variance": variance, "skewness": skewness, "kurtosis": kurtosis}


def loglikelihood(x: ArrayLike, alpha: float = 1.0, beta: float = 1.0) -> float:
    r"""Calculate the log-likelihood of the Inverse Gaussian (Ig) distribution.
    
    Args:
        x (ArrayLike): The data from which to calculate the log-likelihood.
        alpha: The shape parameter of the Inverse Gaussian distribution.
        beta: The rate parameter of the Inverse Gaussian distribution.
    
    Returns:
        float: The log-likelihood of the data given the parameters.
    """
    return _loglikelihood(logpdf_func=logpdf, x=x, 
                          params=ig_params_dict(alpha=alpha, beta=beta))


def aic(x: ArrayLike, alpha: float = 1.0, beta: float = 1.0) -> float:
    r"""Akaike Information Criterion (AIC) of the Inverse Gaussian (Ig)
    distribution.
    
    Args:
        x (ArrayLike): The data.
        alpha: The shape parameter of the Inverse Gaussian distribution.
        beta: The rate parameter of the Inverse Gaussian distribution.
    
    Returns:
        float: The AIC value.
    """
    return _aic(logpdf_func=logpdf, x=x, 
                params=ig_params_dict(alpha=alpha, beta=beta))


def bic(x: ArrayLike, alpha: float = 1.0, beta: float = 1.0) -> float:
    r"""Bayesian Information Criterion (BIC) of the Inverse Gaussian (Ig)
    distribution.
    
    Args:
        x (ArrayLike): The data.
        alpha: The shape parameter of the Inverse Gaussian distribution.
        beta: The rate parameter of the Inverse Gaussian distribution.
    
    Returns:
        float: The BIC value.
    """
    return _bic(logpdf_func=logpdf, x=x, 
                params=ig_params_dict(alpha=alpha, beta=beta))
