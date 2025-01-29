"""Fit metrics for univariate distributions."""
import jax.numpy as jnp
from typing import Callable


def _loglikelihood(logpdf_func: Callable, params: dict, x: jnp.ndarray, **kwargs) -> float:
    return jnp.sum(logpdf_func(x=x, **params, **kwargs))


def _mle_objective(logpdf_func: Callable, params: dict, x: jnp.ndarray, **kwargs) -> float:
    return -_loglikelihood(logpdf_func=logpdf_func, params=params, x=x, **kwargs)


def _aic(logpdf_func: Callable, params: dict, x: jnp.ndarray, k: int, **kwargs) -> float:
    """Akaike Information Criterion (AIC) for model selection.
    Best model selected via minimising AIC.

    Args:
        logpdf_func: The log probability density function of the distribution.
        params: The parameters of the distribution.
        x: The data.
        k: The number of parameters in the model.
    """
    return 2 * k - 2 * _loglikelihood(logpdf_func=logpdf_func, params=params, x=x, **kwargs)


def _bic(logpdf_func: Callable, params: dict, x: jnp.ndarray, k: int, **kwargs) -> float:
    """Bayesian Information Criterion (BIC) for model selection.
    Best model selected via minimising BIC.

    Args:
        logpdf_func: The log probability density function of the distribution.
        params: The parameters of the distribution.
        x: The data.
        k: The number of parameters in the model.
    """
    n: int = x.shape[0]
    return k * jnp.log(n) - 2 * _loglikelihood(logpdf_func=logpdf_func, params=params, x=x, **kwargs)


# def kl_divergence(logpdf_P: Callable, logpdf_Q: Callable, params_P: dict, params_Q: dict, x: jnp.ndarray) -> float:
#     """Kullback-Leibler divergence between two distributions.
#     A smaller value indicates that the approximating distribution Q is closer 
#     to the true distribution P.

#     Note:
#         https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    
#     Args:
#         logpdf_P: The log probability density function of the assumed true distribution P.
#         logpdf_Q: The log probability density function of the approximating distribution Q.
#         params_P: The parameters of the first distribution, as a dictionary.
#         params_Q: The parameters of the second distribution, as a dictionary.
#         x: Data sample drawn from P.
#     """
#     logpdf_vals_P: jnp.ndarray = logpdf_P(x=x, **params_P)
#     pdf_vals_P: jnp.ndarray = jnp.exp(logpdf_vals_P)
#     logpdf_vals_Q: jnp.ndarray = logpdf_Q(x=x, **params_Q)

#     return jnp.sum(pdf_vals_P * (logpdf_vals_P - logpdf_vals_Q)) # todo -> wrong, needs to integrate over x when continuous.
