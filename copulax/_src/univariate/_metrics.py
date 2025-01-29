"""Distribution fit metrics."""
import jax.numpy as jnp
from typing import Callable

from copulax._src.metrics import (
    _loglikelihood, 
    _mle_objective,
      _aic as __aic, 
      _bic as __bic,
      )


def _aic(logpdf_func: Callable, params: dict, x: jnp.ndarray) -> float:
    """Akaike Information Criterion (AIC) for model selection.
    Best model selected via minimising AIC.

    Args:
        logpdf_func: The log probability density function of the distribution.
        params: The parameters of the distribution.
        x: The data.
    """
    return __aic(logpdf_func=logpdf_func, params=params, x=x, k=len(params))


def _bic(logpdf_func: Callable, params: dict, x: jnp.ndarray) -> float:
    """Bayesian Information Criterion (BIC) for model selection.
    Best model selected via minimising BIC.

    Args:
        logpdf_func: The log probability density function of the distribution.
        params: The parameters of the distribution.
        x: The data.
    """
    return __bic(logpdf_func=logpdf_func, params=params, x=x, k=len(params))