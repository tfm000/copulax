"""Sample statistics implemented in JAX.

All functions are JIT-compatible and support gradient flow.
"""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def skew(x: ArrayLike, bias: bool = True) -> Array:
    r"""Compute the sample skewness of a 1-D array.

    Measures the asymmetry of the distribution of values around the mean.
    Positive values indicate a right-skewed distribution; negative values
    indicate a left-skewed distribution.

    The biased (default) estimator is:

    .. math::
        \hat{\gamma}_1 = \frac{\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^3}
                              {\left(\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2\right)^{3/2}}

    The unbiased (Fisher) correction multiplies by :math:`\frac{\sqrt{n(n-1)}}{n-2}`.

    Args:
        x: 1-D array-like of sample values.
        bias: If ``True`` (default), return the biased estimator.
            If ``False``, apply the G1 small-sample correction
            :math:`\sqrt{n(n-1)}\,/\,(n-2)`.

    Returns:
        Scalar JAX array containing the sample skewness.
    """
    x = jnp.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    mu = jnp.mean(x)
    deviations = x - mu
    var = jnp.mean(deviations**2)
    skewness = jnp.mean(deviations**3) / (var**1.5)
    correction = jnp.sqrt(n * (n - 1.0)) / (n - 2.0)
    return jnp.where(bias, skewness, correction * skewness)


def kurtosis(x: ArrayLike, fisher: bool = True, bias: bool = True) -> Array:
    r"""Compute the sample kurtosis of a 1-D array.

    Kurtosis measures the "tailedness" of a distribution. Two conventions
    are supported:

    * **Pearson** (``fisher=False``): raw fourth standardised moment.
    * **Fisher** (``fisher=True``, default): excess kurtosis =
      Pearson − 3, so that a normal distribution has excess kurtosis 0.

    The biased Pearson estimator is:

    .. math::
        \hat{\kappa} = \frac{\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^4}
                            {\left(\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2\right)^{2}}

    The unbiased correction (``bias=False``) follows the standard
    G2 formula:

    .. math::
        \text{kurt}_{\text{unbiased}} = \frac{(n+1)(n-1)}{(n-2)(n-3)}
            \left(\hat{\kappa} - \frac{3(n-1)}{n+1}\right)

    Args:
        x: 1-D array-like of sample values.
        fisher: If ``True`` (default), return excess kurtosis (Pearson − 3).
            If ``False``, return the Pearson kurtosis.
        bias: If ``True`` (default), return the biased estimator.
            If ``False``, apply the G2 small-sample correction.

    Returns:
        Scalar JAX array containing the sample kurtosis.
    """
    x = jnp.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    mu = jnp.mean(x)
    deviations = x - mu
    var = jnp.mean(deviations**2)
    kurt = jnp.mean(deviations**4) / (var**2)

    # G2 unbiased correction; the formula gives unbiased excess kurtosis,
    # so we add 3 to stay in Pearson space throughout
    kurt_unbiased_pearson = (
        (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * kurt - 3.0 * (n - 1.0))
    ) + 3.0
    kurt = jnp.where(bias, kurt, kurt_unbiased_pearson)
    return jnp.where(fisher, kurt - 3.0, kurt)
