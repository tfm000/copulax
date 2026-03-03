"""JAX-jittable goodness-of-fit tests for univariate distributions.

Implements the Kolmogorov-Smirnov and Cramér-von Mises one-sample tests.
Both test statistics and asymptotic p-values are fully jit-compilable.
"""

import jax.numpy as jnp
from jax import jit, lax
from jax.scipy import special

from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src.special import kv


###############################################################################
# Kolmogorov-Smirnov test
###############################################################################
def _ks_pvalue(d: Scalar, n: Scalar) -> Scalar:
    r"""Two-sided Kolmogorov-Smirnov p-value (asymptotic).

    Uses the Kolmogorov survival function:

    .. math::
        P(D_n \ge d) \approx 2 \sum_{k=1}^{K} (-1)^{k+1} e^{-2 k^2 \lambda^2}

    where :math:`\lambda = (\sqrt{n} + 0.12 + 0.11 / \sqrt{n}) \cdot d`.

    Reference: Marsaglia, Tsang & Wang (2003),
    "Evaluating Kolmogorov's Distribution".
    """
    sqrt_n = jnp.sqrt(n)
    lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d

    ks = jnp.arange(1, 101)
    signs = jnp.where(ks % 2 == 1, 1.0, -1.0)
    terms = signs * jnp.exp(-2.0 * ks**2 * lam**2)
    p = 2.0 * jnp.sum(terms)
    return jnp.clip(p, 0.0, 1.0)


@jit
def ks_test(x: jnp.ndarray, dist, params: dict) -> dict:
    r"""One-sample Kolmogorov-Smirnov goodness-of-fit test.

    Tests whether *x* was drawn from the distribution described by
    *dist* and *params*.

    Args:
        x (ArrayLike): Observed sample (1-D).
        dist: A copulAX ``Univariate`` distribution object.
        params (dict): Distribution parameters (as returned by ``fit``).

    Returns:
        dict: ``{'statistic': D_n, 'p_value': p}``
    """
    x, _ = _univariate_input(x)
    x_sorted = jnp.sort(x.flatten())
    n = jnp.asarray(x_sorted.size, dtype=float)

    # theoretical CDF at the sorted observations
    f_x = dist.cdf(x=x_sorted, params=params).flatten()

    # empirical CDF: i/n  (i = 1, ..., n)
    i = jnp.arange(1, x_sorted.size + 1, dtype=float)
    ecdf_upper = i / n  # F_n(x_i)
    ecdf_lower = (i - 1.0) / n  # F_n(x_{i-1})

    d_plus = jnp.max(ecdf_upper - f_x)
    d_minus = jnp.max(f_x - ecdf_lower)
    d_n = jnp.maximum(d_plus, d_minus)

    p_value = _ks_pvalue(d_n, n)
    return {"statistic": d_n, "p_value": p_value}


###############################################################################
# Cramér-von Mises test
###############################################################################
def _cvm_pvalue(w2: Scalar) -> Scalar:
    r"""Asymptotic Cramér-von Mises p-value.

    Uses the representation by Csörgő & Faraway (1996) based on the
    eigenvalue expansion.  The CDF is:

    .. math::
        F(w) = \sum_{j=0}^{\infty} (-1)^j \,
            \frac{\Gamma(j + \tfrac{1}{2})}{j!\,\pi} \,
            \sqrt{\frac{4j+1}{\pi\,w}} \;
            e^{-\frac{(4j+1)^2}{16\,w}} \;
            K_{1/4}\!\!\left(\frac{(4j+1)^2}{16\,w}\right)

    The p-value is :math:`1 - F(w)`.
    """
    j = jnp.arange(0, 20, dtype=float)
    a = 4.0 * j + 1.0
    z = a**2 / (16.0 * w2)

    # K_{1/4}(z) via quadrature
    k_val = kv(0.25, z)

    # coefficients: (-1)^j * Gamma(j+1/2) / (j! * pi)
    log_gamma_half = special.gammaln(j + 0.5)
    log_factorial = special.gammaln(j + 1.0)
    log_coeff = log_gamma_half - log_factorial - jnp.log(jnp.pi)
    coeff = jnp.exp(log_coeff)
    signs = jnp.where(j.astype(int) % 2 == 0, 1.0, -1.0)

    sqrt_term = jnp.sqrt(a / (jnp.pi * w2))
    exp_term = jnp.exp(-z)

    terms = signs * coeff * sqrt_term * exp_term * k_val
    cdf = jnp.sum(terms)
    return jnp.clip(1.0 - cdf, 0.0, 1.0)


@jit
def cvm_test(x: jnp.ndarray, dist, params: dict) -> dict:
    r"""One-sample Cramér-von Mises goodness-of-fit test.

    Tests whether *x* was drawn from the distribution described by
    *dist* and *params*.

    Args:
        x (ArrayLike): Observed sample (1-D).
        dist: A copulAX ``Univariate`` distribution object.
        params (dict): Distribution parameters (as returned by ``fit``).

    Returns:
        dict: ``{'statistic': W2, 'p_value': p}``
    """
    x, _ = _univariate_input(x)
    x_sorted = jnp.sort(x.flatten())
    n = jnp.asarray(x_sorted.size, dtype=float)

    # theoretical CDF at sorted observations
    f_x = dist.cdf(x=x_sorted, params=params).flatten()

    # CvM statistic
    i = jnp.arange(1, x_sorted.size + 1, dtype=float)
    w2 = jnp.sum((f_x - (2.0 * i - 1.0) / (2.0 * n)) ** 2) + 1.0 / (12.0 * n)

    p_value = _cvm_pvalue(w2)
    return {"statistic": w2, "p_value": p_value}
