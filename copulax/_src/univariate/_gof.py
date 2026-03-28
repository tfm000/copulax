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

# Kolmogorov series small-lambda threshold (mirrors scipy cephes/kolmogorov.c).
# Below lam_min, exp(-2*k^2*lam^2) ≈ 1 for all k and the alternating series
# diverges.  Derived from the dtype's log-underflow limit:
#   lam_min = pi / sqrt(8 * |log(tiny)|)
# float64 → ~0.042, float32 → ~0.119.
_KS_LAM_MIN = {
    jnp.float32: jnp.pi / jnp.sqrt(8.0 * (-jnp.log(jnp.finfo(jnp.float32).tiny))),
    jnp.float64: jnp.pi / jnp.sqrt(8.0 * (-jnp.log(jnp.finfo(jnp.float64).tiny))),
}


###############################################################################
# Kolmogorov-Smirnov test
###############################################################################
def _ks_pvalue(d: Scalar, n: Scalar) -> Scalar:
    r"""Two-sided Kolmogorov-Smirnov p-value (asymptotic).

    Uses the Kolmogorov survival function:

    .. math::
        P(D_n \ge d) \approx 2 \sum_{k=1}^{K} (-1)^{k+1} e^{-2 k^2 \lambda^2}

    where :math:`\lambda = (\sqrt{n} + 0.12 + 0.11 / \sqrt{n}) \cdot d`.

    For small :math:`\lambda`, the series terms ``exp(-2 k^2 \lambda^2)``
    all approach 1 and the alternating sum fails to converge.  Following
    scipy's ``cephes/kolmogorov.c``, we return ``p = 1`` when
    :math:`\lambda \le \pi / \sqrt{8 \, |\!\log(\mathrm{tiny})|}`, the
    point below which every term underflows for the input dtype.  The
    threshold is computed once at module load from ``jnp.finfo`` (see
    ``_KS_LAM_MIN``).

    References
    ----------
    Marsaglia, G., Tsang, W. W., & Wang, J. (2003).
    "Evaluating Kolmogorov's Distribution". JOSS 8(18).
    """
    sqrt_n = jnp.sqrt(n)
    lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d

    lam_min = _KS_LAM_MIN.get(d.dtype, _KS_LAM_MIN[jnp.float64])

    ks = jnp.arange(1, 101)
    signs = jnp.where(ks % 2 == 1, 1.0, -1.0)
    terms = signs * jnp.exp(-2.0 * ks**2 * lam**2)
    p_series = 2.0 * jnp.sum(terms)

    # For lam <= lam_min the series is unreliable; true p-value is 1.0.
    # jnp.where evaluates both branches but selects based on condition,
    # keeping gradients clean (no NaN from the unused branch).
    p = jnp.where(lam <= lam_min, 1.0, p_series)
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

    Uses the representation by Csörgő & Faraway (1996) eq. 1.2 based on
    the eigenvalue expansion.  The CDF is a sum of **all-positive** terms
    (no alternating signs):

    .. math::
        F(w) = \sum_{j=0}^{\infty}
            \frac{\Gamma(j + \tfrac{1}{2})}{j!\,\pi} \,
            \sqrt{\frac{4j+1}{\pi\,w}} \;
            e^{-\frac{(4j+1)^2}{16\,w}} \;
            K_{1/4}\!\!\left(\frac{(4j+1)^2}{16\,w}\right)

    The p-value is :math:`1 - F(w)`.

    For :math:`W^2 \ge 5` the 20-term truncation of the series is
    sufficient to establish :math:`F \approx 1` (true *p* < 3.1e-12).
    To prevent a spurious rise in *p* for very large :math:`W^2` (caused
    by :math:`K_{1/4}(z \to 0)` divergence in the truncated tail), the
    CDF is clamped to 1.0 above that threshold.

    References
    ----------
    Csörgő, S. & Faraway, J. (1996). "The Exact and Asymptotic
    Distributions of Cramér-von Mises Statistics." JRSS-B 58(1), 221-234.
    """
    # Guard against w2 <= 0 (degenerate perfect fit): avoid division by
    # zero in z = a^2 / (16 * w2).  Use a safe denominator for tracing,
    # then select p = 1.0 at the end via jnp.where.
    w2_safe = jnp.where(w2 > 0, w2, 1.0)

    j = jnp.arange(0, 20, dtype=float)
    a = 4.0 * j + 1.0
    z = a**2 / (16.0 * w2_safe)

    # K_{1/4}(z) via quadrature
    k_val = kv(0.25, z)

    # coefficients: Gamma(j+1/2) / (j! * pi)  — ALL POSITIVE (no (-1)^j)
    # Reference: Csörgő & Faraway (1996) eq. 1.2; confirmed by scipy source
    log_gamma_half = special.gammaln(j + 0.5)
    log_factorial = special.gammaln(j + 1.0)
    log_coeff = log_gamma_half - log_factorial - jnp.log(jnp.pi)
    coeff = jnp.exp(log_coeff)

    sqrt_term = jnp.sqrt(a / (jnp.pi * w2))
    exp_term = jnp.exp(-z)

    terms = coeff * sqrt_term * exp_term * k_val
    cdf = jnp.sum(terms)

    # Enforce CDF monotonicity.  For W² > ~30 the 20-term truncation
    # becomes unreliable because K_{1/4}(z → 0) diverges, dragging the
    # partial sum spuriously below 1.  At W² = 5 the true p < 3.1e-12,
    # so clamping CDF to 1.0 here loses nothing in practice.
    cdf = jnp.where(w2 >= 5.0, 1.0, cdf)

    p = jnp.where(w2 <= 0, 1.0, 1.0 - cdf)
    return jnp.clip(p, 0.0, 1.0)


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
