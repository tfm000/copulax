"""Modified Bessel function of the second kind and Student's t CDF.

References:
    - Bessel function integral representation:
      https://dlmf.nist.gov/10.32#E10
    - Asymptotic forms:
      https://dlmf.nist.gov/10.30
"""

from jax import lax, vmap
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jax.scipy import special
import jax

from copulax._src.typing import Scalar


# -----------------------------------------------------------------------------
# Legacy kv implementation (adaptive quadax quadrature), retained for reference.
# -----------------------------------------------------------------------------
# def _kv_integrand(w: Array, v: float, x: Array) -> Array:
#     r"""Integrand for the integral representation of $K_v(x)$.
#
#     Uses the substitution $w = e^t$ in the standard integral
#     $K_v(x) = \frac{1}{2}\int_0^\infty w^{v-1} \exp(-x(w + w^{-1})/2)\,dw$.
#     """
#     frac = jnp.pow(w, -1)
#     inner = -0.5 * x * (w + frac)
#     exp = lax.exp(inner)
#     return 0.5 * lax.pow(w, v - 1.0) * exp
#
#
# def _kv_single_x(v: float, xi: float) -> float:
#     """Evaluate $K_v$ at a single scalar point via quadrature."""
#     from quadax import quadgk
#
#     kv_val, _ = quadgk(_kv_integrand, interval=(0.0, jnp.inf), args=(v, xi))
#     return kv_val.reshape(())
#
#
# def kv_legacy(v: float, x: ArrayLike) -> Array:
#     r"""Legacy adaptive-quadrature implementation of $K_v(x)$."""
#     v = jnp.asarray(jnp.abs(v), dtype=float)
#     x = jnp.asarray(x, dtype=float)
#     xshape = x.shape
#     x_flat = x.flatten()
#     kv_raw = vmap(lambda xi: _kv_single_x(v, xi))(x_flat)
#     kv_adj = jnp.where(x_flat < 0, jnp.nan, kv_raw)
#     return kv_adj.reshape(xshape)


# Gauss-Laguerre nodes/weights for \int_0^\infty e^{-t} f(t) dt.
_KV_LAG_ORDER = 64
_KV_LAG_NODES_NP, _KV_LAG_WEIGHTS_NP = np.polynomial.laguerre.laggauss(_KV_LAG_ORDER)
_KV_LAG_NODES = jnp.asarray(_KV_LAG_NODES_NP, dtype=float)
_KV_LAG_WEIGHTS = jnp.asarray(_KV_LAG_WEIGHTS_NP, dtype=float)

_KV_SMALL_X = jnp.asarray(1e-8, dtype=float)
_KV_LARGE_X = jnp.asarray(40.0, dtype=float)
_KV_EULER_GAMMA = jnp.asarray(0.5772156649015329, dtype=float)
_KV_LOG_2 = jnp.log(jnp.asarray(2.0, dtype=float))
_KV_LOG_PI = jnp.log(jnp.asarray(jnp.pi, dtype=float))


def _kv_small_x(v: Array, x: Array) -> Array:
    r"""Small-x asymptotic approximation for K_v(x)."""

    def _k0_branch(_: None) -> Array:
        # K_0(x) ~ -log(x/2) - gamma  as x -> 0
        return -jnp.log(0.5 * x) - _KV_EULER_GAMMA

    def _nonzero_branch(_: None) -> Array:
        # Leading small-x term for v > 0:
        # K_v(x) ~ Gamma(v) * 2^{v-1} * x^{-v}
        v_safe = jnp.maximum(v, jnp.asarray(1e-6, dtype=float))
        leading = jnp.exp(
            special.gammaln(v_safe) + (v_safe - 1.0) * _KV_LOG_2 - v_safe * jnp.log(x)
        )
        # First correction improves accuracy while x is still tiny.
        denom = 1.0 - v_safe
        corr = jnp.where(
            jnp.abs(denom) < 1e-6, 1.0, 1.0 + (x * x) / (4.0 * denom)
        )
        return leading * corr

    return lax.cond(v < 1e-4, _k0_branch, _nonzero_branch, operand=None)


def _kv_large_x(v: Array, x: Array) -> Array:
    r"""Large-x asymptotic approximation for K_v(x)."""
    mu = 4.0 * v * v
    inv8x = 1.0 / (8.0 * x)
    series = (
        1.0
        + (mu - 1.0) * inv8x
        + ((mu - 1.0) * (mu - 9.0)) * 0.5 * (inv8x**2)
        + ((mu - 1.0) * (mu - 9.0) * (mu - 25.0)) * (1.0 / 6.0) * (inv8x**3)
    )
    log_pref = 0.5 * (_KV_LOG_PI - _KV_LOG_2 - jnp.log(x)) - x
    return jnp.exp(log_pref) * series


def _kv_laguerre(v: Array, x: Array) -> Array:
    r"""Fixed-order Gauss-Laguerre quadrature for K_v(x).

    Uses:
        K_v(x) = 0.5 * (2/x)^v * \int_0^\infty t^{v-1} exp(-t - x^2/(4t)) dt
    """
    t = _KV_LAG_NODES
    w = _KV_LAG_WEIGHTS
    log_terms = (v - 1.0) * jnp.log(t) - (x * x) / (4.0 * t)
    m = jnp.max(log_terms)
    weighted = jnp.sum(w * jnp.exp(log_terms - m))
    log_pref = -_KV_LOG_2 + v * (_KV_LOG_2 - jnp.log(x))
    log_val = log_pref + m + jnp.log(weighted + 1e-30)
    return jnp.exp(log_val)


def _kv_single_fast(v: Array, x: Array) -> Array:
    """Single-point dispatcher for fast/stable K_v(x)."""
    x_pos = jnp.maximum(x, jnp.asarray(1e-30, dtype=float))

    core = lax.cond(
        x_pos < _KV_SMALL_X,
        lambda xi: _kv_small_x(v, xi),
        lambda xi: lax.cond(
            xi > _KV_LARGE_X,
            lambda xj: _kv_large_x(v, xj),
            lambda xj: _kv_laguerre(v, xj),
            xi,
        ),
        x_pos,
    )
    core = jnp.where(x < 0.0, jnp.nan, core)
    core = jnp.where(x == 0.0, jnp.inf, core)
    return core


def kv(v: float, x: ArrayLike) -> Array:
    r"""Modified Bessel function of the second kind, $K_v(x)$.

    Implementation details:
    - Pure JAX, JIT-compatible, differentiable.
    - Uses fixed-order Gauss-Laguerre quadrature in the main region.
    - Uses asymptotic approximations for very small/large x.
    """
    v = jnp.asarray(jnp.abs(v), dtype=float).reshape(())
    x = jnp.asarray(x, dtype=float)
    xshape = x.shape
    x_flat = x.reshape(-1)
    vals = vmap(lambda xi: _kv_single_fast(v, xi))(x_flat)
    return vals.reshape(xshape)


def kv_asymptotic(v: float, x: ArrayLike) -> Array:
    """Alias retained for backward compatibility."""
    return kv(v, x)


########################################################################
# igammainv / igammacinv implementation
########################################################################


def _igammainv_impl(a, p):
    """Core computation for igammainv.

    Finds x such that gammainc(a, x) = p.

    Uses a Wilson-Hilferty / Cornish-Fisher initial approximation refined
    by Newton-Halley iterations (3 steps).

    References:
        Didonato, A. and Morris, A. (1986). Computation of the Incomplete
        Gamma Function Ratios and their Inverse.
        ACM Trans. Math. Softw. 12(4), 377-393.
    """
    q = 1.0 - p
    p_safe = jnp.clip(p, 1e-10, 1.0 - 1e-10)

    # --- Initial approximation ---

    # For a >= 1: Wilson-Hilferty / Cornish-Fisher normal approximation
    s = special.ndtri(p_safe)
    t = 1.0 / (9.0 * a)
    x_wh = a * jnp.power(jnp.maximum(1.0 - t + s * jnp.sqrt(t), 1e-4), 3.0)

    # For a < 1: (p * Gamma(a+1))^(1/a)
    x_small = jnp.power(
        jnp.maximum(p_safe * jnp.exp(special.gammaln(a + 1.0)), 1e-30),
        1.0 / jnp.maximum(a, 1e-10),
    )

    x = jnp.where(a >= 1.0, x_wh, x_small)
    # For a == 1 (exponential): -log(1-p) is exact
    x = jnp.where(jnp.equal(a, 1.0), -jnp.log1p(-p_safe), x)
    x = jnp.maximum(x, 1e-30)

    # --- Newton-Halley refinement (3 iterations) ---
    for _ in range(3):
        # fac = x^a * exp(-x) / Gamma(a)  (= x * gamma_pdf)
        fac = jnp.exp(a * jnp.log(x) - x - special.gammaln(a))
        # f / f'  using gammainc or gammaincc for numerical stability
        f_over_fprime = jnp.where(
            p <= 0.9,
            (special.gammainc(a, x) - p) * x / fac,
            -(special.gammaincc(a, x) - q) * x / fac,
        )
        # f'' / f' = -1 + (a - 1) / x
        fprime2_over_fprime = -1.0 + (a - 1.0) / x
        # Halley step (with Newton fallback when f''/f' is infinite)
        step = jnp.where(
            jnp.isinf(fprime2_over_fprime),
            f_over_fprime,
            f_over_fprime / (1.0 - 0.5 * f_over_fprime * fprime2_over_fprime),
        )
        x = jnp.where(jnp.equal(fac, 0.0), x, x - step)
        x = jnp.maximum(x, 1e-30)

    # Boundary conditions
    x = jnp.where(p <= 0.0, 0.0, x)
    x = jnp.where(p >= 1.0, jnp.inf, x)

    return x


def igammainv(a: ArrayLike, p: ArrayLike) -> Array:
    r"""Inverse of the regularized lower incomplete gamma function.

    Finds $x$ such that $\mathrm{gammainc}(a, x) = p$.

    Args:
        a: positive shape parameter.
        p: probability values in $[0, 1]$.

    Returns:
        Array of the same shape as the broadcast of ``a`` and ``p``.
    """
    return _igammainv_impl(jnp.asarray(a, dtype=float), jnp.asarray(p, dtype=float))


def igammacinv(a: ArrayLike, p: ArrayLike) -> Array:
    r"""Inverse of the regularized upper incomplete gamma function.

    Finds $x$ such that $\mathrm{gammaincc}(a, x) = p$,
    equivalently $\mathrm{gammainc}(a, x) = 1 - p$.

    Args:
        a: positive shape parameter.
        p: probability values in $[0, 1]$.

    Returns:
        Array of the same shape as the broadcast of ``a`` and ``p``.
    """
    return _igammainv_impl(
        jnp.asarray(a, dtype=float), 1.0 - jnp.asarray(p, dtype=float)
    )


########################################################################
# stdtr implementation
########################################################################
def _stdtr_impl(df: Scalar, t: Array) -> Array:
    """Primal Student-t CDF implementation."""
    # Use the regularized incomplete beta in a form that avoids
    # cancellation near t=0:
    #   CDF = 0.5 + 0.5 * sign(t) * I_z(1/2, df/2),
    #   z = t^2 / (df + t^2)
    t2 = t * t
    z = t2 / (df + t2)
    ib = special.betainc(0.5, 0.5 * df, z)
    return 0.5 + 0.5 * jnp.sign(t) * ib


def _stdtr_pdf_t(df: Scalar, t: Array) -> Array:
    """Derivative of stdtr(df, t) w.r.t. t (Student-t PDF at t)."""
    log_norm = (
        special.gammaln(0.5 * (df + 1.0))
        - special.gammaln(0.5 * df)
        - 0.5 * (jnp.log(df) + jnp.log(jnp.pi))
    )
    log_kernel = -0.5 * (df + 1.0) * jnp.log1p((t * t) / df)
    return jnp.exp(log_norm + log_kernel)


@jax.custom_vjp
def stdtr(df: Scalar, t: Array) -> Array:
    """Compute the cdf of the standard Student's t-distribution.

    Note:
        Gradient flow is supported for ``t``.
        Gradient flow for ``df`` is explicitly disabled (set to zero)
        because ``jax.scipy.special.betainc`` does not support gradients
        w.r.t. its first two arguments.

    Args:
        df (scalar): degrees of freedom.
        t (Array): values at which to evaluate the cdf.

    Returns:
        Array: cdf values of the standard Student's t-distribution.
    """
    # transforming args
    df: Scalar = jnp.asarray(df, dtype=float).reshape(())
    t: Array = jnp.asarray(t, dtype=float)
    return _stdtr_impl(df, t)


def _stdtr_fwd(df: Scalar, t: Array) -> tuple[Array, tuple[Scalar, Array]]:
    df = jnp.asarray(df, dtype=float).reshape(())
    t = jnp.asarray(t, dtype=float)
    y = _stdtr_impl(df, t)
    return y, (df, t)


def _stdtr_bwd(res: tuple[Scalar, Array], g: Array) -> tuple[Scalar, Array]:
    df, t = res
    pdf_t = _stdtr_pdf_t(df, t)
    d_df = jnp.zeros_like(df)
    d_t = g * pdf_t
    return d_df, d_t


stdtr.defvjp(_stdtr_fwd, _stdtr_bwd)
