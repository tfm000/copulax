"""Modified Bessel function of the second kind and Student's t CDF.

References:
    - Bessel function integral representation:
      https://dlmf.nist.gov/10.32#E10
    - Asymptotic forms:
      https://dlmf.nist.gov/10.30
"""
from jax import lax, vmap
import jax.numpy as jnp
from jax._src.typing import ArrayLike, Array
from quadax import quadgk
from jax.scipy import special
from typing import Callable
import jax

from copulax._src.typing import Scalar


def _kv_integrand(w: Array, v: float, x: Array) -> Array:
    r"""Integrand for the integral representation of $K_v(x)$.

    Uses the substitution $w = e^t$ in the standard integral
    $K_v(x) = \frac{1}{2}\int_0^\infty w^{v-1} \exp(-x(w + w^{-1})/2)\,dw$.
    """
    frac = jnp.pow(w, -1)
    inner = -0.5 * x * (w + frac)
    exp = lax.exp(inner)
    return 0.5 * lax.pow(w, v - 1.0) * exp


def _kv_single_x(v: float, xi: float) -> float:
    """Evaluate $K_v$ at a single scalar point via quadrature."""
    kv_val, info = quadgk(_kv_integrand, interval=(0.0, jnp.inf), args=(v, xi))
    return kv_val.reshape(())


def kv(v: float, x: ArrayLike) -> Array:
    r"""Modified Bessel function of the second kind, $K_v(x)$.

    Evaluated via numerical quadrature, vectorized with ``vmap``.

    Args:
        v: float, order of the Bessel function.
        x: arraylike, value(s) at which to evaluate the function.

    Returns:
        array of values.
    """
    v = jnp.asarray(jnp.abs(v), dtype=float)
    x = jnp.asarray(x, dtype=float)
    xshape = x.shape
    x_flat = x.flatten()

    kv_raw = vmap(lambda xi: _kv_single_x(v, xi))(x_flat)

    kv_adj = jnp.where(x_flat < 0, jnp.nan, kv_raw)
    return kv_adj.reshape(xshape)


def _kv_x_to_0(v: float, x: ArrayLike) -> Array:
    r"""Asymptotic form of $K_v(x)$ as $x \to 0$:
    $K_v(x) \approx \Gamma(v) \cdot 2^{v-1} \cdot x^{-v}$.
    """
    return special.gamma(v) * jnp.power(2, v - 1) * jnp.power(x, -v)


def _kv_x_to_inf(v: float, x: ArrayLike) -> Array:
    r"""Asymptotic form of $K_v(x)$ as $x \to \infty$: $K_v(x) \to 0$."""
    return 0.0


@jax.custom_gradient
def _kv_asymptotic_single(v: float, xi: float) -> float:
    """Single-element $K_v$ with asymptotic gradient switching.

    Uses the exact quadrature value for the forward pass but switches
    to asymptotic approximations for the backward pass when the value
    is very large ($>10$) or very small ($<10^{-10}$) to avoid NaN
    gradients.
    """
    v = jnp.asarray(jnp.abs(v), dtype=float)
    xi = jnp.asarray(xi, dtype=float)

    # forward: exact quadrature value (no gradient tracking)
    kv_val = jax.lax.stop_gradient(kv(v, xi))

    # select gradient branch: 0=exact, 1=x→0 asymptote, 2=x→∞ asymptote
    index = 0
    index = lax.cond(kv_val > 10.0, lambda _: 1, lambda _: index, None)
    index = lax.cond(kv_val < 1e-10, lambda _: 2, lambda _: index, None)

    kv_grad_func: Callable = lambda v_, x_: lax.switch(
        index, (_kv_single_x, _kv_x_to_0, _kv_x_to_inf), v_, x_
    )
    kv_grad: jnp.ndarray = jax.grad(kv_grad_func, argnums=(0, 1))(v, xi)
    return kv_val, lambda g: (g * kv_grad[0], g * kv_grad[1])


def kv_asymptotic(v: float, x: ArrayLike) -> Array:
    r"""Modified Bessel function of the second kind with stable gradients.

    Returns exact $K_v(x)$ values but uses asymptotic forms for
    gradient computation when $K_v(x) > 10$ or $K_v(x) < 10^{-10}$,
    preventing NaN gradients that arise from the quadrature at
    extreme values.

    Vectorized with ``vmap`` over all elements of *x*.

    Args:
        v: float, order of the Bessel function.
        x: arraylike, value(s) at which to evaluate the function.

    Returns:
        array of values.
    """
    v = jnp.asarray(jnp.abs(v), dtype=float)
    x = jnp.asarray(x, dtype=float)
    xshape = x.shape
    x_flat = x.flatten()

    kv_vals = vmap(lambda xi: _kv_asymptotic_single(v, xi))(x_flat)
    return kv_vals.reshape(xshape)


########################################################################
# stdtr implementation
########################################################################
@jax.jit
def stdtr(df: Scalar, t: Array) -> Array:
    """Compute the cdf of the standard Student's t-distribution.

    Note:
        Gradients are not implemented for the first argument, df,
        stemming from the jax.special.betainc implementation.

    Args:
        df (scalar): degrees of freedom.
        t (Array): values at which to evaluate the cdf.

    Returns:
        Array: cdf values of the standard Student's t-distribution.
    """
    # transforming args
    df: Scalar = jnp.asarray(df, dtype=float).reshape(())
    t: Array = jnp.asarray(t, dtype=float)

    # computing the cdf
    x_t = df / ((t**2) + df)
    tail = 0.5 * special.betainc(df * 0.5, 0.5, x_t)
    return jnp.where(t < 0, tail, 1.0 - tail)
