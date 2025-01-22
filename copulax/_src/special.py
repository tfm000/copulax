from jax import lax
import jax.numpy as jnp
from jax._src.typing import ArrayLike, Array
from quadax import quadgk
from jax.scipy import special
from typing import Callable
import jax

# from copulax.univariate import student_t, normal, lognormal
# from copulax._src.univariate._utils import DEFAULT_RANDOM_KEY

# sample_dist, SCALE = normal, 2.0


# sample_dist, SCALE = student_t, 2.0
# sample_dist, SCALE = lognormal, 1.0


# def _kv_integrand(t: Array, v: float, x: Array) -> Array:
#     cosh_t = lax.cosh(t)
#     cosh_tv = lax.cosh(lax.mul(v, t))
#     inner = -x * cosh_t
#     exp = lax.exp(inner)

#     integrand = exp * cosh_tv

#     integrand = jnp.where(jnp.isinf(cosh_t), 0.0, integrand)
#     integrand = jnp.where(jnp.isinf(cosh_tv), 0.0, integrand)
#     integrand = jnp.where(jnp.isinf(exp), 0.0, integrand)
#     # integrand = jnp.where(jnp.isinf(inner), 0.0, integrand)
#     return integrand


# def _kv_integrand(t: Array, v: float, x: Array) -> Array:
#     cosh_t = lax.cosh(t)
#     cosh_t = jnp.where(jnp.isinf(cosh_t), 0.0, cosh_t)

#     cosh_tv = lax.cosh(lax.mul(v, t))
#     cosh_tv = jnp.where(jnp.isinf(cosh_tv), 0.0, cosh_tv)

#     inner = -x * cosh_t
#     exp = lax.exp(inner)
#     exp = jnp.where(jnp.isinf(exp), 0.0, exp)

#     integrand = exp * cosh_tv

#     # integrand = jnp.where(jnp.isinf(cosh_t), 0.0, integrand)
#     # integrand = jnp.where(jnp.isinf(cosh_tv), 0.0, integrand)
#     # integrand = jnp.where(jnp.isinf(exp), 0.0, integrand)
#     # integrand = jnp.where(jnp.isinf(inner), 0.0, integrand)
#     return integrand


def _kv_integrand(w: Array, v: float, x: Array) -> Array:
    STABILITY_TERM = 0 #1e-15
    frac = jnp.pow(w + STABILITY_TERM, -1)
    inner = -0.5*x * (w + frac)
    exp = lax.exp(inner)
    return 0.5 * lax.pow(w + STABILITY_TERM, v - 1.0) * exp


# def _kv_integrand(w: Array, v: float, x: Array) -> Array:
#     STABILITY_TERM = 0#1e-15
#     bottom = lax.pow(lax.pow(w, 2) + lax.pow(x, 2), v + 0.5)
#     vals = lax.cos(w) / bottom
#     vals = jnp.where(jnp.isinf(w), 0.0, vals)
#     vals = jnp.where(jnp.isinf(bottom), 0.0, vals)
#     return vals

# def _kv_integrand(w: Array, v: float, x: Array) -> Array:


def _kv_single_x(v: float, xi: float) -> float:
    # from jax import config
    # config.update("jax_debug_nans", True)
    # config.update("jax_disable_jit", True)

    kv_val, info = quadgk(_kv_integrand, interval=[0.0, jnp.inf], args=(v, xi))
    return kv_val.reshape(())


def kv(v: float, x: ArrayLike) -> Array:
    r"""Modified Bessel function of the second/third kind.

    Args:
        v: float, order of the Bessel function.
        x: arraylike, value(s) at which to evaluate the function.

    Returns:
        array of values.
    """
    v = jnp.asarray(jnp.abs(v), dtype=float)
    x = jnp.asarray(x, dtype=float)
    xshape = x.shape
    x = x.flatten()

    def _iter(carry, xi):
        kv_i = _kv_single_x(v, xi)
        return carry, kv_i
    
    _, kv_raw = lax.scan(_iter, None, x)

    kv_adj = jnp.where(x < 0, jnp.nan, kv_raw)
    return kv_adj.reshape(xshape)


def kv_x_to_0(v: float, x: ArrayLike) -> Array:
    r"""Modified Bessel function of the second/third kind.

    Args:
        v: float, order of the Bessel function.
        x: arraylike, value(s) at which to evaluate the function.

    Returns:
        array of values.
    """
    return special.gamma(v) * jnp.power(2, v - 1) * jnp.power(x, -v)


def kv_x_to_inf(v: float, x: ArrayLike) -> Array:
    return 0.0



# def kv_asymptotic_(v: float, x: ArrayLike) -> Array:
#     r"""Modified Bessel function of the second/third kind.

#     Incorporates limit approximations for x -> 0 and x -> inf. 
#     This allows for a more stable evaluation of gradients at the expense of 
#     precision loss. If the value of the modified Bessel function of the second 
#     kind is important, use kv. If gradient is important use kv_asymptotic.

#     Args:
#         v: float, order of the Bessel function.
#         x: arraylike, value(s) at which to evaluate the function.

#     Returns:
#         array of values.
#     """
#     v = jnp.asarray(jnp.abs(v), dtype=float)
#     x = jnp.asarray(x, dtype=float)
    
#     # preventing nans resulting from undesired gradient propagation via conditioning
#     raw_val = jax.lax.stop_gradient(kv(v, x))

#     # hacky way of implementing if, elif, else statements
#     index = 0
#     index = lax.cond(raw_val > 10.0, lambda _: 1, lambda _: index, None)
#     index = lax.cond(raw_val < 1e-10, lambda _: 2, lambda _: index, None)

#     return lax.switch(index, (kv, kv_x_to_0, kv_x_to_inf), v, x)


@jax.custom_gradient
def kv_asymptotic_single(v: float, xi: float) -> float:
    v = jnp.asarray(jnp.abs(v), dtype=float)
    xi = jnp.asarray(xi, dtype=float)
    
    # preventing nans resulting from undesired gradient propagation via conditioning
    kv_val = jax.lax.stop_gradient(kv(v, xi))

    # hacky way of implementing if, elif, else statements
    index = 0
    index = lax.cond(kv_val > 10.0, lambda _: 1, lambda _: index, None)
    index = lax.cond(kv_val < 1e-10, lambda _: 2, lambda _: index, None)

    # calculating the gradient
    kv_grad_func: Callable = (lambda v_, x_:  lax.switch(index, (_kv_single_x, kv_x_to_0, kv_x_to_inf), v_, x_))
    kv_grad: jnp.ndarray = jax.grad(kv_grad_func, argnums=(0, 1))(v, xi)
    return kv_val, lambda g: (g * kv_grad[0], g * kv_grad[1])


def kv_asymptotic(v: float, x: ArrayLike) -> Array:
    r"""Modified Bessel function of the second/third kind.

    Allows for a more stable evaluation of gradients at the expense of 
    precision loss by using asymptotic forms of the Bessel function when 
    calculating gradients. The kv value itself is not effected.

    Args:
        v: float, order of the Bessel function.
        x: arraylike, value(s) at which to evaluate the function.

    Returns:
        array of values.
    """
    v = jnp.asarray(jnp.abs(v), dtype=float)
    x = jnp.asarray(x, dtype=float)
    xshape = x.shape
    x = x.flatten()

    def _iter(carry, xi):
        kv_i = kv_asymptotic_single(v, xi)
        return carry, kv_i
    
    _, kv_vals = lax.scan(_iter, None, x)
    return kv_vals.reshape(xshape)

# maybe use personalised grad func 

    # return lax.cond(x <= 1e-5, kv_x_to_0, kv_, v, x)

    # return kv_(v, x)


# def _kv_integrand(v: float, x: Array, t: Array) -> Array:
#     cosh_t = lax.cosh(t)
#     cosh_tv = lax.cosh(lax.mul(v, t))
#     inner = -x * cosh_t
#     exp = lax.exp(inner)

#     integrand = exp * cosh_tv

#     integrand = jnp.where(jnp.isinf(cosh_t), 0.0, integrand)
#     integrand = jnp.where(jnp.isinf(cosh_tv), 0.0, integrand)
#     integrand = jnp.where(jnp.isinf(exp), 0.0, integrand)
#     # integrand = jnp.where(jnp.isinf(inner), 0.0, integrand)
#     return integrand


# def kv(v: float, x: ArrayLike, key=DEFAULT_RANDOM_KEY, num_points: int = 100) -> Array:
#     r"""Modified Bessel function of the second/third kind.

#     Approximates the indefinate integral using monte-carlo integration.

#     Args:
#         v: float, order of the Bessel function.
#         x: arraylike, value(s) at which to evaluate the function.
#         key: random.PRNGKey, random key for generating samples to use in 
#         monte-carlo integration.
#         num_points: int, number of random points to sample in monte-carlo 
#         integration.

#     Returns:
#         array of values.
#     """
#     eps = 1e-6
#     v = jnp.asarray(jnp.abs(v), dtype=float)
#     x = jnp.asarray(x, dtype=float)
#     xshape = x.shape
#     x = x.reshape((x.size, 1))
#     num_points: int = int(num_points)

#     t: Array = jnp.abs(sample_dist.rvs(key=key, shape=(num_points,)))
#     t = jnp.where(t < eps, eps, t)
#     samples = _kv_integrand(v, x, t)

#     pdf_vals = sample_dist.pdf(t).flatten() * SCALE
#     pdf_vals = jnp.where(pdf_vals < eps, eps, pdf_vals)
#     values = samples / pdf_vals
#     values = values.mean(axis=1).reshape(xshape)

#     values = jnp.where(x < 0, jnp.nan, values)
#     return values.reshape(xshape)





# def _kv_integrand(v: float, x: Array, t: Array) -> Array:
#     # top = lax.cos(t)
#     top = jnp.cos(t)
#     # top = 1.0
#     bottom = lax.pow(lax.pow(t, 2) + lax.pow(x, 2), v + 0.5)
#     return top / bottom


# def kv(v: float, x: ArrayLike, key=random.PRNGKey(0), num_points: int = 250) -> Array:
#     r"""Modified Bessel function of the second/third kind.

#     Approximates the indefinate integral using monte-carlo integration.

#     Args:
#         v: float, order of the Bessel function.
#         x: arraylike, value(s) at which to evaluate the function.
#         key: random.PRNGKey, random key for generating samples to use in 
#         monte-carlo integration.
#         num_points: int, number of random points to sample in monte-carlo 
#         integration.

#     Returns:
#         array of values.
#     """
#     eps = 1e-6
#     v = jnp.asarray(jnp.abs(v), dtype=float)
#     x = jnp.asarray(x, dtype=float)
#     xshape = x.shape
#     x = x.reshape((x.size, 1))
#     num_points: int = int(num_points)

#     t: Array = jnp.abs(sample_dist.rvs(key=key, shape=(num_points,)))
#     t = jnp.where(t < eps, eps, t)
#     c = lax.exp(lax.lgamma(v + 0.5)) * lax.pow(2.0 * x, v) / lax.sqrt(jnp.pi)
#     samples = lax.mul(c, _kv_integrand(v, x, t))

#     pdf_vals = sample_dist.pdf(t).flatten() * scale
#     pdf_vals = jnp.where(pdf_vals < eps, eps, pdf_vals)
#     values = samples / pdf_vals
#     values = values.mean(axis=1)[:, None]

#     values = jnp.where(x < 0, jnp.nan, values)
#     # if jnp.isnan(values).any():
#     #     breakpoint()
#     return values.reshape(xshape)