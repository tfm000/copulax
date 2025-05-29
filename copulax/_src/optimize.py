import jax.numpy as jnp
import jax
from typing import Callable
import jaxopt.projection as proj
from functools import partial
from jax._src.typing import Array

from copulax._src.typing import Scalar


###############################################################################
# ADAM optimizer
###############################################################################
@jax.jit
def adam(grad: jnp.ndarray, m: jnp.ndarray, v: jnp.ndarray, t: int, 
         beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
         ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Adam optimiser.

    .. math::
        https://arxiv.org/abs/1412.6980

    Args:
    grad: the gradient at the current iteration.
    m: the first moment estimate vector from the prior iteration.
    v: the second raw moment estimate vector from the prior iteration.
    t: the prior iteration count.
    beta1: the beta1 parameter controling exponential decay rate for the m vector.
    beta2: the beta2 parameter controling exponential decay rate for the v vector.
    eps: the epsilon parameter to prevent division by zero. Defaults to 1e-8.

    Returns:
        Adam direction, the first moment estimate vector, the second moment 
        estimate vector, the current iteration.
    """
    t += 1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**(t + 1))
    v_hat = v / (1 - beta2**(t + 1))
    d = m_hat / (jnp.sqrt(v_hat) + eps)
    return d, m, v, t


@partial(jax.jit, static_argnames=("projection",))
def single_update(x: jnp.ndarray, d: jnp.ndarray, lr: float,
                  projection: Callable, projection_options: dict,) -> jnp.ndarray:
    """Update the weights using the projected gradient method.

    Args:
    x: the current parameters from the previous iteration.
    d: the direction to move in in the non-constrained space.
    lr: learning rate used in projected gradient descent.
    projection: the projection function to use.
    projection_options: dictionary of options for the projection function.

    Returns:
        The updated weights.
    """
    # Calculate the new weights
    x_uc: jnp.ndarray = x - lr * d

    # Project the new weights onto the feasible set
    x_uc = x_uc[None].T
    x_proj: jnp.ndarray = projection(x_uc, **projection_options)
    return x_proj.flatten()


def projected_gradient(f: Callable, x0: jnp.ndarray, projection_method: str,
                       lr: float = 1.0, maxiter: int = 100, 
                       adam_options: dict = {}, jit_options: dict = {}, 
                       projection_options: dict = {}, **kwargs
                       ) -> dict:
    """Projected gradient descent for linearly constrained optimization.
    Ninimizes the objective function f using the projected gradient descent algorithm and Adam gradient updates.

    Args:
        f: objective function to minimize. Must be jax.grad and jax.jit compatible and return a scalar value. The first argument must be the parameter vector to optimize.
        x0: initial guess. Must be a flatterned array with the same size as the solution.
        projection_method: name of the projection function to use. All jaxopt constrained optimisation projection functions are supported.
        lr: learning rate used in projected gradient descent.
        maxiter: maximum number of iterations.
        adam_options: dictionary of options for the Adam optimizer. 
        jit_options: kwargs to pass to jax.jit when compiling f.
        projection_options: kwargs to pass to the specified projection function.
        kwargs: additional arguments to pass to the objective function.

    Returns:
        dictionary containing optimal results.
    """
    # JIT compiling the projection and gradient functions
    projection: Callable = getattr(proj, projection_method)
    projection = jax.jit(projection)
    # f_vg: Callable = jax.jit(jax.value_and_grad(f, argnums=0), **jit_options)
    # f_vg: Callable = jax.value_and_grad(f, argnums=0)
    f = jax.jit(f, **jit_options)
    grad: Callable = jax.jit(jax.grad(f, argnums=0), **jit_options)

    def _iter(tup: tuple, it):
        x: jnp.ndarray = tup[0]  # current estimate
        m: jnp.ndarray = tup[2]  # first moment estimate
        v: jnp.ndarray = tup[3]  # second moment estimate
        t: jnp.ndarray = tup[4]  # loop iteration count

        # getting raw gradient
        f_val: jnp.ndarray = f(x, **kwargs)
        f_grad: jnp.ndarray = grad(x, **kwargs)
        # f_val, f_grad = f_vg(x, **kwargs)
        f_grad = jnp.nan_to_num(f_grad)  # replace NaNs with 0s

        # performing Adam step
        d, m, v, t = adam(grad=f_grad, m=m, v=v, t=t, **adam_options)

        # performing projected gradient step
        x = single_update(x=x, d=d, lr=lr, projection=projection, projection_options=projection_options)
        # if jnp.isnan(f_val) or jnp.isnan(f_grad).any() or jnp.isnan(d).any() or jnp.isnan(x).any():
        #     raise ValueError("NaNs detected in the optimization loop.")
        return (x, f_val, m, v, t), it
    

    # initialise the optimization loop
    m0: jnp.ndarray = jnp.zeros_like(x0)
    v0: jnp.ndarray = jnp.zeros_like(x0)
    t: int = 0
    init = x0, jnp.inf, m0, v0, t

    # running projected gradient descent loop
    res, _ = jax.lax.scan(_iter, init, None, length=maxiter)
    # res = init
    # for i in range(maxiter):
    #     res, _ = _iter(res, i)

    # getting optimal values
    x_opt = res[0]
    # val_opt, _ = f_vg(x_opt, **kwargs)
    val_opt = f(x_opt, **kwargs)
    return {'x': x_opt, 'val': val_opt}


###############################################################################
# Brent's method
###############################################################################
@jax.jit
def _brent_new_bounds(a: Scalar, b: Scalar, c: Scalar, ga: Scalar, gb: Scalar, gc: Scalar) -> Array:
    # looking for same sign
    index = jnp.argmax(jnp.array([ga == 0, gb == 0, gc == 0, ga * gc > 0, ga * gc < 0], dtype=int))
    return jax.lax.switch(index, [
        lambda: jnp.array([a, a]),  # ga == 0
        lambda: jnp.array([b, b]),  # gb == 0
        lambda: jnp.array([c, c]),  # gc == 0
        lambda: jnp.array([c, b]),  # ga * gc > 0
        lambda: jnp.array([a, c])])


def _brentb(g: Callable, bounds: jnp.ndarray, **kwargs) -> Scalar:
    r"""Brent's bisection method for root finding.

    Args:
        g: function to find the root of.
        bounds: lower and upper bounds.
        maxiter: maximum number of iterations.

    Returns:
        dictionary containing optimal results.
    """
    a, b = bounds
    c: Scalar = 0.5 * (a + b)

    # evaluating the function at each point
    ga: Scalar = g(a, **kwargs)
    gb: Scalar = g(b, **kwargs)
    gc: Scalar = g(c, **kwargs)

    # returning new bounds
    return _brent_new_bounds(a=a, b=b, c=c, ga=ga, gb=gb, gc=gc)


@jax.jit
def _secant_root(a: Scalar, b: Scalar, ga: Scalar, gb: Scalar) -> Scalar:
    return b - gb * (b - a) / (gb - ga)


def _brents(g: Callable, bounds: jnp.ndarray, **kwargs) -> Scalar:
    r"""Brent's secant method for root finding.

    Args:
        g: function to find the root of.
        bounds: lower and upper bounds.
        maxiter: maximum number of iterations.

    Returns:
        dictionary containing optimal results.
    """
    a, b = bounds

    # evaluating the function at the bounds
    ga: Scalar = g(a, **kwargs)
    gb: Scalar = g(b, **kwargs)

    # deriving the root of the line
    c: Scalar = _secant_root(a=a, b=b, ga=ga, gb=gb)
    gc: Scalar = g(c, **kwargs)

    # returning new bounds
    return _brent_new_bounds(a=a, b=b, c=c, ga=ga, gb=gb, gc=gc)


def _brentq(g: Callable, bounds: jnp.ndarray, **kwargs) -> Scalar:
    r"""Brent's inverse quadratic method for root finding.

    Args:
        g: function to find the root of.
        bounds: lower and upper bounds.
        maxiter: maximum number of iterations.

    Returns:
        dictionary containing optimal results.
    """
    a, b = bounds

    # evaluating the function at the bounds
    ga: Scalar = g(a, **kwargs)
    gb: Scalar = g(b, **kwargs)

    # secant root
    c: Scalar = _secant_root(a=a, b=b, ga=ga, gb=gb)
    gc: Scalar = g(c, **kwargs)

    # finding root of the quadratic
    xq: Scalar = (
        (a * gb * gc) / ((ga - gb) * (ga - gc))
        + (b * ga * gc) / ((gb - ga) * (gb - gc))
        + (c * ga * gb) / ((gc - ga) * (gc - gb))
        )
    gxq: Scalar = g(xq, **kwargs)

    # returning new bounds
    # calc widths between all
    # find points with opposite signs
    # pick the two with smallest width
    combinations = [[a, c, ga, gc], [a, xq, ga, gxq], [b, c, gb, gc], 
                   [b, xq, gb, gxq], [c, xq, gc, gxq]]
    min_width = jnp.abs(b - a)
    for comb in combinations:
        i, j, gi, gj = comb
        width = jnp.abs(j - i)
        
        smaller_interval = jnp.logical_and(width < min_width, gi * gj < 0)
        index = jnp.argmax(jnp.array([gi == 0, gj ==0, smaller_interval, jnp.logical_not(smaller_interval)], dtype=int))
        min_width, bounds = jax.lax.switch(index, [
            lambda: (jnp.array(0.0), jnp.array([i, i])),  # gi == 0
            lambda: (jnp.array(0.0), jnp.array([j, j])),  # gj == 0
            lambda: (width, jnp.array([i, j])),  # smaller interval
            lambda: (min_width, bounds),  # default interval
        ])

    return jnp.sort(bounds)


def _brentsb(g: Callable, bounds: jnp.ndarray, **kwargs) -> Scalar:
    # bisection method
    bisection_bounds = _brentb(g=g, bounds=bounds, **kwargs)

    # secant method
    secant_bounds = _brents(g=g, bounds=bisection_bounds, **kwargs)
    return secant_bounds


def _brentqb(g: Callable, bounds: jnp.ndarray, **kwargs) -> Scalar:
    # bisection method
    bisection_bounds = _brentb(g=g, bounds=bounds, **kwargs)

    # quadratic method 
    quadratic_bounds = _brentq(g=g, bounds=bisection_bounds, **kwargs)
    return quadratic_bounds


def brent(g: Callable, bounds: jnp.ndarray, method: str = 'quadratic-bisection', 
          maxiter: int = 50, **kwargs) -> Scalar:
    r"""Brent's method for root finding.

    Args:
        g: function to find the root of.
        bounds: lower and upper bounds.
        method: method to use for root finding. Can be 'bisection',
            'secant', 'quadratic', 'secant-bisection', 'quadratic-bisection'.
        maxiter: maximum number of iterations.
        kwargs: additional arguments to pass to g.

    Returns:
        dictionary containing optimal results.
    """
    # getting method
    method = method.lower()
    if method == 'bisection':
        brent_method = _brentb
    elif method == 'secant':
        brent_method = _brents
    elif method == 'quadratic':
        brent_method = _brentq
    elif method == 'secant-bisection':
        brent_method = _brentsb
    elif method == 'quadratic-bisection':
        brent_method = _brentqb
    else:
        raise ValueError(f"Unknown method: {method}")

    # standardizing the bounds
    bounds: Array = jnp.asarray(bounds).flatten()
    bounds = jnp.sort(bounds)

    # iterating to find the root
    scan_func: Callable = lambda bounds_, _: (brent_method(g=g, bounds=bounds_, **kwargs), None)
    bounds, _ = jax.lax.scan(scan_func, bounds, None, length=maxiter)
    return bounds.mean()






