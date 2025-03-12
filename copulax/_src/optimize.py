import jax.numpy as jnp
import jax
from typing import Callable
import jaxopt.projection as proj
from functools import partial


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
    f_vg: Callable = jax.jit(jax.value_and_grad(f, argnums=0), **jit_options)
    # f_vg: Callable = jax.value_and_grad(f, argnums=0)

    def _iter(tup: tuple, it):
        x: jnp.ndarray = tup[0]  # current estimate
        m: jnp.ndarray = tup[2]  # first moment estimate
        v: jnp.ndarray = tup[3]  # second moment estimate
        t: jnp.ndarray = tup[4]  # loop iteration count

        # getting raw gradient
        f_val, f_grad = f_vg(x, **kwargs)

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
    val_opt, _ = f_vg(x_opt, **kwargs)
    return {'x': x_opt, 'val': val_opt}

