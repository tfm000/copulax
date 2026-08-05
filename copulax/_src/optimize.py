import jax.numpy as jnp
import jax
from typing import Callable
import optax.projections as proj
from functools import partial
from jax import Array

from copulax._src.typing import Scalar


###############################################################################
# ADAM optimizer
###############################################################################
@jax.jit
def adam(
    grad: jnp.ndarray,
    m: jnp.ndarray,
    v: jnp.ndarray,
    t: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Adam optimiser.

    Reference:
        Kingma, D. P. & Ba, J. (2014). Adam: A Method for Stochastic
        Optimization. https://arxiv.org/abs/1412.6980

    Args:
        grad: the gradient at the current iteration.
        m: the first moment estimate vector from the prior iteration.
        v: the second raw moment estimate vector from the prior iteration.
        t: the prior iteration count.
        beta1: exponential decay rate for the first-moment vector ``m``.
        beta2: exponential decay rate for the second-moment vector ``v``.
        eps: small constant added to the denominator to prevent division
            by zero. Defaults to ``1e-8``.

    Returns:
        Adam direction, the first moment estimate vector, the second moment
        estimate vector, the current iteration.
    """
    t += 1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    d = m_hat / (jnp.sqrt(v_hat) + eps)
    return d, m, v, t


@partial(jax.jit, static_argnames=("projection",))
def single_update(
    x: jnp.ndarray,
    d: jnp.ndarray,
    lr: float,
    projection: Callable,
    projection_options: dict,
) -> jnp.ndarray:
    """Update the weights using the projected gradient method.

    Args:
        x: the current parameters from the previous iteration.
        d: the descent direction in the unconstrained space.
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


def projected_gradient(
    f: Callable,
    x0: jnp.ndarray,
    projection_method: str,
    lr: float = 1.0,
    maxiter: int = 100,
    adam_options: dict = {},
    jit_options: dict = {},
    projection_options: dict = {},
    **kwargs,
) -> dict:
    """Projected gradient descent for linearly constrained optimisation.

    Minimises the objective function ``f`` using projected gradient descent
    with Adam gradient updates.

    Args:
        f: objective function to minimise. Must be ``jax.grad`` and
            ``jax.jit`` compatible and return a scalar value. The first
            argument must be the parameter vector to optimise.
        x0: initial guess. Must be a flattened array with the same size as
            the solution.
        projection_method: name of the projection function to use. All
            ``optax`` constrained-optimisation projection functions are
            supported.
        lr: learning rate used in projected gradient descent.
        maxiter: maximum number of iterations.
        adam_options: dictionary of options for the Adam optimiser.
        jit_options: kwargs to pass to ``jax.jit`` when compiling ``f``.
        projection_options: kwargs to pass to the specified projection
            function.
        kwargs: additional arguments forwarded to the objective function.

    Returns:
        Dictionary of optimisation results with keys:

        ``x``
            The BEST iterate encountered over the whole scan (the
            argmin of the minimised objective), not the last iterate.
            First-order Adam-projected steps are not monotone, so the
            last iterate can be worse than an earlier one; returning the
            best point observed makes the result honest. If the objective
            is non-finite at every evaluated point (a degenerate fit) this
            surfaces a non-finite value in ``val`` rather than a
            finite-but-wrong stalled point.
        ``val``
            The objective evaluated at ``x`` (i.e. at the best iterate).
            Identical to ``best_val``. Preserved for backward compatibility
            with callers that read only ``{"x", "val"}``.
        ``best_val``
            The best (minimum) objective value observed over the scan.
            Equal to ``val`` by construction.
        ``nan_encountered``
            Boolean flag (as a JAX scalar) set ``True`` if the gradient
            was non-finite at any iterate. On such a step the carry is
            FROZEN (the iterate is held) rather than silently stepping on
            a zeroed gradient, so a bad parameter region surfaces loudly
            downstream instead of being masked.
    """
    # JIT compiling the projection and gradient functions
    projection: Callable = getattr(proj, projection_method)
    projection = jax.jit(projection)
    f_vg: Callable = jax.jit(jax.value_and_grad(f, argnums=0), **jit_options)

    def _iter(carry: tuple, it):
        x: jnp.ndarray = carry[0]  # current estimate
        best_x: jnp.ndarray = carry[1]  # best iterate so far
        best_val: jnp.ndarray = carry[2]  # objective at best_x
        m: jnp.ndarray = carry[3]  # first moment estimate
        v: jnp.ndarray = carry[4]  # second moment estimate
        t: jnp.ndarray = carry[5]  # loop iteration count

        # getting value and gradient in a single forward+backward pass
        f_val, f_grad = f_vg(x, **kwargs)

        # HARD-10 (D-11): detect a non-finite gradient and FREEZE the carry
        # on it, rather than nan_to_num-zeroing the gradient and silently
        # stepping on. Zeroing masked bad parameter regions in downstream
        # fitters; freezing holds the iterate and records the flag so a
        # degenerate fit surfaces NaN (the honest signal) instead of a
        # finite-but-wrong stalled point.
        nan_grad = jnp.any(~jnp.isfinite(f_grad))
        f_grad = jnp.where(nan_grad, 0.0, f_grad)  # frozen step (see x_new)

        # performing Adam step
        d, m, v, t = adam(grad=f_grad, m=m, v=v, t=t, **adam_options)

        # performing projected gradient step; on a non-finite gradient hold
        # the current iterate (freeze-carry) instead of stepping.
        step = single_update(
            x=x,
            d=d,
            lr=lr,
            projection=projection,
            projection_options=projection_options,
        )
        x_new = jnp.where(nan_grad, x, step)

        # HARD-04: track the BEST iterate seen (objective is minimised, so
        # smaller is better). best_val is seeded with the objective at x0,
        # so the returned point is at least as good as the start (dossier
        # section 6: "including the evaluation at the start point itself").
        # NaN comparisons are always False, so a non-finite f_val never
        # displaces a finite incumbent, and an all-degenerate fit keeps the
        # NaN seeded from x0 -> val surfaces NaN.
        improved = f_val < best_val
        best_x = jnp.where(improved, x, best_x)
        best_val = jnp.where(improved, f_val, best_val)

        return (x_new, best_x, best_val, m, v, t), nan_grad

    # initialise the optimization loop
    m0: jnp.ndarray = jnp.zeros_like(x0)
    v0: jnp.ndarray = jnp.zeros_like(x0)
    t: int = 0
    # Seed the best-iterate tracker with the start point and its objective
    # so the returned optimum incorporates x0 itself (and so a degenerate
    # start propagates its NaN objective into best_val).
    best_val0: jnp.ndarray = f_vg(x0, **kwargs)[0]
    init = (x0, x0, best_val0, m0, v0, t)

    # running projected gradient descent loop
    (_, best_x, best_val, _, _, _), nan_flags = jax.lax.scan(
        _iter, init, None, length=maxiter
    )

    # HARD-04/HARD-10: return the BEST iterate (not the last) plus the best
    # objective and the non-finite-gradient flag. ``val`` mirrors ``best_val``
    # (the objective at ``best_x``) to preserve the existing {"x", "val"}
    # contract for callers that ignore the new keys.
    nan_encountered = jnp.any(nan_flags)
    return {
        "x": best_x,
        "val": best_val,
        "best_val": best_val,
        "nan_encountered": nan_encountered,
    }


###############################################################################
# Brent's method (Brent 1973, Algorithm 4.1)
###############################################################################
_DENOM_EPS = 1e-30


def _safe_div(num: Scalar, denom: Scalar) -> Scalar:
    """Division guarded against zero denominator."""
    safe_denom = jnp.where(jnp.abs(denom) < _DENOM_EPS, _DENOM_EPS, denom)
    return num / safe_denom


def _brent_classical(
    g: Callable, bounds: jnp.ndarray, maxiter: int = 20, tol: float = 1e-12, **kwargs
) -> Scalar:
    r"""Classical Brent's root-finding algorithm.

    Adaptively selects between inverse quadratic interpolation, secant,
    and bisection with acceptance criteria that guarantee convergence.

    Exactly ``maxiter + 2`` function evaluations are performed
    (2 for the initial bracket, 1 per scan iteration).

    Args:
        g: Scalar-valued function whose root is sought.
        bounds: Two-element array ``[a, b]`` bracketing the root.
        maxiter: Fixed number of iterations (for ``jax.lax.scan``).
        tol: Absolute convergence tolerance.
        kwargs: Extra keyword arguments forwarded to *g*.

    Returns:
        Best root estimate (the bracket endpoint with smallest ``|g|``).

    Reference:
        Brent, R.P. (1973). *Algorithms for Minimization without
        Derivatives*, Chapter 4.
    """
    a, b = bounds
    fa = g(a, **kwargs)
    fb = g(b, **kwargs)

    # Ensure |f(b)| <= |f(a)| so b is the best guess.
    swap = jnp.abs(fa) < jnp.abs(fb)
    a, b = jnp.where(swap, b, a), jnp.where(swap, a, b)
    fa, fb = jnp.where(swap, fb, fa), jnp.where(swap, fa, fb)

    c, fc = a, fa
    d = b - a
    mflag = jnp.array(1.0)  # 1.0 = last step was bisection

    init = (a, b, fa, fb, c, fc, d, mflag)

    def _step(carry, _):
        a_, b_, fa_, fb_, c_, fc_, d_, mflag_ = carry

        # --- interpolation attempt ---
        # IQI when three distinct function values; secant otherwise.
        use_iqi = (fa_ != fc_) & (fb_ != fc_)

        # Secant step: s = b - fb*(b-a)/(fb-fa)
        s_sec = b_ - _safe_div(fb_ * (b_ - a_), fb_ - fa_)

        # Inverse quadratic interpolation
        d1 = (fa_ - fb_) * (fa_ - fc_)
        d2 = (fb_ - fa_) * (fb_ - fc_)
        d3 = (fc_ - fa_) * (fc_ - fb_)
        s_iqi = (
            _safe_div(a_ * fb_ * fc_, d1)
            + _safe_div(b_ * fa_ * fc_, d2)
            + _safe_div(c_ * fa_ * fb_, d3)
        )

        s_interp = jnp.where(use_iqi, s_iqi, s_sec)

        # --- bisection fallback ---
        s_bisect = 0.5 * (a_ + b_)

        # --- Brent's acceptance criteria ---
        # s must lie strictly between (3a+b)/4 and b.
        lo = jnp.minimum(0.75 * a_ + 0.25 * b_, b_)
        hi = jnp.maximum(0.75 * a_ + 0.25 * b_, b_)
        cond1 = (s_interp <= lo) | (s_interp >= hi)

        # Step-size conditions (depend on mflag).
        abs_sb = jnp.abs(s_interp - b_)
        abs_bc = jnp.abs(b_ - c_)
        abs_cd = jnp.abs(c_ - d_)

        cond2 = (mflag_ > 0.5) & (abs_sb >= 0.5 * abs_bc)
        cond3 = (mflag_ <= 0.5) & (abs_sb >= 0.5 * abs_cd)
        cond4 = (mflag_ > 0.5) & (abs_bc < tol)
        cond5 = (mflag_ <= 0.5) & (abs_cd < tol)

        use_bisection = cond1 | cond2 | cond3 | cond4 | cond5

        s = jnp.where(use_bisection, s_bisect, s_interp)
        new_mflag = jnp.where(use_bisection, 1.0, 0.0)

        # --- single function evaluation ---
        fs = g(s, **kwargs)

        # --- update bracket ---
        new_d = c_
        new_c = b_
        new_fc = fb_

        # If fa*fs < 0 the root lies between a and s, so b ← s.
        root_left = fa_ * fs < 0
        new_a = jnp.where(root_left, a_, s)
        new_fa = jnp.where(root_left, fa_, fs)
        new_b = jnp.where(root_left, s, b_)
        new_fb = jnp.where(root_left, fs, fb_)

        # Swap to maintain |f(b)| <= |f(a)|.
        need_swap = jnp.abs(new_fa) < jnp.abs(new_fb)
        fin_a = jnp.where(need_swap, new_b, new_a)
        fin_fa = jnp.where(need_swap, new_fb, new_fa)
        fin_b = jnp.where(need_swap, new_a, new_b)
        fin_fb = jnp.where(need_swap, new_fa, new_fb)

        return (fin_a, fin_b, fin_fa, fin_fb, new_c, new_fc, new_d, new_mflag), None

    final, _ = jax.lax.scan(_step, init, None, length=maxiter)
    return final[1]


def brent(
    g: Callable,
    bounds: jnp.ndarray,
    maxiter: int = 20,
    tol: float = 1e-12,
    **kwargs,
) -> Scalar:
    r"""Find a root of *g* in the interval *bounds* using Brent's method.

    Combines inverse quadratic interpolation, secant, and bisection
    with acceptance criteria that guarantee convergence.  Gradients
    w.r.t. ``**kwargs`` are computed via the implicit function theorem,
    so this function is safe to use inside ``jax.grad``.

    Args:
        g: Scalar-valued function.  Signature ``g(x, **kwargs) -> scalar``.
        bounds: Two-element array ``[a, b]`` bracketing a root of *g*.
        maxiter: Number of Brent iterations (fixed, for ``jax.lax.scan``).
        tol: Absolute convergence tolerance.
        kwargs: Extra keyword arguments forwarded to *g*.

    Returns:
        Scalar root estimate.

    Reference:
        Brent, R.P. (1973). *Algorithms for Minimization without
        Derivatives*, Chapter 4.  Prentice-Hall.
    """
    bounds = jnp.asarray(bounds, dtype=float).flatten()
    bounds = jnp.sort(bounds)

    # Forward solve (no gradients through the iterative loop).
    x_star = jax.lax.stop_gradient(_brent_classical(g, bounds, maxiter, tol, **kwargs))

    # Implicit differentiation via IFT:
    #   x_out = x* - g(x*,θ) / stop_gradient(∂g/∂x)
    # Forward value is exact (g(x*)≈0), gradient is the IFT result.
    #
    # ∂g/∂x is estimated via central finite differences rather than AD,
    # so g need not be differentiable w.r.t. x (e.g. betainc-based CDFs).
    _FD_H = 1e-8
    dg_dx = (g(x_star + _FD_H, **kwargs) - g(x_star - _FD_H, **kwargs)) / (2 * _FD_H)
    g_val = g(x_star, **kwargs)
    correction = _safe_div(g_val, jax.lax.stop_gradient(dg_dx))
    # When the root hasn't converged (g_val far from 0), the correction
    # can overflow.  Clamp to zero so the forward value falls back to
    # x_star (the best Brent found).  The IFT is only valid when
    # g(x*) ≈ 0, so a large correction signals non-convergence.
    bracket_width = jnp.abs(bounds[1] - bounds[0])
    correction = jnp.where(jnp.abs(correction) > bracket_width, 0.0, correction)
    # HARD-10 D-11 (Open Question 3) review verdict: KEEP-WITH-JUSTIFICATION.
    # This nan_to_num is NOT the silent gradient-zeroing that was removed from
    # projected_gradient. It clamps the IFT *correction* only, and a NaN/inf
    # correction here (0/0 at an exactly-converged root, or a residual overflow
    # the bracket_width clamp above cannot express) means the implicit-function
    # premise g(x*)~=0 does not hold -- so falling back to correction=0 returns
    # x_star, the best root Brent found (stop_gradient-wrapped, and covered by
    # the no-sign-change/best-guess tests). That is the mathematically correct
    # fallback to a known-good value, not a masked failure; the denominator is
    # additionally guarded by _safe_div. Freeze-carry does not apply.
    correction = jnp.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
    return x_star - correction
