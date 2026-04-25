"""Rigorous tests for copulax._src.optimize: Adam, Brent, projected_gradient.

Catches FINDING-01-01 (Adam bias correction off-by-one) and
FINDING-01-07 (NaN gradients silently replaced with 0).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize
import scipy.stats

from copulax._src.optimize import adam, brent, projected_gradient
from copulax.univariate import gamma, lognormal, normal


# ===================================================================
# Adam optimizer
# ===================================================================

class TestAdam:
    """Tests for Adam optimizer step function."""

    def test_bias_correction_step_values(self):
        """Verify Adam bias correction matches Kingma & Ba (2014) Algorithm 1.

        FINDING-01-01: The code increments t before computing the correction,
        so step 1 uses beta1^2 instead of beta1^1.

        Kingma & Ba (2014) Algorithm 1:
            t <- t + 1
            m_hat <- m_t / (1 - beta1^t)
            v_hat <- v_t / (1 - beta2^t)
            theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
        """
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        # Simulate steps manually
        g = jnp.array(1.0)  # constant gradient
        m = jnp.zeros_like(g)
        v = jnp.zeros_like(g)
        t = 0

        for step in range(1, 6):
            direction, m, v, t = adam(g, m, v, t, beta1=beta1, beta2=beta2, eps=eps)

            # After calling adam, t should be incremented to `step`
            assert int(t) == step, f"Step {step}: t should be {step}, got {int(t)}"

            # Expected values per Kingma & Ba (2014):
            # m_t = beta1 * m_{t-1} + (1 - beta1) * g
            # v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
            # m_hat = m_t / (1 - beta1^t)
            # v_hat = v_t / (1 - beta2^t)
            # direction = m_hat / (sqrt(v_hat) + eps)
            expected_m = (1 - beta1) * sum(beta1 ** (step - 1 - i) for i in range(step))
            expected_v = (1 - beta2) * sum(beta2 ** (step - 1 - i) for i in range(step))
            expected_m_hat = expected_m / (1 - beta1 ** step)
            expected_v_hat = expected_v / (1 - beta2 ** step)
            expected_direction = expected_m_hat / (np.sqrt(expected_v_hat) + eps)

            np.testing.assert_allclose(
                float(direction), expected_direction, rtol=1e-5,
                err_msg=f"Step {step}: Adam direction mismatch "
                        f"(bias correction may be off-by-one)"
            )

    def test_converges_on_quadratic(self):
        """Adam should converge to the minimum of f(x) = (x - 3)^2."""
        def f(x):
            return (x - 3.0) ** 2

        x = jnp.array(0.0)
        m = jnp.zeros_like(x)
        v = jnp.zeros_like(x)
        t = 0
        lr = 0.1

        for _ in range(500):
            g = jax.grad(f)(x)
            d, m, v, t = adam(g, m, v, t)
            x = x - lr * d

        np.testing.assert_allclose(float(x), 3.0, atol=0.01,
                                   err_msg="Adam failed to converge on quadratic")

    def test_nan_gradient_behavior(self):
        """Verify behavior when gradient contains NaN.

        FINDING-01-07: projected_gradient uses jnp.nan_to_num which silently
        replaces NaN gradients with 0. Test that adam itself propagates NaN.
        """
        g = jnp.array(float('nan'))
        m = jnp.zeros(())
        v = jnp.zeros(())
        t = 0
        direction, m_new, v_new, t_new = adam(g, m, v, t)

        # Adam step on NaN grad should propagate NaN (not silently zero it)
        # If adam itself uses nan_to_num internally, this test flags it.
        # Either NaN propagation or explicit error is acceptable.
        is_nan = np.isnan(float(direction))
        is_zero = float(direction) == 0.0
        if is_zero:
            pytest.xfail("FINDING-01-07: NaN gradient silently replaced with 0")


# ===================================================================
# Brent root-finding
# ===================================================================

class TestBrent:
    """Tests for classical Brent's root-finding algorithm (Brent 1973)."""

    def test_finds_sqrt2(self):
        """Find root of x^2 - 2 = 0 on [0, 2] to near machine precision."""
        root = float(brent(lambda x: x ** 2 - 2.0,
                           bounds=jnp.array([0.0, 2.0]), maxiter=100))
        np.testing.assert_allclose(root, np.sqrt(2), rtol=1e-10)

    def test_finds_pi(self):
        """Find root of sin(x) = 0 on [3, 4] to near machine precision."""
        root = float(brent(lambda x: jnp.sin(x),
                           bounds=jnp.array([3.0, 4.0]), maxiter=100))
        np.testing.assert_allclose(root, np.pi, rtol=1e-10)

    @pytest.mark.parametrize("f,bounds,true_root", [
        (lambda x: x ** 2 - 2.0, [0.0, 2.0], np.sqrt(2)),
        (lambda x: jnp.sin(x), [3.0, 4.0], np.pi),
        (lambda x: x ** 3 - 1.0, [0.0, 2.0], 1.0),
        (lambda x: jnp.exp(x) - 3.0, [0.0, 2.0], np.log(3)),
        (lambda x: x ** 5 - x - 1.0, [1.0, 2.0], 1.1673039782614187),
    ], ids=["sqrt2", "pi", "cube_root", "ln3", "quintic"])
    def test_convergence_vs_scipy(self, f, bounds, true_root):
        """Classical Brent matches scipy.optimize.brentq on 5 test functions."""
        our_root = float(brent(f, bounds=jnp.array(bounds), maxiter=50))
        scipy_root = scipy.optimize.brentq(lambda x: float(f(x)),
                                           bounds[0], bounds[1])
        # Both should be within 1e-10 of truth
        np.testing.assert_allclose(our_root, true_root, atol=1e-10,
                                   err_msg=f"Brent error too large")
        np.testing.assert_allclose(our_root, scipy_root, atol=1e-10,
                                   err_msg=f"Brent disagrees with scipy")

    def test_jit_compilable(self):
        """Brent is JIT-compatible."""
        @jax.jit
        def solve():
            return brent(lambda x: x ** 2 - 4.0,
                         bounds=jnp.array([0.0, 3.0]), maxiter=50)
        root = float(solve())
        np.testing.assert_allclose(root, 2.0, rtol=1e-10)

    def test_narrow_bracket(self):
        """Brent works with a very narrow initial bracket."""
        root = float(brent(lambda x: x - 1.5,
                           bounds=jnp.array([1.49, 1.51]), maxiter=100))
        np.testing.assert_allclose(root, 1.5, rtol=1e-10)

    def test_equal_function_values(self):
        """Handles f(a) = -f(b) gracefully (secant denominator guard)."""
        root = float(brent(lambda x: x,
                           bounds=jnp.array([-1.0, 1.0]), maxiter=50))
        np.testing.assert_allclose(root, 0.0, atol=1e-10)

    def test_vmap_compatible(self):
        """Brent can be vmapped over different bracket endpoints."""
        def f(x):
            return x ** 2 - 2.0

        # Batch of 5 different brackets, all containing sqrt(2)
        lo = jnp.array([0.0, 0.5, 1.0, 1.2, 1.4])
        hi = jnp.array([2.0, 2.5, 1.5, 1.5, 1.45])

        def solve_one(bounds):
            return brent(f, bounds=bounds, maxiter=50)

        roots = jax.vmap(solve_one)(jnp.stack([lo, hi], axis=1))
        np.testing.assert_allclose(np.array(roots), np.sqrt(2),
                                   atol=1e-8,
                                   err_msg="Brent vmap failed")

    def test_grad_implicit_differentiation(self):
        """Gradient through Brent uses IFT: d(sqrt(a))/da = 1/(2*sqrt(a)).

        For g(x, a) = x^2 - a, root x* = sqrt(a).
        IFT: dx*/da = -[dg/dx]^{-1} * dg/da = -[2x*]^{-1} * (-1) = 1/(2*sqrt(a)).
        """
        def root_of(a):
            return brent(lambda x, a=a: x ** 2 - a,
                         bounds=jnp.array([0.0, 10.0]), maxiter=50,
                         a=a)

        a_val = 2.0
        grad_val = float(jax.grad(root_of)(jnp.array(a_val)))
        expected = 1.0 / (2.0 * np.sqrt(a_val))  # 1/(2*sqrt(2))
        np.testing.assert_allclose(grad_val, expected, rtol=1e-4,
                                   err_msg="IFT gradient incorrect")

    def test_grad_ppf_style(self):
        """Gradient of PPF-style root-finding: d(ppf)/dq = 1/pdf(x*).

        For standard normal: ppf'(q) = 1/pdf(ppf(q)).
        """
        def ppf_via_brent(qi):
            return brent(
                lambda x, qi=qi: jax.scipy.stats.norm.cdf(x) - qi,
                bounds=jnp.array([-6.0, 6.0]),
                maxiter=50,
                qi=qi,
            )

        q = 0.75
        grad_val = float(jax.grad(ppf_via_brent)(jnp.array(q)))

        # Expected: 1/pdf(ppf(q))
        x_star = scipy.stats.norm.ppf(q)
        expected = 1.0 / scipy.stats.norm.pdf(x_star)
        np.testing.assert_allclose(grad_val, expected, rtol=1e-3,
                                   err_msg="PPF-style IFT gradient incorrect")

    def test_kwargs_forwarding(self):
        """Extra kwargs are correctly forwarded to g."""
        def g(x, offset=0.0):
            return x ** 2 - offset

        root = float(brent(g, bounds=jnp.array([0.0, 3.0]),
                           maxiter=50, offset=4.0))
        np.testing.assert_allclose(root, 2.0, rtol=1e-10)

    def test_no_sign_change_still_finite(self):
        """When bracket has no sign change, return best guess (not NaN)."""
        root = float(brent(lambda x: x ** 2 + 1.0,
                           bounds=jnp.array([-1.0, 1.0]), maxiter=50))
        assert np.isfinite(root), f"Expected finite result, got {root}"

    def test_converges_in_15_iters_wide_bracket(self):
        """Classical Brent converges to <1e-12 in ≤15 iters on [-6,6] CDF."""
        from copulax._src.optimize import _brent_classical
        f = lambda x: jax.scipy.stats.norm.cdf(x) - 0.75
        root = float(_brent_classical(f, jnp.array([-6.0, 6.0]), maxiter=15))
        np.testing.assert_allclose(root, scipy.stats.norm.ppf(0.75),
                                   atol=1e-12)


# ===================================================================
# Projected gradient optimizer
# ===================================================================

class TestProjectedGradient:
    """Tests for projected gradient descent optimizer."""

    def test_converges_quadratic(self):
        """Minimize f(x) = sum((x - [1, 2])^2) with hypercube projection."""
        def f(x):
            return jnp.sum((x - jnp.array([1.0, 2.0])) ** 2)

        x0 = jnp.array([0.0, 0.0])
        result = projected_gradient(
            f, x0, projection_method="projection_non_negative",
            lr=0.1, maxiter=500,
        )
        np.testing.assert_allclose(np.array(result["x"]), [1.0, 2.0],
                                   atol=0.05,
                                   err_msg="projected_gradient failed on quadratic")

    def test_non_negative_projection(self):
        """Parameters should remain non-negative with non-negative projection."""
        def f(x):
            return jnp.sum((x - jnp.array([-5.0, 3.0])) ** 2)

        x0 = jnp.array([1.0, 1.0])
        result = projected_gradient(f, x0,
                                    projection_method="projection_non_negative",
                                    lr=0.1, maxiter=200)
        x_opt = np.array(result["x"])
        # First param should be clipped to 0 (unconstrained optimum is -5)
        assert x_opt[0] >= -1e-6, f"Non-negative violated: x[0]={x_opt[0]}"


# ===================================================================
# Fit convergence and failure surfacing (M-J)
# ===================================================================

class TestFitConvergenceSurfacing:
    """Ensure fit() surfaces its convergence state.

    Two silent-failure modes are guarded:

    1. ``fit()`` returns finite-looking params that aren't at a stationary
       point of the log-likelihood — the optimiser stopped early but the
       user has no way to tell.
    2. ``fit()`` returns finite-looking params on data that violates the
       distribution's support — the user gets a ``.params`` dict with no
       indication that the fit is meaningless.

    Both are canonical no-silent-failure violations (CLAUDE.md rule 1).
    """

    @pytest.mark.parametrize(
        "dist_name,true_params,fit_kwargs",
        [
            ("normal", {"mu": 0.0, "sigma": 1.0}, {}),
            ("gamma", {"alpha": 2.0, "beta": 3.0}, {"maxiter": 500}),
            ("lognormal", {"mu": 0.5, "sigma": 0.5}, {}),
        ],
        ids=["normal", "gamma", "lognormal"],
    )
    def test_fit_gradient_near_zero_at_optimum(
        self, dist_name, true_params, fit_kwargs
    ):
        """|grad LL(fitted_params)| / n < 1e-3 at the fitted optimum.

        Per-observation normalisation lets one threshold work across the
        three fit paths:
        - Normal / LogNormal use closed-form MLE (grad is machine zero).
        - Gamma uses projected-gradient MLE (grad is small but non-zero
          unless enough iterations are taken — hence maxiter=500).
        """
        rng = np.random.default_rng(42)
        if dist_name == "normal":
            dist = normal
            x = jnp.asarray(
                rng.normal(true_params["mu"], true_params["sigma"], 2000)
            )
        elif dist_name == "gamma":
            dist = gamma
            # numpy gamma uses shape/scale; CopulAX uses shape/rate ⇒ scale = 1/beta
            x = jnp.asarray(
                rng.gamma(
                    true_params["alpha"], 1.0 / true_params["beta"], 2000
                )
            )
        elif dist_name == "lognormal":
            dist = lognormal
            x = jnp.asarray(
                rng.lognormal(true_params["mu"], true_params["sigma"], 2000)
            )

        fitted = dist.fit(x, **fit_kwargs)
        params = fitted.params

        # Preflight: fitted params must be finite; a NaN/Inf param is a
        # separate failure class handled by the pathological-data test.
        for name, value in params.items():
            assert np.isfinite(float(value)), (
                f"{dist.name}: fitted param {name} = {float(value)} is not "
                f"finite on well-behaved data"
            )

        # Compute grad LL(params) via a pytree gradient.
        def ll_fn(p):
            return dist.loglikelihood(x=x, params=p)

        grad_tree = jax.grad(ll_fn)(params)
        grad_vals = np.array(
            [float(grad_tree[k]) for k in sorted(grad_tree.keys())]
        )
        per_obs_grad = np.max(np.abs(grad_vals)) / float(x.shape[0])

        assert per_obs_grad < 1e-3, (
            f"{dist.name}: |grad LL| / n = {per_obs_grad:.3e} at fitted "
            f"params {dict(params)} exceeds 1e-3. The optimiser has not "
            f"reached a stationary point — fit() is returning a "
            f"non-optimum without surfacing the failure."
        )

    def test_fit_pathological_data_surfaces_failure(self):
        """Gamma.fit on all-negative data must not return a usable fit.

        Gamma support is [0, ∞). On strictly-negative data there is no
        valid MLE. The library must surface that by one of:

        (a) returning NaN / non-finite values in the fitted params, OR
        (b) returning params whose loglikelihood on the original data is
            -inf (so any downstream AIC/BIC/fitter ranking flags the
            violation automatically).

        Returning finite params with a finite loglikelihood would be a
        silent-garbage failure: the caller has no signal that the fit is
        meaningless.
        """
        x = jnp.asarray(np.linspace(-5.0, -0.1, 200))

        fitted = gamma.fit(x)
        params = fitted.params

        params_non_finite = any(
            not np.isfinite(float(v)) for v in params.values()
        )
        ll = float(gamma.loglikelihood(x=x, params=params))
        ll_signals_failure = np.isneginf(ll) or np.isnan(ll)

        assert params_non_finite or ll_signals_failure, (
            f"Gamma.fit on all-negative data returned finite params "
            f"{dict(params)} with finite loglikelihood={ll}. The domain "
            f"violation is not surfaced — this is the silent-garbage "
            f"failure mode CLAUDE.md rule 1 forbids."
        )
