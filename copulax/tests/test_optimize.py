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


def _unconstrained_box(n: int) -> dict:
    """projection_box options that act as the identity (unconstrained).

    ``optax.projections`` has no explicit identity projection; the codebase
    uses ``projection_box`` with infinite bounds as the unconstrained case
    (e.g. GARCH fits in ``_garch_base.py``). Bounds are shaped as column
    vectors to match ``single_update``'s ``x[None].T`` reshape.
    """
    return {
        "lower": jnp.full((n, 1), -jnp.inf),
        "upper": jnp.full((n, 1), jnp.inf),
    }


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

    def test_nan_gradient_propagates(self):
        """``adam`` must propagate NaN through the moment updates so a
        bad parameter region surfaces loudly downstream — silent
        zeroing would mask infeasible regions in every fit path that
        uses Adam under projected gradient."""
        g = jnp.array(float('nan'))
        m = jnp.zeros(())
        v = jnp.zeros(())
        direction, _, _, _ = adam(g, m, v, t=0)
        assert np.isnan(float(direction)), (
            "adam silently zeroed a NaN gradient instead of propagating "
            "it; downstream fitters rely on NaN propagation to surface "
            "infeasible parameter regions."
        )


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
# HARD-04: best-iterate return contract
# ===================================================================

class TestBestIterate:
    """``projected_gradient`` must return the BEST iterate encountered
    (argmin of the minimised objective over the scan), not the LAST.

    First-order Adam-projected steps are not monotone, so the last iterate
    can be worse than an earlier one. Returning the last point hands back a
    plausible-but-suboptimal fit (the J1/B7 pathology, dossier section 6).
    These tests assert the returned ``x`` is the best point visited and that
    ``val`` is the objective at that point.
    """

    def test_convex_quadratic_returns_minimiser_and_matching_val(self):
        """On a strictly convex quadratic, x is the minimiser and val == f(x).

        Guards the {x, val} contract under best-iterate: whatever point is
        returned in ``x``, ``val`` must be the objective evaluated there.
        """
        target = jnp.array([1.0, 2.0])

        def f(x):
            return jnp.sum((x - target) ** 2)

        x0 = jnp.array([0.0, 0.0])
        result = projected_gradient(
            f, x0, projection_method="projection_non_negative",
            lr=0.1, maxiter=500,
        )
        np.testing.assert_allclose(np.array(result["x"]), np.array(target),
                                   atol=0.05,
                                   err_msg="best-iterate x not at minimiser")
        # val MUST equal the objective at the returned x (contract).
        np.testing.assert_allclose(
            float(result["val"]), float(f(result["x"])), atol=1e-6,
            err_msg="val is not the objective evaluated at the returned x",
        )

    def test_last_iterate_worse_than_best_is_not_returned(self):
        """An oscillating run must return the best point seen, not the last.

        On f(x) = (x - 1)^2 from x0 = -5 with lr = 0.8, Adam approaches the
        minimiser 1 then oscillates around it: an intermediate iterate lands
        at val ~= 8e-5 (x ~= 0.99) while the LAST iterate sits at val ~= 2.8e-2
        (x ~= 0.83). The trajectory is fully deterministic (fixed data, fixed
        step rule). Best-iterate must return the ~8e-5 point; the last-iterate
        code returns the ~2.8e-2 endpoint. The threshold 5e-3 sits cleanly
        between the two, so this test fails on last-iterate code and passes on
        best-iterate code.
        """
        target = jnp.array([1.0])

        def f(x):
            return jnp.sum((x - target) ** 2)

        x0 = jnp.array([-5.0])
        result = projected_gradient(
            f, x0, projection_method="projection_box",
            projection_options=_unconstrained_box(1),
            lr=0.8, maxiter=60,
        )
        # The returned objective must be the best one visited (~8e-5), well
        # below the last-iterate value (~2.8e-2). The 5e-3 cap separates them.
        assert float(result["val"]) < 5e-3, (
            f"projected_gradient returned val={float(result['val'])} >= 5e-3; "
            f"it returned a non-best (last/oscillating) iterate instead of the "
            f"best point visited."
        )
        # The returned point should be the best one (near the minimiser 1),
        # not the oscillating endpoint (~0.83).
        np.testing.assert_allclose(np.array(result["x"]), np.array(target),
                                   atol=0.05,
                                   err_msg="returned x is not the best iterate")
        np.testing.assert_allclose(
            float(result["val"]), float(f(result["x"])), atol=1e-6,
            err_msg="val is not the objective evaluated at the returned x",
        )

    def test_well_conditioned_fit_no_regression(self):
        """When the last iterate IS the best, best-iterate must not regress.

        A well-conditioned descent that converges monotonically should reach
        the same optimum as before — best == last in the limit.
        """
        target = jnp.array([3.0, -1.0, 4.0])

        def f(x):
            return jnp.sum((x - target) ** 2)

        x0 = jnp.array([0.0, 0.0, 0.0])
        result = projected_gradient(
            f, x0, projection_method="projection_box",
            projection_options=_unconstrained_box(3),
            lr=0.05, maxiter=1000,
        )
        np.testing.assert_allclose(np.array(result["x"]), np.array(target),
                                   atol=1e-3,
                                   err_msg="well-conditioned fit regressed")


# ===================================================================
# HARD-10 (D-11): NaN-gradient freeze-carry, not silent zeroing
# ===================================================================

class TestNaNGradFreeze:
    """On a non-finite gradient the Adam scan must FREEZE the carry (hold
    the current iterate) and record a ``nan_encountered`` flag, rather than
    ``nan_to_num``-zeroing the gradient and silently stepping on.

    FINDING-01-07 / HARD-10 D-11: silent zeroing masks bad parameter
    regions in downstream fitters. A degenerate fit must surface NaN (the
    honest signal), never a finite-but-wrong stalled point.
    """

    def test_return_dict_exposes_new_keys(self):
        """Return dict carries x, val, best_val, and nan_encountered."""
        def f(x):
            return jnp.sum((x - jnp.array([1.0, 2.0])) ** 2)

        result = projected_gradient(
            f, jnp.array([0.0, 0.0]),
            projection_method="projection_non_negative",
            lr=0.1, maxiter=100,
        )
        for key in ("x", "val", "best_val", "nan_encountered"):
            assert key in result, f"return dict missing key {key!r}"

    def test_clean_fit_reports_no_nan(self):
        """A well-behaved fit must report nan_encountered == False."""
        def f(x):
            return jnp.sum((x - jnp.array([1.0, 2.0])) ** 2)

        result = projected_gradient(
            f, jnp.array([0.0, 0.0]),
            projection_method="projection_non_negative",
            lr=0.1, maxiter=200,
        )
        assert not bool(result["nan_encountered"]), (
            "nan_encountered set True on a clean, well-behaved fit"
        )

    def test_nan_gradient_after_k_returns_best_finite_and_flags(self):
        """Gradient goes non-finite past a threshold; the best FINITE point
        is returned and nan_encountered is True.

        f(x) = (x - 5)^2 with a non-finite-gradient region for x < 4.5. From
        x0 = 8 the iterate descends toward the minimiser 5, visiting a point
        at ~4.99 (val ~2e-4), then overshoots below 4.5 on a later step —
        Adam's ~lr-sized step near the flat minimum guarantees the crossing —
        where the gradient is non-finite and the carry FREEZES. The
        deterministic freeze-carry outcome (verified numerically) is
        best_x ~= 4.99, nan_encountered = True. The solver must therefore
        (a) never return a NaN, (b) return the best finite point near 5, and
        (c) flag nan_encountered.
        """
        threshold = 4.5
        target = 5.0

        def f(x):
            xs = x[0]
            base = (xs - target) ** 2
            # Non-finite GRADIENT region for xs < threshold: sqrt of a
            # negative argument yields a nan gradient there, while the
            # objective VALUE is nan-guarded to 0 so best-iterate tracking
            # never prefers the frozen point over the near-optimum visit.
            bad = jnp.where(xs < threshold,
                            jnp.sqrt(jnp.maximum(xs - threshold, 0.0) - 1e-6),
                            0.0)
            return base + jnp.where(jnp.isnan(bad), 0.0, bad)

        x0 = jnp.array([8.0])
        result = projected_gradient(
            f, x0, projection_method="projection_box",
            projection_options=_unconstrained_box(1),
            lr=0.8, maxiter=200,
        )
        x_ret = float(result["x"][0])
        assert np.isfinite(x_ret), (
            f"returned x={x_ret} is non-finite; freeze-carry failed to hold "
            f"a finite iterate"
        )
        assert np.isfinite(float(result["val"])), (
            f"returned val={float(result['val'])} is non-finite"
        )
        # The best finite point should be at/near the true minimiser 5.
        np.testing.assert_allclose(x_ret, target, atol=0.25,
                                   err_msg="did not return the best finite "
                                           "iterate near the minimiser")
        assert bool(result["nan_encountered"]), (
            "nan_encountered False despite a non-finite gradient encountered "
            "during the scan"
        )

    def test_all_degenerate_returns_nan_and_flags(self):
        """Objective/gradient NaN from the first evaluation everywhere.

        There is no valid point, so the honest result is NaN (not a finite
        silently-wrong stall) and nan_encountered must be True. This is the
        no-silent-failure contract (CLAUDE.md rule 1).
        """
        def f(x):
            # sqrt of a negative argument -> NaN objective AND NaN gradient
            # (0.5/sqrt(neg)) at every point, from the very first evaluation.
            return jnp.sum(jnp.sqrt(x))

        x0 = jnp.array([-0.5, -0.5])
        result = projected_gradient(
            f, x0, projection_method="projection_box",
            projection_options=_unconstrained_box(2),
            lr=0.1, maxiter=50,
        )
        assert np.isnan(float(result["val"])) or not np.all(
            np.isfinite(np.array(result["x"]))
        ), (
            f"all-degenerate fit returned finite x={np.array(result['x'])} "
            f"val={float(result['val'])}; the domain violation is not "
            f"surfaced (silent-garbage failure)."
        )
        assert bool(result["nan_encountered"]), (
            "nan_encountered False on an all-degenerate fit whose gradient "
            "was non-finite at every evaluation"
        )

    def test_backcompat_x_val_meaning_preserved(self):
        """A caller reading only x and val observes the best iterate in x
        and its objective in val — the existing contract, unchanged.
        """
        target = jnp.array([2.0, -3.0])

        def f(x):
            return jnp.sum((x - target) ** 2)

        result = projected_gradient(
            f, jnp.array([0.0, 0.0]),
            projection_method="projection_box",
            projection_options=_unconstrained_box(2),
            lr=0.05, maxiter=1000,
        )
        # Two-key view still works and is self-consistent.
        x_only = result["x"]
        val_only = result["val"]
        np.testing.assert_allclose(float(val_only), float(f(x_only)),
                                   atol=1e-6,
                                   err_msg="val != f(x): {x,val} contract "
                                           "broken for two-key callers")
        np.testing.assert_allclose(np.array(x_only), np.array(target),
                                   atol=1e-3,
                                   err_msg="x is not the best (optimal) "
                                           "iterate for a converging fit")


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
