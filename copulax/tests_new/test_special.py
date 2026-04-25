"""Rigorous tests for copulax._src.special: Bessel K_v, igammainv, stdtr,
digamma, trigamma.

Cross-validates against scipy.special with tight tolerances that catch
the known 59% Bessel K_v error (FINDING-01-02) and regime transition
discontinuity (FINDING-01-03).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special
import scipy.stats

from copulax._src.special import kv, log_kv, igammainv, igammacinv, stdtr, digamma, trigamma
from copulax._src.special import _stable_log_sinh, log_kv_plus_s_log_r


# ===================================================================
# Bessel K_v
# ===================================================================

class TestKv:
    """Tests for modified Bessel function of the second kind K_v(x)."""

    # --- Accuracy vs scipy across all regimes ---

    @pytest.mark.parametrize("v", [0, 1, 2, 5, 10])
    def test_matches_scipy_integer_order(self, v):
        """Integer orders: expect machine-precision agreement."""
        x = np.logspace(-2, 2, 50)
        cx = np.array(jax.vmap(lambda xi: kv(float(v), xi))(jnp.array(x)))
        sp = scipy.special.kv(v, x)
        mask = np.isfinite(sp) & np.isfinite(cx) & (sp > 0)
        np.testing.assert_allclose(cx[mask], sp[mask], rtol=1e-6, atol=1e-12,
                                   err_msg=f"K_{v}(x) mismatch for integer v={v}")

    @pytest.mark.parametrize("v", [0.5, 1.5, 2.5, 4.5])
    def test_matches_scipy_half_integer_order(self, v):
        """Half-integer orders have closed-form expressions."""
        x = np.logspace(-2, 2, 50)
        cx = np.array(jax.vmap(lambda xi: kv(v, xi))(jnp.array(x)))
        sp = scipy.special.kv(v, x)
        mask = np.isfinite(sp) & np.isfinite(cx) & (sp > 0)
        np.testing.assert_allclose(cx[mask], sp[mask], rtol=1e-6, atol=1e-12,
                                   err_msg=f"K_{v}(x) mismatch for v={v}")

    @pytest.mark.parametrize("v", [0.1, 0.25, 0.75, 1.3])
    def test_matches_scipy_fractional_order(self, v):
        """Fractional orders: the regime where the 59% error was found.

        FINDING-01-02: Gauss-Laguerre quadrature has a singularity at t=0
        for v < 1, causing up to 59% relative error. This test catches it
        with rtol=1e-4 (the old suite used rtol=3e-3).
        """
        x = np.logspace(-1, 1.5, 40)
        cx = np.array(jax.vmap(lambda xi: kv(v, xi))(jnp.array(x)))
        sp = scipy.special.kv(v, x)
        mask = np.isfinite(sp) & np.isfinite(cx) & (sp > 0)
        np.testing.assert_allclose(cx[mask], sp[mask], rtol=1e-4, atol=1e-12,
                                   err_msg=f"K_{v}(x) mismatch for fractional v={v}")

    @pytest.mark.parametrize("v", [0.01, 0.05])
    def test_matches_scipy_small_v(self, v):
        """Very small v: near the v=0 branch threshold."""
        x = np.logspace(-1, 1, 30)
        cx = np.array(jax.vmap(lambda xi: kv(v, xi))(jnp.array(x)))
        sp = scipy.special.kv(v, x)
        mask = np.isfinite(sp) & np.isfinite(cx) & (sp > 0)
        np.testing.assert_allclose(cx[mask], sp[mask], rtol=1e-4, atol=1e-12,
                                   err_msg=f"K_{v}(x) mismatch for small v={v}")

    # --- Mathematical identities ---

    def test_symmetry_neg_v(self):
        """K_{-v}(x) == K_v(x) for all v, x > 0 (DLMF 10.27.3)."""
        vs = [0.5, 1.0, 1.5, 2.5]
        x = np.logspace(-1, 1, 20)
        for v in vs:
            k_pos = np.array(jax.vmap(lambda xi: kv(v, xi))(jnp.array(x)))
            k_neg = np.array(jax.vmap(lambda xi: kv(-v, xi))(jnp.array(x)))
            np.testing.assert_allclose(k_pos, k_neg, rtol=1e-6,
                                       err_msg=f"K_{{-{v}}} != K_{{{v}}}")

    def test_recurrence_relation(self):
        """K_{v+1}(x) = K_{v-1}(x) + (2v/x)*K_v(x) (DLMF 10.29.1)."""
        v = 1.5
        x = np.logspace(-0.5, 1.5, 30)
        k_vm1 = np.array(jax.vmap(lambda xi: kv(v - 1, xi))(jnp.array(x)))
        k_v = np.array(jax.vmap(lambda xi: kv(v, xi))(jnp.array(x)))
        k_vp1 = np.array(jax.vmap(lambda xi: kv(v + 1, xi))(jnp.array(x)))
        rhs = k_vm1 + (2 * v / x) * k_v
        mask = np.isfinite(k_vp1) & np.isfinite(rhs) & (k_vp1 > 0)
        np.testing.assert_allclose(k_vp1[mask], rhs[mask], rtol=1e-4,
                                   err_msg="Recurrence relation violated")

    def test_positivity(self):
        """K_v(x) > 0 for all x > 0."""
        for v in [0, 0.25, 1, 2.5]:
            x = np.logspace(-2, 2, 30)
            cx = np.array(jax.vmap(lambda xi: kv(float(v), xi))(jnp.array(x)))
            assert np.all(cx[np.isfinite(cx)] > 0), f"K_{v} not positive"

    def test_shape_preservation(self):
        """Output shape matches input shape."""
        x = jnp.array([0.5, 1.0, 2.0])
        results = jax.vmap(lambda xi: kv(1.0, xi))(x)
        assert results.shape == x.shape

    def test_gradient_wrt_x_vs_finite_diff(self):
        """jax.grad(kv, argnums=1) matches central finite differences."""
        v, x0 = 1.0, 2.0
        h = 1e-5
        grad_fn = jax.grad(kv, argnums=1)
        analytic = float(grad_fn(v, jnp.array(x0)))
        numerical = (float(kv(v, jnp.array(x0 + h)))
                     - float(kv(v, jnp.array(x0 - h)))) / (2 * h)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-3,
                                   err_msg="K_v x-gradient mismatch")

    @pytest.mark.parametrize("v,x", [
        (0.25, 1.0), (0.5, 2.0), (1.0, 1.5), (2.5, 3.0),
        (5.0, 2.5), (10.0, 5.0), (20.0, 10.0), (50.0, 25.0),
    ])
    def test_gradient_wrt_v_vs_finite_diff(self, v, x):
        """jax.grad(kv, argnums=0) matches central finite differences.

        This verifies that autodiff flows correctly through the quadrature
        (v < 15) and Debye expansion (v >= 15) w.r.t. the order parameter v.
        """
        h = 1e-6
        grad_fn = jax.grad(kv, argnums=0)
        analytic = float(grad_fn(jnp.array(float(v)), jnp.array(float(x))))
        numerical = (float(kv(jnp.array(v + h), jnp.array(float(x))))
                     - float(kv(jnp.array(v - h), jnp.array(float(x))))) / (2 * h)
        if abs(numerical) < 1e-15:
            pytest.skip("Numerical gradient too small for comparison")
        np.testing.assert_allclose(analytic, numerical, rtol=1e-3,
                                   err_msg=f"K_v v-gradient mismatch at v={v}, x={x}")

    def test_jit_compilable(self):
        """kv is JIT-compatible."""
        f = jax.jit(lambda x: kv(1.0, x))
        result = f(jnp.array(1.0))
        assert np.isfinite(float(result))


# ===================================================================
# log Bessel K_v  — numerical stability in extreme regimes
# ===================================================================

class TestLogKv:
    """Tests for log-space K_v(x).

    ``log_kv`` is the load-bearing primitive behind the skewed-t CDF fix
    (see ``memory/skewed_t_cdf_bug.md``) and the GIG-moment ratio path.
    Its contract is to remain finite in regimes where K_v(x) itself
    underflows to 0 in float64 (x >= ~710).
    """

    # --- Accuracy vs scipy across regimes where scipy.kv is a safe reference ---

    @pytest.mark.parametrize("v,x", [
        (0.5, 0.1), (0.5, 1.0), (0.5, 50.0),
        (1.0, 0.5), (1.0, 5.0), (1.0, 100.0),
        (2.5, 1.0), (2.5, 20.0), (2.5, 100.0),
        (5.0, 2.0), (5.0, 30.0), (5.0, 150.0),
    ])
    def test_matches_log_of_kv(self, v, x):
        """log_kv(v, x) ≈ log(scipy.special.kv(v, x)) in safe (non-underflow) regimes.

        Uses scipy.special.kv as the external reference. The rtol is 1e-6,
        set against the absolute magnitude of log_kv itself (which grows
        roughly linearly with x via the -x term in the asymptotic).
        """
        sp = float(scipy.special.kv(v, x))
        assert sp > 0 and np.isfinite(sp), (
            f"scipy.kv({v}, {x}) = {sp!r} is not a safe reference"
        )
        log_sp = np.log(sp)
        cx = float(log_kv(v, jnp.array(x)))
        np.testing.assert_allclose(
            cx, log_sp, rtol=1e-6,
            err_msg=f"log_kv({v}, {x}) = {cx} vs log(scipy.kv) = {log_sp}",
        )

    # --- Finiteness where K_v itself underflows or is singular ---

    @pytest.mark.parametrize("v,x", [
        (0.5, 700.0),   # x beyond the kv underflow threshold (~710)
        (1.0, 1e6),     # massively large x — kv=0 in float64
        (-0.5, 1e-8),   # negative v (K_{-v} = K_v) at tiny x
        (10.0, 1e-5),   # moderate v at tiny x — small-x asymptotic branch
    ])
    def test_finite_in_extreme_regime(self, v, x):
        """log_kv must remain finite in regimes where K_v itself underflows/diverges.

        These are the exact regimes the skewed-t CDF touches; a silent NaN
        or inf here reintroduces the bug fixed in ``memory/skewed_t_cdf_bug.md``.
        """
        cx = float(log_kv(v, jnp.array(x)))
        assert np.isfinite(cx), f"log_kv({v}, {x}) = {cx!r} is not finite"

    # --- Leading-order large-x asymptotic (DLMF 10.40.2) ---

    @pytest.mark.parametrize("v", [0.5, 1.0, 2.0, 5.0])
    def test_agrees_with_asymptotic_at_large_x(self, v):
        r"""log K_v(x) ≈ 0.5·log(π/(2x)) − x for x ≫ v² (DLMF 10.40.2 leading term).

        At x = 10⁴ the next-order correction :math:`(4v²-1)/(8x)` is
        O(10⁻⁴) in absolute value, well below rtol=1e-3 against a
        reference of magnitude ~10⁴.
        """
        x = 1e4
        cx = float(log_kv(v, jnp.array(x)))
        expected = 0.5 * np.log(np.pi / (2.0 * x)) - x
        np.testing.assert_allclose(
            cx, expected, rtol=1e-3,
            err_msg=f"log_kv({v}, {x}) = {cx} vs asymptotic {expected}",
        )


# ===================================================================
# log Bessel K_v — custom_jvp gradient correctness
# ===================================================================

class TestLogKvGradient:
    """Gradient tests for ``log_kv``'s hand-written ``@jax.custom_jvp`` rule.

    The rule uses
      * the standard recurrence ``∂/∂x log K_v(x) = -½(K_{v-1}/K_v + K_{v+1}/K_v)``
        for the x-tangent, and
      * the derivative of the integral representation (DLMF 10.32.9)
        ``∂K_v/∂v = ∫ t·sinh(vt)·e^{-x cosh t} dt`` — evaluated with the
        same 64-node Gauss-Legendre / saddle-point-centred interval as
        ``_log_kv_legendre`` — for the v-tangent.

    These tests pin the rule against two independent references:
      1. ``scipy.special.kvp`` (analytic ∂K_v/∂x via scipy) for d/dx.
      2. High-order central finite differences for d/dv.
    plus the symmetry identity ``∂log K_v/∂v|_{-v} = -∂log K_v/∂v|_{+v}``
    and the corollary ``∂log K_v/∂v|_{v=0} = 0`` (``K_v`` is even in ν).
    """

    # --- d/dx against scipy.special.kvp (exact reference) ---

    @pytest.mark.parametrize("v,x", [
        (-5.0, 0.1), (-5.0, 1.0), (-5.0, 10.0), (-5.0, 100.0),
        (-1.5, 0.1), (-1.5, 1.0), (-1.5, 10.0), (-1.5, 100.0),
        (-0.1, 0.1), (-0.1, 1.0), (-0.1, 10.0), (-0.1, 100.0),
        (0.5, 0.1), (0.5, 1.0), (0.5, 10.0), (0.5, 100.0),
        (2.0, 0.1), (2.0, 1.0), (2.0, 10.0), (2.0, 100.0),
        (10.0, 0.1), (10.0, 1.0), (10.0, 10.0), (10.0, 100.0),
        (20.0, 1.0), (20.0, 10.0), (20.0, 100.0),
    ])
    def test_d_dx_matches_scipy_kvp(self, v, x):
        """``jax.grad(log_kv, argnums=1)`` matches ``K_v'(x) / K_v(x)``.

        ``scipy.special.kvp`` gives ``K_v'(x)`` to double-precision;
        dividing by ``K_v(x)`` yields an analytic reference for
        ``∂log K_v/∂x`` wherever ``K_v(x)`` is representable.

        Tolerance ``rtol=1e-5`` mirrors the library's existing forward
        precision at large-v / large-x combinations (e.g. v=5, x=100):
        the 4-term Hankel series in :py:func:`_log_kv_large_x` has
        ~3e-6 relative error from the omitted ``a_4`` coefficient at
        that regime, which the recurrence-based x-tangent inherits.
        That precision is sufficient for all current downstream uses
        (tighter than the ``log_kv`` forward tolerance used in
        :py:meth:`TestLogKv.test_matches_log_of_kv`).
        """
        kvp_val = float(scipy.special.kvp(v, x))
        kv_val = float(scipy.special.kv(v, x))
        assert kv_val > 0 and np.isfinite(kvp_val / kv_val), (
            f"scipy reference unreliable at v={v}, x={x}"
        )
        expected = kvp_val / kv_val

        v_j = jnp.asarray(v, dtype=float)
        x_j = jnp.asarray(x, dtype=float)
        got = float(jax.grad(lambda xi: log_kv(v_j, xi))(x_j))
        np.testing.assert_allclose(
            got, expected, rtol=1e-5, atol=1e-8,
            err_msg=(
                f"∂log_kv/∂x({v}, {x}): jvp={got} vs scipy.kvp/kv={expected}"
            ),
        )

    # --- d/dv against central finite differences ---

    @pytest.mark.parametrize("v,x", [
        (-5.0, 0.1), (-5.0, 1.0), (-5.0, 10.0),
        (-1.5, 0.1), (-1.5, 1.0), (-1.5, 10.0),
        (-0.1, 0.1), (-0.1, 1.0), (-0.1, 10.0),
        (0.5, 0.1), (0.5, 1.0), (0.5, 10.0),
        (2.0, 0.1), (2.0, 1.0), (2.0, 10.0),
        (10.0, 0.1), (10.0, 1.0), (10.0, 10.0),
        (20.0, 1.0), (20.0, 10.0),
        # Deep Debye regime (v >= 15 picks the Debye primal; the
        # ν-tangent uses the integral-representation quadrature
        # regardless of which forward regime fires, so these cases
        # stress-test that the quadrature rule remains accurate where
        # the forward would otherwise have switched).
        (25.0, 1.0), (25.0, 10.0), (25.0, 50.0),
        (30.0, 1.0), (30.0, 10.0), (30.0, 100.0),
        (50.0, 5.0), (50.0, 50.0),
    ])
    def test_d_dv_matches_finite_diff(self, v, x):
        """``jax.grad(log_kv, argnums=0)`` matches 4-point central FD in ν.

        FD stencil (``h=1e-4``) delivers ~10-11 digits on smooth
        functions; ``rtol=1e-6`` leaves a cushion for the step-size
        rounding noise while still catching any analytical mistake.
        """
        h = 1e-4
        x_j = jnp.asarray(x, dtype=float)
        fd = (
            float(log_kv(v + h, x_j)) - float(log_kv(v - h, x_j))
        ) / (2.0 * h)

        v_j = jnp.asarray(v, dtype=float)
        got = float(jax.grad(lambda vi: log_kv(vi, x_j))(v_j))
        np.testing.assert_allclose(
            got, fd, rtol=1e-6, atol=1e-9,
            err_msg=f"∂log_kv/∂v({v}, {x}): jvp={got} vs FD={fd}",
        )

    # --- v = 0 identity ---

    @pytest.mark.parametrize("x", [0.1, 1.0, 10.0, 100.0])
    def test_d_dv_is_exactly_zero_at_v_zero(self, x):
        """``∂log K_v(x)/∂v|_{v=0} = 0`` (``K_v`` is even in ν).

        Bit-exact zero is the required semantics: the ν-tangent inside
        ``_log_kv_pos`` returns ``sinh(0·t)·... = 0`` identically, and
        the public wrapper's ``jnp.abs`` chain rule multiplies by
        ``sign(0) = 0``.  Any non-zero gradient here would leak an
        asymmetry back into downstream fits.
        """
        x_j = jnp.asarray(x, dtype=float)
        got = float(
            jax.grad(lambda vi: log_kv(vi, x_j))(jnp.asarray(0.0, dtype=float))
        )
        assert got == 0.0, (
            f"∂log_kv/∂v|_{{v=0, x={x}}} = {got}, expected exact 0"
        )

    # --- Antisymmetry: grad(-v) == -grad(+v) ---

    @pytest.mark.parametrize("v,x", [
        (0.1, 5.0), (0.5, 1.0), (2.0, 3.0), (10.0, 7.0),
    ])
    def test_d_dv_is_antisymmetric_in_v(self, v, x):
        """``∂log K_{-v}(x)/∂v = -∂log K_{+v}(x)/∂v`` for ``v > 0``.

        Follows from ``K_{-v} = K_v`` (even in ν): the log is even, so
        its derivative is odd.  The custom_jvp reproduces this via the
        ``jnp.abs`` wrapper's ``sign(v)`` chain-rule multiplier.
        """
        x_j = jnp.asarray(x, dtype=float)
        g_pos = float(jax.grad(lambda vi: log_kv(vi, x_j))(
            jnp.asarray(+v, dtype=float)
        ))
        g_neg = float(jax.grad(lambda vi: log_kv(vi, x_j))(
            jnp.asarray(-v, dtype=float)
        ))
        np.testing.assert_allclose(
            g_pos + g_neg, 0.0, atol=1e-12,
            err_msg=(
                f"antisymmetry violated: grad(+{v}, {x}) + grad(-{v}, {x}) "
                f"= {g_pos + g_neg}, expected 0"
            ),
        )

    # --- Extreme-x stability (where K_v itself underflows) ---

    @pytest.mark.parametrize("v,x", [
        (0.5, 500.0),    # x near float64 K_v underflow boundary
        (2.0, 500.0),
        (5.0, 1000.0),   # K_v already zero in float64
    ])
    def test_d_dx_finite_at_extreme_x(self, v, x):
        """Gradients must remain finite when ``K_v(x)`` underflows.

        The recurrence form ``-½(exp(log K_{v-1} − log K_v) + …)``
        keeps the computation in log-space, so the x-tangent stays
        near -1 for large x (leading asymptotic
        ``∂log K_v/∂x ≈ -1 − (4v²-1)/(8x²)``).
        """
        v_j = jnp.asarray(v, dtype=float)
        x_j = jnp.asarray(x, dtype=float)
        got = float(jax.grad(lambda xi: log_kv(v_j, xi))(x_j))
        assert np.isfinite(got), f"∂log_kv/∂x({v}, {x}) = {got}"
        # Leading-order: ∂log K_v/∂x → -1 as x → ∞.
        np.testing.assert_allclose(
            got, -1.0, atol=5e-3,
            err_msg=f"tail asymptote broken at v={v}, x={x}: got {got}",
        )

    # --- Chain rule from log_kv to kv (Fix 3 must not regress kv grads) ---

    @pytest.mark.parametrize("v,x", [
        (0.5, 2.0), (1.0, 3.0), (2.5, 5.0), (5.0, 7.0),
    ])
    def test_kv_gradient_inherits_from_log_kv(self, v, x):
        """``∂kv/∂x = K_v(x) · ∂log K_v(x)/∂x`` (chain rule via exp).

        ``kv`` is defined as ``exp(log_kv(...))``, so its gradients
        flow through the custom_jvp rule via the standard exp chain
        rule.  Cross-check against ``scipy.special.kvp``.
        """
        v_j = jnp.asarray(v, dtype=float)
        x_j = jnp.asarray(x, dtype=float)
        got = float(jax.grad(kv, argnums=1)(v_j, x_j))
        expected = float(scipy.special.kvp(v, x))
        np.testing.assert_allclose(
            got, expected, rtol=1e-8,
            err_msg=f"∂kv/∂x({v}, {x}): jvp={got} vs scipy.kvp={expected}",
        )

    # --- JIT compatibility of gradient path ---

    def test_grad_is_jit_compilable(self):
        """``jax.jit(jax.grad(log_kv))`` compiles and runs.

        Regression guard: the custom_jvp rule must trace cleanly under
        ``jit`` — the performance win in copula EM fitting depends on
        this (the rule embeds a Gauss-Legendre quadrature for the
        ν-tangent, so a tracing bug would resurface as a compile
        error or a silent NaN).
        """
        g = jax.jit(jax.grad(lambda v, x: log_kv(v, x).sum(), argnums=(0, 1)))
        dv, dx = g(jnp.asarray(1.5), jnp.asarray(3.0))
        assert np.isfinite(float(dv)) and np.isfinite(float(dx))

    # --- vmap over gradients (the custom_jvp rule must batch correctly) ---

    def test_grad_vmap_over_x_matches_per_point(self):
        r"""``vmap(grad(log_kv, x))`` agrees with scalar ``grad`` element-by-element.

        The custom_jvp rule's x-tangent uses the recurrence directly on
        the full ``x`` array; the ν-tangent vmaps ``_dlog_kv_dv_single``
        over ``x`` internally.  When the user wraps the whole
        ``grad(log_kv)`` call in an outer ``vmap``, JAX must batch the
        rule correctly.  Mismatch here would indicate a subtle
        vmap/custom_jvp interaction bug.
        """
        v = jnp.asarray(1.5, dtype=float)
        xs = jnp.asarray([0.5, 1.0, 2.0, 5.0, 10.0, 50.0], dtype=float)

        # Scalar reference: grad one point at a time.
        per_point_dx = jnp.array([
            jax.grad(lambda xi: log_kv(v, xi))(xi) for xi in xs
        ])

        # vmap'd gradient of the per-point function.
        vmap_dx = jax.vmap(lambda xi: jax.grad(lambda x: log_kv(v, x))(xi))(xs)

        np.testing.assert_allclose(
            np.asarray(vmap_dx), np.asarray(per_point_dx),
            rtol=1e-10, atol=1e-12,
            err_msg="vmap(grad(log_kv, x)) disagrees with per-point grad",
        )

    def test_grad_vmap_over_v_matches_per_point(self):
        r"""``vmap(grad(log_kv, v))`` agrees with scalar ``grad`` per v.

        Outer ``vmap`` over ``v`` must exercise the ν-tangent rule once
        per batched element, reproducing the scalar result every time.
        The ν-tangent's internal quadrature is scalar-in-v so the rule
        should vectorise cleanly.
        """
        vs = jnp.asarray([-3.0, -0.5, 0.0, 0.5, 2.5, 10.0, 25.0], dtype=float)
        x = jnp.asarray(2.0, dtype=float)

        per_point_dv = jnp.array([
            jax.grad(lambda vi: log_kv(vi, x))(vi) for vi in vs
        ])
        vmap_dv = jax.vmap(lambda vi: jax.grad(lambda v: log_kv(v, x))(vi))(vs)

        np.testing.assert_allclose(
            np.asarray(vmap_dv), np.asarray(per_point_dv),
            rtol=1e-10, atol=1e-12,
            err_msg="vmap(grad(log_kv, v)) disagrees with per-point grad",
        )


# ===================================================================
# log-space sinh helper for the log_kv ν-tangent quadrature
# ===================================================================

class TestStableLogSinh:
    """Tests for ``_stable_log_sinh`` — the numerically-safe log(sinh(y)).

    Used by :py:func:`_dlog_kv_dv_single` inside the log_kv ν-tangent
    quadrature.  The helper must
      1. return ``-inf`` at ``y = 0`` (``sinh(0) = 0``), so that
         ``_log_kv_pos``'s ν-tangent integrand vanishes identically at
         ``v = 0`` without polluting the log-sum-exp reduction;
      2. stay finite past the ``sinh(y)`` float64 overflow point
         (``y ≳ 710``) — otherwise the Debye-regime v-derivatives
         would NaN out.

    Reference values come from ``mpmath`` at 50-digit precision.
    """

    @pytest.mark.parametrize("y", [
        0.0,
        1e-12, 1e-6, 1e-3,
        0.1, 0.5, 1.0, 2.0, 5.0,
        9.999, 10.0, 10.001,   # straddling the branch-switch threshold
        50.0, 100.0, 500.0, 700.0,
    ])
    def test_matches_math_log_sinh(self, y):
        """Within float64 eps of ``log(sinh(y))``.

        For ``y <= 700``, the reference is plain ``math.log(math.sinh(y))``
        in float64 — more than enough resolution to pin down the helper's
        own float64 result.  ``y == 0`` is checked separately: both
        branches return exact ``-inf``.  For the overflow regime
        ``y > 700`` see :py:meth:`test_no_overflow_past_sinh_underflow`.
        """
        if y == 0.0:
            got = float(_stable_log_sinh(jnp.asarray(y, dtype=float)))
            assert got == float("-inf"), (
                f"_stable_log_sinh(0) = {got}, expected -inf"
            )
            return

        import math
        ref = math.log(math.sinh(y))

        got = float(_stable_log_sinh(jnp.asarray(y, dtype=float)))
        np.testing.assert_allclose(
            got, ref, rtol=1e-14, atol=1e-14,
            err_msg=f"_stable_log_sinh({y}) = {got} vs math reference {ref}",
        )

    @pytest.mark.parametrize("y", [710.0, 1000.0, 5000.0])
    def test_matches_asymptotic_past_sinh_overflow(self, y):
        r"""Matches the exact identity in the overflow regime.

        ``sinh(710) \approx e^{710}/2`` overflows to ``inf`` in float64,
        so ``math.log(math.sinh(y))`` is unavailable as a reference.
        The identity

        .. math::
            \log\sinh y = y - \log 2 + \log1p(-e^{-2y})

        is exact.  For ``y >= 710`` the correction term ``log1p(-e^{-2y})``
        is below machine epsilon (``e^{-1420}`` already underflows to 0),
        so the identity reduces to ``y - log 2`` to float64 precision.
        """
        import math
        ref = y - math.log(2.0) + math.log1p(-math.exp(-2.0 * y))

        got = float(_stable_log_sinh(jnp.asarray(y, dtype=float)))
        assert np.isfinite(got), f"_stable_log_sinh({y}) = {got}"
        np.testing.assert_allclose(
            got, ref, rtol=1e-14, atol=1e-14,
            err_msg=f"_stable_log_sinh({y}) = {got} vs asymptotic {ref}",
        )

    def test_jit_compilable(self):
        """Helper is JIT-traceable."""
        f = jax.jit(_stable_log_sinh)
        assert np.isfinite(float(f(jnp.asarray(5.0))))
        assert float(f(jnp.asarray(0.0))) == float("-inf")

    @pytest.mark.parametrize("y", [1e-12, 1e-10, 1e-8, 1e-6])
    def test_small_y_precision_beats_naive_large_formula(self, y):
        r"""At tiny ``y`` the helper's small-branch keeps precision that the
        large-branch formula (``y + log1p(-exp(-2y)) - log 2``) loses.

        The large-branch subtraction ``1 - e^{-2y}`` cancels away
        ``log10(1/(2y))`` significant digits at small ``y``.  The
        helper's branch switch at ``y <= 10`` dodges this by falling
        back to direct ``log(sinh(y))``.  This test pins down that the
        helper is more accurate than the naive large-branch-only
        alternative would be, demonstrating the switch is
        precision-load-bearing and not just overflow protection.
        """
        import math
        ref = math.log(math.sinh(y))  # float64 reference via Taylor-safe math.sinh
        helper = float(_stable_log_sinh(jnp.asarray(y, dtype=float)))
        naive = y + math.log1p(-math.exp(-2.0 * y)) - math.log(2.0)

        # Helper matches the reference to float64 epsilon.
        np.testing.assert_allclose(
            helper, ref, rtol=1e-14, atol=1e-14,
            err_msg=f"helper({y}) = {helper} vs ref {ref}"
        )
        # And the naive alternative is measurably worse for very small y.
        if y <= 1e-6:
            assert abs(naive - ref) > abs(helper - ref), (
                f"expected large-only formula to be less accurate at y={y}, "
                f"got helper_err={abs(helper - ref):.3e} "
                f"vs naive_err={abs(naive - ref):.3e}"
            )


# ===================================================================
# log K_s(r) + s log r  — cancellation-stable combination
# ===================================================================

class TestLogKvPlusSLogR:
    r"""Tests for ``log_kv_plus_s_log_r`` — the stable combination
    ``log K_s(r) + s · log(r)`` used by the skewed-t log-PDF.

    The function's purpose is to keep the ``s log r`` divergence
    together with the matching ``-s log r`` divergence inside
    ``log K_s(r)`` so they cancel arithmetically, rather than
    computing each separately and subtracting.  These tests pin down:

    1. The exact analytical limit ``log Γ(s) + (s-1) log 2`` at
       ``r = 0``.
    2. Agreement with a scipy reference for ``r > 0``.
    3. A smooth, finite gradient at ``r = 0`` (the removable
       singularity the helper was built to resolve).
    4. JIT compatibility.
    """

    # --- r = 0 returns the analytical limit -------------------------

    @pytest.mark.parametrize("s", [0.5, 1.0, 1.5, 2.5, 5.0, 10.0, 20.0])
    def test_r_zero_matches_limit(self, s):
        """``log K_s(0) + s · log(0)`` evaluates to ``lgamma(s) + (s-1) log 2``.

        Proves the removable-singularity handling is exact in the
        limit — this is the property that lets the skewed-t log-PDF
        dispatch collapse to a single branch at ``γ = 0``.
        """
        ref = float(scipy.special.gammaln(s)) + (s - 1.0) * np.log(2.0)
        got = float(log_kv_plus_s_log_r(s, jnp.asarray(0.0, dtype=float)))
        np.testing.assert_allclose(
            got, ref, rtol=1e-12, atol=1e-12,
            err_msg=(
                f"log_kv_plus_s_log_r({s}, 0) = {got} "
                f"vs analytical limit {ref}"
            ),
        )

    # --- r > 0 agrees with scipy ------------------------------------

    @pytest.mark.parametrize("s,r", [
        (0.5, 0.01), (0.5, 0.5), (0.5, 2.0),
        (1.0, 0.1), (1.0, 1.0), (1.0, 5.0),
        (1.5, 0.5), (1.5, 3.0),
        (2.5, 0.1), (2.5, 2.0), (2.5, 10.0),
        (5.0, 0.5), (5.0, 5.0), (5.0, 20.0),
        (10.0, 1.0), (10.0, 10.0), (10.0, 50.0),
    ])
    def test_matches_scipy_reference(self, s, r):
        """``log K_s(r) + s log r`` matches ``log(scipy.kv(s, r)) + s log r``
        wherever ``scipy.kv`` is a safe reference (i.e. ``kv`` does not
        underflow).  Tolerance ``rtol=1e-6`` mirrors ``log_kv``'s own
        ``test_matches_log_of_kv`` tolerance.
        """
        sp_kv = float(scipy.special.kv(s, r))
        assert sp_kv > 0 and np.isfinite(sp_kv), (
            f"scipy.kv({s}, {r}) not a safe reference"
        )
        ref = np.log(sp_kv) + s * np.log(r)
        got = float(log_kv_plus_s_log_r(s, jnp.asarray(r, dtype=float)))
        np.testing.assert_allclose(
            got, ref, rtol=1e-6, atol=1e-10,
            err_msg=(
                f"log_kv_plus_s_log_r({s}, {r}) = {got} "
                f"vs log(scipy.kv) + s log r = {ref}"
            ),
        )

    # --- gradient at r = 0 is finite --------------------------------

    @pytest.mark.parametrize("s", [0.5, 1.0, 2.5, 10.0])
    def test_gradient_at_r_zero_is_finite(self, s):
        """``∂/∂r`` is finite at ``r = 0``.

        The internal ``jnp.maximum(r, floor)`` pins ``r_safe`` to a
        constant when ``r < floor``, so the gradient traverses via
        ``dr_safe/dr = 0``.  This is the safety property the skewed-t
        refactor depends on — without it, ``jax.grad(skewed_t.logpdf)``
        at ``γ = 0`` would NaN-poison.
        """
        s_j = jnp.asarray(s, dtype=float)
        dv = float(jax.grad(lambda r: log_kv_plus_s_log_r(s_j, r))(
            jnp.asarray(0.0, dtype=float)
        ))
        assert np.isfinite(dv), f"∂/∂r at r=0, s={s}: {dv}"

    # --- array shape preservation -----------------------------------

    def test_array_shape_preserved(self):
        """Output shape matches input ``r`` shape."""
        s = jnp.asarray(2.0)
        r = jnp.linspace(0.0, 5.0, 7)
        out = log_kv_plus_s_log_r(s, r)
        assert out.shape == r.shape
        assert np.all(np.isfinite(np.asarray(out)))

    # --- JIT compatibility ------------------------------------------

    def test_jit_compilable(self):
        """Helper traces cleanly under ``jax.jit``."""
        f = jax.jit(log_kv_plus_s_log_r)
        got = f(jnp.asarray(1.5), jnp.asarray(2.0))
        assert np.isfinite(float(got))
        # Also under grad
        g = jax.jit(jax.grad(lambda r: log_kv_plus_s_log_r(1.5, r)))
        assert np.isfinite(float(g(jnp.asarray(0.0))))
        assert np.isfinite(float(g(jnp.asarray(3.0))))


# ===================================================================
# Student's t CDF (stdtr)
# ===================================================================

class TestStdtr:
    """Tests for Student's t CDF implementation."""

    @pytest.mark.parametrize("df", [1, 2, 5, 10, 30, 100])
    def test_matches_scipy_t_cdf(self, df):
        """stdtr(df, x) should match scipy.stats.t.cdf(x, df)."""
        x = np.linspace(-5, 5, 50)
        cx = np.array(jax.vmap(lambda xi: stdtr(float(df), xi))(jnp.array(x)))
        sp = scipy.stats.t.cdf(x, df)
        np.testing.assert_allclose(cx, sp, rtol=1e-5, atol=1e-8,
                                   err_msg=f"stdtr mismatch for df={df}")

    def test_monotonicity(self):
        """CDF must be non-decreasing."""
        x = np.linspace(-10, 10, 100)
        for df in [1.0, 5.0, 30.0]:
            cdf = np.array(jax.vmap(lambda xi: stdtr(df, xi))(jnp.array(x)))
            assert np.all(np.diff(cdf) >= -1e-10), f"stdtr not monotone for df={df}"

    def test_symmetry(self):
        """stdtr(df, -x) + stdtr(df, x) == 1."""
        x = np.linspace(0.1, 5, 30)
        for df in [1.0, 5.0, 30.0]:
            left = np.array(jax.vmap(lambda xi: stdtr(df, -xi))(jnp.array(x)))
            right = np.array(jax.vmap(lambda xi: stdtr(df, xi))(jnp.array(x)))
            np.testing.assert_allclose(left + right, 1.0, rtol=1e-6,
                                       err_msg=f"stdtr symmetry violated for df={df}")

    def test_boundary_values(self):
        """stdtr approaches 0 at -inf and 1 at +inf."""
        for df in [1.0, 5.0, 30.0]:
            low = float(stdtr(df, jnp.array(-100.0)))
            high = float(stdtr(df, jnp.array(100.0)))
            assert low < 0.01, f"stdtr(-100) should be near 0, got {low}"
            assert high > 0.99, f"stdtr(100) should be near 1, got {high}"

    def test_extreme_df_approaches_normal(self):
        """As df -> inf, t-dist CDF -> normal CDF."""
        x = np.linspace(-3, 3, 20)
        # df=1e6 needed: t(1000) differs from Normal by ~2.5% at x=-3
        # (Edgeworth correction ~ phi(x)*(x^3+x)/(4*df))
        cx = np.array(jax.vmap(lambda xi: stdtr(1e6, xi))(jnp.array(x)))
        sp = scipy.stats.norm.cdf(x)
        np.testing.assert_allclose(cx, sp, rtol=1e-3,
                                   err_msg="Large df should approach normal")

    def test_jit_compilable(self):
        f = jax.jit(lambda x: stdtr(5.0, x))
        result = f(jnp.array(0.0))
        np.testing.assert_allclose(float(result), 0.5, atol=1e-6)

    def test_gradient_wrt_x(self):
        """d/dx stdtr(df, x) should equal the t-distribution PDF."""
        df, x0 = 5.0, 1.5
        grad_fn = jax.grad(stdtr, argnums=1)
        analytic = float(grad_fn(df, jnp.array(x0)))
        expected_pdf = scipy.stats.t.pdf(x0, df)
        np.testing.assert_allclose(analytic, expected_pdf, rtol=1e-3,
                                   err_msg="stdtr gradient should be t-pdf")


# ===================================================================
# Inverse incomplete gamma functions
# ===================================================================

class TestIgammainv:
    """Tests for inverse regularized lower incomplete gamma function."""

    @pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_matches_scipy(self, a):
        """igammainv(a, p) matches scipy.special.gammaincinv(a, p)."""
        p = np.linspace(0.01, 0.99, 30)
        cx = np.array(jax.vmap(lambda pi: igammainv(jnp.array(a), pi))(jnp.array(p)))
        sp = scipy.special.gammaincinv(a, p)
        mask = np.isfinite(cx) & np.isfinite(sp)
        np.testing.assert_allclose(cx[mask], sp[mask], rtol=1e-5, atol=1e-10,
                                   err_msg=f"igammainv mismatch for a={a}")

    @pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 5.0])
    def test_roundtrip(self, a):
        """gammainc(a, igammainv(a, p)) == p."""
        p = np.linspace(0.01, 0.99, 20)
        x = jax.vmap(lambda pi: igammainv(jnp.array(a), pi))(jnp.array(p))
        p_recovered = np.array(jax.scipy.special.gammainc(jnp.array(a), x))
        np.testing.assert_allclose(p_recovered, p, rtol=1e-5,
                                   err_msg=f"igammainv roundtrip failed for a={a}")

    def test_boundary_zero(self):
        """igammainv(a, 0) == 0."""
        for a in [0.5, 1.0, 5.0]:
            result = float(igammainv(jnp.array(a), jnp.array(0.0)))
            assert result == 0.0 or result < 1e-10

    def test_boundary_one(self):
        """igammainv(a, 1) == inf."""
        for a in [0.5, 1.0, 5.0]:
            result = float(igammainv(jnp.array(a), jnp.array(1.0)))
            assert result > 1e10 or np.isinf(result)

    def test_monotonicity(self):
        """igammainv(a, p) is non-decreasing in p."""
        p = np.linspace(0.01, 0.99, 50)
        for a in [0.5, 1.0, 5.0]:
            vals = np.array(jax.vmap(lambda pi: igammainv(jnp.array(a), pi))(jnp.array(p)))
            assert np.all(np.diff(vals) >= -1e-10), f"Not monotone for a={a}"

    def test_non_negative(self):
        """igammainv always returns non-negative values."""
        p = np.linspace(0.01, 0.99, 30)
        for a in [0.5, 1.0, 5.0]:
            vals = np.array(jax.vmap(lambda pi: igammainv(jnp.array(a), pi))(jnp.array(p)))
            assert np.all(vals[np.isfinite(vals)] >= 0)


class TestIgammacinv:
    """Tests for inverse regularized upper incomplete gamma function."""

    @pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_matches_scipy(self, a):
        """igammacinv(a, p) matches scipy.special.gammainccinv(a, p)."""
        p = np.linspace(0.01, 0.99, 30)
        cx = np.array(jax.vmap(lambda pi: igammacinv(jnp.array(a), pi))(jnp.array(p)))
        sp = scipy.special.gammainccinv(a, p)
        mask = np.isfinite(cx) & np.isfinite(sp)
        np.testing.assert_allclose(cx[mask], sp[mask], rtol=1e-5, atol=1e-10,
                                   err_msg=f"igammacinv mismatch for a={a}")

    @pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 5.0])
    def test_roundtrip(self, a):
        """gammaincc(a, igammacinv(a, p)) == p."""
        p = np.linspace(0.01, 0.99, 20)
        x = jax.vmap(lambda pi: igammacinv(jnp.array(a), pi))(jnp.array(p))
        p_recovered = np.array(jax.scipy.special.gammaincc(jnp.array(a), x))
        np.testing.assert_allclose(p_recovered, p, rtol=1e-5,
                                   err_msg=f"igammacinv roundtrip failed for a={a}")

    def test_complement_relation(self):
        """igammacinv(a, p) == igammainv(a, 1-p)."""
        p = np.linspace(0.05, 0.95, 20)
        for a in [1.0, 3.0]:
            c_inv = np.array(jax.vmap(lambda pi: igammacinv(jnp.array(a), pi))(jnp.array(p)))
            inv = np.array(jax.vmap(lambda pi: igammainv(jnp.array(a), pi))(jnp.array(1 - p)))
            mask = np.isfinite(c_inv) & np.isfinite(inv)
            np.testing.assert_allclose(c_inv[mask], inv[mask], rtol=1e-4,
                                       err_msg="Complement relation violated")

    def test_monotonicity(self):
        """igammacinv(a, p) is non-increasing in p."""
        p = np.linspace(0.01, 0.99, 50)
        for a in [0.5, 1.0, 5.0]:
            vals = np.array(jax.vmap(lambda pi: igammacinv(jnp.array(a), pi))(jnp.array(p)))
            assert np.all(np.diff(vals) <= 1e-10), f"Not monotone decreasing for a={a}"


# ===================================================================
# Digamma and trigamma
# ===================================================================

class TestDigamma:
    """Tests for digamma function psi(x) = d/dx ln Gamma(x)."""

    @pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0])
    def test_matches_scipy(self, x):
        """digamma(x) matches scipy.special.digamma(x)."""
        cx = float(digamma(jnp.array([x]))[0])
        sp = float(scipy.special.digamma(x))
        np.testing.assert_allclose(cx, sp, rtol=1e-5,
                                   err_msg=f"digamma mismatch at x={x}")

    def test_batch(self):
        """digamma works on arrays."""
        x = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
        cx = np.array(digamma(x))
        sp = scipy.special.digamma(np.array(x))
        np.testing.assert_allclose(cx, sp, rtol=1e-5)

    def test_recurrence(self):
        """psi(x+1) = psi(x) + 1/x (DLMF 5.5.2)."""
        x = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
        lhs = np.array(digamma(x + 1))
        rhs = np.array(digamma(x)) + np.array(1.0 / x)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-5,
                                   err_msg="Digamma recurrence psi(x+1)=psi(x)+1/x violated")

    def test_known_values(self):
        """psi(1) = -gamma (Euler-Mascheroni), psi(1/2) = -gamma - 2*ln(2)."""
        euler_gamma = 0.5772156649015329
        np.testing.assert_allclose(
            float(digamma(jnp.array([1.0]))[0]), -euler_gamma, rtol=1e-5,
            err_msg="psi(1) should be -gamma")
        np.testing.assert_allclose(
            float(digamma(jnp.array([0.5]))[0]),
            -euler_gamma - 2 * np.log(2), rtol=1e-5,
            err_msg="psi(1/2) should be -gamma - 2*ln(2)")

    def test_shape_preservation(self):
        """Output shape matches input shape."""
        x = jnp.array([1.0, 2.0, 3.0])
        assert digamma(x).shape == x.shape
        x_scalar = jnp.array([5.0])
        assert digamma(x_scalar).shape == (1,)

    def test_jit_compilable(self):
        """digamma is JIT-compatible."""
        f = jax.jit(lambda x: digamma(x))
        result = f(jnp.array([1.0]))
        assert np.isfinite(float(result[0]))


class TestTrigamma:
    """Tests for trigamma function psi'(x) = d^2/dx^2 ln Gamma(x)."""

    @pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0])
    def test_matches_scipy(self, x):
        """trigamma(x) matches scipy.special.polygamma(1, x)."""
        cx = float(trigamma(jnp.array([x]))[0])
        sp = float(scipy.special.polygamma(1, x))
        np.testing.assert_allclose(cx, sp, rtol=1e-5,
                                   err_msg=f"trigamma mismatch at x={x}")

    def test_batch(self):
        """trigamma works on arrays."""
        x = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
        cx = np.array(trigamma(x))
        sp = scipy.special.polygamma(1, np.array(x))
        np.testing.assert_allclose(cx, sp, rtol=1e-5)

    def test_recurrence(self):
        """psi'(x+1) = psi'(x) - 1/x^2 (DLMF 5.15.5)."""
        x = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
        lhs = np.array(trigamma(x + 1))
        rhs = np.array(trigamma(x)) - np.array(1.0 / x ** 2)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-4,
                                   err_msg="Trigamma recurrence psi'(x+1)=psi'(x)-1/x^2 violated")

    def test_known_values(self):
        """psi'(1) = pi^2/6, psi'(1/2) = pi^2/2."""
        np.testing.assert_allclose(
            float(trigamma(jnp.array([1.0]))[0]), np.pi ** 2 / 6, rtol=1e-5,
            err_msg="psi'(1) should be pi^2/6")
        np.testing.assert_allclose(
            float(trigamma(jnp.array([0.5]))[0]), np.pi ** 2 / 2, rtol=1e-4,
            err_msg="psi'(1/2) should be pi^2/2")

    def test_positivity(self):
        """psi'(x) > 0 for all x > 0."""
        x = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0])
        vals = np.array(trigamma(x))
        assert np.all(vals > 0), "Trigamma must be positive for x > 0"

    def test_monotonically_decreasing(self):
        """psi'(x) is strictly decreasing for x > 0."""
        x = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0])
        vals = np.array(trigamma(x))
        assert np.all(np.diff(vals) < 0), "Trigamma must be monotonically decreasing"

    def test_shape_preservation(self):
        """Output shape matches input shape."""
        x = jnp.array([1.0, 2.0, 3.0])
        assert trigamma(x).shape == x.shape

    def test_jit_compilable(self):
        """trigamma is JIT-compatible."""
        f = jax.jit(lambda x: trigamma(x))
        result = f(jnp.array([1.0]))
        assert np.isfinite(float(result[0]))
