"""Rigorous tests for copulax._src.special: Bessel K_v, igammainv, stdtr.

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

from copulax._src.special import kv, igammainv, igammacinv, stdtr


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
        cx = np.array(jax.vmap(lambda xi: stdtr(1000.0, xi))(jnp.array(x)))
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
