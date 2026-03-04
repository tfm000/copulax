import numpy as np
import jax
import jax.numpy as jnp
import scipy.stats
import scipy.special

from copulax.special import kv, stdtr, igammainv, igammacinv
from copulax.tests.helpers import *


def test_kv():
    # checking values and gradients
    kv_vg = jax.jit(jax.value_and_grad(kv, argnums=(0, 1)))
    for v in np.linspace(-10, 10, 100):
        for x in np.linspace(1e-10, 10, 100):
            val, grad = kv_vg(v, x)

            assert val >= 0, f"kv not positive for v={v}, x={x}"
            assert not np.isnan(val), f"kv contains NaNs for v={v}, x={x}"
            assert (not np.isnan(grad[0])) and (
                not np.isnan(grad[1])
            ), f"kv gradient contains NaNs for v={v}, x={x}"

    # checking shape
    xs = [
        jnp.array([[1.0, 2.0, 3.0], [2.3, 3.4, 5.5]]),
        jnp.array([1.0, 2.0, 3.0]),
        jnp.array(1.0),
    ]
    for x in xs:
        val = kv(0.5, x)
        assert val.shape == x.shape, f"kv shape mismatch for x={x}"


def test_stdtr():
    """comparing the output of stdtr with scipy.special.stdtr"""
    stdtr_jit = jax.jit(stdtr)
    func = lambda x, params: stdtr_jit(params, x)
    for df in [1, 2, 5, 100]:
        for size in [(1,), (1, 1), (2, 2), (10, 5)]:
            x = scipy.stats.t.rvs(df, size=size)
            copulax_val = stdtr_jit(df, x)
            scipy_val = scipy.special.stdtr(df, x)

            # checking properties
            assert isinstance(
                copulax_val, jnp.ndarray
            ), f"copulax stdtr output is not a JAX array for df={df}, size={size}"
            assert (
                copulax_val.shape == x.shape
            ), f"stdtr shape mismatch for df={df}, size={size}"
            assert no_nans(copulax_val), f"copulax stdtr contains NaNs for df={df}"
            assert is_positive(
                copulax_val
            ), f"copulax stdtr is not positive for df={df}"
            assert is_finite(
                copulax_val
            ), f"copulax stdtr contains non-finite values for df={df}"
            assert np.all(0 <= copulax_val) and np.all(
                copulax_val <= 1
            ), f"copulax stdtr is not in [0, 1] for df={df}"
            assert np.allclose(
                copulax_val, scipy_val, atol=1e-4, rtol=1e-4
            ), f"copulax stdtr does not match scipy for df={df}"

            gradients(
                func=func, s="stdtr", data=x, params=jnp.float32(df), params_error=False
            )


class TestIgammainv:
    """Tests for igammainv comparing against scipy.special.gammaincinv."""

    _a_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    _p_interior = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    def test_matches_scipy(self):
        """Interior values match scipy.special.gammaincinv."""
        for a in self._a_values:
            for p in self._p_interior:
                ours = float(igammainv(a, p))
                ref = float(scipy.special.gammaincinv(a, p))
                assert np.allclose(
                    ours, ref, rtol=1e-3, atol=1e-6
                ), f"igammainv({a}, {p}): got {ours}, expected {ref}"

    def test_roundtrip(self):
        """igammainv inverts jax.scipy.special.gammainc."""
        for a in self._a_values:
            for p in self._p_interior:
                x = igammainv(a, p)
                p_rt = float(jax.scipy.special.gammainc(a, x))
                assert np.allclose(
                    p_rt, p, rtol=1e-3, atol=1e-6
                ), f"roundtrip failed for a={a}, p={p}: got {p_rt}"

    def test_boundary_zero(self):
        """igammainv(a, 0) == 0 for all a."""
        for a in self._a_values:
            assert float(igammainv(a, 0.0)) == 0.0, f"igammainv({a}, 0) != 0"

    def test_boundary_one(self):
        """igammainv(a, 1) == inf for all a."""
        for a in self._a_values:
            assert jnp.isinf(igammainv(a, 1.0)), f"igammainv({a}, 1) != inf"

    def test_monotonic(self):
        """igammainv(a, p) is monotonically increasing in p."""
        p_vals = jnp.array(self._p_interior)
        for a in self._a_values:
            x_vals = jnp.array([igammainv(a, float(p)) for p in p_vals])
            assert jnp.all(jnp.diff(x_vals) > 0), f"igammainv not monotonic for a={a}"

    def test_array_input(self):
        """Accepts and broadcasts array inputs."""
        a = jnp.array([1.0, 2.0, 5.0])
        p = jnp.array([0.1, 0.5, 0.9])
        result = igammainv(a, p)
        assert result.shape == (3,), f"shape mismatch: {result.shape}"
        ref = np.array(
            [scipy.special.gammaincinv(float(ai), float(pi)) for ai, pi in zip(a, p)]
        )
        assert np.allclose(result, ref, rtol=1e-3, atol=1e-6)

    def test_scalar_input(self):
        """Works with scalar inputs."""
        result = igammainv(2.0, 0.5)
        assert result.shape == (), f"scalar shape mismatch: {result.shape}"
        ref = scipy.special.gammaincinv(2.0, 0.5)
        assert np.allclose(float(result), ref, rtol=1e-3, atol=1e-6)

    def test_positive_output(self):
        """Output is non-negative for valid inputs."""
        for a in self._a_values:
            for p in self._p_interior:
                assert float(igammainv(a, p)) >= 0.0


class TestIgammacinv:
    """Tests for igammacinv comparing against scipy.special.gammainccinv."""

    _a_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    _p_interior = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    def test_matches_scipy(self):
        """Interior values match scipy.special.gammainccinv."""
        for a in self._a_values:
            for p in self._p_interior:
                ours = float(igammacinv(a, p))
                ref = float(scipy.special.gammainccinv(a, p))
                assert np.allclose(
                    ours, ref, rtol=1e-3, atol=1e-6
                ), f"igammacinv({a}, {p}): got {ours}, expected {ref}"

    def test_roundtrip(self):
        """igammacinv inverts jax.scipy.special.gammaincc."""
        for a in self._a_values:
            for p in self._p_interior:
                x = igammacinv(a, p)
                p_rt = float(jax.scipy.special.gammaincc(a, x))
                assert np.allclose(
                    p_rt, p, rtol=1e-3, atol=1e-6
                ), f"roundtrip failed for a={a}, p={p}: got {p_rt}"

    def test_boundary_zero(self):
        """igammacinv(a, 0) == inf for all a."""
        for a in self._a_values:
            assert jnp.isinf(igammacinv(a, 0.0)), f"igammacinv({a}, 0) != inf"

    def test_boundary_one(self):
        """igammacinv(a, 1) == 0 for all a."""
        for a in self._a_values:
            assert float(igammacinv(a, 1.0)) == 0.0, f"igammacinv({a}, 1) != 0"

    def test_monotonic(self):
        """igammacinv(a, p) is monotonically decreasing in p."""
        p_vals = jnp.array(self._p_interior)
        for a in self._a_values:
            x_vals = jnp.array([igammacinv(a, float(p)) for p in p_vals])
            assert jnp.all(
                jnp.diff(x_vals) < 0
            ), f"igammacinv not monotonically decreasing for a={a}"

    def test_array_input(self):
        """Accepts and broadcasts array inputs."""
        a = jnp.array([1.0, 2.0, 5.0])
        p = jnp.array([0.1, 0.5, 0.9])
        result = igammacinv(a, p)
        assert result.shape == (3,), f"shape mismatch: {result.shape}"
        ref = np.array(
            [scipy.special.gammainccinv(float(ai), float(pi)) for ai, pi in zip(a, p)]
        )
        assert np.allclose(result, ref, rtol=1e-3, atol=1e-6)

    def test_scalar_input(self):
        """Works with scalar inputs."""
        result = igammacinv(2.0, 0.5)
        assert result.shape == (), f"scalar shape mismatch: {result.shape}"
        ref = scipy.special.gammainccinv(2.0, 0.5)
        assert np.allclose(float(result), ref, rtol=1e-3, atol=1e-6)

    def test_complement_relation(self):
        """igammacinv(a, p) == igammainv(a, 1 - p)."""
        for a in self._a_values:
            for p in self._p_interior:
                inv = float(igammainv(a, 1.0 - p))
                cinv = float(igammacinv(a, p))
                assert np.allclose(inv, cinv, rtol=1e-6, atol=1e-10), (
                    f"complement relation failed for a={a}, p={p}: "
                    f"igammainv(a,1-p)={inv}, igammacinv(a,p)={cinv}"
                )

    def test_positive_output(self):
        """Output is non-negative for valid inputs."""
        for a in self._a_values:
            for p in self._p_interior:
                assert float(igammacinv(a, p)) >= 0.0
