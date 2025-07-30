import numpy as np
import jax
import jax.numpy as jnp
import scipy.stats
import scipy.special

from copulax.special import kv, stdtr
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

            gradients(func=func, s="stdtr", data=x, params=df)
