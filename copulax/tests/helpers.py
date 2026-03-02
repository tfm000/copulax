"""Helper functions for testing."""
from jax import grad, jit, lax
from jax import numpy as jnp
import jax.tree_util as jtu
import numpy as np
import warnings

from copulax._src.typing import Scalar


@jit
def jittable(dist):
    return dist.example_params()


def check_uvt_params(params, s):
    assert isinstance(params, dict), f"{s} params is not a dict"
    assert len(params) > 0, f"{s} is empty"
    assert all(isinstance(k, str) for k in params.keys()), f"{s} params keys are not strings"
    assert all(isinstance(v, Scalar) for v in params.values()), f"{s} params values are not scalars"
    assert all(v.ndim == 0 and v.shape == () and v.size == 1 for v in params.values()), f"{s} params values are not scalars"
    assert any(jnp.isnan(v) for v in params.values()) == False, f"{s} params values are NaN"
    assert all(jnp.isfinite(v) for v in params.values()) == True, f"{s} params values are not finite"


def check_mvt_params(params, s):
    assert isinstance(params, dict), f"{s} params is not a dict"
    assert len(params) > 0, f"{s} is empty"
    assert all(isinstance(k, str) for k in params.keys()), f"{s} params keys are not strings"
    assert all(isinstance(v, jnp.ndarray) for v in params.values()), f"{s} params values are not arrays"
    assert all((v.ndim == 0 and v.shape == () and v.size == 1) or (v.ndim == 2 and v.shape == (v.size, 1) and v.size > 1) or (v.ndim == 2 and v.shape == (int(v.size ** 0.5), int(v.size ** 0.5)) and v.size > 1) for v in params.values()), f"{s} params values are not scalars, 1D-vectors or 2D-square matrices"
    assert any(jnp.any(jnp.isnan(v)) for v in params.values()) == False, f"{s} params values are NaN"
    assert all(jnp.all(jnp.isfinite(v)) for v in params.values()) == True, f"{s} params values are not finite"


def correct_uvt_shape(x, output, dist, method):
    """Check if the shape of the univariate output array is correct"""
    assert isinstance(output, jnp.ndarray), f"{method} output is not a JAX array for {dist}"
    assert output.size == x.size, f"{method} size mismatch for {dist}"
    assert output.shape == x.shape, f"{method} shape mismatch for {dist}"
    assert output.ndim == 1, f"{method} is not 1D for {dist}"


def correct_mvt_shape(x, output, dist, method):
    """Check if the shape of the multivariate output array is correct."""
    assert isinstance(output, jnp.ndarray), f"{method} output is not a JAX array for {dist}"
    expected_shape: tuple = (x.shape[0], 1)
    assert output.shape == expected_shape, f"{method} shape mismatch for {dist}"


def two_dim(output):
    """Check if the input array is two-dimensional."""
    return output.ndim == 2


def is_positive(output):
    """Check if the output array is positive."""
    return np.all(output >= 0)


def no_nans(output):
    """Check if the output array has no NaNs."""
    return not np.any(np.isnan(output))


def is_finite(output):
    """Check if the output array contains only finite values."""
    return np.all(np.isfinite(output))


def gradients(func, s, data, params, params_error: bool = True, **kwargs):
    """Calculate the gradients of the output.
    
    When params_error=False, uses stop_gradient on params to safely
    compute data gradients (avoids e.g. betainc ValueError), then
    attempts params gradient separately.
    """
    new_func = lambda x, p: func(x, params=p, **kwargs).sum()

    if params_error:
        x_grad, params_grad = grad(new_func, argnums=[0, 1])(data, params)
        params_grad_leaves = jtu.tree_leaves(params_grad)
        assert no_nans(x_grad), f"{s} gradient contains NaNs for data argument"
        assert is_finite(x_grad), f"{s} gradient contains non-finite values for data argument"
        assert all(no_nans(l) for l in params_grad_leaves), f"{s} gradient contains NaNs for params argument"
        assert all(is_finite(l) for l in params_grad_leaves), f"{s} gradient contains non-finite values for params argument"
    else:
        # Use stop_gradient on params so data gradient doesn't
        # trigger unsupported param gradients (e.g. betainc a, b).
        sg_func = lambda x, p: func(
            x, params=lax.stop_gradient(p), **kwargs
        ).sum()
        x_grad = grad(sg_func, argnums=0)(data, params)
        assert no_nans(x_grad), f"{s} gradient contains NaNs for data argument"
        assert is_finite(x_grad), f"{s} gradient contains non-finite values for data argument"

        try:
            params_grad = grad(new_func, argnums=1)(data, params)
            params_grad_leaves = jtu.tree_leaves(params_grad)
            if not all(no_nans(l) for l in params_grad_leaves):
                warnings.warn(f"{s} gradient contains NaNs for params argument")
            elif not all(is_finite(l) for l in params_grad_leaves):
                warnings.warn(f"{s} gradient contains non-finite values for params argument")
        except (ValueError, Exception) as e:
            warnings.warn(f"{s} params gradient could not be computed: {e}")


def is_scalar(output):
    """Check if the output is a scalar."""
    return np.asarray(output).flatten().size == 1


def check_metric_output(dist, output, metric_name):
    assert isinstance(output, Scalar) and output.shape == () and output.size == 1, f"{dist} {metric_name} is non-scalar."
    assert no_nans(output), f"{dist} {metric_name} contains NaNs."
    # assert is_finite(output), f"{dist} {metric_name} contains non-finite values."