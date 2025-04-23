"""Helper functions for testing."""
from jax import grad, jit
from jax import numpy as jnp
import numpy as np
import warnings

from copulax._src.typing import Scalar


@jit
def jittable(dist):
    return dist.example_params()


def correct_uvt_shape(x, output, dist, method):
    """Check if the shape of the univariate output array is correct"""
    assert isinstance(output, jnp.ndarray), f"{method} output is not a JAX array for {dist}"
    assert output.size == output.size, f"{method} size mismatch for {dist}"
    assert output.shape == output.shape, f"{method} shape mismatch for {dist}"
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


# def gradients(func, s, data, params, params_error: bool = True, **kwargs):
#     """Calculate the gradients of the output."""
#     new_func = lambda x, p: func(x, params=p, **kwargs).sum()
#     x_grad, params_grad = grad(new_func, argnums=[0, 1])(data, params)
#     params_grad = tuple(params_grad.values())
#     assert no_nans(x_grad), f"{s} gradient contains NaNs for data argument"
#     assert is_finite(x_grad), f"{s} gradient contains non-finite values for data argument"

#     params_nans_res = no_nans(params_grad), f"{s} gradient contains NaNs for params argument"
#     params_finite_res = is_finite(params_grad), f"{s} gradient contains non-finite values for params argument"
#     if params_error:
#         assert params_nans_res[0], params_nans_res[1]
#         assert params_finite_res[0], params_finite_res[1]
#     else:
#         if not params_nans_res[0]:
#             warnings.warn(params_nans_res[1])
#         elif not params_finite_res[0]:
#             warnings.warn(params_finite_res[1])


def gradients(func, s, data, params, params_error: bool = True, **kwargs):
    """Calculate the gradients of the output."""
    new_func = lambda x, p: func(x, params=p, **kwargs).sum()
    x_grad = grad(new_func, argnums=[0,])(data, params)
    # params_grad = tuple(params_grad.values())
    assert no_nans(x_grad), f"{s} gradient contains NaNs for data argument"
    assert is_finite(x_grad), f"{s} gradient contains non-finite values for data argument"

    # params_nans_res = no_nans(params_grad), f"{s} gradient contains NaNs for params argument"
    # params_finite_res = is_finite(params_grad), f"{s} gradient contains non-finite values for params argument"
    # if params_error:
    #     assert params_nans_res[0], params_nans_res[1]
    #     assert params_finite_res[0], params_finite_res[1]
    # else:
    #     if not params_nans_res[0]:
    #         warnings.warn(params_nans_res[1])
    #     elif not params_finite_res[0]:
    #         warnings.warn(params_finite_res[1])


def is_scalar(output):
    """Check if the output is a scalar."""
    return np.asarray(output).flatten().size == 1


def check_metric_output(dist, output, metric_name):
    assert isinstance(output, Scalar) and output.shape == () and output.size == 1, f"{dist} {metric_name} is non-scalar."
    assert no_nans(output), f"{dist} {metric_name} contains NaNs."
    # assert is_finite(output), f"{dist} {metric_name} contains non-finite values."