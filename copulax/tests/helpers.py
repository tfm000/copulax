"""Helper functions for testing."""
from jax import grad, jit
import numpy as np


def correct_mvt_shape(x, output):
    """Check if the shape of the output array is correct."""
    expected_shape: tuple = (x.shape[0], 1)
    return output.shape == expected_shape


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


def gradients(func, s, data):
    """Calculate the gradients of the output."""
    new_func = lambda x: func(x).sum()
    grad_output = grad(new_func)(data)
    assert no_nans(grad_output), f"{s} gradient contains NaNs"
    assert is_finite(grad_output), f"{s} gradient contains non-finite values"

def is_scalar(output):
    """Check if the output is a scalar."""
    return np.asarray(output).flatten().size == 1