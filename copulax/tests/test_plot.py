"""Smoke tests for plot methods — verify they don't raise.

Uses matplotlib's non-interactive (Agg) backend to avoid GUI pop-ups.
Marked as ``slow`` because plot() internally calls cdf() and ppf(),
both of which require JIT compilation per distribution.
"""
import pytest
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

from copulax.univariate import (
    gamma, ig, lognormal, normal, skewed_t, student_t, uniform,
)

# GH and GIG are excluded: their PPF (used internally by plot) requires
# minutes of JIT compilation, making them impractical for routine CI.
UVT_DISTS = [gamma, ig, lognormal, normal, skewed_t, student_t, uniform]


@pytest.mark.slow
@pytest.mark.parametrize("dist", UVT_DISTS)
def test_plot_no_sample(dist):
    """plot() without sample data should not raise."""
    params = dist.example_params()
    support = dist.support(params)
    # Use explicit domain to avoid slow PPF calls
    lo = max(float(support[0]), -10.0)
    hi = min(float(support[1]), 10.0)
    if float(support[0]) >= 0:
        lo = max(lo, 0.01)
    try:
        dist.plot(params=params, show=False, num_points=10,
                  domain=(lo, hi))
    finally:
        plt.close('all')


@pytest.mark.slow
@pytest.mark.parametrize("dist", UVT_DISTS)
def test_plot_with_sample(dist):
    """plot() with sample data should not raise."""
    import jax
    params = dist.example_params()
    support = dist.support(params)
    lo = max(float(support[0]), -10.0)
    hi = min(float(support[1]), 10.0)
    if float(support[0]) >= 0:
        lo = max(lo, 0.01)
    key = jax.random.PRNGKey(0)
    sample = dist.rvs(size=(50,), params=params, key=key)
    try:
        dist.plot(params=params, sample=sample, show=False, 
                  num_points=10, bins=10, domain=(lo, hi))
    finally:
        plt.close('all')
