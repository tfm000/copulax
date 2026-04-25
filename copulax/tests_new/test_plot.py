"""Smoke tests for the plotting module.

Verifies that plot() runs without errors for representative distributions.
Uses matplotlib's non-interactive Agg backend to avoid display issues in CI.
"""

import matplotlib
matplotlib.use("Agg")

import jax.numpy as jnp
import numpy as np
import pytest

from copulax.univariate import normal, student_t, gamma


class TestUnivariatePlot:
    """Smoke tests for Univariate.plot()."""

    def test_normal_plot_runs(self):
        """normal.plot() should complete without error."""
        params = {"mu": 0.0, "sigma": 1.0}
        normal.plot(params=params, show=False)

    def test_student_t_plot_runs(self):
        """student_t.plot() should complete without error."""
        params = {"nu": 5.0, "mu": 0.0, "sigma": 1.0}
        student_t.plot(params=params, show=False)

    def test_gamma_plot_runs(self):
        """gamma.plot() should complete without error."""
        params = {"alpha": 2.0, "beta": 1.0}
        gamma.plot(params=params, show=False)

    def test_plot_with_sample(self):
        """plot() with a sample overlay should not error."""
        np.random.seed(42)
        sample = jnp.array(np.random.normal(0, 1, 200))
        params = {"mu": 0.0, "sigma": 1.0}
        normal.plot(params=params, sample=sample, show=False)

    def test_plot_with_custom_domain(self):
        """plot() with explicit domain should not error."""
        params = {"mu": 0.0, "sigma": 1.0}
        normal.plot(params=params, domain=(-5.0, 5.0), show=False)
