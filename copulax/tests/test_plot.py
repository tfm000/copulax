"""Smoke tests for the plotting module.

Verifies that plot() runs without errors for representative distributions.
Uses matplotlib's non-interactive Agg backend to avoid display issues in CI.
"""

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import numpy as np
import pytest

from copulax.univariate import gamma, normal, student_t


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
        rng = np.random.RandomState(42)
        sample = jnp.array(rng.normal(0, 1, 200))
        params = {"mu": 0.0, "sigma": 1.0}
        normal.plot(params=params, sample=sample, show=False)

    def test_plot_with_custom_domain(self):
        """plot() with explicit domain should not error."""
        params = {"mu": 0.0, "sigma": 1.0}
        normal.plot(params=params, domain=(-5.0, 5.0), show=False)


class TestUnivariatePlotDomainDiscoveryBounded:
    """WR-07: the ``plot()`` domain-discovery ``while not isfinite`` loops
    must be bounded.

    Under the library's NaN failure-signalling convention, a degenerate
    fit yields NaN parameters; ``support`` is then ``(-inf, inf)`` and
    ``ppf`` returns NaN for every probed quantile, so the unbounded
    ``eps += delta`` loops never terminate and ``plot()`` hangs.  The
    bounded loops must instead raise an informative ``ValueError`` (in the
    ``_require_fitted`` wording style) telling the user the parameters may
    be invalid and to pass an explicit ``domain``.

    These assertions are timeout-free: after the fix the raise happens
    immediately (the cap is hit within a fixed, small number of
    iterations), so a plain ``pytest.raises`` both proves the fix and can
    never hang the suite once the bound is in place.
    """

    def test_nan_params_raise_valueerror_not_hang(self):
        """``plot()`` with NaN parameters raises ``ValueError`` (bounded
        domain discovery) rather than looping forever."""
        nan_params = {"mu": jnp.nan, "sigma": jnp.nan}
        with pytest.raises(ValueError):
            normal.plot(params=nan_params, show=False)

    def test_nan_params_message_mentions_domain(self):
        """The raised error is informative: it names the invalid-parameter
        cause and the explicit-``domain`` escape hatch."""
        nan_params = {"mu": jnp.nan, "sigma": jnp.nan}
        with pytest.raises(ValueError, match="domain"):
            normal.plot(params=nan_params, show=False)

    def test_explicit_domain_bypasses_discovery_with_nan_params(self):
        """Supplying an explicit ``domain`` skips domain discovery
        entirely — the bound only fences the auto-discovery path, so a
        caller who provides a domain is never blocked by it (the pdf/cdf
        curves themselves may be NaN, which is the honest signal)."""
        nan_params = {"mu": jnp.nan, "sigma": jnp.nan}
        # Must not raise the domain-discovery ValueError and must not hang.
        normal.plot(params=nan_params, domain=(-5.0, 5.0), show=False)
