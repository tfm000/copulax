"""Wald-specific tests covering the high-λ/μ numerical-stability regime.

The generic ``test_univariate.py`` parametrize lists exercise a single
moderate ``(μ=1, λ=2)`` config. The overflow regime where naive
``exp(2λ/μ)`` blows up (``2λ/μ ≳ 709``) is not in the shared config
space, so it needs a dedicated regression test here.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from copulax.univariate import wald


class TestWaldCdfStability:
    """CDF must stay finite and match scipy when ``exp(2λ/μ)`` would overflow."""

    @pytest.mark.parametrize("x", [0.5, 0.9, 1.0, 1.1, 1.5, 2.0])
    def test_cdf_high_lambda_matches_scipy(self, x):
        params = {"mu": 1.0, "lamb": 400.0}
        cx = float(wald.cdf(jnp.asarray(x), params))
        sp = float(scipy.stats.invgauss(mu=1.0 / 400.0, scale=400.0).cdf(x))

        assert np.isfinite(cx), f"cdf is NaN/inf at x={x}, mu=1, lamb=400"
        assert 0.0 <= cx <= 1.0, f"cdf={cx} outside [0,1] at x={x}"
        np.testing.assert_allclose(
            cx, sp, rtol=1e-6,
            err_msg=f"Wald cdf mismatch vs scipy at x={x} (high-lambda regime)"
        )
