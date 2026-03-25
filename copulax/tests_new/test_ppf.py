"""Rigorous tests for PPF (percent point function / quantile function).

Cross-validates against scipy PPF and verifies mathematical properties
(monotonicity, boundary values, CDF-PPF inverse relationship).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copulax.univariate import (
    normal, student_t, gamma, lognormal, uniform,
    ig, gen_normal, gig, gh,
)
from copulax.tests_new.conftest import get_scipy_dist


# Distributions with their test parameters and expected scipy equivalents
PPF_CONFIGS = [
    (normal, {"mu": 2.0, "sigma": 1.5}),
    (student_t, {"nu": 5.0, "mu": 1.0, "sigma": 2.0}),
    (gamma, {"alpha": 3.0, "beta": 2.0}),
    (lognormal, {"mu": 0.5, "sigma": 0.8}),
    (uniform, {"a": -1.0, "b": 3.0}),
    (ig, {"alpha": 4.0, "beta": 2.0}),
    (gen_normal, {"mu": 0.5, "alpha": 1.5, "beta": 2.0}),
]

PPF_IDS = [d.name for d, _ in PPF_CONFIGS]


class TestPPFAgainstScipy:
    """Verify PPF matches scipy quantile function."""

    @pytest.mark.parametrize("dist,params", PPF_CONFIGS, ids=PPF_IDS)
    def test_ppf_matches_scipy(self, dist, params):
        """PPF(q) should match scipy.ppf(q) for q in (0.01, 0.99)."""
        sp = get_scipy_dist(dist, params)
        if sp is None:
            pytest.skip(f"No scipy equivalent for {dist.name}")

        q = np.linspace(0.05, 0.95, 20)
        cx_ppf = np.array(dist.ppf(q=jnp.array(q), params=params,
                                    maxiter=50)).flatten()
        sp_ppf = sp.ppf(q)

        mask = np.isfinite(cx_ppf) & np.isfinite(sp_ppf)
        np.testing.assert_allclose(
            cx_ppf[mask], sp_ppf[mask], rtol=1e-2, atol=1e-4,
            err_msg=f"{dist.name} PPF mismatch vs scipy"
        )


class TestPPFMonotonicity:
    """PPF must be non-decreasing."""

    @pytest.mark.parametrize("dist,params", PPF_CONFIGS, ids=PPF_IDS)
    def test_ppf_non_decreasing(self, dist, params):
        q = jnp.linspace(0.05, 0.95, 20)
        x = np.array(dist.ppf(q=q, params=params, maxiter=50)).flatten()
        diffs = np.diff(x)
        mask = np.isfinite(diffs)
        assert np.all(diffs[mask] >= -1e-6), \
            f"{dist.name} PPF not monotone: min diff = {np.min(diffs[mask])}"


class TestPPFBoundary:
    """PPF boundary values at q=0 and q=1."""

    @pytest.mark.parametrize("dist,params", PPF_CONFIGS, ids=PPF_IDS)
    def test_ppf_at_zero_and_one(self, dist, params):
        support = np.array(dist._support(params)).flatten()
        lo, hi = float(support[0]), float(support[1])

        q_zero = jnp.array([0.0])
        q_one = jnp.array([1.0])

        ppf_0 = float(dist.ppf(q=q_zero, params=params, maxiter=50).flatten()[0])
        ppf_1 = float(dist.ppf(q=q_one, params=params, maxiter=50).flatten()[0])

        if np.isfinite(lo):
            np.testing.assert_allclose(ppf_0, lo, atol=1e-3,
                                       err_msg=f"{dist.name} PPF(0) != lower bound")
        else:
            assert ppf_0 < -100 or np.isneginf(ppf_0), \
                f"{dist.name} PPF(0) should be -inf, got {ppf_0}"

        if np.isfinite(hi):
            np.testing.assert_allclose(ppf_1, hi, atol=1e-3,
                                       err_msg=f"{dist.name} PPF(1) != upper bound")
        else:
            assert ppf_1 > 100 or np.isposinf(ppf_1), \
                f"{dist.name} PPF(1) should be +inf, got {ppf_1}"


class TestPPFCubicVsLinear:
    """Cubic interpolation and direct optimization should agree."""

    @pytest.mark.parametrize("dist,params", [
        (normal, {"mu": 0.0, "sigma": 1.0}),
        (gamma, {"alpha": 3.0, "beta": 2.0}),
        (uniform, {"a": 0.0, "b": 1.0}),
    ], ids=["Normal", "Gamma", "Uniform"])
    def test_cubic_vs_direct(self, dist, params):
        q = jnp.linspace(0.1, 0.9, 10)
        direct = np.array(dist.ppf(q=q, params=params, cubic=False,
                                    maxiter=50)).flatten()
        cubic = np.array(dist.ppf(q=q, params=params, cubic=True,
                                   num_points=200, maxiter=50)).flatten()
        mask = np.isfinite(direct) & np.isfinite(cubic)
        np.testing.assert_allclose(
            direct[mask], cubic[mask], rtol=1e-2, atol=1e-3,
            err_msg=f"{dist.name} cubic vs direct PPF disagree"
        )
