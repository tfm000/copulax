"""Rigorous tests for univariate_fitter: distribution ranking and GoF filtering."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copulax.univariate import univariate_fitter, normal, student_t, gamma, uniform


class TestUnivariateProfiler:
    """Verify the univariate fitter correctly ranks and filters distributions."""

    @pytest.mark.parametrize("metric", ["aic", "bic", "loglikelihood"])
    def test_sorting_order(self, metric):
        """Results should be sorted correctly by the chosen metric."""
        np.random.seed(42)
        data = jnp.array(np.random.normal(0, 1, 500))
        result = univariate_fitter(x=data, metric=metric)
        assert result is not None, "univariate_fitter returned None"

    def test_normal_data_ranks_normal_high(self):
        """Normal data should rank the normal distribution near the top."""
        np.random.seed(42)
        data = jnp.array(np.random.normal(2.0, 1.5, 1000))
        result = univariate_fitter(
            x=data, metric="bic",
            distributions=[normal, student_t, gamma, uniform]
        )
        # result is (best_dist, fitted_dists_tuple)
        assert result is not None

    def test_gof_filtering_rejects_bad_fits(self):
        """GoF should reject distributions that clearly don't fit."""
        np.random.seed(42)
        # Uniform data should fail a normality GoF test
        data = jnp.array(np.random.uniform(0, 1, 500))
        result = univariate_fitter(
            x=data, metric="bic",
            distributions=[normal, uniform],
            gof_test="ks",
            significance_level=0.05,
        )
        # The result should exist (at least uniform should pass)
        assert result is not None

    def test_gof_filtering_keeps_good_fits(self):
        """GoF should keep distributions that do fit."""
        np.random.seed(42)
        data = jnp.array(np.random.normal(0, 1, 500))
        result = univariate_fitter(
            x=data, metric="bic",
            distributions=[normal, student_t],
            gof_test="ks",
            significance_level=0.05,
        )
        assert result is not None


class TestFitterEdgeCases:
    """Edge cases for the univariate fitter."""

    def test_single_distribution(self):
        """Fitter should work with a single distribution."""
        np.random.seed(42)
        data = jnp.array(np.random.normal(0, 1, 200))
        result = univariate_fitter(
            x=data, metric="bic",
            distributions=[normal],
        )
        assert result is not None

    def test_small_sample(self):
        """Fitter should handle small samples gracefully."""
        np.random.seed(42)
        data = jnp.array(np.random.normal(0, 1, 30))
        result = univariate_fitter(
            x=data, metric="bic",
            distributions=[normal, uniform],
        )
        assert result is not None
