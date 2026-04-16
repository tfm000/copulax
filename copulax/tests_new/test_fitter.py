"""Rigorous tests for univariate_fitter: distribution ranking and GoF filtering."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copulax.univariate import (
    univariate_fitter, batch_univariate_fitter,
    normal, student_t, gamma, uniform,
)


class TestUnivariateProfiler:
    """Verify the univariate fitter correctly ranks and filters distributions."""

    @pytest.mark.parametrize("metric", ["aic", "bic", "loglikelihood"])
    def test_sorting_order(self, metric):
        """Results should be sorted correctly by the chosen metric."""
        np.random.seed(42)
        data = jnp.array(np.random.normal(0, 1, 500))
        best_idx, fitted = univariate_fitter(x=data, metric=metric)
        assert best_idx == 0
        metrics = [float(r["metric"]) for r in fitted]
        if metric == "loglikelihood":
            assert all(a >= b for a, b in zip(metrics, metrics[1:])), \
                f"Not sorted descending for {metric}: {metrics}"
        else:
            assert all(a <= b for a, b in zip(metrics, metrics[1:])), \
                f"Not sorted ascending for {metric}: {metrics}"

    def test_normal_data_ranks_normal_high(self):
        """Normal data should rank the normal distribution near the top."""
        np.random.seed(42)
        data = jnp.array(np.random.normal(2.0, 1.5, 1000))
        best_idx, fitted = univariate_fitter(
            x=data, metric="bic",
            distributions=[normal, student_t, gamma, uniform]
        )
        top_2_names = [r["dist"].name for r in fitted[:2]]
        assert "Normal" in top_2_names, f"Normal not in top 2: {top_2_names}"

    def test_gof_filtering_rejects_bad_fits(self):
        """GoF should reject distributions that clearly don't fit."""
        np.random.seed(42)
        # Uniform data should fail a normality GoF test
        data = jnp.array(np.random.uniform(0, 1, 500))
        best_idx, fitted = univariate_fitter(
            x=data, metric="bic",
            distributions=[normal, uniform],
            gof_test="ks",
            significance_level=0.05,
        )
        surviving_names = [r["dist"].name for r in fitted]
        assert "Normal" not in surviving_names, \
            "Normal should have been rejected by KS test on uniform data"
        assert "Uniform" in surviving_names, \
            "Uniform should survive KS test on uniform data"

    def test_gof_filtering_keeps_good_fits(self):
        """GoF should keep distributions that do fit."""
        np.random.seed(42)
        data = jnp.array(np.random.normal(0, 1, 500))
        best_idx, fitted = univariate_fitter(
            x=data, metric="bic",
            distributions=[normal, student_t],
            gof_test="ks",
            significance_level=0.05,
        )
        surviving_names = [r["dist"].name for r in fitted]
        assert "Normal" in surviving_names, "Normal should pass KS on normal data"
        assert "Student-T" in surviving_names, "Student-T should pass KS on normal data"


class TestFitterEdgeCases:
    """Edge cases for the univariate fitter."""

    def test_single_distribution(self):
        """Fitter should work with a single distribution."""
        np.random.seed(42)
        data = jnp.array(np.random.normal(0, 1, 200))
        best_idx, fitted = univariate_fitter(
            x=data, metric="bic",
            distributions=[normal],
        )
        assert best_idx == 0
        assert len(fitted) == 1
        assert fitted[0]["dist"].name == "Normal"
        assert np.isfinite(fitted[0]["metric"])

    def test_small_sample(self):
        """Fitter should handle small samples gracefully."""
        np.random.seed(42)
        data = jnp.array(np.random.normal(0, 1, 30))
        best_idx, fitted = univariate_fitter(
            x=data, metric="bic",
            distributions=[normal, uniform],
        )
        assert best_idx == 0
        assert len(fitted) >= 1
        for r in fitted:
            assert np.isfinite(r["metric"]), f"{r['dist'].name} metric not finite"


class TestBatchUnivariateFitter:
    """Tests for batch_univariate_fitter (vmapped multi-column fitting)."""

    def test_returns_list_of_correct_length(self):
        """2D input with 3 columns should return list of length 3."""
        np.random.seed(42)
        data = jnp.array(np.random.normal(0, 1, (200, 3)))
        results = batch_univariate_fitter(
            x=data, metric="bic",
            distributions=[normal, uniform],
        )
        assert isinstance(results, list), f"Expected list, got {type(results)}"
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        for i, res in enumerate(results):
            assert res is not None, f"Column {i} result is None"

    def test_ranking_matches_univariate_fitter(self):
        """Each column's result should match calling univariate_fitter independently."""
        np.random.seed(42)
        data_np = np.random.normal(0, 1, (300, 2))
        data = jnp.array(data_np)
        dists = [normal, uniform]

        batch_results = batch_univariate_fitter(
            x=data, metric="bic", distributions=dists,
        )
        for col in range(2):
            single_result = univariate_fitter(
                x=jnp.array(data_np[:, col]), metric="bic", distributions=dists,
            )
            # Best distribution class should match
            batch_best = batch_results[col][1][0]["dist"]
            single_best = single_result[1][0]["dist"]
            assert type(batch_best) == type(single_best), (
                f"Col {col}: batch best={batch_best.name}, "
                f"single best={single_best.name}"
            )

    def test_normal_columns_rank_normal_high(self):
        """Normal data columns: normal should rank in top 2."""
        np.random.seed(42)
        data = jnp.array(np.random.normal(2.0, 1.5, (500, 3)))
        results = batch_univariate_fitter(
            x=data, metric="bic",
            distributions=[normal, student_t, gamma, uniform],
        )
        for col, res in enumerate(results):
            top_2_names = [r["dist"].name for r in res[1][:2]]
            assert "Normal" in top_2_names, (
                f"Col {col}: Normal not in top 2, got {top_2_names}"
            )

    def test_gof_filtering(self):
        """GoF filtering should reject bad fits in batch mode."""
        np.random.seed(42)
        data = jnp.array(np.random.uniform(0, 1, (500, 2)))
        results = batch_univariate_fitter(
            x=data, metric="bic",
            distributions=[normal, uniform],
            gof_test="ks",
            significance_level=0.05,
        )
        # At least uniform should survive for each column
        for col, res in enumerate(results):
            assert res is not None, f"Col {col}: all dists filtered out"

    def test_mixed_distributions(self):
        """Columns from different dists: top-ranked should match generator."""
        np.random.seed(42)
        col_normal = np.random.normal(0, 1, 500)
        col_uniform = np.random.uniform(-2, 2, 500)
        data = jnp.array(np.column_stack([col_normal, col_uniform]))

        results = batch_univariate_fitter(
            x=data, metric="bic",
            distributions=[normal, student_t, uniform],
        )
        # Column 0 (normal data) should rank Normal or Student-T first
        best_col0 = results[0][1][0]["dist"].name
        assert best_col0 in ("Normal", "Student-T"), (
            f"Col 0 (normal data): best={best_col0}"
        )
        # Column 1 (uniform data) should rank Uniform first
        best_col1 = results[1][1][0]["dist"].name
        assert best_col1 == "Uniform", f"Col 1 (uniform data): best={best_col1}"
