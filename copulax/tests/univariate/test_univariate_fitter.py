# """Tests for the univariate_fitter function"""
import numpy as np
import pytest

from copulax.univariate import univariate_fitter, normal, student_t, lognormal, uniform
from copulax._src.typing import Scalar


METRICS = tuple(("aic", "bic", "loglikelihood"))
USER_DEFINED_DISTS = tuple((normal, student_t, lognormal))
DISTS = tuple(
    (
        "all",
        "common",
        "continuous",
        "discrete",
        "common continuous",
        "common discrete",
        USER_DEFINED_DISTS,
    )
)
TEST_OPTIONS = tuple((metric, dists) for metric in METRICS for dists in DISTS)


@pytest.mark.parametrize("metric, dists", TEST_OPTIONS)
def test_univariate_fitter(metric, dists, continuous_data):
    # testing univariate fitter works
    best_index, fitted_dists = univariate_fitter(
        x=continuous_data, metric=metric, distributions=dists
    )

    # checking properties
    desc = f"metric = {metric} and dists = {dists}"
    assert (
        best_index == 0
    ), f"Best index should always be 0 (sorted output) when {desc}."
    assert isinstance(
        fitted_dists, tuple
    ), f"Fitted distributions are not a tuple when {desc}."
    assert all(
        isinstance(dist, dict) for dist in fitted_dists
    ), f"Fitted distributions tuple does not contain dictionaries when {desc}."
    assert all(
        ("params" in dist_res and "metric" in dist_res and "dist" in dist_res)
        for dist_res in fitted_dists
    ), f"Fitted distribution outputs do not contain all of 'params', 'metric' and 'dist' keys when {desc}."

    # verify sorted order
    if len(fitted_dists) > 1:
        metrics = [float(r["metric"]) for r in fitted_dists]
        if metric == "loglikelihood":
            # higher is better → descending
            assert all(
                a >= b for a, b in zip(metrics, metrics[1:])
            ), f"Results not sorted descending for loglikelihood when {desc}."
        else:
            # lower is better → ascending
            assert all(
                a <= b for a, b in zip(metrics, metrics[1:])
            ), f"Results not sorted ascending for {metric} when {desc}."


# ── GoF filtering tests ──────────────────────────────────────────────────────
GOF_TESTS = ("ks", "cvm")


@pytest.mark.parametrize("gof", GOF_TESTS)
def test_fitter_gof_includes_results(gof, continuous_data):
    """When gof_test is set, each result dict contains a 'gof' key."""
    best_index, fitted_dists = univariate_fitter(
        x=continuous_data, distributions="common continuous", gof_test=gof
    )
    if fitted_dists:
        assert all(
            "gof" in r for r in fitted_dists
        ), f"Missing 'gof' key in results when gof_test='{gof}'."
        for r in fitted_dists:
            assert "statistic" in r["gof"] and "p_value" in r["gof"]


@pytest.mark.parametrize("gof", GOF_TESTS)
def test_fitter_gof_filters_bad_dists(gof):
    """Fitting uniform data against normal only → should be rejected."""
    np.random.seed(123)
    x = np.random.uniform(0, 1, 300)
    best_index, fitted_dists = univariate_fitter(
        x=x, distributions=(normal,), gof_test=gof, significance_level=0.05
    )
    # normal is a bad fit for uniform data; should likely be filtered out
    assert best_index is None or len(fitted_dists) <= 1


@pytest.mark.parametrize("gof", GOF_TESTS)
def test_fitter_gof_all_rejected(gof):
    """When all distributions fail the GoF test, return (None, ())."""
    np.random.seed(99)
    # very non-normal data: bimodal
    x = np.concatenate(
        [np.random.normal(-10, 0.1, 150), np.random.normal(10, 0.1, 150)]
    )
    best_index, fitted_dists = univariate_fitter(
        x=x, distributions=(uniform,), gof_test=gof, significance_level=0.05
    )
    # uniform is a terrible fit for bimodal → should be rejected
    if not fitted_dists:
        assert best_index is None
        assert fitted_dists == ()


@pytest.mark.parametrize("gof", GOF_TESTS)
def test_fitter_gof_keeps_good_dists(gof):
    """Normal data fitted with normal should survive GoF filtering."""
    np.random.seed(7)
    x = np.random.normal(0, 1, 200)
    best_index, fitted_dists = univariate_fitter(
        x=x, distributions=(normal, student_t), gof_test=gof, significance_level=0.05
    )
    assert len(fitted_dists) >= 1, f"Normal should survive {gof} test on normal data."
    assert best_index == 0
