# """Tests for the univariate_fitter function"""
from jax import jit
import pytest
from typing import Callable

from copulax.univariate import univariate_fitter, normal, student_t, lognormal
from copulax._src.typing import Scalar


METRICS = tuple(("aic", "bic", "loglikelihood"))
USER_DEFINED_DISTS = tuple((normal, student_t, lognormal))
DISTS = tuple(("all", "common", "continuous", "discrete", "common continuous", "common discrete", USER_DEFINED_DISTS))
TEST_OPTIONS = tuple((metric, dists) for metric in METRICS for dists in DISTS)


@pytest.mark.parametrize("metric, dists", TEST_OPTIONS)
def test_univariate_fitter(metric, dists, continuous_data):
    # testing univariate fitter works
    best_index, fitted_dists = univariate_fitter(
        x=continuous_data, metric=metric, distributions=dists)
    
    # checking properties
    desc = f"metric = {metric} and dists = {dists}"
    assert isinstance(best_index, Scalar), f"Best index is not a scalar when {desc}."
    assert isinstance(fitted_dists, tuple), f"Fitted distributions are not a tuple when {desc}."
    assert all(isinstance(dist, dict) for dist in fitted_dists), f"Fitted distributions tuple does not contain dictionaries when {desc}."
    assert all(("params" in dist_res and "metric" in dist_res and "dist" in dist_res) for dist_res in fitted_dists), f"Fitted distribution outputs do not contain all of 'params', 'metric' and 'dist' keys when {desc}."

    # testing univariate fitter works when jitted
    jitted_func: Callable = jit(univariate_fitter, static_argnames=['metric', 'distributions'])
    best_index_jit, fitted_dists_jit = jitted_func(
        x=continuous_data, metric=metric, distributions=dists)
    assert isinstance(best_index_jit, Scalar), f"Best index is not a scalar when jitted for {desc}."
    assert isinstance(fitted_dists_jit, tuple), f"Fitted distributions are not a tuple when jitted for {desc}."
