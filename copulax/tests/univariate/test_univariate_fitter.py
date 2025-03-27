"""Tests for the univariate_fitter function"""
from jax import jit

from copulax.univariate import univariate_fitter
from copulax._src.typing import Scalar

# TODO: test all methods! i.e. aic, bic, loglikelihood - particularly when jitted.
# TODO: implement gof tests for univariate fitter if unjittable


def test_univariate_fitter(continuous_data, continuous_dists):
    # testing univariate fitter works
    best_index, fitted_dists = univariate_fitter(continuous_data)
    assert isinstance(best_index, Scalar), "Best index is not a scalar."
    assert isinstance(fitted_dists, tuple), "Fitted distributions are not a tuple."
    assert len(fitted_dists) == len(continuous_dists), "Fitted distributions tuple is not the correct length."
    assert all(isinstance(dist, dict) for dist in fitted_dists), "Fitted distributions tuple does not contain dictionaries."
    assert all(("params" in dist_res and "metric" in dist_res and "dist" in dist_res) for dist_res in fitted_dists), "Fitted distribution outputs do not contain all of 'params', 'metric' and 'dist' keys."

    # testing univariate fitter works when jitted
    best_index_jit, fitted_dists_jit = jit(univariate_fitter)(continuous_data)
    assert isinstance(best_index_jit, Scalar), "Best index is not a scalar when jitted."
    assert isinstance(fitted_dists_jit, tuple), "Fitted distributions are not a tuple when jitted."