"""Tests for copula distributions."""

import pytest
import jax.numpy as jnp
from jax import jit
import numpy as np

from copulax._src._distributions import Univariate, Multivariate
from copulax._src.copulas._distributions import Copula, CopulaBase
from copulax.tests.helpers import *
from copulax.tests.copulas.conftest import NUM_ASSETS
from copulax.copulas import (
    gaussian_copula,
    student_t_copula,
    skewed_t_copula,
    gh_copula,
)


# Test combinations
FAST_DISTRIBUTIONS = (gaussian_copula, student_t_copula)
SLOW_DISTRIBUTIONS = (
    pytest.param(skewed_t_copula, marks=pytest.mark.slow),
    pytest.param(gh_copula, marks=pytest.mark.slow),
)
DISTRIBUTIONS = FAST_DISTRIBUTIONS + SLOW_DISTRIBUTIONS
DATASETS = ("uncorrelated_sample", "correlated_sample")
COMBINATIONS = tuple(
    (dist, dataset) for dist in FAST_DISTRIBUTIONS for dataset in DATASETS
) + tuple(
    pytest.param(dist, dataset, marks=pytest.mark.slow)
    for dist in (skewed_t_copula, gh_copula)
    for dataset in DATASETS
)
SIZES = (0, 1, 2, 11)
SIZE_COMBINATIONS = tuple(
    (dist, size) for dist in FAST_DISTRIBUTIONS for size in SIZES
) + tuple(
    pytest.param(dist, size, marks=pytest.mark.slow)
    for dist in (skewed_t_copula, gh_copula)
    for size in SIZES
)


def _check_marginals(dist, dim, fitted_marginals):
    assert isinstance(fitted_marginals, dict)
    assert len(fitted_marginals) == 1 and "marginals" in fitted_marginals
    marginals = fitted_marginals["marginals"]
    assert isinstance(marginals, tuple) and len(marginals) == dim
    for marginal_tup in marginals:
        assert isinstance(marginal_tup, tuple) and len(marginal_tup) == 2
        marginal_dist, marginal_params = marginal_tup
        assert isinstance(marginal_dist, Univariate)
        check_uvt_params(marginal_params, f"{dist} marginal params")


def _check_copula_params(dist, fitted_copula):
    assert isinstance(fitted_copula, dict)
    assert len(fitted_copula) == 1 and "copula" in fitted_copula
    check_mvt_params(fitted_copula["copula"], f"{dist} copula_params")


class TestDistributionStructure:
    """Tests for copula distribution object structure and metadata."""

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_is_copula(self, dist):
        assert isinstance(dist, Copula)
        assert isinstance(dist._mvt, Multivariate)
        assert isinstance(dist._uvt, Univariate)

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_all_methods_implemented(self, dist):
        methods: set[str] = {
            "dtype",
            "dist_type",
            "name",
            "support",
            "get_u",
            "get_x_dash",
            "copula_logpdf",
            "copula_pdf",
            "logpdf",
            "pdf",
            "copula_rvs",
            "copula_sample",
            "rvs",
            "sample",
            "fit_marginals",
            "fit_copula",
            "fit",
            "aic",
            "bic",
            "loglikelihood",
            "stats",
            "example_params",
        }
        for method in methods:
            assert hasattr(dist, method), f"{dist} missing method {method}"

        pytree_methods: set[str] = {"tree_flatten", "tree_unflatten"}
        extra_methods = set(dist.__dict__.keys()) - methods - pytree_methods
        extra_methods = {
            m
            for m in extra_methods
            if not m.startswith("_") and callable(getattr(dist, m, None))
        }
        assert not extra_methods, f"{dist} has extra methods: {extra_methods}"

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_name(self, dist):
        assert isinstance(dist.name, str) and dist.name != ""
        assert dist.name == str(dist)

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_dtype(self, dist):
        assert dist.dtype in ("continuous", "discrete")

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_dist_type(self, dist):
        assert dist.dist_type == "copula"

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_example_params(self, dist):
        params = dist.example_params(dim=NUM_ASSETS)
        assert len(params) == 2 and "marginals" in params and "copula" in params

        copula_params = params.copy()
        copula_params.pop("marginals")
        _check_copula_params(dist, copula_params)

        marginal_params = params.copy()
        marginal_params.pop("copula")
        _check_marginals(dist, NUM_ASSETS, marginal_params)

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_jittable(self, dist):
        jittable(dist)


class TestSupport:
    """Tests for copula support."""

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_support(self, dist):
        params = dist.example_params()
        support = dist.support(params)

        assert isinstance(support, jnp.ndarray)
        assert two_dim(support)
        assert np.all(support[:, 0] < support[:, 1])
        assert no_nans(support)

        jit(dist.support)(params)


class TestTransforms:
    """Tests for get_u and get_x_dash transforms."""

    @pytest.mark.parametrize("dist, dataset", COMBINATIONS)
    def test_get_u(self, dist, dataset, datasets):
        params = dist.example_params()
        sample = datasets[dataset]
        u = dist.get_u(sample, params)

        assert isinstance(u, jnp.ndarray) and two_dim(u)
        assert np.all(u >= 0) and np.all(u <= 1)
        assert no_nans(u)

        jit(dist.get_u)(sample, params)
        gradients(
            dist.get_u,
            f"{dist} u",
            sample,
            params,
            params_error=(dist is not student_t_copula),
        )

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_get_x_dash(self, dist, u_sample):
        params = dist.example_params()
        x_dash = dist.get_x_dash(u_sample, params)

        assert isinstance(x_dash, jnp.ndarray) and two_dim(x_dash)
        assert no_nans(x_dash)

        jit(dist.get_x_dash)(u_sample, params)
        gradients(
            dist.get_x_dash,
            f"{dist} x_dash",
            u_sample,
            params,
            params_error=(dist is not student_t_copula),
        )


class TestCopulaDensity:
    """Tests for copula_logpdf/copula_pdf."""

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    @pytest.mark.parametrize("method", ["copula_logpdf", "copula_pdf"])
    def test_copula_density(self, dist, method, u_sample):
        params = dist.example_params()
        output = getattr(dist, method)(u_sample, params)

        correct_mvt_shape(u_sample, output, dist, method)
        assert no_nans(output), f"{dist} {method} contains NaNs"
        if method == "copula_pdf":
            assert is_positive(output), f"{dist} copula_pdf not positive"
            assert is_finite(output), f"{dist} copula_pdf non-finite"

        jit(getattr(dist, method))(u_sample, params)
        gradients(
            getattr(dist, method),
            f"{dist} {method}",
            u_sample,
            params,
            params_error=(dist is not student_t_copula),
        )


class TestJointDensity:
    """Tests for logpdf/pdf via Sklar's decomposition."""

    @pytest.mark.parametrize("dist, dataset", COMBINATIONS)
    @pytest.mark.parametrize("method", ["logpdf", "pdf"])
    def test_joint_density(self, dist, dataset, method, datasets):
        params = dist.example_params()
        sample = datasets[dataset]
        output = getattr(dist, method)(sample, params)

        correct_mvt_shape(sample, output, dist, method)
        assert no_nans(output), f"{dist} {method} contains NaNs"
        if method == "pdf":
            assert is_positive(output), f"{dist} pdf negative"
            assert is_finite(output), f"{dist} pdf non-finite"

        jit(getattr(dist, method))(sample, params)
        gradients(
            getattr(dist, method),
            f"{dist} {method}",
            sample,
            params,
            params_error=(dist is not student_t_copula),
        )


class TestRVS:
    """Tests for copula_rvs and rvs."""

    @pytest.mark.parametrize("dist, size", SIZE_COMBINATIONS)
    @pytest.mark.parametrize("method", ["copula_rvs", "rvs"])
    def test_rvs(self, dist, size, method):
        params = dist.example_params()
        output = getattr(dist, method)(size=size, params=params)

        expected_shape = (size, NUM_ASSETS)
        assert two_dim(output)
        assert output.shape == expected_shape
        assert no_nans(output)
        assert is_finite(output)

        jit(getattr(dist, method), static_argnums=0)(size=size, params=params)


class TestFitting:
    """Tests for fit_marginals, fit_copula, and fit."""

    @pytest.mark.parametrize("dist, dataset", COMBINATIONS)
    def test_fit_marginals(self, dist, dataset, datasets):
        sample = datasets[dataset]
        fitted_marginals = dist.fit_marginals(sample)
        _check_marginals(dist, sample.shape[1], fitted_marginals)

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_fit_copula(self, dist, u_sample):
        fitted_copula = dist.fit_copula(u_sample)
        _check_copula_params(dist, fitted_copula)
        jit(dist.fit_copula)(u_sample)

    @pytest.mark.parametrize("dist, dataset", COMBINATIONS)
    def test_fit(self, dist, dataset, datasets):
        sample = datasets[dataset]
        fitted_joint = dist.fit(sample)

        assert isinstance(fitted_joint, CopulaBase)
        assert fitted_joint.params is not None
        fitted_params = fitted_joint.params
        assert len(fitted_params) == 2
        assert "marginals" in fitted_params and "copula" in fitted_params

        fitted_marginals = fitted_params.copy()
        fitted_marginals.pop("copula")
        _check_marginals(dist, sample.shape[1], fitted_marginals)
        fitted_copula = fitted_params.copy()
        fitted_copula.pop("marginals")
        _check_copula_params(dist, fitted_copula)


class TestMetrics:
    """Tests for loglikelihood, AIC, BIC."""

    @pytest.mark.parametrize("dist, dataset", COMBINATIONS)
    @pytest.mark.parametrize("metric", ["loglikelihood", "aic", "bic"])
    def test_metric(self, dist, dataset, metric, datasets):
        params = dist.example_params()
        sample = datasets[dataset]
        func = getattr(dist, metric)
        output = func(sample, params)

        check_metric_output(dist, output, metric)
        jit(func)(sample, params)
        gradients(
            func,
            f"{dist} {metric}",
            sample,
            params,
            params_error=(dist is not student_t_copula),
        )
