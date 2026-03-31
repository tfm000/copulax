"""Tests for multivariate probability distributions."""

import pytest
import jax.numpy as jnp
from jax import jit
import numpy as np

from copulax._src._distributions import Multivariate
from copulax.tests.helpers import *
from copulax.multivariate import mvt_normal, mvt_student_t, mvt_gh, mvt_skewed_t
from copulax.tests.multivariate.conftest import NUM_ASSETS

DISTRIBUTIONS = tuple((mvt_normal, mvt_student_t, mvt_gh, mvt_skewed_t))
VALID_DATASETS = ("uncorrelated_sample", "correlated_sample")
ERROR_DATASETS = ("too_large_dim_sample",)
ALL_DATASETS = VALID_DATASETS + ERROR_DATASETS
COMBINATIONS = tuple(
    (dist, dataset) for dist in DISTRIBUTIONS for dataset in ALL_DATASETS
)
VALID_COMBINATIONS = tuple(
    (dist, dataset) for dist in DISTRIBUTIONS for dataset in VALID_DATASETS
)


class TestDistributionStructure:
    """Tests for distribution object structure and metadata."""

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_is_multivariate(self, dist):
        assert isinstance(dist, Multivariate)

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_all_methods_implemented(self, dist):
        methods: set[str] = {
            "support",
            "logpdf",
            "pdf",
            "rvs",
            "sample",
            "fit",
            "stats",
            "loglikelihood",
            "aic",
            "bic",
            "dtype",
            "dist_type",
            "name",
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
        assert dist.dist_type == "multivariate"

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_example_params(self, dist):
        check_mvt_params(dist.example_params(), f"{dist} example_params")

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_jittable(self, dist):
        jittable(dist)


class TestSupport:
    """Tests for distribution support."""

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_support(self, dist):
        params = dist.example_params()
        support = dist.support(params)

        assert isinstance(support, jnp.ndarray)
        assert support.ndim == 2
        assert np.all(support[:, 0] < support[:, 1])
        assert no_nans(support)

        jit(dist.support)(params)


class TestDensity:
    """Tests for logpdf/pdf with valid and error datasets."""

    @pytest.mark.parametrize("dist, dataset", VALID_COMBINATIONS)
    @pytest.mark.parametrize("method", ["logpdf", "pdf"])
    def test_density(self, dist, dataset, method, datasets):
        data = datasets[dataset]
        params = dist.example_params()
        output = getattr(dist, method)(data, params=params)

        correct_mvt_shape(data, output, dist, method)
        assert no_nans(output), f"{dist} {method} contains NaNs"
        if method == "pdf":
            assert is_positive(output), f"{dist} pdf negative"
            assert is_finite(output), f"{dist} pdf non-finite"

        jit(getattr(dist, method))(data, params=params)
        gradients(getattr(dist, method), f"{dist} {method}", data, params=params)

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    @pytest.mark.parametrize("method", ["logpdf", "pdf"])
    def test_dim_mismatch_raises(self, dist, method, datasets):
        data = datasets["too_large_dim_sample"]
        params = dist.example_params()
        with pytest.raises((TypeError, ValueError)):
            getattr(dist, method)(data, params=params)


class TestRVS:
    """Tests for random variate sampling."""

    SIZES = (0, 1, 2, 11)

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    @pytest.mark.parametrize("size", SIZES)
    def test_rvs(self, dist, size):
        params = dist.example_params()
        output = dist.rvs(size=size, params=params)

        expected_shape = (size, NUM_ASSETS)
        assert output.shape == expected_shape
        assert no_nans(output)
        assert is_finite(output)

        jit(dist.rvs, static_argnums=0)(size=size, params=params)


class TestFit:
    """Tests for distribution fitting."""

    @pytest.mark.parametrize("dist, dataset", VALID_COMBINATIONS)
    def test_fit(self, dist, dataset, datasets):
        data = datasets[dataset]
        fitted = dist.fit(data)

        assert isinstance(fitted, Multivariate)
        assert fitted.params is not None
        check_mvt_params(fitted.params, f"{dist} fit")

        jitted_fitted = jit(dist.fit)(data)
        assert isinstance(jitted_fitted, Multivariate)


class TestFittedInstanceMethods:
    """Regression tests for fitted-instance method dispatch."""

    @pytest.mark.parametrize("dist, dataset", VALID_COMBINATIONS)
    def test_methods_run_without_explicit_params(self, dist, dataset, datasets):
        fitted = dist._fitted_instance(dist.example_params())
        data = jnp.asarray(datasets[dataset])[:10]

        failures = []
        method_calls = (
            ("support", lambda: fitted.support()),
            ("stats", lambda: fitted.stats()),
            ("logpdf", lambda: fitted.logpdf(data)),
            ("pdf", lambda: fitted.pdf(data)),
            ("rvs", lambda: fitted.rvs(size=4)),
            ("sample", lambda: fitted.sample(size=4)),
            ("loglikelihood", lambda: fitted.loglikelihood(data)),
            ("aic", lambda: fitted.aic(data)),
            ("bic", lambda: fitted.bic(data)),
        )

        for method_name, method in method_calls:
            try:
                method()
            except Exception as exc:
                failures.append(f"{method_name}: {type(exc).__name__}: {exc}")

        assert not failures, f"{fitted} fitted methods failed -> {failures}"


class TestStats:
    """Tests for distribution statistics."""

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_stats(self, dist):
        params = dist.example_params()
        stats = dist.stats(params=params)

        assert isinstance(stats, dict) and len(stats) > 0
        assert all(isinstance(k, str) for k in stats.keys())
        assert all(isinstance(v, jnp.ndarray) for v in stats.values())
        d = params["mu"].shape[0]
        for k, v in stats.items():
            ok = v.ndim == 0 or v.shape == (d, 1) or v.shape == (d, d)
            assert ok, f"{dist} stats['{k}'] unexpected shape {v.shape}"

        jit(dist.stats)(params=params)


class TestMetrics:
    """Tests for loglikelihood, AIC, BIC."""

    @pytest.mark.parametrize("dist, dataset", VALID_COMBINATIONS)
    @pytest.mark.parametrize("metric", ["loglikelihood", "aic", "bic"])
    def test_metric(self, dist, dataset, metric, datasets):
        data = datasets[dataset]
        params = dist.example_params()
        func = getattr(dist, metric)
        output = func(data, params=params)

        check_metric_output(dist, output, metric)
        jit(func)(data, params=params)
        gradients(func, f"{dist} {metric}", data, params)

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    @pytest.mark.parametrize("metric", ["loglikelihood", "aic", "bic"])
    def test_metric_dim_mismatch_raises(self, dist, metric, datasets):
        data = datasets["too_large_dim_sample"]
        params = dist.example_params()
        with pytest.raises((TypeError, ValueError)):
            getattr(dist, metric)(data, params=params)
