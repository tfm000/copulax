"""Tests for univariate probability distributions."""

import numpy as np
from jax import jit
import inspect
import pytest
from jax import numpy as jnp

from copulax._src._distributions import Univariate
from copulax.univariate import distributions
from copulax.univariate.distributions import _all_dists, skewed_t, student_t
from copulax._src.typing import Scalar
from copulax.tests.helpers import *


# Helper
def get_data(dist, continuous_data, discrete_data):
    if dist.dtype == "continuous":
        return continuous_data
    elif dist.dtype == "discrete":
        return discrete_data
    else:
        raise ValueError(f"Unknown distribution type: {dist.dtype}")


# Distribution collection
DISTRIBUTIONS = tuple(
    value for value in distributions.values() if isinstance(value, Univariate)
)
UNNORMALISED_DISTRIBUTIONS = tuple((skewed_t.name, student_t.name))

# Reduced RVS sizes — removed redundant combos like (1,)/(1,1)/1
RVS_SIZES = (
    ((), 1),
    ((3,), 3),
    ((3, 2), 6),
    (0, 0),
)


class TestDistributionStructure:
    """Tests for distribution object structure and metadata."""

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_is_univariate(self, dist):
        assert isinstance(dist, Univariate), f"{dist} is not a Univariate object."

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_all_methods_implemented(self, dist):
        methods: set[str] = {
            "support",
            "logpdf",
            "pdf",
            "logcdf",
            "cdf",
            "ppf",
            "inverse_cdf",
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
        for method_name in methods:
            assert hasattr(dist, method_name), f"{dist} missing {method_name} method."

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
        assert dist.dist_type == "univariate"

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_example_params(self, dist):
        params = dist.example_params()
        check_uvt_params(params, f"{dist} example_params")

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_jittable(self, dist):
        jittable(dist)


class TestSupport:
    """Tests for distribution support."""

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_support(self, dist):
        params = dist.example_params()
        support = dist.support(params)

        assert isinstance(support, jnp.ndarray), f"{dist} support is not a JAX array"
        assert support.shape == (2,), f"{dist} support shape mismatch"
        assert np.all(support[0] < support[1]), f"{dist} bounds not in order"
        assert no_nans(support), f"{dist} support contains NaNs"

        jit(dist.support)(params)


class TestDensity:
    """Tests for logpdf/pdf and logcdf/cdf."""

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    @pytest.mark.parametrize("method", ["logpdf", "pdf"])
    def test_density(self, dist, method, continuous_data, discrete_data):
        data = get_data(dist, continuous_data, discrete_data)
        params = dist.example_params()
        output = getattr(dist, method)(data, params)

        correct_uvt_shape(data, output, dist, method)
        assert no_nans(output), f"{dist} {method} contains NaNs"
        if method == "pdf":
            assert is_positive(output), f"{dist} pdf contains negative values"
            assert is_finite(output), f"{dist} pdf contains non-finite values"

        jit(getattr(dist, method))(data, params)
        gradients(
            getattr(dist, method),
            f"{dist.name} {method}",
            data,
            params,
            params_error=dist.name not in UNNORMALISED_DISTRIBUTIONS,
        )

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    @pytest.mark.parametrize("method", ["logcdf", "cdf"])
    def test_cdf(self, dist, method, continuous_data, discrete_data):
        data = get_data(dist, continuous_data, discrete_data)
        params = dist.example_params()
        output = getattr(dist, method)(data, params)

        correct_uvt_shape(data, output, dist, method)
        assert no_nans(output), f"{dist} {method} contains NaNs"
        if method == "cdf":
            assert is_positive(output), f"{dist} cdf contains negative values"
            assert is_finite(output), f"{dist} cdf contains non-finite values"
            assert np.all(0 <= output) and np.all(
                output <= 1
            ), f"cdf not in [0, 1] for {dist}"

        jit(getattr(dist, method))(data, params)
        gradients(
            getattr(dist, method),
            f"{dist.name} {method}",
            data,
            params,
            params_error=dist.name not in UNNORMALISED_DISTRIBUTIONS,
        )


class TestPPF:
    """Tests for percent point function (inverse CDF)."""

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    @pytest.mark.parametrize("cubic", [False, True])
    def test_ppf(self, dist, cubic, uniform_data):
        data = uniform_data
        params = dist.example_params()
        support = dist.support(params)
        s = " cubic" if cubic else ""

        output = dist.ppf(data, params, cubic=cubic, maxiter=5)

        correct_uvt_shape(data, output, dist, "ppf")
        assert no_nans(output), f"{dist} ppf{s} contains NaNs"
        assert is_finite(output), f"{dist} ppf{s} contains non-finite values"
        assert np.all(output >= support[0]) and np.all(
            output <= support[1]
        ), f"ppf not in support for {dist}{s}"

        jitted = jit(dist.ppf, static_argnames="cubic")
        jitted(data, params, cubic=cubic)
        gradients(jitted, f"{dist.name} ppf", data, params, cubic=cubic)


class TestRVS:
    """Tests for random variate sampling."""

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    @pytest.mark.parametrize("size, num", RVS_SIZES)
    def test_rvs(self, dist, size, num):
        params = dist.example_params()
        output = dist.rvs(size, params)

        shape = (size,) if isinstance(size, int) else size
        assert isinstance(output, jnp.ndarray), f"{dist} rvs is not a JAX array"
        assert output.shape == shape, f"{dist} rvs shape mismatch"
        assert output.size == num, f"{dist} rvs size mismatch"
        assert no_nans(output), f"{dist} rvs contains NaNs"
        assert is_finite(output), f"{dist} rvs contains non-finite values"

        jit(dist.rvs, static_argnames="size")(size=size, params=params)


class TestFit:
    """Tests for distribution fitting."""

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_fit(self, dist, continuous_data, discrete_data):
        data = get_data(dist, continuous_data, discrete_data)
        fitted = dist.fit(data)

        assert isinstance(fitted, type(dist)), f"{dist} fit did not return same type"

        fitted_params = fitted.params
        check_uvt_params(fitted_params, f"{dist} fitted")
        assert set(fitted_params.keys()) == set(
            dist.example_params().keys()
        ), f"{dist} fitted params and example_params mismatch"

        fit_args = inspect.getfullargspec(dist.fit).args
        if "method" in fit_args:
            jit_fit = jit(dist.fit, static_argnames="method")(continuous_data)
        else:
            jit_fit = jit(dist.fit)(continuous_data)
        check_uvt_params(jit_fit.params, f"{dist} jit fitted")


class TestFittedInstanceMethods:
    """Regression tests for fitted-instance method dispatch."""

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_methods_run_without_explicit_params(
        self, dist, continuous_data, discrete_data, uniform_data
    ):
        fitted = dist._fitted_instance(dist.example_params())
        data = jnp.asarray(get_data(dist, continuous_data, discrete_data))[:10]
        quantiles = jnp.asarray(uniform_data)

        failures = []
        method_calls = (
            ("support", lambda: fitted.support()),
            ("stats", lambda: fitted.stats()),
            ("logpdf", lambda: fitted.logpdf(data)),
            ("pdf", lambda: fitted.pdf(data)),
            ("logcdf", lambda: fitted.logcdf(data)),
            ("cdf", lambda: fitted.cdf(data)),
            ("ppf", lambda: fitted.ppf(quantiles, maxiter=5)),
            ("inverse_cdf", lambda: fitted.inverse_cdf(quantiles, maxiter=5)),
            ("rvs", lambda: fitted.rvs(size=(4,))),
            ("sample", lambda: fitted.sample(size=(4,))),
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
        assert all(isinstance(v, Scalar) for v in stats.values())
        assert all(v.ndim == 0 and v.shape == () for v in stats.values())


class TestMetrics:
    """Tests for loglikelihood, AIC, BIC."""

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    @pytest.mark.parametrize("metric", ["loglikelihood", "aic", "bic"])
    def test_metric(self, dist, metric, continuous_data, discrete_data):
        data = get_data(dist, continuous_data, discrete_data)
        params = dist.example_params()
        func = getattr(dist, metric)
        output = func(data, params)

        check_metric_output(dist, output, metric)
        jit(func)(data, params)
        gradients(func, f"{dist} {metric}", data, params)
