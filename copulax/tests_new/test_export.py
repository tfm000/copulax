"""``jax.export`` round-trip coverage of user-facing CopulAX surfaces.

Probes every public method on every distribution / copula / preprocessing
class plus every public top-level utility through
``export → serialize → deserialize``. Passing tests act as regression
guards; xfail tests document surfaces that currently fail.

``flatbuffers`` is required for ``Exported.serialize()`` and is not a
runtime CopulAX dependency, so the file ``importorskip``s it.
"""

import inspect

import pytest

flatbuffers = pytest.importorskip("flatbuffers")

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


def _fit_accepts_method_kwarg(dist) -> bool:
    """True iff ``dist.fit`` accepts a ``method`` keyword argument."""
    return "method" in inspect.signature(dist.fit).parameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round_trip(fn, *arg_specs, static_argnames=()):
    """Export → serialize → deserialize. Returns (loaded_fn, blob_size)."""
    jitted = jax.jit(fn, static_argnames=static_argnames)
    exported = jax.export.export(jitted)(*arg_specs)
    blob = exported.serialize()
    return jax.export.deserialize(blob), len(blob)


# xfail is reserved for surfaces with a documented working alternative
# (``rvs(key=None)`` family — pass an explicit ``key`` to bypass).
_HOST_CB_REASON = (
    "rvs/sample with key=None routes through jax.pure_callback; "
    "jax.export does not yet serialise host_callbacks. "
    "Pass an explicit `key` to bypass."
)
_PURE_CB_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason=_HOST_CB_REASON,
)

KEY_SPEC = jax.ShapeDtypeStruct((), jr.key(0).dtype)


# ---------------------------------------------------------------------------
# Distribution registries
# ---------------------------------------------------------------------------

UNIVARIATE_DISTS = [
    "normal", "student_t", "uniform", "gamma", "lognormal", "ig", "gig",
    "gen_normal", "asym_gen_normal", "skewed_t", "gh", "nig", "wald",
]

MULTIVARIATE_DISTS = ["mvt_normal", "mvt_student_t", "mvt_gh", "mvt_skewed_t"]

MV_COPULAS = ["gaussian_copula", "student_t_copula", "gh_copula", "skewed_t_copula"]
ARCH_COPULAS = ["clayton_copula", "frank_copula", "gumbel_copula",
                "joe_copula", "amh_copula", "independence_copula"]
ALL_COPULAS = MV_COPULAS + ARCH_COPULAS


def _get_uni(name):
    import copulax.univariate as u
    return getattr(u, name)


def _get_mvt(name):
    import copulax.multivariate as m
    return getattr(m, name)


def _get_copula(name):
    import copulax.copulas as c
    return getattr(c, name)


# ---------------------------------------------------------------------------
# Univariate — methods that take (x, params)
# ---------------------------------------------------------------------------

UNI_DATA_METHODS = ["logpdf", "pdf", "cdf", "logcdf",
                    "loglikelihood", "aic", "bic", "ks_test", "cvm_test"]


@pytest.mark.parametrize("dist_name", UNIVARIATE_DISTS)
@pytest.mark.parametrize("method", UNI_DATA_METHODS)
class TestUnivariateDataMethods:
    """Methods of signature ``f(x, params) → array|dict|scalar``."""

    def test_round_trip(self, dist_name, method):
        dist = _get_uni(dist_name)
        params = dist.example_params()
        x_spec = jax.ShapeDtypeStruct((50,), jnp.float64)
        _round_trip(
            lambda x: getattr(dist, method)(x, params=params),
            x_spec,
        )


# ---------------------------------------------------------------------------
# Univariate — quantile methods (ppf, inverse_cdf)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", UNIVARIATE_DISTS)
@pytest.mark.parametrize("method", ["ppf", "inverse_cdf"])
class TestUnivariateQuantileMethods:
    """Quantile-function methods of signature ``f(q, params) → array``."""

    def test_round_trip(self, dist_name, method):
        dist = _get_uni(dist_name)
        params = dist.example_params()
        q_spec = jax.ShapeDtypeStruct((20,), jnp.float64)
        _round_trip(
            lambda q: getattr(dist, method)(q, params=params),
            q_spec,
        )


# ---------------------------------------------------------------------------
# Univariate — random sampling (rvs, sample) with explicit key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", UNIVARIATE_DISTS)
@pytest.mark.parametrize("method", ["rvs", "sample"])
class TestUnivariateRvsExplicitKey:
    """``f(size, params, key) → array`` with explicit key — bypasses callback."""

    def test_round_trip(self, dist_name, method):
        dist = _get_uni(dist_name)
        params = dist.example_params()
        _round_trip(
            lambda k: getattr(dist, method)(size=(20,), params=params, key=k),
            KEY_SPEC,
        )


@pytest.mark.parametrize("dist_name", UNIVARIATE_DISTS)
@pytest.mark.parametrize("method", ["rvs", "sample"])
class TestUnivariateRvsDefaultKey:
    """``rvs(key=None)`` — currently xfailed (host_callback)."""

    @_PURE_CB_XFAIL
    def test_round_trip(self, dist_name, method):
        dist = _get_uni(dist_name)
        params = dist.example_params()
        _round_trip(
            lambda: getattr(dist, method)(size=(20,), params=params, key=None),
        )


# ---------------------------------------------------------------------------
# Univariate — zero-arg methods (stats, support) with params baked in
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", UNIVARIATE_DISTS)
@pytest.mark.parametrize("method", ["stats", "support"])
class TestUnivariateZeroArgMethods:
    """``f(params) → dict|array`` with params as bake-in constant."""

    def test_round_trip(self, dist_name, method):
        dist = _get_uni(dist_name)
        params = dist.example_params()
        _round_trip(lambda: getattr(dist, method)(params=params))


# ---------------------------------------------------------------------------
# Univariate — fit, parametrized over each supported method
# ---------------------------------------------------------------------------

def _uni_fit_cases():
    """Yield ``(dist_name, fit_method)`` per supported method.  When
    ``fit`` does not accept a ``method`` kwarg, yields ``(name, None)``."""
    for name in UNIVARIATE_DISTS:
        dist = _get_uni(name)
        if not _fit_accepts_method_kwarg(dist):
            yield pytest.param(name, None)
            continue
        for m in sorted(dist._supported_methods):
            yield pytest.param(name, m)


@pytest.mark.parametrize("dist_name,fit_method", list(_uni_fit_cases()))
class TestUnivariateFit:
    """``fit(x, method=...)`` per (dist, supported method)."""

    def test_round_trip(self, dist_name, fit_method):
        dist = _get_uni(dist_name)
        rng = np.random.default_rng(0)
        if dist_name in {"gamma", "ig", "gig", "wald", "lognormal"}:
            x = jnp.asarray(rng.gamma(2.0, 1.0, 200))
        elif dist_name == "uniform":
            x = jnp.asarray(rng.uniform(-1.0, 2.0, 200))
        else:
            x = jnp.asarray(rng.normal(size=200))
        x_spec = jax.ShapeDtypeStruct(x.shape, jnp.float64)
        if fit_method is None:
            _round_trip(lambda data: dist.fit(data).params, x_spec)
        else:
            _round_trip(
                lambda data: dist.fit(data, method=fit_method).params,
                x_spec,
            )


# ---------------------------------------------------------------------------
# Multivariate — same shape / size methods, batched over distributions
# ---------------------------------------------------------------------------

MVT_DATA_METHODS = ["logpdf", "pdf", "loglikelihood", "aic", "bic"]


@pytest.mark.parametrize("dist_name", MULTIVARIATE_DISTS)
@pytest.mark.parametrize("method", MVT_DATA_METHODS)
class TestMultivariateDataMethods:
    """``f(x, params)`` for multivariate distributions; x shape ``(n, d)``."""

    def test_round_trip(self, dist_name, method):
        dist = _get_mvt(dist_name)
        params = dist.example_params(d=3)
        x_spec = jax.ShapeDtypeStruct((50, 3), jnp.float64)
        _round_trip(
            lambda x: getattr(dist, method)(x, params=params),
            x_spec,
        )


@pytest.mark.parametrize("dist_name", MULTIVARIATE_DISTS)
@pytest.mark.parametrize("method", ["rvs", "sample"])
class TestMultivariateRvsExplicitKey:
    def test_round_trip(self, dist_name, method):
        dist = _get_mvt(dist_name)
        params = dist.example_params(d=3)
        _round_trip(
            lambda k: getattr(dist, method)(size=20, params=params, key=k),
            KEY_SPEC,
        )


@pytest.mark.parametrize("dist_name", MULTIVARIATE_DISTS)
@pytest.mark.parametrize("method", ["rvs", "sample"])
class TestMultivariateRvsDefaultKey:
    @_PURE_CB_XFAIL
    def test_round_trip(self, dist_name, method):
        dist = _get_mvt(dist_name)
        params = dist.example_params(d=3)
        _round_trip(
            lambda: getattr(dist, method)(size=20, params=params, key=None),
        )


@pytest.mark.parametrize("dist_name", MULTIVARIATE_DISTS)
@pytest.mark.parametrize("method", ["stats", "support"])
class TestMultivariateZeroArgMethods:
    def test_round_trip(self, dist_name, method):
        dist = _get_mvt(dist_name)
        params = dist.example_params(d=3)
        _round_trip(lambda: getattr(dist, method)(params=params))


def _mvt_fit_cases():
    """Yield ``(dist_name, fit_method)`` per supported method."""
    for name in MULTIVARIATE_DISTS:
        dist = _get_mvt(name)
        if not _fit_accepts_method_kwarg(dist):
            yield pytest.param(name, None)
            continue
        for m in sorted(getattr(dist, "_supported_methods", None) or []):
            yield pytest.param(name, m)


@pytest.mark.parametrize("dist_name,fit_method", list(_mvt_fit_cases()))
class TestMultivariateFit:
    def test_round_trip(self, dist_name, fit_method):
        dist = _get_mvt(dist_name)
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.normal(size=(100, 3)))
        x_spec = jax.ShapeDtypeStruct(x.shape, jnp.float64)
        if fit_method is None:
            _round_trip(lambda data: dist.fit(data).params, x_spec)
        else:
            _round_trip(
                lambda data: dist.fit(data, method=fit_method).params,
                x_spec,
            )


# ---------------------------------------------------------------------------
# Copulas — joint x-space methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("copula_name", ALL_COPULAS)
@pytest.mark.parametrize("method", ["logpdf", "pdf", "loglikelihood", "aic", "bic"])
class TestCopulaJointMethods:
    """Copula joint x-space density / scalar-summary methods."""

    def test_round_trip(self, copula_name, method):
        cop = _get_copula(copula_name)
        params = cop.example_params(d=3)
        x_spec = jax.ShapeDtypeStruct((50, 3), jnp.float64)
        _round_trip(
            lambda x: getattr(cop, method)(x, params=params),
            x_spec,
        )


# ---------------------------------------------------------------------------
# Copulas — u-space methods (copula_logpdf, copula_pdf, copula_cdf)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("copula_name", ALL_COPULAS)
@pytest.mark.parametrize("method", ["copula_logpdf", "copula_pdf"])
class TestCopulaUSpaceDensities:
    def test_round_trip(self, copula_name, method):
        cop = _get_copula(copula_name)
        params = cop.example_params(d=3)
        u_spec = jax.ShapeDtypeStruct((50, 3), jnp.float64)
        _round_trip(
            lambda u: getattr(cop, method)(u, params=params),
            u_spec,
        )


@pytest.mark.parametrize("copula_name", ARCH_COPULAS)
class TestArchimedeanCopulaCDF:
    """Only Archimedean copulas expose ``copula_cdf``."""

    def test_round_trip(self, copula_name):
        cop = _get_copula(copula_name)
        params = cop.example_params(d=3)
        u_spec = jax.ShapeDtypeStruct((50, 3), jnp.float64)
        _round_trip(
            lambda u: cop.copula_cdf(u, params=params),
            u_spec,
        )


# ---------------------------------------------------------------------------
# Copulas — sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("copula_name", ALL_COPULAS)
@pytest.mark.parametrize("method", ["rvs", "sample", "copula_rvs", "copula_sample"])
class TestCopulaRvsExplicitKey:
    def test_round_trip(self, copula_name, method):
        cop = _get_copula(copula_name)
        params = cop.example_params(d=3)
        _round_trip(
            lambda k: getattr(cop, method)(size=20, params=params, key=k),
            KEY_SPEC,
        )


@pytest.mark.parametrize("copula_name", ALL_COPULAS)
@pytest.mark.parametrize("method", ["rvs", "sample", "copula_rvs", "copula_sample"])
class TestCopulaRvsDefaultKey:
    @_PURE_CB_XFAIL
    def test_round_trip(self, copula_name, method):
        cop = _get_copula(copula_name)
        params = cop.example_params(d=3)
        _round_trip(
            lambda: getattr(cop, method)(size=20, params=params, key=None),
        )


# ---------------------------------------------------------------------------
# Copulas — zero-arg methods + helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("copula_name", ALL_COPULAS)
@pytest.mark.parametrize("method", ["stats", "support"])
class TestCopulaZeroArgMethods:
    def test_round_trip(self, copula_name, method):
        cop = _get_copula(copula_name)
        params = cop.example_params(d=3)
        _round_trip(lambda: getattr(cop, method)(params=params))


@pytest.mark.parametrize("copula_name", ALL_COPULAS)
class TestCopulaGetU:
    """``get_u`` transforms x-space data to u-space pseudo-observations."""

    def test_round_trip(self, copula_name):
        cop = _get_copula(copula_name)
        params = cop.example_params(d=3)
        x_spec = jax.ShapeDtypeStruct((50, 3), jnp.float64)
        _round_trip(
            lambda x: cop.get_u(x, params=params),
            x_spec,
        )


@pytest.mark.parametrize("copula_name", MV_COPULAS)
class TestMvCopulaGetXDash:
    """``get_x_dash`` (MV-copulas only) inverts u-space back to x' space."""

    def test_round_trip(self, copula_name):
        cop = _get_copula(copula_name)
        params = cop.example_params(d=3)
        u_spec = jax.ShapeDtypeStruct((50, 3), jnp.float64)
        _round_trip(
            lambda u: cop.get_x_dash(u, params=params),
            u_spec,
        )


# ``independence_copula`` has no ``theta`` (generator is just -log(t));
# excluded from the parametrised ``generator``/``generator_inv`` tests.
_THETA_ARCH_COPULAS = [c for c in ARCH_COPULAS if c != "independence_copula"]


@pytest.mark.parametrize("copula_name", _THETA_ARCH_COPULAS)
@pytest.mark.parametrize("method", ["generator", "generator_inv"])
class TestArchimedeanGenerators:
    """Archimedean ``generator`` / ``generator_inv`` (scalar)."""

    def test_round_trip(self, copula_name, method):
        cop = _get_copula(copula_name)
        params = cop.example_params(d=3)
        theta = params["copula"]["theta"]
        scalar_spec = jax.ShapeDtypeStruct((), jnp.float64)
        _round_trip(
            lambda t: getattr(cop, method)(t, theta),
            scalar_spec,
        )


# ---------------------------------------------------------------------------
# Copulas — fit_copula per supported method (skip mean-variance — see notes)
# ---------------------------------------------------------------------------

def _copula_fit_cases():
    """``fit_copula`` parametrised over ``(copula, supported method)``.

    Targets ``fit_copula``, not the top-level ``fit`` (which dispatches
    over a Python tuple of distribution objects during marginal fitting
    and is intentionally not JIT-compatible).
    """
    for name in ALL_COPULAS:
        cop = _get_copula(name)
        for m in sorted(getattr(cop, "_supported_methods", None) or ["fc_mle"]):
            yield pytest.param(name, m)


@pytest.mark.parametrize("copula_name,fit_method", list(_copula_fit_cases()))
class TestCopulaFitCopula:
    """``fit_copula(u, method=...)`` per supported method."""

    def test_round_trip(self, copula_name, fit_method):
        cop = _get_copula(copula_name)
        rng = np.random.default_rng(0)
        u = jnp.asarray(rng.uniform(0.05, 0.95, (60, 3)))
        u_spec = jax.ShapeDtypeStruct(u.shape, jnp.float64)
        _round_trip(
            lambda data: cop.fit_copula(data, method=fit_method),
            u_spec,
        )


# ---------------------------------------------------------------------------
# Top-level utility / special / stats functions
# ---------------------------------------------------------------------------

class TestSpecialFunctions:
    """Public special-function re-exports under ``copulax.special``."""

    def test_kv_round_trip(self):
        from copulax.special import kv
        v_spec = jax.ShapeDtypeStruct((), jnp.float64)
        x_spec = jax.ShapeDtypeStruct((10,), jnp.float64)
        _round_trip(lambda v, x: kv(v, x), v_spec, x_spec)

    def test_log_kv_round_trip(self):
        from copulax.special import log_kv
        v_spec = jax.ShapeDtypeStruct((), jnp.float64)
        x_spec = jax.ShapeDtypeStruct((10,), jnp.float64)
        _round_trip(lambda v, x: log_kv(v, x), v_spec, x_spec)

    def test_stdtr_round_trip(self):
        from copulax.special import stdtr
        df_spec = jax.ShapeDtypeStruct((), jnp.float64)
        x_spec = jax.ShapeDtypeStruct((10,), jnp.float64)
        _round_trip(lambda df, x: stdtr(df, x), df_spec, x_spec)

    def test_igammainv_round_trip(self):
        from copulax.special import igammainv
        a_spec = jax.ShapeDtypeStruct((), jnp.float64)
        p_spec = jax.ShapeDtypeStruct((10,), jnp.float64)
        _round_trip(lambda a, p: igammainv(a, p), a_spec, p_spec)

    def test_igammacinv_round_trip(self):
        from copulax.special import igammacinv
        a_spec = jax.ShapeDtypeStruct((), jnp.float64)
        p_spec = jax.ShapeDtypeStruct((10,), jnp.float64)
        _round_trip(lambda a, p: igammacinv(a, p), a_spec, p_spec)

    def test_digamma_round_trip(self):
        from copulax.special import digamma
        x_spec = jax.ShapeDtypeStruct((10,), jnp.float64)
        _round_trip(lambda x: digamma(x), x_spec)

    def test_trigamma_round_trip(self):
        from copulax.special import trigamma
        x_spec = jax.ShapeDtypeStruct((10,), jnp.float64)
        _round_trip(lambda x: trigamma(x), x_spec)


class TestStatsFunctions:
    """Public stats helpers under ``copulax.stats``."""

    def test_skew_round_trip(self):
        from copulax.stats import skew
        x_spec = jax.ShapeDtypeStruct((100,), jnp.float64)
        _round_trip(lambda x: skew(x), x_spec)

    def test_kurtosis_round_trip(self):
        from copulax.stats import kurtosis
        x_spec = jax.ShapeDtypeStruct((100,), jnp.float64)
        _round_trip(lambda x: kurtosis(x), x_spec)


class TestMultivariateUtilities:
    """``corr``, ``cov``, ``random_correlation``, ``random_covariance``."""

    def test_corr_round_trip(self):
        from copulax.multivariate import corr
        x_spec = jax.ShapeDtypeStruct((100, 3), jnp.float64)
        _round_trip(lambda x: corr(x, method="pearson"), x_spec)

    def test_cov_round_trip(self):
        from copulax.multivariate import cov
        x_spec = jax.ShapeDtypeStruct((100, 3), jnp.float64)
        _round_trip(lambda x: cov(x, method="pearson"), x_spec)

    def test_random_correlation_round_trip(self):
        from copulax.multivariate import random_correlation
        _round_trip(lambda k: random_correlation(size=3, key=k), KEY_SPEC)

    def test_random_covariance_round_trip(self):
        from copulax.multivariate import random_covariance
        vars_arr = jnp.array([1.0, 2.0, 0.5])
        _round_trip(lambda k: random_covariance(vars=vars_arr, key=k), KEY_SPEC)


class TestUnivariateGofFunctions:
    """Module-level ``copulax.univariate.ks_test`` and ``cvm_test``."""

    @pytest.mark.parametrize("dist_name", UNIVARIATE_DISTS)
    def test_ks_test_round_trip(self, dist_name):
        from copulax.univariate import ks_test
        dist = _get_uni(dist_name)
        params = dist.example_params()
        x_spec = jax.ShapeDtypeStruct((50,), jnp.float64)
        _round_trip(
            lambda x: ks_test(x, dist, params),
            x_spec,
        )

    @pytest.mark.parametrize("dist_name", UNIVARIATE_DISTS)
    def test_cvm_test_round_trip(self, dist_name):
        from copulax.univariate import cvm_test
        dist = _get_uni(dist_name)
        params = dist.example_params()
        x_spec = jax.ShapeDtypeStruct((50,), jnp.float64)
        _round_trip(
            lambda x: cvm_test(x, dist, params),
            x_spec,
        )


# ``copulax.univariate.univariate_fitter`` / ``batch_univariate_fitter``
# dispatch over a Python tuple of distribution objects and are
# intentionally not JIT-compatible.  Not tested.


# ---------------------------------------------------------------------------
# DataScaler preprocessing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scaler_method", ["zscore", "minmax", "robust", "maxabs"])
class TestDataScaler:
    """``DataScaler`` ``fit`` / ``transform`` / ``inverse_transform`` / ``fit_transform``."""

    def _data(self):
        rng = np.random.default_rng(0)
        return jnp.asarray(rng.normal(size=(100, 4)))

    def test_fit_round_trip(self, scaler_method):
        from copulax.preprocessing import DataScaler
        x_spec = jax.ShapeDtypeStruct((100, 4), jnp.float64)
        _round_trip(
            lambda x: DataScaler(scaler_method).fit(x).offset,
            x_spec,
        )

    def test_transform_round_trip(self, scaler_method):
        from copulax.preprocessing import DataScaler
        scaler = DataScaler(scaler_method).fit(self._data())
        x_spec = jax.ShapeDtypeStruct((100, 4), jnp.float64)
        _round_trip(
            lambda x: scaler.transform(x),
            x_spec,
        )

    def test_inverse_transform_round_trip(self, scaler_method):
        from copulax.preprocessing import DataScaler
        scaler = DataScaler(scaler_method).fit(self._data())
        z_spec = jax.ShapeDtypeStruct((100, 4), jnp.float64)
        _round_trip(
            lambda z: scaler.inverse_transform(z),
            z_spec,
        )

    def test_fit_transform_round_trip(self, scaler_method):
        from copulax.preprocessing import DataScaler
        x_spec = jax.ShapeDtypeStruct((100, 4), jnp.float64)
        _round_trip(
            lambda x: DataScaler(scaler_method).fit_transform(x)[1],
            x_spec,
        )
