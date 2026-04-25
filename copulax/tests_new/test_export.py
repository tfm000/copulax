"""Documents which user-facing CopulAX surfaces survive ``jax.export``.

``jax.export`` is JAX's AOT (ahead-of-time) compilation API: trace once,
lower to StableHLO, serialise to bytes for use in another process or on
another machine without Python source. This file probes every public,
user-facing method on every distribution / copula / preprocessing class
plus every public top-level utility function, locking in which surfaces
round-trip cleanly through ``export → serialize → deserialize`` and
which currently fail.

Tests that pass act as **regression guards**: a future change that
breaks export of a previously-working surface will fail loudly.

Currently failing paths use ``pytest.mark.xfail(strict=True,
raises=NotImplementedError)`` so they:

  * pass today (because they correctly fail), and
  * become loudly visible when a future JAX or CopulAX change makes
    them exportable, prompting a doc/source update.

Today's failure mode is paths that route through ``jax.pure_callback``
inside a JIT-compiled function — JAX's exporter does not yet implement
serialisation of host callbacks. CopulAX uses ``pure_callback`` only
inside ``copulax._src._utils.get_random_key`` to defeat trace-time seed
pollution when the user does not supply an explicit ``key``.

``flatbuffers`` is required for ``Exported.serialize()`` and is not a
runtime dependency of CopulAX, so the file ``importorskip``s it. To run
locally: ``uv pip install flatbuffers``.
"""

import inspect

import pytest

flatbuffers = pytest.importorskip("flatbuffers")

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


def _fit_accepts_method_kwarg(dist) -> bool:
    """True iff ``dist.fit`` accepts a ``method`` keyword argument.

    Closed-form / single-strategy fits (``gamma``, ``ig``, ``gig``,
    ``mvt_student_t``, etc.) define ``_supported_methods`` for
    advertising purposes but their ``fit`` signature does not include a
    ``method=`` kwarg. Calling ``fit(x, method='mle')`` against those
    raises ``TypeError`` — which is a test-harness issue, not an export
    issue.
    """
    sig = inspect.signature(dist.fit)
    return "method" in sig.parameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round_trip(fn, *arg_specs, static_argnames=()):
    """Export → serialize → deserialize. Returns (loaded_fn, blob_size)."""
    jitted = jax.jit(fn, static_argnames=static_argnames)
    exported = jax.export.export(jitted)(*arg_specs)
    blob = exported.serialize()
    return jax.export.deserialize(blob), len(blob)


# xfail is reserved for surfaces that fail in the keyless mode but have
# a documented working alternative — namely ``rvs(key=None)`` style calls
# whose canonical alternative is to pass an explicit ``key``. Any other
# failure is treated as a real bug surfaced by export and should fail
# loudly here so the source can be fixed.
_HOST_CB_REASON = (
    "rvs/sample with key=None routes through jax.pure_callback "
    "(get_random_key); jax.export does not yet serialise host_callbacks. "
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
    """``rvs(key=None)`` routes through pure_callback and currently fails."""

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
        # zero-arg JIT'd function is fine; it just bakes the constant
        _round_trip(lambda: getattr(dist, method)(params=params))


# ---------------------------------------------------------------------------
# Univariate — fit, parametrized over each supported method
# ---------------------------------------------------------------------------

def _uni_fit_cases():
    """Yield ``(dist_name, fit_method)`` per supported method.

    Distributions whose ``fit`` signature does not accept ``method=`` are
    yielded once with ``fit_method=None`` so the test calls ``fit(x)``
    plain. Failing cases (e.g. ``gh.fit('mle')`` and ``gh.fit('ldmle')``,
    which call ``get_random_key`` unconditionally inside ``_fit_mle`` /
    ``_fit_ldmle``) are *not* marked xfail — they fail as real bugs.
    """
    for name in UNIVARIATE_DISTS:
        dist = _get_uni(name)
        if not _fit_accepts_method_kwarg(dist):
            yield pytest.param(name, None)
            continue
        for m in sorted(dist._supported_methods):
            yield pytest.param(name, m)


@pytest.mark.parametrize("dist_name,fit_method", list(_uni_fit_cases()))
class TestUnivariateFit:
    """``fit(x, method=...)`` round-trips for each (dist, supported method) pair."""

    def test_round_trip(self, dist_name, fit_method):
        dist = _get_uni(dist_name)
        # Generate plausible data so the fit traces in the relevant branch
        rng = np.random.default_rng(0)
        if dist_name in {"gamma", "ig", "gig", "wald", "lognormal"}:
            x = jnp.asarray(rng.gamma(2.0, 1.0, 200))  # positive-support
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
    """Yield ``(dist_name, fit_method)``. Failing cases (e.g.
    ``mvt_gh.fit('ldmle')``, which calls ``get_random_key`` unconditionally
    in ``_ldmle_inputs``) are *not* marked xfail — they fail as real bugs.
    """
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


# independence_copula has no theta parameter — its generator is the
# trivial -log(t), so it would never be called in a parameterised
# fashion. Skip it for the generator/generator_inv parametrisation.
_THETA_ARCH_COPULAS = [c for c in ARCH_COPULAS if c != "independence_copula"]


@pytest.mark.parametrize("copula_name", _THETA_ARCH_COPULAS)
@pytest.mark.parametrize("method", ["generator", "generator_inv"])
class TestArchimedeanGenerators:
    """Archimedean ``generator``/``generator_inv`` are scalar functions.

    ``independence_copula`` is excluded — it has no ``theta`` parameter
    (its example_params['copula'] dict is empty), so the parametrised
    generator API does not apply.
    """

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
    """``fit_copula`` parametrised over ``(copula, method)``.

    Mean-variance copulas dispatch through Python-level marginal-fitting
    when called via the top-level ``fit``; ``fit_copula`` operates on
    pre-computed pseudo-observations and is the JIT-compatible entry
    point. Per the migration checklist (G-10), ``fit_copula`` is the
    correct surface to test for export.

    Methods on ``gh_copula`` / ``skewed_t_copula`` (``mle``, ``ecme``,
    ``ecme_double_gamma``, ``ecme_outer_gamma``) currently fail with
    ``ConcretizationTypeError`` from a Python-level ``float()`` of a
    traced value inside ``copulax._src.copulas._mom_init.mom_gh_params``
    (line 445) — a latent JIT bug surfaced here. They fail as real bugs
    rather than xfailing; the fix is to rewrite ``mom_gh_params`` using
    ``jax.lax.cond``/``while_loop``-style convergence (the same fix
    already applied to ``student_t_copula`` per migration checklist
    G-10).
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
        # signature: random_correlation(size, key=None) — pass explicit key
        _round_trip(lambda k: random_correlation(size=3, key=k), KEY_SPEC)

    def test_random_covariance_round_trip(self):
        from copulax.multivariate import random_covariance
        # vars must be positive; pass as a static array
        vars_arr = jnp.array([1.0, 2.0, 0.5])
        _round_trip(lambda k: random_covariance(vars=vars_arr, key=k), KEY_SPEC)


class TestUnivariateGofFunctions:
    """Module-level ``copulax.univariate.ks_test`` and ``cvm_test``.

    These are the standalone forms of the per-distribution goodness-of-fit
    methods — same signature ``f(x, dist, params)``, returning a dict with
    test statistic and p-value.
    """

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


# ``copulax.univariate.univariate_fitter`` and ``batch_univariate_fitter``
# are intentionally not JIT-compatible — they dispatch over a Python-level
# tuple of distribution objects (see migration checklist G-10) — so they
# cannot be exported. Not tested here.


# ---------------------------------------------------------------------------
# DataScaler preprocessing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scaler_method", ["zscore", "minmax", "robust", "maxabs"])
class TestDataScaler:
    """All four ``DataScaler`` modes: fit, transform, inverse_transform, fit_transform."""

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
        # Pre-fit on training data so the scaler is stateful, then export
        # the transform-only step (the typical inference-time pattern).
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
        # fit_transform returns (scaler, z); export only the array part
        _round_trip(
            lambda x: DataScaler(scaler_method).fit_transform(x)[1],
            x_spec,
        )
