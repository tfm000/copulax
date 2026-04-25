"""Regression tests for the ``Distribution._resolve_params`` contract.

The contract (``copulax/_src/_distributions.py:_resolve_params``):

* Any public method whose signature is ``params: dict = None`` MUST call
  ``self._resolve_params(params)`` so that a *fitted* instance — one
  whose ``_stored_params`` returns a non-``None`` dict — can be invoked
  without re-passing the parameters.
* On an *unfitted* instance (no stored params) calling such a method
  without explicit ``params=`` MUST raise
  ``ValueError("No parameters provided. ...")`` so the user gets a clear
  error rather than a confusing crash deep in JAX.

These tests sweep every concrete distribution family (univariate,
multivariate normal-mixture, archimedean copula, mean-variance copula)
and verify both halves of the contract for every public method that
takes ``params: dict = None``.

The error-path checks deliberately call the *unwrapped* Python methods.
``_resolve_params`` raises at Python (trace) time rather than inside a
JIT-compiled or grad-traced computation, so wrapping the call in
``jax.jit`` / ``jax.grad`` would surface as a trace-time failure rather
than the documented ``ValueError``. Future contributors: do not be
tempted to "JIT-test" this path.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copulax._src.univariate._registry import _registry
from copulax.copulas import (
    amh_copula,
    clayton_copula,
    frank_copula,
    gaussian_copula,
    gh_copula,
    gumbel_copula,
    independence_copula,
    joe_copula,
    skewed_t_copula,
    student_t_copula,
)
from copulax.multivariate import mvt_gh, mvt_normal, mvt_skewed_t, mvt_student_t
from copulax._src.copulas._archimedean import IndependenceCopula


# ---------------------------------------------------------------------------
# Parametrisation sources
# ---------------------------------------------------------------------------

UNIVARIATE_DISTS = list(_registry)

MULTIVARIATE_DISTS = [mvt_normal, mvt_student_t, mvt_gh, mvt_skewed_t]
MULTIVARIATE_IDS = [d.name for d in MULTIVARIATE_DISTS]

ARCHIMEDEAN_COPULAS = [
    clayton_copula,
    frank_copula,
    gumbel_copula,
    joe_copula,
    amh_copula,
    independence_copula,
]
ARCHIMEDEAN_IDS = [c.name for c in ARCHIMEDEAN_COPULAS]

MV_COPULAS_PARAMS = [
    pytest.param(gaussian_copula, id=gaussian_copula.name),
    pytest.param(student_t_copula, id=student_t_copula.name),
    pytest.param(gh_copula, id=gh_copula.name, marks=pytest.mark.slow),
    pytest.param(skewed_t_copula, id=skewed_t_copula.name, marks=pytest.mark.slow),
]


# Methods on IndependenceCopula that are stateless by design — they
# evaluate to constants on the input ``u`` and never consult stored
# params, so they do *not* raise on an unfitted instance.
INDEPENDENCE_STATELESS_METHODS = frozenset({"copula_cdf", "copula_logpdf"})

# Pattern for the documented unfitted-error message.
UNFITTED_ERROR = re.compile(r"No parameters provided")

# Default PRNG key reused across paired sampling calls so the two
# invocations draw the same sample.
SAMPLE_KEY = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _structurally_equal(a, b) -> bool:
    """Recursive structural equality for arrays / scalars / dicts / tuples.

    Both invocations of a fitted-instance method trace the same JIT
    graph against the same stored params, so bit-equal output is the
    natural contract here. Anything looser would let subtle drift slide.
    """
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return False
        return all(_structurally_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        if not isinstance(b, type(a)) or len(a) != len(b):
            return False
        return all(_structurally_equal(x, y) for x, y in zip(a, b))
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape != b_arr.shape:
        return False
    # NaN-aware equality: positions that are NaN in both are equal.
    both_nan = np.isnan(a_arr) & np.isnan(b_arr)
    eq = (a_arr == b_arr) | both_nan
    return bool(np.all(eq))


def _call_pair(method, *args, params, **kwargs):
    """Invoke ``method`` with and without explicit ``params`` and assert
    the two results are structurally identical.
    """
    explicit = method(*args, params=params, **kwargs)
    implicit = method(*args, **kwargs)
    assert _structurally_equal(explicit, implicit), (
        f"{method.__qualname__}: explicit-params and stored-params "
        f"results differ.\nExplicit: {explicit!r}\nImplicit: {implicit!r}"
    )


def _assert_unfitted_raises(method, *args, **kwargs):
    """Assert calling ``method`` (no ``params=``) raises the documented error."""
    with pytest.raises(ValueError, match=UNFITTED_ERROR):
        method(*args, **kwargs)


def _univariate_test_x(fitted) -> jnp.ndarray:
    """Generate 5 sensible interior x-values via ``fitted.ppf`` (uses the
    analytical inverse CDF when available, otherwise the Chebyshev
    spline). Routing through ``ppf`` guarantees the points lie well
    inside the support for *any* distribution in the registry.
    """
    q = jnp.linspace(0.1, 0.9, 5)
    return fitted.ppf(q)


def _multivariate_test_x(d: int = 3, n: int = 6, seed: int = 0) -> jnp.ndarray:
    """Generate a small (n, d) array of standard-normal samples — adequate
    bulk coverage for any normal-mixture / Sklar-joint distribution we
    test, and identical across calls so paired comparisons are pure.
    """
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((n, d)))


def _copula_test_u(d: int = 3, n: int = 6) -> jnp.ndarray:
    """Generate a small (n, d) uniform array clipped away from the unit
    interval boundaries (avoids ppf blow-ups for heavy-tailed marginals).
    """
    rng = np.random.default_rng(1)
    return jnp.asarray(rng.uniform(0.05, 0.95, size=(n, d)))


# ---------------------------------------------------------------------------
# Univariate
# ---------------------------------------------------------------------------


class TestUnivariateResolveParams:
    """Verify ``_resolve_params`` for every univariate in ``_registry``."""

    @pytest.mark.parametrize(
        "dist", UNIVARIATE_DISTS, ids=[d.name for d in UNIVARIATE_DISTS]
    )
    def test_no_params_matches_explicit_params(self, dist):
        params = dist.example_params()
        fitted = dist._fitted_instance(params)
        stored = fitted._stored_params
        x = _univariate_test_x(fitted)
        q = jnp.linspace(0.1, 0.9, 5)

        # Density / distribution functions
        _call_pair(fitted.pdf, x, params=stored)
        _call_pair(fitted.logpdf, x, params=stored)
        _call_pair(fitted.cdf, x, params=stored)
        _call_pair(fitted.logcdf, x, params=stored)

        # Quantile / inverse CDF
        _call_pair(fitted.ppf, q, params=stored)
        _call_pair(fitted.inverse_cdf, q, params=stored)

        # Sampling — pass an explicit fixed key so both calls draw
        # bit-identical samples.
        _call_pair(fitted.rvs, 4, params=stored, key=SAMPLE_KEY)
        _call_pair(fitted.sample, 4, params=stored, key=SAMPLE_KEY)

        # Support / stats
        _call_pair(fitted.support, params=stored)
        _call_pair(fitted.stats, params=stored)

        # Likelihood-based metrics
        _call_pair(fitted.loglikelihood, x, params=stored)
        _call_pair(fitted.aic, x, params=stored)
        _call_pair(fitted.bic, x, params=stored)

        # Goodness-of-fit
        _call_pair(fitted.ks_test, x, params=stored)
        _call_pair(fitted.cvm_test, x, params=stored)

    @pytest.mark.parametrize(
        "dist", UNIVARIATE_DISTS, ids=[d.name for d in UNIVARIATE_DISTS]
    )
    def test_unfitted_raises(self, dist):
        unfitted = type(dist)(name="unfitted-test")
        assert unfitted._stored_params is None
        x = jnp.linspace(0.1, 0.9, 5)
        q = jnp.linspace(0.1, 0.9, 5)

        _assert_unfitted_raises(unfitted.pdf, x)
        _assert_unfitted_raises(unfitted.logpdf, x)
        _assert_unfitted_raises(unfitted.cdf, x)
        _assert_unfitted_raises(unfitted.logcdf, x)
        _assert_unfitted_raises(unfitted.ppf, q)
        _assert_unfitted_raises(unfitted.inverse_cdf, q)
        _assert_unfitted_raises(unfitted.rvs, 4, key=SAMPLE_KEY)
        _assert_unfitted_raises(unfitted.sample, 4, key=SAMPLE_KEY)
        _assert_unfitted_raises(unfitted.support)
        _assert_unfitted_raises(unfitted.loglikelihood, x)
        _assert_unfitted_raises(unfitted.aic, x)
        _assert_unfitted_raises(unfitted.bic, x)
        _assert_unfitted_raises(unfitted.ks_test, x)
        _assert_unfitted_raises(unfitted.cvm_test, x)


# ---------------------------------------------------------------------------
# Multivariate normal-mixture family
# ---------------------------------------------------------------------------


class TestMultivariateResolveParams:
    """Verify ``_resolve_params`` for the four multivariate normal-mixtures."""

    @pytest.mark.parametrize("dist", MULTIVARIATE_DISTS, ids=MULTIVARIATE_IDS)
    def test_no_params_matches_explicit_params(self, dist):
        params = dist.example_params(dim=3)
        fitted = dist._fitted_instance(params)
        stored = fitted._stored_params
        x = _multivariate_test_x(d=3)

        _call_pair(fitted.pdf, x, params=stored)
        _call_pair(fitted.logpdf, x, params=stored)
        _call_pair(fitted.rvs, 4, params=stored, key=SAMPLE_KEY)
        _call_pair(fitted.sample, 4, params=stored, key=SAMPLE_KEY)
        _call_pair(fitted.support, params=stored)
        _call_pair(fitted.stats, params=stored)
        _call_pair(fitted.loglikelihood, x, params=stored)
        _call_pair(fitted.aic, x, params=stored)
        _call_pair(fitted.bic, x, params=stored)

    @pytest.mark.parametrize("dist", MULTIVARIATE_DISTS, ids=MULTIVARIATE_IDS)
    def test_unfitted_raises(self, dist):
        unfitted = type(dist)(name="unfitted-test")
        assert unfitted._stored_params is None
        x = _multivariate_test_x(d=3)

        _assert_unfitted_raises(unfitted.pdf, x)
        _assert_unfitted_raises(unfitted.logpdf, x)
        _assert_unfitted_raises(unfitted.rvs, 4, key=SAMPLE_KEY)
        _assert_unfitted_raises(unfitted.sample, 4, key=SAMPLE_KEY)
        _assert_unfitted_raises(unfitted.support)
        _assert_unfitted_raises(unfitted.loglikelihood, x)
        _assert_unfitted_raises(unfitted.aic, x)
        _assert_unfitted_raises(unfitted.bic, x)


# ---------------------------------------------------------------------------
# Archimedean copulas
# ---------------------------------------------------------------------------


def _arch_dim(copula) -> int:
    """AMH is restricted to d=2; everything else uses d=3."""
    return 2 if copula.name == "AMH-Copula" else 3


class TestArchimedeanCopulaResolveParams:
    """Verify ``_resolve_params`` for the six Archimedean copulas."""

    @pytest.mark.parametrize("copula", ARCHIMEDEAN_COPULAS, ids=ARCHIMEDEAN_IDS)
    def test_no_params_matches_explicit_params(self, copula):
        d = _arch_dim(copula)
        params = copula.example_params(dim=d)
        fitted = copula._fitted_instance(params)
        stored = fitted._stored_params
        u = _copula_test_u(d=d)
        x = _multivariate_test_x(d=d)

        _call_pair(fitted.copula_cdf, u, params=stored)
        _call_pair(fitted.copula_pdf, u, params=stored)
        _call_pair(fitted.copula_logpdf, u, params=stored)
        _call_pair(fitted.copula_rvs, 4, params=stored, key=SAMPLE_KEY)
        _call_pair(fitted.copula_sample, 4, params=stored, key=SAMPLE_KEY)
        _call_pair(fitted.pdf, x, params=stored)
        _call_pair(fitted.logpdf, x, params=stored)
        _call_pair(fitted.rvs, 4, params=stored, key=SAMPLE_KEY)
        _call_pair(fitted.support, params=stored)
        _call_pair(fitted.get_u, x, params=stored)
        _call_pair(fitted.loglikelihood, x, params=stored)
        _call_pair(fitted.aic, x, params=stored)
        _call_pair(fitted.bic, x, params=stored)

    @pytest.mark.parametrize("copula", ARCHIMEDEAN_COPULAS, ids=ARCHIMEDEAN_IDS)
    def test_unfitted_raises(self, copula):
        d = _arch_dim(copula)
        unfitted = type(copula)(name="unfitted-test")
        assert unfitted._stored_params is None
        u = _copula_test_u(d=d)
        x = _multivariate_test_x(d=d)

        # Methods that always require resolved params (marginals or theta).
        _assert_unfitted_raises(unfitted.copula_rvs, 4, key=SAMPLE_KEY)
        _assert_unfitted_raises(unfitted.copula_sample, 4, key=SAMPLE_KEY)
        _assert_unfitted_raises(unfitted.pdf, x)
        _assert_unfitted_raises(unfitted.logpdf, x)
        _assert_unfitted_raises(unfitted.rvs, 4, key=SAMPLE_KEY)
        _assert_unfitted_raises(unfitted.support)
        _assert_unfitted_raises(unfitted.get_u, x)
        _assert_unfitted_raises(unfitted.loglikelihood, x)
        _assert_unfitted_raises(unfitted.aic, x)
        _assert_unfitted_raises(unfitted.bic, x)

        # ``copula_cdf`` / ``copula_logpdf`` consult ``theta`` for every
        # copula EXCEPT IndependenceCopula, which is genuinely stateless
        # — for it those two methods should return finite results on an
        # unfitted instance rather than raise.
        if isinstance(unfitted, IndependenceCopula):
            cdf_out = unfitted.copula_cdf(u)
            logpdf_out = unfitted.copula_logpdf(u)
            assert np.all(np.isfinite(np.asarray(cdf_out)))
            assert np.all(np.isfinite(np.asarray(logpdf_out)))
        else:
            _assert_unfitted_raises(unfitted.copula_cdf, u)
            _assert_unfitted_raises(unfitted.copula_logpdf, u)
            # ``copula_pdf`` is exp(copula_logpdf) so it raises by extension.
            _assert_unfitted_raises(unfitted.copula_pdf, u)


# ---------------------------------------------------------------------------
# Mean-variance copulas
# ---------------------------------------------------------------------------


class TestMVCopulaResolveParams:
    """Verify ``_resolve_params`` for the four mean-variance copulas.

    GH and SkewedT carry the ``slow`` mark via ``MV_COPULAS_PARAMS`` so
    the fast pair (Gaussian, Student-T) still runs under the default
    ``-m "not slow"`` invocation.
    """

    @pytest.mark.parametrize("copula", MV_COPULAS_PARAMS)
    def test_no_params_matches_explicit_params(self, copula):
        d = 3
        params = copula.example_params(dim=d)
        fitted = copula._fitted_instance(params)
        stored = fitted._stored_params
        u = _copula_test_u(d=d)
        x = _multivariate_test_x(d=d)

        _call_pair(
            fitted.copula_logpdf, u, params=stored, brent=False, nodes=100
        )
        _call_pair(
            fitted.copula_pdf, u, params=stored, brent=False, nodes=100
        )
        _call_pair(fitted.copula_rvs, 4, params=stored, key=SAMPLE_KEY)
        _call_pair(fitted.copula_sample, 4, params=stored, key=SAMPLE_KEY)
        _call_pair(fitted.pdf, x, params=stored)
        _call_pair(fitted.logpdf, x, params=stored)
        _call_pair(
            fitted.rvs, 4, params=stored, key=SAMPLE_KEY,
            brent=False, nodes=100,
        )
        _call_pair(fitted.support, params=stored)
        _call_pair(fitted.get_u, x, params=stored)
        _call_pair(fitted.loglikelihood, x, params=stored)
        _call_pair(fitted.aic, x, params=stored)
        _call_pair(fitted.bic, x, params=stored)

    @pytest.mark.parametrize("copula", MV_COPULAS_PARAMS)
    def test_unfitted_raises(self, copula):
        d = 3
        # Mean-variance copulas need their concrete ``_mvt`` / ``_uvt``
        # pair to be instantiable — pull them off the singleton.
        unfitted = type(copula)(
            name="unfitted-test", mvt=copula._mvt, uvt=copula._uvt
        )
        assert unfitted._stored_params is None
        u = _copula_test_u(d=d)
        x = _multivariate_test_x(d=d)

        _assert_unfitted_raises(
            unfitted.copula_logpdf, u, brent=False, nodes=100
        )
        _assert_unfitted_raises(
            unfitted.copula_pdf, u, brent=False, nodes=100
        )
        _assert_unfitted_raises(unfitted.copula_rvs, 4, key=SAMPLE_KEY)
        _assert_unfitted_raises(unfitted.copula_sample, 4, key=SAMPLE_KEY)
        _assert_unfitted_raises(unfitted.pdf, x)
        _assert_unfitted_raises(unfitted.logpdf, x)
        _assert_unfitted_raises(
            unfitted.rvs, 4, key=SAMPLE_KEY, brent=False, nodes=100,
        )
        _assert_unfitted_raises(unfitted.support)
        _assert_unfitted_raises(unfitted.get_u, x)
        _assert_unfitted_raises(unfitted.loglikelihood, x)
        _assert_unfitted_raises(unfitted.aic, x)
        _assert_unfitted_raises(unfitted.bic, x)
