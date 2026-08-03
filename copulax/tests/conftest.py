"""Shared test fixtures, scipy parameter mappings, and assertion helpers
for the CopulAX rigorous test suite.

Design philosophy: every mathematical claim the library makes is
independently verified against scipy or a mathematical identity.
"""

import importlib
import math
import os
import warnings
from types import ModuleType

import jax

# Enable float64 BEFORE any other JAX imports or tracing can occur.
# Must be at module level, not in a fixture, to ensure all JIT-compiled
# functions trace with float64 precision from the start.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special
import scipy.stats
from quadax import quadgk

from copulax.tests._timeseries_helpers import STANDARD, series, shared_fit
from copulax.univariate import normal, student_t

# ---------------------------------------------------------------------------
# Session-wide JAX configuration
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def _enable_x64():
    """Ensure float64 precision is enabled (belt-and-suspenders)."""
    jax.config.update("jax_enable_x64", True)
    yield


# ---------------------------------------------------------------------------
# PRNG key fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng_key():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Third-party oracle availability (strict oracle mode)
# ---------------------------------------------------------------------------

#: Environment variable that switches oracle imports from "skip when
#: missing" to "fail when missing".  Set in every CI leg that runs the
#: test suite, so a leg whose environment silently lost ``statsmodels``
#: or ``arch`` reports red instead of quietly dropping the
#: cross-validation coverage those packages provide.
STRICT_ORACLES_ENV = "COPULAX_STRICT_ORACLES"

#: Values of :data:`STRICT_ORACLES_ENV` that mean "not strict".  Any
#: other non-empty value enables strict mode, so a typo'd truthy value
#: (``"yes"``, ``"True"``, ``"2"``) errs towards enforcement rather than
#: towards silently skipping.
_FALSY_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})


def strict_oracles_enabled() -> bool:
    """Whether third-party oracle imports are mandatory.

    Returns
    -------
    bool
        ``True`` when the ``COPULAX_STRICT_ORACLES`` environment
        variable is set to anything other than ``""``, ``"0"``,
        ``"false"``, ``"no"`` or ``"off"`` (case-insensitive).
    """
    return os.environ.get(STRICT_ORACLES_ENV, "").strip().lower() \
        not in _FALSY_ENV_VALUES


def require_oracle(modname: str) -> ModuleType:
    """Import a third-party oracle package used for cross-validation.

    Drop-in replacement for ``pytest.importorskip`` at every
    ``statsmodels`` / ``arch`` call site.  Outside strict mode the
    semantics are unchanged (a missing oracle skips the dependent
    tests); under strict mode a missing oracle is a hard failure, so an
    environment that lost its dev extras cannot silently erase the
    independent verification those packages provide.

    Parameters
    ----------
    modname : str
        Fully-qualified module path to import, e.g. ``"arch"`` or
        ``"statsmodels.tsa.stattools"``.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    Failed
        If the module cannot be imported and strict oracle mode is
        active (see :func:`strict_oracles_enabled`).
    Skipped
        If the module cannot be imported and strict oracle mode is
        inactive.
    """
    try:
        return importlib.import_module(modname)
    except ImportError as exc:
        reason = f"{modname!r} could not be imported ({exc})"

    if strict_oracles_enabled():
        pytest.fail(
            f"oracle package missing under strict mode "
            f"({STRICT_ORACLES_ENV} is set): {reason}. Cross-validation "
            f"oracles are mandatory in this configuration -- install the "
            f"dev extras (pip install '.[dev]') or unset "
            f"{STRICT_ORACLES_ENV} to restore skip-on-missing behaviour.",
            pytrace=False,
        )
    pytest.skip(f"oracle unavailable: {reason}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Scipy parameter mapping infrastructure
# ---------------------------------------------------------------------------

def _copulax_to_scipy_normal(params):
    return scipy.stats.norm(loc=float(params["mu"]),
                            scale=float(params["sigma"]))


def _copulax_to_scipy_student_t(params):
    return scipy.stats.t(df=float(params["nu"]),
                         loc=float(params["mu"]),
                         scale=float(params["sigma"]))


def _copulax_to_scipy_gamma(params):
    # CopulAX Gamma uses rate parameterization: f(x) propto x^{a-1} exp(-b*x)
    # scipy.stats.gamma uses shape/scale: f(x) propto x^{a-1} exp(-x/scale)
    # so scale = 1/beta
    return scipy.stats.gamma(a=float(params["alpha"]),
                             scale=1.0 / float(params["beta"]))


def _copulax_to_scipy_exponential(params):
    # CopulAX Exponential is rate-parameterised: f(x) = lamb * exp(-lamb * x).
    # scipy.stats.expon uses scale = 1/lamb.
    return scipy.stats.expon(scale=1.0 / float(params["lamb"]))


def _copulax_to_scipy_lognormal(params):
    # CopulAX: X = exp(mu + sigma*Z), Z ~ N(0,1)
    # scipy.stats.lognorm: s=sigma (shape), scale=exp(mu)
    return scipy.stats.lognorm(s=float(params["sigma"]),
                               scale=np.exp(float(params["mu"])))


def _copulax_to_scipy_uniform(params):
    a, b = float(params["a"]), float(params["b"])
    return scipy.stats.uniform(loc=a, scale=b - a)


def _copulax_to_scipy_ig(params):
    # CopulAX IG (Inverse Gamma): f(x) propto x^{-alpha-1} exp(-beta/x)
    # scipy.stats.invgamma: a=alpha, scale=beta
    return scipy.stats.invgamma(a=float(params["alpha"]),
                                scale=float(params["beta"]))


def _copulax_to_scipy_gen_normal(params):
    # CopulAX GenNormal: params (mu, alpha, beta) where alpha=scale, beta=shape
    # scipy.stats.gennorm: beta=shape, loc=mu, scale=alpha
    return scipy.stats.gennorm(beta=float(params["beta"]),
                               loc=float(params["mu"]),
                               scale=float(params["alpha"]))


def _copulax_to_scipy_gig(params):
    # CopulAX GIG: lamb, chi, psi
    # scipy.stats.geninvgauss: p=lamb, b=sqrt(chi*psi), scale=sqrt(chi/psi)
    lam = float(params["lamb"])
    chi = float(params["chi"])
    psi = float(params["psi"])
    b = np.sqrt(chi * psi)
    scale = np.sqrt(chi / psi)
    return scipy.stats.geninvgauss(p=lam, b=b, loc=0, scale=scale)


def _copulax_to_scipy_wald(params):
    # CopulAX Wald (Inverse Gaussian): f(x) = sqrt(lamb/(2*pi*x^3)) * exp(-lamb*(x-mu)^2/(2*mu^2*x))
    # scipy.stats.invgauss uses f(x, mu_sp) with scale param; mapping: mu_sp = mu/lamb, scale = lamb
    mu_cx = float(params["mu"])
    lamb_cx = float(params["lamb"])
    return scipy.stats.invgauss(mu=mu_cx / lamb_cx, scale=lamb_cx)


def _copulax_to_scipy_nig(params):
    # CopulAX NIG: mu (loc), alpha (tail), beta (asymmetry), delta (scale)
    # scipy.stats.norminvgauss(a, b, loc, scale) with a=alpha*delta, b=beta*delta
    mu = float(params["mu"])
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    delta = float(params["delta"])
    return scipy.stats.norminvgauss(a=alpha * delta, b=beta * delta,
                                    loc=mu, scale=delta)


def _copulax_to_scipy_gh(params):
    # CopulAX GH: lamb, chi, psi, mu, sigma, gamma (McNeil 2005)
    # scipy genhyperbolic: p, a, b, loc, scale
    # Mapping for univariate:
    #   p = lamb
    #   delta = sigma * sqrt(chi)
    #   alpha = sqrt(psi + gamma^2/sigma^2) / sigma  (but need alpha*delta form)
    #   a = alpha * delta = sqrt(chi) * sqrt(psi + gamma^2/sigma^2)
    #         = sqrt(chi * psi + chi * gamma^2 / sigma^2)
    #   b = (gamma / sigma^2) * delta = gamma * sqrt(chi) / sigma
    #   loc = mu
    #   scale = delta = sigma * sqrt(chi)
    lam = float(params["lamb"])
    chi = float(params["chi"])
    psi = float(params["psi"])
    mu = float(params["mu"])
    sigma = float(params["sigma"])
    gamma = float(params["gamma"])

    delta = sigma * np.sqrt(chi)
    a = np.sqrt(chi * psi + chi * gamma ** 2 / sigma ** 2)
    b = gamma * np.sqrt(chi) / sigma
    return scipy.stats.genhyperbolic(p=lam, a=a, b=b, loc=mu, scale=delta)


# Central registry: CopulAX distribution name -> converter function
SCIPY_MAP = {
    "Normal": _copulax_to_scipy_normal,
    "Student-T": _copulax_to_scipy_student_t,
    "Gamma": _copulax_to_scipy_gamma,
    "Exponential": _copulax_to_scipy_exponential,
    "LogNormal": _copulax_to_scipy_lognormal,
    "Uniform": _copulax_to_scipy_uniform,
    "IG": _copulax_to_scipy_ig,
    "Gen-Normal": _copulax_to_scipy_gen_normal,
    "GIG": _copulax_to_scipy_gig,
    "GH": _copulax_to_scipy_gh,
    "NIG": _copulax_to_scipy_nig,
    "Wald": _copulax_to_scipy_wald,
}


def get_scipy_dist(dist, params):
    """Convert a CopulAX distribution + params to a frozen scipy dist.

    Returns None if no scipy equivalent is available.
    """
    converter = SCIPY_MAP.get(dist.name)
    if converter is None:
        return None
    return converter(params)


# ---------------------------------------------------------------------------
# Test point generation
# ---------------------------------------------------------------------------

def gen_test_points(dist, params, n=50):
    """Generate *n* test points spread across the distribution's support.

    Uses quantiles from the scipy equivalent when available, otherwise
    linspace within the support with a small margin.
    """
    support = np.array(dist._support(params)).flatten()
    lo, hi = float(support[0]), float(support[1])

    sp = get_scipy_dist(dist, params)
    if sp is not None:
        q = np.linspace(0.005, 0.995, n)
        pts = sp.ppf(q)
        # Filter out any non-finite values
        pts = pts[np.isfinite(pts)]
        if len(pts) >= n // 2:
            return jnp.array(pts)

    # Fallback: linspace within support
    if np.isinf(lo):
        lo = -50.0
    if np.isinf(hi):
        hi = 50.0
    margin = (hi - lo) * 0.01
    return jnp.linspace(lo + margin, hi - margin, n)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def assert_scipy_logpdf_match(dist, params, x, rtol=1e-6, atol=1e-10):
    """Assert CopulAX logpdf matches scipy logpdf at test points *x*.

    Raises pytest.skip if no scipy equivalent exists.
    """
    sp = get_scipy_dist(dist, params)
    if sp is None:
        pytest.skip(f"No scipy equivalent for {dist.name}")

    cx_vals = np.asarray(dist.logpdf(x=jnp.array(x), params=params)).flatten()
    sp_vals = sp.logpdf(np.asarray(x).flatten())

    # Only compare where both are finite (skip tails where both are -inf)
    mask = np.isfinite(sp_vals) & np.isfinite(cx_vals)
    if mask.sum() == 0:
        pytest.skip("No finite comparison points")

    np.testing.assert_allclose(
        cx_vals[mask], sp_vals[mask], rtol=rtol, atol=atol,
        err_msg=f"{dist.name} logpdf mismatch vs scipy"
    )


def assert_scipy_cdf_match(dist, params, x, rtol=1e-5, atol=1e-10):
    """Assert CopulAX CDF matches scipy CDF at test points *x*."""
    sp = get_scipy_dist(dist, params)
    if sp is None:
        pytest.skip(f"No scipy equivalent for {dist.name}")

    cx_vals = np.asarray(dist.cdf(x=jnp.array(x), params=params)).flatten()
    sp_vals = sp.cdf(np.asarray(x).flatten())

    mask = np.isfinite(sp_vals) & np.isfinite(cx_vals)
    if mask.sum() == 0:
        pytest.skip("No finite comparison points")

    np.testing.assert_allclose(
        cx_vals[mask], sp_vals[mask], rtol=rtol, atol=atol,
        err_msg=f"{dist.name} CDF mismatch vs scipy"
    )


def assert_pdf_integrates_to_one(dist, params, rtol=1e-3):
    """Verify that the PDF integrates to ~1 over the support via quadrature."""
    support = np.array(dist._support(params)).flatten()
    lo, hi = float(support[0]), float(support[1])

    def pdf_func(x_val):
        val = dist.pdf(x=jnp.array(x_val), params=params)
        return val.flatten()[0]

    result, _ = quadgk(pdf_func, interval=(lo, hi))
    np.testing.assert_allclose(
        float(result), 1.0, rtol=rtol,
        err_msg=f"{dist.name} PDF integrates to {float(result)}, not 1.0"
    )


def assert_inverse_consistency(dist, params, rtol=1e-3, n_points=20,
                               maxiter=50, brent=True, nodes=100):
    """Assert CDF(PPF(q)) ≈ q for quantiles in (0.05, 0.95).

    Defaults to ``brent=True`` (the machine-epsilon path) so inverse
    consistency can be checked at tight tolerances independent of
    cubic-spline discretisation.  Pass ``brent=False`` to verify the
    cubic path instead — use ``nodes`` to set the spline grid size.
    """
    q = jnp.linspace(0.05, 0.95, n_points)
    x = dist.ppf(q=q, params=params, brent=brent, nodes=nodes, maxiter=maxiter)
    q_recovered = dist.cdf(x=x, params=params).flatten()
    q_np = np.asarray(q)
    qr_np = np.asarray(q_recovered)

    mask = np.isfinite(qr_np) & np.isfinite(q_np)
    np.testing.assert_allclose(
        qr_np[mask], q_np[mask], rtol=rtol,
        err_msg=f"{dist.name} CDF(PPF(q)) != q"
    )


def assert_stats_match_scipy(dist, params, rtol=1e-5):
    """Assert stats() mean and variance match scipy equivalents."""
    sp = get_scipy_dist(dist, params)
    if sp is None:
        pytest.skip(f"No scipy equivalent for {dist.name}")

    cx_stats = dist.stats(params=params)
    sp_mean = sp.mean()
    sp_var = sp.var()

    if np.isfinite(sp_mean):
        np.testing.assert_allclose(
            float(cx_stats["mean"]), sp_mean, rtol=rtol,
            err_msg=f"{dist.name} mean mismatch"
        )

    if np.isfinite(sp_var) and sp_var > 0:
        np.testing.assert_allclose(
            float(cx_stats["variance"]), sp_var, rtol=rtol,
            err_msg=f"{dist.name} variance mismatch"
        )


# ---------------------------------------------------------------------------
# Generic helpers (ported from existing helpers.py)
# ---------------------------------------------------------------------------

def no_nans(output):
    return not np.any(np.isnan(np.asarray(output)))


def is_finite(output):
    return np.all(np.isfinite(np.asarray(output)))


def is_positive(output):
    return np.all(np.asarray(output) >= 0)


# ---------------------------------------------------------------------------
# Shared time-series series and fits
#
# The single home for every frozen series and every tiered fit that TWO
# OR MORE of the nine ``test_timeseries_*.py`` modules consume.  A module
# that is the only consumer of a series or a fit keeps its own fixture;
# the moment a second module wants the same thing, it belongs here.
#
# These fixtures are thin wrappers over
# ``copulax/tests/_timeseries_helpers.py``, which owns the frozen-corpus
# loader (:func:`series`), the process-wide fit registry
# (:func:`shared_fit`) and the tier definitions.  This module supplies
# only the pytest surface: what the fixture is called and which
# ``(series, model, tier)`` it names.
#
# Why function scope, and why no ``scope="module"`` / ``scope="session"``
# ----------------------------------------------------------------------
# The memoisation already happens one layer down and is *process*-wide:
# :func:`series` caches each converted array by name, and
# :func:`shared_fit` caches each fitted model under its full
# ``(tier, model signature, series, tag, fit arguments)`` key.  A fixture
# body here is therefore a dict lookup.  Adding a pytest-level scope on
# top would not save the fit — it would only add a *second*, narrower
# cache whose lifetime is a module or a session, which is precisely the
# sharing boundary the registry exists to remove.  Function scope keeps
# one authority for "has this been computed yet".
#
# The values handed out are safe to share: frozen jax arrays and frozen
# equinox PyTrees that every consumer only reads.
# ``_timeseries_helpers.assert_snapshot_intact`` is the tripwire that
# keeps that true, and
# ``test_timeseries_variance.py::TestSharedRegistryCrossModule`` pins the
# cross-module identity these fixtures depend on.
#
# Import cost
# -----------
# ``_timeseries_helpers`` (and through it the frozen corpus) is imported
# eagerly at the top of this file: measured at 2.8 ms / 3.7 MB, which is
# not worth deferring.  ``copulax.timeseries`` is NOT — it costs a
# measured 118 ms / 36 MB, and pytest loads this conftest for the whole
# test tree, including univariate-only sessions that never touch a
# time-series fixture.  The three fit fixtures below therefore import
# their model class in the fixture body.  ``copulax.univariate`` is
# eager because every test module in the tree imports it anyway, so it
# is free here.
# ---------------------------------------------------------------------------

#: AR(1), phi = 0.5, n = 500.  Consumed by ``test_timeseries_diagnostics``
#: (shape / JIT / plot / p-value-range / unit-root checks) and
#: ``test_timeseries_plotting`` (the joint ArmaGarch plot fits).
SERIES_AR1_P050_N500_S42 = "ar1_p050_n500_s42"

#: AR(1), phi = 0.6, n = 500.  Consumed by ``test_timeseries_diagnostics``
#: and ``test_timeseries_plotting``.
SERIES_AR1_P060_N500_S42 = "ar1_p060_n500_s42"

#: GARCH(1, 1), n = 500.  Consumed by ``test_timeseries_plotting`` and
#: ``test_timeseries_variance``.
SERIES_GARCH11_N500_S2 = "garch11_n500_s2"

#: GARCH(1, 1), n = 1000.  Consumed by ``test_timeseries_diagnostics``
#: and ``test_timeseries_variance``.
SERIES_GARCH11_N1000_S42 = "garch11_n1000_s42"

#: GARCH(1, 1), n = 1500.  Consumed by ``test_timeseries_diagnostics``
#: and ``test_timeseries_summary``.
SERIES_GARCH11_N1500_S42 = "garch11_n1500_s42"

#: GARCH(1, 1), n = 2000.  Consumed by ``test_timeseries_summary`` and
#: ``test_timeseries_variance``.
SERIES_GARCH11_N2000_S2 = "garch11_n2000_s2"


@pytest.fixture
def ar1_p050_n500_s42():
    """Frozen AR(1) series, phi = 0.5, n = 500.

    Returns
    -------
    jax.Array
        The series ``ar1_p050_n500_s42``, shape ``(500,)``.
    """
    return series(SERIES_AR1_P050_N500_S42)


@pytest.fixture
def ar1_p060_n500_s42():
    """Frozen AR(1) series, phi = 0.6, n = 500.

    Returns
    -------
    jax.Array
        The series ``ar1_p060_n500_s42``, shape ``(500,)``.
    """
    return series(SERIES_AR1_P060_N500_S42)


@pytest.fixture
def garch11_n500_s2():
    """Frozen GARCH(1, 1) innovation series, n = 500.

    Returns
    -------
    jax.Array
        The series ``garch11_n500_s2``, shape ``(500,)``.
    """
    return series(SERIES_GARCH11_N500_S2)


@pytest.fixture
def garch11_n1000_s42():
    """Frozen GARCH(1, 1) innovation series, n = 1000.

    Returns
    -------
    jax.Array
        The series ``garch11_n1000_s42``, shape ``(1000,)``.
    """
    return series(SERIES_GARCH11_N1000_S42)


@pytest.fixture
def garch11_n1500_s42():
    """Frozen GARCH(1, 1) innovation series, n = 1500.

    Returns
    -------
    jax.Array
        The series ``garch11_n1500_s42``, shape ``(1500,)``.
    """
    return series(SERIES_GARCH11_N1500_S42)


@pytest.fixture
def garch11_n2000_s2():
    """Frozen GARCH(1, 1) innovation series, n = 2000.

    Returns
    -------
    jax.Array
        The series ``garch11_n2000_s2``, shape ``(2000,)``.
    """
    return series(SERIES_GARCH11_N2000_S2)


@pytest.fixture
def ar1_p060_n500_s42_normal_fit_standard():
    """``AR(p=1)`` with Normal residuals on ``ar1_p060_n500_s42``, STANDARD.

    Consumed by ``test_timeseries_diagnostics`` (ARMA residual
    diagnostics) and ``test_timeseries_plotting`` (the mean-model plot
    surface).  Neither asserts on the location of the optimum, so the
    STANDARD budget applies.

    Returns
    -------
    copulax.timeseries.AR
        The fitted model — the registry's shared instance.
    """
    from copulax.timeseries import AR

    return shared_fit(
        AR(p=1, residual_dist=normal), SERIES_AR1_P060_N500_S42,
        tier=STANDARD,
    )


@pytest.fixture
def garch11_n500_s2_normal_fit_standard():
    """``GARCH(1, 1)`` with Normal residuals on ``garch11_n500_s2``, STANDARD.

    The most-shared fit in the family: ``test_timeseries_plotting`` uses
    it for the variance plot surface, ``test_timeseries_variance`` for
    the NumPy recursion reference and for the cross-module registry
    identity probe.

    Returns
    -------
    copulax.timeseries.GARCH
        The fitted model — the registry's shared instance.
    """
    from copulax.timeseries import GARCH

    return shared_fit(
        GARCH(p=1, q=1, residual_dist=normal), SERIES_GARCH11_N500_S2,
        tier=STANDARD,
    )


@pytest.fixture
def garch11_n2000_s2_student_t_fit_standard():
    """``GARCH(1, 1)`` with Student-T residuals on ``garch11_n2000_s2``.

    STANDARD tier.  Consumed by ``test_timeseries_summary`` (the
    residual-law standard-error keys) and ``test_timeseries_variance``
    (the residual-law swap smoke test).  Both are structural checks, so
    they share one fit.

    Note that ``test_timeseries_summary`` separately fits the same model
    and series at the PRECISION tier for its AD-vs-FD Hessian check.
    That is a different registry key and a different optimum, so it
    stays a single-module fit in that module.

    Returns
    -------
    copulax.timeseries.GARCH
        The fitted model — the registry's shared instance.
    """
    from copulax.timeseries import GARCH

    return shared_fit(
        GARCH(p=1, q=1, residual_dist=student_t), SERIES_GARCH11_N2000_S2,
        tier=STANDARD,
    )


@pytest.fixture
def arch_module():
    """The ``arch`` package, for GARCH-family cross-validation.

    Consumed by ``test_timeseries_variance`` (four cross-validation
    classes), ``test_timeseries_standard_errors`` and
    ``test_timeseries_summary``.  Skips the requesting test when ``arch``
    is unavailable, or fails it under strict oracle mode — see
    :func:`require_oracle`.

    Returns
    -------
    ModuleType
        The imported ``arch`` module.
    """
    return require_oracle("arch")
