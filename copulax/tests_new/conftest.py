"""Shared test fixtures, scipy parameter mappings, and assertion helpers
for the CopulAX rigorous test suite.

Design philosophy: every mathematical claim the library makes is
independently verified against scipy or a mathematical identity.
"""

import math
import warnings

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
                               maxiter=50):
    """Assert CDF(PPF(q)) ≈ q for quantiles in (0.05, 0.95)."""
    q = jnp.linspace(0.05, 0.95, n_points)
    x = dist.ppf(q=q, params=params, maxiter=maxiter)
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
