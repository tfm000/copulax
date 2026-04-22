r"""Method-of-moments initialisers for elliptical copula fitting.

Provides data-driven starting values for the EM/ML optimisers in
:mod:`copulax._src.copulas._distributions`, replacing arbitrary
hard-coded defaults (e.g. ``nu=5``).

Two estimators are implemented:

1. **Student-T / Skewed-T nu** — median-matching via Brent root-finding
   on :math:`h(\nu) = \text{median}(D^2/d) - F_{\text{median}}(d, \nu)`.

2. **GH (λ, χ, ψ)** — self-consistent iteration: project through GH PPF,
   extract raw mixing moments from :math:`D^{2k}`, match to GIG
   Bessel-function moments, damped update.

Both use the Kendall-tau correlation matrix *R* estimated in Stage 1 of the
copula fitting pipeline (independent of the mixing distribution).

Reference
---------
McNeil, Frey & Embrechts (2005), *QRM*, Chapter 5.5.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike
from jax import Array

from copulax._src.optimize import brent
from copulax._src.special import log_kv
from copulax._src.univariate.student_t import student_t
from copulax._src.univariate.gh import gh
from copulax._src.typing import Scalar

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _beta_median(a: Scalar, b: Scalar) -> Scalar:
    r"""Median of Beta(a, b) via Brent on ``betainc(a, b, x) - 0.5``.

    Args:
        a: First shape parameter (> 0).
        b: Second shape parameter (> 0).

    Returns:
        Scalar x such that ``I_x(a, b) = 0.5``.
    """
    def _residual(x, a, b):
        return jax.scipy.special.betainc(a, b, x) - 0.5

    return brent(
        g=_residual,
        bounds=jnp.array([1e-10, 1.0 - 1e-10]),
        maxiter=40,
        tol=1e-10,
        a=a,
        b=b,
    )


def _f_median(d: int, nu: Scalar) -> Scalar:
    r"""Median of the F(d, nu) distribution.

    Uses the relationship :math:`X \sim F(d, \nu)` iff
    :math:`Y = \frac{(d/\nu) X}{1 + (d/\nu) X} \sim \text{Beta}(d/2, \nu/2)`,
    so :math:`F_{\text{median}} = (\nu/d) \cdot y/(1-y)` where
    :math:`y` is the Beta median.

    Args:
        d: Numerator degrees of freedom (positive int).
        nu: Denominator degrees of freedom (> 0).

    Returns:
        Scalar median of F(d, nu).
    """
    d_f = jnp.asarray(d, dtype=float)
    y_med = _beta_median(d_f / 2.0, nu / 2.0)
    return (nu / d_f) * y_med / (1.0 - y_med)


def _bessel_ratio(nu: Scalar, k: int, omega: Scalar) -> Scalar:
    r"""Compute :math:`K_{\nu+k}(\omega) / K_\nu(\omega)` in log-space.

    Args:
        nu: Bessel order.
        k: Order increment (positive int).
        omega: Bessel argument (> 0).

    Returns:
        Scalar ratio.
    """
    log_num = log_kv(nu + k, omega)
    log_den = log_kv(nu, omega)
    return jnp.exp(log_num - log_den)


def _gig_normalized_moments(
    lamb: Scalar, omega: Scalar,
) -> tuple[Scalar, Scalar]:
    r"""Normalised second and third moments of GIG under E[W]=1.

    Given GIG(λ, χ, ψ) with :math:`\omega = \sqrt{\chi \psi}` and
    the constraint E[W]=1:

    .. math::
        m_k = \frac{r_k}{r_1^k}, \quad r_k = K_{\lambda+k}(\omega) / K_\lambda(\omega)

    Args:
        lamb: GIG lamb parameter.
        omega: :math:`\sqrt{\chi \psi}` (> 0).

    Returns:
        ``(m2, m3)`` — normalised moments.
    """
    r1 = _bessel_ratio(lamb, 1, omega)
    r2 = _bessel_ratio(lamb, 2, omega)
    r3 = _bessel_ratio(lamb, 3, omega)
    m2 = r2 / (r1 ** 2)
    m3 = r3 / (r1 ** 3)
    return m2, m3


# ---------------------------------------------------------------------------
# Estimator 1: Student-T / Skewed-T nu
# ---------------------------------------------------------------------------

def _h_nu(nu: Scalar, u_flat: Array, R_inv: Array, d: int, n: int) -> Scalar:
    r"""Root function for nu estimation.

    .. math::
        h(\nu) = \text{median}(D^2 / d) - F_{\text{median}}(d, \nu)

    where :math:`D^2_i = x_i^\top R^{-1} x_i` and
    :math:`x_i = t_\nu^{-1}(u_i)`.

    Args:
        nu: Candidate degrees of freedom.
        u_flat: Flattened pseudo-observations, shape ``(n * d,)``.
        R_inv: Inverse correlation matrix, shape ``(d, d)``.
        d: Dimensionality.
        n: Number of observations.

    Returns:
        Scalar h(nu).
    """
    params = student_t._params_dict(
        nu=nu, mu=jnp.array(0.0), sigma=jnp.array(1.0)
    )
    x_flat = student_t.ppf(u_flat, params=params)
    X = x_flat.reshape((n, d))
    D_sq = jnp.sum((X @ R_inv) * X, axis=1)
    return jnp.median(D_sq / d) - _f_median(d, nu)


def mom_nu_student_t(
    u: Array,
    R_inv: Array,
    d: int,
    nu_lo: float = 2.5,
    nu_hi: float = 200.0,
    tol: float = 0.1,
    maxiter: int = 30,
) -> Scalar:
    r"""Estimate Student-t / Skewed-T nu via median-matching bisection.

    Projects pseudo-observations through the Student-t PPF at candidate
    nu, computes squared Mahalanobis distances, and finds the nu where
    the empirical median of :math:`D^2/d` matches the theoretical
    F(d, nu) median.

    Uses a Python-level bisection loop (not ``jax.lax.scan``) to avoid
    nested JIT compilation with the PPF's own Brent solver.

    Args:
        u: Pseudo-observations, shape ``(n, d)`` in ``[0, 1]``.
        R_inv: Inverse correlation matrix, shape ``(d, d)``.
        d: Dimensionality.
        nu_lo: Lower bracket bound.
        nu_hi: Upper bracket bound.
        tol: Absolute convergence tolerance on nu.
        maxiter: Maximum bisection iterations.

    Returns:
        Estimated nu (scalar).  Falls back to boundary values if
        bracket fails.
    """
    n = u.shape[0]
    u_safe = jnp.clip(u, 1e-10, 1.0 - 1e-10)
    u_flat = u_safe.flatten()

    # Evaluate bracket endpoints
    h_lo = float(_h_nu(jnp.array(nu_lo), u_flat, R_inv, d, n))
    h_hi = float(_h_nu(jnp.array(nu_hi), u_flat, R_inv, d, n))

    # Bracket failure fallbacks
    if h_lo <= 0:
        return jnp.array(nu_lo)   # extremely heavy tails
    if h_hi >= 0:
        return jnp.array(nu_hi)   # near-Gaussian

    # Python-level bisection (each iteration calls JIT-compiled _h_nu)
    a, b = nu_lo, nu_hi
    for _ in range(maxiter):
        mid = 0.5 * (a + b)
        if (b - a) < tol:
            break
        h_mid = float(_h_nu(jnp.array(mid), u_flat, R_inv, d, n))
        if h_mid > 0:
            a = mid
        else:
            b = mid

    return jnp.array(0.5 * (a + b))


# ---------------------------------------------------------------------------
# Estimator 2: GH (lamb, chi, psi)
# ---------------------------------------------------------------------------

def _solve_gig_moments(
    m2_target: Scalar,
    m3_target: Scalar,
    lamb_init: Scalar,
    omega_init: Scalar,
    lr: float = 0.01,
    maxiter: int = 50,
) -> tuple[Scalar, Scalar]:
    r"""Find (lamb, omega) matching target normalised GIG moments.

    Minimises the sum of squared relative errors between theoretical
    and target moments via Adam gradient descent with box constraints.

    Uses a multi-start strategy: 7 starting points covering the major
    GH sub-families.  Returns the solution with lowest objective.

    Args:
        m2_target: Target normalised second moment.
        m3_target: Target normalised third moment.
        lamb_init: Warm-start lamb from previous iteration.
        omega_init: Warm-start omega from previous iteration.
        lr: Learning rate for Adam.
        maxiter: Number of Adam iterations per start.

    Returns:
        ``(lamb_hat, omega_hat)``.
    """
    from copulax._src.optimize import projected_gradient

    def objective(params):
        log_omega, lamb = params[0], params[1]
        omega = jnp.exp(log_omega)
        omega = jnp.maximum(omega, 1e-4)
        m2_th, m3_th = _gig_normalized_moments(lamb, omega)
        err2 = ((m2_th - m2_target) / jnp.maximum(m2_target, 1e-6)) ** 2
        err3 = ((m3_th - m3_target) / jnp.maximum(m3_target, 1e-6)) ** 2
        return err2 + err3

    starts = jnp.array([
        [jnp.log(jnp.maximum(omega_init, 0.01)), lamb_init],
        [jnp.log(1.0), -0.5],      # NIG
        [jnp.log(1.0),  1.0],      # Hyperbolic
        [jnp.log(0.5), -2.0],      # Heavy-tailed
        [jnp.log(2.0),  0.5],      # Moderate
        [jnp.log(0.1), -5.0],      # Very heavy-tailed
        [jnp.log(3.0),  2.0],      # Light-tailed
    ])

    bounds = {
        "lower": jnp.array([[jnp.log(0.01)], [-20.0]]),
        "upper": jnp.array([[jnp.log(50.0)], [20.0]]),
    }

    best_val = jnp.array(jnp.inf)
    best_params = starts[0]

    for i in range(starts.shape[0]):
        res = projected_gradient(
            f=objective,
            x0=starts[i],
            projection_method="projection_box",
            projection_options=bounds,
            lr=lr,
            maxiter=maxiter,
        )
        is_better = res["val"] < best_val
        best_params = jnp.where(is_better, res["x"], best_params)
        best_val = jnp.minimum(best_val, res["val"])

    return best_params[1], jnp.exp(best_params[0])  # lamb, omega


def mom_gh_params(
    u: Array,
    R_inv: Array,
    d: int,
    nu_hat: Scalar,
    max_iter: int = 30,
    alpha: float = 0.7,
) -> tuple[Scalar, Scalar, Scalar]:
    r"""Estimate GH (lamb, chi, psi) via self-consistent iteration.

    **Phase 1**: Initialise from Skewed-T boundary using ``nu_hat``.

    **Phase 2**: Iterate — project through GH PPF, extract raw mixing
    moments from squared Mahalanobis distances, match to GIG
    Bessel-function moments, damped update.

    Args:
        u: Pseudo-observations, shape ``(n, d)`` in ``[0, 1]``.
        R_inv: Inverse correlation matrix, shape ``(d, d)``.
        d: Dimensionality.
        nu_hat: Estimated nu from ``mom_nu_student_t``.
        max_iter: Maximum self-consistent iterations.
        alpha: Damping factor for parameter updates.

    Returns:
        ``(lamb_hat, chi_hat, psi_hat)``.
    """
    n = u.shape[0]
    u_safe = jnp.clip(u, 1e-10, 1.0 - 1e-10)
    d_f = float(d)

    # Chi-squared moment products: E[Q^k] = d*(d+2)*...*(d+2*(k-1))
    eq1 = d_f
    eq2 = d_f * (d_f + 2.0)
    eq3 = d_f * (d_f + 2.0) * (d_f + 4.0)

    # Phase 1: Skewed-T boundary
    lamb = -nu_hat / 2.0
    omega = jnp.array(0.01)

    for it in range(max_iter):
        lamb_old = lamb
        omega_old = omega

        # 1. Recover (chi, psi) from (lamb, omega) under E[W]=1
        r1 = _bessel_ratio(lamb, 1, jnp.maximum(omega, 1e-4))
        chi = omega / r1
        psi = omega * r1

        # Ensure positivity
        chi = jnp.maximum(chi, 1e-6)
        psi = jnp.maximum(psi, 1e-6)

        # 2. Project U -> X via PPF
        #    Compute both paths and select to avoid Python-level branching
        #    on traced values (which would break JIT).
        st_params = student_t._params_dict(
            nu=jnp.maximum(-2.0 * lamb, 2.01),
            mu=jnp.array(0.0),
            sigma=jnp.array(1.0),
        )
        X_st = student_t.ppf(
            u_safe.flatten(), params=st_params
        ).reshape((n, d))

        gh_params = gh._params_dict(
            lamb=lamb, chi=chi, psi=psi,
            mu=jnp.array(0.0), sigma=jnp.array(1.0),
            gamma=jnp.array(0.0),
        )
        X_gh = gh.ppf(
            u_safe.flatten(), params=gh_params, cubic=True
        ).reshape((n, d))

        use_student_t = psi < 0.01
        X = jnp.where(use_student_t, X_st, X_gh)

        # 3. Compute squared Mahalanobis distances
        D_sq = jnp.sum((X @ R_inv) * X, axis=1)

        # 4. Extract raw mixing moments (NO winsorisation)
        E_W1 = jnp.mean(D_sq) / eq1
        E_W2 = jnp.mean(D_sq ** 2) / eq2
        E_W3 = jnp.mean(D_sq ** 3) / eq3

        # 5. Normalise to E[W] = 1
        E_W1_safe = jnp.maximum(E_W1, 1e-8)
        m2_norm = jnp.maximum(E_W2 / E_W1_safe ** 2, 1.0 + 1e-6)
        m3_norm = E_W3 / E_W1_safe ** 3

        # 6. Solve for (lamb_new, omega_new)
        lamb_new, omega_new = _solve_gig_moments(
            m2_norm, m3_norm, lamb, omega
        )
        omega_new = jnp.maximum(omega_new, 0.01)

        # 7. Damped update
        lamb = alpha * lamb_new + (1.0 - alpha) * lamb
        omega = alpha * omega_new + (1.0 - alpha) * omega

        # 8. Convergence check
        if (abs(float(lamb - lamb_old)) < 0.05 and
                abs(float(omega - omega_old)) < 0.05):
            break

    # Final recovery
    r1 = _bessel_ratio(lamb, 1, jnp.maximum(omega, 1e-4))
    chi = jnp.maximum(omega / r1, 1e-6)
    psi = jnp.maximum(omega * r1, 1e-6)
    return lamb, chi, psi
