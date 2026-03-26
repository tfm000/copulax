"""Modified Bessel function of the second kind and Student's t CDF.

References:
    - Bessel function integral representation:
      https://dlmf.nist.gov/10.32#E10
    - Asymptotic forms:
      https://dlmf.nist.gov/10.30
"""

from jax import lax, vmap
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jax.scipy import special
import jax

from copulax._src.typing import Scalar


# -----------------------------------------------------------------------------
# Legacy kv implementation (adaptive quadax quadrature), retained for reference.
# -----------------------------------------------------------------------------
# def _kv_integrand(w: Array, v: float, x: Array) -> Array:
#     r"""Integrand for the integral representation of $K_v(x)$.
#
#     Uses the substitution $w = e^t$ in the standard integral
#     $K_v(x) = \frac{1}{2}\int_0^\infty w^{v-1} \exp(-x(w + w^{-1})/2)\,dw$.
#     """
#     frac = jnp.pow(w, -1)
#     inner = -0.5 * x * (w + frac)
#     exp = lax.exp(inner)
#     return 0.5 * lax.pow(w, v - 1.0) * exp
#
#
# def _kv_single_x(v: float, xi: float) -> float:
#     """Evaluate $K_v$ at a single scalar point via quadrature."""
#     from quadax import quadgk
#
#     kv_val, _ = quadgk(_kv_integrand, interval=(0.0, jnp.inf), args=(v, xi))
#     return kv_val.reshape(())
#
#
# def kv_legacy(v: float, x: ArrayLike) -> Array:
#     r"""Legacy adaptive-quadrature implementation of $K_v(x)$."""
#     v = jnp.asarray(jnp.abs(v), dtype=float)
#     x = jnp.asarray(x, dtype=float)
#     xshape = x.shape
#     x_flat = x.flatten()
#     kv_raw = vmap(lambda xi: _kv_single_x(v, xi))(x_flat)
#     kv_adj = jnp.where(x_flat < 0, jnp.nan, kv_raw)
#     return kv_adj.reshape(xshape)


# ---------------------------------------------------------------------------
# K_v(x) quadrature nodes and regime thresholds
# ---------------------------------------------------------------------------

# Gauss-Legendre nodes/weights for the main quadrature (DLMF 10.32.9).
_KV_GL_ORDER = 64
_KV_GL_NODES_NP, _KV_GL_WEIGHTS_NP = np.polynomial.legendre.leggauss(
    _KV_GL_ORDER
)
_KV_GL_NODES = jnp.asarray(_KV_GL_NODES_NP, dtype=float)
_KV_GL_WEIGHTS = jnp.asarray(_KV_GL_WEIGHTS_NP, dtype=float)

_KV_SMALL_X = jnp.asarray(1e-8, dtype=float)
_KV_LARGE_X = jnp.asarray(40.0, dtype=float)
_KV_EULER_GAMMA = jnp.asarray(0.5772156649015329, dtype=float)
_KV_LOG_2 = jnp.log(jnp.asarray(2.0, dtype=float))
_KV_LOG_PI = jnp.log(jnp.asarray(jnp.pi, dtype=float))


# ---------------------------------------------------------------------------
# Overflow-safe log(cosh(z))
# ---------------------------------------------------------------------------

def _log_cosh(z: Array) -> Array:
    r"""Numerically stable computation of log(cosh(z)).

    For |z| ≤ 20, computes ``log(cosh(z))`` directly.
    For |z| > 20, uses the identity

    .. math::

        \log\cosh z = |z| - \log 2 + \log(1 + e^{-2|z|})
                     \approx |z| - \log 2

    which avoids the overflow of ``cosh(z)`` that occurs for |z| > ~710
    in float64.
    """
    abs_z = jnp.abs(z)
    # Large-|z| branch: |z| - log(2), with small correction
    large = abs_z - _KV_LOG_2 + jnp.log1p(jnp.exp(-2.0 * abs_z))
    # Small-|z| branch: direct
    small = jnp.log(jnp.cosh(z))
    return jnp.where(abs_z > 20.0, large, small)


# ---------------------------------------------------------------------------
# Log-space regime-specific evaluation functions for log(K_v(x))
#
# Each function computes log(K_v(x)) directly, avoiding the exp() that
# causes underflow in K_v(x) for x >= ~710 in float64.
# ---------------------------------------------------------------------------

def _log_kv_small_x(v: Array, x: Array) -> Array:
    r"""Log of K_v(x) for small x (x < 1e-8), dispatching across three regimes.

    **Branch 1 — v < 10⁻⁴ (K₀ leading term, DLMF 10.31.2):**

    .. math::

        K_0(x) \approx -\log(x/2) - \gamma

    Accurate to < 4 × 10⁻⁸ relative error for v < 10⁻⁴.

    **Branch 2 — 10⁻⁴ ≤ v < 1 (two-term formula, DLMF 10.27.4):**

    .. math::

        K_v(x) \approx \frac{\pi}{2}
        \frac{(x/2)^{-v}/\Gamma(1-v) - (x/2)^{v}/\Gamma(v+1)}{\sin(v\pi)}

    This is the leading-order approximation from the definition
    K_v = (π/2)[I_{-v} - I_v]/sin(vπ).  Each I_v series is truncated
    to its m = 0 term, which is exact to O(x²).  For 10⁻⁴ ≤ v < 1 the
    two terms are well-separated (no catastrophic cancellation) and the
    formula achieves < 10⁻¹⁴ relative error against scipy.

    **Branch 3 — v ≥ 1 (single-term dominant, DLMF 10.30.2):**

    .. math::

        K_v(x) \approx \frac{\Gamma(v)}{2}\left(\frac{2}{x}\right)^v

    For v ≥ 1 the (x/2)^{-v} term dominates overwhelmingly and the
    single-term formula achieves < 10⁻¹⁴ relative error.
    """
    log_half_x = jnp.log(0.5 * x)

    def _k0_branch(_: None) -> Array:
        # K_0(x) ≈ -log(x/2) - γ, always positive for x < 1e-8
        return jnp.log(-log_half_x - _KV_EULER_GAMMA)

    def _two_term_branch(_: None) -> Array:
        # K_v(x) = (π/2) * [(x/2)^{-v}/Γ(1-v) - (x/2)^v/Γ(v+1)] / sin(vπ)
        # Compute each term in log space, then subtract.
        log_t1 = -v * log_half_x - special.gammaln(1.0 - v)
        log_t2 = v * log_half_x - special.gammaln(v + 1.0)
        # t1 > t2 always (the (x/2)^{-v} term dominates for x < 1e-8, v > 0)
        # Use log-subtract: log(t1 - t2) = log(t1) + log(1 - exp(log_t2 - log_t1))
        # = log_t1 + log1p(-exp(log_t2 - log_t1))
        diff = jnp.log1p(-jnp.exp(log_t2 - log_t1))
        return jnp.log(jnp.pi / 2.0) + log_t1 + diff - jnp.log(jnp.sin(v * jnp.pi))

    def _single_term_branch(_: None) -> Array:
        # K_v(x) ≈ Γ(v)/2 · (2/x)^v
        v_safe = jnp.maximum(v, jnp.asarray(1e-6, dtype=float))
        return (
            special.gammaln(v_safe)
            + (v_safe - 1.0) * _KV_LOG_2
            - v_safe * jnp.log(x)
        )

    def _nonzero_branch(_: None) -> Array:
        return lax.cond(v < 1.0, _two_term_branch, _single_term_branch, operand=None)

    return lax.cond(v < 1e-4, _k0_branch, _nonzero_branch, operand=None)


def _log_kv_large_x(v: Array, x: Array) -> Array:
    r"""Log of the large-x asymptotic expansion for K_v(x).

    .. math::

        \log K_v(x) \approx \tfrac{1}{2}(\log\pi - \log 2 - \log x) - x
                           + \log\!\bigl(\sum_{k=0}^{3} a_k / (8x)^k\bigr)

    where :math:`a_k = \prod_{j=0}^{k-1} (4v^2 - (2j+1)^2) / k!`.
    (DLMF 10.40.2)
    """
    mu = 4.0 * v * v
    inv8x = 1.0 / (8.0 * x)
    series = (
        1.0
        + (mu - 1.0) * inv8x
        + ((mu - 1.0) * (mu - 9.0)) * 0.5 * (inv8x ** 2)
        + ((mu - 1.0) * (mu - 9.0) * (mu - 25.0))
        * (1.0 / 6.0)
        * (inv8x ** 3)
    )
    log_pref = 0.5 * (_KV_LOG_PI - _KV_LOG_2 - jnp.log(x)) - x
    return log_pref + jnp.log(series)


def _log_kv_debye(v: Array, x: Array) -> Array:
    r"""Log of K_v(x) via the Debye uniform asymptotic expansion.

    Uses the uniform expansion (DLMF 10.41.3):

    .. math::

        K_v(v\,z) \sim \sqrt{\frac{\pi}{2v}}\,
        \frac{e^{-v\,\eta(z)}}{(1+z^2)^{1/4}}\,
        \sum_{k=0}^{5} \frac{(-1)^k\,U_k(p)}{v^k}

    where :math:`z = x/v`, :math:`p = 1/\sqrt{1+z^2}`, and
    :math:`\eta(z) = \sqrt{1+z^2} + \ln\!\bigl(z/(1+\sqrt{1+z^2})\bigr)`.

    The Debye polynomials :math:`U_k(p)` are given explicitly in
    DLMF 10.41.10 / Olver (1954).  Six terms (k=0..5) give ~14-digit
    accuracy for v ≥ 15.

    Returns ``log(K_v(x))`` directly, avoiding the ``exp(-v·η)`` that
    underflows for large v or x.

    References:
        DLMF §10.41; Olver, F.W.J. (1954) "The asymptotic expansion of
        Bessel functions of large order", Phil. Trans. R. Soc. A 247, 328-368.
    """
    z = x / jnp.maximum(v, 1e-30)
    z2 = z * z
    sqrt1z2 = jnp.sqrt(1.0 + z2)
    p = 1.0 / sqrt1z2

    # Debye phase eta(z) = sqrt(1+z^2) + ln(z / (1 + sqrt(1+z^2)))
    eta = sqrt1z2 + jnp.log(z / (1.0 + sqrt1z2))

    # Debye polynomials U_k(p) — coefficients from DLMF 10.41.10
    p2 = p * p
    p3 = p2 * p
    p4 = p2 * p2
    p5 = p4 * p
    p6 = p3 * p3
    p7 = p6 * p
    p8 = p4 * p4
    p9 = p8 * p
    p10 = p5 * p5
    p12 = p6 * p6
    p13 = p12 * p
    p15 = p12 * p3

    u0 = 1.0
    u1 = (3.0 * p - 5.0 * p3) / 24.0
    u2 = (81.0 * p2 - 462.0 * p4 + 385.0 * p6) / 1152.0
    u3 = (30375.0 * p3 - 369603.0 * p5 + 765765.0 * p7
           - 425425.0 * p9) / 414720.0
    u4 = (4465125.0 * p4 - 94121676.0 * p6 + 349922430.0 * p8
           - 446185740.0 * p10 + 185910725.0 * p12) / 39813120.0
    u5 = (1519035525.0 * p5 - 49286948607.0 * p7
           + 284499769554.0 * p9 - 614135872350.0 * p10 * p
           + 566098157625.0 * p13 - 188699385875.0 * p15) / 6688604160.0

    # Series with alternating signs: sum_{k=0}^5 (-1)^k U_k(p) / v^k
    inv_v = 1.0 / jnp.maximum(v, 1e-30)
    series = (u0
              - u1 * inv_v
              + u2 * inv_v ** 2
              - u3 * inv_v ** 3
              + u4 * inv_v ** 4
              - u5 * inv_v ** 5)

    # Log-space prefactor: avoids exp(-v*eta) underflow
    log_pref = (0.5 * (_KV_LOG_PI - _KV_LOG_2 - jnp.log(v))
                - v * eta
                - 0.25 * jnp.log(1.0 + z2))

    return log_pref + jnp.log(series)


def _log_kv_legendre(v: Array, x: Array) -> Array:
    r"""Log of K_v(x) via Gauss-Legendre quadrature on DLMF 10.32.9.

    .. math::

        K_v(x) = \int_0^\infty \cosh(v\,t)\,\exp(-x\,\cosh t)\,\mathrm{d}t

    The integrand is smooth and bounded for all v, x > 0 (no singularity),
    and decays exponentially for large t.

    The integration interval [t_lo, t_hi] adapts to the integrand peak:

    - *Saddle-centred* (sharp peak): when the integrand is dominated by a
      peak at the saddle point :math:`t^* = \mathrm{arcsinh}(v/x)`, the
      interval is centred on :math:`t^*`.
    - *Decay-based* (broad peak): when the peak is broad, the interval is
      ``[0, T_decay]`` where ``T_decay = log(92/x)``.

    Returns ``log(K_v(x))`` directly via the log-sum-exp trick, avoiding
    the ``exp(m)`` that underflows for large x.

    References:
        DLMF 10.32.9; Watson (1944) §6.22; Abramowitz & Stegun 9.6.24.
    """
    x_safe = jnp.maximum(x, 1e-10)

    # Decay-based interval: exp(-x*cosh(T)) < eps when T ≈ log(92/x)
    t_hi_decay = jnp.maximum(jnp.log(92.0 / x_safe), 10.0)

    # Saddle-point interval
    t_star = jnp.arcsinh(v / x_safe)
    cosh_tstar = jnp.cosh(t_star)
    peak_width = 1.0 / jnp.sqrt(jnp.maximum(x_safe * cosh_tstar, 1e-30))
    saddle_half = jnp.maximum(8.0 * peak_width, 4.0)

    # Use saddle-centering when the peak is sharp (half-width < decay interval)
    use_saddle = saddle_half < t_hi_decay

    # Saddle-centred bounds (regime A)
    t_lo_saddle = jnp.maximum(t_star - saddle_half, 0.0)
    t_hi_saddle = t_star + saddle_half

    # Decay bounds (regime B)
    t_lo_decay = jnp.asarray(0.0, dtype=float)

    t_lo = jnp.where(use_saddle, t_lo_saddle, t_lo_decay)
    t_hi = jnp.where(use_saddle, t_hi_saddle, t_hi_decay)

    # When x is small, the integrand at t=0 is cosh(0)*exp(-x) ≈ 1,
    # which is non-negligible.  Force t_lo = 0 so we don't miss this
    # contribution, even when saddle-centering pushes t_lo > 0.
    t_lo = jnp.where(x_safe < 20.0, 0.0, t_lo)

    # Map [-1, 1] -> [t_lo, t_hi]
    t = 0.5 * (t_hi - t_lo) * (_KV_GL_NODES + 1.0) + t_lo
    w = 0.5 * (t_hi - t_lo) * _KV_GL_WEIGHTS

    # Log-space evaluation for numerical stability.
    # _log_cosh prevents overflow of cosh(v*t) for large v*t.
    log_integrand = _log_cosh(v * t) + (-x * jnp.cosh(t))

    # Log-sum-exp: log(Σ w_i * exp(f_i)) = m + log(Σ w_i * exp(f_i - m))
    m = jnp.max(log_integrand)
    result = jnp.sum(w * jnp.exp(log_integrand - m))

    return m + jnp.log(result)


# ---------------------------------------------------------------------------
# Single-point dispatcher and public API
# ---------------------------------------------------------------------------

_KV_DEBYE_V_THRESH = jnp.asarray(15.0, dtype=float)


def _log_kv_single(v: Array, x: Array) -> Array:
    """Evaluate log(K_v(x)) at a single scalar point, dispatching across regimes.

    Regime boundaries:
    - v >= 15: Debye uniform asymptotic expansion (DLMF 10.41.3)
    - x < small_x_threshold: small-x asymptotic (DLMF 10.27.4 / 10.30 / 10.31)
    - x > large_x_threshold: large-x asymptotic (DLMF 10.40.2)
    - otherwise: Gauss-Legendre quadrature on DLMF 10.32.9

    Both thresholds are v-dependent:

    - **Small-x**: ``max(1e-8, v * 1e-5)``.  For large v the leading-term
      asymptotic K_v(x) ≈ Γ(v)/2·(2/x)^v has relative correction
      O(x² / (4(v+1))), which is negligible even at moderately small x.
      The wider threshold keeps the quadrature away from the regime where
      the integrand peak is too sharp for 64 nodes.
    - **Large-x**: ``max(40, 2v² + 20)``.  The 4-term Hankel series
      converges only when x >> v²/2.
    """
    x_pos = jnp.maximum(x, jnp.asarray(1e-30, dtype=float))

    # v-dependent thresholds
    small_x_thresh = jnp.maximum(_KV_SMALL_X, v * 1e-5)
    large_x_thresh = jnp.maximum(_KV_LARGE_X, 2.0 * v * v + 20.0)

    def _moderate_v(xi):
        """Dispatch for v < 15: quadrature or asymptotic."""
        return lax.cond(
            xi < small_x_thresh,
            lambda xj: _log_kv_small_x(v, xj),
            lambda xj: lax.cond(
                xj > large_x_thresh,
                lambda xk: _log_kv_large_x(v, xk),
                lambda xk: _log_kv_legendre(v, xk),
                xj,
            ),
            xi,
        )

    core = lax.cond(
        v >= _KV_DEBYE_V_THRESH,
        lambda xi: _log_kv_debye(v, xi),
        _moderate_v,
        x_pos,
    )
    core = jnp.where(x < 0.0, jnp.nan, core)
    core = jnp.where(x == 0.0, jnp.inf, core)
    return core


def log_kv(v: float, x: ArrayLike) -> Array:
    r"""Log of the modified Bessel function of the second kind, $\log K_v(x)$.

    Computes $\log K_v(x)$ directly in log space, avoiding the underflow
    that occurs in $K_v(x)$ for large $x$.  Since $K_v(x) \propto e^{-x}$,
    the value $K_v(x)$ underflows to zero in float64 for $x \gtrsim 710$.
    In contrast, $\log K_v(x) \approx -x + \text{small corrections}$ remains
    finite and accurate for arbitrarily large $x$.

    This is essential for the GIG distribution, whose moments involve
    the ratio $K_{v+1}(\sqrt{\chi\psi}) / K_v(\sqrt{\chi\psi})$.
    When $\chi\psi$ is large, computing this ratio as
    $\exp(\log K_{v+1}(r) - \log K_v(r))$ avoids the $0/0$ that would
    result from evaluating $K_{v+1}(r)$ and $K_v(r)$ individually.

    Pure JAX, JIT-compatible, and differentiable w.r.t. both *v* and *x*
    via JAX automatic differentiation.

    Four evaluation regimes, selected automatically:

    1. **v ≥ 15**: Debye uniform asymptotic expansion (DLMF 10.41.3),
       6-term series with Olver's polynomials.
    2. **x < 10⁻⁸**: small-x leading asymptotics (DLMF 10.30/10.31).
    3. **x > max(40, 2v²+20)**: large-x Hankel expansion (DLMF 10.40.2),
       4-term series.
    4. **Otherwise**: Gauss-Legendre quadrature (64 points) on the
       integral $K_v(x) = \int_0^\infty \cosh(vt)\,e^{-x\cosh t}\,dt$
       (DLMF 10.32.9), with saddle-point-centred integration interval.

    Args:
        v: Order (real, may be negative — K_{-v} = K_v).
        x: Argument (array-like, must be ≥ 0).

    Returns:
        Array of log(K_v(x)) values with the same shape as *x*.
    """
    v = jnp.asarray(jnp.abs(v), dtype=float).reshape(())
    x = jnp.asarray(x, dtype=float)
    xshape = x.shape
    x_flat = x.reshape(-1)
    vals = vmap(lambda xi: _log_kv_single(v, xi))(x_flat)
    return vals.reshape(xshape)


def kv(v: float, x: ArrayLike) -> Array:
    r"""Modified Bessel function of the second kind, $K_v(x)$.

    Convenience wrapper: ``kv(v, x) = exp(log_kv(v, x))``.

    For large *x* (≳ 710), ``kv`` underflows to zero while ``log_kv``
    remains finite.  Prefer ``log_kv`` when the result will be used
    in log-space arithmetic (e.g. log-likelihood computation).

    Pure JAX, JIT-compatible, and differentiable w.r.t. both *v* and *x*
    via JAX automatic differentiation.

    Args:
        v: Order (real, may be negative — K_{-v} = K_v).
        x: Argument (array-like, must be ≥ 0).

    Returns:
        Array of K_v(x) values with the same shape as *x*.
    """
    return jnp.exp(log_kv(v, x))


def kv_asymptotic(v: float, x: ArrayLike) -> Array:
    """Alias retained for backward compatibility."""
    return kv(v, x)


########################################################################
# igammainv / igammacinv implementation
########################################################################


def _igammainv_impl(a, p, q):
    """Core computation for igammainv.

    Finds x such that gammainc(a, x) = p, where p + q = 1.

    Both ``p`` and ``q`` are accepted so that callers can preserve full
    precision of whichever value they have directly (avoiding catastrophic
    cancellation in ``1 - p`` when p ≈ 1).

    Uses a hybrid initial approximation (Wilson-Hilferty for moderate p with
    a >= 1, log-space left-tail asymptotic otherwise) refined by 6 unrolled
    Newton-Halley iterations.

    References:
        Didonato, A. and Morris, A. (1986). Computation of the Incomplete
        Gamma Function Ratios and their Inverse.
        ACM Trans. Math. Softw. 12(4), 377-393.
    """
    TINY = jnp.finfo(a.dtype).tiny  # ~2.2e-308 for float64

    # --- Safe helpers (avoid log(0), ndtri(0), ndtri(1)) ---
    p_pos = jnp.maximum(p, TINY)
    q_pos = jnp.maximum(q, TINY)
    log_p = jnp.log(p_pos)

    # Normal quantile: use symmetry ndtri(1-q) = -ndtri(q) to always pass
    # the smaller of p, q to ndtri (avoiding precision loss near 1).
    s = jnp.where(
        p <= 0.5,
        special.ndtri(jnp.clip(p_pos, TINY, 0.5)),
        -special.ndtri(jnp.clip(q_pos, TINY, 0.5)),
    )

    # --- Initial approximation: Wilson-Hilferty (good for a >= 1, moderate p) ---
    a_safe = jnp.maximum(a, 1.0)
    t = 1.0 / (9.0 * a_safe)
    w = 1.0 - t + s * jnp.sqrt(t)
    x_wh = a * jnp.power(jnp.maximum(w, TINY), 3.0)

    # --- Initial approximation: log-space left-tail asymptotic (any a, small p) ---
    # From gammainc(a,x) ~ x^a / (a * Gamma(a)) for small x:
    #   x ~ (p * a * Gamma(a))^(1/a) = (p * Gamma(a+1))^(1/a)
    #   log(x) = (log(p) + gammaln(a+1)) / a
    log_x_left = (log_p + special.gammaln(a + 1.0)) / jnp.maximum(a, TINY)
    # Allow underflow to 0 for extremely small values (correct for a << 1)
    x_left = jnp.exp(jnp.minimum(log_x_left, 708.0))

    # --- Initial approximation: exponential-like for right tail ---
    # For a <= 1 with p close to 1, the left-tail asymptotic is wrong.
    # x ~ -log(q) is the exponential (a=1) exact formula and serves as a
    # reasonable starting point for a < 1 right-tail cases.
    x_exp = -jnp.log(q_pos)

    # --- Select initial guess ---
    # Wilson-Hilferty is dramatically better for moderate-to-large p with a >= 1
    # (up to 360,000x at a=100, p=0.5). But when the cube base w collapses
    # (extreme left tail), it fails and the log-space formula takes over.
    # For a < 1 with p > 0.5, the left-tail formula can break down for the
    # right tail (e.g. a=0.75, p=0.99: gives ~0.8 instead of ~3.3). But
    # x_exp = -log(q) can be wrong for very small a (e.g. a=0.01, p=0.9:
    # gives 2.3 instead of 1.5e-5). Neither heuristic is reliable across
    # all (a, p) pairs, so we evaluate gammainc once to pick the better
    # starting point.
    use_wh = (a >= 1.0) & (w > 0.2)
    x_left_safe = jnp.maximum(x_left, TINY)
    x_exp_safe = jnp.maximum(x_exp, TINY)
    err_left = jnp.abs(special.gammainc(a, x_left_safe) - p)
    err_exp = jnp.abs(special.gammainc(a, x_exp_safe) - p)
    use_right = (a < 1.0) & (p > 0.5) & (err_exp < err_left)
    x = jnp.where(use_wh, x_wh, x_left)
    x = jnp.where(use_right, x_exp, x)
    x = jnp.where(jnp.equal(a, 1.0), x_exp, x)
    x = jnp.maximum(x, TINY)

    # --- Newton-Halley refinement (6 iterations, unrolled) ---
    lgamma_a = special.gammaln(a)
    for _ in range(6):
        # fac = x^a * exp(-x) / Gamma(a)  (proportional to x * gamma_pdf)
        log_fac = a * jnp.log(jnp.maximum(x, TINY)) - x - lgamma_a
        fac = jnp.exp(log_fac)
        fac_safe = jnp.maximum(fac, TINY)

        # f / f' using gammainc for p <= 0.5, gammaincc for p > 0.5
        # (uses q directly in the upper branch to avoid cancellation)
        f_over_fprime = jnp.where(
            p <= 0.5,
            (special.gammainc(a, x) - p) * x / fac_safe,
            -(special.gammaincc(a, x) - q) * x / fac_safe,
        )

        # f'' / f' = -1 + (a - 1) / x  (guard overflow for tiny x)
        fprime2_over_fprime = jnp.where(
            x > 1e-100,
            -1.0 + (a - 1.0) / x,
            0.0,
        )

        # Halley step with safe denominator
        halley_denom = 1.0 - 0.5 * f_over_fprime * fprime2_over_fprime
        halley_denom = jnp.where(
            jnp.abs(halley_denom) < 1e-10, 1.0, halley_denom
        )
        step = f_over_fprime / halley_denom

        # Prevent step from making x negative
        step = jnp.maximum(step, -0.9 * x)

        x = jnp.where(fac > 0.0, x - step, x)
        x = jnp.maximum(x, TINY)

    # Boundary conditions
    x = jnp.where(p <= 0.0, 0.0, x)
    x = jnp.where(p >= 1.0, jnp.inf, x)

    return x


def igammainv(a: ArrayLike, p: ArrayLike) -> Array:
    r"""Inverse of the regularized lower incomplete gamma function.

    Finds $x$ such that $\mathrm{gammainc}(a, x) = p$.

    Args:
        a: positive shape parameter.
        p: probability values in $[0, 1]$.

    Returns:
        Array of the same shape as the broadcast of ``a`` and ``p``.
    """
    a = jnp.asarray(a, dtype=float)
    p = jnp.asarray(p, dtype=float)
    return _igammainv_impl(a, p, 1.0 - p)


def igammacinv(a: ArrayLike, p: ArrayLike) -> Array:
    r"""Inverse of the regularized upper incomplete gamma function.

    Finds $x$ such that $\mathrm{gammaincc}(a, x) = p$,
    equivalently $\mathrm{gammainc}(a, x) = 1 - p$.

    Args:
        a: positive shape parameter.
        p: probability values in $[0, 1]$.

    Returns:
        Array of the same shape as the broadcast of ``a`` and ``p``.
    """
    a = jnp.asarray(a, dtype=float)
    q = jnp.asarray(p, dtype=float)
    return _igammainv_impl(a, 1.0 - q, q)


########################################################################
# stdtr implementation
########################################################################
def _stdtr_impl(df: Scalar, t: Array) -> Array:
    """Primal Student-t CDF implementation."""
    # Use the regularized incomplete beta in a form that avoids
    # cancellation near t=0:
    #   CDF = 0.5 + 0.5 * sign(t) * I_z(1/2, df/2),
    #   z = t^2 / (df + t^2)
    t2 = t * t
    z = t2 / (df + t2)
    ib = special.betainc(0.5, 0.5 * df, z)
    return 0.5 + 0.5 * jnp.sign(t) * ib


def _stdtr_pdf_t(df: Scalar, t: Array) -> Array:
    """Derivative of stdtr(df, t) w.r.t. t (Student-t PDF at t)."""
    log_norm = (
        special.gammaln(0.5 * (df + 1.0))
        - special.gammaln(0.5 * df)
        - 0.5 * (jnp.log(df) + jnp.log(jnp.pi))
    )
    log_kernel = -0.5 * (df + 1.0) * jnp.log1p((t * t) / df)
    return jnp.exp(log_norm + log_kernel)


@jax.custom_vjp
def stdtr(df: Scalar, t: Array) -> Array:
    """Compute the cdf of the standard Student's t-distribution.

    Note:
        Gradient flow is supported for ``t``.
        Gradient flow for ``df`` is explicitly disabled (set to zero)
        because ``jax.scipy.special.betainc`` does not support gradients
        w.r.t. its first two arguments.

    Args:
        df (scalar): degrees of freedom.
        t (Array): values at which to evaluate the cdf.

    Returns:
        Array: cdf values of the standard Student's t-distribution.
    """
    # transforming args
    df: Scalar = jnp.asarray(df, dtype=float).reshape(())
    t: Array = jnp.asarray(t, dtype=float)
    return _stdtr_impl(df, t)


def _stdtr_fwd(df: Scalar, t: Array) -> tuple[Array, tuple[Scalar, Array]]:
    df = jnp.asarray(df, dtype=float).reshape(())
    t = jnp.asarray(t, dtype=float)
    y = _stdtr_impl(df, t)
    return y, (df, t)


def _stdtr_bwd(res: tuple[Scalar, Array], g: Array) -> tuple[Scalar, Array]:
    df, t = res
    pdf_t = _stdtr_pdf_t(df, t)
    d_df = jnp.zeros_like(df)
    d_t = g * pdf_t
    return d_df, d_t


stdtr.defvjp(_stdtr_fwd, _stdtr_bwd)
