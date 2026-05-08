"""Bijective reparameterisations enforcing stationarity / positivity /
sign constraints across the time-series subpackage.

Each helper maps an unconstrained ``raw`` vector in :math:`\\mathbb{R}^k`
onto the constrained set the model requires.  The optimiser sees an
unconstrained problem, the recursion sees a feasible parameter vector,
and the inverse exists in closed form so warm-starts can be seeded
directly from a fitted ``params`` dict.

The module is JIT- and autograd-compatible end-to-end: every map is a
composition of element-wise operations, fixed-shape ``jnp`` reductions,
and Python-level loops over **static** orders ``p`` and ``q`` (which
parameterise the compiled graph and never change at runtime).

Citations are given alongside each reparameterisation; the inverses
are derived in the docstrings.
"""

from __future__ import annotations

import jax.nn as jnn
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


# Numerical guard.  Prevents ``arctanh`` overflow / ``log(0)`` when the
# inverse maps are evaluated on a near-boundary fitted point — e.g. an
# AR partial autocorrelation at :math:`\\pm 1`, an IGARCH persistence
# pinned at exactly 1, or a degenerate softmax weight at 0.
_BOUNDARY_EPS: float = 1e-6
# Denominator floor used in the backward Levinson recursion when a
# reflection coefficient lands at :math:`\\pm 1`.  Matches the AR/MA
# boundary clip so the forward and inverse paths share a tolerance.
_DENOM_EPS: float = 1e-6


###############################################################################
# Positive scalars (omega) — softplus / inverse-softplus
###############################################################################
def raw_to_positive(raw: ArrayLike) -> Array:
    r"""Map :math:`\\mathbb{R} \\to (0, \\infty)` via :math:`\\mathrm{softplus}`.

    Used for the GARCH intercept ``ω > 0`` and any other positivity
    constraint that does not need to be jointly bounded with other
    parameters.  Mirrors the existing CopulAX convention in
    ``copulax/_src/multivariate/mvt_gh.py:552``.
    """
    return jnn.softplus(jnp.asarray(raw, dtype=float))


def positive_to_raw(value: ArrayLike) -> Array:
    r"""Inverse of :func:`raw_to_positive` for warm starts.

    .. math::

        \\mathrm{raw} = \\log(e^{\\text{value}} - 1)
                     = \\text{value} + \\log\\!\\left(1 - e^{-\\text{value}}\\right)

    The right-hand form is numerically stable for ``value`` away from
    zero and we additionally floor ``value`` at :data:`_BOUNDARY_EPS`
    to keep the inverse defined right at the boundary ``value → 0``.
    """
    v = jnp.maximum(jnp.asarray(value, dtype=float), _BOUNDARY_EPS)
    return v + jnp.log(-jnp.expm1(-v))


###############################################################################
# AR / MA — partial autocorrelation reflection-coefficient reparam
###############################################################################
def reflection_to_ar(reflection: ArrayLike) -> Array:
    r"""Map a length-``p`` vector of reflection coefficients onto the
    stationary AR(p) coefficients via the Levinson-Durbin recursion.

    Reflection coefficients (partial autocorrelations) lie in
    :math:`(-1, 1)`.  Iterating

    .. math::

        \\phi^{(m+1)}_{m+1} &= \\rho_{m+1},\\\\
        \\phi^{(m+1)}_j     &= \\phi^{(m)}_j
                              - \\rho_{m+1}\\,\\phi^{(m)}_{m+1-j},
                              \\qquad j = 1, \\ldots, m

    for :math:`m = 0, \\ldots, p-1` produces an AR(p) polynomial whose
    roots all lie strictly outside the unit circle — equivalently, the
    AR process is causal / stationary.  The map is bijective onto the
    open stationarity region.

    The same machinery enforces MA invertibility (substitute
    ``θ`` for ``φ``); see :func:`reflection_to_ma`.

    Reference:
        Brockwell & Davis (1991), *Time Series: Theory and Methods*,
        §3.4 (eqn 3.4.11).  Burg (1968), Levinson (1947).

    Args:
        reflection: Array of shape ``(p,)`` with entries in ``(-1, 1)``.

    Returns:
        Array of shape ``(p,)`` containing the stationary AR
        coefficients :math:`(\\phi_1, \\ldots, \\phi_p)`.
    """
    reflection = jnp.asarray(reflection, dtype=float).reshape(-1)
    p = int(reflection.shape[0])
    if p == 0:
        return jnp.zeros((0,), dtype=reflection.dtype)
    phi = jnp.zeros((p,), dtype=reflection.dtype)
    for m in range(1, p + 1):
        rho_m = reflection[m - 1]
        if m > 1:
            head = phi[: m - 1] - rho_m * phi[: m - 1][::-1]
            phi = phi.at[: m - 1].set(head)
        phi = phi.at[m - 1].set(rho_m)
    return phi


def ar_to_reflection(phi: ArrayLike) -> Array:
    r"""Inverse of :func:`reflection_to_ar` (backward Levinson recursion).

    Given stationary AR coefficients :math:`\\phi^{(p)}`, recovers the
    reflection coefficients :math:`\\rho_1, \\ldots, \\rho_p` via

    .. math::

        \\rho_m            &= \\phi^{(m)}_m,\\\\
        \\phi^{(m-1)}_j    &= \\frac{\\phi^{(m)}_j
                                   + \\rho_m\\,\\phi^{(m)}_{m-j}}
                                  {1 - \\rho_m^2},
                              \\qquad j = 1, \\ldots, m-1

    iterating from ``m = p`` down to ``m = 1``.  The denominator is
    floored at :data:`_DENOM_EPS` to keep the inverse well-defined on
    near-unit-root warm starts; final reflections are *not* clipped
    here so the caller can decide whether to push through to
    :func:`raw_to_ar` (which does its own clip-and-arctanh) or
    consume the reflections directly.
    """
    phi = jnp.asarray(phi, dtype=float).reshape(-1)
    p = int(phi.shape[0])
    if p == 0:
        return jnp.zeros((0,), dtype=phi.dtype)
    reflection = jnp.zeros((p,), dtype=phi.dtype)
    for m in range(p, 0, -1):
        rho_m = phi[m - 1]
        reflection = reflection.at[m - 1].set(rho_m)
        if m > 1:
            denom = 1.0 - rho_m * rho_m
            sign = jnp.where(denom >= 0.0, 1.0, -1.0)
            denom = jnp.where(jnp.abs(denom) < _DENOM_EPS, sign * _DENOM_EPS, denom)
            new_head = (phi[: m - 1] + rho_m * phi[: m - 1][::-1]) / denom
            phi = phi.at[: m - 1].set(new_head)
    return reflection


def raw_to_ar(raw: ArrayLike) -> Array:
    r"""Compose ``raw → tanh → reflection → AR`` for the AR fit objective.

    The optimiser sees an unconstrained vector in :math:`\\mathbb{R}^p`;
    the recursion sees stationary AR coefficients.  Smooth and
    autograd-compatible end-to-end.
    """
    reflection = jnp.tanh(jnp.asarray(raw, dtype=float).reshape(-1))
    return reflection_to_ar(reflection)


def ar_to_raw(phi: ArrayLike) -> Array:
    r"""Inverse of :func:`raw_to_ar` for warm starts.

    Reflections are clipped to :math:`(-1 + \\varepsilon, 1 - \\varepsilon)`
    before taking ``arctanh`` (per plan boundary-clip rule —
    :data:`_BOUNDARY_EPS` = 1e-6).  Without the clip a warm start from
    a near-unit-root fit would push ``arctanh`` to ``±inf`` and break
    the trace.
    """
    reflection = ar_to_reflection(phi)
    reflection = jnp.clip(
        reflection, -1.0 + _BOUNDARY_EPS, 1.0 - _BOUNDARY_EPS
    )
    return jnp.arctanh(reflection)


# MA invertibility — same Levinson recursion as AR stationarity, but
# composed with a coefficient sign-flip to match CopulAX's ``+θ`` MA
# polynomial convention.
#
# Setup.  AR(p) uses the polynomial
#
#   .. math::  \\Phi(z) = 1 - \\phi_1 z - \\cdots - \\phi_p z^p
#
# while MA(q) uses
#
#   .. math::  \\Theta(z) = 1 + \\theta_1 z + \\cdots + \\theta_q z^q
#
# so the "subtract AR coefficients" / "add MA coefficients" sign
# convention differs between the two.  Substituting :math:`z = -w` in
# :math:`\\Theta(z)` gives
#
#   .. math::  \\Theta(-w) = 1 - \\theta_1 w + \\theta_2 w^2 - \\cdots
#                          = 1 - \\sum_j (-1)^{j+1} \\theta_j w^j,
#
# which is in AR form with coefficients
# :math:`\\tilde\\phi_j = (-1)^{j+1} \\theta_j`.  Roots of
# :math:`\\Theta(z)` and roots of the substituted polynomial differ by
# a sign (``z = -w``) but **share moduli**, so MA invertibility of
# :math:`\\theta` ⇔ AR stationarity of :math:`\\tilde\\phi`.
#
# The reparameterisation therefore:
#
# 1. Runs the Levinson recursion on the reflection coefficients to get
#    :math:`\\tilde\\phi` with :math:`1 - \\sum \\tilde\\phi_j w^j`
#    AR-stationary.
# 2. Recovers :math:`\\theta_j = (-1)^{j+1} \\tilde\\phi_j`, the
#    sign-flipped coefficients that make :math:`1 + \\sum \\theta_j z^j`
#    MA-invertible.
#
# The naive q ≥ 2 alias-to-AR path silently produces non-invertible θ
# (Levinson stationarity does not imply MA invertibility under the
# ``+θ`` convention); this composition is the correct fix.
def _ma_sign_flip(q: int) -> Array:
    r"""Length-``q`` array of :math:`(-1)^{j+1}` for :math:`j = 1..q`,
    indexed from 0 (so entry 0 is ``+1``, entry 1 is ``-1``, ...).

    The same array is its own inverse — the sign-flip is an involution
    — so ``reflection_to_ma`` and ``ma_to_reflection`` use the same
    pattern.
    """
    return jnp.power(-1.0, jnp.arange(q, dtype=float))


def reflection_to_ma(reflection: ArrayLike) -> Array:
    r"""Map a length-``q`` reflection-coefficient vector onto an
    invertible MA(q) coefficient vector.

    The map is :math:`\\rho \\mapsto \\theta = \\mathrm{sign\\_flip}(
    \\mathrm{Levinson}(\\rho))` where ``sign_flip`` applies
    :math:`(-1)^{j+1}` element-wise.  Bijective onto the open
    invertibility region :math:`\\{ \\theta : \\text{all roots of }
    1 + \\sum_j \\theta_j z^j \\text{ have } |z| > 1 \\}`.

    See the module-level discussion above :func:`reflection_to_ma` for
    the derivation.

    Args:
        reflection: Array of shape ``(q,)`` with entries in ``(-1, 1)``.

    Returns:
        Array of shape ``(q,)`` containing invertible MA coefficients
        :math:`(\\theta_1, \\ldots, \\theta_q)`.
    """
    reflection = jnp.asarray(reflection, dtype=float).reshape(-1)
    q = int(reflection.shape[0])
    if q == 0:
        return jnp.zeros((0,), dtype=reflection.dtype)
    phi = reflection_to_ar(reflection)
    return _ma_sign_flip(q) * phi


def ma_to_reflection(theta: ArrayLike) -> Array:
    r"""Inverse of :func:`reflection_to_ma`.

    Recovers reflection coefficients from invertible MA coefficients by
    sign-flipping back to AR form and running backward Levinson.
    """
    theta = jnp.asarray(theta, dtype=float).reshape(-1)
    q = int(theta.shape[0])
    if q == 0:
        return jnp.zeros((0,), dtype=theta.dtype)
    phi = _ma_sign_flip(q) * theta
    return ar_to_reflection(phi)


def raw_to_ma(raw: ArrayLike) -> Array:
    r"""Compose ``raw → tanh → reflection → MA`` for the MA fit
    objective.  Smooth and autograd-compatible end-to-end; produces θ
    guaranteed to make :math:`1 + \\sum_j \\theta_j z^j` invertible.
    """
    reflection = jnp.tanh(jnp.asarray(raw, dtype=float).reshape(-1))
    return reflection_to_ma(reflection)


def ma_to_raw(theta: ArrayLike) -> Array:
    r"""Inverse of :func:`raw_to_ma` for warm starts.  Reflections are
    clipped to :math:`(-1 + \\varepsilon, 1 - \\varepsilon)` before
    ``arctanh`` to keep the trace finite at near-boundary fits.
    """
    reflection = ma_to_reflection(theta)
    reflection = jnp.clip(
        reflection, -1.0 + _BOUNDARY_EPS, 1.0 - _BOUNDARY_EPS
    )
    return jnp.arctanh(reflection)


###############################################################################
# GARCH-family simplex reparameterisations
###############################################################################
def garch_simplex(
    raw_persistence: ArrayLike,
    raw_weights: ArrayLike,
    p: int,
) -> tuple[Array, Array]:
    r"""Simplex split for vanilla GARCH(p, q) parameters.

    Lets persistence ``s = sigmoid(raw_persistence) ∈ (0, 1)`` and
    distributes it across the ``(p + q)``-element simplex via
    ``softmax``:

    .. math::

        s &= \\sigma(r_s),
        & w &= \\mathrm{softmax}(r_w),\\\\
        (\\alpha_1, \\ldots, \\alpha_p, \\beta_1, \\ldots, \\beta_q)
            &= s \\cdot w.

    The map is differentiable, smooth, and avoids any JAX-side
    projection step — :math:`\\sum \\alpha_i + \\sum \\beta_j = s
    \\in (0, 1)` by construction, so the GARCH stationarity
    constraint is satisfied identically.

    Args:
        raw_persistence: scalar in :math:`\\mathbb{R}`.
        raw_weights: shape ``(p + q,)`` in :math:`\\mathbb{R}`.
        p: Number of α-coefficients (must be a static Python ``int``).

    Returns:
        Tuple ``(alpha, beta)`` of shape ``(p,)`` and ``(q,)``,
        non-negative, with ``Σα + Σβ < 1``.
    """
    raw_weights = jnp.asarray(raw_weights, dtype=float).reshape(-1)
    s = jnn.sigmoid(jnp.asarray(raw_persistence, dtype=float).reshape(()))
    w = jnn.softmax(raw_weights)
    chunks = s * w
    return chunks[:p], chunks[p:]


def garch_unsimplex(
    alpha: ArrayLike, beta: ArrayLike,
) -> tuple[Array, Array]:
    r"""Inverse of :func:`garch_simplex` for warm starts.

    .. math::

        s             &= \\sum_i \\alpha_i + \\sum_j \\beta_j,\\\\
        r_s           &= \\mathrm{logit}(s),\\\\
        r_w           &= \\log\\bigl((\\alpha, \\beta) / s\\bigr).

    ``softmax`` is shift-invariant so taking the element-wise log of
    the normalised weight vector is one valid inverse — the optimiser
    sees an equivalent unconstrained start.  ``s`` is clipped away
    from 0 / 1 by :data:`_BOUNDARY_EPS` to keep ``logit`` and
    :math:`\\log` defined.
    """
    alpha = jnp.asarray(alpha, dtype=float).reshape(-1)
    beta = jnp.asarray(beta, dtype=float).reshape(-1)
    s = jnp.sum(alpha) + jnp.sum(beta)
    s = jnp.clip(s, _BOUNDARY_EPS, 1.0 - _BOUNDARY_EPS)
    raw_persistence = jnp.log(s) - jnp.log1p(-s)
    weights = jnp.concatenate([alpha, beta]) / s
    weights = jnp.maximum(weights, _BOUNDARY_EPS)
    raw_weights = jnp.log(weights)
    return raw_persistence, raw_weights


def igarch_simplex(
    raw_weights: ArrayLike, p: int,
) -> tuple[Array, Array]:
    r"""Simplex split for IGARCH(p, q) — persistence pinned to 1.

    .. math::

        (\\alpha_1, \\ldots, \\alpha_p, \\beta_1, \\ldots, \\beta_q)
            = \\mathrm{softmax}(r_w),
        \\qquad
        \\sum_i \\alpha_i + \\sum_j \\beta_j = 1.

    The intercept :math:`\\omega > 0` is parameterised separately via
    :func:`raw_to_positive`.  Standard IGARCH treatment in both
    ``arch`` and ``rugarch``.
    """
    raw_weights = jnp.asarray(raw_weights, dtype=float).reshape(-1)
    w = jnn.softmax(raw_weights)
    return w[:p], w[p:]


def igarch_unsimplex(
    alpha: ArrayLike, beta: ArrayLike,
) -> Array:
    r"""Inverse of :func:`igarch_simplex` for warm starts."""
    alpha = jnp.asarray(alpha, dtype=float).reshape(-1)
    beta = jnp.asarray(beta, dtype=float).reshape(-1)
    weights = jnp.concatenate([alpha, beta])
    weights = jnp.maximum(weights, _BOUNDARY_EPS)
    return jnp.log(weights)


def gjr_simplex(
    raw_persistence: ArrayLike,
    raw_weights: ArrayLike,
    p: int,
    q: int,
    kappa: ArrayLike,
) -> tuple[Array, Array, Array]:
    r"""Simplex split for GJR-GARCH(p, q) under a (possibly skewed)
    standardised residual law.

    Persistence on the σ²-recursion is

    .. math::

        s = \\sum_i \\alpha_i + \\kappa \\sum_i \\gamma_i + \\sum_j \\beta_j,
        \\qquad
        \\kappa = \\mathbb{E}[z^2 \\mathbf{1}\\{z < 0\\}],

    the *truncated second moment* of the standardised residual.  Under
    symmetric residuals :math:`\\kappa = 1/2` and the textbook
    ``Σα + Σγ/2 + Σβ < 1`` is recovered; under skewed residuals
    :math:`\\kappa \\neq P(z<0)`, so substituting one for the other
    understates persistence and is wrong — see plan §"Stationarity"
    for the worked derivation.

    Implementation: split ``s = sigmoid(raw_persistence) ∈ (0, 1)``
    across the ``(2p + q)``-simplex ``softmax(raw_weights)``, then
    divide the γ-chunk by ``κ`` so the recursion sees raw ``γ_i``:

    .. math::

        (s w)_{1:p}        &= \\alpha,\\\\
        (s w)_{p+1:2p}     &= \\kappa \\gamma,\\\\
        (s w)_{2p+1:2p+q}  &= \\beta.

    ``κ`` is computed elsewhere (typically inside the fit objective
    via ``quadax.quadgk`` integrating ``z^2 · pdf_std(z)`` over
    :math:`(-\\infty, 0)`); gradients flow through ``κ`` into the
    residual's shape parameters automatically.  ``κ`` is floored at
    :data:`_BOUNDARY_EPS` to keep the division defined when the
    optimiser briefly visits a degenerate-residual configuration.

    Args:
        raw_persistence: scalar in :math:`\\mathbb{R}`.
        raw_weights: shape ``(2p + q,)`` in :math:`\\mathbb{R}`.
        p: Number of α / γ coefficients (static).
        q: Number of β coefficients (static).
        kappa: Scalar truncated-second-moment of the standardised
            residual law.

    Returns:
        Tuple ``(alpha, gamma, beta)`` of shapes ``(p,)``, ``(p,)``,
        ``(q,)`` — all non-negative; satisfies ``Σα + κ·Σγ + Σβ < 1``.
    """
    raw_weights = jnp.asarray(raw_weights, dtype=float).reshape(-1)
    s = jnn.sigmoid(jnp.asarray(raw_persistence, dtype=float).reshape(()))
    w = jnn.softmax(raw_weights)
    chunks = s * w
    kappa_safe = jnp.maximum(jnp.asarray(kappa, dtype=float), _BOUNDARY_EPS)
    alpha = chunks[:p]
    gamma = chunks[p : 2 * p] / kappa_safe
    beta = chunks[2 * p : 2 * p + q]
    return alpha, gamma, beta


def gjr_unsimplex(
    alpha: ArrayLike,
    gamma: ArrayLike,
    beta: ArrayLike,
    kappa: ArrayLike,
) -> tuple[Array, Array]:
    r"""Inverse of :func:`gjr_simplex` for warm starts."""
    alpha = jnp.asarray(alpha, dtype=float).reshape(-1)
    gamma = jnp.asarray(gamma, dtype=float).reshape(-1)
    beta = jnp.asarray(beta, dtype=float).reshape(-1)
    kappa_safe = jnp.maximum(jnp.asarray(kappa, dtype=float), _BOUNDARY_EPS)
    chunks = jnp.concatenate([alpha, kappa_safe * gamma, beta])
    s = jnp.sum(chunks)
    s = jnp.clip(s, _BOUNDARY_EPS, 1.0 - _BOUNDARY_EPS)
    raw_persistence = jnp.log(s) - jnp.log1p(-s)
    weights = chunks / s
    weights = jnp.maximum(weights, _BOUNDARY_EPS)
    raw_weights = jnp.log(weights)
    return raw_persistence, raw_weights


def tgarch_simplex(
    raw_persistence: ArrayLike,
    raw_weights: ArrayLike,
    p: int,
    q: int,
    e_pos: ArrayLike,
    e_neg: ArrayLike,
) -> tuple[Array, Array, Array]:
    r"""Simplex split for TGARCH (Zakoian 1994 σ-form).

    The recursion is on :math:`\\sigma_t` (not :math:`\\sigma^2_t`):

    .. math::

        \\sigma_t = \\omega
                  + \\sum_{i=1}^p (\\alpha^{+}_i\\,\\varepsilon^{+}_{t-i}
                                  + \\alpha^{-}_i\\,\\varepsilon^{-}_{t-i})
                  + \\sum_{j=1}^q \\beta_j\\,\\sigma_{t-j},

    with :math:`\\varepsilon^{+} = \\max(\\varepsilon, 0)`,
    :math:`\\varepsilon^{-} = \\max(-\\varepsilon, 0)`.  Taking
    expectations under stationarity (with :math:`z \\perp \\sigma`,
    :math:`\\mathbb{E}[\\sigma \\cdot z^{+}] = \\mathbb{E}[\\sigma]
    \\cdot \\mathbb{E}[z^{+}]`):

    .. math::

        \\sum_i \\bigl(\\alpha^{+}_i \\mathbb{E}[z^{+}]
                     + \\alpha^{-}_i \\mathbb{E}[z^{-}]\\bigr)
        + \\sum_j \\beta_j  < 1,
        \\qquad
        \\mathbb{E}[z^{+}] = \\mathbb{E}[z\\cdot\\mathbf{1}\\{z>0\\}],
        \\quad
        \\mathbb{E}[z^{-}] = -\\mathbb{E}[z\\cdot\\mathbf{1}\\{z<0\\}].

    Under symmetric residuals :math:`\\mathbb{E}[z^{+}] =
    \\mathbb{E}[z^{-}] = \\mathbb{E}[|z|] / 2`; under skew they
    differ.  Both are computed via ``quadax`` integrals against the
    standardised-residual PDF.

    Reference:
        Zakoian, J.M. (1994). *Threshold heteroskedastic models*.
        Journal of Economic Dynamics and Control, 18(5), 931-955.

    Args:
        raw_persistence: scalar.
        raw_weights: shape ``(2p + q,)``.
        p: Number of asymmetric positive / negative coefficients.
        q: Number of β coefficients.
        e_pos: :math:`\\mathbb{E}[z^{+}]` (positive scalar).
        e_neg: :math:`\\mathbb{E}[z^{-}]` (positive scalar).

    Returns:
        Tuple ``(alpha_pos, alpha_neg, beta)`` of shapes ``(p,)``,
        ``(p,)``, ``(q,)``.  All non-negative; satisfies the
        persistence inequality above.
    """
    raw_weights = jnp.asarray(raw_weights, dtype=float).reshape(-1)
    s = jnn.sigmoid(jnp.asarray(raw_persistence, dtype=float).reshape(()))
    w = jnn.softmax(raw_weights)
    chunks = s * w
    e_pos_safe = jnp.maximum(jnp.asarray(e_pos, dtype=float), _BOUNDARY_EPS)
    e_neg_safe = jnp.maximum(jnp.asarray(e_neg, dtype=float), _BOUNDARY_EPS)
    alpha_pos = chunks[:p] / e_pos_safe
    alpha_neg = chunks[p : 2 * p] / e_neg_safe
    beta = chunks[2 * p : 2 * p + q]
    return alpha_pos, alpha_neg, beta


def tgarch_unsimplex(
    alpha_pos: ArrayLike,
    alpha_neg: ArrayLike,
    beta: ArrayLike,
    e_pos: ArrayLike,
    e_neg: ArrayLike,
) -> tuple[Array, Array]:
    r"""Inverse of :func:`tgarch_simplex` for warm starts."""
    alpha_pos = jnp.asarray(alpha_pos, dtype=float).reshape(-1)
    alpha_neg = jnp.asarray(alpha_neg, dtype=float).reshape(-1)
    beta = jnp.asarray(beta, dtype=float).reshape(-1)
    e_pos_safe = jnp.maximum(jnp.asarray(e_pos, dtype=float), _BOUNDARY_EPS)
    e_neg_safe = jnp.maximum(jnp.asarray(e_neg, dtype=float), _BOUNDARY_EPS)
    chunks = jnp.concatenate(
        [e_pos_safe * alpha_pos, e_neg_safe * alpha_neg, beta]
    )
    s = jnp.sum(chunks)
    s = jnp.clip(s, _BOUNDARY_EPS, 1.0 - _BOUNDARY_EPS)
    raw_persistence = jnp.log(s) - jnp.log1p(-s)
    weights = chunks / s
    weights = jnp.maximum(weights, _BOUNDARY_EPS)
    raw_weights = jnp.log(weights)
    return raw_persistence, raw_weights


###############################################################################
# Stationarity diagnostics (for stats() / summary())
###############################################################################
def ar_polynomial_roots(phi: ArrayLike) -> Array:
    r"""Roots of the AR characteristic polynomial
    :math:`1 - \\phi_1 z - \\cdots - \\phi_p z^p`.

    The AR(p) process is causal / stationary iff every root has
    modulus strictly greater than 1.  Used by :meth:`stats()` to
    expose the moduli alongside a stationarity flag.
    """
    phi = jnp.asarray(phi, dtype=float).reshape(-1)
    p = int(phi.shape[0])
    if p == 0:
        return jnp.zeros((0,), dtype=jnp.complex64)
    coeffs = jnp.concatenate([jnp.array([1.0]), -phi])
    return jnp.roots(coeffs[::-1])


def ar_is_stationary(phi: ArrayLike) -> Array:
    r"""``True`` iff every root of the AR polynomial has modulus
    :math:`> 1`.

    Returns a JAX scalar ``bool_`` so the result can be threaded
    through traced code paths (e.g. for diagnostics summaries that
    are themselves jittable).
    """
    phi = jnp.asarray(phi, dtype=float).reshape(-1)
    if int(phi.shape[0]) == 0:
        return jnp.asarray(True)
    moduli = jnp.abs(ar_polynomial_roots(phi))
    return jnp.all(moduli > 1.0 + _BOUNDARY_EPS)


def ma_polynomial_roots(theta: ArrayLike) -> Array:
    r"""Roots of the MA characteristic polynomial
    :math:`1 + \\theta_1 z + \\cdots + \\theta_q z^q`.

    The MA(q) process is invertible iff every root has modulus
    strictly greater than 1.  Note the **plus** signs on the
    :math:`\\theta_j` coefficients — this matches CopulAX's
    :func:`run_arma` recursion ``μ_t = μ + Σ φ_i (y_{t-i} − μ) +
    Σ θ_j ε_{t-j}`` and is also the convention used by
    ``statsmodels.tsa.arima.ARIMA``.  In particular this is *not*
    interchangeable with :func:`ar_polynomial_roots` for q ≥ 2:
    the two polynomials have different coefficients (sign flips on
    every term) and therefore different roots / moduli.
    """
    theta = jnp.asarray(theta, dtype=float).reshape(-1)
    q = int(theta.shape[0])
    if q == 0:
        return jnp.zeros((0,), dtype=jnp.complex64)
    coeffs = jnp.concatenate([jnp.array([1.0]), theta])
    return jnp.roots(coeffs[::-1])


def ma_is_invertible(theta: ArrayLike) -> Array:
    r"""``True`` iff every root of the MA polynomial
    :math:`1 + \\sum_j \\theta_j z^j` has modulus :math:`> 1`.

    Returns a JAX scalar ``bool_``.  Invertibility is required for
    the MA parameters to be uniquely identified from the
    autocovariance function (the "non-invertible mirror" otherwise —
    Brockwell-Davis 1991, §3.1).
    """
    theta = jnp.asarray(theta, dtype=float).reshape(-1)
    if int(theta.shape[0]) == 0:
        return jnp.asarray(True)
    moduli = jnp.abs(ma_polynomial_roots(theta))
    return jnp.all(moduli > 1.0 + _BOUNDARY_EPS)


def garch_persistence(alpha: ArrayLike, beta: ArrayLike) -> Array:
    r"""Vanilla-GARCH persistence :math:`\\sum_i \\alpha_i +
    \\sum_j \\beta_j`.

    The unconditional variance is :math:`\\omega / (1 - \\text{persistence})`
    when persistence ``< 1`` and undefined otherwise.
    """
    alpha = jnp.asarray(alpha, dtype=float).reshape(-1)
    beta = jnp.asarray(beta, dtype=float).reshape(-1)
    return jnp.sum(alpha) + jnp.sum(beta)
