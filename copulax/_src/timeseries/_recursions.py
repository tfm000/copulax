""":func:`jax.lax.scan` recursion kernels for the time-series subpackage.

Every fit objective, ``residuals(...)``, ``conditional_*(...)``, and
``forecast(...)`` method routes through one of the kernels below.
Co-locating the kernels has two benefits:

* Mathematical clarity — each recursion is defined once, with a
  citation, and shared across the family's fit / residual / forecast
  / sample paths.  No hidden divergence between the loss surface and
  the residual it produces.
* JIT cache reuse — the kernel is a pure function of ``(params,
  series, init_state)``; orders ``(p, q)`` enter through the static
  array shapes, so a single compiled trace serves every series of the
  same length and dtype.

Conventions:

* ``y`` denotes the level series for mean models; ``eps`` denotes
  the mean-corrected innovation series for variance models.
* The carry stores the **last p / q** lagged values that the
  recursion needs at the *next* step.  After consuming the full
  series, the final carry is exactly the per-family terminal state
  required by ``forecast(h)``.
* ``_shift(lags, new_value)`` updates a lag buffer by prepending the
  new value and dropping the oldest.  A no-op when the buffer is
  empty (``p = 0`` or ``q = 0`` cases) — handled at trace time via a
  Python-level conditional on the static lag shape.
* All recursions floor their output (``σ²``, ``σ``) at a small
  positive constant so subsequent ``log`` / ``√`` operations remain
  finite.  The floor is below any plausible empirical value but
  above machine epsilon, matching the pattern in
  ``copulax/_src/multivariate/mvt_gh.py:409-410``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


# Lower bound on conditional-variance / -standard-deviation outputs.
# Below this, ``log`` and ``1/σ`` produce non-finite leaves that
# poison gradients via NaN propagation.  The floor is well below any
# plausible fit (variances at this scale would already imply σ ≪
# 1e-7 in returns space) and well above ``finfo(float32).eps``
# (~1.2e-7) so single-precision JAX defaults are safe too.
_VAR_FLOOR: float = 1e-12
_SIGMA_FLOOR: float = 1e-6


def _shift(lags: Array, new_value: ArrayLike) -> Array:
    r"""Prepend ``new_value``, drop the oldest entry; pass through when
    ``lags`` is empty.

    Shape-preserving so the ``lax.scan`` carry stays consistent across
    iterations.  The ``shape[0] == 0`` branch resolves at trace time
    since lag lengths are static.
    """
    if int(lags.shape[0]) == 0:
        return lags
    head = jnp.asarray(new_value, dtype=lags.dtype).reshape((1,))
    return jnp.concatenate([head, lags[:-1]])


###############################################################################
# ARMA(p, q) — mean-equation recursion (centred form, Box-Jenkins / Hamilton)
###############################################################################
def run_arma(
    y: Array,
    phi: Array,
    theta: Array,
    mu: Array,
    init_y_lags: Array,
    init_eps_lags: Array,
) -> tuple[Array, Array, tuple[Array, Array]]:
    r"""ARMA(p, q) one-step-ahead recursion forward over ``y`` (centred form).

    The conditional mean is

    .. math::

        \mu_t = \mu + \sum_{i=1}^p \phi_i\, (y_{t-i} - \mu)
                    + \sum_{j=1}^q \theta_j\, \varepsilon_{t-j},

    where :math:`\mu` is the unconditional mean of the process, and
    the innovation residual is :math:`\varepsilon_t = y_t - \mu_t`.
    This matches the centred convention used by Box-Jenkins,
    Hamilton (1994), and rugarch/statsmodels — :math:`\mu` IS the
    long-run mean rather than a per-step additive drift.

    Both sequences are returned along with the terminal state
    :math:`(y_{n}, \ldots, y_{n-p+1}; \varepsilon_n, \ldots,
    \varepsilon_{n-q+1})`.

    Args:
        y: shape ``(n,)`` — observed series.
        phi: shape ``(p,)`` — AR coefficients (already in the
            constrained / stationary parameterisation).
        theta: shape ``(q,)`` — MA coefficients (already constrained).
        mu: scalar — unconditional mean of the process.
        init_y_lags: shape ``(p,)`` — pre-sample
            :math:`(y_0, y_{-1}, \ldots, y_{-p+1})` ordered with the
            most-recent value first.
        init_eps_lags: shape ``(q,)`` — pre-sample
            :math:`(\varepsilon_0, \varepsilon_{-1}, \ldots,
            \varepsilon_{-q+1})`.

    Returns:
        Tuple ``(mu_seq, eps_seq, terminal_state)`` where ``mu_seq``
        and ``eps_seq`` have shape ``(n,)`` and ``terminal_state ==
        (y_lags_n, eps_lags_n)`` is the post-recursion carry suitable
        for ``forecast(h)`` or chaining into another window.
    """
    y = jnp.asarray(y, dtype=float).reshape(-1)
    phi = jnp.asarray(phi, dtype=float).reshape(-1)
    theta = jnp.asarray(theta, dtype=float).reshape(-1)
    mu = jnp.asarray(mu, dtype=float).reshape(())

    def step(carry, y_t):
        y_lags, eps_lags = carry
        mu_t = mu + jnp.dot(phi, y_lags - mu) + jnp.dot(theta, eps_lags)
        eps_t = y_t - mu_t
        return (
            (_shift(y_lags, y_t), _shift(eps_lags, eps_t)),
            (mu_t, eps_t),
        )

    init_carry = (
        jnp.asarray(init_y_lags, dtype=float).reshape(-1),
        jnp.asarray(init_eps_lags, dtype=float).reshape(-1),
    )
    final_carry, (mu_seq, eps_seq) = jax.lax.scan(step, init_carry, y)
    return mu_seq, eps_seq, final_carry


###############################################################################
# GARCH(p, q) — vanilla σ²-form recursion
###############################################################################
def run_garch(
    eps: Array,
    omega: Array,
    alpha: Array,
    beta: Array,
    init_eps_sq_lags: Array,
    init_var_lags: Array,
) -> tuple[Array, tuple[Array, Array]]:
    r"""GARCH(p, q) σ²-recursion (Bollerslev 1986).

    .. math::

        \sigma^2_t = \omega
                   + \sum_{i=1}^p \alpha_i\, \varepsilon^2_{t-i}
                   + \sum_{j=1}^q \beta_j\, \sigma^2_{t-j}.

    ``omega``, ``alpha``, and ``beta`` must already satisfy
    positivity / stationarity (e.g. via
    :func:`copulax._src.timeseries._stationarity.garch_simplex`).
    The output ``σ²`` is floored at :data:`_VAR_FLOOR` to keep
    downstream ``log`` finite if the optimiser briefly visits a
    near-degenerate point.

    Args:
        eps: shape ``(n,)`` — mean-corrected innovation series.
        omega: scalar.
        alpha: shape ``(p,)``.
        beta: shape ``(q,)``.
        init_eps_sq_lags: shape ``(p,)`` — pre-sample
            :math:`(\varepsilon^2_0, \ldots, \varepsilon^2_{-p+1})`.
        init_var_lags: shape ``(q,)`` — pre-sample
            :math:`(\sigma^2_0, \ldots, \sigma^2_{-q+1})`.

    Returns:
        Tuple ``(var_seq, terminal_state)`` where ``var_seq`` has
        shape ``(n,)`` and ``terminal_state`` is the carry after the
        scan — directly usable as ``forecast(h)`` input.
    """
    eps = jnp.asarray(eps, dtype=float).reshape(-1)
    alpha = jnp.asarray(alpha, dtype=float).reshape(-1)
    beta = jnp.asarray(beta, dtype=float).reshape(-1)
    omega = jnp.asarray(omega, dtype=float).reshape(())

    def step(carry, eps_t):
        eps_sq_lags, var_lags = carry
        var_t = omega + jnp.dot(alpha, eps_sq_lags) + jnp.dot(beta, var_lags)
        var_t = jnp.maximum(var_t, _VAR_FLOOR)
        return (
            (_shift(eps_sq_lags, eps_t * eps_t),
             _shift(var_lags, var_t)),
            var_t,
        )

    init_carry = (
        jnp.asarray(init_eps_sq_lags, dtype=float).reshape(-1),
        jnp.asarray(init_var_lags, dtype=float).reshape(-1),
    )
    final_carry, var_seq = jax.lax.scan(step, init_carry, eps)
    return var_seq, final_carry


###############################################################################
# GJR-GARCH(p, q) — asymmetric σ²-form recursion (Glosten-Jagannathan-Runkle)
###############################################################################
def run_gjr_garch(
    eps: Array,
    omega: Array,
    alpha: Array,
    gamma: Array,
    beta: Array,
    init_eps_sq_lags: Array,
    init_neg_eps_sq_lags: Array,
    init_var_lags: Array,
) -> tuple[Array, tuple[Array, Array, Array]]:
    r"""GJR-GARCH(p, q) σ²-recursion (Glosten-Jagannathan-Runkle 1993).

    .. math::

        \sigma^2_t = \omega
                   + \sum_{i=1}^p \alpha_i\, \varepsilon^2_{t-i}
                   + \sum_{i=1}^p \gamma_i\, \varepsilon^2_{t-i}\,
                                   \mathbf{1}\{\varepsilon_{t-i} < 0\}
                   + \sum_{j=1}^q \beta_j\, \sigma^2_{t-j}.

    The carry maintains both ``ε²`` and the leverage-projected
    ``ε² · 1{ε < 0}`` lag buffers separately so that a single dot
    product per family suffices at each step (no per-element
    ``where`` inside the scan body).

    Args:
        eps: shape ``(n,)``.
        omega: scalar.
        alpha, gamma: shape ``(p,)`` each.
        beta: shape ``(q,)``.
        init_eps_sq_lags, init_neg_eps_sq_lags: shape ``(p,)`` each;
            the second is :math:`\varepsilon^2_{t} \cdot \mathbf{1}\{
            \varepsilon_{t} < 0\}`.
        init_var_lags: shape ``(q,)``.
    """
    eps = jnp.asarray(eps, dtype=float).reshape(-1)
    alpha = jnp.asarray(alpha, dtype=float).reshape(-1)
    gamma = jnp.asarray(gamma, dtype=float).reshape(-1)
    beta = jnp.asarray(beta, dtype=float).reshape(-1)
    omega = jnp.asarray(omega, dtype=float).reshape(())

    def step(carry, eps_t):
        eps_sq_lags, neg_eps_sq_lags, var_lags = carry
        var_t = (
            omega
            + jnp.dot(alpha, eps_sq_lags)
            + jnp.dot(gamma, neg_eps_sq_lags)
            + jnp.dot(beta, var_lags)
        )
        var_t = jnp.maximum(var_t, _VAR_FLOOR)
        eps_t_sq = eps_t * eps_t
        neg_eps_t_sq = jnp.where(eps_t < 0.0, eps_t_sq, 0.0)
        return (
            (_shift(eps_sq_lags, eps_t_sq),
             _shift(neg_eps_sq_lags, neg_eps_t_sq),
             _shift(var_lags, var_t)),
            var_t,
        )

    init_carry = (
        jnp.asarray(init_eps_sq_lags, dtype=float).reshape(-1),
        jnp.asarray(init_neg_eps_sq_lags, dtype=float).reshape(-1),
        jnp.asarray(init_var_lags, dtype=float).reshape(-1),
    )
    final_carry, var_seq = jax.lax.scan(step, init_carry, eps)
    return var_seq, final_carry


###############################################################################
# EGARCH(p, q) — log-variance recursion (Nelson 1991)
###############################################################################
def run_egarch(
    eps: Array,
    omega: Array,
    alpha: Array,
    gamma: Array,
    beta: Array,
    expected_abs_z: Array,
    init_z_lags: Array,
    init_log_var_lags: Array,
) -> tuple[Array, tuple[Array, Array]]:
    r"""EGARCH(p, q) log-variance recursion (Nelson 1991, eqn 2.6).

    .. math::

        \log \sigma^2_t = \omega
                       + \sum_{i=1}^p \alpha_i\, z_{t-i}
                       + \sum_{i=1}^p \gamma_i\, (|z_{t-i}|
                                                  - \mathbb{E}|z|)
                       + \sum_{j=1}^q \beta_j\, \log \sigma^2_{t-j},

    where :math:`z_t = \varepsilon_t / \sigma_t` is the standardised
    residual.  ``alpha`` is the *leverage* coefficient (sign-
    sensitive response to ``z``) and ``gamma`` is the *size*
    coefficient (response to centred ``|z|``); this matches Nelson
    (1991), rugarch, arch, and standard textbooks. The log-variance
    form has no positivity constraint — :math:`\sigma^2_t = \exp(\log
    \sigma^2_t)` is positive identically — so no simplex
    reparameterisation is needed; stationarity is governed by the AR
    polynomial in the lag operator on :math:`\log \sigma^2`.

    ``expected_abs_z`` is the analytic / quadrature-computed
    :math:`\mathbb{E}|z|` under the standardised residual law;
    centring the :math:`|z_{t-i}|` term is essential for
    :math:`\mathbb{E}[\log \sigma^2_t]` to equal :math:`\omega /
    (1 - \sum \beta_j)`.

    Args:
        eps: shape ``(n,)``.
        omega: scalar.
        alpha: shape ``(p,)`` — leverage coefficients on
            :math:`z_{t-i}`.
        gamma: shape ``(p,)`` — size coefficients on
            :math:`|z_{t-i}| - \mathbb{E}|z|`.
        beta: shape ``(q,)``.
        expected_abs_z: scalar — :math:`\mathbb{E}|z|` for the chosen
            standardised residual law.
        init_z_lags: shape ``(p,)`` — pre-sample standardised
            residuals.
        init_log_var_lags: shape ``(q,)`` — pre-sample
            :math:`\log \sigma^2`.

    Returns:
        Tuple ``(log_var_seq, terminal_state)``.
    """
    eps = jnp.asarray(eps, dtype=float).reshape(-1)
    alpha = jnp.asarray(alpha, dtype=float).reshape(-1)
    gamma = jnp.asarray(gamma, dtype=float).reshape(-1)
    beta = jnp.asarray(beta, dtype=float).reshape(-1)
    omega = jnp.asarray(omega, dtype=float).reshape(())
    expected_abs_z = jnp.asarray(expected_abs_z, dtype=float).reshape(())

    def step(carry, eps_t):
        z_lags, log_var_lags = carry
        centred_abs_z_lags = jnp.abs(z_lags) - expected_abs_z
        log_var_t = (
            omega
            + jnp.dot(alpha, z_lags)
            + jnp.dot(gamma, centred_abs_z_lags)
            + jnp.dot(beta, log_var_lags)
        )
        sigma_t = jnp.exp(0.5 * log_var_t)
        sigma_t = jnp.maximum(sigma_t, _SIGMA_FLOOR)
        z_t = eps_t / sigma_t
        return (
            (_shift(z_lags, z_t),
             _shift(log_var_lags, log_var_t)),
            log_var_t,
        )

    init_carry = (
        jnp.asarray(init_z_lags, dtype=float).reshape(-1),
        jnp.asarray(init_log_var_lags, dtype=float).reshape(-1),
    )
    final_carry, log_var_seq = jax.lax.scan(step, init_carry, eps)
    return log_var_seq, final_carry


###############################################################################
# TGARCH(p, q) — Zakoian (1994) σ-form recursion
###############################################################################
def run_tgarch(
    eps: Array,
    omega: Array,
    alpha_pos: Array,
    alpha_neg: Array,
    beta: Array,
    init_eps_pos_lags: Array,
    init_eps_neg_lags: Array,
    init_sigma_lags: Array,
) -> tuple[Array, tuple[Array, Array, Array]]:
    r"""TGARCH(p, q) σ-form recursion (Zakoian 1994).

    .. math::

        \sigma_t = \omega
                + \sum_{i=1}^p (\alpha^{+}_i\, \varepsilon^{+}_{t-i}
                              + \alpha^{-}_i\, \varepsilon^{-}_{t-i})
                + \sum_{j=1}^q \beta_j\, \sigma_{t-j},

    with :math:`\varepsilon^{+} = \max(\varepsilon, 0)`,
    :math:`\varepsilon^{-} = \max(-\varepsilon, 0)`.  Note this is
    the σ-recursion (not σ²) — the persistence condition involves
    *first* moments of the standardised residual; see
    ``_stationarity.tgarch_simplex`` for the corresponding
    reparameterisation.

    Reference:
        Zakoian, J.M. (1994). Threshold heteroskedastic models.
        *Journal of Economic Dynamics and Control*, 18(5), 931-955.

    Args:
        eps: shape ``(n,)``.
        omega: scalar.
        alpha_pos, alpha_neg: shape ``(p,)`` each.
        beta: shape ``(q,)``.
        init_eps_pos_lags, init_eps_neg_lags: shape ``(p,)``;
            non-negative entries holding pre-sample ``ε^±``.
        init_sigma_lags: shape ``(q,)`` — pre-sample σ.
    """
    eps = jnp.asarray(eps, dtype=float).reshape(-1)
    alpha_pos = jnp.asarray(alpha_pos, dtype=float).reshape(-1)
    alpha_neg = jnp.asarray(alpha_neg, dtype=float).reshape(-1)
    beta = jnp.asarray(beta, dtype=float).reshape(-1)
    omega = jnp.asarray(omega, dtype=float).reshape(())

    def step(carry, eps_t):
        eps_pos_lags, eps_neg_lags, sigma_lags = carry
        sigma_t = (
            omega
            + jnp.dot(alpha_pos, eps_pos_lags)
            + jnp.dot(alpha_neg, eps_neg_lags)
            + jnp.dot(beta, sigma_lags)
        )
        sigma_t = jnp.maximum(sigma_t, _SIGMA_FLOOR)
        eps_t_pos = jnp.maximum(eps_t, 0.0)
        eps_t_neg = jnp.maximum(-eps_t, 0.0)
        return (
            (_shift(eps_pos_lags, eps_t_pos),
             _shift(eps_neg_lags, eps_t_neg),
             _shift(sigma_lags, sigma_t)),
            sigma_t,
        )

    init_carry = (
        jnp.asarray(init_eps_pos_lags, dtype=float).reshape(-1),
        jnp.asarray(init_eps_neg_lags, dtype=float).reshape(-1),
        jnp.asarray(init_sigma_lags, dtype=float).reshape(-1),
    )
    final_carry, sigma_seq = jax.lax.scan(step, init_carry, eps)
    return sigma_seq, final_carry


###############################################################################
# QGARCH(1, q) — quadratic-asymmetry recursion (Sentana 1995)
###############################################################################
def run_qgarch(
    eps: Array,
    omega: Array,
    alpha: Array,
    psi: Array,
    beta: Array,
    init_eps_lags: Array,
    init_eps_sq_lags: Array,
    init_var_lags: Array,
) -> tuple[Array, tuple[Array, Array, Array]]:
    r"""QGARCH(p, q) σ²-recursion (Sentana 1995).

    .. math::

        \sigma^2_t = \omega
                   + \sum_{i=1}^p \alpha_i\, \varepsilon^2_{t-i}
                   + \sum_{i=1}^p \psi_i\, \varepsilon_{t-i}
                   + \sum_{j=1}^q \beta_j\, \sigma^2_{t-j}.

    The ``ψ_i ε_{t-i}`` term picks up sign-dependent asymmetry while
    leaving the unconditional variance equal to the symmetric case
    (``E[ψ ε] = 0`` under any zero-mean residual law).  Per plan
    §"Stationarity" v1 restricts ``p = 1`` — positivity for ``p ≥ 2``
    is a *matrix* condition (Sentana 1995, augmented matrix PSD)
    rather than a scalar and is deferred.

    Reference:
        Sentana, E. (1995). Quadratic ARCH Models. *Review of
        Economic Studies*, 62(4), 639-661.
    """
    eps = jnp.asarray(eps, dtype=float).reshape(-1)
    alpha = jnp.asarray(alpha, dtype=float).reshape(-1)
    psi = jnp.asarray(psi, dtype=float).reshape(-1)
    beta = jnp.asarray(beta, dtype=float).reshape(-1)
    omega = jnp.asarray(omega, dtype=float).reshape(())

    def step(carry, eps_t):
        eps_lags, eps_sq_lags, var_lags = carry
        var_t = (
            omega
            + jnp.dot(alpha, eps_sq_lags)
            + jnp.dot(psi, eps_lags)
            + jnp.dot(beta, var_lags)
        )
        var_t = jnp.maximum(var_t, _VAR_FLOOR)
        return (
            (_shift(eps_lags, eps_t),
             _shift(eps_sq_lags, eps_t * eps_t),
             _shift(var_lags, var_t)),
            var_t,
        )

    init_carry = (
        jnp.asarray(init_eps_lags, dtype=float).reshape(-1),
        jnp.asarray(init_eps_sq_lags, dtype=float).reshape(-1),
        jnp.asarray(init_var_lags, dtype=float).reshape(-1),
    )
    final_carry, var_seq = jax.lax.scan(step, init_carry, eps)
    return var_seq, final_carry


###############################################################################
# GARCH-M(p, q) — variance-in-mean recursion
###############################################################################
def run_garch_m(
    y: Array,
    mu: Array,
    lambda_m: Array,
    omega: Array,
    alpha: Array,
    beta: Array,
    init_eps_sq_lags: Array,
    init_var_lags: Array,
) -> tuple[Array, Array, Array, tuple[Array, Array]]:
    r"""GARCH-M(p, q) joint mean-variance recursion (Engle, Lilien & Robins 1987).

    .. math::

        y_t       &= \mu + \lambda_m\, \sigma^2_t + \varepsilon_t,\\
        \sigma^2_t &= \omega
                    + \sum_{i=1}^p \alpha_i\, \varepsilon^2_{t-i}
                    + \sum_{j=1}^q \beta_j\, \sigma^2_{t-j}.

    ``σ²_t`` depends only on the carry, so we compute it first, then
    use it to form ``μ_t`` and ``ε_t = y_t - μ_t`` at each step.  The
    carry is identical to vanilla GARCH; only the per-step output
    expands to the triple ``(μ_t, ε_t, σ²_t)``.

    Reference:
        Engle, R., Lilien, D., & Robins, R. (1987). Estimating Time
        Varying Risk Premia in the Term Structure: The ARCH-M Model.
        *Econometrica*, 55(2), 391-407.
    """
    y = jnp.asarray(y, dtype=float).reshape(-1)
    alpha = jnp.asarray(alpha, dtype=float).reshape(-1)
    beta = jnp.asarray(beta, dtype=float).reshape(-1)
    omega = jnp.asarray(omega, dtype=float).reshape(())
    mu = jnp.asarray(mu, dtype=float).reshape(())
    lambda_m = jnp.asarray(lambda_m, dtype=float).reshape(())

    def step(carry, y_t):
        eps_sq_lags, var_lags = carry
        var_t = omega + jnp.dot(alpha, eps_sq_lags) + jnp.dot(beta, var_lags)
        var_t = jnp.maximum(var_t, _VAR_FLOOR)
        mu_t = mu + lambda_m * var_t
        eps_t = y_t - mu_t
        return (
            (_shift(eps_sq_lags, eps_t * eps_t),
             _shift(var_lags, var_t)),
            (mu_t, eps_t, var_t),
        )

    init_carry = (
        jnp.asarray(init_eps_sq_lags, dtype=float).reshape(-1),
        jnp.asarray(init_var_lags, dtype=float).reshape(-1),
    )
    final_carry, (mu_seq, eps_seq, var_seq) = jax.lax.scan(step, init_carry, y)
    return mu_seq, eps_seq, var_seq, final_carry
