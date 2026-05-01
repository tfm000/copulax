"""Initialisation strategies for the time-series subpackage.

Exposes three building-block estimators and the per-family dispatchers
they compose into:

* :func:`acvf` — biased sample autocovariance function.
* :func:`yule_walker_ar` — closed-form AR(p) coefficient estimate
  via :func:`jax.numpy.linalg.solve` on the symmetric Toeplitz
  Yule-Walker system.
* :func:`innovations_ma` — Brockwell & Davis (1991, §5.2.2)
  Innovations Algorithm truncated to MA(q), implemented with a
  fully-unrolled Python loop over the static order ``n_iter`` —
  total work :math:`O(n_{\\mathrm{iter}} \\cdot q^2)`, well below the
  cost of a YW pre-fit + Hannan-Rissanen second stage.

The dispatcher functions :func:`init_arma_params`, :func:`init_garch_params`,
:func:`arma_pre_sample_state`, and :func:`garch_pre_sample_state`
implement the four documented modes — ``analytical``, ``backcast``,
``sample``, and the warm-start path (handled at the family fit
dispatcher).  The compiled trace key for each mode is constant across
rolling-window calls of the same length and dtype.

References:
    Brockwell, P.J. & Davis, R.A. (1991).  *Time Series: Theory and
    Methods* (2nd ed.).  Springer.  Yule-Walker — §3.4; Innovations
    Algorithm — §5.2.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


# Numerical guard for divisions in the Innovations recursion.  Below
# this magnitude, ``v_k`` is treated as a degenerate denominator and
# the corresponding θ update is masked to zero — ``v_k`` only goes
# negative or vanishes when the input ACVF matrix is rank-deficient
# (e.g. constant series), in which case the MA estimate is degenerate
# anyway.
_V_FLOOR: float = 1e-12


###############################################################################
# Sample autocovariance
###############################################################################
def acvf(y: ArrayLike, max_lag: int) -> Array:
    r"""Biased sample autocovariance function up to ``max_lag``.

    .. math::

        \hat{\gamma}(k) = \frac{1}{n} \sum_{t=1}^{n-k}
                                  (y_t - \bar y)\,(y_{t+k} - \bar y),
        \qquad k = 0, 1, \ldots, \mathrm{max\_lag}.

    The biased estimator (denominator :math:`n`, not :math:`n-k`) is
    the textbook choice when the ACVF is fed into Yule-Walker /
    Innovations algorithms — it guarantees the resulting ACVF
    Toeplitz matrix is positive semi-definite (Brockwell & Davis
    1991, Proposition 1.6.1), which the unbiased estimator does not.

    Args:
        y: shape ``(n,)`` — input series.
        max_lag: Maximum lag to compute (static Python ``int``).

    Returns:
        shape ``(max_lag + 1,)`` array of sample autocovariances.
    """
    y = jnp.asarray(y, dtype=float).reshape(-1)
    n = int(y.shape[0])
    mean = jnp.mean(y)
    centred = y - mean
    max_lag = int(max_lag)

    def gamma_k(k: int) -> Array:
        # Multiply centred[:n-k] by centred[k:] and sum.  Both slices
        # are static-shape views so the resulting graph is fixed per
        # ``k``; the outer loop is unrolled at trace time.
        if k == 0:
            return jnp.sum(centred * centred) / n
        return jnp.sum(centred[: n - k] * centred[k:]) / n

    return jnp.stack([gamma_k(k) for k in range(max_lag + 1)])


###############################################################################
# Yule-Walker AR(p) estimator
###############################################################################
def yule_walker_ar(y: ArrayLike, p: int) -> Array:
    r"""Yule-Walker estimate of the AR(p) coefficient vector.

    Solves the symmetric Toeplitz system :math:`\Gamma \phi = \gamma`
    where :math:`\Gamma_{ij} = \hat\gamma(|i - j|)` for
    :math:`i, j = 1, \ldots, p` and :math:`\gamma = (\hat\gamma(1),
    \ldots, \hat\gamma(p))^\top`.  Single :func:`jnp.linalg.solve` —
    JIT-compatible end-to-end.

    The Toeplitz structure is exploited only to build the matrix; the
    actual solve uses a dense LAPACK ``GESV`` call.  For typical
    AR orders (``p <= 10``) this is faster than Levinson-Durbin in
    JAX since the latter would unroll into a sequence of ``at[].set``
    calls.

    Args:
        y: shape ``(n,)`` — input series.
        p: AR order (static Python ``int``).

    Returns:
        shape ``(p,)`` array of AR coefficients.  Returns an empty
        ``(0,)`` array when ``p == 0``.
    """
    p = int(p)
    if p == 0:
        return jnp.zeros((0,), dtype=float)
    gamma = acvf(y, p)  # shape (p+1,)
    idx = jnp.abs(
        jnp.arange(p, dtype=int)[:, None] - jnp.arange(p, dtype=int)[None, :]
    )
    Gamma = gamma[idx]  # (p, p)
    rhs = gamma[1:]  # (p,)
    # ``jnp.linalg.solve`` is autograd-compatible and respects the
    # implicit-function-theorem chain rule — no separate forward /
    # reverse pass needed.
    return jnp.linalg.solve(Gamma, rhs)


###############################################################################
# Innovations Algorithm for MA(q)
###############################################################################
def innovations_ma(
    y: ArrayLike, q: int, n_iter: int | None = None,
) -> tuple[Array, Array]:
    r"""Innovations Algorithm estimate of the MA(q) coefficient vector.

    Implements the Brockwell-Davis (1991, §5.2.2) recursion

    .. math::

        \theta_{n,\,n-k} &= v_k^{-1}\!\left(
            \hat\gamma(n-k)
            - \sum_{j=0}^{k-1}
                \theta_{k,\,k-j}\,\theta_{n,\,n-j}\,v_j
        \right),
        \qquad k = 0, 1, \ldots, n-1,\\
        v_n &= \hat\gamma(0)
              - \sum_{j=0}^{n-1} \theta_{n,\,n-j}^2\, v_j,

    with the truncation :math:`\theta_{n, m} = 0` for :math:`m > q`
    that obtains in the MA(q) limit.  The recursion is iterated for
    ``n = 1, …, n_iter`` and the last row :math:`\theta_{n_{\mathrm
    {iter}}, 1\ldots q}` is returned as the MA(q) coefficient
    estimate.

    The Python double-loop is fully unrolled at trace time
    (``q``, ``n_iter`` are both static), giving a compiled graph of
    size :math:`O(n_{\mathrm{iter}} \cdot q^2)`.  This is materially
    cheaper than a Hannan-Rissanen second stage (which would require
    a long pre-AR fit) and the only JIT-friendly way to estimate MA
    coefficients at MLE-quality from a closed form.

    Args:
        y: shape ``(n,)`` — input series.
        q: MA order (static Python ``int``).
        n_iter: Number of recursion steps; default
            ``min(n, max(50, 10 * q))`` — large enough for the
            estimator to converge to its asymptotic value on typical
            financial windows, while keeping the unrolled graph
            small enough to compile in seconds.

    Returns:
        Tuple ``(theta, v_final)`` where ``theta`` has shape ``(q,)``
        and ``v_final`` is a scalar — the estimated innovation
        variance after ``n_iter`` steps.
    """
    y = jnp.asarray(y, dtype=float).reshape(-1)
    n = int(y.shape[0])
    q = int(q)
    if n_iter is None:
        n_iter = min(n, max(50, 10 * q))
    n_iter = int(n_iter)
    if q == 0:
        # No MA coefficients to estimate; v_final is the sample
        # variance of the centred series, matching the recursion's
        # ``n_iter -> infinity`` limit at q = 0.
        return jnp.zeros((0,), dtype=float), jnp.var(y)

    gamma = acvf(y, n_iter)  # shape (n_iter+1,)

    # Storage:
    # - theta_history[n, m-1] holds θ_{n, m} for m = 1..q.  Rows for
    #   n < m are zero (recursion has not yet built up to that lag).
    # - v_history[n] holds v_n.
    theta_history = jnp.zeros((n_iter + 1, q), dtype=float)
    v_history = jnp.zeros((n_iter + 1,), dtype=float)
    v_history = v_history.at[0].set(gamma[0])

    for nn in range(1, n_iter + 1):
        # Build the new row θ_{nn, 1..q}.  Iterate m descending so
        # that θ_{nn, m'} for m' > m (which the inner sum needs) are
        # already filled.
        new_row = jnp.zeros((q,), dtype=float)
        for m in range(q, 0, -1):
            k = nn - m
            if k < 0:
                # n has not yet reached order m → θ_{nn, m} = 0.
                continue
            # Inner sum over j; nonzero terms are bounded by
            # j ∈ [max(0, k-q, nn-q), k-1].
            j_lo = max(0, k - q, nn - q)
            inner = jnp.zeros((), dtype=float)
            for j in range(j_lo, k):
                # θ_{k, k-j}: column index (k-j) - 1.
                # θ_{nn, nn-j}: column index (nn-j) - 1.
                theta_kj = theta_history[k, (k - j) - 1]
                theta_nnj = new_row[(nn - j) - 1]
                inner = inner + theta_kj * theta_nnj * v_history[j]
            v_k = v_history[k]
            v_k_safe = jnp.where(jnp.abs(v_k) < _V_FLOOR, _V_FLOOR, v_k)
            theta_nn_m = (gamma[m] - inner) / v_k_safe
            new_row = new_row.at[m - 1].set(theta_nn_m)
        theta_history = theta_history.at[nn].set(new_row)

        # v_n = γ(0) - Σ θ_{nn, nn-j}² v_j; nonzero terms have
        # j ∈ [max(0, nn-q), nn-1].
        j_lo = max(0, nn - q)
        v_nn = gamma[0]
        for j in range(j_lo, nn):
            theta_nnj = new_row[(nn - j) - 1]
            v_nn = v_nn - theta_nnj * theta_nnj * v_history[j]
        v_history = v_history.at[nn].set(v_nn)

    return theta_history[n_iter], v_history[n_iter]


###############################################################################
# ARMA initialisation
###############################################################################
def analytical_arma_params(
    y: ArrayLike, p: int, q: int, n_iter: int | None = None,
) -> dict:
    r"""Closed-form starting parameters for an ARMA(p, q) fit.

    Strategy (chained estimator):

    1. AR coefficients :math:`\phi` from :func:`yule_walker_ar` on
       the centred series.
    2. Mean-corrected AR-filtered residuals
       :math:`r_t = y_t - \bar y - \sum_i \phi_i\,(y_{t-i} - \bar y)`,
       padded with zeros for the pre-sample lags.
    3. MA coefficients :math:`\theta` from :func:`innovations_ma` on
       :math:`r_t`.
    4. Intercept :math:`c = \bar y \cdot (1 - \sum_i \phi_i)` so the
       unconditional mean of the implied ARMA process equals
       :math:`\bar y`.

    Args:
        y: shape ``(n,)`` — input series.
        p, q: ARMA orders (static Python ``int``).
        n_iter: Forwarded to :func:`innovations_ma`.

    Returns:
        ``{"phi": (p,), "theta": (q,), "c": ()}`` — JAX-traceable
        starting values suitable as the optimiser's ``x0``.
    """
    y = jnp.asarray(y, dtype=float).reshape(-1)
    p = int(p)
    q = int(q)

    mean = jnp.mean(y)
    centred = y - mean
    phi = yule_walker_ar(centred, p)

    if q == 0:
        theta = jnp.zeros((0,), dtype=float)
    else:
        n = int(centred.shape[0])
        # Build AR-filtered residuals: r_t = centred[t] - sum_i phi[i] * centred[t-i-1].
        # Pad the first p entries with zeros (no AR contribution available).
        if p == 0:
            r = centred
        else:
            r = jnp.zeros((n,), dtype=float)
            for i in range(p):
                # phi[i] coefficient on lag i+1: contribution at time t is phi[i] * centred[t - i - 1]
                shift = i + 1
                r = r.at[shift:].add(-phi[i] * centred[: n - shift])
            r = r.at[p:].add(centred[p:])
        theta, _ = innovations_ma(r, q, n_iter=n_iter)

    c = mean * (1.0 - jnp.sum(phi))
    return {
        "phi": phi,
        "theta": theta,
        "c": c.reshape(()),
    }


def sample_arma_params(y: ArrayLike, p: int, q: int) -> dict:
    r"""Trivial sample-mean starting parameters for ARMA(p, q).

    Sets every AR / MA coefficient to zero and ``c`` to the sample
    mean.  Useful as a fall-back when the analytical chained estimator
    is not appropriate (e.g. very short series where Yule-Walker is
    near-singular).
    """
    y = jnp.asarray(y, dtype=float).reshape(-1)
    return {
        "phi": jnp.zeros((int(p),), dtype=float),
        "theta": jnp.zeros((int(q),), dtype=float),
        "c": jnp.mean(y).reshape(()),
    }


def arma_pre_sample_state(
    y: ArrayLike, p: int, q: int, mode: str,
    backcast_length: int | None = None,
) -> tuple[Array, Array]:
    r"""Pre-sample lag arrays for the ARMA recursion's initial carry.

    Args:
        y: shape ``(n,)`` — input series.
        p, q: ARMA orders (static).
        mode: ``"backcast"`` — fill ``y_lags`` with the rolling mean
            of the first ``backcast_length`` observations and
            ``eps_lags`` with zeros (no innovations observed before
            the sample window starts);
            ``"sample"`` — same backcast but using the mean of the
            full series, equivalent to setting the recursion's
            pre-sample state to the unconditional level;
            ``"zero"`` — fill both buffers with zero.  Useful when
            the series is already centred or the modeller wants to
            absorb pre-sample uncertainty into the early-sample
            innovations.
        backcast_length: Length of the leading window used for the
            backcast under ``mode="backcast"``; ignored otherwise.

    Returns:
        Tuple ``(y_lags, eps_lags)`` of shapes ``(p,)`` and ``(q,)``.

    Raises:
        ValueError: When ``mode`` is unknown.
    """
    y = jnp.asarray(y, dtype=float).reshape(-1)
    p = int(p)
    q = int(q)
    if mode == "backcast":
        n = int(y.shape[0])
        length = n if backcast_length is None else int(backcast_length)
        length = max(1, min(length, n))
        anchor = jnp.mean(y[:length])
    elif mode == "sample":
        anchor = jnp.mean(y)
    elif mode == "zero":
        anchor = jnp.asarray(0.0, dtype=float)
    else:
        raise ValueError(
            f"Unknown ARMA pre-sample mode {mode!r}; expected one of "
            "{'backcast', 'sample', 'zero'}."
        )
    y_lags = jnp.full((p,), anchor)
    eps_lags = jnp.zeros((q,), dtype=float)
    return y_lags, eps_lags


###############################################################################
# GARCH initialisation
###############################################################################
def ewma_backcast(
    eps: ArrayLike, decay: float = 0.94,
) -> Array:
    r"""Exponentially-weighted variance estimate for GARCH backcast init.

    .. math::

        \hat\sigma^2_0 = (1 - \lambda) \sum_{t=0}^{n-1}
                                       \lambda^t\, \varepsilon_t^2,
        \qquad \lambda = \texttt{decay}.

    Default ``decay = 0.94`` matches the RiskMetrics convention used
    by both ``arch`` and ``rugarch`` for their backcast initialisation.

    Args:
        eps: shape ``(n,)`` — mean-corrected innovation series.
        decay: EWMA decay factor in ``(0, 1)``.
    """
    eps = jnp.asarray(eps, dtype=float).reshape(-1)
    eps_sq = eps * eps
    n = int(eps_sq.shape[0])
    weights = (1.0 - decay) * jnp.power(decay, jnp.arange(n, dtype=float))
    return jnp.sum(weights * eps_sq)


def analytical_garch_params(
    eps: ArrayLike, p: int, q: int,
    alpha_share: float = 0.05, beta_share: float = 0.90,
) -> dict:
    r"""Sample-moment starting parameters for GARCH(p, q).

    Splits the standard ``α + β ≈ 0.95`` industry prior across the
    requested orders:

    .. math::

        \alpha_i &= \texttt{alpha\_share} / p,\\
        \beta_j  &= \texttt{beta\_share}  / q,\\
        \omega   &= (1 - \alpha_{\mathrm{tot}} - \beta_{\mathrm{tot}})
                   \cdot \mathrm{var}(\varepsilon).

    The resulting unconditional variance equals the sample variance
    of the input innovation series, so the optimiser starts with a
    moment-consistent first iterate.

    Defaults match the ``arch`` / ``rugarch`` GARCH(1, 1) priors.
    Generalises cleanly to higher orders without re-tuning.
    """
    eps = jnp.asarray(eps, dtype=float).reshape(-1)
    p = int(p)
    q = int(q)
    alpha = jnp.full((p,), alpha_share / max(p, 1), dtype=float) if p > 0 else jnp.zeros((0,), dtype=float)
    beta = jnp.full((q,), beta_share / max(q, 1), dtype=float) if q > 0 else jnp.zeros((0,), dtype=float)
    persistence = jnp.sum(alpha) + jnp.sum(beta)
    omega = (1.0 - persistence) * jnp.var(eps)
    return {
        "omega": omega.reshape(()),
        "alpha": alpha,
        "beta": beta,
    }


def sample_garch_params(eps: ArrayLike, p: int, q: int) -> dict:
    r"""Sample-variance starting parameters for GARCH(p, q).

    Sets ``α = β = 0`` and ``ω = var(eps)`` — the ARCH(0) (i.i.d.
    Gaussian) hypothesis under which all heteroskedasticity is
    attributed to homoskedastic noise.  Useful as a fallback or as a
    null-comparison starting point.
    """
    eps = jnp.asarray(eps, dtype=float).reshape(-1)
    return {
        "omega": jnp.var(eps).reshape(()),
        "alpha": jnp.zeros((int(p),), dtype=float),
        "beta": jnp.zeros((int(q),), dtype=float),
    }


def garch_pre_sample_state(
    eps: ArrayLike, p: int, q: int, mode: str,
    backcast_length: int | None = None, decay: float = 0.94,
) -> tuple[Array, Array]:
    r"""Pre-sample lag arrays for the GARCH σ²-recursion.

    Args:
        eps: shape ``(n,)`` — mean-corrected innovation series.
        p, q: GARCH orders (static).
        mode: ``"backcast"`` — fill both buffers with the EWMA
            estimate from :func:`ewma_backcast` (computed over the
            first ``backcast_length`` observations);
            ``"sample"`` — fill both buffers with the sample
            variance of the full series;
            ``"zero"`` — fill both buffers with zero (rarely useful
            and prone to early-sample numerical underflow; provided
            for completeness).
        backcast_length: Forwarded to :func:`ewma_backcast` under
            ``mode="backcast"``; ignored otherwise.
        decay: EWMA decay factor under ``mode="backcast"``.

    Returns:
        Tuple ``(eps_sq_lags, var_lags)`` of shapes ``(p,)`` and
        ``(q,)``.
    """
    eps = jnp.asarray(eps, dtype=float).reshape(-1)
    p = int(p)
    q = int(q)
    if mode == "backcast":
        n = int(eps.shape[0])
        length = n if backcast_length is None else int(backcast_length)
        length = max(1, min(length, n))
        var_anchor = ewma_backcast(eps[:length], decay=decay)
    elif mode == "sample":
        var_anchor = jnp.var(eps)
    elif mode == "zero":
        var_anchor = jnp.asarray(0.0, dtype=float)
    else:
        raise ValueError(
            f"Unknown GARCH pre-sample mode {mode!r}; expected one of "
            "{'backcast', 'sample', 'zero'}."
        )
    var_anchor = jnp.maximum(var_anchor, 0.0)
    eps_sq_lags = jnp.full((p,), var_anchor)
    var_lags = jnp.full((q,), var_anchor)
    return eps_sq_lags, var_lags


###############################################################################
# Top-level dispatchers
###############################################################################
_ARMA_INIT_MODES = frozenset({"analytical", "sample", "backcast"})
_GARCH_INIT_MODES = frozenset({"analytical", "sample", "backcast"})


def init_arma_params(
    y: ArrayLike, p: int, q: int, mode: str = "analytical",
    n_iter: int | None = None,
) -> dict:
    r"""Dispatch ARMA(p, q) parameter starting values by mode.

    Modes:
        ``"analytical"`` — :func:`analytical_arma_params` (Yule-Walker
        + Innovations Algorithm); the default.
        ``"backcast"`` — same starting values as ``"sample"`` for the
        AR / MA coefficients, with ``c`` set to the rolling mean over
        the leading observations.  Per plan, the *parameter* init
        under ``"backcast"`` collapses to the sample mean since AR /
        MA coefficients have no closed-form moment estimator
        independent of ACVF input.  The user-controlled
        ``backcast_length`` kwarg primarily affects the recursion's
        pre-sample state via :func:`arma_pre_sample_state`.
        ``"sample"`` — :func:`sample_arma_params` (zero AR / MA + sample mean).

    Raises:
        ValueError: When ``mode`` is unknown.
    """
    if mode not in _ARMA_INIT_MODES:
        raise ValueError(
            f"Unknown ARMA init mode {mode!r}; expected one of "
            f"{sorted(_ARMA_INIT_MODES)}."
        )
    if mode == "analytical":
        return analytical_arma_params(y, p, q, n_iter=n_iter)
    return sample_arma_params(y, p, q)


def init_garch_params(
    eps: ArrayLike, p: int, q: int, mode: str = "analytical",
    backcast_length: int | None = None, decay: float = 0.94,
) -> dict:
    r"""Dispatch GARCH(p, q) parameter starting values by mode.

    Modes:
        ``"analytical"`` — :func:`analytical_garch_params` (industry
        ``α=0.05, β=0.9`` prior split across orders).
        ``"backcast"`` — :func:`analytical_garch_params` with
        ``ω = (1 - α - β) · ewma_backcast(eps, decay)`` instead of
        sample variance — matches the standard backcast prior.
        ``"sample"`` — :func:`sample_garch_params` (ARCH(0) null).

    Raises:
        ValueError: When ``mode`` is unknown.
    """
    if mode not in _GARCH_INIT_MODES:
        raise ValueError(
            f"Unknown GARCH init mode {mode!r}; expected one of "
            f"{sorted(_GARCH_INIT_MODES)}."
        )
    if mode == "analytical":
        return analytical_garch_params(eps, p, q)
    if mode == "sample":
        return sample_garch_params(eps, p, q)
    # backcast: same α / β prior, ω rescaled to backcast-implied unconditional var
    eps = jnp.asarray(eps, dtype=float).reshape(-1)
    n = int(eps.shape[0])
    length = n if backcast_length is None else int(backcast_length)
    length = max(1, min(length, n))
    var_anchor = ewma_backcast(eps[:length], decay=decay)
    base = analytical_garch_params(eps, p, q)
    persistence = jnp.sum(base["alpha"]) + jnp.sum(base["beta"])
    base = dict(base)
    base["omega"] = ((1.0 - persistence) * var_anchor).reshape(())
    return base
