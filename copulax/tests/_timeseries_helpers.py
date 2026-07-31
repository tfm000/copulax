"""Shared data simulators for the time-series test family.

Every ``test_timeseries_*.py`` module previously carried its own copy of
the same handful of simulators — five processes spread across nine
near-duplicate definitions in six files.  This module holds exactly one
definition per process; the test modules import from here.

The module name does not match ``test_*.py``, so pytest never collects
it.  ``copulax/tests/__init__.py`` exists and ``pytest.ini`` sets
``pythonpath = .``, so ``from copulax.tests._timeseries_helpers import
...`` resolves from any test module.

Consolidation invariant
-----------------------
Every function here reproduces the series produced by the local
definition it replaced **byte for byte** at every existing call site.
Where two local definitions implemented genuinely different
initialisations of the same process (MA(1) and ARMA(1, 1): one form
burns a pre-sample innovation, the other starts from a zero
pre-sample state), the difference is exposed as an explicit ``init``
argument rather than silently harmonised — harmonising would change
the simulated series, and the series are inputs to frozen statistical
assertions.

Conventions
-----------
* Mean processes are written in **centred form**,
  ``y_t = mu + phi (y_{t-1} - mu) + ...``, matching the production ARMA
  recursion.  ``mu = 0.0`` (the default) reduces this exactly to the
  uncentred form used by the majority of call sites.
* PRNG keys are threaded explicitly; nothing here reads global random
  state.  The ``*_series`` wrappers at the bottom take an integer seed
  purely as a convenience for call sites that were written that way.
* ``sigma`` scales the innovations.  ``sigma = 1.0`` (the default) is a
  numerical no-op: ``1.0 * x`` returns ``x`` unchanged.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp


__all__ = [
    "ar1_from_innovations",
    "simulate_ar1",
    "simulate_arp",
    "simulate_ma1",
    "simulate_arma11",
    "simulate_garch11",
    "simulate_ar1_garch11",
    "ar1_series",
    "garch11_series",
]


# ---------------------------------------------------------------------------
# Mean processes
# ---------------------------------------------------------------------------

def ar1_from_innovations(
    phi: float,
    eps: jax.Array,
    *,
    mu: float = 0.0,
) -> jax.Array:
    r"""Roll an AR(1) recursion over a supplied innovation sequence.

    Implements the centred recursion

    .. math::
        y_t = \mu + \phi (y_{t-1} - \mu) + \varepsilon_t,

    seeded at :math:`y_{-1} = \mu`, so the first observation is
    :math:`y_0 = \mu + \varepsilon_0`.

    Use this when the innovations are already in hand (for example when
    injecting an AR(1) component into a GARCH residual series); use
    :func:`simulate_ar1` when the innovations should be drawn from a
    standard normal.

    Parameters
    ----------
    phi : float
        Autoregressive coefficient.
    eps : jax.Array
        Innovation sequence, shape ``(n,)``.
    mu : float, optional
        Unconditional mean of the process.  Default ``0.0``.

    Returns
    -------
    jax.Array
        The simulated series, shape ``(n,)``.
    """
    def step(y_prev, eps_t):
        y_t = mu + phi * (y_prev - mu) + eps_t
        return y_t, y_t

    _, ys = jax.lax.scan(step, jnp.asarray(mu, dtype=eps.dtype), eps)
    return ys


def simulate_ar1(
    n: int,
    phi: float,
    key: jax.Array,
    *,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> jax.Array:
    r"""Simulate an AR(1) process with standard-normal innovations.

    .. math::
        y_t = \mu + \phi (y_{t-1} - \mu) + \sigma z_t,
        \qquad z_t \sim N(0, 1),

    seeded at :math:`y_{-1} = \mu`.

    Parameters
    ----------
    n : int
        Number of observations to simulate.
    phi : float
        Autoregressive coefficient.
    key : jax.Array
        PRNG key drawing the ``n`` innovations.
    mu : float, optional
        Unconditional mean of the process.  Default ``0.0``.
    sigma : float, optional
        Innovation standard deviation.  Default ``1.0``.

    Returns
    -------
    jax.Array
        The simulated series, shape ``(n,)``.

    Note
    ----
    If you intend to jit wrap this function, ensure that ``n`` is a
    static argument.
    """
    eps = sigma * jax.random.normal(key, (n,))
    return ar1_from_innovations(phi, eps, mu=mu)


def simulate_arp(
    n: int,
    phi: Sequence[float],
    key: jax.Array,
    *,
    sigma: float = 1.0,
) -> jax.Array:
    r"""Simulate an AR(p) process via a lag-window scan.

    .. math::
        y_t = \sum_{i=1}^{p} \phi_i y_{t-i} + \sigma z_t,
        \qquad z_t \sim N(0, 1),

    seeded at a zero lag window.

    Parameters
    ----------
    n : int
        Number of observations to simulate.
    phi : Sequence[float]
        Autoregressive coefficients, most recent lag first.  Its length
        sets the order ``p``.
    key : jax.Array
        PRNG key drawing the ``n`` innovations.
    sigma : float, optional
        Innovation standard deviation.  Default ``1.0``.

    Returns
    -------
    jax.Array
        The simulated series, shape ``(n,)``.

    Note
    ----
    If you intend to jit wrap this function, ensure that ``n`` and the
    length of ``phi`` are static arguments.
    """
    p = len(phi)
    phi_arr = jnp.asarray(phi, dtype=float)
    eps = sigma * jax.random.normal(key, (n,))

    def step(carry, e):
        # carry holds the last p observations, most recent first.
        mu = jnp.dot(phi_arr, carry)
        new_y = mu + e
        new_carry = jnp.concatenate([new_y.reshape((1,)), carry[:-1]])
        return new_carry, new_y

    init_carry = jnp.zeros((p,), dtype=float)
    _, ys = jax.lax.scan(step, init_carry, eps)
    return ys


def simulate_ma1(
    n: int,
    theta: float,
    key: jax.Array,
    *,
    mu: float = 0.0,
    sigma: float = 1.0,
    init: str = "pre_sample",
) -> jax.Array:
    r"""Simulate an MA(1) process with standard-normal innovations.

    .. math::
        y_t = \mu + \varepsilon_t + \theta \varepsilon_{t-1},
        \qquad \varepsilon_t = \sigma z_t, \quad z_t \sim N(0, 1).

    Parameters
    ----------
    n : int
        Number of observations to simulate.
    theta : float
        Moving-average coefficient.
    key : jax.Array
        PRNG key drawing the innovations — ``n + 1`` of them under
        ``init="pre_sample"``, ``n`` under ``init="zero"``.
    mu : float, optional
        Unconditional mean of the process.  Default ``0.0``.
    sigma : float, optional
        Innovation standard deviation.  Default ``1.0``.
    init : {"pre_sample", "zero"}, optional
        Pre-sample innovation convention.  ``"pre_sample"`` (default)
        draws one extra innovation and uses it as
        :math:`\varepsilon_{-1}`, so every observation carries a genuine
        MA term.  ``"zero"`` sets :math:`\varepsilon_{-1} = 0`, so
        :math:`y_0 = \mu + \varepsilon_0`.  The two conventions produce
        different series from the same key and are **not**
        interchangeable.

    Returns
    -------
    jax.Array
        The simulated series, shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``init`` is not one of the two supported conventions.

    Note
    ----
    If you intend to jit wrap this function, ensure that ``n`` and
    ``init`` are static arguments.
    """
    if init == "pre_sample":
        eps = sigma * jax.random.normal(key, (n + 1,))
        return mu + eps[1:] + theta * eps[:-1]

    if init == "zero":
        eps = sigma * jax.random.normal(key, (n,))

        def step(eps_lag, e):
            y_t = mu + theta * eps_lag + e
            return e, y_t

        _, ys = jax.lax.scan(step, jnp.array(0.0), eps)
        return ys

    raise ValueError(
        f"init must be 'pre_sample' or 'zero', got {init!r}."
    )


def simulate_arma11(
    n: int,
    phi: float,
    theta: float,
    key: jax.Array,
    *,
    mu: float = 0.0,
    sigma: float = 1.0,
    init: str = "pre_sample",
) -> jax.Array:
    r"""Simulate an ARMA(1, 1) process with standard-normal innovations.

    .. math::
        y_t = \mu + \phi (y_{t-1} - \mu) + \varepsilon_t
              + \theta \varepsilon_{t-1},
        \qquad \varepsilon_t = \sigma z_t, \quad z_t \sim N(0, 1).

    Parameters
    ----------
    n : int
        Number of observations to simulate.
    phi : float
        Autoregressive coefficient.
    theta : float
        Moving-average coefficient.
    key : jax.Array
        PRNG key drawing the innovations — ``n + 1`` of them under
        ``init="pre_sample"``, ``n`` under ``init="zero"``.
    mu : float, optional
        Unconditional mean of the process.  Default ``0.0``.
    sigma : float, optional
        Innovation standard deviation.  Default ``1.0``.
    init : {"pre_sample", "zero"}, optional
        Pre-sample state convention.  ``"pre_sample"`` (default) draws
        one extra innovation for :math:`\varepsilon_{-1}` and seeds
        :math:`y_{-1} = \mu`.  ``"zero"`` seeds both the lagged
        observation and the lagged innovation at zero.  The two
        conventions produce different series from the same key and are
        **not** interchangeable.

    Returns
    -------
    jax.Array
        The simulated series, shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``init`` is not one of the two supported conventions.

    Note
    ----
    If you intend to jit wrap this function, ensure that ``n`` and
    ``init`` are static arguments.
    """
    if init == "pre_sample":
        eps = sigma * jax.random.normal(key, (n + 1,))

        def step(carry, inp):
            y_prev, eps_prev = carry
            eps_curr = inp
            y_t = mu + phi * (y_prev - mu) + eps_curr + theta * eps_prev
            return (y_t, eps_curr), y_t

        init_carry = (mu + eps[1] + theta * eps[0], eps[1])
        _, ys = jax.lax.scan(step, init_carry, eps[2:])
        return jnp.concatenate([init_carry[0].reshape((1,)), ys])

    if init == "zero":
        eps = sigma * jax.random.normal(key, (n,))

        def step(carry, e):
            y_lag, eps_lag = carry
            y_t = mu + phi * (y_lag - mu) + theta * eps_lag + e
            return (y_t, e), y_t

        _, ys = jax.lax.scan(
            step, (jnp.array(0.0), jnp.array(0.0)), eps,
        )
        return ys

    raise ValueError(
        f"init must be 'pre_sample' or 'zero', got {init!r}."
    )


# ---------------------------------------------------------------------------
# Variance processes
# ---------------------------------------------------------------------------

def simulate_garch11(
    n: int,
    omega: float,
    alpha: float,
    beta: float,
    key: jax.Array,
) -> jax.Array:
    r"""Simulate a GARCH(1, 1) residual series.

    .. math::
        \sigma^2_t = \omega + \alpha \varepsilon^2_{t-1}
                     + \beta \sigma^2_{t-1},
        \qquad \varepsilon_t = \sigma_t z_t, \quad z_t \sim N(0, 1),

    seeded at the unconditional variance
    :math:`\omega / (1 - \alpha - \beta)` for both the lagged variance
    and the lagged squared residual.

    Parameters
    ----------
    n : int
        Number of observations to simulate.
    omega : float
        Variance intercept.
    alpha : float
        ARCH coefficient.
    beta : float
        GARCH coefficient.
    key : jax.Array
        PRNG key drawing the ``n`` standardised innovations.

    Returns
    -------
    jax.Array
        The simulated residual series, shape ``(n,)``.

    Note
    ----
    If you intend to jit wrap this function, ensure that ``n`` is a
    static argument.
    """
    sigma2_uncond = omega / (1.0 - alpha - beta)
    z = jax.random.normal(key, (n,))

    def step(carry, z_t):
        sigma2_prev, eps2_prev = carry
        sigma2_t = omega + alpha * eps2_prev + beta * sigma2_prev
        eps_t = jnp.sqrt(sigma2_t) * z_t
        return (sigma2_t, eps_t * eps_t), eps_t

    _, eps = jax.lax.scan(step, (sigma2_uncond, sigma2_uncond), z)
    return eps


def simulate_ar1_garch11(
    n: int,
    phi: float,
    omega: float,
    alpha: float,
    beta: float,
    key: jax.Array,
    *,
    mu: float = 0.0,
) -> jax.Array:
    r"""Simulate an AR(1) level process driven by GARCH(1, 1) residuals.

    .. math::
        \sigma^2_t = \omega + \alpha \varepsilon^2_{t-1}
                     + \beta \sigma^2_{t-1},
        \qquad \varepsilon_t = \sigma_t z_t,
        \qquad y_t = \mu + \phi (y_{t-1} - \mu) + \varepsilon_t,

    seeded at :math:`y_{-1} = \mu` and at the unconditional variance for
    both variance-recursion lags.

    ``mu`` is keyword-only and trails the variance parameters: the level
    mean is a refinement of the process, and placing it positionally
    would silently reorder the variance arguments at call sites that
    omit it.

    Parameters
    ----------
    n : int
        Number of observations to simulate.
    phi : float
        Autoregressive coefficient of the level equation.
    omega : float
        Variance intercept.
    alpha : float
        ARCH coefficient.
    beta : float
        GARCH coefficient.
    key : jax.Array
        PRNG key drawing the ``n`` standardised innovations.
    mu : float, optional
        Unconditional mean of the level process.  Default ``0.0``.

    Returns
    -------
    jax.Array
        The simulated level series, shape ``(n,)``.

    Note
    ----
    If you intend to jit wrap this function, ensure that ``n`` is a
    static argument.
    """
    sigma2_uncond = omega / (1.0 - alpha - beta)
    z = jax.random.normal(key, (n,))

    def step(carry, z_t):
        y_lag, sigma2_lag, eps_sq_lag = carry
        sigma2_t = omega + alpha * eps_sq_lag + beta * sigma2_lag
        eps_t = jnp.sqrt(sigma2_t) * z_t
        y_t = mu + phi * (y_lag - mu) + eps_t
        return (y_t, sigma2_t, eps_t * eps_t), y_t

    _, y = jax.lax.scan(
        step, (mu, sigma2_uncond, sigma2_uncond), z,
    )
    return y


# ---------------------------------------------------------------------------
# Seed-taking convenience wrappers
# ---------------------------------------------------------------------------

def ar1_series(
    n: int = 500,
    phi: float = 0.6,
    seed: int = 13,
) -> jax.Array:
    r"""Seed-taking wrapper over :func:`simulate_ar1`.

    Parameters
    ----------
    n : int, optional
        Number of observations to simulate.  Default ``500``.
    phi : float, optional
        Autoregressive coefficient.  Default ``0.6``.
    seed : int, optional
        Integer seed for :func:`jax.random.PRNGKey`.  Default ``13``.

    Returns
    -------
    jax.Array
        The simulated series, shape ``(n,)``.

    Note
    ----
    If you intend to jit wrap this function, ensure that ``n`` is a
    static argument.
    """
    return simulate_ar1(n, phi, jax.random.PRNGKey(seed))


def garch11_series(
    n: int = 500,
    omega: float = 0.05,
    alpha: float = 0.10,
    beta: float = 0.85,
    seed: int = 2,
) -> jax.Array:
    r"""Seed-taking wrapper over :func:`simulate_garch11`.

    Parameters
    ----------
    n : int, optional
        Number of observations to simulate.  Default ``500``.
    omega : float, optional
        Variance intercept.  Default ``0.05``.
    alpha : float, optional
        ARCH coefficient.  Default ``0.10``.
    beta : float, optional
        GARCH coefficient.  Default ``0.85``.
    seed : int, optional
        Integer seed for :func:`jax.random.PRNGKey`.  Default ``2``.

    Returns
    -------
    jax.Array
        The simulated residual series, shape ``(n,)``.

    Note
    ----
    If you intend to jit wrap this function, ensure that ``n`` is a
    static argument.
    """
    return simulate_garch11(n, omega, alpha, beta, jax.random.PRNGKey(seed))
