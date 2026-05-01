"""Time-series models for CopulAX.

Provides AR / MA / ARMA mean-equation models alongside the
forthcoming GARCH-family conditional-variance models and the joint
``arma_garch`` composite estimator.  Every model is JIT-compatible,
autograd-compatible, and supports warm-start fitting for fast
rolling-window refits.

Innovations are drawn from any standardised (mean=0, var=1) law on
the residual whitelist — currently ``normal``, ``student_t``,
``gen_normal``, ``nig``, ``gh``, and ``skewed_t``.

See :mod:`copulax._src.timeseries._mean._arma_base` for the full
mean-model fit / forecast / residual / stats contract.

Example:
    >>> import jax.random
    >>> from copulax.timeseries import arma
    >>> from copulax.univariate import student_t
    >>> y = jax.random.normal(jax.random.PRNGKey(0), (500,))
    >>> fit = arma.fit(y, p=1, q=1, residual_dist=student_t)  # doctest: +SKIP
    >>> fit.params  # doctest: +SKIP
"""

from copulax._src.timeseries._mean import (
    AR, ar,
    ARMA, arma,
    MA, ma,
    ARMABase, ARMATerminalState,
)

__all__ = [
    "ar", "AR",
    "ma", "MA",
    "arma", "ARMA",
    "ARMABase", "ARMATerminalState",
]
