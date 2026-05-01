"""Time-series models for CopulAX.

Provides AR / MA / ARMA mean-equation models alongside the
forthcoming GARCH-family conditional-variance models and the joint
``ArmaGarch`` composite estimator.  Every model is JIT-compatible,
autograd-compatible, and supports warm-start fitting for fast
rolling-window refits.

Innovations are drawn from any standardised (mean=0, var=1) law on
the residual whitelist — currently ``normal``, ``student_t``,
``gen_normal``, ``nig``, ``gh``, and ``skewed_t``.

Models are configured at construction time and fit on data:

.. code-block:: python

    from copulax.timeseries import ARMA, GARCH
    from copulax.univariate import normal, student_t

    arma_fit = ARMA(p=1, q=1, residual_dist=student_t).fit(y)
    garch_fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps)

The ``(p, q, residual_dist)`` triple is part of the model's
**static** configuration — it parameterises the compiled fit graph
and is fixed for the lifetime of the instance.  Construct a new
instance to fit a different specification.
"""

from copulax._src.timeseries._diagnostics import (
    acf,
    arch_lm,
    ljung_box,
    pacf,
    plot_acf,
    plot_pacf,
)
from copulax._src.timeseries._joint import ArmaGarch
from copulax._src.timeseries._mean import AR, ARMA, MA
from copulax._src.timeseries._variance import (
    EGARCH,
    GARCH,
    GARCH_M,
    GJR_GARCH,
    IGARCH,
    QGARCH,
    TGARCH,
)

__all__ = [
    # mean models
    "AR", "MA", "ARMA",
    # variance models
    "GARCH",
    "IGARCH",
    "GJR_GARCH",
    "EGARCH",
    "TGARCH",
    "QGARCH",
    "GARCH_M",
    # joint composite
    "ArmaGarch",
    # diagnostics
    "acf", "pacf", "ljung_box", "arch_lm",
    "plot_acf", "plot_pacf",
]
