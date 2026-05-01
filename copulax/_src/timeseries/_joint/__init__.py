"""Internal joint mean-and-variance composite estimators.

The ``ArmaGarch`` composite couples an ARMA(p, q) mean equation
with a GARCH-family conditional-variance equation under a single
joint MLE — the textbook Bollerslev (1986) estimator.  Public
access is via :mod:`copulax.timeseries`.
"""

from copulax._src.timeseries._joint.arma_garch import (
    ArmaGarch,
    ArmaGarchTerminalState,
)

__all__ = [
    "ArmaGarch", "ArmaGarchTerminalState",
]
