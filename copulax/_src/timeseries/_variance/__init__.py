"""Internal conditional-variance models (GARCH family).

Shared σ²-form scaffolding lives in :mod:`._garch_base`; concrete
classes in :mod:`.garch` (and forthcoming :mod:`.igarch`,
:mod:`.gjr_garch`, :mod:`.egarch`, :mod:`.tgarch`, :mod:`.qgarch`,
:mod:`.garch_m`).  Public access is via :mod:`copulax.timeseries`.
"""

from copulax._src.timeseries._variance._garch_base import (
    GARCHBase,
    GARCHTerminalState,
)
from copulax._src.timeseries._variance.garch import GARCH

__all__ = [
    "GARCHBase", "GARCHTerminalState",
    "GARCH",
]
