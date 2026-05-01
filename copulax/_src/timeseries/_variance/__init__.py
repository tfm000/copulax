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
from copulax._src.timeseries._variance.egarch import EGARCH, EGARCHTerminalState
from copulax._src.timeseries._variance.garch import GARCH
from copulax._src.timeseries._variance.gjr_garch import GJR_GARCH, GJRTerminalState
from copulax._src.timeseries._variance.igarch import IGARCH

__all__ = [
    "GARCHBase", "GARCHTerminalState",
    "EGARCH", "EGARCHTerminalState",
    "GARCH",
    "GJR_GARCH", "GJRTerminalState",
    "IGARCH",
]
