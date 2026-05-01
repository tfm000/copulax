"""Internal mean-equation models (AR / MA / ARMA).

The shared scaffolding lives in :mod:`._arma_base`; concrete
singletons in :mod:`.ar`, :mod:`.ma`, and :mod:`.arma`.  Public access
is via :mod:`copulax.timeseries`.
"""

from copulax._src.timeseries._mean._arma_base import ARMABase, ARMATerminalState
from copulax._src.timeseries._mean.ar import AR, ar
from copulax._src.timeseries._mean.ma import MA, ma
from copulax._src.timeseries._mean.arma import ARMA, arma

__all__ = [
    "ARMABase", "ARMATerminalState",
    "AR", "ar",
    "MA", "ma",
    "ARMA", "arma",
]
