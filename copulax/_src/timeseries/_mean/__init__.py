"""Internal mean-equation models (AR / MA / ARMA).

Shared scaffolding lives in :mod:`._arma_base`; concrete classes in
:mod:`.ar`, :mod:`.ma`, and :mod:`.arma`.  Public access is via
:mod:`copulax.timeseries`.
"""

from copulax._src.timeseries._mean._arma_base import ARMABase, ARMATerminalState
from copulax._src.timeseries._mean.ar import AR
from copulax._src.timeseries._mean.ma import MA
from copulax._src.timeseries._mean.arma import ARMA

__all__ = [
    "ARMABase", "ARMATerminalState",
    "AR", "MA", "ARMA",
]
