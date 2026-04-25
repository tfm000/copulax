"""copulAX — JAX-based probability distributions and copula library."""

from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
    version as _version,
)

from copulax._src._utils import get_random_key
from copulax._src._serialization import load

try:
    __version__ = _version("copulax")
except _PackageNotFoundError:
    __version__ = "0.0.0+unknown"
