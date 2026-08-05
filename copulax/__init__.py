"""copulAX — JAX-based probability distributions and copula library."""

from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
)
from importlib.metadata import (
    version as _version,
)

from copulax._src._serialization import load
from copulax._src._utils import get_random_key

try:
    __version__ = _version("copulax")
except _PackageNotFoundError:
    __version__ = "0.0.0+unknown"
