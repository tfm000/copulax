"""Internal typed-parameter package (Phase 0 skeleton).

Re-exports the guard skeleton for **internal** callers only.  There is no
public ``copulax.params`` / ``copulax.exceptions`` surface in Phase 0 —
the typed-parameter API ships in Phase 3, once every family is migrated
and the guard can actually fire.  Until then this package provides:

* :class:`ParamsBase` — the bare ``equinox.Module`` marker every typed
  params class will subclass (Phase 3 anchor);
* :class:`ParamsTypeError` — the ``TypeError`` subclass raised when a
  migrated family receives a raw dict;
* :func:`guard_params` — the dict-rejection guard wired into the
  parameter-resolution chokepoints;
* :data:`_MIGRATED_FAMILIES` — the (currently empty) set of migrated
  family-name keys the guard consults.
"""

from copulax._src._params._base import ParamsBase
from copulax._src._params._exceptions import ParamsTypeError, migration_hint
from copulax._src._params._guard import _MIGRATED_FAMILIES, guard_params

__all__ = [
    "ParamsBase",
    "ParamsTypeError",
    "migration_hint",
    "guard_params",
    "_MIGRATED_FAMILIES",
]
