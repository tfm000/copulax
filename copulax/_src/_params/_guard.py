"""Dict-rejection guard for the typed-parameter migration.

Defines :func:`guard_params`, the single entry-point check wired into the
library's user-facing parameter-resolution chokepoints
(:meth:`copulax._src._distributions.Distribution._resolve_params` and
:meth:`copulax._src.timeseries._base.TimeSeriesModel._guard_residual_params`).

The guard consults the module-level :data:`_MIGRATED_FAMILIES` set: while
a family name is absent from that set the guard is a pure pass-through, so
every existing dict-based family keeps working unchanged.  Once a family
has been migrated to typed parameters, passing a raw ``dict`` for it
raises :class:`ParamsTypeError` with a ``from_dict`` migration hint.

To migrate a family to typed parameters (Phase 3):

1. Implement the family's typed-parameter class (subclassing
   :class:`copulax._src._params._base.ParamsBase`) with ``from_dict`` /
   ``to_dict`` conversion helpers.
2. Route every user-facing parameter-entry site for that family through
   its typed class, and route the ``.cpx`` load path through an
   **unguarded** internal reconstruction (see the load-path caveat on
   :meth:`TimeSeriesModel._guard_residual_params`) so pre-4.0.0 files
   keep loading.
3. Add the family-name key to :data:`_MIGRATED_FAMILIES` below.  From
   then on :func:`guard_params` rejects raw dicts for that family.

The guard is JIT-safe by construction: it inspects a user-supplied Python
object (``dict`` vs. typed object) at Python / trace time, not a traced
array, so a plain ``isinstance`` check here does not violate the
"no Python control flow over traced values" rule.
"""

from __future__ import annotations

from typing import TypeVar

#: The parameter object flowing through the guard.  The guard is a pure
#: pass-through, so binding the argument and the return to one variable
#: preserves the caller's own parameter type across the call instead of
#: erasing it to ``Any``.
_ParamsT = TypeVar("_ParamsT")

#: Family-name keys whose parameters have been migrated to typed objects.
#: EMPTY through Phases 0-2, so :func:`guard_params` is a behavioural
#: no-op for every family today — raw dicts pass everywhere.  Phase 3
#: adds one name at a time as each family is migrated (see the module
#: docstring for the full migration recipe).
_MIGRATED_FAMILIES: set[str] = set()


def guard_params(name: str, params: _ParamsT) -> _ParamsT:
    r"""Reject raw dicts for migrated families; pass everything else.

    Args:
        name: The family-name key for the caller (e.g. ``"normal"``,
            ``"IGARCH"``).  This is the key looked up in
            :data:`_MIGRATED_FAMILIES`.
        params: The user-supplied parameters — a raw ``dict`` today, a
            typed :class:`~copulax._src._params._base.ParamsBase`
            instance once ``name`` is migrated.

    Returns:
        ``params`` unchanged.  The guard never mutates or copies the
        parameters; it only raises for a rejected raw dict.

    Raises:
        ParamsTypeError: if ``name`` is in :data:`_MIGRATED_FAMILIES`
            **and** ``params`` is a raw ``dict``.  The message names the
            family and the ``from_dict`` migration recipe.
    """
    if name in _MIGRATED_FAMILIES and isinstance(params, dict):
        # Deferred import avoids any import cycle between this guard and
        # the exception module's future dependencies.
        from copulax._src._params._exceptions import (
            ParamsTypeError,
            migration_hint,
        )

        raise ParamsTypeError(migration_hint(name))
    return params
