"""Marker base for the typed-parameter object hierarchy.

Defines :class:`ParamsBase`, the bare :mod:`equinox` ``Module`` marker
that every per-family typed-parameter class will inherit from once the
4.0.0 migration lands (Phase 3).  In Phase 0 the marker ships on its own
so the guard skeleton has a stable ``isinstance`` target and Phase 3 has
a single anchor to extend.

The base is deliberately minimal — a docstring-only ``eqx.Module`` with
no fields, no constructor, no hash / equality overrides, and no field
metadata declarations.  Parameter-hash semantics (identity vs.
static-fields-only vs. frozen-floats) and the self-describing field
metadata machinery are deliberate Phase 3 deferrals: they depend on
use-cases resolved in the params-phase discussion, and pulling any of
them forward now would pre-commit a decision the milestone has explicitly
left open.  This mirrors the shape of
:class:`copulax._src.timeseries._base.TerminalState`, the codebase's
existing bare-marker ``eqx.Module``.
"""

from __future__ import annotations

import equinox as eqx


class ParamsBase(eqx.Module):
    r"""Marker base for per-family typed-parameter PyTrees.

    Every distribution / model family's typed-parameter class will
    subclass :class:`ParamsBase` in Phase 3, replacing the raw
    ``dict`` parameters accepted at the public API today.  Because
    :class:`ParamsBase` is an :mod:`equinox` ``Module``, subclasses are
    immutable, JIT-/vmap-/grad-compatible registered PyTrees out of the
    box.

    The base provides only the marker for type checking and as the
    Phase 3 extension anchor.  It intentionally declares no fields, no
    hash / equality semantics, and no field-metadata machinery — those
    are resolved per the params-phase discussion and added on the
    subclasses (or here) at that point, not before.
    """
