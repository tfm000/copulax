"""Exception type for the typed-parameter migration guard.

Defines :class:`ParamsTypeError`, raised when a family that has been
migrated to typed parameters (Phase 3) is handed a raw ``dict`` at the
public API instead of its typed-parameter object.  The exception
subclasses the built-in :class:`TypeError` so existing ``except
TypeError`` sites keep catching it and callers can assert on the
concrete class.

The module-level :func:`migration_hint` builder centralises the message
text so Phase 3 can refine the wording (e.g. once every family's typed
class name is known) in exactly one place, rather than scattering
f-strings across the guarded call sites.
"""

from __future__ import annotations


def migration_hint(name: str, typed_class_name: str | None = None) -> str:
    r"""Build the user-facing message for a rejected raw-``dict`` params.

    The message states what is wrong, names the family, names the typed
    parameter class the caller should build instead, and gives the exact
    ``<Class>.from_dict(...)`` recipe to migrate an existing dict.  When
    the typed class name is not yet known (Phase 0, before the per-family
    classes exist) it falls back to generic wording that still points at
    the ``from_dict`` conversion path.

    Args:
        name: The family-name key (e.g. ``"normal"``, ``"IGARCH"``) the
            raw dict was passed for.
        typed_class_name: The name of the family's typed-parameter class
            (e.g. ``"NormalParams"``).  Optional — omitted in Phase 0.

    Returns:
        The formatted, user-facing error message.
    """
    if typed_class_name:
        return (
            f"Family {name!r} no longer accepts a raw dict for its "
            f"parameters. Pass a {typed_class_name} instance instead. "
            f"To migrate an existing parameter dict, use "
            f"{typed_class_name}.from_dict(params)."
        )
    return (
        f"Family {name!r} no longer accepts a raw dict for its "
        f"parameters. Pass the family's typed parameter object instead. "
        f"To migrate an existing parameter dict, use the typed params "
        f"class's from_dict(params) constructor."
    )


class ParamsTypeError(TypeError):
    r"""Raised when a migrated family receives a raw dict for its params.

    Subclasses :class:`TypeError` so that callers catching ``TypeError``
    continue to catch this, and tests can assert on the concrete class.
    The message (built via :func:`migration_hint`) names the family, the
    expected typed-parameter class, and the ``from_dict`` migration
    recipe.
    """
