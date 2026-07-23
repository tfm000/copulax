"""Unit tests for the internal ``copulax._src._params`` guard skeleton.

The guard (``copulax/_src/_params/_guard.py``) is the early-landing
dict-rejection chokepoint for the 4.0.0 typed-parameter migration.  Its
contract in Phase 0:

* ``guard_params(name, params)`` is a **no-op** for every family while
  ``_MIGRATED_FAMILIES`` is empty — it returns ``params`` unchanged so
  every existing dict-based family stays green.
* Once a family name is added to ``_MIGRATED_FAMILIES`` (Phase 3), a raw
  ``dict`` passed for that family raises :class:`ParamsTypeError` — a
  ``TypeError`` subclass whose message names the family and the
  ``from_dict`` migration recipe.

The raise-path checks deliberately call the *unwrapped* Python function.
``guard_params`` raises at Python (trace) time rather than inside a
JIT-compiled or grad-traced computation, so wrapping the call in
``jax.jit`` / ``jax.grad`` would surface as a trace-time failure rather
than the documented :class:`ParamsTypeError`.  Future contributors: do
not be tempted to "JIT-test" this path (this mirrors the rationale in
``test_resolve_params.py``).
"""

import equinox as eqx
import pytest

from copulax._src._params import ParamsBase, ParamsTypeError, guard_params
from copulax._src._params import _guard


def test_package_imports():
    """``import copulax._src._params`` succeeds (SC-4)."""
    import copulax._src._params as params_pkg

    assert params_pkg.ParamsBase is ParamsBase
    assert params_pkg.ParamsTypeError is ParamsTypeError
    assert params_pkg.guard_params is guard_params


def test_params_base_is_equinox_module():
    """``ParamsBase`` is a PyTree marker — an ``equinox.Module`` subclass."""
    assert issubclass(ParamsBase, eqx.Module)


def test_params_type_error_is_type_error():
    """``ParamsTypeError`` is catchable as a plain ``TypeError`` (D-12)."""
    assert issubclass(ParamsTypeError, TypeError)
    assert isinstance(ParamsTypeError("x"), TypeError)


def test_guard_is_noop_with_empty_migrated_set():
    """With the real (empty) ``_MIGRATED_FAMILIES`` the guard returns the
    dict unchanged — the Phase 0 no-op contract."""
    # Guard against accidental Phase-3 leakage into this repo state.
    assert _guard._MIGRATED_FAMILIES == set()

    params = {"mu": 0.0, "sigma": 1.0}
    returned = guard_params("normal", params)
    assert returned is params  # identity: unchanged, not merely equal


def test_guard_rejects_dict_when_family_migrated(monkeypatch):
    """When a family is migrated, a raw dict raises ``ParamsTypeError``
    whose message names the family and the ``from_dict`` recipe."""
    monkeypatch.setattr(_guard, "_MIGRATED_FAMILIES", {"normal"})

    with pytest.raises(ParamsTypeError) as excinfo:
        guard_params("normal", {"mu": 0.0, "sigma": 1.0})

    message = str(excinfo.value)
    assert "normal" in message
    assert "from_dict" in message


def test_guard_passes_non_dict_for_migrated_family(monkeypatch):
    """A migrated family still accepts a non-dict (e.g. a typed params
    object stand-in) — the guard only rejects raw ``dict`` instances."""
    monkeypatch.setattr(_guard, "_MIGRATED_FAMILIES", {"normal"})

    sentinel = object()
    assert guard_params("normal", sentinel) is sentinel


def test_guard_passes_dict_for_unmigrated_family(monkeypatch):
    """Even with a non-empty migrated set, a family NOT in the set keeps
    accepting dicts (only migrated families are gated)."""
    monkeypatch.setattr(_guard, "_MIGRATED_FAMILIES", {"normal"})

    params = {"df": 4.0}
    assert guard_params("student_t", params) is params
