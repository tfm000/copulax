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
import jax.numpy as jnp
import pytest

from copulax._src._params import ParamsBase, ParamsTypeError, _guard, guard_params
from copulax._src.timeseries._base import TimeSeriesModel
from copulax.timeseries import ARMA, ArmaGarch
from copulax.univariate import normal


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


# ---------------------------------------------------------------------------
# WR-01 / WR-02 — chokepoint integration + stable-family-key re-keying
# ---------------------------------------------------------------------------
#
# The two guard chokepoints (``Distribution._resolve_params`` and
# ``TimeSeriesModel._guard_residual_params``) must key on a STABLE family
# identifier — ``type(self).__name__`` — rather than the mutable display
# ``name``.  The display name is user-settable and is auto-generated to a
# per-instance value (``FittedNormal-<id>``, ``FittedARMA(1,1)-...``) for
# fitted instances, so keying on it lets a fitted / renamed instance
# bypass the Phase 3 migration guard entirely (WR-01, empirically proven
# in 00-REVIEW.md).  These integration tests exercise the wired
# chokepoints end-to-end (WR-02: previously zero integration coverage,
# including the fitted-instance residual-params path), assert
# rename-immunity, and confirm the Phase 0 no-op-while-empty contract is
# preserved at both sites.
#
# The stable key for both chokepoints is ``type(self).__name__`` (e.g.
# ``"Normal"``, ``"ARMA"``, ``"ArmaGarch"``), which is invariant across
# the singleton, its auto-named fitted instances, and any user rename.


class TestDistributionChokepointGuard:
    """WR-01/02: ``Distribution._resolve_params`` guard chokepoint."""

    def test_noop_passthrough_while_migrated_set_empty(self):
        """Phase 0 contract: with the real (empty) ``_MIGRATED_FAMILIES``
        a raw dict flows through ``_resolve_params`` unchanged."""
        assert _guard._MIGRATED_FAMILIES == set()
        params = {"mu": jnp.array(0.0), "sigma": jnp.array(1.0)}
        # ``_resolve_params`` returns the guarded params object; with an
        # empty migrated set it is the identical dict (pure pass-through).
        assert normal._resolve_params(params) is params

    def test_migrated_family_rejects_dict_through_chokepoint(self, monkeypatch):
        """A raw dict passed through ``_resolve_params`` for a migrated
        family raises ``ParamsTypeError`` — the guard fires end-to-end,
        not merely in isolation."""
        monkeypatch.setattr(_guard, "_MIGRATED_FAMILIES", {"Normal"})
        with pytest.raises(ParamsTypeError):
            normal.logpdf(0.0, {"mu": 0.0, "sigma": 1.0})

    def test_stable_key_survives_instance_rename(self, monkeypatch):
        """WR-01 core: a *renamed* fitted instance is STILL guarded.

        The guard keys on ``type(self).__name__`` (``"Normal"``), not the
        display name, so renaming the instance to something absent from
        ``_MIGRATED_FAMILIES`` does NOT bypass the guard.  Against the
        old display-name keying this raw dict would be silently accepted.
        """
        monkeypatch.setattr(_guard, "_MIGRATED_FAMILIES", {"Normal"})
        renamed = normal._fitted_instance(
            {"mu": jnp.array(0.0), "sigma": jnp.array(1.0)},
            name="TotallyDifferentName",
        )
        # Sanity: the display name really is not a migrated key.
        assert renamed._name not in _guard._MIGRATED_FAMILIES
        with pytest.raises(ParamsTypeError):
            renamed.logpdf(0.0, {"mu": 0.0, "sigma": 1.0})

    def test_auto_named_fitted_instance_is_guarded(self, monkeypatch):
        """An auto-named fitted instance (``FittedNormal-<id>``) is
        guarded too — its per-instance display name would never match a
        ``_MIGRATED_FAMILIES`` entry under display-name keying."""
        monkeypatch.setattr(_guard, "_MIGRATED_FAMILIES", {"Normal"})
        fitted = normal._fitted_instance(
            {"mu": jnp.array(0.0), "sigma": jnp.array(1.0)}
        )
        assert fitted._name.startswith("FittedNormal-")
        with pytest.raises(ParamsTypeError):
            fitted.logpdf(0.0, {"mu": 0.0, "sigma": 1.0})

    def test_unmigrated_distribution_still_accepts_dict(self, monkeypatch):
        """With only ``"Normal"`` migrated, an unrelated family
        (``student_t``) keeps accepting raw dicts."""
        from copulax.univariate import student_t

        monkeypatch.setattr(_guard, "_MIGRATED_FAMILIES", {"Normal"})
        out = student_t.logpdf(0.0, {"nu": 5.0, "mu": 0.0, "sigma": 1.0})
        assert jnp.isfinite(out)


class TestTimeSeriesChokepointGuard:
    """WR-02: the fitted-instance residual-params guard chokepoint.

    This is the path with zero prior integration coverage.  The residual
    law's parameters enter each time-series family via the
    ``residual_params`` constructor kwarg, routed through
    ``TimeSeriesModel._guard_residual_params``.  It must key on the stable
    family class name (``"ARMA"``, ``"ArmaGarch"``), not the display name.
    """

    def test_noop_passthrough_while_migrated_set_empty(self):
        """Phase 0 contract: an empty migrated set is a straight
        pass-through — the residual dict is stored unchanged."""
        assert _guard._MIGRATED_FAMILIES == set()
        residual = {"mu": 0.0, "sigma": 1.0}
        model = ARMA(p=1, q=1, residual_params=residual)
        # Pass-through: stored dict equals the supplied residual dict.
        assert dict(model.residual_params) == residual

    def test_migrated_family_rejects_residual_dict(self, monkeypatch):
        """A raw residual dict for a migrated time-series family raises
        ``ParamsTypeError`` through the constructor chokepoint."""
        monkeypatch.setattr(_guard, "_MIGRATED_FAMILIES", {"ARMA"})
        with pytest.raises(ParamsTypeError):
            ARMA(p=1, q=1, residual_params={"mu": 0.0, "sigma": 1.0})

    def test_stable_key_survives_instance_rename(self, monkeypatch):
        """WR-01 core (time-series): a renamed instance is still guarded.

        The guard keys on ``type(self).__name__`` (``"ARMA"``), so a
        user-supplied ``name=`` that is absent from ``_MIGRATED_FAMILIES``
        does NOT bypass the residual-params guard.  Under the old
        display-name keying, passing ``name="MyModel"`` would silently
        accept the raw dict.
        """
        monkeypatch.setattr(_guard, "_MIGRATED_FAMILIES", {"ARMA"})
        with pytest.raises(ParamsTypeError):
            ARMA(
                p=1,
                q=1,
                name="MyRenamedModel",
                residual_params={"mu": 0.0, "sigma": 1.0},
            )

    def test_armagarch_migrated_family_rejects_residual_dict(self, monkeypatch):
        """The joint ``ArmaGarch`` composite is guarded on its own stable
        class-name key, independent of the mean/variance sub-families."""
        monkeypatch.setattr(_guard, "_MIGRATED_FAMILIES", {"ArmaGarch"})
        with pytest.raises(ParamsTypeError):
            ArmaGarch(
                mean_order=(1, 1),
                var_order=(1, 1),
                name="RenamedJoint",
                residual_params={"mu": 0.0, "sigma": 1.0},
            )

    def test_unmigrated_timeseries_family_still_accepts_dict(self, monkeypatch):
        """With only ``"ArmaGarch"`` migrated, the ``ARMA`` family keeps
        accepting raw residual dicts (per-family gating)."""
        monkeypatch.setattr(_guard, "_MIGRATED_FAMILIES", {"ArmaGarch"})
        model = ARMA(p=1, q=1, residual_params={"mu": 0.0, "sigma": 1.0})
        assert dict(model.residual_params) == {"mu": 0.0, "sigma": 1.0}

    def test_guard_helper_keys_on_class_name_not_display_name(self, monkeypatch):
        """Direct check of the shared helper's key semantics: the guard
        must be reachable via the stable class-name key.  The helper is a
        ``staticmethod`` taking an explicit key, so this documents the
        contract the constructor call sites must satisfy (pass
        ``type(self).__name__``, never the display name)."""
        monkeypatch.setattr(_guard, "_MIGRATED_FAMILIES", {"ARMA"})
        with pytest.raises(ParamsTypeError):
            TimeSeriesModel._guard_residual_params("ARMA", {"mu": 0.0})
        # A display-name-style key that is NOT migrated must pass through.
        passed = {"mu": 0.0}
        assert (
            TimeSeriesModel._guard_residual_params("MyRenamedModel", passed) is passed
        )
