"""Import and JIT-compilation smoke tests for the public copulAX surface.

This module is the seconds-scale tripwire that runs on **every** CI leg,
including the light ones (``-m "not slow and not heavy"``).  Its job is to
turn the two failure modes that make every other test meaningless — a
public subpackage that no longer imports, and a JAX program that no longer
lowers or compiles — into an immediate, unambiguous failure rather than a
wall of downstream errors from whichever numerical module happened to be
collected first.

Relationship to ``scripts/release_smoke_test.py``
-------------------------------------------------
The release script exercises one representative *numerical* call per
subpackage against a freshly installed wheel, and runs only on the four
``package-install`` CI legs.  This module deliberately does **not**
duplicate it: it asserts structure (the names resolve, a program compiles)
rather than results, so it costs no optimiser budget and can therefore be
selected on every light leg, where the release script is never run.
``copulax.timeseries`` is additionally covered here but absent from the
release script's import surface.

Frozen-corpus integrity
-----------------------
``test_frozen_corpus_integrity`` is a structural tripwire of the same shape
as the checks above — light-leg, sub-second and fit-free — so it belongs
beside them rather than in a module of its own.  Its subject is
``copulax/tests/_r_reference/frozen_series_data.py``, which records a
SHA-256 for every series it commits: the regenerator verifies every digest
at write time, but a tripwire that never runs protects nothing.  A generated
module of that size is a classic merge-conflict magnet, and a mangled
conflict resolution or a hand edit (despite "DO NOT EDIT") would silently
become the new truth for every consumer of the corpus, surfacing only as
confusing downstream failures.  Arming the check here runs it wherever the
suite runs, so byte-level corruption of the committed data fails loudly, by
series name, before a single fit reads it.  The ``_r_reference``
regenerators are omitted from coverage measurement by design — they shell
out to R — which makes this test the control standing in for their coverage.

Marker policy
-------------
Nothing in this module carries ``pytest.mark.slow`` or ``pytest.mark.heavy``,
and nothing here may acquire one: a light leg that deselects its own import
smoke has no tripwire at all.
"""

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import copulax
from copulax import (
    copulas,
    multivariate,
    preprocessing,
    special,
    stats,
    timeseries,
    univariate,
)
from copulax.tests._r_reference.frozen_series_data import FROZEN_SERIES

# The seven public subpackages, each paired with one attribute that must
# resolve on it.  The probe attribute is a load-bearing member of that
# subpackage's public API — a re-export break that leaves the module
# importable but empties its namespace is exactly what this catches, and an
# `import` statement alone would not.
_PUBLIC_SUBPACKAGES = (
    pytest.param("univariate", univariate, "normal", id="univariate"),
    pytest.param("multivariate", multivariate, "mvt_normal", id="multivariate"),
    pytest.param("copulas", copulas, "gaussian_copula", id="copulas"),
    pytest.param("preprocessing", preprocessing, "DataScaler", id="preprocessing"),
    pytest.param("special", special, "kv", id="special"),
    pytest.param("stats", stats, "skew", id="stats"),
    pytest.param("timeseries", timeseries, "GARCH", id="timeseries"),
)

# Names re-exported directly from the ``copulax`` package object itself,
# rather than from one of its subpackages.
_TOP_LEVEL_NAMES = ("load", "get_random_key")

# The concrete time-series model classes ``copulax.timeseries`` promises in
# its ``__all__``.  Kept as an explicit tuple rather than derived from
# ``__all__`` so that a class silently dropped from both the imports and the
# ``__all__`` list still fails this test.
_TIMESERIES_MODEL_CLASSES = (
    "AR",
    "MA",
    "ARMA",
    "GARCH",
    "IGARCH",
    "GJR_GARCH",
    "EGARCH",
    "TGARCH",
    "QGARCH",
    "GARCH_M",
    "ArmaGarch",
)


@pytest.mark.parametrize(("name", "module", "attribute"), _PUBLIC_SUBPACKAGES)
def test_public_subpackages_resolve(name, module, attribute):
    """Every public subpackage imports, binds onto ``copulax`` and is populated.

    Parameters
    ----------
    name : str
        Attribute name of the subpackage on the ``copulax`` package object.
    module : module
        The subpackage as imported at module level in this file.
    attribute : str
        A public member that must resolve on ``module``.
    """
    assert module.__name__ == f"copulax.{name}", (
        f"expected copulax.{name}, imported {module.__name__}"
    )
    assert getattr(copulax, name, None) is module, (
        f"copulax.{name} is not bound to the imported subpackage"
    )
    assert hasattr(module, attribute), (
        f"copulax.{name}.{attribute} did not resolve — the subpackage imported "
        f"but its public surface is missing"
    )


def test_top_level_namespace_resolves():
    """``copulax`` itself re-exports its callable top-level API."""
    for name in _TOP_LEVEL_NAMES:
        member = getattr(copulax, name, None)
        assert member is not None, f"copulax.{name} did not resolve"
        assert callable(member), f"copulax.{name} is not callable"


def test_version_attribute():
    """``copulax.__version__`` is present and is a non-empty string."""
    version = copulax.__version__
    assert isinstance(version, str), f"__version__ is {type(version).__name__}, not str"
    assert version.strip(), "__version__ is empty"


def test_trivial_jit_path_compiles():
    """A jitted density call lowers, compiles and returns finite values.

    This is the live probe of the JAX toolchain on each CI leg: it is the
    step the persistent compilation cache populates, so a lowering or
    compilation break surfaces here in seconds rather than inside the first
    model fit that happens to run.
    """
    x = jnp.array([-1.0, 0.0, 1.0])
    params = {"mu": jnp.array(0.0), "sigma": jnp.array(1.0)}

    jitted = jax.jit(univariate.normal.pdf)

    # Lower and compile explicitly instead of inferring compilation from a
    # successful call, so a failure names which of the two stages broke.
    lowered = jitted.lower(x, params)
    assert lowered.as_text(), "jit lowering produced no HLO"
    compiled = lowered.compile()

    density = compiled(x, params)
    assert density.shape == x.shape, (
        f"density shape {density.shape} does not match input {x.shape}"
    )
    assert bool(jnp.all(jnp.isfinite(density))), f"non-finite density: {density}"
    # Strict positivity is a structural property of a density evaluated on its
    # support, not a tolerance check: it rejects an all-zero result that would
    # otherwise satisfy the finiteness assertion above.
    assert bool(jnp.all(density > 0.0)), f"non-positive density: {density}"


def test_timeseries_surface():
    """``copulax.timeseries`` exports its full ``__all__`` and every model class."""
    unresolved = [name for name in timeseries.__all__ if not hasattr(timeseries, name)]
    assert not unresolved, (
        f"copulax.timeseries.__all__ names that do not resolve: {unresolved}"
    )

    for name in _TIMESERIES_MODEL_CLASSES:
        model = getattr(timeseries, name, None)
        assert model is not None, f"copulax.timeseries.{name} did not resolve"
        assert isinstance(model, type), (
            f"copulax.timeseries.{name} is {type(model).__name__}, not a class"
        )
        assert name in timeseries.__all__, (
            f"copulax.timeseries.{name} exists but is absent from __all__"
        )


def test_frozen_corpus_integrity():
    """Every committed series matches its own recorded provenance.

    For each ``FROZEN_SERIES`` entry: the array is 1-D float64
    (consumers downcast at the call site — a committed dtype change
    would alter every digest), the recorded ``n`` equals ``len(y)``,
    and the recorded SHA-256 equals the SHA-256 of the array's bytes —
    the exact check the corpus module's docstring documents, executed
    instead of quoted.

    Deliberately dependency-light: pure ``numpy`` + ``hashlib``, no jax,
    no copulax model code and no fits — the whole corpus verifies in
    well under a second, on whichever leg selects it.
    """
    assert FROZEN_SERIES, "frozen corpus is empty"
    for name, entry in FROZEN_SERIES.items():
        y = np.asarray(entry["y"])
        provenance = entry["provenance"]
        assert y.dtype == np.float64, (name, y.dtype)
        assert y.ndim == 1, (name, y.shape)
        assert len(y) == provenance["n"], (name, len(y), provenance["n"])
        digest = hashlib.sha256(y.tobytes()).hexdigest()
        assert digest == provenance["sha256"], name
