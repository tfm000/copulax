"""Regression test for docstring visibility on subclass overrides.

Python's ``inspect.getdoc()`` does not walk the MRO past an override
whose ``__doc__`` is ``None`` — meaning ``help()``, IPython ``?``, and
IDE hover all show no documentation on subclass overrides that omit a
docstring, even though the parent declares the contract in detail.

The ``Distribution.__init_subclass__`` hook (in
``copulax/_src/_distributions.py``) copies parent docstrings onto
overrides at class-creation time to surface them through
``inspect.getdoc()``.  This test pins that behaviour: every public
method on every public distribution / copula / preprocessing object
must return a non-empty ``inspect.getdoc()``.
"""

import inspect
import re

import pytest

import copulax
from copulax import univariate, multivariate, copulas, preprocessing, special, stats
from copulax import timeseries as _timeseries


# Concrete time-series model classes whose class docstrings must each
# carry a NumPy-style ``References`` section naming the primary
# literature source(s) for their recursion / likelihood (HARD-01 D-01).
# The section is enforced on each class's OWN ``__doc__`` (what
# ``help()`` / IPython ``?`` / IDE hover surface on the concrete class)
# rather than the inherited base docstring, so every user-facing model
# is individually traceable to its source.
_TIMESERIES_MODEL_CLASS_NAMES = (
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


def _public_objects():
    """Yield (label, object) pairs for every public attribute on every
    family submodule."""
    for mod_name in (
        "univariate",
        "multivariate",
        "copulas",
        "preprocessing",
        "special",
        "stats",
    ):
        mod = getattr(copulax, mod_name)
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            attr = getattr(mod, attr_name)
            yield f"{mod_name}.{attr_name}", attr


_JAX_INTERNAL_MODULE_PREFIXES = ("jax.", "jaxlib.", "jaxlib_")


def _is_jax_internal(obj) -> bool:
    """True if ``obj`` is a JAX-provided wrapper (``PjitFunction``,
    ``custom_vjp``, ``custom_jvp``, etc.). Such wrappers expose JAX
    protocol attributes (``clear_cache``, ``fwd``, ``bwd``,
    ``eval_shape``, ``lower``) that are not part of CopulAX's surface
    and shouldn't be audited as if they were."""
    mod = type(obj).__module__
    return any(mod.startswith(p) for p in _JAX_INTERNAL_MODULE_PREFIXES)


def _public_methods(obj):
    """Yield (name, method) pairs for every public callable attribute on
    *obj*, skipping JAX-provided wrappers whose attributes are protocol
    internals rather than CopulAX surface."""
    if _is_jax_internal(obj):
        return
    for name in dir(obj):
        if name.startswith("_"):
            continue
        method = getattr(obj, name, None)
        if method is None or not callable(method):
            continue
        if _is_jax_internal(method):
            continue
        yield name, method


def _collect_invisible():
    """Return [(label, method_name)] for every public method whose
    ``inspect.getdoc()`` is empty."""
    invisible = []
    seen = set()
    for label, obj in _public_objects():
        for name, method in _public_methods(obj):
            mid = id(method)
            if mid in seen:
                continue
            seen.add(mid)
            if not inspect.getdoc(method):
                invisible.append((label, name))
    return invisible


def test_all_public_methods_have_visible_docstrings():
    """No public method on any public distribution / preprocessing /
    stats / special object may have an empty ``inspect.getdoc()``.

    This guards against the regression where a subclass adds an
    override without a docstring and breaks ``help()`` / IPython ``?`` /
    IDE hover for users.
    """
    invisible = _collect_invisible()
    assert not invisible, (
        "Found public methods with no visible docstring (inspect.getdoc "
        "returns None or empty). Either add a docstring to the override "
        "or remove the override if it is a pure super() passthrough.\n"
        "Offenders:\n  " + "\n  ".join(f"{label}.{name}" for label, name in invisible)
    )


@pytest.mark.parametrize(
    "obj_name,method_name",
    [
        # Previously-invisible univariate overrides
        ("gen_normal", "cdf"),
        ("gen_normal", "rvs"),
        ("gen_normal", "stats"),
        ("nig", "stats"),
        ("student_t", "example_params"),
        ("gamma", "example_params"),
        # Inherited methods (must remain visible regardless of patch)
        ("student_t", "fit"),
        ("student_t", "pdf"),
        ("student_t", "logpdf"),
        ("gamma", "logpdf"),  # was a deleted passthrough; now inherited cleanly
        ("gamma", "logcdf"),
        # Previously-invisible Archimedean overrides
        ("clayton_copula", "generator"),
        ("clayton_copula", "generator_inv"),
        ("frank_copula", "generator"),
        ("amh_copula", "generator"),
        ("independence_copula", "generator"),
    ],
)
def test_specific_methods_have_visible_docstrings(obj_name, method_name):
    """Spot-checks for individual high-traffic methods (overrides that
    were previously invisible, and inherited methods that should remain
    visible)."""
    # Resolve the object from the appropriate family submodule
    for mod_name in ("univariate", "copulas"):
        mod = getattr(copulax, mod_name)
        obj = getattr(mod, obj_name, None)
        if obj is not None:
            break
    assert obj is not None, f"Could not find {obj_name} in any family"

    method = getattr(obj, method_name)
    doc = inspect.getdoc(method)
    assert doc, f"{obj_name}.{method_name}: inspect.getdoc returned {doc!r}"


# ---------------------------------------------------------------------------
# Time-series model References sections (HARD-01 D-01)
# ---------------------------------------------------------------------------
#
# Every mean / variance / joint time-series model class must document its
# primary literature source(s) via a NumPy-style ``References`` section in
# its own class docstring, so each recursion / likelihood is traceable to
# the source it was reviewed against (01-MATH-REVIEW.md).  These tests pin
# that: a class missing the section (or the citation line under it) fails.

# A citation line under the References section: a NumPy-style underline
# followed by at least one non-empty content line (the reference itself).
_REFERENCES_HEADER_RE = re.compile(
    r"^\s*References\s*\n\s*-{3,}\s*\n\s*\S", re.MULTILINE
)


def _timeseries_model_classes():
    """Yield ``(name, cls)`` for every registered concrete time-series
    model class, resolved from the public ``copulax.timeseries`` surface."""
    for name in _TIMESERIES_MODEL_CLASS_NAMES:
        cls = getattr(_timeseries, name, None)
        assert cls is not None and inspect.isclass(cls), (
            f"copulax.timeseries.{name} is not a class (got {cls!r}); the "
            "time-series model registry in this test is stale."
        )
        yield name, cls


@pytest.mark.parametrize(
    "model_name", _TIMESERIES_MODEL_CLASS_NAMES,
)
def test_timeseries_model_class_has_references_section(model_name):
    """Each time-series model class docstring must contain a non-empty
    NumPy-style ``References`` section (header + at least one citation
    line) on its OWN ``__doc__``.

    This enforces HARD-01 D-01: every recursion / likelihood is traceable
    to the primary source it was reviewed against in 01-MATH-REVIEW.md.
    The check fails if a registered model class omits the section — the
    audit trail is only as good as the citation it carries (CLAUDE.md
    rule 5).
    """
    cls = getattr(_timeseries, model_name)
    doc = cls.__doc__ or ""
    assert "References" in doc, (
        f"{model_name} class docstring has no 'References' section header. "
        "Add a NumPy-style References section naming the model's primary "
        "literature source(s) — see 01-MATH-REVIEW.md for the citation."
    )
    assert _REFERENCES_HEADER_RE.search(doc), (
        f"{model_name} class docstring has a 'References' heading but no "
        "citation line beneath it (expected 'References' followed by a "
        "'----' underline and at least one non-empty reference line)."
    )


def test_all_timeseries_models_covered_by_references_check():
    """Guard against a model class being added to
    ``copulax.timeseries`` without a corresponding References assertion —
    the registry in this test must stay complete.
    """
    registered = {name for name, _ in _timeseries_model_classes()}
    # Every concrete GARCH / ARMA / joint model exposed publicly must be
    # in the parametrised set above.  Discover the concrete model classes
    # on the public surface and require each is covered.
    from copulax._src.timeseries._base import TimeSeriesModel

    discovered = set()
    for attr_name in dir(_timeseries):
        if attr_name.startswith("_"):
            continue
        attr = getattr(_timeseries, attr_name)
        if inspect.isclass(attr) and issubclass(attr, TimeSeriesModel):
            # Skip abstract bases (ARMABase / GARCHBase are not exposed
            # publicly, but guard anyway): only concrete, instantiable
            # models carry a user-facing References contract.
            if inspect.isabstract(attr):
                continue
            discovered.add(attr_name)

    missing = discovered - registered
    assert not missing, (
        "Public time-series model classes missing from the References "
        f"coverage check: {sorted(missing)}. Add them to "
        "_TIMESERIES_MODEL_CLASS_NAMES."
    )
