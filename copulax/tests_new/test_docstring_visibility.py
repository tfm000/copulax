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

import pytest

import copulax
from copulax import univariate, multivariate, copulas, preprocessing, special, stats


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
