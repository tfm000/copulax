r"""Contains collections of univariate distributions.

Currently the following distribution collections are implemented:
- distributions: contains all implemented distributions, grouped by data type.
- common: contains the most common distributions, grouped by data type.
- continuous: contains all continuous distributions.
- discrete: contains all discrete distributions.

New distributions are auto-discovered: just create a module in
``copulax/_src/univariate/`` that exports a ``Univariate`` instance
and it will appear here automatically.  Only the ``_COMMON_NAMES``
set below needs manual curation.
"""

import importlib
import pkgutil

import copulax._src.univariate as _uvt_pkg
from copulax._src._distributions import Univariate

# ── Auto-discover Univariate instances ─────────────────────────────────────
# Scans every public module in copulax._src.univariate (skipping private
# modules and univariate_fitter) for module-level Univariate instances.
_SKIP_MODULES = frozenset({"univariate_fitter"})

_registry: list = []
_registry_attr_names: list = []
_seen_ids: set = set()

for _importer, _modname, _ispkg in sorted(pkgutil.iter_modules(_uvt_pkg.__path__)):
    if _modname.startswith("_") or _modname in _SKIP_MODULES:
        continue
    _mod = importlib.import_module(f"copulax._src.univariate.{_modname}")
    for _attr in dir(_mod):
        _obj = getattr(_mod, _attr)
        if (
            isinstance(_obj, Univariate)
            and not _attr.startswith("_")
            and id(_obj) not in _seen_ids
            and getattr(_obj, "__module__", None) == _mod.__name__
        ):
            globals()[_attr] = _obj
            _registry.append(_obj)
            _registry_attr_names.append(_attr)
            _seen_ids.add(id(_obj))
            # Also export the unparameterised class (e.g. Uniform)
            _cls = type(_obj)
            _cls_name = _cls.__name__
            if _cls_name not in globals():
                globals()[_cls_name] = _cls

_registry = tuple(sorted(_registry, key=lambda d: d.name.lower()))

# ── Curated "common" collection ───────────────────────────────────────────
# Only edit this set when adding / removing from the common collection.
_COMMON_NAMES = frozenset(
    {
        "Gamma",
        "Gen-Normal",
        "Normal",
        "Student-T",
        "Uniform",
        "LogNormal",
    }
)

_all_dists: list = []
_dist_tree = {"continuous": {}, "discrete": {}}
_dist_tree["common"] = _dist_tree.copy()
_all_dist_objects: list = []
_all_dist_classes: list = []
for dist in _registry:
    _all_dists.append(dist.name)
    _all_dist_objects.append(dist.name.replace("-", "_").lower())
    _cls_name = type(dist).__name__
    if _cls_name not in _all_dist_classes:
        _all_dist_classes.append(_cls_name)

    _dist_tree[dist.name] = dist

    if dist.name in _COMMON_NAMES:
        _dist_tree["common"][dist.dtype][dist.name] = dist

    _dist_tree[dist.dtype][dist.name] = dist

_all_dists: tuple = tuple(_all_dists)
distributions = _dist_tree.copy()
common: dict = distributions.pop("common")
continuous: dict = distributions["continuous"].copy()
discrete: dict = distributions["discrete"].copy()

__all__ = ["distributions", "common", "continuous", "discrete"]
__all__ += _all_dist_objects
__all__ += _all_dist_classes
