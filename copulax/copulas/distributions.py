r"""Collections of copula distributions.

Copula distributions are auto-discovered by scanning
``copulax._src.copulas`` for module-level ``CopulaBase`` instances.
Both instantiated objects (e.g. ``gaussian_copula``) and their
uninstantiated classes (e.g. ``GaussianCopula``) are exported.
"""

import importlib
import pkgutil

import copulax._src.copulas as _cop_pkg
from copulax._src.copulas._distributions import CopulaBase


_TARGET_MODULES = frozenset({"_distributions", "_archimedean"})

_registry: list = []
_registry_attr_names: list = []
_seen_ids: set = set()

for _importer, _modname, _ispkg in sorted(pkgutil.iter_modules(_cop_pkg.__path__)):
    if _modname not in _TARGET_MODULES:
        continue
    _mod = importlib.import_module(f"copulax._src.copulas.{_modname}")
    for _attr in dir(_mod):
        _obj = getattr(_mod, _attr)
        if (
            isinstance(_obj, CopulaBase)
            and not _attr.startswith("_")
            and id(_obj) not in _seen_ids
            and getattr(_obj, "__module__", None) == _mod.__name__
        ):
            globals()[_attr] = _obj
            _registry.append(_obj)
            _registry_attr_names.append(_attr)
            _seen_ids.add(id(_obj))
            _cls = type(_obj)
            _cls_name = _cls.__name__
            if _cls_name not in globals():
                globals()[_cls_name] = _cls

_registry = tuple(sorted(_registry, key=lambda d: d.name.lower()))

_all_dists: tuple = tuple(dist.name for dist in _registry)
_all_dist_objects: list = [dist.name.replace("-", "_").lower() for dist in _registry]
_all_dist_classes: list = []
for _dist in _registry:
    _cls_name = type(_dist).__name__
    if _cls_name not in _all_dist_classes:
        _all_dist_classes.append(_cls_name)

distributions: dict = {dist.name: dist for dist in _registry}

__all__ = ["distributions"]
__all__ += _all_dist_objects
__all__ += _all_dist_classes

