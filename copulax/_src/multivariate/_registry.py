r"""Single source of truth for the multivariate distribution registry.

Imported by :mod:`copulax.multivariate` (the public package) and by
:mod:`copulax._src._serialization`.  Living under ``_src`` keeps it
sibling-importable from other private modules without triggering the
public package's ``__init__``.

To add a new distribution, define a ``Multivariate`` subclass in
``copulax/_src/multivariate/`` exporting a singleton instance, then
add it to the imports below and to ``_registry``.
"""

from copulax._src.multivariate.mvt_gh import MvtGH, mvt_gh
from copulax._src.multivariate.mvt_normal import MvtNormal, mvt_normal
from copulax._src.multivariate.mvt_skewed_t import MvtSkewedT, mvt_skewed_t
from copulax._src.multivariate.mvt_student_t import MvtStudentT, mvt_student_t

_registry: tuple = tuple(
    sorted(
        (mvt_gh, mvt_normal, mvt_skewed_t, mvt_student_t),
        key=lambda d: d.name.lower(),
    )
)

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
