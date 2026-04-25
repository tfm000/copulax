r"""Collections of copula distributions.

To add a new copula, define a ``CopulaBase`` subclass in
``copulax/_src/copulas/`` exporting a singleton instance, then add it
to the imports below and to ``_registry``.
"""

from copulax._src.copulas._mv_copulas import (
    GHCopula,
    GaussianCopula,
    SkewedTCopula,
    StudentTCopula,
    gaussian_copula,
    gh_copula,
    skewed_t_copula,
    student_t_copula,
)
from copulax._src.copulas._archimedean import (
    AMHCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    IndependenceCopula,
    JoeCopula,
    amh_copula,
    clayton_copula,
    frank_copula,
    gumbel_copula,
    independence_copula,
    joe_copula,
)

_registry: tuple = tuple(
    sorted(
        (
            gaussian_copula,
            gh_copula,
            skewed_t_copula,
            student_t_copula,
            amh_copula,
            clayton_copula,
            frank_copula,
            gumbel_copula,
            independence_copula,
            joe_copula,
        ),
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
