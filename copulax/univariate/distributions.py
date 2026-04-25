r"""Contains collections of univariate distributions.

Currently the following distribution collections are implemented:
- distributions: contains all implemented distributions, grouped by data type.
- common: contains the most common distributions, grouped by data type.
- continuous: contains all continuous distributions.
- discrete: contains all discrete distributions.

To add a new distribution, define a ``Univariate`` subclass in
``copulax/_src/univariate/`` exporting a singleton instance, then add
it to the imports below and to ``_registry``.  ``_COMMON_NAMES`` needs
manual curation when the "common" tier changes.
"""

from copulax._src.univariate.asym_gen_normal import AsymGenNormal, asym_gen_normal
from copulax._src.univariate.gamma import Gamma, gamma
from copulax._src.univariate.gen_normal import GenNormal, gen_normal
from copulax._src.univariate.gh import GH, gh
from copulax._src.univariate.gig import GIG, gig
from copulax._src.univariate.ig import IG, ig
from copulax._src.univariate.lognormal import LogNormal, lognormal
from copulax._src.univariate.nig import NIG, nig
from copulax._src.univariate.normal import Normal, normal
from copulax._src.univariate.skewed_t import SkewedT, skewed_t
from copulax._src.univariate.student_t import StudentT, student_t
from copulax._src.univariate.uniform import Uniform, uniform
from copulax._src.univariate.wald import Wald, wald

_registry: tuple = tuple(
    sorted(
        (
            asym_gen_normal,
            gamma,
            gen_normal,
            gh,
            gig,
            ig,
            lognormal,
            nig,
            normal,
            skewed_t,
            student_t,
            uniform,
            wald,
        ),
        key=lambda d: d.name.lower(),
    )
)

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
_dist_tree["common"] = {"continuous": {}, "discrete": {}}
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
