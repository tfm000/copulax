r"""Contains collections of univariate distributions.

Currently the following distribution collections are implemented:
- distributions: contains all implemented distributions, grouped by data type.
- common: contains the most common distributions, grouped by data type.
- continuous: contains all continuous distributions.
- discrete: contains all discrete distributions.
"""
# importing pytree distribution objects
from copulax._src.univariate.gamma import gamma
from copulax._src.univariate.gh import gh
from copulax._src.univariate.gig import gig
from copulax._src.univariate.ig import ig
from copulax._src.univariate.lognormal import lognormal
from copulax._src.univariate.normal import normal
from copulax._src.univariate.skewed_t import skewed_t
from copulax._src.univariate.student_t import student_t
from copulax._src.univariate.uniform import uniform

# importing categorizing map
from copulax._src._distributions import DistMap

__dists = (gamma, gh, gig, ig, lognormal, normal, skewed_t, student_t, uniform)
__common_names = ('Gamma', 'Normal', 'Student-T', 'Uniform', 'LogNormal',)

_all_dists: list = []
_dist_tree = {'continuous': {}, 'discrete': {}}
_dist_tree['common'] = _dist_tree.copy()
_all_dist_objects: list = []
for dist in __dists:
    _all_dists.append(dist.name)
    _all_dist_objects.append(DistMap.name_map[dist.name]['object_name'])

    _dist_tree[dist.name] = dist

    if dist.name in __common_names:
        _dist_tree['common'][dist.dtype][dist.name] = dist
    
    _dist_tree[dist.dtype][dist.name] = dist

_all_dists: tuple = tuple(_all_dists)
distributions = _dist_tree.copy()
common: dict = distributions.pop('common')
continuous: dict = distributions['continuous'].copy()
discrete: dict = distributions['discrete'].copy()

__all__ = ['distributions', 'common', 'continuous', 'discrete']
__all__ += _all_dist_objects
