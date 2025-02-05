r"""Contains collections of univariate distributions.

Currently the following distribution collections are implemented:
- distributions: contains all implemented distributions, grouped by data type.
- common: contains the most common distributions, grouped by data type.
- continuous: contains all continuous distributions.
- discrete: contains all discrete distributions.
"""
from copulax.univariate import gamma, gh, gig, ig, lognormal, normal, skewed_t, student_t, uniform

__dists = (gamma, gh, gig, ig, lognormal, normal, skewed_t, student_t, uniform)
__common_names = ('gamma', 'normal', 'student_t', 'uniform', 'lognormal',)

_all_dists: list = []
_dist_tree = {'continuous': {}, 'discrete': {}}
_dist_tree['common'] = _dist_tree.copy()
for dist in __dists:
    _all_dists.append(dist.name)

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
__all__ += _all_dists
