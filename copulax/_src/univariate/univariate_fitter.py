"""contains the copulAX implementation of a univariate fitter object."""
import jax.numpy as jnp
from jax import jit, lax
from jax._src.typing import ArrayLike
from typing import Iterable
import numpy as np
from functools import partial

from copulax.univariate.distributions import *
from copulax.univariate.distributions import _all_dists, _dist_tree
from copulax._src.typing import Scalar


def _get_dist_objects(dists: Iterable | str) -> tuple:
    if isinstance(dists, str):
        dists: str = dists.lower().strip()
        if dists not in ("all", "common", "continuous", "discrete", "common continuous", "common discrete"):
            raise ValueError(f"Invalid value for 'dists' argument: {dists}." \
                             "If a string, dists must be one of 'all', " \
                                "'common', 'continuous', 'discrete', " \
                                "'common continuous' or 'common disrete'.")
        
        elif dists == "all":
            dists_objs: tuple = (*_dist_tree["continuous"].values(), 
                                 *_dist_tree["discrete"].values())
        elif dists in  ("common continuous", "common discrete"):
            dists_objs: tuple = tuple(_dist_tree["common"][dists.split()[-1]].values())
        elif dists == "common":
            dists_objs: tuple = tuple(*_dist_tree["common"]["continuous"].values(), 
                                      *_dist_tree["common"]["discrete"].values())
        else:
            dists_objs: tuple = tuple(_dist_tree[dists].values())

    elif isinstance(dists, Iterable):
        dists_objs: tuple = tuple(dists)
        for dist in dists:
            if not dist.__name__.startswith("copulax.univariate."):
                raise ValueError(f"Invalid distribution object provided " \
                                 f"within 'dists' iterable: {dist}. " \
                                 f"Distribution objects must be copulAX " \
                                 "distributions.")
    else:
        raise ValueError(f"Invalid value for 'dists' argument: {dists}. " \
                         "Dists must be a string or an iterable of " \
                         "copulAX distribution objects.")
    
    return dists_objs


@partial(jit, static_argnames=('metric',))
def _fit_and_stats(dist, x, metric, **kwargs):
    dist_params = dist.fit(x, **kwargs)
    dist_metric = getattr(dist, metric)(x, **dist_params)
    return {'params': dist_params, 'metric': dist_metric, 'dist': dist, }


# @partial(jit, static_argnames=('metric', 'distributions'))
def univariate_fitter(x: jnp.ndarray, metric: str = "bic", distributions: Iterable | str = "common continuous", **kwargs) -> dict:
    r"""Find and fit the 'best' univariate distribution to the input data 
    according to a specified metric.

    Args:
        x (ArrayLike): The input data to fit a distribution to.
        metric (str): The metric to use when selecting the 'best' distribution. 
        Must be one of 'aic', 'bic' or 'loglikelihood'. Default is 'bic'.
        distributions (Iterable | str): The distribution(s) to fit to the data. If a 
        string, must be one of 'all', 'common', 'continuous', 'discrete', 
        'common continuous' or 'common discrete' corresponding to the 
        classifications within copulax.univariate.distributions.distributions. 
        If an iterable, must contain copulAX distribution objects. Default is 
        'common continuous'.
        kwargs: Additional keyword arguments to pass to each distribution 
        object's fit method.

    Note:
        If you intend to jit wrap this function, ensure that the 'dists' and 
        metrics arguments are static

    Returns:
        res (tuple): The index of the best distribution fit and a tuple of 
        fitted distributions and the metric values.

    Exampless
    --------
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> from copulax.univariate import univariate_fitter
    >>> x = np.random.normal(0, 1, 100)
    >>> univariate_fitter(x)
    """
    # getting distribution objects
    dists_objs = _get_dist_objects(distributions)

    # checking metric
    if metric not in ("aic", "bic", "loglikelihood"):
        raise ValueError(f"Invalid value for 'metric' argument: {metric}. " \
                         "Must be one of 'aic', 'bic' or 'loglikelihood'.")

    # fitting distributions
    output: list = []
    best_index, best_metric = 0, jnp.inf
    for i, dist in enumerate(dists_objs):
        dist_output: dict = _fit_and_stats(dist=dist, x=x, metric=metric, **kwargs)
        dist_metric: Scalar = dist_output["metric"]
        cond: bool = dist_metric < best_metric
        best_index: int = jnp.where(cond, i, best_index)
        best_metric: Scalar = jnp.where(cond, dist_metric, best_metric)
        
        output.append(dist_output)

    return best_index, tuple(output)
 