"""contains the copulAX implementation of a univariate fitter object."""

import jax.numpy as jnp
from jax import jit, lax, vmap
from typing import Iterable
from functools import partial

from copulax.univariate.distributions import *
from copulax.univariate.distributions import _dist_tree
from copulax._src.typing import Scalar
from copulax._src._distributions import Univariate
from copulax._src.univariate._gof import ks_test, cvm_test


_GOF_FUNCS = {"ks": ks_test, "cvm": cvm_test}

# ── Distribution registry ─────────────────────────────────────────────────
# Ordered tuple of every univariate distribution; used by the JIT core as
# the fixed set of ``lax.switch`` branches.
_DIST_REGISTRY: tuple = (
    gamma,
    gh,
    gig,
    ig,
    lognormal,
    normal,
    skewed_t,
    student_t,
    uniform,
)
_MAX_DISTS: int = len(_DIST_REGISTRY)
_MAX_PARAMS: int = max(len(d.example_params()) for d in _DIST_REGISTRY)
_DIST_NAME_TO_INDEX: dict = {d.name: i for i, d in enumerate(_DIST_REGISTRY)}


def _get_dist_objects(dists: Iterable | str) -> tuple:
    if isinstance(dists, str):
        dists: str = dists.lower().strip()
        if dists not in (
            "all",
            "common",
            "continuous",
            "discrete",
            "common continuous",
            "common discrete",
        ):
            raise ValueError(
                f"Invalid value for 'dists' argument: {dists}."
                "If a string, dists must be one of 'all', "
                "'common', 'continuous', 'discrete', "
                "'common continuous' or 'common disrete'."
            )

        elif dists == "all":
            dists_objs: tuple = (
                *_dist_tree["continuous"].values(),
                *_dist_tree["discrete"].values(),
            )
        elif dists in ("common continuous", "common discrete"):
            dists_objs: tuple = tuple(_dist_tree["common"][dists.split()[-1]].values())
        elif dists == "common":
            dists_objs: tuple = tuple(
                (
                    *_dist_tree["common"]["continuous"].values(),
                    *_dist_tree["common"]["discrete"].values(),
                )
            )
        else:
            dists_objs: tuple = tuple(_dist_tree[dists].values())

    elif isinstance(dists, Iterable):
        dists_objs: tuple = tuple(dists)
        for dist in dists:
            if not isinstance(dist, Univariate):
                raise ValueError(
                    f"Invalid distribution object provided "
                    f"within 'dists' iterable: {dist}. "
                    f"Distribution objects must be univariate "
                    "copulax distributions."
                )
    else:
        raise ValueError(
            f"Invalid value for 'dists' argument: {dists}. "
            "Dists must be a string or an iterable of "
            "copulAX distribution objects."
        )

    return dists_objs


def _dist_to_indices(dists_objs: tuple) -> jnp.ndarray:
    """Map distribution objects to their integer indices in _DIST_REGISTRY."""
    return jnp.array([_DIST_NAME_TO_INDEX[d.name] for d in dists_objs], dtype=jnp.int32)


# ── Branch factory ─────────────────────────────────────────────────────────
def _make_branches(metric: str, gof_test: str | None):
    """Build one branch function per registered distribution.

    Each branch has an identical return pytree so that ``lax.switch`` can
    dispatch across them.  The *metric* and *gof_test* strings are
    captured in closures (resolved at trace time).
    """
    gof_func = _GOF_FUNCS.get(gof_test)

    branches = []
    for dist in _DIST_REGISTRY:

        def _branch(x, _dist=dist):
            fitted = _dist.fit(x)
            params = fitted.params
            params_arr = _dist._padded_params_to_array(params, max_params=_MAX_PARAMS)
            metric_val = getattr(_dist, metric)(x=x, params=params)

            if gof_func is not None:
                gof_result = gof_func(x=x, dist=_dist, params=params)
                gof_stat = gof_result["statistic"]
                gof_pval = gof_result["p_value"]
            else:
                gof_stat = jnp.nan
                gof_pval = jnp.nan

            return params_arr, metric_val, gof_stat, gof_pval

        branches.append(_branch)

    return branches


# ── Core implementation ────────────────────────────────────────────────────
def _core_impl(
    x, dist_indices, active_mask, significance_level, branches, ascending, has_gof
):
    """Fit all distributions, score, filter and rank — fully on-device.

    Not JIT-decorated so it can be composed with ``vmap``.

    Args:
        x: Input data array, shape (n,).
        dist_indices: Integer indices into _DIST_REGISTRY, shape (MAX_DISTS,).
        active_mask: Boolean mask, shape (MAX_DISTS,).
        significance_level: GoF p-value threshold.
        branches: Tuple of branch callables (static).
        ascending: Whether lower metric is better (static).
        has_gof: Whether GoF filtering is active (static).

    Returns:
        Tuple of (sorted_order, params_arrs, metrics, gof_stats,
        gof_pvals, final_mask, n_pass).
    """

    def _fit_one(idx):
        return lax.switch(idx, branches, x)

    # lax.map applies _fit_one sequentially over the leading axis
    all_results = lax.map(_fit_one, dist_indices)
    params_arrs, metrics, gof_stats, gof_pvals = all_results

    # GoF filtering
    if has_gof:
        gof_passed = gof_pvals >= significance_level
    else:
        gof_passed = jnp.ones(_MAX_DISTS, dtype=jnp.bool_)

    final_mask = active_mask & gof_passed & jnp.isfinite(metrics)

    # Fill inactive / failed slots with sentinel for sorting
    sentinel = jnp.where(ascending, jnp.inf, -jnp.inf)
    scored = jnp.where(final_mask, metrics, sentinel)

    # Sort (best first)
    if ascending:
        order = jnp.argsort(scored)
    else:
        order = jnp.argsort(-scored)
    n_pass = jnp.sum(final_mask)

    return order, params_arrs, metrics, gof_stats, gof_pvals, final_mask, n_pass


# ── JIT core (single variable) ────────────────────────────────────────────
@partial(jit, static_argnames=("branches", "ascending", "has_gof"))
def _jit_core(
    x, dist_indices, active_mask, significance_level, branches, ascending, has_gof
):
    return _core_impl(
        x,
        dist_indices,
        active_mask,
        significance_level,
        branches,
        ascending,
        has_gof,
    )


# ── JIT core (batched across variables) ───────────────────────────────────
@partial(jit, static_argnames=("branches", "ascending", "has_gof"))
def _batched_jit_core(
    x_batch,
    dist_indices,
    active_mask,
    significance_level,
    branches,
    ascending,
    has_gof,
):
    """``vmap``-ed version of :func:`_core_impl` over the leading axis of
    *x_batch* (shape ``(d, n)``).  All other arguments are broadcast."""

    def _single(xi):
        return _core_impl(
            xi,
            dist_indices,
            active_mask,
            significance_level,
            branches,
            ascending,
            has_gof,
        )

    return vmap(_single)(x_batch)


# ── Public API ─────────────────────────────────────────────────────────────
def univariate_fitter(
    x: jnp.ndarray,
    metric: str = "bic",
    distributions: Iterable | str = "common continuous",
    gof_test: str | None = None,
    significance_level: float = 0.05,
) -> tuple:
    r"""Find and fit the 'best' univariate distribution to the input data
    according to a specified metric.

    The implementation is fully JIT-compiled: a single ``jax.jit``-traced
    function fits every registered distribution via ``lax.switch``,
    computes the chosen metric, optionally applies a goodness-of-fit
    filter, and sorts the results — all on-device in one XLA graph.
    After the first call the compiled graph is cached, so subsequent
    calls (e.g. once per marginal variable in a copula) execute at
    near-zero Python overhead.

    Args:
        x (ArrayLike): The input data to fit a distribution to.
        metric (str): The metric to use when selecting the 'best' distribution.
            Must be one of 'aic', 'bic' or 'loglikelihood'. Default is 'bic'.
        distributions (Iterable | str): The distribution(s) to fit to the data.
            If a string, must be one of 'all', 'common', 'continuous',
            'discrete', 'common continuous' or 'common discrete' corresponding
            to the classifications within
            copulax.univariate.distributions.distributions. If an iterable,
            must contain copulAX distribution objects. Default is
            'common continuous'.
        gof_test (str | None): Optional goodness-of-fit test to apply after
            fitting. One of 'ks' (Kolmogorov-Smirnov), 'cvm' (Cramér-von
            Mises), or None (no test). When set, distributions that fail
            the test at the given *significance_level* are removed from the
            results. Default is None.
        significance_level (float): Significance level for the goodness-of-fit
            test. Distributions with a p-value below this threshold are
            removed. Only used when *gof_test* is not None. Default is 0.05.

    Returns:
        res (tuple): The index of the best distribution fit (always 0)
        and a tuple of fitted distribution results sorted by the metric
        (best first). Each result is a dict with keys 'params', 'metric',
        'dist', and optionally 'gof'. Returns ``(None, ())`` if all
        distributions are filtered out by the goodness-of-fit test.

    Examples:
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> from copulax.univariate import univariate_fitter
    >>> x = np.random.normal(0, 1, 100)
    >>> univariate_fitter(x)
    >>> univariate_fitter(x, gof_test='ks', significance_level=0.05)
    """
    # ── Validation (Python-side, not traced) ──
    dists_objs = _get_dist_objects(distributions)

    if metric not in ("aic", "bic", "loglikelihood"):
        raise ValueError(
            f"Invalid value for 'metric' argument: {metric}. "
            "Must be one of 'aic', 'bic' or 'loglikelihood'."
        )

    if gof_test is not None and gof_test not in _GOF_FUNCS:
        raise ValueError(
            f"Invalid value for 'gof_test' argument: {gof_test}. "
            "Must be one of 'ks', 'cvm' or None."
        )

    ascending = metric != "loglikelihood"
    has_gof = gof_test is not None
    n_active = len(dists_objs)

    # ── Map distributions → fixed-size index & mask arrays ──
    raw_indices = _dist_to_indices(dists_objs)
    # Pad to _MAX_DISTS with index 0 (inactive slots are masked out)
    dist_indices = jnp.zeros(_MAX_DISTS, dtype=jnp.int32).at[:n_active].set(raw_indices)
    active_mask = jnp.arange(_MAX_DISTS) < n_active

    # ── Build branches & call JIT core ──
    branches = tuple(_make_branches(metric, gof_test))

    order, params_arrs, metrics, gof_stats, gof_pvals, final_mask, n_pass = _jit_core(
        x=jnp.asarray(x, dtype=float),
        dist_indices=dist_indices,
        active_mask=active_mask,
        significance_level=jnp.asarray(significance_level, dtype=float),
        branches=branches,
        ascending=ascending,
        has_gof=has_gof,
    )

    # ── Reconstruct Python result dicts from JIT output ──
    if has_gof and int(n_pass) == 0:
        return None, ()

    output = []
    for i in range(_MAX_DISTS):
        idx = int(order[i])
        if not bool(final_mask[idx]):
            continue
        dist = _DIST_REGISTRY[int(dist_indices[idx])]
        n_p = dist.n_params
        keys = tuple(dist.example_params().keys())
        params = dist._args_transform(
            {k: params_arrs[idx, j] for j, k in enumerate(keys)}
        )
        result = {
            "params": params,
            "metric": metrics[idx],
            "dist": dist,
        }
        if has_gof:
            result["gof"] = {
                "statistic": gof_stats[idx],
                "p_value": gof_pvals[idx],
            }
        output.append(result)

    return 0, tuple(output)


# ── Batched public API ─────────────────────────────────────────────────────
def batch_univariate_fitter(
    x: jnp.ndarray,
    metric: str = "bic",
    distributions: Iterable | str = "common continuous",
    gof_test: str | None = None,
    significance_level: float = 0.05,
) -> list[tuple]:
    r"""Fit univariate distributions to every column of *x* simultaneously.

    Equivalent to calling :func:`univariate_fitter` on each column, but
    uses ``jax.vmap`` to process all columns in a **single device call**,
    which is significantly faster for multi-dimensional data.

    Args:
        x (ArrayLike): Input data of shape ``(n, d)``.
        metric (str): Selection metric — ``'aic'``, ``'bic'``, or
            ``'loglikelihood'``.  Default ``'bic'``.
        distributions (Iterable | str): Distributions to try (same as
            :func:`univariate_fitter`).
        gof_test (str | None): Optional goodness-of-fit test (``'ks'``,
            ``'cvm'``, or ``None``).
        significance_level (float): GoF p-value threshold (default 0.05).

    Returns:
        list[tuple]: One ``(best_index, fitted)`` tuple per column, in
        the same format as :func:`univariate_fitter`.
    """
    # ── Shared validation (done once) ──
    dists_objs = _get_dist_objects(distributions)

    if metric not in ("aic", "bic", "loglikelihood"):
        raise ValueError(
            f"Invalid value for 'metric' argument: {metric}. "
            "Must be one of 'aic', 'bic' or 'loglikelihood'."
        )

    if gof_test is not None and gof_test not in _GOF_FUNCS:
        raise ValueError(
            f"Invalid value for 'gof_test' argument: {gof_test}. "
            "Must be one of 'ks', 'cvm' or None."
        )

    ascending = metric != "loglikelihood"
    has_gof = gof_test is not None
    n_active = len(dists_objs)

    raw_indices = _dist_to_indices(dists_objs)
    dist_indices = jnp.zeros(_MAX_DISTS, dtype=jnp.int32).at[:n_active].set(raw_indices)
    active_mask = jnp.arange(_MAX_DISTS) < n_active
    branches = tuple(_make_branches(metric, gof_test))

    # ── Single batched device call ──
    x_batch = jnp.asarray(x, dtype=float).T  # (d, n)

    (
        orders,
        params_arrs_all,
        metrics_all,
        gof_stats_all,
        gof_pvals_all,
        final_masks,
        n_passes,
    ) = _batched_jit_core(
        x_batch=x_batch,
        dist_indices=dist_indices,
        active_mask=active_mask,
        significance_level=jnp.asarray(significance_level, dtype=float),
        branches=branches,
        ascending=ascending,
        has_gof=has_gof,
    )

    # ── Reconstruct per-column Python result dicts ──
    d = x_batch.shape[0]
    results = []
    for dim in range(d):
        order = orders[dim]
        params_arrs = params_arrs_all[dim]
        metrics = metrics_all[dim]
        final_mask = final_masks[dim]
        n_pass = n_passes[dim]

        if has_gof and int(n_pass) == 0:
            results.append((None, ()))
            continue

        output = []
        for i in range(_MAX_DISTS):
            idx = int(order[i])
            if not bool(final_mask[idx]):
                continue
            dist = _DIST_REGISTRY[int(dist_indices[idx])]
            keys = tuple(dist.example_params().keys())
            params = dist._args_transform(
                {k: params_arrs[idx, j] for j, k in enumerate(keys)}
            )
            result = {
                "params": params,
                "metric": metrics[idx],
                "dist": dist,
            }
            if has_gof:
                result["gof"] = {
                    "statistic": gof_stats_all[dim, idx],
                    "p_value": gof_pvals_all[dim, idx],
                }
            output.append(result)

        results.append((0, tuple(output)))

    return results
