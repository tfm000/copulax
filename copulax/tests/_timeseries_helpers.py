"""Frozen-series access and the shared fit registry for the time-series tests.

This module is the single data-and-fit gateway for the nine
``test_timeseries_*.py`` modules.  It does two things:

1. :func:`series` hands out the **frozen** test series committed in
   ``copulax/tests/_r_reference/frozen_series_data.py``.  Nothing here
   simulates a process — every series was generated once by a
   third-party engine (rugarch ``ugarchpath`` / statsmodels
   ``arma_generate_sample``) or by a one-time committed port for the
   handful of DGPs no third-party engine represents, and committed with
   its own SHA-256.  The regenerator is
   ``_r_reference/generate_frozen_series_handrolled.py``.
2. :func:`shared_fit` / :func:`shared_case` hand out **one fitted model
   per distinct fit**, shared across every consumer in every module, with
   an explicit tier that fixes the iteration budget.

The module name does not match ``test_*.py``, so pytest never collects
it.  ``copulax/tests/__init__.py`` exists and ``pytest.ini`` sets
``pythonpath = .``, so ``from copulax.tests._timeseries_helpers import
...`` resolves from any test module.

Why the data is frozen
----------------------
Before the corpus, every test series was rolled at test runtime by a
copulax-authored recursion, which put a copulax formula on both sides of
every statistical assertion — the process the test fits and the estimator
it checks came from the same hand.  Loading committed third-party data
removes the hand-rolled DGP from the test path and makes every series
byte-identical on every machine, every CI leg and every rerun.

Fit tiers
---------
Fitting is the dominant cost of this test family, and before the registry
the same (model, series, budget) triple was refitted in several modules
with a scatter of iteration budgets (200 / 300 / 400 / 500 / 600 / 800 /
1000) that carried no semantic meaning.  Every fit now names a tier, and
the tier fixes the budget:

.. list-table::
   :header-rows: 1

   * - Tier
     - Budget
     - Used by
   * - :data:`REFERENCE`
     - ``init="analytical"``, ``n_starts=`` :data:`N_STARTS_FULL`,
       ``maxiter=`` :data:`MAXITER_REFERENCE`
     - the ``test_timeseries_arma_garch.py`` matrix machinery only —
       the fits that are compared against rugarch.  Values are frozen:
       this tier's arguments must not change.
   * - :data:`STANDARD`
     - ``maxiter=`` :data:`MAXITER_STANDARD`
     - every "any fitted object will do" consumer: serialization,
       plotting, summary rendering, diagnostics smoke tests, JIT probes.
   * - :data:`PRECISION`
     - ``maxiter=`` :data:`MAXITER_PRECISION`
     - parameter-recovery and third-party-agreement fits, where the
       assertion is on the location of the optimum.
   * - :data:`BEHAVIOURAL`
     - caller's own arguments, **never cached**
     - tests whose SUBJECT is the iteration budget or the init path
       (``maxiter=0/2/10/20/40``, warm starts, convergence warnings,
       multi-start statistics).  A behavioural fit must never be served
       from — or written into — the shared cache, because its arguments
       are the thing under test.

:data:`MAXITER_STANDARD` is ``300``: the modal middle-band value in the
suite before consolidation, the production default of
:meth:`ArmaGarch.fit`, and an increase on every site it replaces except
the 400/500 pair in the same band, so no "any fitted object" consumer
loses convergence relative to its old budget in the common case.
:data:`MAXITER_PRECISION` is ``600``: the budget the pre-existing n=2000
GARCH recovery fixture already used, and enough for the third-party
agreement bands (verified per-site — see the 01-15 audit).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any, Callable, Hashable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from copulax.tests._r_reference.frozen_series_data import FROZEN_SERIES


__all__ = [
    # Frozen data
    "FROZEN_SERIES_NAMES",
    "series",
    "series_provenance",
    # Tiers and their canonical arguments
    "REFERENCE",
    "STANDARD",
    "PRECISION",
    "BEHAVIOURAL",
    "TIERS",
    "MAXITER_REFERENCE",
    "MAXITER_STANDARD",
    "MAXITER_PRECISION",
    "N_STARTS_FULL",
    "FIT_LR",
    "tier_kwargs",
    # Registry
    "shared_fit",
    "shared_case",
    "fit_key",
    "fit_snapshot",
    "registry_keys",
    "assert_snapshot_intact",
]


# ---------------------------------------------------------------------------
# Frozen series access
# ---------------------------------------------------------------------------

#: Every frozen series name, sorted.  A test that asks for a name outside
#: this set gets a ``KeyError`` naming the closest matches.
FROZEN_SERIES_NAMES: tuple[str, ...] = tuple(sorted(FROZEN_SERIES))

#: ``{name: jax.Array}`` — one converted array per name, per process.
_SERIES_CACHE: dict[str, jax.Array] = {}


def series(name: str) -> jax.Array:
    """Return the frozen test series ``name`` as a jax array.

    The committed corpus stores float64; jax downcasts to float32 at this
    call unless x64 is enabled, exactly as the other reference modules in
    ``_r_reference/`` behave.  The converted array is cached per process,
    so repeated calls hand out the identical (immutable) instance rather
    than re-converting a multi-thousand-element array.

    Parameters
    ----------
    name : str
        A key of ``FROZEN_SERIES`` — see :data:`FROZEN_SERIES_NAMES`.
        Names encode the DGP, length and seed, e.g. ``garch11_n2000_s2``.

    Returns
    -------
    jax.Array
        The series, shape ``(n,)``.

    Raises
    ------
    KeyError
        If ``name`` is not in the frozen corpus.  The message lists the
        names sharing its family prefix, which is almost always the typo.

    Examples
    --------
    >>> eps = series("garch11_n500_s2")
    >>> eps.shape
    (500,)
    """
    cached = _SERIES_CACHE.get(name)
    if cached is not None:
        return cached

    entry = FROZEN_SERIES.get(name)
    if entry is None:
        prefix = name.split("_")[0]
        near = [n for n in FROZEN_SERIES_NAMES if n.startswith(prefix)]
        raise KeyError(
            f"no frozen series named {name!r}. "
            + (f"Series in the {prefix!r} family: {near}."
               if near else
               f"Known families: "
               f"{sorted({n.split('_')[0] for n in FROZEN_SERIES_NAMES})}.")
        )

    arr = jnp.asarray(entry["y"])
    _SERIES_CACHE[name] = arr
    return arr


def series_provenance(name: str) -> dict[str, Any]:
    """Return the provenance record of the frozen series ``name``.

    Parameters
    ----------
    name : str
        A key of ``FROZEN_SERIES``.

    Returns
    -------
    dict
        ``{"generator", "engine", "spec", "seed", "n", "sha256"}``.

    Raises
    ------
    KeyError
        If ``name`` is not in the frozen corpus.
    """
    series(name)  # reuse the error message and the existence check
    return dict(FROZEN_SERIES[name]["provenance"])


# ---------------------------------------------------------------------------
# Fit tiers
# ---------------------------------------------------------------------------

#: Oracle-comparison tier.  Reserved for the ARMA-GARCH matrix machinery.
REFERENCE = "reference"

#: "Any fitted object will do" tier.
STANDARD = "standard"

#: Parameter-recovery / third-party-agreement tier.
PRECISION = "precision"

#: Tier for tests whose subject IS the fit budget.  Never cached.
BEHAVIOURAL = "behavioural"

#: All four tiers, in increasing order of budget (BEHAVIOURAL last: it
#: has no fixed budget at all).
TIERS: tuple[str, ...] = (STANDARD, PRECISION, REFERENCE, BEHAVIOURAL)

#: Canonical iteration budget of the :data:`REFERENCE` tier.  Frozen —
#: the matrix fits are cross-validated against rugarch.
MAXITER_REFERENCE = 1500

#: Canonical iteration budget of the :data:`STANDARD` tier.
MAXITER_STANDARD = 300

#: Canonical iteration budget of the :data:`PRECISION` tier.
MAXITER_PRECISION = 600

#: Multi-start candidate count used by the :data:`REFERENCE` tier.  The
#: value caps at the number of available candidates (4 joint / 3
#: standalone), so one constant covers both.
N_STARTS_FULL = 4

#: Adam learning rate shared by every tier.  It was already 0.05 —
#: :meth:`fit`'s own default — at every call site in the family.
FIT_LR = 0.05

_TIER_KWARGS: dict[str, dict[str, Any]] = {
    REFERENCE: {
        "init": "analytical",
        "n_starts": N_STARTS_FULL,
        "maxiter": MAXITER_REFERENCE,
        "lr": FIT_LR,
    },
    STANDARD: {"maxiter": MAXITER_STANDARD},
    PRECISION: {"maxiter": MAXITER_PRECISION},
    BEHAVIOURAL: {},
}


def tier_kwargs(tier: str) -> dict[str, Any]:
    """Return the canonical ``fit`` keyword arguments of ``tier``.

    Parameters
    ----------
    tier : str
        One of :data:`TIERS`.

    Returns
    -------
    dict
        A fresh copy of the tier's defaults; mutating it is harmless.

    Raises
    ------
    ValueError
        If ``tier`` is not a known tier.
    """
    if tier not in _TIER_KWARGS:
        raise ValueError(
            f"unknown fit tier {tier!r}; expected one of {TIERS}"
        )
    return dict(_TIER_KWARGS[tier])


# ---------------------------------------------------------------------------
# The shared fit registry
# ---------------------------------------------------------------------------

#: ``{key: fitted model}`` — one entry per distinct (tier, model, data,
#: fit-argument) combination, for the whole pytest process.  Shared
#: across modules: a fit requested by ``test_timeseries_variance.py`` and
#: again by ``test_timeseries_summary.py`` under the same key runs once.
_FIT_REGISTRY: dict[Hashable, Any] = {}

#: ``{key: (flat params, loglikelihood)}`` captured the moment a fit is
#: first built.  A shared instance is only safe while nothing mutates it,
#: so :func:`assert_snapshot_intact` compares the live model against this.
_FIT_SNAPSHOT: dict[Hashable, tuple[dict[str, np.ndarray], float]] = {}


def _flatten(x: Any) -> np.ndarray:
    """Flatten a parameter value to a 1-D float array."""
    return np.asarray(jnp.atleast_1d(jnp.asarray(x, dtype=float))).ravel()


def _describe(value: Any) -> str:
    """Render a model field as a stable, hashable descriptor string."""
    if isinstance(value, type):
        return f"class:{value.__module__}.{value.__qualname__}"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return repr(value)
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(_describe(v) for v in value) + ")"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{k!r}: {_describe(v)}" for k, v in sorted(value.items())
        ) + "}"
    if isinstance(value, (np.ndarray, jax.Array)):
        return "array:" + repr(np.asarray(value).tolist())
    # Distribution singletons and any other module-level object: the
    # concrete type is the identity that matters (``normal`` is the only
    # ``Normal``), and its name is stable across processes.
    return f"obj:{type(value).__module__}.{type(value).__qualname__}"


def _model_signature(model: Any) -> tuple[str, ...]:
    """Describe an UNFITTED model completely enough to key a cache on it.

    Equinox modules are frozen dataclasses whose fitted state (params,
    terminal state, standard errors, ...) is ``None`` before ``fit``
    runs, so the non-``None`` fields of an unfitted model are exactly its
    structural definition: concrete class, name, orders and residual law.

    Parameters
    ----------
    model : Any
        An unfitted copulax time-series model.

    Returns
    -------
    tuple[str, ...]
        A hashable, human-readable signature.  Two models produce the
        same signature if and only if they are structurally identical.
    """
    parts = [f"{type(model).__module__}.{type(model).__qualname__}"]
    for field in dataclasses.fields(model):
        value = getattr(model, field.name, None)
        if value is None:
            continue
        parts.append(f"{field.name}={_describe(value)}")
    return tuple(parts)


def _kwargs_signature(kwargs: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Render resolved ``fit`` keyword arguments as a hashable signature."""
    return tuple(
        (name, _describe(value)) for name, value in sorted(kwargs.items())
    )


def _resolve(tier: str, fit_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Merge a tier's canonical arguments with the caller's overrides."""
    resolved = tier_kwargs(tier)
    resolved.update(fit_kwargs)
    return resolved


def fit_key(
    model: Any,
    series_name: str,
    *,
    tier: str = STANDARD,
    tag: Optional[str] = None,
    **fit_kwargs: Any,
) -> tuple:
    """Return the registry key a :func:`shared_fit` call would use.

    Exposed so the isolation guard can look a fit up in the snapshot
    table without re-deriving the key by hand.

    Parameters
    ----------
    model : Any
        The unfitted model, exactly as passed to :func:`shared_fit`.
    series_name : str
        The frozen series name, exactly as passed to :func:`shared_fit`.
    tier : str, optional
        The fit tier.  Default :data:`STANDARD`.
    tag : str, optional
        Data tag for a locally transformed series — see
        :func:`shared_fit`.
    **fit_kwargs
        Overrides of the tier's canonical arguments.

    Returns
    -------
    tuple
        ``(tier, model signature, series name, tag, fit-argument
        signature)``.
    """
    return (
        tier,
        _model_signature(model),
        series_name,
        tag,
        _kwargs_signature(_resolve(tier, fit_kwargs)),
    )


def shared_fit(
    model: Any,
    series_name: str,
    *,
    tier: str = STANDARD,
    y: Optional[jax.Array] = None,
    tag: Optional[str] = None,
    transform: Optional[Callable[[jax.Array], jax.Array]] = None,
    **fit_kwargs: Any,
) -> Any:
    """Fit ``model`` on a frozen series once per distinct key, process-wide.

    The key is ``(tier, model signature, series name, tag, resolved fit
    arguments)`` — the complete set of inputs that determine the result —
    so two callers anywhere in the test family that ask for the same fit
    share one computation, and two callers that differ in *any* respect
    never collide.

    Fitted models are frozen equinox PyTrees and every consumer only
    reads from them, so handing out the shared instance is safe;
    :func:`assert_snapshot_intact` is the tripwire that keeps it true.

    Parameters
    ----------
    model : Any
        An **unfitted** copulax time-series model.  Constructed fresh by
        the caller; only its structure is used for the key.
    series_name : str
        Name of the frozen series to fit — see :func:`series`.  When
        ``y`` or ``transform`` is supplied this is the name of the
        *base* series, and ``tag`` distinguishes the derived data.
    tier : str, optional
        One of :data:`TIERS`.  Default :data:`STANDARD`.  The tier
        supplies the canonical fit arguments; ``fit_kwargs`` override
        them.  :data:`BEHAVIOURAL` fits are never cached.
    y : jax.Array, optional
        Explicit data, replacing the frozen series.  Requires ``tag``.
    transform : callable, optional
        Applied to the frozen series before fitting.  Requires ``tag``.
    tag : str, optional
        Short label for derived data (``"scaled_500x"``,
        ``"inf_at_n_over_3"``).  Part of the key, so derived data can
        never be served a fit of the base series.
    **fit_kwargs
        Overrides of the tier's canonical ``fit`` arguments.

    Returns
    -------
    Any
        The fitted model — the shared instance for cached tiers, a fresh
        one for :data:`BEHAVIOURAL`.

    Raises
    ------
    ValueError
        If ``y`` or ``transform`` is given without ``tag``, if both are
        given, or if ``tier`` is unknown.

    Examples
    --------
    >>> from copulax.timeseries import GARCH
    >>> from copulax.univariate import normal
    >>> fit = shared_fit(
    ...     GARCH(p=1, q=1, residual_dist=normal), "garch11_n500_s2",
    ... )
    >>> fit is shared_fit(
    ...     GARCH(p=1, q=1, residual_dist=normal), "garch11_n500_s2",
    ... )
    True
    """
    if y is not None and transform is not None:
        raise ValueError("pass either y or transform, not both")
    if (y is not None or transform is not None) and tag is None:
        raise ValueError(
            "derived data needs an explicit tag so it cannot be served "
            "the base series' fit"
        )

    resolved = _resolve(tier, fit_kwargs)
    key = (
        tier,
        _model_signature(model),
        series_name,
        tag,
        _kwargs_signature(resolved),
    )

    if tier == BEHAVIOURAL:
        # The arguments are the thing under test: never cached, never
        # visible to another caller.
        data = y if y is not None else series(series_name)
        if transform is not None:
            data = transform(data)
        return model.fit(data, **resolved)

    cached = _FIT_REGISTRY.get(key)
    if cached is not None:
        return cached

    data = y if y is not None else series(series_name)
    if transform is not None:
        data = transform(data)
    fitted = model.fit(data, **resolved)
    _FIT_REGISTRY[key] = fitted
    _FIT_SNAPSHOT[key] = (
        {
            name: _flatten(value)
            for name, value in fitted.params.items()
            if not isinstance(value, dict)
        },
        float(fitted.loglikelihood()),
    )
    return fitted


def shared_case(
    model: Any,
    series_name: str,
    *,
    tier: str = STANDARD,
    y: Optional[jax.Array] = None,
    tag: Optional[str] = None,
    transform: Optional[Callable[[jax.Array], jax.Array]] = None,
    label: Optional[str] = None,
    **fit_kwargs: Any,
) -> SimpleNamespace:
    """A **fresh** namespace wrapping the shared series and shared fit.

    The cached fit is never handed out inside a shared wrapper:
    consumers assign attributes onto the case they are given, so a
    shared mutable wrapper would leak those writes between fixtures.
    Every call rebuilds the namespace around the same immutable values.

    Parameters
    ----------
    model, series_name, tier, y, tag, transform, **fit_kwargs
        As :func:`shared_fit`.
    label : str, optional
        Value for ``case.label``.  Defaults to ``series_name``.

    Returns
    -------
    types.SimpleNamespace
        ``label``, ``y`` (the data actually fitted), ``fit`` and ``key``.
    """
    data = y if y is not None else series(series_name)
    if transform is not None:
        data = transform(data)
    return SimpleNamespace(
        label=series_name if label is None else label,
        y=data,
        fit=shared_fit(
            model, series_name, tier=tier, y=y, tag=tag,
            transform=transform, **fit_kwargs,
        ),
        key=fit_key(
            model, series_name, tier=tier, tag=tag, **fit_kwargs,
        ),
    )


def fit_snapshot(key: Hashable) -> tuple[dict[str, np.ndarray], float]:
    """Return the ``(params, loglikelihood)`` snapshot taken at build time.

    Parameters
    ----------
    key : Hashable
        A key returned by :func:`fit_key`.

    Returns
    -------
    tuple
        ``({param name: flat array}, loglikelihood)``.

    Raises
    ------
    KeyError
        If no cached fit exists for ``key``.
    """
    return _FIT_SNAPSHOT[key]


def registry_keys() -> tuple[Hashable, ...]:
    """Return every key currently held by the registry, in insertion order."""
    return tuple(_FIT_REGISTRY)


def assert_snapshot_intact(key: Hashable) -> None:
    """Assert the shared fit for ``key`` still matches its build-time snapshot.

    The mutation tripwire behind the whole registry: sharing one fitted
    model between consumers is sound only while nothing writes to it.

    Parameters
    ----------
    key : Hashable
        A key returned by :func:`fit_key`.

    Raises
    ------
    KeyError
        If no cached fit exists for ``key``.
    AssertionError
        If any parameter or the log-likelihood has moved.
    """
    fitted = _FIT_REGISTRY[key]
    snap_params, snap_loglik = _FIT_SNAPSHOT[key]
    for name, expected in snap_params.items():
        np.testing.assert_array_equal(
            _flatten(fitted.params[name]), expected,
            err_msg=f"shared fit param {name!r} mutated (key={key!r})",
        )
    assert float(fitted.loglikelihood()) == snap_loglik, (
        f"shared fit loglikelihood mutated (key={key!r})"
    )
