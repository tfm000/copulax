"""Abstract base class for the ``copulax.timeseries`` model families.

Defines :class:`TimeSeriesModel` and the two intermediate marker
classes :class:`MeanModel` and :class:`VarianceModel`.  Each concrete
family — AR / MA / ARMA mean models, GARCH-family variance models,
and the joint ARMA-GARCH composite — inherits from one of these
intermediates.

The base provides only the cross-cutting machinery shared by every
family:

* equinox PyTree semantics — fitted models are immutable, JIT- and
  autograd-compatible, and round-trip through the shared
  :mod:`copulax._src._serialization` machinery once the timeseries
  branch is registered there;
* the ``name`` / ``params`` / ``_fitted_instance`` API mirrors
  :class:`copulax._src._distributions.Distribution`, so post-fit
  object construction looks identical to a univariate fit;
* shared input validation: every public method that consumes a series
  ``y`` runs :func:`copulax._src.univariate._utils._univariate_input`
  then ravels to 1D before handing off to the recursion.  This is the
  same input convention every univariate distribution uses;
* a :class:`TerminalState` marker base whose subclasses store the
  small constant-size carry state (last few returns / innovations /
  squared residuals / variances) that ``forecast(h)`` rolls forward
  from.  The schema is per-family, so each family base class declares
  its own subclass.

Concrete recursion kernels, stationarity reparameterisations, and
per-family fit objectives live in their own modules — this file
deliberately stays small.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, ClassVar, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import _params_equal
from copulax._src._params import guard_params
from copulax._src.optimize import projected_gradient
from copulax._src.univariate._utils import _univariate_input


###############################################################################
# residual_diagnostics_ serialisation helpers
###############################################################################
_DIAG_TEST_KEYS: tuple[str, ...] = (
    "ljung_box", "ljung_box_sq", "arch_lm", "adf", "kpss",
)
_DIAG_INT_SUBKEYS: frozenset = frozenset({"used_lag", "n_obs", "dof"})


def _serialise_residual_diagnostics(
    diag: dict, arrays: dict, metadata: dict,
) -> None:
    r"""Round-trip the consolidated ``residual_diagnostics_`` bundle
    into the ``(arrays, metadata)`` pair used by
    :meth:`TimeSeriesModel._serialise_traced`.

    Layout:

    * ``"acf"``, ``"pacf"`` — written to ``arrays["diag_acf"]`` /
      ``arrays["diag_pacf"]`` as numpy arrays (they are
      ``(lags + 1,)`` 1-D float arrays).
    * ``"crit_values"`` inside the per-test sub-dicts (ADF / KPSS) —
      written to ``arrays[f"diag_{test}_crit_values"]`` as numpy
      arrays (3-element for ADF, 4-element for KPSS, aligned with
      :data:`copulax._src.timeseries._unit_root.ADF_CRIT_LEVELS` /
      ``KPSS_CRIT_LEVELS``).
    * Every other entry — the ``loglikelihood`` / ``aic`` / ``bic``
      scalars and the remaining sub-keys of each hypothesis-test
      result dict (``statistic`` / ``p_value`` as floats,
      ``used_lag`` / ``n_obs`` / ``dof`` round-tripped as floats and
      restored to ``int32`` on load) — JSON-encodable after casting
      to Python floats; goes under
      ``metadata["residual_diagnostics"]`` as a single nested dict.

    The split keeps array data in the ``.cpx`` numpy section
    (compressed, type-preserved) and small structured data in the
    metadata section (round-trips through the existing JSON
    pipeline).
    """
    import numpy as np

    rest: dict = {}
    for key, val in diag.items():
        if key in ("acf", "pacf"):
            arrays[f"diag_{key}"] = np.asarray(val)
        elif isinstance(val, dict):
            sub: dict = {}
            for sub_key, sub_val in val.items():
                if sub_key == "crit_values":
                    arrays[f"diag_{key}_crit_values"] = np.asarray(sub_val)
                else:
                    sub[sub_key] = float(sub_val)
            rest[key] = sub
        else:
            # Top-level scalar (loglikelihood / aic / bic).
            rest[key] = float(val)
    if rest:
        metadata["residual_diagnostics"] = rest


def _deserialise_residual_diagnostics(
    arrays: dict, metadata: dict,
) -> Optional[dict]:
    r"""Inverse of :func:`_serialise_residual_diagnostics`.

    Returns the rebuilt dict (with the ``acf`` / ``pacf`` and per-test
    ``crit_values`` arrays reattached and every scalar entry cast back
    to a JAX array — ``int32`` for ``used_lag`` / ``n_obs`` / ``dof``,
    ``float`` for everything else — so the cached-default accessors
    return the same dtype contract as a freshly-fitted model), or
    ``None`` when no diagnostic state was serialised.

    ``diag_n_train_`` is deliberately excluded from the ``diag_`` prefix
    scan (WR-03): ``_serialise_traced`` writes it for *every* model with
    ``n_train_`` set — including models with no diagnostics bundle — so
    treating it as a diagnostics marker would resurrect an empty ``{}``
    bundle for a diagnostics-less model.  The constructor stores that
    ``{}`` (it only checks ``is not None``), which then masks the
    designed informative error behind a bare ``KeyError`` / ``TypeError``
    on the cached fast paths.  Returning ``None`` here keeps the
    ``residual_diagnostics_ is None`` gates honest.
    """
    has_meta = "residual_diagnostics" in metadata
    has_arr = any(
        k.startswith("diag_") and k != "diag_n_train_" for k in arrays
    )
    if not has_meta and not has_arr:
        return None
    diag: dict = dict(metadata.get("residual_diagnostics", {}))
    for key in ("loglikelihood", "aic", "bic"):
        if key in diag:
            diag[key] = jnp.asarray(diag[key], dtype=float)
    if "diag_acf" in arrays:
        diag["acf"] = jnp.asarray(arrays["diag_acf"], dtype=float)
    if "diag_pacf" in arrays:
        diag["pacf"] = jnp.asarray(arrays["diag_pacf"], dtype=float)
    for test_key in _DIAG_TEST_KEYS:
        if test_key not in diag:
            continue
        sub = dict(diag[test_key])
        for sk in list(sub.keys()):
            dtype = jnp.int32 if sk in _DIAG_INT_SUBKEYS else float
            sub[sk] = jnp.asarray(sub[sk], dtype=dtype)
        crit_key = f"diag_{test_key}_crit_values"
        if crit_key in arrays:
            sub["crit_values"] = jnp.asarray(arrays[crit_key], dtype=float)
        diag[test_key] = sub
    # Defensive floor (WR-03): if nothing but ``diag_n_train_`` reached
    # this point, ``diag`` is empty — collapse it to ``None`` rather than
    # returning a ``{}`` bundle that would pass every ``is not None`` gate.
    return diag or None


###############################################################################
# Terminal-state marker base
###############################################################################
class TerminalState(eqx.Module):
    r"""Marker base for per-family terminal-state PyTrees.

    Every fitted time-series model carries a ``terminal_state`` field
    populated automatically from the training data at fit time.  The
    schema is per-family — σ²-form GARCH variants store the last
    ``p`` squared residuals and ``q`` conditional variances; EGARCH
    stores standardised residuals and log-variances; ARMA stores the
    last ``p`` returns and ``q`` innovations; the joint composite
    stores both halves — so each family declares its own subclass with
    the appropriate fields.

    The base provides only the marker for type-checking and for
    routing through the shared ``.cpx`` serialiser.  All subclasses
    must hold ``O(max(p, q))`` traced ``Array`` leaves (no training
    data) — this is what keeps ``forecast(h)`` working out-of-the-box
    while preserving the constant-size-on-disk invariant.
    """


###############################################################################
# TimeSeriesModel base
###############################################################################
class TimeSeriesModel(eqx.Module):
    r"""Abstract base for all time-series models in :mod:`copulax.timeseries`.

    Subclasses fall into one of three families:

    1. :class:`MeanModel` — AR / MA / ARMA mean equations.
    2. :class:`VarianceModel` — GARCH-family conditional-variance
       equations (vanilla, IGARCH, GJR, EGARCH, TGARCH, QGARCH,
       GARCH-M).
    3. ``ArmaGarch`` — joint ARMA-GARCH composite (defined in
       ``_joint/arma_garch.py`` and inheriting from this base
       directly, since it owns both a mean and a variance recursion).

    The base enforces the following shared contract:

    * The model is an :mod:`equinox` PyTree.  Static configuration
      (``p``, ``q``, ``residual_dist`` template, ``name``) lives behind
      ``eqx.field(static=True)`` on each subclass; fitted parameters,
      the residual distribution's parameter dict, and the per-family
      terminal state flow through as traced ``Array`` leaves.
    * ``params`` returns a parameter ``dict`` mirroring the canonical
      :class:`copulax._src._distributions.Univariate` shape, and
      :meth:`_fitted_instance` reconstructs a fitted model from such a
      dict — so warm-starts and save/load round-trip through the same
      representation the user already manipulates.
    * Every public method that consumes a series ``y`` first routes it
      through :func:`_univariate_input` and ravels to 1D, matching the
      input convention used by every :class:`Univariate` distribution.

    Notes:
        Concrete subclasses must implement :meth:`fit`,
        :meth:`residuals`, :meth:`conditional_mean`,
        :meth:`conditional_variance`, :meth:`stats`, :meth:`forecast`,
        and :meth:`rvs`.  The dispatcher / type-checking helpers
        defined here are non-abstract and shared.
    """

    _name: str = eqx.field(static=True)

    #: Strings the subclass's :meth:`fit` dispatcher accepts via the
    #: ``init`` kwarg.  Mirrors :attr:`Distribution._supported_methods`.
    _supported_methods: ClassVar[frozenset] = frozenset()

    def __init_subclass__(cls, **kwargs):
        r"""Surface inherited docstrings on subclass overrides.

        Mirrors :meth:`Distribution.__init_subclass__`: ``inspect.getdoc``
        does not walk the MRO past an override whose ``__doc__`` is
        ``None``, so ``help()``, IPython ``?`` and IDE hover tooltips
        show nothing for subclass overrides that omit a docstring even
        when the parent declares the contract in detail.  This hook
        copies the first parent docstring it finds onto each public
        override that lacks its own.
        """
        super().__init_subclass__(**kwargs)
        for name, attr in cls.__dict__.items():
            if not callable(attr) or name.startswith("_"):
                continue
            if getattr(attr, "__doc__", None):
                continue
            for base in cls.__mro__[1:]:
                parent = base.__dict__.get(name)
                parent_doc = getattr(parent, "__doc__", None) if parent else None
                if parent_doc:
                    try:
                        attr.__doc__ = parent_doc
                    except (AttributeError, TypeError):
                        pass
                    break

    def __init__(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        # Object-identity hash mirrors :class:`Distribution.__hash__` —
        # required by equinox/JAX for JIT tracing of bound methods.
        return id(self)

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        sp = self._stored_params
        op = other._stored_params  # type: ignore[union-attr]
        if sp is None and op is None:
            return True
        if sp is None or op is None:
            return False
        return _params_equal(sp, op)

    # ------------------------------------------------------------------
    # name / params / dist_type
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Display name for the model."""
        return self._name

    @property
    def dist_type(self) -> str:
        """Family routing label used by :mod:`copulax._src._serialization`."""
        return "timeseries"

    @property
    def dtype(self) -> str:
        """Data type of the modelled series.  Time series operate on
        continuous-valued returns / innovations."""
        return "continuous"

    @property
    def _stored_params(self) -> Optional[dict]:
        r"""Override in subclasses: return a parameter ``dict`` produced
        by the subclass's ``_params_dict(*arrays)`` classmethod, or
        ``None`` when the model is unfitted."""
        return None

    @property
    def params(self) -> Optional[dict]:
        """Stored model parameters as a JAX-compatible ``dict``, or
        ``None`` when the model is unfitted."""
        return self._stored_params

    @property
    def is_fitted(self) -> bool:
        """``True`` iff the model has stored fitted parameters."""
        return self._stored_params is not None

    # ------------------------------------------------------------------
    # Fit-method dispatch helpers
    # ------------------------------------------------------------------
    def _check_method(self, method: str) -> None:
        r"""Validate ``method`` against the subclass's accepted set.

        Mirrors :meth:`Distribution._check_method`.

        Raises:
            ValueError: when ``method`` is not in
                ``self._supported_methods``.
        """
        if method not in self._supported_methods:
            raise ValueError(
                f"Method {method!r} not supported by "
                f"{type(self).__name__}. Supported: "
                f"{sorted(self._supported_methods)}."
            )

    @staticmethod
    def _validate_series(y: ArrayLike) -> Array:
        r"""Apply the canonical univariate-input check then ravel to 1D.

        :func:`_univariate_input` casts dtype and reshapes to ``(n, 1)``;
        time-series recursions consume a 1D series, so we ravel after.
        This matches the input convention every :class:`Univariate`
        distribution uses, guaranteeing identical handling of pandas
        Series, scalar, list, and ``jnp.ndarray`` inputs across the
        whole library.

        Args:
            y: Input series.  Must be 1D-like (length ``n``).

        Returns:
            1D ``jnp.ndarray`` of shape ``(n,)`` and float dtype.
        """
        arr, _ = _univariate_input(y)
        return arr.ravel()

    @staticmethod
    def _guard_residual_params(family_key: str, residual_params):
        r"""Route residual params through the typed-parameter guard.

        Time-series models have no monolithic ``params`` argument and no
        ``_resolve_params`` chokepoint — the residual law's parameters
        enter each family as the ``residual_params`` constructor kwarg.
        This shared helper is the time-series analog of
        :meth:`Distribution._resolve_params`' guard hook: every family
        ``__init__`` routes its ``residual_params`` through
        :func:`copulax._src._params.guard_params` so that, once the family
        is migrated to typed parameters (Phase 3), a raw dict raises
        :class:`ParamsTypeError`.  While ``_MIGRATED_FAMILIES`` is empty
        the call is a straight pass-through — a behavioural no-op for
        every family today.

        The key is the STABLE family identifier ``type(self).__name__``
        (e.g. ``"ARMA"``, ``"IGARCH"``, ``"ArmaGarch"``), passed by each
        family ``__init__``.  It is deliberately the class name and NOT
        the mutable display ``name``: the display name is user-settable
        and is auto-generated to a per-instance value for fitted
        instances (``FittedARMA(1,1)-...``), so keying on it would let a
        fitted / renamed instance bypass the migration guard entirely
        (WR-01).

        .. warning::

            The ``.cpx`` load path reconstructs fitted models through the
            same family ``__init__`` (e.g.
            :meth:`_build_fitted_instance` / ``_fitted_instance``), so
            this guard is reached on load as well as on user entry.  In
            Phase 0 that is safe because the migrated set is empty and the
            guard never fires.  **Phase 3 must route the load path through
            an UNGUARDED internal reconstruction before adding any family
            to** ``_MIGRATED_FAMILIES`` — otherwise the guard would reject
            the plain dicts deserialised from pre-4.0.0 ``.cpx`` files and
            break backward-compatible loading (PARM-06).

        Args:
            family_key: The STABLE family identifier key (e.g.
                ``"IGARCH"``, ``"ARMA"``) — ``type(self).__name__`` at
                each call site, NOT the display ``name=`` argument.
            residual_params: The residual law's parameters — a raw dict
                today, a typed params object once ``family_key`` is
                migrated.

        Returns:
            ``residual_params`` unchanged (the guard never mutates it).
        """
        return guard_params(family_key, residual_params)

    # ------------------------------------------------------------------
    # Convergence-status packing (D-09, HARD-06)
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_status_leaf(
        value: Optional[ArrayLike], dtype,
    ) -> Optional[Array]:
        r"""Coerce a convergence-status constructor argument to a typed
        array leaf, preserving ``None`` for unfitted instances.

        Mirrors the ``jnp.asarray(x, ...) if x is not None else None``
        idiom used for the other fitted-only leaves, centralised so the
        six D-09 status fields coerce identically across all three family
        constructors.
        """
        if value is None:
            return None
        return jnp.asarray(value, dtype=dtype)

    @staticmethod
    def _multi_start_fit(
        objective: Callable,
        starts: list,
        obj_kwargs: dict,
        lr: float,
        maxiter: int,
    ) -> tuple[dict, dict]:
        r"""Run a candidate-set multi-start fit and return the winner.

        Stacks the supplied candidate start vectors, ``jax.vmap`` s them
        through :func:`copulax._src.optimize.projected_gradient` (which
        returns the best iterate seen over each start's Adam scan, seeded
        with the objective at the start point), then selects the candidate
        with the highest finite log-likelihood.

        The objective is a mean negative log-likelihood the solver
        minimises, so a candidate's ``best_val`` is its (minimised)
        objective and the winning candidate is ``argmax`` of the *negated*
        objective over the candidates whose ``best_val`` is finite.
        Non-finite candidates (a start whose whole scan hit NaN, or whose
        objective is non-finite everywhere) are masked out first so a
        degenerate start can never win over a finite one; if every
        candidate is non-finite the argmax still returns index 0 and the
        winning ``val`` stays non-finite (the honest signal).

        All candidate starts must be the same flat shape so they stack
        into an ``(n_starts, k)`` batch; only the start vector is mapped —
        the objective's extra arguments (the series and pre-sample state)
        are shared across candidates, so every candidate is scored on the
        identical likelihood surface. This is what makes the returned
        argmax the same regardless of which cold-start mode requested the
        fit.

        Args:
            objective: The fit objective closure, signature
                ``objective(raw, *obj_args) -> scalar`` (minimised).
            starts: List of flat candidate start vectors (each shape
                ``(k,)``).
            obj_kwargs: The objective's extra keyword arguments (the
                series and pre-sample state), forwarded verbatim to the
                solver for every candidate.
            lr: Adam learning rate.
            maxiter: Adam iteration budget per candidate start.

        Returns:
            ``(winner_res, candidate_stats)`` where ``winner_res`` is a
            single-start :func:`projected_gradient` result dict for the
            winning candidate (keys ``x`` / ``val`` / ``best_val`` /
            ``nan_encountered``), and ``candidate_stats`` carries the
            ``n_finite_candidates`` (count of finite-``best_val``
            candidates) and ``best_candidate`` (winning candidate index)
            int32 array leaves for the D-09 status packing.
        """
        x0_batch = jnp.stack(starts)
        k = x0_batch.shape[1]
        lower = jnp.full((k, 1), -jnp.inf)
        upper = jnp.full((k, 1), jnp.inf)

        def _run_one(x0: Array) -> dict:
            return projected_gradient(
                f=objective,
                x0=x0,
                projection_method="projection_box",
                projection_options={"lower": lower, "upper": upper},
                lr=lr,
                maxiter=maxiter,
                **obj_kwargs,
            )

        # vmap over the start vectors only; obj_kwargs (series + pre-sample
        # state) are captured in the closure and shared across candidates.
        results = jax.vmap(_run_one)(x0_batch)

        best_vals = jnp.asarray(results["best_val"], dtype=float)
        finite = jnp.isfinite(best_vals)
        # best_val is the minimised objective, so maximise its negation.
        # Non-finite candidates are masked to -inf so they never win when a
        # finite candidate exists (the GH finite-likelihood guard).
        masked = jnp.where(finite, -best_vals, -jnp.inf)
        winner = jnp.argmax(masked)

        winner_res = {
            "x": results["x"][winner],
            "val": best_vals[winner],
            "best_val": best_vals[winner],
            "nan_encountered": jnp.asarray(
                results["nan_encountered"][winner], dtype=bool
            ),
        }
        candidate_stats = {
            "n_finite_candidates": jnp.sum(finite).astype(jnp.int32),
            "best_candidate": winner.astype(jnp.int32),
        }
        return winner_res, candidate_stats

    #: Infinity-norm gradient tolerance below which a fit is declared
    #: converged.  The fit objective is the mean negative log-likelihood,
    #: whose gradient at a converged interior optimum sits at ~1e-6 for a
    #: healthy GARCH/ARMA fit (measured); 1e-3 leaves three orders of
    #: headroom while still flagging a stalled or non-stationary run.
    _CONVERGENCE_GTOL: ClassVar[float] = 1e-3

    def _compute_convergence_status(
        self,
        res: dict,
        objective: Callable,
        x_opt: Array,
        obj_args: tuple,
        maxiter: int,
        candidate_stats: Optional[dict] = None,
    ) -> dict:
        r"""Derive the D-09 convergence-status leaves from a solver result.

        Packs the plain-named (no-trailing-underscore) status leaves that
        every fitted time-series instance carries: ``converged`` (bool),
        ``grad_norm`` (infinity norm of the objective gradient at the
        returned best iterate ``x_opt``), ``n_iterations`` (the fixed Adam
        iteration budget the scan ran), ``nan_encountered`` (from the
        solver's freeze-carry flag), and the multi-start candidate stats
        ``n_finite_candidates`` / ``best_candidate``.

        ``converged`` is ``True`` iff the best-iterate gradient
        infinity-norm is below :attr:`_CONVERGENCE_GTOL` AND the solver did
        not hit a non-finite gradient region.  This is the honest
        first-order-stationarity test on the returned point (Plan 02 makes
        ``x_opt`` the argmin over the whole scan, so its gradient is the
        right quantity to threshold).

        The candidate-stats leaves come from ``candidate_stats`` when the
        caller ran a multi-start fit (:meth:`_multi_start_fit`):
        ``n_finite_candidates`` is the number of candidate starts whose
        best objective was finite and ``best_candidate`` is the index of
        the winning start.  When ``candidate_stats`` is ``None`` (a
        single-start caller) they fall back to ``1`` if the returned
        objective is finite else ``0``, and ``best_candidate = 0`` (the
        sole start).

        All returned values are JAX array leaves so a jitted fit populates
        them and they round-trip through the equinox PyTree machinery.

        Args:
            res: The :func:`copulax._src.optimize.projected_gradient`
                return dict (must carry ``"nan_encountered"`` and
                ``"val"``).
            objective: The fit objective closure the solver minimised —
                signature ``objective(x, *obj_args) -> scalar``.
            x_opt: The best iterate returned by the solver (``res["x"]``).
            obj_args: The positional extra-args tuple the objective takes
                after ``x`` (the same series / pre-sample state passed to
                the solver).
            maxiter: The Adam iteration budget the scan ran.
            candidate_stats: Optional dict from :meth:`_multi_start_fit`
                carrying ``n_finite_candidates`` / ``best_candidate`` for a
                multi-start fit; ``None`` for a single-start caller.

        Returns:
            Dict of the six status leaves, keyed by their (plain) field
            names.
        """
        grad = jax.grad(lambda x: objective(x, *obj_args))(x_opt)
        grad_norm = jnp.max(jnp.abs(grad))
        nan_encountered = jnp.asarray(res["nan_encountered"], dtype=bool)
        converged = jnp.logical_and(
            grad_norm < self._CONVERGENCE_GTOL,
            jnp.logical_not(nan_encountered),
        )
        if candidate_stats is not None:
            n_finite_candidates = jnp.asarray(
                candidate_stats["n_finite_candidates"], dtype=jnp.int32
            )
            best_candidate = jnp.asarray(
                candidate_stats["best_candidate"], dtype=jnp.int32
            )
        else:
            best_finite = jnp.isfinite(jnp.asarray(res["val"], dtype=float))
            n_finite_candidates = jnp.where(
                best_finite,
                jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            )
            best_candidate = jnp.asarray(0, dtype=jnp.int32)
        return {
            "converged": converged,
            "grad_norm": grad_norm,
            "n_iterations": jnp.asarray(int(maxiter), dtype=jnp.int32),
            "nan_encountered": nan_encountered,
            "n_finite_candidates": n_finite_candidates,
            "best_candidate": best_candidate,
        }

    def _deliver_fit_warnings(
        self, status: dict, series_variance: Array,
    ) -> None:
        r"""Fire the fit-diagnostics warnings via one ``jax.debug.callback``.

        Delivers, host-side, a :class:`ConvergenceWarning` when the fit did
        not converge and a :class:`DataScaleWarning` when the series
        variance is outside the well-conditioned range ``[0.1, 10000.0)``
        (D-10).  A single callback carries both flags, so at most one host
        hop per fit.

        ``jax.debug.callback`` executes host-side under BOTH eager and
        ``jax.jit`` evaluation, and — crucially — does NOT fire while JAX
        is merely *tracing* (building the jaxpr).  Placing this at the fit
        tail (outside the per-iteration objective the solver differentiates)
        therefore fires exactly once when a fit is actually run, and never
        during the inner gradient evaluations or an outer trace.

        Args:
            status: The convergence-status leaf dict from
                :meth:`_compute_convergence_status`.
            series_variance: Sample variance of the input series, used for
                the data-scale check.
        """
        from copulax._src.timeseries._warnings import (
            DATA_SCALE_LOWER,
            DATA_SCALE_UPPER,
            data_scale_hint,
        )

        converged = status["converged"]
        grad_norm = status["grad_norm"]
        nan_encountered = status["nan_encountered"]
        out_of_scale = jnp.logical_or(
            series_variance < DATA_SCALE_LOWER,
            series_variance >= DATA_SCALE_UPPER,
        )

        def _emit(conv, gnorm, nan_enc, oos, var):
            import warnings

            from copulax._src.timeseries._warnings import (
                ConvergenceWarning,
                DataScaleWarning,
            )

            if not bool(conv):
                reason = (
                    "hit a non-finite gradient region"
                    if bool(nan_enc)
                    else f"gradient norm {float(gnorm):.3g} exceeds tolerance"
                )
                warnings.warn(
                    f"Fit did not converge ({reason}). Try more iterations, "
                    f"a different init, or rescaling the series with "
                    f"copulax.timeseries.DataScaler.",
                    ConvergenceWarning,
                    stacklevel=2,
                )
            if bool(oos):
                warnings.warn(
                    data_scale_hint(float(var)), DataScaleWarning, stacklevel=2,
                )

        jax.debug.callback(
            _emit, converged, grad_norm, nan_encountered,
            out_of_scale, series_variance,
        )

    @staticmethod
    def _raw_ll_sum(
        wrapper: Any, z: Array, log_sigma: Array, residual_params: dict,
    ) -> Array:
        r"""Raw NaN-propagating conditional log-likelihood sum (WR-05).

        Computes ``Σ_t [log f_z(z_t) - log σ_t]`` with **no** finite-masking,
        so a degenerate fit (any non-finite term) propagates ``NaN`` into
        the sum.  This is the honest reported log-likelihood — identical in
        form to every family's ``_log_likelihood_on_series`` — and must be
        packed at the fit tail instead of the penalised optimiser objective
        (which floors non-finite contributions and would report a large
        finite value like ``-2e9`` for a degenerate fit, making AIC/BIC look
        plausible-but-wrong).

        Args:
            wrapper: The fit's :class:`StandardisedResidual` wrapper.
            z: Standardised training residuals ``ε_t / σ_t``.
            log_sigma: ``log σ_t`` over the training window (the same σ the
                recursion produced; callers pass ``jnp.log`` of their σ path,
                or the log-σ path directly for log-variance families).
            residual_params: The fitted residual-law shape parameters.

        Returns:
            The scalar raw log-likelihood sum (NaN if any term is
            non-finite).
        """
        logpdf = wrapper.logpdf(z, residual_params) - log_sigma
        return jnp.sum(logpdf)

    def _render_convergence_line(self) -> Optional[str]:
        r"""Build the ``summary()`` convergence footer line from this
        instance's D-09 status leaves.

        Returns ``None`` when the model carries no convergence status
        (e.g. reconstructed without the status leaves) so ``summary()``
        omits the line.  Delegates the formatting to
        :func:`copulax._src.timeseries._summary.convergence_line`.
        """
        from copulax._src.timeseries._summary import convergence_line

        if self.converged is None:
            return None
        return convergence_line(
            converged=bool(self.converged),
            grad_norm=(
                None if self.grad_norm is None else float(self.grad_norm)
            ),
            n_iterations=(
                None if self.n_iterations is None
                else int(self.n_iterations)
            ),
            nan_encountered=(
                None if self.nan_encountered is None
                else bool(self.nan_encountered)
            ),
        )

    @staticmethod
    def _validate_orders(
        p: Optional[int],
        q: Optional[int],
        *,
        require_p: bool = True,
        require_q: bool = True,
        min_p: int = 0,
        min_q: int = 0,
    ) -> tuple[int, int]:
        r"""Coerce and validate the static ``(p, q)`` order pair.

        ``p`` and ``q`` are static fields on every model — they
        parameterise the compiled recursion graph and cannot change at
        runtime.  This helper centralises the type / sign checks so
        each family fit signature reads the same.

        Args:
            p: Mean / variance / asymmetry order.  Coerced to ``int``
                and checked against ``min_p``.
            q: MA / β-order.  Coerced to ``int`` and checked against
                ``min_q``.
            require_p: If ``True``, ``p`` must not be ``None``.
            require_q: If ``True``, ``q`` must not be ``None``.
            min_p: Minimum admissible value of ``p``.
            min_q: Minimum admissible value of ``q``.

        Returns:
            Validated ``(p, q)`` as a tuple of plain Python ``int``.
            ``None`` is mapped to ``0`` when the corresponding
            ``require_*`` flag is ``False``.

        Raises:
            ValueError: when an order is missing under
                ``require_*=True`` or below the minimum.
            TypeError: when an order is not an integer.
        """
        if require_p and p is None:
            raise ValueError("Order `p` must be provided.")
        if require_q and q is None:
            raise ValueError("Order `q` must be provided.")
        for label, value, lo in (("p", p, min_p), ("q", q, min_q)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(
                value, (int, jnp.integer)
            ):
                raise TypeError(
                    f"Order `{label}` must be an integer, got "
                    f"{type(value).__name__}."
                )
            if int(value) < lo:
                raise ValueError(
                    f"Order `{label}` must be >= {lo}, got {int(value)}."
                )
        return (
            int(p) if p is not None else 0,
            int(q) if q is not None else 0,
        )

    @staticmethod
    def _validate_backcast_length(
        backcast_length: Optional[int], n: int
    ) -> int:
        r"""Resolve the ``backcast_length`` kwarg for fit / residuals.

        Default ``None`` means use the entire series.  When set
        explicitly the caller must satisfy ``0 < backcast_length <= n``.

        Args:
            backcast_length: User-supplied value or ``None``.
            n: Length of the input series.

        Returns:
            Plain Python ``int`` with the resolved length.

        Raises:
            ValueError / TypeError: per the contract documented in the
                fit signatures.
        """
        if backcast_length is None:
            return int(n)
        if isinstance(backcast_length, bool) or not isinstance(
            backcast_length, (int, jnp.integer)
        ):
            raise TypeError(
                "backcast_length must be an integer or None, got "
                f"{type(backcast_length).__name__}."
            )
        if not (0 < int(backcast_length) <= int(n)):
            raise ValueError(
                f"Require 0 < backcast_length <= len(y); got "
                f"backcast_length={int(backcast_length)}, len(y)={int(n)}."
            )
        return int(backcast_length)

    # ------------------------------------------------------------------
    # Fitted-instance construction & serialisation
    # ------------------------------------------------------------------
    def _fitted_instance(
        self,
        params_dict: dict,
        name: Optional[str] = None,
        **extra: Any,
    ) -> "TimeSeriesModel":
        r"""Construct a new fitted instance carrying ``params_dict``.

        Mirrors :meth:`Distribution._fitted_instance` but allows the
        caller to attach extra static / traced fields (terminal state,
        fit-time diagnostics, observed-information matrix) that the
        base class does not predeclare.  Each family subclass receives
        its own additional kwargs through ``**extra``.

        Args:
            params_dict: Fitted parameter values, in the canonical
                family-specific schema.
            name: Optional custom name for the fitted instance.  When
                ``None`` an auto-generated name including the dict
                identity is used.
            **extra: Additional fields to forward to the subclass
                constructor (terminal state, diagnostics, etc.).

        Returns:
            A new instance of ``type(self)`` with the supplied
            parameters and extras.
        """
        cls = type(self)
        if name is None:
            name = f"Fitted{cls.__name__}-{id(params_dict):x}"
        return cls(name=name, **params_dict, **extra)

    def save(self, path: str) -> None:
        r"""Save the fitted model to a ``.cpx`` file via the shared
        :mod:`copulax._src._serialization` machinery."""
        from copulax._src._serialization import _save_distribution
        _save_distribution(self, path)

    # ------------------------------------------------------------------
    # Serialisation hooks (overridden by subclasses where needed)
    # ------------------------------------------------------------------
    def _serialise_static(self) -> dict:
        r"""Per-class static-config metadata.

        Default implementation handles families whose static config
        is the ``(p, q, residual_dist)`` triple — i.e. all mean and
        variance models in this subpackage.  :class:`ArmaGarch`
        overrides to record ``mean_order`` / ``var_order`` /
        ``var_model`` instead.
        """
        return {
            "p": int(getattr(self, "p", 0)),
            "q": int(getattr(self, "q", 0)),
            "residual_dist_class": type(self.residual_dist).__name__,
        }

    def _serialise_traced(
        self,
    ) -> tuple[dict, dict[str, Any]]:
        r"""Per-class traced-field metadata + arrays.

        Default returns:
        * ``params_to_flat`` of the ``params`` dict (under key
          ``params_flat`` in arrays + ``params_schema`` in metadata),
        * terminal-state leaves (under ``ts_<i>`` keys) with the
          ``ts_class`` name in metadata,
        * the ``n_train_`` sample-size scalar (under ``diag_n_train_``
          in arrays),
        * optional ``cov_matrix_`` and ``standard_errors_`` (using
          the same flat schema as params),
        * optional ``residual_diagnostics_`` bundle — the
          ``"acf"`` / ``"pacf"`` arrays go to ``diag_acf`` /
          ``diag_pacf`` in arrays; everything else (the
          ``loglikelihood`` / ``aic`` / ``bic`` scalars and the five
          hypothesis-test result dicts) is JSON-encodable and goes
          under ``metadata["residual_diagnostics"]``.

        Subclasses can override if they need to serialise additional
        traced fields (e.g. variant-specific carry layouts).
        """
        import jax
        import numpy as np
        from copulax._src.timeseries._se import params_to_flat

        metadata: dict = {}
        arrays: dict[str, Any] = {}

        if self.params is not None:
            flat, schema = params_to_flat(self.params)
            arrays["params_flat"] = np.asarray(flat)
            metadata["params_schema"] = [[k, list(s)] for k, s in schema]

        ts = getattr(self, "terminal_state", None)
        if ts is not None:
            leaves = jax.tree_util.tree_leaves(ts)
            for i, leaf in enumerate(leaves):
                arrays[f"ts_{i}"] = np.asarray(leaf)
            metadata["ts_n_leaves"] = len(leaves)
            metadata["ts_class"] = type(ts).__name__

        n_train = getattr(self, "n_train_", None)
        if n_train is not None:
            arrays["diag_n_train_"] = np.asarray(n_train)

        if (
            hasattr(self, "cov_matrix_")
            and self.cov_matrix_ is not None
        ):
            arrays["cov_matrix_"] = np.asarray(self.cov_matrix_)

        if (
            hasattr(self, "standard_errors_")
            and self.standard_errors_ is not None
        ):
            flat_se, se_schema = params_to_flat(self.standard_errors_)
            arrays["se_flat"] = np.asarray(flat_se)
            metadata["se_schema"] = [
                [k, list(s)] for k, s in se_schema
            ]

        diag = getattr(self, "residual_diagnostics_", None)
        if diag is not None:
            _serialise_residual_diagnostics(diag, arrays, metadata)

        return metadata, arrays

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(self, y: ArrayLike, *args, **kwargs) -> "TimeSeriesModel":
        r"""Fit the model to the input series and return a new fitted
        instance.

        Subclasses must accept ``y`` as the first positional argument.
        The base class is immutable — fitting never mutates ``self``.
        """

    @abstractmethod
    def conditional_mean(self, y: ArrayLike) -> Array:
        r"""One-step-ahead conditional mean trajectory ``μ_t`` over
        ``y``."""

    @abstractmethod
    def conditional_variance(self, y: ArrayLike) -> Array:
        r"""One-step-ahead conditional variance trajectory ``σ²_t``
        over ``y``.

        Mean models that do not parameterise heteroskedasticity may
        return a constant trajectory equal to the unconditional
        residual variance under the chosen standardised residual law.
        """

    @abstractmethod
    def residuals(self, y: ArrayLike, *args, **kwargs):
        r"""Residuals from running the recursion forward over ``y``.

        Mean models return innovation residuals ``ε_t = y_t − μ_t``.
        Variance models return the pair ``(ε_t, z_t)`` with
        ``z_t = ε_t / σ_t``.  The joint composite returns a
        :class:`namedtuple` exposing both halves.
        """

    @abstractmethod
    def stats(self, *args, **kwargs) -> dict:
        r"""Analytic, parameter-only statistics — no data required.

        Concrete subclasses return at minimum the unconditional mean
        and variance under the fitted parameters; variance models
        additionally expose persistence, half-life, and a stationarity
        flag.  Distinct from the data-dependent diagnostics
        (``loglikelihood``, ``aic``, ``bic``, etc.) which take a
        series at call-time.
        """

    @abstractmethod
    def forecast(self, h: int, *args, **kwargs):
        r"""``h``-step-ahead conditional moments rolled forward from
        the stored terminal state (or an explicit ``last_state``).

        Returns a ``ForecastResult`` PyTree carrying ``mean``,
        ``variance``, and (for ``method='simulation'``) simulated
        ``paths``.
        """

    @abstractmethod
    def rvs(self, *args, **kwargs) -> Array:
        r"""Simulate synthetic series from the fitted model.

        See the family-specific implementations for the precise
        signature; the canonical form is
        ``rvs(size=None, *, key=None, u=None, last_state=None)``.
        """


###############################################################################
# Family intermediates
###############################################################################
class MeanModel(TimeSeriesModel):
    r"""Abstract intermediate for ARMA-style mean-equation models.

    Concrete subclasses (``AR``, ``MA``, ``ARMA``) parameterise the
    conditional mean ``μ_t = E[y_t | y_{<t}]`` via an
    autoregressive / moving-average recursion driven by the chosen
    standardised residual law.  The conditional variance under a
    pure mean model is the (constant) residual-distribution variance —
    pair with a :class:`VarianceModel` (or use the joint
    ``arma_garch`` composite) when heteroskedasticity matters.
    """


class VarianceModel(TimeSeriesModel):
    r"""Abstract intermediate for GARCH-family conditional-variance
    models.

    Operates on a mean-corrected innovation series ``ε_t``.  The
    conditional mean of the level series is zero by assumption — to
    fit a non-zero mean alongside the variance, either run an
    :class:`ARMA` mean model first and feed its residuals in, or use
    the joint ``arma_garch`` composite which estimates both stages
    under a single MLE objective.
    """
