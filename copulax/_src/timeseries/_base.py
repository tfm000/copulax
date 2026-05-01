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
from typing import Any, ClassVar, Optional

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import _params_equal
from copulax._src.univariate._utils import _univariate_input


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
        r"""Save the fitted model to a ``.cpx`` file.

        Routed through :func:`copulax._src._serialization._save_distribution`
        once the ``"timeseries"`` ``dist_family`` branch is registered
        there.  Until that wiring lands, calling :meth:`save` raises a
        clear ``ValueError`` from the serialiser — no silent failure.
        """
        from copulax._src._serialization import _save_distribution
        _save_distribution(self, path)

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
