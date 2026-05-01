"""Standardised-residual wrapper used by every time-series fit objective.

Wraps an allowed univariate distribution (see :mod:`_registry`) so
that the time-series optimiser sees only its **shape** parameters as
free fitting variables: ``μ`` is fixed to 0 and any scale-like
parameter (``σ``, ``δ``, ``α``) is set so the resulting distribution
has unit variance.  The full ``(mean = 0, var = 1)`` parameter dict
is reconstructed at every objective evaluation by composing the
shape-only inputs with the per-distribution
:meth:`_standardise_params` classmethod.

This is the only way to make GARCH ``ω`` and the residual scale
jointly identifiable — without unit-variance residuals the two would
trade off freely along the model likelihood.

The wrapper is a plain Python object (not an :mod:`equinox` module)
because:

* the only fields are a singleton :class:`Univariate` and a static
  tuple of strings — neither is a JAX-traced leaf;
* the fit-objective closure captures the wrapper by Python reference,
  hashes by identity (consistent across rolling-window calls), and
  recompiles automatically when the chosen residual law changes
  (the ``residual_dist`` is a static argument of the fit graph).

Public API:

* :meth:`logpdf`, :meth:`pdf`, :meth:`cdf`, :meth:`ppf`,
  :meth:`rvs` / :meth:`sample`, :meth:`stats` — same signatures as
  the corresponding :class:`Univariate` methods, but consume
  ``shape_params`` (a dict of *only* the free shape keys) instead of
  the full parameter dict.
* :meth:`shape_params_to_array` / :meth:`shape_params_from_array` —
  flat-array bridge for the optimiser.
* :meth:`to_distribution` — produce a fitted
  :class:`Univariate` instance from the post-fit shape-params dict;
  this is what the fitted time-series model exposes via its
  ``residual_distribution`` attribute.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src.timeseries._residuals._registry import (
    _ALLOWED_RESIDUAL_DISTS,
    _RESIDUAL_DEFAULT_SHAPE_PARAMS,
    _RESIDUAL_SHAPE_KEYS,
)


class StandardisedResidual:
    r"""Standardised wrapper around an allowed univariate distribution.

    Exposes only the distribution's shape parameters as free fitting
    variables.  ``μ`` and ``σ`` (and any other location / scale
    parameters) are derived inside the wrapped distribution's
    :meth:`_standardise_params` classmethod so the resulting
    distribution has zero mean and unit variance — the contract the
    GARCH-family fit needs for ``ω`` and the residual scale to be
    jointly identifiable.

    Args:
        base_dist: A :class:`Univariate` singleton from
            :data:`_ALLOWED_RESIDUAL_DISTS`.

    Raises:
        ValueError: If ``base_dist`` is not on the residual whitelist.

    Examples:
        >>> from copulax.univariate import skewed_t
        >>> from copulax._src.timeseries._residuals._standardise import (
        ...     StandardisedResidual
        ... )
        >>> wrapper = StandardisedResidual(skewed_t)
        >>> shape = {"nu": 5.0, "gamma": 0.3}
        >>> float(wrapper.logpdf(0.0, shape).reshape(()))  # mean=0
        ... # doctest: +SKIP
    """

    __slots__ = ("base_dist", "shape_keys")

    def __init__(self, base_dist: Univariate):
        cls = type(base_dist)
        if cls not in _RESIDUAL_SHAPE_KEYS:
            allowed = ", ".join(d.name for d in _ALLOWED_RESIDUAL_DISTS)
            raise ValueError(
                f"Distribution {base_dist.name!r} is not on the residual "
                f"whitelist for copulax.timeseries. Allowed: {allowed}. "
                "Add a new entry to copulax/_src/timeseries/_residuals/"
                "_registry.py to extend the whitelist."
            )
        # Store the canonical singleton so identity comparisons match
        # whether the user passes the singleton or a freshly constructed
        # equivalent instance.
        self.base_dist: Univariate = base_dist
        self.shape_keys: tuple[str, ...] = _RESIDUAL_SHAPE_KEYS[cls]

    # ------------------------------------------------------------------
    # Identity / introspection
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Display name of the wrapped distribution."""
        return self.base_dist.name

    @property
    def n_shape_params(self) -> int:
        """Number of free shape parameters under the standardised form."""
        return len(self.shape_keys)

    def __repr__(self) -> str:
        return f"StandardisedResidual({self.base_dist.name})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StandardisedResidual):
            return NotImplemented
        return type(self.base_dist) is type(other.base_dist)

    def __hash__(self) -> int:
        # Identity-style hash on the wrapped class — consistent with
        # equinox / JAX expectations for static fields.
        return hash(type(self.base_dist))

    # ------------------------------------------------------------------
    # Parameter dict construction
    # ------------------------------------------------------------------
    def example_shape_params(self) -> dict:
        r"""A sensible default shape-only parameter dict.

        Returns the per-class entry from
        :data:`_RESIDUAL_DEFAULT_SHAPE_PARAMS` cast to JAX scalars.
        These defaults are moment-matched feasible: the resulting
        standardised distribution always has finite mean, variance,
        and density at zero (the per-distribution
        :meth:`example_params` are *not* used here because they are
        tuned to be illustrative for the unstandardised density and
        do not all satisfy the standardised-form feasibility bounds —
        most prominently, ``Skewed-T``'s ``example_params`` violates
        :math:`\gamma^2 < 1 / \mathrm{Var}[W]` at any practical
        :math:`\nu`).
        """
        defaults = _RESIDUAL_DEFAULT_SHAPE_PARAMS[type(self.base_dist)]
        return {k: jnp.asarray(v, dtype=float) for k, v in defaults.items()}

    def _full_params(self, shape_params: dict) -> dict:
        r"""Compose the user's shape-only dict into a full
        ``(mean = 0, var = 1)`` parameter dict.

        Strategy: start from the wrapped distribution's
        ``example_params()`` (which already has every required key
        with concrete defaults), overlay the user's shape values, then
        run the result through :meth:`_standardise_params` so ``μ`` and
        ``σ`` are derived from the shape inputs.

        This composition is JIT-friendly: the ``example_params``
        defaults are concrete Python floats that JAX casts on first
        trace, the overlaid shape values flow as traced leaves, and
        ``_standardise_params`` is a pure JAX op.

        Raises:
            KeyError: If any expected shape key is missing.
        """
        cls = type(self.base_dist)
        # Start from the wrapped distribution's defaults — provides
        # concrete placeholders for any key that ``_standardise_params``
        # ignores.  These are overwritten next.
        template = dict(self.base_dist.example_params())
        for key in self.shape_keys:
            if key not in shape_params:
                raise KeyError(
                    f"Missing shape parameter {key!r} for "
                    f"StandardisedResidual({self.base_dist.name}); "
                    f"required keys: {self.shape_keys}."
                )
            template[key] = shape_params[key]
        return cls._standardise_params(template)

    # ------------------------------------------------------------------
    # Distribution methods (shape-only signatures)
    # ------------------------------------------------------------------
    def logpdf(self, x: ArrayLike, shape_params: dict) -> Array:
        r"""Log-density of the standardised distribution at ``x``."""
        return self.base_dist.logpdf(x=x, params=self._full_params(shape_params))

    def pdf(self, x: ArrayLike, shape_params: dict) -> Array:
        r"""Density of the standardised distribution at ``x``."""
        return self.base_dist.pdf(x=x, params=self._full_params(shape_params))

    def cdf(self, x: ArrayLike, shape_params: dict) -> Array:
        r"""CDF of the standardised distribution at ``x``."""
        return self.base_dist.cdf(x=x, params=self._full_params(shape_params))

    def ppf(self, q: ArrayLike, shape_params: dict, **kwargs) -> Array:
        r"""Inverse CDF of the standardised distribution at ``q``.

        Extra keyword arguments are forwarded to
        :meth:`Univariate.ppf` (e.g. ``brent``, ``nodes``,
        ``maxiter`` — see that method's docstring for details).
        """
        return self.base_dist.ppf(
            q=q, params=self._full_params(shape_params), **kwargs
        )

    def rvs(
        self,
        size,
        shape_params: dict,
        key: Optional[Array] = None,
    ) -> Array:
        r"""Generate samples from the standardised distribution."""
        return self.base_dist.rvs(
            size=size, params=self._full_params(shape_params), key=key
        )

    sample = rvs

    def stats(self, shape_params: dict) -> dict:
        r"""Analytic statistics of the standardised distribution.

        For correctly-specified shape parameters the returned dict
        carries ``mean ≈ 0`` and ``variance ≈ 1`` to machine precision
        — exposed primarily for diagnostic / regression-test purposes.
        """
        return self.base_dist.stats(params=self._full_params(shape_params))

    # ------------------------------------------------------------------
    # Optimiser bridge: shape-params dict ↔ flat array
    # ------------------------------------------------------------------
    def shape_params_to_array(self, shape_params: dict) -> Array:
        r"""Pack a shape-params dict into a flat 1D array.

        Order matches :data:`_RESIDUAL_SHAPE_KEYS[cls]`.  Used by the
        time-series fit objective to compose the residual half of the
        unconstrained parameter vector.

        Raises:
            KeyError: When any expected shape key is missing.
        """
        for key in self.shape_keys:
            if key not in shape_params:
                raise KeyError(
                    f"Missing shape parameter {key!r}; required keys: "
                    f"{self.shape_keys}."
                )
        if len(self.shape_keys) == 0:
            return jnp.zeros((0,), dtype=float)
        values = [jnp.asarray(shape_params[k], dtype=float).reshape(()) for k in self.shape_keys]
        return jnp.stack(values)

    def shape_params_from_array(self, arr: ArrayLike) -> dict:
        r"""Inverse of :meth:`shape_params_to_array`.

        Element ``i`` of ``arr`` is mapped to key
        ``shape_keys[i]``.  No reparameterisation is applied here —
        the caller is responsible for already having mapped the
        unconstrained optimiser state through any
        bound-enforcing reparam (``softplus`` for ``ν``, ``tanh`` for
        ``γ`` feasibility, etc.).
        """
        arr = jnp.asarray(arr, dtype=float).reshape(-1)
        if arr.shape[0] != len(self.shape_keys):
            raise ValueError(
                f"Expected flat array of length {len(self.shape_keys)} "
                f"matching shape_keys={self.shape_keys}, got length "
                f"{int(arr.shape[0])}."
            )
        return {k: arr[i] for i, k in enumerate(self.shape_keys)}

    # ------------------------------------------------------------------
    # Post-fit finaliser
    # ------------------------------------------------------------------
    def to_distribution(
        self, shape_params: dict, name: Optional[str] = None,
    ) -> Univariate:
        r"""Build a fitted :class:`Univariate` instance from the
        post-fit shape parameters.

        This is what the fitted time-series model exposes as its
        ``residual_distribution`` field — a regular CopulAX univariate
        distribution with ``mean = 0``, ``var = 1``, ready for
        downstream ``.sample``, ``.logpdf``, ``.cdf``, etc. calls
        without going through the wrapper.

        Args:
            shape_params: Shape-only parameter dict produced by the
                fit (or ``shape_params_from_array(arr)``).
            name: Optional custom name for the fitted instance; passed
                through to :meth:`Univariate._fitted_instance`.

        Returns:
            A fitted :class:`Univariate` instance.
        """
        full = self._full_params(shape_params)
        return self.base_dist._fitted_instance(full, name=name)


__all__ = ["StandardisedResidual"]
