"""JAX-native, jittable, autograd-compatible data scaler.

This module provides :class:`DataScaler`, an :mod:`equinox`-based PyTree
object that fits an affine rescaling to input data and exposes
``transform`` / ``inverse_transform`` for applying and undoing the
rescaling on later observations.

Four scaling methods are supported — all reducing to a uniform
``z = (x - offset) / scale`` representation:

* ``"zscore"`` (default): centre at the mean, scale by the standard deviation.
* ``"minmax"``: shift so the minimum is zero, scale so the range is one.
* ``"robust"``: centre at the median, scale by the inter-quantile range
  (default 25/75).
* ``"maxabs"``: no centring, scale by the element-wise absolute maximum.

The class is a proper :class:`equinox.Module`, so fitted instances compose
cleanly with ``jax.jit``, ``jax.grad``, ``jax.vmap``, and ``equinox`` PyTree
utilities.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


_FnPair = Optional[Tuple[Optional[Callable], Optional[Callable]]]
_VALID_METHODS = frozenset({"zscore", "minmax", "robust", "maxabs"})


class DataScaler(eqx.Module):
    r"""Jittable, autograd-compatible data scaler.

    Fits an affine rescaling of the form :math:`z = (x - \text{offset}) /
    \text{scale}` to input data under one of four methods, then applies
    the same rescaling (or its inverse) to later observations.

    All scaling statistics are reduced over axis 0 (the sample axis).
    Any trailing axes are treated as feature dimensions and are preserved
    in the fitted ``offset`` / ``scale`` arrays. Transform and inverse-
    transform operations broadcast naturally over any leading batch shape
    as long as the trailing feature dims match.

    Four methods are supported:

    * ``"zscore"``: ``offset = mean(x, axis=0)``, ``scale = std(x, axis=0)``.
    * ``"minmax"``: ``offset = min(x, axis=0)``, ``scale = max - min``.
    * ``"robust"``: ``offset = median(x, axis=0)``, ``scale = q_high - q_low``.
    * ``"maxabs"``: ``offset = 0``, ``scale = max(|x|, axis=0)``.

    Zero-variance features (a fitted ``scale`` of ``0``) are silently
    clamped to ``1.0`` so division does not break autograd or produce
    NaNs. Optional ``offset_only`` / ``scale_only`` flags restrict
    fitting to centring-only or rescaling-only behaviour. Optional
    ``pre_fns`` / ``post_fns`` tuples attach JAX-compliant forward and
    inverse functions to the pipeline (for example, z-scoring over
    log-transformed data).

    Args:
        method: One of ``"zscore"`` (default), ``"minmax"``, ``"robust"``,
            or ``"maxabs"``.
        q_low: Lower quantile for the ``"robust"`` method. Must satisfy
            ``0 < q_low < q_high < 1``. Defaults to ``0.25``.
        q_high: Upper quantile for the ``"robust"`` method. Defaults to
            ``0.75``.
        offset_only: If ``True``, the fitted ``scale`` is forced to ``1``
            so ``transform`` performs centring only. Mutually exclusive
            with ``scale_only``. Defaults to ``False``.
        scale_only: If ``True``, the fitted ``offset`` is forced to ``0``
            so ``transform`` performs rescaling only. Mutually exclusive
            with ``offset_only``. Defaults to ``False``.
        pre_fns: Optional ``(forward, inverse)`` tuple of JAX-compliant
            functions applied to the data *before* the affine scaling.
            The forward function runs during both :meth:`fit` and
            :meth:`transform`; the inverse runs at the end of
            :meth:`inverse_transform`. Either element may be ``None`` to
            skip that direction. Defaults to ``None``.
        post_fns: Optional ``(forward, inverse)`` tuple applied *after*
            the affine scaling during :meth:`transform` and inverted
            first in :meth:`inverse_transform`. ``post_fns`` is **not**
            applied during :meth:`fit`. Same ``None``-skip semantics as
            ``pre_fns``. Defaults to ``None``.
        offset: Pre-fitted offset array. Normally populated by
            :meth:`fit` rather than passed directly.
        scale: Pre-fitted scale array. Normally populated by :meth:`fit`
            rather than passed directly.

    Attributes:
        offset: Fitted offset, shape ``x.shape[1:]``. ``None`` until fit.
        scale: Fitted scale, shape ``x.shape[1:]``. ``None`` until fit.
        is_fitted: Whether both ``offset`` and ``scale`` are populated.

    Notes:
        ``method``, ``q_low``, ``q_high``, ``offset_only``,
        ``scale_only``, ``pre_fns``, and ``post_fns`` are static PyTree
        fields (``eqx.field(static=True)``). Only ``offset`` and
        ``scale`` are traced leaves. Branching on ``method`` is
        therefore safe under ``jit`` — JIT specialises per method.

    Example:
        >>> import jax.numpy as jnp
        >>> from copulax.preprocessing import DataScaler
        >>> x = jnp.asarray([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]])
        >>> scaler, z = DataScaler("zscore").fit_transform(x)
        >>> bool(jnp.allclose(z.mean(axis=0), 0.0, atol=1e-6))
        True
        >>> bool(jnp.allclose(scaler.inverse_transform(z), x, atol=1e-6))
        True
    """

    method: str = eqx.field(static=True)
    q_low: float = eqx.field(static=True)
    q_high: float = eqx.field(static=True)
    offset_only: bool = eqx.field(static=True)
    scale_only: bool = eqx.field(static=True)
    pre_fns: _FnPair = eqx.field(static=True)
    post_fns: _FnPair = eqx.field(static=True)
    offset: Optional[Array]
    scale: Optional[Array]

    def __init__(
        self,
        method: str = "zscore",
        *,
        q_low: float = 0.25,
        q_high: float = 0.75,
        offset_only: bool = False,
        scale_only: bool = False,
        pre_fns: _FnPair = None,
        post_fns: _FnPair = None,
        offset: Optional[ArrayLike] = None,
        scale: Optional[ArrayLike] = None,
    ):
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown method: {method!r}. "
                f"Expected one of {sorted(_VALID_METHODS)}."
            )
        if not (0.0 < q_low < q_high < 1.0):
            raise ValueError(
                f"Require 0 < q_low < q_high < 1; got q_low={q_low}, q_high={q_high}."
            )
        if offset_only and scale_only:
            raise ValueError(
                "offset_only and scale_only are mutually exclusive "
                "(together they specify the identity transform)."
            )
        for name, fns in (("pre_fns", pre_fns), ("post_fns", post_fns)):
            if fns is not None:
                if not (isinstance(fns, tuple) and len(fns) == 2):
                    raise ValueError(
                        f"{name} must be a (forward, inverse) tuple of length 2; "
                        "use None for missing halves."
                    )
                for half in fns:
                    if half is not None and not callable(half):
                        raise TypeError(
                            f"{name} entries must be callable or None; "
                            f"got {type(half).__name__}."
                        )

        self.method = method
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.offset_only = bool(offset_only)
        self.scale_only = bool(scale_only)
        self.pre_fns = pre_fns
        self.post_fns = post_fns
        self.offset = None if offset is None else jnp.asarray(offset, dtype=float)
        self.scale = None if scale is None else jnp.asarray(scale, dtype=float)

    @property
    def is_fitted(self) -> bool:
        """Whether ``offset`` and ``scale`` have both been populated."""
        return self.offset is not None and self.scale is not None

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"DataScaler(method={self.method!r}, {status})"

    @staticmethod
    def _safe_scale(scale: Array) -> Array:
        """Replace exact zeros in ``scale`` with ``1`` to avoid division-by-zero."""
        return jnp.where(scale == 0, jnp.ones_like(scale), scale)

    @staticmethod
    def _apply(fns: _FnPair, idx: int, x: Array) -> Array:
        """Apply ``fns[idx]`` to ``x``, or pass ``x`` through if missing."""
        if fns is None or fns[idx] is None:
            return x
        return fns[idx](x)

    def _rebuild(self, *, offset: Array, scale: Array) -> "DataScaler":
        """Construct a new instance preserving all static configuration."""
        return DataScaler(
            method=self.method,
            q_low=self.q_low,
            q_high=self.q_high,
            offset_only=self.offset_only,
            scale_only=self.scale_only,
            pre_fns=self.pre_fns,
            post_fns=self.post_fns,
            offset=offset,
            scale=scale,
        )

    def _compute_offset_scale(self, x: Array) -> Tuple[Array, Array]:
        """Compute the method-specific ``(offset, scale)`` from ``x``.

        Assumes ``x`` is already a JAX array with ``pre_fns`` forward
        applied (if any). Applies the method formula, the zero-variance
        safeguard, and the ``offset_only`` / ``scale_only`` overrides.
        """
        if self.method == "zscore":
            offset = x.mean(axis=0)
            scale = x.std(axis=0)
        elif self.method == "minmax":
            lo = x.min(axis=0)
            hi = x.max(axis=0)
            offset = lo
            scale = hi - lo
        elif self.method == "robust":
            # Combine median + low/high quantiles into a single quantile
            # call to avoid three separate reductions over the same array.
            qs = jnp.quantile(
                x,
                jnp.asarray([self.q_low, 0.5, self.q_high], dtype=x.dtype),
                axis=0,
            )
            offset = qs[1]
            scale = qs[2] - qs[0]
        else:  # "maxabs" — other values ruled out by __init__ validation
            offset = jnp.zeros(x.shape[1:], dtype=x.dtype)
            scale = jnp.abs(x).max(axis=0)

        scale = self._safe_scale(scale)
        if self.offset_only:
            scale = jnp.ones_like(scale)
        if self.scale_only:
            offset = jnp.zeros_like(offset)
        return offset, scale

    def fit(self, x: ArrayLike) -> "DataScaler":
        """Fit the scaler to ``x`` and return a new fitted instance.

        Args:
            x: Input data of shape ``(n, *feature_dims)``. Axis 0 is the
                sample axis; all remaining axes are feature dims.

        Returns:
            A new :class:`DataScaler` instance with ``offset`` and
            ``scale`` populated. The original instance is unchanged
            (pure functional).
        """
        x_arr = self._apply(self.pre_fns, 0, jnp.asarray(x, dtype=float))
        offset, scale = self._compute_offset_scale(x_arr)
        return self._rebuild(offset=offset, scale=scale)

    def transform(self, x: ArrayLike) -> Array:
        """Apply the fitted scaling to ``x``.

        The pipeline is ``post_forward((pre_forward(x) - offset) / scale)``;
        missing halves of ``pre_fns`` / ``post_fns`` are no-ops.

        Args:
            x: Data to scale. Trailing dims must match ``offset`` / ``scale``.

        Returns:
            The scaled data, same shape as ``x``.

        Raises:
            ValueError: If the scaler has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError(
                "DataScaler is not fitted. Call .fit(x) or pass offset/scale "
                "to the constructor first."
            )
        x_arr = self._apply(self.pre_fns, 0, jnp.asarray(x, dtype=float))
        z = (x_arr - self.offset) / self.scale
        return self._apply(self.post_fns, 0, z)

    def inverse_transform(self, z: ArrayLike) -> Array:
        """Undo the fitted scaling on ``z``.

        The pipeline is ``pre_inverse(post_inverse(z) * scale + offset)``;
        missing halves of ``pre_fns`` / ``post_fns`` are silently skipped
        — the caller is responsible for providing inverses when full
        round-trip fidelity is required.

        Args:
            z: Previously scaled data. Trailing dims must match
                ``offset`` / ``scale``.

        Returns:
            The unscaled data, same shape as ``z``.

        Raises:
            ValueError: If the scaler has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError(
                "DataScaler is not fitted. Call .fit(x) or pass offset/scale "
                "to the constructor first."
            )
        z_arr = self._apply(self.post_fns, 1, jnp.asarray(z, dtype=float))
        x = z_arr * self.scale + self.offset
        return self._apply(self.pre_fns, 1, x)

    def fit_transform(self, x: ArrayLike) -> Tuple["DataScaler", Array]:
        """Fit the scaler to ``x`` and return ``(fitted_scaler, scaled_x)``.

        Equivalent to ``fitted = self.fit(x); return fitted, fitted.transform(x)``
        but applies ``pre_fns`` forward only once (``fit`` and
        ``transform`` would otherwise each apply it).

        Args:
            x: Input data of shape ``(n, *feature_dims)``.

        Returns:
            A tuple ``(fitted, scaled)`` where ``fitted`` is the fitted
            scaler and ``scaled`` is the transformed data.
        """
        x_arr = self._apply(self.pre_fns, 0, jnp.asarray(x, dtype=float))
        offset, scale = self._compute_offset_scale(x_arr)
        fitted = self._rebuild(offset=offset, scale=scale)
        z = (x_arr - offset) / scale
        z = self._apply(self.post_fns, 0, z)
        return fitted, z

    def save(self, path: str) -> None:
        """Save the fitted scaler to a ``.cpx`` file.

        The file can be loaded back with :func:`copulax.load`. The
        ``.cpx`` extension is appended automatically when missing.

        Any ``pre_fns`` / ``post_fns`` callables are serialised by their
        import path (``{module}.{qualname}``) so they can be rehydrated
        on load without ``pickle``. Lambdas and locally-defined
        closures cannot be serialised this way and will cause a
        :class:`ValueError` at save time — use a module-level function
        instead, or clear the callable(s) before saving.

        Args:
            path: Destination file path.

        Raises:
            ValueError: If the scaler has not been fitted, or any
                attached callable cannot be round-tripped by qualname.
        """
        from copulax._src._serialization import _save_scaler
        _save_scaler(self, path)
