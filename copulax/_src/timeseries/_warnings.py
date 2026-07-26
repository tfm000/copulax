"""Warning types for the time-series fit-diagnostics UX.

Defines :class:`ConvergenceWarning` and :class:`DataScaleWarning`,
emitted from a fitted model's fit tail (via ``jax.debug.callback``) to
signal, respectively, that the optimiser did not reach a stationary
point and that the input series is scaled far outside the range where
the GARCH/ARMA likelihood is numerically well-behaved.

Both subclass the built-in :class:`Warning` so existing ``except
Warning`` / ``warnings.catch_warnings`` sites keep catching them and
callers can filter on the concrete class.

These classes are internal until Phase 3.  The public re-export surface
(``copulax.exceptions`` / ``copulax.warnings``) is decided in Phase 3's
coordinated API break, mirroring the deferral of
:class:`copulax._src._params._exceptions.ParamsTypeError`; until then
they live under ``copulax._src.timeseries``.

The module-level :func:`data_scale_hint` builder centralises the
data-scale message text so the ``DataScaler`` pointer and the accepted
variance-ratio bounds are stated in exactly one place.
"""

from __future__ import annotations


# Variance-ratio bounds outside which a DataScaleWarning is raised.
# The ``arch`` package (Sheppard) warns when the sample variance of the
# series is outside ``[1e-1, 1e4)`` (arch.univariate.base — the
# ``initial_value`` scale check), because the GARCH log-likelihood loses
# conditioning far from unit scale.  We adopt the same bounds and, like
# arch, ship a diagnostic rather than auto-rescaling — the caller decides
# whether to rescale (e.g. via ``copulax.timeseries.DataScaler``).
DATA_SCALE_LOWER: float = 0.1
DATA_SCALE_UPPER: float = 10000.0


def data_scale_hint(variance: float) -> str:
    r"""Build the user-facing :class:`DataScaleWarning` message.

    States the offending sample variance, the accepted ``[lower, upper)``
    range, and points the caller at ``DataScaler`` for rescaling.  No
    rescaling is performed — the warning is purely diagnostic.

    Args:
        variance: The sample variance of the series that fell outside the
            accepted range.

    Returns:
        The formatted, user-facing warning message.
    """
    return (
        f"Series variance {variance:.3g} is outside the well-conditioned "
        f"range [{DATA_SCALE_LOWER:g}, {DATA_SCALE_UPPER:g}); the GARCH / "
        f"ARMA likelihood may be poorly scaled. Rescale the series (for "
        f"example with copulax.timeseries.DataScaler) before fitting. The "
        f"fit was NOT rescaled automatically."
    )


class ConvergenceWarning(Warning):
    r"""Raised when a time-series fit did not reach a stationary point.

    Emitted at the fit tail when the gradient norm at the returned best
    iterate exceeds the convergence tolerance, or when the optimiser hit
    a non-finite gradient region.  Subclasses :class:`Warning` so
    ``warnings.catch_warnings`` and ``warnings.filterwarnings`` handle it
    like any other warning.
    """


class DataScaleWarning(Warning):
    r"""Raised when a fitted series is scaled far from unit variance.

    Emitted at fit entry when the series' sample variance falls outside
    the well-conditioned range ``[0.1, 10000.0)``, at which the
    conditional-likelihood surface loses numerical conditioning.  The
    warning points the caller at ``DataScaler``; the fit itself is left
    un-rescaled.
    """
