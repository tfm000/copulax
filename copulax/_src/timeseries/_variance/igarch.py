r"""IGARCH(p, q) — integrated GARCH (Engle & Bollerslev 1986).

Special case of vanilla GARCH with persistence pinned to 1:

.. math::

    \sigma^2_t = \omega
               + \sum_{i=1}^p \alpha_i\, \varepsilon^2_{t-i}
               + \sum_{j=1}^q \beta_j\, \sigma^2_{t-j},
    \qquad
    \sum_{i=1}^p \alpha_i + \sum_{j=1}^q \beta_j = 1,
    \qquad \omega > 0.

Although the unconditional variance does not exist under integrated
persistence, :math:`\omega` is finite-sample identifiable through its
contribution to the conditional-variance recursion's level over the
observed window — the standard treatment in both ``arch`` and
``rugarch``.

The σ²-recursion is identical to vanilla GARCH so we reuse
:func:`run_garch`; only the reparameterisation pack/unpack changes
to enforce the simplex constraint
:math:`\sum \alpha + \sum \beta = 1` exactly via
:func:`igarch_simplex` instead of :func:`garch_simplex`.

Reference:
    Engle, R.F. & Bollerslev, T. (1986).  *Modelling the persistence
    of conditional variances*.  Econometric Reviews, 5(1), 1-50.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax import Array

from copulax._src._distributions import Univariate
from copulax._src.timeseries._residuals._standardise import StandardisedResidual
from copulax._src.timeseries._stationarity import (
    igarch_simplex,
    igarch_unsimplex,
    positive_to_raw,
    raw_to_positive,
)
from copulax._src.timeseries._variance._garch_base import (
    GARCHBase,
    GARCHTerminalState,
)


class IGARCH(GARCHBase):
    r"""Integrated GARCH(p, q) — persistence pinned to 1.

    Construct with the desired orders and residual law:

    .. code-block:: python

        from copulax.timeseries import IGARCH
        from copulax.univariate import normal
        fit = IGARCH(p=1, q=1, residual_dist=normal).fit(eps)

    Inherits :meth:`fit` / :meth:`forecast` / :meth:`residuals` /
    :meth:`stats` etc. from :class:`GARCHBase`.  The only differences
    from vanilla GARCH are the reparameterisation (simplex over the
    full ``(α, β)`` vector rather than ``s · simplex``) and the
    documentation of :attr:`stats` (unconditional variance is
    ``inf``).
    """

    def __init__(
        self,
        p: int = 0,
        q: int = 0,
        *,
        residual_dist: Optional[Univariate] = None,
        name: str = "IGARCH",
        omega=None,
        alpha=None,
        beta=None,
        residual_params=None,
        terminal_state: Optional[GARCHTerminalState] = None,
        loglikelihood_=None,
        aic_=None,
        bic_=None,
        n_train_: Optional[int] = None,
    ):
        super().__init__(
            name=name,
            p=p,
            q=q,
            residual_dist=residual_dist,
            omega=omega,
            alpha=alpha,
            beta=beta,
            residual_params=residual_params,
            terminal_state=terminal_state,
            loglikelihood_=loglikelihood_,
            aic_=aic_,
            bic_=bic_,
            n_train_=n_train_,
        )

    @property
    def n_params(self) -> int:
        r"""Number of free fitted parameters.

        IGARCH drops one degree of freedom relative to GARCH because
        the simplex over ``(α, β)`` enforces a sum-to-one constraint.
        """
        wrapper = StandardisedResidual(self.residual_dist)
        # ω + (p + q - 1) free simplex coordinates + residual shape
        return 1 + (self.p + self.q - 1) + wrapper.n_shape_params

    # ------------------------------------------------------------------
    # Reparameterisation pack / unpack — simplex sums to 1
    # ------------------------------------------------------------------
    def _pack_x0(
        self,
        params_dict: dict,
        wrapper: StandardisedResidual,
    ) -> Array:
        r"""Pack a constrained ``params_dict`` into the unconstrained
        flat optimiser-state vector.

        Layout: ``[raw_omega (1,), raw_weights (p+q,),
        raw_residual_shape (n_shape_params,)]`` — no
        ``raw_persistence`` slot because persistence is pinned to 1.
        """
        omega = jnp.asarray(params_dict["omega"], dtype=float).reshape(())
        alpha = jnp.asarray(params_dict["alpha"], dtype=float).reshape(-1)
        beta = jnp.asarray(params_dict["beta"], dtype=float).reshape(-1)
        residual = params_dict.get("residual", {}) or {}

        raw_omega = positive_to_raw(jnp.maximum(omega, 1e-6))
        raw_weights = igarch_unsimplex(alpha, beta)
        raw_residual = wrapper.shape_params_to_array(residual)
        return jnp.concatenate(
            [raw_omega.reshape((1,)), raw_weights, raw_residual]
        )

    def _unpack_raw(
        self,
        raw: Array,
        wrapper: StandardisedResidual,
    ) -> tuple[Array, Array, Array, dict]:
        r"""Inverse of :meth:`_pack_x0`."""
        idx = 0
        raw_omega = raw[idx]
        idx += 1
        raw_weights = raw[idx : idx + self.p + self.q]
        idx += self.p + self.q
        raw_residual = raw[idx : idx + wrapper.n_shape_params]

        omega = raw_to_positive(raw_omega)
        alpha, beta = igarch_simplex(raw_weights, p=self.p)
        residual = wrapper.shape_params_from_array(raw_residual)
        return omega, alpha, beta, residual

    # ------------------------------------------------------------------
    # stats() — unconditional variance is undefined under integrated
    # persistence; persistence is identically 1.
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        r"""Analytic, parameter-only diagnostics for IGARCH.

        Unconditional variance does not exist (``inf``) because
        persistence ``= 1`` by construction.  Half-life is also
        infinite under that limit.
        """
        self._require_fitted()
        persistence = jnp.sum(self.alpha) + jnp.sum(self.beta)
        return {
            "unconditional_variance": jnp.asarray(jnp.inf, dtype=float),
            "persistence": persistence,
            "half_life": jnp.asarray(jnp.inf, dtype=float),
            "is_stationary": jnp.asarray(False),
        }
