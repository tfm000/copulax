r"""Pagan-Newey two-stage standard errors for the separable
``ARMA → GARCH-on-residuals`` workflow.

When a user fits ARMA first and then a GARCH-family variance model
on the ARMA residuals, the second-stage MLE depends implicitly on
the first-stage estimate (via :math:`\varepsilon_t = y_t -
\mu_t(\hat\theta_1)`).  Naive plug-in standard errors on the
second-stage parameters ignore the noise contributed by
:math:`\hat\theta_1` and are biased — typically too small.

The Pagan-Newey (1988) sandwich corrects for this by augmenting
the second-stage score with a first-stage influence-function term:

.. math::

    u_t = s_{2,t} - J_{21}\, J_{11}^{-1}\, s_{1,t},
    \qquad
    V_2 = J_{22}^{-1}\, \mathrm{Cov}(u_t)\, J_{22}^{-\top} \big/ n.

This module wires a fitted :class:`ARMABase` and a fitted
:class:`GARCHBase` together into the closures
:func:`pagan_newey_cov` consumes — building the joint
``(\theta_1, \theta_2)`` per-observation likelihood that runs the
ARMA recursion to derive :math:`\varepsilon_t` and then the GARCH
recursion on those :math:`\varepsilon_t`, with full autograd
support so the cross-stage Hessian :math:`J_{21}` is computed
exactly.

Public entry points:

* :func:`two_stage_cov` — corrected covariance matrix on the
  GARCH natural-parameter MLE.
* :func:`two_stage_standard_errors` — :math:`\sqrt{\mathrm{diag}}`
  packed back into a parameter-dict matching the GARCH
  ``params`` schema.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from copulax._src.timeseries._mean._arma_base import ARMABase
from copulax._src.timeseries._recursions import run_arma
from copulax._src.timeseries._residuals._standardise import StandardisedResidual
from copulax._src.timeseries._se import (
    flat_to_params,
    pagan_newey_cov,
    params_to_flat,
)
from copulax._src.timeseries._variance._garch_base import GARCHBase


_VAR_FLOOR: float = 1e-12
_SIGMA_FLOOR: float = 1e-6


def _build_two_stage_closures(
    arma_fit: ARMABase,
    var_fit: GARCHBase,
    y: Array,
    *,
    arma_init: str,
    arma_backcast_length: Optional[int],
    var_init: str,
    var_backcast_length: Optional[int],
):
    r"""Build the four closures :func:`pagan_newey_cov` consumes.

    Returns
    -------
    nll1_total, per_obs_nll1, nll2_joint, per_obs_nll2_joint, schemas

    ``schemas`` is a tuple ``(arma_schema, var_schema)`` so callers
    can recover dict-form parameters from the flat MLE vectors.
    """
    arma_wrapper = StandardisedResidual(arma_fit.residual_dist)
    var_wrapper = StandardisedResidual(var_fit.residual_dist)

    n = int(y.shape[0])
    arma_init_y_lags, arma_init_eps_lags = arma_fit._build_initial_state(
        y, mode=arma_init, backcast_length=arma_backcast_length,
    )

    # Pre-sample state for the variance recursion is built once on the
    # realised ε's at the MLE — matches arch's convention of pinning
    # the GARCH backcast to the optimum's residuals.  This keeps the
    # closure autodifferentiable purely with respect to the parameter
    # vectors.  Each GARCH-family variant's ``_ag_initial_state``
    # returns the correct variant-specific carry shape (vanilla
    # GARCH: ``(eps_sq, var)``; GJR: ``(eps_sq, neg_eps_sq, var)``;
    # EGARCH: ``(z, log_var)``; etc.).
    eps_at_mle = var_fit_residuals(
        arma_fit, y, init=arma_init,
        backcast_length=arma_backcast_length,
    )
    var_init_state = var_fit._ag_initial_state(
        eps_proxy=eps_at_mle,
        mode=var_init,
        backcast_length=var_backcast_length,
        residual_params=var_fit.residual_params,
    )

    arma_params = arma_fit.params
    var_params = var_fit.params
    arma_flat, arma_schema = params_to_flat(arma_params)
    var_flat, var_schema = params_to_flat(var_params)

    def _residuals_from_arma_flat(p1_flat: Array) -> Array:
        p1 = flat_to_params(p1_flat, arma_schema)
        phi = p1["phi"]
        theta = p1["theta"]
        mu = p1["mu"]
        _, eps_seq, _ = run_arma(
            y=y, phi=phi, theta=theta, mu=mu,
            init_y_lags=arma_init_y_lags,
            init_eps_lags=arma_init_eps_lags,
        )
        return eps_seq

    def _arma_per_obs_nll(p1_flat: Array) -> Array:
        p1 = flat_to_params(p1_flat, arma_schema)
        sigma_eps = jnp.maximum(p1["sigma_eps"], _SIGMA_FLOOR)
        residual_shape = p1.get("residual", {}) or {}
        eps_seq = _residuals_from_arma_flat(p1_flat)
        z = eps_seq / sigma_eps
        logpdf = arma_wrapper.logpdf(z, residual_shape) - jnp.log(sigma_eps)
        finite = jnp.isfinite(logpdf)
        safe = jnp.where(finite, logpdf, 0.0)
        return -safe

    def _arma_nll_total(p1_flat: Array) -> Array:
        return jnp.sum(_arma_per_obs_nll(p1_flat))

    def _var_per_obs_nll_joint(p1_flat: Array, p2_flat: Array) -> Array:
        eps_seq = _residuals_from_arma_flat(p1_flat)
        p2 = flat_to_params(p2_flat, var_schema)
        var_keys = var_fit._ag_var_keys()
        var_dict = {k: p2[k] for k in var_keys}
        residual_shape = p2.get("residual", {}) or {}
        var_seq, _ = var_fit._ag_run_recursion(
            eps_seq=eps_seq,
            var_params=var_dict,
            residual_params=residual_shape,
            init_state=var_init_state,
        )
        sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
        z = eps_seq / sigma_seq
        logpdf = var_wrapper.logpdf(z, residual_shape) - jnp.log(sigma_seq)
        finite = jnp.isfinite(logpdf)
        safe = jnp.where(finite, logpdf, 0.0)
        return -safe

    def _var_nll_total_joint(p1_flat: Array, p2_flat: Array) -> Array:
        return jnp.sum(_var_per_obs_nll_joint(p1_flat, p2_flat))

    return (
        _arma_nll_total,
        _arma_per_obs_nll,
        _var_nll_total_joint,
        _var_per_obs_nll_joint,
        (arma_schema, var_schema),
        (arma_flat, var_flat),
    )


def var_fit_residuals(
    arma_fit: ARMABase,
    y: ArrayLike,
    *,
    init: str = "backcast",
    backcast_length: Optional[int] = None,
) -> Array:
    r"""ARMA innovation residuals :math:`\varepsilon_t = y_t - \mu_t`.

    Convenience accessor used both by the two-stage SE machinery and
    directly by callers running the manual ARMA-then-GARCH workflow.
    Returns just the innovation array; for the full
    ``{"residuals", "standardised_residuals"}`` dict call
    ``arma_fit.residuals(y)`` directly.
    """
    return arma_fit.residuals(
        y, init=init, backcast_length=backcast_length,
    )["residuals"]


def two_stage_cov(
    arma_fit: ARMABase,
    var_fit: GARCHBase,
    y: ArrayLike,
    *,
    arma_init: str = "backcast",
    arma_backcast_length: Optional[int] = None,
    var_init: str = "backcast",
    var_backcast_length: Optional[int] = None,
) -> Array:
    r"""Pagan-Newey corrected covariance for the two-stage GARCH MLE.

    See :func:`copulax._src.timeseries._se.pagan_newey_cov` for the
    full derivation.  This wrapper accepts a fitted ARMA model and
    a fitted variance model (the variance model must have been fit
    on ``arma_fit.residuals(y)``), constructs the joint
    ``(\theta_1, \theta_2)`` likelihood closures, and returns the
    corrected ``(k_2, k_2)`` covariance of the variance-stage
    parameters.

    Args:
        arma_fit: Fitted :class:`ARMABase` instance (stage 1).
        var_fit: Fitted :class:`GARCHBase` instance (stage 2),
            previously fit on ``arma_fit.residuals(y)``.
        y: Original series the ARMA was fit on.
        arma_init / arma_backcast_length: Pre-sample state mode for
            the ARMA recursion.  Use the same setting that produced
            ``arma_fit.terminal_state`` for consistency.
        var_init / var_backcast_length: Pre-sample state mode for
            the variance recursion.

    Returns:
        ``(k_2, k_2)`` Pagan-Newey corrected covariance.
    """
    if not arma_fit.is_fitted:
        raise ValueError("arma_fit must be a fitted ARMABase instance.")
    if not var_fit.is_fitted:
        raise ValueError("var_fit must be a fitted GARCHBase instance.")

    y_arr = arma_fit._validate_series(y)
    n = int(y_arr.shape[0])

    (
        nll1_total,
        per_obs_nll1,
        nll2_joint,
        per_obs_nll2_joint,
        _schemas,
        (p1_flat, p2_flat),
    ) = _build_two_stage_closures(
        arma_fit, var_fit, y_arr,
        arma_init=arma_init, arma_backcast_length=arma_backcast_length,
        var_init=var_init, var_backcast_length=var_backcast_length,
    )

    return pagan_newey_cov(
        nll1_total=nll1_total,
        per_obs_nll1=per_obs_nll1,
        nll2_total_joint=nll2_joint,
        per_obs_nll2_joint=per_obs_nll2_joint,
        params1_flat=p1_flat,
        params2_flat=p2_flat,
        n_obs=n,
    )


def two_stage_standard_errors(
    arma_fit: ARMABase,
    var_fit: GARCHBase,
    y: ArrayLike,
    *,
    arma_init: str = "backcast",
    arma_backcast_length: Optional[int] = None,
    var_init: str = "backcast",
    var_backcast_length: Optional[int] = None,
) -> dict:
    r"""Pagan-Newey corrected standard errors as a parameter dict.

    Returns ``\sqrt{\mathrm{diag}(V_2)}`` packed back into the same
    nested-dict layout the variance model uses for its ``params``
    attribute, so per-parameter SEs can be looked up by name.

    See :func:`two_stage_cov` for the underlying covariance.
    """
    cov = two_stage_cov(
        arma_fit, var_fit, y,
        arma_init=arma_init, arma_backcast_length=arma_backcast_length,
        var_init=var_init, var_backcast_length=var_backcast_length,
    )
    se_flat = jnp.sqrt(jnp.maximum(jnp.diag(cov), 0.0))
    _, var_schema = params_to_flat(var_fit.params)
    return flat_to_params(se_flat, var_schema)


__all__ = [
    "two_stage_cov",
    "two_stage_standard_errors",
    "var_fit_residuals",
]
