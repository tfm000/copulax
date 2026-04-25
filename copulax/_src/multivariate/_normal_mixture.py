"""Feasibility reparametrisation shared by normal-mixture LDMLE paths.

For the moment-matching reconstruction
``Sigma = (sample_cov - Var[W] * gamma gamma^T) / E[W]``, positive-definiteness
requires ``gamma^T sample_cov^{-1} gamma < 1 / Var[W]``. Parametrising gamma
from an unconstrained ``z`` via

    tau(z) = 1 / sqrt(1 + ||z||^2)
    v(z)   = tau(z) * z                           # in the open unit ball
    L      = chol(sample_cov)                     # once per fit
    c      = FEASIBILITY_BUFFER / sqrt(Var[W])
    gamma  = c * L @ v(z)

places gamma strictly inside the feasibility ellipsoid, so Sigma is PD by
construction and no silent repair is needed. After substitution, the factor
``Var[W]`` cancels and

    Sigma = L @ (I - FEASIBILITY_BUFFER^2 * v v^T) @ L.T / E[W],

which uses only ``L`` - the sample covariance is never touched during
per-iteration work. The inner matrix ``I - FEASIBILITY_BUFFER^2 * v v^T`` has
smallest eigenvalue ``>= 1 - FEASIBILITY_BUFFER^2 > 0``.
"""

import jax.numpy as jnp
from jax import Array

from copulax._src.multivariate._shape import cov, _corr


FEASIBILITY_BUFFER: float = 0.99
_BUFFER_SQ: float = FEASIBILITY_BUFFER ** 2
_INV_SHRINK: float = 0.95
_EPS: float = 1e-12


def prepare_sample_cov(x: Array, cov_method: str) -> tuple[Array, Array]:
    """Return ``(sample_mean, L)`` with sample covariance PD-enforced and factored.

    Called once per LDMLE fit. ``L`` is passed down as the ``shape`` argument so
    no per-iteration Cholesky is needed.
    """
    d: int = x.shape[1]
    sample_mean: Array = jnp.mean(x, axis=0).reshape((d, 1))
    sample_cov_pd: Array = _corr._rm_incomplete(cov(x=x, method=cov_method), 1e-5)
    L: Array = jnp.linalg.cholesky(sample_cov_pd)
    return sample_mean, L


def forward_reparam(
    z: Array, L: Array, w_mean: Array, w_var: Array,
) -> tuple[Array, Array]:
    """Feasibility-guaranteed ``(gamma, sigma)`` from unconstrained ``z``.

    ``sigma`` is strictly PD for all finite ``z`` by construction.
    """
    d: int = L.shape[0]
    tau: Array = 1.0 / jnp.sqrt(1.0 + jnp.sum(z ** 2))
    v: Array = tau * z
    c: Array = FEASIBILITY_BUFFER / jnp.sqrt(w_var)
    gamma: Array = (c * (L @ v)).reshape((d, 1))
    inner: Array = jnp.eye(d) - _BUFFER_SQ * jnp.outer(v, v)
    sigma: Array = (L @ inner @ L.T) / w_mean
    return gamma, sigma


def invert_gamma_to_z(gamma0: Array, L0: Array, w_var0: Array) -> Array:
    """Invert ``gamma = c * L @ v`` to recover ``z0`` for the optimiser's init.

    ``gamma0`` is shrunk to ``_INV_SHRINK * c0`` in the sample-covariance metric
    whenever it lands on or outside the feasibility ellipsoid, so the inverse
    denominator ``sqrt(c0^2 - ||y0||^2)`` stays strictly positive.
    """
    d: int = L0.shape[0]
    c0: Array = FEASIBILITY_BUFFER / jnp.sqrt(w_var0)
    y0: Array = jnp.linalg.solve(L0, gamma0.reshape(d))
    y0_norm: Array = jnp.linalg.norm(y0)
    y0 = jnp.where(
        y0_norm < _INV_SHRINK * c0,
        y0,
        y0 * (_INV_SHRINK * c0 / (y0_norm + _EPS)),
    )
    z0: Array = y0 / jnp.sqrt(c0 ** 2 - jnp.sum(y0 ** 2) + _EPS)
    return z0
