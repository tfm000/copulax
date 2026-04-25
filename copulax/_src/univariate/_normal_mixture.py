"""Scalar helpers for mean-variance normal-mixture distributions.

Provides the public ``stats()`` assembler and the 1D analogue of the
multivariate γ-feasibility reparametrisation used by LDMLE.

The reparam maps an unconstrained scalar ``z`` onto a strict sub-interval of
the feasibility set ``{γ : γ² < sample_variance / Var[W]}``, so that the
moment-matching reconstruction

    σ² = (sample_variance − Var[W]·γ²) / E[W]

is strictly positive by construction — no silent ``sqrt(abs(...))`` repair is
required. See ``copulax/_src/multivariate/_normal_mixture.py`` for the d-dim
analogue.
"""

import jax.numpy as jnp
from jax import lax


FEASIBILITY_BUFFER: float = 0.99
_BUFFER_SQ: float = FEASIBILITY_BUFFER ** 2
_INV_SHRINK: float = 0.95
_EPS: float = 1e-12


def mean_variance_stats(w_stats: dict, mu: float, sigma: float, gamma: float) -> dict:
    """Compute mean, variance, and std of a mean-variance normal mixture."""
    mean: float = mu + lax.mul(w_stats["mean"], gamma)
    var: float = lax.mul(w_stats["mean"], lax.pow(sigma, 2)) + lax.mul(
        w_stats["variance"], lax.pow(gamma, 2)
    )
    std: float = lax.sqrt(var)
    return {"mean": mean, "variance": var, "std": std}


def forward_reparam_1d(
    z: float, sigma_hat: float, w_mean: float, w_var: float,
) -> tuple[float, float]:
    """Feasibility-guaranteed ``(gamma, sigma)`` from unconstrained ``z``.

    ``sigma`` is strictly positive by construction — ``Var[W]`` cancels inside
    ``sigma_sq`` after substitution, giving

        σ² = σ̂² · (1 − BUFFER² · v²) / E[W],    v = z / √(1 + z²).

    The inner factor ``(1 − BUFFER²·v²)`` has minimum ``1 − BUFFER² > 0``.
    """
    tau = 1.0 / jnp.sqrt(1.0 + z ** 2)
    v = tau * z
    c = FEASIBILITY_BUFFER / jnp.sqrt(w_var)
    gamma = c * sigma_hat * v
    sigma_sq = sigma_hat ** 2 * (1.0 - _BUFFER_SQ * v ** 2) / w_mean
    sigma = jnp.sqrt(sigma_sq)
    return gamma, sigma


def invert_gamma_to_z_1d(
    gamma0: float, sigma_hat: float, w_var0: float,
) -> float:
    """Invert ``gamma = c · sigma_hat · v`` to recover ``z0`` for optimiser init.

    ``gamma0`` is shrunk to ``_INV_SHRINK · c0 · sigma_hat`` in magnitude if it
    lands on or outside the feasibility interval, so the inverse denominator
    stays strictly positive.
    """
    c0 = FEASIBILITY_BUFFER / jnp.sqrt(w_var0)
    y0 = gamma0 / sigma_hat
    y0 = jnp.where(
        jnp.abs(y0) < _INV_SHRINK * c0,
        y0,
        jnp.sign(y0) * _INV_SHRINK * c0,
    )
    z0 = y0 / jnp.sqrt(c0 ** 2 - y0 ** 2 + _EPS)
    return z0
