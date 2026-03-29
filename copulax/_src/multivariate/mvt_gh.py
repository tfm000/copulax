"""File containing the copulAX implementation of the multivariate
generalized hyperbolic (GH) distribution."""

import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, random, jit
from jax import Array
from jax.typing import ArrayLike
from jax.scipy import special

from copulax._src._distributions import NormalMixture
from copulax._src.special import log_kv
from copulax._src.typing import Scalar
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src._utils import _resolve_key, get_local_random_key
from copulax._src.multivariate._shape import cov, _corr
from copulax._src.univariate.gig import gig
from copulax.special import kv

_POS_EPS = 1e-8
_POS_INIT = 1.0


class MvtGH(NormalMixture):
    r"""The multivariate generalized hyperbolic (GH) distribution is a
    generalization of the univariate GH distribution to d > 1
    dimensions. This is a flexible, continuous 6-parameter family of
    distributions that can model a variety of data behaviors, including
    heavy tails and skewness. It contains a number of popular
    distributions as special cases, including the multivariate normal,
    multivariate student-t and multivariate skewed-T distributions.

    We adopt the parameterization used by McNeil et al. (2005)
    """

    _PARAM_KEY_TO_KWARG = {"lambda": "lamb"}

    lamb: Array = None
    chi: Array = None
    psi: Array = None
    mu: Array = None
    gamma: Array = None
    sigma: Array = None

    def __init__(
        self,
        name="Mvt-GH",
        *,
        lamb=None,
        chi=None,
        psi=None,
        mu=None,
        gamma=None,
        sigma=None,
    ):
        """Initialize with optional stored parameters."""
        super().__init__(name)
        self.lamb = (
            jnp.asarray(lamb, dtype=float).reshape(()) if lamb is not None else None
        )
        self.chi = (
            jnp.asarray(chi, dtype=float).reshape(()) if chi is not None else None
        )
        self.psi = (
            jnp.asarray(psi, dtype=float).reshape(()) if psi is not None else None
        )
        self.mu = jnp.asarray(mu, dtype=float) if mu is not None else None
        self.gamma = jnp.asarray(gamma, dtype=float) if gamma is not None else None
        self.sigma = jnp.asarray(sigma, dtype=float) if sigma is not None else None

    @property
    def _stored_params(self):
        """Return stored parameters dict if all are set, else None."""
        if any(
            v is None
            for v in [self.lamb, self.chi, self.psi, self.mu, self.gamma, self.sigma]
        ):
            return None
        return {
            "lambda": self.lamb,
            "chi": self.chi,
            "psi": self.psi,
            "mu": self.mu,
            "gamma": self.gamma,
            "sigma": self.sigma,
        }

    def _classify_params(self, params: dict) -> tuple:
        """Classify parameters into scalar, vector, and shape groups."""
        # return (lamb, chi, psi,), (mu, gamma), (sigma,)
        return super()._classify_params(
            params=params,
            scalar_names=("lambda", "chi", "psi"),
            vector_names=("mu", "gamma"),
            shape_names=("sigma",),
            symmetric_shape_names=("sigma",),
        )

    def _params_dict(
        self,
        lamb: Scalar,
        chi: Scalar,
        psi: Scalar,
        mu: ArrayLike,
        gamma: ArrayLike,
        sigma: ArrayLike,
    ) -> dict:
        """Construct a normalized parameters dict from all six GH parameters."""
        d: dict = {
            "lambda": lamb,
            "chi": chi,
            "psi": psi,
            "mu": mu,
            "gamma": gamma,
            "sigma": sigma,
        }
        return self._args_transform(d)

    def _params_to_tuple(self, params: dict) -> tuple:
        """Extract `(lambda, chi, psi, mu, gamma, sigma)` from a params dict."""
        params = self._args_transform(params)
        return (
            params["lambda"],
            params["chi"],
            params["psi"],
            params["mu"],
            params["gamma"],
            params["sigma"],
        )

    def example_params(self, dim: int = 3, *args, **kwargs) -> dict:
        r"""Example parameters for the multivariate GH distribution.

        This is a six parameter family, defined by the scalar parameters
        `lambda`, `chi`, `psi`, the location vector `mu`, the
        skewness vector `gamma` and the shape matrix `sigma`.

        Args:
            dim: int, number of dimensions of the multivariate GH
                distribution. Default is 3.
        """
        return self._params_dict(
            lamb=0.0,
            chi=1.0,
            psi=1.0,
            mu=jnp.zeros((dim, 1)),
            gamma=jnp.zeros((dim, 1)),
            sigma=jnp.eye(dim, dim),
        )

    def support(self, params: dict = None) -> Array:
        """Return the support: `(-inf, inf)` per dimension."""
        return super().support(params=params)

    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, params: dict) -> Array:
        """Numerically stable log-PDF of the multivariate GH distribution.

        Args:
            stability: Small constant for numerical stability.
            x: Input data of shape (n, d).
            params: Distribution parameters.

        Returns:
            Array of log-density values with shape (n, 1).
        """
        x, yshape, n, d = _multivariate_input(x)
        lamb, chi, psi, mu, gamma, sigma = self._params_to_tuple(params)

        sigma_inv: Array = jnp.linalg.inv(sigma)
        Q: Array = chi + self._calc_Q(x=x, mu=mu, sigma_inv=sigma_inv)
        R: Array = psi + gamma.T @ sigma_inv @ gamma
        QR: Array = Q * R
        # H: Array = self._calc_H(x=x, mu=mu, gamma=gamma, sigma_inv=sigma_inv)
        H: Array = ((x - mu.T) @ sigma_inv @ gamma).flatten()
        log_det_sigma: Scalar = jnp.linalg.slogdet(sigma)[1]
        s: Scalar = lamb - d / 2

        log_c: Scalar = (
            0.5 * lamb * lax.log((psi / (chi + stability)) + stability)
            - s * lax.log(R + stability)
            - 0.5 * d * lax.log(2 * jnp.pi)
            - 0.5 * log_det_sigma
            - log_kv(lamb, lax.sqrt(chi * psi))
        )

        logpdf: Array = (
            log_c
            + log_kv(s, lax.sqrt(QR))
            + H
            + 0.5 * s * (lax.log(QR + stability))
        )
        return logpdf.reshape(yshape)

    # sampling
    def rvs(self, size: int, params: dict = None, key: ArrayLike = None) -> Array:
        """Generate random samples via the GIG normal-variance mixture.

        Args:
            size: Number of samples to draw.
            params: Distribution parameters.
            key: JAX random key.

        Returns:
            Array of shape (size, d).
        """
        params = self._resolve_params(params)
        key = _resolve_key(key)
        lamb, chi, psi, mu, gamma, sigma = self._params_to_tuple(params)

        key, subkey = random.split(key)
        W: Array = gig.rvs(
            size=(size,), key=key, params={"lambda": lamb, "chi": chi, "psi": psi}
        )
        return super()._rvs(key=subkey, n=size, W=W, mu=mu, gamma=gamma, sigma=sigma)

    # stats
    def stats(self, params: dict = None) -> dict:
        """Compute distribution statistics using GIG mixing moments."""
        params = self._resolve_params(params)
        lamb, chi, psi, mu, gamma, sigma = self._params_to_tuple(params)
        gig_stats = gig.stats(params={"lambda": lamb, "chi": chi, "psi": psi})
        return self._stats(w_stats=gig_stats, mu=mu, gamma=gamma, sigma=sigma)

    # fitting
    def _ldmle_inputs(self, d, x=None):
        """Generate initial parameter array and bounds for LD-MLE optimization."""
        lc = jnp.full((d + 3, 1), -jnp.inf)
        uc = jnp.full((d + 3, 1), jnp.inf)

        key1, key2 = random.split(get_local_random_key())
        key2, key3 = random.split(key2)
        pos0 = _POS_INIT + jnp.abs(random.normal(key2, (2,)))
        pos0_raw = jnp.log(jnp.expm1(pos0))
        params0 = jnp.array(
            [
                random.normal(key1),
                *pos0_raw,
                *random.normal(key3, (d,)),
            ]
        ).flatten()
        return {"lower": lc, "upper": uc}, params0

    def _reconstruct_ldmle_params(self, params_arr, loc, shape):
        """Reconstruct lambda, chi, psi, mu, gamma, sigma from LD-MLE output."""
        d: int = loc.size
        scalars = lax.dynamic_slice_in_dim(params_arr, 0, 3)
        lamb, chi_, psi_ = scalars
        chi = jnn.softplus(chi_) + _POS_EPS
        psi = jnn.softplus(psi_) + _POS_EPS
        gamma: Array = lax.dynamic_slice_in_dim(params_arr, 3, d).reshape((d, 1))
        gig_stats: dict = gig.stats(params={"lambda": lamb, "chi": chi, "psi": psi})

        mu: Array = loc - gig_stats["mean"] * gamma
        sigma_: Array = (
            shape - gig_stats["variance"] * jnp.outer(gamma, gamma)
        ) / gig_stats["mean"]
        sigma: Array = _corr._rm_incomplete(sigma_, 1e-5)
        return lamb, chi, psi, mu, gamma, sigma

    def _reconstruct_ldmle_copula_params(self, params_arr, loc, shape):
        d: int = loc.size
        scalars = lax.dynamic_slice_in_dim(params_arr, 0, 3)
        lamb, chi_, psi_ = scalars
        chi = jnn.softplus(chi_) + _POS_EPS
        psi = jnn.softplus(psi_) + _POS_EPS
        gamma = lax.dynamic_slice_in_dim(params_arr, 3, d).reshape((d, 1))
        return lamb, chi, psi, loc, gamma, shape


mvt_gh = MvtGH("Mvt-GH")
