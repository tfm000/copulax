"""File containing the copulAX implementation of the multivariate
skewed-T distribution."""

import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, random, jit
from jax import Array
from jax.typing import ArrayLike
from jax.scipy import special

from copulax._src._distributions import NormalMixture
from copulax._src.typing import Scalar
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src._utils import _resolve_key, get_local_random_key
from copulax._src.multivariate._shape import cov, _corr
from copulax._src.univariate.ig import ig
from copulax._src.univariate.skewed_t import skewed_t
from copulax.special import kv
from copulax._src.multivariate.mvt_student_t import mvt_student_t

_NU_EPS = 1e-6
_NU_INIT = 4.0


class MvtSkewedT(NormalMixture):
    r"""The multivariate skewed-T distribution is a generalization of
    the univariate skewed-T distribution to d > 1 dimensions, which
    itself is a generalization of the student-t distribution which
    allows for skewness. It can also be expressed as a limiting case of
    the multivariate generalized hyperbolic distribution (GH) when
    phi -> 0 in addition to lambda = -0.5*chi.

    We use the 4 parameter McNeil et al (2005) specification of the
    distribution.
    """

    nu: Array = None
    mu: Array = None
    gamma: Array = None
    sigma: Array = None

    def __init__(
        self, name="Mvt-Skewed-T", *, nu=None, mu=None, gamma=None, sigma=None
    ):
        """Initialize with optional stored parameters `nu`, `mu`, `gamma`, and `sigma`."""
        super().__init__(name)
        self.nu = jnp.asarray(nu, dtype=float).reshape(()) if nu is not None else None
        self.mu = jnp.asarray(mu, dtype=float) if mu is not None else None
        self.gamma = jnp.asarray(gamma, dtype=float) if gamma is not None else None
        self.sigma = jnp.asarray(sigma, dtype=float) if sigma is not None else None

    @property
    def _stored_params(self):
        """Return stored parameters dict if all are set, else None."""
        if any(v is None for v in [self.nu, self.mu, self.gamma, self.sigma]):
            return None
        return {"nu": self.nu, "mu": self.mu, "gamma": self.gamma, "sigma": self.sigma}

    def _classify_params(self, params: dict) -> tuple:
        """Classify parameters into scalar, vector, and shape groups."""
        return super()._classify_params(
            params=params,
            scalar_names=("nu",),
            vector_names=("mu", "gamma"),
            shape_names=("sigma",),
            symmetric_shape_names=("sigma",),
        )

    def _params_dict(
        self, nu: Scalar, mu: ArrayLike, gamma: ArrayLike, sigma: ArrayLike
    ) -> dict:
        """Construct a normalized parameters dict from `nu`, `mu`, `gamma`, and `sigma`."""
        d: dict = {"nu": nu, "mu": mu, "gamma": gamma, "sigma": sigma}
        return self._args_transform(d)

    def _params_to_tuple(self, params: dict) -> tuple:
        """Extract `(nu, mu, gamma, sigma)` tuple from a parameters dict."""
        params = self._args_transform(params)
        return (params["nu"], params["mu"], params["gamma"], params["sigma"])

    def example_params(self, dim: int = 3, *args, **kwargs):
        """Example parameters for the multivariate skewed-t distribution.

        Args:
            dim: Number of dimensions. Default is 3.
        """
        return self._params_dict(
            nu=4.5,
            mu=jnp.zeros((dim, 1)),
            gamma=jnp.zeros((dim, 1)),
            sigma=jnp.eye(dim, dim),
        )

    def support(self, params: dict = None) -> Array:
        """Return the support: `(-inf, inf)` per dimension."""
        return super().support(params=params)

    def _skt_stable_logpdf(
        self, stability: Scalar, x: ArrayLike, params: dict
    ) -> Array:
        """Stable log-PDF for the skewed-t branch (gamma != 0).

        Uses the modified Bessel function of the second kind to handle
        the skewness term.

        Args:
            stability: Small constant for numerical stability.
            x: Input data of shape (n, d).
            params: Distribution parameters.

        Returns:
            Array of log-density values.
        """
        x, yshape, n, d = _multivariate_input(x)
        nu, mu, gamma, sigma = self._params_to_tuple(params)

        # clamp gamma away from zero to avoid kv(s, 0) singularity
        gamma = jnp.where(gamma == 0, 1e-30, gamma)

        sigma_inv: Array = jnp.linalg.inv(sigma)
        Q: Array = self._calc_Q(x=x, mu=mu, sigma_inv=sigma_inv)
        P: Array = sigma_inv @ gamma
        R: Array = gamma.T @ P
        s: Scalar = 0.5 * (nu + d)
        log_det_sigma: Scalar = jnp.linalg.slogdet(sigma)[1]

        log_c: Scalar = (
            (1 - s) * jnp.log(2)
            - lax.lgamma(0.5 * nu)
            - 0.5 * (d * lax.log(nu * jnp.pi + stability) + log_det_sigma)
        )

        log_T: Array = (
            lax.log(kv(s, jnp.sqrt((nu + Q) * R).flatten()))
            + ((x.T - mu).T @ P).flatten()
        )

        log_B: Array = (
            s
            * (
                lax.log(1 + Q / (nu + stability))
                - 0.5 * lax.log(stability + (nu + Q) * R)
            ).flatten()
        )

        logpdf: Array = log_c + log_T - log_B
        return logpdf.reshape(yshape)

    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, params: dict) -> Array:
        """Stable log-PDF dispatching to student-t or skewed-t branches.

        When `gamma = 0`, delegates to the symmetric student-t log-PDF
        to avoid the Bessel function singularity at zero.
        """
        gamma: Array = jnp.asarray(params["gamma"])
        is_symmetric = jnp.all(gamma == 0)
        student_t_result = mvt_student_t._stable_logpdf(
            stability=stability, x=x, params=params
        )
        # When gamma=0, the skewed branch has a kv(s, ~0) singularity
        # whose gradient diverges.  jnp.where differentiates BOTH
        # branches, so we substitute a safe non-zero gamma to keep
        # the unchosen branch finite during backprop.
        safe_gamma = jnp.where(is_symmetric, jnp.ones_like(gamma), gamma)
        safe_params = {**params, "gamma": safe_gamma}
        skewed_result = self._skt_stable_logpdf(
            stability=stability, x=x, params=safe_params
        )
        return jnp.where(is_symmetric, student_t_result, skewed_result)

    # sampling
    def rvs(self, size: int, params: dict = None, key: ArrayLike = None) -> Array:
        """Generate random samples via the normal-variance mixture.

        Args:
            size: Number of samples to draw.
            params: Distribution parameters.
            key: JAX random key.

        Returns:
            Array of shape (size, d).
        """
        params = self._resolve_params(params)
        key = _resolve_key(key)
        nu, mu, gamma, sigma = self._params_to_tuple(params)

        key, subkey = random.split(key)
        W: Array = ig.rvs(
            size=(size,), key=key, params={"alpha": 0.5 * nu, "beta": 0.5 * nu}
        )
        return super()._rvs(key=subkey, n=size, W=W, mu=mu, gamma=gamma, sigma=sigma)

    # stats
    def stats(self, params: dict = None) -> dict:
        """Compute distribution statistics using inverse-gamma mixing moments."""
        params = self._resolve_params(params)
        nu, mu, gamma, sigma = self._params_to_tuple(params)
        ig_stats = ig.stats(params={"alpha": 0.5 * nu, "beta": 0.5 * nu})
        return self._stats(w_stats=ig_stats, mu=mu, gamma=gamma, sigma=sigma)

    # fitting
    def _ldmle_inputs(self, d, x=None):
        """Generate initial parameter array and bounds for LD-MLE optimization."""
        lc = jnp.full((d + 1, 1), -jnp.inf)
        uc = jnp.full((d + 1, 1), jnp.inf)

        key1, key2 = random.split(get_local_random_key())
        key2, key3 = random.split(key2)
        # softplus(raw_nu) = nu enforces nu > 0 without hard lower clipping.
        nu0 = jnp.log(jnp.expm1(_NU_INIT + jnp.abs(random.normal(key1))))
        params0 = jnp.array(
            [nu0, *random.normal(key3, (d,))]
        ).flatten()
        return {"lower": lc, "upper": uc}, params0

    def _reconstruct_ldmle_params(self, params_arr, loc, shape):
        """Reconstruct nu, mu, gamma, sigma from LD-MLE optimizer output."""
        d: int = loc.size
        nu_: Scalar = lax.dynamic_slice_in_dim(params_arr, 0, 1)
        nu: Scalar = (jnn.softplus(nu_) + _NU_EPS).flatten()
        gamma: Array = lax.dynamic_slice_in_dim(params_arr, 1, d).reshape((d, 1))
        ig_stats = skewed_t._get_w_stats(nu=nu)

        mu: Array = loc - ig_stats["mean"] * gamma
        sigma_: Array = (
            shape - ig_stats["variance"] * jnp.outer(gamma, gamma)
        ) / ig_stats["mean"]
        sigma: Array = _corr._rm_incomplete(sigma_, 1e-5)
        return nu, mu, gamma, sigma

    def _reconstruct_ldmle_copula_params(self, params_arr, loc, shape):
        """Reconstruct copula parameters from LD-MLE optimizer output."""
        d: int = loc.size
        nu_: Scalar = lax.dynamic_slice_in_dim(params_arr, 0, 1)
        nu: Scalar = (jnn.softplus(nu_) + _NU_EPS).flatten()
        gamma = lax.dynamic_slice_in_dim(params_arr, 1, d).reshape((d, 1))
        return nu, loc, gamma, shape


mvt_skewed_t = MvtSkewedT("Mvt-Skewed-T")
