"""File containing the copulAX implementation of the multivariate
student-t distribution."""

import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, random, jit
from jax import Array
from jax.typing import ArrayLike
from jax.scipy import special

from copulax._src._distributions import NormalMixture
from copulax._src.typing import Scalar
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src._utils import _resolve_key
from copulax._src.stats import kurtosis
from copulax._src.multivariate._shape import cov
from copulax._src.univariate.ig import ig

_NU_EPS = 1e-8


class MvtStudentT(NormalMixture):
    r"""The multivariate student-t distribution is a generalization of
    the univariate student-t distribution to d > 1 dimensions.

    https://en.wikipedia.org/wiki/Multivariate_t-distribution

    :math:`\mu` is the mean vector and :math:`\sigma` the shape matrix,
    which for this parameterization is not the variance-covariance
    matrix of the data distribution. :math:`\nu` is the degrees of
    freedom parameter.
    """

    nu: Array = None
    mu: Array = None
    sigma: Array = None

    def __init__(self, name="Mvt-Student-T", *, nu=None, mu=None, sigma=None):
        """Initialize with optional stored parameters `nu`, `mu`, and `sigma`."""
        super().__init__(name)
        self.nu = jnp.asarray(nu, dtype=float).reshape(()) if nu is not None else None
        self.mu = jnp.asarray(mu, dtype=float) if mu is not None else None
        self.sigma = jnp.asarray(sigma, dtype=float) if sigma is not None else None

    @property
    def _stored_params(self):
        """Return stored parameters dict if all are set, else None."""
        if self.nu is None or self.mu is None or self.sigma is None:
            return None
        return {"nu": self.nu, "mu": self.mu, "sigma": self.sigma}

    def _classify_params(self, params: dict) -> dict:
        """Classify parameters into scalar, vector, and shape groups."""
        return super()._classify_params(
            params=params,
            scalar_names=("nu",),
            vector_names=("mu",),
            shape_names=("sigma",),
            symmetric_shape_names=("sigma",),
        )

    def _params_dict(self, nu: Scalar, mu: ArrayLike, sigma: ArrayLike) -> dict:
        """Construct a normalized parameters dict from `nu`, `mu`, and `sigma`."""
        d: dict = {"nu": nu, "mu": mu, "sigma": sigma}
        return self._args_transform(d)

    def _params_to_tuple(self, params: dict) -> tuple:
        """Extract `(nu, mu, sigma)` tuple from a parameters dict."""
        params = self._args_transform(params)
        return params["nu"], params["mu"], params["sigma"]

    def example_params(self, dim: int = 3, *args, **kwargs) -> dict:
        r"""Example parameters for the multivariate student-t distribution.

        This is a three parameter family, defined by the degrees of
        freedom scalar `nu`, the mean / location vector `mu` and the
        shape matrix `sigma`.

        Args:
            dim: int, number of dimensions of the multivariate student-t
                distribution. Default is 3.
        """
        return self._params_dict(
            nu=2.5, mu=jnp.zeros((dim, 1)), sigma=jnp.eye(dim, dim)
        )

    def support(self, params: dict = None) -> Array:
        """Return the support: `(-inf, inf)` per dimension."""
        return super().support(params=params)

    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, params: dict) -> Array:
        """Numerically stable log-PDF of the multivariate student-t.

        Args:
            stability: Small constant added for numerical stability.
            x: Input data of shape (n, d).
            params: Distribution parameters.

        Returns:
            Array of log-density values with shape (n, 1).
        """
        x, yshape, n, d = _multivariate_input(x)
        nu, mu, sigma = self._params_to_tuple(params)

        s: Scalar = 0.5 * (nu + d)
        sigma_inv: Array = jnp.linalg.inv(sigma)
        Q: Array = self._calc_Q(x=x, mu=mu, sigma_inv=sigma_inv)
        log_det_sigma: Scalar = jnp.linalg.slogdet(sigma)[1]
        logpdf: Array = (
            lax.lgamma(s)
            - lax.lgamma(0.5 * nu)
            - 0.5 * d * jnp.log(jnp.pi * nu + stability)
            - 0.5 * log_det_sigma
            - s * jnp.log1p(Q / nu)
        )
        return logpdf.reshape(yshape)

    # sampling
    def rvs(self, size: int, params: dict = None, key: ArrayLike = None) -> Array:
        """Generate random samples via the normal-variance mixture.

        Sampling uses an inverse-gamma mixing variable W and the
        base class normal-variance mixture sampler.

        Args:
            size: Number of samples to draw.
            params: Distribution parameters.
            key: JAX random key.

        Returns:
            Array of shape (size, d).
        """
        params = self._resolve_params(params)
        key = _resolve_key(key)
        nu, mu, sigma = self._params_to_tuple(params)

        key, subkey = random.split(key)
        W: Array = ig.rvs(
            size=(size,), key=key, params={"alpha": 0.5 * nu, "beta": 0.5 * nu}
        )
        gamma: Array = jnp.zeros_like(mu)
        return super()._rvs(key=subkey, n=size, W=W, mu=mu, gamma=gamma, sigma=sigma)

    # stats
    def stats(self, params: dict = None) -> dict:
        """Compute distribution statistics (mean, median, mode, cov, skewness)."""
        params = self._resolve_params(params)
        nu, mu, sigma = self._params_to_tuple(params)

        mean: Array = jnp.where(nu > 1, mu, jnp.full_like(mu, jnp.nan))
        scale: Scalar = jnp.where(nu > 2, nu / (nu - 2), jnp.nan)
        cov: Array = scale * sigma
        return {
            "mean": mean,
            "median": mu,
            "mode": mu,
            "cov": cov,
            "skewness": jnp.zeros_like(mu),
        }

    # fitting
    def _ldmle_inputs(self, d, x=None):
        """Generate initial parameter array and bounds for LD-MLE optimization."""
        lc = jnp.full((1, 1), -jnp.inf)
        uc = jnp.full((1, 1), jnp.inf)

        # MoM: average marginal excess kurtosis -> nu = 4 + 6/kappa
        kappas = jnp.array([kurtosis(x[:, j], fisher=True) for j in range(d)])
        kappa = jnp.mean(kappas)
        nu0 = jnp.clip(4.0 + 6.0 / jnp.maximum(kappa, 0.06), 2.5, 100.0)

        raw_nu0 = jnp.log(jnp.expm1(nu0))
        params0: jnp.ndarray = jnp.array([raw_nu0])
        return {"lower": lc, "upper": uc}, params0

    def _reconstruct_ldmle_params(self, params_arr, loc, shape):
        """Reconstruct nu, mu, sigma from LD-MLE optimizer output."""
        raw_nu: Scalar = params_arr.reshape(())
        nu: Scalar = jnn.softplus(raw_nu) + _NU_EPS
        scale: Scalar = jnp.where(nu > 2, (nu - 2) / nu, 1.0)
        return nu, loc, scale * shape

    def _reconstruct_ldmle_copula_params(self, params, loc, shape):
        """Reconstruct copula parameters from LD-MLE optimizer output."""
        raw_nu: Scalar = params.reshape(())
        nu: Scalar = jnn.softplus(raw_nu) + _NU_EPS
        return nu, loc, shape


mvt_student_t = MvtStudentT("Mvt-Student-T")
