"""File containing the copulAX implementation of the multivariate 
student-t distribution."""
import jax.numpy as jnp
from jax import lax, random, jit
from jax._src.typing import ArrayLike, Array
from jax.scipy import special

from copulax._src._distributions import NormalMixture
from copulax._src.typing import Scalar
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.multivariate._shape import cov
from copulax._src.univariate.ig import ig


class MvtStudentT(NormalMixture):
    r"""The multivariate student-t distribution is a generalization of 
    the univariate student-t distribution to d > 1 dimensions.
    
    https://en.wikipedia.org/wiki/Multivariate_t-distribution

    :math:`\mu` is the mean vector and :math:`\sigma` the shape matrix,
    which for this parameterization is not the variance-covariance 
    matrix of the data distribution. :math:`\nu` is the degrees of 
    freedom parameter.
    """
    def _classify_params(self, params: dict) -> dict:
        return super()._classify_params(
            params=params, scalar_names=('nu',), vector_names=('mu',), 
            shape_names=('sigma',), symmetric_shape_names=('sigma',))
    
    def _params_dict(self, nu: Scalar, mu: ArrayLike, sigma: ArrayLike) -> dict:
        d: dict = {"nu": nu, "mu": mu, "sigma": sigma}
        return self._args_transform(d)
    
    def _params_to_tuple(self, params: dict) -> tuple:
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
        return self._params_dict(nu=2.5, mu=jnp.zeros((dim, 1)), 
                                 sigma=jnp.eye(dim, dim))
    
    def support(self, params: dict) -> Array:
        return super().support(params=params)
    
    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, 
                       params: dict) -> Array:
        x, yshape, n, d = _multivariate_input(x)
        nu, mu, sigma = self._params_to_tuple(params)

        s: Scalar = 0.5 * (nu + d)
        sigma_inv: Array = jnp.linalg.inv(sigma)
        Q: Array = self._calc_Q(x=x, mu=mu, sigma_inv=sigma_inv)
        log_det_sigma: Scalar = jnp.linalg.slogdet(sigma)[1]
        logpdf: Array = (lax.lgamma(s) 
                         - lax.lgamma(0.5 * nu) 
                         - 0.5 * d * jnp.log(jnp.pi * nu + stability) 
                         - 0.5 * log_det_sigma 
                         - s * jnp.log1p(Q / nu))
        return logpdf.reshape(yshape)
    
    # sampling
    def rvs(self, size: Scalar, params: dict, 
            key: ArrayLike=DEFAULT_RANDOM_KEY) -> Array:
        nu, mu, sigma = self._params_to_tuple(params)
        size: Scalar = self._size_input(size)

        key, subkey = random.split(key)
        W: Array = ig.rvs(size=(size, ), key=key, params={'alpha': 0.5 * nu, 'beta': 0.5 * nu})
        gamma: Array = jnp.zeros_like(mu)
        return super()._rvs(key=subkey, n=size, W=W, mu=mu, gamma=gamma, sigma=sigma)
    
    # stats
    def stats(self, params: dict) -> dict:
        nu, mu, sigma = self._params_to_tuple(params)

        mean: Array = jnp.where(nu > 1, mu, jnp.full_like(mu, jnp.nan))
        scale: Scalar = jnp.where(nu > 2, nu / (nu - 2), jnp.nan)
        cov: Array = scale * sigma
        return {
            "mean": mean, 
            "median": mu, 
            "mode": mu, 
            "cov": cov, 
            "skewness": jnp.zeros_like(mu),}
    
    # fitting
    def _ldmle_inputs(self, d):
        constraints: tuple = (jnp.array([[1e-8]]).T, 
                            jnp.array([[jnp.inf]]).T)
        params0: jnp.ndarray = jnp.exp(random.normal(key=DEFAULT_RANDOM_KEY, shape=(1, )) * 2.5)
        return {'hyperparams': constraints}, params0

    def _reconstruct_ldmle_params(self, params_arr, loc, shape):
        nu: Scalar = params_arr.reshape(())
        scale: Scalar = jnp.where(nu > 2, (nu - 2) / 2, 1.0)
        return nu, loc, scale * shape
    
    def _reconstruct_ldmle_copula_params(self, params, loc, shape):
        nu: Scalar = params.reshape(())
        return nu, loc, shape
        

    
    
        

mvt_student_t = MvtStudentT("Mvt-Student-T")