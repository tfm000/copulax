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
    def _classify_params(self, nu: Scalar, mu: ArrayLike, sigma: ArrayLike
                         ) -> tuple:
        return (nu,), (mu,), (sigma,)
    
    def _params_dict(self, nu: Scalar, mu: ArrayLike, sigma: ArrayLike) -> dict:
        (nu,), (mu,), (sigma,) = self._args_transform(nu, mu, sigma)
        return {"nu": nu, "mu": mu, "sigma": sigma}
    
    def support(self, nu: Scalar=1.0, mu: ArrayLike=jnp.zeros((2, 1)), 
                sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().support(nu=nu, mu=mu, sigma=sigma)
    
    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, 
                       nu: Scalar, mu: ArrayLike, sigma: ArrayLike) -> Array:
        x, yshape, n, d = _multivariate_input(x)
        (nu,), (mu,), (sigma,) = self._args_transform(nu, mu, sigma)

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
    
    def logpdf(self, x: ArrayLike, nu: Scalar=1.0, mu: ArrayLike=jnp.zeros((2, 1)),
               sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().logpdf(x=x, nu=nu, mu=mu, sigma=sigma)
    
    def pdf(self, x: ArrayLike, nu: Scalar=1.0, mu: ArrayLike=jnp.zeros((2, 1)),
            sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().pdf(x=x, nu=nu, mu=mu, sigma=sigma)
    
    # sampling
    def rvs(self, size: int = 1, key: ArrayLike=DEFAULT_RANDOM_KEY,
            nu: Scalar=1.0, mu: ArrayLike=jnp.zeros((2, 1)),
            sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        (nu,), (mu,), (sigma,) = self._args_transform(nu, mu, sigma)
        key, subkey = random.split(key)
        W: Array = ig.rvs(size=(size, ), key=key, alpha=0.5 * nu, beta=0.5 * nu)
        gamma: Array = jnp.zeros_like(mu)
        return super().rvs(key=subkey, n=size, W=W, mu=mu, gamma=gamma, sigma=sigma)

    def sample(self, size: int = 1, key: ArrayLike=DEFAULT_RANDOM_KEY,
               nu: Scalar=1.0, mu: ArrayLike=jnp.zeros((2, 1)),
               sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().sample(size=size, key=key, nu=nu, mu=mu, sigma=sigma)
    
    # stats
    def stats(self, nu: Scalar=1.0, mu: ArrayLike=jnp.zeros((2, 1)),
              sigma: ArrayLike=jnp.eye(2, 2)) -> dict:
        mean: Array = jnp.where(nu > 1, mu, jnp.full_like(mu, jnp.nan))
        scale: Scalar = jnp.where(nu > 2, nu / (nu - 2), jnp.nan)
        cov: Array = scale * sigma
        return {"mean": mean, "median": mu, "mode": mu, "cov": cov, 
                "skewness": jnp.zeros_like(mu),}

    # metrics
    def loglikelihood(self, x: ArrayLike, nu: Scalar=1.0,
                      mu: ArrayLike=jnp.zeros((2, 1)),
                      sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().loglikelihood(x, nu, mu, sigma)
    
    def aic(self, x: ArrayLike, nu: Scalar=1.0, mu: ArrayLike=jnp.zeros((2, 1)),
            sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().aic(x, nu, mu, sigma)
    
    def bic(self, x: ArrayLike, nu: Scalar=1.0, mu: ArrayLike=jnp.zeros((2, 1)),
            sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().bic(x, nu, mu, sigma)
    
    # fitting
    def _reconstruct_ldmle_params(self, params, sample_mean, sample_cov):
        nu: Scalar = params.reshape(())
        scale: Scalar = jnp.where(nu > 2, (nu - 2) / 2, 1.0)
        return nu, sample_mean, scale * sample_cov
    
    def _ldmle_inputs(self, d):
        constraints: tuple = (jnp.array([[1e-8]]).T, 
                            jnp.array([[jnp.inf]]).T)
        params0: jnp.ndarray = jnp.abs(random.normal(key=DEFAULT_RANDOM_KEY, shape=(1, )))
        return {'hyperparams': constraints}, params0
        

mvt_student_t = MvtStudentT("Mvt-Student-T")