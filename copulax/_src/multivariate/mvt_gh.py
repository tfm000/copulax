"""File containing the copulAX implementation of the multivariate 
generalized hyperbolic (GH) distribution."""
import jax.numpy as jnp
from jax import lax, random, jit
from jax._src.typing import ArrayLike, Array
from jax.scipy import special

from copulax._src._distributions import NormalMixture
from copulax._src.typing import Scalar
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.multivariate._shape import cov, _corr
from copulax._src.univariate.gig import gig
from copulax.special import kv


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
    def _classify_params(self, lamb: Scalar, chi: Scalar, psi: Scalar, 
                         mu: ArrayLike, gamma: ArrayLike, sigma: ArrayLike
                         ) -> tuple:
        return (lamb, chi, psi,), (mu, gamma), (sigma,)
    
    def _params_dict(self, lamb: Scalar, chi: Scalar, psi: Scalar,
                     mu: ArrayLike, gamma: ArrayLike, sigma: ArrayLike) -> dict:
        (lamb, chi, psi,), (mu, gamma,), (sigma,) = self._args_transform(
            lamb, chi, psi, mu, gamma, sigma)
        return {"lamb": lamb, "chi": chi, "psi": psi, "mu": mu,
                "gamma": gamma, "sigma": sigma}
    
    def support(self, lamb: Scalar=0.0, chi: Scalar=1.0, psi: Scalar=1.0, 
                mu: ArrayLike=jnp.zeros((2, 1)), 
                gamma: ArrayLike=jnp.zeros((2, 1)), 
                sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().support(lamb=lamb, chi=chi, psi=psi, mu=mu, gamma=gamma, 
                               sigma=sigma)
    
    @jit
    def _single_hi(self, carry: tuple, xi: jnp.ndarray) -> jnp.ndarray:
        mu, gamma, sigma_inv = carry
        return carry, lax.sub(xi, mu).T @ sigma_inv @ gamma
    
    def _calc_H(self, x: ArrayLike, mu: ArrayLike, gamma: ArrayLike, sigma_inv: ArrayLike) -> Array:
        r""""Calculates the H vector (x - mu).T @ sigma^-1 @ gamma."""
        return lax.scan(f=self._single_hi, xs=x,
                        init=(mu.flatten(), gamma.flatten(), sigma_inv))[1]  
    
    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, lamb: Scalar=0.0, 
                       chi: Scalar=1.0, psi: Scalar=1.0, mu: ArrayLike=jnp.zeros((2, 1)), 
                       gamma: ArrayLike=jnp.zeros((2, 1)), sigma: ArrayLike=jnp.eye(2, 2)
                       ) -> Array:
        x, yshape, n, d = _multivariate_input(x)
        (lamb, chi, psi,), (mu, gamma,), (sigma,) = self._args_transform(lamb, chi, psi, mu, gamma, sigma)
        # sigma: Array = _corr._rm(sigma, 1e-5)
        sigma_inv: Array = jnp.linalg.inv(sigma)
        Q: Array = chi + self._calc_Q(x=x, mu=mu, sigma_inv=sigma_inv)
        R: Array = psi + gamma.T @ sigma_inv @ gamma
        QR: Array = Q * R
        H: Array = self._calc_H(x=x, mu=mu, gamma=gamma, sigma_inv=sigma_inv)
        log_det_sigma: Scalar = jnp.linalg.slogdet(sigma)[1]
        s: Scalar = lamb - d / 2

        log_c: Scalar = (0.5 * lamb * lax.log((chi / (psi + stability)) + stability) - s * lax.log(R + stability) - 0.5 * d * lax.log(2 * jnp.pi) - 0.5 * log_det_sigma)
        
        logpdf: Array = (log_c 
                         + lax.log(kv(s, lax.sqrt(QR)) + stability) 
                         + H 
                         - 0.5 * s * (lax.log(QR + stability)))
        return logpdf.reshape(yshape)
    
    def logpdf(self, x: ArrayLike, lamb: Scalar=0.0, chi: Scalar=1.0, psi: Scalar=1.0,
               mu: ArrayLike=jnp.zeros((2, 1)), gamma: ArrayLike=jnp.zeros((2, 1)), 
               sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().logpdf(x=x, lamb=lamb, chi=chi, psi=psi, mu=mu, gamma=gamma, sigma=sigma)
    
    def pdf(self, x: ArrayLike, lamb: Scalar=0.0, chi: Scalar=1.0, psi: Scalar=1.0,
            mu: ArrayLike=jnp.zeros((2, 1)), gamma: ArrayLike=jnp.zeros((2, 1)), 
            sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().pdf(x=x, lamb=lamb, chi=chi, psi=psi, mu=mu, gamma=gamma, sigma=sigma)
    
    # sampling
    def rvs(self, size: int = 1, key: ArrayLike=DEFAULT_RANDOM_KEY, 
            lamb: Scalar=0.0, chi: Scalar=1.0, psi: Scalar=1.0, 
            mu: ArrayLike=jnp.zeros((2, 1)), gamma: ArrayLike=jnp.zeros((2, 1)), 
            sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        (lamb, chi, psi,), (mu, gamma,), (sigma,) = self._args_transform(
            lamb, chi, psi, mu, gamma, sigma)
        key, subkey = random.split(key)
        W: Array = gig.rvs(size=(size,), key=key, lamb=lamb, chi=chi, psi=psi)
        return super().rvs(key=subkey, n=size, W=W, mu=mu, gamma=gamma, sigma=sigma)
    
    def sample(self, size: int = 1, key: ArrayLike=DEFAULT_RANDOM_KEY,
               lamb: Scalar=0.0, chi: Scalar=1.0, psi: Scalar=1.0, 
               mu: ArrayLike=jnp.zeros((2, 1)), gamma: ArrayLike=jnp.zeros((2, 1)), 
               sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().sample(size=size, key=key, lamb=lamb, chi=chi, psi=psi, 
                              mu=mu, gamma=gamma, sigma=sigma)
    
    # stats
    def stats(self, lamb: Scalar=0.0, chi: Scalar=1.0, psi: Scalar=1.0, 
              mu: ArrayLike=jnp.zeros((2, 1)), gamma: ArrayLike=jnp.zeros((2, 1)), 
              sigma: ArrayLike=jnp.eye(2, 2)) -> dict:
        
        gig_stats = gig.stats(lamb=lamb, chi=chi, psi=psi)
        mean: Array = mu + gig_stats["mean"] * gamma
        cov: Array = gig_stats["mean"] * sigma + gig_stats["variance"] * jnp.outer(gamma, gamma)
        return {"mean": mean, "cov": cov, "skewness": gamma,}
    
    # metrics
    def loglikelihood(self, x: ArrayLike, lamb: Scalar=0.0, chi: Scalar=1.0, 
                      psi: Scalar=1.0, mu: ArrayLike=jnp.zeros((2, 1)), 
                      gamma: ArrayLike=jnp.zeros((2, 1)), 
                      sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().loglikelihood(x, lamb, chi, psi, mu, gamma, sigma)
    
    def aic(self, x: ArrayLike, lamb: Scalar=0.0, chi: Scalar=1.0, psi: Scalar=1.0,
            mu: ArrayLike=jnp.zeros((2, 1)), gamma: ArrayLike=jnp.zeros((2, 1)), 
            sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().aic(x, lamb, chi, psi, mu, gamma, sigma)
    
    def bic(self, x: ArrayLike, lamb: Scalar=0.0, chi: Scalar=1.0, psi: Scalar=1.0,
            mu: ArrayLike=jnp.zeros((2, 1)), gamma: ArrayLike=jnp.zeros((2, 1)), 
            sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().bic(x, lamb, chi, psi, mu, gamma, sigma)
    
    # fitting
    def _ldmle_inputs(self, d):
        eps = 1e-8
        lc = jnp.vstack((jnp.array([[-jnp.inf, -jnp.inf, -jnp.inf]]).T, jnp.full((d,1), -jnp.inf)))
        uc = jnp.full((d + 3,1), jnp.inf)
        
        key1, key2 = random.split(DEFAULT_RANDOM_KEY)
        key2, key3 = random.split(key2)
        params0 = jnp.array([random.normal(key1), *lax.exp(random.normal(key2, (2,))), *random.normal(key3, (d,))]).flatten()
        return {'hyperparams': (lc, uc)}, params0
    
    def _reconstruct_ldmle_params(self, params, loc, shape):
        d: int = loc.size
        scalars = lax.dynamic_slice_in_dim(params, 0, 3)
        lamb, chi_, psi_ = scalars
        chi, psi = jnp.exp(chi_), jnp.exp(psi_)
        gamma = lax.dynamic_slice_in_dim(params, 3, d).reshape((d, 1))
        gig_stats = gig.stats(lamb=lamb, chi=chi, psi=psi)

        mu: Array = loc - gig_stats["mean"] * gamma
        sigma_: Array = (shape - gig_stats["variance"] * jnp.outer(gamma, gamma)) / gig_stats["mean"]
        sigma: Array = _corr._rm_invalid(sigma_, 1e-5)
        return lamb, chi, psi, mu, gamma, sigma
    

    def _reconstruct_ldmle_copula_params(self, params, loc, shape):
        d: int = loc.size
        scalars = lax.dynamic_slice_in_dim(params, 0, 3)
        lamb, chi_, psi_ = scalars
        chi, psi = jnp.exp(chi_), jnp.exp(psi_)
        gamma = lax.dynamic_slice_in_dim(params, 3, d).reshape((d, 1))
        return lamb, chi, psi, loc, gamma, shape
    
    

mvt_gh = MvtGH("Mvt-GH")
