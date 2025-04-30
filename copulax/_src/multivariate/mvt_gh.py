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
    def _classify_params(self, params: dict) -> tuple:
        # return (lamb, chi, psi,), (mu, gamma), (sigma,)
        return super()._classify_params(
            params=params, scalar_names=('lambda', 'chi', 'psi'),
            vector_names=('mu', 'gamma'), shape_names=('sigma',),
            symmetric_shape_names=('sigma',))
    
    def _params_dict(self, lamb: Scalar, chi: Scalar, psi: Scalar,
                     mu: ArrayLike, gamma: ArrayLike, sigma: ArrayLike) -> dict:
        d: dict = {"lambda": lamb, "chi": chi, "psi": psi, "mu": mu, 
                   "gamma": gamma, "sigma": sigma}
        return self._args_transform(d)
    
    def _params_to_tuple(self, params: dict) -> tuple:
        params = self._args_transform(params)
        return (params["lambda"], params["chi"], params["psi"],
                params["mu"], params["gamma"], params["sigma"])
    
    def example_params(self, dim: int = 3, *args, **kwargs) -> dict:
        r"""Example parameters for the multivariate GH distribution.
        
        This is a six parameter family, defined by the scalar parameters 
        `lambda`, `chi`, `psi`, the location vector `mu`, the 
        skewness vector `gamma` and the shape matrix `sigma`.

        Args:
            dim: int, number of dimensions of the multivariate GH 
                distribution. Default is 3.
        """
        return self._params_dict(lamb=0.0, chi=1.0, psi=1.0, 
                                 mu=jnp.zeros((dim, 1)), 
                                 gamma=jnp.zeros((dim, 1)), 
                                 sigma=jnp.eye(dim, dim))
    
    def support(self, params: dict) -> Array:
        return super().support(params=params)
    
    # @jit
    # def _single_hi(self, carry: tuple, xi: jnp.ndarray) -> jnp.ndarray:
    #     mu, gamma, sigma_inv = carry
    #     return carry, lax.sub(xi, mu).T @ sigma_inv @ gamma
    
    # def _calc_H(self, x: ArrayLike, mu: ArrayLike, gamma: ArrayLike, sigma_inv: ArrayLike) -> Array:
    #     r""""Calculates the H vector (x - mu).T @ sigma^-1 @ gamma."""
    #     return lax.scan(f=self._single_hi, xs=x, 
    #                     init=(mu.flatten(), gamma.flatten(), sigma_inv))[1]
    
    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, 
                       params: dict) -> Array:
        x, yshape, n, d = _multivariate_input(x)
        lamb, chi, psi, mu, gamma, sigma = self._params_to_tuple(params)
        # sigma: Array = _corr._rm(sigma, 1e-5)

        sigma_inv: Array = jnp.linalg.inv(sigma)
        Q: Array = chi + self._calc_Q(x=x, mu=mu, sigma_inv=sigma_inv)
        R: Array = psi + gamma.T @ sigma_inv @ gamma
        QR: Array = Q * R
        # H: Array = self._calc_H(x=x, mu=mu, gamma=gamma, sigma_inv=sigma_inv)
        H: Array = (x @ sigma_inv @ gamma).flatten()
        log_det_sigma: Scalar = jnp.linalg.slogdet(sigma)[1]
        s: Scalar = lamb - d / 2

        log_c: Scalar = (0.5 * lamb * lax.log((chi / (psi + stability)) + stability) - s * lax.log(R + stability) - 0.5 * d * lax.log(2 * jnp.pi) - 0.5 * log_det_sigma)
        
        logpdf: Array = (log_c 
                         + lax.log(kv(s, lax.sqrt(QR)) + stability) 
                         + H 
                         - 0.5 * s * (lax.log(QR + stability)))
        return logpdf.reshape(yshape)
    
    # sampling
    def rvs(self, size: int, params: dict, 
            key: ArrayLike=DEFAULT_RANDOM_KEY) -> Array:
        lamb, chi, psi, mu, gamma, sigma = self._params_to_tuple(params)

        key, subkey = random.split(key)
        W: Array = gig.rvs(size=(size,), key=key, 
                           params={'lambda': lamb, 'chi': chi, 'psi': psi})
        return super()._rvs(key=subkey, n=size, W=W, mu=mu, gamma=gamma, sigma=sigma)
    
    # stats
    def stats(self, params: dict) -> dict:
        lamb, chi, psi, mu, gamma, sigma = self._params_to_tuple(params)
        gig_stats = gig.stats(params={'lambda': lamb, 'chi': chi, 'psi': psi})
        return self._stats(w_stats=gig_stats, mu=mu, gamma=gamma, sigma=sigma)
    
    # fitting
    def _ldmle_inputs(self, d):
        lc = jnp.full((d + 3,1), -jnp.inf)
        uc = jnp.full((d + 3,1), jnp.inf)
        
        key1, key2 = random.split(DEFAULT_RANDOM_KEY)
        key2, key3 = random.split(key2)
        params0 = jnp.array([random.normal(key1), *lax.exp(random.normal(key2, (2,))), *random.normal(key3, (d,))]).flatten()
        return {'hyperparams': (lc, uc)}, params0
    
    def _reconstruct_ldmle_params(self, params_arr, loc, shape):
        d: int = loc.size
        scalars = lax.dynamic_slice_in_dim(params_arr, 0, 3)
        lamb, chi_, psi_ = scalars
        chi, psi = jnp.exp(chi_), jnp.exp(psi_)
        gamma: Array = lax.dynamic_slice_in_dim(params_arr, 3, d).reshape((d, 1))
        gig_stats: dict = gig.stats(params={'lambda': lamb, 'chi': chi, 'psi': psi})

        mu: Array = loc - gig_stats["mean"] * gamma
        sigma_: Array = (shape - gig_stats["variance"] * jnp.outer(gamma, gamma)) / gig_stats["mean"]
        sigma: Array = _corr._rm_incomplete(sigma_, 1e-5)
        return lamb, chi, psi, mu, gamma, sigma
    

    def _reconstruct_ldmle_copula_params(self, params_arr, loc, shape):
        d: int = loc.size
        scalars = lax.dynamic_slice_in_dim(params_arr, 0, 3)
        lamb, chi_, psi_ = scalars
        chi, psi = jnp.exp(chi_), jnp.exp(psi_)
        gamma = lax.dynamic_slice_in_dim(params_arr, 3, d).reshape((d, 1))
        return lamb, chi, psi, loc, gamma, shape


mvt_gh = MvtGH("Mvt-GH")
