"""File containing the copulAX implementation of the multivariate normal 
distribution."""
import jax.numpy as jnp
from jax import lax, random, jit
from jax._src.typing import ArrayLike, Array
from jax.scipy import special

from copulax._src._distributions import Multivariate
from copulax._src.typing import Scalar
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.multivariate._shape import cov


class MvtNormal(Multivariate):
    r"""The multivariate normal / Gaussian distribution is a 
    generalization of the univariate normal distribution to d > 1 
    dimensions.

    https://en.wikipedia.org/wiki/Multivariate_normal_distribution

    .. math::

        f(x|\mu, \Sigma) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)

    where :math:`\mu` is the mean vector and :math:`\sigma` the 
    variance-covariance matrix of the data distribution.
    """
    def _classify_params(self, mu: ArrayLike, sigma: ArrayLike) -> tuple:
        return (), (mu,), (sigma,)

    def _params_dict(self, mu: ArrayLike, sigma: ArrayLike) -> dict:
        _, (mu,), (sigma,) = self._args_transform(mu, sigma)
        return {"mu": mu, "sigma": sigma}
    
    def support(self, mu: ArrayLike=jnp.zeros((2, 1)), 
                sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().support(mu=mu, sigma=sigma)

    def logpdf(self, x: ArrayLike, mu: ArrayLike=jnp.zeros((2, 1)), 
               sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        x, yshape, n, d = _multivariate_input(x)
        _, (mu,), (sigma,) = self._args_transform(mu, sigma)

        const: jnp.ndarray = -0.5 * (d * jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(sigma)[1])

        sigma_inv: Array = jnp.linalg.inv(sigma)
        Q: jnp.ndarray = self._calc_Q(x=x, mu=mu, sigma_inv=sigma_inv)

        logpdf: jnp.ndarray = -0.5 * Q + const
        return logpdf.reshape(yshape)
    
    def pdf(self, x: ArrayLike, mu: ArrayLike=jnp.zeros((2, 1)), 
            sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().pdf(x=x, mu=mu, sigma=sigma)
    
    # sampling
    def rvs(self, size: int = 1, key:ArrayLike=DEFAULT_RANDOM_KEY, 
            mu: ArrayLike=jnp.zeros((2, 1)), 
            sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        _, (mu,), (sigma,) = self._args_transform(mu, sigma)
        return random.multivariate_normal(
            key=key, mean=mu.flatten(), cov=sigma, shape=(size, ))
    
    def sample(self, size: int = 1, key: ArrayLike=DEFAULT_RANDOM_KEY, 
               mu: ArrayLike=jnp.zeros((2, 1)), 
               sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().sample(size=size, key=key, mu=mu, sigma=sigma)
    
    # stats
    def stats(self, mu: ArrayLike=jnp.zeros((2, 1)), 
              sigma: ArrayLike=jnp.eye(2, 2)) -> dict:
        return {"mean": mu, "median": mu, "mode": mu, "cov": sigma, 
                "skewness": jnp.zeros_like(mu),}
    
    # metrics
    def loglikelihood(self, x: ArrayLike, mu: ArrayLike=jnp.zeros((2, 1)), 
                      sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().loglikelihood(x, mu, sigma)
    
    def aic(self, x: ArrayLike, mu: ArrayLike=jnp.zeros((2, 1)), 
            sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().aic(x, mu, sigma)
    
    def bic(self, x: ArrayLike, mu: ArrayLike=jnp.zeros((2, 1)),
            sigma: ArrayLike=jnp.eye(2, 2)) -> Array:
        return super().bic(x, mu, sigma)

    # fitting
    def fit(self, x: ArrayLike, sigma_method: str = 'pearson', 
            *args, **kwargs) -> dict:
        r"""Fits the multivariate normal distribution to the data.

        Note:
            If you intend to jit wrap this function, ensure that 
            'sigma_method' is a static argument.

        Args:
            x: arraylike, data to fit the distribution to.
            sigma_method: str, method to estimate the covariance matrix,
                sigma. See copulax.multivariate.cov for available 
                methods.

        Returns:
            dict containing the fitted parameters.
        """
        x, _, _, d = _multivariate_input(x)
        mu: jnp.ndarray = jnp.mean(x, axis=0)
        sigma: jnp.ndarray = cov(x=x, method=sigma_method)
        return self._params_dict(mu=mu, sigma=sigma)
    
    def _fit_copula(self, u, corr_method = 'pearson', *args, **kwargs):
        d: dict = super()._fit_copula(u, corr_method, *args, **kwargs)
        return {'mu': d['mu'], 'sigma': d['sigma']}
        

mvt_normal = MvtNormal("Mvt-Normal")