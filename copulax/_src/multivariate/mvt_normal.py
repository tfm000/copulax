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
    def _classify_params(self, params: dict) -> dict:
        return super()._classify_params(
            params=params, vector_names=('mu',), shape_names=('sigma',), 
            symmetric_shape_names=('sigma',))

    def _params_dict(self, mu: ArrayLike, sigma: ArrayLike) -> dict:
        d: dict = {"mu": mu, "sigma": sigma}
        return self._args_transform(d)
    
    def _params_to_tuple(self, params: dict) -> tuple:
        params = self._args_transform(params)
        return params["mu"], params["sigma"]
    
    def example_params(self, dim: int = 3, *args, **kwargs) -> dict:
        r"""Example parameters for the multivariate normal distribution.
        
        This is a two parameter family, defined by the mean / location 
        vector `mu` and the variance-covariance matrix `sigma`.

        Args:
            dim: int, number of dimensions of the multivariate normal 
                distribution. Default is 3.
        """
        return self._params_dict(mu=jnp.zeros((dim, 1)), 
                                 sigma=jnp.eye(dim, dim))
    
    def support(self, params: dict) -> Array:
        return super().support(params=params)

    def logpdf(self, x: ArrayLike, params: dict) -> Array:
        x, yshape, n, d = _multivariate_input(x)
        mu, sigma = self._params_to_tuple(params)

        const: jnp.ndarray = -0.5 * (d * jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(sigma)[1])

        sigma_inv: Array = jnp.linalg.inv(sigma)
        Q: jnp.ndarray = self._calc_Q(x=x, mu=mu, sigma_inv=sigma_inv)

        logpdf: jnp.ndarray = -0.5 * Q + const
        return logpdf.reshape(yshape)
    
    # sampling
    def rvs(self, size: int, params: dict, key=DEFAULT_RANDOM_KEY) -> Array:
        mu, sigma = self._params_to_tuple(params)
        return random.multivariate_normal(key=key, mean=mu.flatten(), 
                                          cov=sigma, shape=(size, ))
    
    # stats
    def stats(self, params: dict) -> dict:
        mu, sigma = self._params_to_tuple(params)
        return {
            "mean": mu, 
            "median": mu, 
            "mode": mu, 
            "cov": sigma, 
            "skewness": jnp.zeros_like(mu),}

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
        return self._params_dict(mu=d["mu"], sigma=d["sigma"])
        

mvt_normal = MvtNormal("Mvt-Normal")

# TODO: i believe i have finished updating mvt_normal. check if works correctly and if so move onto other mvts