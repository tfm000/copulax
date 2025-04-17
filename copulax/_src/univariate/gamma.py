"""File containing the copulAX implementation of the Gamma distribution."""
import jax.numpy as jnp
from jax import lax, random, scipy
from jax._src.typing import ArrayLike, Array
from tensorflow_probability.substrates import jax as tfp

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.optimize import projected_gradient
from copulax._src.univariate.lognormal import lognormal


class Gamma(Univariate):
    r"""The gamma distribution is a two-parameter family of continuous probability
    distributions, which includes the exponential, Erlang and chi-squared 
    distributions as special cases.

    We use the rate parameterization of the gamma distribution specified by 
    McNeil et al (2005).

    https://en.wikipedia.org/wiki/Gamma_distribution"""
    def _params_dict(self, alpha: Scalar, beta: Scalar) -> dict:
        d: dict = {"alpha": alpha, "beta": beta}
        return self._args_transform(d)

    def _params_to_tuple(self, params: dict):
        params = self._args_transform(params)
        return params["alpha"], params["beta"]
    
    def example_params(self, *args, **kwargs):
        r"""Example parameters for the gamma distribution.
        
        This is a two parameter family, defined by alpha and beta 
        parameters. Here we adopt the rate parameterization of the gamma.
        """
        return self._params_dict(alpha=1.0, beta=1.0)
    
    def support(self, *args, **kwargs) -> Array:
        return jnp.array([0.0, jnp.inf])
    
    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, params: dict) -> Array:
        x, xshape = _univariate_input(x)
        alpha, beta = self._params_to_tuple(params)
        
        logpdf: jnp.ndarray = (alpha * jnp.log(beta + stability) 
                               - lax.lgamma(alpha) 
                               + (alpha - 1) * jnp.log(x) 
                               - beta * x)
        return logpdf.reshape(xshape)

    def logpdf(self, x: ArrayLike, params: dict) -> Array:
        return super().logpdf(x=x, params=params)

    def pdf(self, x: ArrayLike, params: dict) -> Array:
        return super().pdf(x=x, params=params)
    
    def logcdf(self, x: ArrayLike, params: dict) -> Array:
        return super().logcdf(x=x, params=params)
    
    def cdf(self, x: ArrayLike, params: dict) -> Array:
        x, xshape = _univariate_input(x)
        alpha, beta = self._params_to_tuple(params)
        cdf: jnp.ndarray = scipy.special.gammainc(a=alpha, x=beta*x)
        return cdf.reshape(xshape)
    
    # ppf
    def _get_x0(self, params: dict):
        return self.stats(params=params)["mean"]

    def _ppf(self, q: ArrayLike, params: dict, *args, **kwargs) -> Array:
        alpha, beta = self._params_to_tuple(params)
        return tfp.math.igammainv(a=alpha, p=q) / beta
    
    # sampling
    def rvs(self, size: tuple | Scalar, params: dict, key: Array = DEFAULT_RANDOM_KEY) -> Array:
        alpha, beta = self._params_to_tuple(params)
        unscales_rvs: jnp.ndarray = random.gamma(key, shape=size, a=alpha)
        return unscales_rvs / beta
    
    # stats
    def stats(self, params: dict) -> dict:
        alpha, beta = self._params_to_tuple(params)
        mean: float = alpha / beta
        mode: float = jnp.where(alpha >= 1.0, (alpha - 1) / beta, 0.0)
        variance: float = alpha / (beta ** 2)
        std: float = jnp.sqrt(variance)
        skewness: float = 2.0 / jnp.sqrt(alpha)
        kurtosis: float = 6.0 / alpha
        return {"mean": mean, "mode": mode, "variance": variance, "std": std, 
                "skewness": skewness, "kurtosis": kurtosis}
    
    # fitting
    # def _params_from_array(self, params_arr, *args, **kwargs):
    #     alpha, beta = params_arr
    #     return self._args_transform({})

    def _fit_mle(self, x: ArrayLike, lr: float, maxiter: int) -> dict:
        beta0: float = self.rvs(size=(), params=self.example_params())
        alpha0: float = x.mean() * beta0
        params0: jnp.ndarray = jnp.array([alpha0, beta0])

        res = projected_gradient(f=self._mle_objective, x0=params0, 
                                projection_method='projection_non_negative', 
                                x=x, lr=lr, maxiter=maxiter)
        alpha, beta = res['x']
        return self._params_dict(alpha=alpha, beta=beta)#, res['fun']
    
    def fit(self, x: ArrayLike, lr: float = 1.0, maxiter: int = 100) -> dict:
        r"""Fit the distribution to the input data.
        
        Args:
            x (ArrayLike): The input data to fit the distribution to.
            lr (float): Learning rate for the fitting process.
            maxiter (int): Maximum number of iterations for the fitting process.
        
        Returns:
            dict: The fitted distribution parameters.
        """
        x: jnp.ndarray = _univariate_input(x)[0]
        return self._fit_mle(x=x, lr=lr, maxiter=maxiter)
    

gamma = Gamma("Gamma")