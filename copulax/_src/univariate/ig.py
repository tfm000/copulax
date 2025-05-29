"""File containing the copulAX implementation of the Inverse Gaussian distribution."""
import jax.numpy as jnp
from jax import lax, random, scipy
from jax._src.typing import ArrayLike, Array
from tensorflow_probability.substrates import jax as tfp

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.optimize import projected_gradient
from copulax._src.univariate.gamma import gamma


class IG(Univariate):
    r"""The inverse gamma distribution is a two-parameter family of continuous 
    probability distributions which represents the reciprocal of gamma distributed 
    random variables.

    We use the rate parameterization of the inverse gamma distribution specified by
    McNeil et al (2005).

    https://en.wikipedia.org/wiki/Inverse-gamma_distribution
    """
    @classmethod
    def _params_dict(cls, alpha: Scalar, beta: Scalar) -> dict:
        d: dict = {"alpha": alpha, "beta": beta}
        return cls._args_transform(d)

    def _params_to_tuple(self, params):
        params = self._args_transform(params)
        return params["alpha"], params["beta"]
    
    def example_params(self, *args, **kwargs) -> dict:
        r"""Example parameters for the inverse gamma distribution.
        
        This is a two parameter family, with the inverse gamma being 
        defined by parameters `alpha` and `beta`.
        """
        return self._params_dict(alpha=1.0, beta=1.0)

    @classmethod
    def _support(cls, *args, **kwargs) -> tuple:
        return 0.0, jnp.inf
    
    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, params: dict) -> Array:
        x, xshape = _univariate_input(x)
        alpha, beta = self._params_to_tuple(params)
        
        logpdf: jnp.ndarray = (alpha * jnp.log(beta + stability) 
                               - lax.lgamma(alpha) 
                               - (alpha + 1) * jnp.log(x) 
                               - (beta / x))
        return logpdf.reshape(xshape)

    def cdf(self, x: ArrayLike, params: dict) -> Array:
        x, xshape = _univariate_input(x)
        alpha, beta = self._params_to_tuple(params)
        cdf: jnp.ndarray = scipy.special.gammaincc(a=alpha, x=(beta / x))
        return cdf.reshape(xshape)
    
    # ppf
    # def _get_x0(self, params):
    #     return self.stats(params=params)['mode']

    def _ppf(self, q: ArrayLike, params: dict, *args, **kwargs) -> Array:
        alpha, beta = self._params_to_tuple(params)
        return beta / tfp.math.igammacinv(a=alpha, p=q)
    
    # sampling
    def rvs(self, size: tuple | Scalar, params: dict, key: Array = DEFAULT_RANDOM_KEY) -> Array:
        return 1.0 / gamma.rvs(size=size, key=key, params=params)
    
    # stats
    def stats(self, params: dict) -> dict:
        alpha, beta = self._params_to_tuple(params)
        mean: float = jnp.where(alpha > 1.0, beta / (alpha - 1), jnp.nan)
        mode: float = beta / (alpha + 1)
        variance: float = jnp.where(alpha > 2.0, lax.pow(beta, 2) / (lax.pow(alpha - 1, 2) * (alpha - 1)), jnp.nan)
        std: float = jnp.sqrt(variance)
        skewness: float = jnp.where(alpha > 3.0, 4 * jnp.sqrt(alpha - 2) / (alpha - 3), jnp.nan)
        kurtosis: float = jnp.where(alpha > 4.0, 6 * (5*alpha - 11) / ((alpha - 3) * (alpha - 4)), jnp.nan)
        return self._scalar_transform({
            "mean": mean, 
            "mode": mode, 
            "variance": variance, 
            "std": std, 
            "skewness": skewness, 
            "kurtosis": kurtosis})
    
    # fitting
    def _fit_mle(self, x: ArrayLike, lr: float, maxiter: int) -> dict:
        key1, key2 = random.split(DEFAULT_RANDOM_KEY)

        gamma_params: dict = gamma.example_params()
        params0: jnp.ndarray = jnp.array([gamma.rvs(size=(), key=key1, params=gamma_params), 
                                        gamma.rvs(size=(), key=key2, params=gamma_params)])
        
        res = projected_gradient(
            f=self._mle_objective, x0=params0, x=x, lr=lr, maxiter=maxiter, 
            projection_method='projection_non_negative')
        
        alpha, beta = res["x"]
        return self._params_dict(alpha=alpha, beta=beta)#, res["fun"]
    
    def fit(self, x: ArrayLike, lr: float = 1.0, maxiter: int = 100) -> dict:
        x: jnp.ndarray = _univariate_input(x)[0]
        return self._fit_mle(x=x, lr=lr, maxiter=maxiter)
    

ig = IG("IG")