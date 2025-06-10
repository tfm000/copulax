"""File containing the copulAX implementation of the continuous uniform 
distribution."""
import jax.numpy as jnp
from jax import lax, random
from jax._src.typing import ArrayLike, Array

from copulax._src._distributions import Univariate
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.typing import Scalar


class Uniform(Univariate):
    r"""The continuous uniform distribution.
    
    The continuous uniform distribution is defined as:
    
    .. math::
    
        f(x|a, b) = \frac{1}{b - a}
    
    where :math:`a` is the lower bound of the distribution and :math:`b` is the 
    upper bound.
    """
    @classmethod
    def _params_dict(cls, a: Scalar, b: Scalar) -> dict:
        d: dict = {"a": a, "b": b}
        return cls._args_transform(d)
    
    @classmethod
    def _params_to_tuple(cls, params: dict) -> tuple:
        params = cls._args_transform(params)
        return params["a"], params["b"]
    
    def example_params(self, *args, **kwargs) -> dict:
        r"""Example parameters for the uniform distribution.
        
        This is a two parameter family, with the uniform being defined by 
        its lower and upper bounds.
        """
        return self._params_dict(a=0.0, b=1.0)
    
    @classmethod
    def _support(cls, params: dict) -> tuple:
        a, b = cls._params_to_tuple(params)
        return a, b

    def logpdf(self, x: ArrayLike, params: dict) -> Array:
        x, xshape = _univariate_input(x)
        a, b = self._params_to_tuple(params)

        log_pdf: jnp.ndarray = -jnp.log(lax.sub(b, a))
        log_pdf = jnp.where(jnp.logical_and(x >= a, x <= b), log_pdf, -jnp.inf)
        return log_pdf.reshape(xshape)
    
    def logcdf(self, x: ArrayLike, params : dict) -> Array:
        x, xshape = _univariate_input(x)
        a, b = self._params_to_tuple(params)

        log_cdf: jnp.ndarray = jnp.log(lax.sub(x, a)) - jnp.log(lax.sub(b, a))
        log_cdf = jnp.where(jnp.logical_and(x >= a, x <= b), log_cdf, -jnp.inf)
        return log_cdf.reshape(xshape)
    
    def cdf(self, x: ArrayLike, params: dict) -> Array:
        return jnp.exp(self.logcdf(x=x, params=params))
    
    # ppf
    # def _get_x0(self, params: dict) -> Scalar:
    #     stats: dict = self.stats(params=params)
    #     return stats["mean"]
    
    def _ppf(self, q: ArrayLike, params: dict, *args, **kwargs) -> Array:
        q, qshape = _univariate_input(q)
        a, b = self._params_to_tuple(params)
        
        ppf_values: jnp.ndarray = lax.add(a, lax.mul(q, lax.sub(b, a)))
        ppf_values = jnp.where(jnp.logical_and(q >= 0, q <= 1), ppf_values, jnp.nan)
        return ppf_values.reshape(qshape)
    
    # sampling
    def rvs(self, size: tuple | Scalar, params: dict, key=DEFAULT_RANDOM_KEY) -> Array:
        a, b = self._params_to_tuple(params)
        return random.uniform(key=key, shape=size, minval=a, maxval=b)
    
    # stats
    def stats(self, params: dict) -> dict:
        a, b = self._params_to_tuple(params)

        mean: Scalar = (a + b) / 2
        variance: Scalar = lax.pow(b - a, 2) / 12
        std: Scalar = jnp.sqrt(variance)
        return self._scalar_transform({
            'mean': mean, 
            'median': mean, 
            'variance': variance,
            'std': std, 
            'skewness': 0.0, 
            'kurtosis': -6 / 5})

    # fitting
    def fit(self, x: ArrayLike, *args, **kwargs) -> dict:
        x: jnp.ndarray = _univariate_input(x)[0]
        a: Scalar = jnp.min(x)
        b: Scalar = jnp.max(x)
        return self._params_dict(a=a, b=b)
    

uniform = Uniform("Uniform")