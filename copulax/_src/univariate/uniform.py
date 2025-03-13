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
    def _params_dict(self, a: Scalar, b: Scalar) -> dict:
        a, b = self._args_transform(a, b)
        return {"a": a, "b": b}
    
    def support(self, a: Scalar = 0.0, b: Scalar = 1.0) -> tuple[Scalar, Scalar]:
        return self._args_transform(a, b)

    def logpdf(self, x: ArrayLike, a: Scalar = 0.0, b: Scalar = 1.0) -> Array:
        x, xshape = _univariate_input(x)
        a, b = self._args_transform(a, b)

        log_pdf: jnp.ndarray = -jnp.log(lax.sub(b, a))
        log_pdf = jnp.where(jnp.logical_and(x >= a, x <= b), log_pdf, -jnp.inf)
        return log_pdf.reshape(xshape)
    
    def pdf(self, x: ArrayLike, a: Scalar = 0.0, b: Scalar = 1.0) -> Array:
        return super().pdf(x=x, a=a, b=b)
    
    def logcdf(self, x: ArrayLike, a: Scalar = 0.0, b: Scalar = 1.0) -> Array:
        x, xshape = _univariate_input(x)
        a, b = self._args_transform(a, b)

        log_cdf: jnp.ndarray = jnp.log(lax.sub(x, a)) - jnp.log(lax.sub(b, a))
        log_cdf = jnp.where(jnp.logical_and(x >= a, x <= b), log_cdf, -jnp.inf)
        return log_cdf.reshape(xshape)
    
    def cdf(self, x: ArrayLike, a: Scalar = 0.0, b: Scalar = 1.0) -> Array:
        return jnp.exp(self.logcdf(x=x, a=a, b=b))
    
    def ppf(self, q: ArrayLike, a: Scalar = 0.0, b: Scalar = 1.0) -> Array:
        q, qshape = _univariate_input(q)
        a, b = self._args_transform(a, b)
        
        ppf_values: jnp.ndarray = lax.add(a, lax.mul(q, lax.sub(b, a)))
        ppf_values = jnp.where(jnp.logical_and(q >= 0, q <= 1), ppf_values, jnp.nan)
        return ppf_values.reshape(qshape)
    
    def inverse_cdf(self, q: ArrayLike, a: Scalar = 0.0, b: Scalar = 1.0) -> Array:
        return super().inverse_cdf(q=q, a=a, b=b)
    
    # sampling
    def rvs(self, size: tuple | Scalar = (), key=DEFAULT_RANDOM_KEY, a: Scalar = 0.0, b: Scalar = 1.0) -> Array:
        a, b = self._args_transform(a, b)
        return random.uniform(key=key, shape=size, minval=a, maxval=b)
    
    def sample(self, size: tuple | Scalar = (), key=DEFAULT_RANDOM_KEY, a: Scalar = 0.0, b: Scalar = 1.0) -> Array:
        return super().sample(size=size, key=key, a=a, b=b)
    
    # stats
    def stats(self, a: Scalar = 0.0, b: Scalar = 1.0) -> dict:
        a, b = self._args_transform(a, b)

        mean: Scalar = (a + b) / 2
        variance: Scalar = lax.pow(b - a, 2) / 12
        return {'mean': mean, 'median': mean, 'variance': variance, 
                'skewness': 0.0, 'kurtosis': -6 / 5}
    
    # metrics
    def loglikelihood(self, x, a: Scalar = 0.0, b: Scalar = 1.0) -> Scalar:
        return super().loglikelihood(x, a=a, b=b)
    
    def aic(self, x, a: Scalar = 0.0, b: Scalar = 1.0) -> Scalar:
        return super().aic(x, a=a, b=b)
    
    def bic(self, x, a: Scalar = 0.0, b: Scalar = 1.0) -> Scalar:
        return super().bic(x, a=a, b=b)

    # fitting
    def fit(self, x: ArrayLike, a: Scalar = 0.0, b: Scalar = 1.0, *args, **kwargs) -> dict:
        x: jnp.ndarray = _univariate_input(x)[0]
        a: Scalar = jnp.min(x)
        b: Scalar = jnp.max(x)
        return self._params_dict(a=a, b=b)
    

uniform = Uniform("Uniform")