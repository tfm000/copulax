"""File containing the copulAX implementation of the Gamma distribution."""
import jax.numpy as jnp
from jax import lax, random, scipy
from jax._src.typing import ArrayLike, Array
from tensorflow_probability.substrates import jax as tfp

from copulax._src.univariate._distributions import Univariate
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
        alpha, beta = self._args_transform(alpha, beta)
        return {"alpha": alpha, "beta": beta}
    
    def support(self, *args, **kwargs) -> tuple[Scalar, Scalar]:
        return jnp.array(0.0), jnp.array(jnp.inf)
    
    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, alpha: Scalar = 1.0, beta: Scalar = 1.0) -> Array:
        x, xshape = _univariate_input(x)
        alpha, beta = self._args_transform(alpha, beta)
        
        logpdf: jnp.ndarray = (alpha * jnp.log(beta + stability) - lax.lgamma(alpha) + (alpha - 1) * jnp.log(x) - beta * x)
        return logpdf.reshape(xshape)

    def logpdf(self, x: ArrayLike, alpha: Scalar = 1.0, beta: Scalar = 1.0) -> Array:
        return super().logpdf(x=x, alpha=alpha, beta=beta)

    def pdf(self, x: ArrayLike, alpha: Scalar = 1.0, beta: Scalar = 1.0) -> Array:
        return super().pdf(x=x, alpha=alpha, beta=beta)
    
    def logcdf(self, x: ArrayLike, alpha: Scalar = 1.0, beta: Scalar = 1.0) -> Array:
        return super().logcdf(x=x, alpha=alpha, beta=beta)
    
    def cdf(self, x: ArrayLike, alpha: Scalar = 1.0, beta: Scalar = 1.0) -> Array:
        x, xshape = _univariate_input(x)
        alpha, beta = self._args_transform(alpha, beta)
        cdf: jnp.ndarray = scipy.special.gammainc(a=alpha, x=beta*x)
        return cdf.reshape(xshape)
    
    def ppf(self, q: ArrayLike, alpha: Scalar = 1.0, beta: Scalar = 1.0) -> Array:
        q, qshape = _univariate_input(q)
        alpha, beta = self._args_transform(alpha, beta)
        ppf: jnp.ndarray = tfp.math.igammainv(a=alpha, p=q) / beta
        return ppf.reshape(qshape)
    
    def inverse_cdf(self, q: ArrayLike, alpha: Scalar = 1.0, beta: Scalar = 1.0) -> Array:
        return super().inverse_cdf(q=q, alpha=alpha, beta=beta)
    
    # sampling
    def rvs(self, shape=(), key: Array = DEFAULT_RANDOM_KEY, alpha: Scalar = 1.0, beta: Scalar = 1.0) -> Array:
        alpha, beta = self._args_transform(alpha, beta)

        unscales_rvs: jnp.ndarray = random.gamma(key, shape=shape, a=alpha)
        return unscales_rvs / beta
    
    def sample(shape=(), key: Array = DEFAULT_RANDOM_KEY, alpha: Scalar = 1.0, beta: Scalar = 1.0) -> Array:
        return super().sample(shape=shape, key=key, alpha=alpha, beta=beta)
    
    # stats
    def stats(self, alpha: Scalar = 1.0, beta: Scalar = 1.0) -> dict:
        alpha, beta = self._args_transform(alpha, beta)
        mean: float = alpha / beta
        mode: float = jnp.where(alpha >= 1.0, (alpha - 1) / beta, 0.0)
        variance: float = alpha / (beta ** 2)
        skewness: float = 2.0 / jnp.sqrt(alpha)
        kurtosis: float = 6.0 / alpha
        return {"mean": mean, "mode": mode, "variance": variance, "skewness": skewness, "kurtosis": kurtosis}
    
    # metrics
    def loglikelihood(self, x: ArrayLike, alpha: Scalar = 1.0, beta: Scalar = 1.0) -> Scalar:
        return super().loglikelihood(x=x, alpha=alpha, beta=beta)
    
    def aic(self, x: ArrayLike, alpha: Scalar = 1.0, beta: Scalar = 1.0) -> Scalar:
        return super().aic(x=x, alpha=alpha, beta=beta)
    
    def bic(self, x: ArrayLike, alpha: Scalar = 1.0, beta: Scalar = 1.0) -> Scalar:
        return super().bic(x=x, alpha=alpha, beta=beta)
    
    # fitting
    def _fit_mle(self, x: ArrayLike, *args, **kwargs) -> dict:
        beta0: float = lognormal.rvs(shape=())
        alpha0: float = x.mean() * beta0
        params0: jnp.ndarray = jnp.array([alpha0, beta0])

        res = projected_gradient(f=self._mle_objective, x0=params0, 
                                projection_method='projection_non_negative', x=x)
        alpha, beta = res['x']
        return self._params_dict(alpha=alpha, beta=beta)#, res['fun']
    
    def fit(self, x: ArrayLike, *args, **kwargs) -> dict:
        x: jnp.ndarray = _univariate_input(x)[0]
        return self._fit_mle(x=x, *args, **kwargs)
    

gamma = Gamma("Gamma")