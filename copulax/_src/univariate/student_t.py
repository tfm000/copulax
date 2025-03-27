"""File containing the copulAX implementation of the student-T distribution."""
import jax.numpy as jnp
from jax import lax, custom_vjp, random
from jax.scipy import special
from jax._src.typing import ArrayLike, Array
from copy import deepcopy

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.univariate._cdf import _cdf, cdf_bwd, _cdf_fwd
from copulax._src.optimize import projected_gradient


class StudentTBase(Univariate):
    @staticmethod
    def _params_dict(nu, mu, sigma):
        nu, mu, sigma = StudentTBase._args_transform(nu, mu, sigma)
        return {'nu': nu, 'mu': mu, 'sigma': sigma}
    
    @staticmethod
    def support(*args, **kwargs) -> Array:
        return jnp.array([-jnp.inf, jnp.inf])
    
    @staticmethod
    def _stable_logpdf(stability: Scalar, x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        x, xshape = _univariate_input(x)
        nu, mu, sigma = StudentTBase._args_transform(nu, mu, sigma)

        z: jnp.ndarray = lax.div(lax.sub(x, mu), sigma)

        const: jnp.ndarray = special.gammaln(0.5 * (nu + 1)) - special.gammaln(0.5 * nu) - 0.5 * lax.log(stability + (nu * jnp.pi * sigma))
        e: jnp.ndarray = lax.mul(lax.log(stability + lax.add(1.0, lax.div(lax.pow(z, 2.0), nu))), -0.5 * (nu + 1))
        logpdf = lax.add(const, e)
        return logpdf.reshape(xshape)
    
    @staticmethod
    def logpdf(x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return StudentTBase._stable_logpdf(stability=0.0, x=x, nu=nu, mu=mu, sigma=sigma)
    
    @staticmethod
    def pdf(x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return jnp.exp(StudentTBase.logpdf(x=x, nu=nu, mu=mu, sigma=sigma))
    
    def logcdf(self, x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return super().logcdf(x=x, nu=nu, mu=mu, sigma=sigma)
    
    def ppf(self, q: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return super().ppf(x0=mu, q=q, nu=nu, mu=mu, sigma=sigma)
    
    def inverse_cdf(self, q: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return super().inverse_cdf(q=q, nu=nu, mu=mu, sigma=sigma)

    # sampling
    def rvs(self, size: tuple | Scalar= (), key: Array = DEFAULT_RANDOM_KEY, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        nu, mu, sigma = self._args_transform(nu, mu, sigma)
        z: jnp.ndarray = random.t(key=key, df=nu, shape=size)
        return lax.add(lax.mul(z, sigma), mu)

    def sample(self, size: tuple | Scalar = (), key: Array = DEFAULT_RANDOM_KEY, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return super().sample(size=size, key=key, nu=nu, mu=mu, sigma=sigma)
    
    # stats
    def stats(self, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> dict:
        nu, mu, sigma = self._args_transform(nu, mu, sigma)
        mean: float = jnp.where(nu > 1, mu, jnp.nan)
        variance: float = jnp.where(nu > 2, nu / (nu - 2), jnp.nan)
        skewness: float = jnp.where(nu > 3, 0.0, jnp.nan)
        kurtosis: float = jnp.where(nu > 4, 6 / (nu - 4), jnp.inf)
        kurtosis = jnp.where(nu <= 2, jnp.nan, kurtosis)

        return {"mean": mean, "median": mu, "mode": mu, "variance": variance, 
                "skewness": skewness, "kurtosis": kurtosis}
    
    # metrics
    def loglikelihood(self, x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Scalar:
        return super().loglikelihood(x=x, nu=nu, mu=mu, sigma=sigma)
    
    def aic(self, x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Scalar:
        return super().aic(x=x, nu=nu, mu=mu, sigma=sigma)
    
    def bic(self, x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Scalar:
        return super().bic(x=x, nu=nu, mu=mu, sigma=sigma)
    
    # fitting
    def _fit_mle(self, x: jnp.ndarray, *args, **kwargs) -> dict:
        eps = 1e-8
        constraints: tuple = (jnp.array([[eps, -jnp.inf, eps]]).T, 
                            jnp.array([[jnp.inf, jnp.inf, jnp.inf]]).T)
        
        projection_options: dict = {'hyperparams': constraints}
        params0: jnp.ndarray = jnp.array([1.0, x.mean(), x.std()])

        res = projected_gradient(
            f=self._mle_objective, x0=params0, projection_method='projection_box', 
            projection_options=projection_options, x=x)
        nu, mu, sigma = res['x']

        return self._params_dict(nu=nu, mu=mu, sigma=sigma)#, res['val']
    
    def _ldmle_objective(self, params: jnp.ndarray, x: jnp.ndarray, mu: Scalar) -> jnp.ndarray:
        nu, sigma = params
        return self._mle_objective(params=jnp.array([nu, mu, sigma]), x=x)
    
    def _fit_ldmle(self, x: jnp.ndarray, *args, **kwargs) -> dict:
        params0: jnp.ndarray = jnp.array([1.0, x.std()])
        sample_mean: float = x.mean()
        res = projected_gradient(
            f=self._ldmle_objective, x0=params0, 
            projection_method='projection_non_negative', x=x, mu=sample_mean)
        nu, sigma = res['x']

        return self._params_dict(nu=nu, mu=sample_mean, sigma=sigma)#, res['val']
    
    def fit(self, x: ArrayLike, method: str = 'LDMLE', *args, **kwargs) -> dict:
        """Fit the distribution to the input data.

        Note:
            If you intend to jit wrap this function, ensure that 'method' is a 
            static argument.
        
        Args:
            x (ArrayLike): The input data to fit the distribution to.
            method (str): The fitting method to use.  Options are 
            'MLE' for maximum likelihood estimation, and 'LDMLE' for low-dimensional 
            maximum likelihood estimation. Defaults to 'LDMLE'. 
            kwargs: Additional keyword arguments to pass to the fit method.
        
        Returns:
            dict: The fitted distribution parameters.
        """
        x = _univariate_input(x)[0]
        if method == 'MLE':
            return self._fit_mle(x, *args, **kwargs)
        else:
            return self._fit_ldmle(x, *args, **kwargs)

# cdf
def _vjp_cdf(x: ArrayLike, nu: Scalar, mu: Scalar, sigma: Scalar) -> Array:
    params: dict = StudentTBase._params_dict(nu=nu, mu=mu, sigma=sigma)
    return _cdf(pdf_func=StudentTBase.pdf, lower_bound=StudentTBase.support()[0], 
                x=x, params=params)


_vjp_cdf_copy = deepcopy(_vjp_cdf)
_vjp_cdf = custom_vjp(_vjp_cdf)


def cdf_fwd(x: ArrayLike, nu, mu, sigma) -> tuple[Array, tuple]:
    params: dict = StudentTBase._params_dict(nu=nu, mu=mu, sigma=sigma)
    cdf_vals, (pdf_vals, param_grads) = _cdf_fwd(pdf_func=StudentTBase.pdf, cdf_func=_vjp_cdf_copy, x=x, params=params)
    return cdf_vals, (pdf_vals, param_grads['nu'], param_grads['mu'], param_grads['sigma'])


_vjp_cdf.defvjp(cdf_fwd, cdf_bwd)


class StudentT(StudentTBase):
    r"""The student-T distribution is a 3 parameter family of continuous 
    distributions which generalize the normal distribuion, allowing it to have 
    heavier tails.

    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    """
    def cdf(self, x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0) -> Array:
        return _vjp_cdf(x=x, nu=nu, mu=mu, sigma=sigma)
    

student_t = StudentT("Student-T")