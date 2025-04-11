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
    def _params_dict(nu: Scalar, mu: Scalar, sigma: Scalar) -> dict:
        d: dict = {"nu": nu, "mu": mu, "sigma": sigma}
        return StudentTBase._args_transform(d)
    
    @staticmethod
    def _params_to_tuple(params: dict) -> tuple:
        params = StudentTBase._args_transform(params)
        return params["nu"], params["mu"], params["sigma"]
    
    @staticmethod
    def _params_to_array(params: dict) -> Array:
        return jnp.asarray(StudentTBase._params_to_tuple(params)).flatten()
    
    def example_params(self, *args, **kwargs) -> dict:
        r"""Example parameters for the student-T distribution.
        
        This is a three parameter family, with the student-T being defined by 
        its degrees of freedom `nu`, location `mu` and scale `sigma`.
        """
        return self._params_dict(nu=2.5, mu=0.0, sigma=1.0)
    
    @staticmethod
    def support(*args, **kwargs) -> Array:
        return jnp.array([-jnp.inf, jnp.inf])
    
    @staticmethod
    def _stable_logpdf(stability: Scalar, x: ArrayLike, params: dict) -> Array:
        x, xshape = _univariate_input(x)
        nu, mu, sigma = StudentTBase._params_to_tuple(params)

        z: jnp.ndarray = lax.div(lax.sub(x, mu), sigma)

        const: jnp.ndarray = special.gammaln(0.5 * (nu + 1)) - special.gammaln(0.5 * nu) - 0.5 * lax.log(stability + (nu * jnp.pi * sigma))
        e: jnp.ndarray = lax.mul(lax.log(stability + lax.add(1.0, lax.div(lax.pow(z, 2.0), nu))), -0.5 * (nu + 1))
        logpdf = lax.add(const, e)
        return logpdf.reshape(xshape)
    
    @staticmethod
    def logpdf(x: ArrayLike, params: dict) -> Array:
        return StudentTBase._stable_logpdf(stability=0.0, x=x, params=params)
    
    @staticmethod
    def pdf(x: ArrayLike, params: dict) -> Array:
        return jnp.exp(StudentTBase.logpdf(x=x, params=params))
    
    # ppf
    def _get_x0(self, params: dict) -> Scalar:
        return self._args_transform(params)["mu"]

    # sampling
    def rvs(self, size: tuple | Scalar, params: dict, 
            key: Array = DEFAULT_RANDOM_KEY) -> Array:
        nu, mu, sigma = self._params_to_tuple(params)
        z: jnp.ndarray = random.t(key=key, df=nu, shape=size)
        return lax.add(lax.mul(z, sigma), mu)
    
    # stats
    def stats(self, params: dict) -> dict:
        nu, mu, sigma = self._params_to_tuple(params)
        mean: float = jnp.where(nu > 1, mu, jnp.nan)
        variance: float = jnp.where(nu > 2, nu / (nu - 2), jnp.nan)
        std: float = jnp.sqrt(variance)
        skewness: float = jnp.where(nu > 3, 0.0, jnp.nan)
        kurtosis: float = jnp.where(nu > 4, 6 / (nu - 4), jnp.inf)
        kurtosis = jnp.where(nu <= 2, jnp.nan, kurtosis)

        return {"mean": mean, "median": mu, "mode": mu, "variance": variance, 
                "std": std, "skewness": skewness, "kurtosis": kurtosis}
    
    # fitting
    @staticmethod
    def _params_from_array(params_arr, *args, **kwargs):
        nu, mu, sigma = params_arr
        return StudentTBase._args_transform({"nu": nu, "mu": mu, "sigma": sigma})

    def _fit_mle(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        eps = 1e-8
        constraints: tuple = (jnp.array([[eps, -jnp.inf, eps]]).T, 
                            jnp.array([[jnp.inf, jnp.inf, jnp.inf]]).T)
        
        projection_options: dict = {'hyperparams': constraints}
        params0: jnp.ndarray = jnp.array([
            jnp.exp(random.normal(key=DEFAULT_RANDOM_KEY) * 2.5), 
            x.mean(), x.std()])

        res = projected_gradient(
            f=self._mle_objective, x0=params0, projection_method='projection_box', 
            projection_options=projection_options, x=x, lr=lr, maxiter=maxiter)
        nu, mu, sigma = res['x']

        return self._params_dict(nu=nu, mu=mu, sigma=sigma)
    
    def _ldmle_objective(self, params: jnp.ndarray, x: jnp.ndarray, sample_mean: Scalar) -> jnp.ndarray:
        nu, sigma = params
        return self._mle_objective(params=jnp.array([nu, sample_mean, sigma]), x=x)
    
    def _fit_ldmle(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        params0: jnp.ndarray = jnp.array([1.0, x.std()])
        sample_mean: float = x.mean()
        res = projected_gradient(
            f=self._ldmle_objective, x0=params0, 
            projection_method='projection_non_negative', x=x, 
            sample_mean=sample_mean, lr=lr, maxiter=maxiter)
        nu, sigma = res['x']

        return self._params_dict(nu=nu, mu=sample_mean, sigma=sigma)
    
    def fit(self, x: ArrayLike, method: str = 'LDMLE', lr: float = 1.0, 
            maxiter: int = 100,) -> dict:
        r"""Fit the distribution to the input data.

        Note:
            If you intend to jit wrap this function, ensure that 'method' is a 
            static argument.
        
        Args:
            x (ArrayLike): The input data to fit the distribution to.
            method (str): The fitting method to use.  Options are 
                'MLE' for maximum likelihood estimation, and 'LDMLE' 
                for low-dimensional maximum likelihood estimation. 
                Defaults to 'LDMLE'. 
            lr (float): Learning rate for the fitting process.
            maxiter (int): Maximum number of iterations for the fitting process.
        
        Returns:
            dict: The fitted distribution parameters.
        """
        x = _univariate_input(x)[0]
        if method == 'MLE':
            return self._fit_mle(x, lr=lr, maxiter=maxiter)
        else:
            return self._fit_ldmle(x, lr=lr, maxiter=maxiter)
        
    # cdf
    @staticmethod
    def _pdf_for_cdf(x: ArrayLike, *params_tuple) -> Array:
        params_array: jnp.ndarray = jnp.asarray(params_tuple).flatten()
        params: dict = StudentTBase._params_from_array(params_array)
        return StudentTBase.pdf(x=x, params=params)
    
def _vjp_cdf(x: ArrayLike, params: dict) -> Array:
    params = StudentTBase._args_transform(params)
    return _cdf(dist=StudentTBase, x=x, params=params)


_vjp_cdf_copy = deepcopy(_vjp_cdf)
_vjp_cdf = custom_vjp(_vjp_cdf)


def cdf_fwd(x: ArrayLike, params: dict) -> tuple[Array, tuple]:
    params = StudentTBase._args_transform(params)
    return _cdf_fwd(dist=StudentTBase, cdf_func=_vjp_cdf_copy, x=x, params=params)


_vjp_cdf.defvjp(cdf_fwd, cdf_bwd)


class StudentT(StudentTBase):
    r"""The student-T distribution is a 3 parameter family of continuous 
    distributions which generalize the normal distribuion, allowing it to have 
    heavier tails.

    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    """
    def cdf(self, x: ArrayLike, params: dict) -> Array:
        return _vjp_cdf(x=x, params=params)
    

student_t = StudentT("Student-T")