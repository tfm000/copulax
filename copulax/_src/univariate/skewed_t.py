"""File containing the copuLAX implementation of the skewed-T distribution."""
import jax.numpy as jnp
from jax import lax, custom_vjp, random
from jax._src.typing import ArrayLike, Array
from copy import deepcopy

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax.special import kv
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.univariate._cdf import _cdf, cdf_bwd, _cdf_fwd
from copulax._src.optimize import projected_gradient
from copulax._src.univariate.student_t import student_t
from copulax._src.univariate.ig import ig
from copulax._src.univariate._mean_variance import mean_variance_ldmle_params, mean_variance_stats
from copulax._src.univariate._rvs import mean_variance_sampling


class SkewedTBase(Univariate):
    @staticmethod
    def _params_dict(nu: Scalar, mu: Scalar, sigma: Scalar, 
                     gamma: Scalar) -> dict:
        d: dict = {"nu": nu, "mu": mu, "sigma": sigma, "gamma": gamma}
        return SkewedTBase._args_transform(d)
    
    @staticmethod
    def _params_to_tuple(params: dict) -> tuple:
        params = SkewedTBase._args_transform(params)
        return params["nu"], params["mu"], params["sigma"], params["gamma"]
    
    @staticmethod
    def _params_to_array(params: dict) -> Array:
        return jnp.asarray(SkewedTBase._params_to_tuple(params)).flatten()

    @staticmethod
    def support(*args, **kwargs) -> Array:
        return jnp.array([-jnp.inf, jnp.inf])
    

    def example_params(self, *args, **kwargs) -> dict:
        r"""Example parameters for the skewed-T distribution.
        
        This is a four parameter family, with the skewed-T being defined 
        by its degrees of freedom `nu`, location `mu`, scale `sigma` and 
        skewness `gamma`. It is a generalisation of the student-T 
        distribution, which it includes as a special case when gamma is 
        zero. Here, we adopt the parameterization used by McNeil et al. 
        (2005).
        """
        return self._params_dict(nu=2.5, mu=0.0, sigma=1.0, gamma=1.0)
    
    @staticmethod
    def _stable_logpdf(stability: float, x: ArrayLike, params: dict) -> Array:
        nu, mu, sigma, gamma = SkewedTBase._params_to_tuple(params)
        x, xshape = _univariate_input(x)

        s: float = 0.5 * (nu + 1)
        c: float = (jnp.log(2.0) * (1 - s) 
                    - lax.lgamma(0.5 * nu) 
                    - 0.5 * jnp.log(jnp.pi * nu + stability) 
                    - jnp.log(sigma + stability))

        P: jnp.ndarray = (x - mu) * lax.pow(sigma, -2)
        Q: jnp.ndarray = P * (x - mu)
        R: jnp.ndarray = lax.pow(gamma / sigma, 2)

        T: jnp.ndarray = (jnp.log(kv(s, lax.sqrt((nu + Q) * R)) + stability) 
                          + P * gamma)
        B: jnp.ndarray = (-s * 0.5 * jnp.log((nu + Q) * R + stability) 
                          + s * jnp.log(1 + Q / nu) + stability)

        logpdf: jnp.ndarray = c + T - B
        return logpdf.reshape(xshape)

    @staticmethod
    def _unnormalized_logpdf(x: ArrayLike, params: dict, stability: float = 0.0) -> Array:
        gamma: Scalar = params["gamma"]
        return lax.cond(gamma == 0, 
                        lambda x: student_t.logpdf(x=x, params=params), 
                        lambda x: SkewedTBase._stable_logpdf(x=x, params=params, stability=stability), x)

    @staticmethod
    def _unnormalized_pdf(x: ArrayLike, params: dict, stability: float = 0.0) -> Array:
        return jnp.exp(SkewedTBase._unnormalized_logpdf(x=x, params=params, stability=stability))
    
    @staticmethod
    def logpdf(x: ArrayLike, params: dict) -> Array:
        params = SkewedTBase._args_transform(params)
        normalising_constant: float = _cdf(dist=SkewedTBase, x=jnp.inf, params=params)
        return SkewedTBase._unnormalized_logpdf(x=x, params=params) - jnp.log(normalising_constant)
    
    @staticmethod
    def pdf(x: ArrayLike, params: dict) -> Array:
        return jnp.exp(SkewedTBase.logpdf(x=x, params=params))
    
    # ppf
    def _get_x0(self, params: dict) -> Scalar:
        return self._args_transform(params)["mu"]
    
    # sampling
    def rvs(self, size: tuple | Scalar, params: dict, key: Array = DEFAULT_RANDOM_KEY) -> Array:
        nu, mu, sigma, gamma = self._params_to_tuple(params)
        
        key1, key2 = random.split(key)
        W: jnp.ndarray = ig.rvs(size=size, key=key1, alpha=nu*0.5, beta=nu*0.5)
        return mean_variance_sampling(key=key2, W=W, shape=size, mu=mu, sigma=sigma, gamma=gamma)
    
    # stats
    def _get_w_stats(self, nu: float) -> dict:
        ig_stats: dict = ig.stats(params={'alpha': nu*0.5, 'beta': nu*0.5})
        w_mean =  jnp.where(jnp.isnan(ig_stats['mean']), ig_stats['mode'], ig_stats['mean'])
        w_variance = jnp.where(jnp.isnan(ig_stats['variance']), w_mean ** 2, ig_stats['variance'])
        return {'mean': w_mean, 'variance': w_variance}

    def stats(self, params: dict) -> dict:
        nu, mu, sigma, gamma = self._params_to_tuple(params)
        w_stats: dict = self._get_w_stats(nu)
        return mean_variance_stats(mu=mu, sigma=sigma, gamma=gamma, w_stats=w_stats)
    
    # fitting
    def _mle_objective(self, params_arr: jnp.ndarray, x: jnp.ndarray, 
                       *args, **kwargs) -> Scalar:
        # overriding base method to use unnormalized-loglikelihood for faster iterations
        params: dict = self._params_from_array(params_arr, *args, **kwargs)
        return self._unnormalized_logpdf(x=x, params=params, stability=1e-30).sum()

    def _fit_mle(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        eps: float = 1e-8
        constraints: tuple = (jnp.array([[eps, -jnp.inf, eps, -jnp.inf]]).T, 
                            jnp.array([[jnp.inf, jnp.inf, jnp.inf, jnp.inf]]).T)
        
        projection_options: dict = {'hyperparams': constraints}

        key1, key2 = random.split(DEFAULT_RANDOM_KEY)
        params0: jnp.ndarray = jnp.array([
            jnp.abs(random.normal(key1, ())), 
            x.mean(),
            x.std(),
            random.normal(key2, ())])
        
        params0: jnp.ndarray = jnp.array([
            1.0, 
            x.mean(),
            x.std(),
            1.0])
        
        res: dict = projected_gradient(f=self._mle_objective, x0=params0,
                                    projection_method='projection_box', 
                                    projection_options=projection_options, 
                                    x=x, lr=lr, maxiter=maxiter)
        nu, mu, sigma, gamma = res['x']
        return self._params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)#, res['fun']

    def _ldmle_objective(self, params: jnp.ndarray, x: jnp.ndarray, sample_mean: Scalar, sample_variance: Scalar) -> jnp.ndarray:
        nu, gamma = params
        ig_stats: dict = self._get_w_stats(nu=nu)
        mu, sigma = mean_variance_ldmle_params(
            stats=ig_stats, gamma=gamma, sample_mean=sample_mean, 
            sample_variance=sample_variance)
        return self._mle_objective(params_arr=jnp.array([nu, mu, sigma, gamma]), x=x)
    
    def _fit_ldmle(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        eps: float = 1e-8
        min_nu: float = 4.0 
        constraints: tuple = (jnp.array([[min_nu + eps, -jnp.inf]]).T, 
                            jnp.array([[jnp.inf, jnp.inf]]).T)

        key1, key2 = random.split(DEFAULT_RANDOM_KEY)
        params0: jnp.ndarray = jnp.array([min_nu+jnp.abs(random.normal(key1, ())), 
                                        random.normal(key2, ())])
        
        projection_options: dict = {'hyperparams': constraints}

        sample_mean, sample_variance = x.mean(), x.var()
        res: dict = projected_gradient(
            f=self._ldmle_objective, x0=params0, projection_method='projection_box', 
            projection_options=projection_options, x=x, sample_mean=sample_mean, 
            sample_variance=sample_variance, lr=lr, maxiter=maxiter)
        nu, gamma = res['x']
        ig_stats: dict = self._get_w_stats(nu=nu)
        mu, sigma = mean_variance_ldmle_params(
            stats=ig_stats, gamma=gamma, sample_mean=sample_mean, 
            sample_variance=sample_variance)
        return self._params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)#, res['fun']

    def fit(self, x: ArrayLike, method: str = 'LDMLE', 
            lr=1.0, maxiter: int = 100) -> dict:
        r"""Fit the distribution to the input data.

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
            return self._fit_mle(x=x, lr=lr, maxiter=maxiter)
        else:
            return self._fit_ldmle(x=x, lr=lr, maxiter=maxiter)
        
    # cdf
    @staticmethod
    def _params_from_array(params_arr: jnp.ndarray, *args, **kwargs) -> dict:
        nu, mu, sigma, gamma = params_arr
        return SkewedTBase._params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)

    @staticmethod
    def _pdf_for_cdf(x: ArrayLike, *params_tuple) -> Array:
        params_array: jnp.ndarray = jnp.asarray(params_tuple).flatten()
        params: dict = SkewedTBase._params_from_array(params_array)
        return SkewedTBase._unnormalized_pdf(x=x, params=params)


def _vjp_cdf(x: ArrayLike, params: dict) -> Array:
    params = SkewedTBase._args_transform(params)


    normalising_constant: Scalar = _cdf(dist=SkewedTBase, x=jnp.inf, params=params)
    cdf: jnp.ndarray = _cdf(dist=SkewedTBase, x=x, params=params) / normalising_constant

    # TODO: just append inf to x
    return jnp.where(cdf > 1.0, 1.0, cdf)


_vjp_cdf_copy = deepcopy(_vjp_cdf)
_vjp_cdf = custom_vjp(_vjp_cdf)


def cdf_fwd(x: ArrayLike, params: dict) -> tuple[Array, tuple]:
    params = SkewedTBase._args_transform(params)
    return _cdf_fwd(dist=SkewedTBase, cdf_func=_vjp_cdf_copy, x=x, params=params)


_vjp_cdf.defvjp(cdf_fwd, cdf_bwd)


class SkewedT(SkewedTBase):
    r"""The skewed-t distribution is a generalisation of the continuous Student's 
    t-distribution that allows for skewness. It can also be expressed as a limiting 
    case of the Generalized Hyperbolic distribution when phi -> 0 in addition to 
    lambda = -0.5*chi.

    We use the 4 parameter McNeil et al (2005) specification of the distribution.
    """
    def cdf(self, x: ArrayLike, params: dict) -> Array:
        return _vjp_cdf(x=x, params=params)
    

skewed_t = SkewedT("Skewed-T")