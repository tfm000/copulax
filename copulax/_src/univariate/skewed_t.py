"""File containing the copuLAX implementation of the skewed-T distribution."""
import jax.numpy as jnp
from jax import lax, custom_vjp, random
from jax._src.typing import ArrayLike, Array
from copy import deepcopy

from copulax._src.univariate._distributions import Univariate
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
    def _params_dict(nu, mu, sigma, gamma):
        nu, mu, sigma, gamma = SkewedTBase._args_transform(nu, mu, sigma, gamma)
        return {'nu': nu, 'mu': mu, 'sigma': sigma, 'gamma': gamma}

    @staticmethod
    def support(*args, **kwargs):
        return jnp.array(-jnp.inf), jnp.array(jnp.inf)
    
    @staticmethod
    def _stable_logpdf(stability: float, x: ArrayLike, nu: float, mu: float, sigma: float, gamma: float) -> Array:
        # stability term is here to be used when fitting, as for small values of 
        # sigma, log(kv) can blow up as kv -> 0.
        # including a small value to the kv term ensures that the logpdf does not 
        # blow up, but will lead to undersirable results when evaluating the 
        # pdf/cdf and sampling, hence it is only used when fitting.
        x, xshape = _univariate_input(x)

        s: float = 0.5 * (nu + 1)
        c: float = jnp.log(2.0) * (1 - s) - lax.lgamma(0.5 * nu) - 0.5 * jnp.log(jnp.pi * nu) - jnp.log(sigma)

        P: jnp.ndarray = (x - mu) * lax.pow(sigma, -2)
        Q: jnp.ndarray = P * (x - mu)
        R: jnp.ndarray = lax.pow(gamma / sigma, 2)

        T: jnp.ndarray = jnp.log(kv(s, lax.sqrt((nu + Q) * R)) + stability) + P * gamma
        B: jnp.ndarray = -s * 0.5 * jnp.log((nu + Q) * R) + s * jnp.log(1 + Q / nu)

        logpdf: jnp.ndarray = c + T - B
        return logpdf.reshape(xshape)

    @staticmethod
    def _unnormalized_logpdf(x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0, stability: float = 0.0) -> Array:
        nu, mu, sigma, gamma = SkewedTBase._args_transform(nu=nu, mu=mu, sigma=sigma, gamma=gamma)
        return lax.cond(gamma == 0, lambda x: student_t.logpdf(x=x, nu=nu, mu=mu, sigma=sigma), lambda x: SkewedTBase._stable_logpdf(x=x, nu=nu, mu=mu, sigma=sigma, gamma=gamma, stability=stability), x)

    @staticmethod
    def _unnormalized_pdf(x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0, stability: float = 0.0) -> Array:
        return jnp.exp(SkewedTBase._unnormalized_logpdf(x=x, nu=nu, mu=mu, sigma=sigma, gamma=gamma, stability=stability))
    
    @staticmethod
    def logpdf(x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        params = SkewedTBase._params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)

        support: tuple = SkewedTBase.support()
        normalising_constant: float = _cdf(pdf_func=SkewedTBase._unnormalized_pdf, lower_bound=support[0], x=support[1], params=params)
        return SkewedTBase._unnormalized_logpdf(x=x, nu=nu, mu=mu, sigma=sigma, gamma=gamma) - jnp.log(normalising_constant)
    
    @staticmethod
    def pdf(x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        return jnp.exp(SkewedTBase.logpdf(x=x, nu=nu, mu=mu, sigma=sigma, gamma=gamma))
    
    def logcdf(self, x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        return super().logcdf(x=x, nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    
    def ppf(self, q: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        mean: Scalar = self.stats(nu=nu, mu=mu, sigma=sigma, gamma=gamma)['mean']
        return super().ppf(x0=mean, q=q, nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    
    def inverse_cdf(self, q: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        return super().inverse_cdf(q=q, nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    
    # sampling
    def rvs(self, shape: tuple = (), key: Array = DEFAULT_RANDOM_KEY, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        nu, mu, sigma, gamma = self._args_transform(nu, mu, sigma, gamma)
        
        key1, key2 = random.split(key)
        W: jnp.ndarray = ig.rvs(shape=shape, key=key1, alpha=nu*0.5, beta=nu*0.5)
        return mean_variance_sampling(key=key2, W=W, shape=shape, mu=mu, sigma=sigma, gamma=gamma)
    
    def sample(self, shape: tuple = (), key: Array = DEFAULT_RANDOM_KEY, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        return super().sample(shape=shape, key=key, nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    
    # stats
    def _get_w_stats(self, nu: float) -> dict:
        ig_stats: dict = ig.stats(alpha=nu*0.5, beta=nu*0.5)
        w_mean =  jnp.where(jnp.isnan(ig_stats['mean']), ig_stats['mode'], ig_stats['mean'])
        w_variance = jnp.where(jnp.isnan(ig_stats['variance']), w_mean ** 2, ig_stats['variance'])
        return {'mean': w_mean, 'variance': w_variance}

    def stats(self, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> dict:
        nu, mu, sigma, gamma = self._args_transform(nu, mu, sigma, gamma)
        w_stats: dict = self._get_w_stats(nu)
        return mean_variance_stats(mu=mu, sigma=sigma, gamma=gamma, w_stats=w_stats)
    
    # metrics
    def loglikelihood(self, x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        return super().loglikelihood(x=x, nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    
    def aic(self, x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        return super().aic(x=x, nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    
    def bic(self, x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        return super().bic(x=x, nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    
    # fitting
    def _mle_objective(self, params: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        # overriding base method to use unnormalized-loglikelihood
        return self._unnormalized_logpdf(x, *params, stability=1e-30).sum()

    def _fit_mle(self, x: jnp.ndarray, *args, **kwargs) -> dict:
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
                                    projection_options=projection_options, x=x)
        nu, mu, sigma, gamma = res['x']
        return self._params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)#, res['fun']

    def _ldmle_objective(self, params: jnp.ndarray, x: jnp.ndarray, sample_mean: Scalar, sample_variance: Scalar) -> jnp.ndarray:
        nu, gamma = params
        ig_stats: dict = self._get_w_stats(nu=nu)
        mu, sigma = mean_variance_ldmle_params(
            stats=ig_stats, gamma=gamma, sample_mean=sample_mean, 
            sample_variance=sample_variance)
        return self._mle_objective(params=jnp.array([nu, mu, sigma, gamma]), x=x)
    
    def _fit_ldmle(self, x: jnp.ndarray, *args, **kwargs) -> dict:
        eps: float = 1e-8
        min_nu: float = 4.0 
        constraints: tuple = (jnp.array([[min_nu + eps, -jnp.inf]]).T, 
                            jnp.array([[jnp.inf, jnp.inf]]).T)

        key1, key2 = random.split(DEFAULT_RANDOM_KEY)
        params0: jnp.ndarray = jnp.array([min_nu+jnp.abs(random.normal(key1, ())), 
                                        random.normal(key2, ())])
        
        projection_options: dict = {'hyperparams': constraints}

        sample_mean, sample_variance = x.mean(), x.var()
        res: dict = projected_gradient(f=self._ldmle_objective, x0=params0,
                                    projection_method='projection_box', 
                                    projection_options=projection_options, x=x, 
                                    sample_mean=sample_mean, sample_variance=sample_variance)
        nu, gamma = res['x']
        ig_stats: dict = self._get_w_stats(nu=nu)
        mu, sigma = mean_variance_ldmle_params(
            stats=ig_stats, gamma=gamma, sample_mean=sample_mean, 
            sample_variance=sample_variance)
        return self._params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)#, res['fun']

    def fit(self, x: ArrayLike, method: str = 'LDMLE', *args, **kwargs):
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

# cdf - TODO: make this use unnormalised pdf and normalize within this.  then have cdf only call this. aka no normalization within
def _vjp_cdf(x: ArrayLike, nu: Scalar, mu: Scalar, sigma: Scalar, gamma: Scalar) -> Array:
    params: dict = SkewedTBase._params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)

    support: tuple = SkewedTBase.support()
    normalising_constant: float = _cdf(pdf_func=SkewedTBase._unnormalized_pdf, lower_bound=support[0], x=support[1], params=params)
    cdf: jnp.ndarray = _cdf(pdf_func=SkewedTBase._unnormalized_pdf, lower_bound=support[0], x=x, params=params) / normalising_constant
    return jnp.where(cdf > 1.0, 1.0, cdf)


_vjp_cdf_copy = deepcopy(_vjp_cdf)
_vjp_cdf = custom_vjp(_vjp_cdf)


def cdf_fwd(x: ArrayLike, nu: float, mu: float, sigma: float, gamma: float) -> tuple[Array, tuple]:
    params: dict = SkewedTBase._params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    cdf_vals, (pdf_vals, param_grads) = _cdf_fwd(pdf_func=SkewedTBase.pdf, cdf_func=_vjp_cdf_copy, x=x, params=params)
    return cdf_vals, (pdf_vals, param_grads['nu'], param_grads['mu'], param_grads['sigma'], param_grads['gamma'])


_vjp_cdf.defvjp(cdf_fwd, cdf_bwd)


class SkewedT(SkewedTBase):
    r"""The skewed-t distribution is a generalisation of the continuous Student's 
    t-distribution that allows for skewness. It can also be expressed as a limiting 
    case of the Generalized Hyperbolic distribution when phi -> 0 in addition to 
    lambda = -0.5*chi.

    We use the 4 parameter McNeil et al (2005) specification of the distribution.
    """
    def cdf(self, x: ArrayLike, nu: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        return _vjp_cdf(x=x, nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    

skewed_t = SkewedT("Skewed-T")