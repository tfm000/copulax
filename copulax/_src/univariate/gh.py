"""File containing the copulAX implementation of the generalized hyperbolic distribution."""
import jax.numpy as jnp
from jax import lax, custom_vjp, random
from jax._src.typing import ArrayLike, Array
from copy import deepcopy

from copulax._src._distributions import Univariate
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.typing import Scalar
from copulax._src.univariate._cdf import _cdf, cdf_bwd, _cdf_fwd
from copulax.special import kv
from copulax._src.univariate._rvs import mean_variance_sampling
from copulax._src.univariate._mean_variance import mean_variance_stats, mean_variance_ldmle_params
from copulax._src.univariate.gig import gig
from copulax._src.optimize import projected_gradient


class GHBase(Univariate):
    @classmethod
    def _params_dict(cls, lamb: Scalar, chi: Scalar, psi: Scalar, mu: Scalar, sigma: Scalar, gamma: Scalar) -> dict:
        r"""Convert parameters to a dictionary."""
        d: dict = {"lambda": lamb, "chi": chi, "psi": psi, 
                   "mu": mu, "sigma": sigma, "gamma": gamma}
        return cls._args_transform(d)
                
    @staticmethod
    def _params_to_tuple(params: dict) -> tuple:
        params = GHBase._args_transform(params)
        return (params["lambda"], params["chi"], params["psi"], 
                params["mu"], params["sigma"], params["gamma"])
    
    @staticmethod
    def _params_to_array(params: dict) -> Array:
        return jnp.asarray(GHBase._params_to_tuple(params)).flatten()
    
    @classmethod
    def _support(cls, *args, **kwargs) -> tuple:
        return -jnp.inf, jnp.inf
    
    def example_params(self, *args, **kwargs) -> dict:
        r"""Example parameters for the generalized hyperbolic 
        distribution.
        
        This is a six parameter family, with the generalized hyperbolic
        being defined by location `mu`, dispersion `sigma` and skewness 
        `gamma` in addition to `lambda`, `chi` and `psi` shape parameters. 
        Here, we adopt the parameterization used by McNeil et al. (2005).
        """
        return self._params_dict(lamb=0.0, chi=1.0, psi=1.0, 
                                 mu=0.0, sigma=1.0, gamma=0.0)
    
    @staticmethod
    def _stable_logpdf(stability: Scalar, x: ArrayLike, params: dict) -> Array:
        lamb, chi, psi, mu, sigma, gamma = GHBase._params_to_tuple(params)
        x, xshape = _univariate_input(x)

        r: float = lax.sqrt(lax.mul(chi, psi))
        s: float = 0.5 - lamb
        h: float = lax.add(psi, lax.pow(lax.div(gamma, sigma), 2))
        g = lax.div(lax.sub(x, mu), lax.pow(sigma, 2))

        m = lax.sqrt(lax.mul(lax.add(chi, lax.mul(g, lax.sub(x, mu))), h))

        T = lax.add(lax.log(kv(-s, m) + stability), lax.mul(g, gamma))
        B = lax.mul(lax.log(m + stability), s)

        cT = lax.add(lax.mul(lamb, lax.log((psi / (r + stability)) + stability)), lax.mul(lax.log(h), s)) 
        cB = lax.add(lax.add(lax.log(sigma), lax.log(lax.sqrt(2*jnp.pi))), lax.log(kv(lamb, r) + stability))

        c = lax.sub(cT, cB)
        logpdf: jnp.ndarray = lax.add(lax.sub(T, B), c)
        return logpdf.reshape(xshape)
    
    @staticmethod
    def logpdf(x: ArrayLike, params: dict) -> Array:
        return GHBase._stable_logpdf(stability=0.0, x=x, params=params)
    
    @staticmethod
    def pdf(x: ArrayLike, params: dict) -> Array:
        return lax.exp(GHBase.logpdf(x=x, params=params))
    
    # # ppf
    # def _get_x0(self, params: dict) -> Scalar:
    #     return self.stats(params=params)["mean"]
    
    # sampling
    def rvs(self, size: tuple | Scalar, params: dict, key: Array = DEFAULT_RANDOM_KEY) -> Array:
        lamb, chi, psi, mu, sigma, gamma = self._params_to_tuple(params)

        key1, key2 = random.split(key)
        W = gig.rvs(key=key1, size=size, params={'lambda': lamb, 'chi': chi, 'psi': psi})
        return mean_variance_sampling(key=key2, W=W, shape=size, mu=mu, sigma=sigma, gamma=gamma)

    # stats
    def _get_w_stats(self, lamb: Scalar, chi: Scalar, psi: Scalar) -> dict:
        return gig.stats(params={'lambda': lamb, 'chi': chi, 'psi': psi})

    def stats(self, params: dict) -> dict:
        lamb, chi, psi, mu, sigma, gamma = self._params_to_tuple(params) 
        gig_stats: dict = self._get_w_stats(lamb=lamb, chi=chi, psi=psi)
        return self._scalar_transform(mean_variance_stats(w_stats=gig_stats, mu=mu, sigma=sigma, gamma=gamma))

    # fitting
    def _fit_mle(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        eps: float = 1e-8
        constraints: tuple = (jnp.array([[-jnp.inf, eps, eps, -jnp.inf, eps, -jnp.inf]]).T, 
                          jnp.array([[jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf]]).T)
        
        projection_options: dict = {'hyperparams': constraints}

        key1, key = random.split(DEFAULT_RANDOM_KEY)
        key2, key3 = random.split(key)
        params0: jnp.ndarray = jnp.array([
            random.normal(key1, ()), 
            random.uniform(key2, (), minval=eps), 
            random.uniform(key3, (), minval=eps),  
            x.mean(), 
            x.std(), 
            0.0])
        
        res: dict = projected_gradient(
            f=self._mle_objective, x0=params0, projection_method='projection_box', 
            projection_options=projection_options, x=x, lr=lr, maxiter=maxiter)
        lamb, chi, psi, mu, sigma, gamma = res['x']
        return GHBase._params_dict(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)
    
    def _ldmle_objective(self, params: jnp.ndarray, x: jnp.ndarray, sample_mean: Scalar, sample_variance: Scalar) -> Scalar:
        lamb, chi, psi, gamma = params
        gig_stats: dict = self._get_w_stats(lamb=lamb, chi=chi, psi=psi)
        mu, sigma = mean_variance_ldmle_params(stats=gig_stats, gamma=gamma, sample_mean=sample_mean, sample_variance=sample_variance)
        return self._mle_objective(params_arr=jnp.array([lamb, chi, psi, mu, sigma, gamma]), x=x)
    
    def _fit_ldmle(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        eps = 1e-8
        constraints: tuple = (jnp.array([[-jnp.inf, eps, eps, -jnp.inf]]).T, 
                            jnp.array([[jnp.inf, jnp.inf, jnp.inf, jnp.inf]]).T)
        
        key1, key = random.split(DEFAULT_RANDOM_KEY)
        key2, key3 = random.split(key)
        params0: jnp.ndarray = jnp.array([random.normal(key1, ()), 
                                        random.uniform(key2, (), minval=eps), 
                                        random.uniform(key3, (), minval=eps),  
                                        0.0])

        projection_options: dict = {'hyperparams': constraints}

        sample_mean, sample_variance = x.mean(), x.var()
        res = projected_gradient(
            f=self._ldmle_objective, x0=params0, x=x, lr=lr, maxiter=maxiter,
            projection_method='projection_box', projection_options=projection_options, 
            sample_mean=sample_mean, sample_variance=sample_variance)
        lamb, chi, psi, gamma = res['x']
        gig_stats: dict = self._get_w_stats(lamb=lamb, chi=chi, psi=psi)
        mu, sigma = mean_variance_ldmle_params(stats=gig_stats, gamma=gamma, sample_mean=sample_mean, sample_variance=sample_variance)
        return self._params_dict(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)
    
    def fit(self, x: ArrayLike, method: str = 'LDMLE', lr: float = 1.0, 
            maxiter: int = 100) -> dict:
        r"""Fit the distribution to the input data.

        Note:
            If you intend to jit wrap this function, ensure that 'method' is a 
            static argument.
        
        Args:
            x (ArrayLike): The input data to fit the distribution to.
            method (str): The fitting method to use.  Options are 
                'MLE' for maximum likelihood estimation, 
                and 'LDMLE' for low-dimensional maximum likelihood 
                estimation. Defaults to 'LDMLE'. 
            lr (float): Learning rate for optimization.
            maxiter (int): Maximum number of iterations for optimization.
        
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
    def _params_from_array(params_arr: jnp.ndarray, *args, **kwargs) -> dict:
        lamb, chi, psi, mu, sigma, gamma = params_arr
        return GHBase._params_dict(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)

    @staticmethod
    def _pdf_for_cdf(x: ArrayLike, *params_tuple) -> Array:
        params_array: jnp.ndarray = jnp.asarray(params_tuple).flatten()
        params: dict = GHBase._params_from_array(params_array)
        return GHBase.pdf(x=x, params=params)
    
def _vjp_cdf(x: ArrayLike, params: dict) -> Array:
    params = GHBase._args_transform(params)
    return _cdf(dist=GHBase, x=x, params=params)
    

_vjp_cdf_copy = deepcopy(_vjp_cdf)
_vjp_cdf = custom_vjp(_vjp_cdf)


def cdf_fwd(x: ArrayLike, params: dict) -> tuple[Array, tuple]:
    params = GHBase._args_transform(params)
    return _cdf_fwd(dist=GHBase, cdf_func=_vjp_cdf_copy, x=x, params=params)

_vjp_cdf.defvjp(cdf_fwd, cdf_bwd)


class GH(GHBase):
    r"""The generalized hyperbolic distribution. This is a flexible, 
    continuous 6-parameter family of distributions that can model a variety 
    of data behaviors, including heavy tails and skewness. It contains 
    a number of popular distributions as special cases, including the
    normal, student-t, hyperbolic, laplace, and skewed-T distributions.

    We adopt the parameterization used by McNeil et al. (2005):
    
    .. math::

        f(x|\mu, \sigma, \chi, \psi, \gamma, \lambda) \propto e^{\frac{(x-\mu)\gamma}{\sigma^2}} \frac{K_{\lambda - 0.5}(\sqrt{(\chi + (\frac{x-\mu}{\sigma})^2)(\psi + (\frac{\gamma}{\sigma})^2 })}{(\sqrt{(\chi + (\frac{x-\mu}{\sigma})^2)(\psi + (\frac{\gamma}{\sigma})^2 })^{0.5-\lambda}}

    where :math:`K_{\lambda}` is the modified Bessel function of the second 
    kind, :math:`\mu` is the location parameter, :math:`\sigma` is the scale,
    :math: `\gamma` is the skewness and :math:`\lambda`, :math:`\chi` and 
    :math:`\psi` relate to the shape of the distribution.
    """
    def cdf(self, x: ArrayLike, params: dict) -> Array:
        return _vjp_cdf(x=x, params=params)
    

gh = GH("GH")