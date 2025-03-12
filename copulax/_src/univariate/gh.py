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
    @staticmethod
    def _params_dict(lamb, chi, psi, mu, sigma, gamma):
        lamb, chi, psi, mu, sigma, gamma = GHBase._args_transform(lamb, chi, psi, mu, sigma, gamma)
        return {"lamb": lamb, "chi": chi, "psi": psi, "mu": mu, 
                "sigma": sigma, "gamma": gamma}
    
    @staticmethod
    def support(*args, **kwargs) -> tuple[Scalar, Scalar]:
        return jnp.array(-jnp.inf), jnp.array(jnp.inf)
    
    @staticmethod
    def _stable_logpdf(stability: Scalar, x: ArrayLike, lamb: Scalar = 0.0, chi: Scalar = 1.0, psi: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        x, xshape = _univariate_input(x)
        lamb, chi, psi, mu, sigma, gamma = GHBase._args_transform(lamb, chi, psi, mu, sigma, gamma)

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
    def logpdf(x: ArrayLike, lamb: Scalar = 0.0, chi: Scalar = 1.0, psi: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        return GHBase._stable_logpdf(stability=0.0, x=x, lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)
    
    @staticmethod
    def pdf(x: ArrayLike, lamb: Scalar = 0.0, chi: Scalar = 1.0, psi: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        return lax.exp(GHBase.logpdf(x=x, lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma))
    
    def logcdf(self, x: ArrayLike, lamb: Scalar = 0.0, chi: Scalar = 1.0, psi: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        return super().logcdf(x=x, lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)
    
    def ppf(self, q: ArrayLike, lamb: Scalar = 0.0, chi: Scalar = 1.0, psi: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        mean: Scalar = self.stats(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)["mean"]
        return super().ppf(x0=mean, q=q, lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)
    
    def inverse_cdf(self, q: ArrayLike, lamb: Scalar = 0.0, chi: Scalar = 1.0, psi: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        return super().inverse_cdf(q=q, lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)
    
    # sampling
    def rvs(self, size: tuple = (), key: Array = DEFAULT_RANDOM_KEY, lamb: Scalar = 0.0, chi: Scalar = 1.0, psi: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0,  gamma: Scalar = 0.0) -> Array:
        lamb, chi, psi, mu, sigma, gamma = self._args_transform(lamb, chi, psi, mu, sigma, gamma)

        key1, key2 = random.split(key)
        W = gig.rvs(key=key1, size=size, chi=chi, psi=psi, lamb=lamb)
        return mean_variance_sampling(key=key2, W=W, shape=size, mu=mu, sigma=sigma, gamma=gamma)
    
    def sample(self, size: tuple = (), key: Array = DEFAULT_RANDOM_KEY, lamb: Scalar = 0.0, chi: Scalar = 1.0, psi: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0,  gamma: Scalar = 0.0) -> Array:
        return super().sample(size=size, key=key, lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)

    # stats
    def stats(self, lamb: Scalar = 0.0, chi: Scalar = 1.0, psi: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0,  gamma: Scalar = 0.0) -> dict:
        lamb, chi, psi, mu, sigma, gamma = self._args_transform(lamb, chi, psi, mu, sigma, gamma) 
        gig_stats: dict = gig.stats(lamb=lamb, chi=chi, psi=psi)
        return mean_variance_stats(w_stats=gig_stats, mu=mu, sigma=sigma, gamma=gamma)
    
    # metrics
    def loglikelihood(self, x, lamb: Scalar = 0.0, chi: Scalar = 1.0, psi: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0,  gamma: Scalar = 0.0) -> Scalar:
        return super().loglikelihood(x=x, lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)

    def aic(self, x, lamb: Scalar = 0.0, chi: Scalar = 1.0, psi: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0,  gamma: Scalar = 0.0) -> Scalar:
        return super().aic(x=x, lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)
    
    def bic(self, x, lamb: Scalar = 0.0, chi: Scalar = 1.0, psi: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0,  gamma: Scalar = 0.0) -> Scalar:
        return super().bic(x=x, lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)

    # fitting
    def _fit_mle(self, x: jnp.ndarray, *args, **kwargs) -> dict:
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
        
        res: dict = projected_gradient(f=self._mle_objective, x0=params0, projection_method='projection_box', projection_options=projection_options, x=x)
        lamb, chi, psi, mu, sigma, gamma = res['x']
        return self._params_dict(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)
    
    def _ldmle_objective(self, params: jnp.ndarray, x: jnp.ndarray, sample_mean: Scalar, sample_variance: Scalar) -> Scalar:
        lamb, chi, psi, gamma = params
        gig_stats: dict = gig.stats(lamb=lamb, chi=chi, psi=psi)
        mu, sigma = mean_variance_ldmle_params(stats=gig_stats, gamma=gamma, sample_mean=sample_mean, sample_variance=sample_variance)
        return self._mle_objective(params=jnp.array([lamb, chi, psi, mu, sigma, gamma]), x=x)
    
    def _fit_ldmle(self, x: jnp.ndarray, *args, **kwargs) -> dict:
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
        res = projected_gradient(f=self._ldmle_objective, x0=params0, projection_method='projection_box', projection_options=projection_options, x=x, sample_mean=sample_mean, sample_variance=sample_variance)
        lamb, chi, psi, gamma = res['x']
        gig_stats: dict = gig.stats(lamb=lamb, chi=chi, psi=psi)
        mu, sigma = mean_variance_ldmle_params(stats=gig_stats, gamma=gamma, sample_mean=sample_mean, sample_variance=sample_variance)
        return self._params_dict(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)
    
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
        
# cdf
def _vjp_cdf(x: ArrayLike, lamb: Scalar, chi: Scalar, psi: Scalar, mu: Scalar, sigma: Scalar, gamma: Scalar) -> Array:
    params: dict = GHBase._params_dict(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)
    return _cdf(pdf_func=GHBase.pdf, lower_bound=GHBase.support()[0], x=x, params=params)
    

_vjp_cdf_copy = deepcopy(_vjp_cdf)
_vjp_cdf = custom_vjp(_vjp_cdf)


def cdf_fwd(x: ArrayLike, lamb, chi, psi, mu, sigma, gamma) -> tuple[Array, tuple]:
    params: dict = GHBase._params_dict(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)
    cdf_vals, (pdf_vals, param_grads) = _cdf_fwd(pdf_func=GHBase.pdf, cdf_func=_vjp_cdf_copy, x=x, params=params)
    return cdf_vals, (pdf_vals, param_grads['lamb'], param_grads['chi'], param_grads['psi'], param_grads['mu'], param_grads['sigma'], param_grads['gamma'])


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
    def cdf(self, x: ArrayLike, lamb: Scalar = 0.0, chi: Scalar = 1.0, psi: Scalar = 1.0, mu: Scalar = 0.0, sigma: Scalar = 1.0, gamma: Scalar = 0.0) -> Array:
        return _vjp_cdf(x=x, lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)
    

gh = GH("GH")