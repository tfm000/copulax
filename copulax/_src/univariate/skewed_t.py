"""File containing the copuLAX implementation of the skewed-T distribution."""
import jax.numpy as jnp
from jax import lax, custom_vjp, random
from jax._src.typing import ArrayLike, Array
from copy import deepcopy
from typing import Callable

from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.univariate._ppf import _ppf
from copulax._src.univariate._cdf import _cdf, cdf_bwd, _cdf_fwd
from copulax._src.optimize import projected_gradient
from copulax.special import kv
from copulax._src.univariate import student_t, ig, normal
from copulax._src.univariate._mean_variance import _get_ldmle_params, _get_stats
from copulax._src.univariate._rvs import mean_variance_sampling
from copulax._src.univariate._metrics import (_loglikelihood, _aic, _bic, _mle_objective as __mle_objective)


def skewed_t_args_check(nu: float | ArrayLike, mu: float | ArrayLike, sigma: float | ArrayLike, gamma: float | ArrayLike) -> tuple:
    return jnp.asarray(nu, dtype=float), jnp.asarray(mu, dtype=float), jnp.asarray(sigma, dtype=float), jnp.asarray(gamma, dtype=float)


def skewed_t_params_dict(nu: float | ArrayLike, mu: float | ArrayLike, sigma: float | ArrayLike, gamma: float | ArrayLike) -> dict:
    nu, mu, sigma, gamma = skewed_t_args_check(nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    return {'nu': nu, 'mu': mu, 'sigma': sigma, 'gamma': gamma}


#f(x|\nu, \mu, \sigma, \gamma) = \frac{2}{\sigma} \frac{\Gamma\left(\frac{\nu+1}{2}\right)}
# {\Gamma\left(\frac{\nu}{2}\right)} \frac{1}{\sqrt{\nu\pi}} \frac{1}{\sigma} 
# \frac{1}{\sqrt{1+\gamma^2}} \left[1 + \frac{1}{\nu}\left(\frac{x-\mu}{\sigma}\right)^2\right]^{-\frac{\nu+1}{2}}


def support(*args, **kwargs) -> tuple[float, float]:
    r"""The support of the distribution is the subset of x for which the pdf 
    is non-zero. 
    
    Returns:
        (float, float): Tuple containing the support of the distribution.
    """
    return -jnp.inf, jnp.inf


def _mcneil_logpdf(x: ArrayLike, nu: float, mu: float, sigma: float, gamma: float, stability: float) -> Array:
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


def _azzalini_logpdf(x: ArrayLike, nu: float, mu: float, sigma: float, gamma: float) -> Array:
    x, xshape = _univariate_input(x)
    
    Q: jnp.ndarray = lax.pow((x - mu) / sigma, 2)
    T1_arg: jnp.ndarray = gamma * (x - mu) * lax.sqrt((nu + 1)/ (Q + nu)) / sigma

    logpdf: jnp.ndarray = jnp.log(2) + student_t.logpdf(x=x, nu=nu, mu=mu, sigma=sigma) + student_t.logpdf(x=T1_arg, nu=nu + 1, mu=mu, sigma=sigma)
    return logpdf.reshape(xshape)


_log_pdf_func: Callable = _mcneil_logpdf


def _unnormalised_logpdf(x: ArrayLike, nu: float, mu: float, sigma: float, gamma: float, stability: float=0.0) -> Array:
    nu, mu, sigma, gamma = skewed_t_args_check(nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    return lax.cond(gamma == 0, lambda x: student_t.logpdf(x=x, nu=nu, mu=mu, sigma=sigma), lambda x: _log_pdf_func(x=x, nu=nu, mu=mu, sigma=sigma, gamma=gamma, stability=stability), x)


def _unnormalised_pdf(x: ArrayLike, nu: float, mu: float, sigma: float, gamma: float, stability: float=0.0) -> Array:
    return jnp.exp(_unnormalised_logpdf(x=x, nu=nu, mu=mu, sigma=sigma, gamma=gamma, stability=stability))


def logpdf(x: ArrayLike, nu: float = 1.0, mu: float = 0.0, sigma: float = 1.0, gamma: float = 0.0) -> Array:
    r"""Log-probability density function of the skewed-t distribution.
    
    Args:
        x: arraylike, value(s) at which to evaluate the pdf.
        nu: Degrees of freedom of the skewed-t distribution.
        mu: Mean/location of the skewed-t distribution.
        sigma: Scale parameter of the skewed-t distribution.
        gamma: Skewness parameter of the skewed-t distribution.
    
    Returns:
        array of log-pdf values.
    """
    params = skewed_t_params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)

    normalising_constant: float = _cdf(pdf_func=_unnormalised_pdf, lower_bound=-jnp.inf, x=jnp.inf, params=params)
    return _unnormalised_logpdf(x=x, nu=nu, mu=mu, sigma=sigma, gamma=gamma) - jnp.log(normalising_constant)

    

def pdf(x: ArrayLike, nu: float = 1.0, mu: float = 0.0, sigma: float = 1.0, gamma: float = 0.0) -> Array:
    r"""Probability density function of the skewed-t distribution.
    
    Args:
        x: arraylike, value(s) at which to evaluate the pdf.
        nu: Degrees of freedom of the skewed-t distribution.
        mu: Mean/location of the skewed-t distribution.
        sigma: Scale parameter of the skewed-t distribution.
        gamma: Skewness parameter of the skewed-t distribution.
    
    Returns:
        array of pdf values.
    """
    return jnp.exp(logpdf(x=x, nu=nu, mu=mu, sigma=sigma, gamma=gamma))


def cdf(x: ArrayLike, nu: float = 1.0, mu: float = 0.0, sigma: float = 1.0, gamma: float = 0.0) -> Array:
    r"""Cumulative distribution function of the skewed-t distribution.
    
    Args:
        x: arraylike, value(s) at which to evaluate the cdf.
        nu: Degrees of freedom of the skewed-t distribution.
        mu: Mean/location of the skewed-t distribution.
        sigma: Scale parameter of the skewed-t distribution.
        gamma: Skewness parameter of the skewed-t distribution.
    
    Returns:
        array of cdf values.
    """
    params: dict = skewed_t_params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)

    lb: float = support()[0]
    normalising_constant: float = _cdf(pdf_func=_unnormalised_pdf, lower_bound=lb, x=jnp.inf, params=params)
    cdf: jnp.ndarray = _cdf(pdf_func=_unnormalised_pdf, lower_bound=lb, x=x, params=params) / normalising_constant
    return jnp.where(cdf > 1.0, 1.0, cdf)


__cdf = deepcopy(cdf)
cdf = custom_vjp(cdf)


def cdf_fwd(x: ArrayLike, nu: float, mu: float, sigma: float, gamma: float) -> tuple[Array, tuple]:
    params: dict = skewed_t_params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    cdf_vals, (pdf_vals, param_grads) = _cdf_fwd(pdf_func=pdf, cdf_func=__cdf, x=x, params=params)
    return cdf_vals, (pdf_vals, param_grads['nu'], param_grads['mu'], param_grads['sigma'], param_grads['gamma'])


cdf.defvjp(cdf_fwd, cdf_bwd)


def logcdf(x: ArrayLike, nu: float = 1.0, mu: float = 0.0, sigma: float = 1.0, gamma: float = 0.0) -> Array:
    r"""Log-cumulative distribution function of the skewed-t distribution.
    
    Args:
        x: arraylike, value(s) at which to evaluate the log-cdf.
        nu: Degrees of freedom of the skewed-t distribution.
        mu: Mean/location of the skewed-t distribution.
        sigma: Scale parameter of the skewed-t distribution.
        gamma: Skewness parameter of the skewed-t distribution.
    
    Returns:
        array of log-cdf values.
    """
    return jnp.log(cdf(x=x, nu=nu, mu=mu, sigma=sigma, gamma=gamma))


def ppf(q: ArrayLike, nu: float = 1.0, mu: float = 0.0, sigma: float = 1.0, gamma: float = 0.0) -> Array:
    r"""Percent point function (inverse cdf) of the skewed-t distribution.
    
    Args:
        q: arraylike, quantile(s) at which to evaluate the ppf.
        nu: Degrees of freedom of the skewed-t distribution.
        mu: Mean/location of the skewed-t distribution.
        sigma: Scale parameter of the skewed-t distribution.
        gamma: Skewness parameter of the skewed-t distribution.
    
    Returns:
        array of ppf values.
    """
    params: dict = skewed_t_params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    return _ppf(cdf_func=cdf, bounds=support(), q=q, params=params, x0=mu)


def rvs(shape: tuple = (1,), key: Array = DEFAULT_RANDOM_KEY, nu: float = 1.0, mu: float = 0.0, sigma: float = 1.0, gamma: float = 0.0) -> Array:
    r"""Generate random samples from the skewed-t distribution.
    
    Args:
        shape: The shape of the output array.
        key: Key for random number generation.
        nu: Degrees of freedom of the skewed-t distribution.
        mu: Mean/location of the skewed-t distribution.
        sigma: Scale parameter of the skewed-t distribution.
        gamma: Skewness parameter of the skewed-t distribution.

    Note:
        If you intend to jit wrap this function, ensure that 'shape' is a 
        static argument.
    
    Returns:
        array of random samples.
    """
    nu, mu, sigma, gamma = skewed_t_args_check(nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    
    key1, key2 = random.split(key)
    W: jnp.ndarray = ig.rvs(key=key1, alpha=nu*0.5, beta=nu*0.5, shape=shape)
    return mean_variance_sampling(key=key2, W=W, shape=shape, mu=mu, sigma=sigma, gamma=gamma)


def _mle_objective(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    nu, mu, sigma, gamma = params
    return __mle_objective(
        logpdf_func=_unnormalised_logpdf, x=x, stability=1e-30, 
        params=skewed_t_params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma))


def _fit_mle(x: ArrayLike) -> dict:
    eps: float = 1e-8
    constraints: tuple = (jnp.array([[eps, -jnp.inf, eps, -jnp.inf]]).T, 
                          jnp.array([[jnp.inf, jnp.inf, jnp.inf, jnp.inf]]).T)
    
    projection_options: dict = {'hyperparams': constraints}

    key1, key2 = random.split(DEFAULT_RANDOM_KEY)
    params0: jnp.ndarray = jnp.array([jnp.abs(random.normal(key1, ())), 
                                      x.mean(),
                                      x.std(),
                                      random.normal(key2, ())])
    
    params0: jnp.ndarray = jnp.array([1.0, 
                                      x.mean(),
                                      x.std(),
                                      1.0])
    
    res: dict = projected_gradient(f=_mle_objective, x0=params0,
                                   projection_method='projection_box', 
                                   projection_options=projection_options, x=x)
    nu, mu, sigma, gamma = res['x']
    return skewed_t_params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)#, res['fun']


def _get_w_stats(nu: float) -> dict:
    ig_stats: dict = ig.stats(alpha=nu*0.5, beta=nu*0.5)
    w_mean =  jnp.where(jnp.isnan(ig_stats['mean']), ig_stats['mode'], ig_stats['mean'])
    w_variance = jnp.where(jnp.isnan(ig_stats['variance']), w_mean ** 2, ig_stats['variance'])
    return {'mean': w_mean, 'variance': w_variance}


def _ldmle_objective(params: jnp.ndarray, x: jnp.ndarray, sample_mean: float, sample_variance: float) -> jnp.ndarray:
    nu, gamma = params
    ig_stats: dict = _get_w_stats(nu=nu)
    mu, sigma = _get_ldmle_params(stats=ig_stats, gamma=gamma, sample_mean=sample_mean, sample_variance=sample_variance)
    return _mle_objective(params=jnp.array([nu, mu, sigma, gamma]), x=x)


def _fit_ldmle(x: ArrayLike) -> tuple[dict, float]:
    eps: float = 1e-8
    min_nu: float = 4.0 
    constraints: tuple = (jnp.array([[min_nu + eps, -jnp.inf]]).T, 
                          jnp.array([[jnp.inf, jnp.inf]]).T)

    key1, key2 = random.split(DEFAULT_RANDOM_KEY)
    params0: jnp.ndarray = jnp.array([min_nu+jnp.abs(random.normal(key1, ())), 
                                      random.normal(key2, ())])
    
    projection_options: dict = {'hyperparams': constraints}

    sample_mean, sample_variance = x.mean(), x.var()
    res: dict = projected_gradient(f=_ldmle_objective, x0=params0,
                                   projection_method='projection_box', 
                                   projection_options=projection_options, x=x, 
                                   sample_mean=sample_mean, sample_variance=sample_variance)
    nu, gamma = res['x']
    ig_stats: dict = _get_w_stats(nu=nu)
    mu, sigma = _get_ldmle_params(stats=ig_stats, gamma=gamma, sample_mean=sample_mean, sample_variance=sample_variance)
    return skewed_t_params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)#, res['fun']



def fit(x: ArrayLike, method: str = 'LDMLE') -> dict:
    r"""Fit the skewed-t distribution to data.
    
    Note:
        If you intend to jit wrap this function, ensure that 'method' is a 
        static argument.

    Args:
        x: arraylike, data to fit the distribution to.
        method: str, method to use for fitting the distribution. Options are 
        'MLE' for maximum likelihood estimation, and 'LDMLE' for low-dimensional 
        maximum likelihood estimation. Defaults to 'LDMLE'.
    
    Returns:
        dict of fitted parameters.
    """
    x: jnp.ndarray = _univariate_input(x)[0]
    if method == 'MLE':
        return _fit_mle(x)
    else:
        return _fit_ldmle(x)


def stats(nu: float = 1.0, mu: float = 0.0, sigma: float = 1.0, gamma: float = 0.0) -> dict:
    r"""Distribution statistics for the skewed-t distribution.
    Returns the mean and variance of the distribution.
    
    Args:
        nu: Degrees of freedom of the skewed-t distribution.
        mu: Mean/location of the skewed-t distribution.
        sigma: Scale parameter of the skewed-t distribution.
        gamma: Skewness parameter of the skewed-t distribution.
    
    Returns:
        dict of distribution statistics.
    """
    nu, mu, sigma, gamma = skewed_t_args_check(nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    ig_stats: dict = ig.stats(alpha=nu*0.5, beta=nu*0.5)
    return _get_stats(w_stats=ig_stats, mu=mu, sigma=sigma, gamma=gamma)


def loglikelihood(x: ArrayLike, nu: float = 1.0, mu: float = 0.0, sigma: float = 1.0, gamma: float = 0.0) -> float:
    r"""Log-likelihood of the skewed-t distribution.
    
    Args:
        x: arraylike, data to evaluate the log-likelihood at.
        nu: Degrees of freedom of the skewed-t distribution.
        mu: Mean/location of the skewed-t distribution.
        sigma: Scale parameter of the skewed-t distribution.
        gamma: Skewness parameter of the skewed-t distribution.
    
    Returns:
        float log-likelihood value.
    """
    return _loglikelihood(
        logpdf_func=logpdf, x=x, 
        params=skewed_t_params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma))


def aic(x: ArrayLike, nu: float = 1.0, mu: float = 0.0, sigma: float = 1.0, gamma: float = 0.0) -> float:
    r"""Akaike Information Criterion (AIC) of the skewed-t distribution.
    
    Args:
        x: arraylike, data to evaluate the AIC at.
        nu: Degrees of freedom of the skewed-t distribution.
        mu: Mean/location of the skewed-t distribution.
        sigma: Scale parameter of the skewed-t distribution.
        gamma: Skewness parameter of the skewed-t distribution.
    
    Returns:
        float AIC value.
    """
    return _aic(logpdf_func=logpdf, x=x, 
                params=skewed_t_params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma))


def bic(x: ArrayLike, nu: float = 1.0, mu: float = 0.0, sigma: float = 1.0, gamma: float = 0.0) -> float:
    r"""Bayesian Information Criterion (BIC) of the skewed-t distribution.
    
    Args:
        x: arraylike, data to evaluate the BIC at.
        nu: Degrees of freedom of the skewed-t distribution.
        mu: Mean/location of the skewed-t distribution.
        sigma: Scale parameter of the skewed-t distribution.
        gamma: Skewness parameter of the skewed-t distribution.
    
    Returns:
        float BIC value.
    """
    return _bic(logpdf_func=logpdf, x=x, 
                params=skewed_t_params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma))