"""File containing the copuLAX implementation of the skewed-T distribution."""
import jax.numpy as jnp
from jax import lax, custom_vjp, random
from jax._src.typing import ArrayLike, Array
from copy import deepcopy
from typing import Callable

from copulax._src.univariate._utils import _univariate_input, DEFAULT_RANDOM_KEY
from copulax._src.univariate._ppf import _ppf
from copulax._src.univariate._cdf import _cdf, cdf_bwd, _cdf_fwd
from copulax._src.optimize import projected_gradient
from copulax.special import kv
from copulax._src.univariate import student_t
from copulax._src.univariate._rvs import inverse_transform_sampling


def skewed_t_args_check(nu: float | ArrayLike, mu: float | ArrayLike, sigma: float | ArrayLike, gamma: float | ArrayLike) -> tuple:
    return jnp.asarray(nu, dtype=float), jnp.asarray(mu, dtype=float), jnp.asarray(sigma, dtype=float), jnp.asarray(gamma, dtype=float)


def skewed_t_params_dict(nu: float | ArrayLike, mu: float | ArrayLike, sigma: float | ArrayLike, gamma: float | ArrayLike) -> dict:
    nu, mu, sigma, gamma = skewed_t_args_check(nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    return {'nu': nu, 'mu': mu, 'sigma': sigma, 'gamma': gamma}


#f(x|\nu, \mu, \sigma, \gamma) = \frac{2}{\sigma} \frac{\Gamma\left(\frac{\nu+1}{2}\right)}
# {\Gamma\left(\frac{\nu}{2}\right)} \frac{1}{\sqrt{\nu\pi}} \frac{1}{\sigma} 
# \frac{1}{\sqrt{1+\gamma^2}} \left[1 + \frac{1}{\nu}\left(\frac{x-\mu}{\sigma}\right)^2\right]^{-\frac{\nu+1}{2}}


def support(*args) -> tuple[float, float]:
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

    normalising_constant: float = _cdf(pdf_func=_unnormalised_pdf, lower_bound=-jnp.inf, x=jnp.inf, params=params)
    cdf: jnp.ndarray = _cdf(pdf_func=_unnormalised_pdf, lower_bound=-jnp.inf, x=x, params=params) / normalising_constant
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
    return _ppf(cdf_func=cdf, bounds=(-jnp.inf, jnp.inf), q=q, params=params, x0=mu)


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
    params: dict = skewed_t_params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)
    return inverse_transform_sampling(ppf_func=ppf, shape=shape, params=params, key=key)


def _mle_objective(params: dict, x: ArrayLike) -> Array:
    nu, mu, sigma, gamma = params
    return -jnp.sum(_unnormalised_logpdf(x=x, nu=nu, mu=mu, sigma=sigma, gamma=gamma, stability=1e-30))


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
                                   lr=0.1,
                                   projection_method='projection_box', 
                                   projection_options=projection_options, x=x)
    nu, mu, sigma, gamma = res['x']
    return skewed_t_params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)#, res['fun']


def fit(x: Array) -> dict:
    r"""Fit the skewed-t distribution to data using maximum likelihood estimation.
    
    Args:
        x: arraylike, data to fit the distribution to.
    
    Returns:
        dict of fitted parameters.
    """
    x: jnp.ndarray = _univariate_input(x)[0]
    return _fit_mle(x)


def stats(nu: float = 1.0, mu: float = 0.0, sigma: float = 1.0, gamma: float = 0.0) -> dict:
    r"""Currently not implemented for the skewed-T distribution
    
    Args:
        nu: Degrees of freedom of the skewed-t distribution.
        mu: Mean/location of the skewed-t distribution.
        sigma: Scale parameter of the skewed-t distribution.
        gamma: Skewness parameter of the skewed-t distribution.
    
    Returns:
        dict of distribution statistics.
    """
    return {}