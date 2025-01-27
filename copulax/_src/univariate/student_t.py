"""File containing the copulAX implementation of the student-T distribution."""
import jax.numpy as jnp
from jax import random
from jax import lax
from jax._src.typing import ArrayLike, Array
from jax.scipy import special
from jax import custom_vjp
from copy import deepcopy

from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.univariate._ppf import _ppf
from copulax._src.optimize import projected_gradient
from copulax._src.univariate._cdf import _cdf, cdf_bwd, _cdf_fwd


def student_t_args_check(nu: float | ArrayLike, mu: float | ArrayLike, sigma: float | ArrayLike) -> tuple:
    return jnp.asarray(nu, dtype=float), jnp.asarray(mu, dtype=float), jnp.asarray(sigma, dtype=float)


def student_t_params_dict(nu: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> dict:
    nu, mu, sigma = student_t_args_check(nu=nu, mu=mu, sigma=sigma)
    return {"nu": nu, "mu": mu, "sigma": sigma}


def support(*args) -> tuple[float, float]:
    r"""The support of the distribution is the subset of x for which the pdf 
    is non-zero. 
    
    Returns:
        (float, float): Tuple containing the support of the distribution.
    """
    return -jnp.inf, jnp.inf


def logpdf(x: ArrayLike, nu: float = 1.0, mu: float = 0.0, sigma: float = 1.0) -> Array:
    r"""Log-probability density function of the student-T distribution.
    
    The student-T pdf is defined as:

    .. math::

        f(x|\nu, \mu, \sigma) = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi}\sigma} \left(1 + \frac{(x - \mu)^2}{\nu\sigma^2}\right)^{-\frac{\nu + 1}{2}}

    Args:
        x: arraylike, value(s) at which to evaluate the pdf.
        nu: Degrees of freedom of the student-T distribution.
        mu: Mean/location of the student-T distribution.
        sigma: Scale parameter of the student-T distribution.

    Returns:
        array of log-pdf values.
    """
    x, xshape = _univariate_input(x)
    nu, mu, sigma = student_t_args_check(nu, mu, sigma)

    z: jnp.ndarray = lax.div(lax.sub(x, mu), sigma)

    const: jnp.ndarray = special.gammaln(0.5 * (nu + 1)) - special.gammaln(0.5 * nu) - 0.5 * jnp.log(nu * jnp.pi) - jnp.log(sigma)
    e: jnp.ndarray = lax.mul(lax.log(lax.add(1.0, lax.div(lax.pow(z, 2.0), nu))), -0.5 * (nu + 1))
    logpdf = lax.add(const, e)
    return logpdf.reshape(xshape)


def pdf(x: ArrayLike, nu: float=1.0, mu: float=0.0, sigma: float=1.0) -> Array:
    r"""Probability density function of the student-T distribution.
    
    The student-T pdf is defined as:

    .. math::

        f(x|\nu, \mu, \sigma) = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi}\sigma} \left(1 + \frac{(x - \mu)^2}{\nu\sigma^2}\right)^{-\frac{\nu + 1}{2}}
    
    Args:
        x: arraylike, value(s) at which to evaluate the pdf.
        nu: Degrees of freedom of the student-T distribution.
        mu: Mean/location of the student-T distribution.
        sigma: Scale parameter of the student-T distribution.

    Returns:
        array of pdf values.
    """
    return jnp.exp(logpdf(x=x, nu=nu, mu=mu, sigma=sigma))


def cdf(x: ArrayLike, nu: float=1.0, mu: float=0.0, sigma: float=1.0) -> Array:
    r"""Cumulative distribution function of the student-T distribution.
    
    The student-T pdf is defined as:

    .. math::

        f(x|\nu, \mu, \sigma) = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi}\sigma} \left(1 + \frac{(x - \mu)^2}{\nu\sigma^2}\right)^{-\frac{\nu + 1}{2}}

    Args:
        x: arraylike, value(s) at which to evaluate the cdf.
        nu: Degrees of freedom of the student-T distribution.
        mu: Mean/location of the student-T distribution.
        sigma: Scale parameter of the student-T distribution.

    Returns:
        array of cdf values.
    """
    params = student_t_params_dict(nu=nu, mu=mu, sigma=sigma)
    return _cdf(pdf_func=pdf, lower_bound=-jnp.inf, x=x, params=params)


__cdf = deepcopy(cdf)
cdf = custom_vjp(cdf)


def cdf_fwd(x: ArrayLike, nu, mu, sigma) -> tuple[Array, tuple]:
    params=student_t_params_dict(nu=nu, mu=mu, sigma=sigma)
    cdf_vals, (pdf_vals, param_grads) = _cdf_fwd(pdf_func=pdf, cdf_func=__cdf, x=x, params=params)
    return cdf_vals, (pdf_vals, param_grads['nu'], param_grads['mu'], param_grads['sigma'])


cdf.defvjp(cdf_fwd, cdf_bwd)


def logcdf(x: ArrayLike, nu: float=1.0, mu: float=0.0, sigma: float=1.0) -> Array:
    r"""Log-cumulative distribution function of the student-T distribution.
    
    The student-T pdf is defined as:

    .. math::

        f(x|\nu, \mu, \sigma) = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi}\sigma} \left(1 + \frac{(x - \mu)^2}{\nu\sigma^2}\right)^{-\frac{\nu + 1}{2}}

    Args:
        x: arraylike, value(s) at which to evaluate the cdf.
        nu: Degrees of freedom of the student-T distribution.
        mu: Mean/location of the student-T distribution.
        sigma: Scale parameter of the student-T distribution.

    Returns:
        array of log-cdf values.
    """
    return jnp.log(cdf(x=x, nu=nu, mu=mu, sigma=sigma))

def ppf(q: ArrayLike, nu: float=1.0, mu: float=0.0, sigma: float=1.0) -> Array:
    r"""Percent point function (inverse of cdf) of the student-T distribution.
    
    The student-T pdf is defined as:

    .. math::

        f(x|\nu, \mu, \sigma) = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi}\sigma} \left(1 + \frac{(x - \mu)^2}{\nu\sigma^2}\right)^{-\frac{\nu + 1}{2}}
    
    Args:
        q: arraylike, value(s) at which to evaluate the ppf.
        nu: Degrees of freedom of the student-T distribution.
        mu: Mean/location of the student-T distribution.
        sigma: Scale parameter of the student-T distribution.

    Returns:
        array of ppf values.
    """
    return _ppf(cdf_func=cdf, bounds=(-jnp.inf, jnp.inf), x0=mu,
                      q=q, params=student_t_params_dict(nu=nu, mu=mu, sigma=sigma))


def rvs(shape: tuple = (1, ), key: Array=DEFAULT_RANDOM_KEY, nu: float=1.0, mu: float=0.0, sigma: float=1.0) -> Array:
    r"""Generate random variates from the student-T distribution.

    The student-T pdf is defined as:

    .. math::

        f(x|\nu, \mu, \sigma) = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi}\sigma} \left(1 + \frac{(x - \mu)^2}{\nu\sigma^2}\right)^{-\frac{\nu + 1}{2}}
    
    Args:
        shape: Shape of the output array.
        key: PRNGKey for random number generation.
        nu: Degrees of freedom of the student-T distribution.
        mu: Mean/location of the student-T distribution.
        sigma: Scale parameter of the student-T distribution.

    Returns:
        array of random variates.
    """
    nu, mu, sigma = student_t_args_check(nu=nu, mu=mu, sigma=sigma)
    z: jnp.ndarray = random.t(key=key, df=nu, shape=shape)
    return lax.add(lax.mul(z, sigma), mu)


# def cdf_approx(key: Array, x: ArrayLike, nu=1.0, mu=0.0, sigma=1.0, num_points: int = 100) -> Array:
#     r"""Approximate the cumulative distribution function of the student-T distribution by 
#     evaluating it at a set of points. Uses a linear interpolation to reduce the 
#     required number of similated random variables.

#     Args:
#         key: PRNGKey for random number generation.
#         x: arraylike, value(s) at which to evaluate the cdf.
#         nu: Degrees of freedom of the student-T distribution.
#         mu: Mean/location of the student-T distribution.
#         sigma: Scale parameter of the student-T distribution.
#         num_points: Number of simulated random points at which to evaluate the 
#         cdf.

#     Returns:
#         array of cdf values.
#     """
#     return _cdf_approx(key=key, rvs_func=rvs, x=x, num_points=num_points, params=(nu, mu, sigma))


# def ppf_approx(key: Array, q: ArrayLike, nu=1.0, mu=0.0, sigma=1.0, num_points: int = 100) -> Array:
#     r"""Approximate the percent point function (inverse of cdf) of the student-T distribution 
#     by evaluating it at a set of points. Uses a linear interpolation to reduce the required 
#     number of simulated random variables.

#     Args:
#         key: PRNGKey for random number generation.
#         q: arraylike, value(s) at which to evaluate the ppf.
#         nu: Degrees of freedom of the student-T distribution.
#         mu: Mean/location of the student-T distribution.
#         sigma: Scale parameter of the student-T distribution.
#         num_points: Number of simulated random points at which to evaluate the 
#         cdf.
#     """
#     return _ppf_approx(key=key, rvs_func=rvs, q=q, num_points=num_points, params=(nu, mu, sigma))


def _mle_objective(params: jnp.ndarray, x: jnp.ndarray) -> float:
    r"""Objective function for maximum likelihood estimation of the student-T distribution.
    
    Args:
        params: array, parameters of the student-T distribution.
        x: array, data to fit the distribution to.
    
    Returns:
        negative log-likelihood value.
    """
    nu, mu, sigma = params
    return -jnp.sum(logpdf(x=x, nu=nu, mu=mu, sigma=sigma))

def _fit_mle(x: ArrayLike) -> tuple[dict, float]:
    r"""Fit the student-T distribution using maximum likelihood estimation.
    
    Args:
        x: arraylike, data to fit the distribution to.

    Returns:
        dictionary of fitted parameters and the negative log-likelihood value.
    """
    eps = 1e-8
    constraints: tuple = (jnp.array([[eps, -jnp.inf, eps]]).T, 
                          jnp.array([[jnp.inf, jnp.inf, jnp.inf]]).T)
    
    projection_options: dict = {'hyperparams': constraints}
    params0: jnp.ndarray = jnp.array([1.0, x.mean(), x.std()])

    res = projected_gradient(
        f=_mle_objective, x0=params0, projection_method='projection_box', 
        projection_options=projection_options, x=x)
    nu, mu, sigma = res['x']

    return {"nu": nu, "mu": mu, "sigma": sigma}#, res['val']


def _ldmle_objective(params: jnp.ndarray, x: jnp.ndarray, mu) -> jnp.ndarray:
    r"""Objective function for low-dimensional maximum likelihood estimation 
    of the student-T distribution.
    
    Args:
        params: array, parameters of the student-T distribution.
        x: array, data to fit the distribution to.
    
    Returns:
        array of negative log-likelihood values.
    """
    nu, sigma = params
    return _mle_objective(params=jnp.array([nu, mu, sigma]), x=x)


def _fit_ldmle(x: ArrayLike) -> tuple[dict, float]:
    r"""Fit the student-T distribution using low-dimensional maximum 
    likelihood estimation. This assumes mu = x.mean(), removing a variable 
    from the optimzation loop.
    
    Args:
        x: arraylike, data to fit the distribution to.

    Returns:
        dictionary of fitted parameters and the negative log-likelihood value.
    """
    params0: jnp.ndarray = jnp.array([1.0, x.std()])

    res = projected_gradient(
        f=_ldmle_objective, x0=params0, 
        projection_method='projection_non_negative', x=x, mu=x.mean())
    nu, sigma = res['x']

    return {"nu": nu, "mu": x.mean(), "sigma": sigma}#, res['val']


def fit(x: ArrayLike, method: str = 'LDMLE') -> tuple[dict, float]:
    r"""Fit the parameters of a student-T distribution to the data using 
    maximum likelihood estimation.

    Note:
        If you intend to jit wrap this function, ensure that 'method' is a 
        static argument.
    
    Args:
        x: arraylike, data to fit the distribution to.
        method: method to use for fitting the distribution. Options are: 
            'MLE' for maximum likelihood estimation.
            'LDMLE' for low-dimensional maximum likelihood estimation, which uses mu = x.mean(), removing this parameter from the optimization loop.

    Returns:
        dictionary of fitted parameters and the negative log-likelihood value.
    """
    x: jnp.ndarray = _univariate_input(x)[0]
    method = method.upper().strip()
    if method == 'MLE':
        return _fit_mle(x)
    else:
        return _fit_ldmle(x)
        

def stats(nu: float = 1.0, mu: float = 0.0, sigma: float = 1.0) -> dict:
    r"""Distribution statistics for the student-T distribution. Returns the 
    mean, median, mode, variance, skewness and (excess) kurtosis of the 
    distribution.

    Args:
        nu: Degrees of freedom of the student-T distribution.
        mu: Mean/location of the student-T distribution.
        sigma: Scale parameter of the student-T distribution.

    Returns:
        dictionary of distribution statistics.
    """
    nu, mu, sigma = student_t_args_check(nu, mu, sigma)

    mean: float = jnp.where(nu > 1, mu, jnp.nan)
    variance: float = jnp.where(nu > 2, nu / (nu - 2), jnp.nan)
    skewness: float = jnp.where(nu > 3, 0.0, jnp.nan)
    kurtosis: float = jnp.where(nu > 4, 6 / (nu - 4), jnp.inf)
    kurtosis = jnp.where(nu <= 2, jnp.nan, kurtosis)

    stats_dict = {"mean": mean, "median": mu, "mode": mu, "variance": variance, "skewness": skewness, "kurtosis": kurtosis}
    return stats_dict


