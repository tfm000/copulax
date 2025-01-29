"""File containing the copuLAX implementaqtion of the Generalized Inverse 
Gaussian distribution."""
import jax.numpy as jnp
from jax import random, lax, custom_vjp, jit
from jax._src.typing import ArrayLike, Array
from copy import deepcopy

from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.univariate._ppf import _ppf
from copulax._src.univariate._cdf import _cdf, cdf_bwd, _cdf_fwd
from copulax._src.optimize import projected_gradient
from copulax.special import kv
from copulax._src.univariate._metrics import (_loglikelihood, _aic, _bic, _mle_objective as __mle_objective)


def gig_args_check(lamb: float | ArrayLike, chi: float | ArrayLike, psi: float | ArrayLike) -> tuple:
    return jnp.asarray(lamb, dtype=float), jnp.asarray(chi, dtype=float), jnp.asarray(psi, dtype=float)


def gig_params_dict(lamb: float | ArrayLike, chi: float | ArrayLike, psi: float | ArrayLike) -> dict:
    lamb, chi, psi = gig_args_check(lamb=lamb, chi=chi, psi=psi)
    return {'lamb': lamb, 'chi': chi, 'psi': psi}


def support(*args, **kwargs) -> tuple[float, float]:
    r"""The support of the distribution is the subset of x for which the pdf 
    is non-zero. 
    
    Returns:
        (float, float): Tuple containing the support of the distribution.
    """
    return 0.0, jnp.inf


def logpdf(x: ArrayLike, lamb: float = 1.0, chi: float = 1.0, psi: float = 1.0) -> Array:
    r"""Log-probability density function of the Generalized Inverse Gaussian 
    (GIG) distribution.
    
    Args:
        x: The input array.
        chi: Distrubition parameter.
        psi: Distrubition parameter.
        lamb: Distrubition parameter.

    Returns:
        Array: The log-probability density function values.
    """
    eps = 1e-30

    x, xshape = _univariate_input(x)
    lamb, chi, psi = gig_args_check(lamb=lamb, chi=chi, psi=psi)

    var = lax.add(lax.mul(lamb - 1, lax.log(x)), 
                  -0.5 * (lax.mul(chi, lax.pow(x, -1)) + lax.mul(psi, x)))

    cT = lax.mul(0.5*lamb, lax.sub(lax.log(psi), lax.log(chi)) )
    kv_val = kv(lamb, lax.pow(lax.mul(chi, psi), 0.5))
    kv_val = jnp.where(kv_val < eps, eps, kv_val)  
    cB = lax.log(2 * kv_val)
    
    c = lax.sub(cT, cB)
    pdf_raw = lax.add(var, c)
    logpdf: jnp.ndarray = jnp.where(jnp.isnan(pdf_raw), -jnp.inf, pdf_raw)
    return logpdf.reshape(xshape)


def pdf(x: ArrayLike, lamb: float = 1.0, chi: float = 1.0, psi: float = 1.0) -> Array:
    r"""Probability density function of the Generalized Inverse Gaussian (GIG) 
    distribution.
    
    Args:
        x: The input array.
        chi: Distrubition parameter.
        psi: Distrubition parameter.
        lamb: Distrubition parameter.

    Returns:
        Array: The probability density function values.
    """
    return lax.exp(logpdf(x=x, lamb=lamb, chi=chi, psi=psi))


def cdf(x: ArrayLike, lamb: float = 1.0, chi: float = 1.0, psi: float = 1.0) -> Array:
    r"""Cumulative distribution function of the Generalized Inverse Gaussian
    (GIG) distribution.
    
    Args:
        x: The input array.
        chi: Distrubition parameter.
        psi: Distrubition parameter.
        lamb: Distrubition parameter.

    Returns:
        Array: The cumulative distribution function values.
    """
    params = gig_params_dict(lamb=lamb, chi=chi, psi=psi)
    return _cdf(pdf_func=pdf, lower_bound=support()[0], x=x, params=params)


__cdf = deepcopy(cdf)
cdf = custom_vjp(cdf)


def cdf_fwd(x: ArrayLike, lamb: float, chi: float, psi: float) -> tuple[Array, tuple]:
    params = gig_params_dict(lamb=lamb, chi=chi, psi=psi)
    cdf_vals, (pdf_vals, param_grads) = _cdf_fwd(pdf_func=pdf, cdf_func=__cdf, x=x, params=params)
    return cdf_vals, (pdf_vals, param_grads['lamb'], param_grads['chi'], param_grads['psi'])


cdf.defvjp(cdf_fwd, cdf_bwd)


def logcdf(x: ArrayLike, lamb: float = 1.0, chi: float = 1.0, psi: float = 1.0) -> Array:
    r"""Log-cumulative distribution function of the Generalized Inverse Gaussian
    (GIG) distribution.
    
    Args:
        x: The input array.
        chi: Distrubition parameter.
        psi: Distrubition parameter.
        lamb: Distrubition parameter.

    Returns:
        Array: The log-cumulative distribution function values.
    """
    return lax.log(cdf(x=x, lamb=lamb, chi=chi, psi=psi))


def ppf(q: ArrayLike, lamb: float = 1.0, chi: float = 1.0, psi: float = 1.0) -> Array:
    r"""Percent point function (inverse of cdf) of the Generalized Inverse 
    Gaussian (GIG) distribution.
    
    Args:
        q: arraylike, value(s) at which to evaluate the ppf.
        chi: Distrubition parameter.
        psi: Distrubition parameter.
        lamb: Distrubition parameter.

    Returns:
        Array: The percent point function values.
    """
    lamb, chi, psi = gig_args_check(lamb=lamb, chi=chi, psi=psi)
    params = gig_params_dict(lamb=lamb, chi=chi, psi=psi)
    mean: float = stats(lamb=lamb, chi=chi, psi=psi)['mean']
    return _ppf(cdf_func=cdf, bounds=support(), q=q, params=params, x0=mean)


def _devroye(x, alpha, lamb):
    return -alpha * (jnp.cosh(x) - 1) - lamb * (jnp.exp(x) - x - 1)


def _devroye_grad(x, alpha, lamb):
    return -alpha * jnp.sinh(x) - lamb * (jnp.exp(x) - 1)


def _new_single_rv(carry, _):
    key, _, stop, count, constants = carry
    lamb, alpha, t, s, t_, s_, eta, zeta, theta, xi, p, r, q = constants


    key, subkey = random.split(key)
    u, v, w = random.uniform(subkey, shape=(3, ))

    x = jnp.where(u < (q + r) / (q + p + r), t_ + r * lax.log(1 / v), -s_ - p * lax.log(1 / v ))
    x = jnp.where(u < q / (q + p + r), -s_ + q * v, x)

    # checking stopping condition
    chi = jnp.where(jnp.logical_and(-s_ < x, x < t_), 1.0, 0.0) + jnp.where(t_ < x, jnp.exp(-eta - zeta * (x - t)) , 0.0) + jnp.where(x < -s_, jnp.exp(-theta + xi * (x + s)), 0.0)
    stop = w * chi <= jnp.exp(_devroye(x, alpha, lamb))

    return (key, x, stop, count + 1, constants), None


@jit
def _generate_single_rv(key: Array, constants: tuple) -> tuple[Array, Array]:
    maxiter = 10
    init = (key, jnp.array(jnp.nan), False, 0, constants)
    res = lax.scan((lambda carry, _: lax.cond(carry[2], (lambda carry, _: (carry, _)), _new_single_rv, carry, None)), init, None, maxiter)[0]
    return res[0], res[1]


def rvs(shape: tuple = (1, ), key: Array=DEFAULT_RANDOM_KEY, lamb: float = 1.0, chi: float = 1.0, psi: float = 1.0) -> Array:
    r"""Generate random samples from the Generalized Inverse Gaussian (GIG) 
    distribution.

    Uses the method outlined by Luc Devroye in "Random variate generation for 
    the generalized inverse Gaussian distribution" (2014).

    Note:
        If you intend to jit wrap this function, ensure that 'shape' is a 
        static argument.
    
    Args:
        shape: The shape of the random number array to generate.
        key: Key for random number generation.
        chi: Distrubition parameter.
        psi: Distrubition parameter.
        lamb: Distrubition parameter.
    """
    # getting parameters
    lamb, chi, psi = gig_args_check(lamb=lamb, chi=chi, psi=psi)
    sign_lamb: int = jnp.where(jnp.sign(lamb) >= 0, 1, -1)
    lamb: float = jnp.abs(lamb)
    omega: float = lax.sqrt(chi * psi)
    alpha: float = lax.sqrt(jnp.pow(omega, 2) + jnp.pow(lamb, 2)) - lamb

    # getting positive constant t
    _devroye_1: float = _devroye(x=1, alpha=alpha, lamb=lamb)
    t: float = jnp.where(-_devroye_1 > 2, lax.sqrt(2 / (alpha + lamb)), 1)
    t = jnp.where(-_devroye_1 < 0.5, lax.log(4/(alpha + 2*lamb)), t)
    
    # getting positive constant s
    _devroye_minus_1: float = _devroye(x=-1, alpha=alpha, lamb=lamb)
    s: float = jnp.where(-_devroye_minus_1 > 2, lax.sqrt(4 / (alpha * jnp.cosh(1) + lamb)), 1)
    s = jnp.where(-_devroye_minus_1 < 0.5, jnp.min(jnp.array([1 / lamb, lax.log(1 + (1 / alpha) + lax.sqrt(jnp.pow(alpha, -2) + (2 / alpha))),]))  , s)
    
    # Computing constants
    eta, zeta, theta, xi = -_devroye(x=t, alpha=alpha, lamb=lamb), -_devroye_grad(x=t, alpha=alpha, lamb=lamb), -_devroye(x=-s, alpha=alpha, lamb=lamb), _devroye_grad(x=-s, alpha=alpha, lamb=lamb)
    p, r = 1 / xi, 1 / zeta
    t_: float = t - r * eta
    s_: float = s - p * theta
    q : float = t_ + s_

    # Generating random variables
    constants: tuple = (lamb, alpha, t, s, t_, s_, eta, zeta, theta, xi, p, r, q)
    num_samples: int = 1
    for _ in shape:
        num_samples *= _
    X: jnp.ndarray = lax.scan((lambda key, _ : _generate_single_rv(key, constants)), key, None, num_samples)[1]

    frac: float = lax.div(lamb, omega)
    c: float = frac + lax.sqrt(1 + lax.pow(frac, 2))

    return jnp.pow((c * jnp.exp(X)), sign_lamb).reshape(shape)


def _mle_objective(params: dict, x: Array) -> float:
    lamb, chi, psi = params
    return __mle_objective(
        logpdf_func=logpdf, x=x, 
        params=gig_params_dict(lamb=lamb, chi=chi, psi=psi))


def _fit_mle(x: Array) -> tuple[dict, float]:
    eps = 1e-8
    constraints: tuple = (jnp.array([[-jnp.inf, eps, eps]]).T, 
                          jnp.array([[jnp.inf, jnp.inf, jnp.inf]]).T)
    
    projection_options: dict = {'hyperparams': constraints}

    key1, key = random.split(DEFAULT_RANDOM_KEY)
    key2, key3 = random.split(key)
    params0: jnp.ndarray = jnp.array([random.normal(key1, ()), 
                                      random.uniform(key2, (), minval=eps), 
                                      random.uniform(key3, (), minval=eps)])
    
    res = projected_gradient(f=_mle_objective, x0=params0, projection_method='projection_box', projection_options=projection_options, x=x)
    lamb, chi, psi = res['x']
    return gig_params_dict(lamb=lamb, chi=chi, psi=psi)#, res['fun']


def fit(x: Array) -> dict:
    r"""Fit the Generalized Inverse Gaussian distribution to the data using MLE.
    
    Args:
        x: arraylike, data to fit the distribution to.

    Returns:
        dictionary of fitted parameters.
    """
    x: jnp.ndarray = _univariate_input(x)[0]
    return _fit_mle(x)


def stats(lamb = 1.0, chi = 1.0, psi = 1.0) -> dict:
    r"""Distribution statistics for the Generalized Inverse Gaussian 
    distribution (GIG). Returns the mean, variance and mode of the 
    distribution.
    
    Args:
        chi: Distrubition parameter.
        psi: Distrubition parameter.
        lamb: Distrubition parameter.

    Returns:
        dict: Dictionary containing the distribution statistics.
    """
    lamb, chi, psi = gig_args_check(lamb=lamb, chi=chi, psi=psi)

    # calculating mean
    r: float = lax.sqrt(lax.mul(chi, psi))
    frac: float = lax.div(chi, psi)
    kv_lamb: float = kv(lamb, r)
    kv_lamb_plus_1: float = kv(lamb + 1, r)
    mean: float = lax.mul(lax.pow(frac, 0.5), lax.div(kv_lamb_plus_1, kv_lamb))

    # calculating variance
    kv_lamb_plus_2: float = kv(lamb + 2, r)
    second_moment: float = lax.mul(frac, lax.div(kv_lamb_plus_2, kv_lamb))
    variance: float = lax.sub(second_moment, lax.pow(mean, 2))

    # mode
    mode: float = lax.div((lamb - 1) + lax.sqrt(lax.pow(lamb - 1, 2) + lax.mul(chi, psi)), psi)

    return {'mean': mean, 'variance': variance, 'mode': mode}


def loglikelihood(x: ArrayLike, lamb: float = 1.0, chi: float = 1.0, psi: float = 1.0) -> float:
    r"""Log-likelihood of the Generalized Inverse Gaussian distribution.
    
    Args:
        x: arraylike, data to evaluate the log-likelihood at.
        chi: Distrubition parameter.
        psi: Distrubition parameter.
        lamb: Distrubition parameter.

    Returns:
        float: The log-likelihood of the data given the parameters.
    """
    return _loglikelihood(
        logpdf_func=logpdf, x=x, 
        params=gig_params_dict(lamb=lamb, chi=chi, psi=psi))


def aic(x: ArrayLike, lamb: float = 1.0, chi: float = 1.0, psi: float = 1.0) -> float:
    r"""Akaike Information Criterion (AIC) of the Generalized Inverse Gaussian 
    (GIG) distribution.

    Args:
        x: arraylike, data to evaluate the AIC at.
        chi: Distrubition parameter.
        psi: Distrubition parameter.
        lamb: Distrubition parameter.

    Returns:
        float: The AIC of the data given the parameters.
    """
    return _aic(
        logpdf_func=logpdf, x=x, 
        params=gig_params_dict(lamb=lamb, chi=chi, psi=psi))


def bic(x: ArrayLike, lamb: float = 1.0, chi: float = 1.0, psi: float = 1.0) -> float:
    r"""Bayesian Information Criterion (BIC) of the Generalized Inverse Gaussian 
    (GIG) distribution.

    Args:
        x: arraylike, data to evaluate the BIC at.
        chi: Distrubition parameter.
        psi: Distrubition parameter.
        lamb: Distrubition parameter.

    Returns:
        float: The BIC of the data given the parameters.
    """
    return _bic(
        logpdf_func=logpdf, x=x, 
        params=gig_params_dict(lamb=lamb, chi=chi, psi=psi))