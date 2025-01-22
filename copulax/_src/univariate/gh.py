"""File containing the copulAX implementation of the generalized hyperbolic distribution."""
import jax.numpy as jnp
from jax import lax, custom_vjp, random
from jax._src.typing import ArrayLike, Array
from copy import deepcopy

from copulax._src.univariate._utils import _univariate_input, DEFAULT_RANDOM_KEY
from copulax._src.univariate._ppf import _ppf
from copulax._src.univariate._cdf import _cdf, cdf_bwd, _cdf_fwd
from copulax._src.optimize import projected_gradient
from copulax.special import kv
from copulax._src.univariate import gig, normal


def gh_args_check(lamb: float | ArrayLike, chi: float | ArrayLike, psi: float | ArrayLike, mu: float | ArrayLike, sigma: float | ArrayLike, gamma: float | ArrayLike) -> tuple:
    return (jnp.asarray(lamb, dtype=float), jnp.asarray(chi, dtype=float), jnp.asarray(psi, dtype=float), jnp.asarray(mu, dtype=float), jnp.asarray(sigma, dtype=float), jnp.asarray(gamma, dtype=float))


def gh_params_dict(lamb: ArrayLike, chi: ArrayLike, psi: ArrayLike, mu: ArrayLike, sigma: ArrayLike, gamma: ArrayLike) -> dict:
    lamb, chi, psi, mu, sigma, gamma = gh_args_check(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)
    return {
        'lamb': lamb,
        'chi': chi,
        'psi': psi,
        'mu': mu,
        'sigma': sigma,
        'gamma': gamma,
        
    }


def support(*args) -> tuple[float, float]:
    r"""The support of the distribution is the subset of x for which the pdf 
    is non-zero. 
    
    Returns:
        (float, float): Tuple containing the support of the distribution.
    """
    return -jnp.inf, jnp.inf


def logpdf(x: ArrayLike, lamb: float = 0.0, chi: float = 1.0, psi: float = 1.0, mu: float = 0.0, sigma: float = 1.0,  gamma: float = 0.0) -> Array:
    r"""Log-probability density function of the generalized hyperbolic distribution.
    
    The generalized hyperbolic pdf is defined as:

    .. math::

        f(x|\mu, \sigma, \chi, \psi, \gamma, \lambda) \propto e^{\frac{(x-\mu)\gamma}{\sigma^2}} \frac{K_{\lambda - 0.5}(\sqrt{(\chi + (\frac{x-\mu}{\sigma})^2)(\psi + (\frac{\gamma}{\sigma})^2 })}{(\sqrt{(\chi + (\frac{x-\mu}{\sigma})^2)(\psi + (\frac{\gamma}{\sigma})^2 })^{0.5-\lambda}}

    where :math:`K_{\lambda}` is the modified Bessel function of the second kind.

    Args:
        x: arraylike, value(s) at which to evaluate the pdf.
        mu: location parameter of the generalized hyperbolic distribution.
        sigma: Scale / dispersion parameter of the generalized hyperbolic distribution.
        chi: Shape parameter of the generalized hyperbolic distribution.
        psi: Shape parameter of the generalized hyperbolic distribution.
        gamma: Skewness parameter of the generalized hyperbolic distribution.
        lamb: Shape parameter of the generalized hyperbolic distribution.

    Returns:
        array of log-pdf values.
    """
    x, xshape = _univariate_input(x)
    lamb, chi, psi, mu, sigma, gamma = gh_args_check(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)

    r: float = lax.sqrt(lax.mul(chi, psi))
    s: float = 0.5 - lamb
    h: float = lax.add(psi, lax.pow(lax.div(gamma, sigma), 2))
    g = lax.div(lax.sub(x, mu), lax.pow(sigma, 2))

    m = lax.sqrt(lax.mul(lax.add(chi, lax.mul(g, lax.sub(x, mu))), h))

    T = lax.add(lax.log(kv(-s, m)), lax.mul(g, gamma))
    B = lax.mul(lax.log(m), s)

    cT = lax.add(lax.mul(lamb, lax.sub(lax.log(psi), lax.log(r))), lax.mul(lax.log(h), s)) 
    cB = lax.add(lax.add(lax.log(sigma), lax.log(lax.sqrt(2*jnp.pi))), lax.log(kv(lamb, r)))

    c = lax.sub(cT, cB)
    logpdf: jnp.ndarray = lax.add(lax.sub(T, B), c)
    return logpdf.reshape(xshape)


def pdf(x: ArrayLike, lamb: float = 0.0, chi: float = 1.0, psi: float = 1.0, mu: float = 0.0, sigma: float = 1.0,  gamma: float = 0.0) -> Array:
    r"""Probability density function of the generalized hyperbolic distribution.
    
    The generalized hyperbolic pdf is defined as:

    .. math::

        f(x|\mu, \sigma, \chi, \psi, \gamma, \lambda) \propto e^{\frac{(x-\mu)\gamma}{\sigma^2}} \frac{K_{\lambda - 0.5}(\sqrt{(\chi + (\frac{x-\mu}{\sigma})^2)(\psi + (\frac{\gamma}{\sigma})^2 })}{(\sqrt{(\chi + (\frac{x-\mu}{\sigma})^2)(\psi + (\frac{\gamma}{\sigma})^2 })^{0.5-\lambda}}

    where :math:`K_{\lambda}` is the modified Bessel function of the second kind.

    Args:
        x: arraylike, value(s) at which to evaluate the pdf.
        mu: location parameter of the generalized hyperbolic distribution.
        sigma: Scale / dispersion parameter of the generalized hyperbolic distribution.
        chi: Shape parameter of the generalized hyperbolic distribution.
        psi: Shape parameter of the generalized hyperbolic distribution.
        gamma: Skewness parameter of the generalized hyperbolic distribution.
        lamb: Shape parameter of the generalized hyperbolic distribution.

    Returns:
        array of pdf values.
    """
    return jnp.exp(logpdf(x=x, lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma))


def cdf(x: ArrayLike, lamb = 0.0, chi = 1.0, psi = 1.0, mu = 0.0, sigma = 1.0,  gamma = 0.0) -> Array:
    r"""Cumulative distribution function of the generalized hyperbolic distribution.
    
    The generalized hyperbolic pdf is defined as:

    .. math::

        f(x|\mu, \sigma, \chi, \psi, \gamma, \lambda) \propto e^{\frac{(x-\mu)\gamma}{\sigma^2}} \frac{K_{\lambda - 0.5}(\sqrt{(\chi + (\frac{x-\mu}{\sigma})^2)(\psi + (\frac{\gamma}{\sigma})^2 })}{(\sqrt{(\chi + (\frac{x-\mu}{\sigma})^2)(\psi + (\frac{\gamma}{\sigma})^2 })^{0.5-\lambda}}

    where :math:`K_{\lambda}` is the modified Bessel function of the second kind.

    Args:
        x: arraylike, value(s) at which to evaluate the pdf.
        mu: location parameter of the generalized hyperbolic distribution.
        sigma: Scale / dispersion parameter of the generalized hyperbolic distribution.
        chi: Shape parameter of the generalized hyperbolic distribution.
        psi: Shape parameter of the generalized hyperbolic distribution.
        gamma: Skewness parameter of the generalized hyperbolic distribution.
        lamb: Shape parameter of the generalized hyperbolic distribution.

    Returns:
        array of cdf values.
    """
    params = gh_params_dict(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)
    return _cdf(pdf_func=pdf, lower_bound=-jnp.inf, x=x, params=params)


__cdf = deepcopy(cdf)
cdf = custom_vjp(cdf)


def cdf_fwd(x: ArrayLike, lamb, chi, psi, mu, sigma, gamma) -> tuple[Array, tuple]:
    params = gh_params_dict(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)
    cdf_vals, (pdf_vals, param_grads) = _cdf_fwd(pdf_func=pdf, cdf_func=__cdf, x=x, params=params)
    return cdf_vals, (pdf_vals, param_grads['lamb'], param_grads['chi'], param_grads['psi'], param_grads['mu'], param_grads['sigma'], param_grads['gamma'])


cdf.defvjp(cdf_fwd, cdf_bwd)


def logcdf(x: ArrayLike, lamb: float = 0.0, chi: float = 1.0, psi: float = 1.0, mu: float = 0.0, sigma: float = 1.0,  gamma: float = 0.0) -> Array:
    r"""Log of the cumulative distribution function of the generalized 
    hyperbolic distribution.
    
    The generalized hyperbolic pdf is defined as:

    .. math::

        f(x|\mu, \sigma, \chi, \psi, \gamma, \lambda) \propto e^{\frac{(x-\mu)\gamma}{\sigma^2}} \frac{K_{\lambda - 0.5}(\sqrt{(\chi + (\frac{x-\mu}{\sigma})^2)(\psi + (\frac{\gamma}{\sigma})^2 })}{(\sqrt{(\chi + (\frac{x-\mu}{\sigma})^2)(\psi + (\frac{\gamma}{\sigma})^2 })^{0.5-\lambda}}

    where :math:`K_{\lambda}` is the modified Bessel function of the second kind.

    Args:
        x: arraylike, value(s) at which to evaluate the pdf.
        mu: location parameter of the generalized hyperbolic distribution.
        sigma: Scale / dispersion parameter of the generalized hyperbolic distribution.
        chi: Shape parameter of the generalized hyperbolic distribution.
        psi: Shape parameter of the generalized hyperbolic distribution.
        gamma: Skewness parameter of the generalized hyperbolic distribution.
        lamb: Shape parameter of the generalized hyperbolic distribution.

    Returns:
        array of log-cdf values.
    """
    return jnp.log(cdf(x, lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma))


def ppf(q: ArrayLike, lamb: float = 0.0, chi: float = 1.0, psi: float = 1.0, mu: float = 0.0, sigma: float = 1.0,  gamma: float = 0.0) -> Array:
    r"""Percent point function (inverse of cdf) of the generalized hyperbolic distribution.
    
    The generalized hyperbolic pdf is defined as:

    .. math::

        f(x|\mu, \sigma, \chi, \psi, \gamma, \lambda) \propto e^{\frac{(x-\mu)\gamma}{\sigma^2}} \frac{K_{\lambda - 0.5}(\sqrt{(\chi + (\frac{x-\mu}{\sigma})^2)(\psi + (\frac{\gamma}{\sigma})^2 })}{(\sqrt{(\chi + (\frac{x-\mu}{\sigma})^2)(\psi + (\frac{\gamma}{\sigma})^2 })^{0.5-\lambda}}
    
    where :math:`K_{\lambda}` is the modified Bessel function of the second kind.

    Args:
        q: arraylike, value(s) at which to evaluate the ppf.
        mu: location parameter of the generalized hyperbolic distribution.
        sigma: Scale / dispersion parameter of the generalized hyperbolic distribution.
        chi: Shape parameter of the generalized hyperbolic distribution.
        psi: Shape parameter of the generalized hyperbolic distribution.
        gamma: Skewness parameter of the generalized hyperbolic distribution.
        lamb: Shape parameter of the generalized hyperbolic distribution.

    Returns:
        array of ppf values.
    """
    mean: float = stats(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)['mean']
    return _ppf(cdf_func=cdf, bounds=(-jnp.inf, jnp.inf), q=q, x0=mean,
                params=gh_params_dict(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma))



def rvs(shape: tuple = (1,), key: Array = DEFAULT_RANDOM_KEY, lamb: float = 0.0, chi: float = 1.0, psi: float = 1.0, mu: float = 0.0, sigma: float = 1.0,  gamma: float = 0.0) -> Array:
    r"""Generate random variates from the generalized hyperbolic distribution.
    
    The generalized hyperbolic pdf is defined as:

    .. math::

        f(x|\mu, \sigma, \chi, \psi, \gamma, \lambda) \propto e^{\frac{(x-\mu)\gamma}{\sigma^2}} \frac{K_{\lambda - 0.5}(\sqrt{(\chi + (\frac{x-\mu}{\sigma})^2)(\psi + (\frac{\gamma}{\sigma})^2 })}{(\sqrt{(\chi + (\frac{x-\mu}{\sigma})^2)(\psi + (\frac{\gamma}{\sigma})^2 })^{0.5-\lambda}}
    
    where :math:`K_{\lambda}` is the modified Bessel function of the second kind.

    Note:
        If you intend to jit wrap this function, ensure that 'shape' is a 
        static argument.

    Args:
        shape: The shape of the random number array to generate.
        key: Key for random number generation.
        mu: location parameter of the generalized hyperbolic distribution.
        sigma: Scale / dispersion parameter of the generalized hyperbolic distribution.
        chi: Shape parameter of the generalized hyperbolic distribution.
        psi: Shape parameter of the generalized hyperbolic distribution.
        gamma: Skewness parameter of the generalized hyperbolic distribution.
        lamb: Shape parameter of the generalized hyperbolic distribution.

    """
    lamb, chi, psi, mu, sigma, gamma = gh_args_check(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)

    key1, key2 = random.split(key)
    W = gig.rvs(key=key1, shape=shape, chi=chi, psi=psi, lamb=lamb)
    Z = normal.rvs(key=key2, shape=shape, mu=0.0, sigma=1.0)

    m = mu + W * gamma
    s = lax.sqrt(W) * sigma * Z
    s = lax.mul(lax.sqrt(W) * sigma, Z)
    X = m + s
    return X.reshape(shape)


def _mle_objective(params, x) -> jnp.ndarray:
    lamb, chi, psi, mu, sigma, gamma = params
    return -jnp.sum(logpdf(x=x, lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma))


def _fit_mle(x: ArrayLike) -> tuple[dict, float]:
    eps = 1e-8
    constraints: tuple = (jnp.array([[-jnp.inf, eps, eps, -jnp.inf, eps, -jnp.inf]]).T, 
                          jnp.array([[jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf]]).T)
    
    projection_options: dict = {'hyperparams': constraints}

    key1, key = random.split(DEFAULT_RANDOM_KEY)
    key2, key3 = random.split(key)
    params0: jnp.ndarray = jnp.array([random.normal(key1, ()), 
                                      random.uniform(key2, (), minval=eps), 
                                      random.uniform(key3, (), minval=eps),  
                                      x.mean(), 
                                      x.std(), 
                                      0.0])
    
    res: dict = projected_gradient(f=_mle_objective, x0=params0, projection_method='projection_box', projection_options=projection_options, x=x)
    lamb, chi, psi, mu, sigma, gamma = res['x']
    return gh_params_dict(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)#, res['fun']


def _get_ldmle_params(lamb: float, chi: float, psi: float, gamma: float, sample_mean: float, sample_variance: float) -> float:
    # obtaining mu and sigma estimates
    gig_stats: dict = gig.stats(lamb=lamb, chi=chi, psi=psi)
    mu: float = sample_mean - gig_stats['mean'] * gamma
    sigma_sq: float = (sample_variance - gig_stats['variance'] * lax.pow(gamma, 2)) / gig_stats['mean'] 
    sigma: float = lax.sqrt(jnp.abs(sigma_sq))
    return mu, sigma


def _ldmle_objective(params: jnp.ndarray, x: jnp.ndarray, sample_mean: float, sample_variance: float) -> jnp.ndarray:
    lamb, chi, psi, gamma = params
    mu, sigma = _get_ldmle_params(lamb=lamb, chi=chi, psi=psi, gamma=gamma, sample_mean=sample_mean, sample_variance=sample_variance)
    return -jnp.sum(logpdf(x=x, lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma))


def _fit_ldmle(x: ArrayLike) -> tuple[dict, float]:
    r"""Fit the generalized hyperbolic distribution using low-dimensional 
    maximum likelihood estimation. This uses estimates of the sample mean and 
    variance of the data to remove the mu and sigma parameters from the 
    numerical optimization problem.
    
    Args:
        x: arraylike, data to fit the distribution to.

    Returns:
        dictionary of fitted parameters.
    """
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
    res = projected_gradient(f=_ldmle_objective, x0=params0, projection_method='projection_box', projection_options=projection_options, x=x, sample_mean=sample_mean, sample_variance=sample_variance)
    lamb, chi, psi, gamma = res['x']
    mu, sigma = _get_ldmle_params(lamb=lamb, chi=chi, psi=psi, gamma=gamma, sample_mean=sample_mean, sample_variance=sample_variance)
    return gh_params_dict(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)#, res['fun']


def fit(x: ArrayLike, method: str = 'LDMLE') -> tuple[dict, float]:
    r"""Fit the generalized hyperbolic distribution to the data.

    Note:
        If you intend to jit wrap this function, ensure that 'method' is a 
        static argument.
    
    Args:
        x: arraylike, data to fit the distribution to.
        method: str, method to use for fitting the distribution. Options are 
        'MLE' for maximum likelihood estimation, and 'LDMLE' for low-dimensional 
        maximum likelihood estimation. Defaults to 'LDMLE'.

    Returns:
        dictionary of fitted parameters.
    """
    x = _univariate_input(x)[0]
    if method == 'MLE':
        return _fit_mle(x)
    else:
        return _fit_ldmle(x)


def stats(lamb = 0.0, chi = 1.0, psi = 1.0, mu = 0.0, sigma = 1.0,  gamma = 0.0) -> dict:
    r"""Distribution stats for the generalized hyperbolic distribution.
    Returns the mean and variance of the distribution.
    
    The generalized hyperbolic pdf is defined as:

    .. math::

        f(x|\mu, \sigma, \chi, \psi, \gamma, \lambda) \propto e^{\frac{(x-\mu)\gamma}{\sigma^2}} \frac{K_{\lambda - 0.5}(\sqrt{(\chi + (\frac{x-\mu}{\sigma})^2)(\psi + (\frac{\gamma}{\sigma})^2 })}{(\sqrt{(\chi + (\frac{x-\mu}{\sigma})^2)(\psi + (\frac{\gamma}{\sigma})^2 })^{0.5-\lambda}}
    
    where :math:`K_{\lambda}` is the modified Bessel function of the second kind.

    Args:
        mu: location parameter of the generalized hyperbolic distribution.
        sigma: Scale / dispersion parameter of the generalized hyperbolic distribution.
        chi: Shape parameter of the generalized hyperbolic distribution.
        psi: Shape parameter of the generalized hyperbolic distribution.
        gamma: Skewness parameter of the generalized hyperbolic distribution.
        lamb: Shape parameter of the generalized hyperbolic distribution.

    Returns:
        dict: Dictionary containing the distribution statistics.
    """
    lamb, chi, psi, mu, sigma, gamma = gh_args_check(lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma)
    
    gig_stats: dict = gig.stats(lamb=lamb, chi=chi, psi=psi)
    mean: float = mu + lax.mul(gig_stats['mean'], gamma)
    var: float = lax.mul(gig_stats['mean'], lax.pow(sigma, 2)) + lax.mul(gig_stats['variance'], lax.pow(gamma, 2))
    return {'mean': mean,'variance': var}
