"""File containing the copuLAX implementaqtion of the Generalized Inverse 
Gaussian distribution."""
import jax.numpy as jnp
from jax import random, lax, custom_vjp, jit
from jax._src.typing import ArrayLike, Array
from copy import deepcopy

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.univariate._cdf import _cdf, cdf_bwd, _cdf_fwd
from copulax._src.optimize import projected_gradient
from copulax.special import kv


class GIGBase(Univariate):
    @classmethod
    def _params_dict(cls, lamb: Scalar, chi: Scalar, psi: Scalar) -> dict:
        d: dict = {"lambda": lamb, "chi": chi, "psi": psi}
        return cls._args_transform(d)
    
    @staticmethod
    def _params_to_tuple(params: dict) -> tuple:
        params = GIGBase._args_transform(params)
        return params["lambda"], params["chi"], params["psi"]
    
    @staticmethod
    def _params_to_array(params: dict) -> Array:
        return jnp.asarray(GIGBase._params_to_tuple(params)).flatten()
    
    @classmethod
    def _support(cls, *args, **kwargs) -> tuple:
        return 0.0, jnp.inf
    
    def example_params(self, *args, **kwargs):
        r"""Example parameters for the GIG distribution.
        
        This is a three parameter family of continuous distributions, 
        with the GIG being defined by shape parameters `lambda`, `chi`, 
        and `psi`. Here, we adopt the parameterization used by McNeil 
        et al. (2005)"""
        return self._params_dict(lamb=1.0, chi=1.0, psi=1.0)
    
    @staticmethod
    def _stable_logpdf(stability: Scalar, x: ArrayLike, params: dict) -> Array:
        lamb, chi, psi = GIGBase._params_to_tuple(params)
        x, xshape = _univariate_input(x)

        var = lax.add(lax.mul(lamb - 1, lax.log(x)), 
                    -0.5 * (lax.mul(chi, lax.pow(x, -1)) + lax.mul(psi, x)))

        cT = lax.mul(0.5*lamb, lax.log((psi/chi) + stability))
        kv_val = kv(lamb, lax.pow(lax.mul(chi, psi), 0.5))
        cB = lax.log(stability + 2 * kv_val)
        
        c = lax.sub(cT, cB)
        logpdf_raw = lax.add(var, c)
        logpdf: jnp.ndarray = jnp.where(jnp.isnan(logpdf_raw), -jnp.inf, logpdf_raw)
        return logpdf.reshape(xshape)
    
    @staticmethod
    def logpdf(x: ArrayLike, params: dict) -> Array:
        return GIGBase._stable_logpdf(stability=0.0, x=x, params=params)
    
    @staticmethod
    def pdf(x: ArrayLike, params: dict) -> Array:
        return lax.exp(GIGBase.logpdf(x=x, params=params))
    
    # ppf
    # def _get_x0(self, params: dict) -> Scalar:
    #     return self.stats(params=params)['mode']

    # sampling
    # Uses the method outlined by Luc Devroye in "Random variate generation for 
    # the generalized inverse Gaussian distribution" (2014).
    def _devroye(self, x, alpha, lamb):
        return -alpha * (jnp.cosh(x) - 1) - lamb * (jnp.exp(x) - x - 1)

    def _devroye_grad(self, x, alpha, lamb):
        return -alpha * jnp.sinh(x) - lamb * (jnp.exp(x) - 1)
    
    def _new_single_rv(self, carry, _):
        key, _, stop, count, constants = carry
        lamb, alpha, t, s, t_, s_, eta, zeta, theta, xi, p, r, q = constants


        key, subkey = random.split(key)
        u, v, w = random.uniform(subkey, shape=(3, ))

        x = jnp.where(u < (q + r) / (q + p + r), t_ + r * lax.log(1 / v), -s_ - p * lax.log(1 / v ))
        x = jnp.where(u < q / (q + p + r), -s_ + q * v, x)

        # checking stopping condition
        chi = jnp.where(jnp.logical_and(-s_ < x, x < t_), 1.0, 0.0) + jnp.where(t_ < x, jnp.exp(-eta - zeta * (x - t)) , 0.0) + jnp.where(x < -s_, jnp.exp(-theta + xi * (x + s)), 0.0)
        stop = w * chi <= jnp.exp(self._devroye(x, alpha, lamb))

        return (key, x, stop, count + 1, constants), None
    
    @jit
    def _generate_single_rv(self, key: Array, constants: tuple) -> tuple[Array, Array]:
        maxiter = 10
        init = (key, jnp.array(jnp.nan), False, 0, constants)
        res = lax.scan((lambda carry, _: lax.cond(carry[2], (lambda carry, _: (carry, _)), self._new_single_rv, carry, None)), init, None, maxiter)[0]
        return res[0], res[1]
    
    def rvs(self, size: tuple | Scalar, params: dict, key: Array=DEFAULT_RANDOM_KEY) -> Array:
        # getting parameters
        lamb, chi, psi = self._params_to_tuple(params)
        sign_lamb: int = jnp.where(jnp.sign(lamb) >= 0, 1, -1)
        lamb: float = jnp.abs(lamb)
        omega: float = lax.sqrt(chi * psi)
        alpha: float = lax.sqrt(jnp.pow(omega, 2) + jnp.pow(lamb, 2)) - lamb

        # getting positive constant t
        _devroye_1: float = self._devroye(x=1, alpha=alpha, lamb=lamb)
        t: float = jnp.where(-_devroye_1 > 2, lax.sqrt(2 / (alpha + lamb)), 1)
        t = jnp.where(-_devroye_1 < 0.5, lax.log(4/(alpha + 2*lamb)), t)
        
        # getting positive constant s
        _devroye_minus_1: float = self._devroye(x=-1, alpha=alpha, lamb=lamb)
        s: float = jnp.where(-_devroye_minus_1 > 2, lax.sqrt(4 / (alpha * jnp.cosh(1) + lamb)), 1)
        s = jnp.where(-_devroye_minus_1 < 0.5, jnp.min(jnp.array([1 / lamb, lax.log(1 + (1 / alpha) + lax.sqrt(jnp.pow(alpha, -2) + (2 / alpha))),]))  , s)
        
        # Computing constants
        eta, zeta, theta, xi = -self._devroye(x=t, alpha=alpha, lamb=lamb), -self._devroye_grad(x=t, alpha=alpha, lamb=lamb), -self._devroye(x=-s, alpha=alpha, lamb=lamb), self._devroye_grad(x=-s, alpha=alpha, lamb=lamb)
        p, r = 1 / xi, 1 / zeta
        t_: float = t - r * eta
        s_: float = s - p * theta
        q : float = t_ + s_

        # Generating random variables
        constants: tuple = (lamb, alpha, t, s, t_, s_, eta, zeta, theta, xi, p, r, q)
        if isinstance(size, (int, float)):
            num_samples: int = int(size)
        else:
            num_samples: int = 1
            for number in size:
                num_samples *= number

        X: jnp.ndarray = lax.scan((lambda key, _ : self._generate_single_rv(key, constants)), key, None, num_samples)[1]

        frac: float = lax.div(lamb, omega)
        c: float = frac + lax.sqrt(1 + lax.pow(frac, 2))

        return jnp.pow((c * jnp.exp(X)), sign_lamb).reshape(size)
    
    # stats
    def _sample_estimates(self, params: dict, analytical_mean: Scalar, analytical_variance: Scalar) -> tuple[Scalar, Scalar]:
        sample = self.rvs(size=1000, params=params)
        sample_mean = jnp.mean(sample)
        sample_variance = jnp.var(sample)
        return (sample_mean, sample_variance)

    def stats(self, params: dict) -> dict:
        lamb, chi, psi = self._params_to_tuple(params)

        # calculating mean
        r: float = lax.sqrt(lax.mul(chi, psi))
        frac: float = lax.div(chi, psi)
        kv_lamb: float = kv(lamb, r)
        kv_lamb_plus_1: float = kv(lamb + 1, r)
        analytical_mean: float = lax.mul(lax.pow(frac, 0.5), lax.div(kv_lamb_plus_1, kv_lamb))

        # calculating variance
        kv_lamb_plus_2: float = kv(lamb + 2, r)
        second_moment: float = lax.mul(frac, lax.div(kv_lamb_plus_2, kv_lamb))
        analytical_variance: float = lax.sub(second_moment, lax.pow(analytical_mean, 2))

        # accounting for numerical instability
        # when psi is very large, the first and second moments can be 
        # NaN due to divide by zero error. Resolving this by using sample
        # mean and variance estimates in these cases.
        mean, variance = lax.cond(
            jnp.logical_or(jnp.isnan(analytical_mean), jnp.isnan(analytical_variance)), 
            self._sample_estimates, 
            (lambda params, analytical_mean, analytical_variance: (analytical_mean, analytical_variance)), params, analytical_mean, analytical_variance)
        
        std: float = lax.sqrt(variance)

        # mode
        mode: float = lax.div((lamb - 1) + lax.sqrt(lax.pow(lamb - 1, 2) + lax.mul(chi, psi)), psi)
        
        return self._scalar_transform({
            'mean': mean, 
            'variance': variance, 
            'std': std,
            'mode': mode})
    
    # fitting
    def _fit_mle(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        eps = 1e-8
        constraints: tuple = (jnp.array([[-jnp.inf, eps, eps]]).T, 
                            jnp.array([[jnp.inf, jnp.inf, jnp.inf]]).T)
        
        projection_options: dict = {'hyperparams': constraints}

        key1, key = random.split(DEFAULT_RANDOM_KEY)
        key2, key3 = random.split(key)
        params0: jnp.ndarray = jnp.array([
            random.normal(key1, ()), 
            random.uniform(key2, (), minval=eps), 
            random.uniform(key3, (), minval=eps)])
        
        res = projected_gradient(
            f=self._mle_objective, x0=params0, projection_method='projection_box', 
            projection_options=projection_options, x=x, lr=lr, maxiter=maxiter)
        lamb, chi, psi = res['x']
        return self._params_dict(lamb=lamb, chi=chi, psi=psi)#, res['fun']
    
    def fit(self, x: ArrayLike, lr: float = 0.1, maxiter: int = 100) -> dict:
        r"""Fit the distribution to the input data.
        
        Args:
            x (ArrayLike): The input data to fit the distribution to.
            lr (float): Learning rate for optimization.
            maxiter (int): Maximum number of iterations for optimization.
        
        Returns:
            dict: The fitted distribution parameters.
        """ 
        x: jnp.ndarray = _univariate_input(x)[0]
        return self._fit_mle(x=x, lr=lr, maxiter=maxiter)
    
    # cdf
    @staticmethod
    def _params_from_array(params_arr, *args, **kwargs) -> dict:
        lamb, chi, psi = params_arr
        return GIGBase._params_dict(lamb=lamb, chi=chi, psi=psi)

    @staticmethod
    def _pdf_for_cdf(x: ArrayLike, *params_tuple) -> Array:
        params_array: jnp.ndarray = jnp.asarray(params_tuple).flatten()
        params: dict = GIGBase._params_from_array(params_array)
        return GIG.pdf(x=x, params=params)
    

def _vjp_cdf(x: ArrayLike, params: dict) -> Array:
    params: dict = GIGBase._args_transform(params)
    return _cdf(dist=GIGBase, x=x, params=params)


_vjp_cdf_copy = deepcopy(_vjp_cdf)
_vjp_cdf = custom_vjp(_vjp_cdf)


def cdf_fwd(x: ArrayLike, params: dict) -> tuple[Array, tuple]:
    params = GIGBase._args_transform(params)
    return _cdf_fwd(dist=GIGBase, cdf_func=_vjp_cdf_copy, x=x, params=params)


_vjp_cdf.defvjp(cdf_fwd, cdf_bwd)


class GIG(GIGBase):
    r"""The Generalized Inverse Gaussian distribution is a 3 parameter family 
    of continuous distributions.
    
    We adopt the parameterization used by McNeil et al. (2005)

    https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution

    :math: `\lambda` is real-valued.
    :math: `\chi` is strictly positive.
    :math: `\psi` is strictly positive.
    """
    def cdf(self, x: ArrayLike, params: dict) -> Array:
        return _vjp_cdf(x=x, params=params)


gig = GIG("GIG")