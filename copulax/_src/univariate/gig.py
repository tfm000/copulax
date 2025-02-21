"""File containing the copuLAX implementaqtion of the Generalized Inverse 
Gaussian distribution."""
import jax.numpy as jnp
from jax import random, lax, custom_vjp, jit
from jax._src.typing import ArrayLike, Array
from copy import deepcopy

from copulax._src.univariate._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.univariate._cdf import _cdf, cdf_bwd, _cdf_fwd
from copulax._src.optimize import projected_gradient
from copulax.special import kv


class GIGBase(Univariate):
    @staticmethod
    def _params_dict(lamb, chi, psi):
        lamb, chi, psi = GIGBase._args_transform(lamb, chi, psi)
        return {"lamb": lamb, "chi": chi, "psi": psi}
    
    @staticmethod
    def support(*args, **kwargs) -> tuple[Scalar, Scalar]:
        return jnp.array(0.0), jnp.array(jnp.inf)
    
    @staticmethod
    def _stable_logpdf(stability: Scalar, x: ArrayLike, lamb: Scalar = 1.0, chi: Scalar = 1.0, psi: Scalar = 1.0) -> Array:
        x, xshape = _univariate_input(x)
        lamb, chi, psi = GIGBase._args_transform(lamb, chi, psi)

        var = lax.add(lax.mul(lamb - 1, lax.log(x)), 
                    -0.5 * (lax.mul(chi, lax.pow(x, -1)) + lax.mul(psi, x)))

        cT = lax.mul(0.5*lamb, lax.log((psi/chi) + stability))
        kv_val = kv(lamb, lax.pow(lax.mul(chi, psi), 0.5))
        cB = lax.log(stability + 2 * kv_val)
        
        c = lax.sub(cT, cB)
        pdf_raw = lax.add(var, c)
        logpdf: jnp.ndarray = jnp.where(jnp.isnan(pdf_raw), -jnp.inf, pdf_raw)
        return logpdf.reshape(xshape)
    
    @staticmethod
    def logpdf(x: ArrayLike, lamb: Scalar = 1.0, chi: Scalar = 1.0, psi: Scalar = 1.0) -> Array:
        return GIGBase._stable_logpdf(stability=0.0, x=x, lamb=lamb, chi=chi, psi=psi)
    
    @staticmethod
    def pdf(x: ArrayLike, lamb: Scalar = 1.0, chi: Scalar = 1.0, psi: Scalar = 1.0) -> Array:
        return lax.exp(GIGBase.logpdf(x=x, lamb=lamb, chi=chi, psi=psi))
    
    def logcdf(self, x: ArrayLike, lamb: Scalar = 1.0, chi: Scalar = 1.0, psi: Scalar = 1.0) -> Array:
        return super().logcdf(x=x, lamb=lamb, chi=chi, psi=psi)
    
    def ppf(self, q: ArrayLike, lamb: Scalar = 1.0, chi: Scalar = 1.0, psi: Scalar = 1.0) -> Array:
        mean: Scalar = self.stats(lamb=lamb, chi=chi, psi=psi)['mean']
        return super().ppf(x0=mean, q=q, lamb=lamb, chi=chi, psi=psi)
    
    def inverse_cdf(self, q: ArrayLike, lamb: Scalar = 1.0, chi: Scalar = 1.0, psi: Scalar = 1.0) -> Array:
        return super().inverse_cdf(q=q, lamb=lamb, chi=chi, psi=psi)

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
    
    def rvs(self, shape: tuple = (), key: Array=DEFAULT_RANDOM_KEY, lamb: Scalar = 1.0, chi: Scalar = 1.0, psi: Scalar = 1.0) -> Array:
        # getting parameters
        lamb, chi, psi = self._args_transform(lamb=lamb, chi=chi, psi=psi)
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
        num_samples: int = 1
        for _ in shape:
            num_samples *= _
        X: jnp.ndarray = lax.scan((lambda key, _ : self._generate_single_rv(key, constants)), key, None, num_samples)[1]

        frac: float = lax.div(lamb, omega)
        c: float = frac + lax.sqrt(1 + lax.pow(frac, 2))

        return jnp.pow((c * jnp.exp(X)), sign_lamb).reshape(shape)
    
    def sample(self, shape: tuple = (), key: Array=DEFAULT_RANDOM_KEY, lamb: Scalar = 1.0, chi: Scalar = 1.0, psi: Scalar = 1.0) -> Array:
        return super().sample(shape=shape, key=key, lamb=lamb, chi=chi, psi=psi)
    
    # stats
    def stats(self, lamb: Scalar = 1.0, chi: Scalar = 1.0, psi: Scalar = 1.0) -> dict:
        lamb, chi, psi = self._args_transform(lamb=lamb, chi=chi, psi=psi)

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
    
    # metrics
    def loglikelihood(self, x: ArrayLike, lamb: Scalar = 1.0, chi: Scalar = 1.0, psi: Scalar = 1.0) -> Scalar:
        return super().loglikelihood(x=x, lamb=lamb, chi=chi, psi=psi)
    
    def aic(self, x: ArrayLike, lamb: Scalar = 1.0, chi: Scalar = 1.0, psi: Scalar = 1.0) -> Scalar:
        return super().aic(x=x, lamb=lamb, chi=chi, psi=psi)
    
    def bic(self, x: ArrayLike, lamb: Scalar = 1.0, chi: Scalar = 1.0, psi: Scalar = 1.0) -> Scalar:
        return super().bic(x=x, lamb=lamb, chi=chi, psi=psi)
    
    # fitting
    def _fit_mle(self, x: jnp.ndarray, *args, **kwargs) -> dict:
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
        
        res = projected_gradient(f=self._mle_objective, x0=params0, projection_method='projection_box', projection_options=projection_options, x=x)
        lamb, chi, psi = res['x']
        return self._params_dict(lamb=lamb, chi=chi, psi=psi)#, res['fun']
    
    def fit(self, x: ArrayLike, *args, **kwargs) -> dict:
        x: jnp.ndarray = _univariate_input(x)[0]
        return self._fit_mle(x=x, *args, **kwargs)
    
# cdf
def _vjp_cdf(x: ArrayLike, lamb: Scalar, chi: Scalar, psi: Scalar) -> Array:
    params: dict = GIGBase._params_dict(lamb=lamb, chi=chi, psi=psi)
    return _cdf(pdf_func=GIGBase.pdf, lower_bound=GIGBase.support()[0], x=x, params=params)


_vjp_cdf_copy = deepcopy(_vjp_cdf)
_vjp_cdf = custom_vjp(_vjp_cdf)


def cdf_fwd(x: ArrayLike, lamb: float, chi: float, psi: float) -> tuple[Array, tuple]:
    params = GIGBase._params_dict(lamb=lamb, chi=chi, psi=psi)
    cdf_vals, (pdf_vals, param_grads) = _cdf_fwd(pdf_func=GIGBase.pdf, cdf_func=_vjp_cdf_copy, x=x, params=params)
    return cdf_vals, (pdf_vals, param_grads['lamb'], param_grads['chi'], param_grads['psi'])


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
    def cdf(self, x: ArrayLike, lamb: Scalar = 1.0, chi: Scalar = 1.0, psi: Scalar = 1.0) -> Array:
        return _vjp_cdf(x=x, lamb=lamb, chi=chi, psi=psi)


gig = GIG("GIG")