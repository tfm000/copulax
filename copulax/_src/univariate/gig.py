"""File containing the copulAX implementation of the Generalized Inverse
Gaussian distribution."""

import jax.numpy as jnp
from jax import random, lax, custom_vjp, jit
from jax import Array
from jax.typing import ArrayLike
from copy import deepcopy

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import _resolve_key, get_local_random_key
from copulax._src.univariate._cdf import _cdf, cdf_bwd, _cdf_fwd
from copulax._src.optimize import projected_gradient
from copulax.special import kv, log_kv


class GIG(Univariate):
    r"""The Generalized Inverse Gaussian distribution is a 3 parameter family
    of continuous distributions.

    We adopt the parameterization used by McNeil et al. (2005)

    https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution

    :math: `\lambda` is real-valued.
    :math: `\chi` is strictly positive.
    :math: `\psi` is strictly positive.
    """

    lamb: Array = None
    chi: Array = None
    psi: Array = None

    def __init__(self, name="GIG", *, lamb=None, chi=None, psi=None):
        """Initialize the Generalized Inverse Gaussian distribution.

        Args:
            name: Display name for the distribution.
            lamb: Shape parameter (real-valued).
            chi: Concentration parameter (strictly positive).
            psi: Rate parameter (strictly positive).
        """
        super().__init__(name)
        self.lamb = (
            jnp.asarray(lamb, dtype=float).reshape(()) if lamb is not None else None
        )
        self.chi = (
            jnp.asarray(chi, dtype=float).reshape(()) if chi is not None else None
        )
        self.psi = (
            jnp.asarray(psi, dtype=float).reshape(()) if psi is not None else None
        )

    @property
    def _stored_params(self):
        """Return stored parameters if all are set, else None."""
        if self.lamb is None or self.chi is None or self.psi is None:
            return None
        return {"lamb": self.lamb, "chi": self.chi, "psi": self.psi}

    @classmethod
    def _params_dict(cls, lamb: Scalar, chi: Scalar, psi: Scalar) -> dict:
        """Create a parameter dictionary from lamb, chi, and psi values."""
        d: dict = {"lamb": lamb, "chi": chi, "psi": psi}
        return cls._args_transform(d)

    @staticmethod
    def _params_to_tuple(params: dict) -> tuple:
        """Extract (lamb, chi, psi) from the parameter dictionary."""
        params = GIG._args_transform(params)
        return params["lamb"], params["chi"], params["psi"]

    @staticmethod
    def _params_to_array(params: dict) -> Array:
        """Convert the parameter dictionary to a flat array."""
        return jnp.asarray(GIG._params_to_tuple(params)).flatten()

    @classmethod
    def _support(cls, *args, **kwargs) -> Array:
        """Return the support ``(0, inf)``."""
        return jnp.array([0.0, jnp.inf])

    def example_params(self, *args, **kwargs):
        r"""Example parameters for the GIG distribution.

        This is a three parameter family of continuous distributions,
        with the GIG being defined by shape parameters `lamb`, `chi`,
        and `psi`. Here, we adopt the parameterization used by McNeil
        et al. (2005)"""
        return self._params_dict(lamb=1.0, chi=1.0, psi=1.0)

    @staticmethod
    def _stable_logpdf(stability: Scalar, x: ArrayLike, params: dict) -> Array:
        """Compute the numerically stabilized log-PDF of the GIG distribution."""
        lamb, chi, psi = GIG._params_to_tuple(params)
        x, xshape = _univariate_input(x)

        var = lax.add(
            lax.mul(lamb - 1, lax.log(x)),
            -0.5 * (lax.mul(chi, lax.pow(x, -1)) + lax.mul(psi, x)),
        )

        cT = lax.mul(0.5 * lamb, lax.log((psi / (chi + stability)) + stability))
        cB = log_kv(lamb, lax.pow(lax.mul(chi, psi), 0.5)) + jnp.log(2.0)
        # kv_val = kv(lamb, lax.pow(lax.mul(chi, psi), 0.5))
        # cB = lax.log(stability + 2 * kv_val)

        c = lax.sub(cT, cB)
        logpdf_raw = lax.add(var, c)
        logpdf: jnp.ndarray = jnp.where(jnp.isnan(logpdf_raw), -jnp.inf, logpdf_raw)
        return logpdf.reshape(xshape)

    def logpdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the log probability density function."""
        params = self._resolve_params(params)
        logpdf = GIG._stable_logpdf(stability=0.0, x=x, params=params)
        return self._enforce_support_on_logpdf(x=x, logpdf=logpdf, params=params)

    # sampling
    # Uses the method outlined by Luc Devroye in "Random variate generation for
    # the generalized inverse Gaussian distribution" (2014).
    def _devroye(self, x, alpha, lamb):
        """Evaluate the Devroye (2014) acceptance log-density."""
        return -alpha * (jnp.cosh(x) - 1) - lamb * (jnp.exp(x) - x - 1)

    def _devroye_grad(self, x, alpha, lamb):
        """Gradient of the Devroye acceptance log-density."""
        return -alpha * jnp.sinh(x) - lamb * (jnp.exp(x) - 1)

    def _new_single_rv(self, carry, _):
        """One iteration of the Devroye rejection sampler."""
        key, _, stop, count, constants = carry
        lamb, alpha, t, s, t_, s_, eta, zeta, theta, xi, p, r, q = constants

        key, subkey = random.split(key)
        u, v, w = random.uniform(subkey, shape=(3,))

        x = jnp.where(
            u < (q + r) / (q + p + r), t_ + r * lax.log(1 / v), -s_ - p * lax.log(1 / v)
        )
        x = jnp.where(u < q / (q + p + r), -s_ + q * v, x)

        # checking stopping condition
        chi = (
            jnp.where(jnp.logical_and(-s_ < x, x < t_), 1.0, 0.0)
            + jnp.where(t_ < x, jnp.exp(-eta - zeta * (x - t)), 0.0)
            + jnp.where(x < -s_, jnp.exp(-theta + xi * (x + s)), 0.0)
        )
        stop = w * chi <= jnp.exp(self._devroye(x, alpha, lamb))

        return (key, x, stop, count + 1, constants), None

    @jit
    def _generate_single_rv(self, key: Array, constants: tuple) -> tuple[Array, Array]:
        """Generate a single GIG random variate using the Devroye (2014) algorithm."""
        maxiter = 10
        init = (key, jnp.array(jnp.nan), False, 0, constants)
        res = lax.scan(
            (
                lambda carry, _: lax.cond(
                    carry[2],
                    (lambda carry, _: (carry, _)),
                    self._new_single_rv,
                    carry,
                    None,
                )
            ),
            init,
            None,
            maxiter,
        )[0]
        return res[0], res[1]

    def rvs(
        self, size: tuple | Scalar, params: dict = None, key: Array = None
    ) -> Array:
        """Generate random variates using the Devroye (2014) rejection algorithm.

        Args:
            size: Shape of the output array.
            params: Distribution parameters. Uses stored parameters if None.
            key: JAX PRNG key. A default key is used if None.

        Returns:
            Array of GIG random samples.
        """
        params = self._resolve_params(params)
        key = _resolve_key(key)
        # getting parameters
        lamb, chi, psi = self._params_to_tuple(params)
        sign_lamb: int = jnp.where(jnp.sign(lamb) >= 0, 1, -1)
        lamb: float = jnp.abs(lamb)
        omega: float = lax.sqrt(chi * psi)
        alpha: float = lax.sqrt(jnp.pow(omega, 2) + jnp.pow(lamb, 2)) - lamb

        # getting positive constant t
        _devroye_1: float = self._devroye(x=1, alpha=alpha, lamb=lamb)
        t: float = jnp.where(-_devroye_1 > 2, lax.sqrt(2 / (alpha + lamb)), 1)
        t = jnp.where(-_devroye_1 < 0.5, lax.log(4 / (alpha + 2 * lamb)), t)

        # getting positive constant s
        _devroye_minus_1: float = self._devroye(x=-1, alpha=alpha, lamb=lamb)
        s: float = jnp.where(
            -_devroye_minus_1 > 2, lax.sqrt(4 / (alpha * jnp.cosh(1) + lamb)), 1
        )
        s = jnp.where(
            -_devroye_minus_1 < 0.5,
            jnp.min(
                jnp.array(
                    [
                        1 / lamb,
                        lax.log(
                            1 + (1 / alpha) + lax.sqrt(jnp.pow(alpha, -2) + (2 / alpha))
                        ),
                    ]
                )
            ),
            s,
        )

        # Computing constants
        eta, zeta, theta, xi = (
            -self._devroye(x=t, alpha=alpha, lamb=lamb),
            -self._devroye_grad(x=t, alpha=alpha, lamb=lamb),
            -self._devroye(x=-s, alpha=alpha, lamb=lamb),
            self._devroye_grad(x=-s, alpha=alpha, lamb=lamb),
        )
        p, r = 1 / xi, 1 / zeta
        t_: float = t - r * eta
        s_: float = s - p * theta
        q: float = t_ + s_

        # Generating random variables
        constants: tuple = (lamb, alpha, t, s, t_, s_, eta, zeta, theta, xi, p, r, q)
        if isinstance(size, (int, float)):
            num_samples: int = int(size)
        else:
            num_samples: int = 1
            for number in size:
                num_samples *= number

        X: jnp.ndarray = lax.scan(
            (lambda key, _: self._generate_single_rv(key, constants)),
            key,
            None,
            num_samples,
        )[1]

        frac: float = lax.div(lamb, omega)
        c: float = frac + lax.sqrt(1 + lax.pow(frac, 2))
        scale = lax.sqrt(lax.div(chi, psi))
        return (scale * jnp.pow((c * jnp.exp(X)), sign_lamb)).reshape(size)

    # stats
    def stats(self, params: dict = None) -> dict:
        """Compute distribution statistics (mean, variance, std, mode).

        Uses analytical formulas based on modified Bessel functions.
        Falls back to sample estimates when numerical instability causes NaN.
        """
        params = self._resolve_params(params)
        lamb, chi, psi = self._params_to_tuple(params)

        # calculating mean
        r: float = lax.sqrt(lax.mul(chi, psi))
        # frac: float = lax.div(chi, psi)
        # kv_lamb: float = kv(lamb, r)
        # kv_lamb_plus_1: float = kv(lamb + 1, r)
        # mean: float = lax.mul(
        #     lax.pow(frac, 0.5), lax.div(kv_lamb_plus_1, kv_lamb)
        # )
        log_frac: float = lax.log(chi) - lax.log(psi)
        log_kv_lamb: float = log_kv(lamb, r)
        log_kv_lamb_plus_1: float = log_kv(lamb + 1, r)
        log_mean: float = 0.5 * log_frac + log_kv_lamb_plus_1 - log_kv_lamb
        mean = jnp.exp(log_mean)

        # calculating variance
        # kv_lamb_plus_2: float = kv(lamb + 2, r)
        # second_moment: float = lax.mul(frac, lax.div(kv_lamb_plus_2, kv_lamb))
        # variance: float = lax.sub(second_moment, lax.pow(mean, 2))
        log_kv_lamb_plus_2: float = log_kv(lamb + 2, r)
        log_second_moment: float = log_frac + log_kv_lamb_plus_2 - log_kv_lamb
        second_moment: float = jnp.exp(log_second_moment)
        variance: float = lax.sub(second_moment, lax.pow(mean, 2))
        std: float = jnp.sqrt(variance)

        # mode
        mode: float = lax.div(
            (lamb - 1) + lax.sqrt(lax.pow(lamb - 1, 2) + lax.mul(chi, psi)), psi
        )

        return self._scalar_transform(
            {"mean": mean, "variance": variance, "std": std, "mode": mode}
        )

    # fitting
    @staticmethod
    def _sample_moments(x: jnp.ndarray) -> tuple:
        """Compute method-of-moments initial estimates for (lamb, chi, psi).

        Uses the large-r asymptotic approximation where K_{λ+1}(r)/K_λ(r) ≈ 1:
            E[X] ≈ sqrt(chi/psi)
            Var(X) ≈ sqrt(chi/psi) / sqrt(chi*psi) = E[X] / r

        Solving for chi and psi:
            r ≈ mean² / var   (from Var ≈ mean / r)
            chi ≈ mean * r    (from chi = sqrt(chi/psi * chi*psi) = mean * r)
            psi ≈ r / mean    (from psi = sqrt(chi*psi / (chi/psi)) = r / mean)
        """
        m = jnp.mean(x)
        v = jnp.var(x)
        # r = sqrt(chi*psi) ≈ mean^2 / var
        r0 = jnp.clip(m ** 2 / (v + 1e-10), 0.5, 50.0)
        chi0 = jnp.clip(m * r0, 1e-4, 100.0)
        psi0 = jnp.clip(r0 / (m + 1e-10), 1e-4, 100.0)
        lamb0 = 1.0
        return lamb0, chi0, psi0

    def _fit_mle(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        """Fit via projected gradient MLE with box constraints on chi and psi."""
        eps = 1e-8
        constraints: tuple = (
            jnp.array([[-jnp.inf, eps, eps]]).T,
            jnp.array([[jnp.inf, jnp.inf, jnp.inf]]).T,
        )

        projection_options: dict = {"lower": constraints[0], "upper": constraints[1]}

        lamb0, chi0, psi0 = self._sample_moments(x)
        params0: jnp.ndarray = jnp.array([lamb0, chi0, psi0])

        res = projected_gradient(
            f=self._mle_objective,
            x0=params0,
            projection_method="projection_box",
            projection_options=projection_options,
            x=x,
            lr=lr,
            maxiter=maxiter,
        )
        lamb, chi, psi = res["x"]
        return self._params_dict(lamb=lamb, chi=chi, psi=psi)  # , res['fun']

    def fit(
        self, x: ArrayLike, lr: float = 0.1, maxiter: int = 100, name: str = None
    ):
        r"""Fit the distribution to the input data.

        Args:
            x (ArrayLike): The input data to fit the distribution to.
            lr (float): Learning rate for optimization.
            maxiter (int): Maximum number of iterations for optimization.
            name (str): Optional custom name for the fitted instance.

        Returns:
            dict: The fitted distribution parameters.
        """
        x: jnp.ndarray = _univariate_input(x)[0]
        return self._fitted_instance(
            self._fit_mle(x=x, lr=lr, maxiter=maxiter), name=name
        )

    # cdf
    @staticmethod
    def _params_from_array(params_arr, *args, **kwargs) -> dict:
        """Reconstruct a parameter dictionary from a flat array."""
        lamb, chi, psi = params_arr
        return GIG._params_dict(lamb=lamb, chi=chi, psi=psi)

    @staticmethod
    def _pdf_for_cdf(x: ArrayLike, *params_tuple) -> Array:
        """Evaluate the PDF for numerical CDF integration."""
        params_array: jnp.ndarray = jnp.asarray(params_tuple).flatten()
        params: dict = GIG._params_from_array(params_array)
        return lax.exp(GIG._stable_logpdf(stability=0.0, x=x, params=params))

    def cdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the CDF via numerical integration with a custom VJP."""
        params = self._resolve_params(params)
        cdf = _vjp_cdf(x=x, params=params)
        return self._enforce_support_on_cdf(x=x, cdf=cdf, params=params)


gig = GIG("GIG")


def _vjp_cdf(x: ArrayLike, params: dict) -> Array:
    params: dict = GIG._args_transform(params)
    return _cdf(dist=gig, x=x, params=params)


_vjp_cdf_copy = deepcopy(_vjp_cdf)
_vjp_cdf = custom_vjp(_vjp_cdf)


def cdf_fwd(x: ArrayLike, params: dict) -> tuple[Array, tuple]:
    params = GIG._args_transform(params)
    return _cdf_fwd(dist=gig, cdf_func=_vjp_cdf_copy, x=x, params=params)


_vjp_cdf.defvjp(cdf_fwd, cdf_bwd)
