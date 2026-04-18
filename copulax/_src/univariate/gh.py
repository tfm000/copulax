"""File containing the copulAX implementation of the generalized hyperbolic distribution."""

import jax.numpy as jnp
from jax import lax, custom_vjp, random, jit, value_and_grad
from jax import Array
from jax.typing import ArrayLike
from copy import deepcopy

from copulax._src._distributions import Univariate
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import _resolve_key, get_local_random_key
from copulax._src.typing import Scalar
from copulax._src.univariate._cdf import _cdf, cdf_bwd, _cdf_fwd
from copulax.special import log_kv
from copulax._src.univariate._rvs import mean_variance_sampling
from copulax._src.univariate._mean_variance import (
    mean_variance_stats,
    mean_variance_ldmle_params,
)
from copulax._src.univariate.gig import gig
from copulax._src.optimize import projected_gradient


class GH(Univariate):
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

    lamb: Array = None
    chi: Array = None
    psi: Array = None
    mu: Array = None
    sigma: Array = None
    gamma: Array = None

    def __init__(
        self,
        name="GH",
        *,
        lamb=None,
        chi=None,
        psi=None,
        mu=None,
        sigma=None,
        gamma=None,
    ):
        """Initialize the Generalized Hyperbolic distribution.

        Args:
            name: Display name for the distribution.
            lamb: Shape parameter (real-valued).
            chi: Concentration parameter (strictly positive).
            psi: Rate parameter (strictly positive).
            mu: Location parameter.
            sigma: Scale / dispersion parameter.
            gamma: Skewness parameter.
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
        self.mu = jnp.asarray(mu, dtype=float).reshape(()) if mu is not None else None
        self.sigma = (
            jnp.asarray(sigma, dtype=float).reshape(()) if sigma is not None else None
        )
        self.gamma = (
            jnp.asarray(gamma, dtype=float).reshape(()) if gamma is not None else None
        )

    @property
    def _stored_params(self):
        """Return stored parameters if all are set, else None."""
        if any(
            v is None
            for v in [self.lamb, self.chi, self.psi, self.mu, self.sigma, self.gamma]
        ):
            return None
        return {
            "lamb": self.lamb,
            "chi": self.chi,
            "psi": self.psi,
            "mu": self.mu,
            "sigma": self.sigma,
            "gamma": self.gamma,
        }

    @classmethod
    def _params_dict(
        cls,
        lamb: Scalar,
        chi: Scalar,
        psi: Scalar,
        mu: Scalar,
        sigma: Scalar,
        gamma: Scalar,
    ) -> dict:
        r"""Convert parameters to a dictionary."""
        d: dict = {
            "lamb": lamb,
            "chi": chi,
            "psi": psi,
            "mu": mu,
            "sigma": sigma,
            "gamma": gamma,
        }
        return cls._args_transform(d)

    @staticmethod
    def _params_to_tuple(params: dict) -> tuple:
        """Extract (lamb, chi, psi, mu, sigma, gamma) from the parameter dictionary."""
        params = GH._args_transform(params)
        return (
            params["lamb"],
            params["chi"],
            params["psi"],
            params["mu"],
            params["sigma"],
            params["gamma"],
        )

    @staticmethod
    def _params_to_array(params: dict) -> Array:
        """Convert the parameter dictionary to a flat array."""
        return jnp.asarray(GH._params_to_tuple(params)).flatten()

    @classmethod
    def _support(cls, *args, **kwargs) -> Array:
        """Return the support ``[-inf, inf]``."""
        return jnp.array([-jnp.inf, jnp.inf])

    def example_params(self, *args, **kwargs) -> dict:
        r"""Example parameters for the generalized hyperbolic
        distribution.

        This is a six parameter family, with the generalized hyperbolic
        being defined by location `mu`, dispersion `sigma` and skewness
        `gamma` in addition to `lamb`, `chi` and `psi` shape parameters.
        Here, we adopt the parameterization used by McNeil et al. (2005).
        """
        return self._params_dict(
            lamb=0.0, chi=1.0, psi=1.0, mu=0.0, sigma=1.0, gamma=0.0
        )

    @staticmethod
    def _stable_logpdf(stability: Scalar, x: ArrayLike, params: dict) -> Array:
        """Compute the numerically stabilized log-PDF of the GH distribution."""
        lamb, chi, psi, mu, sigma, gamma = GH._params_to_tuple(params)
        x, xshape = _univariate_input(x)

        r: float = lax.sqrt(lax.mul(chi, psi))
        s: float = 0.5 - lamb
        h: float = lax.add(psi, lax.pow(lax.div(gamma, sigma), 2))
        g = lax.div(lax.sub(x, mu), lax.pow(sigma, 2))

        m = lax.sqrt(lax.mul(lax.add(chi, lax.mul(g, lax.sub(x, mu))), h))

        T = lax.add(log_kv(-s, m), lax.mul(g, gamma))
        B = lax.mul(lax.log(m + stability), s)

        cT = lax.add(
            lax.mul(lamb, lax.log((psi / (r + stability)) + stability)),
            lax.mul(lax.log(h), s),
        )
        cB = lax.add(
            lax.add(lax.log(sigma), lax.log(lax.sqrt(2 * jnp.pi))),
            log_kv(lamb, r),
        )

        c = lax.sub(cT, cB)
        logpdf: jnp.ndarray = lax.add(lax.sub(T, B), c)
        return logpdf.reshape(xshape)

    def logpdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the log probability density function."""
        params = self._resolve_params(params)
        logpdf = GH._stable_logpdf(stability=0.0, x=x, params=params)
        return self._enforce_support_on_logpdf(x=x, logpdf=logpdf, params=params)

    def pdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the probability density function."""
        params = self._resolve_params(params)
        return lax.exp(GH._stable_logpdf(stability=0.0, x=x, params=params))

    # sampling
    def rvs(
        self, size: tuple | Scalar, params: dict = None, key: Array = None
    ) -> Array:
        """Generate random variates via GIG-normal mean-variance mixture."""
        params = self._resolve_params(params)
        key = _resolve_key(key)
        lamb, chi, psi, mu, sigma, gamma = self._params_to_tuple(params)

        key1, key2 = random.split(key)
        W = gig.rvs(
            key=key1, size=size, params={"lamb": lamb, "chi": chi, "psi": psi}
        )
        return mean_variance_sampling(
            key=key2, W=W, shape=size, mu=mu, sigma=sigma, gamma=gamma
        )

    # stats
    def _get_w_stats(self, lamb: Scalar, chi: Scalar, psi: Scalar) -> dict:
        """Compute statistics of the GIG mixing variable W."""
        return gig.stats(params={"lamb": lamb, "chi": chi, "psi": psi})

    def stats(self, params: dict = None) -> dict:
        """Compute distribution statistics derived from the GIG-normal mixture representation."""
        params = self._resolve_params(params)
        lamb, chi, psi, mu, sigma, gamma = self._params_to_tuple(params)
        gig_stats: dict = self._get_w_stats(lamb=lamb, chi=chi, psi=psi)
        return self._scalar_transform(
            mean_variance_stats(w_stats=gig_stats, mu=mu, sigma=sigma, gamma=gamma)
        )

    # fitting
    @staticmethod
    def _gig_expected_w(lamb: Scalar, chi: Scalar, psi: Scalar) -> Array:
        """E[W] for W ~ GIG(lamb, chi, psi) using log_kv ratios."""
        r = lax.sqrt(jnp.maximum(lax.mul(chi, psi), 1e-8))
        log_ew = 0.5 * lax.log(lax.div(chi, psi)) + log_kv(lamb + 1, r) - log_kv(lamb, r)
        return jnp.exp(log_ew)

    @staticmethod
    def _gig_expected_inv_w(lamb: Scalar, chi: Scalar, psi: Scalar) -> Array:
        """E[1/W] for W ~ GIG(lamb, chi, psi) using log_kv ratios."""
        r = lax.sqrt(jnp.maximum(lax.mul(chi, psi), 1e-8))
        log_einv = 0.5 * lax.log(lax.div(psi, chi)) + log_kv(lamb - 1, r) - log_kv(lamb, r)
        return jnp.exp(log_einv)

    @staticmethod
    @jit
    def _nll_value_and_grad(all_params: Array, x: Array) -> tuple:
        """Compute negative log-likelihood and its gradient w.r.t. all 6 parameters."""
        def _nll(params_arr, x):
            params = GH._params_from_array(params_arr)
            return -jnp.mean(GH._stable_logpdf(1e-30, x, params))
        return value_and_grad(_nll)(all_params, x)

    def _fit_mle(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        """Fit all six parameters via projected gradient MLE with box constraints."""
        eps: float = 1e-8
        constraints: tuple = (
            jnp.array([[-jnp.inf, eps, eps, -jnp.inf, eps, -jnp.inf]]).T,
            jnp.array([[jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf]]).T,
        )

        projection_options: dict = {"lower": constraints[0], "upper": constraints[1]}

        key1, key = random.split(get_local_random_key())
        key2, key3 = random.split(key)
        params0: jnp.ndarray = jnp.array(
            [
                random.normal(key1, ()),
                random.uniform(key2, (), minval=eps),
                random.uniform(key3, (), minval=eps),
                x.mean(),
                x.std(),
                0.0,
            ]
        )

        res: dict = projected_gradient(
            f=self._mle_objective,
            x0=params0,
            projection_method="projection_box",
            projection_options=projection_options,
            x=x,
            lr=lr,
            maxiter=maxiter,
        )
        lamb, chi, psi, mu, sigma, gamma = res["x"]
        return GH._params_dict(
            lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma
        )

    @staticmethod
    def _em_body(carry: tuple, _: None, x: Array, lr: float, shape_steps: int) -> tuple:
        """Single ECME iteration as a pure function for use with lax.scan.

        Args:
            carry: Tuple of (lamb, chi, psi, mu, sigma, gamma).
            _: Unused scan input.
            x: Data array (static).
            lr: Shape learning rate (static).
            shape_steps: Number of inner gradient steps (static).

        Returns:
            Updated carry and None (no stacked output).
        """
        eps: float = 1e-8
        lamb, chi, psi, mu, sigma, gamma = carry

        # --- E-step ---
        Q = lax.pow(lax.div(lax.sub(x, mu), sigma), 2)
        psi_bar = psi + lax.pow(lax.div(gamma, sigma), 2)
        lam_post = lamb - 0.5
        chi_post = chi + Q

        delta = jnp.clip(GH._gig_expected_w(lam_post, chi_post, psi_bar), eps, 1e10)
        eta = jnp.clip(GH._gig_expected_inv_w(lam_post, chi_post, psi_bar), eps, 1e10)

        # --- CM-step 1: closed-form update for mu, gamma, sigma^2 ---
        delta_bar = jnp.mean(delta)
        eta_bar = jnp.mean(eta)
        x_bar = jnp.mean(x)
        x_eta_bar = jnp.mean(x * eta)

        denom = eta_bar - 1.0 / delta_bar
        denom = jnp.where(jnp.abs(denom) < eps, eps, denom)
        mu = (x_eta_bar - x_bar / delta_bar) / denom
        gamma = (x_bar - mu) / delta_bar
        sigma_sq = jnp.mean(
            (x - mu) ** 2 * eta - 2 * (x - mu) * gamma + delta * gamma ** 2
        )
        sigma = jnp.sqrt(jnp.maximum(sigma_sq, eps))

        # --- CM-step 2: gradient descent for lamb, chi, psi ---
        def _shape_step(shape_carry, _):
            l, c, p = shape_carry
            all_p = jnp.array([l, c, p, mu, sigma, gamma])
            _, g = GH._nll_value_and_grad(all_p, x)
            g_shape = jnp.nan_to_num(g[:3], nan=0.0)
            l = l - lr * g_shape[0]
            c = jnp.maximum(c - lr * g_shape[1], eps)
            p = jnp.maximum(p - lr * g_shape[2], eps)
            return (l, c, p), None

        (lamb, chi, psi), _ = lax.scan(
            _shape_step, (lamb, chi, psi), None, length=shape_steps
        )

        return (lamb, chi, psi, mu, sigma, gamma), None

    def _fit_em(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        """Fit via ECME algorithm (McNeil et al. 2005, Section 3.4.2).

        The EM algorithm treats the GIG mixing variable W as latent data.
        It avoids the mu/gamma/sigma identifiability ridge by updating these
        parameters in closed form from the expected sufficient statistics,
        while the shape parameters (lamb, chi, psi) are updated via
        gradient descent on the observed log-likelihood.

        The entire loop is compiled via ``lax.scan`` for performance.

        Args:
            x: Input data array.
            lr: Learning rate for shape parameter gradient steps.
            maxiter: Number of EM iterations.

        Returns:
            Fitted parameter dictionary.
        """
        # Initialize from sample moments
        sample_mean: Scalar = x.mean()
        sample_std: Scalar = x.std()
        z: jnp.ndarray = (x - sample_mean) / sample_std
        sample_skew: Scalar = jnp.mean(z ** 3)

        init_carry: tuple = (
            jnp.array(0.0),                        # lamb
            jnp.array(1.0),                        # chi
            jnp.array(1.0),                        # psi
            sample_mean,                            # mu
            sample_std,                             # sigma
            sample_skew * sample_std * 0.25,        # gamma
        )

        shape_steps: int = 10
        em_step = lambda carry, _: self._em_body(carry, _, x, lr, shape_steps)
        final_carry, _ = lax.scan(em_step, init_carry, None, length=maxiter)
        lamb, chi, psi, mu, sigma, gamma = final_carry

        return GH._params_dict(
            lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma
        )

    def _ldmle_objective(
        self,
        params: jnp.ndarray,
        x: jnp.ndarray,
        sample_mean: Scalar,
        sample_variance: Scalar,
    ) -> Scalar:
        """LDMLE objective that optimizes (lamb, chi, psi, gamma) and derives mu, sigma."""
        lamb, chi, psi, gamma = params
        gig_stats: dict = self._get_w_stats(lamb=lamb, chi=chi, psi=psi)
        mu, sigma = mean_variance_ldmle_params(
            stats=gig_stats,
            gamma=gamma,
            sample_mean=sample_mean,
            sample_variance=sample_variance,
        )
        return self._mle_objective(
            params_arr=jnp.array([lamb, chi, psi, mu, sigma, gamma]), x=x
        )

    def _fit_ldmle(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        """Fit via low-dimensional MLE, optimizing (lamb, chi, psi, gamma) with mu and sigma derived."""
        eps = 1e-8
        constraints: tuple = (
            jnp.array([[-jnp.inf, eps, eps, -jnp.inf]]).T,
            jnp.array([[jnp.inf, jnp.inf, jnp.inf, jnp.inf]]).T,
        )

        key1, key = random.split(get_local_random_key())
        key2, key3 = random.split(key)
        params0: jnp.ndarray = jnp.array(
            [
                random.normal(key1, ()),
                random.uniform(key2, (), minval=eps),
                random.uniform(key3, (), minval=eps),
                0.0,
            ]
        )

        projection_options: dict = {"lower": constraints[0], "upper": constraints[1]}

        sample_mean, sample_variance = x.mean(), x.var()
        res = projected_gradient(
            f=self._ldmle_objective,
            x0=params0,
            x=x,
            lr=lr,
            maxiter=maxiter,
            projection_method="projection_box",
            projection_options=projection_options,
            sample_mean=sample_mean,
            sample_variance=sample_variance,
        )
        lamb, chi, psi, gamma = res["x"]
        gig_stats: dict = self._get_w_stats(lamb=lamb, chi=chi, psi=psi)
        mu, sigma = mean_variance_ldmle_params(
            stats=gig_stats,
            gamma=gamma,
            sample_mean=sample_mean,
            sample_variance=sample_variance,
        )
        return self._params_dict(
            lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma
        )

    def fit(
        self,
        x: ArrayLike,
        method: str = "EM",
        lr: float = 0.1,
        maxiter: int = 100,
        name: str = None,
    ):
        r"""Fit the distribution to the input data.

        Note:
            If you intend to jit wrap this function, ensure that 'method' is a
            static argument.

        Args:
            x (ArrayLike): The input data to fit the distribution to.
            method (str): The fitting method to use.  Options are
                'EM' for the ECME algorithm (McNeil et al. 2005),
                'MLE' for direct projected gradient maximum likelihood,
                and 'LDMLE' for low-dimensional maximum likelihood
                estimation. Defaults to 'EM'.
            lr (float): Learning rate for optimization.
            maxiter (int): Maximum number of iterations for optimization.
            name (str): Optional custom name for the fitted instance.

        Returns:
            dict: The fitted distribution parameters.
        """
        x = _univariate_input(x)[0]
        if method == "MLE":
            return self._fitted_instance(
                self._fit_mle(x, lr=lr, maxiter=maxiter), name=name
            )
        elif method == "EM":
            return self._fitted_instance(
                self._fit_em(x, lr=lr, maxiter=maxiter), name=name
            )
        else:
            return self._fitted_instance(
                self._fit_ldmle(x, lr=lr, maxiter=maxiter), name=name
            )

    # cdf
    @staticmethod
    def _params_from_array(params_arr: jnp.ndarray, *args, **kwargs) -> dict:
        """Reconstruct a parameter dictionary from a flat array."""
        lamb, chi, psi, mu, sigma, gamma = params_arr
        return GH._params_dict(
            lamb=lamb, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma
        )

    @staticmethod
    def _pdf_for_cdf(x: ArrayLike, *params_tuple) -> Array:
        """Evaluate the PDF for numerical CDF integration."""
        params_array: jnp.ndarray = jnp.asarray(params_tuple).flatten()
        params: dict = GH._params_from_array(params_array)
        return lax.exp(GH._stable_logpdf(stability=0.0, x=x, params=params))

    def cdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the CDF via numerical integration with a custom VJP."""
        params = self._resolve_params(params)
        cdf = _vjp_cdf(x=x, params=params)
        return self._enforce_support_on_cdf(x=x, cdf=cdf, params=params)


gh = GH("GH")


def _vjp_cdf(x: ArrayLike, params: dict) -> Array:
    params = GH._args_transform(params)
    return _cdf(dist=gh, x=x, params=params)


_vjp_cdf_copy = deepcopy(_vjp_cdf)
_vjp_cdf = custom_vjp(_vjp_cdf)


def cdf_fwd(x: ArrayLike, params: dict) -> tuple[Array, tuple]:
    params = GH._args_transform(params)
    return _cdf_fwd(dist=gh, cdf_func=_vjp_cdf_copy, x=x, params=params)


_vjp_cdf.defvjp(cdf_fwd, cdf_bwd)
