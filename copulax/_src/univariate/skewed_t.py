"""File containing the copulAX implementation of the skewed-T distribution."""

import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, custom_vjp, random, jit, value_and_grad
from jax import Array
from jax.typing import ArrayLike
from copy import deepcopy

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src.special import log_kv_plus_s_log_r
from copulax._src._utils import _resolve_key, get_local_random_key
from copulax._src.univariate._cdf import _cdf, cdf_bwd, _cdf_fwd
from copulax._src.optimize import projected_gradient
from copulax._src.univariate.ig import ig
from copulax._src.univariate._normal_mixture import (
    forward_reparam_1d,
    invert_gamma_to_z_1d,
    mean_variance_stats,
)
from copulax._src.univariate._rvs import mean_variance_sampling
from copulax._src.univariate.gh import GH

_NU_EPS = 1e-8
_NU_INIT = 4.0
_NU_LDMLE_MIN = 4.0 + 1e-3


class SkewedT(Univariate):
    r"""The skewed-t distribution is a generalisation of the continuous Student's
    t-distribution that allows for skewness. It can also be expressed as a limiting
    case of the Generalized Hyperbolic distribution when phi -> 0 in addition to
    lamb = -0.5*chi.

    We use the 4 parameter McNeil et al (2005) specification of the distribution.
    """

    nu: Array = None
    mu: Array = None
    sigma: Array = None
    gamma: Array = None

    def __init__(self, name="Skewed-T", *, nu=None, mu=None, sigma=None, gamma=None):
        """Initialize the Skewed-T distribution.

        Args:
            name: Display name for the distribution.
            nu: Degrees of freedom parameter.
            mu: Location parameter.
            sigma: Scale parameter.
            gamma: Skewness parameter (zero recovers the Student-T).
        """
        super().__init__(name)
        self.nu = jnp.asarray(nu, dtype=float).reshape(()) if nu is not None else None
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
        if any(v is None for v in [self.nu, self.mu, self.sigma, self.gamma]):
            return None
        return {"nu": self.nu, "mu": self.mu, "sigma": self.sigma, "gamma": self.gamma}

    @classmethod
    def _params_dict(cls, nu: Scalar, mu: Scalar, sigma: Scalar, gamma: Scalar) -> dict:
        """Create a parameter dictionary from nu, mu, sigma, and gamma values."""
        d: dict = {"nu": nu, "mu": mu, "sigma": sigma, "gamma": gamma}
        return cls._args_transform(d)

    @staticmethod
    def _params_to_tuple(params: dict) -> tuple:
        """Extract (nu, mu, sigma, gamma) from the parameter dictionary."""
        params = SkewedT._args_transform(params)
        return params["nu"], params["mu"], params["sigma"], params["gamma"]

    @staticmethod
    def _params_to_array(params: dict) -> Array:
        """Convert the parameter dictionary to a flat array."""
        return jnp.asarray(SkewedT._params_to_tuple(params)).flatten()

    @classmethod
    def _support(cls, *args, **kwargs) -> Array:
        """Return the support ``[-inf, inf]``."""
        return jnp.array([-jnp.inf, jnp.inf])

    def example_params(self, *args, **kwargs) -> dict:
        r"""Example parameters for the skewed-T distribution.

        This is a four parameter family, with the skewed-T being defined
        by its degrees of freedom `nu`, location `mu`, scale `sigma` and
        skewness `gamma`. It is a generalisation of the student-T
        distribution, which it includes as a special case when gamma is
        zero. Here, we adopt the parameterization used by McNeil et al.
        (2005).
        """
        return self._params_dict(nu=4.5, mu=0.0, sigma=1.0, gamma=1.0)

    @staticmethod
    def _stable_logpdf(stability: float, x: ArrayLike, params: dict) -> Array:
        r"""Skewed-t log-PDF (McNeil, Frey & Embrechts 2005, §3.2).

        .. math::

            \log f(x)
              = c
                + \bigl[\log K_s(r) + s \log r\bigr]
                + P \gamma
                - s \log\!\bigl(1 + Q/\nu\bigr),

        where ``s = (ν+1)/2``, ``P = (x-μ)/σ²``, ``Q = P·(x-μ)``,
        ``R = (γ/σ)²`` and ``r = sqrt((ν+Q)·R)``.

        The bracketed combination ``log K_s(r) + s log r`` has
        divergent individual terms as ``γ → 0`` (``log K_s(0) = +∞``,
        ``s log 0 = −∞``) but a finite analytical limit
        ``log Γ(s) + (s−1) log 2`` — computed as a single
        cancellation-stable object by :py:func:`log_kv_plus_s_log_r`.
        At ``γ = 0`` exactly the formula evaluates to the Student-t
        log-PDF to float64 eps.
        """
        nu, mu, sigma, gamma = SkewedT._params_to_tuple(params)
        x, xshape = _univariate_input(x)

        s: float = 0.5 * (nu + 1)
        c: float = (
            jnp.log(2.0) * (1 - s)
            - lax.lgamma(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * nu + stability)
            - jnp.log(sigma + stability)
        )

        P: jnp.ndarray = (x - mu) * lax.pow(sigma, -2)
        Q: jnp.ndarray = P * (x - mu)
        R: jnp.ndarray = lax.pow(gamma / sigma, 2)

        # jnp.maximum on the sqrt argument keeps ∂r/∂γ finite at γ=0
        # (otherwise ∂√z/∂z = 1/(2√z) → ∞ at z=0 multiplies against
        # the upstream ∂z/∂γ = 0 to give NaN).  The 1e-24 floor is
        # the square of log_kv_plus_s_log_r's internal r-floor, so
        # the helper's direct-sum path is reached for any γ > 0.
        r = lax.sqrt(jnp.maximum((nu + Q) * R, 1e-24))
        log_kv_plus = log_kv_plus_s_log_r(s, r)

        logpdf: jnp.ndarray = (
            c
            + log_kv_plus
            + P * gamma
            - s * jnp.log(1 + Q / (nu + stability))
        )
        return logpdf.reshape(xshape)

    # sampling
    def rvs(
        self, size: tuple | Scalar, params: dict = None, key: Array = None
    ) -> Array:
        """Generate random variates via mean-variance mixture of normals."""
        params = self._resolve_params(params)
        key = _resolve_key(key)
        nu, mu, sigma, gamma = self._params_to_tuple(params)

        key1, key2 = random.split(key)
        W: jnp.ndarray = ig.rvs(
            size=size, key=key1, params={"alpha": nu * 0.5, "beta": nu * 0.5}
        )
        return mean_variance_sampling(
            key=key2, W=W, shape=size, mu=mu, sigma=sigma, gamma=gamma
        )

    # stats
    def _get_w_stats(self, nu: float) -> dict:
        """Compute mean and variance of the inverse-gamma mixing variable W.

        Divergent moments propagate as ``+inf``: mean diverges for ``nu <= 2``,
        variance diverges for ``nu <= 4``.
        """
        ig_params: dict = {"alpha": nu * 0.5, "beta": nu * 0.5}
        ig_stats: dict = ig.stats(params=ig_params)
        return {"mean": ig_stats["mean"], "variance": ig_stats["variance"]}

    def stats(self, params: dict = None) -> dict:
        """Compute distribution statistics derived from the mean-variance mixture representation."""
        params = self._resolve_params(params)
        nu, mu, sigma, gamma = self._params_to_tuple(params)
        w_stats: dict = self._get_w_stats(nu)
        return self._scalar_transform(
            mean_variance_stats(mu=mu, sigma=sigma, gamma=gamma, w_stats=w_stats)
        )

    # fitting
    @staticmethod
    def _sample_moments(x: jnp.ndarray) -> tuple:
        """Compute sample mean, std, skewness, and excess kurtosis."""
        sample_mean = x.mean()
        sample_std = x.std()
        z = (x - sample_mean) / sample_std
        sample_skew = jnp.mean(z ** 3)
        sample_kurt = jnp.mean(z ** 4) - 3.0
        return sample_mean, sample_std, sample_skew, sample_kurt

    @staticmethod
    @jit
    def _nll_value_and_grad(all_params: Array, x: Array) -> tuple:
        """Compute negative log-likelihood and its gradient w.r.t. all 4 parameters."""
        def _nll(params_arr, x):
            params = SkewedT._params_from_array(params_arr)
            return -jnp.mean(SkewedT._stable_logpdf(1e-30, x, params))
        return value_and_grad(_nll)(all_params, x)

    @staticmethod
    def _em_body(carry: tuple, _: None, x: Array, lr: float, shape_steps: int) -> tuple:
        """Single ECME iteration for skewed-t, compatible with lax.scan.

        E-step:  posterior W_i|x_i ~ GIG(-(nu+1)/2, nu+Q_i, gamma^2/sigma^2)
        CM-step 1: closed-form update for mu, gamma, sigma
        CM-step 2: gradient descent on nu

        Args:
            carry: Tuple of (nu, mu, sigma, gamma).
            _: Unused scan input.
            x: Data array (static).
            lr: Shape learning rate (static).
            shape_steps: Number of inner gradient steps for nu (static).

        Returns:
            Updated carry and None.
        """
        eps: float = 1e-8
        nu, mu, sigma, gamma = carry

        # --- E-step ---
        Q = lax.pow(lax.div(lax.sub(x, mu), sigma), 2)
        psi_bar = lax.pow(lax.div(gamma, sigma), 2)
        lam_post = -(nu + 1.0) / 2.0
        chi_post = nu + Q

        delta = jnp.clip(GH._gig_expected_w(lam_post, chi_post, psi_bar), eps, 1e10)
        eta = jnp.clip(GH._gig_expected_inv_w(lam_post, chi_post, psi_bar), eps, 1e10)

        # --- CM-step 1: closed-form update for mu, gamma, sigma ---
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

        # --- CM-step 2: gradient descent for nu ---
        def _shape_step(shape_carry, _):
            n = shape_carry[0]
            all_p = jnp.array([n, mu, sigma, gamma])
            _, g = SkewedT._nll_value_and_grad(all_p, x)
            g_nu = jnp.nan_to_num(g[0], nan=0.0)
            n = jnp.maximum(n - lr * g_nu, eps)
            return (n,), None

        (nu,), _ = lax.scan(_shape_step, (nu,), None, length=shape_steps)

        return (nu, mu, sigma, gamma), None

    def _fit_em(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        """Fit via ECME algorithm (McNeil et al. 2005, Section 3.4.2).

        The EM algorithm treats the IG mixing variable W as latent data.
        It avoids the mu/gamma/sigma identifiability ridge by updating these
        parameters in closed form, while nu is updated via gradient descent.
        The entire loop is compiled via ``lax.scan`` for performance.

        Args:
            x: Input data array.
            lr: Learning rate for nu gradient steps.
            maxiter: Number of EM iterations.

        Returns:
            Fitted parameter dictionary.
        """
        sample_mean, sample_std, sample_skew, sample_kurt = self._sample_moments(x)

        nu0 = jnp.clip(4.0 + 6.0 / jnp.maximum(sample_kurt, 0.1), _NU_INIT, 60.0)

        init_carry: tuple = (
            nu0,
            sample_mean,
            sample_std,
            sample_skew * sample_std * 0.25,
        )

        shape_steps: int = 10
        em_step = lambda carry, _: self._em_body(carry, _, x, lr, shape_steps)
        final_carry, _ = lax.scan(em_step, init_carry, None, length=maxiter)
        nu, mu, sigma, gamma = final_carry

        return self._params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)

    def _fit_mle(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        """Fit all four parameters via projected gradient MLE with box constraints."""
        eps: float = 1e-8
        sample_mean, sample_std, sample_skew, sample_kurt = self._sample_moments(x)

        # Data-driven box constraints to prevent divergence
        sigma_lo = 0.1 * sample_std
        gamma_bound = 2.0 * sample_std
        mu_bound = 2.0 * sample_std

        constraints: tuple = (
            jnp.array([[eps, sample_mean - mu_bound, sigma_lo + eps, -gamma_bound]]).T,
            jnp.array([[jnp.inf, sample_mean + mu_bound, jnp.inf, gamma_bound]]).T,
        )
        projection_options: dict = {"lower": constraints[0], "upper": constraints[1]}

        # Method-of-moments initial estimates
        nu0 = jnp.clip(4.0 + 6.0 / jnp.maximum(sample_kurt, 0.1), 4.0 + eps, 60.0)
        gamma0 = jnp.clip(sample_skew * sample_std * 0.5, -gamma_bound, gamma_bound)
        ew = nu0 / (nu0 - 2.0)
        mu0 = jnp.clip(sample_mean - ew * gamma0, sample_mean - mu_bound, sample_mean + mu_bound)
        sigma0 = jnp.maximum(
            jnp.sqrt(jnp.maximum(sample_std**2 - ew * gamma0**2, eps) / ew),
            sigma_lo + eps,
        )

        params0: jnp.ndarray = jnp.array([nu0, mu0, sigma0, gamma0])

        res: dict = projected_gradient(
            f=self._mle_objective,
            x0=params0,
            projection_method="projection_box",
            projection_options=projection_options,
            x=x,
            lr=lr,
            maxiter=maxiter,
        )
        nu, mu, sigma, gamma = res["x"]
        return self._params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)

    def _ldmle_objective(
        self,
        params: jnp.ndarray,
        x: jnp.ndarray,
        sample_mean: Scalar,
        sample_variance: Scalar,
    ) -> jnp.ndarray:
        """LDMLE objective over (raw_nu, z). gamma follows from z via the
        feasibility reparam; mu and sigma follow from moment-matching. sigma is
        strictly positive by construction.
        """
        raw_nu, z = params
        nu = jnn.softplus(raw_nu) + _NU_LDMLE_MIN
        sigma_hat = jnp.sqrt(sample_variance)
        ig_stats: dict = self._get_w_stats(nu=nu)
        gamma, sigma = forward_reparam_1d(
            z, sigma_hat, ig_stats["mean"], ig_stats["variance"],
        )
        mu = sample_mean - ig_stats["mean"] * gamma
        return self._mle_objective(params_arr=jnp.array([nu, mu, sigma, gamma]), x=x)

    def _fit_ldmle(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        """Fit via LDMLE. Optimises (raw_nu, z): gamma is reparametrised so
        feasibility of the moment-matching reconstruction is structural.
        """
        _, sample_std, sample_skew, sample_kurt = self._sample_moments(x)

        # z is unconstrained in both directions.
        constraints: tuple = (
            jnp.array([[-jnp.inf, -jnp.inf]]).T,
            jnp.array([[jnp.inf, jnp.inf]]).T,
        )

        # Method-of-moments initial estimates. nu floored above _NU_LDMLE_MIN so
        # the moment-matching reconstruction stays in the Var[W] < inf regime.
        nu_lower = _NU_LDMLE_MIN + 0.5
        nu0 = jnp.clip(4.0 + 6.0 / jnp.maximum(sample_kurt, 0.1), nu_lower, 60.0)
        raw_nu0 = jnp.log(jnp.expm1(nu0 - _NU_LDMLE_MIN))
        gamma0 = sample_skew * sample_std * 0.5
        w_var0 = self._get_w_stats(nu=nu0)["variance"]
        z0 = invert_gamma_to_z_1d(gamma0, sample_std, w_var0)
        params0: jnp.ndarray = jnp.array([raw_nu0, z0])

        projection_options: dict = {"lower": constraints[0], "upper": constraints[1]}

        sample_mean, sample_variance = x.mean(), x.var()
        res: dict = projected_gradient(
            f=self._ldmle_objective,
            x0=params0,
            projection_method="projection_box",
            projection_options=projection_options,
            x=x,
            sample_mean=sample_mean,
            sample_variance=sample_variance,
            lr=lr,
            maxiter=maxiter,
        )
        raw_nu, z = res["x"]
        nu = jnn.softplus(raw_nu) + _NU_LDMLE_MIN
        sigma_hat = jnp.sqrt(sample_variance)
        ig_stats: dict = self._get_w_stats(nu=nu)
        gamma, sigma = forward_reparam_1d(
            z, sigma_hat, ig_stats["mean"], ig_stats["variance"],
        )
        mu = sample_mean - ig_stats["mean"] * gamma
        return self._params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)

    def fit(
        self,
        x: ArrayLike,
        method: str = "EM",
        lr=0.1,
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
                'MLE' for projected gradient maximum likelihood,
                and 'LDMLE' for low-dimensional maximum likelihood
                estimation. Defaults to 'EM'.
            lr (float): Learning rate for optimization.
            maxiter (int): Maximum number of iterations.
            name (str): Optional custom name for the fitted instance.

        Returns:
            dict: The fitted distribution parameters.
        """
        x = _univariate_input(x)[0]
        if method == "MLE":
            return self._fitted_instance(
                self._fit_mle(x=x, lr=lr, maxiter=maxiter), name=name
            )
        elif method == "EM":
            return self._fitted_instance(
                self._fit_em(x=x, lr=lr, maxiter=maxiter), name=name
            )
        else:
            return self._fitted_instance(
                self._fit_ldmle(x=x, lr=lr, maxiter=maxiter), name=name
            )

    # cdf
    @staticmethod
    def _params_from_array(params_arr: jnp.ndarray, *args, **kwargs) -> dict:
        """Reconstruct a parameter dictionary from a flat array."""
        nu, mu, sigma, gamma = params_arr
        return SkewedT._params_dict(nu=nu, mu=mu, sigma=sigma, gamma=gamma)

    @staticmethod
    def _pdf_for_cdf(x: ArrayLike, *params_tuple) -> Array:
        """Evaluate the PDF for numerical CDF integration."""
        params_array: jnp.ndarray = jnp.asarray(params_tuple).flatten()
        params: dict = SkewedT._params_from_array(params_array)
        return jnp.exp(SkewedT._stable_logpdf(stability=1e-30, x=x, params=params))

    def _cdf_anchor_scales(self, params: dict) -> Array:
        """Use the intrinsic scale parameter sigma, not sqrt(variance).

        The default sqrt(variance) formula for skewed-T requires
        ``nu > 2`` to be finite. For ``1 < nu <= 2`` the mean exists
        but variance is infinite; the base-class default would produce
        ``inf`` or ``nan`` for the scale and break the breakpoint
        grid. The sigma shape parameter is always positive and
        well-defined regardless of ``nu``, giving a clean bulk scale.
        """
        _, _, sigma, _ = SkewedT._params_to_tuple(params)
        return jnp.asarray(sigma).reshape((1,))

    def cdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the CDF via numerical integration with a custom VJP."""
        params = self._resolve_params(params)
        cdf = _vjp_cdf(x=x, params=params)
        return self._enforce_support_on_cdf(x=x, cdf=cdf, params=params)


skewed_t = SkewedT("Skewed-T")


def _vjp_cdf(x: ArrayLike, params: dict) -> Array:
    params = SkewedT._args_transform(params)
    return _cdf(dist=skewed_t, x=x, params=params)


_vjp_cdf_copy = deepcopy(_vjp_cdf)
_vjp_cdf = custom_vjp(_vjp_cdf)


def cdf_fwd(x: ArrayLike, params: dict) -> tuple[Array, tuple]:
    params = SkewedT._args_transform(params)
    return _cdf_fwd(dist=skewed_t, cdf_func=_vjp_cdf_copy, x=x, params=params)


_vjp_cdf.defvjp(cdf_fwd, cdf_bwd)
