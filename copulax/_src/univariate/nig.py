"""File containing the copulAX implementation of the Normal-Inverse Gaussian distribution."""

import jax.numpy as jnp
from jax import lax, custom_vjp, random
from jax import Array
from jax.typing import ArrayLike
from copy import deepcopy

from copulax._src._distributions import Univariate
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import _resolve_key
from copulax._src.typing import Scalar
from copulax._src.univariate._cdf import _cdf, cdf_bwd, _cdf_fwd
from copulax.special import log_kv
from copulax._src.univariate.wald import wald
from copulax._src.optimize import projected_gradient
from copulax._src.stats import skew, kurtosis


class NIG(Univariate):
    r"""The Normal-Inverse Gaussian distribution. This is a flexible,
    continuous 4-parameter distribution that can capture skewness and heavy
    tails. It is a special case of the Generalized Hyperbolic distribution,
    obtained by fixing :math:`\lambda = -\tfrac{1}{2}`.

    We adopt the parameterization used on Wikipedia (and by Karlis 2002):

    .. math::

        f(x|\mu, \alpha, \beta, \delta) = \frac{\alpha \delta
        K_{1}\left(\alpha \sqrt{\delta^2 + (x-\mu)^2}\right)}
        {\pi \sqrt{\delta^2 + (x-\mu)^2}}
        e^{\delta \sqrt{\alpha^2 - \beta^2} + \beta (x-\mu)}

    where :math:`K_{1}` is the modified Bessel function of the second kind
    of order 1, :math:`\mu \in \mathbb{R}` is the location parameter,
    :math:`\delta > 0` is the scale parameter, :math:`\alpha > 0` controls
    the tail heaviness, and :math:`\beta \in (-\alpha, \alpha)` controls
    the asymmetry / skewness.
    """

    mu: Array = None
    alpha: Array = None
    beta: Array = None
    delta: Array = None

    def __init__(
            self,
            name="NIG",
            *,
            mu=None,
            alpha=None,
            beta=None,
            delta=None,
        ):
        """Initialize the NIG distribution.

        Args:
            name: Name of the distribution.
            mu: Location parameter (real-valued).
            alpha: Tail heaviness parameter (positive).
            beta: Asymmetry parameter (between -alpha and alpha).
            delta: Scale parameter (positive).
        """
        super().__init__(name=name)
        self.mu = (
            jnp.asarray(mu, dtype=float).reshape(()) if mu is not None else None
        )
        self.alpha = (
            jnp.asarray(alpha, dtype=float).reshape(()) if alpha is not None else None
        )
        self.beta = (
            jnp.asarray(beta, dtype=float).reshape(()) if beta is not None else None
        )
        self.delta = (
            jnp.asarray(delta, dtype=float).reshape(()) if delta is not None else None
        )

    @property
    def _stored_params(self):
        """Return stored parameters as a dict if all are set, else None."""
        if any(v is None for v in [self.mu, self.alpha, self.beta, self.delta]):
            return None
        return {
            "mu": self.mu,
            "alpha": self.alpha,
            "beta": self.beta,
            "delta": self.delta,
        }

    @classmethod
    def _params_dict(cls, mu: Scalar, alpha: Scalar, beta: Scalar, delta: Scalar) -> dict:
        d: dict = {"mu": mu, "alpha": alpha, "beta": beta, "delta": delta}
        return cls._args_transform(d)

    @staticmethod
    def _params_to_tuple(params: dict) -> tuple:
        params = NIG._args_transform(params)
        return params["mu"], params["alpha"], params["beta"], params["delta"]

    @staticmethod
    def _params_to_array(params: dict) -> Array:
        return jnp.asarray(NIG._params_to_tuple(params), dtype=float).flatten()

    @classmethod
    def _support(cls, *args, **kwargs) -> Array:
        return jnp.array([-jnp.inf, jnp.inf])

    def example_params(self, *args, **kwargs) -> dict:
        r"""Example parameters for the NIG distribution.

        This is a four-parameter family with the NIG being defined by its
        location ``mu``, tail heaviness ``alpha``, asymmetry ``beta``, and
        scale ``delta``.
        """
        return self._params_dict(mu=0.0, alpha=2.5, beta=1.5, delta=1.0)

    @staticmethod
    def _stable_logpdf(stability: Scalar, x: ArrayLike, params: dict) -> Array:
        """Compute the numerically stabilised log-pdf of the NIG distribution."""
        mu, alpha, beta, delta = NIG._params_to_tuple(params)
        x, xshape = _univariate_input(x)

        gamma = jnp.sqrt(jnp.maximum(alpha ** 2 - beta ** 2, stability))
        diff = x - mu
        r = jnp.sqrt(delta ** 2 + diff ** 2)

        log_exponent = delta * gamma + beta * diff
        T = log_kv(1, alpha * r) + jnp.log(alpha + stability) + jnp.log(delta + stability)
        B = jnp.log(jnp.pi) + jnp.log(r + stability)
        logpdf = log_exponent + T - B
        return logpdf.reshape(xshape)

    # sampling
    def rvs(self, size: tuple | Scalar, params: dict = None, key: Array = None) -> Array:
        r"""Generate random variates via an IG-normal variance-mean mixture."""
        params = self._resolve_params(params)
        key = _resolve_key(key)
        mu, alpha, beta, delta = NIG._params_to_tuple(params)
        gamma = jnp.sqrt(alpha ** 2 - beta ** 2)

        key1, key2 = random.split(key)
        W = wald.rvs(size=size, params={"mu": delta / gamma, "lamb": delta ** 2}, key=key1)
        Z = random.normal(key2, shape=W.shape)
        return mu + beta * W + jnp.sqrt(W) * Z

    # stats
    def stats(self, params: dict = None) -> dict:
        params = self._resolve_params(params)
        mu, alpha, beta, delta = NIG._params_to_tuple(params)
        gamma = jnp.sqrt(alpha ** 2 - beta ** 2)

        mean = mu + delta * beta / gamma
        variance = delta * alpha ** 2 / gamma ** 3
        skewness = 3.0 * beta / (alpha * jnp.sqrt(delta * gamma))
        kurt = 3.0 * (1.0 + 4.0 * beta ** 2 / alpha ** 2) / (delta * gamma)

        return self._scalar_transform({
            "mean": mean,
            "variance": variance,
            "skewness": skewness,
            "kurtosis": kurt,
        })

    # -------------------------------------------------------------------- #
    # Fitting
    # -------------------------------------------------------------------- #
    @staticmethod
    def _fit_mom(x: jnp.ndarray) -> dict:
        """Fit the NIG distribution to data using method of moments (Karlis 2002, §3.1).

        For the moment estimator to exist we need ``3·kurt − 5·skew² > 0``.
        When this condition fails we fall back to a symmetric-NIG moment
        match ``(β = 0, α = 1/s, δ = s)`` where ``s`` is the sample standard
        deviation; this keeps the estimator in the admissible region for any
        input and gives a safe EM/MLE starting point.
        """
        eps = 1e-8
        sample_mean = jnp.mean(x)
        sample_var = jnp.var(x, ddof=1)
        sample_std = jnp.sqrt(jnp.maximum(sample_var, eps))
        sample_skew = skew(x)
        sample_kurt = kurtosis(x, fisher=True)  # excess kurtosis γ₂

        cond_value = 3.0 * sample_kurt - 5.0 * sample_skew ** 2

        def _regular_branch(_):
            gamma = 3.0 / (sample_std * jnp.sqrt(jnp.maximum(cond_value, eps)))
            beta = sample_skew * sample_std * gamma ** 2 / 3.0
            delta = sample_var * gamma ** 3 / jnp.maximum(beta ** 2 + gamma ** 2, eps)
            mu = sample_mean - beta * delta / jnp.maximum(gamma, eps)
            alpha = jnp.sqrt(beta ** 2 + gamma ** 2)
            return mu, alpha, beta, delta

        def _fallback_branch(_):
            mu = sample_mean
            delta = sample_std
            alpha = 1.0 / jnp.maximum(sample_std, eps)
            beta = jnp.asarray(0.0, dtype=sample_std.dtype)
            return mu, alpha, beta, delta

        mu, alpha, beta, delta = lax.cond(
            cond_value > 0.0, _regular_branch, _fallback_branch, operand=None
        )
        return NIG._params_dict(mu=mu, alpha=alpha, beta=beta, delta=delta)

    @staticmethod
    def _em_body(carry: tuple, _: None, x: Array) -> tuple:
        """Single Karlis EM iteration as a pure function, suitable for ``lax.scan``.

        Every update is closed form — no inner gradient step, no ECME.
        """
        eps = 1e-12
        mu, alpha, beta, delta = carry

        # --- E-step: posterior expectations of the IG mixing variable.
        # Karlis (2002) eqs (4)-(5): the posterior of Z|x is GIG(-1, δ√φ(x), α),
        # whose first moments reduce to ratios of Bessel K functions.
        diff = x - mu
        t = jnp.sqrt(delta ** 2 + diff ** 2)  # = δ·√φ(x)
        u = alpha * t  # argument shared by every Bessel K in the E-step

        # Log-space ratios protect against underflow in K_v(u) at large u
        # and overflow at small u (c.f. the skewed-T Bessel underflow fix).
        log_r_s = log_kv(0, u) - log_kv(1, u)
        log_r_w = log_kv(2, u) - log_kv(1, u)
        s_i = (t / alpha) * jnp.exp(log_r_s)          # E[Z|x_i]
        w_i = (alpha / t) * jnp.exp(log_r_w)          # E[Z^{-1}|x_i]

        # --- M-step: closed-form updates (Karlis 2002 p. 47-48).
        x_bar = jnp.mean(x)
        s_bar = jnp.mean(s_i)
        w_bar = jnp.mean(w_i)
        xw_bar = jnp.mean(x * w_i)

        # δ update: Λ̂ = 1 / mean(w_i − 1/s̄).
        inv_term = w_bar - 1.0 / jnp.maximum(s_bar, eps)
        lam = 1.0 / jnp.maximum(inv_term, eps)
        delta_new = jnp.sqrt(jnp.maximum(lam, eps))

        gamma_new = delta_new / jnp.maximum(s_bar, eps)

        # β update: ML regression coefficient for E[x|z] = μ + β·z with Var(x|z)=z.
        denom_b = 1.0 - s_bar * w_bar
        denom_b = jnp.where(jnp.abs(denom_b) < eps, eps, denom_b)
        beta_new = (xw_bar - x_bar * w_bar) / denom_b

        mu_new = x_bar - beta_new * s_bar
        alpha_new = jnp.sqrt(gamma_new ** 2 + beta_new ** 2)

        return (mu_new, alpha_new, beta_new, delta_new), None

    def _fit_em(self, x: jnp.ndarray, maxiter: int) -> dict:
        """Fit the NIG distribution via the Karlis (2002) EM algorithm.

        The IG mixing variable ``Z`` is treated as latent. The GIG
        conjugacy of the IG prior gives a closed-form posterior, so both
        the E-step and M-step are analytic. Compiles via ``lax.scan``.
        """
        init_params = self._fit_mom(x)
        init_carry = NIG._params_to_tuple(init_params)

        em_step = lambda carry, _: NIG._em_body(carry, _, x)
        final_carry, _ = lax.scan(em_step, init_carry, None, length=maxiter)
        mu, alpha, beta, delta = final_carry
        return NIG._params_dict(mu=mu, alpha=alpha, beta=beta, delta=delta)

    def _mle_objective_3p(
        self,
        params_arr: jnp.ndarray,
        x: jnp.ndarray,
        sample_mean: Scalar,
    ) -> Scalar:
        """3-parameter NIG objective exploiting the exact β-score identity.

        Karlis (2002) Lemma: ``∂L/∂β = 0`` gives ``x̄ = μ + δβ/γ`` exactly,
        so the observed-data MLE over ``(μ, α, β, δ)`` equals the MLE over
        ``(γ, β, δ)`` with ``μ = x̄ − δβ/γ`` and ``α = √(γ² + β²)``. We
        optimise in ``(γ, β, log δ)`` so ``δ`` stays strictly positive
        without a boundary constraint.
        """
        gamma, beta, log_delta = params_arr
        delta = jnp.exp(log_delta)
        alpha = jnp.sqrt(gamma ** 2 + beta ** 2)
        mu = sample_mean - delta * beta / gamma
        full_params = jnp.array([mu, alpha, beta, delta])
        return self._mle_objective(params_arr=full_params, x=x)

    def _fit_mle(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        """Fit via projected-gradient MLE over ``(γ, β, log δ)``.

        This is a *genuine* MLE — the β-score identity ``μ = x̄ − δβ/γ``
        is an exact first-order condition, so eliminating ``μ`` loses no
        optimality relative to a 4-D search.
        """
        eps = 1e-6
        constraints = (
            jnp.array([[eps, -jnp.inf, -jnp.inf]]).T,
            jnp.array([[jnp.inf, jnp.inf, jnp.inf]]).T,
        )
        projection_options = {"lower": constraints[0], "upper": constraints[1]}

        # Initialise from MoM so γ, β, log δ start in the admissible region.
        mom = self._fit_mom(x)
        mu0, alpha0, beta0, delta0 = NIG._params_to_tuple(mom)
        gamma0 = jnp.sqrt(jnp.maximum(alpha0 ** 2 - beta0 ** 2, eps))
        params0 = jnp.array([gamma0, beta0, jnp.log(jnp.maximum(delta0, eps))])

        sample_mean = x.mean()
        res = projected_gradient(
            f=self._mle_objective_3p,
            x0=params0,
            projection_method="projection_box",
            projection_options=projection_options,
            x=x,
            sample_mean=sample_mean,
            lr=lr,
            maxiter=maxiter,
        )
        gamma, beta, log_delta = res["x"]
        delta = jnp.exp(log_delta)
        alpha = jnp.sqrt(gamma ** 2 + beta ** 2)
        mu = sample_mean - delta * beta / gamma
        return NIG._params_dict(mu=mu, alpha=alpha, beta=beta, delta=delta)

    _supported_methods = frozenset({"em", "mle", "mom"})

    def fit(
        self,
        x: ArrayLike,
        method: str = "em",
        lr: float = 0.1,
        maxiter: int = 100,
        name: str = None,
    ):
        r"""Fit the NIG distribution to the input data.

        Note:
            If you intend to ``jit``-wrap this function, ensure that
            ``method`` is a static argument.

        Args:
            x (ArrayLike): The input data to fit the distribution to.
            method (str): Fitting method.  One of:
                ``'em'`` — iterated Karlis (2002) EM step (numerical;
                **default**);
                ``'mle'`` — 3-parameter projected-gradient MLE via the
                exact β-score identity (numerical);
                ``'mom'`` — **closed-form** method of moments.
            lr (float): Learning rate for the projected-gradient MLE.
                Ignored for ``'em'`` and ``'mom'``.
            maxiter (int): Maximum number of iterations for iterative
                methods. Ignored for ``'mom'``.
            name (str): Optional custom name for the fitted instance.

        Returns:
            NIG: A fitted ``NIG`` instance.

        Raises:
            ValueError: If ``method`` is not one of the accepted
                strings listed above.
        """
        self._check_method(method)
        x = _univariate_input(x)[0]
        if method == "mle":
            return self._fitted_instance(self._fit_mle(x, lr=lr, maxiter=maxiter), name=name)
        elif method == "em":
            return self._fitted_instance(self._fit_em(x, maxiter=maxiter), name=name)
        elif method == "mom":
            return self._fitted_instance(self._fit_mom(x), name=name)
        else:
            raise ValueError(
                f"Unknown NIG fit method {method!r}. "
                f"Expected one of: {sorted(self._supported_methods)}."
            )

    # -------------------------------------------------------------------- #
    # CDF (numerical integration with custom VJP)
    # -------------------------------------------------------------------- #
    @staticmethod
    def _params_from_array(params_arr: jnp.ndarray, *args, **kwargs) -> dict:
        mu, alpha, beta, delta = params_arr
        return NIG._params_dict(mu=mu, alpha=alpha, beta=beta, delta=delta)

    @staticmethod
    def _pdf_for_cdf(x: ArrayLike, *params_tuple) -> Array:
        """PDF evaluator used by the numerical CDF integrator.

        Overrides the base ``Univariate._pdf_for_cdf`` classmethod, which
        calls ``cls.pdf`` as though ``pdf`` were a classmethod — it is an
        instance method, so the base implementation raises
        ``TypeError: missing 1 required positional argument: 'self'``
        when invoked through the quadrature driver. We call the static
        ``_stable_logpdf`` directly instead.
        """
        params_array: jnp.ndarray = jnp.asarray(params_tuple).flatten()
        params: dict = NIG._params_from_array(params_array)
        return lax.exp(NIG._stable_logpdf(stability=0.0, x=x, params=params))

    def _cdf_anchor_scales(self, params: dict) -> Array:
        """Use the intrinsic scale parameter delta.

        The default sqrt(variance) formula for NIG is
        ``delta * alpha^2 / (alpha^2 - beta^2)^(3/2)``, which blows up
        as |beta| approaches alpha (near-boundary case). The scale
        parameter ``delta`` is always finite and positive and gives a
        numerically robust bulk scale for the t-space breakpoint grid.
        """
        _, _, _, delta = NIG._params_to_tuple(params)
        return jnp.asarray(delta).reshape((1,))

    def cdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the CDF via numerical integration with a custom VJP."""
        params = self._resolve_params(params)
        cdf_vals = _vjp_cdf(x=x, params=params)
        return self._enforce_support_on_cdf(x=x, cdf=cdf_vals, params=params)


nig = NIG("NIG")


def _vjp_cdf(x: ArrayLike, params: dict) -> Array:
    params = NIG._args_transform(params)
    return _cdf(dist=nig, x=x, params=params)


_vjp_cdf_copy = deepcopy(_vjp_cdf)
_vjp_cdf = custom_vjp(_vjp_cdf)


def cdf_fwd(x: ArrayLike, params: dict) -> tuple[Array, tuple]:
    params = NIG._args_transform(params)
    return _cdf_fwd(dist=nig, cdf_func=_vjp_cdf_copy, x=x, params=params)


_vjp_cdf.defvjp(cdf_fwd, cdf_bwd)
