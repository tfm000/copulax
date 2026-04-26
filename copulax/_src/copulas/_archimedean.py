"""CopulAX implementation of Archimedean copula distributions.

An Archimedean copula is defined by a generator function
φ: [0,1] → [0,∞) with φ(1) = 0, and its pseudo-inverse ψ = φ⁻¹:

    C(u₁,...,u_d) = ψ(φ(u₁) + ... + φ(u_d))

References:
    Nelsen, R. B. (2006). An Introduction to Copulas, 2nd ed.
        Springer Series in Statistics.
    McNeil, A. J., Frey, R., & Embrechts, P. (2005).
        Quantitative Risk Management. Princeton University Press.
    McNeil, A. J. & Nešlehová, J. (2009). Multivariate Archimedean
        copulas, d-monotone functions and l₁-norm symmetric
        distributions. Annals of Statistics, 37(5B), 3059-3097.
    Hofert, M. (2011). Efficiently Sampling Nested Archimedean
        Copulas. Computational Statistics & Data Analysis, 55(1), 57-70.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, random, lax
from jax import Array
from jax.typing import ArrayLike
from typing import Callable

from copulax._src.copulas._distributions import CopulaBase
from copulax._src._distributions import (
    Univariate,
)
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src._utils import _resolve_key
from copulax._src.typing import Scalar
from copulax._src.multivariate._shape import corr
from copulax._src.optimize import brent


###############################################################################
# ArchimedeanCopula Base Class
###############################################################################
class ArchimedeanCopula(CopulaBase):
    r"""Base class for Archimedean copula distributions.

    An Archimedean copula is defined by a generator function
    φ: [0,1] → [0,∞) with φ(1) = 0, and its pseudo-inverse ψ = φ⁻¹:

        C(u₁,...,u_d) = ψ(φ(u₁) + ... + φ(u_d))

    The copula density is:

    .. math::

        c(u) = \bigl|\psi^{(d)}\bigl(\textstyle\sum_i \phi(u_i)\bigr)\bigr|
            \cdot \prod_i \bigl|\phi'(u_i)\bigr|

    where :math:`\psi^{(d)}` is the d-th derivative of the inverse
    generator.

    References:
        Nelsen, R. B. (2006). An Introduction to Copulas, 2nd ed.
            Springer Series in Statistics.
        McNeil, A. J. & Nešlehová, J. (2009). Multivariate Archimedean
            copulas, d-monotone functions and l₁-norm symmetric
            distributions. Annals of Statistics, 37(5B), 3059-3097.
    """

    def __init__(self, name, *, marginals=None, copula=None):
        super().__init__(name)
        self._marginals = marginals if marginals is not None else None
        self._copula_params = copula if copula is not None else None

    # --- Abstract interface (subclasses must implement) ---

    def generator(self, t: Scalar, theta: Scalar) -> Scalar:
        r"""Generator function φ(t; θ).

        Must satisfy φ(1) = 0, φ is strictly decreasing and convex.
        """
        raise NotImplementedError

    def generator_inv(self, s: Scalar, theta: Scalar) -> Scalar:
        r"""Inverse generator ψ(s; θ) = φ⁻¹(s; θ).

        Also known as the Laplace-Stieltjes transform.
        """
        raise NotImplementedError

    def _tau_to_theta(self, tau: Scalar) -> Scalar:
        r"""Convert Kendall's tau to the copula parameter θ."""
        raise NotImplementedError

    def _rvs_frailty(self, key: Array, theta: Scalar, size: int) -> Array:
        r"""Sample V from the frailty distribution for Marshall-Olkin.

        The frailty distribution F_θ has Laplace transform ψ(s; θ),
        i.e., E[exp(-sV)] = ψ(s; θ).
        """
        raise NotImplementedError

    def _default_theta(self) -> float:
        r"""Default θ value for example_params."""
        raise NotImplementedError

    def _theta_bounds(self) -> tuple:
        r"""Parameter bounds (lower, upper) for θ."""
        raise NotImplementedError

    # --- Shared implementations ---

    def _params_to_tuple(self, params: dict) -> tuple:
        return (params["copula"]["theta"],)

    def example_params(self, dim: int = 3, *args, **kwargs) -> dict:
        r"""Example parameters for the Archimedean copula distribution.

        Args:
            dim: Number of dimensions. Default is 3.

        Returns:
            dict with keys 'marginals' and 'copula'.
        """
        from copulax._src.univariate.normal import normal

        marginals = tuple((normal, normal.example_params(dim=dim)) for _ in range(dim))
        return {
            "marginals": marginals,
            "copula": {"theta": jnp.float32(self._default_theta())},
        }

    # --- Copula CDF ---

    def copula_cdf(self, u: ArrayLike, params: dict = None) -> Array:
        r"""Copula CDF: C(u₁,...,u_d) = ψ(φ(u₁) + ... + φ(u_d)).

        Args:
            u: Uniform marginal values of shape (n, d).
            params: Must contain 'copula' → {'theta': scalar}.

        Returns:
            Array of shape (n, 1).
        """
        u_arr: jnp.ndarray = _multivariate_input(u)[0]
        params = self._resolve_params(params)
        theta: Scalar = params["copula"]["theta"]
        phi = lambda t: self.generator(t, theta)
        psi = lambda s: self.generator_inv(s, theta)
        phi_u: jnp.ndarray = vmap(vmap(phi))(u_arr)  # (n, d)
        s: jnp.ndarray = phi_u.sum(axis=1)  # (n,)
        return vmap(psi)(s)[:, None]

    # --- Copula log-PDF ---

    def copula_logpdf(self, u: ArrayLike, params: dict = None, **kwargs) -> Array:
        r"""Copula log-density via generator derivatives.

        Uses the formula:
            log c(u) = log|ψ⁽ᵈ⁾(∑ φ(uᵢ))| + ∑ log|φ'(uᵢ)|

        where ψ⁽ᵈ⁾ is computed via d nested applications of jax.grad.

        Args:
            u: Uniform marginal values of shape (n, d).
            params: Must contain 'copula' → {'theta': scalar}.

        Returns:
            Array of shape (n, 1).
        """
        u_arr: jnp.ndarray = _multivariate_input(u)[0]
        params = self._resolve_params(params)
        theta: Scalar = params["copula"]["theta"]
        d: int = u_arr.shape[1]

        phi = lambda t: self.generator(t, theta)
        psi_fn = lambda s: self.generator_inv(s, theta)
        phi_prime = jax.grad(phi)

        # Compute ψ⁽ᵈ⁾ via nested autodiff
        psi_d = psi_fn
        for _ in range(d):
            psi_d = jax.grad(psi_d)

        def _single_logpdf(u_row):
            phi_vals = vmap(phi)(u_row)  # (d,)
            s = phi_vals.sum()
            log_abs_phi_prime = vmap(lambda t: jnp.log(jnp.abs(phi_prime(t))))(u_row)
            log_abs_psi_d = jnp.log(jnp.abs(psi_d(s)))
            return log_abs_psi_d + log_abs_phi_prime.sum()

        return vmap(_single_logpdf)(u_arr)[:, None]

    # --- Copula RVS (Marshall-Olkin algorithm) ---

    def copula_rvs(self, size: Scalar, params: dict = None, key: Array = None) -> Array:
        r"""Sample from the copula using the Marshall-Olkin algorithm.

        Algorithm (Marshall & Olkin, 1988):
            1. Sample V ~ frailty distribution F_θ
            2. Sample E₁,...,E_d iid ~ Exp(1)
            3. Uᵢ = ψ(Eᵢ / V)

        Args:
            size: Number of samples to generate.
            params: Must contain 'marginals' (for dimension) and
                'copula' → {'theta': scalar}.
            key: JAX random key.

        Returns:
            Array of shape (size, d) with values in (0, 1).
        """
        key = _resolve_key(key)
        params = self._resolve_params(params)
        d: int = self._get_dim(params)
        theta: Scalar = params["copula"]["theta"]

        key1, key2 = random.split(key)
        V: jnp.ndarray = self._rvs_frailty(key1, theta, size)  # (size,)
        E: jnp.ndarray = random.exponential(key2, shape=(size, d))

        ratios: jnp.ndarray = E / V[:, None]  # (size, d)
        psi = lambda s: self.generator_inv(s, theta)
        u: jnp.ndarray = vmap(vmap(psi))(ratios)
        return jnp.clip(u, 1e-7, 1 - 1e-7)

    # --- Metrics ---

    def aic(self, x: ArrayLike, params: dict = None) -> float:
        r"""Akaike Information Criterion."""
        params = self._resolve_params(params)
        k: int = 1  # theta
        return 2 * k - 2 * self.loglikelihood(x=x, params=params)

    def bic(self, x: ArrayLike, params: dict = None) -> float:
        r"""Bayesian Information Criterion."""
        params = self._resolve_params(params)
        x_arr, _, n, _ = _multivariate_input(x)
        k: int = 1  # theta
        return k * jnp.log(n) - 2 * self.loglikelihood(x=x_arr, params=params)

    # --- Fitting ---

    _supported_methods: frozenset = frozenset({"kendall"})

    def fit_copula(self, u: ArrayLike, method: str = "kendall", **kwargs) -> dict:
        r"""Fit the copula parameter θ.

        Args:
            u: Uniform marginal values of shape (n, d).
            method: Fitting algorithm. Only ``'kendall'`` is currently
                supported (Kendall's tau inversion).

        Returns:
            dict with key 'copula' → {'theta': fitted_theta}.

        Raises:
            ValueError: If ``method`` is not ``'kendall'``, or if
                ``kwargs`` contains any keys (the Kendall method takes
                no tuning parameters).
        """
        self._check_method(method)
        if kwargs:
            raise ValueError(
                f"Method {method!r} does not accept kwargs "
                f"{sorted(kwargs)}. Accepted: []."
            )
        return self._fit_copula_kendall(u)

    def _fit_copula_kendall(self, u: ArrayLike) -> dict:
        r"""Fit θ via average pairwise Kendall's tau inversion.

        Computes the average pairwise Kendall's tau from the data,
        then applies the copula-specific τ(θ) inversion.
        """
        u_arr: jnp.ndarray = _multivariate_input(u)[0]
        tau_matrix: jnp.ndarray = corr(u_arr, method="kendall")
        d: int = tau_matrix.shape[0]
        mask: jnp.ndarray = 1.0 - jnp.eye(d)
        tau_avg: Scalar = (tau_matrix * mask).sum() / (d * (d - 1))
        theta: Scalar = self._tau_to_theta(tau_avg)
        return {"copula": {"theta": theta}}


###############################################################################
# Clayton Copula
###############################################################################
class ClaytonCopula(ArchimedeanCopula):
    r"""Clayton copula with generator φ(t) = t^{-θ} - 1.

    Parameters:
        θ ∈ (0, ∞)

    Properties:
        - Lower tail dependence: λ_L = 2^{-1/θ}
        - Upper tail dependence: λ_U = 0
        - Kendall's tau: τ = θ/(θ+2)

    References:
        Clayton, D. G. (1978). A model for association in bivariate
            life tables and its application in epidemiological studies
            of familial tendency in chronic disease incidence.
            Biometrika, 65(1), 141-151.
        Nelsen (2006), Example 4.2.
    """

    def generator(self, t, theta):
        return jnp.power(t, -theta) - 1.0

    def generator_inv(self, s, theta):
        return jnp.power(1.0 + s, -1.0 / theta)

    def _tau_to_theta(self, tau):
        # τ = θ/(θ+2) ⟹ θ = 2τ/(1-τ)
        return 2.0 * tau / (1.0 - tau)

    def _rvs_frailty(self, key, theta, size):
        # V ~ Gamma(1/θ, 1)
        return random.gamma(key, 1.0 / theta, shape=(size,))

    def _default_theta(self):
        return 2.0

    def _theta_bounds(self):
        return (1e-6, jnp.inf)

    def copula_logpdf(self, u, params=None, **kwargs):
        r"""Closed-form Clayton copula log-density.

        log c(u) = ∑_{k=1}^{d-1} log(1 + kθ)
                   - (1/θ + d) · log(∑ uᵢ^{-θ} - (d-1))
                   - (θ + 1) · ∑ log(uᵢ)

        Derivation from ψ⁽ᵈ⁾(s) = (-1)^d · ∏_{k=0}^{d-1}(1/θ+k) · (1+s)^{-1/θ-d}.
        """
        u_arr: jnp.ndarray = _multivariate_input(u)[0]
        params = self._resolve_params(params)
        theta: Scalar = params["copula"]["theta"]
        d: int = u_arr.shape[1]

        def _single(u_row):
            # Prefactor: ∑ log(1 + kθ) for k = 1, ..., d-1
            ks = jnp.arange(1, d, dtype=float)
            log_prefactor = jnp.sum(jnp.log(1.0 + ks * theta))

            # S = ∑ uᵢ^{-θ} - (d-1)
            S = jnp.sum(jnp.power(u_row, -theta)) - (d - 1.0)

            return (
                log_prefactor
                - (1.0 / theta + d) * jnp.log(S)
                - (theta + 1.0) * jnp.sum(jnp.log(u_row))
            )

        return vmap(_single)(u_arr)[:, None]


clayton_copula = ClaytonCopula("Clayton-Copula")


###############################################################################
# Frank Copula
###############################################################################
class FrankCopula(ArchimedeanCopula):
    r"""Frank copula with generator φ(t) = -ln((e^{-θt}-1)/(e^{-θ}-1)).

    Parameters:
        θ ∈ ℝ \ {0}

    Properties:
        - No tail dependence: λ_L = λ_U = 0
        - Allows negative dependence (θ < 0)
        - Kendall's tau: τ = 1 - 4/θ · (1 - D₁(θ))
          where D₁ is the first Debye function

    References:
        Frank, M. J. (1979). On the simultaneous associativity of
            F(x,y) and x + y - F(x,y). Aequationes Mathematicae,
            19, 194-226.
        Nelsen (2006), Example 4.5.
    """

    def generator(self, t, theta):
        # φ(t) = -ln((e^{-θt} - 1) / (e^{-θ} - 1))
        return -jnp.log(jnp.expm1(-theta * t) / jnp.expm1(-theta))

    def generator_inv(self, s, theta):
        # ψ(s) = -1/θ · ln(1 + e^{-s} · (e^{-θ} - 1))
        return -jnp.log1p(jnp.exp(-s) * jnp.expm1(-theta)) / theta

    @staticmethod
    def _debye1(x):
        r"""First Debye function D₁(x) = (1/x) ∫₀ˣ t/(eᵗ-1) dt.

        Computed via the series:
            D₁(x) = (1/x) · ∑_{n=1}^N [1/n² - (x/n + 1/n²)e^{-nx}]

        which converges rapidly for all x > 0.
        """
        abs_x = jnp.abs(x)
        ns = jnp.arange(1.0, 51.0)
        terms = 1.0 / ns**2 - (abs_x / ns + 1.0 / ns**2) * jnp.exp(-ns * abs_x)
        return jnp.where(abs_x < 1e-10, 1.0, terms.sum() / abs_x)

    def _tau_to_theta(self, tau):
        r"""Invert τ = 1 - 4/θ · (1 - D₁(θ)) via Brent's method."""

        def residual(theta):
            return 1.0 - 4.0 / theta * (1.0 - self._debye1(theta)) - tau

        # Search bounds depend on sign of tau
        lo = jnp.where(tau >= 0, 0.1, -100.0)
        hi = jnp.where(tau >= 0, 100.0, -0.1)
        bounds = jnp.array([lo, hi])
        return brent(residual, bounds=bounds)

    def _rvs_frailty(self, key, theta, size):
        r"""Sample :math:`V \sim \mathrm{Logarithmic}(1 - e^{-|\theta|})` via truncated PMF.

        The Logarithmic distribution has PMF
        ``P(V=k) = -p^k / (k · ln(1-p)),  k = 1, 2, ...``
        with ``p = 1 - exp(-|θ|)``.

        Uses categorical sampling from a truncated PMF with K_max terms.
        """
        p = -jnp.expm1(-jnp.abs(theta))
        K_max = 500
        ks = jnp.arange(1, K_max + 1, dtype=float)
        log_pmf = ks * jnp.log(p) - jnp.log(ks)
        log_pmf = log_pmf - jax.nn.logsumexp(log_pmf)  # normalize
        return random.categorical(key, log_pmf, shape=(size,)).astype(float) + 1.0

    def _default_theta(self):
        return 5.0

    def _theta_bounds(self):
        return (-100.0, 100.0)


frank_copula = FrankCopula("Frank-Copula")


###############################################################################
# Gumbel Copula
###############################################################################
class GumbelCopula(ArchimedeanCopula):
    r"""Gumbel copula with generator φ(t) = (-ln t)^θ.

    Parameters:
        θ ∈ [1, ∞)

    Properties:
        - No lower tail dependence: λ_L = 0
        - Upper tail dependence: λ_U = 2 - 2^{1/θ}
        - Kendall's tau: τ = 1 - 1/θ
        - θ = 1 gives the independence copula

    References:
        Gumbel, E. J. (1960). Distributions des valeurs extrêmes en
            plusieurs dimensions. Publications de l'Institut de
            Statistique de l'Université de Paris, 9, 171-173.
        Nelsen (2006), Example 4.4.
    """

    def generator(self, t, theta):
        return jnp.power(-jnp.log(t), theta)

    def generator_inv(self, s, theta):
        return jnp.exp(-jnp.power(s, 1.0 / theta))

    def _tau_to_theta(self, tau):
        # τ = 1 - 1/θ ⟹ θ = 1/(1-τ)
        return 1.0 / (1.0 - tau)

    def _rvs_frailty(self, key, theta, size):
        r"""Sample V ~ Stable(1/θ) via Hofert (2011) Algorithm 1.

        For α = 1/θ ∈ (0, 1], generates positive stable V with
        Laplace transform E[e^{-sV}] = exp(-s^α).

        Implementation mirrors R copula::rPosStableS (Hofert 2011):
            Θ ~ Uniform(0, π), W ~ Exp(1), I_a = 1 - α
            a = sin(I_a·Θ) · [sin(αΘ)^α / sin(Θ)]^{1/I_a}
            V = (a/W)^{I_a/α}

        For θ = 1 (α = 1), V = 1 deterministically (independence).
        """
        alpha = 1.0 / theta
        I_a = 1.0 - alpha

        key1, key2 = random.split(key)
        # Avoid boundary values 0 and π for numerical stability
        U = random.uniform(key1, shape=(size,), minval=1e-10, maxval=jnp.pi - 1e-10)
        W = random.exponential(key2, shape=(size,))

        a = jnp.sin(I_a * U) * jnp.power(
            jnp.power(jnp.sin(alpha * U), alpha) / jnp.sin(U),
            1.0 / I_a,
        )
        V = jnp.power(a / W, I_a / alpha)

        # Handle θ = 1 (α = 1): V = 1 deterministically
        return jnp.where(theta <= 1.0 + 1e-8, 1.0, V)

    def _default_theta(self):
        return 2.0

    def _theta_bounds(self):
        return (1.0, jnp.inf)


gumbel_copula = GumbelCopula("Gumbel-Copula")


###############################################################################
# Joe Copula
###############################################################################
class JoeCopula(ArchimedeanCopula):
    r"""Joe copula with generator φ(t) = -ln(1 - (1-t)^θ).

    Parameters:
        θ ∈ [1, ∞)

    Properties:
        - No lower tail dependence: λ_L = 0
        - Upper tail dependence: λ_U = 2 - 2^{1/θ}
        - Kendall's tau: τ = 1 - 4·∑_{k=1}^∞ 1/(k(kθ+2)((k-2)θ+2))

    References:
        Joe, H. (1993). Parametric families of multivariate
            distributions with given margins. Journal of Multivariate
            Analysis, 46(2), 262-282.
        Nelsen (2006), Example 4.10.
    """

    def generator(self, t, theta):
        return -jnp.log1p(-jnp.power(1.0 - t, theta))

    def generator_inv(self, s, theta):
        return 1.0 - jnp.power(-jnp.expm1(-s), 1.0 / theta)

    @staticmethod
    def _theta_to_tau(theta):
        r"""Kendall's tau for Joe copula via numerical quadrature.

        Uses the general Archimedean formula:
            τ = 1 + 4·∫₀¹ φ(t)/φ'(t) dt

        Reformulated for numerical stability as:
            φ/φ' = [-ln(1-x)]·(1-x)·(1-t)^{1-θ} / θ
            where x = (1-t)^θ.
        """
        ts = jnp.linspace(1e-6, 1 - 1e-6, 1000)
        one_minus_t = 1.0 - ts
        x = jnp.power(one_minus_t, theta)  # (1-t)^θ
        neg_log = -jnp.log1p(-x)  # -ln(1 - x) ≥ 0
        # φ/φ' = -ln(1-x)·(1-x)·(1-t)^{1-θ}/θ  (negative, since φ'>0 here)
        integrand = neg_log * (1.0 - x) * jnp.power(one_minus_t, 1.0 - theta) / theta
        integrand = jnp.where(jnp.isfinite(integrand), integrand, 0.0)
        # The integrand represents φ/φ' which is negative
        return 1.0 - 4.0 * jnp.trapezoid(integrand, ts)

    def _tau_to_theta(self, tau):
        r"""Invert τ(θ) via bisection.

        Since τ is monotonically increasing in θ ∈ [1, ∞),
        bisection is guaranteed to converge.
        """
        lo = 1.0 + 1e-6
        hi = 100.0

        def _bisect_step(state, _):
            lo, hi = state
            mid = (lo + hi) / 2.0
            tau_mid = self._theta_to_tau(mid)
            lo = jnp.where(tau_mid < tau, mid, lo)
            hi = jnp.where(tau_mid >= tau, mid, hi)
            return (lo, hi), None

        (lo, hi), _ = lax.scan(_bisect_step, (lo, hi), None, length=60)
        return (lo + hi) / 2.0

    def _rvs_frailty(self, key, theta, size):
        r"""Sample V ~ Sibuya(1/θ) via Hofert (2011) Proposition 3.2.

        Implementation mirrors R copula::rSibuya (src/rSibuya.c).
        Sibuya(α) is heavy-tailed (P(V>k) ~ k^{-α}/Γ(1-α)), so
        truncated-PMF approaches lose mass and bias V downward.
        Hofert's algorithm samples exactly via inversion of the
        asymptotic CDF approximation with a single Beta-function
        correction step:

            U ~ U(0,1)
            if U <= α: return 1
            else:
                Ginv = ((1-U)·Γ(1-α))^{-1/α}
                if Ginv > 1/ε:        return floor(Ginv)
                if 1-U < 1/(⌊Ginv⌋·B(⌊Ginv⌋, 1-α)):
                                       return ceil(Ginv)
                else:                  return floor(Ginv)
        """
        from jax.scipy.special import gammaln

        alpha = 1.0 / theta
        U = random.uniform(key, shape=(size,))
        one_minus_U = 1.0 - U

        gamma_1_a = jnp.exp(gammaln(1.0 - alpha))
        Ginv = jnp.power(one_minus_U * gamma_1_a, -1.0 / alpha)
        fGinv = jnp.floor(Ginv)
        cGinv = jnp.ceil(Ginv)

        # log B(fGinv, 1-α) = lgamma(fGinv) + lgamma(1-α) - lgamma(fGinv + 1-α)
        log_beta = (gammaln(fGinv) + gammaln(1.0 - alpha)
                    - gammaln(fGinv + 1.0 - alpha))
        # 1-U < 1/(fGinv · B) ⟺ log(1-U) < -log(fGinv) - log_beta
        bump_up = jnp.log(one_minus_U) < (-jnp.log(fGinv) - log_beta)

        xMax = 1.0 / jnp.finfo(jnp.float64).eps
        # Three-way nested where: U<=α → 1; else if Ginv>xMax → fGinv;
        # else if bump_up → cGinv; else → fGinv.
        return jnp.where(
            U <= alpha,
            1.0,
            jnp.where(
                Ginv > xMax,
                fGinv,
                jnp.where(bump_up, cGinv, fGinv),
            ),
        )

    def _default_theta(self):
        return 2.0

    def _theta_bounds(self):
        return (1.0, jnp.inf)


joe_copula = JoeCopula("Joe-Copula")


###############################################################################
# Ali-Mikhail-Haq (AMH) Copula
###############################################################################
class AMHCopula(ArchimedeanCopula):
    r"""Ali-Mikhail-Haq copula with generator φ(t) = ln((1-θ(1-t))/t).

    Parameters:
        θ ∈ [-1, 1)

    Properties:
        - Only valid for dimension d ≤ 2
        - Weak dependence: τ ∈ [-0.1817, 1/3)
        - No tail dependence: λ_L = λ_U = 0

    Note:
        The AMH generator is only completely monotone (hence valid for
        all dimensions) when θ ∈ [0, 1). For θ ∈ [-1, 0), the copula
        is only valid for d = 2 (2-monotone but not completely monotone).
        The Marshall-Olkin sampling uses a Geometric frailty distribution
        which requires θ ∈ [0, 1). For θ < 0, frailty V is set to 1,
        providing approximate sampling.

    References:
        Ali, M. M., Mikhail, N. N., & Haq, M. S. (1978). A class
            of bivariate distributions including the bivariate logistic.
            Journal of Multivariate Analysis, 8(3), 405-412.
        Nelsen (2006), Example 4.8.
    """

    def generator(self, t, theta):
        return jnp.log((1.0 - theta * (1.0 - t)) / t)

    def generator_inv(self, s, theta):
        return (1.0 - theta) / (jnp.exp(s) - theta)

    @staticmethod
    def _theta_to_tau(theta):
        r"""Kendall's tau for AMH copula.

        τ = 1 - 2(θ + (1-θ)²·ln(1-θ)) / (3θ²)

        For :math:`|\theta| < \varepsilon`, uses the Taylor approximation
        :math:`\tau \approx 2\theta/9`.
        """
        safe_theta = jnp.clip(theta, -0.999, 0.999)
        numerator = safe_theta + jnp.square(1.0 - safe_theta) * jnp.log(
            1.0 - safe_theta
        )
        full_result = 1.0 - 2.0 * numerator / (3.0 * jnp.square(safe_theta))
        # Taylor approximation near θ = 0 to avoid 0/0
        return jnp.where(
            jnp.abs(safe_theta) < 0.01,
            2.0 * safe_theta / 9.0,
            full_result,
        )

    def _tau_to_theta(self, tau):
        r"""Invert τ(θ) via Brent's method."""

        def residual(theta):
            return self._theta_to_tau(theta) - tau

        # For τ > 0: θ > 0; for τ < 0: θ < 0
        lo = jnp.where(tau >= 0, 0.01, -0.99)
        hi = jnp.where(tau >= 0, 0.99, -0.01)
        bounds = jnp.array([lo, hi])
        return brent(residual, bounds=bounds)

    def _rvs_frailty(self, key, theta, size):
        r"""Sample V ~ Geometric(1-θ) for θ ∈ [0, 1).

        P(V = k) = (1-θ)·θ^{k-1}, k = 1, 2, ...
        Sampled as V = 1 + floor(log(U) / log(θ)), U ~ Uniform(0,1).

        For θ = 0, V = 1 (independence). For θ < 0, V = 1 is used
        as an approximation.
        """
        safe_theta = jnp.clip(theta, 1e-10, 1.0 - 1e-6)
        U = random.uniform(key, shape=(size,))
        V = 1.0 + jnp.floor(jnp.log(U) / jnp.log(safe_theta))
        # For theta ≤ 0, V = 1
        return jnp.where(theta <= 0.0, 1.0, V)

    def _default_theta(self):
        return 0.5

    def _theta_bounds(self):
        return (-1.0, 1.0)

    def example_params(self, dim: int = 2, *args, **kwargs) -> dict:
        r"""Example parameters for AMH copula (d=2 only).

        Args:
            dim: Must be 2 for AMH copula.

        Raises:
            ValueError: If dim != 2.
        """
        if dim != 2:
            raise ValueError(
                "AMH copula only supports dimension d=2. " f"Got dim={dim}."
            )
        return super().example_params(dim=2, *args, **kwargs)


amh_copula = AMHCopula("AMH-Copula")


###############################################################################
# Independence (Product) Copula
###############################################################################
class IndependenceCopula(ArchimedeanCopula):
    r"""Independence copula (product copula) C(u₁,...,u_d) = ∏ uᵢ.

    The independence copula corresponds to the Archimedean generator
    φ(t) = −ln(t) with inverse ψ(s) = e^{−s}. It has no free
    parameters and represents the case of stochastically independent
    margins.

    Properties:
        - No tail dependence: λ_L = λ_U = 0
        - Kendall's tau: τ = 0
        - Copula density: c(u) = 1 for all u ∈ (0,1)^d
        - No parameters to fit
        - Valid for any dimension d ≥ 2

    The independence copula is useful as a null / benchmark model
    for comparing against parametric copula fits.

    References:
        Nelsen, R. B. (2006). An Introduction to Copulas, 2nd ed.
            Springer Series in Statistics, Section 2.5.
    """

    def generator(self, t, theta):
        return -jnp.log(t)

    def generator_inv(self, s, theta):
        return jnp.exp(-s)

    def _tau_to_theta(self, tau):
        return 1.0

    def _rvs_frailty(self, key, theta, size):
        return jnp.ones((size,))

    def _default_theta(self):
        return 1.0

    def _theta_bounds(self):
        return (1.0, 1.0)

    def _params_to_tuple(self, params: dict) -> tuple:
        return ()

    # --- Parameter handling (no copula params) ---

    def example_params(self, dim: int = 3, *args, **kwargs) -> dict:
        r"""Example parameters for the independence copula.

        Args:
            dim: Number of dimensions. Default is 3.

        Returns:
            dict with keys 'marginals' and 'copula' (empty dict).
        """
        from copulax._src.univariate.normal import normal

        marginals = tuple((normal, normal.example_params(dim=dim)) for _ in range(dim))
        return {"marginals": marginals, "copula": {}}

    # --- Copula CDF: C(u) = ∏ uᵢ ---

    def copula_cdf(self, u, params=None, **kwargs):
        r"""Independence copula CDF: C(u₁,...,u_d) = ∏ uᵢ.

        Args:
            u: Uniform marginal values of shape (n, d).
            params: Ignored (no copula parameters).

        Returns:
            Array of shape (n, 1).
        """
        u_arr = _multivariate_input(u)[0]
        return jnp.prod(u_arr, axis=1, keepdims=True)

    # --- Copula log-PDF: log c(u) = 0 ---

    def copula_logpdf(self, u, params=None, **kwargs):
        r"""Independence copula log-density: log c(u) = 0.

        Args:
            u: Uniform marginal values of shape (n, d).
            params: Ignored (no copula parameters).

        Returns:
            Array of zeros with shape (n, 1).
        """
        u_arr = _multivariate_input(u)[0]
        return jnp.zeros((u_arr.shape[0], 1))

    # --- Copula RVS: independent uniforms ---

    def copula_rvs(self, size, params=None, key=None):
        r"""Sample independent uniform margins.

        Args:
            size: Number of samples to generate.
            params: Must contain 'marginals' (for dimension inference).
            key: JAX random key.

        Returns:
            Array of shape (size, d) with iid Uniform(0,1) entries.
        """
        key = _resolve_key(key)
        params = self._resolve_params(params)
        d = self._get_dim(params)
        return jax.random.uniform(key, shape=(size, d), minval=1e-7, maxval=1 - 1e-7)

    # --- Fitting (nothing to fit) ---

    def fit_copula(self, u, method: str = "kendall", **kwargs):
        r"""No parameters to fit for the independence copula.

        Validates ``method`` (only ``'kendall'`` is accepted) and
        rejects any kwargs for parity with the rest of the family,
        then returns an empty copula-params dict.

        Returns:
            dict with key 'copula' → {} (empty).
        """
        self._check_method(method)
        if kwargs:
            raise ValueError(
                f"Method {method!r} does not accept kwargs "
                f"{sorted(kwargs)}. Accepted: []."
            )
        return {"copula": {}}

    # --- Metrics (k=0 free parameters) ---

    def aic(self, x, params=None):
        r"""Akaike Information Criterion with k=0 free parameters."""
        params = self._resolve_params(params)
        return -2.0 * self.loglikelihood(x=x, params=params)

    def bic(self, x, params=None):
        r"""Bayesian Information Criterion with k=0 free parameters."""
        params = self._resolve_params(params)
        return -2.0 * self.loglikelihood(x=x, params=params)


independence_copula = IndependenceCopula("Independence-Copula")
