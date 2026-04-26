"""File containing the copulAX implementation of the multivariate
skewed-T distribution."""

import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, random, jit, value_and_grad
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import NormalMixture
from copulax._src.typing import Scalar
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src._utils import _resolve_key, get_random_key
from copulax._src.multivariate._shape import cov, _corr
from copulax._src.multivariate._normal_mixture import (
    prepare_sample_cov,
    forward_reparam,
    invert_gamma_to_z,
)
from copulax._src.univariate.ig import ig
from copulax._src.univariate.skewed_t import skewed_t
from copulax._src.univariate.gh import GH
from copulax._src.special import log_kv_plus_s_log_r
from copulax._src.stats import kurtosis

_NU_LDMLE_MIN = 4.0 + 1e-3
_NU_INIT = 4.0


class MvtSkewedT(NormalMixture):
    r"""The multivariate skewed-T distribution is a generalization of
    the univariate skewed-T distribution to d > 1 dimensions, which
    itself is a generalization of the student-t distribution which
    allows for skewness. It can also be expressed as a limiting case of
    the multivariate generalized hyperbolic distribution (GH) when
    phi -> 0 in addition to lamb = -0.5*chi.

    We use the 4 parameter McNeil et al (2005) specification of the
    distribution.
    """

    nu: Array = None
    mu: Array = None
    gamma: Array = None
    sigma: Array = None

    def __init__(
        self, name="Mvt-Skewed-T", *, nu=None, mu=None, gamma=None, sigma=None
    ):
        """Initialize with optional stored parameters ``nu``, ``mu``, ``gamma``, and ``sigma``."""
        super().__init__(name)
        self.nu = jnp.asarray(nu, dtype=float).reshape(()) if nu is not None else None
        self.mu = jnp.asarray(mu, dtype=float) if mu is not None else None
        self.gamma = jnp.asarray(gamma, dtype=float) if gamma is not None else None
        self.sigma = jnp.asarray(sigma, dtype=float) if sigma is not None else None

    @property
    def _stored_params(self):
        """Return stored parameters dict if all are set, else None."""
        if any(v is None for v in [self.nu, self.mu, self.gamma, self.sigma]):
            return None
        return {"nu": self.nu, "mu": self.mu, "gamma": self.gamma, "sigma": self.sigma}

    def _classify_params(self, params: dict) -> tuple:
        """Classify parameters into scalar, vector, and shape groups."""
        return super()._classify_params(
            params=params,
            scalar_names=("nu",),
            vector_names=("mu", "gamma"),
            shape_names=("sigma",),
            symmetric_shape_names=("sigma",),
        )

    def _params_dict(
        self, nu: Scalar, mu: ArrayLike, gamma: ArrayLike, sigma: ArrayLike
    ) -> dict:
        """Construct a normalized parameters dict from ``nu``, ``mu``, ``gamma``, and ``sigma``."""
        d: dict = {"nu": nu, "mu": mu, "gamma": gamma, "sigma": sigma}
        return self._args_transform(d)

    def _params_to_tuple(self, params: dict) -> tuple:
        """Extract ``(nu, mu, gamma, sigma)`` tuple from a parameters dict."""
        params = self._args_transform(params)
        return (params["nu"], params["mu"], params["gamma"], params["sigma"])

    def example_params(self, dim: int = 3, *args, **kwargs):
        """Example parameters for the multivariate skewed-t distribution.

        Args:
            dim: Number of dimensions. Default is 3.
        """
        return self._params_dict(
            nu=4.5,
            mu=jnp.zeros((dim, 1)),
            gamma=jnp.zeros((dim, 1)),
            sigma=jnp.eye(dim, dim),
        )

    def support(self, params: dict = None) -> Array:
        """Return the support: ``(-inf, inf)`` per dimension."""
        return super().support(params=params)

    @staticmethod
    def _logpdf_core(
        stability: Scalar,
        x: Array,
        nu: Scalar,
        mu: Array,
        gamma: Array,
        sigma: Array,
    ) -> Array:
        r"""Core log-PDF computation for the multivariate skewed-t distribution.

        This is a static, pure function suitable for use inside
        ``value_and_grad``.  Both the public ``_stable_logpdf`` and the
        ECME shape-parameter gradient call this.

        Implements McNeil et al. (2005) equation (3.32):

        .. math::

            \log f(x) = \log c + \log K_s(\sqrt{A})
                         + H + \tfrac{s}{2}\log A
                         - s\,\log(1 + Q/\nu)

        where :math:`s = (\nu+d)/2`,
        :math:`A = (\nu + Q)\,R_\gamma`,
        :math:`Q = (x-\mu)'\Sigma^{-1}(x-\mu)`,
        :math:`R_\gamma = \gamma'\Sigma^{-1}\gamma`,
        :math:`H = (x-\mu)'\Sigma^{-1}\gamma`.

        Args:
            stability: Small constant for numerical stability.
            x: Input data of shape (n, d).
            nu: Degrees of freedom (scalar).
            mu: Location vector of shape (d, 1).
            gamma: Skewness vector of shape (d, 1).
            sigma: Covariance matrix of shape (d, d).

        Returns:
            Array of log-density values with shape (n,).
        """
        d: int = x.shape[1]

        sigma_inv: Array = jnp.linalg.inv(sigma)
        diff: Array = x - mu.flatten()
        Q: Array = jnp.sum(diff @ sigma_inv * diff, axis=1)
        P: Array = sigma_inv @ gamma
        R: Array = (gamma.T @ P).squeeze()
        s: Scalar = 0.5 * (nu + d)
        log_det_sigma: Scalar = jnp.linalg.slogdet(sigma)[1]

        log_c: Scalar = (
            (1 - s) * jnp.log(2)
            - lax.lgamma(0.5 * nu)
            - 0.5 * (d * lax.log(nu * jnp.pi + stability) + log_det_sigma)
        )

        # log K_s(r) + (s/2) log((ν+Q)·R) = log K_s(r) + s log r,
        # computed as one cancellation-stable object via
        # :py:func:`log_kv_plus_s_log_r`.  The jnp.maximum floor on
        # the sqrt argument keeps ``∂r/∂γ`` finite at ``γ = 0``
        # (otherwise ``∂sqrt(z)/∂z = 1/(2√z) = ∞`` at ``z = 0``
        # multiplies against the upstream ``∂z/∂γ = 0`` to give NaN).
        # The 1e-24 floor matches the 1e-12 internal floor of
        # ``log_kv_plus_s_log_r`` (squared), so the direct-sum path
        # inside the helper is reached for any γ > 0.
        r = jnp.sqrt(jnp.maximum((nu + Q) * R, 1e-24))
        log_kv_plus = log_kv_plus_s_log_r(s, r)

        return (
            log_c
            + log_kv_plus
            + ((x - mu.T) @ P).flatten()
            - s * lax.log(1 + Q / (nu + stability))
        )

    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, params: dict) -> Array:
        """Stable log-PDF, wrapping :py:meth:`_logpdf_core` with shape handling.

        :py:meth:`_logpdf_core` handles the ``γ = 0`` removable
        singularity internally via :py:func:`log_kv_plus_s_log_r`, so
        a single forward pass is enough — no γ-branch dispatch and no
        doubled autograd trace.
        """
        x, yshape, _, _ = _multivariate_input(x)
        nu, mu, gamma, sigma = self._params_to_tuple(params)
        return MvtSkewedT._logpdf_core(
            stability, x, nu, mu, gamma, sigma
        ).reshape(yshape)

    # sampling
    def rvs(self, size: int, params: dict = None, key: ArrayLike = None) -> Array:
        """Generate random samples via the normal-variance mixture.

        Args:
            size: Number of samples to draw.
            params: Distribution parameters.
            key: JAX random key.

        Returns:
            Array of shape (size, d).
        """
        params = self._resolve_params(params)
        key = _resolve_key(key)
        nu, mu, gamma, sigma = self._params_to_tuple(params)

        key, subkey = random.split(key)
        W: Array = ig.rvs(
            size=(size,), key=key, params={"alpha": 0.5 * nu, "beta": 0.5 * nu}
        )
        return super()._rvs(key=subkey, n=size, W=W, mu=mu, gamma=gamma, sigma=sigma)

    # stats
    def stats(self, params: dict = None) -> dict:
        """Compute distribution statistics using inverse-gamma mixing moments."""
        params = self._resolve_params(params)
        nu, mu, gamma, sigma = self._params_to_tuple(params)
        ig_stats = ig.stats(params={"alpha": 0.5 * nu, "beta": 0.5 * nu})
        return self._stats(w_stats=ig_stats, mu=mu, gamma=gamma, sigma=sigma)

    # fitting — ECME algorithm (McNeil et al. 2005, Algorithm 3.14, skewed-t case)
    @staticmethod
    @jit
    def _nll_nu_value_and_grad(
        nu: Array, mu: Array, gamma: Array, sigma: Array, x: Array
    ) -> tuple:
        """Compute NLL and gradient w.r.t. the scalar nu parameter.

        This implements the ECME variant of CM-step 2 from McNeil et al.
        (2005, p. 83): maximize the original likelihood (3.33) w.r.t. nu
        with the other parameters held fixed.

        For the skewed-t, nu is the only shape parameter (unlike the
        general GH which has lamb, chi, psi).

        Args:
            nu: Degrees of freedom (scalar array of shape (1,) or ()).
            mu: Location vector of shape (d, 1).
            gamma: Skewness vector of shape (d, 1).
            sigma: Covariance matrix of shape (d, d).
            x: Data array of shape (n, d).

        Returns:
            Tuple of (nll_value, gradient) where gradient is a scalar.
        """
        def _nll(nu_val, mu, gamma, sigma, x):
            logpdf = MvtSkewedT._logpdf_core(1e-30, x, nu_val, mu, gamma, sigma)
            return -jnp.mean(logpdf)

        return value_and_grad(_nll)(nu, mu, gamma, sigma, x)

    @staticmethod
    def _em_body(
        carry: tuple,
        _: None,
        x: Array,
        log_det_S: Scalar,
        lr: float,
        shape_steps: int,
    ) -> tuple:
        """Single ECME iteration for the multivariate skewed-t.

        Follows McNeil et al. (2005) Algorithm 3.14, adapted for the
        skewed-t special case (lamb = -nu/2, chi = nu, psi = 0).

        Notation follows the book (eq. 3.37):

        - delta_i = E[W_i^{-1} | X_i; theta^{[k]}]
        - eta_i   = E[W_i     | X_i; theta^{[k]}]

        Steps:

        (2) E-step — compute weights delta_i, eta_i from posterior
            W_i | X_i ~ GIG(-nu/2 - d/2, nu + Q_i, R_gamma)

        (3) Update gamma

        (4) Update mu, Psi, then Sigma with determinant constraint
            ``|Sigma| = |S|``

        (5)-(6) CM-step 2 — ECME: maximize observed log-likelihood
                w.r.t. nu via gradient descent

        Args:
            carry: Tuple of (nu, mu, gamma, sigma).
            _: Unused scan input.
            x: Data array of shape (n, d) (static).
            log_det_S: ``log|S|`` where ``S`` is the sample covariance (static).
            lr: Shape learning rate (static).
            shape_steps: Number of inner gradient steps (static).

        Returns:
            Updated carry and None (no stacked output).
        """
        eps: float = 1e-8
        nu, mu, gamma, sigma = carry
        n, d = x.shape[0], x.shape[1]

        # --- Step (2): E-step — posterior GIG expectations (eq. 3.36) ---
        # For skewed-t: W_i | X_i ~ GIG(-(nu+d)/2, nu + Q_i, R_gamma)
        sigma_inv: Array = jnp.linalg.inv(sigma)
        diff: Array = x - mu.flatten()  # (n, d)
        Q: Array = jnp.sum(diff @ sigma_inv * diff, axis=1)  # (n,)
        R: Scalar = (gamma.T @ sigma_inv @ gamma).squeeze()  # scalar

        lam_post: Scalar = -nu / 2.0 - d / 2.0
        chi_post: Array = nu + Q    # (n,)
        # Floor R at eps: when gamma≈0, R_γ=γ'Σ⁻¹γ≈0 which causes
        # log(chi/psi)→inf in _gig_expected_w.  The floor prevents
        # this singularity while having negligible effect on the
        # expectations when R is already positive.
        psi_post: Scalar = jnp.maximum(R, eps)  # scalar (psi=0 + R)

        # delta_i = E[1/W_i | X_i] (eq. 3.37)
        delta: Array = jnp.clip(
            GH._gig_expected_inv_w(lam_post, chi_post, psi_post), eps, 1e10
        )
        # eta_i = E[W_i | X_i] (eq. 3.37)
        eta: Array = jnp.clip(
            GH._gig_expected_w(lam_post, chi_post, psi_post), eps, 1e10
        )

        delta_bar: Scalar = jnp.mean(delta)
        eta_bar: Scalar = jnp.mean(eta)
        x_bar: Array = jnp.mean(x, axis=0).reshape((d, 1))

        # --- Step (3): gamma update (Algorithm 3.14, step 3) ---
        x_delta_bar: Array = jnp.mean(
            x * delta[:, None], axis=0
        ).reshape((d, 1))
        denom: Scalar = delta_bar * eta_bar - 1.0
        denom = jnp.where(jnp.abs(denom) < eps, eps, denom)
        gamma = (delta_bar * x_bar - x_delta_bar) / denom

        # --- Step (4): mu, Psi, Sigma update (Algorithm 3.14, step 4) ---
        mu = (x_delta_bar - gamma) / delta_bar

        diff = x - mu.flatten()  # (n, d) — recompute with updated mu
        psi_mat: Array = (
            jnp.mean(
                delta[:, None, None] * (diff[:, :, None] * diff[:, None, :]),
                axis=0,
            )
            - eta_bar * (gamma @ gamma.T)
        )

        # PSD repair, then determinant constraint
        psi_mat = _corr._rm_incomplete(psi_mat, 1e-5)

        # Determinant constraint: |Sigma| = |S| (identifiability)
        log_det_psi: Scalar = jnp.linalg.slogdet(psi_mat)[1]
        scale: Scalar = jnp.exp((log_det_S - log_det_psi) / d)
        sigma = scale * psi_mat

        # --- Steps (5)-(6): CM-step 2 — ECME variant ---
        # Maximize original log-likelihood w.r.t. nu only
        def _shape_step(shape_carry, _):
            n_val = shape_carry[0]
            _, g = MvtSkewedT._nll_nu_value_and_grad(
                n_val, mu, gamma, sigma, x
            )
            g = jnp.nan_to_num(g, nan=0.0)
            n_val = jnp.maximum(n_val - lr * g, eps)
            return (n_val,), None

        (nu,), _ = lax.scan(
            _shape_step, (nu,), None, length=shape_steps
        )

        return (nu, mu, gamma, sigma), None

    def _fit_em(
        self, x: jnp.ndarray, lr: float = 0.1, maxiter: int = 100
    ) -> dict:
        """Fit via ECME algorithm (McNeil et al. 2005, Algorithm 3.14).

        The EM algorithm treats the IG mixing variable W as latent data.
        Steps (3)-(4) update (gamma, mu, Sigma) in closed form from the
        expected sufficient statistics, with Sigma constrained so that
        ``|Sigma| = |S|`` for identifiability. Steps (5)-(6) use the
        ECME variant: maximize the observed log-likelihood w.r.t. nu
        via gradient descent.

        The entire loop is compiled via ``lax.scan`` for performance.

        Args:
            x: Input data array of shape (n, d).
            lr: Learning rate for nu gradient steps.
            maxiter: Number of EM iterations.

        Returns:
            Fitted parameter dictionary.
        """
        x, _, n, d = _multivariate_input(x)
        sample_mean: Array = jnp.mean(x, axis=0).reshape((d, 1))
        sample_cov: Array = cov(x=x, method="pearson")
        log_det_S: Scalar = jnp.linalg.slogdet(sample_cov)[1]

        # Step (1): starting values — MoM init for nu and gamma
        kappas = jnp.array(
            [kurtosis(x[:, j], fisher=True) for j in range(d)]
        )
        kappa = jnp.mean(kappas)
        nu0 = jnp.clip(4.0 + 6.0 / jnp.maximum(kappa, 0.06), 2.5, 100.0)

        # MoM init for gamma: marginal skewness direction
        x_std = jnp.std(x, axis=0)
        z = (x - jnp.mean(x, axis=0)) / jnp.where(x_std > 1e-8, x_std, 1.0)
        skew = jnp.mean(z ** 3, axis=0)
        gamma0 = (skew * x_std * 0.25).reshape((d, 1))

        init_carry: tuple = (
            nu0,                      # nu via MoM
            sample_mean,              # mu = X_bar
            gamma0,                   # gamma via MoM skewness
            sample_cov,               # sigma = S
        )

        shape_steps: int = 10
        em_step = lambda carry, _: self._em_body(
            carry, _, x, log_det_S, lr, shape_steps
        )
        final_carry, _ = lax.scan(em_step, init_carry, None, length=maxiter)
        nu, mu, gamma, sigma = final_carry

        return self._params_dict(nu=nu, mu=mu, gamma=gamma, sigma=sigma)

    _supported_methods = frozenset({"em", "ldmle"})

    def fit(
        self,
        x: ArrayLike,
        method: str = "em",
        cov_method: str = "pearson",
        lr: float = 0.1,
        maxiter: int = 100,
        name: str = None,
    ):
        r"""Fit the multivariate skewed-t distribution to data.

        Note:
            If you intend to jit wrap this function, ensure that
            ``method`` and ``cov_method`` are static arguments.

        Args:
            x: Input data of shape ``(n, d)``.
            method: Fitting method. One of:
                ``'em'`` — ECME algorithm (McNeil et al. 2005,
                Section 3.2.4); updates ``(mu, gamma, Sigma)`` in closed
                form via E-step sufficient statistics and ``nu`` via
                gradient descent; generally more robust and
                faster-converging than LDMLE (default);
                ``'ldmle'`` — low-dimensional MLE via projected ADAM
                gradient descent, optimising ``(nu, gamma)`` while
                deriving ``(mu, Sigma)`` analytically from sample
                moments.
            cov_method: Covariance estimator used for initialisation
                (both methods) and throughout the LDMLE path.
                Forwarded to :func:`copulax.multivariate.cov`.
            lr: Learning rate. Default ``0.1`` is tuned for EM; LDMLE
                may require a lower rate.
            maxiter: Maximum number of iterations.
            name: Optional custom name for the fitted instance.

        Returns:
            MvtSkewedT: A fitted ``MvtSkewedT`` instance.

        Raises:
            ValueError: If ``method`` is not one of the accepted
                strings listed above.
        """
        self._check_method(method)
        if method == "em":
            params = self._fit_em(x=x, lr=lr, maxiter=maxiter)
            return self._fitted_instance(params, name=name)
        x_arr, _, _, d = _multivariate_input(x)
        sample_mean, L = prepare_sample_cov(x_arr, cov_method)
        params = self._general_fit(
            x=x_arr, d=d, loc=sample_mean, shape=L, lr=lr, maxiter=maxiter,
        )
        return self._fitted_instance(params, name=name)

    # LDMLE fitting
    def _ldmle_inputs(self, d, x=None):
        """Generate initial parameter array and bounds for LD-MLE optimization.

        When data ``x`` is provided, nu is initialized from the average
        marginal excess kurtosis and gamma from the marginal skewness
        direction rather than random noise.
        """
        lc = jnp.full((d + 1, 1), -jnp.inf)
        uc = jnp.full((d + 1, 1), jnp.inf)

        # MoM init for nu: average marginal excess kurtosis. Clipped above
        # _NU_LDMLE_MIN so softplus inversion stays valid and the optimiser
        # starts in the Var[W] < infinity regime required by the moment-
        # matching reconstruction.
        nu_lower = _NU_LDMLE_MIN + 0.5
        if x is not None:
            kappas = jnp.array(
                [kurtosis(x[:, j], fisher=True) for j in range(d)]
            )
            kappa = jnp.mean(kappas)
            nu0 = jnp.clip(4.0 + 6.0 / jnp.maximum(kappa, 0.06), nu_lower, 100.0)
        else:
            nu0 = jnp.maximum(
                _NU_INIT + jnp.abs(random.normal(get_random_key())),
                nu_lower,
            )

        # nu = softplus(raw_nu) + _NU_LDMLE_MIN enforces nu > 4 strictly.
        raw_nu0 = jnp.log(jnp.expm1(nu0 - _NU_LDMLE_MIN))

        # MoM init for gamma: marginal skewness direction.
        if x is not None:
            x_std = jnp.std(x, axis=0)
            z_data = (x - jnp.mean(x, axis=0)) / jnp.where(x_std > 1e-8, x_std, 1.0)
            skew = jnp.mean(z_data ** 3, axis=0)
            gamma0 = skew * x_std * 0.25
            sample_cov0 = _corr._rm_incomplete(cov(x=x, method="pearson"), 1e-5)
        else:
            key = get_random_key()
            gamma0 = random.normal(key, (d,))
            sample_cov0 = jnp.eye(d)

        L0 = jnp.linalg.cholesky(sample_cov0)
        w_var0 = skewed_t._get_w_stats(nu=nu0)["variance"]
        z0 = invert_gamma_to_z(gamma0, L0, w_var0)

        params0 = jnp.array([raw_nu0, *z0]).flatten()
        return {"lower": lc, "upper": uc}, params0

    def _reconstruct_ldmle_params(self, params_arr, loc, shape):
        """Reconstruct nu, mu, gamma, sigma from LD-MLE optimizer output.

        ``shape`` is the Cholesky factor L of the PD-enforced sample covariance,
        precomputed once in ``fit``. γ is obtained via the feasibility
        reparametrisation γ = c(ν) · L · v with v = z/√(1+‖z‖²); by construction
        this keeps γᵀ Σ̂⁻¹ γ < 1/Var[W], so the reconstructed Σ is strictly PD
        and no silent repair is needed. Σ = L·(I − 0.9801·vvᵀ)·Lᵀ / E[W] is
        obtained after the Var[W] factor cancels inside the reconstruction.
        """
        L: Array = shape
        d: int = L.shape[0]
        nu_: Scalar = lax.dynamic_slice_in_dim(params_arr, 0, 1)
        nu: Scalar = (jnn.softplus(nu_) + _NU_LDMLE_MIN).flatten()
        z: Array = lax.dynamic_slice_in_dim(params_arr, 1, d)

        ig_stats = skewed_t._get_w_stats(nu=nu)
        gamma, sigma = forward_reparam(z, L, ig_stats["mean"], ig_stats["variance"])
        mu: Array = loc - ig_stats["mean"] * gamma
        return nu, mu, gamma, sigma



mvt_skewed_t = MvtSkewedT("Mvt-Skewed-T")
