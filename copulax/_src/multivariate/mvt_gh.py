"""File containing the copulAX implementation of the multivariate
generalized hyperbolic (GH) distribution."""

import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, random, jit, value_and_grad
from jax import Array
from jax.typing import ArrayLike
from jax.scipy import special

from copulax._src._distributions import NormalMixture
from copulax._src.special import log_kv
from copulax._src.typing import Scalar
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src._utils import _resolve_key, get_random_key
from copulax._src.multivariate._shape import cov, _corr
from copulax._src.multivariate._normal_mixture import (
    prepare_sample_cov,
    forward_reparam,
    invert_gamma_to_z,
)
from copulax._src.univariate.gig import gig
from copulax._src.univariate.gh import GH
from copulax.special import kv

_POS_EPS = 1e-8
_POS_INIT = 1.0


class MvtGH(NormalMixture):
    r"""The multivariate generalized hyperbolic (GH) distribution is a
    generalization of the univariate GH distribution to d > 1
    dimensions. This is a flexible, continuous 6-parameter family of
    distributions that can model a variety of data behaviors, including
    heavy tails and skewness. It contains a number of popular
    distributions as special cases, including the multivariate normal,
    multivariate student-t and multivariate skewed-T distributions.

    We adopt the parameterization used by McNeil et al. (2005)
    """

    lamb: Array = None
    chi: Array = None
    psi: Array = None
    mu: Array = None
    gamma: Array = None
    sigma: Array = None

    def __init__(
        self,
        name="Mvt-GH",
        *,
        lamb=None,
        chi=None,
        psi=None,
        mu=None,
        gamma=None,
        sigma=None,
    ):
        """Initialize with optional stored parameters."""
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
        self.mu = jnp.asarray(mu, dtype=float) if mu is not None else None
        self.gamma = jnp.asarray(gamma, dtype=float) if gamma is not None else None
        self.sigma = jnp.asarray(sigma, dtype=float) if sigma is not None else None

    @property
    def _stored_params(self):
        """Return stored parameters dict if all are set, else None."""
        if any(
            v is None
            for v in [self.lamb, self.chi, self.psi, self.mu, self.gamma, self.sigma]
        ):
            return None
        return {
            "lamb": self.lamb,
            "chi": self.chi,
            "psi": self.psi,
            "mu": self.mu,
            "gamma": self.gamma,
            "sigma": self.sigma,
        }

    def _classify_params(self, params: dict) -> tuple:
        """Classify parameters into scalar, vector, and shape groups."""
        # return (lamb, chi, psi,), (mu, gamma), (sigma,)
        return super()._classify_params(
            params=params,
            scalar_names=("lamb", "chi", "psi"),
            vector_names=("mu", "gamma"),
            shape_names=("sigma",),
            symmetric_shape_names=("sigma",),
        )

    def _params_dict(
        self,
        lamb: Scalar,
        chi: Scalar,
        psi: Scalar,
        mu: ArrayLike,
        gamma: ArrayLike,
        sigma: ArrayLike,
    ) -> dict:
        """Construct a normalized parameters dict from all six GH parameters."""
        d: dict = {
            "lamb": lamb,
            "chi": chi,
            "psi": psi,
            "mu": mu,
            "gamma": gamma,
            "sigma": sigma,
        }
        return self._args_transform(d)

    def _params_to_tuple(self, params: dict) -> tuple:
        """Extract `(lamb, chi, psi, mu, gamma, sigma)` from a params dict."""
        params = self._args_transform(params)
        return (
            params["lamb"],
            params["chi"],
            params["psi"],
            params["mu"],
            params["gamma"],
            params["sigma"],
        )

    def example_params(self, dim: int = 3, *args, **kwargs) -> dict:
        r"""Example parameters for the multivariate GH distribution.

        This is a six parameter family, defined by the scalar parameters
        `lamb`, `chi`, `psi`, the location vector `mu`, the
        skewness vector `gamma` and the shape matrix `sigma`.

        Args:
            dim: int, number of dimensions of the multivariate GH
                distribution. Default is 3.
        """
        return self._params_dict(
            lamb=0.0,
            chi=1.0,
            psi=1.0,
            mu=jnp.zeros((dim, 1)),
            gamma=jnp.zeros((dim, 1)),
            sigma=jnp.eye(dim, dim),
        )

    def support(self, params: dict = None) -> Array:
        """Return the support: `(-inf, inf)` per dimension."""
        return super().support(params=params)

    @staticmethod
    def _logpdf_core(
        stability: Scalar,
        x: Array,
        lamb: Scalar,
        chi: Scalar,
        psi: Scalar,
        mu: Array,
        gamma: Array,
        sigma: Array,
    ) -> Array:
        """Core log-PDF computation for the multivariate GH distribution.

        This is a static, pure function suitable for use inside
        ``value_and_grad``. Both the public ``_stable_logpdf`` and the
        ECME shape-parameter gradient call this.

        Args:
            stability: Small constant for numerical stability.
            x: Input data of shape (n, d).
            lamb: Shape parameter lamb.
            chi: Shape parameter chi.
            psi: Shape parameter psi.
            mu: Location vector of shape (d, 1).
            gamma: Skewness vector of shape (d, 1).
            sigma: Covariance matrix of shape (d, d).

        Returns:
            Array of log-density values with shape (n,).
        """
        d: int = x.shape[1]
        sigma_inv: Array = jnp.linalg.inv(sigma)
        diff: Array = x - mu.flatten()
        Q: Array = chi + jnp.sum(diff @ sigma_inv * diff, axis=1)
        R: Array = psi + (gamma.T @ sigma_inv @ gamma).squeeze()
        QR: Array = Q * R
        H: Array = ((x - mu.T) @ sigma_inv @ gamma).flatten()
        log_det_sigma: Scalar = jnp.linalg.slogdet(sigma)[1]
        s: Scalar = lamb - d / 2.0

        log_c: Scalar = (
            0.5 * lamb * lax.log((psi / (chi + stability)) + stability)
            - s * lax.log(R + stability)
            - 0.5 * d * lax.log(2 * jnp.pi)
            - 0.5 * log_det_sigma
            - log_kv(lamb, lax.sqrt(chi * psi))
        )

        logpdf: Array = (
            log_c
            + log_kv(s, lax.sqrt(QR))
            + H
            + 0.5 * s * lax.log(QR + stability)
        )
        return logpdf

    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, params: dict) -> Array:
        """Numerically stable log-PDF of the multivariate GH distribution.

        Args:
            stability: Small constant for numerical stability.
            x: Input data of shape (n, d).
            params: Distribution parameters.

        Returns:
            Array of log-density values with shape (n, 1).
        """
        x, yshape, n, d = _multivariate_input(x)
        lamb, chi, psi, mu, gamma, sigma = self._params_to_tuple(params)
        logpdf = MvtGH._logpdf_core(
            stability, x, lamb, chi, psi, mu, gamma, sigma
        )
        return logpdf.reshape(yshape)

    # sampling
    def rvs(self, size: int, params: dict = None, key: ArrayLike = None) -> Array:
        """Generate random samples via the GIG normal-variance mixture.

        Args:
            size: Number of samples to draw.
            params: Distribution parameters.
            key: JAX random key.

        Returns:
            Array of shape (size, d).
        """
        params = self._resolve_params(params)
        key = _resolve_key(key)
        lamb, chi, psi, mu, gamma, sigma = self._params_to_tuple(params)

        key, subkey = random.split(key)
        W: Array = gig.rvs(
            size=(size,), key=key, params={"lamb": lamb, "chi": chi, "psi": psi}
        )
        return super()._rvs(key=subkey, n=size, W=W, mu=mu, gamma=gamma, sigma=sigma)

    # stats
    def stats(self, params: dict = None) -> dict:
        """Compute distribution statistics using GIG mixing moments."""
        params = self._resolve_params(params)
        lamb, chi, psi, mu, gamma, sigma = self._params_to_tuple(params)
        gig_stats = gig.stats(params={"lamb": lamb, "chi": chi, "psi": psi})
        return self._stats(w_stats=gig_stats, mu=mu, gamma=gamma, sigma=sigma)

    # fitting — ECME algorithm (McNeil et al. 2005, Algorithm 3.14)
    @staticmethod
    @jit
    def _nll_shape_value_and_grad(
        shape_params: Array, mu: Array, gamma: Array, sigma: Array, x: Array
    ) -> tuple:
        """Compute NLL and gradient w.r.t. shape parameters [lamb, chi, psi].

        This implements the ECME variant of CM-step 2 from McNeil et al.
        (2005, p. 83): "instead of maximizing Q2 we may maximize the
        original likelihood (3.33) with respect to lamb, chi and psi
        with the other parameters held fixed."

        Args:
            shape_params: Array of shape (3,) containing [lamb, chi, psi].
            mu: Location vector of shape (d, 1).
            gamma: Skewness vector of shape (d, 1).
            sigma: Covariance matrix of shape (d, d).
            x: Data array of shape (n, d).

        Returns:
            Tuple of (nll_value, gradient) where gradient has shape (3,).
        """
        def _nll(sp, mu, gamma, sigma, x):
            lamb, chi, psi = sp
            logpdf = MvtGH._logpdf_core(1e-30, x, lamb, chi, psi, mu, gamma, sigma)
            return -jnp.mean(logpdf)

        return value_and_grad(_nll)(shape_params, mu, gamma, sigma, x)

    @staticmethod
    def _em_body(
        carry: tuple,
        _: None,
        x: Array,
        log_det_S: Scalar,
        lr: float,
        shape_steps: int,
    ) -> tuple:
        """Single ECME iteration following McNeil et al. Algorithm 3.14.

        Notation follows the book (eq. 3.37):

        - delta_i = E[W_i^{-1} | X_i; theta^{[k]}]
        - eta_i   = E[W_i     | X_i; theta^{[k]}]

        Steps:

        (2) E-step — compute weights delta_i, eta_i from posterior
            W_i | X_i ~ GIG(lamb - d/2, chi + Q_i, psi + R)

        (3) Update gamma (symmetric model: gamma = 0)

        (4) Update mu, Psi, then Sigma with determinant constraint
            |Sigma| = |S|

        (5)-(6) CM-step 2 — ECME: maximize observed log-likelihood
                w.r.t. (lamb, chi, psi) via gradient descent

        Args:
            carry: Tuple of (lamb, chi, psi, mu, gamma, sigma).
            _: Unused scan input.
            x: Data array of shape (n, d) (static).
            log_det_S: log|S| where S is the sample covariance (static).
            lr: Shape learning rate (static).
            shape_steps: Number of inner gradient steps (static).

        Returns:
            Updated carry and None (no stacked output).
        """
        eps: float = 1e-8
        lamb, chi, psi, mu, gamma, sigma = carry
        n, d = x.shape[0], x.shape[1]

        # --- Step (2): E-step — posterior GIG expectations (eq. 3.36) ---
        # W_i | X_i ~ GIG(lamb - d/2, chi + Q_i, psi + gamma' Sigma^{-1} gamma)
        sigma_inv: Array = jnp.linalg.inv(sigma)
        diff: Array = x - mu.flatten()  # (n, d)
        Q: Array = jnp.sum(diff @ sigma_inv * diff, axis=1)  # (n,)
        R: Scalar = (gamma.T @ sigma_inv @ gamma).squeeze()  # scalar

        lam_post: Scalar = lamb - d / 2.0
        chi_post: Array = chi + Q    # (n,)
        psi_post: Scalar = psi + R   # scalar

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
        # gamma = [n^{-1} sum delta_i (X_bar - X_i)] / (delta_bar * eta_bar - 1)
        x_delta_bar: Array = jnp.mean(
            x * delta[:, None], axis=0
        ).reshape((d, 1))
        denom: Scalar = delta_bar * eta_bar - 1.0
        denom = jnp.where(jnp.abs(denom) < eps, eps, denom)
        gamma = (delta_bar * x_bar - x_delta_bar) / denom

        # --- Step (4): mu, Psi, Sigma update (Algorithm 3.14, step 4) ---
        # mu = (n^{-1} sum delta_i X_i - gamma) / delta_bar
        mu = (x_delta_bar - gamma) / delta_bar

        # Psi = (1/n) sum delta_i (X_i - mu)(X_i - mu)' - eta_bar * gamma gamma'
        diff = x - mu.flatten()  # (n, d) — recompute with updated mu
        psi_mat: Array = (
            jnp.mean(
                delta[:, None, None] * (diff[:, :, None] * diff[:, None, :]),
                axis=0,
            )
            - eta_bar * (gamma @ gamma.T)
        )

        # PSD repair first, then determinant constraint (order matters:
        # _rm_incomplete changes eigenvalues which changes the determinant,
        # so we must apply it before the rescaling, not after).
        psi_mat = _corr._rm_incomplete(psi_mat, 1e-5)

        # Determinant constraint: |Sigma| = |S| (identifiability, McNeil p. 82)
        # Sigma = |S|^{1/d} * Psi / |Psi|^{1/d}
        log_det_psi: Scalar = jnp.linalg.slogdet(psi_mat)[1]
        scale: Scalar = jnp.exp((log_det_S - log_det_psi) / d)
        sigma = scale * psi_mat

        # --- Steps (5)-(6): CM-step 2 — ECME variant (McNeil p. 83) ---
        # Maximize original log-likelihood w.r.t. (lamb, chi, psi)
        # with (mu, gamma, Sigma) held fixed.
        def _shape_step(shape_carry, _):
            l, c, p = shape_carry
            _, g = MvtGH._nll_shape_value_and_grad(
                jnp.array([l, c, p]), mu, gamma, sigma, x
            )
            g = jnp.nan_to_num(g, nan=0.0)
            l = l - lr * g[0]
            c = jnp.maximum(c - lr * g[1], eps)
            p = jnp.maximum(p - lr * g[2], eps)
            return (l, c, p), None

        (lamb, chi, psi), _ = lax.scan(
            _shape_step, (lamb, chi, psi), None, length=shape_steps
        )

        return (lamb, chi, psi, mu, gamma, sigma), None

    def _fit_em(
        self, x: jnp.ndarray, lr: float = 0.1, maxiter: int = 100
    ) -> dict:
        """Fit via ECME algorithm (McNeil et al. 2005, Algorithm 3.14).

        The EM algorithm treats the GIG mixing variable W as latent data.
        Steps (3)-(4) update (gamma, mu, Sigma) in closed form from the
        expected sufficient statistics, with Sigma constrained so that
        |Sigma| = |S| for identifiability. Steps (5)-(6) use the ECME
        variant: maximize the observed log-likelihood w.r.t. (lamb,
        chi, psi) via gradient descent.

        The entire loop is compiled via ``lax.scan`` for performance.

        Args:
            x: Input data array of shape (n, d).
            lr: Learning rate for shape parameter gradient steps.
            maxiter: Number of EM iterations.

        Returns:
            Fitted parameter dictionary.
        """
        x, _, n, d = _multivariate_input(x)
        sample_mean: Array = jnp.mean(x, axis=0).reshape((d, 1))
        sample_cov: Array = cov(x=x, method="pearson")
        log_det_S: Scalar = jnp.linalg.slogdet(sample_cov)[1]

        # Step (1): starting values (Algorithm 3.14, step 1)
        init_carry: tuple = (
            jnp.array(0.0),       # lamb
            jnp.array(1.0),       # chi
            jnp.array(1.0),       # psi
            sample_mean,          # mu = X_bar
            jnp.zeros((d, 1)),    # gamma = 0
            sample_cov,           # sigma = S
        )

        shape_steps: int = 10
        em_step = lambda carry, _: self._em_body(
            carry, _, x, log_det_S, lr, shape_steps
        )
        final_carry, _ = lax.scan(em_step, init_carry, None, length=maxiter)
        lamb, chi, psi, mu, gamma, sigma = final_carry

        return self._params_dict(
            lamb=lamb, chi=chi, psi=psi, mu=mu, gamma=gamma, sigma=sigma,
        )

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
        r"""Fit the multivariate GH distribution to data.

        Note:
            If you intend to jit wrap this function, ensure that
            ``method`` and ``cov_method`` are static arguments.

        Args:
            x: Input data of shape ``(n, d)``.
            method: Fitting method. One of:
                ``'em'`` — ECME algorithm (McNeil et al. 2005,
                Section 3.4.2); updates ``(mu, gamma, Sigma)`` in closed
                form via E-step sufficient statistics and
                ``(lamb, chi, psi)`` via gradient descent; generally more
                robust and faster-converging than LDMLE (default);
                ``'ldmle'`` — low-dimensional MLE via projected ADAM
                gradient descent, optimising ``(lamb, chi, psi, gamma)``
                while deriving ``(mu, Sigma)`` analytically from sample
                moments.
            cov_method: Covariance estimator used for initialisation
                (both methods) and throughout the LDMLE path.
                Forwarded to :func:`copulax.multivariate.cov`.
            lr: Learning rate. Default ``0.1`` is tuned for EM; LDMLE
                may require a lower rate.
            maxiter: Maximum number of iterations.
            name: Optional custom name for the fitted instance.

        Returns:
            MvtGH: A fitted ``MvtGH`` instance.

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

    def _ldmle_inputs(self, d, x=None):
        """Generate initial parameter array and bounds for LD-MLE optimization.

        When data ``x`` is provided, gamma is initialized from the marginal
        sample skewness direction rather than random noise. The params slot
        for gamma stores the unconstrained ``z`` vector that drives the
        feasibility reparametrisation; the init inverts ``gamma0`` through
        the same map.
        """
        lc = jnp.full((d + 3, 1), -jnp.inf)
        uc = jnp.full((d + 3, 1), jnp.inf)

        key1, key2 = random.split(get_random_key())
        key2, key3 = random.split(key2)
        pos0 = _POS_INIT + jnp.abs(random.normal(key2, (2,)))
        pos0_raw = jnp.log(jnp.expm1(pos0))
        lamb0 = random.normal(key1)

        if x is not None:
            x_std = jnp.std(x, axis=0)
            z_data = (x - jnp.mean(x, axis=0)) / jnp.where(x_std > 1e-8, x_std, 1.0)
            skew = jnp.mean(z_data ** 3, axis=0)
            gamma0 = skew * x_std * 0.25
            sample_cov0 = _corr._rm_incomplete(cov(x=x, method="pearson"), 1e-5)
        else:
            gamma0 = random.normal(key3, (d,))
            sample_cov0 = jnp.eye(d)

        L0 = jnp.linalg.cholesky(sample_cov0)
        chi0 = jnn.softplus(pos0_raw[0]) + _POS_EPS
        psi0 = jnn.softplus(pos0_raw[1]) + _POS_EPS
        w_var0 = gig.stats(
            params={"lamb": lamb0, "chi": chi0, "psi": psi0}
        )["variance"]
        z0 = invert_gamma_to_z(gamma0, L0, w_var0)

        params0 = jnp.array([lamb0, *pos0_raw, *z0]).flatten()
        return {"lower": lc, "upper": uc}, params0

    def _reconstruct_ldmle_params(self, params_arr, loc, shape):
        """Reconstruct lamb, chi, psi, mu, gamma, sigma from LD-MLE output.

        ``shape`` is ``L = chol(sample_cov_pd)``, precomputed in ``fit``.
        gamma is obtained via the feasibility reparametrisation so the
        reconstructed sigma is strictly PD by construction; no silent repair
        and no per-step matrix decomposition.
        """
        L: Array = shape
        d: int = L.shape[0]
        scalars = lax.dynamic_slice_in_dim(params_arr, 0, 3)
        lamb, chi_, psi_ = scalars
        chi = jnn.softplus(chi_) + _POS_EPS
        psi = jnn.softplus(psi_) + _POS_EPS
        z: Array = lax.dynamic_slice_in_dim(params_arr, 3, d)

        gig_stats: dict = gig.stats(params={"lamb": lamb, "chi": chi, "psi": psi})
        gamma, sigma = forward_reparam(
            z, L, gig_stats["mean"], gig_stats["variance"]
        )
        mu: Array = loc - gig_stats["mean"] * gamma
        return lamb, chi, psi, mu, gamma, sigma



mvt_gh = MvtGH("Mvt-GH")
