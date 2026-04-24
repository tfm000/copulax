"""CopulAX implementation of the popular Copula distributions."""

from abc import abstractmethod
from jax import Array
from jax.typing import ArrayLike
from typing import Callable
import jax
from jax import numpy as jnp
from jax import jit, vmap, lax
import jax.nn as jnn

from copulax._src._distributions import (
    GeneralMultivariate,
    Multivariate,
    Univariate,
)
from copulax.univariate import univariate_fitter
from copulax._src.univariate.univariate_fitter import batch_univariate_fitter
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src._utils import _resolve_key
from copulax._src.typing import Scalar
from copulax._src.multivariate._shape import corr, _corr
from copulax._src.optimize import projected_gradient, adam
from copulax._src.univariate._utils import _univariate_input
from functools import partial

from copulax._src.multivariate.mvt_normal import mvt_normal
from copulax._src.univariate.normal import normal
from copulax._src.multivariate.mvt_student_t import mvt_student_t, MvtStudentT
from copulax._src.univariate.student_t import student_t
from copulax._src.multivariate.mvt_gh import mvt_gh, MvtGH
from copulax._src.univariate.gh import gh, GH
from copulax._src.multivariate.mvt_skewed_t import mvt_skewed_t, MvtSkewedT
from copulax._src.univariate.skewed_t import skewed_t
from copulax._src.copulas._mom_init import mom_nu_student_t, mom_gh_params
from collections import defaultdict

# Module-level constants for copula parameter constraints
_NU_EPS: float = 1e-6
_POS_EPS: float = 1e-8

# Fitting constants
_GRAD_CLIP: float = 10.0
_EPS: float = 1e-8

# Per-method accepted kwargs for ``MeanVarianceCopulaBase.fit_copula``.
# Used to fail fast on inapplicable kwargs (e.g. passing ``tol`` with
# ``method='fc_mle'``) instead of silently dropping them.  ``corr_method``
# is accepted by every method (Stage 1 correlation estimator).
_METHOD_KWARGS: dict[str, frozenset[str]] = {
    "fc_mle":            frozenset({"lr", "maxiter"}),
    "mle":               frozenset({"lr", "maxiter", "tol", "patience", "brent", "nodes"}),
    "ecme":              frozenset({"lr", "maxiter", "tol", "patience", "brent", "nodes"}),
    "ecme_double_gamma": frozenset({"lr", "maxiter", "tol", "patience", "brent", "nodes"}),
    "ecme_outer_gamma":  frozenset({"lr", "maxiter", "tol", "patience", "brent", "nodes"}),
}


def _inv_softplus(x: jnp.ndarray) -> jnp.ndarray:
    r"""Numerically stable inverse of ``jax.nn.softplus``.

    For large x, ``softplus(x) ≈ x`` so ``inv_softplus(x) ≈ x``.
    For small x, ``inv_softplus(x) = log(expm1(x))``.  The crossover
    at x=20 avoids float32 overflow in ``expm1``.

    Args:
        x: Input array (positive values).

    Returns:
        Array y such that ``softplus(y) ≈ x``.
    """
    return jnp.where(x > 20.0, x, jnp.log(jnp.expm1(jnp.minimum(x, 20.0))))


###############################################################################
# Shared copula fitting helpers
###############################################################################
def _reset_adam_state(
    adam_state: tuple[jnp.ndarray, jnp.ndarray, int],
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    r"""Fully reset Adam first/second moment and step counter to zero.

    Used between outer iterations of the copula EM/MLE fitting loops.
    Each outer iteration is treated as a fresh subproblem starting from
    the EM-warm-started parameters: any momentum carried from the
    previous outer iteration was computed under a different
    (gamma, sigma, x') configuration and is therefore stale.  Even
    parameters that did not move directly (e.g. nu) live on a loss
    surface that has shifted, so their stored gradient direction is no
    longer reliable.  A full reset removes the staleness uniformly.

    Args:
        adam_state: Tuple ``(m, v, t)`` — first moment, second moment,
            step counter.

    Returns:
        Zeroed Adam state ``(0, 0, 0)`` with the same shapes/dtypes
        as the input.
    """
    m, v, _ = adam_state
    return (jnp.zeros_like(m), jnp.zeros_like(v), jnp.array(0))


def _adam_gradient_step(
    nll_fn: Callable,
    opt_arr: jnp.ndarray,
    adam_state: tuple[jnp.ndarray, jnp.ndarray, int],
    lr: float,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, int]]:
    r"""Compute NLL gradient, clip, and apply one Adam update.

    This is a plain Python function (not JIT-decorated) that is called
    inside JIT-compiled closures.  JAX traces through it during
    compilation, so all operations must be JAX-traceable.

    Args:
        nll_fn: Scalar-valued negative log-likelihood function of
            ``opt_arr``.
        opt_arr: Current parameter vector.
        adam_state: Tuple ``(m, v, t)``.
        lr: Learning rate.

    Returns:
        Tuple of (updated opt_arr, new adam_state).
    """
    _, grad = jax.value_and_grad(nll_fn)(opt_arr)
    grad = jnp.nan_to_num(grad, nan=0.0)
    grad = jnp.clip(grad, -_GRAD_CLIP, _GRAD_CLIP)
    m, v, t = adam_state
    direction, m, v, t = adam(grad, m, v, t)
    return opt_arr - lr * direction, (m, v, t)


def _skewed_t_gig_posteriors(
    nu: jnp.ndarray,
    sigma_inv: jnp.ndarray,
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    eps: float = _EPS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""GIG posterior parameters for the Skewed-T copula.

    For the Skewed-T distribution,
    :math:`W_i | X_i \sim \text{GIG}(-(\nu+d)/2, \nu + Q_i, R_\gamma)`.

    Args:
        nu: Degrees of freedom (scalar).
        sigma_inv: Inverse correlation matrix, shape ``(d, d)``.
        x: Centred data (mu=0), shape ``(n, d)``.
        gamma: Skewness vector, shape ``(d, 1)``.
        eps: Floor for psi_post.

    Returns:
        ``(lam_post, chi_post, psi_post)`` — per-sample GIG parameters.
    """
    d = x.shape[1]
    Q = jnp.sum((x @ sigma_inv) * x, axis=1)
    R = (gamma.T @ sigma_inv @ gamma).squeeze()
    lam_post = -nu / 2.0 - d / 2.0
    chi_post = nu + Q
    psi_post = jnp.maximum(R, eps)
    return lam_post, chi_post, psi_post


def _gh_gig_posteriors(
    lamb: jnp.ndarray,
    chi: jnp.ndarray,
    psi: jnp.ndarray,
    sigma_inv: jnp.ndarray,
    x: jnp.ndarray,
    gamma: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""GIG posterior parameters for the GH copula.

    For the GH distribution,
    :math:`W_i | X_i \sim \text{GIG}(\lambda - d/2,
    \chi + Q_i, \psi + R_\gamma)`.

    Args:
        lamb: GH lamb parameter (scalar).
        chi: GH chi parameter (scalar).
        psi: GH psi parameter (scalar).
        sigma_inv: Inverse correlation matrix, shape ``(d, d)``.
        x: Centred data (mu=0), shape ``(n, d)``.
        gamma: Skewness vector, shape ``(d, 1)``.

    Returns:
        ``(lam_post, chi_post, psi_post)`` — per-sample GIG parameters.
    """
    d = x.shape[1]
    Q = jnp.sum((x @ sigma_inv) * x, axis=1)
    R = (gamma.T @ sigma_inv @ gamma).squeeze()
    lam_post = lamb - d / 2.0
    chi_post = chi + Q
    psi_post = psi + R
    return lam_post, chi_post, psi_post


@partial(jax.jit, static_argnames=("update_gamma",))
def _copula_inner_em_body(
    gamma: jnp.ndarray,
    sigma: jnp.ndarray,
    x: jnp.ndarray,
    lam_post: jnp.ndarray,
    chi_post: jnp.ndarray,
    psi_post: jnp.ndarray,
    update_gamma: bool,
    eps: float = _EPS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Shared inner EM body for copula fitting.

    Computes GIG posterior expectations, optionally updates gamma
    (mu=0 constraint), and updates the correlation matrix sigma.

    Args:
        gamma: Skewness vector, shape ``(d, 1)``.
        sigma: Current correlation matrix, shape ``(d, d)``.
        x: Centred data (mu=0), shape ``(n, d)``.
        lam_post: GIG lamb posterior, shape ``(n,)``.
        chi_post: GIG chi posterior, shape ``(n,)``.
        psi_post: GIG psi posterior, shape ``(n,)`` or scalar.
        update_gamma: Whether to update gamma (True for em/em2,
            False for em3).
        eps: Numerical stability constant.

    Returns:
        ``(gamma_new, sigma_new)``.
    """
    d = x.shape[1]

    delta = jnp.clip(
        GH._gig_expected_inv_w(lam_post, chi_post, psi_post), eps, 1e10
    )
    eta = jnp.clip(
        GH._gig_expected_w(lam_post, chi_post, psi_post), eps, 1e10
    )

    eta_bar = jnp.mean(eta)

    if update_gamma:
        x_bar = jnp.mean(x, axis=0).reshape((d, 1))
        eta_bar_safe = jnp.maximum(eta_bar, eps)
        gamma = jnp.clip(x_bar / eta_bar_safe, -10.0, 10.0)

    psi_mat = (
        jnp.mean(
            delta[:, None, None] * (x[:, :, None] * x[:, None, :]),
            axis=0,
        )
        - eta_bar * (gamma @ gamma.T)
    )
    psi_mat = _corr._rm_incomplete(psi_mat, 1e-5)
    sigma = _corr._corr_from_cov(psi_mat)
    sigma = _corr._ensure_valid(sigma)

    return gamma, sigma


@partial(jax.jit, static_argnames=("update_gamma",))
def _inner_em_step_skewed_t(
    gamma: jnp.ndarray,
    sigma: jnp.ndarray,
    x: jnp.ndarray,
    nu: jnp.ndarray,
    update_gamma: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Inner EM step for the Skewed-T copula.

    Computes GIG posterior parameters for the Skewed-T family,
    then delegates to the shared EM body.

    Args:
        gamma: Skewness vector, shape ``(d, 1)``.
        sigma: Correlation matrix, shape ``(d, d)``.
        x: Centred data (mu=0), shape ``(n, d)``.
        nu: Degrees of freedom (scalar).
        update_gamma: Whether to update gamma.

    Returns:
        ``(gamma_new, sigma_new)``.
    """
    sigma_inv = jnp.linalg.inv(sigma)
    lp, cp, pp = _skewed_t_gig_posteriors(nu, sigma_inv, x, gamma)
    return _copula_inner_em_body(gamma, sigma, x, lp, cp, pp, update_gamma)


@partial(jax.jit, static_argnames=("update_gamma",))
def _inner_em_step_gh(
    gamma: jnp.ndarray,
    sigma: jnp.ndarray,
    x: jnp.ndarray,
    lamb: jnp.ndarray,
    chi: jnp.ndarray,
    psi: jnp.ndarray,
    update_gamma: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Inner EM step for the GH copula.

    Computes GIG posterior parameters for the GH family,
    then delegates to the shared EM body.

    Args:
        gamma: Skewness vector, shape ``(d, 1)``.
        sigma: Correlation matrix, shape ``(d, d)``.
        x: Centred data (mu=0), shape ``(n, d)``.
        lamb: GH lamb (scalar).
        chi: GH chi (scalar).
        psi: GH psi (scalar).
        update_gamma: Whether to update gamma.

    Returns:
        ``(gamma_new, sigma_new)``.
    """
    sigma_inv = jnp.linalg.inv(sigma)
    lp, cp, pp = _gh_gig_posteriors(lamb, chi, psi, sigma_inv, x, gamma)
    return _copula_inner_em_body(gamma, sigma, x, lp, cp, pp, update_gamma)


###############################################################################
# CopulaBase — shared logic for elliptical and Archimedean copulas
###############################################################################
class CopulaBase(GeneralMultivariate):
    r"""Base class for all copula distributions.

    Provides Sklar's theorem implementations for the joint distribution,
    common marginal-fitting logic, and sampling via inverse-transform
    of copula samples.
    """

    _marginals: tuple = None
    _copula_params: dict = None

    @property
    def _stored_params(self):
        """Return stored parameters dict if marginals and copula are set."""
        if self._marginals is None or self._copula_params is None:
            return None
        return {"marginals": self._marginals, "copula": self._copula_params}

    @property
    def dist_type(self) -> str:
        """Distribution family type identifier."""
        return "copula"

    def _get_dim(self, params: dict) -> int:
        """Infer dimensionality from the number of marginal distributions."""
        return len(params["marginals"])

    def support(self, params: dict = None) -> Array:
        r"""Support of the joint distribution."""
        params = self._resolve_params(params)
        marginals: tuple = params["marginals"]
        return jnp.vstack([dist.support(params=mparams) for dist, mparams in marginals])

    @staticmethod
    def _grouped_marginal_apply(func_name, x_arr, marginals, **func_kwargs):
        """Apply a univariate function across dimensions, vmapping over
        groups that share the same distribution type for efficiency.

        Args:
            func_name: Name of the univariate method to call (e.g. 'cdf').
            x_arr: Input data of shape (n, d).
            marginals: Tuple of (distribution, params) per dimension.
            **func_kwargs: Extra keyword arguments forwarded to each call.

        Returns:
            Array of shape (n, d) with the function evaluated per column.
        """
        d = len(marginals)
        groups = defaultdict(list)
        for i, (dist, mparams) in enumerate(marginals):
            groups[dist.name].append((i, mparams))

        out = None
        for _, items in groups.items():
            dim_indices = [item[0] for item in items]
            idx_arr = jnp.asarray(dim_indices, dtype=int)
            param_dicts = [item[1] for item in items]
            dist = marginals[dim_indices[0]][0]
            func = getattr(dist, func_name)

            batched_params = {
                k: jnp.stack([p[k] for p in param_dicts]) for k in param_dicts[0].keys()
            }
            x_group = x_arr[:, idx_arr]

            def _apply(xi_col, p, _f=func):
                return _f(xi_col, params=p, **func_kwargs)

            result = vmap(_apply, in_axes=(1, 0), out_axes=1)(x_group, batched_params)
            if out is None:
                out = jnp.empty((x_arr.shape[0], d), dtype=result.dtype)
            out = out.at[:, idx_arr].set(result)

        if out is None:
            return jnp.empty((x_arr.shape[0], 0), dtype=x_arr.dtype)
        return out

    def get_u(self, x: ArrayLike, params: dict = None) -> Array:
        r"""Compute marginal CDF values u = (F_1(x_1), ..., F_d(x_d)).

        Args:
            x: Input data of shape (n, d).
            params: Distribution parameters with 'marginals' key.

        Returns:
            Array of shape (n, d) with values in [0, 1].
        """
        x_arr: jnp.ndarray = _multivariate_input(x)[0]
        params = self._resolve_params(params)
        return self._grouped_marginal_apply("cdf", x_arr, params["marginals"])

    # --- copula densities (abstract) ---

    @abstractmethod
    def copula_logpdf(self, u: ArrayLike, params: dict = None, **kwargs) -> Array:
        r"""Log-density of the copula (subclasses must implement)."""

    def copula_pdf(self, u: ArrayLike, params: dict = None, **kwargs) -> Array:
        r"""Density of the copula: c(u) = exp(copula_logpdf(u))."""
        return jnp.exp(self.copula_logpdf(u, params, **kwargs))

    @abstractmethod
    def copula_rvs(self, size: Scalar, params: dict, key: Array = None) -> Array:
        r"""Generate random samples from the copula (subclasses must implement)."""

    def copula_sample(
        self, size: Scalar, params: dict = None, key: Array = None
    ) -> Array:
        r"""Alias for copula_rvs."""
        return self.copula_rvs(size=size, params=params, key=key)

    # --- joint distribution (Sklar's theorem) ---

    def logpdf(self, x: ArrayLike, params: dict = None, **kwargs) -> Array:
        r"""Joint log-PDF via Sklar's theorem.

        log f(x) = log c(F_1(x_1),...,F_d(x_d)) + sum log f_i(x_i)

        Args:
            x: Input data of shape (n, d).
            params: Distribution parameters with 'marginals' and
                'copula' keys.

        Returns:
            Array of shape (n, 1).
        """
        x_arr, _, n, d = _multivariate_input(x)
        params = self._resolve_params(params)
        marginal_logpdf_sum: jnp.ndarray = self._grouped_marginal_apply(
            "logpdf", x_arr, params["marginals"]
        ).sum(axis=1, keepdims=True)
        u: jnp.ndarray = self.get_u(x_arr, params)
        copula_lp: jnp.ndarray = self.copula_logpdf(u, params, **kwargs)
        return copula_lp + marginal_logpdf_sum

    def pdf(self, x: ArrayLike, params: dict = None, **kwargs) -> Array:
        r"""Joint PDF."""
        return jnp.exp(self.logpdf(x, params, **kwargs))

    # --- sampling ---

    def rvs(
        self,
        size: Scalar,
        params: dict = None,
        key: Array = None,
        brent: bool = False,
        nodes: int = 100,
    ) -> Array:
        r"""Sample from the joint distribution.

        1. Sample u from copula
        2. Transform u to x via marginal PPFs

        Args:
            size: Number of samples.
            params: Distribution parameters.
            key: JAX random key.
            brent: Forwarded to the marginal :py:meth:`Univariate.ppf`.
                ``False`` (default) uses the analytical inverse CDF
                when available and otherwise the Chebyshev cubic
                spline; ``True`` forces per-quantile Brent
                root-finding (slower but machine-epsilon accurate).
            nodes: Number of Chebyshev-Lobatto nodes when the cubic
                path is used.  Ignored for analytical marginals and
                when ``brent=True``.

        Returns:
            Array of shape (size, d).
        """
        key = _resolve_key(key)
        params = self._resolve_params(params)
        u_raw: jnp.ndarray = self.copula_rvs(size=size, params=params, key=key)
        eps: float = 1e-4
        u: jnp.ndarray = jnp.clip(u_raw, eps, 1 - eps)
        return self._grouped_marginal_apply(
            "ppf", u, params["marginals"], brent=brent, nodes=nodes
        )

    # --- fitting ---

    def fit_marginals(
        self,
        x: ArrayLike,
        univariate_fitter_options: tuple[dict] | dict = None,
    ) -> dict:
        r"""Fit univariate marginal distributions to each dimension.

        Args:
            x: Input data of shape (n, d).
            univariate_fitter_options: Options for the univariate
                fitter. Dict applies same options to all dimensions;
                tuple of dicts applies per-dimension options.

        Note:
            Not jitable.

        Returns:
            dict with key 'marginals' containing fitted distributions.
        """
        x_arr, _, n, d = _multivariate_input(x)
        if univariate_fitter_options is None:
            univariate_fitter_options = ({},) * d
        elif isinstance(univariate_fitter_options, dict):
            univariate_fitter_options = (univariate_fitter_options,) * d
        elif isinstance(univariate_fitter_options, tuple):
            if len(univariate_fitter_options) != d:
                raise ValueError(
                    "univariate_fitter_options tuple must have "
                    "an entry for each variable in x."
                )
        else:
            raise ValueError("univariate_fitter_options must be a tuple or dictionary.")

        # Group dimensions by options for batched fitting
        groups: dict[str, list[int]] = defaultdict(list)
        for i, opts in enumerate(univariate_fitter_options):
            key = str(sorted(opts.items())) if opts else ""
            groups[key].append(i)

        marginals: list = [None] * d
        for key, dim_indices in groups.items():
            opts = univariate_fitter_options[dim_indices[0]]
            x_batch = x_arr[:, jnp.array(dim_indices)]
            batch_results = batch_univariate_fitter(x_batch, **opts)
            for j, (best_index, fitted) in enumerate(batch_results):
                dist: Univariate = fitted[best_index]["dist"]
                params: dict = fitted[best_index]["params"]
                marginals[dim_indices[j]] = (dist, params)

        return {"marginals": tuple(marginals)}

    @abstractmethod
    def fit_copula(self, u: ArrayLike, **kwargs) -> dict:
        r"""Fit copula parameters (subclasses must implement)."""

    def fit(
        self,
        x: ArrayLike,
        univariate_fitter_options: tuple[dict] | dict = None,
        name: str = None,
        **kwargs,
    ) -> dict:
        r"""Fit marginals and copula to the data.

        Equivalent to calling fit_marginals then fit_copula.

        Args:
            x: Input data of shape (n, d).
            univariate_fitter_options: Options for marginal fitting.
            name: Optional custom name for the fitted instance.
            **kwargs: Additional arguments forwarded to ``fit_copula``
                (``method``, ``lr``, ``maxiter``, ``tol``, ``patience``).

        Note:
            Not jitable.

        Returns:
            dict with keys 'marginals' and 'copula'.
        """
        marginals: dict = self.fit_marginals(x, univariate_fitter_options)
        u: jnp.ndarray = self.get_u(x, marginals)
        copula: dict = self.fit_copula(u, **kwargs)
        params = {**marginals, **copula}
        return self._fitted_instance(params, name=name)


###############################################################################
# Mean-Variance Copula Base Hierarchy
###############################################################################
class MeanVarianceCopulaBase(CopulaBase):
    r"""Umbrella base class for normal-mixture copula distributions.

    Holds the shared ``fit_copula`` dispatcher, ``_METHOD_KWARGS``
    validation, correlation estimation, and other machinery common to
    both normal *variance* mixture copulas (true elliptical, γ=0; see
    :class:`EllipticalCopula`) and normal *mean-variance* mixture
    copulas (with skewness γ; see :class:`MeanVarianceCopula`).

    Reference:
        McNeil, Frey, Embrechts (2005) *Quantitative Risk Management*,
        §3.2.1 (variance mixtures) and §3.2.2 (mean-variance mixtures).
    """

    # Concrete sub-bases override.  The umbrella alone supports nothing.
    _supported_methods: frozenset = frozenset()

    _mvt: Multivariate
    _uvt: Univariate

    # initialisation
    def __init__(
        self,
        name,
        mvt: Multivariate,
        uvt: Univariate,
        *,
        marginals=None,
        copula=None,
    ):
        super().__init__(name)
        self._mvt: Multivariate = mvt  # multivariate pytree object
        self._uvt: Univariate = uvt  # univariate pytree object
        self._marginals = marginals if marginals is not None else None
        self._copula_params = copula if copula is not None else None

    def _fitted_instance(self, params_dict: dict, name: str = None):
        """Create a fitted Copula instance (passes mvt/uvt positional args).

        Args:
            params_dict: Fitted parameter values.
            name: Optional custom name for the fitted instance. If ``None``,
                an auto-generated name is used.

        Returns:
            A new Copula instance with the given parameters.
        """
        cls = type(self)
        if name is None:
            name = f"Fitted{cls.__name__}-{id(params_dict):x}"
        return cls(name, self._mvt, self._uvt, **params_dict)

    def _params_to_tuple(self, params: dict) -> tuple:
        """Return an empty tuple (elliptical copula params held in dict)."""
        return tuple()

    def example_params(self, dim: int = 3, *args, **kwargs):
        r"""Example parameters for the copula distribution.

        Generates example marginal and copula parameters for the overall
        joint distribution.

        Args:
            dim: int, number of dimensions of the copula distribution.
                Default is 3.
        """
        # copula parameters
        mvt_params: dict = self._mvt.example_params(dim=dim, *args, **kwargs)
        mvt_params["sigma"] = jnp.eye(dim, dim)

        # marginal parameters
        marginal_params: tuple = tuple(
            (self._uvt, self._uvt.example_params(dim=dim)) for _ in range(dim)
        )

        # joint parameters
        return {"marginals": marginal_params, "copula": mvt_params}

    def _get_uvt_params(self, params: dict) -> tuple:
        """Returns the univariate distribution parameters."""
        return tuple()

    def _scan_uvt_func(self, func: Callable, x: Array, params: dict, **kwargs) -> Array:
        """Applies func per dimension, vectorized with vmap."""
        batched_params: dict = self._get_uvt_params(params)

        def _per_dim(xi_col, p_slice):
            return func(xi_col, params=p_slice, **kwargs)

        return vmap(_per_dim, in_axes=(1, 0), out_axes=1)(x, batched_params)

    def get_x_dash(
        self,
        u: ArrayLike,
        params: dict,
        brent: bool = False,
        nodes: int = 100,
    ) -> Array:
        r"""Computes x' values, which represent the mappings of the
        independent marginal cdf values (U) to the domain of the joint
        multivariate distribution.

        Routes through :py:meth:`Univariate.ppf`, so distributions
        with an analytical inverse CDF (Normal, Gamma, LogNormal, IG,
        Uniform, Gen-Normal) use the closed-form path automatically
        and ignore ``nodes``.

        Note:
            If you intend to jit wrap this function, both ``brent``
            and ``nodes`` must be static arguments.

        Args:
            u (ArrayLike): The independent univariate marginal cdf
                values (U) for each dimension, shape ``(n, d)``.
            params (dict): The copula and marginal distribution
                parameters.
            brent (bool): If ``False`` (default), use the analytical
                inverse CDF when available and otherwise the
                Chebyshev-node cubic spline approximation.  If
                ``True``, force per-quantile Brent root-finding
                (machine-epsilon accurate, slower).
            nodes (int): Number of Chebyshev-Lobatto nodes used by the
                cubic spline path.  Ignored for analytical marginals
                and when ``brent=True``.

        Returns:
            ``x'`` values of shape ``(n, d)``.
        """
        u_raw: jnp.ndarray = _multivariate_input(u)[0]
        eps: float = 1e-4
        u_clipped: jnp.ndarray = jnp.clip(u_raw, eps, 1 - eps)
        uvt = self._uvt
        batched_params: dict = self._get_uvt_params(params)

        def _per_dim(xi_col, p_slice):
            p = uvt._resolve_params(p_slice)
            return uvt.ppf(xi_col, params=p, brent=brent, nodes=nodes)

        return vmap(_per_dim, in_axes=(1, 0), out_axes=1)(u_clipped, batched_params)

    # densities
    def copula_logpdf(
        self,
        u: ArrayLike,
        params: dict = None,
        brent: bool = False,
        nodes: int = 100,
    ) -> Array:
        r"""Computes the log-pdf of the copula distribution.

        Note:
            If you intend to jit wrap this function, both ``brent``
            and ``nodes`` must be static arguments.

        Args:
            u (ArrayLike): The independent univariate marginal cdf
                values (u) for each dimension.
            params (dict): The copula and marginal distribution
                parameters.
            brent (bool): Forwarded to :py:meth:`get_x_dash`.  ``False``
                (default) uses the analytical inverse CDF when
                available and otherwise the Chebyshev cubic spline;
                ``True`` forces per-quantile Brent root-finding.
            nodes (int): Number of Chebyshev-Lobatto nodes used by the
                cubic spline path.  Ignored for analytical marginals
                and when ``brent=True``.

        Returns:
            logpdf (Array): The log-pdf values of the copula
                distribution.
        """
        # mapping u to x' space
        params = self._resolve_params(params)
        x_dash: jnp.ndarray = self.get_x_dash(u, params, brent=brent, nodes=nodes)

        # computing univariate logpdfs
        uvt_logpdf: jnp.ndarray = self._scan_uvt_func(
            func=self._uvt.logpdf, x=x_dash, params=params
        )

        # computing copula logpdf
        mvt_params: dict = params["copula"]
        mvt_logpdf: jnp.ndarray = self._mvt.logpdf(x_dash, params=mvt_params)
        return mvt_logpdf - uvt_logpdf.sum(axis=1, keepdims=True)

    # sampling
    def copula_rvs(self, size: Scalar, params: dict = None, key: Array = None) -> Array:
        r"""Generates random samples from the copula distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'size'
            is a static argument.

        Args:
            size (Scalar): size (Scalar): The size / shape of the generated
                output array of random numbers. Must be scalar.
                Generates an (size, d) array of random numbers, where
                d is the number of dimensions inferred from the provided
                distribution parameters.
            params (dict): The copula and marginal distribution
                parameters.
            key (Array): The Key for random number generation.
        """
        params = self._resolve_params(params)
        key = _resolve_key(key)
        # generating random samples from x'
        x_dash: jnp.ndarray = self._mvt.rvs(size=size, key=key, params=params["copula"])

        # projecting x' to u space
        return self._scan_uvt_func(self._uvt.cdf, x=x_dash, params=params)

    # fitting
    def _estimate_copula_correlation(
        self, u: jnp.ndarray, corr_method: str
    ) -> Array:
        r"""Estimate the copula correlation matrix from pseudo-observations.

        For elliptical copulas, the recommended method is ``rm_pp_kendall``
        which computes Kendall's tau, applies :math:`\sin(\pi/2 \cdot \tau)`
        to recover the linear correlation parameter (Proposition 5.37,
        McNeil et al. 2005), and denoises via eigenvalue clamping to
        ensure positive semi-definiteness.

        Args:
            u: Pseudo-observations of shape ``(n, d)`` in ``[0, 1]``.
            corr_method: Correlation estimation method. Recommended:
                ``'rm_pp_kendall'`` (default). See
                ``copulax.multivariate.corr`` for all methods.

        Returns:
            Estimated correlation matrix of shape ``(d, d)``.
        """
        return corr(x=u, method=corr_method)

    def _build_initial_copula_params(self, d: int, sigma: Array) -> dict:
        r"""Construct initial copula parameters with the estimated
        correlation matrix and sensible defaults for other parameters.

        Subclasses must override to add distribution-specific parameters
        (e.g. nu for Student-t, gamma for skewed-t).

        Args:
            d: Dimensionality.
            sigma: Estimated correlation matrix of shape ``(d, d)``.

        Returns:
            Initial copula parameter dictionary.
        """
        return self._mvt._params_dict(
            mu=jnp.zeros((d, 1)), sigma=sigma
        )

    def _copula_nll(
        self,
        opt_arr: jnp.ndarray,
        u: jnp.ndarray,
        sigma: jnp.ndarray,
        dummy_marginals: tuple,
    ) -> Scalar:
        r"""Negative copula log-likelihood for optimisation.

        Gradients flow through the PPF via the implicit function
        theorem (custom JVP on the PPF), giving exact derivatives
        without differentiating through the root-finder or cubic
        spline.

        Args:
            opt_arr: Flat array of parameters being optimised.
            u: Pseudo-observations, shape ``(n, d)``.
            sigma: Fixed correlation matrix, shape ``(d, d)``.
            dummy_marginals: Tuple of (dist, params) for dimension
                inference.

        Returns:
            Scalar negative log-likelihood.
        """
        d: int = sigma.shape[0]
        copula_params: dict = self._reconstruct_copula_opt_params(
            opt_arr, sigma, d
        )
        full_params: dict = {
            "marginals": dummy_marginals, "copula": copula_params
        }
        logpdf: Array = self.copula_logpdf(u, params=full_params)
        n = logpdf.shape[0]
        finite_mask = jnp.isfinite(logpdf)
        safe_logpdf = jnp.where(finite_mask, logpdf, 0.0)
        n_invalid = (~finite_mask).astype(float).sum()
        return -safe_logpdf.sum() / n + 1e6 * n_invalid / n

    def _get_opt_params_and_bounds(
        self, d: int
    ) -> tuple[jnp.ndarray, dict]:
        r"""Return initial optimisation vector and box bounds.

        Subclasses that need parameter optimisation must override.

        Returns:
            Tuple of (initial_params_array, projection_options_dict).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not require parameter optimisation."
        )

    def _reconstruct_copula_opt_params(
        self,
        opt_arr: jnp.ndarray,
        sigma: jnp.ndarray,
        d: int,
    ) -> dict:
        r"""Rebuild copula params dict from optimiser output + fixed sigma.

        Subclasses that need parameter optimisation must override.
        """
        raise NotImplementedError

    def _optimize_copula_params(
        self,
        u: jnp.ndarray,
        sigma: jnp.ndarray,
        d: int,
        lr: float,
        maxiter: int,
    ) -> dict:
        r"""Optimise non-correlation copula parameters via ML.

        Uses ``projected_gradient`` to minimise the negative copula
        log-likelihood with the correlation matrix ``sigma`` held fixed.

        Args:
            u: Pseudo-observations, shape ``(n, d)``.
            sigma: Fixed correlation matrix, shape ``(d, d)``.
            d: Dimensionality.
            lr: Learning rate.
            maxiter: Maximum optimisation iterations.

        Returns:
            Fitted copula parameter dictionary.
        """
        params0, proj_opts = self._get_opt_params_and_bounds(d)
        dummy_marginals: tuple = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )
        res: dict = projected_gradient(
            f=self._copula_nll,
            x0=params0,
            projection_method="projection_box",
            projection_options=proj_opts,
            u=u,
            sigma=sigma,
            dummy_marginals=dummy_marginals,
            lr=lr,
            maxiter=maxiter,
        )
        return self._reconstruct_copula_opt_params(res["x"], sigma, d)

    def fit_copula(
        self,
        u: ArrayLike,
        corr_method: str = "rm_pp_kendall",
        method: str = "fc_mle",
        **kwargs,
    ) -> dict:
        r"""Fit copula parameters from pseudo-observations.

        Two-stage estimation following McNeil, Frey & Embrechts (2005),
        Section 5.5:

        **Stage 1** — Estimate the copula correlation matrix *P* from
        the pseudo-observations *u* using rank correlation.  The default
        ``rm_pp_kendall`` computes Kendall's tau, applies the inversion
        :math:`\hat\rho_{ij} = \sin(\tfrac{\pi}{2}\,\hat\tau_{ij})`
        (Proposition 5.37), and ensures positive semi-definiteness via
        eigenvalue clamping.

        **Stage 2** — Estimate remaining parameters (e.g. *ν*, *γ*) by
        maximising the copula log-likelihood, with *P* either held
        fixed at the Stage 1 estimate or jointly re-optimised, depending
        on ``method``.

        Args:
            u: Pseudo-observations of shape ``(n, d)`` in ``[0, 1]``.
            corr_method: Correlation estimation method for Stage 1.
                Default ``'rm_pp_kendall'``. See
                ``copulax.multivariate.corr`` for all methods.
            method: Fitting algorithm for Stage 2.  Must be a member of
                ``self._supported_methods``.
                ``'fc_mle'`` — *Fixed-Correlation MLE*: shape parameters
                optimised via projected gradient with Σ held at the
                Stage 1 Kendall-τ estimate. Available for every
                concrete subclass.
                ``'mle'`` — *Full joint MLE*: all parameters (shape +
                Σ off-diagonals) optimised together via Adam, with Σ
                tanh-parameterised onto the correlation manifold.
                Mean-variance subclasses (GH, SkewedT) only.
                ``'ecme'`` — Inner EM updates (P, γ); outer gradient
                descent on the copula log-likelihood for the remaining
                shape parameters (McNeil §3.2.4 ECME variant).
                Mean-variance subclasses only.
                ``'ecme_double_gamma'`` — Like ``ecme`` but γ is
                additionally re-optimised in the outer numerical M-step
                (so γ is updated twice per outer iteration).
                Mean-variance subclasses only.
                ``'ecme_outer_gamma'`` — Inner EM updates Σ only (γ
                frozen); outer MLE on all shape parameters including γ.
                Mean-variance subclasses only.
            **kwargs: Method-specific keyword arguments.  Each
                ``method`` accepts only the kwargs in
                ``_METHOD_KWARGS[method]``; passing an inapplicable
                kwarg raises ``ValueError``.  Common kwargs:
                ``lr`` (float, all methods), ``maxiter`` (int, all),
                ``tol`` (float, all except ``fc_mle``),
                ``patience`` (int, all except ``fc_mle``),
                ``brent`` (bool, all except ``fc_mle``),
                ``nodes`` (int, all except ``fc_mle``).

        Returns:
            dict with key ``'copula'`` containing fitted parameters.

        Raises:
            ValueError: If ``method`` is not in
                ``self._supported_methods``, or if ``kwargs`` contains
                a key not accepted by the chosen method.
        """
        # --- Validate method + kwargs (Python-level; happens at trace
        # time when fit_copula is JIT-wrapped with method as a static
        # arg, so the dispatcher remains JIT- and autograd-safe). ---
        if method not in self._supported_methods:
            raise ValueError(
                f"Method {method!r} not supported by {type(self).__name__}. "
                f"Supported: {sorted(self._supported_methods)}."
            )
        allowed = _METHOD_KWARGS[method]
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(
                f"Method {method!r} does not accept kwargs "
                f"{sorted(unknown)}. Accepted: {sorted(allowed)}."
            )

        # --- Resolve kwargs with documented defaults. ---
        lr = kwargs.get("lr", 1e-2)
        maxiter = kwargs.get("maxiter", 200)
        tol = kwargs.get("tol", 1e-6)
        patience = kwargs.get("patience", 5)
        brent = kwargs.get("brent", False)
        nodes = kwargs.get("nodes", 100)

        u_arr, _, n, d = _multivariate_input(u)

        # Stage 1: estimate correlation matrix P
        sigma: jnp.ndarray = self._estimate_copula_correlation(
            u_arr, corr_method
        )

        # Stage 2: estimate remaining parameters
        if method == "fc_mle":
            copula_params = self._fit_copula_fc_mle(
                u_arr, sigma, d, lr, maxiter,
            )
        elif method == "ecme":
            copula_params = self._fit_copula_ecme(
                u_arr, sigma, d, lr, maxiter, tol, patience, brent, nodes,
            )
        elif method == "ecme_double_gamma":
            copula_params = self._fit_copula_ecme_double_gamma(
                u_arr, sigma, d, lr, maxiter, tol, patience, brent, nodes,
            )
        elif method == "ecme_outer_gamma":
            copula_params = self._fit_copula_ecme_outer_gamma(
                u_arr, sigma, d, lr, maxiter, tol, patience, brent, nodes,
            )
        elif method == "mle":
            copula_params = self._fit_copula_mle(
                u_arr, sigma, d, lr, maxiter, tol, patience, brent, nodes,
            )
        else:
            # Should be unreachable thanks to the _supported_methods
            # guard above, but kept as a defensive backstop.
            raise ValueError(
                f"Unhandled supported method {method!r} on "
                f"{type(self).__name__}; implementation missing."
            )

        return {"copula": copula_params}

    def _fit_copula_fc_mle(
        self,
        u: jnp.ndarray,
        sigma: jnp.ndarray,
        d: int,
        lr: float,
        maxiter: int,
    ) -> dict:
        r"""Fixed-Correlation MLE: shape-parameter MLE with Σ held fixed.

        Σ is taken as-is from the Stage 1 Kendall-τ estimate supplied
        by ``MeanVarianceCopulaBase.fit_copula``.  Shape parameters are
        optimised via :func:`projected_gradient` on the negative copula
        log-likelihood.

        For the Gaussian copula this returns the correlation matrix
        directly (no additional shape parameters).  Subclasses with
        shape parameters override this method; mean-variance subclasses
        additionally implement ``_fit_copula_mle`` /
        ``_fit_copula_ecme*`` for joint Σ optimisation.
        """
        return self._build_initial_copula_params(d, sigma)


###############################################################################
# Sub-base classes (taxonomic split: variance vs mean-variance mixtures)
###############################################################################
class EllipticalCopula(MeanVarianceCopulaBase):
    r"""True elliptical copulas (normal *variance* mixtures, γ=0).

    Concrete subclasses (:class:`GaussianCopula`, :class:`StudentTCopula`)
    only support the ``'fc_mle'`` Stage 2 fitting method.

    Reference:
        McNeil, Frey, Embrechts (2005) *Quantitative Risk Management*,
        §3.2.1 Normal Variance Mixtures.
    """

    _supported_methods: frozenset = frozenset({"fc_mle"})


class MeanVarianceCopula(MeanVarianceCopulaBase):
    r"""Normal mean-variance mixture copulas with skewness γ.

    Concrete subclasses (:class:`GHCopula`, :class:`SkewedTCopula`)
    additionally implement γ-aware fitting methods (``mle``, ``ecme``,
    ``ecme_double_gamma``, ``ecme_outer_gamma``) on top of ``fc_mle``.

    Note:
        ``MeanVarianceCopulaBase`` is the broader umbrella covering this
        class **and** :class:`EllipticalCopula` (the γ=0 special case).
        This class is the proper γ≠0 specialisation.

    Reference:
        McNeil, Frey, Embrechts (2005) *Quantitative Risk Management*,
        §3.2.2 Normal Mean-Variance Mixtures, §3.2.4 Algorithm 3.14.
    """

    _supported_methods: frozenset = frozenset({
        "fc_mle", "mle", "ecme", "ecme_double_gamma", "ecme_outer_gamma",
    })

    @abstractmethod
    def _fit_copula_mle(
        self, u, sigma, d, lr, maxiter, tol=1e-6, patience=5,
        brent: bool = False, nodes: int = 100,
    ) -> dict:
        r"""Full joint MLE over Σ off-diagonals **and** all shape /
        skewness parameters.

        Σ is re-optimised via tanh-parameterisation of the off-diagonal
        correlations projected onto the correlation manifold, alongside
        the shape parameters, with Adam steps under an outer
        ``tol``/``patience`` early-stopping loop.  Unlike
        :py:meth:`_fit_copula_fc_mle`, which holds Σ fixed at the
        Kendall-τ rank-correlation estimate supplied by the base
        dispatcher, ``mle`` re-optimises Σ jointly with the shape
        parameters.

        Subclasses implement; abstract here for safety.
        """

    @abstractmethod
    def _fit_copula_ecme(
        self, u, sigma, d, lr, maxiter, tol=1e-6, patience=5,
        brent: bool = False, nodes: int = 100,
    ) -> dict:
        r"""ECME fitting: inner EM updates (Σ, γ); outer numerical
        maximisation of the original copula log-likelihood with respect
        to the remaining shape parameters (e.g. λ, χ, ψ for GH; ν for
        SkewedT) with γ and Σ held fixed at the inner-EM values
        (McNeil §3.2.4 ECME variant)."""

    @abstractmethod
    def _fit_copula_ecme_double_gamma(
        self, u, sigma, d, lr, maxiter, tol=1e-6, patience=5,
        brent: bool = False, nodes: int = 100,
    ) -> dict:
        r"""ECME variant in which γ is updated *twice* per outer
        iteration: first by the inner EM step (alongside Σ), then again
        by the outer numerical M-step alongside the other shape
        parameters."""

    @abstractmethod
    def _fit_copula_ecme_outer_gamma(
        self, u, sigma, d, lr, maxiter, tol=1e-6, patience=5,
        brent: bool = False, nodes: int = 100,
    ) -> dict:
        r"""ECME variant in which the inner EM updates Σ only (γ is
        held fixed in the inner step) and γ is optimised together with
        the other shape parameters in the outer numerical M-step."""


###############################################################################
# Copula Distributions
###############################################################################
# Normal Mixture Copulas
class GaussianCopula(EllipticalCopula):
    r"""The Gaussian Copula is a copula that uses the multivariate normal
    distribution to model the dependencies between random variables.

    The copula is parameterised by the correlation matrix *P* only.
    Fitting estimates *P* from pseudo-observations via rank correlation
    (McNeil et al. 2005, Example 5.58).

    https://en.wikipedia.org/wiki/Copula_(statistics)
    """

    @jit
    def _get_uvt_params(self, params: dict) -> dict:
        """Extract univariate parameters for the Gaussian copula margins."""
        d: int = self._get_dim(params)
        return {"mu": jnp.zeros(d), "sigma": jnp.ones(d)}


gaussian_copula = GaussianCopula("Gaussian-Copula", mvt_normal, normal)


class StudentTCopula(EllipticalCopula):
    r"""The Student-T Copula is a copula that uses the multivariate
    Student-T distribution to model the dependencies between random
    variables.

    The copula is parameterised by degrees of freedom *ν* and correlation
    matrix *P*.  Fitting estimates *P* via Kendall's tau inversion and
    *ν* by maximising the copula log-likelihood (McNeil et al. 2005,
    Examples 5.54 and 5.59).

    https://en.wikipedia.org/wiki/Copula_(statistics)
    """

    @jit
    def _get_uvt_params(self, params: dict) -> dict:
        """Extract univariate parameters for the student-t copula margins."""
        nu: Scalar = params["copula"]["nu"]
        d: int = self._get_dim(params)
        return {"nu": jnp.full(d, nu), "mu": jnp.zeros(d), "sigma": jnp.ones(d)}

    def _build_initial_copula_params(self, d: int, sigma: Array) -> dict:
        return self._mvt._params_dict(
            nu=jnp.array(5.0),
            mu=jnp.zeros((d, 1)),
            sigma=sigma,
        )

    def _get_opt_params_and_bounds(self, d: int):
        # Optimise raw_nu (1 parameter); nu = softplus(raw_nu)
        raw_nu0 = jnp.log(jnp.expm1(jnp.array(5.0)))
        params0 = raw_nu0.reshape((1,))
        proj_opts = {
            "lower": jnp.full((1, 1), -10.0),
            "upper": jnp.full((1, 1), 10.0),
        }
        return params0, proj_opts

    def _reconstruct_copula_opt_params(self, opt_arr, sigma, d):
        raw_nu = opt_arr[0]
        nu = jnn.softplus(raw_nu) + _NU_EPS
        return self._mvt._params_dict(
            nu=nu,
            mu=jnp.zeros((d, 1)),
            sigma=sigma,
        )

    def _fit_copula_fc_mle(self, u, sigma, d, lr, maxiter):
        # MoM initialization for nu
        R_inv = jnp.linalg.inv(sigma)
        nu_hat = mom_nu_student_t(u, R_inv, d)
        raw_nu0 = _inv_softplus(jnp.clip(nu_hat, 2.5, 200.0))
        params0 = raw_nu0.reshape((1,))
        proj_opts = {
            "lower": jnp.full((1, 1), -10.0),
            "upper": jnp.full((1, 1), 10.0),
        }
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )
        res = projected_gradient(
            f=self._copula_nll,
            x0=params0,
            projection_method="projection_box",
            projection_options=proj_opts,
            u=u, sigma=sigma,
            dummy_marginals=dummy_marginals,
            lr=lr, maxiter=maxiter,
        )
        return self._reconstruct_copula_opt_params(res["x"], sigma, d)


student_t_copula = StudentTCopula("Student-T-Copula", mvt_student_t, student_t)


class GHCopula(MeanVarianceCopula):
    r"""The GH Copula is a copula that uses the multivariate generalized
    hyperbolic (GH) distribution to model the dependencies between
    random variables.

    The copula is parameterised by (λ, χ, ψ, γ) and correlation matrix
    *P*.  Fitting estimates *P* via Kendall's tau inversion and the
    remaining parameters via ML or EM (McNeil et al. 2005, Section 5.5).

    https://en.wikipedia.org/wiki/Copula_(statistics)
    """

    @jit
    def _get_uvt_params(self, params: dict) -> dict:
        """Extract univariate parameters for the GH copula margins."""
        d: int = self._get_dim(params)
        lamb: Scalar = params["copula"]["lamb"]
        chi: Scalar = params["copula"]["chi"]
        psi: Scalar = params["copula"]["psi"]
        gamma: Array = params["copula"]["gamma"]
        return {
            "lamb": jnp.full(d, lamb),
            "chi": jnp.full(d, chi),
            "psi": jnp.full(d, psi),
            "mu": jnp.zeros(d),
            "sigma": jnp.ones(d),
            "gamma": gamma.flatten(),
        }

    def _build_initial_copula_params(self, d: int, sigma: Array) -> dict:
        return self._mvt._params_dict(
            lamb=jnp.array(0.0),
            chi=jnp.array(1.0),
            psi=jnp.array(1.0),
            mu=jnp.zeros((d, 1)),
            gamma=jnp.zeros((d, 1)),
            sigma=sigma,
        )

    def _get_opt_params_and_bounds(self, d: int):
        # Optimise [lamb, raw_chi, raw_psi, gamma_1..gamma_d]
        params0 = jnp.concatenate([
            jnp.array([0.0]),                                    # lamb
            jnp.log(jnp.expm1(jnp.array([1.0]))),               # raw_chi
            jnp.log(jnp.expm1(jnp.array([1.0]))),               # raw_psi
            jnp.zeros(d),                                        # gamma
        ])
        n_params = 3 + d
        proj_opts = {
            "lower": jnp.full((n_params, 1), -10.0),
            "upper": jnp.full((n_params, 1), 10.0),
        }
        return params0, proj_opts

    def _reconstruct_copula_opt_params(self, opt_arr, sigma, d):
        lamb = opt_arr[0]
        chi = jnn.softplus(opt_arr[1]) + _POS_EPS
        psi = jnn.softplus(opt_arr[2]) + _POS_EPS
        gamma = opt_arr[3:3 + d].reshape((d, 1))
        return self._mvt._params_dict(
            lamb=lamb, chi=chi, psi=psi,
            mu=jnp.zeros((d, 1)),
            gamma=gamma, sigma=sigma,
        )

    def _fit_copula_fc_mle(self, u, sigma, d, lr, maxiter):
        return self._optimize_copula_params(u, sigma, d, lr, maxiter)

    def _gh_copula_nll_closure(self, d, mu, eps=_EPS):
        r"""Build a JIT-compiled copula NLL function for the GH family.

        Returns a function ``nll(opt_arr, sigma, x) -> scalar`` where
        ``opt_arr = [lamb, raw_chi, raw_psi, gamma_1..gamma_d]``.
        """
        mvt = self._mvt
        uvt = self._uvt

        def _copula_nll(opt_arr, sigma_, x):
            l = opt_arr[0]
            c = jnn.softplus(opt_arr[1]) + eps
            p = jnn.softplus(opt_arr[2]) + eps
            g = opt_arr[3:].reshape((d, 1))
            copula_p = mvt._params_dict(
                lamb=l, chi=c, psi=p,
                mu=mu, gamma=g, sigma=sigma_,
            )
            mvt_ll = mvt.logpdf(x, params=copula_p)
            uvt_params = {
                "lamb": jnp.full(d, l),
                "chi": jnp.full(d, c),
                "psi": jnp.full(d, p),
                "mu": jnp.zeros(d),
                "sigma": jnp.ones(d),
                "gamma": g.flatten(),
            }
            uvt_ll = vmap(
                lambda xi, pr: uvt.logpdf(xi, params=pr),
                in_axes=(1, 0), out_axes=1,
            )(x, uvt_params).sum(axis=1, keepdims=True)
            logpdf = mvt_ll - uvt_ll
            n = logpdf.shape[0]
            finite_mask = jnp.isfinite(logpdf)
            safe = jnp.where(finite_mask, logpdf, 0.0)
            return -safe.sum() / n + 1e6 * (~finite_mask).sum() / n

        return _copula_nll

    def _gh_copula_ll(self, d, mu):
        r"""Build a JIT-compiled copula LL evaluator for convergence
        monitoring.  Returns ``ll(x, lamb, chi, psi, gamma, sigma) ->
        scalar``."""
        mvt = self._mvt
        uvt = self._uvt

        @jax.jit
        def _ll(x, lamb, chi, psi, gamma, sigma_):
            copula_p = mvt._params_dict(
                lamb=lamb, chi=chi, psi=psi,
                mu=mu, gamma=gamma, sigma=sigma_,
            )
            mvt_ll = mvt.logpdf(x, params=copula_p)
            uvt_params = {
                "lamb": jnp.full(d, lamb),
                "chi": jnp.full(d, chi),
                "psi": jnp.full(d, psi),
                "mu": jnp.zeros(d),
                "sigma": jnp.ones(d),
                "gamma": gamma.flatten(),
            }
            uvt_ll = vmap(
                lambda xi, pr: uvt.logpdf(xi, params=pr),
                in_axes=(1, 0), out_axes=1,
            )(x, uvt_params).sum(axis=1, keepdims=True)
            logpdf = mvt_ll - uvt_ll
            n = logpdf.shape[0]
            finite_mask = jnp.isfinite(logpdf)
            return jnp.where(finite_mask, logpdf, 0.0).sum() / n

        return _ll

    def _fit_copula_ecme(
        self, u, sigma, d, lr, maxiter, tol=1e-6, patience=5,
        brent: bool = False, nodes: int = 100,
    ):
        r"""EM-MLE copula fitting for the GH copula.

        Alternates between inner EM (updates P and γ) and outer
        CM-step (gradient descent on λ, χ, ψ).

        Uses ``lax.scan`` for inner loops, fresh Adam state per outer
        iteration, and early stopping.
        """
        eps = _EPS
        mu = jnp.zeros((d, 1))
        inner_maxiter: int = 5
        shape_steps: int = 10
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )

        copula_nll_fn = self._gh_copula_nll_closure(d, mu, eps)
        copula_ll_fn = self._gh_copula_ll(d, mu)

        # --- JIT: inner EM scan (gamma + sigma) ---
        @jax.jit
        def _run_inner_em(gamma, sigma_, x_dash, lamb, chi, psi):
            def _scan_body(carry, _):
                g, s = carry
                g, s = _inner_em_step_gh(g, s, x_dash, lamb, chi, psi, True)
                return (g, s), None
            (g, s), _ = lax.scan(
                _scan_body, (gamma, sigma_), None, length=inner_maxiter
            )
            return g, s

        # --- JIT: shape CM scan (lamb, chi, psi) ---
        @jax.jit
        def _run_shape_steps(lamb, chi, psi, gamma, sigma_, adam_state, x_dash):
            def _copula_nll_shape(shape_arr):
                opt = jnp.concatenate([shape_arr, gamma.flatten()])
                return copula_nll_fn(opt, sigma_, x_dash)

            def _scan_body(carry, _):
                l, c, p, a_s = carry
                raw_c = _inv_softplus(jnp.maximum(c, eps))
                raw_p = _inv_softplus(jnp.maximum(p, eps))
                shape_arr = jnp.array([l, raw_c, raw_p])
                shape_arr, a_s = _adam_gradient_step(
                    _copula_nll_shape, shape_arr, a_s, lr
                )
                l = jnp.clip(shape_arr[0], -10.0, 10.0)
                c = jnp.clip(jnn.softplus(shape_arr[1]) + eps, eps, 100.0)
                p = jnp.clip(jnn.softplus(shape_arr[2]) + eps, eps, 100.0)
                return (l, c, p, a_s), None

            (lamb, chi, psi, adam_state), _ = lax.scan(
                _scan_body, (lamb, chi, psi, adam_state), None,
                length=shape_steps,
            )
            return lamb, chi, psi, adam_state

        # --- MoM initialization ---
        R_inv = jnp.linalg.inv(sigma)
        nu_hat = mom_nu_student_t(u, R_inv, d)
        lamb, chi, psi = mom_gh_params(u, R_inv, d, nu_hat)

        # --- Python outer loop ---
        gamma = jnp.zeros((d, 1))
        adam_state = (jnp.zeros(3), jnp.zeros(3), jnp.array(0))
        _get_x_dash_jit = jax.jit(
            self.get_x_dash, static_argnames=("brent", "nodes")
        )
        prev_ll = -1e20
        no_improve_count = 0

        for _iter in range(maxiter):
            copula_params = self._mvt._params_dict(
                lamb=lamb, chi=chi, psi=psi,
                mu=mu, gamma=gamma, sigma=sigma,
            )
            full_params = {
                "marginals": dummy_marginals, "copula": copula_params,
            }
            x_dash = _get_x_dash_jit(
                u, full_params, brent=brent, nodes=nodes
            )

            # Early stopping (evaluate with fresh x_dash + current params)
            if tol > 0:
                current_ll = float(copula_ll_fn(
                    x_dash, lamb, chi, psi, gamma, sigma
                ))
                rel_imp = (current_ll - prev_ll) / max(abs(float(prev_ll)), 1.0)
                if rel_imp < tol:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                if no_improve_count >= patience:
                    break
                prev_ll = current_ll

            gamma, sigma = _run_inner_em(
                gamma, sigma, x_dash, lamb, chi, psi
            )

            adam_state = _reset_adam_state(adam_state)
            lamb, chi, psi, adam_state = _run_shape_steps(
                lamb, chi, psi, gamma, sigma, adam_state, x_dash
            )

        return self._mvt._params_dict(
            lamb=lamb, chi=chi, psi=psi,
            mu=mu, gamma=gamma, sigma=sigma,
        )

    def _fit_copula_ecme_double_gamma(
        self, u, sigma, d, lr, maxiter, tol=1e-6, patience=5,
        brent: bool = False, nodes: int = 100,
    ):
        r"""EM-MLE variant 2 for the GH copula.

        Outer MLE optimises (λ, χ, ψ, γ) jointly.  Inner EM updates
        both γ and P, providing a warm-start for each outer step.
        """
        eps = _EPS
        mu = jnp.zeros((d, 1))
        inner_maxiter: int = 5
        shape_steps: int = 10
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )

        copula_nll_fn = self._gh_copula_nll_closure(d, mu, eps)
        copula_ll_fn = self._gh_copula_ll(d, mu)

        # --- JIT: inner EM scan (gamma + sigma) ---
        @jax.jit
        def _run_inner_em(gamma, sigma_, x_dash, lamb, chi, psi):
            def _scan_body(carry, _):
                g, s = carry
                g, s = _inner_em_step_gh(g, s, x_dash, lamb, chi, psi, True)
                return (g, s), None
            (g, s), _ = lax.scan(
                _scan_body, (gamma, sigma_), None, length=inner_maxiter
            )
            return g, s

        # --- JIT: outer MLE scan (lamb, chi, psi, gamma) ---
        @jax.jit
        def _run_outer_mle(lamb, chi, psi, gamma, sigma_, adam_state, x_dash):
            def _scan_body(carry, _):
                l, c, p, g, a_s = carry
                raw_c = _inv_softplus(jnp.maximum(c, eps))
                raw_p = _inv_softplus(jnp.maximum(p, eps))
                opt_arr = jnp.concatenate([
                    jnp.array([l, raw_c, raw_p]), g.flatten()
                ])
                opt_arr, a_s = _adam_gradient_step(
                    lambda arr: copula_nll_fn(arr, sigma_, x_dash),
                    opt_arr, a_s, lr,
                )
                l = jnp.clip(opt_arr[0], -10.0, 10.0)
                c = jnp.clip(jnn.softplus(opt_arr[1]) + eps, eps, 100.0)
                p = jnp.clip(jnn.softplus(opt_arr[2]) + eps, eps, 100.0)
                g = opt_arr[3:].reshape((d, 1))
                return (l, c, p, g, a_s), None

            (lamb, chi, psi, gamma, adam_state), _ = lax.scan(
                _scan_body,
                (lamb, chi, psi, gamma, adam_state),
                None, length=shape_steps,
            )
            return lamb, chi, psi, gamma, adam_state

        # --- MoM initialization ---
        R_inv = jnp.linalg.inv(sigma)
        nu_hat = mom_nu_student_t(u, R_inv, d)
        lamb, chi, psi = mom_gh_params(u, R_inv, d, nu_hat)

        # --- Python outer loop ---
        gamma = jnp.zeros((d, 1))
        adam_state = (jnp.zeros(3 + d), jnp.zeros(3 + d), jnp.array(0))
        _get_x_dash_jit = jax.jit(
            self.get_x_dash, static_argnames=("brent", "nodes")
        )
        prev_ll = -1e20
        no_improve_count = 0

        for _iter in range(maxiter):
            copula_params = self._mvt._params_dict(
                lamb=lamb, chi=chi, psi=psi,
                mu=mu, gamma=gamma, sigma=sigma,
            )
            full_params = {
                "marginals": dummy_marginals, "copula": copula_params,
            }
            x_dash = _get_x_dash_jit(
                u, full_params, brent=brent, nodes=nodes
            )

            # Early stopping (evaluate with fresh x_dash + current params)
            if tol > 0:
                current_ll = float(copula_ll_fn(
                    x_dash, lamb, chi, psi, gamma, sigma
                ))
                rel_imp = (current_ll - prev_ll) / max(abs(float(prev_ll)), 1.0)
                if rel_imp < tol:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                if no_improve_count >= patience:
                    break
                prev_ll = current_ll

            gamma, sigma = _run_inner_em(
                gamma, sigma, x_dash, lamb, chi, psi
            )

            adam_state = _reset_adam_state(adam_state)
            lamb, chi, psi, gamma, adam_state = _run_outer_mle(
                lamb, chi, psi, gamma, sigma, adam_state, x_dash
            )

        return self._mvt._params_dict(
            lamb=lamb, chi=chi, psi=psi,
            mu=mu, gamma=gamma, sigma=sigma,
        )

    def _fit_copula_ecme_outer_gamma(
        self, u, sigma, d, lr, maxiter, tol=1e-6, patience=5,
        brent: bool = False, nodes: int = 100,
    ):
        r"""EM-MLE variant 3 for the GH copula.

        Inner EM updates P only (γ frozen).  Outer MLE optimises
        (λ, χ, ψ, γ) jointly.
        """
        eps = _EPS
        mu = jnp.zeros((d, 1))
        inner_maxiter: int = 5
        shape_steps: int = 10
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )

        copula_nll_fn = self._gh_copula_nll_closure(d, mu, eps)
        copula_ll_fn = self._gh_copula_ll(d, mu)

        # --- JIT: inner EM scan (sigma only, gamma frozen) ---
        @jax.jit
        def _run_inner_em(gamma, sigma_, x_dash, lamb, chi, psi):
            def _scan_body(carry, _):
                g, s = carry
                g, s = _inner_em_step_gh(g, s, x_dash, lamb, chi, psi, False)
                return (g, s), None
            (_, s), _ = lax.scan(
                _scan_body, (gamma, sigma_), None, length=inner_maxiter
            )
            return s

        # --- JIT: outer MLE scan (lamb, chi, psi, gamma) ---
        @jax.jit
        def _run_outer_mle(lamb, chi, psi, gamma, sigma_, adam_state, x_dash):
            def _scan_body(carry, _):
                l, c, p, g, a_s = carry
                raw_c = _inv_softplus(jnp.maximum(c, eps))
                raw_p = _inv_softplus(jnp.maximum(p, eps))
                opt_arr = jnp.concatenate([
                    jnp.array([l, raw_c, raw_p]), g.flatten()
                ])
                opt_arr, a_s = _adam_gradient_step(
                    lambda arr: copula_nll_fn(arr, sigma_, x_dash),
                    opt_arr, a_s, lr,
                )
                l = jnp.clip(opt_arr[0], -10.0, 10.0)
                c = jnp.clip(jnn.softplus(opt_arr[1]) + eps, eps, 100.0)
                p = jnp.clip(jnn.softplus(opt_arr[2]) + eps, eps, 100.0)
                g = opt_arr[3:].reshape((d, 1))
                return (l, c, p, g, a_s), None

            (lamb, chi, psi, gamma, adam_state), _ = lax.scan(
                _scan_body,
                (lamb, chi, psi, gamma, adam_state),
                None, length=shape_steps,
            )
            return lamb, chi, psi, gamma, adam_state

        # --- MoM initialization ---
        R_inv = jnp.linalg.inv(sigma)
        nu_hat = mom_nu_student_t(u, R_inv, d)
        lamb, chi, psi = mom_gh_params(u, R_inv, d, nu_hat)

        # --- Python outer loop ---
        gamma = jnp.zeros((d, 1))
        adam_state = (jnp.zeros(3 + d), jnp.zeros(3 + d), jnp.array(0))
        _get_x_dash_jit = jax.jit(
            self.get_x_dash, static_argnames=("brent", "nodes")
        )
        prev_ll = -1e20
        no_improve_count = 0

        for _iter in range(maxiter):
            copula_params = self._mvt._params_dict(
                lamb=lamb, chi=chi, psi=psi,
                mu=mu, gamma=gamma, sigma=sigma,
            )
            full_params = {
                "marginals": dummy_marginals, "copula": copula_params,
            }
            x_dash = _get_x_dash_jit(
                u, full_params, brent=brent, nodes=nodes
            )

            # Early stopping (evaluate with fresh x_dash + current params)
            if tol > 0:
                current_ll = float(copula_ll_fn(
                    x_dash, lamb, chi, psi, gamma, sigma
                ))
                rel_imp = (current_ll - prev_ll) / max(abs(float(prev_ll)), 1.0)
                if rel_imp < tol:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                if no_improve_count >= patience:
                    break
                prev_ll = current_ll

            sigma = _run_inner_em(gamma, sigma, x_dash, lamb, chi, psi)

            adam_state = _reset_adam_state(adam_state)
            lamb, chi, psi, gamma, adam_state = _run_outer_mle(
                lamb, chi, psi, gamma, sigma, adam_state, x_dash
            )

        return self._mvt._params_dict(
            lamb=lamb, chi=chi, psi=psi,
            mu=mu, gamma=gamma, sigma=sigma,
        )

    def _fit_copula_mle(
        self, u, sigma, d, lr, maxiter, tol=1e-6, patience=5,
        brent: bool = False, nodes: int = 100,
    ):
        r"""Full MLE for the GH copula.

        All parameters (λ, χ, ψ, γ, P) optimised via gradient descent
        on the copula LL.  P is represented via ``d*(d-1)/2`` free
        off-diagonal entries mapped through ``tanh``.
        """
        eps = _EPS
        mu = jnp.zeros((d, 1))
        shape_steps: int = 10
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )
        tril_rows, tril_cols = jnp.tril_indices(d, k=-1)
        mvt = self._mvt
        uvt = self._uvt

        def _sigma_from_raw(raw_corr):
            rho = jnp.tanh(raw_corr)
            P = jnp.eye(d)
            P = P.at[tril_rows, tril_cols].set(rho)
            P = P.at[tril_cols, tril_rows].set(rho)
            P = _corr._rm_incomplete(P, 1e-5)
            P = _corr._corr_from_cov(P)
            P = _corr._ensure_valid(P)
            return P

        def _raw_from_sigma(sigma_):
            rho = sigma_[tril_rows, tril_cols]
            return jnp.arctanh(jnp.clip(rho, -0.999, 0.999))

        copula_ll_fn = self._gh_copula_ll(d, mu)

        @jax.jit
        def _run_mle_steps(opt_arr, adam_state, x_dash):
            def _copula_nll(arr):
                l = arr[0]
                c = jnn.softplus(arr[1]) + eps
                p = jnn.softplus(arr[2]) + eps
                g = arr[3:3 + d].reshape((d, 1))
                sigma_ = _sigma_from_raw(arr[3 + d:])
                copula_p = mvt._params_dict(
                    lamb=l, chi=c, psi=p,
                    mu=mu, gamma=g, sigma=sigma_,
                )
                mvt_ll = mvt.logpdf(x_dash, params=copula_p)
                uvt_params = {
                    "lamb": jnp.full(d, l),
                    "chi": jnp.full(d, c),
                    "psi": jnp.full(d, p),
                    "mu": jnp.zeros(d),
                    "sigma": jnp.ones(d),
                    "gamma": g.flatten(),
                }
                uvt_ll = vmap(
                    lambda xi, pr: uvt.logpdf(xi, params=pr),
                    in_axes=(1, 0), out_axes=1,
                )(x_dash, uvt_params).sum(axis=1, keepdims=True)
                logpdf = mvt_ll - uvt_ll
                n = logpdf.shape[0]
                finite_mask = jnp.isfinite(logpdf)
                safe = jnp.where(finite_mask, logpdf, 0.0)
                return -safe.sum() / n + 1e6 * (~finite_mask).sum() / n

            def _scan_body(carry, _):
                arr, a_s = carry
                arr, a_s = _adam_gradient_step(_copula_nll, arr, a_s, lr)
                return (arr, a_s), None

            (opt_arr, adam_state), _ = lax.scan(
                _scan_body, (opt_arr, adam_state), None, length=shape_steps
            )
            return opt_arr, adam_state

        # --- Python outer loop ---
        # MoM initialization
        R_inv = jnp.linalg.inv(sigma)
        nu_hat = mom_nu_student_t(u, R_inv, d)
        lamb, chi, psi = mom_gh_params(u, R_inv, d, nu_hat)

        gamma = jnp.zeros((d, 1))
        n_corr = d * (d - 1) // 2
        n_opt = 3 + d + n_corr
        adam_state = (jnp.zeros(n_opt), jnp.zeros(n_opt), jnp.array(0))
        _get_x_dash_jit = jax.jit(
            self.get_x_dash, static_argnames=("brent", "nodes")
        )
        prev_ll = -1e20
        no_improve_count = 0

        for _iter in range(maxiter):
            copula_params = self._mvt._params_dict(
                lamb=lamb, chi=chi, psi=psi,
                mu=mu, gamma=gamma, sigma=sigma,
            )
            full_params = {
                "marginals": dummy_marginals, "copula": copula_params,
            }
            x_dash = _get_x_dash_jit(
                u, full_params, brent=brent, nodes=nodes
            )

            # Early stopping (evaluate with fresh x_dash + current params)
            if tol > 0:
                current_ll = float(copula_ll_fn(
                    x_dash, lamb, chi, psi, gamma, sigma
                ))
                rel_imp = (current_ll - prev_ll) / max(abs(float(prev_ll)), 1.0)
                if rel_imp < tol:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                if no_improve_count >= patience:
                    break
                prev_ll = current_ll

            raw_chi = _inv_softplus(jnp.maximum(chi, eps))
            raw_psi = _inv_softplus(jnp.maximum(psi, eps))
            raw_corr = _raw_from_sigma(sigma)
            opt_arr = jnp.concatenate([
                jnp.array([lamb, raw_chi, raw_psi]),
                gamma.flatten(),
                raw_corr,
            ])

            adam_state = _reset_adam_state(adam_state)
            opt_arr, adam_state = _run_mle_steps(opt_arr, adam_state, x_dash)

            lamb = jnp.clip(opt_arr[0], -10.0, 10.0)
            chi = jnp.clip(jnn.softplus(opt_arr[1]) + eps, eps, 100.0)
            psi = jnp.clip(jnn.softplus(opt_arr[2]) + eps, eps, 100.0)
            gamma = opt_arr[3:3 + d].reshape((d, 1))
            sigma = _sigma_from_raw(opt_arr[3 + d:])

        return self._mvt._params_dict(
            lamb=lamb, chi=chi, psi=psi,
            mu=mu, gamma=gamma, sigma=sigma,
        )


gh_copula = GHCopula("GH-Copula", mvt_gh, gh)


class SkewedTCopula(MeanVarianceCopula):
    r"""The Skewed-T Copula is a copula that uses the multivariate
    skewed-T distribution to model the dependencies between random
    variables.

    The copula is parameterised by degrees of freedom *ν*, skewness
    vector *γ*, and correlation matrix *P*.  Fitting estimates *P* via
    Kendall's tau inversion and (*ν*, *γ*) via ML or EM (McNeil et al.
    2005, Section 5.5).

    https://en.wikipedia.org/wiki/Copula_(statistics)
    """

    def _get_uvt_params(self, params: dict) -> dict:
        """Extract univariate parameters for the skewed-t copula margins."""
        d: int = self._get_dim(params)
        nu: Scalar = params["copula"]["nu"]
        gamma: Array = params["copula"]["gamma"]
        return {
            "nu": jnp.full(d, nu),
            "mu": jnp.zeros(d),
            "sigma": jnp.ones(d),
            "gamma": gamma.flatten(),
        }

    def _build_initial_copula_params(self, d: int, sigma: Array) -> dict:
        return self._mvt._params_dict(
            nu=jnp.array(5.0),
            mu=jnp.zeros((d, 1)),
            gamma=jnp.zeros((d, 1)),
            sigma=sigma,
        )

    def _get_opt_params_and_bounds(self, d: int):
        # Optimise [raw_nu, gamma_1..gamma_d]
        raw_nu0 = jnp.log(jnp.expm1(jnp.array(5.0)))
        params0 = jnp.concatenate([
            raw_nu0.reshape((1,)),
            jnp.zeros(d),
        ])
        n_params = 1 + d
        proj_opts = {
            "lower": jnp.full((n_params, 1), -10.0),
            "upper": jnp.full((n_params, 1), 10.0),
        }
        return params0, proj_opts

    def _reconstruct_copula_opt_params(self, opt_arr, sigma, d):
        raw_nu = opt_arr[0]
        nu = jnn.softplus(raw_nu) + _NU_EPS
        gamma = opt_arr[1:1 + d].reshape((d, 1))
        return self._mvt._params_dict(
            nu=nu,
            mu=jnp.zeros((d, 1)),
            gamma=gamma,
            sigma=sigma,
        )

    def _fit_copula_fc_mle(self, u, sigma, d, lr, maxiter):
        return self._optimize_copula_params(u, sigma, d, lr, maxiter)

    def _st_copula_nll_closure(self, d, mu, eps=_EPS):
        r"""Build a copula NLL function for the Skewed-T family.

        Returns ``nll(opt_arr, sigma, x) -> scalar`` where
        ``opt_arr = [raw_nu, gamma_1..gamma_d]``.
        """
        mvt = self._mvt
        uvt = self._uvt

        def _copula_nll(opt_arr, sigma_, x):
            n_val = jnn.softplus(opt_arr[0]) + eps
            g = opt_arr[1:].reshape((d, 1))
            copula_p = mvt._params_dict(
                nu=n_val, mu=mu, gamma=g, sigma=sigma_,
            )
            mvt_ll = mvt.logpdf(x, params=copula_p)
            uvt_params = {
                "nu": jnp.full(d, n_val),
                "mu": jnp.zeros(d),
                "sigma": jnp.ones(d),
                "gamma": g.flatten(),
            }
            uvt_ll = vmap(
                lambda xi, pr: uvt.logpdf(xi, params=pr),
                in_axes=(1, 0), out_axes=1,
            )(x, uvt_params).sum(axis=1, keepdims=True)
            logpdf = mvt_ll - uvt_ll
            n = logpdf.shape[0]
            finite_mask = jnp.isfinite(logpdf)
            safe = jnp.where(finite_mask, logpdf, 0.0)
            return -safe.sum() / n + 1e6 * (~finite_mask).sum() / n

        return _copula_nll

    def _st_copula_ll(self, d, mu):
        r"""Build a JIT-compiled copula LL evaluator for convergence
        monitoring.  Returns ``ll(x, nu, gamma, sigma) -> scalar``."""
        mvt = self._mvt
        uvt = self._uvt

        @jax.jit
        def _ll(x, nu, gamma, sigma_):
            copula_p = mvt._params_dict(
                nu=nu, mu=mu, gamma=gamma, sigma=sigma_,
            )
            mvt_ll = mvt.logpdf(x, params=copula_p)
            uvt_params = {
                "nu": jnp.full(d, nu),
                "mu": jnp.zeros(d),
                "sigma": jnp.ones(d),
                "gamma": gamma.flatten(),
            }
            uvt_ll = vmap(
                lambda xi, pr: uvt.logpdf(xi, params=pr),
                in_axes=(1, 0), out_axes=1,
            )(x, uvt_params).sum(axis=1, keepdims=True)
            logpdf = mvt_ll - uvt_ll
            n = logpdf.shape[0]
            finite_mask = jnp.isfinite(logpdf)
            return jnp.where(finite_mask, logpdf, 0.0).sum() / n

        return _ll

    def _fit_copula_ecme(
        self, u, sigma, d, lr, maxiter, tol=1e-6, patience=5,
        brent: bool = False, nodes: int = 100,
    ):
        r"""EM-MLE copula fitting for the Skewed-T copula.

        Alternates between inner EM (updates P and γ) and outer
        CM-step (gradient descent on ν).
        """
        eps = _EPS
        mu = jnp.zeros((d, 1))
        inner_maxiter: int = 5
        shape_steps: int = 10
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )

        copula_nll_fn = self._st_copula_nll_closure(d, mu, eps)
        copula_ll_fn = self._st_copula_ll(d, mu)

        # --- JIT: inner EM scan (gamma + sigma) ---
        @jax.jit
        def _run_inner_em(gamma, sigma_, x_dash, nu):
            def _scan_body(carry, _):
                g, s = carry
                g, s = _inner_em_step_skewed_t(g, s, x_dash, nu, True)
                return (g, s), None
            (g, s), _ = lax.scan(
                _scan_body, (gamma, sigma_), None, length=inner_maxiter
            )
            return g, s

        # --- JIT: shape CM scan (nu only) ---
        @jax.jit
        def _run_shape_steps(nu, gamma, sigma_, adam_state, x_dash):
            def _copula_nll_nu(raw_nu_arr):
                opt = jnp.concatenate([raw_nu_arr, gamma.flatten()])
                return copula_nll_fn(opt, sigma_, x_dash)

            def _scan_body(carry, _):
                n, a_s = carry
                raw_nu = _inv_softplus(jnp.maximum(n, eps))
                raw_arr = raw_nu.reshape((1,))
                raw_arr, a_s = _adam_gradient_step(
                    _copula_nll_nu, raw_arr, a_s, lr
                )
                n = jnn.softplus(raw_arr[0]) + eps
                return (n, a_s), None

            (nu, adam_state), _ = lax.scan(
                _scan_body, (nu, adam_state), None, length=shape_steps
            )
            return nu, adam_state

        # --- MoM initialization ---
        R_inv = jnp.linalg.inv(sigma)
        nu = jnp.clip(mom_nu_student_t(u, R_inv, d), 2.5, 60.0)

        # --- Python outer loop ---
        gamma = jnp.zeros((d, 1))
        adam_state = (jnp.zeros(1), jnp.zeros(1), jnp.array(0))
        _get_x_dash_jit = jax.jit(
            self.get_x_dash, static_argnames=("brent", "nodes")
        )
        prev_ll = -1e20
        no_improve_count = 0

        for _iter in range(maxiter):
            copula_params = self._mvt._params_dict(
                nu=nu, mu=mu, gamma=gamma, sigma=sigma,
            )
            full_params = {
                "marginals": dummy_marginals, "copula": copula_params,
            }
            x_dash = _get_x_dash_jit(
                u, full_params, brent=brent, nodes=nodes
            )

            # Early stopping (evaluate with fresh x_dash + current params)
            if tol > 0:
                current_ll = float(copula_ll_fn(
                    x_dash, nu, gamma, sigma
                ))
                rel_imp = (current_ll - prev_ll) / max(abs(float(prev_ll)), 1.0)
                if rel_imp < tol:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                if no_improve_count >= patience:
                    break
                prev_ll = current_ll

            gamma, sigma = _run_inner_em(gamma, sigma, x_dash, nu)

            adam_state = _reset_adam_state(adam_state)
            nu, adam_state = _run_shape_steps(
                nu, gamma, sigma, adam_state, x_dash
            )

        return self._mvt._params_dict(
            nu=nu, mu=mu, gamma=gamma, sigma=sigma,
        )

    def _fit_copula_ecme_double_gamma(
        self, u, sigma, d, lr, maxiter, tol=1e-6, patience=5,
        brent: bool = False, nodes: int = 100,
    ):
        r"""EM-MLE variant 2 for the Skewed-T copula.

        Outer MLE optimises (ν, γ) jointly.  Inner EM updates both
        γ and P.
        """
        eps = _EPS
        mu = jnp.zeros((d, 1))
        inner_maxiter: int = 5
        shape_steps: int = 10
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )

        copula_nll_fn = self._st_copula_nll_closure(d, mu, eps)
        copula_ll_fn = self._st_copula_ll(d, mu)

        # --- JIT: inner EM scan (gamma + sigma) ---
        @jax.jit
        def _run_inner_em(gamma, sigma_, x_dash, nu):
            def _scan_body(carry, _):
                g, s = carry
                g, s = _inner_em_step_skewed_t(g, s, x_dash, nu, True)
                return (g, s), None
            (g, s), _ = lax.scan(
                _scan_body, (gamma, sigma_), None, length=inner_maxiter
            )
            return g, s

        # --- JIT: outer MLE scan (nu, gamma) ---
        @jax.jit
        def _run_outer_mle(nu, gamma, sigma_, adam_state, x_dash):
            def _scan_body(carry, _):
                n, g, a_s = carry
                raw_nu = _inv_softplus(jnp.maximum(n, eps))
                opt_arr = jnp.concatenate([raw_nu.reshape((1,)), g.flatten()])
                opt_arr, a_s = _adam_gradient_step(
                    lambda arr: copula_nll_fn(arr, sigma_, x_dash),
                    opt_arr, a_s, lr,
                )
                n = jnn.softplus(opt_arr[0]) + eps
                g = opt_arr[1:].reshape((d, 1))
                return (n, g, a_s), None

            (nu, gamma, adam_state), _ = lax.scan(
                _scan_body,
                (nu, gamma, adam_state),
                None, length=shape_steps,
            )
            return nu, gamma, adam_state

        # --- MoM initialization ---
        R_inv = jnp.linalg.inv(sigma)
        nu = jnp.clip(mom_nu_student_t(u, R_inv, d), 2.5, 60.0)

        # --- Python outer loop ---
        gamma = jnp.zeros((d, 1))
        adam_state = (jnp.zeros(1 + d), jnp.zeros(1 + d), jnp.array(0))
        _get_x_dash_jit = jax.jit(
            self.get_x_dash, static_argnames=("brent", "nodes")
        )
        prev_ll = -1e20
        no_improve_count = 0

        for _iter in range(maxiter):
            copula_params = self._mvt._params_dict(
                nu=nu, mu=mu, gamma=gamma, sigma=sigma,
            )
            full_params = {
                "marginals": dummy_marginals, "copula": copula_params,
            }
            x_dash = _get_x_dash_jit(
                u, full_params, brent=brent, nodes=nodes
            )

            # Early stopping (evaluate with fresh x_dash + current params)
            if tol > 0:
                current_ll = float(copula_ll_fn(
                    x_dash, nu, gamma, sigma
                ))
                rel_imp = (current_ll - prev_ll) / max(abs(float(prev_ll)), 1.0)
                if rel_imp < tol:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                if no_improve_count >= patience:
                    break
                prev_ll = current_ll

            gamma, sigma = _run_inner_em(gamma, sigma, x_dash, nu)

            adam_state = _reset_adam_state(adam_state)
            nu, gamma, adam_state = _run_outer_mle(
                nu, gamma, sigma, adam_state, x_dash
            )

        return self._mvt._params_dict(
            nu=nu, mu=mu, gamma=gamma, sigma=sigma,
        )

    def _fit_copula_ecme_outer_gamma(
        self, u, sigma, d, lr, maxiter, tol=1e-6, patience=5,
        brent: bool = False, nodes: int = 100,
    ):
        r"""EM-MLE variant 3 for the Skewed-T copula.

        Inner EM updates P only (γ frozen).  Outer MLE optimises
        (ν, γ) jointly.
        """
        eps = _EPS
        mu = jnp.zeros((d, 1))
        inner_maxiter: int = 5
        shape_steps: int = 10
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )

        copula_nll_fn = self._st_copula_nll_closure(d, mu, eps)
        copula_ll_fn = self._st_copula_ll(d, mu)

        # --- JIT: inner EM scan (sigma only, gamma frozen) ---
        @jax.jit
        def _run_inner_em(gamma, sigma_, x_dash, nu):
            def _scan_body(carry, _):
                g, s = carry
                g, s = _inner_em_step_skewed_t(g, s, x_dash, nu, False)
                return (g, s), None
            (_, s), _ = lax.scan(
                _scan_body, (gamma, sigma_), None, length=inner_maxiter
            )
            return s

        # --- JIT: outer MLE scan (nu, gamma) ---
        @jax.jit
        def _run_outer_mle(nu, gamma, sigma_, adam_state, x_dash):
            def _scan_body(carry, _):
                n, g, a_s = carry
                raw_nu = _inv_softplus(jnp.maximum(n, eps))
                opt_arr = jnp.concatenate([raw_nu.reshape((1,)), g.flatten()])
                opt_arr, a_s = _adam_gradient_step(
                    lambda arr: copula_nll_fn(arr, sigma_, x_dash),
                    opt_arr, a_s, lr,
                )
                n = jnn.softplus(opt_arr[0]) + eps
                g = opt_arr[1:].reshape((d, 1))
                return (n, g, a_s), None

            (nu, gamma, adam_state), _ = lax.scan(
                _scan_body,
                (nu, gamma, adam_state),
                None, length=shape_steps,
            )
            return nu, gamma, adam_state

        # --- MoM initialization ---
        R_inv = jnp.linalg.inv(sigma)
        nu = jnp.clip(mom_nu_student_t(u, R_inv, d), 2.5, 60.0)

        # --- Python outer loop ---
        gamma = jnp.zeros((d, 1))
        adam_state = (jnp.zeros(1 + d), jnp.zeros(1 + d), jnp.array(0))
        _get_x_dash_jit = jax.jit(
            self.get_x_dash, static_argnames=("brent", "nodes")
        )
        prev_ll = -1e20
        no_improve_count = 0

        for _iter in range(maxiter):
            copula_params = self._mvt._params_dict(
                nu=nu, mu=mu, gamma=gamma, sigma=sigma,
            )
            full_params = {
                "marginals": dummy_marginals, "copula": copula_params,
            }
            x_dash = _get_x_dash_jit(
                u, full_params, brent=brent, nodes=nodes
            )

            # Early stopping (evaluate with fresh x_dash + current params)
            if tol > 0:
                current_ll = float(copula_ll_fn(
                    x_dash, nu, gamma, sigma
                ))
                rel_imp = (current_ll - prev_ll) / max(abs(float(prev_ll)), 1.0)
                if rel_imp < tol:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                if no_improve_count >= patience:
                    break
                prev_ll = current_ll

            sigma = _run_inner_em(gamma, sigma, x_dash, nu)

            adam_state = _reset_adam_state(adam_state)
            nu, gamma, adam_state = _run_outer_mle(
                nu, gamma, sigma, adam_state, x_dash
            )

        return self._mvt._params_dict(
            nu=nu, mu=mu, gamma=gamma, sigma=sigma,
        )

    def _fit_copula_mle(
        self, u, sigma, d, lr, maxiter, tol=1e-6, patience=5,
        brent: bool = False, nodes: int = 100,
    ):
        r"""Full MLE for the Skewed-T copula.

        All parameters (ν, γ, P) optimised via gradient descent
        on the copula LL.  P is represented via ``d*(d-1)/2``
        off-diagonal entries mapped through ``tanh``.
        """
        eps = _EPS
        mu = jnp.zeros((d, 1))
        shape_steps: int = 10
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )
        tril_rows, tril_cols = jnp.tril_indices(d, k=-1)
        mvt = self._mvt
        uvt = self._uvt

        def _sigma_from_raw(raw_corr):
            rho = jnp.tanh(raw_corr)
            P = jnp.eye(d)
            P = P.at[tril_rows, tril_cols].set(rho)
            P = P.at[tril_cols, tril_rows].set(rho)
            P = _corr._rm_incomplete(P, 1e-5)
            P = _corr._corr_from_cov(P)
            P = _corr._ensure_valid(P)
            return P

        def _raw_from_sigma(sigma_):
            rho = sigma_[tril_rows, tril_cols]
            return jnp.arctanh(jnp.clip(rho, -0.999, 0.999))

        copula_ll_fn = self._st_copula_ll(d, mu)

        @jax.jit
        def _run_mle_steps(opt_arr, adam_state, x_dash):
            def _copula_nll(arr):
                n_val = jnn.softplus(arr[0]) + eps
                g = arr[1:1 + d].reshape((d, 1))
                sigma_ = _sigma_from_raw(arr[1 + d:])
                copula_p = mvt._params_dict(
                    nu=n_val, mu=mu, gamma=g, sigma=sigma_,
                )
                mvt_ll = mvt.logpdf(x_dash, params=copula_p)
                uvt_params = {
                    "nu": jnp.full(d, n_val),
                    "mu": jnp.zeros(d),
                    "sigma": jnp.ones(d),
                    "gamma": g.flatten(),
                }
                uvt_ll = vmap(
                    lambda xi, pr: uvt.logpdf(xi, params=pr),
                    in_axes=(1, 0), out_axes=1,
                )(x_dash, uvt_params).sum(axis=1, keepdims=True)
                logpdf = mvt_ll - uvt_ll
                n = logpdf.shape[0]
                finite_mask = jnp.isfinite(logpdf)
                safe = jnp.where(finite_mask, logpdf, 0.0)
                return -safe.sum() / n + 1e6 * (~finite_mask).sum() / n

            def _scan_body(carry, _):
                arr, a_s = carry
                arr, a_s = _adam_gradient_step(_copula_nll, arr, a_s, lr)
                return (arr, a_s), None

            (opt_arr, adam_state), _ = lax.scan(
                _scan_body, (opt_arr, adam_state), None, length=shape_steps
            )
            return opt_arr, adam_state

        # --- MoM initialization ---
        R_inv = jnp.linalg.inv(sigma)
        nu = jnp.clip(mom_nu_student_t(u, R_inv, d), 2.5, 60.0)

        # --- Python outer loop ---
        gamma = jnp.zeros((d, 1))
        n_corr = d * (d - 1) // 2
        n_opt = 1 + d + n_corr
        adam_state = (jnp.zeros(n_opt), jnp.zeros(n_opt), jnp.array(0))
        _get_x_dash_jit = jax.jit(
            self.get_x_dash, static_argnames=("brent", "nodes")
        )
        prev_ll = -1e20
        no_improve_count = 0

        for _iter in range(maxiter):
            copula_params = self._mvt._params_dict(
                nu=nu, mu=mu, gamma=gamma, sigma=sigma,
            )
            full_params = {
                "marginals": dummy_marginals, "copula": copula_params,
            }
            x_dash = _get_x_dash_jit(
                u, full_params, brent=brent, nodes=nodes
            )

            # Early stopping (evaluate with fresh x_dash + current params)
            if tol > 0:
                current_ll = float(copula_ll_fn(
                    x_dash, nu, gamma, sigma
                ))
                rel_imp = (current_ll - prev_ll) / max(abs(float(prev_ll)), 1.0)
                if rel_imp < tol:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                if no_improve_count >= patience:
                    break
                prev_ll = current_ll

            raw_nu = _inv_softplus(jnp.maximum(nu, eps))
            raw_corr = _raw_from_sigma(sigma)
            opt_arr = jnp.concatenate([
                raw_nu.reshape((1,)),
                gamma.flatten(),
                raw_corr,
            ])

            adam_state = _reset_adam_state(adam_state)
            opt_arr, adam_state = _run_mle_steps(opt_arr, adam_state, x_dash)

            nu = jnn.softplus(opt_arr[0]) + eps
            gamma = opt_arr[1:1 + d].reshape((d, 1))
            sigma = _sigma_from_raw(opt_arr[1 + d:])

        return self._mvt._params_dict(
            nu=nu, mu=mu, gamma=gamma, sigma=sigma,
        )


skewed_t_copula = SkewedTCopula("Skewed-T-Copula", mvt_skewed_t, skewed_t)
