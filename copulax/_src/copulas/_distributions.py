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

from copulax._src.multivariate.mvt_normal import mvt_normal
from copulax._src.univariate.normal import normal
from copulax._src.multivariate.mvt_student_t import mvt_student_t, MvtStudentT
from copulax._src.univariate.student_t import student_t
from copulax._src.multivariate.mvt_gh import mvt_gh, MvtGH
from copulax._src.univariate.gh import gh, GH
from copulax._src.multivariate.mvt_skewed_t import mvt_skewed_t, MvtSkewedT
from copulax._src.univariate.skewed_t import skewed_t
from collections import defaultdict

# Module-level constants for copula parameter constraints
_NU_EPS: float = 1e-6
_POS_EPS: float = 1e-8


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
        cubic: bool = True,
    ) -> Array:
        r"""Sample from the joint distribution.

        1. Sample u from copula
        2. Transform u to x via marginal PPFs

        Args:
            size: Number of samples.
            params: Distribution parameters.
            key: JAX random key.
            cubic: Whether to use cubic spline PPF approximation.

        Returns:
            Array of shape (size, d).
        """
        key = _resolve_key(key)
        params = self._resolve_params(params)
        u_raw: jnp.ndarray = self.copula_rvs(size=size, params=params, key=key)
        eps: float = 1e-4
        u: jnp.ndarray = jnp.clip(u_raw, eps, 1 - eps)
        return self._grouped_marginal_apply("ppf", u, params["marginals"], cubic=cubic)

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
            **kwargs: Additional arguments forwarded to fit_copula.

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
# Copula Class (elliptical copulas)
###############################################################################
class Copula(CopulaBase):
    r"""Base class for copula distributions."""

    _mvt: Multivariate
    _uvt: Univariate

    _PARAM_KEY_TO_KWARG = {"copula": "copula_params"}

    # initialisation
    def __init__(
        self,
        name,
        mvt: Multivariate,
        uvt: Univariate,
        *,
        marginals=None,
        copula_params=None,
    ):
        super().__init__(name)
        self._mvt: Multivariate = mvt  # multivariate pytree object
        self._uvt: Univariate = uvt  # univariate pytree object
        self._marginals = marginals if marginals is not None else None
        self._copula_params = copula_params if copula_params is not None else None

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
        key_map = getattr(cls, "_PARAM_KEY_TO_KWARG", {})
        kwargs = {key_map.get(k, k): v for k, v in params_dict.items()}
        return cls(name, self._mvt, self._uvt, **kwargs)

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

    def get_x_dash(self, u: ArrayLike, params: dict, cubic: bool = True) -> Array:
        r"""Computes x' values, which represent the mappings of the
        independent marginal cdf values (U) to the domain of the joint
        multivariate distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'cubic'
            is a static argument.

        Args:
            u (ArrayLike): The independent univariate marginal cdf
                values (U) for each dimension.
            params (dict): The copula and marginal distribution
                parameters.
            cubic (bool): Whether to use a cubic spline approximation
                of the univariate ppf function for faster computation.
                This can also improve gradient estimates.

        Returns:
            x_dash (Array): The x' values for each dimension.
        """
        u_raw: jnp.ndarray = _multivariate_input(u)[0]
        eps: float = 1e-4
        u: jnp.ndarray = jnp.clip(u_raw, eps, 1 - eps)
        return self._scan_uvt_func(func=self._uvt.ppf, x=u, params=params, cubic=cubic)

    # densities
    def copula_logpdf(
        self, u: ArrayLike, params: dict = None, cubic: bool = True
    ) -> Array:
        r"""Computes the log-pdf of the copula distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'cubic'
            is a static argument.

        Args:
            u (ArrayLike): The independent univariate marginal cdf
                values (u) for each dimension.
            params (dict): The copula and marginal distribution
                parameters.
            cubic (bool): Whether to use a cubic spline approximation
                of the univariate ppf function for faster computation.
                This can also improve gradient estimates.

        Returns:
            logpdf (Array): The log-pdf values of the copula
                distribution.
        """
        # mapping u to x' space
        params = self._resolve_params(params)
        x_dash: jnp.ndarray = self.get_x_dash(u, params, cubic=cubic)

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
        finite_mask = jnp.isfinite(logpdf)
        safe_logpdf = jnp.where(finite_mask, logpdf, 0.0)
        n_invalid = (~finite_mask).astype(float).sum()
        return -safe_logpdf.sum() + 1e6 * n_invalid

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
        method: str = "ml",
        lr: float = 1e-3,
        maxiter: int = 200,
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
        maximising the copula log-likelihood with *P* fixed.

        Args:
            u: Pseudo-observations of shape ``(n, d)`` in ``[0, 1]``.
            corr_method: Correlation estimation method for Stage 1.
                Default ``'rm_pp_kendall'``. See
                ``copulax.multivariate.corr`` for all methods.
            method: Fitting algorithm for Stage 2.
                ``'ml'`` — projected gradient on copula NLL (all copulas).
                ``'em'`` — ECME algorithm (Skewed-T and GH only).
            lr: Learning rate for optimisation.
            maxiter: Maximum number of iterations.

        Returns:
            dict with key ``'copula'`` containing fitted parameters.
        """
        u_arr, _, n, d = _multivariate_input(u)

        # Stage 1: estimate correlation matrix P
        sigma: jnp.ndarray = self._estimate_copula_correlation(
            u_arr, corr_method
        )

        # Stage 2: estimate remaining parameters
        if method == "em":
            copula_params = self._fit_copula_em(u_arr, sigma, d, lr, maxiter)
        elif method == "em2":
            copula_params = self._fit_copula_em2(u_arr, sigma, d, lr, maxiter)
        elif method == "em2_lowlr":
            copula_params = self._fit_copula_em2(u_arr, sigma, d, lr * 0.1, maxiter)
        elif method == "em3":
            copula_params = self._fit_copula_em3(u_arr, sigma, d, lr, maxiter)
        elif method == "mle":
            copula_params = self._fit_copula_full_mle(u_arr, sigma, d, lr, maxiter)
        else:
            copula_params = self._fit_copula_ml(u_arr, sigma, d, lr, maxiter)

        return {"copula": copula_params}

    def _fit_copula_ml(
        self,
        u: jnp.ndarray,
        sigma: jnp.ndarray,
        d: int,
        lr: float,
        maxiter: int,
    ) -> dict:
        r"""ML-based copula fitting (Stage 2).

        For the Gaussian copula this returns the correlation matrix
        directly. For other copulas it optimises additional parameters.
        """
        return self._build_initial_copula_params(d, sigma)

    def _fit_copula_em(
        self,
        u: jnp.ndarray,
        sigma: jnp.ndarray,
        d: int,
        lr: float,
        maxiter: int,
    ) -> dict:
        r"""EM-based copula fitting (Stage 2).

        Only supported for Skewed-T and GH copulas. Raises
        ``NotImplementedError`` by default.
        """
        raise NotImplementedError(
            f"EM fitting is not supported for {type(self).__name__}. "
            f"Use method='ml' instead."
        )

    def _fit_copula_em2(
        self,
        u: jnp.ndarray,
        sigma: jnp.ndarray,
        d: int,
        lr: float,
        maxiter: int,
    ) -> dict:
        r"""EM-MLE variant 2: γ in both inner EM and outer MLE.

        Only supported for Skewed-T and GH copulas. Raises
        ``NotImplementedError`` by default.
        """
        raise NotImplementedError(
            f"EM2 fitting is not supported for {type(self).__name__}. "
            f"Use method='ml' or method='em' instead."
        )

    def _fit_copula_em3(
        self,
        u: jnp.ndarray,
        sigma: jnp.ndarray,
        d: int,
        lr: float,
        maxiter: int,
    ) -> dict:
        r"""EM-MLE variant 3: inner EM updates P only, outer MLE
        optimises all non-P params (shape + γ).

        Only supported for Skewed-T and GH copulas. Raises
        ``NotImplementedError`` by default.
        """
        raise NotImplementedError(
            f"EM3 fitting is not supported for {type(self).__name__}. "
            f"Use method='ml' or method='em' instead."
        )

    def _fit_copula_full_mle(
        self,
        u: jnp.ndarray,
        sigma: jnp.ndarray,
        d: int,
        lr: float,
        maxiter: int,
    ) -> dict:
        r"""Full MLE: all params including P optimised via gradient
        descent on copula LL.  After each step sigma is projected
        to a valid correlation matrix.

        Only supported for Skewed-T and GH copulas. Raises
        ``NotImplementedError`` by default.
        """
        raise NotImplementedError(
            f"Full MLE fitting is not supported for {type(self).__name__}. "
            f"Use method='ml' instead."
        )


###############################################################################
# Copula Distributions
###############################################################################
# Normal Mixture Copulas
class GaussianCopula(Copula):
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


class StudentTCopula(Copula):
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

    def _fit_copula_ml(self, u, sigma, d, lr, maxiter):
        return self._optimize_copula_params(u, sigma, d, lr, maxiter)


student_t_copula = StudentTCopula("Student-T-Copula", mvt_student_t, student_t)


class GHCopula(Copula):
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
        lamb: Scalar = params["copula"]["lambda"]
        chi: Scalar = params["copula"]["chi"]
        psi: Scalar = params["copula"]["psi"]
        gamma: Array = params["copula"]["gamma"]
        return {
            "lambda": jnp.full(d, lamb),
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
        # Optimise [lambda, raw_chi, raw_psi, gamma_1..gamma_d]
        params0 = jnp.concatenate([
            jnp.array([0.0]),                                    # lambda
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

    def _fit_copula_ml(self, u, sigma, d, lr, maxiter):
        return self._optimize_copula_params(u, sigma, d, lr, maxiter)

    def _fit_copula_em(self, u, sigma, d, lr, maxiter):
        r"""EM-MLE copula fitting for the GH copula.

        Alternates between:

        - **Inner EM** (on frozen x' and shape): updates P and γ via
          standard EM with mu=0.
        - **Outer CM-step**: gradient descent on (λ, χ, ψ) w.r.t. the
          copula log-likelihood evaluated on the frozen x'.

        Args:
            u: Pseudo-observations, shape ``(n, d)``.
            sigma: Initial correlation matrix from Stage 1.
            d: Dimensionality.
            lr: Learning rate for shape gradient steps.
            maxiter: Number of outer iterations.

        Returns:
            Fitted copula parameter dictionary.
        """
        eps: float = 1e-8
        mu = jnp.zeros((d, 1))
        inner_maxiter: int = 5
        shape_steps: int = 10
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )

        # --- JIT-compiled inner EM step (no PPF, fixed x' and shape) ---
        @jax.jit
        def _inner_em_step(carry, x, lamb, chi, psi):
            gamma, sigma_ = carry

            sigma_inv = jnp.linalg.inv(sigma_)
            Q = jnp.sum((x @ sigma_inv) * x, axis=1)
            R = (gamma.T @ sigma_inv @ gamma).squeeze()

            lam_post = lamb - d / 2.0
            chi_post = chi + Q
            psi_post = psi + R

            delta = jnp.clip(
                GH._gig_expected_inv_w(lam_post, chi_post, psi_post),
                eps, 1e10,
            )
            eta = jnp.clip(
                GH._gig_expected_w(lam_post, chi_post, psi_post),
                eps, 1e10,
            )

            delta_bar = jnp.mean(delta)
            eta_bar = jnp.mean(eta)
            x_bar = jnp.mean(x, axis=0).reshape((d, 1))

            eta_bar_safe = jnp.maximum(eta_bar, eps)
            gamma = jnp.clip(x_bar / eta_bar_safe, -2.0, 2.0)

            psi_mat = (
                jnp.mean(
                    delta[:, None, None] * (x[:, :, None] * x[:, None, :]),
                    axis=0,
                )
                - eta_bar * (gamma @ gamma.T)
            )
            psi_mat = _corr._rm_incomplete(psi_mat, 1e-5)
            sigma_ = _corr._corr_from_cov(psi_mat)
            sigma_ = _corr._ensure_valid(sigma_)

            return (gamma, sigma_)

        # --- JIT-compiled CM-step: copula LL gradient w.r.t. shape ---
        @jax.jit
        def _shape_cm_step(lamb, chi, psi, gamma, sigma_, adam_state, x):
            def _copula_nll_shape(shape_arr):
                l, c_, p_ = shape_arr[0], shape_arr[1], shape_arr[2]
                c = jnn.softplus(c_) + eps
                p = jnn.softplus(p_) + eps
                copula_p = self._mvt._params_dict(
                    lamb=l, chi=c, psi=p,
                    mu=mu, gamma=gamma, sigma=sigma_,
                )
                mvt_ll = self._mvt.logpdf(x, params=copula_p)
                uvt_params = {
                    "lambda": jnp.full(d, l),
                    "chi": jnp.full(d, c),
                    "psi": jnp.full(d, p),
                    "mu": jnp.zeros(d),
                    "sigma": jnp.ones(d),
                    "gamma": gamma.flatten(),
                }
                uvt_ll = vmap(
                    lambda xi, p: self._uvt.logpdf(xi, params=p),
                    in_axes=(1, 0), out_axes=1,
                )(x, uvt_params).sum(axis=1, keepdims=True)
                logpdf = mvt_ll - uvt_ll
                finite_mask = jnp.isfinite(logpdf)
                safe = jnp.where(finite_mask, logpdf, 0.0)
                return -safe.sum() + 1e6 * (~finite_mask).sum()

            raw_chi = jnp.log(jnp.expm1(jnp.maximum(chi, eps)))
            raw_psi = jnp.log(jnp.expm1(jnp.maximum(psi, eps)))
            shape_arr = jnp.array([lamb, raw_chi, raw_psi])

            _, g = jax.value_and_grad(_copula_nll_shape)(shape_arr)
            g = jnp.nan_to_num(g, nan=0.0)
            g = jnp.clip(g, -1.0, 1.0)
            m, v, t = adam_state
            direction, m, v, t = adam(g, m, v, t)
            shape_arr = shape_arr - lr * direction

            lamb = jnp.clip(shape_arr[0], -10.0, 10.0)
            chi = jnp.clip(jnn.softplus(shape_arr[1]) + eps, eps, 100.0)
            psi = jnp.clip(jnn.softplus(shape_arr[2]) + eps, eps, 100.0)
            return lamb, chi, psi, (m, v, t)

        # --- Python outer loop ---
        lamb = jnp.array(0.0)
        chi = jnp.array(1.0)
        psi = jnp.array(1.0)
        gamma = jnp.zeros((d, 1))
        _get_x_dash_jit = jax.jit(self.get_x_dash, static_argnames=("cubic",))

        for _ in range(maxiter):
            copula_params = self._mvt._params_dict(
                lamb=lamb, chi=chi, psi=psi,
                mu=mu, gamma=gamma, sigma=sigma,
            )
            full_params = {
                "marginals": dummy_marginals, "copula": copula_params,
            }
            x_dash = _get_x_dash_jit(u, full_params, cubic=True)

            carry = (gamma, sigma)
            for _ in range(inner_maxiter):
                carry = _inner_em_step(carry, x_dash, lamb, chi, psi)
            gamma, sigma = carry

            # Adam state reset each outer iter (x' changed)
            _as = (jnp.zeros(3), jnp.zeros(3), 0)
            for _ in range(shape_steps):
                lamb, chi, psi, _as = _shape_cm_step(
                    lamb, chi, psi, gamma, sigma, _as, x_dash
                )

        return self._mvt._params_dict(
            lamb=lamb, chi=chi, psi=psi,
            mu=mu, gamma=gamma, sigma=sigma,
        )

    def _fit_copula_em2(self, u, sigma, d, lr, maxiter):
        r"""EM-MLE copula fitting for the GH copula (variant 2).

        Like ``_fit_copula_em`` but the outer MLE step optimises
        (λ, χ, ψ, **γ**) jointly, giving the gradient-based optimiser
        direct control over skewness.  The inner EM still updates
        both γ and P, providing a warm-start for each outer step.

        Args:
            u: Pseudo-observations, shape ``(n, d)``.
            sigma: Initial correlation matrix from Stage 1.
            d: Dimensionality.
            lr: Learning rate for outer MLE gradient steps.
            maxiter: Number of outer iterations.

        Returns:
            Fitted copula parameter dictionary.
        """
        eps: float = 1e-8
        mu = jnp.zeros((d, 1))
        inner_maxiter: int = 5
        shape_steps: int = 10
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )

        # --- JIT-compiled inner EM step (same as _fit_copula_em) ---
        @jax.jit
        def _inner_em_step(carry, x, lamb, chi, psi):
            gamma, sigma_ = carry

            sigma_inv = jnp.linalg.inv(sigma_)
            Q = jnp.sum((x @ sigma_inv) * x, axis=1)
            R = (gamma.T @ sigma_inv @ gamma).squeeze()

            lam_post = lamb - d / 2.0
            chi_post = chi + Q
            psi_post = psi + R

            delta = jnp.clip(
                GH._gig_expected_inv_w(lam_post, chi_post, psi_post),
                eps, 1e10,
            )
            eta = jnp.clip(
                GH._gig_expected_w(lam_post, chi_post, psi_post),
                eps, 1e10,
            )

            delta_bar = jnp.mean(delta)
            eta_bar = jnp.mean(eta)
            x_bar = jnp.mean(x, axis=0).reshape((d, 1))

            eta_bar_safe = jnp.maximum(eta_bar, eps)
            gamma = jnp.clip(x_bar / eta_bar_safe, -2.0, 2.0)

            psi_mat = (
                jnp.mean(
                    delta[:, None, None] * (x[:, :, None] * x[:, None, :]),
                    axis=0,
                )
                - eta_bar * (gamma @ gamma.T)
            )
            psi_mat = _corr._rm_incomplete(psi_mat, 1e-5)
            sigma_ = _corr._corr_from_cov(psi_mat)
            sigma_ = _corr._ensure_valid(sigma_)

            return (gamma, sigma_)

        # --- JIT-compiled outer MLE: copula LL grad w.r.t. (λ,χ,ψ,γ) ---
        @jax.jit
        def _outer_mle_step(lamb, chi, psi, gamma, sigma_, adam_state, x):
            def _copula_nll(opt_arr):
                l = opt_arr[0]
                c = jnn.softplus(opt_arr[1]) + eps
                p = jnn.softplus(opt_arr[2]) + eps
                g = opt_arr[3:].reshape((d, 1))
                copula_p = self._mvt._params_dict(
                    lamb=l, chi=c, psi=p,
                    mu=mu, gamma=g, sigma=sigma_,
                )
                mvt_ll = self._mvt.logpdf(x, params=copula_p)
                uvt_params = {
                    "lambda": jnp.full(d, l),
                    "chi": jnp.full(d, c),
                    "psi": jnp.full(d, p),
                    "mu": jnp.zeros(d),
                    "sigma": jnp.ones(d),
                    "gamma": g.flatten(),
                }
                uvt_ll = vmap(
                    lambda xi, p: self._uvt.logpdf(xi, params=p),
                    in_axes=(1, 0), out_axes=1,
                )(x, uvt_params).sum(axis=1, keepdims=True)
                logpdf = mvt_ll - uvt_ll
                finite_mask = jnp.isfinite(logpdf)
                safe = jnp.where(finite_mask, logpdf, 0.0)
                return -safe.sum() + 1e6 * (~finite_mask).sum()

            raw_chi = jnp.log(jnp.expm1(jnp.maximum(chi, eps)))
            raw_psi = jnp.log(jnp.expm1(jnp.maximum(psi, eps)))
            opt_arr = jnp.concatenate([
                jnp.array([lamb, raw_chi, raw_psi]),
                gamma.flatten(),
            ])

            _, grad = jax.value_and_grad(_copula_nll)(opt_arr)
            grad = jnp.nan_to_num(grad, nan=0.0)
            grad = jnp.clip(grad, -1.0, 1.0)
            m, v, t = adam_state
            direction, m, v, t = adam(grad, m, v, t)
            opt_arr = opt_arr - lr * direction

            lamb = jnp.clip(opt_arr[0], -10.0, 10.0)
            chi = jnp.clip(jnn.softplus(opt_arr[1]) + eps, eps, 100.0)
            psi = jnp.clip(jnn.softplus(opt_arr[2]) + eps, eps, 100.0)
            gamma = opt_arr[3:].reshape((d, 1))
            return lamb, chi, psi, gamma, (m, v, t)

        # --- Python outer loop ---
        lamb = jnp.array(0.0)
        chi = jnp.array(1.0)
        psi = jnp.array(1.0)
        gamma = jnp.zeros((d, 1))
        _get_x_dash_jit = jax.jit(self.get_x_dash, static_argnames=("cubic",))

        for _ in range(maxiter):
            copula_params = self._mvt._params_dict(
                lamb=lamb, chi=chi, psi=psi,
                mu=mu, gamma=gamma, sigma=sigma,
            )
            full_params = {
                "marginals": dummy_marginals, "copula": copula_params,
            }
            x_dash = _get_x_dash_jit(u, full_params, cubic=True)

            carry = (gamma, sigma)
            for _ in range(inner_maxiter):
                carry = _inner_em_step(carry, x_dash, lamb, chi, psi)
            gamma, sigma = carry

            _as = (jnp.zeros(3 + d), jnp.zeros(3 + d), 0)
            for _ in range(shape_steps):
                lamb, chi, psi, gamma, _as = _outer_mle_step(
                    lamb, chi, psi, gamma, sigma, _as, x_dash
                )

        return self._mvt._params_dict(
            lamb=lamb, chi=chi, psi=psi,
            mu=mu, gamma=gamma, sigma=sigma,
        )


    def _fit_copula_em3(self, u, sigma, d, lr, maxiter):
        r"""EM-MLE variant 3 for the GH copula.

        Inner EM updates **P only** (γ frozen). Outer MLE optimises
        (λ, χ, ψ, γ) jointly via gradient descent on the copula LL.
        """
        eps: float = 1e-8
        mu = jnp.zeros((d, 1))
        inner_maxiter: int = 5
        shape_steps: int = 10
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )

        # --- Inner EM: updates P only (gamma frozen) ---
        @jax.jit
        def _inner_em_step(sigma_, x, lamb, chi, psi, gamma):
            sigma_inv = jnp.linalg.inv(sigma_)
            Q = jnp.sum((x @ sigma_inv) * x, axis=1)
            R = (gamma.T @ sigma_inv @ gamma).squeeze()

            lam_post = lamb - d / 2.0
            chi_post = chi + Q
            psi_post = psi + R

            delta = jnp.clip(
                GH._gig_expected_inv_w(lam_post, chi_post, psi_post),
                eps, 1e10,
            )
            eta = jnp.clip(
                GH._gig_expected_w(lam_post, chi_post, psi_post),
                eps, 1e10,
            )

            eta_bar = jnp.mean(eta)

            # Sigma update only (gamma is frozen)
            psi_mat = (
                jnp.mean(
                    delta[:, None, None] * (x[:, :, None] * x[:, None, :]),
                    axis=0,
                )
                - eta_bar * (gamma @ gamma.T)
            )
            psi_mat = _corr._rm_incomplete(psi_mat, 1e-5)
            sigma_ = _corr._corr_from_cov(psi_mat)
            sigma_ = _corr._ensure_valid(sigma_)

            return sigma_

        # --- Outer MLE: copula LL grad w.r.t. (λ,χ,ψ,γ) ---
        @jax.jit
        def _outer_mle_step(lamb, chi, psi, gamma, sigma_, adam_state, x):
            def _copula_nll(opt_arr):
                l = opt_arr[0]
                c = jnn.softplus(opt_arr[1]) + eps
                p = jnn.softplus(opt_arr[2]) + eps
                g = opt_arr[3:].reshape((d, 1))
                copula_p = self._mvt._params_dict(
                    lamb=l, chi=c, psi=p,
                    mu=mu, gamma=g, sigma=sigma_,
                )
                mvt_ll = self._mvt.logpdf(x, params=copula_p)
                uvt_params = {
                    "lambda": jnp.full(d, l),
                    "chi": jnp.full(d, c),
                    "psi": jnp.full(d, p),
                    "mu": jnp.zeros(d),
                    "sigma": jnp.ones(d),
                    "gamma": g.flatten(),
                }
                uvt_ll = vmap(
                    lambda xi, p: self._uvt.logpdf(xi, params=p),
                    in_axes=(1, 0), out_axes=1,
                )(x, uvt_params).sum(axis=1, keepdims=True)
                logpdf = mvt_ll - uvt_ll
                finite_mask = jnp.isfinite(logpdf)
                safe = jnp.where(finite_mask, logpdf, 0.0)
                return -safe.sum() + 1e6 * (~finite_mask).sum()

            raw_chi = jnp.log(jnp.expm1(jnp.maximum(chi, eps)))
            raw_psi = jnp.log(jnp.expm1(jnp.maximum(psi, eps)))
            opt_arr = jnp.concatenate([
                jnp.array([lamb, raw_chi, raw_psi]),
                gamma.flatten(),
            ])

            _, grad = jax.value_and_grad(_copula_nll)(opt_arr)
            grad = jnp.nan_to_num(grad, nan=0.0)
            grad = jnp.clip(grad, -1.0, 1.0)
            m, v, t = adam_state
            direction, m, v, t = adam(grad, m, v, t)
            opt_arr = opt_arr - lr * direction

            lamb = jnp.clip(opt_arr[0], -10.0, 10.0)
            chi = jnp.clip(jnn.softplus(opt_arr[1]) + eps, eps, 100.0)
            psi = jnp.clip(jnn.softplus(opt_arr[2]) + eps, eps, 100.0)
            gamma = opt_arr[3:].reshape((d, 1))
            return lamb, chi, psi, gamma, (m, v, t)

        # --- Python outer loop ---
        lamb = jnp.array(0.0)
        chi = jnp.array(1.0)
        psi = jnp.array(1.0)
        gamma = jnp.zeros((d, 1))
        _get_x_dash_jit = jax.jit(self.get_x_dash, static_argnames=("cubic",))

        for _ in range(maxiter):
            copula_params = self._mvt._params_dict(
                lamb=lamb, chi=chi, psi=psi,
                mu=mu, gamma=gamma, sigma=sigma,
            )
            full_params = {
                "marginals": dummy_marginals, "copula": copula_params,
            }
            x_dash = _get_x_dash_jit(u, full_params, cubic=True)

            # Inner EM: P only
            for _ in range(inner_maxiter):
                sigma = _inner_em_step(sigma, x_dash, lamb, chi, psi, gamma)

            # Outer MLE: (λ,χ,ψ,γ)
            _as = (jnp.zeros(3 + d), jnp.zeros(3 + d), 0)
            for _ in range(shape_steps):
                lamb, chi, psi, gamma, _as = _outer_mle_step(
                    lamb, chi, psi, gamma, sigma, _as, x_dash
                )

        return self._mvt._params_dict(
            lamb=lamb, chi=chi, psi=psi,
            mu=mu, gamma=gamma, sigma=sigma,
        )

    def _fit_copula_full_mle(self, u, sigma, d, lr, maxiter):
        r"""Full MLE for the GH copula.

        All parameters (λ, χ, ψ, γ, P) optimised via gradient descent
        on the copula LL.  P is represented via its ``d*(d-1)/2``
        free off-diagonal entries mapped through ``tanh`` to stay in
        ``(-1, 1)``.  After each step the matrix is rebuilt and
        projected to a valid correlation matrix.

        x' is re-transformed from u at each outer iteration.
        """
        eps: float = 1e-8
        mu = jnp.zeros((d, 1))
        shape_steps: int = 10
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )
        n_corr = d * (d - 1) // 2
        tril_rows, tril_cols = jnp.tril_indices(d, k=-1)

        def _sigma_from_raw(raw_corr):
            """Rebuild correlation matrix from raw off-diagonal params."""
            rho = jnp.tanh(raw_corr)
            P = jnp.eye(d)
            P = P.at[tril_rows, tril_cols].set(rho)
            P = P.at[tril_cols, tril_rows].set(rho)
            # PSD repair
            P = _corr._rm_incomplete(P, 1e-5)
            P = _corr._corr_from_cov(P)
            P = _corr._ensure_valid(P)
            return P

        def _raw_from_sigma(sigma_):
            """Extract raw off-diagonal params from correlation matrix."""
            rho = sigma_[tril_rows, tril_cols]
            return jnp.arctanh(jnp.clip(rho, -0.999, 0.999))

        @jax.jit
        def _mle_step(opt_arr, adam_state, x):
            def _copula_nll(arr):
                l = arr[0]
                c = jnn.softplus(arr[1]) + eps
                p = jnn.softplus(arr[2]) + eps
                g = arr[3:3 + d].reshape((d, 1))
                raw_c = arr[3 + d:]
                sigma_ = _sigma_from_raw(raw_c)
                copula_p = self._mvt._params_dict(
                    lamb=l, chi=c, psi=p,
                    mu=mu, gamma=g, sigma=sigma_,
                )
                mvt_ll = self._mvt.logpdf(x, params=copula_p)
                uvt_params = {
                    "lambda": jnp.full(d, l),
                    "chi": jnp.full(d, c),
                    "psi": jnp.full(d, p),
                    "mu": jnp.zeros(d),
                    "sigma": jnp.ones(d),
                    "gamma": g.flatten(),
                }
                uvt_ll = vmap(
                    lambda xi, p: self._uvt.logpdf(xi, params=p),
                    in_axes=(1, 0), out_axes=1,
                )(x, uvt_params).sum(axis=1, keepdims=True)
                logpdf = mvt_ll - uvt_ll
                finite_mask = jnp.isfinite(logpdf)
                safe = jnp.where(finite_mask, logpdf, 0.0)
                return -safe.sum() + 1e6 * (~finite_mask).sum()

            _, grad = jax.value_and_grad(_copula_nll)(opt_arr)
            grad = jnp.nan_to_num(grad, nan=0.0)
            grad = jnp.clip(grad, -1.0, 1.0)
            m, v, t = adam_state
            direction, m, v, t = adam(grad, m, v, t)
            return opt_arr - lr * direction, (m, v, t)

        # --- Python outer loop ---
        lamb = jnp.array(0.0)
        chi = jnp.array(1.0)
        psi = jnp.array(1.0)
        gamma = jnp.zeros((d, 1))
        _get_x_dash_jit = jax.jit(self.get_x_dash, static_argnames=("cubic",))

        for _ in range(maxiter):
            # Transform u → x'
            copula_params = self._mvt._params_dict(
                lamb=lamb, chi=chi, psi=psi,
                mu=mu, gamma=gamma, sigma=sigma,
            )
            full_params = {
                "marginals": dummy_marginals, "copula": copula_params,
            }
            x_dash = _get_x_dash_jit(u, full_params, cubic=True)

            # Build optimisation vector
            raw_chi = jnp.log(jnp.expm1(jnp.maximum(chi, eps)))
            raw_psi = jnp.log(jnp.expm1(jnp.maximum(psi, eps)))
            raw_corr = _raw_from_sigma(sigma)
            opt_arr = jnp.concatenate([
                jnp.array([lamb, raw_chi, raw_psi]),
                gamma.flatten(),
                raw_corr,
            ])

            # Gradient steps on ALL params
            n_opt = opt_arr.shape[0]
            _as = (jnp.zeros(n_opt), jnp.zeros(n_opt), 0)
            for _ in range(shape_steps):
                opt_arr, _as = _mle_step(opt_arr, _as, x_dash)

            # Reconstruct params
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


class SkewedTCopula(Copula):
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

    def _fit_copula_ml(self, u, sigma, d, lr, maxiter):
        return self._optimize_copula_params(u, sigma, d, lr, maxiter)

    def _fit_copula_em(self, u, sigma, d, lr, maxiter):
        r"""EM-MLE copula fitting for the Skewed-T copula.

        Alternates between:

        - **Inner EM** (on frozen x' and ν): updates P and γ via
          standard EM with mu=0.
        - **Outer CM-step**: gradient descent on ν w.r.t. the copula
          log-likelihood evaluated on the frozen x'.

        Args:
            u: Pseudo-observations, shape ``(n, d)``.
            sigma: Initial correlation matrix from Stage 1.
            d: Dimensionality.
            lr: Learning rate for nu gradient steps.
            maxiter: Number of outer iterations.

        Returns:
            Fitted copula parameter dictionary.
        """
        eps: float = 1e-8
        mu = jnp.zeros((d, 1))
        inner_maxiter: int = 5
        shape_steps: int = 10
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )

        # --- JIT-compiled inner EM step (no PPF, fixed x' and nu) ---
        @jax.jit
        def _inner_em_step(carry, x, nu):
            gamma, sigma_ = carry

            sigma_inv = jnp.linalg.inv(sigma_)
            Q = jnp.sum((x @ sigma_inv) * x, axis=1)
            R = (gamma.T @ sigma_inv @ gamma).squeeze()

            lam_post = -nu / 2.0 - d / 2.0
            chi_post = nu + Q
            psi_post = jnp.maximum(R, eps)

            delta = jnp.clip(
                GH._gig_expected_inv_w(lam_post, chi_post, psi_post),
                eps, 1e10,
            )
            eta = jnp.clip(
                GH._gig_expected_w(lam_post, chi_post, psi_post),
                eps, 1e10,
            )

            delta_bar = jnp.mean(delta)
            eta_bar = jnp.mean(eta)
            x_bar = jnp.mean(x, axis=0).reshape((d, 1))

            eta_bar_safe = jnp.maximum(eta_bar, eps)
            gamma = jnp.clip(x_bar / eta_bar_safe, -2.0, 2.0)

            psi_mat = (
                jnp.mean(
                    delta[:, None, None] * (x[:, :, None] * x[:, None, :]),
                    axis=0,
                )
                - eta_bar * (gamma @ gamma.T)
            )
            psi_mat = _corr._rm_incomplete(psi_mat, 1e-5)
            sigma_ = _corr._corr_from_cov(psi_mat)
            sigma_ = _corr._ensure_valid(sigma_)

            return (gamma, sigma_)

        # --- JIT-compiled CM-step: copula LL gradient w.r.t. nu ---
        @jax.jit
        def _shape_cm_step(nu, gamma, sigma_, adam_state, x):
            def _copula_nll_nu(raw_nu):
                n_val = jnn.softplus(raw_nu) + eps
                copula_p = self._mvt._params_dict(
                    nu=n_val, mu=mu, gamma=gamma, sigma=sigma_,
                )
                mvt_ll = self._mvt.logpdf(x, params=copula_p)
                uvt_params = {
                    "nu": jnp.full(d, n_val),
                    "mu": jnp.zeros(d),
                    "sigma": jnp.ones(d),
                    "gamma": gamma.flatten(),
                }
                uvt_ll = vmap(
                    lambda xi, p: self._uvt.logpdf(xi, params=p),
                    in_axes=(1, 0), out_axes=1,
                )(x, uvt_params).sum(axis=1, keepdims=True)
                logpdf = mvt_ll - uvt_ll
                finite_mask = jnp.isfinite(logpdf)
                safe = jnp.where(finite_mask, logpdf, 0.0)
                return -safe.sum() + 1e6 * (~finite_mask).sum()

            raw_nu = jnp.log(jnp.expm1(jnp.maximum(nu, eps)))
            _, g = jax.value_and_grad(_copula_nll_nu)(raw_nu)
            g = jnp.nan_to_num(g, nan=0.0)
            m, v, t = adam_state
            direction, m, v, t = adam(g.reshape((1,)), m, v, t)
            raw_nu = raw_nu - lr * direction[0]
            nu = jnn.softplus(raw_nu) + eps
            return nu, (m, v, t)

        # --- Python outer loop ---
        nu = jnp.array(5.0)
        gamma = jnp.zeros((d, 1))
        _get_x_dash_jit = jax.jit(self.get_x_dash, static_argnames=("cubic",))

        for _ in range(maxiter):
            # 1. Transform u → x' (forward only, JIT-cached)
            copula_params = self._mvt._params_dict(
                nu=nu, mu=mu, gamma=gamma, sigma=sigma,
            )
            full_params = {
                "marginals": dummy_marginals, "copula": copula_params,
            }
            x_dash = _get_x_dash_jit(u, full_params, cubic=True)

            # 2. Inner EM: update (gamma, sigma→P) on fixed x'
            carry = (gamma, sigma)
            for _ in range(inner_maxiter):
                carry = _inner_em_step(carry, x_dash, nu)
            gamma, sigma = carry

            # 3. CM-step: gradient steps on nu w.r.t. copula LL
            _as = (jnp.zeros(1), jnp.zeros(1), 0)
            for _ in range(shape_steps):
                nu, _as = _shape_cm_step(nu, gamma, sigma, _as, x_dash)

        return self._mvt._params_dict(
            nu=nu, mu=mu, gamma=gamma, sigma=sigma,
        )

    def _fit_copula_em2(self, u, sigma, d, lr, maxiter):
        r"""EM-MLE variant 2 for the Skewed-T copula.

        Outer MLE optimises (ν, γ) jointly. Inner EM updates both
        γ and P, providing a warm-start for each outer step.
        """
        eps: float = 1e-8
        mu = jnp.zeros((d, 1))
        inner_maxiter: int = 5
        shape_steps: int = 10
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )

        # --- JIT inner EM (same as _fit_copula_em) ---
        @jax.jit
        def _inner_em_step(carry, x, nu):
            gamma, sigma_ = carry

            sigma_inv = jnp.linalg.inv(sigma_)
            Q = jnp.sum((x @ sigma_inv) * x, axis=1)
            R = (gamma.T @ sigma_inv @ gamma).squeeze()

            lam_post = -nu / 2.0 - d / 2.0
            chi_post = nu + Q
            psi_post = jnp.maximum(R, eps)

            delta = jnp.clip(
                GH._gig_expected_inv_w(lam_post, chi_post, psi_post),
                eps, 1e10,
            )
            eta = jnp.clip(
                GH._gig_expected_w(lam_post, chi_post, psi_post),
                eps, 1e10,
            )

            delta_bar = jnp.mean(delta)
            eta_bar = jnp.mean(eta)
            x_bar = jnp.mean(x, axis=0).reshape((d, 1))

            eta_bar_safe = jnp.maximum(eta_bar, eps)
            gamma = jnp.clip(x_bar / eta_bar_safe, -2.0, 2.0)

            psi_mat = (
                jnp.mean(
                    delta[:, None, None] * (x[:, :, None] * x[:, None, :]),
                    axis=0,
                )
                - eta_bar * (gamma @ gamma.T)
            )
            psi_mat = _corr._rm_incomplete(psi_mat, 1e-5)
            sigma_ = _corr._corr_from_cov(psi_mat)
            sigma_ = _corr._ensure_valid(sigma_)

            return (gamma, sigma_)

        # --- JIT outer MLE: copula LL grad w.r.t. (ν, γ) ---
        @jax.jit
        def _outer_mle_step(nu, gamma, sigma_, adam_state, x):
            def _copula_nll(opt_arr):
                n_val = jnn.softplus(opt_arr[0]) + eps
                g = opt_arr[1:].reshape((d, 1))
                copula_p = self._mvt._params_dict(
                    nu=n_val, mu=mu, gamma=g, sigma=sigma_,
                )
                mvt_ll = self._mvt.logpdf(x, params=copula_p)
                uvt_params = {
                    "nu": jnp.full(d, n_val),
                    "mu": jnp.zeros(d),
                    "sigma": jnp.ones(d),
                    "gamma": g.flatten(),
                }
                uvt_ll = vmap(
                    lambda xi, p: self._uvt.logpdf(xi, params=p),
                    in_axes=(1, 0), out_axes=1,
                )(x, uvt_params).sum(axis=1, keepdims=True)
                logpdf = mvt_ll - uvt_ll
                finite_mask = jnp.isfinite(logpdf)
                safe = jnp.where(finite_mask, logpdf, 0.0)
                return -safe.sum() + 1e6 * (~finite_mask).sum()

            raw_nu = jnp.log(jnp.expm1(jnp.maximum(nu, eps)))
            opt_arr = jnp.concatenate([raw_nu.reshape((1,)), gamma.flatten()])

            _, grad = jax.value_and_grad(_copula_nll)(opt_arr)
            grad = jnp.nan_to_num(grad, nan=0.0)
            grad = jnp.clip(grad, -10.0, 10.0)
            m, v, t = adam_state
            direction, m, v, t = adam(grad, m, v, t)
            opt_arr = opt_arr - lr * direction

            nu = jnn.softplus(opt_arr[0]) + eps
            gamma = opt_arr[1:].reshape((d, 1))
            return nu, gamma, (m, v, t)

        # --- Python outer loop ---
        nu = jnp.array(5.0)
        gamma = jnp.zeros((d, 1))
        _get_x_dash_jit = jax.jit(self.get_x_dash, static_argnames=("cubic",))

        for _ in range(maxiter):
            copula_params = self._mvt._params_dict(
                nu=nu, mu=mu, gamma=gamma, sigma=sigma,
            )
            full_params = {
                "marginals": dummy_marginals, "copula": copula_params,
            }
            x_dash = _get_x_dash_jit(u, full_params, cubic=True)

            carry = (gamma, sigma)
            for _ in range(inner_maxiter):
                carry = _inner_em_step(carry, x_dash, nu)
            gamma, sigma = carry

            _as = (jnp.zeros(1 + d), jnp.zeros(1 + d), 0)
            for _ in range(shape_steps):
                nu, gamma, _as = _outer_mle_step(nu, gamma, sigma, _as, x_dash)

        return self._mvt._params_dict(
            nu=nu, mu=mu, gamma=gamma, sigma=sigma,
        )

    def _fit_copula_em3(self, u, sigma, d, lr, maxiter):
        r"""EM-MLE variant 3 for the Skewed-T copula.

        Inner EM updates **P only** (γ frozen). Outer MLE optimises
        (ν, γ) jointly via gradient descent on the copula LL.
        """
        eps: float = 1e-8
        mu = jnp.zeros((d, 1))
        inner_maxiter: int = 5
        shape_steps: int = 10
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )

        # --- Inner EM: updates P only (gamma frozen) ---
        @jax.jit
        def _inner_em_step(sigma_, x, nu, gamma):
            sigma_inv = jnp.linalg.inv(sigma_)
            Q = jnp.sum((x @ sigma_inv) * x, axis=1)
            R = (gamma.T @ sigma_inv @ gamma).squeeze()

            lam_post = -nu / 2.0 - d / 2.0
            chi_post = nu + Q
            psi_post = jnp.maximum(R, eps)

            delta = jnp.clip(
                GH._gig_expected_inv_w(lam_post, chi_post, psi_post),
                eps, 1e10,
            )
            eta = jnp.clip(
                GH._gig_expected_w(lam_post, chi_post, psi_post),
                eps, 1e10,
            )

            eta_bar = jnp.mean(eta)

            # Sigma update only (gamma frozen)
            psi_mat = (
                jnp.mean(
                    delta[:, None, None] * (x[:, :, None] * x[:, None, :]),
                    axis=0,
                )
                - eta_bar * (gamma @ gamma.T)
            )
            psi_mat = _corr._rm_incomplete(psi_mat, 1e-5)
            sigma_ = _corr._corr_from_cov(psi_mat)
            sigma_ = _corr._ensure_valid(sigma_)

            return sigma_

        # --- Outer MLE: copula LL grad w.r.t. (ν, γ) ---
        @jax.jit
        def _outer_mle_step(nu, gamma, sigma_, adam_state, x):
            def _copula_nll(opt_arr):
                n_val = jnn.softplus(opt_arr[0]) + eps
                g = opt_arr[1:].reshape((d, 1))
                copula_p = self._mvt._params_dict(
                    nu=n_val, mu=mu, gamma=g, sigma=sigma_,
                )
                mvt_ll = self._mvt.logpdf(x, params=copula_p)
                uvt_params = {
                    "nu": jnp.full(d, n_val),
                    "mu": jnp.zeros(d),
                    "sigma": jnp.ones(d),
                    "gamma": g.flatten(),
                }
                uvt_ll = vmap(
                    lambda xi, p: self._uvt.logpdf(xi, params=p),
                    in_axes=(1, 0), out_axes=1,
                )(x, uvt_params).sum(axis=1, keepdims=True)
                logpdf = mvt_ll - uvt_ll
                finite_mask = jnp.isfinite(logpdf)
                safe = jnp.where(finite_mask, logpdf, 0.0)
                return -safe.sum() + 1e6 * (~finite_mask).sum()

            raw_nu = jnp.log(jnp.expm1(jnp.maximum(nu, eps)))
            opt_arr = jnp.concatenate([raw_nu.reshape((1,)), gamma.flatten()])

            _, grad = jax.value_and_grad(_copula_nll)(opt_arr)
            grad = jnp.nan_to_num(grad, nan=0.0)
            grad = jnp.clip(grad, -10.0, 10.0)
            m, v, t = adam_state
            direction, m, v, t = adam(grad, m, v, t)
            opt_arr = opt_arr - lr * direction

            nu = jnn.softplus(opt_arr[0]) + eps
            gamma = opt_arr[1:].reshape((d, 1))
            return nu, gamma, (m, v, t)

        # --- Python outer loop ---
        nu = jnp.array(5.0)
        gamma = jnp.zeros((d, 1))
        _get_x_dash_jit = jax.jit(self.get_x_dash, static_argnames=("cubic",))

        for _ in range(maxiter):
            copula_params = self._mvt._params_dict(
                nu=nu, mu=mu, gamma=gamma, sigma=sigma,
            )
            full_params = {
                "marginals": dummy_marginals, "copula": copula_params,
            }
            x_dash = _get_x_dash_jit(u, full_params, cubic=True)

            # Inner EM: P only
            for _ in range(inner_maxiter):
                sigma = _inner_em_step(sigma, x_dash, nu, gamma)

            # Outer MLE: (ν, γ)
            _as = (jnp.zeros(1 + d), jnp.zeros(1 + d), 0)
            for _ in range(shape_steps):
                nu, gamma, _as = _outer_mle_step(nu, gamma, sigma, _as, x_dash)

        return self._mvt._params_dict(
            nu=nu, mu=mu, gamma=gamma, sigma=sigma,
        )

    def _fit_copula_full_mle(self, u, sigma, d, lr, maxiter):
        r"""Full MLE for the Skewed-T copula.

        All parameters (ν, γ, P) optimised via gradient descent
        on the copula LL.  P is represented via its ``d*(d-1)/2``
        free off-diagonal entries mapped through ``tanh``.
        """
        eps: float = 1e-8
        mu = jnp.zeros((d, 1))
        shape_steps: int = 10
        dummy_marginals = tuple(
            (self._uvt, self._uvt.example_params()) for _ in range(d)
        )
        n_corr = d * (d - 1) // 2
        tril_rows, tril_cols = jnp.tril_indices(d, k=-1)

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

        @jax.jit
        def _mle_step(opt_arr, adam_state, x):
            def _copula_nll(arr):
                n_val = jnn.softplus(arr[0]) + eps
                g = arr[1:1 + d].reshape((d, 1))
                sigma_ = _sigma_from_raw(arr[1 + d:])
                copula_p = self._mvt._params_dict(
                    nu=n_val, mu=mu, gamma=g, sigma=sigma_,
                )
                mvt_ll = self._mvt.logpdf(x, params=copula_p)
                uvt_params = {
                    "nu": jnp.full(d, n_val),
                    "mu": jnp.zeros(d),
                    "sigma": jnp.ones(d),
                    "gamma": g.flatten(),
                }
                uvt_ll = vmap(
                    lambda xi, p: self._uvt.logpdf(xi, params=p),
                    in_axes=(1, 0), out_axes=1,
                )(x, uvt_params).sum(axis=1, keepdims=True)
                logpdf = mvt_ll - uvt_ll
                finite_mask = jnp.isfinite(logpdf)
                safe = jnp.where(finite_mask, logpdf, 0.0)
                return -safe.sum() + 1e6 * (~finite_mask).sum()

            _, grad = jax.value_and_grad(_copula_nll)(opt_arr)
            grad = jnp.nan_to_num(grad, nan=0.0)
            grad = jnp.clip(grad, -10.0, 10.0)
            m, v, t = adam_state
            direction, m, v, t = adam(grad, m, v, t)
            return opt_arr - lr * direction, (m, v, t)

        # --- Python outer loop ---
        nu = jnp.array(5.0)
        gamma = jnp.zeros((d, 1))
        _get_x_dash_jit = jax.jit(self.get_x_dash, static_argnames=("cubic",))

        for _ in range(maxiter):
            copula_params = self._mvt._params_dict(
                nu=nu, mu=mu, gamma=gamma, sigma=sigma,
            )
            full_params = {
                "marginals": dummy_marginals, "copula": copula_params,
            }
            x_dash = _get_x_dash_jit(u, full_params, cubic=True)

            raw_nu = jnp.log(jnp.expm1(jnp.maximum(nu, eps)))
            raw_corr = _raw_from_sigma(sigma)
            opt_arr = jnp.concatenate([
                raw_nu.reshape((1,)),
                gamma.flatten(),
                raw_corr,
            ])

            n_opt = opt_arr.shape[0]
            _as = (jnp.zeros(n_opt), jnp.zeros(n_opt), 0)
            for _ in range(shape_steps):
                opt_arr, _as = _mle_step(opt_arr, _as, x_dash)

            nu = jnn.softplus(opt_arr[0]) + eps
            gamma = opt_arr[1:1 + d].reshape((d, 1))
            sigma = _sigma_from_raw(opt_arr[1 + d:])

        return self._mvt._params_dict(
            nu=nu, mu=mu, gamma=gamma, sigma=sigma,
        )


skewed_t_copula = SkewedTCopula("Skewed-T-Copula", mvt_skewed_t, skewed_t)
