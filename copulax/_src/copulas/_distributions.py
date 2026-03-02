"""CopulAX implementation of the popular Copula distributions."""

from abc import abstractmethod
from jax._src.typing import ArrayLike, Array
from typing import Callable
from jax import numpy as jnp
from jax import jit, vmap

from copulax._src._distributions import (
    GeneralMultivariate,
    Multivariate,
    Univariate,
)
from copulax.univariate import univariate_fitter
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src._utils import _resolve_key
from copulax._src.typing import Scalar

from copulax._src.multivariate.mvt_normal import mvt_normal
from copulax._src.univariate.normal import normal
from copulax._src.multivariate.mvt_student_t import mvt_student_t
from copulax._src.univariate.student_t import student_t
from copulax._src.multivariate.mvt_gh import mvt_gh
from copulax._src.univariate.gh import gh
from copulax._src.multivariate.mvt_skewed_t import mvt_skewed_t
from copulax._src.univariate.skewed_t import skewed_t


###############################################################################
# CopulaBase — shared logic for elliptical and Archimedean copulas
###############################################################################
class CopulaBase(GeneralMultivariate):
    r"""Base class for all copula distributions.

    Provides Sklar's theorem implementations for the joint distribution,
    common marginal-fitting logic, and sampling via inverse-transform
    of copula samples.
    """

    @property
    def dist_type(self) -> str:
        return 'copula'

    def _get_dim(self, params: dict) -> int:
        return len(params["marginals"])

    def support(self, params: dict) -> Array:
        r"""Support of the joint distribution."""
        marginals: tuple = params["marginals"]
        return jnp.vstack(
            [dist.support(params=mparams) for dist, mparams in marginals]
        )

    def get_u(self, x: ArrayLike, params: dict) -> Array:
        r"""Compute marginal CDF values u = (F_1(x_1), ..., F_d(x_d)).

        Args:
            x: Input data of shape (n, d).
            params: Distribution parameters with 'marginals' key.

        Returns:
            Array of shape (n, d) with values in [0, 1].
        """
        x_arr: jnp.ndarray = _multivariate_input(x)[0]
        marginals: tuple = params["marginals"]
        u = [dist.cdf(x_arr[:, i][:, None], params=mparams)
             for i, (dist, mparams) in enumerate(marginals)]
        return jnp.concat(u, axis=1)

    # --- copula densities (abstract) ---

    @abstractmethod
    def copula_logpdf(self, u: ArrayLike, params: dict, **kwargs) -> Array:
        r"""Log-density of the copula (subclasses must implement)."""

    def copula_pdf(self, u: ArrayLike, params: dict, **kwargs) -> Array:
        r"""Density of the copula: c(u) = exp(copula_logpdf(u))."""
        return jnp.exp(self.copula_logpdf(u, params, **kwargs))

    @abstractmethod
    def copula_rvs(
        self, size: Scalar, params: dict, key: Array = None
    ) -> Array:
        r"""Generate random samples from the copula (subclasses must implement)."""

    def copula_sample(
        self, size: Scalar, params: dict, key: Array = None
    ) -> Array:
        r"""Alias for copula_rvs."""
        return self.copula_rvs(size=size, params=params, key=key)

    # --- joint distribution (Sklar's theorem) ---

    def logpdf(self, x: ArrayLike, params: dict, **kwargs) -> Array:
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
        marginals: tuple = params["marginals"]
        marginal_logpdf_sum: jnp.ndarray = sum(
            jit(dist.logpdf)(x_arr[:, i][:, None], params=mparams)
            for i, (dist, mparams) in enumerate(marginals)
        )
        u: jnp.ndarray = self.get_u(x_arr, params)
        copula_lp: jnp.ndarray = self.copula_logpdf(u, params, **kwargs)
        return copula_lp + marginal_logpdf_sum

    def pdf(self, x: ArrayLike, params: dict, **kwargs) -> Array:
        r"""Joint PDF."""
        return jnp.exp(self.logpdf(x, params, **kwargs))

    # --- sampling ---

    def rvs(
        self,
        size: Scalar,
        params: dict,
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
        u_raw: jnp.ndarray = self.copula_rvs(
            size=size, params=params, key=key
        )
        eps: float = 1e-4
        u: jnp.ndarray = jnp.clip(u_raw, eps, 1 - eps)

        marginals: tuple = params["marginals"]
        x_cols = [
            jit(dist.ppf, static_argnames="cubic")(
                u[:, i][:, None], params=mparams, cubic=cubic
            )
            for i, (dist, mparams) in enumerate(marginals)
        ]
        return jnp.concat(x_cols, axis=1)

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
            raise ValueError(
                "univariate_fitter_options must be a tuple or dictionary."
            )

        jitted_ufitter: Callable = jit(
            univariate_fitter, static_argnames=("metric", "distributions")
        )
        marginals = []
        for i, options in enumerate(univariate_fitter_options):
            xi: jnp.ndarray = x_arr[:, i]
            best_index, fitted = jitted_ufitter(xi, **options)
            dist: Univariate = fitted[best_index]["dist"]
            params: dict = fitted[best_index]["params"]
            marginals.append((dist, params))

        return {"marginals": tuple(marginals)}

    @abstractmethod
    def fit_copula(self, u: ArrayLike, **kwargs) -> dict:
        r"""Fit copula parameters (subclasses must implement)."""

    def fit(
        self,
        x: ArrayLike,
        univariate_fitter_options: tuple[dict] | dict = None,
        **kwargs,
    ) -> dict:
        r"""Fit marginals and copula to the data.

        Equivalent to calling fit_marginals then fit_copula.

        Args:
            x: Input data of shape (n, d).
            univariate_fitter_options: Options for marginal fitting.
            **kwargs: Additional arguments forwarded to fit_copula.

        Note:
            Not jitable.

        Returns:
            dict with keys 'marginals' and 'copula'.
        """
        marginals: dict = self.fit_marginals(x, univariate_fitter_options)
        u: jnp.ndarray = self.get_u(x, marginals)
        copula: dict = self.fit_copula(u, **kwargs)
        return {**marginals, **copula}


###############################################################################
# Copula Class (elliptical copulas)
###############################################################################
class Copula(CopulaBase):
    r"""Base class for copula distributions."""
    _mvt: Multivariate
    _uvt: Univariate

    # initialisation
    def __init__(self, name, mvt: Multivariate, uvt: Univariate):
        super().__init__(name)
        self._mvt: Multivariate = mvt  # multivariate pytree object
        self._uvt: Univariate = uvt  # univariate pytree object

    def _params_to_tuple(self, params: dict) -> tuple:
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
            return func(xi_col[:, None], params=p_slice, **kwargs).squeeze(-1)

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
        uvt_ppf: Callable = jit(self._uvt.ppf, static_argnames=("cubic",))
        return self._scan_uvt_func(func=uvt_ppf, x=u, params=params, cubic=cubic)

    # densities
    def copula_logpdf(self, u: ArrayLike, params: dict, cubic: bool = True) -> Array:
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
        x_dash: jnp.ndarray = self.get_x_dash(u, params, cubic=cubic)

        # computing univariate logpdfs
        uvt_logpdf: jnp.ndarray = self._scan_uvt_func(
            func=jit(self._uvt.logpdf), x=x_dash, params=params
        )

        # computing copula logpdf
        mvt_params: dict = params["copula"]
        mvt_logpdf: jnp.ndarray = self._mvt.logpdf(x_dash, params=mvt_params)
        return mvt_logpdf - uvt_logpdf.sum(axis=1, keepdims=True)

    # sampling
    def copula_rvs(
        self, size: Scalar, params: dict, key: Array = None
    ) -> Array:
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
        key = _resolve_key(key)
        # generating random samples from x'
        x_dash: jnp.ndarray = self._mvt.rvs(size=size, key=key, params=params["copula"])

        # projecting x' to u space
        return self._scan_uvt_func(jit(self._uvt.cdf), x=x_dash, params=params)

    # fitting
    def fit_copula(
        self,
        u: ArrayLike,
        corr_method: str = "pearson",
        lr: float = 1e-4,
        maxiter: int = 100,
    ) -> dict:
        r"""Fits the copula parameters to the data.

        Args:
            u (ArrayLike): The independent univariate marginal cdf
                values (u) for each dimension.
            corr_method: str, method to estimate the sample correlation
                matrix, sigma. See copulax.multivariate.corr
                for available methods. Used for copula's which require
                a correlation matrix to be estimated, namely Gaussian,
                Student-T, Skewed-T and GH copulas.
            lr (float): Learning rate for optimization. Used for
                copula's which require numerical methods for parameter
                estimation, namely Student-T, Skewed-T and GH copulas.
            maxiter (int): Maximum number of iterations for optimization.
                Used for copula's which require numerical methods for
                parameter estimation, namely Student-T, Skewed-T and GH
                copulas.

        Returns:
            dict: A params dict containing the fitted copula parameters.
        """
        # fitting copula
        copula: dict = self._mvt._fit_copula(
            u, corr_method=corr_method, lr=lr, maxiter=maxiter
        )
        return {"copula": copula}


###############################################################################
# Copula Distributions
###############################################################################
# Normal Mixture Copulas
class GaussianCopula(Copula):
    r"""The Gaussian Copula is a copula that uses the multivariate normal
    distribution to model the dependencies between random variables.

    https://en.wikipedia.org/wiki/Copula_(statistics)
    """

    @jit
    def _get_uvt_params(self, params: dict) -> dict:
        d: int = self._get_dim(params)
        return {"mu": jnp.zeros(d), "sigma": jnp.ones(d)}


gaussian_copula = GaussianCopula("Gaussian-Copula", mvt_normal, normal)


class StudentTCopula(Copula):
    r"""The Student-T Copula is a copula that uses the multivariate
    Student-T distribution to model the dependencies between random
    variables.

    https://en.wikipedia.org/wiki/Copula_(statistics)
    """

    @jit
    def _get_uvt_params(self, params: dict) -> dict:
        nu: Scalar = params["copula"]["nu"]
        d: int = self._get_dim(params)
        return {"nu": jnp.full(d, nu), "mu": jnp.zeros(d), "sigma": jnp.ones(d)}


student_t_copula = StudentTCopula("Student-T-Copula", mvt_student_t, student_t)


class GHCopula(Copula):
    r"""The GH Copula is a copula that uses the multivariate generalized
    hyperbolic (GH) distribution to model the dependencies between
    random variables.

    https://en.wikipedia.org/wiki/Copula_(statistics)
    """

    @jit
    def _get_uvt_params(self, params: dict) -> dict:
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


gh_copula = GHCopula("GH-Copula", mvt_gh, gh)


class SkewedTCopula(Copula):
    r"""The Skewed-T Copula is a copula that uses the multivariate
    skewed-T distribution to model the dependencies between random
    variables.

    https://en.wikipedia.org/wiki/Copula_(statistics)
    """

    def _get_uvt_params(self, params: dict) -> dict:
        d: int = self._get_dim(params)
        nu: Scalar = params["copula"]["nu"]
        gamma: Array = params["copula"]["gamma"]
        return {
            "nu": jnp.full(d, nu),
            "mu": jnp.zeros(d),
            "sigma": jnp.ones(d),
            "gamma": gamma.flatten(),
        }


skewed_t_copula = SkewedTCopula("Skewed-T-Copula", mvt_skewed_t, skewed_t)
