"""CopulAX implementation of the popular Copula distributions."""

from jax._src.typing import ArrayLike, Array
from typing import Callable
from collections import deque
from jax import numpy as jnp
from jax import jit

from copulax._src._distributions import (
    GeneralMultivariate,
    Multivariate,
    Univariate,
    dist_map,
)
from copulax.univariate import univariate_fitter
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src._utils import DEFAULT_RANDOM_KEY
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
# Copula Class
# (Must be located here to prevent circular import issues)
###############################################################################
class Copula(GeneralMultivariate):
    r"""Base class for copula distributions."""

    # initialisation
    def __init__(self, name, mvt: Multivariate, uvt: Univariate):
        super().__init__(name)
        self._mvt: Multivariate = mvt  # multivariate pytree object
        self._uvt: Univariate = uvt  # univariate pytree object

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, values: tuple, **init_kwargs):
        id_: int = aux_data[0]
        return cls(dist_map.id_map[id_]["name"], *values, **init_kwargs)

    def tree_flatten(self):
        children = (self._mvt, self._uvt)  # arrays and pytrees
        aux_data = (self._id,)  # static, hashable data
        return children, aux_data

    # standard functions
    def _get_dim(self, params) -> int:
        return len(params["marginals"])

    def support(self, params: dict) -> Array:
        marginals: tuple = params["marginals"]

        support: deque = deque()
        for dist, mparams in marginals:
            msupport = dist.support(params=mparams)
            support.append(msupport)

        return jnp.vstack(support)

    def _example_copula_params(self, dim, *args, **kwargs) -> dict:
        # override if sigma is not the shape matrix for the mulviariate dist
        mvt_params: dict = self._mvt.example_params(dim=dim, *args, **kwargs)
        mvt_params["sigma"] = jnp.eye(dim, dim)
        return mvt_params

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

    def get_u(self, x: ArrayLike, params: dict) -> Array:
        r"""Computes the independent univariate marginal cdf values (u)
        for each dimension.

        Args:
            x (ArrayLike): The multivariate input data to evaluate the
                cdf.
            params (dict): The marginal distribution parameters.

        Returns:
            u (Array): The independent univariate marginal cdf values
                for each dimension.
        """
        x: jnp.ndarray = _multivariate_input(x)[0]
        marginals: tuple = params["marginals"]
        u: deque = deque()
        for i, (dist, mparams) in enumerate(marginals):
            ui: jnp.ndarray = dist.cdf(x[:, i][:, None], params=mparams)
            u.append(ui)
        return jnp.concat(u, axis=1)

    def _get_uvt_params(self, params: dict) -> tuple:
        """Returns the univariate distribution parameters."""
        return tuple()

    def _scan_uvt_func(self, func: Callable, x: Array, params: dict, **kwargs) -> Array:
        """Scans the univariate distribution function over the data."""
        uvt_params_tuple: tuple = self._get_uvt_params(params)
        d: int = self._get_dim(params)
        output: deque = deque(maxlen=d)
        for i, uvt_params in enumerate(uvt_params_tuple):
            output_i: Array = func(x[:, i][:, None], params=uvt_params, **kwargs)
            output.append(output_i)
        return jnp.concat(output, axis=1)

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

    def copula_pdf(self, u: ArrayLike, params: dict, cubic: bool = True) -> Array:
        r"""Computes the pdf of the copula distribution.

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
            pdf (Array): The pdf values of the copula distribution.
        """
        return jnp.exp(self.copula_logpdf(u, params, cubic=cubic))

    def logpdf(self, x: ArrayLike, params: dict, cubic: bool = True) -> Array:
        r"""The log-probability density function (pdf) of the overall
        joint copula and marginal distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'cubic'
            is a static argument.

        Args:
            x (ArrayLike): The input at which to evaluate the log-pdf.
                Must be in the shape (n, d) where n is the number of
                samples and d is the number of dimensions.
            params (dict): The copula and marginal distribution
                parameters.
            cubic (bool): Whether to use a cubic spline approximation
                of the univariate ppf function for faster computation.
                This can also improve gradient estimates.

        Returns:
            Array: The log-pdf values of the overall joint copula and
                marginal distribution.
        """
        # calculating marginal logpdfs
        x, _, n, d = _multivariate_input(x)
        marginals: tuple = params["marginals"]
        marginal_logpdf_sum: jnp.ndarray = jnp.zeros((n, 1))
        for i, (dist, mparams) in enumerate(marginals):
            marginal_logpdf_sum += jit(dist.logpdf)(x[:, i][:, None], params=mparams)

        # calculating copula logpdf
        u: jnp.ndarray = self.get_u(x, params)
        copula_logpdf: jnp.ndarray = self.copula_logpdf(u, params, cubic=cubic)
        return copula_logpdf + marginal_logpdf_sum

    def pdf(self, x: ArrayLike, params: dict, cubic: bool = True) -> Array:
        r"""The probability density function (pdf) of the overall joint
        copula and marginal distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'cubic'
            is a static argument.

        Args:
            x (ArrayLike): The input at which to evaluate the pdf.
                Must be in the shape (n, d) where n is the number of
                samples and d is the number of dimensions.
            params (dict): The copula and marginal distribution
                parameters.
            cubic (bool): Whether to use a cubic spline approximation
                of the univariate ppf function for faster computation.
                This can also improve gradient estimates.

        Returns:
            Array: The pdf values of the overall joint copula
                and marginal distribution.
        """
        return jnp.exp(self.logpdf(x, params, cubic=cubic))

    # sampling
    def copula_rvs(
        self, size: Scalar, params: dict, key: Array = DEFAULT_RANDOM_KEY
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
        # generating random samples from x'
        x_dash: jnp.ndarray = self._mvt.rvs(size=size, key=key, params=params["copula"])

        # projecting x' to u space
        return self._scan_uvt_func(jit(self._uvt.cdf), x=x_dash, params=params)

    def copula_sample(
        self, size: Scalar, params: dict, key: Array = DEFAULT_RANDOM_KEY
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
        return self.copula_rvs(size=size, key=key, params=params)

    def rvs(
        self,
        size: Scalar,
        params: dict,
        key: Array = DEFAULT_RANDOM_KEY,
        cubic: bool = True,
    ) -> Array:
        r"""Generates random samples from the overall joint copula and
        marginal distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'size'
            and 'cubic' are static arguments.

        Args:
            size (Scalar): size (Scalar): The size / shape of the generated
                output array of random numbers. Must be scalar.
                Generates an (size, d) array of random numbers, where
                d is the number of dimensions inferred from the provided
                distribution parameters.
            params (dict): The copula and marginal distribution
                parameters.
            key (Array): The Key for random number generation.
            cubic (bool): Whether to use a cubic spline approximation
                of the univariate ppf function for faster computation.
                This can also improve gradient estimates.
        """
        # sampling from copula distribution
        u_raw: jnp.ndarray = self.copula_rvs(size=size, key=key, params=params)
        eps: float = 1e-4
        u: jnp.ndarray = jnp.clip(u_raw, eps, 1 - eps)

        # projecting u to x space
        marginals: tuple = params["marginals"]
        x: deque = deque()

        for i, (dist, mparams) in enumerate(marginals):
            xi: jnp.ndarray = jit(dist.ppf, static_argnames="cubic")(
                u[:, i][:, None], params=mparams, cubic=cubic
            )
            x.append(xi)

        return jnp.concat(x, axis=1)

    def sample(
        self,
        size: Scalar,
        params: dict,
        key: Array = DEFAULT_RANDOM_KEY,
        cubic: bool = True,
    ) -> Array:
        r"""Generates random samples from the overall joint copula and
        marginal distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'size'
            and 'cubic' are static arguments.

        Args:
            size (Scalar): size (Scalar): The size / shape of the generated
                output array of random numbers. Must be scalar.
                Generates an (size, d) array of random numbers, where
                d is the number of dimensions inferred from the provided
                distribution parameters.
            params (dict): The copula and marginal distribution
                parameters.
            key (Array): The Key for random number generation.
            cubic (bool): Whether to use a cubic spline approximation
                of the univariate ppf function for faster computation.
                This can also improve gradient estimates.
        """
        return self.rvs(size=size, key=key, params=params, cubic=cubic)

    # fitting
    def fit_marginals(
        self,
        x: ArrayLike,
        univariate_fitter_options: tuple[dict] | dict = None,
    ) -> dict:
        r"""Fits the univariate marginal distributions to the data.

        Args:
            x (ArrayLike): The input data to fit the distribution too.
                Must be in the shape (n, d) where n is the number of
                samples and d is the number of dimensions.
            univariate_fitter_options (tuple | dict): The options for
                the univariate fitter. If a dictionary is provided,
                specify the desired univariate argument names and values
                as key value pairs. These will then be used for each
                distribution. If a tuple is provided, it must contain
                dictionaries for each distribution using the same
                indexing as in 'x'.

        Note:
            Not jitable.

        Returns:
            dict: A params dict containing the fitted univariate
            marginal distributions and their parameters.
        """
        # determining univariate fitter options
        x, _, n, d = _multivariate_input(x)
        if univariate_fitter_options is None:
            univariate_fitter_options: tuple = ({},) * d
        elif isinstance(univariate_fitter_options, dict):
            univariate_fitter_options: tuple = (univariate_fitter_options,) * d
        elif isinstance(univariate_fitter_options, tuple):
            if len(univariate_fitter_options) != d:
                raise ValueError(
                    f"If univariate_fitter_options is a tuple, "
                    "it must have an entry for each variable in x."
                )
        else:
            raise ValueError(
                f"univariate_fitter_options must be a tuple or " "dictionary."
            )

        # fitting marginals
        jitted_ufitter: Callable = jit(
            univariate_fitter, static_argnames=("metric", "distributions")
        )
        marginals: deque = deque(maxlen=d)
        for i, options in enumerate(univariate_fitter_options):
            xi: jnp.ndarray = x[:, i]
            best_index, fitted = jitted_ufitter(xi, **options)
            dist: Univariate = fitted[best_index]["dist"]
            params: dict = fitted[best_index]["params"]
            marginals.append((dist, params))

        return {"marginals": tuple(marginals)}

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

    def fit(
        self,
        x: ArrayLike,
        univariate_fitter_options: tuple[dict] | dict = None,
        corr_method: str = "pearson",
        lr: float = 1e-4,
        maxiter: int = 100,
    ) -> dict:
        r"""Fits the joint copula and marginal distribution to the data.
        This is equivalent to calling 'fit_marginals' and 'fit_copula'
        successively.

        Args:
            x (ArrayLike): The input data to fit the distribution too.
                Must be in the shape (n, d) where n is the number of
                samples and d is the number of dimensions.
            univariate_fitter_options (tuple | dict): The options for
                the univariate fitter. If a dictionary is provided,
                specify the desired univariate argument names and values
                as key value pairs. These will then be used for each
                distribution. If a tuple is provided, it must contain
                dictionaries for each distribution using the same
                indexing as in 'x'.
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

        Note:
            Not jitable.

        Returns:
            dict: A params dict containing the fitted univariate
                marginal distributions and their parameters, in addition
                to the fitted copula parameters.
        """
        # fitting marginals
        marginals: dict = self.fit_marginals(x, univariate_fitter_options)

        # projecting x to u space
        u: jnp.ndarray = self.get_u(x, marginals)

        # fitting copula
        copula: dict = self.fit_copula(
            u, corr_method=corr_method, lr=lr, maxiter=maxiter
        )

        # fitting joint distribution
        return {**marginals, **copula}


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
        return tuple(({"mu": 0.0, "sigma": 1.0},) * d)


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
        return tuple(({"nu": nu, "mu": 0.0, "sigma": 1.0},) * d)


student_t_copula = StudentTCopula("Student-T-Copula", mvt_student_t, student_t)


class GHCopula(Copula):
    r"""The GH Copula is a copula that uses the multivariate generalized
    hyperbolic (GH) distribution to model the dependencies between
    random variables.

    https://en.wikipedia.org/wiki/Copula_(statistics)
    """

    @jit
    def _get_uvt_params(self, params: dict) -> dict:
        lamb: Scalar = params["copula"]["lambda"]
        chi: Scalar = params["copula"]["chi"]
        psi: Scalar = params["copula"]["psi"]
        gamma: Array = params["copula"]["gamma"]

        uvt_params: deque = deque()
        for i, gamma_i in enumerate(gamma.flatten()):
            params_i: dict = {
                "lambda": lamb,
                "chi": chi,
                "psi": psi,
                "mu": 0.0,
                "sigma": 1.0,
                "gamma": gamma_i,
            }
            uvt_params.append(params_i)
        return tuple(uvt_params)


gh_copula = GHCopula("GH-Copula", mvt_gh, gh)


class SkewedTCopula(Copula):
    r"""The Skewed-T Copula is a copula that uses the multivariate
    skewed-T distribution to model the dependencies between random
    variables.

    https://en.wikipedia.org/wiki/Copula_(statistics)
    """

    def _get_uvt_params(self, params: dict) -> dict:
        nu: Scalar = params["copula"]["nu"]
        gamma: Array = params["copula"]["gamma"]

        uvt_params: deque = deque()
        for i, gamma_i in enumerate(gamma.flatten()):
            params_i: dict = {"nu": nu, "mu": 0.0, "sigma": 1.0, "gamma": gamma_i}
            uvt_params.append(params_i)
        return tuple(uvt_params)


skewed_t_copula = SkewedTCopula("Skewed-T-Copula", mvt_skewed_t, skewed_t)
