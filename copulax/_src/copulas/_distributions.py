"""CopulAX implementation of the popular Copula distributions."""
from jax._src.typing import ArrayLike, Array
from typing import Callable
from collections import deque
from jax import numpy as jnp
from jax import jit

from copulax._src._distributions import GeneralMultivariate, Multivariate, Univariate, dist_map
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
        return cls(dist_map.id_map[id_]['name'], *values, **init_kwargs)
    
    def tree_flatten(self):
        children = (self._mvt, self._uvt)  # arrays and pytrees
        aux_data = (self._id,)  # static, hashable data
        return children, aux_data
    
    # standard functions
    def _get_dim(self, params) -> int:
        return len(params['marginals'])

    def support(self, params: dict) -> Array:
        marginals: tuple = params['marginals']
        
        support: deque = deque()
        for dist, mparams in marginals:
            msupport = dist.support(**mparams)
            support.append(msupport)
        
        return jnp.vstack(support)

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
        marginals: tuple = params['marginals']
        
        u: deque = deque()
        for i, (dist, mparams) in enumerate(marginals):
            ui: jnp.ndarray = dist.cdf(x[:, i][:, None], **mparams)
            u.append(ui)
        
        return jnp.concat(u, axis=1)

    def _get_uvt_params(self, params: dict) -> dict:
        """Returns the univariate distribution parameters."""
        return {}

    def get_x_dash(self, u: ArrayLike, params: dict) -> Array:
        r"""Computes x' values, which represent the mappings of the 
        independent marginal cdf values (U) to the domain of the joint 
        multivariate distribution.
        
        Args:
            u (ArrayLike): The independent univariate marginal cdf 
                values (U) for each dimension.
            params (dict): The copula and marginal distribution 
                parameters.

        Returns:
            x_dash (Array): The x' values for each dimension.
        """
        u_raw: jnp.ndarray = _multivariate_input(u)[0]
        eps: float = 1e-4
        u: jnp.ndarray = jnp.clip(u_raw, eps, 1 - eps)
        uvt_params: dict = self._get_uvt_params(params)
        uvt_ppf: Callable = jit(self._uvt.ppf)
        uvt_ppf(jnp.array([[0.5]])) # to trigger compilation
        return uvt_ppf(u, **uvt_params)
    
    # densities
    def copula_logpdf(self, u: ArrayLike, params: dict) -> Array:
        r"""Computes the log-pdf of the copula distribution.
        
        Args:
            u (ArrayLike): The independent univariate marginal cdf 
                values (u) for each dimension.
            params (dict): The copula and marginal distribution 
                parameters.

        Returns:
            logpdf (Array): The log-pdf values of the copula 
                distribution.
        """
        # mapping u to x' space
        x_dash: jnp.ndarray = self.get_x_dash(u, params)

        # computing univariate logpdfs
        uvt_params: dict = self._get_uvt_params(params)
        uvt_logpdf: jnp.ndarray = self._uvt.logpdf(x_dash, **uvt_params)
        
        # computing copula logpdf
        mvt_params: dict = params['copula']
        mvt_logpdf: jnp.ndarray = self._mvt.logpdf(x_dash, **mvt_params)
        return mvt_logpdf - uvt_logpdf.sum(axis=1, keepdims=True)
    
    def copula_pdf(self, u: ArrayLike, params: dict) -> Array:
        r"""Computes the pdf of the copula distribution.

        Args:
            u (ArrayLike): The independent univariate marginal cdf 
                values (U) for each dimension.
            params (dict): The copula and marginal distribution 
                parameters.

        Returns:
            pdf (Array): The pdf values of the copula distribution.
        """
        return jnp.exp(self.copula_logpdf(u, params))
    
    def logpdf(self, x: ArrayLike, params: dict) -> Array:
        r"""The log-probability density function (pdf) of the overall 
        joint copula and marginal distribution.

        Args:
            x (ArrayLike): The input at which to evaluate the log-pdf.
                Must be in the shape (n, d) where n is the number of
                samples and d is the number of dimensions.
            params (dict): The copula and marginal distribution 
                parameters.

        Returns:
            Array: The log-pdf values of the overall joint copula and 
                marginal distribution.
        """
        # calculating marginal logpdfs
        x, _, n, d = _multivariate_input(x)
        marginals: tuple = params['marginals']
        marginal_logpdf_sum: jnp.ndarray = jnp.zeros((n, 1))
        for i, (dist, mparams) in enumerate(marginals):
            marginal_logpdf_sum += dist.logpdf(x[:, i][:, None], **mparams)

        # calculating copula logpdf
        u: jnp.ndarray = self.get_u(x, params)
        copula_logpdf: jnp.ndarray = self.copula_logpdf(u, params)
        return copula_logpdf + marginal_logpdf_sum

    def pdf(self, x: ArrayLike, params: dict) -> Array:
        r"""The probability density function (pdf) of the overall joint 
        copula and marginal distribution.

        Args:
            x (ArrayLike): The input at which to evaluate the pdf.
                Must be in the shape (n, d) where n is the number of
                samples and d is the number of dimensions.
            params (dict): The copula and marginal distribution 
                parameters.

        Returns:
            Array: The pdf values of the overall joint copula
                and marginal distribution.
        """
        return jnp.exp(self.logpdf(x, params))

    # sampling
    def copula_rvs(self, size: Scalar, params: dict, 
                   key: Array = DEFAULT_RANDOM_KEY) -> Array:
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
        x_dash: jnp.ndarray = self._mvt.rvs(size=size, key=key, **params['copula'])

        # projecting x' to u space
        uvt_params: dict = self._get_uvt_params(params)
        return self._uvt.cdf(x_dash, **uvt_params)

    def copula_sample(self, size: Scalar, params: dict, 
                   key: Array = DEFAULT_RANDOM_KEY) -> Array:
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
    
    def rvs(self, size: Scalar, params: dict, 
            key: Array = DEFAULT_RANDOM_KEY) -> Array:
        r"""Generates random samples from the overall joint copula and 
        marginal distribution.

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
        # sampling from copula distribution
        u_raw: jnp.ndarray = self.copula_rvs(size=size, key=key, params=params)
        eps: float = 1e-4
        u: jnp.ndarray = jnp.clip(u_raw, eps, 1 - eps)
        
        # projecting u to x space
        marginals: tuple = params['marginals']
        x: deque = deque()
        
        for i, (dist, mparams) in enumerate(marginals):
            xi: jnp.ndarray = dist.ppf(u[:, i][:, None], **mparams)
            x.append(xi)
        
        return jnp.concat(x, axis=1)

    def sample(self, size: Scalar, params: dict, 
            key: Array = DEFAULT_RANDOM_KEY) -> Array:
        r"""Generates random samples from the overall joint copula and 
        marginal distribution.

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
        return self.rvs(size=size, key=key, params=params)

    # fitting
    def fit_marginals(self, x: ArrayLike, 
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
                raise ValueError(f"If univariate_fitter_options is a tuple, "
                                 "it must have an entry for each variable in x.")
        else:
            raise ValueError(f"univariate_fitter_options must be a tuple or "
                             "dictionary.")
        
        # fitting marginals
        jitted_ufitter: Callable = jit(univariate_fitter)
        marginals: deque = deque(maxlen=d)
        for i, options in enumerate(univariate_fitter_options):
            xi: jnp.ndarray = x[:, i]
            best_index, fitted = jitted_ufitter(xi, **options)
            dist: Univariate = fitted[best_index]['dist']
            params: dict = fitted[best_index]['params']
            marginals.append((dist, params))
        
        return {'marginals':  tuple(marginals)}
    
# """            marginals (dict): The marginal distribution's and their 
#                 parameters. Must be a dictionary with a 'marginals' key 
#                 with a tuple value; this tuple must contain 'd' tuples 
#                 (one for each variable), with a univariate distribution 
#                 and their dictionary parameters. 
#                 Obtainable by calling 'fit_marginals'."""

    def fit_copula(self, u: ArrayLike, **kwargs) -> dict:
        r"""Fits the copula parameters to the data.

        Args:
            u (ArrayLike): The independent univariate marginal cdf 
                values (u) for each dimension.
            kwargs: Additional keyword arguments to pass to the
                copula fitting function. For gaussian_copula, 
                student_t_copula and gh_copula's this includes 
                'corr_method' which specifies which correlation matrix 
                estimation method to use; see copulax.multivariate.corr 
                for details.

        Returns:
            dict: A params dict containing the fitted copula parameters.
        """
        # fitting copula
        copula: dict = self._mvt._fit_copula(u, **kwargs)
        return {'copula': copula}

    def fit(self, x: ArrayLike, univariate_fitter_options: tuple[dict] | dict = None, **kwargs) -> dict:
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
            kwargs: Additional keyword arguments to pass to the
                copula fitting function. For gaussian_copula, 
                student_t_copula and gh_copula's this includes 
                'corr_method' which specifies which correlation matrix 
                estimation method to use; see copulax.multivariate.corr 
                for details.

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
        copula: dict = self.fit_copula(u, **kwargs)
        return {**marginals, **copula}

    # metrics
    # get num params -> rest (loglikelihood, aic, bic) ss follow from inheritience

###############################################################################
# Copula Distributions
###############################################################################
# Normal Mixture Copulas
class GaussianCopula(Copula):
    """The Gaussian Copula is a copula that uses the multivariate normal
    distribution to model the dependencies between random variables.
    
    https://en.wikipedia.org/wiki/Copula_(statistics)
    """
    def _get_uvt_params(self, params: dict) -> dict:
        return {"mu": 0.0, "sigma": 1.0}


gaussian_copula = GaussianCopula('Gaussian-Copula', mvt_normal, normal)


class StudentTCopula(Copula):
    """The Student-T Copula is a copula that uses the multivariate 
    Student-T distribution to model the dependencies between random 
    variables.
    
    https://en.wikipedia.org/wiki/Copula_(statistics)
    """
    def _get_uvt_params(self, params: dict) -> dict:
        nu: Scalar = params["copula"]["nu"]
        return {"nu": nu, "mu": 0.0, "sigma": 1.0}
    

student_t_copula = StudentTCopula('Student-T-Copula', mvt_student_t, student_t)


class GHCopula(Copula):
    """The GH Copula is a copula that uses the multivariate generalized 
    hyperbolic (GH) distribution to model the dependencies between 
    random variables.

    https://en.wikipedia.org/wiki/Copula_(statistics)
    """
    def _get_uvt_params(self, params: dict) -> dict:
        lamb: Scalar = params["copula"]["lamb"]
        chi: Scalar = params["copula"]["chi"]
        psi: Scalar = params["copula"]["psi"]
        return {"lamb": lamb, "chi": chi, "psi": psi}
    

gh_copula = GHCopula('GH-Copula', mvt_gh, gh)