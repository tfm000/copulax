"""Module containing base classes for all distributions to inherit from.
"""
from dataclasses import dataclass, field
from jax.tree_util import register_pytree_node
from abc import abstractmethod
from jax._src.typing import ArrayLike, Array
import jax.numpy as jnp
from jax import lax, jit, random


from copulax._src.typing import Scalar
from copulax._src.univariate._ppf import _ppf
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.univariate._rvs import inverse_transform_sampling
from copulax._src.typing import Scalar
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src.multivariate._shape import cov, corr
from copulax._src.optimize import projected_gradient

###############################################################################
# Record of implemented distributions
###############################################################################
@dataclass(frozen=True)
class DistMap:
    continuous_names: tuple # tuple of continuous univariate distribution names
    discrete_names: tuple # tuple of discrete univariate distribution names
    mvt_names: tuple
    copula_names: tuple

    univariate_names: tuple = field(init=False)

    ids: tuple = field(init=False)  # tuple of all distribution ids
    names: tuple = field(init=False)  # tuple of all distribution names
    dtypes: tuple = field(init=False)  # tuple of all distribution data types
    dist_types: tuple = field(init=False)  # tuple of all distribution types
    object_names: tuple = field(init=False)  # tuple of all distribution object names

    id_map: dict = field(init=False)  # dict mapping distribution ids to names and dist_type
    name_map: dict = field(init=False)  # dict mapping distribution names to ids and dist_type
    object_name_map: dict = field(init=False)  # dict mapping distribution object names to ids and dist_type

    def __post_init__(self):
        univariate_names: tuple = self.continuous_names + self.discrete_names

        id_map: dict = {}
        name_map: dict = {}
        object_name_map: dict = {}

        d: dict = {'univariate': univariate_names, 
                   'multivariate': self.mvt_names, 
                   'copula': self.copula_names}
        start: int = 0
        for s, names in d.items():
            for i, name in enumerate(names, start=start):
                dtype: str = 'discrete' if name in discrete_names else 'continuous'
                object_name: str = name.replace('-', '_').lower()
                entry: dict = {'id': i, 
                               'name': name, 
                               'object_name': object_name, 
                               'dtype': dtype, 
                               'dist_type': s}
                id_map[i] = entry
                name_map[name] = entry
                object_name_map[object_name] = entry
            start = i + 1

        super().__setattr__('univariate_names', univariate_names)
        super().__setattr__('ids', tuple(id_map.keys()))
        super().__setattr__('names', tuple(name_map.keys()))
        super().__setattr__('object_names', tuple(object_name_map.keys()))
        super().__setattr__('dtypes', ('continuous', 'discrete'))
        super().__setattr__('dist_types', tuple(d.keys()))
        super().__setattr__('id_map', id_map)
        super().__setattr__('name_map', name_map)
        super().__setattr__('object_name_map', object_name_map)



continuous_names: tuple = ("Uniform", "Normal", "LogNormal", "Student-T", 
                           "Gamma", "Skewed-T", "GIG", "GH", "IG")
discrete_names: tuple = ()
mvt_names: tuple = ("Mvt-Normal", "Mvt-Student-T", "Mvt-GH")
copula_names: tuple = ("Gaussian-Copula", "Student-T-Copula", "GH-Copula")

dist_map = DistMap(continuous_names=continuous_names, 
                   discrete_names=discrete_names, mvt_names=mvt_names, 
                   copula_names=copula_names)


###############################################################################
# Distribution PyTree / base class
###############################################################################
class Distribution:
    r"""Base class for all implemented copulAX distributions."""
    def __init__(self, name: str):
        self._id: int = dist_map.name_map[name]['id']

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, values: tuple, **init_kwargs):
        """
        Args:
            aux_data:
                Data that will be treated as constant through JAX 
                operations.
            values:
                A JAX PyTree of values from which the object is 
                constructed.
        Returns:
            A constructed object.
        """
        id_: int = aux_data[0]
        return cls(dist_map.id_map[id_]['name'], **init_kwargs)
    
    def tree_flatten(self):
        """
        Returns:
            values: A JAX PyTree of values representing the object.
            aux_data:
                Data that will be treated as constant through JAX 
                operations.
        """
        children = ()  # arrays and pytrees
        aux_data = (self._id,)  # static, hashable data
        return children, aux_data
    
    def __init_subclass__(cls, **kwargs):
        # https://github.com/jax-ml/jax/issues/2916
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    @property
    def name(self) -> str:
        """The name of the distribution.
        
        Returns:
            str: The name of the distribution.
        """
        return dist_map.id_map[self._id]['name']
    
    @property
    def dist_type(self) -> str:
        """The type of copulAX distribution.
        
        Returns:
            str: The type of the distribution.
        """
        return dist_map.id_map[self._id]['dist_type']
    
    @property
    def dtype(self) -> str:
        """The data type of the distribution.
        
        Returns:
            str: The data type of the distribution.
        """
        return dist_map.id_map[self._id]['dtype']
    
    @staticmethod
    def _scalar_transform(params: dict) -> dict:
        return {key: jnp.asarray(value, dtype=float).reshape(()) 
                for key, value in params.items()}
    
    @abstractmethod
    def _args_transform(self, params: dict) -> dict:
        r"""Transforms the input arguments to the correct dtype and 
        shape.
        """

    def _params_from_array(self, params_arr: jnp.ndarray, *args, **kwargs) -> dict:
        r"""Returns a dictionary from an array of params"""
        pass

    @abstractmethod
    def _params_to_tuple(self, params: dict) -> tuple:
        r"""Returns a tuple of params from a dictionary. 
        Reduces code when extracting params."""
        pass
    
    @abstractmethod
    def fit(self, x: ArrayLike, *args, **kwargs):
        r"""Fit the distribution to the input data.
        
        Args:
            x (ArrayLike): The input data to fit the distribution to.
            kwargs: Additional keyword arguments to pass to the fit 
                method.
        
        Returns:
            dict: The fitted distribution parameters.
        """

    def _params_to_array(self, params) -> Array:
        r"""Returns a flattened array of params from a dictionary. 
        Reduces code when extracting params."""
        return jnp.asarray(self._params_to_tuple(params)).flatten()
    
    @abstractmethod
    def rvs(self, params: dict, *args, **kwargs) -> Array:
        r"""Generate random variates from the distribution.
        
        Args:
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.  
            kwargs: Additional keyword arguments to pass to the rvs 
            method.
        
        Returns:
            jnp.ndarray: The generated random variates.
        """

    def sample(self, params: dict, *args, **kwargs) -> Array:
        """Alias for the rvs method."""
        return self.rvs(params=params, *args, **kwargs)

    @abstractmethod
    def fit(self, x: ArrayLike, *args, **kwargs):
        """Fit the distribution to the input data.
        
        Args:
            x (ArrayLike): The input data to fit the distribution to.
            kwargs: Additional keyword arguments to pass to the fit 
                method.
        
        Returns:
            dict: The fitted distribution parameters.
        """

    # fitting
    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, 
                       params: dict) -> Array:
        r"""Stable log-pdf function for distribution fitting.

        Args:
            stability (Scalar): A stability parameter for the distribution.
            x (ArrayLike): The input data to evaluate the stable log-pdf.
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.  

        Returns:
            Array: The stable log-pdf values.
        """

        pass

    @abstractmethod
    def logpdf(self, x: ArrayLike, params: dict) -> Array:
        r"""The log-probability density function (pdf) of the 
        distribution.

        Args:
            x (ArrayLike): The input at which to evaluate the log-pdf.
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.  

        Returns:
            Array: The log-pdf values.
        """
        return self._stable_logpdf(stability=0.0, x=x, params=params)

    @abstractmethod
    def pdf(self, x: ArrayLike, params: dict) -> Array:
        r"""The probability density function (pdf) of the distribution.

        Args:
            x (ArrayLike): The input at which to evaluate the pdf.
            params (dict): Parameters describing the distribution. See
                    the specific distribution class or the 'example_params' 
                    method for details.
        
        Returns:
            Array: The pdf values.
        """
        return jnp.exp(self.logpdf(x=x, params=params))
    
    # stats
    @abstractmethod
    def stats(self, params: dict) -> dict:
        r"""Distribution statistics for the distribution.

        Args:
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.

        Returns:
            stats (dict): A dictionary containing the distribution 
            statistics.
        """
        return {}
    
    # metrics
    @abstractmethod
    def loglikelihood(self, x: ArrayLike, params: dict) -> Scalar:
        r"""Log-likelihood of the distribution given the data.

        Args:
            x (ArrayLike): The input data to evaluate the 
            log-likelihood.
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.  
        
        Returns:
            loglikelihood (Scalar): The log-likelihood value.
        """
        return self.logpdf(x=x, params=params).sum()

    @abstractmethod
    def aic(self, k: int, x: ArrayLike, params: dict) -> Scalar:
        r"""Akaike Information Criterion (AIC) of the distribution 
        given the data. Can be used as a crude metric for model 
        selection, by minimising.

        Args:
            x (ArrayLike): The input data to evaluate the AIC.
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.  

        Returns:
            aic (Scalar): The AIC value.
        """
        return 2 * k - 2 * self.loglikelihood(x=x, params=params)
    
    @abstractmethod
    def bic(self, k: int, n: int, x: ArrayLike, params: dict) -> Scalar:
        r"""Bayesian Information Criterion (BIC) of the distribution 
        given the data. Can be used as a crude metric for model 
        selection, by minimising.

        Args:
            x (ArrayLike): The input data to evaluate the BIC.
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.  

        Returns:
            bic (Scalar): The BIC value.
        """
        return k * jnp.log(n) - 2 * self.loglikelihood(x=x, params=params)
    
    @abstractmethod
    def example_params(self, *args, **kwargs) -> dict:
        r"""Returns example parameters for the distribution.

        Returns:
            dict: A dictionary containing example distribution 
            parameters.
        """
        pass


###############################################################################
# Univariate Base Class
###############################################################################
class Univariate(Distribution):
    r"""Base class for univariate distributions."""
    @staticmethod
    def _args_transform(params: dict) -> dict:
        return Distribution._scalar_transform(params)

    @abstractmethod
    def support(self, *args, **kwargs) -> Array:
        r"""The support of the distribution is the subset of x for which 
        the pdf is non-zero. 
        
        Returns:
            Array: Flattened array containing the support of the 
            distribution.
        """
    
    @abstractmethod
    def logcdf(self, x: ArrayLike, params: dict) -> Array:
        r"""The log-cumulative distribution function of the 
        distribution.

        Args:
            x (ArrayLike): The input at which to evaluate the log-cdf.
            params (dict): Parameters describing the distribution. See
                the specific distribution class or the 'example_params'
                method for details.

        Returns:
            Array: The log-cdf values.
        """
        return jnp.log(self.cdf(x=x, params=params))
    
    @abstractmethod
    def cdf(self, x: ArrayLike, params: dict) -> Array:
        r"""Cumulative distribution function of the distribution.

        Args:
            x (ArrayLike): The input at which to evaluate the cdf.
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.

        Returns:
            Array: The cdf values.
        """

    @abstractmethod
    def ppf(self, x0: float, q: ArrayLike, params: dict) -> Array:
        r"""Percent point function (inverse of the CDF) of the 
        distribution.

        Args:
            q (ArrayLike): The quantile values. at which to evaluate the 
            ppf.
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.

        Returns:
            Array: The inverse CDF values.
        """
        return _ppf(cdf_func=self.cdf, bounds=self.support(), q=q, 
                    params=params, x0=x0)
    
    @abstractmethod
    def inverse_cdf(self, q: ArrayLike, params: dict) -> Array:
        r"""Percent point function (inverse of the CDF) of the 
        distribution.

        Args:
            q (ArrayLike): The quantile values. at which to evaluate the 
            ppf.
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.

        Returns:
            Array: The inverse CDF values.
        """
        return self.ppf(q=q, params=params)

    # sampling
    @abstractmethod
    def rvs(self, size: Scalar | tuple, params: dict,
            key: Array = DEFAULT_RANDOM_KEY) -> Array:
        r"""Generates random samples from the distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'size' 
            is a static argument.

        Args:
            size (tuple | Scalar): The size / shape of the generated 
                output array of random numbers. If a scalar is provided, 
                the output array will have shape (size,), otherwise it will 
                match the shape specified by this tuple.
            params (dict): Parameters describing the distribution. See
                the specific distribution class or the 'example_params'
                method for details.
            key (Array): The Key for random number generation.
        """
        return inverse_transform_sampling(ppf_func=self.ppf, shape=size, 
                                          params=params, key=key)        
    
    @abstractmethod
    def sample(self, size: Scalar | tuple, params: dict, 
               key: Array = DEFAULT_RANDOM_KEY) -> Array:
        r"""Generates random samples from the distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'size' 
            is a static argument.

        Args:
            size (tuple | Scalar): The size / shape of the generated 
                output array of random numbers. If a scalar is provided, 
                the output array will have shape (size,), otherwise it will 
                match the shape specified by this tuple.
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.
            key (Array): The Key for random number generation.
        """
        return self.rvs(size=size, params=params, key=key)
    
    def _mle_objective(self, params_arr: jnp.ndarray, x: jnp.ndarray, 
                       *args, **kwargs) -> Scalar:
        r"""Negative log-likelihood of the distribution given the data.

        Args:
            x (ArrayLike): The input data to evaluate the 
                negative log-likelihood.

        Returns:
            mle_objective (float): The negative log-likelihood value.
        """
        params: dict = self._params_from_array(params_arr, *args, **kwargs)
        return -self._stable_logpdf(stability=1e-30, x=x, params=params).sum()
    
    # metrics
    @abstractmethod
    def aic(self, x: ArrayLike, params: dict) -> float:
        k: int = len(params)
        return super().aic(k=k, x=x, params=params)
    
    @abstractmethod
    def bic(self, x: ArrayLike, params: dict) -> float:
        k: int = len(params)
        n: int  = x.size
        return super().bic(k=k, n=n, x=x, params=params)
    

###############################################################################
# Multivariate Base Class
###############################################################################
class GeneralMultivariate(Distribution):
    r"""Base class for multivariate and copula distributions."""
    @abstractmethod
    def _classify_params(self, *args, **kwargs) -> tuple[tuple[Array]]:
        r"""Classify the distribution parameters into scalars, vectors
        and shapes."""
        pass

    @abstractmethod
    def _get_dim(self, *args, **kwargs) -> int:
        r"""Returns the number of dimensions of the distribution."""

    def _args_transform(self, *args, **kwargs) -> tuple[ArrayLike]:
        scalars, vectors, shapes = self._classify_params(*args, **kwargs)  # todo: issue is here where scalars are being included. need to think about how to do this
        d: int = self._get_dim(scalars, vectors, shapes)

        # scalars
        transformed_scalars: tuple = Distribution._scalar_transform(*scalars)
        
        # vectors
        transformed_vectors: tuple = tuple(
            jnp.asarray(v, dtype=float).reshape((d, 1)) for v in vectors
            )
        
        # shapes
        transformed_shapes: tuple = tuple(
            jnp.asarray(shape, dtype=float).reshape((d, d)) for shape in shapes
            )
        
        return transformed_scalars, transformed_vectors, transformed_shapes
    
    @staticmethod
    def _get_num_params(self, 
                        scalars: tuple[Array] = tuple(), 
                        vectors: tuple[Array] = (jnp.array([]),),
                        symmetric_shapes: tuple[Array] = (jnp.array([]),),
                        symmetric_shapes_non_diagonal: tuple[Array] = (jnp.array([]),),
                        non_symmetric_shapes: tuple[Array] = (jnp.array([]),),
                        ) -> int:
        r"""Returns the number of parameters of the distribution.
        
        Args:
            scalars (tuple[Array]): Tuple of scalar parameters.
            vectors (tuple[Array]): Tuple of vector parameters.
            symmetric_shapes (tuple[Array]): Tuple of symmetric shape 
                parameters. Diagonal elements are included in the count.
            symmetric_shapes_non_diagonal (tuple[Array]): Tuple of 
                symmetric shape parameters. Diagonal elements are not 
                included in the count.
            non_symmetric_shapes: Tuple of non-symmetric shape 
                parameters.

        Returns:
            int: The number of parameters.
        """
        # scalars
        n_scalars: int = len(scalars)
    
        # vectors
        d_vect: int = vectors[0].size
        n_vectors: int = len(vectors) * d_vect

        # symmetric shapes
        d_symm_shape: int = symmetric_shapes[0].shape[0]
        n_symm_shapes: int = len(symmetric_shapes) * (d_symm_shape * (d_symm_shape + 1) // 2)

        # symmetric shapes non-diagonal
        d_symm_shape_nd: int = symmetric_shapes_non_diagonal[0].shape[0]
        n_symm_shapes_nd: int = len(symmetric_shapes_non_diagonal) * (d_symm_shape_nd * (d_symm_shape_nd - 1) // 2)

        # non-symmetric shapes
        d_non_symm_shape: int = non_symmetric_shapes[0].shape[0]
        n_non_symm_shapes: int = len(non_symmetric_shapes) * d_non_symm_shape ** 2
        
        # total
        return n_scalars + n_vectors + n_symm_shapes + n_symm_shapes_nd + n_non_symm_shapes
    

    @abstractmethod
    def support(self, params: dict) -> Array:
        r"""The support of the distribution is the subset of 
        multivariate x for which the pdf is non-zero. 

        Args:
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.
        
        Returns:
            Array: Array containing the support of each variable in
            the distribution, in an (d, 2) shape.
        """

    # sampling
    @abstractmethod
    def rvs(self, size: Scalar, params: dict, 
            key: Array = DEFAULT_RANDOM_KEY) -> Array:
        r"""Generates random samples from the distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'size' 
            is a static argument.

        Args:
            size (Scalar): The size / shape of the generated 
                output array of random numbers. Must be scalar. 
                Generates an (size, d) array of random numbers, where
                d is the number of dimensions inferred from the provided
                distribution parameters.
            params (dict): Parameters describing the distribution. See
                    the specific distribution class or the 'example_params' 
                    method for details.
            key (Array): The Key for random number generation.
        """
    
    @abstractmethod
    def sample(self, size: Scalar | tuple, params: dict, 
               key: Array = DEFAULT_RANDOM_KEY) -> Array:
        r"""Generates random samples from the distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'size' 
            is a static argument.

        Args:
            size (Scalar): The size / shape of the generated 
                output array of random numbers. Must be scalar. 
                Generates an (size, d) array of random numbers, where
                d is the number of dimensions inferred from the provided
                distribution parameters.
            key (Array): The Key for random number generation.
        """
        return super().sample(size=size, params=params, key=key)
    
    # metrics
    @abstractmethod
    def aic(self, x: ArrayLike, *args, **kwargs) -> float:
        scalars, vectors, shapes = self._args_transform(*args, **kwargs)
        k: int = self._get_num_params(scalars, vectors, shapes)
        return super().aic(k, x, *scalars, *vectors, *shapes)

    @abstractmethod
    def bic(self, x: ArrayLike, *args, **kwargs) -> float:
        x, _, n, _ = _multivariate_input(x)
        scalars, vectors, shapes = self._args_transform(*args, **kwargs)
        k: int = self._get_num_params(scalars, vectors, shapes)
        return super().bic(k, n, x, *scalars, *vectors, *shapes)

    # fitting
    @abstractmethod
    def fit(self, x: ArrayLike, *args, **kwargs) -> dict:
        r"""Fit the distribution to the input data.

        Note:
            Data must be in the shape (n, d) where n is the number of 
            samples and d is the number of dimensions.
        
        Args:
            x (ArrayLike): The input data to fit the distribution too.
                Must be in the shape (n, d) where n is the number of
                samples and d is the number of dimensions.
            kwargs: Additional keyword arguments to pass to the fit 
                method.
        
        Returns:
            dict: The fitted distribution parameters.
        """


class Multivariate(GeneralMultivariate):
    r"""Base class for multivariate distributions."""
    def _get_dim(self, scalars: tuple, vectors: tuple, shapes: tuple) -> int:
        return jnp.asarray(vectors[0]).size

    def support(self, marginal_support: tuple = (-jnp.inf, jnp.inf), 
                *args, **kwargs) -> Array:
        scalars, vectors, shapes = self._classify_params(*args, **kwargs)
        d: int = self._get_dim(scalars, vectors, shapes)
        return jnp.concat([
            jnp.full((d, 1), marginal_support[0]), 
            jnp.full((d, 1), marginal_support[1])], 
            axis=1)
    
    @jit
    def _single_qi(self, carry: tuple, xi: jnp.ndarray) -> jnp.ndarray:
        mu, sigma_inv = carry
        return carry, lax.sub(xi, mu).T @ sigma_inv @ lax.sub(xi, mu)
    
    def _calc_Q(self, x: jnp.ndarray, mu: jnp.ndarray, sigma_inv: jnp.ndarray
                ) -> jnp.ndarray:
        r"""Calculates the Q vector (x - mu)^T @ sigma^-1 @ (x - mu)"""
        return lax.scan(f=self._single_qi, xs=x, 
                        init=(mu.flatten(), sigma_inv))[1]

    def logpdf(self, x: ArrayLike, *args, **kwargs) -> Array:
        r"""The log-probability density function (pdf) of the 
        distribution.

        Args:
            x (ArrayLike): The input at which to evaluate the log-pdf.
                Must be in the shape (n, d) where n is the number of
                samples and d is the number of dimensions.

        Returns:
            Array: The log-pdf values.
        """
        return super().logpdf(x, *args, **kwargs)


    def pdf(self, x: ArrayLike, *args, **kwargs) -> Array:
        r"""The probability density function (pdf) of the distribution.

        Args:
            x (ArrayLike): The input at which to evaluate the pdf.
                Must be in the shape (n, d) where n is the number of
                samples and d is the number of dimensions.

        Returns:
            Array: The pdf values.
        """
        return super().pdf(x, *args, **kwargs)
    
    @abstractmethod
    def _fit_copula(self, u: ArrayLike, corr_method: str = 'pearson', 
                    *args, **kwargs) -> dict:
        r"""Fits the copula distribution to the data.

        Note:
            If you intend to jit wrap this function, ensure that 
            'corr_method' is a static argument.
        
        Args:
            x (ArrayLike): The input data to fit the distribution too.
                Must be in the shape (n, d) where n is the number of
                samples and d is the number of dimensions.
            kwargs: Additional keyword arguments to pass to the fit 
                method.
        
        Returns:
            dict: The fitted copula distribution parameters.
        """
        u, _, n, d = _multivariate_input(u)
        mu: jnp.darray = jnp.zeros((d, 1))
        sigma: jnp.ndarray = corr(x=u, method=corr_method)
        return {'mu': mu, 'sigma': sigma, 'n': n, 'd': d, 'u': u}

    
class NormalMixture(Multivariate):
    r"""Base class for normal mixture distributions."""
    def rvs(self, key, n: int, W: Array, mu: Array, gamma: Array, sigma: Array) -> Array:
        r"""Generates random samples from the distribution."""
        d: int = mu.size

        m: jnp.ndarray = mu + W * gamma

        Z: jnp.ndarray = random.normal(key, shape=(d, n))
        A: jnp.ndarray = jnp.linalg.cholesky(sigma)
        r: jnp.ndarray = jnp.sqrt(W) * (A @ Z)
        return (m + r).T
    
    # fitting
    # def _reconstruct_ldmle_params(self, params: jnp.ndarray, 
    #                               sample_mean: jnp.ndarray, 
    #                               sample_cov: jnp.ndarray) -> tuple:
    #     """Reconstructs the low dim MLE parameters from a flat array."""
    #     pass

    # def _ldmle_objective(self, params: jnp.ndarray, x: jnp.ndarray, 
    #                      sample_mean: jnp.ndarray, sample_cov: jnp.ndarray) -> Scalar:
    #     reconstructed_params: tuple = self._reconstruct_ldmle_params(params, sample_mean, sample_cov)
    #     return -self._stable_logpdf(1e-30, x, *reconstructed_params).sum()
    
    @abstractmethod
    def _ldmle_inputs(self, d: int) -> tuple:
        """Returns the input arguments for the low dimensional MLE.
        Specifically, the projection options containing the constraints
        and the initial guess."""
        pass

    # def fit(self, x: ArrayLike, cov_method: str = 'pearson', 
    #         *args, **kwargs) -> dict:
    #     r"""Fits the multivariate distribution to the data.

    #     Note:
    #         If you intend to jit wrap this function, ensure that 
    #         'cov_method' is a static argument.

    #     Algorithm:
    #         1. Estimate the sample mean vector and sample covariance 
    #             matrix, potentially using robust estimators.
    #         2. Remove the location and shape matrices from the 
    #             optimisation process, as these can be inferred from 
    #             scalar parameters and skewness.
    #         3. We maximise the log-likelihood using ADAM.

    #     Args:
    #         x: arraylike, data to fit the distribution to.
    #         cov_method: str, method to estimate the sample covariance 
    #             matrix, sigma. See copulax.multivariate.cov and/or 
    #             copulax.multivariate.corr for available methods.

    #     Returns:
    #         dict containing the fitted parameters.
    #     """
    #     # estimating the sample mean and covariance
    #     x, _, _, d = _multivariate_input(x)
    #     sample_mean: jnp.ndarray = jnp.mean(x, axis=0).reshape((d, 1))
    #     sample_cov: jnp.ndarray = cov(x=x, method=cov_method)

    #     # optimisation constraints and initial guess
    #     projection_options, params0 = self._ldmle_inputs(d)

    #     # ADAM gradient descent
    #     res: dict = projected_gradient(f=self._ldmle_objective, x0=params0, 
    #                                    projection_method='projection_box', 
    #                                    projection_options=projection_options, 
    #                                    x=x, sample_mean=sample_mean, 
    #                                    sample_cov=sample_cov)
        
    #     # reconstructing the parameters
    #     optimised_params: jnp.ndarray = res['x']
    #     reconstructed_params: tuple = self._reconstruct_ldmle_params(optimised_params, sample_mean, sample_cov)
    #     return self._params_dict(*reconstructed_params)
    
    def _general_fit(self, x: ArrayLike, d: int, loc: jnp.ndarray, shape: jnp.ndarray, 
                     reconstruct_func_id: int, *args, **kwargs) -> dict:
        # optimisation constraints and initial guess
        projection_options, params0 = self._ldmle_inputs(d)

        # ADAM gradient descent
        res: dict = projected_gradient(f=self._ldmle_objective, x0=params0, 
                                       projection_method='projection_box', 
                                       projection_options=projection_options, 
                                       x=x, loc=loc, shape=shape, 
                                       reconstruct_func_id=reconstruct_func_id)
        
        # reconstructing the parameters
        optimised_params: jnp.ndarray = res['x']
        reconstructed_params: tuple = self._reconstruct_ldmle_func(
            func_id=reconstruct_func_id, params=optimised_params, 
            loc=loc, shape=shape)
        return self._params_dict(*reconstructed_params)
    
    def _ldmle_objective(self, params: jnp.ndarray, x: jnp.ndarray, 
                         loc: jnp.ndarray, shape: jnp.ndarray, reconstruct_func_id) -> Scalar:
        reconstructed_params: tuple = self._reconstruct_ldmle_func(
            func_id=reconstruct_func_id, params=params, loc=loc, shape=shape)
        return -self._stable_logpdf(1e-30, x, *reconstructed_params).sum()
    
    @abstractmethod
    def _ldmle_inputs(self, d: int) -> tuple:
        """Returns the input arguments for the low dimensional MLE.
        Specifically, the projection options containing the constraints
        and the initial guess."""
        pass
    
    def fit(self, x: ArrayLike, cov_method: str = 'pearson', 
            *args, **kwargs) -> dict:
        r"""Fits the multivariate distribution to the data.

        Note:
            If you intend to jit wrap this function, ensure that 
            'cov_method' is a static argument.

        Algorithm:
            1. Estimate the sample mean vector and sample covariance 
                matrix, potentially using robust estimators.
            2. Remove the location and shape matrices from the 
                optimisation process, as these can be inferred from 
                scalar parameters and skewness.
            3. We maximise the log-likelihood using ADAM.

        Args:
            x: arraylike, data to fit the distribution to.
            cov_method: str, method to estimate the sample covariance 
                matrix, sigma. See copulax.multivariate.cov and/or 
                copulax.multivariate.corr for available methods.

        Returns:
            dict containing the fitted parameters.
        """
        # estimating the sample mean and covariance
        x, _, _, d = _multivariate_input(x)
        sample_mean: jnp.ndarray = jnp.mean(x, axis=0).reshape((d, 1))
        sample_cov: jnp.ndarray = cov(x=x, method=cov_method)

        # optimising
        return self._general_fit(
            x=x, d=d, loc=sample_mean, shape=sample_cov, 
            reconstruct_func_id=0, *args, **kwargs)
    
    @abstractmethod
    def _reconstruct_ldmle_params(self, params: jnp.ndarray, 
                                  loc: jnp.ndarray, 
                                  shape: jnp.ndarray) -> tuple:
        """Reconstructs the low dim MLE parameters from a flat array."""
        pass
    
    @abstractmethod
    def _reconstruct_ldmle_copula_params(self, params: jnp.ndarray,
                                         loc: jnp.ndarray, 
                                         shape: jnp.ndarray) -> tuple:
        """Reconstructs the low dim MLE parameters from a flat array for 
        copula fitting."""
        pass

    def _reconstruct_ldmle_func(self, func_id: int, params: jnp.ndarray, 
                                loc: jnp.ndarray, shape: jnp.ndarray) -> tuple:
        """Reconstructs the low dim MLE parameters from a flat array."""
        # if func_id == 0:
        #     return self._reconstruct_ldmle_params(params, loc, shape)
        # elif func_id == 1:
        #     return self._reconstruct_ldmle_copula_params(params, loc, shape)
        return lax.cond(func_id == 0, 
                        self._reconstruct_ldmle_params, 
                        self._reconstruct_ldmle_copula_params, 
                        params, loc, shape)

    def _fit_copula(self, u: jnp.ndarray, corr_method: str = 'pearson', 
                    *args, **kwargs):
        d: dict = super()._fit_copula(u, corr_method, *args, **kwargs)
        return self._general_fit(
            x=d['u'], d=d['d'], loc=d['mu'], shape=d['sigma'], 
            reconstruct_func_id=1, *args, **kwargs)
        

