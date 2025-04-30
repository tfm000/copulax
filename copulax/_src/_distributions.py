"""Module containing base classes for all distributions to inherit from.
"""
from dataclasses import dataclass, field
from jax.tree_util import register_pytree_node
from abc import abstractmethod
from jax._src.typing import ArrayLike, Array
import jax.numpy as jnp
from jax import lax, jit, random
import matplotlib.pyplot as plt
from typing import Iterable


from copulax._src.typing import Scalar
from copulax._src.univariate._ppf import _ppf
from copulax._src._utils import DEFAULT_RANDOM_KEY
from copulax._src.univariate._rvs import inverse_transform_sampling
from copulax._src.typing import Scalar
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src.multivariate._shape import cov, corr
from copulax._src.optimize import projected_gradient
from copulax._src.univariate._utils import _univariate_input

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
mvt_names: tuple = ("Mvt-Normal", "Mvt-Student-T", "Mvt-GH", "Mvt-Skewed-T",)
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
        r"""Returns a dictionary from an array / iterable of params"""
        return self._params_dict(*params_arr)

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
    def rvs(self, size, params: dict, key: Array = DEFAULT_RANDOM_KEY, 
            *args, **kwargs) -> Array:
        r"""Generate random variates from the distribution.
        
        Args:
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.
        
        Returns:
            jnp.ndarray: The generated random variates.
        """

    def sample(self, size, params: dict, key: Array = DEFAULT_RANDOM_KEY, 
               *args, **kwargs) -> Array:
        """Alias for the rvs method."""
        return self.rvs(size=size, params=params, key=key, *args, **kwargs)

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
        Utilises a stability term to help prevent the logpdf from 
        blowing to inf / nan during numerical optimisation, typically 
        resulting from log and 1 / x functions.

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
    def _support(self, *args, **kwargs) -> tuple:
        """needed for ppf."""
        pass
    
    @classmethod
    def support(cls, *args, **kwargs) -> Array:
        r"""The support of the distribution is the subset of x for which 
        the pdf is non-zero. 
        
        Returns:
            Array: Flattened array containing the support of the 
            distribution.
        """
        return jnp.array(cls._support(*args, **kwargs)).flatten()
    
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
    
    # cdf
    @classmethod
    def _params_from_array(cls, params_arr, *args, **kwargs):
        return cls._params_dict(*params_arr)

    @classmethod
    def _pdf_for_cdf(cls, x: ArrayLike, *params_tuple) -> Array:
        params_array: jnp.ndarray = jnp.asarray(params_tuple).flatten()
        params: dict = cls._params_from_array(params_array)
        return cls.pdf(x=x, params=params)
    
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

    # ppf
    @abstractmethod
    def _get_x0(self, params: dict) -> Scalar:
        """Returns the initial guess for the ppf function."""
        pass 
    

    def _ppf(self, x0: float, q: ArrayLike, params: dict, cubic: bool, 
             num_points: int, lr: float, maxiter: int) -> Array:
        return _ppf(dist=self, x0=x0, q=q, params=params, cubic=cubic, 
                    num_points=num_points, lr=lr, maxiter=maxiter)

    def ppf(self, q: ArrayLike, params: dict, cubic: bool = False, 
            num_points: int = 100, lr: float = 0.1, maxiter: int = 100) -> Array:
        r"""Percent point function (inverse of the CDF) of the 
        distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'cubic' 
            is a static argument.


        Args:
            q (ArrayLike): The quantile values. at which to evaluate the 
            ppf.
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.
            cubic (bool): Whether to use a cubic spline approximation 
                of the ppf function for faster computation. This can 
                also improve gradient estimates.
            num_points (int): The number of points to use for the cubic 
                spline approximation when approx is True.
            lr (float): The learning rate to use when numerically 
                solving for the ppf function via ADAM based gradient 
                descent.
            maxiter (int): The maximum number of iterations to use when 
                solving for the ppf function via ADAM based gradient 
                descent.

        Returns:
            Array: The inverse CDF values.
        """
        q, qshape = _univariate_input(q)
        x0: Scalar = self._get_x0(params=params)
        if cubic: 
            # approximating even if an analytical / more efficient solution exists
            x: jnp.ndarray = _ppf(dist=self, x0=x0, q=q, params=params, 
                                  cubic=True, num_points=num_points, lr=lr, 
                                  maxiter=maxiter)
        else: 
            x: jnp.ndarray = self._ppf(x0=x0, q=q, params=params, cubic=False, 
                                       num_points=num_points, lr=lr, 
                                       maxiter=maxiter)
        return x.reshape(qshape)
    
    @abstractmethod
    def inverse_cdf(self, q: ArrayLike, params: dict, cubic: bool = False, 
            num_points: int = 100, lr: float = 1.0, maxiter: int = 100) -> Array:
        r"""Percent point function (inverse of the CDF) of the 
        distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'cubic' 
            is a static argument.


        Args:
            q (ArrayLike): The quantile values. at which to evaluate the 
            ppf.
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.
            cubic (bool): Whether to use a cubic spline approximation 
                of the ppf function for faster computation. This can 
                also improve gradient estimates.
            num_points (int): The number of points to use for the cubic 
                spline approximation when approx is True.
            lr (float): The learning rate to use when numerically 
                solving for the ppf function via ADAM based gradient 
                descent.
            maxiter (int): The maximum number of iterations to use when 
                solving for the ppf function via ADAM based gradient 
                descent.

        Returns:
            Array: The inverse CDF values.
        """
        return self.ppf(q=q, params=params, cubic=cubic, num_points=num_points, 
                        lr=lr, maxiter=maxiter)

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
    
    # fitting
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
    
    def plot(self, params: dict, sample: jnp.ndarray = None, domain: tuple = None, bins: int=50, num_points: int = 100, figsize: tuple = (16, 8), grid: bool = True, show: bool = True, ppf_options: dict = None):
        r"""Plots the pdf, cdf and ppf of the distribution.

        Note:
            Not jittable.

        Args:
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.
            sample (jnp.ndarray): Sample data to plot alongside the 
                distribution, allowing for a visual goodness of fit via 
                QQ plots. Must be a univariate sample if provided. 
            domain (tuple): The domain of the distribution to plot over.
                Must be a tuple of the form (min, max). If None, the ppf 
                function will be used to generate the domain.
            bins (int): The number of bins to use for the histogram of
                the sample data, if provided.
            num_points (int): The number of points to use for plotting.
            figsize (tuple): The size of the figure for the plot.
            grid (bool): Whether to display a grid on the plot.
            show (bool): Whether to automatically show the plot by 
                internally calling plt.show().
            ppf_options (dict): Options for the ppf function. This is 
                kwargs style dictionary with keyword arguments for the
                keys paired with their assigned values. See the ppf 
                method function documentation for more details.
   Returns:
            None
        """
        params: dict = self._args_transform(params=params)

        # getting pdf and cdf domain
        jitted_ppf = jit(self.ppf, static_argnames=('cubic', 'maxiter'))
        delta: float = 1e-3
        if domain is None:
            support = self.support(params=params)

            # lower bound
            min_val, eps = support[0], 0.0
            while not jnp.isfinite(min_val):
                eps += delta
                min_val = jitted_ppf(q=jnp.array(eps), params=params, **ppf_options)

            # upper bound
            max_val, eps = support[1], 0.0
            while not jnp.isfinite(max_val):
                eps += delta
                max_val = jitted_ppf(q=jnp.array(1 - eps), params=params, **ppf_options)
        else:
            if (not isinstance(domain, Iterable)) or len(domain) != 2:
                raise ValueError("Domain must be a tuple of the form (min, max).")
            if not all(isinstance(i, Scalar) for i in domain):
                raise ValueError("Domain elements must be scalar values")
            if domain[0] >= domain[1]:
                raise ValueError("Domain must be a tuple of the form (min, max).")
            
            min_val, max_val = jnp.asarray(domain).flatten()

        x: Array = jnp.linspace(min_val, max_val, num_points)
        q: Array = jnp.linspace(delta, 1-delta, num_points)

        # pdf, cdf, ppf and rvs values
        pdf_vals: Array = self.pdf(x=x, params=params)
        cdf_vals: Array = self.cdf(x=x, params=params)
        ppf_vals: Array = jitted_ppf(q=q, params=params, **ppf_options)

        # plotting setup
        values: tuple = [pdf_vals, cdf_vals, ppf_vals]
        domains: tuple = [x, x, q]
        titles: tuple = ("PDF", "CDF", "Inverse CDF", "QQ-Plot")
        xlabels: tuple = ("x", "x", "P(X <= q)", "Theoretical Quantiles")
        ylabels: tuple = ("PDF", "P(X <= x)", "q", "Empirical Quantiles")
        dtype = float if self.dtype == 'continuous' else int
        printable_params: str = str({k: round(dtype(v), 2) for k, v in params.items()})
        name_with_params: str = f"{self.name}({printable_params})"

        # qq-plot
        if sample is not None:
            sample: Array = _univariate_input(sample)[0]
            sorted_sample: Array = sample.flatten().sort()
            N: int = sample.size
            empirical_sample_cdf = jnp.array([(sorted_sample <= xi).sum() / N 
                                              for xi in sorted_sample])
            theoretical_sample_cdf = self.cdf(x=sorted_sample, params=params)
            domains.append(theoretical_sample_cdf)
            values.append(empirical_sample_cdf)
        
        # plotting
        num_subplots = len(values)
        fig, ax = plt.subplots(1, num_subplots, figsize=figsize)
        fig.suptitle(name_with_params, fontsize=16)
        for i in range(num_subplots):
            if i == 0 and num_subplots == 4:
                ax[i].hist(sample, bins=bins, density=True, color='blue', label='Sample', zorder=0)
                ax[i].set_xlim(min_val, max_val)
            elif i == 3:
                ax[i].plot(cdf_vals, cdf_vals, color='blue', zorder=0, label='y=x')

            # plotting distribution
            if i < 3:
                ax[i].plot(domains[i], values[i], label=name_with_params, color='black', zorder=1)
            else:
                ax[i].scatter(domains[i], values[i], label=name_with_params, color='black', zorder=1)

            # labeling
            ax[i].set_title(titles[i])
            ax[i].set_xlabel(xlabels[i])
            ax[i].set_ylabel(ylabels[i])
            ax[i].grid(grid)
            ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                         fancybox=True, shadow=True, ncol=1)
        
        plt.tight_layout()
        if show:
            plt.show()
        
###############################################################################
# Multivariate Base Class
###############################################################################
class GeneralMultivariate(Distribution):
    r"""Base class for multivariate and copula distributions."""
    @abstractmethod
    def _classify_params(self, params: dict, 
                         scalar_names: tuple = tuple(), 
                         vector_names: tuple = tuple(), 
                         shape_names: tuple = tuple(), 
                         symmetric_shape_names: tuple = tuple(), 
                         corr_like_shape_names: tuple = tuple(), 
                         non_symmetric_shape_names: tuple = tuple()) -> dict:
        r"""Classify the distribution parameters into scalars, vectors
        and shapes.
        
        Args:
            params (dict)
            scalar_names (tuple[str]): Tuple of scalar parameter names.
            vector_names (tuple[str]): Tuple of vector parameter names.
            symmetric_shape_names (tuple[str]): Tuple of symmetric shape 
                parameter names. Diagonal elements are included in the 
                parameter count.
            corr_like_shape_names (tuple[str]): Tuple of correlation 
                matrix like symmetric shape parameter names. Diagonal 
                elements are not included in the parameter count.
            non_symmetric_shape_names (tuple[str]): Tuple of 
                non-symmetric shape parameter names.
        """

        classifications = {
            "scalars": {name: params[name] for name in scalar_names},
            "vectors": {name: params[name] for name in vector_names},
            "shapes": {name: params[name] for name in shape_names},
            "symmetric_shapes": {name: params[name] for name in symmetric_shape_names},
            "corr_like_shapes": {name: params[name] for name in corr_like_shape_names},
            "non_symmetric_shapes": {name: params[name] for name in non_symmetric_shape_names},
        }
        return classifications

    @abstractmethod
    def _get_dim(self, params: dict) -> int:
        r"""Returns the number of dimensions of the distribution."""

    def _args_transform(self, params: dict) -> dict:
        classifications: dict = self._classify_params(params=params)  # todo: issue is here where scalars are being included. need to think about how to do this
        d: int = self._get_dim(params=params)

        # scalars
        transformed_scalars: dict = Distribution._scalar_transform(classifications["scalars"])
        
        # vectors
        transformed_vectors: dict = {k: jnp.asarray(v, dtype=float).reshape((d, 1)) 
                                     for k, v in classifications["vectors"].items()}
        
        # shapes
        transformed_shapes: dict = {k: jnp.asarray(v, dtype=float).reshape((d, d))
                                     for k, v in classifications["shapes"].items()}
        
        return {**transformed_scalars, **transformed_vectors, **transformed_shapes}
    
    def _get_num_params(self, params: dict) -> int:
        r"""Returns the number of parameters of the distribution.

        Returns:
            int: The number of parameters.
        """
        classifications: dict = self._classify_params(params=params)
        dim: int = self._get_dim(params=params)

        # scalars
        scalars: tuple = classifications["scalars"]
        n_scalars: int = len(scalars)
    
        # vectors
        vectors: dict = classifications["vectors"]
        n_vectors: int = len(vectors) * dim

        # symmetric shapes
        symmetric_shapes: dict = classifications["symmetric_shapes"]
        n_symm_shapes: int = len(symmetric_shapes) * (dim * (dim + 1) // 2)

        # correlation-like shapes
        corr_like_shapes: dict = classifications["corr_like_shapes"]
        n_corr_like_shapes: int = len(corr_like_shapes) * (dim * (dim - 1) // 2)

        # non-symmetric shapes
        non_symmetric_shapes: dict = classifications["non_symmetric_shapes"]
        n_non_symm_shapes: int = len(non_symmetric_shapes) * (dim ** 2)
        
        # total
        return n_scalars + n_vectors + n_symm_shapes + n_corr_like_shapes + n_non_symm_shapes
    

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
    def rvs(self, size: int, params: dict, 
            key: Array = DEFAULT_RANDOM_KEY) -> Array:
        r"""Generates random samples from the distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'size' 
            is a static argument.

        Args:
            size (int): The size of the generated 
                output array of random numbers. Must be an integer. 
                Generates an (size, d) array of random numbers, where
                d is the number of dimensions inferred from the provided
                distribution parameters.
            params (dict): Parameters describing the distribution. See
                    the specific distribution class or the 'example_params' 
                    method for details.
            key (Array): The Key for random number generation.
        """
    
    @abstractmethod
    def sample(self, size: int, params: dict, 
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
    def aic(self, x: ArrayLike, params: dict) -> float:
        k: int = self._get_num_params(params=params)
        return super().aic(k=k, x=x, params=params)

    @abstractmethod
    def bic(self, x: ArrayLike, params: dict) -> float:
        x, _, n, _ = _multivariate_input(x)
        k: int = self._get_num_params(params=params)
        return super().bic(k=k, n=n, x=x, params=params)

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
    def _get_dim(self, params: dict) -> int:
        classifications: dict = self._classify_params(params)
        return jnp.asarray(list(classifications["vectors"].values())[0]).size

    def support(self, params: dict, 
                marginal_support: tuple = (-jnp.inf, jnp.inf), 
                *args, **kwargs) -> Array:
        d: int = self._get_dim(params=params)
        return jnp.concat([jnp.full((d, 1), marginal_support[0]), 
                           jnp.full((d, 1), marginal_support[1])], axis=1)
    
    @jit
    def _single_qi(self, carry: tuple, xi: jnp.ndarray) -> jnp.ndarray:
        mu, sigma_inv = carry
        return carry, lax.sub(xi, mu).T @ sigma_inv @ lax.sub(xi, mu)
    
    def _calc_Q(self, x: jnp.ndarray, mu: jnp.ndarray, sigma_inv: jnp.ndarray
                ) -> jnp.ndarray:
        r"""Calculates the Q vector (x - mu)^T @ sigma^-1 @ (x - mu)"""
        return lax.scan(f=self._single_qi, xs=x, 
                        init=(mu.flatten(), sigma_inv))[1]

    def logpdf(self, x: ArrayLike, params: dict) -> Array:
        r"""The log-probability density function (pdf) of the 
        distribution.

        Args:
            x (ArrayLike): The input at which to evaluate the log-pdf.
                Must be in the shape (n, d) where n is the number of
                samples and d is the number of dimensions.
            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.  

        Returns:
            Array: The log-pdf values.
        """
        return super().logpdf(x=x, params=params)


    def pdf(self, x: ArrayLike, params: dict) -> Array:
        r"""The probability density function (pdf) of the distribution.

        Args:
            x (ArrayLike): The input at which to evaluate the pdf.
                Must be in the shape (n, d) where n is the number of
                samples and d is the number of dimensions.

            params (dict): Parameters describing the distribution. See 
                the specific distribution class or the 'example_params' 
                method for details.  

        Returns:
            Array: The pdf values.
        """
        return super().pdf(x=x, params=params)
    
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
    # sampling
    def _rvs(self, key, n: int, W: Array, mu: Array, gamma: Array, sigma: Array) -> Array:
        r"""Generates random samples from the normal-mixture 
        distribution."""
        d: int = mu.size

        m: jnp.ndarray = mu + W * gamma

        Z: jnp.ndarray = random.normal(key, shape=(d, n))
        A: jnp.ndarray = jnp.linalg.cholesky(sigma)
        r: jnp.ndarray = jnp.sqrt(W) * (A @ Z)
        return (m + r).T
    
    # stats
    def _stats(self, w_stats: dict, mu: Array, gamma: Array, sigma: Array) -> dict:
        mean: Array = mu + w_stats["mean"] * gamma
        cov: Array = w_stats["mean"] * sigma + w_stats["variance"] * jnp.outer(gamma, gamma)
        return {
            "mean": mean, 
            "cov": cov, 
            "skewness": gamma,}

    # fitting
    @abstractmethod
    def _ldmle_inputs(self, d: int) -> tuple:
        """Returns the input arguments for the low dimensional MLE.
        Specifically, the projection options containing the constraints
        and the initial guess."""
        pass
    
    def _general_fit(self, x: ArrayLike, d: int, loc: jnp.ndarray, shape: jnp.ndarray, 
                     reconstruct_func_id: int, lr: float, maxiter: int) -> dict:
        # optimisation constraints and initial guess
        projection_options, params0 = self._ldmle_inputs(d)

        # ADAM gradient descent
        res: dict = projected_gradient(f=self._ldmle_objective, x0=params0, 
                                       projection_method='projection_box', 
                                       projection_options=projection_options, 
                                       x=x, loc=loc, shape=shape, 
                                       reconstruct_func_id=reconstruct_func_id,
                                       lr=lr, maxiter=maxiter,)
        
        # reconstructing the parameters
        optimised_params_arr: jnp.ndarray = res['x']
        optimised_params: tuple = self._reconstruct_ldmle_func(
            func_id=reconstruct_func_id, params_arr=optimised_params_arr, 
            loc=loc, shape=shape)
        return optimised_params
    
    def _ldmle_objective(self, params_arr: jnp.ndarray, x: jnp.ndarray, 
                         loc: jnp.ndarray, shape: jnp.ndarray, reconstruct_func_id) -> Scalar:
        params: dict = self._reconstruct_ldmle_func(
            func_id=reconstruct_func_id, params_arr=params_arr, loc=loc, shape=shape)
        return -self._stable_logpdf(stability=1e-30, x=x, params=params).sum()
    
    @abstractmethod
    def _ldmle_inputs(self, d: int) -> tuple:
        """Returns the input arguments for the low dimensional MLE.
        Specifically, the projection options containing the constraints
        and the initial guess."""
        pass
    
    def fit(self, x: ArrayLike, cov_method: str = 'pearson', 
            lr: float = 1e-4, maxiter: int = 100) -> dict:
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
            lr (float): Learning rate for optimization.
            maxiter (int): Maximum number of iterations for optimization.

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
            reconstruct_func_id=0, lr=lr, maxiter=maxiter,)
    
    @abstractmethod
    def _reconstruct_ldmle_params(self, params_arr: jnp.ndarray, 
                                  loc: jnp.ndarray, 
                                  shape: jnp.ndarray) -> dict:
        """Reconstructs the low dim MLE parameters from a flat array."""
        pass
    
    @abstractmethod
    def _reconstruct_ldmle_copula_params(self, params_arr: jnp.ndarray,
                                         loc: jnp.ndarray, 
                                         shape: jnp.ndarray) -> tuple:
        """Reconstructs the low dim MLE parameters from a flat array for 
        copula fitting."""
        pass

    def _reconstruct_ldmle_func(self, func_id: int, params_arr: jnp.ndarray, 
                                loc: jnp.ndarray, shape: jnp.ndarray) -> tuple:
        """Reconstructs the low dim MLE parameters from a flat array."""
        params_tuple: tuple = lax.cond(func_id == 0, 
                        self._reconstruct_ldmle_params, 
                        self._reconstruct_ldmle_copula_params, 
                        params_arr, loc, shape)
        return self._params_from_array(params_tuple)

    def _fit_copula(self, u: jnp.ndarray, corr_method: str = 'pearson', 
                    *args, **kwargs):
        d: dict = super()._fit_copula(u, corr_method, *args, **kwargs)
        return self._general_fit(
            x=d['u'], d=d['d'], loc=d['mu'], shape=d['sigma'], 
            reconstruct_func_id=1, *args, **kwargs)
        

