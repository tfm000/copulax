# """Module containing base classes for univariate distributions to inherit from.
# """
# from abc import abstractmethod
# from jax._src.typing import ArrayLike, Array
# import jax.numpy as jnp

# from copulax._src._distributions import Distribution
# from copulax._src.univariate._ppf import _ppf
# from copulax._src._utils import DEFAULT_RANDOM_KEY
# from copulax._src.univariate._rvs import inverse_transform_sampling
# from copulax._src.typing import Scalar


# class Univariate(Distribution):
#     r"""Base class for univariate distributions."""
#     @staticmethod
#     def _args_transform(*args, **kwargs) -> tuple[Scalar]:
#         return Distribution._scalar_transform(*args, **kwargs)


#     @abstractmethod
#     def support(*args, **kwargs) -> tuple[Scalar, Scalar]:
#         r"""The support of the distribution is the subset of x for which 
#         the pdf is non-zero. 
        
#         Returns:
#             (Scalar, Scalar): Tuple containing the support of the 
#             distribution.
#         """

#     def _stable_logpdf(self, stability: Scalar, x: ArrayLike, *args, **kwargs
#                        ) -> Array:
#         r"""Stable log-pdf function for distribution fitting."""

#     @abstractmethod
#     def logpdf(self, x: ArrayLike, *args, **kwargs) -> Array:
#         r"""The log-probability density function (pdf) of the 
#         distribution.

#         Args:
#             x (ArrayLike): The input at which to evaluate the log-pdf.

#         Returns:
#             Array: The log-pdf values.
#         """
#         return self._stable_logpdf(stability=0.0, x=x, *args, **kwargs)
    
#     @abstractmethod
#     def pdf(self, x: ArrayLike, *args, **kwargs) -> Array:
#         r"""The probability density function (pdf) of the distribution.

#         Args:
#             x (ArrayLike): The input at which to evaluate the pdf.
        
#         Returns:
#             Array: The pdf values.
#         """
#         return jnp.exp(self.logpdf(x, *args, **kwargs))
    
#     @abstractmethod
#     def logcdf(self, x: ArrayLike, *args, **kwargs) -> Array:
#         r"""The log-cumulative distribution function of the distribution.

#         Args:
#             x (ArrayLike): The input at which to evaluate the log-cdf.

#         Returns:
#             Array: The log-cdf values.
#         """
#         return jnp.log(self.cdf(x, *args, **kwargs))
    
#     @abstractmethod
#     def cdf(self, x: ArrayLike, *args, **kwargs) -> Array:
#         r"""Cumulative distribution function of the distribution.

#         Args:
#             x (ArrayLike): The input at which to evaluate the cdf.

#         Returns:
#             Array: The cdf values.
#         """

#     @abstractmethod
#     def ppf(self, x0: float, q: ArrayLike, *args, **kwargs) -> Array:
#         r"""Percent point function (inverse of the CDF) of the distribution.

#         Args:
#             q (ArrayLike): The quantile values. at which to evaluate the ppf.

#         Returns:
#             Array: The inverse CDF values.
#         """
#         params: dict = self._params_dict(*args, **kwargs)
#         return _ppf(cdf_func=self.cdf, bounds=self.support(), q=q, 
#                     params=params, x0=x0)
    
#     @abstractmethod
#     def inverse_cdf(self, q: ArrayLike, *args, **kwargs) -> Array:
#         r"""Percent point function (inverse of the CDF) of the distribution.

#         Args:
#             q (ArrayLike): The quantile values. at which to evaluate the ppf.

#         Returns:
#             Array: The inverse CDF values.
#         """

#     # sampling
#     @abstractmethod
#     def rvs(self, shape: tuple = (), key: Array = DEFAULT_RANDOM_KEY, *args, **kwargs) -> Array:
#         r"""Generates random samples from the distribution.

#         Note:
#             If you intend to jit wrap this function, ensure that 'shape' is a 
#             static argument.

#         Args:
#             shape (tuple): The shape of the output array.
#             key (Array): The Key for random number generation.
#         """
#         params: dict = self._params_dict(*args, **kwargs)
#         return inverse_transform_sampling(ppf_func=self.ppf, shape=shape, 
#                                           params=params, key=key)        
    
#     @abstractmethod
#     def sample(self, shape: tuple = (), key: Array = DEFAULT_RANDOM_KEY, *args, **kwargs) -> Array:
#         r"""Generates random samples from the distribution.

#         Note:
#             If you intend to jit wrap this function, ensure that 'shape' is a 
#             static argument.

#         Args:
#             shape (tuple): The shape of the output array.
#             key (Array): The Key for random number generation.
#         """
#         return self.rvs(shape=shape, key=key, *args, **kwargs)
    
#     # stats
#     @abstractmethod
#     def stats(self, *args, **kwargs) -> dict:
#         r"""Distribution statistics for the distribution.

#         Returns:
#             stats (dict): A dictionary containing the distribution statistics.
#         """
#         return {}
    
#     # metrics
#     @abstractmethod
#     def loglikelihood(self, x: ArrayLike, *args, **kwargs) -> float:
#         r"""Log-likelihood of the distribution given the data.

#         Args:
#             x (ArrayLike): The input data to evaluate the log-likelihood.
        
#         Returns:
#             loglikelihood (float): The log-likelihood value
#         """
#         return self.logpdf(x=x, *args, **kwargs).sum()

#     @abstractmethod
#     def aic(self, x: ArrayLike, *args, **kwargs) -> float:
#         r"""Akaike Information Criterion (AIC) of the distribution given the 
#         data. Can be used as a crude metric for model selection, by minimising.

#         Args:
#             x (ArrayLike): The input data to evaluate the AIC.

#         Returns:
#             aic (float): The AIC value.
#         """
#         k: int = len(args) + len(kwargs)
#         return 2 * k - 2 * self.loglikelihood(x=x, *args, **kwargs)
    
#     @abstractmethod
#     def bic(self, x: ArrayLike, *args, **kwargs) -> float:
#         r"""Bayesian Information Criterion (BIC) of the distribution given the 
#         data. Can be used as a crude metric for model selection, by minimising.

#         Args:
#             x (ArrayLike): The input data to evaluate the BIC.

#         Returns:
#             bic (float): The BIC value.
#         """
#         k: int = len(args) + len(kwargs)
#         n: int  = x.size
#         return k * jnp.log(n) - 2 * self.loglikelihood(x=x, *args, **kwargs)
    
#     # fitting
#     def _mle_objective(self, params: jnp.ndarray, x: jnp.ndarray, *args, **kwargs) -> Scalar:
#         r"""Negative log-likelihood of the distribution given the data.

#         Args:
#             x (ArrayLike): The input data to evaluate the negative log-likelihood.

#         Returns:
#             mle_objective (float): The negative log-likelihood value.
#         """
#         return -self._stable_logpdf(1e-30, x, *params, *args, **kwargs).sum()
