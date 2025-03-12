# """Module containing base classes for multivariate distributions to 
# inherit from."""
# from abc import abstractmethod
# from jax._src.typing import ArrayLike, Array
# import jax.numpy as jnp
# from typing import Iterable

# from copulax._src._distributions import Distribution
# from copulax._src._utils import DEFAULT_RANDOM_KEY
# from copulax._src.typing import Scalar


# class Multivariate(Distribution):
#     r"""Base class for multivariate distributions."""
#     @staticmethod
#     def _args_transform(scalars: Iterable[Scalar], 
#                         vectors: Iterable[ArrayLike], 
#                         shapes: Iterable[ArrayLike], 
#                         n: int = jnp.nan) -> tuple[ArrayLike]:
#         # scalars
#         transformed_scalars: tuple = Distribution._scalar_transform(*scalars)
        
#         # vectors
#         transformed_vectors: tuple = tuple(
#             jnp.asarray(v, dtype=float).reshape((n, 1)) for v in vectors
#             )
        
#         # shapes
#         transformed_shapes: tuple = tuple(
#             jnp.asarray(shape, dtype=float).reshape((n, n)) for shape in shapes
#             )
        
#         return transformed_scalars, transformed_vectors, transformed_shapes
    

#     @abstractmethod
#     def support(self, *args, **kwargs) -> Array:
#         r"""The support of the distribution is the subset of 
#         multivariate x for which the pdf is non-zero. 
        
#         Returns:
#             Array: Array contatining the support of each variable in
#             the distribution, in an (d, 2) shape.
#         """

#     def _stable_logpdf(self, stability: Scalar, x: ArrayLike, *args, **kwargs
#                        ) -> Array:
#         r"""Stable log-pdf function for distribution fitting."""

#     @abstractmethod
#     def fit(self, x: ArrayLike, *args, **kwargs):
#         """Fit the distribution to the input data.

#         Note:
#             Data must be in the shape (n, d) where n is the number of 
#             samples and d is the number of dimensions.
        
#         Args:
#             x (ArrayLike): The input data to fit the distribution to.
#             kwargs: Additional keyword arguments to pass to the fit 
#                 method.
        
#         Returns:
#             dict: The fitted distribution parameters.
#         """