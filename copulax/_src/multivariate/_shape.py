"""File containing the copulAX implementation of various covariance matrices."""
import jax.numpy as jnp
from jax import lax, random
from jax._src.typing import ArrayLike, Array

from copulax._src.multivariate._utils import _multivariate_input


def _pearson(x: jnp.ndarray) -> Array:
    return jnp.corrcoef(x, rowvar=False)


def _spearman(x: jnp.ndarray) -> Array:
    pass


def _kendall(x: jnp.ndarray) -> Array:
    pass


def _pp_kendall(x: jnp.ndarray) -> Array:
    pass


def _rm_pearson(x: jnp.ndarray) -> Array:
    pass


def _rm_kendall(x: jnp.ndarray) -> Array:
    pass


def _rm_spearman(x: jnp.ndarray) -> Array:
    pass


def _rm_pp_kendall(x: jnp.ndarray) -> Array:
    pass


def _laloux_pearson(x: jnp.ndarray) -> Array:
    pass


def _laloux_spearman(x: jnp.ndarray) -> Array:
    pass


def _laloux_kendall(x: jnp.ndarray) -> Array:
    pass


def _laloux_pp_kendall(x: jnp.ndarray) -> Array:
    pass


# def _ledoit_wolf(x: jnp.ndarray) -> Array:
#     pass


def corr(x: ArrayLike, method: str = 'laloux_pp_kendall') -> Array:
    r"""Compute the correlation matrix of the input data.

    Args:
        x: arraylike, input data.
        method: str, method to use for computing the correlation matrix. 
            Options are 'pearson', 'spearman', 'kendall', 'pp_kendall', 
            'rm_pearson', 'rm_kendall', 'rm_spearman', 'rm_pp_kendall', 
            'laloux_pearson', 'laloux_spearman', 'laloux_kendall' and 
            'laloux_pp_kendall'.
    
    Note:
        If you intend to jit wrap this function, ensure that 'method' is a 
        static argument.

    Returns:
        array, correlation matrix of the input data.
    """
    return _pearson(x)  # TODO: Implement other methods


def cov(x: ArrayLike, method: str = 'laloux_pp_kendall') -> Array:
    r"""Compute the covariance matrix of the input data. Note that many of the 
    available methods are correlation matrix-based and hence the 
    pseudo-covariance matrix (=covariance when method = 'pearson') 
    of sigma_diag * corr * sigma_diag is returned.

    Args:
        x: arraylike, input data.
        method: str, method to use for computing the correlation matrix. 
            Options are 'pearson', 'spearman', 'kendall', 'pp_kendall', 
            'rm_pearson', 'rm_kendall', 'rm_spearman', 'rm_pp_kendall', 
            'laloux_pearson', 'laloux_spearman', 'laloux_kendall' and 
            'laloux_pp_kendall'.
    
    Note:
        If you intend to jit wrap this function, ensure that 'method' is a 
        static argument.

    Returns:
        array, covariance matrix of the input data.
    """
    # calculating correlation matrix
    corr_matrix: jnp.ndarray = corr(x=x, method=method)

    # calculating the diagonal matrix of standard deviations
    sigma_diag: jnp.ndarray = jnp.diag(jnp.std(x, axis=0))

    # returning the pseudo covariance matrix
    return sigma_diag @ corr_matrix @ sigma_diag