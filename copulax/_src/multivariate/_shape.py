"""File containing the copulAX implementation of various covariance 
matrices."""
from jax.tree_util import register_pytree_node
import jax.numpy as jnp
from jax import lax, random, jit
from jax._src.typing import ArrayLike, Array
import jax.scipy.stats as stats
from itertools import combinations
from typing import Callable

from copulax._src.multivariate._utils import _multivariate_input
from copulax._src.typing import Scalar


class Correlation:
    r"""Class for computing correlation matrices."""

    # Making object a pytree
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        register_pytree_node(cls,
                             cls.tree_flatten,
                             cls.tree_unflatten)

    @classmethod
    def tree_unflatten(cls, aux_data, values, **init_kwargs):
        return cls(**init_kwargs)

    def tree_flatten(self):
        return (), None

    # Standard correlation matrix implementations
    def _insure_valid(self, A: Array) -> Array:
        lower_triangular: jnp.ndarray = jnp.tril(A)
        return jnp.fill_diagonal(lower_triangular + lower_triangular.T, 1.0, inplace=False)

    def pearson(self, x: jnp.ndarray) -> Array:
        r"""Pearson correlation matrix."""
        pearson: jnp.ndarray = jnp.corrcoef(x, rowvar=False)
        return self._insure_valid(pearson)


    def spearman(self, x: jnp.ndarray) -> Array:
        r"""Spearman-rank correlation matrix."""
        ranks: jnp.ndarray = stats.rankdata(x, axis=0)
        spearman: jnp.ndarray = self.pearson(ranks)
        return self._insure_valid(spearman)

    @jit
    def _kendall_pair_calc(x: jnp.ndarray, y: jnp.ndarray, 
                           i: int, j: int) -> Scalar:
        return jnp.where(i < j, 
                         jnp.sign(x[i] - x[j]) * jnp.sign(y[i] - y[j]), 
                         0.0)
    
    @staticmethod 
    @jit  
    def _kendall_iter(carry, j: int):
        """scans over i, s.t. i < j"""
        pair_sum, x, y, js = carry
        _iter: Callable = lambda ps, i: (
            ps + Correlation._kendall_pair_calc(x=x, y=y, i=i, j=j), None)
        pair_sum += lax.scan(_iter, 0.0, js)[0]
        return (pair_sum, x, y, js), None

    @staticmethod
    @jit
    def _kendall_pair(x: jnp.ndarray, y: jnp.ndarray, n: int, js: jnp.ndarray
                      ) -> Scalar:
        """scans over j"""
        scale: Scalar = 2 / (n * (n - 1))
        res: tuple = lax.scan(Correlation._kendall_iter, (0.0, x, y, js), js)
        return res[0][0] * scale

    @staticmethod
    @jit
    def _fill_kendall(carry: tuple, indices: tuple) -> Array:
        x, kendall, n, js = carry
        i, j = indices
        kendall_pair: Scalar = Correlation._kendall_pair(x[:, i], x[:, j], n, js)
        kendall = kendall.at[i, j].set(kendall_pair)
        kendall = kendall.at[j, i].set(kendall_pair)
        return (x, kendall, n, js), None
    
    def kendall(self, x: jnp.ndarray) -> Array:
        r"""Kendall-tau correlation matrix.
        
        Note:
            This implementation is essentially a brute force version of 
            the O(n^2) algorithm (O(d * n^2) when calculating the entire 
            correlation matrix) and hence is not efficient and should be
            avoided for large datasets.
        """
        n, d = x.shape
        js: jnp.ndarray = jnp.arange(0, n, 1)
        indices: tuple = jnp.array(tuple(combinations(range(d), 2)))
        kendall: jnp.ndarray = jnp.ones((d, d))
        kendall = lax.scan(self._fill_kendall, (x, kendall, n, js), indices)[0][1]
        return self._insure_valid(kendall)

    # Alternative correlation matrix implementations
    def pp_kendall(self, x: jnp.ndarray) -> Array:
        """Pseudo-Pearson Kendall correlation matrix.
        
        Note:
            This assumes that the data is elliptically distributed and 
            hence has no skewness. It does however provide a method of 
            estimating the correlation matrix when variances/covariances
            are undefined or infinate.
        """
        kendall: jnp.ndarray = self.kendall(x)
        pp_kendall: jnp.ndarray = jnp.sin(0.5 * jnp.pi * kendall)
        return self._insure_valid(pp_kendall)
    
    # Rousseeuw and Molenberghs's denoising technique
    def _rm_denoising(self, A: jnp.ndarray, delta) -> tuple:
        eigenvalues, eigenvectors = jnp.linalg.eigh(A)
        positive_eigenvalues = jnp.where(eigenvalues > 0.0, eigenvalues, delta)
        return positive_eigenvalues.real, eigenvectors.real
    
    def _rm_incomplete(self, A: jnp.ndarray, delta: Scalar) -> Array:
        positive_eigenvalues, eigenvectors = self._rm_denoising(A, delta)
        new_A: jnp.ndarray = (eigenvectors 
                              @ jnp.diag(positive_eigenvalues) 
                              @ jnp.linalg.inv(eigenvectors))
        return new_A

    def _rm(self, A: jnp.ndarray, delta: Scalar) -> Array:
        new_A: jnp.ndarray = self._rm_incomplete(A, delta)
        return self._insure_valid(new_A)

    def rm_pearson(self, x: jnp.ndarray, delta: Scalar = 1e-5) -> Array:
        return self._rm(self.pearson(x), delta)
    
    def rm_spearman(self, x: jnp.ndarray, delta: Scalar = 1e-5) -> Array:
        return self._rm(self.spearman(x), delta)

    def rm_kendall(self, x: jnp.ndarray, delta: Scalar = 1e-5) -> Array:
        return self._rm(self.kendall(x), delta)
    
    def rm_pp_kendall(self, x: jnp.ndarray, delta: Scalar = 1e-5) -> Array:
        return self._rm(self.pp_kendall(x), delta)
    
    # Laloux et al.'s denoising technique
    def _laloux(self, x: jnp.ndarray, A: jnp.ndarray, delta: Scalar) -> Array:
        # performing RM denoising
        positive_eigenvalues, eigenvectors = self._rm_denoising(A, delta)

        # calculating the Bulk
        n, d = x.shape
        Q: Scalar = n / d
        bulk_ub: Scalar = (1 + jnp.pow(Q, -0.5))**2
        
        # replacing eigenvalues with mean
        cond: jnp.ndarray = positive_eigenvalues > bulk_ub
        k: Scalar = jnp.sum(cond)
        fill_val: Scalar = jnp.where(k > 0, positive_eigenvalues[~cond].mean(), 0.0)
        new_eigenvalues: jnp.ndarray = jnp.where(cond, positive_eigenvalues, fill_val)

        # reconstructing the matrix
        laloux: jnp.ndarray = eigenvectors @ jnp.diag(new_eigenvalues) @ jnp.linalg.inv(eigenvectors)
        return self._insure_valid(laloux)
    
    def laloux_pearson(self, x: jnp.ndarray, delta: Scalar = 1e-5) -> Array:
        return self._laloux(x, self.pearson(x), delta)
    
    def laloux_spearman(self, x: jnp.ndarray, delta: Scalar = 1e-5) -> Array:
        return self._laloux(x, self.spearman(x), delta)
    
    def laloux_kendall(self, x: jnp.ndarray, delta: Scalar = 1e-5) -> Array:
        return self._laloux(x, self.kendall(x), delta)
    
    def laloux_pp_kendall(self, x: jnp.ndarray, delta: Scalar = 1e-5) -> Array:
        return self._laloux(x, self.pp_kendall(x), delta)

    # def ledoit_wolf(self, x: jnp.ndarray) -> Array:
    #     pass

_corr: Correlation = Correlation()


def corr(x: ArrayLike, method: str = 'pearson', **kwargs) -> Array:
    r"""Compute the correlation matrix of the input data.

    Args:
        x: arraylike, input data.
        method: str, method to use for computing the correlation matrix. 
            Options are 'pearson', 'spearman', 'kendall', 'pp_kendall', 
            'rm_pearson', 'rm_kendall', 'rm_spearman', 'rm_pp_kendall', 
            'laloux_pearson', 'laloux_spearman', 'laloux_kendall' and 
            'laloux_pp_kendall'.
    
    Note:
        If you intend to jit wrap this function, ensure that 'method' is 
        a static argument.

    Returns:
        array, correlation matrix of the input data.
    """
    method: str = method.lower().strip()
    func: Callable = getattr(_corr, method, 'pearson')
    return func(x=x, **kwargs)


def cov(x: ArrayLike, method: str = 'pearson', **kwargs) -> Array:
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
    corr_matrix: jnp.ndarray = corr(x=x, method=method, **kwargs)

    # calculating the diagonal matrix of standard deviations
    sigma_diag: jnp.ndarray = jnp.diag(jnp.std(x, axis=0))

    # returning the pseudo covariance matrix
    return sigma_diag @ corr_matrix @ sigma_diag