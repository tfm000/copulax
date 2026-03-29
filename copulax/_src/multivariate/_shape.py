"""File containing the copulAX implementation of various covariance
matrices."""

import jax.numpy as jnp
from jax import lax, random, jit, vmap
from jax import Array
from jax.typing import ArrayLike
import jax.scipy.stats as stats
import equinox as eqx
from itertools import combinations
from typing import Callable
from copulax._src.univariate._utils import _univariate_input

from copulax._src.typing import Scalar
from copulax._src._utils import _resolve_key


class Correlation(eqx.Module):
    r"""Class for computing correlation matrices."""

    # Standard correlation matrix implementations
    def _ensure_valid(self, A: Array) -> Array:
        """Enforce symmetry and unit diagonal on a correlation matrix."""
        lower_triangular: jnp.ndarray = jnp.tril(A)
        return jnp.fill_diagonal(
            lower_triangular + lower_triangular.T, 1.0, inplace=False
        )

    def pearson(self, x: jnp.ndarray) -> Array:
        r"""Pearson correlation matrix."""
        pearson: jnp.ndarray = jnp.corrcoef(x, rowvar=False)
        return self._ensure_valid(pearson)

    def spearman(self, x: jnp.ndarray) -> Array:
        r"""Spearman-rank correlation matrix."""
        ranks: jnp.ndarray = stats.rankdata(x, axis=0)
        return self.pearson(ranks)

    @staticmethod
    @jit
    def _kendall_pair_vectorized(x_col: jnp.ndarray, y_col: jnp.ndarray) -> Scalar:
        r"""Compute Kendall's tau for a single pair of variables.

        Uses fully vectorized pairwise concordance via broadcasting,
        counting only the upper triangle ($i < j$).
        """
        n = x_col.shape[0]
        # (n,) -> (n,1) - (1,n) = (n,n) pairwise differences
        dx = x_col[:, None] - x_col[None, :]
        dy = y_col[:, None] - y_col[None, :]
        concordance = jnp.sign(dx) * jnp.sign(dy)
        # zero out lower triangle + diagonal, sum upper triangle
        mask = jnp.triu(jnp.ones((n, n)), k=1)
        return (concordance * mask).sum() * 2.0 / (n * (n - 1))

    def kendall(self, x: jnp.ndarray) -> Array:
        r"""Kendall's tau correlation matrix.

        Vectorized: pairwise concordances are computed via broadcasting
        for each dimension pair, then ``vmap`` parallelizes across all
        $\binom{d}{2}$ pairs.
        """
        n, d = x.shape
        indices = jnp.array(list(combinations(range(d), 2)))

        # Pre-extract column pairs: (num_pairs, n)
        cols_i = x[:, indices[:, 0]].T
        cols_j = x[:, indices[:, 1]].T

        taus = vmap(self._kendall_pair_vectorized)(cols_i, cols_j)

        # fill symmetric matrix
        kendall = jnp.eye(d)
        kendall = kendall.at[indices[:, 0], indices[:, 1]].set(taus)
        kendall = kendall.at[indices[:, 1], indices[:, 0]].set(taus)
        return self._ensure_valid(kendall)

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
        return self._ensure_valid(pp_kendall)

    # Rousseeuw and Molenberghs's denoising technique
    def _rm_denoising(self, A: jnp.ndarray, delta) -> tuple:
        """Rousseeuw-Molenberghs eigenvalue denoising.

        Replaces non-positive eigenvalues with `delta` to ensure
        positive semi-definiteness.

        Args:
            A: Input matrix.
            delta: Replacement value for non-positive eigenvalues.

        Returns:
            Tuple of (clamped eigenvalues, eigenvectors).
        """
        eigenvalues, eigenvectors = jnp.linalg.eigh(A)
        positive_eigenvalues = jnp.where(eigenvalues > 0.0, eigenvalues, delta)
        return positive_eigenvalues.real, eigenvectors.real

    def _rm_incomplete(self, A: jnp.ndarray, delta: Scalar) -> Array:
        """Rousseeuw-Molenberghs denoising without enforcing unit diagonal."""
        positive_eigenvalues, eigenvectors = self._rm_denoising(A, delta)
        new_A: jnp.ndarray = (
            eigenvectors @ jnp.diag(positive_eigenvalues) @ eigenvectors.T
        )
        return new_A

    def _rm(self, A: jnp.ndarray, delta: Scalar) -> Array:
        """Full Rousseeuw-Molenberghs denoising with valid correlation output."""
        new_A: jnp.ndarray = self._rm_incomplete(A, delta)
        return self._ensure_valid(new_A)

    def rm_pearson(self, x: jnp.ndarray, delta: Scalar = 1e-5) -> Array:
        """Denoised Pearson correlation matrix via Rousseeuw-Molenberghs."""
        return self._rm(self.pearson(x), delta)

    def rm_spearman(self, x: jnp.ndarray, delta: Scalar = 1e-5) -> Array:
        """Denoised Spearman correlation matrix via Rousseeuw-Molenberghs."""
        return self._rm(self.spearman(x), delta)

    def rm_kendall(self, x: jnp.ndarray, delta: Scalar = 1e-5) -> Array:
        """Denoised Kendall correlation matrix via Rousseeuw-Molenberghs."""
        return self._rm(self.kendall(x), delta)

    def rm_pp_kendall(self, x: jnp.ndarray, delta: Scalar = 1e-5) -> Array:
        """Denoised pseudo-Pearson Kendall matrix via Rousseeuw-Molenberghs."""
        return self._rm(self.pp_kendall(x), delta)

    # Laloux et al.'s denoising technique
    def _laloux(self, x: jnp.ndarray, A: jnp.ndarray, delta: Scalar) -> Array:
        """Laloux et al. random-matrix-theory denoising.

        Eigenvalues inside the Marchenko-Pastur bulk are replaced by
        their mean, while signal eigenvalues above the bulk upper
        bound are preserved.

        Args:
            x: Input data of shape (n, d), used to compute Q = n/d.
            A: Correlation matrix to denoise.
            delta: Floor for non-positive eigenvalues.

        Returns:
            Denoised correlation matrix.
        """
        # performing RM denoising
        positive_eigenvalues, eigenvectors = self._rm_denoising(A, delta)

        # calculating the Bulk
        n, d = x.shape
        Q: Scalar = n / d
        bulk_ub: Scalar = (1 + jnp.pow(Q, -0.5)) ** 2

        # replacing eigenvalues with mean
        cond: jnp.ndarray = positive_eigenvalues > bulk_ub
        k: Scalar = jnp.sum(cond)
        denominator: Scalar = jnp.where(d - k > 0, d - k, 1.0)
        fill_val: Scalar = (
            jnp.where(~cond, positive_eigenvalues, 0.0).sum() / denominator
        )
        # fill_val: Scalar = jnp.where(k > 0, positive_eigenvalues[~cond].mean(), 0.0)
        new_eigenvalues: jnp.ndarray = jnp.where(cond, positive_eigenvalues, fill_val)

        # reconstructing the matrix
        laloux: jnp.ndarray = (
            eigenvectors @ jnp.diag(new_eigenvalues) @ eigenvectors.T
        )
        return self._ensure_valid(laloux)

    def laloux_pearson(self, x: jnp.ndarray, delta: Scalar = 1e-5) -> Array:
        """Denoised Pearson correlation matrix via Laloux et al."""
        return self._laloux(x, self.pearson(x), delta)

    def laloux_spearman(self, x: jnp.ndarray, delta: Scalar = 1e-5) -> Array:
        """Denoised Spearman correlation matrix via Laloux et al."""
        return self._laloux(x, self.spearman(x), delta)

    def laloux_kendall(self, x: jnp.ndarray, delta: Scalar = 1e-5) -> Array:
        """Denoised Kendall correlation matrix via Laloux et al."""
        return self._laloux(x, self.kendall(x), delta)

    def laloux_pp_kendall(self, x: jnp.ndarray, delta: Scalar = 1e-5) -> Array:
        """Denoised pseudo-Pearson Kendall matrix via Laloux et al."""
        return self._laloux(x, self.pp_kendall(x), delta)

    # helper functions
    def _corr_from_cov(self, C: jnp.ndarray) -> Array:
        """Convert covariance matrix to correlation matrix."""
        sigma_inv: jnp.ndarray = 1.0 / jnp.sqrt(jnp.diag(C))
        R: jnp.ndarray = C * jnp.outer(sigma_inv, sigma_inv)
        return R

    def _cov_from_vars(self, vars: jnp.ndarray, R: jnp.ndarray) -> Array:
        """Convert variances and correlation matrix to covariance matrix."""
        # calculating the diagonal matrix of standard deviations
        sigma_diag: jnp.ndarray = jnp.diag(jnp.sqrt(vars.flatten()))

        # returning the implied pseudo covariance matrix
        return sigma_diag @ R @ sigma_diag

    def _cov_from_corr(self, x: jnp.ndarray, R: jnp.ndarray) -> Array:
        """Convert correlation matrix to covariance matrix."""
        # calculating the variances of the input data
        vars: jnp.ndarray = jnp.var(x, axis=0)
        return self._cov_from_vars(vars=vars, R=R)


_corr: Correlation = Correlation()


def corr(x: ArrayLike, method: str = "pearson", **kwargs) -> Array:
    r"""Compute the correlation matrix of the input data.

    Args:
        x: arraylike, input data.
        method: str, method to use for computing the correlation matrix.
            Options are 'pearson', 'spearman', 'kendall', 'pp_kendall',
            'rm_pearson', 'rm_kendall', 'rm_spearman', 'rm_pp_kendall',
            'laloux_pearson', 'laloux_spearman', 'laloux_kendall' and
            'laloux_pp_kendall'.

    Note:
        If you intend to jit wrap this function, ensure that 'method'
        is a static argument.

    Returns:
        array, correlation matrix of the input data.
    """
    method: str = method.lower().strip()
    func: Callable = getattr(_corr, method, None)
    if func is None:
        raise ValueError(
            f"Unknown correlation method '{method}'."
        )
    return func(x=x, **kwargs)


def cov(x: ArrayLike, method: str = "pearson", **kwargs) -> Array:
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
        If you intend to jit wrap this function, ensure that 'method'
        is a static argument.

    Returns:
        array, covariance matrix of the input data.
    """
    # calculating correlation matrix
    corr_matrix: jnp.ndarray = corr(x=x, method=method, **kwargs)

    # returning the implied pseudo covariance matrix
    return _corr._cov_from_corr(x=x, R=corr_matrix)


def random_correlation(size: int, key: Array = None) -> Array:
    r"""Efficiently generates a random correlation matrix of given size.

    Note:
        If you intend to jit wrap this function, ensure that 'size'
        is a static argument.

        Uses the factors method described in:
        https://stats.stackexchange.com/questions/124538/how-to-generate-a-large-full-rank-random-correlation-matrix-with-some-strong-cor

    Args:
        size (int): size of the correlation matrix.
        key (jax.random.PRNGKey): jax random key, used for generating
            random numbers.

    Returns:
        rand_corr: random correlation matrix of given size.
    """
    key = _resolve_key(key)
    # generating random covariance matrix
    key, subkey = random.split(key)
    W: Array = random.uniform(key=key, shape=(size, size), minval=-1.0, maxval=1.0)
    D: Array = jnp.diag(
        random.uniform(key=subkey, shape=(size,), minval=0.0, maxval=1.0)
    )
    C: Array = W @ W.T + D

    # converting covariance matrix to correlation matrix
    R: Array = _corr._corr_from_cov(C=C)
    return R


def random_covariance(vars: Array, key: Array = None) -> Array:
    r"""Efficiently generates a random covariance matrix of given size.

    Args:
        vars (Array): Variances of the covariates and implies the size
            of the covariance matrix.
        key (jax.random.PRNGKey): jax random key, used for generating
            random numbers.

    Returns:
        rand_cov: random covariance matrix of given size.
    """
    key = _resolve_key(key)
    # we could simply use the same approach as in random_correlation,
    # to generate the covariance matrix C. However, whilst this would
    # be more efficient and would negate the need for the vars argument,
    # the scale of the covariances in C can become large and disjoint
    # from any relevant data distribution.
    vars, _ = _univariate_input(vars)
    R: Array = random_correlation(size=vars.size, key=key)
    return _corr._cov_from_vars(vars=vars, R=R)
