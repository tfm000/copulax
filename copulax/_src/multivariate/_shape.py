"""Correlation and covariance matrix estimation with optional denoising.

Provides 12 correlation estimators (4 base methods x 3 denoising
variants) and corresponding pseudo-covariance estimators, plus random
PSD matrix generation utilities.

Public API:
    corr              — correlation matrix estimation
    cov               — covariance matrix estimation
    random_correlation — generate a random valid correlation matrix
    random_covariance  — generate a random valid covariance matrix
"""

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
        counting only the upper triangle (:math:`i < j`).
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
        :math:`\binom{d}{2}` pairs.
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
        """Full Rousseeuw-Molenberghs denoising with valid correlation output.

        Uses diagonal rescaling (Rebonato-Jackel, 1999) to restore unit
        diagonal. This is a congruence transformation (D⁻¹AD⁻¹) which
        is guaranteed to preserve positive semi-definiteness.
        """
        new_A: jnp.ndarray = self._rm_incomplete(A, delta)
        return self._corr_from_cov(new_A)

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
        new_eigenvalues: jnp.ndarray = jnp.where(cond, positive_eigenvalues, fill_val)

        # reconstructing the matrix
        laloux: jnp.ndarray = (
            eigenvectors @ jnp.diag(new_eigenvalues) @ eigenvectors.T
        )
        return self._corr_from_cov(laloux)

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
        vars: jnp.ndarray = jnp.var(x, axis=0, ddof=1)
        return self._cov_from_vars(vars=vars, R=R)


_corr: Correlation = Correlation()


def corr(x: ArrayLike, method: str = "pearson", **kwargs) -> Array:
    r"""Compute the correlation matrix of the input data.

    Returns a symmetric, positive semi-definite matrix with unit
    diagonal and entries in [-1, 1].

    Four base estimators are available, each optionally combined with
    one of two eigenvalue-denoising techniques:

    **Base estimators:**

    - ``'pearson'`` — standard linear (Pearson) correlation.
    - ``'spearman'`` — Spearman rank correlation (Pearson applied to
      ranks).
    - ``'kendall'`` — Kendall's tau, a concordance-based rank
      correlation. More robust to outliers than Pearson/Spearman.
    - ``'pp_kendall'`` — pseudo-Pearson Kendall: converts Kendall's
      tau to Pearson via the elliptical identity
      :math:`\rho = \sin(\pi \tau / 2)`. Useful when
      variances/covariances are undefined or infinite (e.g. heavy-
      tailed elliptical distributions).

    **Denoising variants** (prefix + base estimator name):

    - ``'rm_*'`` — Rousseeuw-Molenberghs (1993) denoising. Clamps
      non-positive eigenvalues to ``delta`` (default 1e-5), then
      rescales to restore unit diagonal. Guarantees positive semi-
      definiteness. Use when the raw estimator may produce a non-PSD
      matrix (e.g. Kendall/Spearman on small samples).
    - ``'laloux_*'`` — Laloux et al. (1999) random-matrix-theory
      denoising. Eigenvalues inside the Marchenko-Pastur noise bulk
      are replaced by their mean; signal eigenvalues above the bulk
      upper bound :math:`(1 + \sqrt{d/n})^2` are preserved. Use when
      n/d is moderate and you want to separate signal from sampling
      noise.

    Both denoising methods accept a ``delta`` keyword argument
    (default 1e-5) controlling the eigenvalue floor.

    Args:
        x (ArrayLike): Input data of shape ``(n, d)`` where ``n`` is
            the number of observations and ``d`` is the number of
            variables.
        method (str): Correlation method. One of ``'pearson'``,
            ``'spearman'``, ``'kendall'``, ``'pp_kendall'``,
            ``'rm_pearson'``, ``'rm_spearman'``, ``'rm_kendall'``,
            ``'rm_pp_kendall'``, ``'laloux_pearson'``,
            ``'laloux_spearman'``, ``'laloux_kendall'``,
            ``'laloux_pp_kendall'``.
        **kwargs: Passed to the underlying method (e.g. ``delta``
            for denoised variants).

    Returns:
        Array: Correlation matrix of shape ``(d, d)``.

    Raises:
        ValueError: If ``method`` is not a recognised method name.

    Note:
        If you intend to jit wrap this function, ensure that ``method``
        is a static argument.
    """
    method: str = method.lower().strip()
    func: Callable = getattr(_corr, method, None)
    if func is None:
        raise ValueError(
            f"Unknown correlation method '{method}'."
        )
    return func(x=x, **kwargs)


def cov(x: ArrayLike, method: str = "pearson", **kwargs) -> Array:
    r"""Compute the covariance matrix of the input data.

    Constructs the covariance matrix as
    :math:`\Sigma = D \, R \, D` where :math:`R` is the correlation
    matrix from :func:`corr` and :math:`D = \text{diag}(\hat\sigma)`
    is the diagonal matrix of sample standard deviations (``ddof=1``).

    When ``method='pearson'`` this is equivalent to the standard
    sample covariance matrix (i.e. ``numpy.cov(x, rowvar=False)``).
    For non-Pearson methods the result is a *pseudo-covariance*:
    sample variances combined with an alternative correlation
    estimator.

    Args:
        x (ArrayLike): Input data of shape ``(n, d)`` where ``n`` is
            the number of observations and ``d`` is the number of
            variables.
        method (str): Correlation method passed to :func:`corr`.
            See :func:`corr` for available options.
        **kwargs: Passed to :func:`corr` (e.g. ``delta`` for denoised
            variants).

    Returns:
        Array: Covariance matrix of shape ``(d, d)``.

    Raises:
        ValueError: If ``method`` is not a recognised method name.

    Note:
        If you intend to jit wrap this function, ensure that ``method``
        is a static argument.
    """
    # calculating correlation matrix
    corr_matrix: jnp.ndarray = corr(x=x, method=method, **kwargs)

    # returning the implied pseudo covariance matrix
    return _corr._cov_from_corr(x=x, R=corr_matrix)


def random_correlation(size: int, key: Array = None) -> Array:
    r"""Generate a random positive-definite correlation matrix.

    Produces a symmetric matrix with unit diagonal, entries in
    [-1, 1], and strictly positive eigenvalues. Useful for testing,
    simulation, and initialisation of multivariate models.

    Uses the factors method: :math:`C = W W^\top + D` where
    :math:`W \sim \text{Uniform}(-1, 1)^{d \times d}` and
    :math:`D` is diagonal with entries in [0, 1]. The PSD matrix
    :math:`C` is then rescaled to a correlation matrix via
    :math:`R_{ij} = C_{ij} / \sqrt{C_{ii} C_{jj}}`.

    Args:
        size (int): Dimension ``d`` of the ``(d, d)`` output matrix.
        key (jax.random.PRNGKey, optional): JAX PRNG key. If ``None``,
            a key is generated automatically.

    Returns:
        Array: Random correlation matrix of shape ``(size, size)``.

    Note:
        If you intend to jit wrap this function, ensure that ``size``
        is a static argument.
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
    r"""Generate a random positive-definite covariance matrix with
    prescribed variances.

    Constructs :math:`\Sigma = D \, R \, D` where :math:`R` is a
    random correlation matrix from :func:`random_correlation` and
    :math:`D = \text{diag}(\sqrt{\text{vars}})`. The diagonal of the
    output equals the input ``vars``.

    Args:
        vars (Array): Variances of each variable. A 1-d array of
            length ``d``; the output shape will be ``(d, d)``.
        key (jax.random.PRNGKey, optional): JAX PRNG key. If ``None``,
            a key is generated automatically.

    Returns:
        Array: Random covariance matrix of shape ``(d, d)``.
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
