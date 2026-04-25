"""File containing the copulAX implementation of the multivariate normal
distribution."""

import jax.numpy as jnp
from jax import random
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Multivariate
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src._utils import _resolve_key
from copulax._src.multivariate._shape import cov


class MvtNormal(Multivariate):
    r"""The multivariate normal / Gaussian distribution is a
    generalization of the univariate normal distribution to d > 1
    dimensions.

    https://en.wikipedia.org/wiki/Multivariate_normal_distribution

    .. math::

        f(x|\mu, \Sigma) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)

    where :math:`\mu` is the mean vector and :math:`\sigma` the
    variance-covariance matrix of the data distribution.
    """

    mu: Array = None
    sigma: Array = None

    def __init__(self, name="Mvt-Normal", *, mu=None, sigma=None):
        """Initialize with optional stored parameters `mu` and `sigma`."""
        super().__init__(name)
        self.mu = jnp.asarray(mu, dtype=float) if mu is not None else None
        self.sigma = jnp.asarray(sigma, dtype=float) if sigma is not None else None

    @property
    def _stored_params(self):
        """Return stored parameters dict if all are set, else None."""
        if self.mu is None or self.sigma is None:
            return None
        return {"mu": self.mu, "sigma": self.sigma}

    def _classify_params(self, params: dict) -> dict:
        """Classify parameters into vector and shape groups."""
        return super()._classify_params(
            params=params,
            vector_names=("mu",),
            shape_names=("sigma",),
            symmetric_shape_names=("sigma",),
        )

    def _params_dict(self, mu: ArrayLike, sigma: ArrayLike) -> dict:
        """Construct a normalized parameters dict from `mu` and `sigma`."""
        d: dict = {"mu": mu, "sigma": sigma}
        return self._args_transform(d)

    def _params_to_tuple(self, params: dict) -> tuple:
        """Extract `(mu, sigma)` tuple from a parameters dict."""
        params = self._args_transform(params)
        return params["mu"], params["sigma"]

    def example_params(self, dim: int = 3, *args, **kwargs) -> dict:
        r"""Example parameters for the multivariate normal distribution.

        This is a two parameter family, defined by the mean / location
        vector `mu` and the variance-covariance matrix `sigma`.

        Args:
            dim: int, number of dimensions of the multivariate normal
                distribution. Default is 3.
        """
        return self._params_dict(mu=jnp.zeros((dim, 1)), sigma=jnp.eye(dim, dim))

    def support(self, params: dict = None) -> Array:
        """Return the support of the distribution: `(-inf, inf)` per dimension."""
        return super().support(params=params)

    def logpdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Log-probability density function of the multivariate normal.

        Args:
            x: Input data of shape (n, d).
            params: Distribution parameters with keys 'mu' and 'sigma'.

        Returns:
            Array of log-density values with shape (n, 1).
        """
        params = self._resolve_params(params)
        x, yshape, n, d = _multivariate_input(x)
        mu, sigma = self._params_to_tuple(params)

        const: jnp.ndarray = -0.5 * (
            d * jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(sigma)[1]
        )

        sigma_inv: Array = jnp.linalg.inv(sigma)
        Q: jnp.ndarray = self._calc_Q(x=x, mu=mu, sigma_inv=sigma_inv)

        logpdf: jnp.ndarray = -0.5 * Q + const
        return logpdf.reshape(yshape)

    # sampling
    def rvs(self, size: int, params: dict = None, key=None) -> Array:
        """Generate random samples from the multivariate normal.

        Args:
            size: Number of samples to draw.
            params: Distribution parameters.
            key: JAX random key.

        Returns:
            Array of shape (size, d).
        """
        params = self._resolve_params(params)
        key = _resolve_key(key)
        mu, sigma = self._params_to_tuple(params)
        return random.multivariate_normal(
            key=key, mean=mu.flatten(), cov=sigma, shape=(size,)
        )

    # stats
    def stats(self, params: dict = None) -> dict:
        """Compute distribution statistics (mean, median, mode, cov, skewness)."""
        params = self._resolve_params(params)
        mu, sigma = self._params_to_tuple(params)
        return {
            "mean": mu,
            "median": mu,
            "mode": mu,
            "cov": sigma,
            "skewness": jnp.zeros_like(mu),
        }

    # fitting
    _supported_methods = frozenset({"mle"})

    def fit(
        self, x: ArrayLike, sigma_method: str = "pearson", *args, name: str = None, **kwargs
    ) -> dict:
        r"""Fit the multivariate normal to data via **closed-form** MLE:
        :math:`\hat\mu = \operatorname{mean}(x)` (row-wise), and
        :math:`\hat\Sigma` via :func:`copulax.multivariate.cov` using
        the estimator chosen by ``sigma_method``.

        Note:
            If you intend to jit wrap this function, ensure that
            ``sigma_method`` is a static argument.

        Args:
            x: Input data of shape ``(n, d)``.
            sigma_method: Covariance estimator name forwarded to
                :func:`copulax.multivariate.cov` (default
                ``'pearson'``).
            name: Optional custom name for the fitted instance.

        Returns:
            MvtNormal: A fitted ``MvtNormal`` instance.
        """
        x, _, _, d = _multivariate_input(x)
        mu: jnp.ndarray = jnp.mean(x, axis=0)
        sigma: jnp.ndarray = cov(x=x, method=sigma_method)
        params = self._params_dict(mu=mu, sigma=sigma)
        return self._fitted_instance(params, name=name)



mvt_normal = MvtNormal("Mvt-Normal")
