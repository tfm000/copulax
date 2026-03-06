"""File containing the copulAX implementation of the Inverse Gaussian distribution."""

import jax.numpy as jnp
from jax import lax, random, scipy
from jax import Array
from jax.typing import ArrayLike
from copulax._src.special import igammacinv

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import _resolve_key, get_local_random_key
from copulax._src.optimize import projected_gradient
from copulax._src.univariate.gamma import gamma


class IG(Univariate):
    r"""The inverse gamma distribution is a two-parameter family of continuous
    probability distributions which represents the reciprocal of gamma distributed
    random variables.

    We use the rate parameterization of the inverse gamma distribution specified by
    McNeil et al (2005).

    https://en.wikipedia.org/wiki/Inverse-gamma_distribution
    """

    alpha: Array = None
    beta: Array = None

    def __init__(self, name="IG", *, alpha=None, beta=None):
        """Initialize the Inverse Gamma distribution.

        Args:
            name: Display name for the distribution.
            alpha: Shape parameter.
            beta: Scale parameter.
        """
        super().__init__(name)
        self.alpha = (
            jnp.asarray(alpha, dtype=float).reshape(()) if alpha is not None else None
        )
        self.beta = (
            jnp.asarray(beta, dtype=float).reshape(()) if beta is not None else None
        )

    @property
    def _stored_params(self):
        """Return stored parameters if all are set, else None."""
        if self.alpha is None or self.beta is None:
            return None
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def _params_dict(cls, alpha: Scalar, beta: Scalar) -> dict:
        """Create a parameter dictionary from alpha (shape) and beta (scale)."""
        d: dict = {"alpha": alpha, "beta": beta}
        return cls._args_transform(d)

    def _params_to_tuple(self, params):
        """Extract (alpha, beta) from the parameter dictionary."""
        params = self._args_transform(params)
        return params["alpha"], params["beta"]

    def example_params(self, *args, **kwargs) -> dict:
        r"""Example parameters for the inverse gamma distribution.

        This is a two parameter family, with the inverse gamma being
        defined by parameters `alpha` and `beta`.
        """
        return self._params_dict(alpha=1.0, beta=1.0)

    @classmethod
    def _support(cls, *args, **kwargs) -> Array:
        """Return the support ``(0, inf)``."""
        return jnp.array([0.0, jnp.inf])

    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, params: dict) -> Array:
        """Compute the numerically stabilized log-PDF of the Inverse Gamma distribution."""
        x, xshape = _univariate_input(x)
        alpha, beta = self._params_to_tuple(params)

        logpdf: jnp.ndarray = (
            alpha * jnp.log(beta + stability)
            - lax.lgamma(alpha)
            - (alpha + 1) * jnp.log(x)
            - (beta / x)
        )
        return logpdf.reshape(xshape)

    def cdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the CDF via the upper regularized incomplete gamma function."""
        params = self._resolve_params(params)
        x, xshape = _univariate_input(x)
        alpha, beta = self._params_to_tuple(params)
        cdf: jnp.ndarray = scipy.special.gammaincc(a=alpha, x=(beta / x))
        return cdf.reshape(xshape)

    # ppf
    def _ppf(self, q: ArrayLike, params: dict, *args, **kwargs) -> Array:
        """Compute the PPF via ``igammacinv``."""
        alpha, beta = self._params_to_tuple(params)
        return beta / igammacinv(a=alpha, p=q)

    # sampling
    def rvs(
        self, size: tuple | Scalar, params: dict = None, key: Array = None
    ) -> Array:
        """Generate random variates as the reciprocal of Gamma variates."""
        params = self._resolve_params(params)
        key = _resolve_key(key)
        return 1.0 / gamma.rvs(size=size, key=key, params=params)

    # stats
    def stats(self, params: dict = None) -> dict:
        """Compute distribution statistics.

        Returns NaN for moments that are undefined given the current alpha.
        """
        params = self._resolve_params(params)
        alpha, beta = self._params_to_tuple(params)
        mean: float = jnp.where(alpha > 1.0, beta / (alpha - 1), jnp.nan)
        mode: float = beta / (alpha + 1)
        variance: float = jnp.where(
            alpha > 2.0,
            lax.pow(beta, 2) / (lax.pow(alpha - 1, 2) * (alpha - 1)),
            jnp.nan,
        )
        std: float = jnp.sqrt(variance)
        skewness: float = jnp.where(
            alpha > 3.0, 4 * jnp.sqrt(alpha - 2) / (alpha - 3), jnp.nan
        )
        kurtosis: float = jnp.where(
            alpha > 4.0, 6 * (5 * alpha - 11) / ((alpha - 3) * (alpha - 4)), jnp.nan
        )
        return self._scalar_transform(
            {
                "mean": mean,
                "mode": mode,
                "variance": variance,
                "std": std,
                "skewness": skewness,
                "kurtosis": kurtosis,
            }
        )

    # fitting
    def _fit_mle(self, x: ArrayLike, lr: float, maxiter: int) -> dict:
        """Fit alpha and beta via projected gradient MLE."""
        key1, key2 = random.split(get_local_random_key())

        gamma_params: dict = gamma.example_params()
        params0: jnp.ndarray = jnp.array(
            [
                gamma.rvs(size=(), key=key1, params=gamma_params),
                gamma.rvs(size=(), key=key2, params=gamma_params),
            ]
        )

        res = projected_gradient(
            f=self._mle_objective,
            x0=params0,
            x=x,
            lr=lr,
            maxiter=maxiter,
            projection_method="projection_non_negative",
        )

        alpha, beta = res["x"]
        return self._params_dict(alpha=alpha, beta=beta)  # , res["fun"]

    def fit(self, x: ArrayLike, lr: float = 0.1, maxiter: int = 100):
        """Fit the distribution to the input data.

        Args:
            x: Input data to fit.
            lr: Learning rate for optimization.
            maxiter: Maximum number of iterations.

        Returns:
            A new fitted IG instance.
        """
        x: jnp.ndarray = _univariate_input(x)[0]
        return self._fitted_instance(self._fit_mle(x=x, lr=lr, maxiter=maxiter))


ig = IG("IG")
