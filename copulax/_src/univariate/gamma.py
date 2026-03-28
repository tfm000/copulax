"""File containing the copulAX implementation of the Gamma distribution."""

import jax.numpy as jnp
from jax import lax, random, scipy
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src.special import igammainv
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import _resolve_key
from copulax._src.optimize import projected_gradient
from copulax._src.univariate.lognormal import lognormal


class Gamma(Univariate):
    r"""The gamma distribution is a two-parameter family of continuous probability
    distributions, which includes the exponential, Erlang and chi-squared
    distributions as special cases.

    We use the rate parameterization of the gamma distribution specified by
    McNeil et al (2005).

    https://en.wikipedia.org/wiki/Gamma_distribution"""

    alpha: Array = None
    beta: Array = None

    def __init__(self, name="Gamma", *, alpha=None, beta=None):
        """Initialize the Gamma distribution.

        Args:
            name: Display name for the distribution.
            alpha: Shape parameter.
            beta: Rate parameter.
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
        """Create a parameter dictionary from alpha (shape) and beta (rate)."""
        d: dict = {"alpha": alpha, "beta": beta}
        return cls._args_transform(d)

    def _params_to_tuple(self, params: dict):
        """Extract (alpha, beta) from the parameter dictionary."""
        params = self._args_transform(params)
        return params["alpha"], params["beta"]

    def example_params(self, *args, **kwargs):
        r"""Example parameters for the gamma distribution.

        This is a two parameter family, defined by alpha and beta
        parameters. Here we adopt the rate parameterization of the gamma.
        """
        return self._params_dict(alpha=1.0, beta=1.0)

    @classmethod
    def _support(cls, *args, **kwargs) -> Array:
        """Return the support ``[0, inf)``."""
        return jnp.array([0.0, jnp.inf])

    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, params: dict) -> Array:
        """Compute the numerically stabilized log-PDF of the Gamma distribution."""
        x, xshape = _univariate_input(x)
        alpha, beta = self._params_to_tuple(params)

        logpdf: jnp.ndarray = (
            alpha * jnp.log(beta + stability)
            - lax.lgamma(alpha)
            + (alpha - 1) * jnp.log(x)
            - beta * x
        )
        return logpdf.reshape(xshape)

    def logpdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the log probability density function."""
        return super().logpdf(x=x, params=params)

    def pdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the probability density function."""
        return super().pdf(x=x, params=params)

    def logcdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the log cumulative distribution function."""
        return super().logcdf(x=x, params=params)

    def cdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the CDF via the regularized incomplete gamma function."""
        params = self._resolve_params(params)
        x, xshape = _univariate_input(x)
        alpha, beta = self._params_to_tuple(params)
        cdf: jnp.ndarray = scipy.special.gammainc(a=alpha, x=beta * x)
        return self._enforce_support_on_cdf(
            x=x, cdf=cdf.reshape(xshape), params=params
        )

    # ppf
    def _ppf(self, q: ArrayLike, params: dict, *args, **kwargs) -> Array:
        """Compute the percent-point function (inverse CDF) via ``igammainv``."""
        alpha, beta = self._params_to_tuple(params)
        return igammainv(a=alpha, p=q) / beta

    # sampling
    def rvs(
        self, size: tuple | Scalar, params: dict = None, key: Array = None
    ) -> Array:
        """Generate random variates from the Gamma distribution."""
        params = self._resolve_params(params)
        key = _resolve_key(key)
        alpha, beta = self._params_to_tuple(params)
        unscales_rvs: jnp.ndarray = random.gamma(key, shape=size, a=alpha)
        return unscales_rvs / beta

    # stats
    def stats(self, params: dict = None) -> dict:
        """Compute distribution statistics (mean, mode, variance, std, skewness, kurtosis)."""
        params = self._resolve_params(params)
        alpha, beta = self._params_to_tuple(params)
        mean: float = alpha / beta
        mode: float = jnp.where(alpha >= 1.0, (alpha - 1) / beta, 0.0)
        variance: float = alpha / (beta**2)
        std: float = jnp.sqrt(variance)
        skewness: float = 2.0 / jnp.sqrt(alpha)
        kurtosis: float = 6.0 / alpha
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
        beta0: float = self.rvs(size=(), params=self.example_params())
        alpha0: float = x.mean() * beta0
        params0: jnp.ndarray = jnp.array([alpha0, beta0])

        res = projected_gradient(
            f=self._mle_objective,
            x0=params0,
            projection_method="projection_non_negative",
            x=x,
            lr=lr,
            maxiter=maxiter,
        )
        alpha, beta = res["x"]
        return self._params_dict(alpha=alpha, beta=beta)  # , res['fun']

    def fit(
        self, x: ArrayLike, lr: float = 0.1, maxiter: int = 100, name: str = None
    ):
        r"""Fit the distribution to the input data.

        Args:
            x (ArrayLike): The input data to fit the distribution to.
            lr (float): Learning rate for the fitting process.
            maxiter (int): Maximum number of iterations for the fitting process.
            name (str): Optional custom name for the fitted instance.

        Returns:
            dict: The fitted distribution parameters.
        """
        x: jnp.ndarray = _univariate_input(x)[0]
        return self._fitted_instance(
            self._fit_mle(x=x, lr=lr, maxiter=maxiter), name=name
        )


gamma = Gamma("Gamma")
