"""File containing the copulAX implementaion of the Generalized normal distribution."""

import jax.numpy as jnp
from jax import lax, random, scipy
from jax.scipy import special
from jax._src.typing import ArrayLike, Array
from tensorflow_probability.substrates import jax as tfp

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import _resolve_key, get_local_random_key
from copulax._src.optimize import projected_gradient
from copulax._src.univariate.gamma import gamma


class GenNormal(Univariate):
    r"""The symmetric generalized normal distribution is a three-parameter family of
    continuous probability distributions which generalizes the normal distribution
    by allowing for heavier or lighter tails. It includes both the normal distribution and the Laplace distribution as special cases.

    https://en.wikipedia.org/wiki/Generalized_normal_distribution
    """

    mu: Array = None
    alpha: Array = None
    beta: Array = None

    def __init__(self, name="GenNormal", *, mu=None, alpha=None, beta=None):
        super().__init__(name)
        self.mu = jnp.asarray(mu, dtype=float).reshape(()) if mu is not None else None
        self.alpha = (
            jnp.asarray(alpha, dtype=float).reshape(()) if alpha is not None else None
        )
        self.beta = (
            jnp.asarray(beta, dtype=float).reshape(()) if beta is not None else None
        )

    @property
    def _stored_params(self):
        if self.mu is None or self.alpha is None or self.beta is None:
            return None
        return {"mu": self.mu, "alpha": self.alpha, "beta": self.beta}

    @classmethod
    def _params_dict(cls, mu: Scalar, alpha: Scalar, beta: Scalar) -> dict:
        d: dict = {"mu": mu, "alpha": alpha, "beta": beta}
        return cls._args_transform(d)

    def _params_to_tuple(self, params: dict) -> tuple:
        params = self._args_transform(params)
        return params["mu"], params["alpha"], params["beta"]

    def example_params(self, *args, **kwargs) -> dict:
        r"""Example parameters for the generalized normal distribution.

        This is a three parameter family, with the generalized normal being defined by
        its location `mu`, scale `alpha` and shape `beta`.
        """
        return self._params_dict(mu=0.0, alpha=1.0, beta=2.0)

    @classmethod
    def _support(cls, *args, **kwargs) -> Array:
        return jnp.array([-jnp.inf, jnp.inf])

    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, params: dict) -> Array:
        x, xshape = _univariate_input(x)
        mu, alpha, beta = self._params_to_tuple(params)

        log_c: Scalar = (
            jnp.log(beta + stability)
            - jnp.log(2.0 * alpha)
            - special.gammaln(1.0 / (beta + stability))
        )
        logpdf: Array = log_c - (jnp.abs(x - mu) / (alpha)) ** beta
        return logpdf.reshape(xshape)

    def cdf(self, x: ArrayLike, params: dict = None) -> Array:
        params = self._resolve_params(params)
        x, xshape = _univariate_input(x)
        mu, alpha, beta = self._params_to_tuple(params)

        z: Array = (x - mu) / alpha
        incomplete_gamma_component = scipy.special.gammainc(
            a=1.0 / beta, x=(jnp.abs(z) ** beta)
        )
        cdf: Array = 0.5 * (1.0 + jnp.sign(z) * incomplete_gamma_component)
        return cdf.reshape(xshape)

    def _ppf(self, q: ArrayLike, params: dict = None) -> Array:
        params = self._resolve_params(params)
        q, qshape = _univariate_input(q)
        mu, alpha, beta = self._params_to_tuple(params)

        ppf_tensor = tfp.distributions.GeneralizedNormal(
            loc=mu, scale=alpha, power=beta
        ).quantile(q)
        return jnp.asarray(ppf_tensor).reshape(qshape)

    # sampling
    def rvs(
        self, size: tuple | Scalar, params: dict = None, key: Array = None
    ) -> Array:
        params = self._resolve_params(params)
        key = _resolve_key(key)
        mu, alpha, beta = self._params_to_tuple(params)
        rvs_tensor = tfp.distributions.GeneralizedNormal(
            loc=mu, scale=alpha, power=beta
        ).sample(sample_shape=size, seed=key)
        return jnp.asarray(rvs_tensor)

    # stats
    def stats(self, params: dict = None) -> dict:
        params = self._resolve_params(params)
        mu, alpha, beta = self._params_to_tuple(params)

        variance = alpha**2 * special.gamma(3.0 / beta) / special.gamma(1.0 / beta)
        kurtosis = (
            special.gamma(5.0 / beta)
            * special.gamma(1.0 / beta)
            / (special.gamma(3.0 / beta) ** 2)
        )
        return {
            "mean": mu,
            "median": mu,
            "mode": mu,
            "variance": variance,
            "skewness": 0.0,
            "kurtosis": kurtosis,
        }

    # fitting
    def _ldmle_objective(
        self, params_arr: jnp.ndarray, x: jnp.ndarray, sample_mean: Scalar
    ) -> jnp.ndarray:
        beta = params_arr[0]
        alpha = jnp.power(beta * jnp.mean(jnp.abs(x - sample_mean) ** beta), 1.0 / beta)
        return self._mle_objective(
            params_arr=jnp.array([sample_mean, alpha, beta]), x=x
        )

    def _fit_ldmle(self, x: ArrayLike, lr: float, maxiter: int) -> dict:
        x, _ = _univariate_input(x)
        sample_mean = jnp.mean(x)
        initial_params_arr = jnp.array(
            [gamma.rvs(size=(), params=gamma.example_params())]
        )  # Initial guess for beta, alpha will be computed from this and the data
        res = projected_gradient(
            f=self._ldmle_objective,
            x0=initial_params_arr,
            projection_method="projection_non_negative",
            x=x,
            sample_mean=sample_mean,
            lr=lr,
            maxiter=maxiter,
        )
        beta = res["x"][0]
        alpha = jnp.power(beta * jnp.mean(jnp.abs(x - sample_mean) ** beta), 1.0 / beta)
        return self._params_dict(mu=sample_mean, alpha=alpha, beta=beta)

    def fit(
        self,
        x: ArrayLike,
        method: str = "LDMLE",
        lr: float = 0.1,
        maxiter: int = 100,
    ):
        r"""Fit the distribution to the input data.

        Note:
            If you intend to jit wrap this function, ensure that 'method'
            is a static argument.

        Args:
            x: ArrayLike, input data to fit the distribution to.
            method: str, method to use for fitting. Currently only supports "LDMLE".
            lr: float, learning rate for the optimization algorithm. Only used if method is "LDMLE".
            maxiter: int, maximum number of iterations for the optimization algorithm. Only used if method is "LDMLE".

        Returns:
            GenNormal: An instance of the GenNormal distribution with fitted parameters.
        """
        x: jnp.ndarray = _univariate_input(x)[0]
        if method == "LDMLE":
            return self._fitted_instance(self._fit_ldmle(x, lr, maxiter))
        else:
            raise ValueError(f"Unsupported fitting method: {method}")


gen_normal = GenNormal("Gen-Normal")
