"""File containing the copulAX implementaion of the Generalized normal distribution."""

import jax.numpy as jnp
from jax import random, scipy
from jax.scipy import special
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src.special import igammainv
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import _resolve_key
from copulax._src.optimize import projected_gradient
from copulax._src.univariate.gamma import gamma
from copulax._src.univariate.normal import normal
from copulax._src.stats import skew


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
        """Initialize the Generalized Normal distribution.

        Args:
            name: Display name for the distribution.
            mu: Location parameter.
            alpha: Scale parameter.
            beta: Shape parameter (beta=2 gives the normal distribution).
        """
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
        """Return stored parameters if all are set, else None."""
        if self.mu is None or self.alpha is None or self.beta is None:
            return None
        return {"mu": self.mu, "alpha": self.alpha, "beta": self.beta}

    @classmethod
    def _params_dict(cls, mu: Scalar, alpha: Scalar, beta: Scalar) -> dict:
        """Create a parameter dictionary from mu, alpha, and beta values."""
        d: dict = {"mu": mu, "alpha": alpha, "beta": beta}
        return cls._args_transform(d)

    def _params_to_tuple(self, params: dict) -> tuple:
        """Extract (mu, alpha, beta) from the parameter dictionary."""
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
        """Return the support ``[-inf, inf]``."""
        return jnp.array([-jnp.inf, jnp.inf])

    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, params: dict) -> Array:
        """Compute the numerically stabilized log-PDF of the Generalized Normal."""
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

    def _ppf(self, q: ArrayLike, params: dict = None, *args, **kwargs) -> Array:
        """Compute the PPF via the inverse regularized incomplete gamma function."""
        params = self._resolve_params(params)
        q, qshape = _univariate_input(q)
        mu, alpha, beta = self._params_to_tuple(params)

        z = 2.0 * q - 1.0
        x = mu + jnp.sign(z) * alpha * jnp.power(
            igammainv(a=1.0 / beta, p=jnp.abs(z)), 1.0 / beta
        )
        return x.reshape(qshape)

    # sampling
    def rvs(
        self, size: tuple | Scalar, params: dict = None, key: Array = None
    ) -> Array:
        params = self._resolve_params(params)
        key = _resolve_key(key)
        mu, alpha, beta = self._params_to_tuple(params)
        key1, key2 = random.split(key)
        G = gamma.rvs(size=size, key=key1, params={"alpha": 1.0 / beta, "beta": 1.0})
        sign = 2.0 * random.bernoulli(key2, 0.5, shape=size).astype(float) - 1.0
        return mu + alpha * sign * jnp.power(G, 1.0 / beta)

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
            "skewness": jnp.float32(0.0),
            "kurtosis": kurtosis,
        }

    # fitting
    def _ldmle_objective(
        self, params_arr: jnp.ndarray, x: jnp.ndarray, sample_mean: Scalar
    ) -> jnp.ndarray:
        """LDMLE objective that optimizes beta and derives alpha from the data."""
        beta = params_arr[0]
        alpha = jnp.power(beta * jnp.mean(jnp.abs(x - sample_mean) ** beta), 1.0 / beta)
        return self._mle_objective(
            params_arr=jnp.array([sample_mean, alpha, beta]), x=x
        )

    def _fit_ldmle(self, x: ArrayLike, lr: float, maxiter: int) -> dict:
        """Fit via low-dimensional MLE, fixing mu to the sample mean."""
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
        lr: float = 0.1,
        maxiter: int = 100,
    ):
        """Fit the distribution to data using LDMLE.

        Args:
            x: Input data to fit.
            lr: Learning rate for optimization.
            maxiter: Maximum number of iterations.

        Returns:
            A new fitted GenNormal instance.
        """
        x: jnp.ndarray = _univariate_input(x)[0]
        return self._fitted_instance(self._fit_ldmle(x, lr, maxiter))


gen_normal = GenNormal("Gen-Normal")


class AsymGenNormal(Univariate):
    r"""The asymmetric generalized normal distribution is a three-parameter family of
    continuous probability distributions which generalizes the normal distribution
    by allowing for heavier or lighter tails, as well as skewness. It includes both the normal distribution and the Laplace distribution as special cases.

    https://en.wikipedia.org/wiki/Generalized_normal_distribution
    """

    zeta: Array = None
    alpha: Array = None
    kappa: Array = None

    def __init__(self, name="AsymGenNormal", *, zeta=None, alpha=None, kappa=None):
        """Initialize the Asymmetric Generalized Normal distribution.

        Args:
            name: Display name for the distribution.
            zeta: Location parameter.
            alpha: Scale parameter.
            kappa: Shape parameter controlling skewness.
        """
        super().__init__(name)
        self.zeta = (
            jnp.asarray(zeta, dtype=float).reshape(()) if zeta is not None else None
        )
        self.alpha = (
            jnp.asarray(alpha, dtype=float).reshape(()) if alpha is not None else None
        )
        self.kappa = (
            jnp.asarray(kappa, dtype=float).reshape(()) if kappa is not None else None
        )

    @property
    def _stored_params(self):
        """Return stored parameters if all are set, else None."""
        if self.zeta is None or self.alpha is None or self.kappa is None:
            return None
        return {"zeta": self.zeta, "alpha": self.alpha, "kappa": self.kappa}

    @classmethod
    def _params_dict(cls, zeta: Scalar, alpha: Scalar, kappa: Scalar) -> dict:
        """Create a parameter dictionary from zeta, alpha, and kappa values."""
        d: dict = {"zeta": zeta, "alpha": alpha, "kappa": kappa}
        return cls._args_transform(d)

    @classmethod
    def _params_to_tuple(cls, params: dict) -> tuple:
        """Extract (zeta, alpha, kappa) from the parameter dictionary."""
        params = cls._args_transform(params)
        return params["zeta"], params["alpha"], params["kappa"]

    def example_params(self, *args, **kwargs) -> dict:
        r"""Example parameters for the asymmetric generalized normal distribution.

        This is a three parameter family, with the asymmetric generalized normal being defined by
        its location `zeta`, scale `alpha` and shape `kappa`.
        """
        return self._params_dict(zeta=0.0, alpha=1.0, kappa=-0.5)

    @classmethod
    def _support(cls, params: dict) -> Array:
        """Return the support, which depends on kappa.

        When ``kappa < 0`` the support is ``[zeta + alpha/kappa, inf)``;
        when ``kappa > 0`` it is ``(-inf, zeta + alpha/kappa]``;
        when ``kappa == 0`` it is ``(-inf, inf)``.
        """
        zeta, alpha, kappa = cls._params_to_tuple(params)
        val = jnp.where(kappa == 0, jnp.inf, zeta + alpha / kappa)
        support = jnp.where(
            kappa < 0, jnp.array([val, jnp.inf]), jnp.array([-jnp.inf, val])
        )
        return support

    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, params: dict) -> Array:
        """Compute the numerically stabilized log-PDF of the Asymmetric Generalized Normal."""
        x, xshape = _univariate_input(x)
        zeta, alpha, kappa = self._params_to_tuple(params)

        z = (x - zeta) / (alpha + stability)
        y = jnp.where(
            kappa == 0, z, (-1.0 / (kappa + stability)) * jnp.log1p(-kappa * z)
        )
        log_pdf = normal.logpdf(y, params={"mu": 0.0, "sigma": 1.0}) - jnp.log(
            alpha - kappa * (x - zeta)
        )
        return log_pdf.reshape(xshape)

    def cdf(self, x: ArrayLike, params: dict = None) -> Array:
        """Compute the CDF via transformation to the standard normal."""
        params = self._resolve_params(params)
        x, xshape = _univariate_input(x)
        zeta, alpha, kappa = self._params_to_tuple(params)

        z = (x - zeta) / alpha
        y = jnp.where(kappa == 0, z, (-1.0 / kappa) * jnp.log1p(-kappa * z))
        cdf = normal.cdf(y, params={"mu": 0.0, "sigma": 1.0})
        return cdf.reshape(xshape)

    # sampling
    def rvs(
        self, size: tuple | Scalar, params: dict = None, key: Array = None
    ) -> Array:
        """Generate random variates via transformation of standard normals."""
        params = self._resolve_params(params)
        zeta, alpha, kappa = self._params_to_tuple(params)

        Z = normal.rvs(size=size, key=key, params={"mu": 0.0, "sigma": 1.0})
        X = jnp.where(
            kappa == 0,
            zeta + alpha * Z,
            zeta + alpha * (1 - jnp.exp(-kappa * Z)) / kappa,
        )
        return X

    # stats
    def stats(self, params: dict = None) -> dict:
        """Compute distribution statistics (mean, median, mode, variance, skewness, kurtosis)."""
        params = self._resolve_params(params)
        zeta, alpha, kappa = self._params_to_tuple(params)

        kappa_sq_exp = jnp.exp(kappa**2)
        mean = jnp.where(
            kappa == 0, zeta, zeta - (alpha / kappa) * (jnp.exp(0.5 * kappa**2) - 1.0)
        )
        variance = jnp.where(
            kappa == 0,
            alpha**2,
            (alpha / kappa) ** 2 * kappa_sq_exp * (kappa_sq_exp - 1.0),
        )
        skewness = jnp.where(
            kappa == 0,
            0.0,
            jnp.sign(kappa)
            * (3 * kappa_sq_exp - jnp.exp(3 * kappa**2) - 2)
            / ((kappa_sq_exp - 1) ** 1.5),
        )
        kurtosis = (
            jnp.exp(4 * kappa**2)
            + 2 * jnp.exp(3 * kappa**2)
            + 3 * jnp.exp(2 * kappa**2)
            - 3
        )
        return {
            "mean": mean,
            "median": zeta,
            "mode": zeta,
            "variance": variance,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

    # fitting
    def _fit_mle(self, x: ArrayLike, lr: float, maxiter: int) -> dict:
        """Fit all three parameters via projected gradient MLE with box constraints."""
        eps: float = 1e-8
        constraints: tuple = (
            jnp.array([[-jnp.inf, eps, -jnp.inf]]).T,
            jnp.array([[jnp.inf, jnp.inf, jnp.inf]]).T,
        )
        projection_options: dict = {"lower": constraints[0], "upper": constraints[1]}
        x, _ = _univariate_input(x)
        _kappa0 = normal.rvs(
            size=(), params=normal.example_params()
        )  # Initial guess for kappa, alpha and zeta will be computed from this and the data
        kappa0 = (jnp.abs(_kappa0) + eps) * jnp.sign(skew(x)) * -1
        params0: jnp.ndarray = jnp.array([jnp.median(x), jnp.std(x), kappa0])

        res: dict = projected_gradient(
            f=self._mle_objective,
            x0=params0,
            projection_method="projection_box",
            projection_options=projection_options,
            x=x,
            lr=lr,
            maxiter=maxiter,
        )
        zeta, alpha, kappa = res["x"]
        return self._params_dict(zeta=zeta, alpha=alpha, kappa=kappa)

    # def _ldmle_objective(
    #     self,
    #     params_arr: jnp.ndarray,
    #     x: jnp.ndarray,
    #     sample_median: Scalar,
    #     sample_mean: Scalar,
    # ) -> jnp.ndarray:
    #     kappa = params_arr[0]

    #     zeta_est = sample_median
    #     alpha_est = jnp.where(
    #         jnp.logical_or(kappa == 0, sample_median == sample_mean),
    #         jnp.std(x),
    #         kappa * (zeta_est - sample_mean) / (jnp.exp(0.5 * kappa**2) - 1),
    #     )
    #     return self._mle_objective(
    #         params_arr=jnp.array([zeta_est, alpha_est, kappa]), x=x
    #     )

    # def _fit_ldmle(self, x: ArrayLike, lr: float, maxiter: int) -> dict:
    #     x, _ = _univariate_input(x)
    #     sample_median = jnp.median(x)
    #     sample_mean = jnp.mean(x)
    #     initial_params_arr = jnp.array(
    #         [normal.rvs(size=(), params=normal.example_params())]
    #     )  # Initial guess for kappa, alpha and zeta will be computed from this and the data

    #     projection_options: dict = {
    #         "lower": jnp.array([jnp.inf]),
    #         "upper": jnp.array([jnp.inf]),
    #     }

    #     res = projected_gradient(
    #         f=self._ldmle_objective,
    #         x0=initial_params_arr,
    #         projection_method="projection_box",
    #         projection_options=projection_options,
    #         x=x,
    #         sample_median=sample_median,
    #         sample_mean=sample_mean,
    #         lr=lr,
    #         maxiter=maxiter,
    #     )
    #     kappa = res["x"][0]

    #     zeta_est = sample_median
    #     alpha_est = jnp.where(
    #         jnp.logical_or(kappa == 0, sample_median == sample_mean),
    #         jnp.std(x),
    #         kappa * (zeta_est - sample_mean) / (jnp.exp(0.5 * kappa**2) - 1),
    #     )
    #     return self._params_dict(zeta=zeta_est, alpha=alpha_est, kappa=kappa)

    def fit(self, x: ArrayLike, lr: float = 0.1, maxiter: int = 100):
        """Fit the distribution to data using MLE.

        Args:
            x: Input data to fit.
            lr: Learning rate for optimization.
            maxiter: Maximum number of iterations.

        Returns:
            A new fitted AsymGenNormal instance.
        """
        x: jnp.ndarray = _univariate_input(x)[0]
        # return self._fitted_instance(self._fit_ldmle(x, lr, maxiter))
        return self._fitted_instance(self._fit_mle(x, lr, maxiter))


asym_gen_normal = AsymGenNormal("Asym-Gen-Normal")
