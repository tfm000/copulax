"""File containing the copulAX implementation of the student-T distribution."""

import jax.numpy as jnp
from jax import lax, random
from jax.scipy import special
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import _resolve_key, get_local_random_key
from copulax._src.optimize import projected_gradient
from copulax._src.special import stdtr


class StudentT(Univariate):
    r"""The student-T distribution is a 3 parameter family of continuous
    distributions which generalize the normal distribuion, allowing it to have
    heavier tails.

    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    """

    nu: Array = None
    mu: Array = None
    sigma: Array = None

    def __init__(self, name="Student-T", *, nu=None, mu=None, sigma=None):
        """Initialize the Student-T distribution.

        Args:
            name: Display name for the distribution.
            nu: Degrees of freedom parameter.
            mu: Location parameter.
            sigma: Scale parameter.
        """
        super().__init__(name)
        self.nu = jnp.asarray(nu, dtype=float).reshape(()) if nu is not None else None
        self.mu = jnp.asarray(mu, dtype=float).reshape(()) if mu is not None else None
        self.sigma = (
            jnp.asarray(sigma, dtype=float).reshape(()) if sigma is not None else None
        )

    @property
    def _stored_params(self):
        """Return stored parameters if all are set, else None."""
        if self.nu is None or self.mu is None or self.sigma is None:
            return None
        return {"nu": self.nu, "mu": self.mu, "sigma": self.sigma}

    @classmethod
    def _params_dict(cls, nu: Scalar, mu: Scalar, sigma: Scalar) -> dict:
        """Create a parameter dictionary from nu, mu, and sigma values."""
        d: dict = {"nu": nu, "mu": mu, "sigma": sigma}
        return cls._args_transform(d)

    def _params_to_tuple(self, params: dict) -> tuple:
        """Extract (nu, mu, sigma) from the parameter dictionary."""
        params = self._args_transform(params)
        return params["nu"], params["mu"], params["sigma"]

    def example_params(self, *args, **kwargs) -> dict:
        return self._params_dict(nu=2.5, mu=0.0, sigma=1.0)

    @classmethod
    def _support(cls, *args, **kwargs) -> Array:
        """Return the support ``[-inf, inf]``."""
        return jnp.array([-jnp.inf, jnp.inf])

    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, params: dict) -> Array:
        """Compute the numerically stabilized log-PDF of the Student-T distribution."""
        x, xshape = _univariate_input(x)
        nu, mu, sigma = self._params_to_tuple(params)

        z: jnp.ndarray = lax.div(lax.sub(x, mu), sigma)

        const: jnp.ndarray = (
            special.gammaln(0.5 * (nu + 1))
            - special.gammaln(0.5 * nu)
            - 0.5 * lax.log(stability + (nu * jnp.pi)) - lax.log(sigma + stability)
        )
        e: jnp.ndarray = lax.mul(
            lax.log(stability + lax.add(1.0, lax.div(lax.pow(z, 2.0), nu))),
            -0.5 * (nu + 1),
        )
        logpdf = lax.add(const, e)
        return logpdf.reshape(xshape)

    def cdf(
        self,
        x: ArrayLike,
        params: dict = None,
    ) -> Array:
        """Compute the cumulative distribution function via ``stdtr``.

        Args:
            x: Input values at which to evaluate the CDF.
            params: Distribution parameters. Uses stored parameters if None.

        Returns:
            CDF values with the same shape as ``x``.
        """
        params = self._resolve_params(params)
        x, xshape = _univariate_input(x)
        nu, mu, sigma = self._params_to_tuple(params)

        t = lax.div(lax.sub(x, mu), sigma)
        cdf = stdtr(nu, t)
        return self._enforce_support_on_cdf(
            x=x, cdf=cdf.reshape(xshape), params=params
        )

    # sampling
    def rvs(
        self, size: tuple | Scalar, params: dict = None, key: Array = None
    ) -> Array:
        """Generate random variates from the Student-T distribution.

        Args:
            size: Shape of the output array.
            params: Distribution parameters. Uses stored parameters if None.
            key: JAX PRNG key. A default key is used if None.

        Returns:
            Array of random samples.
        """
        params = self._resolve_params(params)
        key = _resolve_key(key)
        nu, mu, sigma = self._params_to_tuple(params)
        z: jnp.ndarray = random.t(key=key, df=nu, shape=size)
        return lax.add(lax.mul(z, sigma), mu)

    # stats
    def stats(self, params: dict = None) -> dict:
        """Statistics conditional on degrees of freedom ``nu``; NaN where undefined."""
        params = self._resolve_params(params)
        nu, mu, sigma = self._params_to_tuple(params)
        mean: float = jnp.where(nu > 1, mu, jnp.nan)
        variance: float = jnp.where(nu > 2, sigma**2 * nu / (nu - 2), jnp.nan)
        std: float = jnp.sqrt(variance)
        skewness: float = jnp.where(nu > 3, 0.0, jnp.nan)
        kurtosis: float = jnp.where(nu > 4, 6 / (nu - 4), jnp.inf)
        kurtosis = jnp.where(nu <= 2, jnp.nan, kurtosis)

        return self._scalar_transform(
            {
                "mean": mean,
                "median": mu,
                "mode": mu,
                "variance": variance,
                "std": std,
                "skewness": skewness,
                "kurtosis": kurtosis,
            }
        )

    # fitting
    @staticmethod
    def _sample_moments(x: jnp.ndarray) -> tuple:
        """Compute method-of-moments initial estimates for (nu, mu, sigma).

        Uses sample mean, variance, and excess kurtosis to invert the
        analytical moment expressions:
            mu = mean(x)
            nu = 4 + 6 / kurtosis   (when kurtosis > 0)
            sigma = std(x) * sqrt((nu - 2) / nu)
        """
        mu0 = jnp.mean(x)
        s2 = jnp.var(x)
        n = x.shape[0]
        # Excess kurtosis (Fisher) — unbiased estimator
        m4 = jnp.mean((x - mu0) ** 4)
        kurt = m4 / (s2 ** 2) - 3.0
        # Invert kurtosis = 6 / (nu - 4) => nu = 4 + 6 / kurt
        nu0 = jnp.where(kurt > 0.05, 4.0 + 6.0 / kurt, 10.0)
        nu0 = jnp.clip(nu0, 4.0, 100.0)
        sigma0 = jnp.sqrt(s2 * (nu0 - 2.0) / nu0)
        return nu0, mu0, sigma0

    def _fit_mle(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        """Fit all three parameters via projected gradient MLE."""
        eps = 1e-8
        constraints: tuple = (
            jnp.array([[eps, -jnp.inf, eps]]).T,
            jnp.array([[jnp.inf, jnp.inf, jnp.inf]]).T,
        )

        projection_options: dict = {"lower": constraints[0], "upper": constraints[1]}
        nu0, mu0, sigma0 = self._sample_moments(x)
        params0: jnp.ndarray = jnp.array([nu0, mu0, sigma0])

        res = projected_gradient(
            f=self._mle_objective,
            x0=params0,
            projection_method="projection_box",
            projection_options=projection_options,
            x=x,
            lr=lr,
            maxiter=maxiter,
        )
        nu, mu, sigma = res["x"]

        return self._params_dict(nu=nu, mu=mu, sigma=sigma)

    def _ldmle_objective(
        self, params_arr: jnp.ndarray, x: jnp.ndarray, sample_mean: Scalar, sample_var: Scalar
    ) -> jnp.ndarray:
        """LDMLE objective that fixes mu to the sample mean and optimizes (nu, sigma)."""
        nu = params_arr.squeeze()
        sigma = jnp.sqrt(sample_var * (nu - 2) / nu)
        return self._mle_objective(params_arr=jnp.array([nu, sample_mean, sigma]), x=x)

    def _fit_ldmle(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        """Fit via low-dimensional MLE, fixing mu to the sample mean."""
        eps = 1e-8
        constraints: tuple = (
            jnp.array([[2 + eps]]).T,
            jnp.array([[jnp.inf]]).T,
        )

        projection_options: dict = {"lower": constraints[0], "upper": constraints[1]}
        nu0, mu0, sigma0 = self._sample_moments(x)
        params0: jnp.ndarray = jnp.array([nu0])

        sample_mean: float = mu0
        sample_var: float = x.var()
        res = projected_gradient(
            f=self._ldmle_objective,
            x0=params0,
            projection_method="projection_box",
            projection_options=projection_options,
            x=x,
            sample_mean=sample_mean,
            sample_var=sample_var,
            lr=lr,
            maxiter=maxiter,
        )
        nu = res["x"]
        sigma = jnp.sqrt(sample_var * (nu - 2) / nu)

        return self._params_dict(nu=nu, mu=sample_mean, sigma=sigma)

    _supported_methods = frozenset({"mle", "ldmle"})

    def fit(
        self,
        x: ArrayLike,
        method: str = "ldmle",
        lr: float = 0.1,
        maxiter: int = 100,
        name: str = None,
    ):
        r"""Fit the distribution to the input data via numerical MLE.

        Note:
            If you intend to jit wrap this function, ensure that
            ``method`` is a static argument.

        Args:
            x (ArrayLike): The input data to fit the distribution to.
            method (str): The fitting method to use.  One of:
                ``'mle'`` — full maximum likelihood estimation;
                ``'ldmle'`` — low-dimensional maximum likelihood
                estimation.  Defaults to ``'ldmle'``.
            lr (float): Learning rate for the fitting process.
            maxiter (int): Maximum number of iterations for the fitting
                process.
            name (str): Optional custom name for the fitted instance.

        Returns:
            StudentT: A fitted ``StudentT`` instance.

        Raises:
            ValueError: If ``method`` is not one of the accepted
                strings listed above.
        """
        self._check_method(method)
        x = _univariate_input(x)[0]
        if method == "mle":
            return self._fitted_instance(
                self._fit_mle(x, lr=lr, maxiter=maxiter), name=name
            )
        elif method == "ldmle":
            return self._fitted_instance(
                self._fit_ldmle(x, lr=lr, maxiter=maxiter), name=name
            )
        else:
            raise ValueError(
                f"Unknown Student-T fit method {method!r}. "
                f"Expected one of: {sorted(self._supported_methods)}."
            )


student_t = StudentT("Student-T")
