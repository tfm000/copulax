"""File containing the copulAX implementation of the Generalized normal distribution."""

import jax.numpy as jnp
from jax import random, scipy
from jax.scipy import special
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src.special import igammainv, digamma
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src._utils import _resolve_key
from copulax._src.optimize import brent
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
        return self._enforce_support_on_cdf(
            x=x, cdf=cdf.reshape(xshape), params=params
        )

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
            - 3.0
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
    @staticmethod
    def _sample_moments(x: jnp.ndarray) -> Scalar:
        r"""Sample-median initial estimate for mu (robust under symmetry; preferred over sample mean for heavy-tailed / small-beta regimes)."""
        return jnp.median(x)

    @staticmethod
    def _mle_score(beta: Scalar, x: jnp.ndarray, mu: Scalar) -> Scalar:
        r"""Score function g(beta) whose root is the MLE of beta.

        From Wikipedia (Generalized normal distribution, Version 1):

        .. math::

            g(\beta) = 1 + \frac{\psi(1/\beta)}{\beta}
                     - \frac{\sum |x_i - \mu|^\beta \log|x_i - \mu|}
                            {\sum |x_i - \mu|^\beta}
                     + \frac{\log\!\bigl(\frac{\beta}{N}\sum |x_i-\mu|^\beta\bigr)}
                            {\beta}

        where psi is the digamma function.

        Args:
            beta: Shape parameter (scalar, > 0).
            x: Data array.
            mu: Location parameter (fixed).

        Returns:
            Scalar value of g(beta).
        """
        n = x.shape[0]
        abs_dev = jnp.abs(x - mu) + 1e-30  # avoid log(0)
        log_abs_dev = jnp.log(abs_dev)
        abs_dev_beta = abs_dev ** beta

        sum_abs_dev_beta = jnp.sum(abs_dev_beta)
        sum_weighted_log = jnp.sum(abs_dev_beta * log_abs_dev)

        inv_beta = 1.0 / beta
        psi_val = digamma(jnp.atleast_1d(inv_beta))[0]

        term1 = 1.0 + psi_val * inv_beta
        term2 = sum_weighted_log / sum_abs_dev_beta
        term3 = jnp.log(beta / n * sum_abs_dev_beta) * inv_beta

        return term1 - term2 + term3

    @staticmethod
    def _mu_score(mu: Scalar, x: jnp.ndarray, beta: Scalar) -> Scalar:
        r"""Derivative of sum |x_i - mu|^beta w.r.t. mu.

        .. math::

            \frac{d}{d\mu}\sum|x_i-\mu|^\beta
                = -\beta \sum |x_i-\mu|^{\beta-1}\,\mathrm{sign}(x_i-\mu)

        The root of this function is the MLE of mu given beta.

        Args:
            mu: Location parameter (scalar).
            x: Data array.
            beta: Shape parameter (fixed).

        Returns:
            Scalar derivative value.
        """
        diff = x - mu
        abs_diff = jnp.abs(diff) + 1e-30
        return -beta * jnp.sum(abs_diff ** (beta - 1.0) * jnp.sign(diff))

    def _fit_mle(self, x: jnp.ndarray) -> dict:
        r"""Fit via Wikipedia's MLE algorithm using Brent's method.

        Algorithm (single pass):
            1. mu_0 = mean(x)
            2. Solve g(beta) = 0 for beta via Brent (with mu fixed at mu_0)
            3. Solve d/dmu sum|x_i - mu|^beta = 0 for mu via Brent (with beta fixed)
            4. alpha = (beta/N * sum|x_i - mu|^beta)^(1/beta)

        Reference:
            https://en.wikipedia.org/wiki/Generalized_normal_distribution
        """
        n = x.shape[0]
        mu = self._sample_moments(x)

        # Step 1: Solve g(beta) = 0 for beta with mu fixed
        beta = brent(
            g=self._mle_score,
            bounds=jnp.array([0.1, 10.0]),
            maxiter=30,
            x=x,
            mu=mu,
        )
        beta = jnp.clip(beta, 0.1, 10.0)

        # Step 2: Solve d/dmu sum|x_i - mu|^beta = 0 for mu with beta fixed
        mu = brent(
            g=self._mu_score,
            bounds=jnp.array([jnp.min(x), jnp.max(x)]),
            maxiter=30,
            x=x,
            beta=beta,
        )

        # Step 3: Derive alpha analytically
        alpha = jnp.power(
            beta / n * jnp.sum(jnp.abs(x - mu) ** beta), 1.0 / beta
        )

        return self._params_dict(mu=mu, alpha=alpha, beta=beta)

    def _fit_mom(self, x: jnp.ndarray) -> dict:
        """Fit via method of moments (no MLE refinement).

        Uses the sample median as mu, solves the MLE score equation for
        beta via Brent, then derives alpha analytically.

        Args:
            x: Data array.

        Returns:
            Parameter dictionary with MoM estimates.
        """
        n = x.shape[0]
        mu = self._sample_moments(x)
        beta = brent(
            g=self._mle_score,
            bounds=jnp.array([0.1, 10.0]),
            maxiter=30,
            x=x,
            mu=mu,
        )
        beta = jnp.clip(beta, 0.1, 10.0)
        alpha = jnp.power(
            beta / n * jnp.sum(jnp.abs(x - mu) ** beta), 1.0 / beta
        )
        return self._params_dict(mu=mu, alpha=alpha, beta=beta)

    _supported_methods = frozenset({"mle", "mom"})

    def fit(self, x: ArrayLike, method: str = "mle", name: str = None):
        r"""Fit the distribution to data.

        Note:
            If you intend to jit wrap this function, ensure that
            ``method`` is a static argument.

        Args:
            x: Input data to fit.
            method: Fitting method.  One of:
                ``'mle'`` — MLE algorithm using Brent's method
                (derivative-free numerical root-finding; default);
                ``'mom'`` — **closed-form** method of moments (faster,
                no μ refinement step).
            name: Optional custom name for the fitted instance.

        Returns:
            GenNormal: A fitted ``GenNormal`` instance.

        Raises:
            ValueError: If ``method`` is not one of the accepted
                strings listed above.
        """
        self._check_method(method)
        x: jnp.ndarray = _univariate_input(x)[0]
        if method == "mle":
            return self._fitted_instance(self._fit_mle(x), name=name)
        elif method == "mom":
            return self._fitted_instance(self._fit_mom(x), name=name)
        else:
            raise ValueError(
                f"Unknown Gen-Normal fit method {method!r}. "
                f"Expected one of: {sorted(self._supported_methods)}."
            )


gen_normal = GenNormal("Gen-Normal")
