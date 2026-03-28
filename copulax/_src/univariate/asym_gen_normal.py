"""File containing the copulAX implementation of the Asymmetric Generalized Normal distribution."""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src.typing import Scalar
from copulax._src.univariate._utils import _univariate_input
from copulax._src.optimize import projected_gradient, brent
from copulax._src.univariate.normal import normal
from copulax._src.stats import skew, kurtosis as sample_kurtosis


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
        return self._enforce_support_on_cdf(
            x=x, cdf=cdf.reshape(xshape), params=params
        )

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
            - 6.0
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
    @staticmethod
    def _kurtosis_score(kappa_abs: Scalar, sample_kurt: Scalar) -> Scalar:
        r"""Residual of the excess kurtosis equation for |kappa|.

        The excess kurtosis of the AsymGenNormal is purely a function of kappa^2:

        .. math::

            \kappa_4(\kappa) = e^{4\kappa^2} + 2e^{3\kappa^2} + 3e^{2\kappa^2} - 6

        This is monotonically increasing in |kappa|, so Brent's method can
        find the unique root on [0, 2].

        Args:
            kappa_abs: Absolute value of the shape parameter (scalar, >= 0).
            sample_kurt: Sample excess kurtosis to match.

        Returns:
            Residual: theoretical kurtosis - sample kurtosis.
        """
        k2 = kappa_abs ** 2
        theoretical = jnp.exp(4 * k2) + 2 * jnp.exp(3 * k2) + 3 * jnp.exp(2 * k2) - 6.0
        return theoretical - sample_kurt

    @staticmethod
    def _sample_moments(x: jnp.ndarray) -> dict:
        r"""Compute method-of-moments estimates for (zeta, alpha, kappa).

        Algorithm:
            1. Compute sample excess kurtosis and skewness.
            2. Solve kurtosis(|kappa|) = sample_kurt via Brent on [0, 2]
               (kurtosis is symmetric in kappa and monotonically increasing in |kappa|).
            3. Determine sign: negative skew => kappa > 0, positive skew => kappa < 0.
            4. zeta = median(x) (the median is the MLE of zeta for this family).
            5. alpha from sample variance and the variance formula:
               Var(X) = (alpha/kappa)^2 * exp(kappa^2) * (exp(kappa^2) - 1).

        Returns:
            Parameter dictionary with MoM estimates.
        """
        sample_mean = jnp.mean(x)
        sample_std = jnp.std(x)
        sample_kurt = sample_kurtosis(x, fisher=True, bias=True)
        sample_skew = skew(x, bias=True)

        # Clip kurtosis to valid range [0, kurtosis(2)]
        # kurtosis(0) = 0, kurtosis(2) ≈ 9.2M
        sample_kurt = jnp.clip(sample_kurt, 0.01, 9e6)

        # Solve for |kappa| via Brent
        kappa_abs = brent(
            g=AsymGenNormal._kurtosis_score,
            bounds=jnp.array([0.0, 2.0]),
            maxiter=30,
            sample_kurt=sample_kurt,
        )
        kappa_abs = jnp.clip(kappa_abs, 0.01, 2.0)

        # Sign: negative skew => kappa > 0, positive skew => kappa < 0
        kappa = jnp.where(sample_skew < 0, kappa_abs, -kappa_abs)

        # zeta = median
        zeta = jnp.median(x)

        # alpha from variance
        var_scale = (kappa**2) / (jnp.exp(kappa**2) * (jnp.exp(kappa**2) - 1.0))
        alpha = jnp.sqrt(var_scale) * sample_std

        # Support safety: ensure all data is within the implied support.
        # For kappa < 0: support is [zeta + alpha/kappa, inf).
        #   Need: min(x) > zeta + alpha/kappa  =>  |kappa| < alpha / (zeta - min(x))
        # For kappa > 0: support is (-inf, zeta + alpha/kappa].
        #   Need: max(x) < zeta + alpha/kappa  =>  kappa < alpha / (max(x) - zeta)
        # Scale |kappa| down with 0.95 safety margin if it violates.
        margin = 0.95
        kappa_max_neg = margin * alpha / (zeta - jnp.min(x) + 1e-30)
        kappa_max_pos = margin * alpha / (jnp.max(x) - zeta + 1e-30)
        kappa = jnp.where(
            kappa < 0,
            jnp.maximum(kappa, -kappa_max_neg),  # clamp toward 0
            jnp.minimum(kappa, kappa_max_pos),    # clamp toward 0
        )

        return AsymGenNormal._params_dict(zeta=zeta, alpha=alpha, kappa=kappa)

    def _fit_mom(self, x: jnp.ndarray) -> dict:
        """Fit via method of moments (no MLE refinement).

        Returns parameter estimates derived purely from sample moments:
        kurtosis → |kappa| via Brent, sign from skewness, zeta from median,
        alpha from the variance formula.

        Args:
            x: Data array.

        Returns:
            Parameter dictionary with MoM estimates.
        """
        return self._sample_moments(x)

    def _fit_mle(self, x: jnp.ndarray, lr: float, maxiter: int) -> dict:
        """Fit via projected gradient MLE, initialized from method of moments.

        Uses MoM estimates (kurtosis inversion for kappa, median for zeta,
        variance formula for alpha) as starting point, then refines all three
        parameters via projected gradient descent on the negative log-likelihood.

        Args:
            x: Data array.
            lr: Learning rate for optimization.
            maxiter: Maximum number of iterations.

        Returns:
            Parameter dictionary with MLE estimates.
        """
        eps: float = 1e-8
        constraints: tuple = (
            jnp.array([[-jnp.inf, eps, -jnp.inf]]).T,
            jnp.array([[jnp.inf, jnp.inf, jnp.inf]]).T,
        )
        projection_options: dict = {"lower": constraints[0], "upper": constraints[1]}

        # MoM initialization
        mom_params = self._sample_moments(x)
        zeta0, alpha0, kappa0 = self._params_to_tuple(mom_params)
        params0: jnp.ndarray = jnp.array([zeta0, alpha0, kappa0])

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

    def fit(
        self,
        x: ArrayLike,
        method: str = "MLE",
        lr: float = 0.01,
        maxiter: int = 200,
        name: str = None,
    ):
        """Fit the distribution to data.

        Note:
            If you intend to jit wrap this function, ensure that 'method' is a
            static argument.

        Args:
            x: Input data to fit.
            method: Fitting method. Options are 'MLE' (default) for projected
                gradient maximum likelihood with MoM initialization, or 'MOM'
                for method-of-moments only (faster, no gradient refinement).
            lr: Learning rate for optimization (MLE only). Default 0.01.
            maxiter: Maximum number of iterations (MLE only).
            name: Optional custom name for the fitted instance.

        Returns:
            A new fitted AsymGenNormal instance.
        """
        x: jnp.ndarray = _univariate_input(x)[0]
        if method == "MLE":
            return self._fitted_instance(self._fit_mle(x, lr, maxiter), name=name)
        elif method == "MOM":
            return self._fitted_instance(self._fit_mom(x), name=name)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'MLE' or 'MOM'.")


asym_gen_normal = AsymGenNormal("Asym-Gen-Normal")
