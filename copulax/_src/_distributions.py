"""Module containing base classes for all distributions to inherit from."""

from abc import abstractmethod
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax import lax, jit, random
import matplotlib.pyplot as plt
from typing import Iterable, ClassVar
import equinox as eqx


from copulax._src.typing import Scalar
from copulax._src.univariate._ppf import _ppf
from copulax._src._utils import _resolve_key
from copulax._src.univariate._rvs import inverse_transform_sampling
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src.multivariate._shape import cov, corr
from copulax._src.optimize import projected_gradient
from copulax._src.univariate._utils import _univariate_input


###############################################################################
# Parameter equality helpers
###############################################################################
def _params_equal(a: dict, b: dict) -> bool:
    """Recursively compare two parameter dictionaries for equality.

    Handles JAX/NumPy arrays via ``jnp.array_equal``, tuples of
    ``(Distribution, params_dict)`` pairs (copula marginals), and nested
    dicts.
    """
    if a.keys() != b.keys():
        return False
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, tuple) and isinstance(vb, tuple):
            # Copula marginals: tuple of (dist, params_dict) pairs
            if len(va) != len(vb):
                return False
            for (da, pa), (db, pb) in zip(va, vb):
                if type(da) is not type(db):
                    return False
                if not _params_equal(pa, pb):
                    return False
        elif isinstance(va, dict) and isinstance(vb, dict):
            if not _params_equal(va, vb):
                return False
        elif isinstance(va, (jnp.ndarray,)) or hasattr(va, "shape"):
            if not jnp.array_equal(va, vb):
                return False
        else:
            if va != vb:
                return False
    return True


###############################################################################
# Distribution PyTree / base class
###############################################################################
class Distribution(eqx.Module):
    r"""Base class for all implemented copulAX distributions."""

    _name: str = eqx.field(static=True)

    def __init__(self, name: str):
        self._name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __hash__(self):
        # Object-identity hash: required by equinox/JAX for JIT tracing
        # of bound methods.  Does NOT imply value-based identity —
        # use __eq__ for semantic comparison.
        return id(self)

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        sp = self._stored_params
        op = other._stored_params
        if sp is None and op is None:
            return True
        if sp is None or op is None:
            return False
        return _params_equal(sp, op)

    @property
    def _stored_params(self):
        """Override in subclasses to return a params dict from stored fields."""
        return None

    def _resolve_params(self, params):
        """Return params if provided, else fall back to stored params."""
        if params is not None:
            return params
        sp = self._stored_params
        if sp is not None:
            return sp
        raise ValueError(
            "No parameters provided. Pass a params dict or create "
            "the distribution with parameters."
        )

    @property
    def name(self) -> str:
        """The name of the distribution."""
        return self._name

    @property
    def params(self):
        """The stored distribution parameters, or None."""
        return self._stored_params

    def save(self, path: str) -> None:
        """Save the fitted distribution to a ``.cpx`` file.

        The file can be loaded back with :func:`copulax.load`.

        Args:
            path: File path to save to.  The ``.cpx`` extension is
                added automatically if not present.

        Raises:
            ValueError: If the distribution has no fitted parameters.
        """
        from copulax._src._serialization import _save_distribution
        _save_distribution(self, path)

    def _fitted_instance(self, params_dict: dict, name: str = None):
        """Create a new instance of this distribution with fitted parameters.

        Args:
            params_dict: Fitted parameter values.
            name: Optional custom name for the fitted instance. If ``None``,
                an auto-generated name is used.

        Returns:
            A new distribution instance with the given parameters.
        """
        cls = type(self)
        if name is None:
            name = f"Fitted{cls.__name__}-{id(params_dict):x}"
        return cls(name=name, **params_dict)

    @property
    def dist_type(self) -> str:
        """The type of copulAX distribution."""
        return "distribution"

    @property
    def dtype(self) -> str:
        """The data type of the distribution."""
        return "continuous"

    @staticmethod
    def _scalar_transform(params: dict) -> dict:
        """Cast every value in *params* to a scalar float JAX array."""
        return {
            key: jnp.asarray(value, dtype=float).reshape(())
            for key, value in params.items()
        }

    @abstractmethod
    def _args_transform(self, params: dict) -> dict:
        r"""Transforms the input arguments to the correct dtype and
        shape.
        """

    def _params_from_array(self, params_arr: jnp.ndarray, *args, **kwargs) -> dict:
        r"""Returns a dictionary from an array / iterable of params"""
        return self._params_dict(*params_arr)

    @abstractmethod
    def _params_to_tuple(self, params: dict) -> tuple:
        r"""Returns a tuple of params from a dictionary.
        Reduces code when extracting params."""
        pass

    @abstractmethod
    def fit(self, x: ArrayLike, *args, **kwargs):
        r"""Fit the distribution to the input data.

        Args:
            x (ArrayLike): The input data to fit the distribution to.
            kwargs: Additional keyword arguments to pass to the fit
                method.

        Returns:
            dict: The fitted distribution parameters.
        """

    def _params_to_array(self, params) -> Array:
        r"""Returns a flattened array of params from a dictionary.
        Reduces code when extracting params."""
        return jnp.asarray(self._params_to_tuple(params)).flatten()

    @abstractmethod
    def rvs(self, size, params: dict, key: Array = None, *args, **kwargs) -> Array:
        r"""Generate random variates from the distribution.

        Args:
            params (dict): Parameters describing the distribution. See
                the specific distribution class or the 'example_params'
                method for details.

        Returns:
            jnp.ndarray: The generated random variates.
        """

    def sample(
        self, size, params: dict = None, key: Array = None, *args, **kwargs
    ) -> Array:
        """Alias for the rvs method."""
        return self.rvs(size=size, params=params, key=key, *args, **kwargs)

    # fitting
    def _stable_logpdf(self, stability: Scalar, x: ArrayLike, params: dict) -> Array:
        r"""Stable log-pdf function for distribution fitting.
        Utilises a stability term to help prevent the logpdf from
        blowing to inf / nan during numerical optimisation, typically
        resulting from log and 1 / x functions.

        Args:
            stability (Scalar): A stability parameter for the distribution.
            x (ArrayLike): The input data to evaluate the stable log-pdf.
            params (dict): Parameters describing the distribution. See
                the specific distribution class or the 'example_params'
                method for details.

        Returns:
            Array: The stable log-pdf values.
        """

        pass

    def logpdf(self, x: ArrayLike, params: dict = None) -> Array:
        r"""The log-probability density function (pdf) of the
        distribution.

        Args:
            x (ArrayLike): The input at which to evaluate the log-pdf.
            params (dict): Parameters describing the distribution. See
                the specific distribution class or the 'example_params'
                method for details. If None, uses stored parameters.

        Returns:
            Array: The log-pdf values.
        """
        params = self._resolve_params(params)
        return self._stable_logpdf(stability=0.0, x=x, params=params)

    def pdf(self, x: ArrayLike, params: dict = None) -> Array:
        r"""The probability density function (pdf) of the distribution.

        Args:
            x (ArrayLike): The input at which to evaluate the pdf.
            params (dict): Parameters describing the distribution. See
                    the specific distribution class or the 'example_params'
                    method for details. If None, uses stored parameters.

        Returns:
            Array: The pdf values.
        """
        params = self._resolve_params(params)
        return jnp.exp(self.logpdf(x=x, params=params))

    # stats
    def stats(self, params: dict = None) -> dict:
        r"""Distribution statistics for the distribution.

        Args:
            params (dict): Parameters describing the distribution. See
                the specific distribution class or the 'example_params'
                method for details. If None, uses stored parameters.

        Returns:
            stats (dict): A dictionary containing the distribution
            statistics.
        """
        return {}

    # metrics
    def loglikelihood(self, x: ArrayLike, params: dict = None) -> Scalar:
        r"""Log-likelihood of the distribution given the data.

        Args:
            x (ArrayLike): The input data to evaluate the
            log-likelihood.
            params (dict): Parameters describing the distribution. See
                the specific distribution class or the 'example_params'
                method for details. If None, uses stored parameters.

        Returns:
            loglikelihood (Scalar): The log-likelihood value.
        """
        params = self._resolve_params(params)
        return self.logpdf(x=x, params=params).sum()

    def aic(self, k: int, x: ArrayLike, params: dict) -> Scalar:
        r"""Akaike Information Criterion (AIC) of the distribution
        given the data. Can be used as a crude metric for model
        selection, by minimising.

        Args:
            x (ArrayLike): The input data to evaluate the AIC.
            params (dict): Parameters describing the distribution. See
                the specific distribution class or the 'example_params'
                method for details.

        Returns:
            aic (Scalar): The AIC value.
        """
        return 2 * k - 2 * self.loglikelihood(x=x, params=params)

    def bic(self, k: int, n: int, x: ArrayLike, params: dict) -> Scalar:
        r"""Bayesian Information Criterion (BIC) of the distribution
        given the data. Can be used as a crude metric for model
        selection, by minimising.

        Args:
            x (ArrayLike): The input data to evaluate the BIC.
            params (dict): Parameters describing the distribution. See
                the specific distribution class or the 'example_params'
                method for details.

        Returns:
            bic (Scalar): The BIC value.
        """
        return k * jnp.log(n) - 2 * self.loglikelihood(x=x, params=params)

    @abstractmethod
    def example_params(self, *args, **kwargs) -> dict:
        r"""Returns example parameters for the distribution.

        Returns:
            dict: A dictionary containing example distribution
            parameters.
        """
        pass


###############################################################################
# Univariate Base Class
###############################################################################
# Fallback value; the canonical dynamic value is _MAX_PARAMS in univariate_fitter.py
MAX_UNIVARIATE_PARAMS: int = 6


class Univariate(Distribution):
    r"""Base class for univariate distributions."""

    _FIT_INVALID_PENALTY: ClassVar[float] = 1e6
    _FIT_SUPPORT_PENALTY: ClassVar[float] = 1e6

    @property
    def dist_type(self) -> str:
        """The type of copulAX distribution."""
        return "univariate"

    @staticmethod
    def _args_transform(params: dict) -> dict:
        """Validate and cast distribution parameters."""
        return Distribution._scalar_transform(params)

    @abstractmethod
    def _support(self, *args, **kwargs) -> Array:
        """Return the support bounds as a JAX array of shape ``(2,)``."""
        pass

    def support(self, params=None, *args, **kwargs) -> Array:
        r"""The support of the distribution is the subset of x for which
        the pdf is non-zero.

        Args:
            params (dict): Parameters describing the distribution.
                If None, uses stored parameters.

        Returns:
            Array: Flattened array of shape ``(2,)`` containing the
            lower and upper bounds of the support.
        """
        params = self._resolve_params(params)
        return jnp.asarray(self._support(params, *args, **kwargs)).flatten()

    def _support_bounds(self, params: dict) -> tuple[Scalar, Scalar]:
        """Return sanitized support bounds (lower, upper) for robust masking."""
        bounds = jnp.asarray(self._support(params)).flatten()
        lower = jnp.where(jnp.isnan(bounds[0]), -jnp.inf, bounds[0])
        upper = jnp.where(jnp.isnan(bounds[1]), jnp.inf, bounds[1])
        return lower.reshape(()), upper.reshape(())

    def _support_violation(self, x: ArrayLike, params: dict) -> Array:
        """Return per-point support violation distance (0 means in-support)."""
        x_arr, _ = _univariate_input(x)
        lower, upper = self._support_bounds(params)
        below = jnp.maximum(lower - x_arr, 0.0)
        above = jnp.maximum(x_arr - upper, 0.0)
        return below + above

    def _enforce_support_on_logpdf(
        self, x: ArrayLike, logpdf: ArrayLike, params: dict
    ) -> Array:
        """Map values outside support to ``-inf`` in log-density outputs."""
        out = jnp.asarray(logpdf, dtype=float)
        x_arr = jnp.asarray(x, dtype=float).reshape(out.shape)
        lower, upper = self._support_bounds(params)
        outside = jnp.logical_or(x_arr < lower, x_arr > upper)
        return jnp.where(outside, -jnp.inf, out)

    def _enforce_support_on_cdf(
        self, x: ArrayLike, cdf: ArrayLike, params: dict
    ) -> Array:
        """Map values outside support and saturating infinities to 0/1."""
        out = jnp.asarray(cdf, dtype=float)
        x_arr = jnp.asarray(x, dtype=float).reshape(out.shape)
        lower, upper = self._support_bounds(params)
        out = jnp.where(x_arr < lower, 0.0, out)
        out = jnp.where(x_arr > upper, 1.0, out)
        # Saturating infinities: F(+inf) = 1, F(-inf) = 0 regardless of
        # support bounds. Makes the contract explicit and catches any
        # NaN leakage from upstream.
        out = jnp.where(jnp.isinf(x_arr) & (x_arr > 0), 1.0, out)
        out = jnp.where(jnp.isinf(x_arr) & (x_arr < 0), 0.0, out)
        return out

    def logpdf(self, x: ArrayLike, params: dict = None) -> Array:
        r"""The log-probability density function (pdf) of the
        distribution.

        Values outside support are mapped to ``-inf``.
        """
        params = self._resolve_params(params)
        raw = self._stable_logpdf(stability=0.0, x=x, params=params)
        return self._enforce_support_on_logpdf(x=x, logpdf=raw, params=params)

    def logcdf(self, x: ArrayLike, params: dict = None) -> Array:
        r"""The log-cumulative distribution function of the
        distribution.

        Args:
            x (ArrayLike): The input at which to evaluate the log-cdf.
            params (dict): Parameters describing the distribution. See
                the specific distribution class or the 'example_params'
                method for details. If None, uses stored parameters.

        Returns:
            Array: The log-cdf values.
        """
        params = self._resolve_params(params)
        return jnp.log(self.cdf(x=x, params=params))

    # cdf
    @classmethod
    def _params_from_array(cls, params_arr, *args, **kwargs):
        """Reconstruct a params dict from a flat parameter array."""
        return cls._params_dict(*params_arr)

    @classmethod
    def _pdf_for_cdf(cls, x: ArrayLike, *params_tuple) -> Array:
        """PDF wrapper used by the numerical CDF integrator."""
        params_array: jnp.ndarray = jnp.asarray(params_tuple).flatten()
        params: dict = cls._params_from_array(params_array)
        return cls.pdf(x=x, params=params)

    # Offset grid (in units of the distribution's standard deviation)
    # used by the default ``_cdf_breakpoints``. Multi-scale coverage
    # from the deep left tail to the deep right tail ensures that the
    # sorted-grid segments in the piecewise CDF path are always
    # narrow enough for fixed-order Gauss-Legendre quadrature to
    # resolve the PDF decay. Light-tailed families (GH, NIG) are fully
    # covered by +/- 20 sigma; heavier-tailed families (Skewed-T) are
    # covered out to +/- 100 sigma. Values outside this range always
    # yield near-zero segment contributions because the PDF has
    # decayed below machine epsilon.
    _CDF_BREAKPOINT_OFFSETS: ClassVar[tuple] = (
        -100.0, -50.0, -20.0, -10.0, -5.0, -2.0, -1.0,
        0.0,
        1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
    )

    def _cdf_anchors(self, params: dict) -> Array:
        r"""Anchor points for the CDF breakpoint grid.

        Returns a 1D array of shape ``(M,)`` giving the locations
        around which breakpoints are placed. For unimodal distributions
        M=1 and this is a single anchor wrapped in a length-1 array.
        For multi-modal distributions, M equals the number of modes.

        The default returns ``[stats["mean"]]``. Subclasses should
        override when (a) the mean does not exist or is numerically
        unreliable, (b) the median or mode is a tighter bulk anchor
        (heavy-skew case, closed-form mode), or (c) the distribution
        is multi-modal.

        Args:
            params (dict): Parameters describing the distribution.

        Returns:
            Array: shape ``(M,)`` array of anchor points.
        """
        return jnp.asarray(self.stats(params)["mean"]).reshape((1,))

    def _cdf_anchor_scales(self, params: dict) -> Array:
        r"""Per-anchor scale for the CDF breakpoint grid.

        Returns a 1D array of shape ``(M,)`` (same length as
        ``_cdf_anchors``) giving the scale at each anchor. The
        breakpoint grid for anchor ``m`` is
        ``anchors[m] + _CDF_BREAKPOINT_OFFSETS * scales[m]``.

        The default returns ``[sqrt(stats["variance"])]``. Subclasses
        should override when (a) variance does not exist (Cauchy,
        Student-t with nu <= 2), (b) the distribution's intrinsic
        scale parameter (e.g. sigma for GH / skewed-T, delta for NIG)
        is more reliable than the moment-based formula, or (c) the
        distribution is multi-modal with per-mode scales.

        Args:
            params (dict): Parameters describing the distribution.

        Returns:
            Array: shape ``(M,)`` array of strictly positive scales.
        """
        variance = jnp.asarray(self.stats(params)["variance"])
        return jnp.sqrt(jnp.maximum(variance, 1e-30)).reshape((1,))

    def _cdf_breakpoints(self, params: dict) -> Array:
        r"""Breakpoints used by the numerical CDF integrator.

        Returns a 1D array of shape ``(M*K,)`` containing ascending
        scalars that span the distribution's bulk at multiple scales.
        The grid is built from ``_cdf_anchors`` (per-anchor location,
        shape ``(M,)``) and ``_cdf_anchor_scales`` (per-anchor scale,
        shape ``(M,)``) via a Kronecker-like product with
        ``_CDF_BREAKPOINT_OFFSETS`` (shape ``(K,)``).

        Multi-modal subclasses get correct coverage by overriding
        ``_cdf_anchors`` / ``_cdf_anchor_scales`` to return length-M
        arrays; the total breakpoint count becomes ``M*K`` with a
        cluster of K breakpoints centred on each mode.

        Breakpoints are clamped into the open support interior.
        Duplicates from clamping are harmless: they produce zero-length
        segments in the downstream prefix sum that contribute zero, and
        quadax's interval transforms handle boundary-adjacent values
        cleanly.

        Args:
            params (dict): Parameters describing the distribution.

        Returns:
            Array: 1D ascending array of breakpoints, shape ``(M*K,)``.
        """
        anchors = jnp.asarray(self._cdf_anchors(params)).flatten()       # (M,)
        scales = jnp.asarray(self._cdf_anchor_scales(params)).flatten()  # (M,)
        offsets = jnp.asarray(
            self._CDF_BREAKPOINT_OFFSETS, dtype=anchors.dtype
        )  # (K,)
        bps = anchors[:, None] + offsets[None, :] * scales[:, None]  # (M, K)
        bps = bps.flatten()  # (M*K,)

        lower, upper = self._support_bounds(params)
        span = jnp.where(
            jnp.isfinite(upper) & jnp.isfinite(lower), upper - lower, 1.0
        )
        margin = 1e-6 * span
        low_clip = jnp.where(jnp.isfinite(lower), lower + margin, -jnp.inf)
        high_clip = jnp.where(jnp.isfinite(upper), upper - margin, jnp.inf)
        return jnp.sort(jnp.clip(bps, low_clip, high_clip))

    @abstractmethod
    def cdf(self, x: ArrayLike, params: dict) -> Array:
        r"""Cumulative distribution function of the distribution.

        Args:
            x (ArrayLike): The input at which to evaluate the cdf.
            params (dict): Parameters describing the distribution. See
                the specific distribution class or the 'example_params'
                method for details.

        Returns:
            Array: The cdf values.
        """

    # ppf
    def _ppf(
        self, q: ArrayLike, params: dict, cubic: bool, num_points: int, maxiter: int
    ) -> Array:
        """Internal dispatch for the percent-point function."""
        return _ppf(
            dist=self,
            q=q,
            params=params,
            cubic=cubic,
            num_points=num_points,
            maxiter=maxiter,
        )

    def ppf(
        self,
        q: ArrayLike,
        params: dict = None,
        cubic: bool = False,
        num_points: int = 100,
        maxiter: int = 20,
    ) -> Array:
        r"""Percent point function (inverse of the CDF) of the
        distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'cubic'
            is a static argument.


        Args:
            q (ArrayLike): The quantile values. at which to evaluate the
            ppf.
            params (dict): Parameters describing the distribution. See
                the specific distribution class or the 'example_params'
                method for details.
            cubic (bool): Whether to use a cubic spline approximation
                of the ppf function for faster computation. This can
                also improve gradient estimates.
            num_points (int): The number of points to use for the cubic
                spline approximation when approx is True.
            maxiter (int): The maximum number of iterations to use when
                solving for the ppf function via brents method.

        Returns:
            Array: The inverse CDF values.
        """
        params = self._resolve_params(params)
        q, qshape = _univariate_input(q)
        if cubic:
            # approximating even if an analytical / more efficient solution exists
            x: jnp.ndarray = _ppf(
                dist=self,
                q=q,
                params=params,
                cubic=True,
                num_points=num_points,
                maxiter=maxiter,
            )
        else:
            x: jnp.ndarray = self._ppf(
                q=q, params=params, cubic=False, num_points=num_points, maxiter=maxiter
            )
        return x.reshape(qshape)

    def inverse_cdf(
        self,
        q: ArrayLike,
        params: dict = None,
        cubic: bool = False,
        num_points: int = 100,
        maxiter: int = 20,
    ) -> Array:
        r"""Percent point function (inverse of the CDF) of the
        distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'cubic'
            is a static argument.

        Args:
            q (ArrayLike): The quantile values. at which to evaluate the
            ppf.
            params (dict): Parameters describing the distribution. See
                the specific distribution class or the 'example_params'
                method for details.
            cubic (bool): Whether to use a cubic spline approximation
                of the ppf function for faster computation. This can
                also improve gradient estimates.
            num_points (int): The number of points to use for the cubic
                spline approximation when approx is True.
            lr (float): The learning rate to use when numerically
                solving for the ppf function via ADAM based gradient
                descent.
            maxiter (int): The maximum number of iterations to use when
                solving for the ppf function via ADAM based gradient
                descent.

        Returns:
            Array: The inverse CDF values.
        """
        return self.ppf(
            q=q, params=params, cubic=cubic, num_points=num_points, maxiter=maxiter
        )

    # sampling
    def rvs(
        self, size: Scalar | tuple, params: dict = None, key: Array = None
    ) -> Array:
        r"""Generates random samples from the distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'size'
            is a static argument.

        Args:
            size (tuple | Scalar): The size / shape of the generated
                output array of random numbers. If a scalar is provided,
                the output array will have shape (size,), otherwise it will
                match the shape specified by this tuple.
            params (dict): Parameters describing the distribution. See
                the specific distribution class or the 'example_params'
                method for details.
            key (Array): The Key for random number generation.
        """
        params = self._resolve_params(params)
        key = _resolve_key(key)
        return inverse_transform_sampling(
            ppf_func=self.ppf, shape=size, params=params, key=key
        )

    # fitting
    def _mle_objective(
        self, params_arr: jnp.ndarray, x: jnp.ndarray, *args, **kwargs
    ) -> Scalar:
        r"""Penalized negative log-likelihood objective used for fitting.

        The objective is robust to parameter proposals that imply invalid
        support or non-finite log-density values:
        - evaluates ``_stable_logpdf`` on a clipped in-support ``x_safe``
          to avoid undefined operations during AD;
        - applies a large penalty for out-of-support / non-finite points;
        - applies an additional penalty when support bounds are invalid.

        Args:
            x (ArrayLike): The input data to evaluate the
                negative log-likelihood.

        Returns:
            mle_objective (float): The negative log-likelihood value.
        """
        params: dict = self._params_from_array(params_arr, *args, **kwargs)
        x_arr, _ = _univariate_input(x)
        lower, upper = self._support_bounds(params)
        eps = 1e-6
        safe_lower = jnp.where(jnp.isfinite(lower), lower + eps, lower)
        safe_upper = jnp.where(jnp.isfinite(upper), upper - eps, upper)
        invalid_interval = safe_lower > safe_upper
        x_safe = jnp.where(
            invalid_interval,
            x_arr,
            jnp.clip(x_arr, min=safe_lower, max=safe_upper),
        )

        logpdf_raw: Array = self._stable_logpdf(stability=1e-30, x=x_safe, params=params)
        finite_mask = jnp.isfinite(logpdf_raw)
        in_support = jnp.logical_and(x_arr >= lower, x_arr <= upper)
        valid_mask = jnp.logical_and(finite_mask, in_support)
        safe_logpdf = jnp.where(valid_mask, logpdf_raw, 0.0)
        invalid_penalty = self._FIT_INVALID_PENALTY * (~valid_mask).mean()

        support_violation = self._support_violation(x=x, params=params)
        support_penalty = self._FIT_SUPPORT_PENALTY * (support_violation**2).mean()

        invalid_bounds = jnp.logical_or(jnp.isnan(lower), jnp.isnan(upper))
        invalid_bounds = jnp.logical_or(invalid_bounds, lower > upper)
        invalid_bounds = jnp.logical_or(invalid_bounds, invalid_interval)
        bounds_penalty = self._FIT_SUPPORT_PENALTY * jnp.where(
            invalid_bounds, 1.0, 0.0
        )

        return -safe_logpdf.mean() + invalid_penalty + support_penalty + bounds_penalty

    # metrics
    # goodness-of-fit tests
    def ks_test(self, x: ArrayLike, params: dict = None) -> dict:
        r"""One-sample Kolmogorov-Smirnov goodness-of-fit test.

        Tests whether *x* was drawn from this distribution with the
        given *params*.

        Args:
            x (ArrayLike): Observed sample.
            params (dict): Distribution parameters.

        Returns:
            dict: ``{'statistic': D_n, 'p_value': p}``
        """
        from copulax._src.univariate._gof import ks_test

        params = self._resolve_params(params)
        return ks_test(x=x, dist=self, params=params)

    def cvm_test(self, x: ArrayLike, params: dict = None) -> dict:
        r"""One-sample Cramér-von Mises goodness-of-fit test.

        Tests whether *x* was drawn from this distribution with the
        given *params*.

        Args:
            x (ArrayLike): Observed sample.
            params (dict): Distribution parameters.

        Returns:
            dict: ``{'statistic': W2, 'p_value': p}``
        """
        from copulax._src.univariate._gof import cvm_test

        params = self._resolve_params(params)
        return cvm_test(x=x, dist=self, params=params)

    @property
    def n_params(self) -> int:
        """Number of distribution parameters."""
        return len(self.example_params())

    def _padded_params_to_array(
        self, params: dict, max_params: int = None
    ) -> jnp.ndarray:
        """Convert params dict to a fixed-length padded array.

        Returns a 1-D array of length *max_params*, with the real
        parameter values followed by NaN padding.  When *max_params*
        is ``None``, uses ``MAX_UNIVARIATE_PARAMS``.
        """
        if max_params is None:
            max_params = MAX_UNIVARIATE_PARAMS
        arr = self._params_to_array(params)
        pad_width = max_params - arr.shape[0]
        return jnp.concatenate([arr, jnp.full(pad_width, jnp.nan)])

    def aic(self, x: ArrayLike, params: dict = None) -> float:
        """Akaike Information Criterion for the fitted distribution."""
        params = self._resolve_params(params)
        k: int = len(params)
        return super().aic(k=k, x=x, params=params)

    def bic(self, x: ArrayLike, params: dict = None) -> float:
        """Bayesian Information Criterion for the fitted distribution."""
        params = self._resolve_params(params)
        k: int = len(params)
        n: int = x.size
        return super().bic(k=k, n=n, x=x, params=params)

    def plot(
        self,
        params: dict = None,
        sample: jnp.ndarray = None,
        domain: tuple = None,
        bins: int = 50,
        num_points: int = 100,
        figsize: tuple = (16, 8),
        grid: bool = True,
        show: bool = True,
        ppf_options: dict = None,
    ):
        r"""Plots the pdf, cdf and ppf of the distribution.

             Note:
                 Not jittable.

             Args:
                 params (dict): Parameters describing the distribution. See
                     the specific distribution class or the 'example_params'
                     method for details.
                 sample (jnp.ndarray): Sample data to plot alongside the
                     distribution, allowing for a visual goodness of fit via
                     QQ plots. Must be a univariate sample if provided.
                 domain (tuple): The domain of the distribution to plot over.
                     Must be a tuple of the form (min, max). If None, the ppf
                     function will be used to generate the domain.
                 bins (int): The number of bins to use for the histogram of
                     the sample data, if provided.
                 num_points (int): The number of points to use for plotting.
                 figsize (tuple): The size of the figure for the plot.
                 grid (bool): Whether to display a grid on the plot.
                 show (bool): Whether to automatically show the plot by
                     internally calling plt.show().
                 ppf_options (dict): Options for the ppf function. This is
                     kwargs style dictionary with keyword arguments for the
                     keys paired with their assigned values. See the ppf
                     method function documentation for more details.
        Returns:
                 None
        """
        params = self._resolve_params(params)
        params: dict = self._args_transform(params=params)

        # getting pdf and cdf domain
        if ppf_options is None:
            ppf_options = {}
        jitted_ppf = jit(self.ppf, static_argnames=("cubic", "maxiter"))
        delta: float = 1e-2
        if domain is None:
            support = self.support(params=params)

            # lower bound
            min_val, eps = support[0], 0.0
            while not jnp.isfinite(min_val):
                eps += delta
                min_val = jitted_ppf(q=jnp.array(eps), params=params, **ppf_options)

            # upper bound
            max_val, eps = support[1], 0.0
            while not jnp.isfinite(max_val):
                eps += delta
                max_val = jitted_ppf(q=jnp.array(1 - eps), params=params, **ppf_options)
        else:
            if (not isinstance(domain, Iterable)) or len(domain) != 2:
                raise ValueError("Domain must be a tuple of the form (min, max).")
            if not all(isinstance(i, Scalar) for i in domain):
                raise ValueError("Domain elements must be scalar values")
            if domain[0] >= domain[1]:
                raise ValueError("Domain must be a tuple of the form (min, max).")

            min_val, max_val = jnp.asarray(domain).flatten()

        x: Array = jnp.linspace(min_val, max_val, num_points)
        q: Array = jnp.linspace(delta, 1 - delta, num_points)

        # pdf, cdf, ppf and rvs values
        pdf_vals: Array = self.pdf(x=x, params=params)
        cdf_vals: Array = self.cdf(x=x, params=params)
        ppf_vals: Array = jitted_ppf(q=q, params=params, **ppf_options)

        # plotting setup
        values: tuple = [pdf_vals, cdf_vals, ppf_vals]
        domains: tuple = [x, x, q]
        titles: tuple = ("PDF", "CDF", "Inverse CDF", "QQ-Plot")
        xlabels: tuple = ("x", "x", "P(X <= q)", "Theoretical Quantiles")
        ylabels: tuple = ("PDF", "P(X <= x)", "q", "Empirical Quantiles")
        dtype = float if self.dtype == "continuous" else int
        printable_params: str = str({k: round(dtype(v), 2) for k, v in params.items()})
        name_with_params: str = f"{self.name}({printable_params})"

        # qq-plot
        if sample is not None:
            sample: Array = _univariate_input(sample)[0]
            sorted_sample: Array = sample.flatten().sort()
            N: int = sample.size
            empirical_sample_cdf = jnp.array(
                [(sorted_sample <= xi).sum() / N for xi in sorted_sample]
            )
            theoretical_sample_cdf = self.cdf(x=sorted_sample, params=params)
            domains.append(theoretical_sample_cdf)
            values.append(empirical_sample_cdf)

        # plotting
        num_subplots = len(values)
        fig, ax = plt.subplots(1, num_subplots, figsize=figsize)
        fig.suptitle(name_with_params, fontsize=16)
        for i in range(num_subplots):
            if i == 0 and num_subplots == 4:
                ax[i].hist(
                    sample,
                    bins=bins,
                    density=True,
                    color="blue",
                    label="Sample",
                    zorder=0,
                )
                ax[i].set_xlim(min_val, max_val)
            elif i == 3:
                ax[i].plot(cdf_vals, cdf_vals, color="blue", zorder=0, label="y=x")

            # plotting distribution
            if i < 3:
                ax[i].plot(
                    domains[i],
                    values[i],
                    label=name_with_params,
                    color="black",
                    zorder=1,
                )
            else:
                ax[i].scatter(
                    domains[i],
                    values[i],
                    label=name_with_params,
                    color="black",
                    zorder=1,
                )

            # labeling
            ax[i].set_title(titles[i])
            ax[i].set_xlabel(xlabels[i])
            ax[i].set_ylabel(ylabels[i])
            ax[i].grid(grid)
            ax[i].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                fancybox=True,
                shadow=True,
                ncol=1,
            )

        plt.tight_layout()
        if show:
            plt.show()


###############################################################################
# Multivariate Base Class
###############################################################################
class GeneralMultivariate(Distribution):
    r"""Base class for multivariate and copula distributions."""

    @property
    def dist_type(self) -> str:
        """The type of copulAX distribution."""
        return "multivariate"

    def _classify_params(
        self,
        params: dict,
        scalar_names: tuple = tuple(),
        vector_names: tuple = tuple(),
        shape_names: tuple = tuple(),
        symmetric_shape_names: tuple = tuple(),
        corr_like_shape_names: tuple = tuple(),
        non_symmetric_shape_names: tuple = tuple(),
    ) -> dict:
        r"""Classify the distribution parameters into scalars, vectors
        and shapes.

        Args:
            params (dict)
            scalar_names (tuple[str]): Tuple of scalar parameter names.
            vector_names (tuple[str]): Tuple of vector parameter names.
            symmetric_shape_names (tuple[str]): Tuple of symmetric shape
                parameter names. Diagonal elements are included in the
                parameter count.
            corr_like_shape_names (tuple[str]): Tuple of correlation
                matrix like symmetric shape parameter names. Diagonal
                elements are not included in the parameter count.
            non_symmetric_shape_names (tuple[str]): Tuple of
                non-symmetric shape parameter names.
        """

        classifications = {
            "scalars": {name: params[name] for name in scalar_names},
            "vectors": {name: params[name] for name in vector_names},
            "shapes": {name: params[name] for name in shape_names},
            "symmetric_shapes": {name: params[name] for name in symmetric_shape_names},
            "corr_like_shapes": {name: params[name] for name in corr_like_shape_names},
            "non_symmetric_shapes": {
                name: params[name] for name in non_symmetric_shape_names
            },
        }
        return classifications

    @abstractmethod
    def _get_dim(self, params: dict) -> int:
        r"""Returns the number of dimensions of the distribution."""

    def _args_transform(self, params: dict) -> dict:
        """Validate and reshape distribution parameters by category."""
        classifications: dict = self._classify_params(
            params=params
        )  # todo: issue is here where scalars are being included. need to think about how to do this
        d: int = self._get_dim(params=params)

        # scalars
        transformed_scalars: dict = Distribution._scalar_transform(
            classifications["scalars"]
        )

        # vectors
        transformed_vectors: dict = {
            k: jnp.asarray(v, dtype=float).reshape((d, 1))
            for k, v in classifications["vectors"].items()
        }

        # shapes
        transformed_shapes: dict = {
            k: jnp.asarray(v, dtype=float).reshape((d, d))
            for k, v in classifications["shapes"].items()
        }

        return {**transformed_scalars, **transformed_vectors, **transformed_shapes}

    def _get_num_params(self, params: dict) -> int:
        r"""Returns the number of parameters of the distribution.

        Returns:
            int: The number of parameters.
        """
        classifications: dict = self._classify_params(params=params)
        dim: int = self._get_dim(params=params)

        # scalars
        scalars: tuple = classifications["scalars"]
        n_scalars: int = len(scalars)

        # vectors
        vectors: dict = classifications["vectors"]
        n_vectors: int = len(vectors) * dim

        # symmetric shapes
        symmetric_shapes: dict = classifications["symmetric_shapes"]
        n_symm_shapes: int = len(symmetric_shapes) * (dim * (dim + 1) // 2)

        # correlation-like shapes
        corr_like_shapes: dict = classifications["corr_like_shapes"]
        n_corr_like_shapes: int = len(corr_like_shapes) * (dim * (dim - 1) // 2)

        # non-symmetric shapes
        non_symmetric_shapes: dict = classifications["non_symmetric_shapes"]
        n_non_symm_shapes: int = len(non_symmetric_shapes) * (dim**2)

        # total
        return (
            n_scalars
            + n_vectors
            + n_symm_shapes
            + n_corr_like_shapes
            + n_non_symm_shapes
        )

    @abstractmethod
    def support(self, params: dict) -> Array:
        r"""The support of the distribution is the subset of
        multivariate x for which the pdf is non-zero.

        Args:
            params (dict): Parameters describing the distribution. See
                the specific distribution class or the 'example_params'
                method for details.

        Returns:
            Array: Array containing the support of each variable in
            the distribution, in an (d, 2) shape.
        """

    # sampling
    @abstractmethod
    def rvs(self, size: int, params: dict, key: Array = None) -> Array:
        r"""Generates random samples from the distribution.

        Note:
            If you intend to jit wrap this function, ensure that 'size'
            is a static argument.

        Args:
            size (int): The size of the generated
                output array of random numbers. Must be an integer.
                Generates an (size, d) array of random numbers, where
                d is the number of dimensions inferred from the provided
                distribution parameters.
            params (dict): Parameters describing the distribution. See
                    the specific distribution class or the 'example_params'
                    method for details.
            key (Array): The Key for random number generation.
        """

    # metrics
    def aic(self, x: ArrayLike, params: dict = None) -> float:
        """Akaike Information Criterion for the fitted distribution."""
        params = self._resolve_params(params)
        k: int = self._get_num_params(params=params)
        return super().aic(k=k, x=x, params=params)

    def bic(self, x: ArrayLike, params: dict = None) -> float:
        """Bayesian Information Criterion for the fitted distribution."""
        params = self._resolve_params(params)
        x, _, n, _ = _multivariate_input(x)
        k: int = self._get_num_params(params=params)
        return super().bic(k=k, n=n, x=x, params=params)

    # fitting
    @abstractmethod
    def fit(self, x: ArrayLike, *args, **kwargs) -> dict:
        r"""Fit the distribution to the input data.

        Note:
            Data must be in the shape (n, d) where n is the number of
            samples and d is the number of dimensions.

        Args:
            x (ArrayLike): The input data to fit the distribution too.
                Must be in the shape (n, d) where n is the number of
                samples and d is the number of dimensions.
            kwargs: Additional keyword arguments to pass to the fit
                method.

        Returns:
            dict: The fitted distribution parameters.
        """


class Multivariate(GeneralMultivariate):
    r"""Base class for multivariate distributions."""

    def _get_dim(self, params: dict) -> int:
        """Infer the number of dimensions from the parameter vectors."""
        classifications: dict = self._classify_params(params)
        return jnp.asarray(list(classifications["vectors"].values())[0]).size

    def support(
        self,
        params: dict = None,
        marginal_support: tuple = (-jnp.inf, jnp.inf),
        *args,
        **kwargs,
    ) -> Array:
        """Return the support as a ``(d, 2)`` array of per-dimension bounds."""
        params = self._resolve_params(params)
        d: int = self._get_dim(params=params)
        return jnp.concat(
            [
                jnp.full((d, 1), marginal_support[0]),
                jnp.full((d, 1), marginal_support[1]),
            ],
            axis=1,
        )

    def _calc_Q(
        self, x: jnp.ndarray, mu: jnp.ndarray, sigma_inv: jnp.ndarray
    ) -> jnp.ndarray:
        r"""Calculates the Mahalanobis distance vector.

        .. math::
            Q_i = (x_i - \mu)^T \Sigma^{-1} (x_i - \mu)

        Args:
            x: Input data of shape ``(n, d)``.
            mu: Mean vector of shape ``(d, 1)`` or ``(d,)``.
            sigma_inv: Inverse covariance matrix of shape ``(d, d)``.

        Returns:
            Array of shape ``(n,)`` containing the quadratic forms.
        """
        diff: jnp.ndarray = x - mu.flatten()  # (n, d)
        return jnp.sum(diff @ sigma_inv * diff, axis=1)



class NormalMixture(Multivariate):
    r"""Base class for normal mixture distributions."""

    # sampling
    def _rvs(
        self, key, n: int, W: Array, mu: Array, gamma: Array, sigma: Array
    ) -> Array:
        r"""Generates random samples from the normal-mixture
        distribution."""
        d: int = mu.size

        m: jnp.ndarray = mu + W * gamma

        Z: jnp.ndarray = random.normal(key, shape=(d, n))
        A: jnp.ndarray = jnp.linalg.cholesky(sigma)
        r: jnp.ndarray = jnp.sqrt(W) * (A @ Z)
        return (m + r).T

    # stats
    def _stats(self, w_stats: dict, mu: Array, gamma: Array, sigma: Array) -> dict:
        """Compute distribution mean and covariance from mixing variable statistics."""
        mean: Array = mu + w_stats["mean"] * gamma
        cov: Array = w_stats["mean"] * sigma + w_stats["variance"] * jnp.outer(
            gamma, gamma
        )
        return {
            "mean": mean,
            "cov": cov,
            "skewness": gamma,
        }

    # fitting
    @abstractmethod
    def _ldmle_inputs(self, d: int, x: jnp.ndarray = None) -> tuple:
        """Returns the input arguments for the low dimensional MLE.
        Specifically, the projection options containing the constraints
        and the initial guess."""
        pass

    def _general_fit(
        self,
        x: ArrayLike,
        d: int,
        loc: jnp.ndarray,
        shape: jnp.ndarray,
        lr: float,
        maxiter: int,
    ) -> dict:
        """Run low-dimensional MLE via projected ADAM gradient descent."""
        # optimisation constraints and initial guess
        projection_options, params0 = self._ldmle_inputs(d, x=x)

        # ADAM gradient descent
        res: dict = projected_gradient(
            f=self._ldmle_objective,
            x0=params0,
            projection_method="projection_box",
            projection_options=projection_options,
            x=x,
            loc=loc,
            shape=shape,
            lr=lr,
            maxiter=maxiter,
        )

        # reconstructing the parameters
        optimised_params_arr: jnp.ndarray = res["x"]
        optimised_params: tuple = self._reconstruct_ldmle_func(
            params_arr=optimised_params_arr,
            loc=loc,
            shape=shape,
        )
        return optimised_params

    _LDMLE_INVALID_PENALTY: ClassVar[float] = 1e6

    def _ldmle_objective(
        self,
        params_arr: jnp.ndarray,
        x: jnp.ndarray,
        loc: jnp.ndarray,
        shape: jnp.ndarray,
    ) -> Scalar:
        """Negative log-likelihood objective for low-dimensional MLE.

        Non-finite log-density values (NaN / ±inf) are replaced with a
        large penalty so the optimiser receives a finite gradient signal
        pointing away from degenerate parameter regions.
        """
        params: dict = self._reconstruct_ldmle_func(
            params_arr=params_arr, loc=loc, shape=shape
        )
        logpdf: Array = self._stable_logpdf(stability=1e-30, x=x, params=params)
        finite_mask = jnp.isfinite(logpdf)
        safe_logpdf = jnp.where(finite_mask, logpdf, 0.0)
        return -safe_logpdf.mean() + self._LDMLE_INVALID_PENALTY * (~finite_mask).mean()

    def fit(
        self,
        x: ArrayLike,
        cov_method: str = "pearson",
        lr: float = 0.1,
        maxiter: int = 100,
        name: str = None,
    ) -> dict:
        r"""Fits the multivariate distribution to the data.

        Note:
            If you intend to jit wrap this function, ensure that
            'cov_method' is a static argument.

        Algorithm:
            1. Estimate the sample mean vector and sample covariance
                matrix, potentially using robust estimators.
            2. Remove the location and shape matrices from the
                optimisation process, as these can be inferred from
                scalar parameters and skewness.
            3. We maximise the log-likelihood using ADAM.

        Args:
            x: arraylike, data to fit the distribution to.
            cov_method: str, method to estimate the sample covariance
                matrix, sigma. See copulax.multivariate.cov and/or
                copulax.multivariate.corr for available methods.
            lr (float): Learning rate for optimization.
            maxiter (int): Maximum number of iterations for optimization.
            name (str): Optional custom name for the fitted instance.

        Returns:
            dict containing the fitted parameters.
        """
        # estimating the sample mean and covariance
        x, _, _, d = _multivariate_input(x)
        sample_mean: jnp.ndarray = jnp.mean(x, axis=0).reshape((d, 1))
        sample_cov: jnp.ndarray = cov(x=x, method=cov_method)

        # optimising
        params = self._general_fit(
            x=x,
            d=d,
            loc=sample_mean,
            shape=sample_cov,
            lr=lr,
            maxiter=maxiter,
        )
        return self._fitted_instance(params, name=name)

    @abstractmethod
    def _reconstruct_ldmle_params(
        self, params_arr: jnp.ndarray, loc: jnp.ndarray, shape: jnp.ndarray
    ) -> dict:
        """Reconstructs the low dim MLE parameters from a flat array."""
        pass

    def _reconstruct_ldmle_func(
        self,
        params_arr: jnp.ndarray,
        loc: jnp.ndarray,
        shape: jnp.ndarray,
    ) -> tuple:
        """Reconstructs the low dim MLE parameters from a flat array."""
        params_tuple: tuple = self._reconstruct_ldmle_params(
            params_arr, loc, shape
        )
        return self._params_from_array(params_tuple)
