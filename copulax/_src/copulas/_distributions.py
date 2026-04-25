"""CopulAX shared copula base class.

Houses :class:`CopulaBase`, the universal abstract base for every
copula family in copulax (Archimedean, mean-variance, future
extensions).  It carries the Sklar / marginal-fitting / sampling
machinery common to all copulas.

Mean-variance copulas (Gaussian, Student-T, GH, Skewed-T) live in
``_mv_copulas.py``; Archimedean copulas live in ``_archimedean.py``.
"""

from abc import abstractmethod
from jax import Array
from jax.typing import ArrayLike
from jax import numpy as jnp
from jax import vmap

from copulax._src._distributions import GeneralMultivariate, Univariate
from copulax._src.multivariate._utils import _multivariate_input
from copulax._src._utils import _resolve_key
from copulax._src.typing import Scalar
from collections import defaultdict


class CopulaBase(GeneralMultivariate):
    r"""Base class for all copula distributions.

    Provides Sklar's theorem implementations for the joint distribution,
    common marginal-fitting logic, and sampling via inverse-transform
    of copula samples.
    """

    _marginals: tuple = None
    _copula_params: dict = None

    @property
    def _stored_params(self):
        """Return stored parameters dict if marginals and copula are set."""
        if self._marginals is None or self._copula_params is None:
            return None
        return {"marginals": self._marginals, "copula": self._copula_params}

    @property
    def dist_type(self) -> str:
        """Distribution family type identifier."""
        return "copula"

    def _get_dim(self, params: dict) -> int:
        """Infer dimensionality from the number of marginal distributions."""
        return len(params["marginals"])

    def support(self, params: dict = None) -> Array:
        r"""Support of the joint distribution."""
        params = self._resolve_params(params)
        marginals: tuple = params["marginals"]
        return jnp.vstack([dist.support(params=mparams) for dist, mparams in marginals])

    @staticmethod
    def _grouped_marginal_apply(func_name, x_arr, marginals, **func_kwargs):
        """Apply a univariate function across dimensions, vmapping over
        groups that share the same distribution type for efficiency.

        Args:
            func_name: Name of the univariate method to call (e.g. 'cdf').
            x_arr: Input data of shape (n, d).
            marginals: Tuple of (distribution, params) per dimension.
            **func_kwargs: Extra keyword arguments forwarded to each call.

        Returns:
            Array of shape (n, d) with the function evaluated per column.
        """
        d = len(marginals)
        groups = defaultdict(list)
        for i, (dist, mparams) in enumerate(marginals):
            groups[dist.name].append((i, mparams))

        out = None
        for _, items in groups.items():
            dim_indices = [item[0] for item in items]
            idx_arr = jnp.asarray(dim_indices, dtype=int)
            param_dicts = [item[1] for item in items]
            dist = marginals[dim_indices[0]][0]
            func = getattr(dist, func_name)

            batched_params = {
                k: jnp.stack([p[k] for p in param_dicts]) for k in param_dicts[0].keys()
            }
            x_group = x_arr[:, idx_arr]

            def _apply(xi_col, p, _f=func):
                return _f(xi_col, params=p, **func_kwargs)

            result = vmap(_apply, in_axes=(1, 0), out_axes=1)(x_group, batched_params)
            if out is None:
                out = jnp.empty((x_arr.shape[0], d), dtype=result.dtype)
            out = out.at[:, idx_arr].set(result)

        if out is None:
            return jnp.empty((x_arr.shape[0], 0), dtype=x_arr.dtype)
        return out

    def get_u(self, x: ArrayLike, params: dict = None) -> Array:
        r"""Compute marginal CDF values u = (F_1(x_1), ..., F_d(x_d)).

        Args:
            x: Input data of shape (n, d).
            params: Distribution parameters with 'marginals' key.

        Returns:
            Array of shape (n, d) with values in [0, 1].
        """
        x_arr: jnp.ndarray = _multivariate_input(x)[0]
        params = self._resolve_params(params)
        return self._grouped_marginal_apply("cdf", x_arr, params["marginals"])

    # --- copula densities (abstract) ---

    @abstractmethod
    def copula_logpdf(self, u: ArrayLike, params: dict = None, **kwargs) -> Array:
        r"""Log-density of the copula (subclasses must implement)."""

    def copula_pdf(self, u: ArrayLike, params: dict = None, **kwargs) -> Array:
        r"""Density of the copula: c(u) = exp(copula_logpdf(u))."""
        return jnp.exp(self.copula_logpdf(u, params, **kwargs))

    @abstractmethod
    def copula_rvs(self, size: Scalar, params: dict, key: Array = None) -> Array:
        r"""Generate random samples from the copula (subclasses must implement)."""

    def copula_sample(
        self, size: Scalar, params: dict = None, key: Array = None
    ) -> Array:
        r"""Alias for copula_rvs."""
        return self.copula_rvs(size=size, params=params, key=key)

    # --- joint distribution (Sklar's theorem) ---

    def logpdf(self, x: ArrayLike, params: dict = None, **kwargs) -> Array:
        r"""Joint log-PDF via Sklar's theorem.

        log f(x) = log c(F_1(x_1),...,F_d(x_d)) + sum log f_i(x_i)

        Args:
            x: Input data of shape (n, d).
            params: Distribution parameters with 'marginals' and
                'copula' keys.

        Returns:
            Array of shape (n, 1).
        """
        x_arr, _, n, d = _multivariate_input(x)
        params = self._resolve_params(params)
        marginal_logpdf_sum: jnp.ndarray = self._grouped_marginal_apply(
            "logpdf", x_arr, params["marginals"]
        ).sum(axis=1, keepdims=True)
        u: jnp.ndarray = self.get_u(x_arr, params)
        copula_lp: jnp.ndarray = self.copula_logpdf(u, params, **kwargs)
        return copula_lp + marginal_logpdf_sum

    def pdf(self, x: ArrayLike, params: dict = None, **kwargs) -> Array:
        r"""Joint PDF."""
        return jnp.exp(self.logpdf(x, params, **kwargs))

    # --- sampling ---

    def rvs(
        self,
        size: Scalar,
        params: dict = None,
        key: Array = None,
        brent: bool = False,
        nodes: int = 100,
    ) -> Array:
        r"""Sample from the joint distribution.

        1. Sample u from copula
        2. Transform u to x via marginal PPFs

        Args:
            size: Number of samples.
            params: Distribution parameters.
            key: JAX random key.
            brent: Forwarded to the marginal :py:meth:`Univariate.ppf`.
                ``False`` (default) uses the analytical inverse CDF
                when available and otherwise the Chebyshev cubic
                spline; ``True`` forces per-quantile Brent
                root-finding (slower but machine-epsilon accurate).
            nodes: Number of Chebyshev-Lobatto nodes when the cubic
                path is used.  Ignored for analytical marginals and
                when ``brent=True``.

        Returns:
            Array of shape (size, d).
        """
        key = _resolve_key(key)
        params = self._resolve_params(params)
        u_raw: jnp.ndarray = self.copula_rvs(size=size, params=params, key=key)
        eps: float = 1e-4
        u: jnp.ndarray = jnp.clip(u_raw, eps, 1 - eps)
        return self._grouped_marginal_apply(
            "ppf", u, params["marginals"], brent=brent, nodes=nodes
        )

    # --- fitting ---

    def fit_marginals(
        self,
        x: ArrayLike,
        univariate_fitter_options: tuple[dict] | dict = None,
    ) -> dict:
        r"""Fit univariate marginal distributions to each dimension.

        Args:
            x: Input data of shape (n, d).
            univariate_fitter_options: Options for the univariate
                fitter. Dict applies same options to all dimensions;
                tuple of dicts applies per-dimension options.

        Note:
            Not jitable.

        Returns:
            dict with key 'marginals' containing fitted distributions.
        """
        x_arr, _, n, d = _multivariate_input(x)
        if univariate_fitter_options is None:
            univariate_fitter_options = ({},) * d
        elif isinstance(univariate_fitter_options, dict):
            univariate_fitter_options = (univariate_fitter_options,) * d
        elif isinstance(univariate_fitter_options, tuple):
            if len(univariate_fitter_options) != d:
                raise ValueError(
                    "univariate_fitter_options tuple must have "
                    "an entry for each variable in x."
                )
        else:
            raise ValueError("univariate_fitter_options must be a tuple or dictionary.")

        # Local import to break a copulas → univariate cycle at module load.
        from copulax._src.univariate.univariate_fitter import batch_univariate_fitter

        # Group dimensions by options for batched fitting
        groups: dict[str, list[int]] = defaultdict(list)
        for i, opts in enumerate(univariate_fitter_options):
            key = str(sorted(opts.items())) if opts else ""
            groups[key].append(i)

        marginals: list = [None] * d
        for key, dim_indices in groups.items():
            opts = univariate_fitter_options[dim_indices[0]]
            x_batch = x_arr[:, jnp.array(dim_indices)]
            batch_results = batch_univariate_fitter(x_batch, **opts)
            for j, (best_index, fitted) in enumerate(batch_results):
                dist: Univariate = fitted[best_index]["dist"]
                params: dict = fitted[best_index]["params"]
                marginals[dim_indices[j]] = (dist, params)

        return {"marginals": tuple(marginals)}

    @abstractmethod
    def fit_copula(self, u: ArrayLike, **kwargs) -> dict:
        r"""Fit copula parameters (subclasses must implement)."""

    def fit(
        self,
        x: ArrayLike,
        univariate_fitter_options: tuple[dict] | dict = None,
        name: str = None,
        **kwargs,
    ) -> dict:
        r"""Fit marginals and copula to the data.

        Equivalent to calling fit_marginals then fit_copula.

        Args:
            x: Input data of shape (n, d).
            univariate_fitter_options: Options for marginal fitting.
            name: Optional custom name for the fitted instance.
            **kwargs: Additional arguments forwarded to ``fit_copula``
                (``method``, ``lr``, ``maxiter``, ``tol``, ``patience``).

        Note:
            Not jitable.

        Returns:
            dict with keys 'marginals' and 'copula'.
        """
        marginals: dict = self.fit_marginals(x, univariate_fitter_options)
        u: jnp.ndarray = self.get_u(x, marginals)
        copula: dict = self.fit_copula(u, **kwargs)
        params = {**marginals, **copula}
        return self._fitted_instance(params, name=name)

