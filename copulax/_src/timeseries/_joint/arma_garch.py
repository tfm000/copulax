r"""ARMA(p, q) - GARCH-family(p', q') joint composite estimator.

Single composite class fitting the mean and variance equations
under one MLE objective:

.. math::

    y_t       &= c + \sum_{i=1}^p \phi_i\, y_{t-i}
                  + \sum_{j=1}^q \theta_j\, \varepsilon_{t-j}
                  + \varepsilon_t,\\
    \varepsilon_t &= \sigma_t\, z_t,
                  \quad z_t \sim f_z\,(\text{mean}=0, \mathrm{var}=1),\\
    \sigma^2_t   &\;=\; \mathrm{variance\_recursion}(\varepsilon, \theta_{\mathrm{var}}).

The variance equation can be any of the GARCH-family variants:
``GARCH``, ``IGARCH``, ``GJR_GARCH``, ``EGARCH``, ``TGARCH``,
``QGARCH``.  Each variant exposes a uniform ``_ag_*`` backend
interface (pack / unpack / recursion / cold-start / forecast-step
/ rvs-step + flags for analytical-h-step support and
terminal-state class) that this composite consumes — so the joint
fit reuses each variant's existing kernel and reparameterisation
without duplication.

The joint conditional log-likelihood (Bollerslev 1986):

.. math::

    \ell(\theta) = \sum_t \bigl[
        \log f_z(\varepsilon_t / \sigma_t) - \log \sigma_t
      \bigr],

minimised over the combined unconstrained parameter vector via
:func:`projected_gradient`.  Standard errors are correct under
joint MLE — see :class:`copulax._src.timeseries._se` and the
``cov_matrix`` / ``standard_errors`` / ``summary`` methods.

API:

.. code-block:: python

    from copulax.timeseries import ArmaGarch, GJR_GARCH
    from copulax.univariate import skewed_t
    fit = ArmaGarch(
        mean_order=(1, 1),
        var_model=GJR_GARCH,
        var_order=(1, 1),
        residual_dist=skewed_t,
    ).fit(y)
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src._utils import _resolve_key
from copulax._src.optimize import projected_gradient
from copulax._src.timeseries._base import TerminalState, TimeSeriesModel
from copulax._src.timeseries._init import (
    arma_pre_sample_state,
    init_arma_params,
)
from copulax._src.timeseries._recursions import run_arma
from copulax._src.timeseries._residuals._standardise import StandardisedResidual
from copulax._src.timeseries._se import (
    compute_param_cov,
    flat_to_params,
    params_to_flat,
)
from copulax._src.timeseries._stationarity import (
    ar_to_raw,
    ma_to_raw,
    raw_to_ar,
    raw_to_ma,
)
from copulax._src.timeseries._variance._garch_base import GARCHBase
from copulax._src.timeseries._variance.garch import GARCH


_VAR_FLOOR: float = 1e-12
_SIGMA_FLOOR: float = 1e-6

#: Whitelist of variance models supported by the joint composite.
#: GARCH-M is excluded because it has its own mean equation that
#: would conflict with the ARMA mean here.
_SUPPORTED_VAR_MODELS: tuple = ()  # populated lazily; see _is_supported_var.


def _is_supported_var(var_model_cls: type) -> bool:
    """Whitelist check; populated lazily to avoid circular imports."""
    from copulax._src.timeseries._variance.egarch import EGARCH
    from copulax._src.timeseries._variance.gjr_garch import GJR_GARCH
    from copulax._src.timeseries._variance.igarch import IGARCH
    from copulax._src.timeseries._variance.qgarch import QGARCH
    from copulax._src.timeseries._variance.tgarch import TGARCH
    return var_model_cls in (
        GARCH, IGARCH, GJR_GARCH, EGARCH, TGARCH, QGARCH,
    )


class ArmaGarchTerminalState(TerminalState):
    r"""Composite terminal state — ARMA carry plus a variant-specific
    variance carry.

    ``y_lags`` and ``eps_lags`` come from the ARMA recursion;
    ``var_state`` is a variant-specific tuple matching the carry
    layout the chosen GARCH-family kernel consumes (e.g. for vanilla
    GARCH it's ``(eps_sq_lags, var_lags)``; for GJR it's
    ``(eps_sq_lags, neg_eps_sq_lags, var_lags)``).
    """
    y_lags: Array
    eps_lags: Array
    var_state: tuple


class ArmaGarch(TimeSeriesModel):
    r"""Joint ARMA(p, q) - GARCH-family(p', q') composite estimator.

    See module docstring for the model and the API contract.
    """

    # ---- static configuration -------------------------------------------
    mean_order: tuple = eqx.field(static=True)   # (p, q)
    var_order: tuple = eqx.field(static=True)    # (p', q')
    var_model: type = eqx.field(static=True)     # GARCH-family class
    residual_dist: Univariate = eqx.field(static=True)

    # ---- traced fitted parameters --------------------------------------
    phi: Optional[Array] = None
    theta: Optional[Array] = None
    c: Optional[Array] = None
    var_params: Optional[dict] = None
    residual_params: Optional[dict] = None
    terminal_state: Optional[ArmaGarchTerminalState] = None

    # ---- diagnostics ---------------------------------------------------
    loglikelihood_: Optional[Array] = None
    aic_: Optional[Array] = None
    bic_: Optional[Array] = None
    n_train_: Optional[int] = None
    cov_matrix_: Optional[Array] = None
    standard_errors_: Optional[dict] = None

    _supported_methods: ClassVar[frozenset] = frozenset(
        {"analytical", "backcast", "sample", "warm"}
    )

    def __init__(
        self,
        mean_order: tuple = (0, 0),
        var_model: type = GARCH,
        var_order: tuple = (0, 0),
        *,
        residual_dist: Optional[Univariate] = None,
        name: str = "ArmaGarch",
        phi=None, theta=None, c=None,
        var_params: Optional[dict] = None,
        residual_params: Optional[dict] = None,
        terminal_state: Optional[ArmaGarchTerminalState] = None,
        loglikelihood_=None, aic_=None, bic_=None,
        n_train_: Optional[int] = None,
        cov_matrix_=None, standard_errors_=None,
    ):
        if not _is_supported_var(var_model):
            raise NotImplementedError(
                f"ArmaGarch does not support {var_model.__name__!r} as a "
                "variance variant.  Supported: GARCH, IGARCH, GJR_GARCH, "
                "EGARCH, TGARCH, QGARCH.  GARCH-M has its own mean "
                "equation and is incompatible with the ARMA mean — fit it "
                "as a standalone model instead."
            )
        if (not isinstance(mean_order, tuple)) or len(mean_order) != 2:
            raise ValueError(
                f"mean_order must be a (p, q) tuple; got {mean_order!r}."
            )
        if (not isinstance(var_order, tuple)) or len(var_order) != 2:
            raise ValueError(
                f"var_order must be a (p, q) tuple; got {var_order!r}."
            )
        super().__init__(name=name)
        self.mean_order = (int(mean_order[0]), int(mean_order[1]))
        self.var_order = (int(var_order[0]), int(var_order[1]))
        self.var_model = var_model
        if residual_dist is None:
            from copulax._src.univariate.normal import normal as _normal
            residual_dist = _normal
        self.residual_dist = residual_dist
        self.phi = (
            jnp.asarray(phi, dtype=float).reshape(-1) if phi is not None else None
        )
        self.theta = (
            jnp.asarray(theta, dtype=float).reshape(-1) if theta is not None else None
        )
        self.c = jnp.asarray(c, dtype=float).reshape(()) if c is not None else None
        self.var_params = (
            {k: jnp.asarray(v, dtype=float) for k, v in var_params.items()}
            if var_params is not None else None
        )
        self.residual_params = residual_params
        self.terminal_state = terminal_state
        self.loglikelihood_ = (
            jnp.asarray(loglikelihood_, dtype=float).reshape(())
            if loglikelihood_ is not None else None
        )
        self.aic_ = (
            jnp.asarray(aic_, dtype=float).reshape(())
            if aic_ is not None else None
        )
        self.bic_ = (
            jnp.asarray(bic_, dtype=float).reshape(())
            if bic_ is not None else None
        )
        self.n_train_ = int(n_train_) if n_train_ is not None else None
        self.cov_matrix_ = (
            jnp.asarray(cov_matrix_, dtype=float)
            if cov_matrix_ is not None else None
        )
        self.standard_errors_ = (
            dict(standard_errors_) if standard_errors_ is not None else None
        )

    # ------------------------------------------------------------------
    # Order accessors / variance backend
    # ------------------------------------------------------------------
    @property
    def p(self) -> int:
        return self.mean_order[0]

    @property
    def q(self) -> int:
        return self.mean_order[1]

    @property
    def p_var(self) -> int:
        return self.var_order[0]

    @property
    def q_var(self) -> int:
        return self.var_order[1]

    @property
    def _var_backend(self) -> GARCHBase:
        r"""Unfitted variance instance providing the ``_ag_*`` backend.

        Cheap to construct (just static fields); reconstructed on
        every call rather than stored as a field so the ArmaGarch
        instance stays a clean PyTree.
        """
        return self.var_model(
            p=self.p_var, q=self.q_var,
            residual_dist=self.residual_dist,
        )

    # ------------------------------------------------------------------
    # params property
    # ------------------------------------------------------------------
    @property
    def _stored_params(self) -> Optional[dict]:
        r"""Canonical parameter dict.

        Schema (variant-dependent for the variance section):

        ``{
            "phi":       (p,),
            "theta":     (q,),
            "c":         (),
            <variance keys: variant-specific>,
            "residual":  {<shape-only dict>},
        }``
        """
        if (
            self.phi is None or self.theta is None or self.c is None
            or self.var_params is None or self.residual_params is None
        ):
            return None
        return {
            "phi": self.phi,
            "theta": self.theta,
            "c": self.c,
            **self.var_params,
            "residual": dict(self.residual_params),
        }

    @property
    def n_params(self) -> int:
        wrapper = StandardisedResidual(self.residual_dist)
        backend = self._var_backend
        # phi(p) + theta(q) + c(1) + variance natural params + residual shape
        n_var = sum(
            jnp.atleast_1d(jnp.asarray(v, dtype=float)).size
            for v in self._var_backend._ag_cold_start(
                jnp.zeros((10,), dtype=float),  # placeholder
                "backcast", None, wrapper,
            ).values()
        )
        return self.p + self.q + 1 + int(n_var) + wrapper.n_shape_params

    def _wrapper(self) -> StandardisedResidual:
        return StandardisedResidual(self.residual_dist)

    # ------------------------------------------------------------------
    # Pack / unpack — ARMA section + variance section + residual section
    # ------------------------------------------------------------------
    def _pack_x0(
        self,
        params_dict: dict,
        wrapper: StandardisedResidual,
    ) -> Array:
        r"""Layout: ``[raw_phi(p), raw_theta(q), c(1), var_section,
        raw_residual_shape]``."""
        backend = self._var_backend
        phi = jnp.asarray(params_dict["phi"], dtype=float).reshape(-1)
        theta = jnp.asarray(params_dict["theta"], dtype=float).reshape(-1)
        c = jnp.asarray(params_dict["c"], dtype=float).reshape(())
        residual = params_dict.get("residual", {}) or {}
        # Extract variance-specific keys.
        var_keys = backend._ag_var_keys()
        var_dict = {k: params_dict[k] for k in var_keys}

        raw_phi = ar_to_raw(phi) if self.p > 0 else jnp.zeros((0,), dtype=float)
        raw_theta = ma_to_raw(theta) if self.q > 0 else jnp.zeros((0,), dtype=float)
        raw_var = backend._ag_pack_x0(var_dict, wrapper, residual)
        raw_residual = wrapper.shape_params_to_array(residual)
        return jnp.concatenate(
            [raw_phi, raw_theta, c.reshape((1,)), raw_var, raw_residual]
        )

    def _unpack_raw(
        self,
        raw: Array,
        wrapper: StandardisedResidual,
    ) -> tuple[Array, Array, Array, dict, dict]:
        r"""Inverse of :meth:`_pack_x0`.

        Returns ``(phi, theta, c, var_dict, residual_dict)``.
        """
        backend = self._var_backend
        idx = 0
        raw_phi = raw[idx : idx + self.p]
        idx += self.p
        raw_theta = raw[idx : idx + self.q]
        idx += self.q
        c = raw[idx]
        idx += 1
        n_var_raw = backend._ag_n_raw(wrapper)
        raw_var = raw[idx : idx + n_var_raw]
        idx += n_var_raw
        raw_residual = raw[idx : idx + wrapper.n_shape_params]

        phi = raw_to_ar(raw_phi) if self.p > 0 else jnp.zeros((0,), dtype=float)
        theta = raw_to_ma(raw_theta) if self.q > 0 else jnp.zeros((0,), dtype=float)
        residual = wrapper.shape_params_from_array(raw_residual)
        var_dict = backend._ag_unpack_raw(raw_var, wrapper, residual)
        return phi, theta, c, var_dict, residual

    # ------------------------------------------------------------------
    # Recursion (ARMA → variance) using the backend
    # ------------------------------------------------------------------
    def _run_recursion(
        self,
        y: Array,
        phi: Array, theta: Array, c: Array,
        var_params: dict, residual_params: dict,
        init_y_lags: Array, init_eps_lags: Array,
        init_var_state: tuple,
    ) -> tuple[Array, Array, Array, ArmaGarchTerminalState]:
        backend = self._var_backend
        mu_seq, eps_seq, arma_terminal = run_arma(
            y=y, phi=phi, theta=theta, c=c,
            init_y_lags=init_y_lags, init_eps_lags=init_eps_lags,
        )
        var_seq, var_terminal_tuple = backend._ag_run_recursion(
            eps_seq, var_params, residual_params, init_var_state,
        )
        terminal_state = ArmaGarchTerminalState(
            y_lags=arma_terminal[0],
            eps_lags=arma_terminal[1],
            var_state=var_terminal_tuple,
        )
        return mu_seq, eps_seq, var_seq, terminal_state

    # ------------------------------------------------------------------
    # Initial state for the recursion
    # ------------------------------------------------------------------
    def _build_initial_state(
        self,
        y: Array,
        mode: str,
        backcast_length: Optional[int],
        residual_params: dict,
    ) -> tuple[Array, Array, tuple]:
        init_y_lags, init_eps_lags = arma_pre_sample_state(
            y, p=self.p, q=self.q, mode=mode, backcast_length=backcast_length,
        )
        init_var_state = self._var_backend._ag_initial_state(
            y, mode=mode, backcast_length=backcast_length,
            residual_params=residual_params,
        )
        return init_y_lags, init_eps_lags, init_var_state

    # ------------------------------------------------------------------
    # Cold-start
    # ------------------------------------------------------------------
    def _build_cold_start(
        self,
        y: Array,
        wrapper: StandardisedResidual,
        init: str,
        backcast_length: Optional[int],
    ) -> dict:
        arma_seed = init_arma_params(y, p=self.p, q=self.q, mode=init)
        if self.p > 0:
            y_arr = jnp.asarray(y, dtype=float).reshape(-1)
            y_centred = y_arr - jnp.mean(y_arr)
            n = int(y_arr.shape[0])
            r = jnp.zeros((n,), dtype=float)
            for i in range(self.p):
                r = r.at[i + 1:].add(-arma_seed["phi"][i] * y_centred[: n - (i + 1)])
            r = r + y_centred
            eps_proxy = r
        else:
            eps_proxy = (
                jnp.asarray(y, dtype=float).reshape(-1)
                - jnp.mean(jnp.asarray(y, dtype=float))
            )
        var_seed = self._var_backend._ag_cold_start(
            eps_proxy, mode=init, backcast_length=backcast_length, wrapper=wrapper,
        )
        return {
            "phi": arma_seed["phi"],
            "theta": arma_seed["theta"],
            "c": arma_seed["c"],
            **var_seed,
            "residual": wrapper.example_shape_params(),
        }

    # ------------------------------------------------------------------
    # Fit objective
    # ------------------------------------------------------------------
    def _make_objective(self, wrapper: StandardisedResidual):
        backend = self._var_backend

        def objective(
            raw: Array,
            y: Array,
            init_y_lags: Array,
            init_eps_lags: Array,
            init_var_state: tuple,
        ) -> Array:
            phi, theta, c, var_dict, residual = self._unpack_raw(raw, wrapper)
            _, eps_seq, var_seq, _ = self._run_recursion(
                y, phi, theta, c, var_dict, residual,
                init_y_lags, init_eps_lags, init_var_state,
            )
            sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
            z = eps_seq / sigma_seq
            logpdf = wrapper.logpdf(z, residual) - jnp.log(sigma_seq)
            finite = jnp.isfinite(logpdf)
            safe_logpdf = jnp.where(finite, logpdf, 0.0)
            invalid_penalty = 1e6 * (~finite).mean()
            return -safe_logpdf.mean() + invalid_penalty

        return objective

    # ------------------------------------------------------------------
    # Public fit
    # ------------------------------------------------------------------
    def fit(
        self,
        y: ArrayLike,
        *,
        init: str = "analytical",
        init_params: Optional[dict] = None,
        backcast_length: Optional[int] = None,
        maxiter: int = 300,
        lr: float = 0.05,
        name: Optional[str] = None,
    ) -> "ArmaGarch":
        r"""Fit the joint ARMA-GARCH composite to a level series ``y``.

        Single MLE over the combined parameter vector.
        """
        self._check_method(init)
        wrapper = StandardisedResidual(self.residual_dist)
        backend = self._var_backend
        var_keys = backend._ag_var_keys()
        y_arr = self._validate_series(y)
        n = int(y_arr.shape[0])
        self._validate_backcast_length(backcast_length, n)

        if init == "warm":
            if init_params is None:
                raise ValueError(
                    "init='warm' requires init_params (a parameter dict "
                    "matching the schema returned by `model.params`)."
                )
            cold = dict(init_params)
            for key in ("phi", "theta", "c", "residual"):
                if key not in cold:
                    raise KeyError(
                        f"Warm-start init_params missing required key {key!r}."
                    )
            for key in var_keys:
                if key not in cold:
                    raise KeyError(
                        f"Warm-start init_params missing variance key {key!r}."
                    )
        else:
            cold = self._build_cold_start(
                y_arr, wrapper, init=init, backcast_length=backcast_length,
            )

        x0 = self._pack_x0(cold, wrapper)

        _state_mode = "sample" if init == "sample" else "backcast"
        init_y_lags, init_eps_lags, init_var_state = self._build_initial_state(
            y_arr, mode=_state_mode, backcast_length=backcast_length,
            residual_params=cold.get("residual", wrapper.example_shape_params()),
        )

        objective = self._make_objective(wrapper)
        res = projected_gradient(
            f=objective,
            x0=x0,
            projection_method="projection_box",
            projection_options={
                "lower": jnp.full((x0.shape[0], 1), -jnp.inf),
                "upper": jnp.full((x0.shape[0], 1), jnp.inf),
            },
            y=y_arr,
            init_y_lags=init_y_lags,
            init_eps_lags=init_eps_lags,
            init_var_state=init_var_state,
            lr=lr,
            maxiter=maxiter,
        )
        x_opt = res["x"]
        phi, theta, c, var_dict, residual = self._unpack_raw(x_opt, wrapper)

        _, _, _, terminal = self._run_recursion(
            y_arr, phi, theta, c, var_dict, residual,
            init_y_lags, init_eps_lags, init_var_state,
        )
        nll = objective(
            x_opt, y_arr, init_y_lags, init_eps_lags, init_var_state,
        )
        loglike = -nll * n
        # n_params: phi + theta + c + variance + residual.
        n_var = sum(
            jnp.atleast_1d(jnp.asarray(v, dtype=float)).size
            for v in var_dict.values()
        )
        n_params_total = (
            self.p + self.q + 1
            + int(n_var)
            + wrapper.n_shape_params
        )
        aic = 2.0 * n_params_total - 2.0 * loglike
        bic = (
            n_params_total * jnp.log(jnp.asarray(n, dtype=float))
            - 2.0 * loglike
        )

        # Compute joint robust SE in natural-parameter space.
        params_dict_for_se = {
            "phi": phi, "theta": theta, "c": c,
            **var_dict,
            "residual": residual,
        }
        cov_const, _, se_dict = self._compute_se(
            params_dict=params_dict_for_se,
            wrapper=wrapper,
            y_arr=y_arr,
            init_y_lags=init_y_lags,
            init_eps_lags=init_eps_lags,
            init_var_state=init_var_state,
            n_obs=n,
        )

        if name is None:
            name = (
                f"FittedArmaGarch(({self.p},{self.q})x"
                f"({self.p_var},{self.q_var}))-{self.var_model.__name__}"
                f"-{self.residual_dist.name}"
            )
        return type(self)(
            mean_order=self.mean_order,
            var_model=self.var_model,
            var_order=self.var_order,
            residual_dist=self.residual_dist,
            name=name,
            phi=phi, theta=theta, c=c,
            var_params=var_dict,
            residual_params=residual,
            terminal_state=terminal,
            loglikelihood_=loglike, aic_=aic, bic_=bic, n_train_=n,
            cov_matrix_=cov_const,
            standard_errors_=se_dict,
        )

    # ------------------------------------------------------------------
    # Conditional moments / residuals
    # ------------------------------------------------------------------
    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError(
                f"Model {self.name!r} is not fitted; call `.fit(y)` first."
            )

    def _recursion_inputs(
        self,
        y: ArrayLike,
        init: str,
        backcast_length: Optional[int],
    ):
        y_arr = self._validate_series(y)
        n = int(y_arr.shape[0])
        self._validate_backcast_length(backcast_length, n)
        state = self._build_initial_state(
            y_arr, mode=init, backcast_length=backcast_length,
            residual_params=self.residual_params,
        )
        return y_arr, state

    def conditional_mean(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        self._require_fitted()
        y_arr, (init_y_lags, init_eps_lags, init_var_state) = (
            self._recursion_inputs(y, init, backcast_length)
        )
        mu_seq, _, _, _ = self._run_recursion(
            y_arr, self.phi, self.theta, self.c,
            self.var_params, self.residual_params,
            init_y_lags, init_eps_lags, init_var_state,
        )
        return mu_seq

    def conditional_variance(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        self._require_fitted()
        y_arr, (init_y_lags, init_eps_lags, init_var_state) = (
            self._recursion_inputs(y, init, backcast_length)
        )
        _, _, var_seq, _ = self._run_recursion(
            y_arr, self.phi, self.theta, self.c,
            self.var_params, self.residual_params,
            init_y_lags, init_eps_lags, init_var_state,
        )
        return var_seq

    def residuals(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> dict:
        r"""Returns ``{"mean_residuals": ε_t,
        "standardised_residuals": z_t}``."""
        self._require_fitted()
        y_arr, (init_y_lags, init_eps_lags, init_var_state) = (
            self._recursion_inputs(y, init, backcast_length)
        )
        _, eps_seq, var_seq, _ = self._run_recursion(
            y_arr, self.phi, self.theta, self.c,
            self.var_params, self.residual_params,
            init_y_lags, init_eps_lags, init_var_state,
        )
        sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
        return {
            "mean_residuals": eps_seq,
            "standardised_residuals": eps_seq / sigma_seq,
        }

    def terminal_state_from(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> ArmaGarchTerminalState:
        self._require_fitted()
        y_arr, (init_y_lags, init_eps_lags, init_var_state) = (
            self._recursion_inputs(y, init, backcast_length)
        )
        _, _, _, terminal = self._run_recursion(
            y_arr, self.phi, self.theta, self.c,
            self.var_params, self.residual_params,
            init_y_lags, init_eps_lags, init_var_state,
        )
        return terminal

    # ------------------------------------------------------------------
    # Loglikelihood / aic / bic
    # ------------------------------------------------------------------
    def _log_likelihood_on_series(
        self,
        y: ArrayLike,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        self._require_fitted()
        wrapper = self._wrapper()
        y_arr, (init_y_lags, init_eps_lags, init_var_state) = (
            self._recursion_inputs(y, init, backcast_length)
        )
        _, eps_seq, var_seq, _ = self._run_recursion(
            y_arr, self.phi, self.theta, self.c,
            self.var_params, self.residual_params,
            init_y_lags, init_eps_lags, init_var_state,
        )
        sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
        z = eps_seq / sigma_seq
        logpdf = wrapper.logpdf(z, self.residual_params) - jnp.log(sigma_seq)
        return jnp.sum(logpdf)

    def loglikelihood(
        self,
        y: Optional[ArrayLike] = None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        if y is None:
            self._require_fitted()
            return self.loglikelihood_
        return self._log_likelihood_on_series(
            y, init=init, backcast_length=backcast_length,
        )

    def aic(
        self,
        y: Optional[ArrayLike] = None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        if y is None:
            self._require_fitted()
            return self.aic_
        ll = self._log_likelihood_on_series(
            y, init=init, backcast_length=backcast_length,
        )
        return 2.0 * self.n_params - 2.0 * ll

    def bic(
        self,
        y: Optional[ArrayLike] = None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        if y is None:
            self._require_fitted()
            return self.bic_
        y_arr = self._validate_series(y)
        ll = self._log_likelihood_on_series(
            y_arr, init=init, backcast_length=backcast_length,
        )
        n_obs = jnp.asarray(int(y_arr.shape[0]), dtype=float)
        return self.n_params * jnp.log(n_obs) - 2.0 * ll

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        r"""Analytic, parameter-only diagnostics for the joint composite.

        Mean side: ARMA stationarity / AR-root moduli.  Variance
        side: defers to the variance backend's ``.stats()`` (a
        temporary fitted instance is constructed for the call).
        """
        self._require_fitted()
        from copulax._src.timeseries._stationarity import (
            ar_is_stationary,
            ar_polynomial_roots,
            ma_is_invertible,
            ma_polynomial_roots,
        )
        ar_roots = ar_polynomial_roots(self.phi)
        ma_roots = (
            ma_polynomial_roots(self.theta)
            if self.q > 0 else jnp.zeros((0,), dtype=jnp.complex64)
        )
        mean_is_stat = ar_is_stationary(self.phi)
        mean_is_inv = (
            ma_is_invertible(self.theta) if self.q > 0 else jnp.asarray(True)
        )
        ar_factor = 1.0 - jnp.sum(self.phi) if self.p > 0 else 1.0
        ar_factor_safe = jnp.where(jnp.abs(ar_factor) < 1e-12, 1e-12, ar_factor)
        unconditional_mean = self.c / ar_factor_safe

        # Build a fitted variance instance to query its stats() directly.
        var_helper_fitted = self.var_model(
            p=self.p_var, q=self.q_var, residual_dist=self.residual_dist,
            **{k: v for k, v in self.var_params.items()},
            residual_params=self.residual_params,
        )
        var_stats = var_helper_fitted.stats()
        return {
            "unconditional_mean": unconditional_mean,
            **{f"var_{k}": v for k, v in var_stats.items()},
            "mean_is_stationary": mean_is_stat,
            "mean_is_invertible": mean_is_inv,
            "ar_root_moduli": jnp.abs(ar_roots),
            "ma_root_moduli": jnp.abs(ma_roots),
        }

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------
    def forecast(
        self,
        h: int,
        *,
        method: str = "analytical",
        n_paths: int = 0,
        key: Optional[Array] = None,
        last_state: Optional[ArmaGarchTerminalState] = None,
    ) -> dict:
        self._require_fitted()
        h = int(h)
        if h <= 0:
            raise ValueError(f"forecast horizon h must be > 0; got {h}.")
        state = last_state if last_state is not None else self.terminal_state
        if state is None:
            raise ValueError(
                "No terminal state available; pass `last_state` explicitly "
                "or fit on a series first."
            )
        backend = self._var_backend

        if method == "analytical":
            if h >= 2 and not backend._ag_supports_analytical_h_step():
                raise ValueError(
                    f"{self.var_model.__name__} variance does not support "
                    f"analytical forecasts at h>=2 (closed-form moments "
                    "of the residual law are unavailable).  Use "
                    "method='simulation' instead."
                )
            # Mean rollout: ε_t replaced by 0 for unobserved future.
            mu_path = []
            y_lags = state.y_lags
            eps_lags = state.eps_lags
            for _ in range(h):
                ar_term = jnp.dot(self.phi, y_lags) if self.p > 0 else 0.0
                ma_term = jnp.dot(self.theta, eps_lags) if self.q > 0 else 0.0
                mu = self.c + ar_term + ma_term
                mu_path.append(mu)
                if self.p > 0:
                    y_lags = jnp.concatenate(
                        [mu.reshape((1,)), y_lags[:-1]]
                    )
                if self.q > 0:
                    eps_lags = jnp.concatenate(
                        [jnp.zeros((1,), dtype=float), eps_lags[:-1]]
                    )
            mean = jnp.stack(mu_path)
            # Variance rollout via backend forecast step.
            var_path = []
            var_state = state.var_state
            for _ in range(h):
                var_t, var_state = backend._ag_forecast_step(
                    self.var_params, self.residual_params, var_state,
                )
                var_path.append(var_t)
            variance = jnp.stack(var_path)
            return {"mean": mean, "variance": variance, "paths": None}

        elif method == "simulation":
            if n_paths <= 0:
                raise ValueError("method='simulation' requires n_paths > 0.")
            key = _resolve_key(key)
            paths = self.rvs(
                size=(int(n_paths), h), key=key, last_state=state,
            )
            return {
                "mean": jnp.mean(paths, axis=0),
                "variance": jnp.var(paths, axis=0),
                "paths": paths,
            }

        else:
            raise ValueError(
                f"Unknown forecast method {method!r}; expected "
                "'analytical' or 'simulation'."
            )

    # ------------------------------------------------------------------
    # rvs — simulate y (level series)
    # ------------------------------------------------------------------
    def _roll_path(
        self, z: Array, state: ArmaGarchTerminalState,
    ) -> Array:
        backend = self._var_backend
        c = self.c
        phi = self.phi
        theta = self.theta
        var_params = self.var_params
        residual_params = self.residual_params

        def step(carry, z_t):
            y_lags, eps_lags, var_state = carry
            ar_term = jnp.dot(phi, y_lags) if self.p > 0 else 0.0
            ma_term = jnp.dot(theta, eps_lags) if self.q > 0 else 0.0
            mu = c + ar_term + ma_term
            # Variance at this step depends on prior eps² lags (in
            # var_state); compute σ_t first using a placeholder ε.
            # We need σ_t before computing eps_t — for σ²-form
            # variants this works because var depends only on lags
            # of ε² (carry only).  Backend handles the family-
            # specific update.
            var_t, _ = backend._ag_rvs_step(
                var_params, residual_params, var_state, jnp.zeros((), dtype=float),
            )
            sigma_t = jnp.sqrt(jnp.maximum(var_t, _VAR_FLOOR))
            eps_t = sigma_t * z_t
            y_t = mu + eps_t
            # Re-run the variance step with the actual eps_t so the
            # state update reflects the realised innovation.
            _, new_var_state = backend._ag_rvs_step(
                var_params, residual_params, var_state, eps_t,
            )
            new_y_lags = (
                jnp.concatenate([y_t.reshape((1,)), y_lags[:-1]])
                if self.p > 0 else y_lags
            )
            new_eps_lags = (
                jnp.concatenate([eps_t.reshape((1,)), eps_lags[:-1]])
                if self.q > 0 else eps_lags
            )
            return (new_y_lags, new_eps_lags, new_var_state), y_t

        init_carry = (state.y_lags, state.eps_lags, state.var_state)
        _, y_seq = jax.lax.scan(step, init_carry, z)
        return y_seq

    def rvs(
        self,
        size=None,
        *,
        key: Optional[Array] = None,
        u: Optional[ArrayLike] = None,
        last_state: Optional[ArmaGarchTerminalState] = None,
    ) -> Array:
        self._require_fitted()
        wrapper = self._wrapper()
        state = last_state if last_state is not None else self.terminal_state
        if state is None:
            raise ValueError(
                "No terminal state available; pass `last_state` explicitly "
                "or fit on a series first."
            )

        if u is not None:
            u_arr = jnp.asarray(u, dtype=float)
            shape = u_arr.shape
            if u_arr.ndim == 1:
                z = wrapper.ppf(u_arr, self.residual_params)
                return self._roll_path(z, state)
            elif u_arr.ndim == 2:
                z = wrapper.ppf(u_arr.reshape(-1), self.residual_params).reshape(shape)
                return jax.vmap(lambda zi: self._roll_path(zi, state))(z)
            else:
                raise ValueError(f"u must have ndim 1 or 2; got ndim={u_arr.ndim}.")

        if size is None:
            raise ValueError(
                "rvs requires either `size` or `u` to determine output shape."
            )
        if isinstance(size, int):
            shape = (size,)
        else:
            shape = tuple(int(s) for s in size)
        key = _resolve_key(key)

        if len(shape) == 1:
            h = shape[0]
            z = wrapper.rvs(size=(h,), shape_params=self.residual_params, key=key)
            return self._roll_path(z, state)
        elif len(shape) == 2:
            n_paths, h = shape
            keys = jax.random.split(key, n_paths)
            z_batch = jax.vmap(
                lambda k: wrapper.rvs(size=(h,), shape_params=self.residual_params, key=k)
            )(keys)
            return jax.vmap(lambda z: self._roll_path(z, state))(z_batch)
        else:
            raise ValueError(
                f"size must be 1- or 2-dimensional; got {shape}."
            )

    @property
    def residual_distribution(self) -> Univariate:
        self._require_fitted()
        return self._wrapper().to_distribution(
            self.residual_params,
            name=f"{self.residual_dist.name}-stdresid-{self.name}",
        )

    # ------------------------------------------------------------------
    # Standard errors
    # ------------------------------------------------------------------
    def _natural_objective_closures(
        self,
        wrapper: StandardisedResidual,
        y_arr: Array,
        init_y_lags: Array, init_eps_lags: Array,
        init_var_state: tuple,
    ) -> tuple[Callable, Callable, list]:
        backend = self._var_backend
        var_keys = backend._ag_var_keys()
        # Schema from a canonical example params dict.
        example_var = backend._ag_cold_start(
            jnp.zeros((100,), dtype=float),  # dummy series
            "backcast", None, wrapper,
        )
        example_params = {
            "phi": jnp.zeros((self.p,), dtype=float),
            "theta": jnp.zeros((self.q,), dtype=float),
            "c": jnp.asarray(0.0, dtype=float),
            **example_var,
            "residual": wrapper.example_shape_params(),
        }
        _, schema = params_to_flat(example_params)

        def per_obs_nll(flat: Array) -> Array:
            params = flat_to_params(flat, schema)
            phi_ = params["phi"]
            theta_ = params["theta"]
            c_ = params["c"]
            var_dict_ = {k: params[k] for k in var_keys}
            residual_ = params.get("residual", {}) or {}
            _, eps_seq, var_seq, _ = self._run_recursion(
                y_arr, phi_, theta_, c_, var_dict_, residual_,
                init_y_lags, init_eps_lags, init_var_state,
            )
            sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
            z = eps_seq / sigma_seq
            logpdf = wrapper.logpdf(z, residual_) - jnp.log(sigma_seq)
            return -jnp.where(jnp.isfinite(logpdf), logpdf, 0.0)

        def nll_total(flat: Array) -> Array:
            return jnp.sum(per_obs_nll(flat))

        return nll_total, per_obs_nll, schema

    def _compute_se(
        self,
        params_dict: dict,
        wrapper: StandardisedResidual,
        y_arr: Array,
        init_y_lags: Array, init_eps_lags: Array,
        init_var_state: tuple,
        n_obs: int,
        cov_type: str = "robust",
    ) -> tuple[Array, Array, dict]:
        nll_total, per_obs_nll, schema = self._natural_objective_closures(
            wrapper, y_arr, init_y_lags, init_eps_lags, init_var_state,
        )
        params_flat, _ = params_to_flat(params_dict)
        cov = compute_param_cov(
            nll_total=nll_total,
            per_obs_nll=per_obs_nll,
            params_flat=params_flat,
            n_obs=n_obs,
            cov_type=cov_type,
        )
        se_flat = jnp.sqrt(jnp.maximum(jnp.diag(cov), 0.0))
        se_dict = flat_to_params(se_flat, schema)
        return cov, se_flat, se_dict

    def cov_matrix(
        self,
        y: Optional[ArrayLike] = None,
        *,
        cov_type: str = "robust",
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        self._require_fitted()
        if y is None and cov_type == "robust":
            return self.cov_matrix_
        if y is None:
            raise ValueError(
                f"cov_type={cov_type!r} requires explicit y for recomputation."
            )
        return self._recompute_se(
            y, init=init, backcast_length=backcast_length, cov_type=cov_type,
        )[0]

    def standard_errors(
        self,
        y: Optional[ArrayLike] = None,
        *,
        cov_type: str = "robust",
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> dict:
        self._require_fitted()
        if y is None and cov_type == "robust":
            return self.standard_errors_
        if y is None:
            raise ValueError(
                f"cov_type={cov_type!r} requires explicit y for recomputation."
            )
        return self._recompute_se(
            y, init=init, backcast_length=backcast_length, cov_type=cov_type,
        )[2]

    def _recompute_se(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
        cov_type: str = "robust",
    ) -> tuple[Array, Array, dict]:
        wrapper = self._wrapper()
        y_arr, (init_y_lags, init_eps_lags, init_var_state) = (
            self._recursion_inputs(y, init, backcast_length)
        )
        n_obs = int(y_arr.shape[0])
        return self._compute_se(
            params_dict=self.params,
            wrapper=wrapper, y_arr=y_arr,
            init_y_lags=init_y_lags, init_eps_lags=init_eps_lags,
            init_var_state=init_var_state,
            n_obs=n_obs, cov_type=cov_type,
        )

    def confidence_intervals(self, alpha: float = 0.05) -> dict:
        self._require_fitted()
        if self.standard_errors_ is None or self.params is None:
            raise ValueError(
                "Confidence intervals require both `standard_errors_` "
                "and `params` to be populated."
            )
        from jax.scipy.stats import norm
        z = float(norm.ppf(1.0 - alpha / 2.0))
        cis: dict = {}
        for key, val in self.params.items():
            if isinstance(val, dict):
                cis[key] = {
                    sub: (
                        jnp.asarray(val[sub], dtype=float)
                        - z * jnp.asarray(self.standard_errors_[key][sub], dtype=float),
                        jnp.asarray(val[sub], dtype=float)
                        + z * jnp.asarray(self.standard_errors_[key][sub], dtype=float),
                    )
                    for sub in val
                }
            else:
                v = jnp.asarray(val, dtype=float)
                s = jnp.asarray(self.standard_errors_[key], dtype=float)
                cis[key] = (v - z * s, v + z * s)
        return cis

    def summary(
        self,
        y: Optional[ArrayLike] = None,
        *,
        alpha: float = 0.05,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> str:
        self._require_fitted()
        from jax.scipy.stats import norm
        z_crit = float(norm.ppf(1.0 - alpha / 2.0))

        if y is None:
            se = self.standard_errors_
            ll = float(self.loglikelihood_)
            aic = float(self.aic_)
            bic = float(self.bic_)
        else:
            _, _, se = self._recompute_se(
                y, init=init, backcast_length=backcast_length,
            )
            ll = float(self.loglikelihood(y, init=init, backcast_length=backcast_length))
            aic = float(self.aic(y, init=init, backcast_length=backcast_length))
            bic = float(self.bic(y, init=init, backcast_length=backcast_length))

        backend = self._var_backend
        var_keys = backend._ag_var_keys()

        rows: list[tuple[str, float, float]] = []
        for key in ("phi", "theta", "c", *var_keys):
            est = self.params[key]
            se_val = se[key]
            est_arr = jnp.atleast_1d(jnp.asarray(est, dtype=float))
            se_arr = jnp.atleast_1d(jnp.asarray(se_val, dtype=float))
            for i in range(est_arr.shape[0]):
                if est_arr.shape[0] > 1 or key in ("phi", "theta", "alpha", "beta",
                                                    "gamma", "psi",
                                                    "alpha_pos", "alpha_neg"):
                    label = f"{key}[{i + 1}]"
                else:
                    label = key
                rows.append((label, float(est_arr[i]), float(se_arr[i])))
        for sub_key, sub_est in self.params["residual"].items():
            sub_se = se["residual"][sub_key]
            rows.append((
                f"residual.{sub_key}",
                float(jnp.asarray(sub_est, dtype=float).reshape(())),
                float(jnp.asarray(sub_se, dtype=float).reshape(())),
            ))

        header_label = (
            f"ArmaGarch({self.p},{self.q}) × "
            f"{self.var_model.__name__}({self.p_var},{self.q_var}) — "
            f"{self.residual_dist.name} residuals"
        )
        line = "=" * 78
        out: list[str] = [header_label, line]
        out.append(
            f"{'param':<14} {'estimate':>12} {'std err':>12} "
            f"{'z':>8} {'P>|z|':>10} {'[lower, upper]':>20}"
        )
        out.append("-" * 78)
        for label, est, s in rows:
            if s > 0.0 and float(jnp.isfinite(jnp.asarray(s))):
                z_stat = est / s
                p_val = 2.0 * (1.0 - float(norm.cdf(abs(z_stat))))
                lo = est - z_crit * s
                hi = est + z_crit * s
                out.append(
                    f"{label:<14} {est:>12.4f} {s:>12.4f} "
                    f"{z_stat:>8.2f} {p_val:>10.4f} "
                    f"[{lo:>+8.4f}, {hi:>+8.4f}]"
                )
            else:
                out.append(
                    f"{label:<14} {est:>12.4f} {s:>12.4f} "
                    f"{'--':>8} {'--':>10} {'--':>20}"
                )
        out.append("-" * 78)
        out.append(
            f"loglikelihood: {ll:.4f}  AIC: {aic:.4f}  BIC: {bic:.4f}  "
            f"n_train: {self.n_train_}"
        )
        out.append(line)
        return "\n".join(out)

    # ------------------------------------------------------------------
    # Diagnostics — on the joint composite's standardised residuals
    # ------------------------------------------------------------------
    def _standardised_residuals(
        self,
        y: ArrayLike,
        init: str,
        backcast_length: Optional[int],
    ) -> Array:
        return self.residuals(
            y, init=init, backcast_length=backcast_length,
        )["standardised_residuals"]

    def acf(
        self, y: ArrayLike, lags: int = 20,
        *, init: str = "backcast", backcast_length: Optional[int] = None,
    ) -> Array:
        from copulax._src.timeseries._diagnostics import acf as _acf
        return _acf(self._standardised_residuals(y, init, backcast_length), lags)

    def pacf(
        self, y: ArrayLike, lags: int = 20, method: str = "yule_walker",
        *, init: str = "backcast", backcast_length: Optional[int] = None,
    ) -> Array:
        from copulax._src.timeseries._diagnostics import pacf as _pacf
        return _pacf(
            self._standardised_residuals(y, init, backcast_length),
            lags, method=method,
        )

    def ljung_box(
        self, y: ArrayLike, lags: int = 10,
        *, init: str = "backcast", backcast_length: Optional[int] = None,
    ) -> tuple[Array, Array]:
        from copulax._src.timeseries._diagnostics import ljung_box as _lb
        return _lb(self._standardised_residuals(y, init, backcast_length), lags)

    def arch_lm(
        self, y: ArrayLike, lags: int = 5,
        *, init: str = "backcast", backcast_length: Optional[int] = None,
    ) -> tuple[Array, Array]:
        from copulax._src.timeseries._diagnostics import arch_lm as _alm
        return _alm(self._standardised_residuals(y, init, backcast_length), lags)

    def plot_acf(
        self, y: ArrayLike, lags: int = 20, alpha: float = 0.05, ax=None,
        *, init: str = "backcast", backcast_length: Optional[int] = None,
    ):
        from copulax._src.timeseries._diagnostics import plot_acf as _plot_acf
        return _plot_acf(
            self._standardised_residuals(y, init, backcast_length),
            lags=lags, alpha=alpha, ax=ax,
        )

    def plot_pacf(
        self, y: ArrayLike, lags: int = 20, method: str = "yule_walker",
        alpha: float = 0.05, ax=None,
        *, init: str = "backcast", backcast_length: Optional[int] = None,
    ):
        from copulax._src.timeseries._diagnostics import plot_pacf as _plot_pacf
        return _plot_pacf(
            self._standardised_residuals(y, init, backcast_length),
            lags=lags, method=method, alpha=alpha, ax=ax,
        )

    # ------------------------------------------------------------------
    # Model-fit plots — composite (mean overlay + variance bands)
    # ------------------------------------------------------------------
    def plot_timeseries(
        self,
        y: ArrayLike,
        h: int = 0,
        m: int = 5,
        show_rolling: bool = True,
        alpha: tuple = (0.05, 0.95),
        axes=None,
    ) -> tuple:
        r"""Two-panel time-series chart: top = ``y`` with conditional-
        mean overlay (and optional ``h``-step extension); bottom =
        ``ε_t`` with VaR bands.  Returns ``(ax_mean, ax_vol)``."""
        from copulax._src.timeseries._plotting import plot_timeseries_joint
        return plot_timeseries_joint(
            self, y, h=h, m=m, show_rolling=show_rolling,
            alpha=alpha, axes=axes,
        )

    def plot_scatter(
        self,
        y: ArrayLike,
        m: int = 5,
        axes=None,
    ) -> tuple:
        r"""Three-panel diagnostic: actual-vs-forecast scatter,
        :math:`\sigma_t` scatter, and Q-Q plot of standardised
        residuals.  Returns ``(ax_mean, ax_vol, ax_qq)``."""
        from copulax._src.timeseries._plotting import plot_scatter_joint
        return plot_scatter_joint(self, y, m=m, axes=axes)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def _serialise_static(self) -> dict:
        return {
            "mean_order": list(self.mean_order),
            "var_order": list(self.var_order),
            "var_model_class": self.var_model.__name__,
            "residual_dist_class": type(self.residual_dist).__name__,
        }

    @classmethod
    def _deserialise(
        cls,
        metadata: dict,
        arrays: dict,
        residual_dist,
        name: Optional[str] = None,
    ) -> "ArmaGarch":
        r"""Reconstruct an ArmaGarch fitted instance from saved state.

        The saved ``params`` dict has variance-keys flattened to the
        top level (matching the canonical ``params`` schema users
        read), so we re-nest them into ``var_params`` here using the
        backend's ``_ag_var_keys`` to know which keys belong to the
        variance section.
        """
        from copulax._src.timeseries._se import flat_to_params
        from copulax._src.timeseries._variance._garch_base import (
            _lookup_terminal_state_class,
        )

        # Look up the variance class from its name.
        var_model_name = metadata["var_model_class"]
        var_class_table = _variance_class_table()
        if var_model_name not in var_class_table:
            raise ValueError(
                f"Unknown variance class {var_model_name!r}.  Update "
                "`_variance_class_table` in arma_garch.py."
            )
        var_model_class = var_class_table[var_model_name]

        kwargs: dict = {
            "mean_order": tuple(metadata["mean_order"]),
            "var_order": tuple(metadata["var_order"]),
            "var_model": var_model_class,
            "residual_dist": residual_dist,
        }
        if name is not None:
            kwargs["name"] = name

        if "params_schema" in metadata:
            schema = [(k, tuple(s)) for k, s in metadata["params_schema"]]
            params = flat_to_params(arrays["params_flat"], schema)
            kwargs["phi"] = params.get("phi")
            kwargs["theta"] = params.get("theta")
            kwargs["c"] = params.get("c")
            if "residual" in params:
                kwargs["residual_params"] = params["residual"]
            # Re-nest variance keys via the backend's var_keys.
            unfitted = var_model_class(
                p=metadata["var_order"][0],
                q=metadata["var_order"][1],
                residual_dist=residual_dist,
            )
            var_keys = unfitted._ag_var_keys()
            kwargs["var_params"] = {
                k: params[k] for k in var_keys if k in params
            }

        if "ts_n_leaves" in metadata:
            n_leaves = int(metadata["ts_n_leaves"])
            leaves = [arrays[f"ts_{i}"] for i in range(n_leaves)]
            # ArmaGarchTerminalState fields: y_lags, eps_lags, var_state.
            # First two are arrays, var_state is a tuple of arrays.
            y_lags = leaves[0]
            eps_lags = leaves[1]
            var_state = tuple(leaves[2:])
            kwargs["terminal_state"] = ArmaGarchTerminalState(
                y_lags=y_lags, eps_lags=eps_lags, var_state=var_state,
            )

        for key in ("loglikelihood_", "aic_", "bic_", "n_train_"):
            arr_key = f"diag_{key}"
            if arr_key in arrays:
                kwargs[key] = arrays[arr_key]

        if "cov_matrix_" in arrays:
            kwargs["cov_matrix_"] = arrays["cov_matrix_"]
        if "se_flat" in arrays and "se_schema" in metadata:
            se_schema = [(k, tuple(s)) for k, s in metadata["se_schema"]]
            kwargs["standard_errors_"] = flat_to_params(
                arrays["se_flat"], se_schema,
            )

        return cls(**kwargs)


def _variance_class_table() -> dict[str, type]:
    r"""Resolve the supported variance-class names to classes.

    Used at deserialisation time to map a saved ``var_model_class``
    string back to the actual class.
    """
    from copulax._src.timeseries._variance.egarch import EGARCH
    from copulax._src.timeseries._variance.gjr_garch import GJR_GARCH
    from copulax._src.timeseries._variance.igarch import IGARCH
    from copulax._src.timeseries._variance.qgarch import QGARCH
    from copulax._src.timeseries._variance.tgarch import TGARCH
    return {
        "GARCH": GARCH,
        "IGARCH": IGARCH,
        "GJR_GARCH": GJR_GARCH,
        "EGARCH": EGARCH,
        "TGARCH": TGARCH,
        "QGARCH": QGARCH,
    }
