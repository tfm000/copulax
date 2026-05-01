r"""ARMA(p, q) - GARCH(p', q') joint composite estimator.

Single composite class fitting the mean and variance equations
under one MLE objective:

.. math::

    y_t       &= c + \sum_{i=1}^p \phi_i\, y_{t-i}
                  + \sum_{j=1}^q \theta_j\, \varepsilon_{t-j}
                  + \varepsilon_t,\\
    \varepsilon_t &= \sigma_t\, z_t,
                  \quad z_t \sim f_z\,(\text{mean}=0, \mathrm{var}=1),\\
    \sigma^2_t   &= \omega
                  + \sum_{i=1}^{p'} \alpha_i\, \varepsilon^2_{t-i}
                  + \sum_{j=1}^{q'} \beta_j\, \sigma^2_{t-j}.

The joint conditional log-likelihood (Bollerslev 1986):

.. math::

    \ell(\theta_{\mathrm{mean}},
         \theta_{\mathrm{var}},
         \theta_{\mathrm{resid}})
    = \sum_t \bigl[
        \log f_z\!\bigl(\varepsilon_t / \sigma_t\bigr)
        - \log \sigma_t
      \bigr].

Single MLE over the combined unconstrained parameter vector via
the same :func:`projected_gradient` engine used by every other
fit in the subpackage.  Because the estimator is fully joint,
**standard errors are correct here** — see plan §"Standard errors
(v1)" for the asymptotic-theory treatment that distinguishes the
joint case from the separable two-stage path.

v1 supports vanilla :class:`GARCH` as the variance equation.
Extensions to :class:`IGARCH`, :class:`GJR_GARCH`, :class:`EGARCH`,
:class:`TGARCH`, and :class:`QGARCH` are straightforward — each
needs its own pack/unpack and recursion-call (the composite
infrastructure here is shared) — and follow in subsequent commits.

API (matches the rest of the subpackage — orders / variant /
residual at construction time, ``fit(y)`` data-only):

.. code-block:: python

    from copulax.timeseries import ArmaGarch, GARCH
    from copulax.univariate import skewed_t
    fit = ArmaGarch(
        mean_order=(1, 1),
        var_model=GARCH,
        var_order=(1, 1),
        residual_dist=skewed_t,
    ).fit(y)
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

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
    garch_pre_sample_state,
    init_arma_params,
    init_garch_params,
)
from copulax._src.timeseries._mean._arma_base import ARMATerminalState
from copulax._src.timeseries._recursions import run_arma, run_garch
from copulax._src.timeseries._residuals._standardise import StandardisedResidual
from copulax._src.timeseries._stationarity import (
    ar_to_raw,
    garch_simplex,
    garch_unsimplex,
    ma_to_raw,
    positive_to_raw,
    raw_to_ar,
    raw_to_positive,
    raw_to_ma,
)
from copulax._src.timeseries._variance._garch_base import GARCHTerminalState
from copulax._src.timeseries._variance.garch import GARCH


_VAR_FLOOR: float = 1e-12
_SIGMA_FLOOR: float = 1e-6


class ArmaGarchTerminalState(TerminalState):
    r"""Composite terminal state — union of ARMA and GARCH carries.

    Stores the last ``p`` returns and ``q`` innovations on the mean
    side, plus the last ``p'`` squared residuals and ``q'``
    conditional variances on the variance side.
    """
    y_lags: Array          # ARMA: last p y values
    eps_lags: Array        # ARMA: last q innovations
    eps_sq_lags: Array     # GARCH: last p' squared residuals
    var_lags: Array        # GARCH: last q' conditional variances


class ArmaGarch(TimeSeriesModel):
    r"""Joint ARMA(p, q) - GARCH(p', q') composite estimator.

    Inherits from :class:`TimeSeriesModel` directly (rather than
    :class:`MeanModel` or :class:`VarianceModel`) because it owns
    *both* a mean and a variance recursion under a single MLE.

    Construction:

    .. code-block:: python

        from copulax.timeseries import ArmaGarch, GARCH
        from copulax.univariate import normal
        model = ArmaGarch(
            mean_order=(1, 1),
            var_model=GARCH,
            var_order=(1, 1),
            residual_dist=normal,
        )
        fit = model.fit(y)
    """

    # ---- static configuration -------------------------------------------
    mean_order: tuple = eqx.field(static=True)   # (p, q)
    var_order: tuple = eqx.field(static=True)    # (p', q')
    var_model: type = eqx.field(static=True)     # GARCH-family class
    residual_dist: Univariate = eqx.field(static=True)

    # ---- traced fitted parameters --------------------------------------
    # Mean
    phi: Optional[Array] = None
    theta: Optional[Array] = None
    c: Optional[Array] = None
    # Variance (vanilla GARCH for v1)
    omega: Optional[Array] = None
    alpha: Optional[Array] = None
    beta: Optional[Array] = None
    # Residual + terminal state + diagnostics
    residual_params: Optional[dict] = None
    terminal_state: Optional[ArmaGarchTerminalState] = None
    loglikelihood_: Optional[Array] = None
    aic_: Optional[Array] = None
    bic_: Optional[Array] = None
    n_train_: Optional[int] = None

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
        phi=None,
        theta=None,
        c=None,
        omega=None,
        alpha=None,
        beta=None,
        residual_params=None,
        terminal_state: Optional[ArmaGarchTerminalState] = None,
        loglikelihood_=None,
        aic_=None,
        bic_=None,
        n_train_: Optional[int] = None,
    ):
        if var_model is not GARCH:
            raise NotImplementedError(
                f"ArmaGarch v1 supports only vanilla `GARCH` as the variance "
                f"equation; got {var_model.__name__!r}.  Extensions to "
                "IGARCH / GJR_GARCH / EGARCH / TGARCH / QGARCH are planned "
                "in subsequent releases — track the project plan for "
                "details."
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
        self.c = (
            jnp.asarray(c, dtype=float).reshape(()) if c is not None else None
        )
        self.omega = (
            jnp.asarray(omega, dtype=float).reshape(())
            if omega is not None else None
        )
        self.alpha = (
            jnp.asarray(alpha, dtype=float).reshape(-1)
            if alpha is not None else None
        )
        self.beta = (
            jnp.asarray(beta, dtype=float).reshape(-1)
            if beta is not None else None
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

    # ------------------------------------------------------------------
    # Order accessors
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

    # ------------------------------------------------------------------
    # params property
    # ------------------------------------------------------------------
    @property
    def _stored_params(self) -> Optional[dict]:
        r"""Canonical params dict.

        Schema:

        ``{
            "phi":       (p,),
            "theta":     (q,),
            "c":         (),
            "omega":     (),
            "alpha":     (p',),
            "beta":      (q',),
            "residual":  {<shape-only dict>},
        }``
        """
        if (
            self.phi is None or self.theta is None or self.c is None
            or self.omega is None or self.alpha is None or self.beta is None
            or self.residual_params is None
        ):
            return None
        return {
            "phi": self.phi,
            "theta": self.theta,
            "c": self.c,
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "residual": dict(self.residual_params),
        }

    @property
    def n_params(self) -> int:
        r"""Number of free fitted parameters."""
        wrapper = StandardisedResidual(self.residual_dist)
        return (
            self.p + self.q + 1               # phi + theta + c
            + 1 + self.p_var + self.q_var     # omega + alpha + beta
            + wrapper.n_shape_params           # residual shape
        )

    def _wrapper(self) -> StandardisedResidual:
        return StandardisedResidual(self.residual_dist)

    # ------------------------------------------------------------------
    # Pack / unpack
    # ------------------------------------------------------------------
    def _pack_x0(
        self,
        params_dict: dict,
        wrapper: StandardisedResidual,
    ) -> Array:
        r"""Pack a constrained ``params_dict`` to a flat unconstrained
        vector.

        Layout: ``[raw_phi (p,), raw_theta (q,), c (1,), raw_omega
        (1,), raw_persistence (1,), raw_weights (p'+q',),
        raw_residual_shape (n_shape,)]``.
        """
        phi = jnp.asarray(params_dict["phi"], dtype=float).reshape(-1)
        theta = jnp.asarray(params_dict["theta"], dtype=float).reshape(-1)
        c = jnp.asarray(params_dict["c"], dtype=float).reshape(())
        omega = jnp.asarray(params_dict["omega"], dtype=float).reshape(())
        alpha = jnp.asarray(params_dict["alpha"], dtype=float).reshape(-1)
        beta = jnp.asarray(params_dict["beta"], dtype=float).reshape(-1)
        residual = params_dict.get("residual", {}) or {}

        raw_phi = ar_to_raw(phi) if self.p > 0 else jnp.zeros((0,), dtype=float)
        raw_theta = ma_to_raw(theta) if self.q > 0 else jnp.zeros((0,), dtype=float)
        raw_omega = positive_to_raw(jnp.maximum(omega, _SIGMA_FLOOR))
        raw_persistence, raw_weights = garch_unsimplex(alpha, beta)
        raw_residual = wrapper.shape_params_to_array(residual)
        return jnp.concatenate(
            [
                raw_phi,
                raw_theta,
                c.reshape((1,)),
                raw_omega.reshape((1,)),
                raw_persistence.reshape((1,)),
                raw_weights,
                raw_residual,
            ]
        )

    def _unpack_raw(
        self,
        raw: Array,
        wrapper: StandardisedResidual,
    ) -> tuple[Array, Array, Array, Array, Array, Array, dict]:
        r"""Inverse of :meth:`_pack_x0`.

        Returns ``(phi, theta, c, omega, alpha, beta,
        residual_shape_dict)``.
        """
        idx = 0
        raw_phi = raw[idx : idx + self.p]
        idx += self.p
        raw_theta = raw[idx : idx + self.q]
        idx += self.q
        c = raw[idx]
        idx += 1
        raw_omega = raw[idx]
        idx += 1
        raw_persistence = raw[idx]
        idx += 1
        raw_weights = raw[idx : idx + self.p_var + self.q_var]
        idx += self.p_var + self.q_var
        raw_residual = raw[idx : idx + wrapper.n_shape_params]

        phi = raw_to_ar(raw_phi) if self.p > 0 else jnp.zeros((0,), dtype=float)
        theta = raw_to_ma(raw_theta) if self.q > 0 else jnp.zeros((0,), dtype=float)
        omega = raw_to_positive(raw_omega)
        alpha, beta = garch_simplex(raw_persistence, raw_weights, p=self.p_var)
        residual = wrapper.shape_params_from_array(raw_residual)
        return phi, theta, c, omega, alpha, beta, residual

    # ------------------------------------------------------------------
    # Recursion (chained ARMA → GARCH)
    # ------------------------------------------------------------------
    def _run_recursion(
        self,
        y: Array,
        phi: Array, theta: Array, c: Array,
        omega: Array, alpha: Array, beta: Array,
        init_y_lags: Array, init_eps_lags: Array,
        init_eps_sq_lags: Array, init_var_lags: Array,
    ) -> tuple[Array, Array, Array, ArmaGarchTerminalState]:
        r"""Run the chained ARMA → GARCH recursion.

        Returns ``(mu_seq, eps_seq, var_seq, terminal_state)``.

        Implementation note: we run the two recursions sequentially.
        ARMA produces the innovation series :math:`\varepsilon_t` from
        :math:`y_t`; GARCH consumes that series and produces
        :math:`\sigma^2_t`.  Two ``lax.scan`` calls — JAX's
        compiler handles the dataflow fine, and keeping them
        separate makes warm-start surgery (e.g. holding the mean
        params fixed and refitting only the variance half) trivial.
        """
        mu_seq, eps_seq, arma_terminal = run_arma(
            y=y, phi=phi, theta=theta, c=c,
            init_y_lags=init_y_lags, init_eps_lags=init_eps_lags,
        )
        var_seq, var_terminal = run_garch(
            eps=eps_seq, omega=omega, alpha=alpha, beta=beta,
            init_eps_sq_lags=init_eps_sq_lags, init_var_lags=init_var_lags,
        )
        terminal_state = ArmaGarchTerminalState(
            y_lags=arma_terminal[0],
            eps_lags=arma_terminal[1],
            eps_sq_lags=var_terminal[0],
            var_lags=var_terminal[1],
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
    ) -> tuple[Array, Array, Array, Array]:
        r"""Pre-sample state for both halves of the recursion.

        ARMA half: ``y_lags`` from the leading-window mean of ``y``
        (or zeros under ``mode="zero"``); ``eps_lags`` zero (the
        marginal mean of standardised innovations is zero under any
        whitelisted residual law).

        GARCH half: ``eps_sq_lags`` and ``var_lags`` both set to the
        leading-window EWMA / sample variance of ``y`` — a
        slightly biased anchor relative to the post-ARMA innovation
        variance, but the bias washes out within the first
        ``max(p', q')`` recursion steps.
        """
        init_y_lags, init_eps_lags = arma_pre_sample_state(
            y, p=self.p, q=self.q, mode=mode, backcast_length=backcast_length,
        )
        init_eps_sq_lags, init_var_lags = garch_pre_sample_state(
            y, p=self.p_var, q=self.q_var,
            mode=mode, backcast_length=backcast_length,
        )
        return init_y_lags, init_eps_lags, init_eps_sq_lags, init_var_lags

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
        r"""Cold-start params dict for the joint composite.

        Strategy: use the standalone analytic init for each half
        with the provided ``init`` mode.  The standalone ARMA init
        produces ``(phi, theta, c)``; we discard its
        ``sigma_eps`` (the GARCH recursion replaces it).  The
        standalone GARCH init runs on the AR-filtered residuals
        (computed by re-applying the seed φ to ``y``) so the
        ``(omega, alpha, beta)`` prior is consistent with the
        actual innovation series the variance equation will
        consume.
        """
        arma_seed = init_arma_params(y, p=self.p, q=self.q, mode=init)
        # Compute AR-filtered residuals using the seed φ — gives a
        # reasonable proxy for the innovation series the GARCH half
        # will see at fit time.
        if self.p > 0:
            y_arr = jnp.asarray(y, dtype=float).reshape(-1)
            y_centred = y_arr - jnp.mean(y_arr)
            n = int(y_arr.shape[0])
            r = jnp.zeros((n,), dtype=float)
            for i in range(self.p):
                r = r.at[i + 1:].add(-arma_seed["phi"][i] * y_centred[: n - (i + 1)])
            r = r + y_centred  # residuals = y - sum(phi[i]*y_{t-i-1})
            eps_proxy = r
        else:
            eps_proxy = jnp.asarray(y, dtype=float).reshape(-1) - jnp.mean(jnp.asarray(y, dtype=float))
        garch_seed = init_garch_params(
            eps_proxy, p=self.p_var, q=self.q_var, mode=init,
            backcast_length=backcast_length,
        )
        return {
            "phi": arma_seed["phi"],
            "theta": arma_seed["theta"],
            "c": arma_seed["c"],
            "omega": garch_seed["omega"],
            "alpha": garch_seed["alpha"],
            "beta": garch_seed["beta"],
            "residual": wrapper.example_shape_params(),
        }

    # ------------------------------------------------------------------
    # Fit objective
    # ------------------------------------------------------------------
    def _make_objective(self, wrapper: StandardisedResidual):
        r"""Build the joint negative log-likelihood objective.

        :math:`\ell = \sum_t \log f_z(\varepsilon_t / \sigma_t)
        - \log \sigma_t`, with non-finite contributions masked to a
        finite penalty.
        """
        def objective(
            raw: Array,
            y: Array,
            init_y_lags: Array,
            init_eps_lags: Array,
            init_eps_sq_lags: Array,
            init_var_lags: Array,
        ) -> Array:
            phi, theta, c, omega, alpha, beta, residual_shape = self._unpack_raw(
                raw, wrapper,
            )
            _, eps_seq, var_seq, _ = self._run_recursion(
                y, phi, theta, c, omega, alpha, beta,
                init_y_lags, init_eps_lags,
                init_eps_sq_lags, init_var_lags,
            )
            sigma_seq = jnp.sqrt(jnp.maximum(var_seq, _VAR_FLOOR))
            z = eps_seq / sigma_seq
            logpdf = wrapper.logpdf(z, residual_shape) - jnp.log(sigma_seq)
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

        Single MLE over the combined parameter vector — see module
        docstring for the conditional log-likelihood and asymptotic-
        theory rationale.

        Args:
            y: shape ``(n,)`` — observed return series.
            init: One of ``"analytical"`` (Yule-Walker + Innovations
                Algorithm for the mean half, industry α=0.05, β=0.9
                prior for the variance half — default),
                ``"backcast"``, ``"sample"``, or ``"warm"``.
            init_params: Warm-start parameter dict matching the
                schema returned by :attr:`params`.
            backcast_length: Window for the EWMA backcast under
                ``init="backcast"``.
            maxiter: Adam iterations.  Default 300 — slightly more
                than the standalone defaults because the joint
                landscape has more directions to walk.
            lr: Adam learning rate.
            name: Optional custom name for the fitted instance.

        Returns:
            A fitted ``ArmaGarch`` instance.
        """
        self._check_method(init)
        wrapper = StandardisedResidual(self.residual_dist)
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
            for key in ("phi", "theta", "c", "omega", "alpha", "beta", "residual"):
                if key not in cold:
                    raise KeyError(
                        f"Warm-start init_params missing required key {key!r}."
                    )
        else:
            cold = self._build_cold_start(
                y_arr, wrapper, init=init, backcast_length=backcast_length,
            )

        x0 = self._pack_x0(cold, wrapper)

        _state_mode = "sample" if init == "sample" else "backcast"
        (
            init_y_lags, init_eps_lags,
            init_eps_sq_lags, init_var_lags,
        ) = self._build_initial_state(
            y_arr, mode=_state_mode, backcast_length=backcast_length,
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
            init_eps_sq_lags=init_eps_sq_lags,
            init_var_lags=init_var_lags,
            lr=lr,
            maxiter=maxiter,
        )
        x_opt = res["x"]
        phi, theta, c, omega, alpha, beta, residual = self._unpack_raw(
            x_opt, wrapper,
        )

        _, _, _, terminal = self._run_recursion(
            y_arr, phi, theta, c, omega, alpha, beta,
            init_y_lags, init_eps_lags,
            init_eps_sq_lags, init_var_lags,
        )
        nll = objective(
            x_opt, y_arr,
            init_y_lags, init_eps_lags,
            init_eps_sq_lags, init_var_lags,
        )
        loglike = -nll * n
        n_params_total = self.n_params
        aic = 2.0 * n_params_total - 2.0 * loglike
        bic = (
            n_params_total * jnp.log(jnp.asarray(n, dtype=float))
            - 2.0 * loglike
        )

        if name is None:
            name = (
                f"FittedArmaGarch(({self.p},{self.q})x"
                f"({self.p_var},{self.q_var}))-{self.residual_dist.name}"
            )
        return type(self)(
            mean_order=self.mean_order,
            var_model=self.var_model,
            var_order=self.var_order,
            residual_dist=self.residual_dist,
            name=name,
            phi=phi, theta=theta, c=c,
            omega=omega, alpha=alpha, beta=beta,
            residual_params=residual,
            terminal_state=terminal,
            loglikelihood_=loglike, aic_=aic, bic_=bic, n_train_=n,
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
        y_arr, (init_y_lags, init_eps_lags, init_eps_sq_lags, init_var_lags) = (
            self._recursion_inputs(y, init, backcast_length)
        )
        mu_seq, _, _, _ = self._run_recursion(
            y_arr, self.phi, self.theta, self.c,
            self.omega, self.alpha, self.beta,
            init_y_lags, init_eps_lags,
            init_eps_sq_lags, init_var_lags,
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
        y_arr, (init_y_lags, init_eps_lags, init_eps_sq_lags, init_var_lags) = (
            self._recursion_inputs(y, init, backcast_length)
        )
        _, _, var_seq, _ = self._run_recursion(
            y_arr, self.phi, self.theta, self.c,
            self.omega, self.alpha, self.beta,
            init_y_lags, init_eps_lags,
            init_eps_sq_lags, init_var_lags,
        )
        return var_seq

    def residuals(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> dict:
        r"""Returns ``{"mean_residuals": ε_t, "standardised_residuals":
        z_t}`` per plan §"Residuals API" — both halves in one
        recursion pass."""
        self._require_fitted()
        y_arr, (init_y_lags, init_eps_lags, init_eps_sq_lags, init_var_lags) = (
            self._recursion_inputs(y, init, backcast_length)
        )
        _, eps_seq, var_seq, _ = self._run_recursion(
            y_arr, self.phi, self.theta, self.c,
            self.omega, self.alpha, self.beta,
            init_y_lags, init_eps_lags,
            init_eps_sq_lags, init_var_lags,
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
        y_arr, (init_y_lags, init_eps_lags, init_eps_sq_lags, init_var_lags) = (
            self._recursion_inputs(y, init, backcast_length)
        )
        _, _, _, terminal = self._run_recursion(
            y_arr, self.phi, self.theta, self.c,
            self.omega, self.alpha, self.beta,
            init_y_lags, init_eps_lags,
            init_eps_sq_lags, init_var_lags,
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
        y_arr, (init_y_lags, init_eps_lags, init_eps_sq_lags, init_var_lags) = (
            self._recursion_inputs(y, init, backcast_length)
        )
        _, eps_seq, var_seq, _ = self._run_recursion(
            y_arr, self.phi, self.theta, self.c,
            self.omega, self.alpha, self.beta,
            init_y_lags, init_eps_lags,
            init_eps_sq_lags, init_var_lags,
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

        Returns the ARMA-side stationarity / AR-root moduli plus the
        GARCH-side persistence / unconditional variance / half-life,
        unified into a single dict.
        """
        self._require_fitted()
        from copulax._src.timeseries._stationarity import (
            ar_is_stationary, ar_polynomial_roots,
        )
        # Mean
        ar_roots = ar_polynomial_roots(self.phi)
        ma_roots = (
            ar_polynomial_roots(self.theta)
            if self.q > 0 else jnp.zeros((0,), dtype=jnp.complex64)
        )
        mean_is_stat = ar_is_stationary(self.phi)
        mean_is_inv = (
            ar_is_stationary(self.theta) if self.q > 0 else jnp.asarray(True)
        )
        ar_factor = 1.0 - jnp.sum(self.phi) if self.p > 0 else 1.0
        ar_factor_safe = jnp.where(jnp.abs(ar_factor) < 1e-12, 1e-12, ar_factor)
        unconditional_mean = self.c / ar_factor_safe
        # Variance (vanilla GARCH)
        var_persistence = jnp.sum(self.alpha) + jnp.sum(self.beta)
        var_is_stat = var_persistence < 1.0
        var_denom = jnp.where(var_is_stat, 1.0 - var_persistence, _VAR_FLOOR)
        unconditional_variance = jnp.where(
            var_is_stat, self.omega / var_denom, jnp.inf,
        )
        log_pers = jnp.log(jnp.maximum(var_persistence, _VAR_FLOOR))
        half_life = jnp.where(
            jnp.logical_and(var_is_stat, var_persistence > 0.0),
            jnp.log(0.5) / log_pers,
            jnp.inf,
        )
        return {
            "unconditional_mean": unconditional_mean,
            "unconditional_variance": unconditional_variance,
            "var_persistence": var_persistence,
            "half_life": half_life,
            "mean_is_stationary": mean_is_stat,
            "mean_is_invertible": mean_is_inv,
            "var_is_stationary": var_is_stat,
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
        r"""``h``-step-ahead conditional moments.

        Analytical: rolls the ARMA mean recursion forward (with
        future innovations replaced by their zero conditional
        expectation) and the GARCH variance recursion forward (with
        future :math:`\varepsilon^2` replaced by their conditional
        expectation :math:`\sigma^2`) — both pieces have closed
        forms.

        Simulation: full Monte Carlo via :meth:`rvs` for any
        residual law.
        """
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

        if method == "analytical":
            # Mean rollout: ε_t is replaced by 0 for unobserved future.
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
            # Variance rollout: ε² replaced by σ² for unobserved future.
            var_path = []
            eps_sq_lags = state.eps_sq_lags
            var_lags = state.var_lags
            for _ in range(h):
                ar_term = jnp.dot(self.alpha, eps_sq_lags) if self.p_var > 0 else 0.0
                ma_term = jnp.dot(self.beta, var_lags) if self.q_var > 0 else 0.0
                var_t = self.omega + ar_term + ma_term
                var_t = jnp.maximum(var_t, _VAR_FLOOR)
                var_path.append(var_t)
                if self.p_var > 0:
                    eps_sq_lags = jnp.concatenate(
                        [var_t.reshape((1,)), eps_sq_lags[:-1]]
                    )
                if self.q_var > 0:
                    var_lags = jnp.concatenate(
                        [var_t.reshape((1,)), var_lags[:-1]]
                    )
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
        c = self.c
        phi = self.phi
        theta = self.theta
        omega = self.omega
        alpha = self.alpha
        beta = self.beta

        def step(carry, z_t):
            y_lags, eps_lags, eps_sq_lags, var_lags = carry
            # Mean
            ar_term = jnp.dot(phi, y_lags) if self.p > 0 else 0.0
            ma_term = jnp.dot(theta, eps_lags) if self.q > 0 else 0.0
            mu = c + ar_term + ma_term
            # Variance
            v_ar = jnp.dot(alpha, eps_sq_lags) if self.p_var > 0 else 0.0
            v_ma = jnp.dot(beta, var_lags) if self.q_var > 0 else 0.0
            var_t = omega + v_ar + v_ma
            var_t = jnp.maximum(var_t, _VAR_FLOOR)
            sigma_t = jnp.sqrt(var_t)
            eps_t = sigma_t * z_t
            y_t = mu + eps_t
            new_y_lags = (
                jnp.concatenate([y_t.reshape((1,)), y_lags[:-1]])
                if self.p > 0 else y_lags
            )
            new_eps_lags = (
                jnp.concatenate([eps_t.reshape((1,)), eps_lags[:-1]])
                if self.q > 0 else eps_lags
            )
            new_eps_sq_lags = (
                jnp.concatenate([(eps_t * eps_t).reshape((1,)), eps_sq_lags[:-1]])
                if self.p_var > 0 else eps_sq_lags
            )
            new_var_lags = (
                jnp.concatenate([var_t.reshape((1,)), var_lags[:-1]])
                if self.q_var > 0 else var_lags
            )
            return (
                new_y_lags, new_eps_lags, new_eps_sq_lags, new_var_lags,
            ), y_t

        init_carry = (
            state.y_lags, state.eps_lags,
            state.eps_sq_lags, state.var_lags,
        )
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
        r"""Simulate synthetic level paths from the joint model."""
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
        r"""The fitted standardised residual distribution as a regular
        :class:`Univariate` instance."""
        self._require_fitted()
        return self._wrapper().to_distribution(
            self.residual_params,
            name=f"{self.residual_dist.name}-stdresid-{self.name}",
        )
