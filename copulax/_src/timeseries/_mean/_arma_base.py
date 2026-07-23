"""Shared scaffolding for AR(p) / MA(q) / ARMA(p, q) mean-equation models.

The :class:`ARMABase` class implements the full mean-model contract
defined on :class:`copulax._src.timeseries._base.TimeSeriesModel`:

* :meth:`fit` — Adam-driven negative-log-likelihood minimisation
  through :func:`copulax._src.optimize.projected_gradient` over an
  unconstrained parameter vector built from the canonical AR / MA
  reflection-coefficient reparameterisation, ``softplus`` for the
  positive innovation scale, and the residual law's free shape
  parameters.  JIT-compatible end-to-end so rolling-window calls
  reuse the compiled trace.
* :meth:`conditional_mean` — one-step-ahead :math:`\\mu_t` over a
  supplied series via :func:`run_arma`.
* :meth:`conditional_variance` — constant trajectory equal to the
  unconditional innovation variance under the fitted scale and
  standardised residual law (mean models do not parameterise
  heteroskedasticity; pair with a :class:`VarianceModel` when that
  matters).
* :meth:`residuals` — innovation residuals
  :math:`\\varepsilon_t = y_t - \\mu_t`.
* :meth:`stats` — analytic, parameter-only statistics: unconditional
  mean / variance / persistence / characteristic-root moduli /
  stationarity / invertibility flags.
* :meth:`forecast` — analytical h-step ARMA forecast (closed form
  in :math:`\\mu_t`); simulation path support via :meth:`rvs`.
* :meth:`rvs` — synthesise paths from the fitted model with full
  shared-noise / inverse-transform / antithetic support via the
  optional ``u`` kwarg.
* :meth:`loglikelihood`, :meth:`aic`, :meth:`bic` — scalar
  diagnostics that take an optional series; default to the stored
  fit-time values.

The three concrete singletons (``ar``, ``ma``, ``arma``) inherit
from this base; they differ only in which orders are pinned to zero
and in the ``fit`` signature exposed to the user.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src._utils import _resolve_key
from copulax._src.optimize import projected_gradient
from copulax._src.timeseries._base import MeanModel, TerminalState
from copulax._src.timeseries._diagnostics import (
    acf as _diag_acf,
    arch_lm as _diag_arch_lm,
    ljung_box as _diag_ljung_box,
    pacf as _diag_pacf,
)
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
    positive_to_raw,
    raw_to_ar,
    raw_to_ma,
    raw_to_positive,
)
from copulax._src.timeseries._summary import (
    ParamSection,
    build_diagnostic_rows,
    display_residual_name,
    format_summary,
    iter_param_rows,
    residual_section,
)
from copulax._src.timeseries._unit_root import adf as _diag_adf, kpss as _diag_kpss


_VAR_FLOOR: float = 1e-12
_SIGMA_FLOOR: float = 1e-6


###############################################################################
# Per-family terminal state
###############################################################################
class ARMATerminalState(TerminalState):
    r"""Constant-size carry state for ARMA(p, q) ``forecast(h)``.

    Stores the last ``p`` returns and the last ``q`` innovation
    residuals — exactly what :func:`run_arma` consumes as its
    initial ``(y_lags, eps_lags)``.  ``O(max(p, q))`` in size, so
    the fitted-model save file stays bounded by the order, not by
    the training-series length.
    """
    y_lags: Array
    eps_lags: Array


###############################################################################
# ARMABase
###############################################################################
class ARMABase(MeanModel):
    r"""Concrete ARMA(p, q) mean-equation model.

    See module docstring for the contract this class implements.
    Concrete user-facing singletons (``ar``, ``ma``, ``arma``)
    inherit from this base and only override the public ``fit``
    signature (e.g. dropping the ``q`` kwarg for ``ar``).

    Notes:
        * ``p``, ``q`` are static fields — they parameterise the
          compiled fit graph and cannot change at runtime.
        * ``residual_dist`` is a traced field carrying a
          :class:`Univariate` instance: pre-fit it stores the user
          template (defaults to ``normal``); post-fit it stores
          the fully-fitted standardised (mean=0, var=1) residual
          distribution, so ``fit.residual_dist.params`` /
          ``fit.residual_dist.sample(key, ...)`` etc. work
          directly without any wrapper indirection.  The residual
          law's *type* is what triggers JIT recompilation across
          fits; same-type-with-different-parameters reuses the
          compiled graph.
        * Fitted parameters (``phi``, ``theta``, ``mu``,
          ``sigma_eps``) and the residual law's free shape
          parameters (``residual_params``) are traced ``Array``
          leaves — they flow through ``jax.grad`` cleanly.  ``mu``
          is the **unconditional mean** of the process under the
          centred-form recursion :math:`y_t = \\mu + \\sum_i \\phi_i
          (y_{t-i} - \\mu) + \\sum_j \\theta_j \\varepsilon_{t-j} +
          \\varepsilon_t`.
        * ``residual_diagnostics_`` is the canonical bundle of
          every fit-time scalar / array / test result (model-fit
          scalars, ACF/PACF arrays, hypothesis-test dicts) — see
          the field's class-level docstring for the schema.
          Populated by :meth:`fit`; ``None`` for unfitted
          instances.
        * ``terminal_state`` carries the last ``p`` returns and
          ``q`` innovations; ``forecast(h)`` rolls forward from
          there by default.
    """

    # ---- static configuration -------------------------------------------
    p: int = eqx.field(static=True)
    q: int = eqx.field(static=True)

    # ---- residual distribution (traced PyTree) -------------------------
    # Pre-fit: the user-supplied template singleton (defaults to
    # ``normal``); only its *type* drives JIT recompilation across
    # fits.  Post-fit: the fully-fitted standardised (mean=0, var=1)
    # residual distribution returned by
    # ``StandardisedResidual.to_distribution``, so
    # ``fit.residual_dist.params``, ``fit.residual_dist.sample(...)``,
    # ``fit.residual_dist.logpdf(...)`` all work directly.  Same-type
    # parameter changes do not trigger recompilation; different
    # residual types do (different PyTree structure).
    residual_dist: Optional[Univariate] = None

    # ---- traced fitted parameters ---------------------------------------
    phi: Optional[Array] = None
    theta: Optional[Array] = None
    mu: Optional[Array] = None
    sigma_eps: Optional[Array] = None
    residual_params: Optional[dict] = None

    # ---- per-fit terminal carry + sample size --------------------------
    terminal_state: Optional[ARMATerminalState] = None
    n_train_: Optional[int] = None

    # ---- post-fit standard errors (observed-Hessian / "classic") -------
    cov_matrix_: Optional[Array] = None
    standard_errors_: Optional[dict] = None

    # ---- post-fit residual diagnostics (cached default-arg results) ----
    # Single canonical bundle of every fit-time scalar / array / test
    # result on the standardised training residuals.  Schema:
    #
    # * ``"loglikelihood"``, ``"aic"``, ``"bic"`` — model-fit scalars
    #   (replaces the previous top-level ``loglikelihood_`` / ``aic_``
    #   / ``bic_`` fields).
    # * ``"acf"``, ``"pacf"`` — shape ``(21,)`` autocorrelation arrays
    #   at the default ``lags=20`` (PACF uses ``method="yule_walker"``).
    # * ``"ljung_box"``, ``"ljung_box_sq"``, ``"arch_lm"``, ``"adf"``,
    #   ``"kpss"`` — standardised hypothesis-test result dicts (see
    #   :mod:`copulax._src.timeseries._diagnostics` /
    #   :mod:`copulax._src.timeseries._unit_root` for the schema).
    #
    # ``y=None`` defaults across :meth:`loglikelihood`, :meth:`aic`,
    # :meth:`bic`, :meth:`acf`, :meth:`pacf`, :meth:`ljung_box`,
    # :meth:`arch_lm`, :meth:`adf_residuals`, :meth:`kpss_residuals`,
    # :meth:`plot_acf`, :meth:`plot_pacf` all read from this dict.
    residual_diagnostics_: Optional[dict] = None

    # ---- supported init-mode strings (mirrors Distribution._supported_methods)
    _supported_methods: ClassVar[frozenset] = frozenset(
        {"analytical", "backcast", "sample", "warm"}
    )

    def __init__(
        self,
        name: str = "ARMA",
        *,
        p: int = 0,
        q: int = 0,
        residual_dist: Univariate = None,
        phi: Optional[ArrayLike] = None,
        theta: Optional[ArrayLike] = None,
        mu: Optional[ArrayLike] = None,
        sigma_eps: Optional[ArrayLike] = None,
        residual_params: Optional[dict] = None,
        terminal_state: Optional[ARMATerminalState] = None,
        n_train_: Optional[int] = None,
        cov_matrix_: Optional[ArrayLike] = None,
        standard_errors_: Optional[dict] = None,
        residual_diagnostics_: Optional[dict] = None,
    ):
        super().__init__(name=name)
        self.p = int(p)
        self.q = int(q)
        # Default to ``normal`` if no residual_dist is supplied so the
        # unfitted singleton has a well-defined static type — picking
        # any other allowed law during fit triggers a fresh JIT trace,
        # which is the documented behaviour.
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
        self.mu = (
            jnp.asarray(mu, dtype=float).reshape(()) if mu is not None else None
        )
        self.sigma_eps = (
            jnp.asarray(sigma_eps, dtype=float).reshape(())
            if sigma_eps is not None else None
        )
        self.residual_params = residual_params
        self.terminal_state = terminal_state
        self.n_train_ = int(n_train_) if n_train_ is not None else None
        self.cov_matrix_ = (
            jnp.asarray(cov_matrix_, dtype=float)
            if cov_matrix_ is not None else None
        )
        self.standard_errors_ = (
            dict(standard_errors_) if standard_errors_ is not None else None
        )
        self.residual_diagnostics_ = (
            dict(residual_diagnostics_)
            if residual_diagnostics_ is not None else None
        )

    # ------------------------------------------------------------------
    # params property
    # ------------------------------------------------------------------
    @property
    def _stored_params(self) -> Optional[dict]:
        """Canonical parameter dict, or ``None`` for an unfitted instance.

        Schema:

        ``{
            "phi":        (p,),
            "theta":      (q,),
            "mu":         (),
            "sigma_eps":  (),
            "residual":   {<shape-only dict matching the
                           StandardisedResidual schema>},
        }``

        ``mu`` is the unconditional mean of the process under the
        centred-form recursion (Box-Jenkins / Hamilton).
        """
        if (
            self.phi is None or self.theta is None or self.mu is None
            or self.sigma_eps is None or self.residual_params is None
        ):
            return None
        return {
            "phi": self.phi,
            "theta": self.theta,
            "mu": self.mu,
            "sigma_eps": self.sigma_eps,
            "residual": dict(self.residual_params),
        }

    @property
    def n_params(self) -> int:
        r"""Number of free fitted parameters in the model."""
        wrapper = StandardisedResidual(self.residual_dist)
        return self.p + self.q + 1 + 1 + wrapper.n_shape_params

    # ------------------------------------------------------------------
    # Recursion helpers
    # ------------------------------------------------------------------
    def _wrapper(self) -> StandardisedResidual:
        """Cache-friendly accessor for the standardised residual wrapper."""
        return StandardisedResidual(self.residual_dist)

    def _build_initial_state(
        self,
        y: Array,
        mode: str,
        backcast_length: Optional[int],
    ) -> tuple[Array, Array]:
        """Pre-sample ``(y_lags, eps_lags)`` for the recursion's initial carry."""
        return arma_pre_sample_state(
            y, p=self.p, q=self.q, mode=mode,
            backcast_length=backcast_length,
        )

    def _run_recursion(
        self,
        y: Array,
        phi: Array,
        theta: Array,
        mu: Array,
        init_y_lags: Array,
        init_eps_lags: Array,
    ) -> tuple[Array, Array, ARMATerminalState]:
        """Run :func:`run_arma` and wrap the terminal carry."""
        mu_seq, eps_seq, terminal = run_arma(
            y=y, phi=phi, theta=theta, mu=mu,
            init_y_lags=init_y_lags, init_eps_lags=init_eps_lags,
        )
        terminal_state = ARMATerminalState(
            y_lags=terminal[0], eps_lags=terminal[1],
        )
        return mu_seq, eps_seq, terminal_state

    # ------------------------------------------------------------------
    # Fit-objective machinery
    # ------------------------------------------------------------------
    def _pack_x0(
        self,
        params_dict: dict,
        wrapper: StandardisedResidual,
    ) -> Array:
        r"""Pack a constrained ``params_dict`` into the unconstrained
        flat optimiser-state vector.

        Layout: ``[raw_phi (p,), raw_theta (q,), mu (), raw_sigma_eps (),
        raw_residual_shape (n_shape_params)]``.
        """
        phi = jnp.asarray(params_dict["phi"], dtype=float).reshape(-1)
        theta = jnp.asarray(params_dict["theta"], dtype=float).reshape(-1)
        mu = jnp.asarray(params_dict["mu"], dtype=float).reshape(())
        sigma_eps = jnp.asarray(params_dict["sigma_eps"], dtype=float).reshape(())
        residual = params_dict.get("residual", {}) or {}
        raw_phi = ar_to_raw(phi) if self.p > 0 else jnp.zeros((0,), dtype=float)
        raw_theta = ma_to_raw(theta) if self.q > 0 else jnp.zeros((0,), dtype=float)
        raw_sigma = positive_to_raw(jnp.maximum(sigma_eps, _SIGMA_FLOOR))
        raw_residual = wrapper.shape_params_to_array(residual)
        return jnp.concatenate(
            [
                raw_phi,
                raw_theta,
                mu.reshape((1,)),
                raw_sigma.reshape((1,)),
                raw_residual,
            ]
        )

    def _unpack_raw(
        self,
        raw: Array,
        wrapper: StandardisedResidual,
    ) -> tuple[Array, Array, Array, Array, dict]:
        r"""Inverse of :meth:`_pack_x0`.

        Returns ``(phi, theta, mu, sigma_eps, residual_shape_dict)`` —
        every component already pushed through its bound-enforcing
        reparameterisation.  ``mu`` is unconstrained (the
        unconditional mean is real-valued with no positivity
        requirement).
        """
        idx = 0
        raw_phi = raw[idx : idx + self.p]
        idx += self.p
        raw_theta = raw[idx : idx + self.q]
        idx += self.q
        mu = raw[idx]
        idx += 1
        raw_sigma = raw[idx]
        idx += 1
        raw_residual = raw[idx : idx + wrapper.n_shape_params]

        phi = raw_to_ar(raw_phi) if self.p > 0 else jnp.zeros((0,), dtype=float)
        theta = raw_to_ma(raw_theta) if self.q > 0 else jnp.zeros((0,), dtype=float)
        sigma_eps = raw_to_positive(raw_sigma)
        residual = wrapper.shape_params_from_array(raw_residual)
        return phi, theta, mu, sigma_eps, residual

    def _make_objective(
        self, wrapper: StandardisedResidual,
    ):
        r"""Build a closure over the standardised-residual wrapper that
        :func:`projected_gradient` can JIT-compile cleanly.

        The wrapper is a plain Python object (not a JAX type) so it
        cannot flow through the optimiser as a traced argument; we
        capture it via closure so the resulting function is
        positional-only over JAX-compatible arguments and the
        wrapper is part of the static trace identity (the wrapper's
        hash keys on ``type(base_dist)``, so rolling-window calls
        with the same residual law reuse a single compiled trace).

        Returns:
            Callable with signature
            ``(raw, y, init_y_lags, init_eps_lags) -> scalar`` —
            the negative mean log-likelihood under the ARMA(p, q) +
            standardised-residual specification:

            .. math::

                \\ell(\\theta; y) = \\sum_t
                    \\bigl[
                        \\log f_z\\!\\bigl((y_t - \\mu_t)
                                           / \\sigma_\\varepsilon\\bigr)
                        - \\log \\sigma_\\varepsilon
                    \\bigr].

            Non-finite log-density contributions are masked to a
            finite penalty so the gradient points away from
            degenerate parameter regions rather than poisoning the
            trace.
        """
        def objective(
            raw: Array,
            y: Array,
            init_y_lags: Array,
            init_eps_lags: Array,
        ) -> Array:
            phi, theta, mu, sigma_eps, residual_shape = self._unpack_raw(
                raw, wrapper,
            )
            sigma_eps_safe = jnp.maximum(sigma_eps, _SIGMA_FLOOR)
            _, eps_seq, _ = run_arma(
                y=y, phi=phi, theta=theta, mu=mu,
                init_y_lags=init_y_lags, init_eps_lags=init_eps_lags,
            )
            z = eps_seq / sigma_eps_safe
            logpdf = wrapper.logpdf(z, residual_shape) - jnp.log(sigma_eps_safe)
            finite = jnp.isfinite(logpdf)
            safe_logpdf = jnp.where(finite, logpdf, 0.0)
            invalid_penalty = 1e6 * (~finite).mean()
            return -safe_logpdf.mean() + invalid_penalty

        return objective

    # ------------------------------------------------------------------
    # Public fit
    # ------------------------------------------------------------------
    def _fit_internal(
        self,
        y: Array,
        wrapper: StandardisedResidual,
        init: str,
        init_params: Optional[dict],
        backcast_length: Optional[int],
        maxiter: int,
        lr: float,
    ) -> tuple[dict, ARMATerminalState, Array]:
        r"""Optimisation core.  Returns
        ``(params_dict, terminal_state, neg_log_likelihood_at_optimum)``.
        """
        n = int(y.shape[0])
        if init == "warm":
            if init_params is None:
                raise ValueError(
                    "init='warm' requires init_params (a parameter dict "
                    "matching the schema returned by `model.params`)."
                )
            cold = dict(init_params)
            for key in ("phi", "theta", "mu", "sigma_eps", "residual"):
                if key not in cold:
                    raise KeyError(
                        f"Warm-start init_params missing required key {key!r}."
                    )
        else:
            arma_seed = init_arma_params(y, p=self.p, q=self.q, mode=init)
            # Sigma_eps starts from the in-sample std of the seed
            # ARMA innovations (a tighter prior than data std for a
            # large-AR fit).
            mu_seed, eps_seed, _ = run_arma(
                y=y,
                phi=arma_seed["phi"],
                theta=arma_seed["theta"],
                mu=arma_seed["mu"],
                init_y_lags=arma_pre_sample_state(
                    y, self.p, self.q, mode="backcast",
                    backcast_length=backcast_length,
                )[0],
                init_eps_lags=arma_pre_sample_state(
                    y, self.p, self.q, mode="backcast",
                    backcast_length=backcast_length,
                )[1],
            )
            sigma_seed = jnp.maximum(jnp.std(eps_seed), _SIGMA_FLOOR)
            cold = {
                "phi": arma_seed["phi"],
                "theta": arma_seed["theta"],
                "mu": arma_seed["mu"],
                "sigma_eps": sigma_seed,
                "residual": wrapper.example_shape_params(),
            }

        x0 = self._pack_x0(cold, wrapper)

        # Map fit-time `init` mode (parameter starting values) to the
        # recursion's pre-sample-state mode (which only accepts
        # ``backcast``, ``sample``, ``zero``).  ``analytical`` and
        # ``warm`` both default to ``backcast`` — mean-of-leading-
        # window is a safer pre-sample anchor than ``zero`` and
        # avoids amplifying optimiser sensitivity to the initial
        # iteration's transient.
        _recursion_state_mode = (
            "sample" if init == "sample" else "backcast"
        )
        recursion_init_y_lags, recursion_init_eps_lags = arma_pre_sample_state(
            y, self.p, self.q,
            mode=_recursion_state_mode,
            backcast_length=backcast_length,
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
            y=y,
            init_y_lags=recursion_init_y_lags,
            init_eps_lags=recursion_init_eps_lags,
            lr=lr,
            maxiter=maxiter,
        )
        x_opt = res["x"]
        phi, theta, mu, sigma_eps, residual = self._unpack_raw(x_opt, wrapper)

        # Terminal state from a final pass at the optimum.
        _, _, terminal = self._run_recursion(
            y=y, phi=phi, theta=theta, mu=mu,
            init_y_lags=recursion_init_y_lags,
            init_eps_lags=recursion_init_eps_lags,
        )
        nll = objective(
            x_opt, y, recursion_init_y_lags, recursion_init_eps_lags,
        )
        params_dict = {
            "phi": phi,
            "theta": theta,
            "mu": mu,
            "sigma_eps": sigma_eps,
            "residual": residual,
        }
        return params_dict, terminal, nll

    # ------------------------------------------------------------------
    # Natural-parameter NLL closures + SE machinery
    # ------------------------------------------------------------------
    def _natural_objective_closures(
        self,
        wrapper: StandardisedResidual,
        params_dict: dict,
        y_arr: Array,
        init_y_lags: Array,
        init_eps_lags: Array,
    ):
        r"""Build per-observation / sum negative-log-likelihood closures
        over the **flat natural-parameter vector** at the constrained
        MLE — used for observed-Fisher-information SE computation.

        Mirrors :meth:`copulax._src.timeseries._joint.arma_garch.ArmaGarch
        ._natural_objective_closures`.  The optimiser's reparameterised
        flat vector is discarded; the SE pipeline operates directly on
        the natural parameters ``(phi, theta, mu, sigma_eps,
        residual_shape...)`` so the inverse-Hessian rescales correctly
        without an unwinding chain-rule term.

        The schema is built from ``params_dict`` rather than ``self``
        so it sees the post-fit residual-shape keys; this method is
        called from ``fit()`` on an unfitted instance whose
        ``self.residual_params`` is still ``None``.

        Args:
            wrapper: Standardised-residual wrapper.
            params_dict: Constrained-natural-params dict at the
                MLE — the source of the flat-vector schema and the
                values whose flattening defines the parameter
                ordering.
            y_arr: shape ``(n,)`` — observed series.
            init_y_lags / init_eps_lags: Pre-sample recursion state
                from :meth:`_recursion_inputs`.

        Returns:
            ``(nll_total, per_obs_nll, schema)`` — closures consumed
            by :func:`compute_param_cov`, plus the schema needed to
            unflatten the resulting SE vector back to a params-dict.
        """
        _, schema = params_to_flat(params_dict)

        def per_obs_nll(flat: Array) -> Array:
            params = flat_to_params(flat, schema)
            phi_ = params.get("phi", jnp.zeros((0,), dtype=float))
            theta_ = params.get("theta", jnp.zeros((0,), dtype=float))
            mu_ = params.get("mu", jnp.asarray(0.0, dtype=float))
            sigma_eps_ = params.get("sigma_eps", jnp.asarray(1.0, dtype=float))
            residual_ = params.get("residual", {}) or {}
            sigma_safe = jnp.maximum(sigma_eps_, _SIGMA_FLOOR)
            _, eps_seq, _ = run_arma(
                y=y_arr, phi=phi_, theta=theta_, mu=mu_,
                init_y_lags=init_y_lags, init_eps_lags=init_eps_lags,
            )
            z = eps_seq / sigma_safe
            logpdf = wrapper.logpdf(z, residual_) - jnp.log(sigma_safe)
            # Mask non-finite contributions to a finite sentinel so
            # ``jax.hessian`` doesn't produce NaN gradients on
            # numerically-degenerate observations (rare at the MLE
            # but cheap insurance).
            return -jnp.where(jnp.isfinite(logpdf), logpdf, 0.0)

        def nll_total(flat: Array) -> Array:
            return jnp.sum(per_obs_nll(flat))

        return nll_total, per_obs_nll, schema

    def _compute_se(
        self,
        params_dict: dict,
        wrapper: StandardisedResidual,
        y_arr: Array,
        init_y_lags: Array,
        init_eps_lags: Array,
        n_obs: int,
    ) -> tuple[Array, Array, dict]:
        r"""Observed-Hessian asymptotic covariance + standard errors
        at the constrained MLE.

        Uses ``cov_type="classic"`` (inverse observed information /
        Hessian only — no Bollerslev-Wooldridge sandwich).  Standalone
        ARMA fits assume the residual law is correctly specified
        (the user has selected it explicitly), so the QMLE robust
        sandwich is unnecessary.

        Returns:
            ``(cov, se_flat, se_dict)`` —
            ``cov``: ``(k, k)`` asymptotic covariance.
            ``se_flat``: ``sqrt(diag(cov))`` as a flat vector.
            ``se_dict``: same SEs repacked into the ``params``-shaped
            nested dict (consumed by ``standard_errors_``).
        """
        nll_total, per_obs_nll, schema = self._natural_objective_closures(
            wrapper, params_dict, y_arr, init_y_lags, init_eps_lags,
        )
        params_flat, _ = params_to_flat(params_dict)
        cov = compute_param_cov(
            nll_total=nll_total,
            per_obs_nll=per_obs_nll,
            params_flat=params_flat,
            n_obs=n_obs,
            cov_type="classic",
        )
        se_flat = jnp.sqrt(jnp.maximum(jnp.diag(cov), 0.0))
        se_dict = flat_to_params(se_flat, schema)
        return cov, se_flat, se_dict

    def _recompute_se(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> tuple[Array, Array, dict]:
        r"""Recompute SEs against an alternate series.  Used by
        :meth:`standard_errors` and :meth:`cov_matrix` when ``y`` is
        supplied at call time.
        """
        wrapper = self._wrapper()
        y_arr, init_y_lags, init_eps_lags = self._recursion_inputs(
            y, init, backcast_length,
        )
        n_obs = int(y_arr.shape[0])
        return self._compute_se(
            params_dict=self.params,
            wrapper=wrapper, y_arr=y_arr,
            init_y_lags=init_y_lags, init_eps_lags=init_eps_lags,
            n_obs=n_obs,
        )

    # ------------------------------------------------------------------
    # Residual-diagnostic caching
    # ------------------------------------------------------------------
    def _compute_residual_diagnostics(
        self,
        z: Array,
        *,
        loglikelihood: Array,
        aic: Array,
        bic: Array,
    ) -> dict:
        r"""Build the canonical fit-time diagnostics bundle used by
        every cached default-arg accessor and by ``summary()``.

        Combines:

        * the model-fit scalars ``loglikelihood`` / ``aic`` / ``bic``
          (passed in by ``fit()`` so they remain consistent with the
          objective evaluated at the optimum);
        * the standardised-residual autocorrelation arrays
          ``acf`` / ``pacf`` at the convenience-wrapper defaults
          (``lags=20``, PACF ``method="yule_walker"``);
        * the five hypothesis-test result dicts on the standardised
          residuals: ``ljung_box`` / ``ljung_box_sq`` / ``arch_lm``
          / ``adf`` / ``kpss``.

        The Ljung-Box dof correction matches the convenience
        wrapper's default (``lags - p - q``, floored at 1).
        """
        dof_corr = max(10 - self.p - self.q, 1)
        return {
            "loglikelihood": loglikelihood,
            "aic":           aic,
            "bic":           bic,
            "acf":           _diag_acf(z, 20),
            "pacf":          _diag_pacf(z, 20, method="yule_walker"),
            "ljung_box":     _diag_ljung_box(z, 10, dof=dof_corr),
            "ljung_box_sq":  _diag_ljung_box(z * z, 10, dof=dof_corr),
            "arch_lm":       _diag_arch_lm(z, 5),
            "adf":           _diag_adf(z, regression="c"),
            "kpss":          _diag_kpss(z, regression="c"),
        }

    def fit(
        self,
        y: ArrayLike,
        *,
        init: str = "analytical",
        init_params: Optional[dict] = None,
        backcast_length: Optional[int] = None,
        maxiter: int = 200,
        lr: float = 0.05,
        name: Optional[str] = None,
    ) -> "ARMABase":
        r"""Fit the ARMA(p, q) model to a series via Adam-driven MLE.

        Orders ``(p, q)`` and the residual distribution are set at
        construction time — see :meth:`__init__`.  ``fit`` itself is
        data-only.

        Args:
            y: shape ``(n,)`` — observed return series.
            init: One of ``"analytical"`` (Yule-Walker + Innovations
                Algorithm — default), ``"backcast"``, ``"sample"``, or
                ``"warm"`` (requires ``init_params``).
            init_params: Warm-start parameter dict matching the
                schema returned by :attr:`params`.  Required when
                ``init="warm"``; ignored otherwise.
            backcast_length: Length of the leading window used to
                seed the recursion's pre-sample state under
                ``init="backcast"`` / ``"sample"``.  ``None`` uses
                the entire series.
            maxiter: Number of Adam iterations.
            lr: Adam learning rate.
            name: Optional custom name for the fitted instance.

        Returns:
            A fitted instance of the same concrete class with
            ``params``, ``terminal_state``, ``n_train_``,
            ``cov_matrix_`` / ``standard_errors_``, and the
            consolidated ``residual_diagnostics_`` bundle (which
            holds ``loglikelihood`` / ``aic`` / ``bic`` / ``acf``
            / ``pacf`` plus the five hypothesis-test result dicts)
            populated.
        """
        self._check_method(init)
        wrapper = StandardisedResidual(self.residual_dist)
        y_arr = self._validate_series(y)
        n = int(y_arr.shape[0])
        self._validate_backcast_length(backcast_length, n)

        params_dict, terminal_state, nll = self._fit_internal(
            y_arr, wrapper, init=init, init_params=init_params,
            backcast_length=backcast_length, maxiter=maxiter, lr=lr,
        )

        # Diagnostics: nll is the *mean* negative log-likelihood; the
        # log-likelihood SUM follows by re-multiplying by n.
        loglike = -nll * n
        n_params_total = (
            self.p + self.q + 1 + 1 + wrapper.n_shape_params
        )
        aic = 2.0 * n_params_total - 2.0 * loglike
        bic = n_params_total * jnp.log(jnp.asarray(n, dtype=float)) - 2.0 * loglike

        # Pre-sample state for the SE / diagnostic recursions —
        # mirrors the convention used by the optimiser path above.
        recursion_init_y_lags, recursion_init_eps_lags = arma_pre_sample_state(
            y_arr, self.p, self.q,
            mode=("sample" if init == "sample" else "backcast"),
            backcast_length=backcast_length,
        )

        # Observed-Hessian asymptotic covariance + standard errors at
        # the constrained MLE.  Reusing the natural-parameter dict
        # built by ``_fit_internal`` keeps everything in sync.
        cov, _, se_dict = self._compute_se(
            params_dict=params_dict,
            wrapper=wrapper, y_arr=y_arr,
            init_y_lags=recursion_init_y_lags,
            init_eps_lags=recursion_init_eps_lags,
            n_obs=n,
        )

        # Cache the five default-arg residual diagnostics so
        # ``summary()`` and the ``*_residuals(y=None)`` accessors can
        # return them without recomputation.
        sigma_safe = jnp.maximum(params_dict["sigma_eps"], _SIGMA_FLOOR)
        _, eps_seq, _ = run_arma(
            y=y_arr,
            phi=params_dict["phi"], theta=params_dict["theta"],
            mu=params_dict["mu"],
            init_y_lags=recursion_init_y_lags,
            init_eps_lags=recursion_init_eps_lags,
        )
        z_train = eps_seq / sigma_safe
        diagnostics = self._compute_residual_diagnostics(
            z_train, loglikelihood=loglike, aic=aic, bic=bic,
        )

        # Promote the unfitted template to the fitted standardised
        # distribution so ``fit.residual_dist`` is the canonical
        # accessor — no separate ``residual_distribution`` property.
        fitted_residual_dist = wrapper.to_distribution(
            params_dict["residual"],
            name=f"{self.residual_dist.name}-stdresid",
        )

        cls = type(self)
        if name is None:
            name = (
                f"Fitted{cls.__name__}({self.p},{self.q})"
                f"-{self.residual_dist.name}"
            )
        return cls(
            name=name,
            p=self.p,
            q=self.q,
            residual_dist=fitted_residual_dist,
            phi=params_dict["phi"],
            theta=params_dict["theta"],
            mu=params_dict["mu"],
            sigma_eps=params_dict["sigma_eps"],
            residual_params=params_dict["residual"],
            terminal_state=terminal_state,
            n_train_=n,
            cov_matrix_=cov,
            standard_errors_=se_dict,
            residual_diagnostics_=diagnostics,
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
    ) -> tuple[Array, Array, Array]:
        y_arr = self._validate_series(y)
        n = int(y_arr.shape[0])
        self._validate_backcast_length(backcast_length, n)
        init_y_lags, init_eps_lags = self._build_initial_state(
            y_arr, mode=init, backcast_length=backcast_length,
        )
        return y_arr, init_y_lags, init_eps_lags

    def conditional_mean(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""One-step-ahead conditional mean ``μ_t`` over ``y``."""
        self._require_fitted()
        y_arr, init_y_lags, init_eps_lags = self._recursion_inputs(
            y, init, backcast_length,
        )
        mu_seq, _, _ = self._run_recursion(
            y_arr, self.phi, self.theta, self.mu,
            init_y_lags=init_y_lags, init_eps_lags=init_eps_lags,
        )
        return mu_seq

    def conditional_variance(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Constant trajectory equal to :math:`\\sigma_\\varepsilon^2`.

        Mean models do not parameterise heteroskedasticity — the
        conditional variance is just the unconditional innovation
        variance under the fitted scale.  Returned as a length-``n``
        array (rather than a scalar) so callers can compose with
        variance-model outputs without per-call broadcasting.
        """
        self._require_fitted()
        y_arr = self._validate_series(y)
        n = int(y_arr.shape[0])
        return jnp.full((n,), self.sigma_eps ** 2)

    def residuals(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> dict:
        r"""Innovation and standardised residuals over ``y``.

        Returns ``{"residuals": ε_t, "standardised_residuals": z_t}``
        where :math:`\varepsilon_t = y_t - \mu_t` and
        :math:`z_t = \varepsilon_t / \sigma_\varepsilon`.  The dict
        return shape is uniform across ARMA / GARCH / ArmaGarch
        so user code does not need per-family branching.

        Requires ``y``; the model does not retain the training
        series — pass it explicitly.
        """
        self._require_fitted()
        y_arr, init_y_lags, init_eps_lags = self._recursion_inputs(
            y, init, backcast_length,
        )
        _, eps_seq, _ = self._run_recursion(
            y_arr, self.phi, self.theta, self.mu,
            init_y_lags=init_y_lags, init_eps_lags=init_eps_lags,
        )
        sigma = jnp.maximum(self.sigma_eps, _SIGMA_FLOOR)
        return {
            "residuals": eps_seq,
            "standardised_residuals": eps_seq / sigma,
        }

    def standardised_residuals(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Innovation residuals divided by the fitted scale —
        :math:`z_t = \\varepsilon_t / \\sigma_\\varepsilon`.

        Convenience accessor; equivalent to
        ``residuals(y)["standardised_residuals"]``.
        """
        return self.residuals(
            y, init=init, backcast_length=backcast_length,
        )["standardised_residuals"]

    def terminal_state_from(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> ARMATerminalState:
        r"""Produce a :class:`ARMATerminalState` from a (possibly new)
        series — used to roll :meth:`forecast` forward from a window
        other than the one the model was fit on.
        """
        self._require_fitted()
        y_arr, init_y_lags, init_eps_lags = self._recursion_inputs(
            y, init, backcast_length,
        )
        _, _, terminal = self._run_recursion(
            y_arr, self.phi, self.theta, self.mu,
            init_y_lags=init_y_lags, init_eps_lags=init_eps_lags,
        )
        return terminal

    # ------------------------------------------------------------------
    # Forecast + sampling
    # ------------------------------------------------------------------
    def forecast(
        self,
        h: int,
        *,
        method: str = "analytical",
        n_paths: int = 0,
        key: Optional[Array] = None,
        last_state: Optional[ARMATerminalState] = None,
    ) -> dict:
        r"""``h``-step-ahead conditional moments.

        Args:
            h: Forecast horizon.  Static Python ``int``.
            method: ``"analytical"`` (default) returns the closed-form
                ARMA conditional-mean recursion; ``"simulation"``
                returns the empirical mean / variance from
                ``n_paths`` Monte Carlo sample paths via :meth:`rvs`.
            n_paths: Required when ``method="simulation"``.
            key: Optional PRNG key for the simulation path; resolved
                to a deterministic default if ``None``.
            last_state: Optional override for the carry state.
                Defaults to ``self.terminal_state``.

        Returns:
            ``{"mean": (h,), "variance": (h,), "paths": Optional[(n_paths, h)]}``.

        Note:
            Under ``method="analytical"`` the returned ``variance``
            is **per-step** — every entry equals
            :math:`\sigma_\varepsilon^2`, the conditional one-step
            innovation variance.  This is *not* the cumulative
            :math:`h`-step forecast-error variance, which for an
            ARMA(p, q) process is :math:`\sigma_\varepsilon^2 \cdot
            \sum_{j=0}^{h-1} \psi_j^2` with :math:`\psi_j` the Wold
            MA(:math:`\infty`) representation (Hamilton 1994 eqn
            4.2.4).  Use ``method="simulation"`` to obtain empirical
            cumulative forecast variances directly.
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
        var_per_step = self.sigma_eps ** 2

        if method == "analytical":
            # Roll the centred-form ARMA conditional-mean recursion
            # forward h steps with zero future innovations (analytic
            # E[ε_{t+i}] = 0 under the residual law).  Returned
            # variance is the per-step innovation variance σ_ε² (see
            # docstring Note for the cumulative-variance formula).
            mu_path = []
            y_lags = state.y_lags
            eps_lags = state.eps_lags
            for _ in range(h):
                ar_term = (
                    jnp.dot(self.phi, y_lags - self.mu) if self.p > 0 else 0.0
                )
                ma_term = jnp.dot(self.theta, eps_lags) if self.q > 0 else 0.0
                mu_t = self.mu + ar_term + ma_term
                mu_path.append(mu_t)
                # Future innovations are zero under analytical mean
                # forecast — y_t = μ_t, ε_t = 0.
                if self.p > 0:
                    y_lags = jnp.concatenate(
                        [mu_t.reshape((1,)), y_lags[:-1]]
                    )
                if self.q > 0:
                    eps_lags = jnp.concatenate(
                        [jnp.zeros((1,), dtype=float), eps_lags[:-1]]
                    )
            mean = jnp.stack(mu_path)
            variance = jnp.full((h,), var_per_step)
            return {"mean": mean, "variance": variance, "paths": None}

        elif method == "simulation":
            if n_paths <= 0:
                raise ValueError(
                    "method='simulation' requires n_paths > 0."
                )
            key = _resolve_key(key)
            paths = self.rvs(
                size=(int(n_paths), h),
                key=key,
                last_state=state,
            )
            mean = jnp.mean(paths, axis=0)
            variance = jnp.var(paths, axis=0)
            return {"mean": mean, "variance": variance, "paths": paths}

        else:
            raise ValueError(
                f"Unknown forecast method {method!r}; expected "
                "'analytical' or 'simulation'."
            )

    def rvs(
        self,
        size=None,
        *,
        key: Optional[Array] = None,
        u: Optional[ArrayLike] = None,
        last_state: Optional[ARMATerminalState] = None,
    ) -> Array:
        r"""Simulate synthetic paths from the fitted ARMA model.

        See :class:`copulax._src.timeseries._base.TimeSeriesModel` for
        the full ``size`` / ``key`` / ``u`` / ``last_state`` contract.
        Inverse-transform support via ``u`` lets callers couple the
        path to a copula or use antithetic / stratified sampling.
        """
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
                # Single path
                z = wrapper.ppf(u_arr, self.residual_params)
                return self._roll_path(z, state)
            elif u_arr.ndim == 2:
                # Batch of n_paths
                z = wrapper.ppf(u_arr.reshape(-1), self.residual_params).reshape(shape)
                return jax.vmap(lambda zi: self._roll_path(zi, state))(z)
            else:
                raise ValueError(
                    f"u must have ndim 1 or 2; got ndim={u_arr.ndim}."
                )

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

    def _roll_path(self, z: Array, state: ARMATerminalState) -> Array:
        r"""Roll a single innovation series ``z`` forward through the
        centred-form ARMA recursion to produce a level-series path.
        """
        sigma = self.sigma_eps
        mu = self.mu
        phi = self.phi
        theta = self.theta

        def step(carry, z_t):
            y_lags, eps_lags = carry
            ar_term = jnp.dot(phi, y_lags - mu) if self.p > 0 else 0.0
            ma_term = jnp.dot(theta, eps_lags) if self.q > 0 else 0.0
            mu_t = mu + ar_term + ma_term
            eps_t = sigma * z_t
            y_t = mu_t + eps_t
            new_y_lags = (
                jnp.concatenate([y_t.reshape((1,)), y_lags[:-1]])
                if self.p > 0 else y_lags
            )
            new_eps_lags = (
                jnp.concatenate([eps_t.reshape((1,)), eps_lags[:-1]])
                if self.q > 0 else eps_lags
            )
            return (new_y_lags, new_eps_lags), y_t

        init_carry = (state.y_lags, state.eps_lags)
        _, y_seq = jax.lax.scan(step, init_carry, z)
        return y_seq

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        r"""Analytic, parameter-only statistics for the fitted model.

        Returns:
            ``{
                "mean":         unconditional mean (when finite),
                "variance":     unconditional variance,
                "sigma_eps":    fitted innovation scale,
                "is_stationary": bool flag,
                "is_invertible": bool flag,
                "ar_root_moduli": (p,) array, or empty,
                "ma_root_moduli": (q,) array, or empty,
            }``
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
            if self.q > 0
            else jnp.zeros((0,), dtype=jnp.complex64)
        )
        is_stat = ar_is_stationary(self.phi)
        is_inv = (
            ma_is_invertible(self.theta) if self.q > 0 else jnp.asarray(True)
        )
        # Centred-form ARMA: μ IS the unconditional mean (no AR
        # rescaling required).
        unconditional_mean = self.mu
        # Approximate unconditional variance: for small (p, q) and
        # stationary processes, var(y) is dominated by the innovation
        # variance scaled by 1 / (1 - phi^T phi) for AR(p) — a
        # simple lower bound.  An exact expression requires solving
        # the Yule-Walker equations on the fitted parameters.
        unconditional_variance = jnp.where(
            self.p > 0,
            self.sigma_eps ** 2 / jnp.maximum(1.0 - jnp.sum(self.phi ** 2), 1e-12),
            self.sigma_eps ** 2,
        )
        return {
            "mean": unconditional_mean,
            "variance": unconditional_variance,
            "sigma_eps": self.sigma_eps,
            "is_stationary": is_stat,
            "is_invertible": is_inv,
            "ar_root_moduli": jnp.abs(ar_roots),
            "ma_root_moduli": jnp.abs(ma_roots),
        }

    # ------------------------------------------------------------------
    # Loglikelihood / AIC / BIC
    # ------------------------------------------------------------------
    def _log_likelihood_on_series(
        self,
        y: ArrayLike,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        self._require_fitted()
        wrapper = self._wrapper()
        y_arr, init_y_lags, init_eps_lags = self._recursion_inputs(
            y, init, backcast_length,
        )
        _, eps_seq, _ = self._run_recursion(
            y_arr, self.phi, self.theta, self.mu,
            init_y_lags=init_y_lags, init_eps_lags=init_eps_lags,
        )
        sigma = jnp.maximum(self.sigma_eps, _SIGMA_FLOOR)
        z = eps_seq / sigma
        logpdf = wrapper.logpdf(z, self.residual_params) - jnp.log(sigma)
        return jnp.sum(logpdf)

    def loglikelihood(
        self,
        y: Optional[ArrayLike] = None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Log-likelihood of the fitted model.

        With ``y=None`` (default) returns the stored fit-time value
        ``residual_diagnostics_["loglikelihood"]``; pass ``y`` to
        recompute against a held-out series.
        """
        if y is None:
            self._require_fitted()
            return self.residual_diagnostics_["loglikelihood"]
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
        r"""Akaike Information Criterion.

        Returns ``residual_diagnostics_["aic"]`` when ``y`` is
        omitted; recomputes against ``y`` otherwise.
        """
        if y is None:
            self._require_fitted()
            return self.residual_diagnostics_["aic"]
        ll = self._log_likelihood_on_series(y, init=init, backcast_length=backcast_length)
        return 2.0 * self.n_params - 2.0 * ll

    def bic(
        self,
        y: Optional[ArrayLike] = None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Bayesian Information Criterion.

        Returns ``residual_diagnostics_["bic"]`` when ``y`` is
        omitted; recomputes against ``y`` otherwise.
        """
        if y is None:
            self._require_fitted()
            return self.residual_diagnostics_["bic"]
        y_arr = self._validate_series(y)
        ll = self._log_likelihood_on_series(y_arr, init=init, backcast_length=backcast_length)
        n = jnp.asarray(int(y_arr.shape[0]), dtype=float)
        return self.n_params * jnp.log(n) - 2.0 * ll

    # ------------------------------------------------------------------
    # Diagnostics — route through ``_diagnostics`` on the standardised
    # residual series ``z_t = ε_t / σ_ε``
    # ------------------------------------------------------------------
    def acf(
        self,
        y: Optional[ArrayLike] = None,
        lags: int = 20,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Sample ACF of the standardised residuals.

        With ``y=None`` (default) AND ``lags`` at its default,
        returns the cached value from
        ``residual_diagnostics_["acf"]`` populated at fit time.
        Pass ``y`` to recompute against an alternate series;
        non-default ``lags`` requires ``y`` explicitly.
        """
        if y is None:
            self._require_fitted()
            if lags == 20 and self.residual_diagnostics_ is not None:
                return self.residual_diagnostics_["acf"]
            raise ValueError(
                "y is required when overriding the default kwargs of "
                "acf() — only the default-arg result is cached."
            )
        z = self.standardised_residuals(
            y, init=init, backcast_length=backcast_length,
        )
        return _diag_acf(z, lags)

    def pacf(
        self,
        y: Optional[ArrayLike] = None,
        lags: int = 20,
        method: str = "yule_walker",
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Sample PACF of the standardised residuals.

        With ``y=None`` (default) AND ``lags`` / ``method`` at
        their defaults, returns the cached value from
        ``residual_diagnostics_["pacf"]`` populated at fit time.
        Pass ``y`` to recompute against an alternate series;
        non-default kwargs require ``y`` explicitly.
        """
        if y is None:
            self._require_fitted()
            if (
                lags == 20 and method == "yule_walker"
                and self.residual_diagnostics_ is not None
            ):
                return self.residual_diagnostics_["pacf"]
            raise ValueError(
                "y is required when overriding the default kwargs of "
                "pacf() — only the default-arg result is cached."
            )
        z = self.standardised_residuals(
            y, init=init, backcast_length=backcast_length,
        )
        return _diag_pacf(z, lags, method=method)

    def ljung_box(
        self,
        y: Optional[ArrayLike] = None,
        lags: int = 10,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
        dof_correction: bool = True,
    ) -> dict:
        r"""Ljung-Box Q-test on the standardised residuals.

        H0: the standardised residuals are white noise — passing
        means the mean-model residuals look IID at lags ``1..lags``.

        ``dof_correction`` (default ``True``) sets the asymptotic
        :math:`\chi^2` degrees-of-freedom to ``lags - p - q`` so the
        null distribution accounts for the fitted ARMA parameters
        (Box-Jenkins-Reinsel §8.2.2).  Set ``False`` to recover the
        primitive form on the standardised residual series.

        With ``y=None`` (default) AND every other kwarg at its
        default, returns the cached value from
        ``residual_diagnostics_["ljung_box"]`` populated at fit
        time.  Pass ``y`` to recompute against an alternate series;
        non-default kwargs require ``y`` explicitly.

        Returns the standardised result dict from
        :func:`copulax.timeseries.ljung_box` —
        ``{"statistic", "p_value", "used_lag", "n_obs", "dof"}``.
        """
        if y is None:
            self._require_fitted()
            if (
                lags == 10 and dof_correction is True
                and self.residual_diagnostics_ is not None
            ):
                return self.residual_diagnostics_["ljung_box"]
            raise ValueError(
                "y is required when overriding the default kwargs of "
                "ljung_box() — only the default-arg result is cached."
            )
        z = self.standardised_residuals(
            y, init=init, backcast_length=backcast_length,
        )
        dof = (lags - self.p - self.q) if dof_correction else lags
        return _diag_ljung_box(z, lags, dof=dof)

    def arch_lm(
        self,
        y: Optional[ArrayLike] = None,
        lags: int = 5,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> dict:
        r"""Engle's ARCH-LM test on the standardised residuals.

        H0: no remaining ARCH effect — passing means the mean
        model has captured everything; failing motivates a
        variance-equation extension.

        With ``y=None`` (default) AND ``lags`` at its default,
        returns the cached value from
        ``residual_diagnostics_["arch_lm"]`` populated at fit time.
        Pass ``y`` to recompute against an alternate series;
        non-default kwargs require ``y`` explicitly.

        Returns the standardised result dict from
        :func:`copulax.timeseries.arch_lm` —
        ``{"statistic", "p_value", "used_lag", "n_obs", "dof"}``.
        """
        if y is None:
            self._require_fitted()
            if lags == 5 and self.residual_diagnostics_ is not None:
                return self.residual_diagnostics_["arch_lm"]
            raise ValueError(
                "y is required when overriding the default kwargs of "
                "arch_lm() — only the default-arg result is cached."
            )
        z = self.standardised_residuals(
            y, init=init, backcast_length=backcast_length,
        )
        return _diag_arch_lm(z, lags)

    def adf_residuals(
        self,
        y: Optional[ArrayLike] = None,
        *,
        regression: str = "c",
        lags: Optional[int] = None,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> dict:
        r"""Augmented Dickey-Fuller test on the standardised residuals.

        Sanity check that the model has captured all unit-root
        behaviour in the input series — for a healthy fit, ADF on
        residuals should **reject** the unit-root null.

        Cached default-arg behaviour mirrors :meth:`ljung_box` /
        :meth:`arch_lm`.

        Returns the standardised result dict from
        :func:`copulax.timeseries.adf` —
        ``{"statistic", "p_value", "used_lag", "n_obs",
        "crit_values"}``.
        """
        if y is None:
            self._require_fitted()
            if (
                regression == "c" and lags is None
                and self.residual_diagnostics_ is not None
            ):
                return self.residual_diagnostics_["adf"]
            raise ValueError(
                "y is required when overriding the default kwargs of "
                "adf_residuals() — only the default-arg result is cached."
            )
        z = self.standardised_residuals(
            y, init=init, backcast_length=backcast_length,
        )
        return _diag_adf(z, regression=regression, lags=lags)

    def kpss_residuals(
        self,
        y: Optional[ArrayLike] = None,
        *,
        regression: str = "c",
        lags: Optional[int] = None,
        lags_choice: str = "short",
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> dict:
        r"""KPSS stationarity test on the standardised residuals.

        Complements :meth:`adf_residuals` — for a healthy fit, KPSS
        should **fail to reject** the stationarity null.

        Cached default-arg behaviour mirrors :meth:`ljung_box` /
        :meth:`arch_lm`.

        Returns the standardised result dict from
        :func:`copulax.timeseries.kpss` —
        ``{"statistic", "p_value", "used_lag", "n_obs",
        "crit_values"}``.
        """
        if y is None:
            self._require_fitted()
            if (
                regression == "c" and lags is None
                and lags_choice == "short"
                and self.residual_diagnostics_ is not None
            ):
                return self.residual_diagnostics_["kpss"]
            raise ValueError(
                "y is required when overriding the default kwargs of "
                "kpss_residuals() — only the default-arg result is cached."
            )
        z = self.standardised_residuals(
            y, init=init, backcast_length=backcast_length,
        )
        return _diag_kpss(
            z, regression=regression, lags=lags, lags_choice=lags_choice,
        )

    # ------------------------------------------------------------------
    # Standard errors / confidence intervals / summary
    # ------------------------------------------------------------------
    def cov_matrix(
        self,
        y: Optional[ArrayLike] = None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Asymptotic covariance matrix of the natural-parameter MLE.

        Inverse observed Hessian (``cov_type="classic"``) — assumes
        the residual law is correctly specified, which the user
        chose explicitly via ``residual_dist=``.  Use
        :class:`copulax.timeseries.ArmaGarch` for the
        Bollerslev-Wooldridge sandwich form on a joint MLE, or
        :func:`copulax.timeseries.two_stage_cov` for the
        Pagan-Newey two-stage correction on the separable workflow.

        With ``y=None`` (default) returns the cached
        ``cov_matrix_``; pass ``y`` to recompute against an
        alternate series.
        """
        self._require_fitted()
        if y is None:
            return self.cov_matrix_
        return self._recompute_se(
            y, init=init, backcast_length=backcast_length,
        )[0]

    def standard_errors(
        self,
        y: Optional[ArrayLike] = None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> dict:
        r"""Asymptotic standard errors structured to mirror :attr:`params`.

        With ``y=None`` (default) returns the cached
        ``standard_errors_``; pass ``y`` to recompute.
        """
        self._require_fitted()
        if y is None:
            return self.standard_errors_
        return self._recompute_se(
            y, init=init, backcast_length=backcast_length,
        )[2]

    def confidence_intervals(self, alpha: float = 0.05) -> dict:
        r"""``(lower, upper)`` confidence intervals at significance
        ``alpha`` for every fitted parameter, structured to mirror
        :attr:`params`.

        Built from the stored ``standard_errors_`` via the
        Wald-statistic CI :math:`\hat\theta \pm z_{\alpha/2}
        \cdot \widehat{\mathrm{SE}}(\hat\theta)`.
        """
        self._require_fitted()
        if self.standard_errors_ is None or self.params is None:
            raise ValueError(
                "Confidence intervals require both `standard_errors_` "
                "and `params` to be populated."
            )
        from jax.scipy.stats import norm
        z_crit = float(norm.ppf(1.0 - alpha / 2.0))
        cis: dict = {}
        for key, val in self.params.items():
            if key == "residual":
                cis[key] = {}
                for sub_key, sub_val in val.items():
                    sub_se = self.standard_errors_[key][sub_key]
                    sub_arr = jnp.asarray(sub_val, dtype=float)
                    sub_se_arr = jnp.asarray(sub_se, dtype=float)
                    cis[key][sub_key] = (
                        sub_arr - z_crit * sub_se_arr,
                        sub_arr + z_crit * sub_se_arr,
                    )
            else:
                est = jnp.asarray(val, dtype=float)
                se = jnp.asarray(self.standard_errors_[key], dtype=float)
                cis[key] = (est - z_crit * se, est + z_crit * se)
        return cis

    def summary(self) -> str:
        r"""Render a printable parameter / diagnostics table.

        Sections (mean equation + residual distribution + residual
        diagnostics) are separated by inline-labelled dashed lines;
        empty sections (e.g. residual distribution under
        ``normal``-law fits, where there are no free shape
        parameters) are silently suppressed.

        See :mod:`copulax._src.timeseries._summary` for the visual
        contract.
        """
        self._require_fitted()
        if (
            self.standard_errors_ is None
            or self.residual_diagnostics_ is None
        ):
            raise ValueError(
                "summary() requires `standard_errors_` and "
                "`residual_diagnostics_` to be populated.  Refit the "
                "model on a recent CopulAX version, or load it from a "
                "checkpoint that includes these fields."
            )
        mean_keys = ("phi", "theta", "mu", "sigma_eps")
        mean_section = ParamSection(
            label=self._mean_section_label(),
            rows=iter_param_rows(
                {k: self.params[k] for k in mean_keys},
                {k: self.standard_errors_[k] for k in mean_keys},
                vector_keys=("phi", "theta"),
            ),
        )
        res_section = residual_section(
            self.params["residual"],
            self.standard_errors_["residual"],
            dist_name=display_residual_name(self.residual_dist.name),
        )
        return format_summary(
            header=self._summary_header(),
            param_sections=[mean_section, res_section],
            diagnostic_rows=build_diagnostic_rows(self.residual_diagnostics_),
            loglikelihood=float(self.residual_diagnostics_["loglikelihood"]),
            aic=float(self.residual_diagnostics_["aic"]),
            bic=float(self.residual_diagnostics_["bic"]),
            n_train=int(self.n_train_),
        )

    def _summary_header(self) -> str:
        r"""Top-line of :meth:`summary` — overridable by subclasses
        that want to drop unused orders (``AR`` / ``MA``)."""
        return (
            f"ARMA({self.p}, {self.q}) — "
            f"{display_residual_name(self.residual_dist.name)} residuals"
        )

    def _mean_section_label(self) -> str:
        r"""Label embedded in the mean-equation section's separator
        line — overridable by subclasses."""
        return f"Mean equation — ARMA({self.p}, {self.q})"

    def plot_acf(
        self,
        y: Optional[ArrayLike] = None,
        lags: int = 20,
        alpha: float = 0.05,
        ax=None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ):
        r"""ACF stem plot for the standardised residuals.

        With ``y=None`` (default) AND ``lags`` at its default,
        renders the plot from the cached
        ``residual_diagnostics_["acf"]`` array (no recursion).
        Pass ``y`` to recompute against an alternate series;
        non-default ``lags`` requires ``y`` explicitly.
        """
        from copulax._src.timeseries._diagnostics import (
            plot_acf as _plot_acf,
            plot_acf_from_corr as _plot_acf_from_corr,
        )
        if y is None:
            self._require_fitted()
            if lags == 20 and self.residual_diagnostics_ is not None:
                return _plot_acf_from_corr(
                    self.residual_diagnostics_["acf"],
                    n_obs=int(self.n_train_),
                    alpha=alpha, ax=ax,
                )
            raise ValueError(
                "y is required when overriding the default kwargs of "
                "plot_acf() — only the default-arg result is cached."
            )
        z = self.standardised_residuals(
            y, init=init, backcast_length=backcast_length,
        )
        return _plot_acf(z, lags=lags, alpha=alpha, ax=ax)

    def plot_pacf(
        self,
        y: Optional[ArrayLike] = None,
        lags: int = 20,
        method: str = "yule_walker",
        alpha: float = 0.05,
        ax=None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ):
        r"""PACF stem plot for the standardised residuals.

        With ``y=None`` (default) AND ``lags`` / ``method`` at
        their defaults, renders the plot from the cached
        ``residual_diagnostics_["pacf"]`` array.  Pass ``y`` to
        recompute against an alternate series; non-default kwargs
        require ``y`` explicitly.
        """
        from copulax._src.timeseries._diagnostics import (
            plot_pacf as _plot_pacf,
            plot_pacf_from_corr as _plot_pacf_from_corr,
        )
        if y is None:
            self._require_fitted()
            if (
                lags == 20 and method == "yule_walker"
                and self.residual_diagnostics_ is not None
            ):
                return _plot_pacf_from_corr(
                    self.residual_diagnostics_["pacf"],
                    n_obs=int(self.n_train_),
                    alpha=alpha, ax=ax,
                )
            raise ValueError(
                "y is required when overriding the default kwargs of "
                "plot_pacf() — only the default-arg result is cached."
            )
        z = self.standardised_residuals(
            y, init=init, backcast_length=backcast_length,
        )
        return _plot_pacf(z, lags=lags, method=method, alpha=alpha, ax=ax)

    # ------------------------------------------------------------------
    # Model-fit plots
    # ------------------------------------------------------------------
    def plot_timeseries(self, y: ArrayLike, h: int = 0, ax=None):
        r"""Time-series chart with conditional-mean overlay (and an
        optional ``h``-step forecast extension)."""
        from copulax._src.timeseries._plotting import plot_timeseries_mean
        return plot_timeseries_mean(self, y, h=h, ax=ax)

    def plot_scatter(self, y: ArrayLike, ax=None) -> tuple:
        r"""Scatter of actual ``y_t`` vs forecast ``μ_t`` with
        ``y = x`` reference.  Returns ``(ax,)``."""
        from copulax._src.timeseries._plotting import plot_scatter_mean
        return plot_scatter_mean(self, y, ax=ax)

    # ------------------------------------------------------------------
    # Deserialisation
    # ------------------------------------------------------------------
    @classmethod
    def _deserialise(
        cls,
        metadata: dict,
        arrays: dict,
        residual_dist,
        name: Optional[str] = None,
    ) -> "ARMABase":
        r"""Reconstruct an ARMABase fitted instance from saved metadata
        and arrays.  Inverse of :meth:`TimeSeriesModel._serialise_traced`.

        Used by :func:`copulax._src._serialization.load`.
        """
        from copulax._src.timeseries._se import flat_to_params
        from copulax._src.timeseries._mean._arma_base import ARMATerminalState

        kwargs: dict = {
            "p": int(metadata["p"]),
            "q": int(metadata["q"]),
            "residual_dist": residual_dist,
        }
        if name is not None:
            kwargs["name"] = name

        if "params_schema" in metadata:
            schema = [(k, tuple(s)) for k, s in metadata["params_schema"]]
            params = flat_to_params(arrays["params_flat"], schema)
            kwargs["phi"] = params.get("phi")
            kwargs["theta"] = params.get("theta")
            kwargs["mu"] = params.get("mu")
            kwargs["sigma_eps"] = params.get("sigma_eps")
            if "residual" in params:
                kwargs["residual_params"] = params["residual"]
                # Promote the unfitted template back to a fitted
                # instance so ``loaded.residual_dist.params`` round-
                # trips the same as ``fit.residual_dist.params``.
                kwargs["residual_dist"] = StandardisedResidual(
                    residual_dist,
                ).to_distribution(
                    params["residual"],
                    name=f"{residual_dist.name}-stdresid",
                )

        if "ts_n_leaves" in metadata:
            n_leaves = int(metadata["ts_n_leaves"])
            leaves = [arrays[f"ts_{i}"] for i in range(n_leaves)]
            kwargs["terminal_state"] = ARMATerminalState(*leaves)

        if "diag_n_train_" in arrays:
            kwargs["n_train_"] = arrays["diag_n_train_"]

        if "cov_matrix_" in arrays:
            kwargs["cov_matrix_"] = arrays["cov_matrix_"]
        if "se_flat" in arrays and "se_schema" in metadata:
            se_schema = [(k, tuple(s)) for k, s in metadata["se_schema"]]
            kwargs["standard_errors_"] = flat_to_params(
                arrays["se_flat"], se_schema,
            )

        from copulax._src.timeseries._base import (
            _deserialise_residual_diagnostics,
        )
        diagnostics = _deserialise_residual_diagnostics(arrays, metadata)
        if diagnostics is not None:
            kwargs["residual_diagnostics_"] = diagnostics

        return cls(**kwargs)
