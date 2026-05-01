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
from copulax._src.timeseries._init import (
    arma_pre_sample_state,
    init_arma_params,
)
from copulax._src.timeseries._recursions import run_arma
from copulax._src.timeseries._residuals._standardise import StandardisedResidual
from copulax._src.timeseries._stationarity import (
    ar_to_raw,
    ma_to_raw,
    positive_to_raw,
    raw_to_ar,
    raw_to_ma,
    raw_to_positive,
)


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
        * ``p``, ``q``, ``residual_dist`` (the singleton template) are
          static fields — they parameterise the compiled fit graph
          and cannot change at runtime.
        * Fitted parameters (``phi``, ``theta``, ``c``,
          ``sigma_eps``) and the residual law's free shape
          parameters (``residual_params``) are traced ``Array``
          leaves — they flow through ``jax.grad`` cleanly.
        * Stored fit-time diagnostics (``loglikelihood_``, ``aic_``,
          ``bic_``, ``n_train_``) are populated by :meth:`fit`; they
          are ``None`` for unfitted instances.
        * ``terminal_state`` carries the last ``p`` returns and
          ``q`` innovations; ``forecast(h)`` rolls forward from
          there by default.
    """

    # ---- static configuration -------------------------------------------
    p: int = eqx.field(static=True)
    q: int = eqx.field(static=True)
    residual_dist: Univariate = eqx.field(static=True)

    # ---- traced fitted parameters ---------------------------------------
    phi: Optional[Array] = None
    theta: Optional[Array] = None
    c: Optional[Array] = None
    sigma_eps: Optional[Array] = None
    residual_params: Optional[dict] = None

    # ---- per-fit terminal carry + diagnostics --------------------------
    terminal_state: Optional[ARMATerminalState] = None
    loglikelihood_: Optional[Array] = None
    aic_: Optional[Array] = None
    bic_: Optional[Array] = None
    n_train_: Optional[int] = None

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
        c: Optional[ArrayLike] = None,
        sigma_eps: Optional[ArrayLike] = None,
        residual_params: Optional[dict] = None,
        terminal_state: Optional[ARMATerminalState] = None,
        loglikelihood_: Optional[ArrayLike] = None,
        aic_: Optional[ArrayLike] = None,
        bic_: Optional[ArrayLike] = None,
        n_train_: Optional[int] = None,
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
        self.c = (
            jnp.asarray(c, dtype=float).reshape(()) if c is not None else None
        )
        self.sigma_eps = (
            jnp.asarray(sigma_eps, dtype=float).reshape(())
            if sigma_eps is not None else None
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
    # params property
    # ------------------------------------------------------------------
    @property
    def _stored_params(self) -> Optional[dict]:
        """Canonical parameter dict, or ``None`` for an unfitted instance.

        Schema:

        ``{
            "phi":        (p,),
            "theta":      (q,),
            "c":          (),
            "sigma_eps":  (),
            "residual":   {<shape-only dict matching the
                           StandardisedResidual schema>},
        }``
        """
        if (
            self.phi is None or self.theta is None or self.c is None
            or self.sigma_eps is None or self.residual_params is None
        ):
            return None
        return {
            "phi": self.phi,
            "theta": self.theta,
            "c": self.c,
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
        c: Array,
        init_y_lags: Array,
        init_eps_lags: Array,
    ) -> tuple[Array, Array, ARMATerminalState]:
        """Run :func:`run_arma` and wrap the terminal carry."""
        mu_seq, eps_seq, terminal = run_arma(
            y=y, phi=phi, theta=theta, c=c,
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

        Layout: ``[raw_phi (p,), raw_theta (q,), c (), raw_sigma_eps (),
        raw_residual_shape (n_shape_params)]``.
        """
        phi = jnp.asarray(params_dict["phi"], dtype=float).reshape(-1)
        theta = jnp.asarray(params_dict["theta"], dtype=float).reshape(-1)
        c = jnp.asarray(params_dict["c"], dtype=float).reshape(())
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
                c.reshape((1,)),
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

        Returns ``(phi, theta, c, sigma_eps, residual_shape_dict)`` —
        every component already pushed through its bound-enforcing
        reparameterisation.
        """
        idx = 0
        raw_phi = raw[idx : idx + self.p]
        idx += self.p
        raw_theta = raw[idx : idx + self.q]
        idx += self.q
        c = raw[idx]
        idx += 1
        raw_sigma = raw[idx]
        idx += 1
        raw_residual = raw[idx : idx + wrapper.n_shape_params]

        phi = raw_to_ar(raw_phi) if self.p > 0 else jnp.zeros((0,), dtype=float)
        theta = raw_to_ma(raw_theta) if self.q > 0 else jnp.zeros((0,), dtype=float)
        sigma_eps = raw_to_positive(raw_sigma)
        residual = wrapper.shape_params_from_array(raw_residual)
        return phi, theta, c, sigma_eps, residual

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
            phi, theta, c, sigma_eps, residual_shape = self._unpack_raw(
                raw, wrapper,
            )
            sigma_eps_safe = jnp.maximum(sigma_eps, _SIGMA_FLOOR)
            _, eps_seq, _ = run_arma(
                y=y, phi=phi, theta=theta, c=c,
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
            for key in ("phi", "theta", "c", "sigma_eps", "residual"):
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
                c=arma_seed["c"],
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
                "c": arma_seed["c"],
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
        phi, theta, c, sigma_eps, residual = self._unpack_raw(x_opt, wrapper)

        # Terminal state from a final pass at the optimum.
        _, _, terminal = self._run_recursion(
            y=y, phi=phi, theta=theta, c=c,
            init_y_lags=recursion_init_y_lags,
            init_eps_lags=recursion_init_eps_lags,
        )
        nll = objective(
            x_opt, y, recursion_init_y_lags, recursion_init_eps_lags,
        )
        params_dict = {
            "phi": phi,
            "theta": theta,
            "c": c,
            "sigma_eps": sigma_eps,
            "residual": residual,
        }
        return params_dict, terminal, nll

    def _resolve_orders_and_dist(
        self,
        p: Optional[int],
        q: Optional[int],
        residual_dist: Optional[Univariate],
    ) -> tuple[int, int, Univariate]:
        r"""Snapshot ``(p, q, residual_dist)`` for the fit, falling back
        to the singleton-level defaults when not supplied."""
        p_resolved = self.p if p is None else int(p)
        q_resolved = self.q if q is None else int(q)
        rd_resolved = self.residual_dist if residual_dist is None else residual_dist
        if p_resolved < 0 or q_resolved < 0:
            raise ValueError(
                f"AR / MA orders must be non-negative; got p={p_resolved}, q={q_resolved}."
            )
        return p_resolved, q_resolved, rd_resolved

    def fit(
        self,
        y: ArrayLike,
        *,
        p: Optional[int] = None,
        q: Optional[int] = None,
        residual_dist: Optional[Univariate] = None,
        init: str = "analytical",
        init_params: Optional[dict] = None,
        backcast_length: Optional[int] = None,
        maxiter: int = 200,
        lr: float = 0.05,
        name: Optional[str] = None,
    ) -> "ARMABase":
        r"""Fit the ARMA(p, q) model to a series via Adam-driven MLE.

        See module docstring for the full optimisation contract.

        Args:
            y: shape ``(n,)`` — observed return series.
            p: AR order; defaults to ``self.p``.  Static.
            q: MA order; defaults to ``self.q``.  Static.
            residual_dist: A :class:`Univariate` singleton on the
                whitelist (``normal``, ``student_t``, ``gen_normal``,
                ``nig``, ``gh``, ``skewed_t``).  Defaults to
                ``self.residual_dist``.  Static.
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
            A fitted :class:`ARMABase` instance with ``params``,
            ``terminal_state``, and the ``loglikelihood_`` /
            ``aic_`` / ``bic_`` / ``n_train_`` diagnostics populated.
        """
        self._check_method(init)
        p_resolved, q_resolved, rd_resolved = self._resolve_orders_and_dist(
            p, q, residual_dist,
        )

        # Build the family-static template that owns the resolved orders /
        # residual class so the fit graph keys on a stable trace identity.
        template = type(self)(
            name=self.name, p=p_resolved, q=q_resolved,
            residual_dist=rd_resolved,
        )
        wrapper = StandardisedResidual(rd_resolved)
        y_arr = template._validate_series(y)
        n = int(y_arr.shape[0])
        template._validate_backcast_length(backcast_length, n)

        params_dict, terminal_state, nll = template._fit_internal(
            y_arr, wrapper, init=init, init_params=init_params,
            backcast_length=backcast_length, maxiter=maxiter, lr=lr,
        )

        # Diagnostics: nll is the *mean* negative log-likelihood; the
        # log-likelihood SUM follows by re-multiplying by n.
        loglike = -nll * n
        n_params_total = (
            p_resolved + q_resolved + 1 + 1 + wrapper.n_shape_params
        )
        aic = 2.0 * n_params_total - 2.0 * loglike
        bic = n_params_total * jnp.log(jnp.asarray(n, dtype=float)) - 2.0 * loglike

        cls = type(self)
        if name is None:
            name = f"Fitted{cls.__name__}({p_resolved},{q_resolved})-{rd_resolved.name}"
        return cls(
            name=name,
            p=p_resolved,
            q=q_resolved,
            residual_dist=rd_resolved,
            phi=params_dict["phi"],
            theta=params_dict["theta"],
            c=params_dict["c"],
            sigma_eps=params_dict["sigma_eps"],
            residual_params=params_dict["residual"],
            terminal_state=terminal_state,
            loglikelihood_=loglike,
            aic_=aic,
            bic_=bic,
            n_train_=n,
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
            y_arr, self.phi, self.theta, self.c,
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
    ) -> Array:
        r"""Innovation residuals ``ε_t = y_t − μ_t`` over ``y``."""
        self._require_fitted()
        y_arr, init_y_lags, init_eps_lags = self._recursion_inputs(
            y, init, backcast_length,
        )
        _, eps_seq, _ = self._run_recursion(
            y_arr, self.phi, self.theta, self.c,
            init_y_lags=init_y_lags, init_eps_lags=init_eps_lags,
        )
        return eps_seq

    def standardised_residuals(
        self,
        y: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Innovation residuals divided by the fitted scale —
        :math:`z_t = \\varepsilon_t / \\sigma_\\varepsilon`.

        Convenient input for the variance-model fit when running the
        ARMA-GARCH two-stage pipeline manually.
        """
        eps = self.residuals(y, init=init, backcast_length=backcast_length)
        return eps / jnp.maximum(self.sigma_eps, _SIGMA_FLOOR)

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
            y_arr, self.phi, self.theta, self.c,
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
            # Roll the conditional-mean recursion forward h steps with
            # zero future innovations (analytic E[ε_{t+i}] = 0 under
            # the residual law).  Variance per step is sigma_eps^2;
            # the cumulative h-step *forecast* variance for an ARMA(p,q)
            # process equals sigma_eps^2 * sum_{j=0}^{h-1} psi_j^2,
            # where psi_j is the Wold MA(infty) representation.  For
            # the default one-step-ahead conditional variance we
            # return the per-step value; callers wanting cumulative
            # forecast variance should integrate this themselves
            # (or use simulation).
            zero_inputs = jnp.zeros((h,), dtype=float)
            mu_seq, _, _ = run_arma(
                y=zero_inputs + jnp.nan,  # ε path, will be overwritten
                phi=self.phi, theta=self.theta, c=self.c,
                init_y_lags=state.y_lags, init_eps_lags=state.eps_lags,
            )
            # The above used y=NaN to drive the scan; we actually want
            # *no* observed future, so we re-derive via a pure roll-out
            # that treats future ε as zero.
            mu_path = []
            y_lags = state.y_lags
            eps_lags = state.eps_lags
            for _ in range(h):
                ar_term = jnp.dot(self.phi, y_lags) if self.p > 0 else 0.0
                ma_term = jnp.dot(self.theta, eps_lags) if self.q > 0 else 0.0
                mu = self.c + ar_term + ma_term
                mu_path.append(mu)
                # Future innovations are zero under analytical mean
                # forecast — y_t = mu_t, eps_t = 0.
                if self.p > 0:
                    y_lags = jnp.concatenate(
                        [mu.reshape((1,)), y_lags[:-1]]
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
        ARMA recursion to produce a level-series path.
        """
        sigma = self.sigma_eps
        c = self.c
        phi = self.phi
        theta = self.theta

        def step(carry, z_t):
            y_lags, eps_lags = carry
            ar_term = jnp.dot(phi, y_lags) if self.p > 0 else 0.0
            ma_term = jnp.dot(theta, eps_lags) if self.q > 0 else 0.0
            mu = c + ar_term + ma_term
            eps_t = sigma * z_t
            y_t = mu + eps_t
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
        )
        ar_roots = ar_polynomial_roots(self.phi)
        ma_roots = ar_polynomial_roots(self.theta) if self.q > 0 else jnp.zeros((0,), dtype=jnp.complex64)
        is_stat = ar_is_stationary(self.phi)
        is_inv = ar_is_stationary(self.theta) if self.q > 0 else jnp.asarray(True)
        # Unconditional mean of an ARMA(p, q) with constant c is
        # E[y] = c / (1 - sum(phi)) when the AR polynomial is stationary.
        ar_factor = 1.0 - jnp.sum(self.phi) if self.p > 0 else 1.0
        ar_factor_safe = jnp.where(jnp.abs(ar_factor) < 1e-12, 1e-12, ar_factor)
        unconditional_mean = self.c / ar_factor_safe
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
            y_arr, self.phi, self.theta, self.c,
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
        :attr:`loglikelihood_`; pass ``y`` to recompute against a
        held-out series.
        """
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
        r"""Akaike Information Criterion.

        Returns the stored value when ``y`` is omitted; recomputes
        against ``y`` otherwise.
        """
        if y is None:
            self._require_fitted()
            return self.aic_
        ll = self._log_likelihood_on_series(y, init=init, backcast_length=backcast_length)
        return 2.0 * self.n_params - 2.0 * ll

    def bic(
        self,
        y: Optional[ArrayLike] = None,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Bayesian Information Criterion."""
        if y is None:
            self._require_fitted()
            return self.bic_
        y_arr = self._validate_series(y)
        ll = self._log_likelihood_on_series(y_arr, init=init, backcast_length=backcast_length)
        n = jnp.asarray(int(y_arr.shape[0]), dtype=float)
        return self.n_params * jnp.log(n) - 2.0 * ll

    @property
    def residual_distribution(self) -> Univariate:
        r"""The fitted standardised residual distribution as a regular
        :class:`Univariate` instance, ready for ``sample`` / ``logpdf``
        / ``cdf`` calls without going through the wrapper.
        """
        self._require_fitted()
        return self._wrapper().to_distribution(
            self.residual_params,
            name=f"{self.residual_dist.name}-stdresid-{self.name}",
        )
