r"""TGARCH(p, q) — threshold GARCH on σ (Zakoian 1994).

The recursion is on the conditional standard deviation
:math:`\sigma_t` (not :math:`\sigma^2_t`), with sign-split
asymmetry coefficients on the lagged innovations:

.. math::

    \sigma_t = \omega
             + \sum_{i=1}^p (\alpha^{+}_i\, \varepsilon^{+}_{t-i}
                           + \alpha^{-}_i\, \varepsilon^{-}_{t-i})
             + \sum_{j=1}^q \beta_j\, \sigma_{t-j},

with :math:`\varepsilon^{+} = \max(\varepsilon, 0)` and
:math:`\varepsilon^{-} = \max(-\varepsilon, 0)`.

Stationarity requires *first-moment* persistence (since the
recursion is on σ, not σ²):

.. math::

    \sum_i \bigl(\alpha^{+}_i \mathbb{E}[z^{+}]
                + \alpha^{-}_i \mathbb{E}[z^{-}]\bigr)
    + \sum_j \beta_j  < 1,

where :math:`\mathbb{E}[z^{+}] = \mathbb{E}[z \cdot \mathbf{1}\{z>0\}]`
and :math:`\mathbb{E}[z^{-}] = -\mathbb{E}[z \cdot \mathbf{1}\{z<0\}]`
under the standardised residual law.  Under symmetric residuals
:math:`\mathbb{E}[z^{+}] = \mathbb{E}[z^{-}] = \mathbb{E}[|z|] / 2`;
under skew they differ — both are computed via 100-point
Gauss-Legendre quadrature inside the fit objective so autograd
flows into the residual's shape parameters.

This is **not** the σ²-threshold variant of TGARCH — that
collapses to GJR-GARCH (a deliberate naming overlap in the
literature).  Use :class:`copulax.timeseries.GJR_GARCH` for the
σ²-form asymmetric model.

Reference:
    Zakoian, J.M. (1994).  *Threshold heteroskedastic models*.
    Journal of Economic Dynamics and Control, 18(5), 931-955.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src.optimize import projected_gradient
from copulax._src.timeseries._base import TerminalState
from copulax._src.timeseries._init import garch_pre_sample_state
from copulax._src.timeseries._recursions import run_tgarch
from copulax._src.timeseries._residuals._standardise import StandardisedResidual
from copulax._src.timeseries._stationarity import (
    positive_to_raw,
    raw_to_positive,
    tgarch_simplex,
    tgarch_unsimplex,
)
from copulax._src.timeseries._variance._garch_base import GARCHBase


_VAR_FLOOR: float = 1e-12
_SIGMA_FLOOR: float = 1e-6


class TGARCHTerminalState(TerminalState):
    r"""Constant-size carry for TGARCH ``forecast(h)``.

    Stores last ``p`` positive-part innovations
    :math:`\varepsilon^{+}`, last ``p`` negative-part innovations
    :math:`\varepsilon^{-}`, and last ``q`` conditional standard
    deviations :math:`\sigma`.
    """
    eps_pos_lags: Array
    eps_neg_lags: Array
    sigma_lags: Array


class TGARCH(GARCHBase):
    r"""TGARCH(p, q) σ-form recursion (Zakoian 1994).

    Construct with the desired orders and residual law:

    .. code-block:: python

        from copulax.timeseries import TGARCH
        from copulax.univariate import skewed_t
        fit = TGARCH(p=1, q=1, residual_dist=skewed_t).fit(eps)
    """

    # The σ-form positive-shock coefficient is stored on the inherited
    # ``self.alpha`` field for code reuse with the GARCHBase scaffolding;
    # the user-facing kwarg / schema key is ``alpha_pos`` to match
    # Zakoian (1994) and the joint-composite reconstructor (which
    # splats schema keys into the constructor).  ``alpha_neg`` is the
    # new field for the negative-shock coefficient.
    alpha_neg: Optional[Array] = None
    terminal_state: Optional[TGARCHTerminalState] = None

    def __init__(
        self,
        p: int = 0,
        q: int = 0,
        *,
        residual_dist: Optional[Univariate] = None,
        name: str = "TGARCH",
        omega=None,
        alpha_pos=None,
        alpha_neg=None,
        beta=None,
        residual_params=None,
        terminal_state: Optional[TGARCHTerminalState] = None,
        n_train_: Optional[int] = None,
        cov_matrix_=None,
        standard_errors_=None,
        residual_diagnostics_=None,
    ):
        super().__init__(
            name=name,
            p=p,
            q=q,
            residual_dist=residual_dist,
            omega=omega,
            alpha=alpha_pos,
            beta=beta,
            residual_params=residual_params,
            terminal_state=terminal_state,
            n_train_=n_train_,
            cov_matrix_=cov_matrix_,
            standard_errors_=standard_errors_,
            residual_diagnostics_=residual_diagnostics_,
        )
        self.alpha_neg = (
            jnp.asarray(alpha_neg, dtype=float).reshape(-1)
            if alpha_neg is not None else None
        )

    @property
    def _stored_params(self) -> Optional[dict]:
        r"""Canonical params dict.

        Schema:

        ``{
            "omega":      (),
            "alpha_pos":  (p,),    (named ``alpha`` in the family
                                    constructor for GARCHBase symmetry)
            "alpha_neg":  (p,),
            "beta":       (q,),
            "residual":   {<shape-only dict>},
        }``
        """
        if (
            self.omega is None or self.alpha is None or self.alpha_neg is None
            or self.beta is None or self.residual_params is None
        ):
            return None
        return {
            "omega": self.omega,
            "alpha_pos": self.alpha,
            "alpha_neg": self.alpha_neg,
            "beta": self.beta,
            "residual": dict(self.residual_params),
        }

    @property
    def n_params(self) -> int:
        wrapper = StandardisedResidual(self.residual_dist)
        return 1 + 2 * self.p + self.q + wrapper.n_shape_params

    # ------------------------------------------------------------------
    # Reparameterisation pack / unpack
    # ------------------------------------------------------------------
    def _pack_x0_tgarch(
        self, params_dict: dict, wrapper: StandardisedResidual,
    ) -> Array:
        omega = jnp.asarray(params_dict["omega"], dtype=float).reshape(())
        alpha_pos = jnp.asarray(params_dict["alpha_pos"], dtype=float).reshape(-1)
        alpha_neg = jnp.asarray(params_dict["alpha_neg"], dtype=float).reshape(-1)
        beta = jnp.asarray(params_dict["beta"], dtype=float).reshape(-1)
        residual = params_dict.get("residual", {}) or {}

        raw_omega = positive_to_raw(jnp.maximum(omega, _SIGMA_FLOOR))
        e_pos = wrapper.expected_z_pos(residual)
        e_neg = wrapper.expected_z_neg(residual)
        raw_persistence, raw_weights = tgarch_unsimplex(
            alpha_pos, alpha_neg, beta, e_pos, e_neg,
        )
        raw_residual = wrapper.shape_params_to_array(residual)
        return jnp.concatenate(
            [
                raw_omega.reshape((1,)),
                raw_persistence.reshape((1,)),
                raw_weights,
                raw_residual,
            ]
        )

    def _unpack_raw_tgarch(
        self, raw: Array, wrapper: StandardisedResidual,
    ) -> tuple[Array, Array, Array, Array, dict]:
        idx = 0
        raw_omega = raw[idx]
        idx += 1
        raw_persistence = raw[idx]
        idx += 1
        raw_weights = raw[idx : idx + 2 * self.p + self.q]
        idx += 2 * self.p + self.q
        raw_residual = raw[idx : idx + wrapper.n_shape_params]

        residual = wrapper.shape_params_from_array(raw_residual)
        e_pos = wrapper.expected_z_pos(residual)
        e_neg = wrapper.expected_z_neg(residual)

        omega = raw_to_positive(raw_omega)
        alpha_pos, alpha_neg, beta = tgarch_simplex(
            raw_persistence, raw_weights,
            p=self.p, q=self.q, e_pos=e_pos, e_neg=e_neg,
        )
        return omega, alpha_pos, alpha_neg, beta, residual

    # ------------------------------------------------------------------
    # Recursion + initial state
    # ------------------------------------------------------------------
    def _initial_state_tgarch(
        self,
        eps: Array,
        mode: str,
        backcast_length: Optional[int],
    ) -> tuple[Array, Array, Array]:
        eps_sq_lags, var_lags = garch_pre_sample_state(
            eps, p=self.p, q=self.q,
            mode=mode, backcast_length=backcast_length,
        )
        # Convert variance anchor to σ; split equally between the
        # positive / negative ε buffers (correct under symmetric
        # innovations and a small bias for skewed ones — absorbed by
        # the recursion's first ``p`` steps).
        sigma_anchor = jnp.sqrt(jnp.maximum(eps_sq_lags[0] if self.p > 0
                                            else var_lags[0], _VAR_FLOOR))
        eps_pos_lags = jnp.full((self.p,), 0.5 * sigma_anchor)
        eps_neg_lags = jnp.full((self.p,), 0.5 * sigma_anchor)
        sigma_lags = jnp.full((self.q,), sigma_anchor)
        return eps_pos_lags, eps_neg_lags, sigma_lags

    def _run_recursion_tgarch(
        self,
        eps: Array,
        omega: Array,
        alpha_pos: Array,
        alpha_neg: Array,
        beta: Array,
        init_state: tuple[Array, Array, Array],
    ) -> tuple[Array, TGARCHTerminalState]:
        eps_pos, eps_neg, sigma_lags = init_state
        sigma_seq, terminal = run_tgarch(
            eps=eps, omega=omega,
            alpha_pos=alpha_pos, alpha_neg=alpha_neg, beta=beta,
            init_eps_pos_lags=eps_pos,
            init_eps_neg_lags=eps_neg,
            init_sigma_lags=sigma_lags,
        )
        # We expose σ² as the canonical "conditional_variance"
        # downstream; the kernel returns σ.
        return sigma_seq, TGARCHTerminalState(
            eps_pos_lags=terminal[0],
            eps_neg_lags=terminal[1],
            sigma_lags=terminal[2],
        )

    # ------------------------------------------------------------------
    # Cold-start
    # ------------------------------------------------------------------
    def _build_cold_start(
        self,
        eps: Array,
        wrapper: StandardisedResidual,
        init: str,
        backcast_length: Optional[int],
    ) -> dict:
        r"""Cold-start params for TGARCH.

        Symmetric prior: ``α^+ = α^-`` summing with ``β`` to a
        modest persistence (~0.95 in σ-units).  ``ω`` set so the
        unconditional σ matches the sample standard deviation.
        """
        eps_arr = jnp.asarray(eps, dtype=float)
        sigma_sample = jnp.std(eps_arr)
        e_pos_default = wrapper.expected_z_pos(wrapper.example_shape_params())
        e_neg_default = wrapper.expected_z_neg(wrapper.example_shape_params())

        beta_share = 0.85
        # Split residual share between α^+ and α^- equally; persistence
        # weight on each is E[z^+] / E[z^-].
        residual_persistence = 0.10
        if self.p > 0:
            alpha_pos = jnp.full(
                (self.p,),
                0.5 * residual_persistence / (self.p * jnp.maximum(e_pos_default, 1e-6)),
            )
            alpha_neg = jnp.full(
                (self.p,),
                0.5 * residual_persistence / (self.p * jnp.maximum(e_neg_default, 1e-6)),
            )
        else:
            alpha_pos = jnp.zeros((0,), dtype=float)
            alpha_neg = jnp.zeros((0,), dtype=float)
        beta = jnp.full((self.q,), beta_share / max(self.q, 1), dtype=float) if self.q > 0 else jnp.zeros((0,), dtype=float)
        # ω = (1 - persistence) · sample σ
        persistence = (
            jnp.sum(e_pos_default * alpha_pos)
            + jnp.sum(e_neg_default * alpha_neg)
            + jnp.sum(beta)
        )
        omega = (1.0 - persistence) * sigma_sample
        return {
            "omega": omega,
            "alpha_pos": alpha_pos,
            "alpha_neg": alpha_neg,
            "beta": beta,
            "residual": wrapper.example_shape_params(),
        }

    def _make_objective_tgarch(self, wrapper: StandardisedResidual):
        def objective(
            raw: Array,
            eps: Array,
            init_eps_pos_lags: Array,
            init_eps_neg_lags: Array,
            init_sigma_lags: Array,
        ) -> Array:
            omega, alpha_pos, alpha_neg, beta, residual_shape = self._unpack_raw_tgarch(
                raw, wrapper,
            )
            init_state = (init_eps_pos_lags, init_eps_neg_lags, init_sigma_lags)
            sigma_seq, _ = self._run_recursion_tgarch(
                eps, omega, alpha_pos, alpha_neg, beta, init_state,
            )
            sigma_seq = jnp.maximum(sigma_seq, _SIGMA_FLOOR)
            z = eps / sigma_seq
            logpdf = wrapper.logpdf(z, residual_shape) - jnp.log(sigma_seq)
            finite = jnp.isfinite(logpdf)
            safe_logpdf = jnp.where(finite, logpdf, 0.0)
            invalid_penalty = 1e6 * (~finite).mean()
            return -safe_logpdf.mean() + invalid_penalty
        return objective

    def fit(
        self,
        eps: ArrayLike,
        *,
        init: str = "analytical",
        init_params: Optional[dict] = None,
        backcast_length: Optional[int] = None,
        maxiter: int = 200,
        lr: float = 0.05,
        name: Optional[str] = None,
    ) -> "TGARCH":
        r"""Fit TGARCH(p, q) to a mean-corrected innovation series."""
        self._check_method(init)
        wrapper = StandardisedResidual(self.residual_dist)
        eps_arr = self._validate_series(eps)
        n = int(eps_arr.shape[0])
        self._validate_backcast_length(backcast_length, n)

        if init == "warm":
            if init_params is None:
                raise ValueError(
                    "init='warm' requires init_params (a parameter dict "
                    "matching the schema returned by `model.params`)."
                )
            cold = dict(init_params)
            for key in ("omega", "alpha_pos", "alpha_neg", "beta", "residual"):
                if key not in cold:
                    raise KeyError(
                        f"Warm-start init_params missing required key {key!r}."
                    )
        else:
            cold = self._build_cold_start(
                eps_arr, wrapper, init=init, backcast_length=backcast_length,
            )

        x0 = self._pack_x0_tgarch(cold, wrapper)
        _state_mode = "sample" if init == "sample" else "backcast"
        init_eps_pos, init_eps_neg, init_sigma = self._initial_state_tgarch(
            eps_arr, mode=_state_mode, backcast_length=backcast_length,
        )

        objective = self._make_objective_tgarch(wrapper)
        res = projected_gradient(
            f=objective,
            x0=x0,
            projection_method="projection_box",
            projection_options={
                "lower": jnp.full((x0.shape[0], 1), -jnp.inf),
                "upper": jnp.full((x0.shape[0], 1), jnp.inf),
            },
            eps=eps_arr,
            init_eps_pos_lags=init_eps_pos,
            init_eps_neg_lags=init_eps_neg,
            init_sigma_lags=init_sigma,
            lr=lr,
            maxiter=maxiter,
        )
        x_opt = res["x"]
        omega, alpha_pos, alpha_neg, beta, residual = self._unpack_raw_tgarch(
            x_opt, wrapper,
        )

        sigma_seq, terminal = self._run_recursion_tgarch(
            eps_arr, omega, alpha_pos, alpha_neg, beta,
            init_state=(init_eps_pos, init_eps_neg, init_sigma),
        )
        nll = objective(
            x_opt, eps_arr, init_eps_pos, init_eps_neg, init_sigma,
        )
        loglike = -nll * n
        n_params_total = 1 + 2 * self.p + self.q + wrapper.n_shape_params
        aic = 2.0 * n_params_total - 2.0 * loglike
        bic = (
            n_params_total * jnp.log(jnp.asarray(n, dtype=float))
            - 2.0 * loglike
        )

        # Standardised training-window residuals + observed-Hessian
        # SEs.  TGARCH stores params under the "alpha_pos" / "alpha_neg"
        # keys (see ``_stored_params``); the SE pipeline keys off
        # ``_ag_var_keys`` which the variant should override to expose
        # this naming.
        sigma_train = jnp.maximum(sigma_seq, _SIGMA_FLOOR)
        z_train = eps_arr / sigma_train
        params_dict = {
            "omega": omega, "alpha_pos": alpha_pos, "alpha_neg": alpha_neg,
            "beta": beta, "residual": residual,
        }
        cov, se_dict, diagnostics = self._post_fit_se_and_diagnostics(
            params_dict=params_dict,
            wrapper=wrapper, eps_arr=eps_arr,
            init_state=(init_eps_pos, init_eps_neg, init_sigma),
            z_train=z_train,
            loglikelihood=loglike, aic=aic, bic=bic,
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
            residual_dist=self.residual_dist,
            omega=omega,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            beta=beta,
            residual_params=residual,
            terminal_state=terminal,
            n_train_=n,
            cov_matrix_=cov,
            standard_errors_=se_dict,
            residual_diagnostics_=diagnostics,
        )

    # ------------------------------------------------------------------
    # Conditional moments / residuals
    # ------------------------------------------------------------------
    def _tgarch_recursion_inputs(
        self,
        eps: ArrayLike,
        init: str,
        backcast_length: Optional[int],
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        eps_arr = self._validate_series(eps)
        n = int(eps_arr.shape[0])
        self._validate_backcast_length(backcast_length, n)
        init_state = self._initial_state_tgarch(
            eps_arr, mode=init, backcast_length=backcast_length,
        )
        return eps_arr, init_state

    def conditional_variance(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        r"""Returns σ²_t (squared conditional standard deviation)."""
        self._require_fitted()
        eps_arr, init_state = self._tgarch_recursion_inputs(
            eps, init, backcast_length,
        )
        sigma_seq, _ = self._run_recursion_tgarch(
            eps_arr, self.omega, self.alpha, self.alpha_neg, self.beta,
            init_state,
        )
        return sigma_seq ** 2

    def residuals(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> dict:
        self._require_fitted()
        eps_arr, init_state = self._tgarch_recursion_inputs(
            eps, init, backcast_length,
        )
        sigma_seq, _ = self._run_recursion_tgarch(
            eps_arr, self.omega, self.alpha, self.alpha_neg, self.beta,
            init_state,
        )
        sigma_safe = jnp.maximum(sigma_seq, _SIGMA_FLOOR)
        return {
            "residuals": eps_arr,
            "standardised_residuals": eps_arr / sigma_safe,
        }

    def terminal_state_from(
        self,
        eps: ArrayLike,
        *,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> TGARCHTerminalState:
        self._require_fitted()
        eps_arr, init_state = self._tgarch_recursion_inputs(
            eps, init, backcast_length,
        )
        _, terminal = self._run_recursion_tgarch(
            eps_arr, self.omega, self.alpha, self.alpha_neg, self.beta,
            init_state,
        )
        return terminal

    # ------------------------------------------------------------------
    # Loglikelihood / aic / bic
    # ------------------------------------------------------------------
    def _log_likelihood_on_series(
        self,
        eps: ArrayLike,
        init: str = "backcast",
        backcast_length: Optional[int] = None,
    ) -> Array:
        self._require_fitted()
        wrapper = self._wrapper()
        eps_arr, init_state = self._tgarch_recursion_inputs(
            eps, init, backcast_length,
        )
        sigma_seq, _ = self._run_recursion_tgarch(
            eps_arr, self.omega, self.alpha, self.alpha_neg, self.beta,
            init_state,
        )
        sigma_safe = jnp.maximum(sigma_seq, _SIGMA_FLOOR)
        z = eps_arr / sigma_safe
        logpdf = wrapper.logpdf(z, self.residual_params) - jnp.log(sigma_safe)
        return jnp.sum(logpdf)

    # ------------------------------------------------------------------
    # Forecast — h=1 closed form, h>=2 simulation only
    # ------------------------------------------------------------------
    def forecast(
        self,
        h: int,
        *,
        method: str = "analytical",
        n_paths: int = 0,
        key: Optional[Array] = None,
        last_state: Optional[TGARCHTerminalState] = None,
    ) -> dict:
        r"""``h``-step-ahead conditional moments.

        Note:
            ``method="analytical"`` is supported only at ``h = 1``.
            For ``h >= 2`` the σ-form recursion's expectation
            requires :math:`\mathbb{E}[z^{+}]^k`,
            :math:`\mathbb{E}[z^{-}]^k` cross-products that have
            closed forms only under normal residuals — not provided
            in v1.  Use ``method="simulation"`` for any horizon
            beyond 1.
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
        mean = jnp.zeros((h,), dtype=float)

        if method == "analytical":
            if h == 1:
                ar_term_pos = (
                    jnp.dot(self.alpha, state.eps_pos_lags)
                    if self.p > 0 else 0.0
                )
                ar_term_neg = (
                    jnp.dot(self.alpha_neg, state.eps_neg_lags)
                    if self.p > 0 else 0.0
                )
                ma_term = (
                    jnp.dot(self.beta, state.sigma_lags)
                    if self.q > 0 else 0.0
                )
                sigma_t1 = self.omega + ar_term_pos + ar_term_neg + ma_term
                sigma_t1 = jnp.maximum(sigma_t1, _SIGMA_FLOOR)
                variance = (sigma_t1 ** 2).reshape((1,))
                return {"mean": mean, "variance": variance, "paths": None}
            raise ValueError(
                "TGARCH analytical forecast for h>=2 requires closed-form "
                "moments of z⁺ / z⁻ that are only available for normal "
                "residuals; use method='simulation'."
            )

        elif method == "simulation":
            if n_paths <= 0:
                raise ValueError("method='simulation' requires n_paths > 0.")
            from copulax._src._utils import _resolve_key
            key = _resolve_key(key)
            paths = self.rvs(
                size=(int(n_paths), h), key=key, last_state=state,
            )
            mc_mean = jnp.mean(paths, axis=0)
            mc_var = jnp.var(paths, axis=0)
            return {"mean": mc_mean, "variance": mc_var, "paths": paths}

        else:
            raise ValueError(
                f"Unknown forecast method {method!r}; expected "
                "'analytical' or 'simulation'."
            )

    # ------------------------------------------------------------------
    # rvs roll-path
    # ------------------------------------------------------------------
    def _roll_path(self, z: Array, state: TGARCHTerminalState) -> Array:
        omega = self.omega
        alpha_pos = self.alpha
        alpha_neg = self.alpha_neg
        beta = self.beta

        def step(carry, z_t):
            eps_pos_lags, eps_neg_lags, sigma_lags = carry
            ar_term_pos = (
                jnp.dot(alpha_pos, eps_pos_lags) if self.p > 0 else 0.0
            )
            ar_term_neg = (
                jnp.dot(alpha_neg, eps_neg_lags) if self.p > 0 else 0.0
            )
            ma_term = jnp.dot(beta, sigma_lags) if self.q > 0 else 0.0
            sigma_t = omega + ar_term_pos + ar_term_neg + ma_term
            sigma_t = jnp.maximum(sigma_t, _SIGMA_FLOOR)
            eps_t = sigma_t * z_t
            eps_pos_t = jnp.maximum(eps_t, 0.0)
            eps_neg_t = jnp.maximum(-eps_t, 0.0)
            new_eps_pos = (
                jnp.concatenate([eps_pos_t.reshape((1,)), eps_pos_lags[:-1]])
                if self.p > 0 else eps_pos_lags
            )
            new_eps_neg = (
                jnp.concatenate([eps_neg_t.reshape((1,)), eps_neg_lags[:-1]])
                if self.p > 0 else eps_neg_lags
            )
            new_sigma = (
                jnp.concatenate([sigma_t.reshape((1,)), sigma_lags[:-1]])
                if self.q > 0 else sigma_lags
            )
            return (new_eps_pos, new_eps_neg, new_sigma), eps_t

        init_carry = (
            state.eps_pos_lags, state.eps_neg_lags, state.sigma_lags,
        )
        _, eps_seq = jax.lax.scan(step, init_carry, z)
        return eps_seq

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        r"""Analytic, parameter-only diagnostics for TGARCH.

        Persistence:
        :math:`\sum \alpha^{+} \mathbb{E}[z^{+}]
              + \sum \alpha^{-} \mathbb{E}[z^{-}]
              + \sum \beta`,
        with :math:`\mathbb{E}[z^{+}]` and :math:`\mathbb{E}[z^{-}]`
        from the fitted residual law.

        Note:
            The unconditional **standard deviation** of ``ε`` is
            :math:`\omega / (1 - \text{persistence})` — surfaced as
            ``unconditional_sigma``.  The unconditional **variance**
            is :math:`\mathbb{E}[\sigma_t^2] = \mathbb{E}[\sigma_t]^2
            + \mathrm{Var}[\sigma_t]`, which depends on higher
            moments of ``z`` and is not closed-form for non-normal
            residuals; we expose ``unconditional_variance``
            assuming :math:`\sigma_t` is approximately constant
            (i.e. equal to ``unconditional_sigma**2``).  For the
            exact value use a simulation-based forecast or refer to
            Zakoian's TGARCH moments paper.
        """
        self._require_fitted()
        wrapper = self._wrapper()
        e_pos = wrapper.expected_z_pos(self.residual_params)
        e_neg = wrapper.expected_z_neg(self.residual_params)
        persistence = (
            jnp.sum(e_pos * self.alpha)
            + jnp.sum(e_neg * self.alpha_neg)
            + jnp.sum(self.beta)
        )
        is_stat = persistence < 1.0
        denom = jnp.where(is_stat, 1.0 - persistence, _VAR_FLOOR)
        unconditional_sigma = jnp.where(
            is_stat, self.omega / denom, jnp.inf,
        )
        log_pers = jnp.log(jnp.maximum(persistence, _VAR_FLOOR))
        half_life = jnp.where(
            jnp.logical_and(is_stat, persistence > 0.0),
            jnp.log(0.5) / log_pers,
            jnp.inf,
        )
        return {
            "unconditional_sigma": unconditional_sigma,
            "unconditional_variance": unconditional_sigma ** 2,
            "persistence": persistence,
            "expected_z_pos": e_pos,
            "expected_z_neg": e_neg,
            "half_life": half_life,
            "is_stationary": is_stat,
        }

    # ------------------------------------------------------------------
    # ArmaGarch backend — TGARCH-specific overrides
    # ------------------------------------------------------------------
    def _ag_var_keys(self) -> tuple:
        return ("omega", "alpha_pos", "alpha_neg", "beta")

    def _ag_n_raw(self, wrapper: StandardisedResidual) -> int:
        return 1 + 1 + 2 * self.p + self.q

    def _ag_pack_x0(
        self,
        var_params: dict,
        wrapper: StandardisedResidual,
        residual_params: dict,
    ) -> Array:
        omega = jnp.asarray(var_params["omega"], dtype=float).reshape(())
        alpha_pos = jnp.asarray(var_params["alpha_pos"], dtype=float).reshape(-1)
        alpha_neg = jnp.asarray(var_params["alpha_neg"], dtype=float).reshape(-1)
        beta = jnp.asarray(var_params["beta"], dtype=float).reshape(-1)
        raw_omega = positive_to_raw(jnp.maximum(omega, _SIGMA_FLOOR))
        e_pos = wrapper.expected_z_pos(residual_params)
        e_neg = wrapper.expected_z_neg(residual_params)
        raw_persistence, raw_weights = tgarch_unsimplex(
            alpha_pos, alpha_neg, beta, e_pos, e_neg,
        )
        return jnp.concatenate(
            [raw_omega.reshape((1,)), raw_persistence.reshape((1,)), raw_weights]
        )

    def _ag_unpack_raw(
        self,
        raw_section: Array,
        wrapper: StandardisedResidual,
        residual_params: dict,
    ) -> dict:
        idx = 0
        raw_omega = raw_section[idx]
        idx += 1
        raw_persistence = raw_section[idx]
        idx += 1
        raw_weights = raw_section[idx : idx + 2 * self.p + self.q]
        e_pos = wrapper.expected_z_pos(residual_params)
        e_neg = wrapper.expected_z_neg(residual_params)
        omega = raw_to_positive(raw_omega)
        alpha_pos, alpha_neg, beta = tgarch_simplex(
            raw_persistence, raw_weights,
            p=self.p, q=self.q, e_pos=e_pos, e_neg=e_neg,
        )
        return {
            "omega": omega,
            "alpha_pos": alpha_pos,
            "alpha_neg": alpha_neg,
            "beta": beta,
        }

    def _ag_initial_state(
        self,
        eps_proxy: Array,
        mode: str,
        backcast_length: Optional[int],
        residual_params: dict,
    ) -> tuple:
        return self._initial_state_tgarch(
            eps_proxy, mode=mode, backcast_length=backcast_length,
        )

    def _ag_run_recursion(
        self,
        eps_seq: Array,
        var_params: dict,
        residual_params: dict,
        init_state: tuple,
    ) -> tuple[Array, tuple]:
        omega = var_params["omega"]
        alpha_pos = var_params["alpha_pos"]
        alpha_neg = var_params["alpha_neg"]
        beta = var_params["beta"]
        sigma_seq, terminal = run_tgarch(
            eps=eps_seq, omega=omega,
            alpha_pos=alpha_pos, alpha_neg=alpha_neg, beta=beta,
            init_eps_pos_lags=init_state[0],
            init_eps_neg_lags=init_state[1],
            init_sigma_lags=init_state[2],
        )
        # Return σ² (squared σ) so downstream sees the same units as
        # the σ²-form variants.
        var_seq = sigma_seq ** 2
        return var_seq, (terminal[0], terminal[1], terminal[2])

    def _ag_cold_start(
        self,
        eps_proxy: Array,
        mode: str,
        backcast_length: Optional[int],
        wrapper: StandardisedResidual,
    ) -> dict:
        base = self._build_cold_start(
            eps_proxy, wrapper, init=mode, backcast_length=backcast_length,
        )
        return {k: v for k, v in base.items() if k != "residual"}

    def _ag_forecast_step(
        self,
        var_params: dict,
        residual_params: dict,
        terminal_state: tuple,
    ) -> tuple[Array, tuple]:
        r"""1-step closed-form forecast in σ-form.  ``h ≥ 2``
        requires moments of ``z⁺ / z⁻`` that have closed form only
        for Normal residuals; ArmaGarch routes ``h ≥ 2`` through
        simulation."""
        omega = var_params["omega"]
        alpha_pos = var_params["alpha_pos"]
        alpha_neg = var_params["alpha_neg"]
        beta = var_params["beta"]
        eps_pos_lags, eps_neg_lags, sigma_lags = terminal_state
        ar_pos = jnp.dot(alpha_pos, eps_pos_lags) if self.p > 0 else 0.0
        ar_neg = jnp.dot(alpha_neg, eps_neg_lags) if self.p > 0 else 0.0
        ma_term = jnp.dot(beta, sigma_lags) if self.q > 0 else 0.0
        sigma_next = jnp.maximum(omega + ar_pos + ar_neg + ma_term, _SIGMA_FLOOR)
        var_next = sigma_next ** 2
        # State updates use σ_next · E[z⁺] / E[z⁻] as the
        # "expected ε⁺ / ε⁻" — only correct at h=1.
        wrapper = self._wrapper()
        e_pos = wrapper.expected_z_pos(residual_params)
        e_neg = wrapper.expected_z_neg(residual_params)
        new_eps_pos_lags = (
            jnp.concatenate(
                [(sigma_next * e_pos).reshape((1,)), eps_pos_lags[:-1]]
            )
            if self.p > 0 else eps_pos_lags
        )
        new_eps_neg_lags = (
            jnp.concatenate(
                [(sigma_next * e_neg).reshape((1,)), eps_neg_lags[:-1]]
            )
            if self.p > 0 else eps_neg_lags
        )
        new_sigma_lags = (
            jnp.concatenate([sigma_next.reshape((1,)), sigma_lags[:-1]])
            if self.q > 0 else sigma_lags
        )
        return var_next, (new_eps_pos_lags, new_eps_neg_lags, new_sigma_lags)

    def _ag_rvs_step(
        self,
        var_params: dict,
        residual_params: dict,
        terminal_state: tuple,
        z_t: Array,
    ) -> tuple[Array, Array, tuple]:
        omega = var_params["omega"]
        alpha_pos = var_params["alpha_pos"]
        alpha_neg = var_params["alpha_neg"]
        beta = var_params["beta"]
        eps_pos_lags, eps_neg_lags, sigma_lags = terminal_state
        ar_pos = jnp.dot(alpha_pos, eps_pos_lags) if self.p > 0 else 0.0
        ar_neg = jnp.dot(alpha_neg, eps_neg_lags) if self.p > 0 else 0.0
        ma_term = jnp.dot(beta, sigma_lags) if self.q > 0 else 0.0
        sigma_t = jnp.maximum(omega + ar_pos + ar_neg + ma_term, _SIGMA_FLOOR)
        var_t = sigma_t ** 2
        eps_t = sigma_t * z_t
        eps_t_pos = jnp.maximum(eps_t, 0.0)
        eps_t_neg = jnp.maximum(-eps_t, 0.0)
        new_eps_pos_lags = (
            jnp.concatenate([eps_t_pos.reshape((1,)), eps_pos_lags[:-1]])
            if self.p > 0 else eps_pos_lags
        )
        new_eps_neg_lags = (
            jnp.concatenate([eps_t_neg.reshape((1,)), eps_neg_lags[:-1]])
            if self.p > 0 else eps_neg_lags
        )
        new_sigma_lags = (
            jnp.concatenate([sigma_t.reshape((1,)), sigma_lags[:-1]])
            if self.q > 0 else sigma_lags
        )
        return var_t, eps_t, (new_eps_pos_lags, new_eps_neg_lags, new_sigma_lags)

    @staticmethod
    def _ag_supports_analytical_h_step() -> bool:
        return False  # σ-form needs higher moments of z⁺/z⁻; h≥2 → simulation.

    def _ag_var_terminal_state_class(self) -> type:
        return TGARCHTerminalState

    @classmethod
    def _deserialise_extra_kwargs(cls, params: dict) -> dict:
        # TGARCH's params dict uses the schema names ``alpha_pos`` /
        # ``alpha_neg`` and its ``__init__`` accepts the same names —
        # the base ``_deserialise`` does not handle these because they
        # aren't in the GARCH/IGARCH schema, so we thread them in here.
        return {
            "alpha_pos": params.get("alpha_pos"),
            "alpha_neg": params.get("alpha_neg"),
        }
