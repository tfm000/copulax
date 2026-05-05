"""End-to-end tests for the joint ARMA-GARCH composite estimator.

Coverage:

* Constructor validation: order tuples, variance whitelist,
  GARCH-M rejection, default residual law.
* Parameter recovery on simulated data, asserted within an
  asymptotic standard-error budget rather than a hand-tuned
  ``atol``.
* ``n_params`` matches the sum of fitted-parameter sizes across an
  (mean_order, variance_variant, var_order, residual_law) grid.
* Joint MLE log-likelihood is at least as high as the two-stage
  separable fit on the same data, verified by evaluating the joint
  objective at the separable parameter point via warm-start with
  ``maxiter=0``.
* Residual contract: ``residuals(y)["residuals"] == y - mean``,
  standardised residuals divide by ``sqrt(conditional_variance)``,
  empirical mean / variance close to ``(0, 1)`` for every config.
* Cached residual diagnostics (loglikelihood, aic, bic, acf, pacf,
  ljung_box, ljung_box_sq, arch_lm, adf, kpss) match the explicit
  ``y``-recompute paths.
* ``stats()`` values match the analytical formulas for each
  variance variant; AR / MA root moduli match ``np.roots``.
* ``forecast`` is finite, converges to the unconditional mean and
  variance at long horizons, and the simulation path average
  matches the analytical mean / variance within Monte-Carlo error.
* ``rvs`` is deterministic under fixed ``u``, batches correctly,
  and the long-run sample moments match the unconditional moments.
* JIT compatibility of conditional moments, residuals,
  log-likelihood, forecast, and rvs.
* Warm-start contract: zero iterations reproduce input parameters,
  short refit reaches the cold log-likelihood, missing keys raise.
* Each ``init`` mode runs cleanly; an unknown mode raises.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copulax.timeseries import (
    ARMA,
    ArmaGarch,
    EGARCH,
    GARCH,
    GARCH_M,
    GJR_GARCH,
    IGARCH,
    QGARCH,
    TGARCH,
)
from copulax.univariate import gh, normal, skewed_t, student_t
from copulax._src.timeseries._residuals._standardise import (
    StandardisedResidual,
)


# ---------------------------------------------------------------------------
# Truth parameters
# ---------------------------------------------------------------------------

_TRUTH_GARCH = {
    "phi": (0.5,), "theta": (0.3,), "c": 0.05,
    "omega": 0.05, "alpha": (0.10,), "beta": (0.85,),
}
_TRUTH_GARCH_AR2 = {
    "phi": (0.4, 0.2), "theta": (0.3,), "c": 0.05,
    "omega": 0.05, "alpha": (0.10,), "beta": (0.85,),
}
_TRUTH_GARCH_PQ12 = {
    "phi": (0.5,), "theta": (0.3,), "c": 0.05,
    "omega": 0.05, "alpha": (0.05,), "beta": (0.45, 0.45),
}
_TRUTH_GARCH_22 = {
    "phi": (0.4, 0.2), "theta": (0.3, 0.1), "c": 0.05,
    "omega": 0.05, "alpha": (0.10,), "beta": (0.85,),
}
_TRUTH_IGARCH = {
    "phi": (0.5,), "theta": (0.3,), "c": 0.05,
    "omega": 0.02, "alpha": (0.10,), "beta": (0.90,),
}
_TRUTH_GJR = {
    "phi": (0.5,), "theta": (0.3,), "c": 0.05,
    "omega": 0.05, "alpha": (0.05,), "gamma": (0.10,), "beta": (0.85,),
}
_TRUTH_EGARCH = {
    "phi": (0.5,), "theta": (0.3,), "c": 0.05,
    "omega": 0.0, "alpha": (-0.05,), "gamma": (0.10,), "beta": (0.95,),
}
_TRUTH_TGARCH = {
    "phi": (0.5,), "theta": (0.3,), "c": 0.05,
    "omega": 0.02, "alpha_pos": (0.10,), "alpha_neg": (0.20,),
    "beta": (0.70,),
}
_TRUTH_QGARCH = {
    "phi": (0.5,), "theta": (0.3,), "c": 0.05,
    "omega": 0.05, "alpha": (0.10,), "psi": -0.05, "beta": (0.85,),
}


# ---------------------------------------------------------------------------
# Generic ARMA-GARCH simulator
# ---------------------------------------------------------------------------

def _draw_z(residual_dist, residual_shape, n, key):
    """Draw n iid standardised residuals (mean 0, variance 1)."""
    if isinstance(residual_dist, type(normal)) and type(residual_dist) is type(normal):
        return jax.random.normal(key, (n,))
    wrapper = StandardisedResidual(residual_dist)
    return wrapper.rvs(size=(n,), shape_params=residual_shape, key=key)


def _variance_uncond_init(var_model_cls, params):
    p_v_alpha = np.asarray(params.get("alpha", [])).reshape(-1)
    beta = np.asarray(params.get("beta", [])).reshape(-1)
    omega = float(params["omega"])
    name = var_model_cls.__name__
    if name == "GARCH":
        denom = 1.0 - p_v_alpha.sum() - beta.sum()
        return omega / max(denom, 1e-3)
    if name == "IGARCH":
        return 1.0
    if name == "GJR_GARCH":
        gamma = np.asarray(params.get("gamma", [])).reshape(-1)
        denom = 1.0 - p_v_alpha.sum() - 0.5 * gamma.sum() - beta.sum()
        return omega / max(denom, 1e-3)
    if name == "EGARCH":
        denom = 1.0 - beta.sum()
        log_uncond = omega / max(abs(denom), 1e-3)
        return float(np.exp(log_uncond))
    if name == "TGARCH":
        alpha_pos = np.asarray(params.get("alpha_pos", [])).reshape(-1)
        alpha_neg = np.asarray(params.get("alpha_neg", [])).reshape(-1)
        e_pos = np.sqrt(2.0 / np.pi)
        e_neg = e_pos
        sigma_uncond = omega / max(
            1.0 - (alpha_pos * e_pos).sum()
            - (alpha_neg * e_neg).sum() - beta.sum(),
            1e-3,
        )
        return float(sigma_uncond * sigma_uncond)
    if name == "QGARCH":
        denom = 1.0 - p_v_alpha.sum() - beta.sum()
        return omega / max(denom, 1e-3)
    raise ValueError(f"unknown variance variant {name!r}")


def _variance_step(var_model_cls, params, state):
    """One step of the variance recursion. Returns (sigma2_t, aux).

    ``aux`` carries variant-specific scalars needed by the lag-update
    step: ``log_sigma2_t`` for EGARCH, ``sigma_t`` for TGARCH, ``None``
    otherwise.
    """
    name = var_model_cls.__name__
    omega = float(params["omega"])
    beta = np.asarray(params.get("beta", [])).reshape(-1)
    q_v = beta.shape[0]

    if name in ("GARCH", "IGARCH"):
        alpha = np.asarray(params["alpha"]).reshape(-1)
        eps_sq_lags, var_lags = state
        sigma2_t = (
            omega
            + (alpha * eps_sq_lags[: alpha.shape[0]]).sum()
            + (beta * var_lags[:q_v]).sum()
        )
        return float(max(sigma2_t, 1e-12)), None
    if name == "GJR_GARCH":
        alpha = np.asarray(params["alpha"]).reshape(-1)
        gamma = np.asarray(params["gamma"]).reshape(-1)
        eps_sq_lags, eps_lags, var_lags = state
        ind = (eps_lags[: alpha.shape[0]] < 0.0).astype(float)
        sigma2_t = (
            omega
            + (alpha * eps_sq_lags[: alpha.shape[0]]).sum()
            + (gamma * ind * eps_sq_lags[: alpha.shape[0]]).sum()
            + (beta * var_lags[:q_v]).sum()
        )
        return float(max(sigma2_t, 1e-12)), None
    if name == "EGARCH":
        alpha = np.asarray(params["alpha"]).reshape(-1)
        gamma = np.asarray(params["gamma"]).reshape(-1)
        z_lags, log_var_lags = state
        e_abs_z = np.sqrt(2.0 / np.pi)
        log_sigma2_t = float(
            omega
            + (alpha * z_lags[: alpha.shape[0]]).sum()
            + (
                gamma
                * (np.abs(z_lags[: alpha.shape[0]]) - e_abs_z)
            ).sum()
            + (beta * log_var_lags[:q_v]).sum()
        )
        sigma2_t = float(np.exp(log_sigma2_t))
        return max(sigma2_t, 1e-12), log_sigma2_t
    if name == "TGARCH":
        alpha_pos = np.asarray(params["alpha_pos"]).reshape(-1)
        alpha_neg = np.asarray(params["alpha_neg"]).reshape(-1)
        eps_pos_lags, eps_neg_lags, sigma_lags = state
        sigma_t_form = (
            omega
            + (alpha_pos * eps_pos_lags[: alpha_pos.shape[0]]).sum()
            + (alpha_neg * eps_neg_lags[: alpha_neg.shape[0]]).sum()
            + (beta * sigma_lags[:q_v]).sum()
        )
        sigma_t = float(max(sigma_t_form, 1e-6))
        return sigma_t * sigma_t, sigma_t
    if name == "QGARCH":
        alpha = np.asarray(params["alpha"]).reshape(-1)
        psi = float(params["psi"])
        eps_sq_lags, eps_lags, var_lags = state
        sigma2_t = (
            omega
            + (alpha * eps_sq_lags[: alpha.shape[0]]).sum()
            + psi * eps_lags[: alpha.shape[0]].sum()
            + (beta * var_lags[:q_v]).sum()
        )
        return float(max(sigma2_t, 1e-12)), None
    raise ValueError(f"unknown variance variant {name!r}")


def _variance_state_init(var_model_cls, params, sigma2_uncond):
    name = var_model_cls.__name__
    p_v_alpha = np.asarray(params.get("alpha", params.get("alpha_pos", []))).reshape(-1)
    p_v = max(p_v_alpha.shape[0], 1)
    beta = np.asarray(params.get("beta", [])).reshape(-1)
    q_v = max(beta.shape[0], 1)
    if name in ("GARCH", "IGARCH"):
        return (
            np.full(p_v, sigma2_uncond),
            np.full(q_v, sigma2_uncond),
        )
    if name == "GJR_GARCH":
        return (
            np.full(p_v, sigma2_uncond),
            np.zeros(p_v),
            np.full(q_v, sigma2_uncond),
        )
    if name == "EGARCH":
        return (
            np.zeros(p_v),
            np.full(q_v, np.log(sigma2_uncond)),
        )
    if name == "TGARCH":
        sigma_uncond = float(np.sqrt(sigma2_uncond))
        return (
            np.zeros(p_v),
            np.zeros(p_v),
            np.full(q_v, sigma_uncond),
        )
    if name == "QGARCH":
        return (
            np.full(p_v, sigma2_uncond),
            np.zeros(p_v),
            np.full(q_v, sigma2_uncond),
        )
    raise ValueError(f"unknown variance variant {name!r}")


def _variance_state_update(var_model_cls, state, eps_t, sigma2_t, z_t, aux=None):
    name = var_model_cls.__name__
    if name in ("GARCH", "IGARCH"):
        eps_sq_lags, var_lags = state
        eps_sq_lags = np.concatenate([[eps_t * eps_t], eps_sq_lags[:-1]])
        var_lags = np.concatenate([[sigma2_t], var_lags[:-1]])
        return (eps_sq_lags, var_lags)
    if name == "GJR_GARCH":
        eps_sq_lags, eps_lags, var_lags = state
        eps_sq_lags = np.concatenate([[eps_t * eps_t], eps_sq_lags[:-1]])
        eps_lags = np.concatenate([[eps_t], eps_lags[:-1]])
        var_lags = np.concatenate([[sigma2_t], var_lags[:-1]])
        return (eps_sq_lags, eps_lags, var_lags)
    if name == "EGARCH":
        z_lags, log_var_lags = state
        log_sigma2_t = float(aux)
        z_lags = np.concatenate([[z_t], z_lags[:-1]])
        log_var_lags = np.concatenate([[log_sigma2_t], log_var_lags[:-1]])
        return (z_lags, log_var_lags)
    if name == "TGARCH":
        eps_pos_lags, eps_neg_lags, sigma_lags = state
        sigma_t = float(aux)
        eps_pos_lags = np.concatenate(
            [[max(eps_t, 0.0)], eps_pos_lags[:-1]]
        )
        eps_neg_lags = np.concatenate(
            [[-min(eps_t, 0.0)], eps_neg_lags[:-1]]
        )
        sigma_lags = np.concatenate([[sigma_t], sigma_lags[:-1]])
        return (eps_pos_lags, eps_neg_lags, sigma_lags)
    if name == "QGARCH":
        eps_sq_lags, eps_lags, var_lags = state
        eps_sq_lags = np.concatenate([[eps_t * eps_t], eps_sq_lags[:-1]])
        eps_lags = np.concatenate([[eps_t], eps_lags[:-1]])
        var_lags = np.concatenate([[sigma2_t], var_lags[:-1]])
        return (eps_sq_lags, eps_lags, var_lags)
    raise ValueError(f"unknown variance variant {name!r}")


def _simulate(
    n, mean_order, var_model_cls, var_order, params,
    residual_dist, residual_shape, key,
):
    p, q = mean_order
    phi = np.asarray(params.get("phi", [0.0] * p)).reshape(-1)
    theta = np.asarray(params.get("theta", [0.0] * q)).reshape(-1)
    c = float(params["c"])

    z = np.asarray(_draw_z(residual_dist, residual_shape, n, key))
    sigma2_uncond = _variance_uncond_init(var_model_cls, params)
    var_state = _variance_state_init(var_model_cls, params, sigma2_uncond)

    if p > 0:
        ar_factor = max(1.0 - phi.sum(), 1e-3)
        y_lags = np.full(p, c / ar_factor)
    else:
        y_lags = np.zeros(0)
    eps_arma_lags = np.zeros(q)

    y = np.zeros(n)
    for t in range(n):
        sigma2_t, aux = _variance_step(var_model_cls, params, var_state)
        sigma_t = float(np.sqrt(sigma2_t))
        eps_t = sigma_t * float(z[t])
        ar_term = float((phi * y_lags[:p]).sum()) if p > 0 else 0.0
        ma_term = (
            float((theta * eps_arma_lags[:q]).sum()) if q > 0 else 0.0
        )
        y_t = c + ar_term + ma_term + eps_t
        y[t] = y_t

        var_state = _variance_state_update(
            var_model_cls, var_state, eps_t, sigma2_t, float(z[t]),
            aux=aux,
        )
        if p > 0:
            y_lags = np.concatenate([[y_t], y_lags[:-1]])
        if q > 0:
            eps_arma_lags = np.concatenate([[eps_t], eps_arma_lags[:-1]])

    return jnp.asarray(y)


# ---------------------------------------------------------------------------
# Matrix configuration
# ---------------------------------------------------------------------------

_KEY = jax.random.PRNGKey(13)
_FIT_MAXITER = 800
_FIT_LR = 0.05
_FIT_N = 4000

_MATRIX = [
    ("arma11_garch11_normal",
     (1, 1), GARCH, (1, 1), normal, {}, _TRUTH_GARCH),
    ("arma21_garch11_normal",
     (2, 1), GARCH, (1, 1), normal, {}, _TRUTH_GARCH_AR2),
    ("arma11_garch12_normal",
     (1, 1), GARCH, (1, 2), normal, {}, _TRUTH_GARCH_PQ12),
    ("arma22_garch11_normal",
     (2, 2), GARCH, (1, 1), normal, {}, _TRUTH_GARCH_22),
    ("arma11_garch11_studentt",
     (1, 1), GARCH, (1, 1), student_t, {"nu": 6.0}, _TRUTH_GARCH),
    ("arma11_igarch11_normal",
     (1, 1), IGARCH, (1, 1), normal, {}, _TRUTH_IGARCH),
    ("arma11_gjr11_normal",
     (1, 1), GJR_GARCH, (1, 1), normal, {}, _TRUTH_GJR),
    ("arma11_egarch11_normal",
     (1, 1), EGARCH, (1, 1), normal, {}, _TRUTH_EGARCH),
    ("arma11_tgarch11_normal",
     (1, 1), TGARCH, (1, 1), normal, {}, _TRUTH_TGARCH),
    ("arma11_qgarch11_normal",
     (1, 1), QGARCH, (1, 1), normal, {}, _TRUTH_QGARCH),
]


def _make_fit(label, mean_order, var_model, var_order, dist, shape, truth, n=_FIT_N):
    key = jax.random.fold_in(_KEY, hash(label) & 0xFFFFFFFF)
    y = _simulate(n, mean_order, var_model, var_order, truth, dist, shape, key)
    fit = ArmaGarch(
        mean_order=mean_order, var_model=var_model, var_order=var_order,
        residual_dist=dist,
    ).fit(y, init="analytical", maxiter=_FIT_MAXITER, lr=_FIT_LR)
    return SimpleNamespace(
        label=label, y=y, fit=fit, truth=truth,
        mean_order=mean_order, var_model=var_model, var_order=var_order,
        residual_dist=dist, residual_shape=shape,
    )


@pytest.fixture(scope="module", params=_MATRIX, ids=[c[0] for c in _MATRIX])
def matrix_fit(request):
    return _make_fit(*request.param)


@pytest.fixture(scope="module")
def base_fit():
    cfg = _MATRIX[0]
    return _make_fit(*cfg)


@pytest.fixture(scope="module")
def base_fit_t():
    cfg = _MATRIX[4]
    return _make_fit(*cfg)


@pytest.fixture(scope="module")
def large_fit():
    return _make_fit(
        "arma11_garch11_normal_large",
        (1, 1), GARCH, (1, 1), normal, {}, _TRUTH_GARCH, n=10000,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(x):
    return np.asarray(jnp.atleast_1d(jnp.asarray(x, dtype=float))).ravel()


def _truth_array(truth_dict, key, fitted_array):
    raw = truth_dict.get(key, None)
    if raw is None:
        return None
    arr = np.atleast_1d(np.asarray(raw, dtype=float))
    return arr.reshape(np.asarray(fitted_array).shape)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_invalid_mean_order_raises(self):
        with pytest.raises(ValueError, match="mean_order"):
            ArmaGarch(
                mean_order=1, var_model=GARCH, var_order=(1, 1),
                residual_dist=normal,
            )
        with pytest.raises(ValueError, match="mean_order"):
            ArmaGarch(
                mean_order=(1, 1, 1), var_model=GARCH, var_order=(1, 1),
                residual_dist=normal,
            )

    def test_invalid_var_order_raises(self):
        with pytest.raises(ValueError, match="var_order"):
            ArmaGarch(
                mean_order=(1, 1), var_model=GARCH, var_order=(1,),
                residual_dist=normal,
            )

    def test_garch_m_raises(self):
        with pytest.raises(NotImplementedError, match="GARCH-M"):
            ArmaGarch(
                mean_order=(1, 1), var_model=GARCH_M, var_order=(1, 1),
                residual_dist=normal,
            )

    @pytest.mark.parametrize(
        "var_model", [GARCH, IGARCH, GJR_GARCH, EGARCH, TGARCH, QGARCH],
    )
    def test_supported_variants_construct_cleanly(self, var_model):
        m = ArmaGarch(
            mean_order=(1, 1), var_model=var_model, var_order=(1, 1),
            residual_dist=normal,
        )
        assert m.var_model is var_model
        assert m.is_fitted is False

    def test_default_residual_dist_is_normal(self):
        m = ArmaGarch(mean_order=(1, 1), var_model=GARCH, var_order=(1, 1))
        assert type(m.residual_dist) is type(normal)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

class TestRecovery:
    def _se_budget_check(self, fit, truth, key, multiplier=4.0, floor=5e-3):
        fitted = _flatten(fit.params[key])
        target = _truth_array(truth, key, fitted)
        if target is None:
            return
        se = _flatten(fit.standard_errors_[key])
        budget = multiplier * se + floor
        diff = np.abs(fitted - target)
        np.testing.assert_array_less(
            diff, budget,
            err_msg=(
                f"key={key!r} fitted={fitted} target={target} "
                f"se={se} diff={diff} budget={budget}"
            ),
        )

    def test_recovery_within_se_budget(self, large_fit):
        fit = large_fit.fit
        truth = large_fit.truth
        for k in ("phi", "theta", "c", "omega", "alpha", "beta"):
            self._se_budget_check(fit, truth, k)

    @pytest.mark.parametrize(
        "var_model,truth",
        [
            (GARCH, _TRUTH_GARCH),
            (IGARCH, _TRUTH_IGARCH),
            (GJR_GARCH, _TRUTH_GJR),
            (EGARCH, _TRUTH_EGARCH),
            (TGARCH, _TRUTH_TGARCH),
            (QGARCH, _TRUTH_QGARCH),
        ],
    )
    def test_recovery_per_variant(self, var_model, truth):
        cfg_label = f"recovery_{var_model.__name__.lower()}"
        bundle = _make_fit(
            cfg_label, (1, 1), var_model, (1, 1), normal, {}, truth, n=_FIT_N,
        )
        for k in ("phi", "theta", "c"):
            self._se_budget_check(bundle.fit, truth, k)


# ---------------------------------------------------------------------------
# n_params
# ---------------------------------------------------------------------------

class TestNParams:
    @pytest.mark.parametrize(
        "mean_order,var_model,var_order,residual_dist",
        [
            ((1, 0), GARCH, (1, 1), normal),
            ((0, 1), GARCH, (1, 1), normal),
            ((1, 1), GARCH, (1, 1), normal),
            ((2, 1), GARCH, (1, 2), normal),
            ((1, 2), GARCH, (2, 1), normal),
            ((2, 2), GARCH, (1, 1), normal),
            ((1, 1), IGARCH, (1, 1), normal),
            ((1, 1), GJR_GARCH, (1, 1), normal),
            ((1, 1), EGARCH, (1, 1), normal),
            ((1, 1), TGARCH, (1, 1), normal),
            ((1, 1), QGARCH, (1, 1), normal),
            ((1, 1), GARCH, (1, 1), student_t),
            ((1, 1), GARCH, (1, 1), gh),
            ((1, 1), GARCH, (1, 1), skewed_t),
        ],
    )
    def test_n_params_matches_param_dict(
        self, mean_order, var_model, var_order, residual_dist,
    ):
        m = ArmaGarch(
            mean_order=mean_order, var_model=var_model, var_order=var_order,
            residual_dist=residual_dist,
        )
        wrapper = StandardisedResidual(residual_dist)
        # Closed form: phi(p) + theta(q) + c(1) + variance natural-param
        # sizes from the cold-start dict + residual shape params.
        cold = m._var_backend._ag_cold_start(
            jnp.zeros((10,), dtype=float), "backcast", None, wrapper,
        )
        n_var = sum(
            int(jnp.atleast_1d(jnp.asarray(v, dtype=float)).size)
            for v in cold.values()
        )
        expected = (
            mean_order[0] + mean_order[1] + 1
            + n_var + wrapper.n_shape_params
        )
        assert m.n_params == expected


# ---------------------------------------------------------------------------
# Joint vs separable
# ---------------------------------------------------------------------------

class TestJointVsSeparable:
    def _evaluate_joint_at(self, mean_order, var_model, var_order, residual_dist, y, init_params):
        return ArmaGarch(
            mean_order=mean_order, var_model=var_model, var_order=var_order,
            residual_dist=residual_dist,
        ).fit(y, init="warm", init_params=init_params, maxiter=0)

    def _separable_init_params(self, mean_order, var_order, residual_dist, y):
        p, q = mean_order
        p_v, q_v = var_order
        arma_fit = ARMA(
            p=p, q=q, residual_dist=residual_dist,
        ).fit(y, init="analytical", maxiter=_FIT_MAXITER, lr=_FIT_LR)
        eps = arma_fit.residuals(y)["residuals"]
        garch_fit = GARCH(
            p=p_v, q=q_v, residual_dist=residual_dist,
        ).fit(eps, init="analytical", maxiter=_FIT_MAXITER, lr=_FIT_LR)
        sep = {
            "phi": arma_fit.params["phi"],
            "theta": arma_fit.params["theta"],
            "c": arma_fit.params["c"],
            "omega": garch_fit.params["omega"],
            "alpha": garch_fit.params["alpha"],
            "beta": garch_fit.params["beta"],
            "residual": dict(garch_fit.params["residual"]),
        }
        return sep

    def test_joint_at_least_as_high_as_separable_normal(self, base_fit):
        y = base_fit.y
        sep_params = self._separable_init_params(
            (1, 1), (1, 1), normal, y,
        )
        sep_eval = self._evaluate_joint_at(
            (1, 1), GARCH, (1, 1), normal, y, sep_params,
        )
        sep_ll = float(sep_eval.loglikelihood())
        joint_ll = float(base_fit.fit.loglikelihood())
        assert joint_ll >= sep_ll - 1e-3

    def test_joint_at_least_as_high_as_separable_studentt(self, base_fit_t):
        y = base_fit_t.y
        sep_params = self._separable_init_params(
            (1, 1), (1, 1), student_t, y,
        )
        sep_eval = self._evaluate_joint_at(
            (1, 1), GARCH, (1, 1), student_t, y, sep_params,
        )
        sep_ll = float(sep_eval.loglikelihood())
        joint_ll = float(base_fit_t.fit.loglikelihood())
        assert joint_ll >= sep_ll - 1e-3


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

class TestResiduals:
    def test_residuals_match_y_minus_conditional_mean(self, matrix_fit):
        fit = matrix_fit.fit
        y = matrix_fit.y
        d = fit.residuals(y)
        expected = np.asarray(y) - np.asarray(fit.conditional_mean(y))
        np.testing.assert_allclose(
            np.asarray(d["residuals"]), expected, rtol=1e-6, atol=1e-8,
        )

    def test_standardised_residuals_match_residuals_over_sigma(self, matrix_fit):
        fit = matrix_fit.fit
        y = matrix_fit.y
        d = fit.residuals(y)
        sigma = np.sqrt(np.asarray(fit.conditional_variance(y)))
        np.testing.assert_allclose(
            np.asarray(d["standardised_residuals"]),
            np.asarray(d["residuals"]) / sigma,
            rtol=1e-6, atol=1e-8,
        )

    def test_standardised_residuals_unit_variance(self, matrix_fit):
        z = np.asarray(
            matrix_fit.fit.residuals(matrix_fit.y)["standardised_residuals"]
        )
        np.testing.assert_allclose(z.mean(), 0.0, atol=0.06)
        np.testing.assert_allclose(z.var(), 1.0, atol=0.10)

    def test_residuals_finite(self, matrix_fit):
        d = matrix_fit.fit.residuals(matrix_fit.y)
        assert np.all(np.isfinite(np.asarray(d["residuals"])))
        assert np.all(np.isfinite(np.asarray(d["standardised_residuals"])))


# ---------------------------------------------------------------------------
# Cached diagnostics parity
# ---------------------------------------------------------------------------

class TestCachedDiagnosticsParity:
    def test_loglikelihood_aic_bic_parity(self, matrix_fit):
        fit = matrix_fit.fit
        y = matrix_fit.y
        np.testing.assert_allclose(
            float(fit.loglikelihood()), float(fit.loglikelihood(y)),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(fit.aic()), float(fit.aic(y)), rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(fit.bic()), float(fit.bic(y)), rtol=1e-5,
        )

    def test_acf_pacf_parity(self, matrix_fit):
        fit = matrix_fit.fit
        y = matrix_fit.y
        np.testing.assert_allclose(
            np.asarray(fit.acf()), np.asarray(fit.acf(y)),
            rtol=1e-5, atol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(fit.pacf()), np.asarray(fit.pacf(y)),
            rtol=1e-5, atol=1e-8,
        )

    def test_hypothesis_test_parity(self, matrix_fit):
        fit = matrix_fit.fit
        y = matrix_fit.y
        for accessor in (
            "ljung_box", "arch_lm", "adf_residuals", "kpss_residuals",
        ):
            cached = getattr(fit, accessor)()
            recomp = getattr(fit, accessor)(y)
            for k, cached_v in cached.items():
                recomp_v = recomp[k]
                cached_arr = np.asarray(
                    jnp.asarray(cached_v) if not isinstance(cached_v, dict)
                    else list(cached_v.values()),
                    dtype=float,
                )
                recomp_arr = np.asarray(
                    jnp.asarray(recomp_v) if not isinstance(recomp_v, dict)
                    else list(recomp_v.values()),
                    dtype=float,
                )
                np.testing.assert_allclose(
                    cached_arr, recomp_arr,
                    rtol=1e-5, atol=1e-8,
                    err_msg=f"{accessor}.{k}",
                )

    def test_cached_diagnostics_dict_consistent(self, matrix_fit):
        fit = matrix_fit.fit
        d = fit.residual_diagnostics_
        np.testing.assert_allclose(
            float(d["loglikelihood"]), float(fit.loglikelihood()),
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            float(d["aic"]), float(fit.aic()), rtol=1e-7,
        )
        np.testing.assert_allclose(
            float(d["bic"]), float(fit.bic()), rtol=1e-7,
        )
        np.testing.assert_allclose(
            np.asarray(d["acf"]), np.asarray(fit.acf()),
            rtol=1e-7,
        )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_unconditional_mean_formula(self, matrix_fit):
        fit = matrix_fit.fit
        s = fit.stats()
        phi = np.asarray(fit.params["phi"]).reshape(-1)
        c = float(fit.params["c"])
        ar_factor = 1.0 - phi.sum()
        if abs(ar_factor) > 1e-9:
            np.testing.assert_allclose(
                float(s["unconditional_mean"]), c / ar_factor,
                rtol=1e-6, atol=1e-8,
            )

    def test_var_persistence_formula(self, matrix_fit):
        fit = matrix_fit.fit
        s = fit.stats()
        name = matrix_fit.var_model.__name__
        if name in ("GARCH", "IGARCH"):
            expected = (
                np.asarray(fit.params["alpha"]).sum()
                + np.asarray(fit.params["beta"]).sum()
            )
        elif name == "GJR_GARCH":
            kappa = float(s["var_kappa"])
            expected = (
                np.asarray(fit.params["alpha"]).sum()
                + kappa * np.asarray(fit.params["gamma"]).sum()
                + np.asarray(fit.params["beta"]).sum()
            )
        elif name == "EGARCH":
            expected = np.asarray(fit.params["beta"]).sum()
        elif name == "TGARCH":
            e_pos = float(s["var_expected_z_pos"])
            e_neg = float(s["var_expected_z_neg"])
            expected = (
                e_pos * np.asarray(fit.params["alpha_pos"]).sum()
                + e_neg * np.asarray(fit.params["alpha_neg"]).sum()
                + np.asarray(fit.params["beta"]).sum()
            )
        elif name == "QGARCH":
            expected = (
                float(np.asarray(fit.params["alpha"]).reshape(-1)[0])
                + np.asarray(fit.params["beta"]).sum()
            )
        else:
            pytest.skip(f"unknown variant {name}")
        np.testing.assert_allclose(
            float(s["var_persistence"]), float(expected),
            rtol=1e-6, atol=1e-8,
        )

    def test_var_unconditional_variance_formula(self, matrix_fit):
        fit = matrix_fit.fit
        s = fit.stats()
        name = matrix_fit.var_model.__name__
        omega = float(fit.params["omega"])
        persistence = float(s["var_persistence"])
        if name == "IGARCH":
            assert np.isinf(float(s["var_unconditional_variance"]))
            return
        if name == "EGARCH":
            log_uncond = omega / (1.0 - persistence)
            expected_var = float(np.exp(log_uncond))
            np.testing.assert_allclose(
                float(s["var_unconditional_variance"]), expected_var,
                rtol=1e-4,
            )
            return
        if name == "TGARCH":
            sigma_u = omega / (1.0 - persistence)
            np.testing.assert_allclose(
                float(s["var_unconditional_sigma"]), sigma_u, rtol=1e-6,
            )
            np.testing.assert_allclose(
                float(s["var_unconditional_variance"]), sigma_u ** 2,
                rtol=1e-6,
            )
            return
        # GARCH, GJR_GARCH, QGARCH share omega / (1 - persistence).
        if persistence < 1.0:
            np.testing.assert_allclose(
                float(s["var_unconditional_variance"]),
                omega / (1.0 - persistence),
                rtol=1e-6, atol=1e-8,
            )

    def test_var_half_life_formula(self, matrix_fit):
        fit = matrix_fit.fit
        s = fit.stats()
        if matrix_fit.var_model is IGARCH:
            assert np.isinf(float(s["var_half_life"]))
            return
        persistence = float(s["var_persistence"])
        if 0.0 < persistence < 1.0:
            expected = np.log(0.5) / np.log(abs(persistence))
            np.testing.assert_allclose(
                float(s["var_half_life"]), expected, rtol=1e-6,
            )

    def test_ar_root_moduli_match_numpy(self, matrix_fit):
        fit = matrix_fit.fit
        phi = np.asarray(fit.params["phi"]).reshape(-1)
        if phi.size == 0:
            return
        coeffs = np.concatenate([(-phi)[::-1], [1.0]])
        ref = np.sort(np.abs(np.roots(coeffs)))
        got = np.sort(np.asarray(fit.stats()["ar_root_moduli"]))
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-7)

    def test_ma_root_moduli_match_numpy(self, matrix_fit):
        fit = matrix_fit.fit
        theta = np.asarray(fit.params["theta"]).reshape(-1)
        if theta.size == 0:
            return
        coeffs = np.concatenate([theta[::-1], [1.0]])
        ref = np.sort(np.abs(np.roots(coeffs)))
        got = np.sort(np.asarray(fit.stats()["ma_root_moduli"]))
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-7)

    def test_mean_stationary_iff_root_moduli_above_one(self, matrix_fit):
        fit = matrix_fit.fit
        s = fit.stats()
        ar_mod = np.asarray(s["ar_root_moduli"])
        if ar_mod.size == 0:
            assert bool(s["mean_is_stationary"])
            return
        assert bool(s["mean_is_stationary"]) == bool(np.all(ar_mod > 1.0))

    def test_stats_keys_present(self, matrix_fit):
        s = matrix_fit.fit.stats()
        required = {
            "unconditional_mean",
            "var_persistence", "var_is_stationary",
            "mean_is_stationary", "mean_is_invertible",
            "ar_root_moduli", "ma_root_moduli",
        }
        assert required <= set(s)


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

_ANALYTICAL_VARIANTS = (GARCH, IGARCH, GJR_GARCH, QGARCH)
_NO_ANALYTICAL_VARIANTS = (EGARCH, TGARCH)


class TestForecast:
    def test_analytical_forecast_finite(self, matrix_fit):
        if matrix_fit.var_model in _NO_ANALYTICAL_VARIANTS:
            pytest.skip("no analytical h>=2")
        fc = matrix_fit.fit.forecast(h=20, method="analytical")
        assert fc["paths"] is None
        assert fc["mean"].shape == (20,)
        assert fc["variance"].shape == (20,)
        assert np.all(np.isfinite(np.asarray(fc["mean"])))
        assert np.all(np.isfinite(np.asarray(fc["variance"])))

    def test_simulation_forecast_finite(self, matrix_fit):
        fc = matrix_fit.fit.forecast(
            h=10, method="simulation", n_paths=50,
            key=jax.random.PRNGKey(7),
        )
        assert fc["mean"].shape == (10,)
        assert fc["variance"].shape == (10,)
        assert fc["paths"].shape == (50, 10)
        assert np.all(np.isfinite(np.asarray(fc["paths"])))

    def test_analytical_mean_converges_to_unconditional(self, matrix_fit):
        if matrix_fit.var_model in _NO_ANALYTICAL_VARIANTS:
            pytest.skip("no analytical h>=2")
        fit = matrix_fit.fit
        s = fit.stats()
        if not bool(s["mean_is_stationary"]):
            pytest.skip("non-stationary mean")
        fc = fit.forecast(h=2000, method="analytical")
        np.testing.assert_allclose(
            float(fc["mean"][-1]), float(s["unconditional_mean"]),
            rtol=1e-3, atol=1e-4,
        )

    def test_analytical_variance_converges_to_unconditional(self, matrix_fit):
        if matrix_fit.var_model is IGARCH:
            pytest.skip("IGARCH has no unconditional variance")
        if matrix_fit.var_model in _NO_ANALYTICAL_VARIANTS:
            pytest.skip("no analytical h>=2")
        fit = matrix_fit.fit
        s = fit.stats()
        if not bool(s["var_is_stationary"]):
            pytest.skip("non-stationary variance")
        fc = fit.forecast(h=2000, method="analytical")
        np.testing.assert_allclose(
            float(fc["variance"][-1]),
            float(s["var_unconditional_variance"]),
            rtol=0.01,
        )

    def test_simulation_mean_matches_analytical(self, matrix_fit):
        if matrix_fit.var_model in _NO_ANALYTICAL_VARIANTS:
            pytest.skip("no analytical reference")
        fit = matrix_fit.fit
        h = 5
        n_paths = 4000
        analytical = fit.forecast(h=h, method="analytical")
        sim = fit.forecast(
            h=h, method="simulation", n_paths=n_paths,
            key=jax.random.PRNGKey(11),
        )
        ana_mean = np.asarray(analytical["mean"])
        sim_mean = np.asarray(sim["mean"])
        ana_var = np.asarray(analytical["variance"])
        # Monte-Carlo SE on the path mean: sqrt(var/n_paths).
        mc_se = np.sqrt(ana_var / n_paths)
        np.testing.assert_array_less(np.abs(sim_mean - ana_mean), 5.0 * mc_se)

    def test_h_zero_raises(self, base_fit):
        with pytest.raises(ValueError):
            base_fit.fit.forecast(h=0, method="analytical")

    def test_h_negative_raises(self, base_fit):
        with pytest.raises(ValueError):
            base_fit.fit.forecast(h=-1, method="analytical")

    def test_simulation_requires_n_paths(self, base_fit):
        with pytest.raises(ValueError):
            base_fit.fit.forecast(h=5, method="simulation", n_paths=0)

    def test_unknown_method_raises(self, base_fit):
        with pytest.raises(ValueError):
            base_fit.fit.forecast(h=5, method="bogus")

    @pytest.mark.parametrize("var_model", _NO_ANALYTICAL_VARIANTS)
    def test_h2_analytical_raises_for_no_analytical_variants(self, var_model):
        truth = {EGARCH: _TRUTH_EGARCH, TGARCH: _TRUTH_TGARCH}[var_model]
        bundle = _make_fit(
            f"fc_h2_{var_model.__name__}",
            (1, 1), var_model, (1, 1), normal, {}, truth, n=1500,
        )
        # h=1 must work.
        fc1 = bundle.fit.forecast(h=1, method="analytical")
        assert fc1["variance"].shape == (1,)
        with pytest.raises(ValueError, match=var_model.__name__):
            bundle.fit.forecast(h=10, method="analytical")


# ---------------------------------------------------------------------------
# Rvs
# ---------------------------------------------------------------------------

class TestRvs:
    def test_rvs_deterministic_under_u(self, matrix_fit):
        fit = matrix_fit.fit
        u = jnp.linspace(0.01, 0.99, 30)
        a = fit.rvs(u=u)
        b = fit.rvs(u=u)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))

    def test_rvs_different_keys_differ(self, matrix_fit):
        fit = matrix_fit.fit
        a = fit.rvs(size=(20,), key=jax.random.PRNGKey(1))
        b = fit.rvs(size=(20,), key=jax.random.PRNGKey(2))
        assert not np.allclose(np.asarray(a), np.asarray(b))

    def test_rvs_2d_shape(self, matrix_fit):
        fit = matrix_fit.fit
        out = fit.rvs(size=(7, 12), key=jax.random.PRNGKey(0))
        assert out.shape == (7, 12)
        assert np.all(np.isfinite(np.asarray(out)))

    def test_rvs_size_or_u_required(self, base_fit):
        with pytest.raises(ValueError):
            base_fit.fit.rvs()

    def test_rvs_3d_u_raises(self, base_fit):
        u = jnp.zeros((2, 3, 4)) + 0.5
        with pytest.raises(ValueError):
            base_fit.fit.rvs(u=u)

    def test_rvs_long_run_moments(self, base_fit):
        fit = base_fit.fit
        s = fit.stats()  # base_fit is GARCH, so stats() works
        if not bool(s["var_is_stationary"]):
            pytest.skip("non-stationary fit")
        n_paths = 600
        h = 600
        paths = fit.rvs(size=(n_paths, h), key=jax.random.PRNGKey(31))
        terminal = np.asarray(paths[:, -1])
        np.testing.assert_allclose(
            terminal.mean(), float(s["unconditional_mean"]),
            atol=0.2,
        )


# ---------------------------------------------------------------------------
# Variant invariants
# ---------------------------------------------------------------------------

class TestVariantInvariants:
    def test_igarch_persistence_pinned(self):
        bundle = _make_fit(
            "inv_igarch", (1, 1), IGARCH, (1, 1), normal, {},
            _TRUTH_IGARCH, n=2000,
        )
        persistence = (
            float(bundle.fit.params["alpha"][0])
            + float(bundle.fit.params["beta"][0])
        )
        np.testing.assert_allclose(persistence, 1.0, atol=1e-6)

    def test_qgarch_positivity_invariant(self):
        bundle = _make_fit(
            "inv_qgarch", (1, 1), QGARCH, (1, 1), normal, {},
            _TRUTH_QGARCH, n=2000,
        )
        omega = float(bundle.fit.params["omega"])
        alpha = float(np.asarray(bundle.fit.params["alpha"]).reshape(-1)[0])
        psi = float(np.asarray(bundle.fit.params["psi"]).reshape(-1)[0])
        assert omega + 1e-9 >= psi * psi / (4.0 * alpha)

    def test_gjr_persistence_below_one(self):
        bundle = _make_fit(
            "inv_gjr", (1, 1), GJR_GARCH, (1, 1), normal, {},
            _TRUTH_GJR, n=2000,
        )
        s = bundle.fit.stats()
        assert float(s["var_persistence"]) < 1.0


# ---------------------------------------------------------------------------
# JIT
# ---------------------------------------------------------------------------

class TestJIT:
    @pytest.mark.parametrize(
        "method",
        [
            "conditional_mean",
            "conditional_variance",
            "residuals",
            "loglikelihood",
        ],
    )
    def test_jit_matches_eager(self, matrix_fit, method):
        fit = matrix_fit.fit
        y = matrix_fit.y
        eager = getattr(fit, method)(y)
        jitted = jax.jit(getattr(fit, method))(y)
        if isinstance(eager, dict):
            for k in eager:
                np.testing.assert_allclose(
                    np.asarray(jitted[k]), np.asarray(eager[k]),
                    rtol=1e-6, atol=1e-8,
                )
        else:
            np.testing.assert_allclose(
                np.asarray(jitted), np.asarray(eager),
                rtol=1e-6, atol=1e-8,
            )

    def test_jit_rvs(self, base_fit):
        fit = base_fit.fit
        key = jax.random.PRNGKey(42)
        eager = fit.rvs(size=(8, 12), key=key)
        jitted = jax.jit(
            lambda k: fit.rvs(size=(8, 12), key=k)
        )(key)
        np.testing.assert_allclose(
            np.asarray(jitted), np.asarray(eager),
            rtol=1e-6, atol=1e-8,
        )


# ---------------------------------------------------------------------------
# Warm start
# ---------------------------------------------------------------------------

class TestWarmStart:
    def test_warm_zero_iter_reproduces_init(self, base_fit):
        cold = base_fit.fit
        warm = ArmaGarch(
            mean_order=base_fit.mean_order, var_model=base_fit.var_model,
            var_order=base_fit.var_order, residual_dist=base_fit.residual_dist,
        ).fit(base_fit.y, init="warm", init_params=cold.params, maxiter=0)
        for k in ("phi", "theta", "c", "omega", "alpha", "beta"):
            np.testing.assert_allclose(
                _flatten(warm.params[k]), _flatten(cold.params[k]),
                rtol=1e-5, atol=1e-6,
            )
        np.testing.assert_allclose(
            float(warm.loglikelihood()), float(cold.loglikelihood()),
            rtol=1e-5,
        )

    def test_warm_short_refit_reaches_cold_ll(self, base_fit):
        cold = base_fit.fit
        warm = ArmaGarch(
            mean_order=base_fit.mean_order, var_model=base_fit.var_model,
            var_order=base_fit.var_order, residual_dist=base_fit.residual_dist,
        ).fit(
            base_fit.y, init="warm", init_params=cold.params,
            maxiter=20, lr=_FIT_LR,
        )
        np.testing.assert_allclose(
            float(warm.loglikelihood()), float(cold.loglikelihood()),
            rtol=5e-3,
        )

    def test_warm_missing_key_raises(self, base_fit):
        partial = dict(base_fit.fit.params)
        partial.pop("phi")
        with pytest.raises(KeyError):
            ArmaGarch(
                mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
                residual_dist=normal,
            ).fit(base_fit.y, init="warm", init_params=partial, maxiter=0)

    def test_warm_missing_init_params_raises(self, base_fit):
        with pytest.raises(ValueError):
            ArmaGarch(
                mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
                residual_dist=normal,
            ).fit(base_fit.y, init="warm", maxiter=0)

    def test_warm_missing_var_key_raises(self, base_fit):
        partial = dict(base_fit.fit.params)
        partial.pop("alpha")
        with pytest.raises(KeyError):
            ArmaGarch(
                mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
                residual_dist=normal,
            ).fit(base_fit.y, init="warm", init_params=partial, maxiter=0)


# ---------------------------------------------------------------------------
# Init modes
# ---------------------------------------------------------------------------

class TestInitModes:
    @pytest.mark.parametrize("mode", ["analytical", "backcast", "sample"])
    def test_init_mode_runs_cleanly(self, base_fit, mode):
        fit = ArmaGarch(
            mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(base_fit.y, init=mode, maxiter=50, lr=_FIT_LR)
        assert fit.is_fitted
        assert jnp.isfinite(fit.loglikelihood())

    def test_unknown_init_mode_raises(self, base_fit):
        with pytest.raises(ValueError):
            ArmaGarch(
                mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
                residual_dist=normal,
            ).fit(base_fit.y, init="bogus", maxiter=10)
