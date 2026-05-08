"""End-to-end tests for the joint ARMA-GARCH composite estimator.

Anchored on rugarch reference data for joint-fit cross-validation.
The rugarch regenerator script and committed Python reference module
live in ``copulax/tests/_r_reference/``; this file loads the hardcoded
``RUGARCH_REFERENCE`` dict at import time and treats it as ground
truth for every matrix entry rugarch covers.

Coverage:

* Constructor validation, residual whitelist, variant whitelist.
* Parameter recovery within an asymptotic SE budget.
* ``n_params`` matches the sum of fitted-parameter sizes across an
  (mean_order, variance_variant, var_order, residual_law) grid.
* Joint MLE log-likelihood is at least as high as the two-stage
  separable fit on the same data, evaluated at the separable
  parameter point via warm-start with ``maxiter=0``.
* Residual contract, cached residual diagnostics, and stats
  formulas for every variant in the matrix.
* Forecast finiteness, convergence to unconditional moments, and
  agreement with rugarch on the cases where rugarch supports the
  variant.
* Long-run rvs path empirical moments match the unconditional
  moments within Monte-Carlo error.
* JIT compatibility of every public method on the post-fit object,
  plus end-to-end fit JIT for every matrix combination.
* Init-mode convergence (analytical / backcast / sample) verified
  against rugarch on every variant.
* AIC / BIC ranking across (GARCH, IGARCH, GJR, EGARCH) matches
  rugarch.
* Cached Ljung-Box and Q-stat on squared residuals match rugarch.
* Robustness: differentiability, determinism, near-stationary edge
  cases, and simulation-based moment checks.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
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
from copulax.univariate import gen_normal, gh, nig, normal, skewed_t, student_t
from copulax._src.timeseries._residuals._standardise import (
    StandardisedResidual,
)


# ---------------------------------------------------------------------------
# Load the rugarch reference module
# ---------------------------------------------------------------------------

_RUGARCH_REF_PATH = (
    Path(__file__).parent / "_r_reference" / "arma_garch_reference_data.py"
)
_rg_spec = _ilu.spec_from_file_location(
    "_arma_garch_rugarch_reference", _RUGARCH_REF_PATH,
)
_rg_module = _ilu.module_from_spec(_rg_spec)
_rg_spec.loader.exec_module(_rg_module)
RUGARCH_REFERENCE = _rg_module.RUGARCH_REFERENCE


_VAR_MODEL_FROM_NAME = {
    "GARCH": GARCH, "IGARCH": IGARCH, "GJR_GARCH": GJR_GARCH,
    "EGARCH": EGARCH, "TGARCH": TGARCH, "QGARCH": QGARCH,
}

_RESIDUAL_DIST_FROM_NAME = {
    "normal": normal, "student_t": student_t, "gen_normal": gen_normal,
    "nig": nig, "gh": gh, "skewed_t": skewed_t,
}


# ---------------------------------------------------------------------------
# Hand-rolled cases (TGARCH, QGARCH, gh, skewed_t; no rugarch parity)
# ---------------------------------------------------------------------------

_HANDROLLED_LABELS = (
    "arma11_tgarch11_normal",
    "arma11_qgarch11_normal",
    "arma11_garch11_gh",
    "arma11_garch11_skewedt",
)

_HANDROLLED_TRUTH = {
    "arma11_tgarch11_normal": {
        "phi": (0.5,), "theta": (0.3,), "mu": 0.10,
        "omega": 0.02,
        "alpha_pos": (0.10,), "alpha_neg": (0.20,), "beta": (0.70,),
    },
    "arma11_qgarch11_normal": {
        "phi": (0.5,), "theta": (0.3,), "mu": 0.10,
        "omega": 0.05,
        "alpha": (0.10,), "psi": -0.05, "beta": (0.85,),
    },
    "arma11_garch11_gh": {
        "phi": (0.5,), "theta": (0.3,), "mu": 0.10,
        "omega": 0.05, "alpha": (0.10,), "beta": (0.85,),
    },
    "arma11_garch11_skewedt": {
        "phi": (0.5,), "theta": (0.3,), "mu": 0.10,
        "omega": 0.05, "alpha": (0.10,), "beta": (0.85,),
    },
}

_HANDROLLED_RESIDUAL_TRUTH = {
    "arma11_tgarch11_normal": {},
    "arma11_qgarch11_normal": {},
    "arma11_garch11_gh": {
        "lamb": 0.0, "chi": 1.0, "psi": 1.0, "gamma": 0.0,
    },
    "arma11_garch11_skewedt": {"nu": 6.0, "gamma": 0.2},
}

_HANDROLLED_VAR_MODEL = {
    "arma11_tgarch11_normal": TGARCH,
    "arma11_qgarch11_normal": QGARCH,
    "arma11_garch11_gh": GARCH,
    "arma11_garch11_skewedt": GARCH,
}

_HANDROLLED_RESIDUAL_DIST = {
    "arma11_tgarch11_normal": normal,
    "arma11_qgarch11_normal": normal,
    "arma11_garch11_gh": gh,
    "arma11_garch11_skewedt": skewed_t,
}

_BURN_IN = 500


def _draw_z(residual_dist, residual_shape, n, key):
    """Draw n iid standardised residuals via the production wrapper."""
    return StandardisedResidual(residual_dist).rvs(
        size=(n,), shape_params=residual_shape, key=key,
    )


def _simulate_handrolled(label, n=2000, key=None):
    """Hand-rolled ARMA(1,1)-variant simulator for cases rugarch can't reach.

    The variance lags are seeded to unit variance and the simulator
    burns ``_BURN_IN`` steps before keeping the final ``n``. No
    closed-form unconditional-variance formulas appear here.
    """
    truth = _HANDROLLED_TRUTH[label]
    var_model = _HANDROLLED_VAR_MODEL[label]
    residual_dist = _HANDROLLED_RESIDUAL_DIST[label]
    residual_shape = _HANDROLLED_RESIDUAL_TRUTH[label]
    key = jax.random.PRNGKey(0) if key is None else key
    total = n + _BURN_IN

    z = np.asarray(
        _draw_z(residual_dist, residual_shape, total, key)
    )
    phi = float(truth["phi"][0])
    theta = float(truth["theta"][0])
    mu = float(truth["mu"])
    omega = float(truth["omega"])

    eps_lag = 0.0
    # Centred-form ARMA: unconditional mean of y_t IS μ (no AR
    # rescaling required).  Seed the AR lag at the unconditional mean.
    y_lag = mu

    if var_model is TGARCH:
        ap = float(truth["alpha_pos"][0])
        an = float(truth["alpha_neg"][0])
        beta = float(truth["beta"][0])
        sigma_lag = 1.0
    elif var_model is QGARCH:
        a = float(truth["alpha"][0])
        psi = float(truth["psi"])
        beta = float(truth["beta"][0])
        eps_sq_lag = 1.0
        var_lag = 1.0
    elif var_model is GARCH:
        a = float(truth["alpha"][0])
        beta = float(truth["beta"][0])
        eps_sq_lag = 1.0
        var_lag = 1.0
    else:
        raise ValueError(f"unsupported handrolled variant {var_model.__name__}")

    y = np.zeros(total)
    for t in range(total):
        if var_model is TGARCH:
            sigma_t = max(
                omega + ap * max(eps_lag, 0.0)
                + an * max(-eps_lag, 0.0) + beta * sigma_lag,
                1e-6,
            )
            sigma2_t = sigma_t * sigma_t
        elif var_model is QGARCH:
            sigma2_t = max(
                omega + a * eps_sq_lag + psi * eps_lag + beta * var_lag,
                1e-12,
            )
            sigma_t = float(np.sqrt(sigma2_t))
        else:
            sigma2_t = max(
                omega + a * eps_sq_lag + beta * var_lag, 1e-12,
            )
            sigma_t = float(np.sqrt(sigma2_t))

        eps_t = sigma_t * float(z[t])
        mu_t = mu + phi * (y_lag - mu) + theta * eps_lag
        y_t = mu_t + eps_t
        y[t] = y_t

        # Lag updates.
        if var_model is TGARCH:
            sigma_lag = sigma_t
        elif var_model is QGARCH:
            eps_sq_lag = eps_t * eps_t
            var_lag = sigma2_t
        else:
            eps_sq_lag = eps_t * eps_t
            var_lag = sigma2_t
        eps_lag = eps_t
        y_lag = y_t

    return jnp.asarray(y[_BURN_IN:])


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------

_MATRIX_LABELS = list(RUGARCH_REFERENCE.keys()) + list(_HANDROLLED_LABELS)
_RUGARCH_LABELS = tuple(RUGARCH_REFERENCE.keys())
_FIT_MAXITER = 1500
_FIT_LR = 0.05


def _build_case(label):
    if label in RUGARCH_REFERENCE:
        c = RUGARCH_REFERENCE[label]
        return SimpleNamespace(
            label=label,
            mean_order=c["mean_order"],
            var_model=_VAR_MODEL_FROM_NAME[c["var_model"]],
            var_order=c["var_order"],
            residual_dist=_RESIDUAL_DIST_FROM_NAME[c["residual_dist"]],
            residual_shape_truth=c["residual_shape_truth"],
            y=jnp.asarray(c["y"]),
            rugarch=c,
            handrolled=False,
        )
    truth_phi = _HANDROLLED_TRUTH[label]["phi"]
    truth_theta = _HANDROLLED_TRUTH[label]["theta"]
    mean_order = (len(truth_phi), len(truth_theta))
    seed = abs(hash(label)) % (2**31)
    y = _simulate_handrolled(label, n=2000, key=jax.random.PRNGKey(seed))
    return SimpleNamespace(
        label=label,
        mean_order=mean_order,
        var_model=_HANDROLLED_VAR_MODEL[label],
        var_order=(1, 1),
        residual_dist=_HANDROLLED_RESIDUAL_DIST[label],
        residual_shape_truth=_HANDROLLED_RESIDUAL_TRUTH[label],
        y=y,
        rugarch=None,
        handrolled=True,
    )


def _fit_case(case):
    return ArmaGarch(
        mean_order=case.mean_order,
        var_model=case.var_model,
        var_order=case.var_order,
        residual_dist=case.residual_dist,
    ).fit(case.y, init="analytical", maxiter=_FIT_MAXITER, lr=_FIT_LR)


@pytest.fixture(scope="module", params=_MATRIX_LABELS, ids=lambda x: x)
def matrix_fit(request):
    case = _build_case(request.param)
    case.fit = _fit_case(case)
    return case


@pytest.fixture(scope="module", params=_RUGARCH_LABELS, ids=lambda x: x)
def rugarch_fit(request):
    case = _build_case(request.param)
    case.fit = _fit_case(case)
    return case


@pytest.fixture(scope="module")
def base_fit():
    case = _build_case("arma11_garch11_normal")
    case.fit = _fit_case(case)
    return case


@pytest.fixture(scope="module")
def large_fit():
    """GARCH(1,1)-Normal fit at the rugarch reference n=2000.

    The recovery test treats rugarch's converged parameters as the
    finite-sample target and asserts copulax matches them within an
    SE-scaled budget. Higher n would tighten the budget but require
    a separate rugarch run; the n=2000 reference is sufficient.
    """
    case = _build_case("arma11_garch11_normal")
    case.fit = _fit_case(case)
    return case


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(x):
    return np.asarray(jnp.atleast_1d(jnp.asarray(x, dtype=float))).ravel()


def _wold_psi_coefs(
    phi: np.ndarray, theta: np.ndarray, K: int,
) -> np.ndarray:
    r"""Wold (MA-∞) coefficients ψ_0, ψ_1, …, ψ_K of the ARMA(p, q)
    process with the supplied (φ, θ).

    Recursion (Brockwell-Davis 1991 §3.3):
        ψ_0 = 1,  ψ_k = θ_k + Σ_{j=1}^{min(k,p)} φ_j ψ_{k-j}     (k ≥ 1)
    with θ_k = 0 for k > q.

    Used to construct the cumulative h-step forecast variance
    Var(y_{n+h} | F_n) = Σ_{j=0}^{h-1} ψ_j² · σ²_{n+h-j} for the
    forecast simulation-vs-analytical mean comparison.
    """
    phi = np.asarray(phi, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    p, q = phi.size, theta.size
    psi = np.zeros(K + 1, dtype=float)
    psi[0] = 1.0
    for k in range(1, K + 1):
        v = theta[k - 1] if (k - 1) < q else 0.0
        m = min(k, p)
        for j in range(1, m + 1):
            v += phi[j - 1] * psi[k - j]
        psi[k] = v
    return psi


def _wold_psi_factor(
    phi: np.ndarray, theta: np.ndarray, K: int = 200,
) -> float:
    r"""Σ_{k=0}^{K} ψ_k² for the Wold (MA-∞) representation of the
    stationary ARMA(p, q) process with the supplied (φ, θ).

    Used to derive the **analytical** Var(y_stationary) = σ_ε² · ψ_factor
    for tests that compare simulation-derived sample moments against
    a target — using ``Var(sample)`` to bound ``|mean(sample) − target|``
    is self-consistent but not robust (a broken sampler with inflated
    variance and inflated mean would pass).  The Wold sum bypasses
    that circularity: it depends only on the fitted (φ, θ), not on
    the simulation output.

    K=200 is far past the geometric-decay threshold for any matrix
    entry (every fitted |φ| < 0.95, so |φ|^200 < 1e-4 — truncation
    error on Σψ_k² is below 1e-6 even at the worst case).
    """
    return float(np.sum(_wold_psi_coefs(phi, theta, K) ** 2))


def _residual_kurtosis_via_mc(
    residual_dist, residual_params, key, n: int = 20_000,
) -> float:
    r"""Excess-kurtosis-aware estimate of κ = E[z⁴] / Var(z)² for the
    fitted standardised residual law via independent MC draws.

    Used to derive the per-law SE of a sample variance:
        Var(S²)  ≈  (κ − 1) · σ⁴ / n
    so SE(S²) on standardised residuals (σ=1) is √((κ−1)/n).

    Independence is the point — drawing fresh i.i.d. samples from the
    residual law (with the **fitted** shape parameters) gives a κ
    estimate that's decoupled from the joint-fit's standardised
    residual sample, so the unit-variance test isn't comparing the
    sample to itself.
    """
    z = StandardisedResidual(residual_dist).rvs(
        size=(n,), shape_params=residual_params, key=key,
    )
    z = np.asarray(z)
    z = z[np.isfinite(z)]
    if z.size < n // 2:
        # Heavy-tail / parameter region where the law's moments may
        # not exist; fall back to a conservative κ matching Student-t
        # at ν = 5 (κ = 9).  Surfaces in the assertion message if it
        # ever drives a failure.
        return 9.0
    var_z = float(np.var(z))
    if var_z <= 0.0:
        return 9.0
    return float(np.mean(z ** 4) / var_z ** 2)


def _se_budget_assert(
    fitted_params, target_params, fitted_se, key,
    multiplier=4.0, floor=5e-3, label="",
):
    """Two independent MLEs on the same data agree within a multiple
    of the asymptotic standard error. NaN SEs (constrained params)
    fall back to ``floor``."""
    fitted = _flatten(fitted_params[key])
    target = _flatten(target_params[key])
    if target.size == 0:
        return
    se = _flatten(fitted_se[key])
    budget = multiplier * se + floor
    budget = np.where(np.isfinite(budget), budget, floor)
    diff = np.abs(fitted - target)
    np.testing.assert_array_less(
        diff, budget,
        err_msg=(
            f"{label} key={key!r} fitted={fitted} target={target} "
            f"se={se} diff={diff} budget={budget}"
        ),
    )


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
    """Asymptotic SE-budget recovery against rugarch reference truth.

    rugarch fits on the same simulated y series produce a finite-sample
    parameter estimate; copulax should agree within ~3 standard errors,
    which is the same budget rule used elsewhere in the suite.
    """

    def test_recovery_arma11_garch11_normal(self, base_fit):
        ref = base_fit.rugarch
        target = ref["params"]
        for k in ("phi", "theta", "mu", "omega", "alpha", "beta"):
            _se_budget_assert(
                base_fit.fit.params, target,
                base_fit.fit.standard_errors_, k,
                label=base_fit.label,
            )

    # ARMA(p+q>=3) cases admit multiple near-equivalent optima
    # (Wold-representation roots cancel with MA roots in different
    # arrangements at the same likelihood). copulax and rugarch
    # converge to different but valid optima; SE-budget recovery
    # against rugarch is not the right metric for these cases.
    _HIGH_ORDER_ARMA = frozenset({
        "arma21_garch11_normal", "arma12_garch11_normal",
        "arma22_garch11_normal",
    })

    @pytest.mark.parametrize("label", _RUGARCH_LABELS)
    def test_recovery_per_rugarch_case(self, label):
        if label in self._HIGH_ORDER_ARMA:
            pytest.skip("ARMA(p+q>=3) admits multiple equivalent MLEs")
        case = _build_case(label)
        fit = _fit_case(case)
        target = case.rugarch["params"]
        for k in ("phi", "theta", "mu", "omega", "alpha", "beta", "gamma"):
            if k in target:
                _se_budget_assert(
                    fit.params, target, fit.standard_errors_, k,
                    label=label,
                )


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
            ((1, 1), GARCH, (1, 1), gen_normal),
            ((1, 1), GARCH, (1, 1), nig),
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
    """Joint MLE log-likelihood is at least as high as the two-stage
    separable fit, evaluated at the separable parameter point via
    warm-init with maxiter=0.  Runs across every matrix entry — every
    variance variant in the joint whitelist (GARCH, IGARCH, GJR_GARCH,
    EGARCH, TGARCH, QGARCH) exposes a standalone ``.fit(eps)`` method
    via :class:`GARCHBase`, so the two-stage fit is well-defined for
    all of them."""

    def _separable_warm_eval(self, case):
        p, q = case.mean_order
        p_v, q_v = case.var_order
        arma_fit = ARMA(
            p=p, q=q, residual_dist=case.residual_dist,
        ).fit(case.y, init="analytical", maxiter=_FIT_MAXITER, lr=_FIT_LR)
        eps = arma_fit.residuals(case.y)["residuals"]
        var_fit = case.var_model(
            p=p_v, q=q_v, residual_dist=case.residual_dist,
        ).fit(eps, init="analytical", maxiter=_FIT_MAXITER, lr=_FIT_LR)
        sep = {
            "phi": arma_fit.params["phi"],
            "theta": arma_fit.params["theta"],
            "mu": arma_fit.params["mu"],
            **{k: var_fit.params[k] for k in var_fit._ag_var_keys()},
            "residual": dict(var_fit.params["residual"]),
        }
        return ArmaGarch(
            mean_order=case.mean_order, var_model=case.var_model,
            var_order=case.var_order, residual_dist=case.residual_dist,
        ).fit(case.y, init="warm", init_params=sep, maxiter=0)

    def test_joint_at_least_as_high_as_separable(self, matrix_fit):
        sep_eval = self._separable_warm_eval(matrix_fit)
        joint_ll = float(matrix_fit.fit.loglikelihood())
        sep_ll = float(sep_eval.loglikelihood())
        assert joint_ll >= sep_ll - 1e-3, (
            f"{matrix_fit.label}: joint_ll={joint_ll} < sep_ll={sep_ll}"
        )


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

class TestResiduals:
    def test_residuals_match_y_minus_conditional_mean(self, matrix_fit):
        d = matrix_fit.fit.residuals(matrix_fit.y)
        expected = (
            np.asarray(matrix_fit.y)
            - np.asarray(matrix_fit.fit.conditional_mean(matrix_fit.y))
        )
        np.testing.assert_allclose(
            np.asarray(d["residuals"]), expected, rtol=1e-6, atol=1e-8,
        )

    def test_standardised_residuals_match_residuals_over_sigma(self, matrix_fit):
        d = matrix_fit.fit.residuals(matrix_fit.y)
        sigma = np.sqrt(np.asarray(matrix_fit.fit.conditional_variance(matrix_fit.y)))
        np.testing.assert_allclose(
            np.asarray(d["standardised_residuals"]),
            np.asarray(d["residuals"]) / sigma,
            rtol=1e-6, atol=1e-8,
        )

    def test_standardised_residuals_unit_variance(self, matrix_fit):
        r"""Standardised residuals satisfy mean=0, var=1 within MC SE
        derived per residual law.

        Tolerances:
          * mean: SE(z̄) = 1/√n under the standardisation contract,
            independent of the residual law.
          * var:  SE(S²) = √((κ − 1) / n), where κ is the kurtosis
            of the residual law — heavy-tailed laws (Student-t,
            skewed_t, NIG, GH) have κ > 3 and need a wider bound.
            κ is estimated via independent MC draws from
            ``fit.residual_dist`` so the bound is decoupled from the
            sample being tested.

        4σ on each, ~6e-5 false-positive rate per matrix entry.
        """
        fit = matrix_fit.fit
        z = np.asarray(fit.residuals(matrix_fit.y)["standardised_residuals"])
        n = z.size

        se_mean = 1.0 / np.sqrt(n)
        kappa = _residual_kurtosis_via_mc(
            fit.residual_dist,
            fit.residual_params,
            jax.random.PRNGKey(7),
        )
        se_var = np.sqrt(max(kappa - 1.0, 0.0) / n)

        np.testing.assert_allclose(
            z.mean(), 0.0, atol=4.0 * se_mean,
            err_msg=f"{matrix_fit.label}: mean(z) outside 4·SE",
        )
        np.testing.assert_allclose(
            z.var(), 1.0, atol=4.0 * se_var,
            err_msg=(
                f"{matrix_fit.label}: var(z) outside 4·SE "
                f"(κ={kappa:.3f}, n={n}, SE={se_var:.4f})"
            ),
        )

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
            for k in cached:
                cv, rv = cached[k], recomp[k]
                if isinstance(cv, dict):
                    continue
                np.testing.assert_allclose(
                    np.asarray(jnp.asarray(cv, dtype=float)),
                    np.asarray(jnp.asarray(rv, dtype=float)),
                    rtol=1e-5, atol=1e-8,
                    err_msg=f"{accessor}.{k}",
                )

    def test_residual_diagnostics_dict_keys(self, matrix_fit):
        expected = {
            "loglikelihood", "aic", "bic", "acf", "pacf",
            "ljung_box", "ljung_box_sq", "arch_lm", "adf", "kpss",
        }
        assert set(matrix_fit.fit.residual_diagnostics_) == expected


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_unconditional_mean_formula(self, matrix_fit):
        fit = matrix_fit.fit
        s = fit.stats()
        # Centred-form ARMA: stats["unconditional_mean"] is a trivial
        # accessor returning μ directly (no AR rescaling).
        np.testing.assert_allclose(
            float(s["unconditional_mean"]), float(fit.params["mu"]),
            rtol=1e-6, atol=1e-8,
        )

    def test_var_persistence_consistency(self, matrix_fit):
        """Production-library persistence is ≥ 0 and (for stationary
        variants) < 1 + small float-noise tolerance."""
        s = matrix_fit.fit.stats()
        p = float(s["var_persistence"])
        assert p >= 0.0 - 1e-9
        if matrix_fit.var_model is IGARCH:
            np.testing.assert_allclose(p, 1.0, atol=1e-6)
        else:
            assert p < 1.0 + 1e-3

    def test_var_unconditional_variance_consistency(self, matrix_fit):
        s = matrix_fit.fit.stats()
        if matrix_fit.var_model is IGARCH:
            assert np.isinf(float(s["var_unconditional_variance"]))
            return
        v = float(s["var_unconditional_variance"])
        assert v > 0.0
        assert np.isfinite(v)

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
        s = matrix_fit.fit.stats()
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

    def test_unconditional_moments_via_simulation(self, matrix_fit):
        """Long Monte-Carlo paths' terminal sample mean agrees with
        ``stats()["unconditional_mean"]`` and per-path innovation
        variance agrees with ``var_unconditional_variance`` within
        MC error.

        Innovations are extracted via ``fit.residuals(path)`` so the
        variance check is on ``Var(eps)`` (matching the analytical
        target), not ``Var(y)`` which carries the ARMA-factor scale.
        """
        fit = matrix_fit.fit
        s = fit.stats()
        if matrix_fit.var_model is IGARCH:
            pytest.skip("IGARCH has no unconditional moments")
        if not bool(s["mean_is_stationary"]):
            pytest.skip("non-stationary mean")
        if not bool(s["var_is_stationary"]):
            pytest.skip("non-stationary variance")
        n_paths = 400
        h = 600
        paths = fit.rvs(
            size=(n_paths, h), key=jax.random.PRNGKey(2024),
        )
        # Mean check on y.  ``terminal_y`` is n_paths i.i.d. draws from
        # the stationary distribution (h ≫ AR transient at every matrix
        # entry), so the sample mean has standard error
        # sqrt(Var(y_stationary) / n_paths).
        #
        # Var(y_stationary) is computed *analytically* from the fitted
        # (φ, θ) via the Wold ψ-factor — using the simulation's
        # empirical variance to bound its empirical mean would be
        # circular: a broken sampler producing both inflated mean and
        # inflated variance would pass.  The analytical bound depends
        # only on the fitted parameters, so a wrong-mean failure is
        # detected even if the variance is also wrong.
        terminal_y = np.asarray(paths[:, -1])
        target_mean = float(s["unconditional_mean"])
        target_var_eps = float(s["var_unconditional_variance"])
        psi_factor = _wold_psi_factor(
            np.asarray(fit.params["phi"]),
            np.asarray(fit.params["theta"]),
        )
        target_var_y = target_var_eps * psi_factor
        mc_se_mean = float(np.sqrt(target_var_y / n_paths))
        np.testing.assert_array_less(
            np.abs(terminal_y.mean() - target_mean),
            4.0 * mc_se_mean,
        )
        # Variance check on eps: extract residuals from each path,
        # compute the per-path empirical variance over the post-
        # transient half of the trajectory, then pool across paths.
        # Each path contributes an i.i.d. estimate of σ²_ε, so the
        # pooled mean has standard error √(Var(per_path_var) / n_paths)
        # — pooling raw eps² values would inflate N spuriously
        # (autocorrelation in eps² under GARCH), and mishandling
        # effective N is what the original ``rtol=0.15`` was hiding.
        #
        # EGARCH skipped here: ``stats()["var_unconditional_variance"]``
        # returns ``exp(ω / (1 − Σβ))`` — the *geometric* mean of σ²_t
        # under the stationary distribution — while the simulation
        # produces the *arithmetic* mean ``E[σ²_t]``.  The two differ
        # by a Jensen-inequality factor that depends on Var(log σ²_t)
        # (i.e. α and γ).  This is the **industry convention**:
        # rugarch's ``uncvariance(fit)`` returns the same formula
        # for EGARCH (verified empirically).  Not a bug, but the
        # simulation-vs-stats comparison can't reconcile it; other
        # variants agree to <1.5% MC noise.
        if matrix_fit.var_model is EGARCH:
            return
        eps_per_path = jax.vmap(
            lambda yi: fit.residuals(yi)["residuals"]
        )(paths)
        eps_late = np.asarray(eps_per_path[:, h // 2:])
        per_path_var = np.var(eps_late, axis=1, ddof=1)  # (n_paths,)
        pooled_var = float(np.mean(per_path_var))
        mc_se_var = float(np.sqrt(np.var(per_path_var, ddof=1) / n_paths))
        np.testing.assert_array_less(
            np.abs(pooled_var - target_var_eps),
            4.0 * mc_se_var,
            err_msg=(
                f"{matrix_fit.label}: pooled_var={pooled_var:.4f}, "
                f"target={target_var_eps:.4f}, mc_se={mc_se_var:.4f}"
            ),
        )


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

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
        r"""h-step ARMA mean forecast converges geometrically to μ at
        rate ``decay_rate = 1 / min(|AR roots|)``:

            mu_h − μ  =  (mu_0 − μ) · decay_rate^h     (asymptotically)

        At h=2000 with AR-root moduli ≥ 1.05 (decay ≤ 0.95) the
        residual gap is ≤ 1e-44, well below float64 round-off.  Use
        a bound calibrated to the fitted decay rate rather than a
        flat ``rtol=1e-3`` (which was 5+ orders of magnitude looser
        than the math required).
        """
        if matrix_fit.var_model in _NO_ANALYTICAL_VARIANTS:
            pytest.skip("no analytical h>=2")
        fit = matrix_fit.fit
        s = fit.stats()
        if not bool(s["mean_is_stationary"]):
            pytest.skip("non-stationary mean")
        h = 2000
        fc = fit.forecast(h=h, method="analytical")
        target = float(s["unconditional_mean"])

        # Geometric decay rate from the AR characteristic-polynomial
        # roots.  For pure-MA models (ar_root_moduli empty), the mean
        # forecast equals μ exactly after q steps — gap is 0 modulo
        # round-off.
        ar_moduli = np.asarray(s["ar_root_moduli"])
        decay = 1.0 / float(np.min(ar_moduli)) if ar_moduli.size > 0 else 0.0
        # Initial gap |mu_0 − μ| from the first analytical forecast
        # point.  Multiplying by decay^h gives the residual at h.
        initial_gap = abs(float(fc["mean"][0]) - target)
        # 1e-9 floor absorbs float64 round-off accumulating over h
        # scan steps.
        residual_bound = max(initial_gap * decay ** h, 1e-9)
        np.testing.assert_allclose(
            float(fc["mean"][-1]), target,
            atol=residual_bound,
            err_msg=(
                f"{matrix_fit.label}: decay={decay:.4f}, "
                f"initial_gap={initial_gap:.4e}, "
                f"residual_bound={residual_bound:.2e}"
            ),
        )

    def test_analytical_variance_converges_to_unconditional(self, matrix_fit):
        r"""At horizon h, the residual gap to the unconditional variance
        decays geometrically as ``persistence^h``.  At h=2000 with
        persistence < 0.99 the residual is below 2e-9 of the gap; even
        at persistence=0.999 (boundary IGARCH) it would be ~0.13.  Use
        a bound calibrated to the fitted persistence rather than a
        flat tolerance.
        """
        if matrix_fit.var_model is IGARCH:
            pytest.skip("IGARCH has no unconditional variance")
        if matrix_fit.var_model in _NO_ANALYTICAL_VARIANTS:
            pytest.skip("no analytical h>=2")
        fit = matrix_fit.fit
        s = fit.stats()
        if not bool(s["var_is_stationary"]):
            pytest.skip("non-stationary variance")
        h = 2000
        fc = fit.forecast(h=h, method="analytical")
        target = float(s["var_unconditional_variance"])
        persistence = float(s["var_persistence"])
        # Theoretical residual gap bound: with σ²_0 floored at 0 and
        # target σ²_∞ = ω/(1−persistence), the residual at h is
        # (σ²_0 − σ²_∞) · persistence^h.  Bound below by 1e-9 to
        # absorb float64 round-off accumulating over 2000 scan steps.
        residual_bound = max(
            target * persistence ** h,
            1e-9,
        )
        np.testing.assert_allclose(
            float(fc["variance"][-1]), target,
            atol=residual_bound,
            err_msg=(
                f"{matrix_fit.label}: persistence={persistence:.4f}, "
                f"residual bound={residual_bound:.2e}"
            ),
        )

    def test_simulation_mean_matches_analytical(self, matrix_fit):
        r"""Simulation-mean trajectory matches the analytical mean
        trajectory within MC standard error.

        ``sim["mean"][t]`` is the mean of n_paths simulated paths at
        horizon t.  Its sampling variance equals the **conditional
        h-step forecast variance** of the level series:

            Var(y_{n+t} | F_n)  =  Σ_{k=0}^{t-1} ψ_k² · σ²_{n+t-k}

        where ψ_k are the Wold-MA(∞) coefficients of the ARMA part
        and σ²_{n+s} are the GARCH variance forecasts at horizons
        s = 1 … t (= ``analytical["variance"][s-1]``).  The original
        bound used ``analytical["variance"][t-1]`` as a stand-in for
        Var(y_{n+t}|F_n), which is **wrong-shape** — it's the per-step
        innovation variance, not the cumulative forecast variance,
        and the cumulative form is strictly larger by a factor of
        Σψ_k² (~1.3-2× for the matrix entries).  A 5× multiplier
        was hiding the shape error; the principled bound uses the
        correct cumulative variance with a 4σ z-bound.
        """
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

        # Cumulative h-step forecast variance per horizon t ∈ [1, h].
        # Var(y_{n+t}|F_n) = Σ_{k=0}^{t-1} ψ_k² · σ²_{n+t-k}
        #                 = Σ_{s=1}^{t}  ψ_{t-s}² · ana_var[s-1]
        psi = _wold_psi_coefs(
            np.asarray(fit.params["phi"]),
            np.asarray(fit.params["theta"]),
            K=h - 1,
        )
        forecast_var = np.array([
            float(np.sum(
                (psi[: t][::-1] ** 2) * ana_var[: t]
            ))
            for t in range(1, h + 1)
        ])
        mc_se = np.sqrt(forecast_var / n_paths)
        np.testing.assert_array_less(
            np.abs(sim_mean - ana_mean),
            4.0 * mc_se,
            err_msg=f"{matrix_fit.label}: forecast mean MC mismatch",
        )

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
        label = (
            "arma11_egarch11_normal" if var_model is EGARCH
            else "arma11_tgarch11_normal"
        )
        case = _build_case(label)
        fit = _fit_case(case)
        fc1 = fit.forecast(h=1, method="analytical")
        assert fc1["variance"].shape == (1,)
        with pytest.raises(ValueError, match=var_model.__name__):
            fit.forecast(h=10, method="analytical")


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
        out = matrix_fit.fit.rvs(size=(7, 12), key=jax.random.PRNGKey(0))
        assert out.shape == (7, 12)
        assert np.all(np.isfinite(np.asarray(out)))

    def test_rvs_size_or_u_required(self, base_fit):
        with pytest.raises(ValueError):
            base_fit.fit.rvs()

    def test_rvs_3d_u_raises(self, base_fit):
        u = jnp.zeros((2, 3, 4)) + 0.5
        with pytest.raises(ValueError):
            base_fit.fit.rvs(u=u)

    def test_ag_rvs_step_var_t_independent_of_z_t(self, matrix_fit):
        """The variance backend's ``_ag_rvs_step`` returns a ``var_t``
        that does not depend on ``z_t``; the joint composite scan
        relies on this for a single-pass step. This test pins the
        contract from BOTH directions: var_t is z-invariant, AND the
        new state IS z-sensitive (else the variant is degenerate)."""
        fit = matrix_fit.fit
        backend = fit._var_backend
        var_state = fit.terminal_state.var_state
        var_t_a, eps_a, state_a = backend._ag_rvs_step(
            fit.var_params, fit.residual_params, var_state,
            jnp.asarray(0.5, dtype=float),
        )
        var_t_b, eps_b, state_b = backend._ag_rvs_step(
            fit.var_params, fit.residual_params, var_state,
            jnp.asarray(-1.7, dtype=float),
        )
        np.testing.assert_allclose(
            np.asarray(var_t_a), np.asarray(var_t_b),
        )
        assert not np.isclose(np.asarray(eps_a), np.asarray(eps_b))
        leaves_a = jax.tree_util.tree_leaves(state_a)
        leaves_b = jax.tree_util.tree_leaves(state_b)
        any_diff = any(
            not np.allclose(np.asarray(a), np.asarray(b), atol=1e-12)
            for a, b in zip(leaves_a, leaves_b)
        )
        assert any_diff, (
            f"{matrix_fit.label}: new var_state did not change in "
            "response to z_t; the variant either ignores z_t or has "
            "a broken state-update path."
        )


# ---------------------------------------------------------------------------
# Variant invariants
# ---------------------------------------------------------------------------

class TestVariantInvariants:
    def test_igarch_persistence_pinned(self):
        case = _build_case("arma11_igarch11_normal")
        fit = _fit_case(case)
        persistence = (
            float(fit.params["alpha"][0]) + float(fit.params["beta"][0])
        )
        np.testing.assert_allclose(persistence, 1.0, atol=1e-6)

    def test_qgarch_positivity_invariant(self):
        case = _build_case("arma11_qgarch11_normal")
        fit = _fit_case(case)
        omega = float(fit.params["omega"])
        alpha = float(np.asarray(fit.params["alpha"]).reshape(-1)[0])
        psi = float(np.asarray(fit.params["psi"]).reshape(-1)[0])
        assert omega + 1e-9 >= psi * psi / (4.0 * alpha)

    def test_gjr_persistence_below_one(self):
        case = _build_case("arma11_gjr11_normal")
        fit = _fit_case(case)
        s = fit.stats()
        assert float(s["var_persistence"]) < 1.0


# ---------------------------------------------------------------------------
# JIT
# ---------------------------------------------------------------------------

class TestJIT:
    """Two layers, both matrix-parametrised:

    Layer 1: the post-fit object's full ``y``-consuming surface runs
    cleanly under a single ``jax.jit`` wrapper for every matrix
    combination.

    Layer 2: the entire fit pipeline (``ArmaGarch(...).fit(y)``) runs
    under ``jax.jit`` for every matrix combination. This is the
    contract a downstream user wrapping the fit in an outer JAX loop
    relies on.
    """

    def test_jit_object_full_surface(self, matrix_fit):
        fit = matrix_fit.fit
        y = matrix_fit.y

        @jax.jit
        def call_all(yy):
            return {
                "ll":        fit.loglikelihood(yy),
                "aic":       fit.aic(yy),
                "bic":       fit.bic(yy),
                "cond_mean": fit.conditional_mean(yy),
                "cond_var":  fit.conditional_variance(yy),
                "resid":     fit.residuals(yy)["residuals"],
                "z":         fit.residuals(yy)["standardised_residuals"],
                "acf":       fit.acf(yy),
                "pacf":      fit.pacf(yy),
            }
        jitted = call_all(y)
        eager = {
            "ll": fit.loglikelihood(y),
            "aic": fit.aic(y),
            "bic": fit.bic(y),
            "cond_mean": fit.conditional_mean(y),
            "cond_var": fit.conditional_variance(y),
            "resid": fit.residuals(y)["residuals"],
            "z": fit.residuals(y)["standardised_residuals"],
            "acf": fit.acf(y),
            "pacf": fit.pacf(y),
        }
        for k in eager:
            np.testing.assert_allclose(
                np.asarray(jitted[k]), np.asarray(eager[k]),
                rtol=1e-5, atol=1e-7,
                err_msg=f"{matrix_fit.label}.{k}",
            )

    def test_jit_rvs(self, matrix_fit):
        fit = matrix_fit.fit
        key = jax.random.PRNGKey(42)
        eager = fit.rvs(size=(8, 12), key=key)
        jitted = jax.jit(
            lambda k: fit.rvs(size=(8, 12), key=k)
        )(key)
        np.testing.assert_allclose(
            np.asarray(jitted), np.asarray(eager),
            rtol=1e-6, atol=1e-8,
        )

    def test_jit_forecast_simulation(self, matrix_fit):
        fit = matrix_fit.fit
        key = jax.random.PRNGKey(7)
        eager = fit.forecast(
            h=5, method="simulation", n_paths=20, key=key,
        )
        jitted = jax.jit(
            lambda k: fit.forecast(
                h=5, method="simulation", n_paths=20, key=k,
            )
        )(key)
        for f in ("mean", "variance", "paths"):
            np.testing.assert_allclose(
                np.asarray(jitted[f]), np.asarray(eager[f]),
                rtol=1e-6, atol=1e-8,
            )

    def test_jit_fit_end_to_end(self, matrix_fit):
        cfg = matrix_fit
        y = cfg.y

        def fit_fn(yy):
            return ArmaGarch(
                mean_order=cfg.mean_order, var_model=cfg.var_model,
                var_order=cfg.var_order, residual_dist=cfg.residual_dist,
            ).fit(yy, init="analytical", maxiter=100, lr=_FIT_LR)

        eager = fit_fn(y)
        jitted = jax.jit(fit_fn)(y)
        for k in ("phi", "theta", "mu"):
            np.testing.assert_allclose(
                _flatten(jitted.params[k]), _flatten(eager.params[k]),
                rtol=1e-5, atol=1e-6,
                err_msg=f"{cfg.label}.{k}",
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
        for k in ("phi", "theta", "mu", "omega", "alpha", "beta"):
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
# Init-mode convergence (rugarch-anchored)
# ---------------------------------------------------------------------------

_INIT_MODES = ("analytical", "backcast", "sample")
_PAIRWISE_LABELS = (
    "arma11_garch11_normal",
    "arma11_igarch11_normal",
    "arma11_gjr11_normal",
    "arma11_egarch11_normal",
    "arma11_tgarch11_normal",
    "arma11_qgarch11_normal",
)


class TestInitModesConvergence:
    """Three init modes (analytical / backcast / sample) converge to
    the same MLE on every supported variant. For variants with a
    rugarch reference, every mode must also match rugarch's converged
    fit. Replaces the prior smoke ``TestInitModes``."""

    def _fit_with_init(self, label, mode, maxiter=2000):
        case = _build_case(label)
        return ArmaGarch(
            mean_order=case.mean_order, var_model=case.var_model,
            var_order=case.var_order, residual_dist=case.residual_dist,
        ).fit(case.y, init=mode, maxiter=maxiter, lr=_FIT_LR)

    @pytest.mark.parametrize("label", _PAIRWISE_LABELS)
    @pytest.mark.parametrize(
        "modes",
        [
            ("analytical", "backcast"),
            ("analytical", "sample"),
            ("backcast", "sample"),
        ],
    )
    def test_pairwise_convergence(self, label, modes):
        m1, m2 = modes
        f1 = self._fit_with_init(label, m1)
        f2 = self._fit_with_init(label, m2)
        np.testing.assert_allclose(
            float(f1.loglikelihood()), float(f2.loglikelihood()),
            rtol=5e-3,
        )

    @pytest.mark.parametrize(
        "label", [l for l in _PAIRWISE_LABELS if l in RUGARCH_REFERENCE],
    )
    @pytest.mark.parametrize("mode", _INIT_MODES)
    def test_each_mode_matches_rugarch(self, label, mode):
        ref = RUGARCH_REFERENCE[label]
        fit = self._fit_with_init(label, mode)
        for k in ("phi", "theta", "mu", "omega", "alpha", "beta"):
            if k in ref["params"] and len(_flatten(ref["params"][k])) > 0:
                _se_budget_assert(
                    fit.params, ref["params"], fit.standard_errors_, k,
                    label=f"{label} mode={mode}",
                )

    def test_unknown_init_mode_raises(self, base_fit):
        with pytest.raises(ValueError):
            ArmaGarch(
                mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
                residual_dist=normal,
            ).fit(base_fit.y, init="bogus", maxiter=10)


# ---------------------------------------------------------------------------
# Rugarch reference cross-validation
# ---------------------------------------------------------------------------

class TestRugarchReference:
    """Joint-fit parameter, log-likelihood, AIC/BIC, forecast, and
    standard-error agreement with rugarch on every reference case."""

    def test_params_match_rugarch(self, rugarch_fit):
        """copulax and rugarch fit the same model on the same data;
        their MLEs agree within ~4 standard errors. Skipped on
        ARMA(p+q>=3) cases where multiple equivalent optima exist."""
        if rugarch_fit.label in TestRecovery._HIGH_ORDER_ARMA:
            pytest.skip("ARMA(p+q>=3) admits multiple equivalent MLEs")
        ref = rugarch_fit.rugarch
        fit = rugarch_fit.fit
        for k in ("phi", "theta", "mu", "omega", "alpha", "beta", "gamma"):
            if k in ref["params"]:
                _se_budget_assert(
                    fit.params, ref["params"], fit.standard_errors_, k,
                    label=rugarch_fit.label,
                )

    def test_loglikelihood_matches_rugarch(self, rugarch_fit):
        """copulax and rugarch use different solvers (Adam projected
        gradient vs L-BFGS-B with restarts); a sub-1% LL gap is
        expected and tolerated."""
        ref = rugarch_fit.rugarch
        np.testing.assert_allclose(
            float(rugarch_fit.fit.loglikelihood()),
            float(ref["loglikelihood"]),
            rtol=1e-2,
        )

    def test_aic_bic_match_rugarch(self, rugarch_fit):
        ref = rugarch_fit.rugarch
        np.testing.assert_allclose(
            float(rugarch_fit.fit.aic()), float(ref["aic"]), rtol=1e-2,
        )
        np.testing.assert_allclose(
            float(rugarch_fit.fit.bic()), float(ref["bic"]), rtol=1e-2,
        )

    def test_forecast_matches_rugarch(self, rugarch_fit):
        r"""Forecast trajectories agree within solver-noise tolerance.

        Bounds are split per-variant from the per-case enumeration:

        * Mean forecasts converge geometrically to μ; per-case max
          absolute error is < 0.02 across the matrix.  ``rtol`` is
          mostly redundant since most forecast points have target
          near zero.
        * Variance forecasts agree to <1.5% on every variant *except*
          IGARCH, whose constrained simplex (α + β = 1) makes the
          variance trajectory diverge; small MLE differences amplify
          to ~5% rel / 0.22 abs by h=20.

        Skipped on ARMA(p+q>=3) cases where copulax and rugarch
        converge to different equivalent optima.
        """
        if rugarch_fit.label in TestRecovery._HIGH_ORDER_ARMA:
            pytest.skip("ARMA(p+q>=3) admits multiple equivalent MLEs")
        if rugarch_fit.var_model in _NO_ANALYTICAL_VARIANTS:
            pytest.skip("no analytical h>=2")
        ref = rugarch_fit.rugarch
        fc = rugarch_fit.fit.forecast(h=20, method="analytical")
        # Mean is a contraction toward μ; tight bound is principled.
        np.testing.assert_allclose(
            np.asarray(fc["mean"]), np.asarray(ref["forecast_mean"]),
            rtol=1e-2, atol=2e-2,
            err_msg=f"{rugarch_fit.label}: mean trajectory",
        )
        # IGARCH variance diverges; needs a wider bound on a divergent
        # trajectory amplified by small cross-library MLE differences.
        if rugarch_fit.var_model is IGARCH:
            np.testing.assert_allclose(
                np.asarray(fc["variance"]),
                np.asarray(ref["forecast_variance"]),
                rtol=0.10, atol=0.30,
                err_msg=f"{rugarch_fit.label}: IGARCH variance trajectory",
            )
        else:
            np.testing.assert_allclose(
                np.asarray(fc["variance"]),
                np.asarray(ref["forecast_variance"]),
                rtol=2e-2, atol=2e-2,
                err_msg=f"{rugarch_fit.label}: variance trajectory",
            )

    def test_standard_errors_match_rugarch(self, rugarch_fit):
        r"""rugarch reports inverse-Hessian (classical) standard errors
        in its matcoef table.  copulax's ``cov_type="classic"`` is the
        same estimator (inverse observed Hessian at the MLE), so the
        two should agree to solver-noise tolerance.

        Per-parameter enumeration: most matrix entries agree to
        <1.5% rel; a few outliers reach 3-3.5% (driven by 1% LL gap
        propagating through the Hessian inversion).  ``rtol=0.05``
        admits every case in the matrix with comfortable margin.

        EGARCH is skipped: its log-variance reparameterisation gives
        the omega/beta SEs a different scaling between the two
        libraries even after the alpha-gamma label swap.
        ARMA(p+q>=3) cases are skipped because the underlying
        parameter optima differ between libraries.
        """
        if rugarch_fit.var_model is EGARCH:
            pytest.skip(
                "EGARCH log-form reparameterises omega/beta; "
                "classical SEs differ across libraries even at the same MLE"
            )
        if rugarch_fit.label in TestRecovery._HIGH_ORDER_ARMA:
            pytest.skip("ARMA(p+q>=3) admits multiple equivalent MLEs")
        ref = rugarch_fit.rugarch
        fit_se = rugarch_fit.fit.standard_errors(
            rugarch_fit.y, cov_type="classic",
        )
        for k in ("phi", "theta", "mu", "omega", "alpha", "beta", "gamma"):
            if k not in ref["standard_errors"]:
                continue
            target = _flatten(ref["standard_errors"][k])
            if target.size == 0:
                continue
            mask = np.isfinite(target)
            if not np.any(mask):
                continue
            np.testing.assert_allclose(
                _flatten(fit_se[k])[mask], target[mask],
                rtol=0.05, atol=2e-3,
                err_msg=f"{rugarch_fit.label}.{k}",
            )


# ---------------------------------------------------------------------------
# Diagnostics cross-validation (rugarch)
# ---------------------------------------------------------------------------

class TestDiagnosticsCrossValidation:
    r"""Cached Ljung-Box and Q-stat-on-squared-residuals match rugarch
    on every reference case.  ADF / KPSS aren't part of rugarch's
    standard fit summary; those are validated against statsmodels in
    :mod:`test_timeseries_diagnostics`.

    Tolerance reasoning (from the per-case enumeration):

    * Q is computed by both libraries via the textbook formula
      ``Q = n(n+2) Σ ρ̂_k² / (n−k)`` on the **standardised** residuals
      at lag 10 — implementations agree to <0.5% when both are at
      the same MLE.
    * ``HIGH_ORDER_ARMA`` cases admit multiple equivalent optima;
      copulax and rugarch settle at slightly different points, and
      the standardised-residual ACF (and hence Q) differs at the
      ~5-15% scale.  Skipped here for the same reason as
      :meth:`TestRecovery.test_recovery_per_rugarch_case`.
    * IGARCH's constrained simplex (α + β = 1) places the MLE at
      a slightly different point than rugarch's solver does — the
      Q-divergence on residuals is ~5%, and on squared residuals
      ~10%.  Within the ``rtol=0.10`` cross-library budget.
    """

    def test_ljung_box_matches_rugarch(self, rugarch_fit):
        if rugarch_fit.label in TestRecovery._HIGH_ORDER_ARMA:
            pytest.skip("ARMA(p+q>=3) admits multiple equivalent MLEs")
        ref = rugarch_fit.rugarch
        cx = rugarch_fit.fit.ljung_box()
        np.testing.assert_allclose(
            float(cx["statistic"]),
            float(ref["ljung_box_statistic"]),
            rtol=0.10, atol=0.2,
            err_msg=f"{rugarch_fit.label}",
        )

    def test_ljung_box_sq_matches_rugarch(self, rugarch_fit):
        if rugarch_fit.label in TestRecovery._HIGH_ORDER_ARMA:
            pytest.skip("ARMA(p+q>=3) admits multiple equivalent MLEs")
        ref = rugarch_fit.rugarch
        # ljung_box_sq is a cached residual_diagnostics_ entry; the
        # canonical accessor name is .ljung_box(on='squared') in
        # production but the cached path is keyed differently.
        # Read directly from residual_diagnostics_.
        cx_sq = rugarch_fit.fit.residual_diagnostics_["ljung_box_sq"]
        np.testing.assert_allclose(
            float(cx_sq["statistic"]),
            float(ref["ljung_box_sq_statistic"]),
            rtol=0.10, atol=0.2,
            err_msg=f"{rugarch_fit.label}",
        )


# ---------------------------------------------------------------------------
# Model-selection consistency (rugarch-anchored)
# ---------------------------------------------------------------------------

_MODEL_RANK_LABELS = (
    "arma11_garch11_normal",
    "arma11_igarch11_normal",
    "arma11_gjr11_normal",
    "arma11_egarch11_normal",
)


class TestModelSelectionConsistency:
    """AIC and BIC rankings across (GARCH, IGARCH, GJR, EGARCH) on
    the same series produce the same ordering in copulax as rugarch.
    Catches a defect in the IC formula without requiring exact
    agreement on absolute values."""

    def test_aic_ranking_matches_rugarch(self):
        # All four variants are fitted on the same simulated y from
        # the GARCH-Normal rugarch case.
        y = jnp.asarray(RUGARCH_REFERENCE["arma11_garch11_normal"]["y"])
        cx_aics = {}
        rg_aics = {}
        for label in _MODEL_RANK_LABELS:
            ref = RUGARCH_REFERENCE[label]
            cls = _VAR_MODEL_FROM_NAME[ref["var_model"]]
            fit = ArmaGarch(
                mean_order=ref["mean_order"], var_model=cls,
                var_order=ref["var_order"], residual_dist=normal,
            ).fit(y, init="analytical", maxiter=_FIT_MAXITER, lr=_FIT_LR)
            cx_aics[label] = float(fit.aic())
            rg_aics[label] = float(ref["aic"])
        cx_rank = sorted(cx_aics, key=lambda k: cx_aics[k])
        rg_rank = sorted(rg_aics, key=lambda k: rg_aics[k])
        assert cx_rank == rg_rank, (
            f"AIC rank mismatch: copulax={cx_rank} rugarch={rg_rank}"
        )

    def test_bic_ranking_matches_rugarch(self):
        y = jnp.asarray(RUGARCH_REFERENCE["arma11_garch11_normal"]["y"])
        cx_bics = {}
        rg_bics = {}
        for label in _MODEL_RANK_LABELS:
            ref = RUGARCH_REFERENCE[label]
            cls = _VAR_MODEL_FROM_NAME[ref["var_model"]]
            fit = ArmaGarch(
                mean_order=ref["mean_order"], var_model=cls,
                var_order=ref["var_order"], residual_dist=normal,
            ).fit(y, init="analytical", maxiter=_FIT_MAXITER, lr=_FIT_LR)
            cx_bics[label] = float(fit.bic())
            rg_bics[label] = float(ref["bic"])
        cx_rank = sorted(cx_bics, key=lambda k: cx_bics[k])
        rg_rank = sorted(rg_bics, key=lambda k: rg_bics[k])
        assert cx_rank == rg_rank, (
            f"BIC rank mismatch: copulax={cx_rank} rugarch={rg_rank}"
        )


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_loglikelihood_grad_finite(self, matrix_fit):
        """``jax.grad`` of the log-likelihood w.r.t. fitted parameters
        is finite on every variant. Catches non-differentiable paths
        through the recursion."""
        fit = matrix_fit.fit
        y = matrix_fit.y

        def ll(phi, theta, mu):
            m = eqx.tree_at(
                lambda f: (f.phi, f.theta, f.mu), fit, (phi, theta, mu),
            )
            return m.loglikelihood(y)

        g = jax.grad(ll, argnums=(0, 1, 2))(fit.phi, fit.theta, fit.mu)
        for arr in jax.tree_util.tree_leaves(g):
            assert jnp.all(jnp.isfinite(arr)), (
                f"{matrix_fit.label}: non-finite gradient"
            )

    def test_determinism_same_data_same_init(self, base_fit):
        """Same data + same init + same maxiter -> reproducible fit."""
        y = base_fit.y
        a = ArmaGarch(
            mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, init="analytical", maxiter=300, lr=_FIT_LR)
        b = ArmaGarch(
            mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, init="analytical", maxiter=300, lr=_FIT_LR)
        for k in ("phi", "theta", "mu", "omega", "alpha", "beta"):
            np.testing.assert_allclose(
                _flatten(a.params[k]), _flatten(b.params[k]),
                rtol=1e-12, atol=1e-12,
            )

    def test_short_series_fits(self, base_fit):
        """Short n=120 series produces a finite log-likelihood."""
        y_short = base_fit.y[:120]
        fit = ArmaGarch(
            mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y_short, init="analytical", maxiter=200, lr=_FIT_LR)
        assert jnp.isfinite(fit.loglikelihood())

    def test_near_stationary_garch_converges(self):
        r"""High-persistence GARCH(1,1) (α+β = 0.99) recovers the truth
        parameters within an SE-budget that's adapted to the
        boundary regime.

        Near the integrated-GARCH boundary the Hessian becomes
        near-singular and individual SEs blow up; finite-sample MLE
        bias on (ω, α, β) is also well-known (Lumsdaine 1995).  The
        budgets below are calibrated to that regime — strict enough
        to fail an optimiser that diverges or produces a garbage fit,
        loose enough to admit standard finite-sample bias at
        n=1500.

        Replaces the prior pure-finiteness check, which passed
        vacuously on any non-NaN result regardless of correctness.
        """
        n = 1500
        key = jax.random.PRNGKey(99)
        truth = {
            "phi": (0.3,), "theta": (0.0,), "mu": 0.0,
            "omega": 0.001, "alpha": (0.05,), "beta": (0.94,),
        }
        # Hand-rolled centred-form simulator used here to keep the
        # case parametrised by truth, not by rugarch reference data.
        z = np.asarray(jax.random.normal(key, (n + _BURN_IN,)))
        eps_sq_lag = 1.0
        var_lag = 1.0
        y = np.zeros(n + _BURN_IN)
        y_lag = float(truth["mu"])
        for t in range(n + _BURN_IN):
            sigma2 = max(
                truth["omega"] + truth["alpha"][0] * eps_sq_lag
                + truth["beta"][0] * var_lag,
                1e-12,
            )
            sigma = float(np.sqrt(sigma2))
            eps = sigma * float(z[t])
            y_t = truth["mu"] + truth["phi"][0] * (y_lag - truth["mu"]) + eps
            y[t] = y_t
            eps_sq_lag = eps * eps
            var_lag = sigma2
            y_lag = y_t
        y_short = jnp.asarray(y[_BURN_IN:])
        fit = ArmaGarch(
            mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y_short, init="analytical", maxiter=600, lr=_FIT_LR)

        # All fitted params are finite (non-NaN).  Pre-condition for
        # any recovery check.
        for k in ("phi", "mu", "omega", "alpha", "beta"):
            assert np.all(np.isfinite(_flatten(fit.params[k]))), (
                f"non-finite {k}: {fit.params[k]}"
            )

        # Recovery budgets calibrated to high-persistence regime.
        # ω is on a small absolute scale (truth=0.001) and is the
        # parameter most affected by near-IGARCH bias; allow a wide
        # absolute slack relative to its own scale.  φ, μ are
        # mean-equation parameters with standard √n bias; α, β must
        # stay close to truth or the variance-equation persistence
        # is mis-recovered.
        budgets = {
            "phi":   0.10,   # ~3.3× SE at n=1500
            "mu":    0.10,   # ~3.3× SE at n=1500
            "omega": 0.005,  # 5× truth — finite-sample bias scale
            "alpha": 0.05,   # 1× truth — persistence-decomposition slack
            "beta":  0.05,   # ~1.3× SE; tighter to pin persistence
        }
        for k, atol in budgets.items():
            fitted = _flatten(fit.params[k])
            target = np.asarray(truth[k], dtype=float).reshape(-1)
            np.testing.assert_array_less(
                np.abs(fitted - target),
                atol + 1e-12,
                err_msg=(
                    f"recovery: {k} fitted={fitted} target={target} "
                    f"budget={atol}"
                ),
            )
        # Persistence (α + β) recovery is the operationally important
        # statistic for high-persistence GARCH — pin it tightly.
        persistence = (
            float(fit.params["alpha"][0]) + float(fit.params["beta"][0])
        )
        np.testing.assert_allclose(
            persistence, 0.99,
            atol=0.05,
            err_msg=f"persistence={persistence} far from truth 0.99",
        )
