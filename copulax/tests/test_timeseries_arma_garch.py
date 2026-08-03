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
from copulax.tests._timeseries_helpers import (
    BEHAVIOURAL,
    MAXITER_REFERENCE,
    N_STARTS_FULL,
    PRECISION,
    REFERENCE,
    STANDARD,
    assert_snapshot_intact,
    fit_key,
    series,
    shared_fit,
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


# ---------------------------------------------------------------------------
# Load the COMMON-SERIES model-selection reference (rugarch fits all four
# variants on ONE shared series; see generate_model_selection_reference.R).
# ---------------------------------------------------------------------------

_MODEL_SELECTION_REF_PATH = (
    Path(__file__).parent / "_r_reference" / "model_selection_reference_data.py"
)
_ms_spec = _ilu.spec_from_file_location(
    "_model_selection_reference", _MODEL_SELECTION_REF_PATH,
)
_ms_module = _ilu.module_from_spec(_ms_spec)
_ms_spec.loader.exec_module(_ms_module)
MODEL_SELECTION_REFERENCE = _ms_module.MODEL_SELECTION_REFERENCE
MODEL_SELECTION_Y = _ms_module.MODEL_SELECTION_Y


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

#: Frozen-series name for each hand-rolled matrix label.  The four
#: labels are the ones no third-party engine can represent (TGARCH's
#: sigma-form recursion, QGARCH's asymmetric psi term, and the copulax
#: ``gh`` / ``skewed_t`` residual parameterisations), so they were frozen
#: from a one-time committed port of the recursion this module used to
#: roll at collection time — see
#: ``_r_reference/generate_frozen_series_handrolled.py``.  The frozen
#: values are bit-identical to what the runtime path produced, so no
#: matrix assertion moved when they were adopted.
_HANDROLLED_FROZEN_NAME = {
    label: f"matrix_{label}_n2000" for label in _HANDROLLED_LABELS
}


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------

_MATRIX_LABELS = list(RUGARCH_REFERENCE.keys()) + list(_HANDROLLED_LABELS)
_RUGARCH_LABELS = tuple(RUGARCH_REFERENCE.keys())

#: The REFERENCE tier's iteration budget and learning rate.  Aliased
#: from the shared registry so the matrix machinery and the tier table
#: can never drift apart.
_FIT_MAXITER = MAXITER_REFERENCE
_FIT_LR = 0.05

#: Number of optimiser starts that pin the structural multi-start
#: guarantees.  ``fit`` defaults to a single start (``n_starts=1``, seeded
#: at the two-stage separable warm start under the default
#: ``init="separable"``); the properties that depend on the full HARD-04
#: candidate set — the joint init-mode invariance (J1
#: ``test_pairwise_convergence``), joint>=separable (B7
#: ``TestJointVsSeparable``), the rugarch/dominance references, and the
#: GH/QGARCH/TGARCH finite-argmax — are properties OF the multi-start path,
#: so every such fit explicitly opts in with the full candidate count.  The
#: value caps at the available candidates (4 joint / 3 standalone), so this
#: single constant covers both the joint fits (4 candidates: chosen init
#: seed + separable warm start + the remaining cold init modes) and the
#: standalone variance fits (3 init-mode candidates).
#: Alias of the registry constant, kept for readability at the call
#: sites that opt into the full multi-start candidate set.
_N_STARTS_FULL = N_STARTS_FULL


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
    y = series(_HANDROLLED_FROZEN_NAME[label])
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


def _matrix_series_name(label):
    """Registry series name for a matrix ``label``.

    Rugarch-reference labels carry their series in
    ``arma_garch_reference_data.py`` (a committed rugarch fixture with
    its own regenerator), not in the frozen-series corpus, so they are
    registered under a ``rugarch_reference_*`` name.  The four
    hand-rolled labels use their frozen-corpus name directly.
    """
    if label in RUGARCH_REFERENCE:
        return f"rugarch_reference_{label}"
    return _HANDROLLED_FROZEN_NAME[label]


def _matrix_tag(label):
    """Registry data tag for a matrix ``label``."""
    return "rugarch_reference" if label in RUGARCH_REFERENCE else "frozen"


def _matrix_model(case):
    """The unfitted joint model for a matrix ``case``."""
    return ArmaGarch(
        mean_order=case.mean_order,
        var_model=case.var_model,
        var_order=case.var_order,
        residual_dist=case.residual_dist,
    )


def _matrix_fit_key(label):
    """The shared-registry key of the REFERENCE fit for ``label``.

    Mirrors :func:`_fit_case` exactly — including ``y=case.y`` — so the
    key's data-digest component matches the one the fit registered
    under.
    """
    case = _cached_case(label)
    return fit_key(
        _matrix_model(case), _matrix_series_name(label), tier=REFERENCE,
        y=case.y, tag=_matrix_tag(label),
    )


def _fit_case(case):
    # REFERENCE tier: init="analytical", the full multi-start candidate
    # set and maxiter=1500.  The matrix-fit consumers (B7
    # joint>=separable, the rugarch/dominance references, the
    # GH/QGARCH/TGARCH finite-argmax, candidate-stats) all rely on the
    # structural multi-start guarantee, which is no longer the default,
    # and every one of these fits is cross-validated against rugarch —
    # so this tier's arguments are frozen.
    return shared_fit(
        _matrix_model(case), _matrix_series_name(case.label),
        tier=REFERENCE, y=case.y, tag=_matrix_tag(case.label),
    )


#: Module-scoped joint-fit cache keyed by ``(label, init_mode, n_starts,
#: maxiter)``.  ``TestInitModesConvergence`` refits the SAME (label, mode)
#: joint model repeatedly — the ``analytical`` seed alone is fit once per
#: mode-pair across the pairwise parametrisation AND again in
#: ``test_each_mode_matches_rugarch`` — and each joint fit is expensive
#: (n=2000, maxiter=2000, four multi-start candidates).  Caching by the
#: fit-determining key collapses those identical computations to one run
#: per key.  Fitted models are frozen equinox PyTrees, so returning the
#: shared instance is safe (the tests only read from it).
_INIT_MODE_FIT_CACHE: dict = {}


def _cached_init_mode_fit(label, mode, n_starts, maxiter):
    r"""Return the joint fit for ``(label, mode, n_starts, maxiter)``,
    computing it once per distinct key and caching module-wide.

    The key is the full set of inputs that determine the fit result
    (``lr`` is fixed at :data:`_FIT_LR` for every init-mode fit), so two
    callers with the same key share the identical fitted model instead of
    recomputing it.
    """
    key = (label, mode, int(n_starts), int(maxiter))
    cached = _INIT_MODE_FIT_CACHE.get(key)
    if cached is None:
        case = _build_case(label)
        cached = ArmaGarch(
            mean_order=case.mean_order, var_model=case.var_model,
            var_order=case.var_order, residual_dist=case.residual_dist,
        ).fit(
            case.y, init=mode, n_starts=int(n_starts),
            maxiter=int(maxiter), lr=_FIT_LR,
        )
        _INIT_MODE_FIT_CACHE[key] = cached
    return cached


#: Module-scoped cache for the matrix CASES, keyed by ``label``.
#: ``label`` is the complete key: ``_build_case`` is a pure function of
#: it (a frozen-corpus series for the hand-rolled labels, a committed
#: rugarch fixture otherwise).  The cache holds immutable values only —
#: a jnp series array and the case metadata — so every consumer reads
#: the identical instance.
#:
#: The matrix FITS live in the shared cross-module registry
#: (``_timeseries_helpers.shared_fit``) at the REFERENCE tier, keyed by
#: ``(tier, model signature, series name, tag, data digest, fit
#: arguments)``.  Before
#: caching, the 14 labels shared by ``matrix_fit`` and ``rugarch_fit``
#: were fitted twice and ``arma11_garch11_normal`` three times; the
#: registry collapses those to one fit per label, and the isolation
#: guard below pins identity, wrapper freshness and immutability.
_MATRIX_CASE_CACHE: dict = {}


def _cached_case(label):
    """Return the case for ``label``, building it once."""
    cached = _MATRIX_CASE_CACHE.get(label)
    if cached is None:
        cached = _build_case(label)
        _MATRIX_CASE_CACHE[label] = cached
    return cached


def _cached_matrix_fit(label):
    """Return the REFERENCE-tier joint fit for ``label``, computed once."""
    return _fit_case(_cached_case(label))


def _matrix_case_view(label):
    """A FRESH wrapper around the cached case and fit for ``label``.

    The cached namespace is never handed out directly: consumers assign
    ``case.fit``, so a shared mutable wrapper would leak attribute
    writes between fixtures.  Each call rebuilds the namespace around
    the same immutable values, giving distinct wrappers over one shared
    series and one shared fitted model.
    """
    view = SimpleNamespace(**vars(_cached_case(label)))
    view.fit = _cached_matrix_fit(label)
    view.fit_key = _matrix_fit_key(label)
    return view


@pytest.fixture(scope="module", params=_MATRIX_LABELS, ids=lambda x: x)
def matrix_fit(request):
    return _matrix_case_view(request.param)


@pytest.fixture(scope="module", params=_RUGARCH_LABELS, ids=lambda x: x)
def rugarch_fit(request):
    return _matrix_case_view(request.param)


@pytest.fixture(scope="module")
def base_fit():
    return _matrix_case_view("arma11_garch11_normal")


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


# ---------------------------------------------------------------------------
# D-08 Layer-2 gate: one-sided LL dominance (fit-vs-fit)
# ---------------------------------------------------------------------------
#
# The retired ``_se_budget_assert`` scaled a same-data solver comparison by
# the asymptotic standard error.  SE is *sampling* error (spread across
# hypothetical re-draws), irrelevant to a deterministic two-solver
# comparison on ONE fixed series; GARCH SEs on n=2000 are wide enough that a
# genuinely wrong optimum (a J1-class 0.75% LL gap) fits inside a k*SE band
# (lessons.md, 2026-07-24).  The Layer-2 gate is instead one-sided LL
# dominance: our fit must be at least as good as the reference's params
# evaluated under OUR likelihood.  Layer-1 (test_timeseries_variance.py)
# proves our likelihood matches rugarch's at fixed params, so beating the
# reference's own params is a legitimate success criterion.

#: One-sided dominance slack (LL units).  Our fit must satisfy
#: ``ll_ours >= ll_ref - _LL_DOMINANCE_EPS``.  Every non-flat-ridge
#: reference case measured strictly POSITIVE margin (we meet or beat the
#: reference), so this small slack only absorbs solver-noise dips; it is a
#: convergence tolerance, never a statistical (SE-scaled) band.
_LL_DOMINANCE_EPS = 1e-1

#: ARMA(p+q>=3) likelihoods have a flat phi-theta ridge admitting multiple
#: near-equivalent optima; copulax and rugarch legitimately land on
#: DIFFERENT points of the ridge, so one-sided dominance is replaced by a
#: measured DELTA-LL-equivalence bound (both param vectors give the same
#: likelihood within this many LL units).  Measured max |margin| among the
#: high-order cases is ~0.92 on n=2000 (~0.03%/obs); 1.5 gives headroom.
#: This is the ONLY sanctioned non-cap flat-ridge justification (D-08) — a
#: widened param cap is never used.
_FLAT_RIDGE_DELTA_LL = 1.5

#: Frozen same-optimum parameter cap (absolute).  Asserted ONLY when the
#: dominance margin is ~0 (both solvers converged to the same optimum) and
#: only for single-lag variance models.  Measured max clean same-optimum
#: diff across the reference matrix is ~6.1e-3 (ma1 beta); 1e-2 gives ~1.6x
#: headroom.  Slack source: finite-sample MLE agreement between copulax's
#: Adam projected-gradient and rugarch's L-BFGS-B on the SAME n=2000 series
#: (both valid MLEs with DELTA-LL ~0) — a convergence/solver cap, not an SE
#: band.  Multi-lag variance (p_var>1 or q_var>1) has a flat lag-split
#: direction (e.g. GARCH(1,2) splits beta across two lags at equal
#: likelihood); those params are recorded, not capped — the ~0 margin is
#: their DELTA-LL-equivalence justification.
_PARAM_MATCH_CAP = 1e-2

#: Margin below which the two fits are treated as the SAME optimum, so the
#: parameter caps are asserted.  Above it we materially dominate (the
#: reference under-converged) and param equality is recorded, not asserted.
_SAME_OPTIMUM_MARGIN = 1e-1


def _ll_at_ref_params(case) -> float:
    r"""Reference params evaluated under OUR likelihood (D-08 RHS).

    Warm-starts an ``ArmaGarch`` at the reference parameter dict with
    ``maxiter=0`` (no optimisation) and reads back the fit-time raw
    log-likelihood — i.e. our recursion + likelihood evaluated exactly at
    the reference's converged params.  This is the same mechanism the
    joint-vs-separable test uses to evaluate a fixed parameter point.
    """
    ref_eval = ArmaGarch(
        mean_order=case.mean_order, var_model=case.var_model,
        var_order=case.var_order, residual_dist=case.residual_dist,
    ).fit(case.y, init="warm", init_params=case.rugarch["params"], maxiter=0)
    return float(ref_eval.loglikelihood())


#: ARMA(p+q>=3) cases admit multiple near-equivalent optima
#: (Wold-representation roots cancel with MA roots in different
#: arrangements at the same likelihood). copulax and rugarch converge to
#: different but valid optima, so the D-08 gate uses a
#: DELTA-LL-equivalence bound (not one-sided dominance) for these.
_HIGH_ORDER_ARMA = frozenset({
    "arma21_garch11_normal", "arma12_garch11_normal",
    "arma22_garch11_normal",
})


def _assert_ll_dominance(fit, case, label=""):
    r"""D-08 Layer-2 gate: one-sided LL dominance (or flat-ridge
    DELTA-LL-equivalence), with same-optimum parameter caps.

    * Flat-ridge cases (ARMA p+q>=3): assert DELTA-LL-equivalence
      ``|ll_ours - ll_ref| <= _FLAT_RIDGE_DELTA_LL`` (multiple equivalent
      optima; no dominance direction, no param caps).
    * Otherwise: assert one-sided dominance
      ``ll_ours >= ll_ref - _LL_DOMINANCE_EPS``.  When the margin is ~0
      (both solvers at the same optimum) additionally assert the frozen
      single-lag parameter caps; when we materially dominate (reference
      under-converged) param equality is RECORDED in the message, not
      asserted.
    """
    ll_ours = float(fit.loglikelihood())
    ll_ref = _ll_at_ref_params(case)
    margin = ll_ours - ll_ref

    if label in _HIGH_ORDER_ARMA:
        assert abs(margin) <= _FLAT_RIDGE_DELTA_LL, (
            f"{label}: flat-ridge DELTA-LL-equivalence violated: "
            f"ll_ours={ll_ours} ll_ref={ll_ref} |margin|={abs(margin)} "
            f"> {_FLAT_RIDGE_DELTA_LL}"
        )
        return

    assert margin >= -_LL_DOMINANCE_EPS, (
        f"{label}: one-sided LL dominance violated: ll_ours={ll_ours} "
        f"< ll_ref={ll_ref} - {_LL_DOMINANCE_EPS} (margin={margin})"
    )

    ref = case.rugarch["params"]
    multi_lag = case.var_order[0] > 1 or case.var_order[1] > 1
    if margin <= _SAME_OPTIMUM_MARGIN and not multi_lag:
        # Same optimum, single-lag variance: assert the frozen param caps.
        for k in ("phi", "theta", "mu", "omega", "alpha", "beta", "gamma"):
            if k not in ref:
                continue
            fitted = _flatten(fit.params[k])
            target = _flatten(ref[k])
            if target.size == 0 or fitted.size != target.size:
                continue
            diff = np.abs(fitted - target)
            np.testing.assert_array_less(
                diff, _PARAM_MATCH_CAP,
                err_msg=(
                    f"{label} key={k!r} same-optimum param cap exceeded: "
                    f"fitted={fitted} target={target} diff={diff} "
                    f"cap={_PARAM_MATCH_CAP} (margin={margin})"
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

    #: Joint-vs-separable slack (LL units).  The joint fit opts into the
    #: full multi-start set (``_fit_case`` -> ``n_starts=4``), so the
    #: two-stage separable warm start is candidate 1 and the best-iterate
    #: solver keeps the point at least as good as it: ``joint_ll >=
    #: sep_ll`` is structural.  The production separable warm start runs the
    #: SAME eager sub-fit computations as ``_separable_warm_eval`` (same
    #: data / init / maxiter / lr / compiled executables) -> a bit-identical
    #: separable point, so the only residual noise converting the vmapped
    #: mean-objective comparison to the unbatched reported LL sum is x64
    #: reassociation noise (~1e-9 absolute LL units at n=2000).  1e-6 sits
    #: three orders above that floor and three below the retired 1e-3 slack
    #: (01-REBASELINE.md section 6).  It is a convergence/reassociation
    #: tolerance, not a statistical band.
    _JOINT_SEP_SLACK = 1e-6

    def test_joint_at_least_as_high_as_separable(self, matrix_fit):
        sep_eval = self._separable_warm_eval(matrix_fit)
        joint_ll = float(matrix_fit.fit.loglikelihood())
        sep_ll = float(sep_eval.loglikelihood())
        assert joint_ll >= sep_ll - self._JOINT_SEP_SLACK, (
            f"{matrix_fit.label}: joint_ll={joint_ll} < sep_ll={sep_ll} "
            f"(slack={self._JOINT_SEP_SLACK})"
        )


# ---------------------------------------------------------------------------
# Multi-start candidate stats (HARD-04)
# ---------------------------------------------------------------------------

class TestMultiStartCandidateStats:
    """The HARD-04 multi-start fit populates the D-09 candidate-stats
    leaves (``n_finite_candidates`` / ``best_candidate``) with the real
    per-fit aggregates, not the single-start placeholders Plan 08 left."""

    def test_joint_candidate_stats_are_multi_start(self, matrix_fit):
        # The joint candidate set is the three cold-start init modes UNION
        # the two-stage separable warm start -> four candidates.
        fit = matrix_fit.fit
        n_finite = int(fit.n_finite_candidates)
        best = int(fit.best_candidate)
        assert n_finite >= 2, (
            f"{matrix_fit.label}: n_finite_candidates={n_finite} is not a "
            "multi-start aggregate (placeholder would be <=1)"
        )
        assert n_finite <= 4
        # The winning candidate index must fall within the candidate set.
        assert 0 <= best < 4, f"{matrix_fit.label}: best_candidate={best}"

    def test_standalone_variance_candidate_stats_are_multi_start(self):
        # A standalone GARCH multi-start fit assembles the three init-mode
        # candidates (n_starts caps at the 3 available); a healthy fit
        # leaves all three finite and the winner in range.
        case = _build_case("arma11_garch11_normal")
        eps = ARMA(
            p=1, q=1, residual_dist=normal,
        ).fit(case.y, init="analytical", maxiter=_FIT_MAXITER,
              lr=_FIT_LR).residuals(case.y)["residuals"]
        vf = GARCH(p=1, q=1, residual_dist=normal).fit(
            eps, init="analytical", n_starts=_N_STARTS_FULL,
            maxiter=_FIT_MAXITER, lr=_FIT_LR,
        )
        assert int(vf.n_finite_candidates) == 3
        assert 0 <= int(vf.best_candidate) < 3


# ---------------------------------------------------------------------------
# Default single-start semantics (post-rework)
# ---------------------------------------------------------------------------

class TestSingleStartDefault:
    """``fit`` defaults to a single optimiser start (``n_starts=1``): only
    the chosen init seed is used.  These tests pin the explicit cold-init
    escape hatch — with ``init="analytical"`` the two-stage separable warm
    start is NOT run (the default ``init="separable"`` seed is covered by
    ``TestSeparableDefaultInit``).  The candidate-stats leaves report the
    single start truthfully (``n_finite_candidates`` in {0, 1},
    ``best_candidate == 0``) under both eager and jitted evaluation, and an
    explicit ``n_starts > 1`` restores the multi-start aggregates."""

    def _y(self):
        key = jax.random.PRNGKey(4)
        return jax.random.normal(key, (700,)) * 0.6 + 0.05

    def test_joint_default_is_single_start(self):
        y = self._y()
        fit = ArmaGarch(
            mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, init="analytical", maxiter=300, lr=_FIT_LR)
        assert int(fit.n_finite_candidates) == 1, (
            f"default joint fit must be single-start; got "
            f"n_finite_candidates={int(fit.n_finite_candidates)}"
        )
        assert int(fit.best_candidate) == 0

    def test_standalone_default_is_single_start(self):
        y = self._y()
        eps = ARMA(p=1, q=1, residual_dist=normal).fit(
            y, init="analytical", maxiter=300, lr=_FIT_LR,
        ).residuals(y)["residuals"]
        vf = GARCH(p=1, q=1, residual_dist=normal).fit(
            eps, init="analytical", maxiter=300, lr=_FIT_LR,
        )
        assert int(vf.n_finite_candidates) == 1
        assert int(vf.best_candidate) == 0

    def test_joint_default_single_start_under_jit(self):
        y = self._y()

        def fit_fn(yy):
            return ArmaGarch(
                mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
                residual_dist=normal,
            ).fit(yy, init="analytical", maxiter=200, lr=_FIT_LR)

        eager = fit_fn(y)
        jitted = jax.jit(fit_fn)(y)
        # Status leaves populate under jit and match the eager fit.
        assert int(jitted.n_finite_candidates) == 1
        assert int(jitted.best_candidate) == 0
        np.testing.assert_allclose(
            float(jitted.loglikelihood()), float(eager.loglikelihood()),
            rtol=1e-5,
        )

    def test_joint_n_starts_gt_one_populates_multi_start(self):
        y = self._y()
        fit = ArmaGarch(
            mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, init="analytical", n_starts=_N_STARTS_FULL,
              maxiter=300, lr=_FIT_LR)
        # Full joint candidate set is four (chosen seed + separable warm
        # start + two other init modes); a healthy fit leaves >=2 finite.
        assert int(fit.n_finite_candidates) >= 2
        assert 0 <= int(fit.best_candidate) < 4

    def test_default_and_full_multistart_are_at_least_as_good(self):
        # The multi-start fit explores a superset of the default single
        # start (its chosen-seed candidate is candidate 0), so its returned
        # log-likelihood is at least the single-start fit's (best-iterate +
        # finite-LL argmax).  This is the structural direction the opt-in
        # buys; it must never be WORSE than the default.
        y = self._y()
        single = ArmaGarch(
            mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, init="analytical", maxiter=400, lr=_FIT_LR)
        multi = ArmaGarch(
            mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y, init="analytical", n_starts=_N_STARTS_FULL,
              maxiter=400, lr=_FIT_LR)
        ll_single = float(single.loglikelihood())
        ll_multi = float(multi.loglikelihood())
        assert ll_multi >= ll_single - 1e-6, (
            f"multi-start LL {ll_multi} < single-start LL {ll_single}"
        )

    def test_n_starts_validation(self):
        y = self._y()
        with pytest.raises(ValueError):
            ArmaGarch(
                mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
                residual_dist=normal,
            ).fit(y, init="analytical", n_starts=0, maxiter=10)
        with pytest.raises(TypeError):
            ArmaGarch(
                mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
                residual_dist=normal,
            ).fit(y, init="analytical", n_starts=True, maxiter=10)


# ---------------------------------------------------------------------------
# Separable default init (fit seeds at the two-stage warm start)
# ---------------------------------------------------------------------------

class TestSeparableDefaultInit:
    """``fit`` defaults to ``init="separable"``: the joint MLE is seeded at
    the two-stage separable warm start, which is also the highest-priority
    multi-start candidate — so ``joint_ll >= separable_ll`` is structural
    for every default fit, and ``n_starts`` truncation always keeps the
    default seed as candidate 0 (larger ``n_starts`` only appends the
    cold-start modes)."""

    _MAXITER = 300

    def _y(self):
        key = jax.random.PRNGKey(11)
        return jax.random.normal(key, (700,)) * 0.6 + 0.05

    def _model(self):
        return ArmaGarch(
            mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        )

    def _composed_separable_params(self, y, maxiter):
        arma_fit = ARMA(p=1, q=1, residual_dist=normal).fit(
            y, init="analytical", maxiter=maxiter, lr=_FIT_LR,
        )
        eps = arma_fit.residuals(y)["residuals"]
        var_fit = GARCH(p=1, q=1, residual_dist=normal).fit(
            eps, init="analytical", maxiter=maxiter, lr=_FIT_LR,
        )
        return {
            "phi": arma_fit.params["phi"],
            "theta": arma_fit.params["theta"],
            "mu": arma_fit.params["mu"],
            **{k: var_fit.params[k] for k in var_fit._ag_var_keys()},
            "residual": dict(var_fit.params["residual"]),
        }

    def test_default_init_is_separable(self):
        # The bare default and the explicit mode run the identical path.
        y = self._y()
        default = self._model().fit(y, maxiter=self._MAXITER, lr=_FIT_LR)
        explicit = self._model().fit(
            y, init="separable", maxiter=self._MAXITER, lr=_FIT_LR,
        )
        np.testing.assert_allclose(
            float(default.loglikelihood()), float(explicit.loglikelihood()),
            rtol=0.0, atol=0.0,
        )
        for k in ("phi", "theta", "mu", "omega", "alpha", "beta"):
            np.testing.assert_allclose(
                np.asarray(default.params[k]),
                np.asarray(explicit.params[k]),
                rtol=0.0, atol=0.0,
            )

    def test_default_matches_composed_two_stage_warm_fit(self):
        # The default fit equals an explicitly composed two-stage warm
        # fit: identical sub-fits produce an identical separable point,
        # and the joint optimisation from that point is the same
        # computation on both paths.
        y = self._y()
        default = self._model().fit(y, maxiter=self._MAXITER, lr=_FIT_LR)
        sep = self._composed_separable_params(y, self._MAXITER)
        warm = self._model().fit(
            y, init="warm", init_params=sep,
            maxiter=self._MAXITER, lr=_FIT_LR,
        )
        np.testing.assert_allclose(
            float(default.loglikelihood()), float(warm.loglikelihood()),
            rtol=0.0, atol=0.0,
        )

    def test_default_dominates_separable_two_stage(self):
        # The structural guarantee at default settings: the joint fit
        # starts at the separable point and keeps its best iterate, so it
        # never ends below the two-stage log-likelihood.
        y = self._y()
        default = self._model().fit(y, maxiter=self._MAXITER, lr=_FIT_LR)
        sep = self._composed_separable_params(y, self._MAXITER)
        sep_eval = self._model().fit(
            y, init="warm", init_params=sep, maxiter=0,
        )
        assert (
            float(default.loglikelihood())
            >= float(sep_eval.loglikelihood()) - 1e-6
        )

    def test_default_single_start_stats(self):
        # n_starts=1 stays the default: one candidate, truthful stats.
        y = self._y()
        fit = self._model().fit(y, maxiter=self._MAXITER, lr=_FIT_LR)
        assert int(fit.n_finite_candidates) == 1
        assert int(fit.best_candidate) == 0

    def test_separable_multistart_is_monotone_in_n_starts(self):
        # Under the default init the candidate list is [separable,
        # analytical, backcast, sample][:n_starts]; the finite-likelihood
        # argmax over a superset is monotone (1e-6 covers vmap-width
        # reassociation noise only).
        y = self._y()
        lls = {}
        for k in (1, 2, 4):
            fit = self._model().fit(
                y, n_starts=k, maxiter=self._MAXITER, lr=_FIT_LR,
            )
            assert 1 <= int(fit.n_finite_candidates) <= k
            assert 0 <= int(fit.best_candidate) < k
            lls[k] = float(fit.loglikelihood())
        assert lls[2] >= lls[1] - 1e-6
        assert lls[4] >= lls[2] - 1e-6


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

    def test_unknown_method_raises(self, base_fit):
        with pytest.raises(ValueError):
            base_fit.fit.forecast(h=5, method="bogus")

    @pytest.mark.parametrize("var_model", _NO_ANALYTICAL_VARIANTS)
    def test_h2_analytical_raises_for_no_analytical_variants(self, var_model):
        label = (
            "arma11_egarch11_normal" if var_model is EGARCH
            else "arma11_tgarch11_normal"
        )
        fit = _cached_matrix_fit(label)
        fc1 = fit.forecast(h=1, method="analytical")
        assert fc1["variance"].shape == (1,)
        with pytest.raises(ValueError, match=var_model.__name__):
            fit.forecast(h=10, method="analytical")


# ---------------------------------------------------------------------------
# forecast(u=...) parity with rvs(u=...) — HARD-09
# ---------------------------------------------------------------------------

def _fit_standalone_garch(seed: int = 0, n: int = 500):
    """Fit a standalone GARCH(1,1)-Normal variance model on iid
    mean-zero innovations, returning a fitted model with a terminal
    state (so ``forecast`` / ``rvs`` are well-defined).

    STANDARD tier: the forecast/rvs parity assertions compare two code
    paths through the SAME fitted model, so any well-behaved interior
    fit serves.

    Note:
        If you intend to jit wrap this function, ensure that ``n`` is a
        static argument.
    """
    eps = jax.random.normal(jax.random.PRNGKey(seed), (n,)) * 0.5
    return shared_fit(
        GARCH(p=1, q=1, residual_dist=normal), f"iid_normal_n{n}_s{seed}",
        tier=STANDARD, y=eps, tag="scaled_0.5",
    )


class TestForecastU:
    """``forecast(u=...)`` must forward pre-drawn uniforms through the
    identical ppf path as ``rvs(u=...)`` — full parity — on both the
    joint ``ArmaGarch`` model and the variance-only base.  Previously
    ``forecast`` had no ``u`` parameter (unlike ``rvs``), so
    copula-drawn uniforms could not be routed through it.
    """

    # --- Joint ArmaGarch ---

    def test_forecast_u_2d_matches_rvs_2d(self, base_fit):
        """2D ``u`` (n_paths, h): forecast paths and moments equal the
        same ``u`` fed through ``rvs(u=, last_state=terminal_state)``."""
        fit = base_fit.fit
        n_paths, h = 16, 10
        u = jnp.asarray(
            np.random.default_rng(0).uniform(0.01, 0.99, size=(n_paths, h))
        )
        fc = fit.forecast(h=h, method="simulation", u=u)
        ref_paths = fit.rvs(u=u, last_state=fit.terminal_state)
        assert fc["paths"].shape == (n_paths, h)
        np.testing.assert_allclose(
            np.asarray(fc["paths"]), np.asarray(ref_paths), rtol=1e-6, atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(fc["mean"]),
            np.asarray(jnp.mean(ref_paths, axis=0)),
            rtol=1e-6, atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(fc["variance"]),
            np.asarray(jnp.var(ref_paths, axis=0)),
            rtol=1e-6, atol=1e-6,
        )

    def test_forecast_u_1d_matches_rvs_1d(self, base_fit):
        """1D ``u`` (h,): a single deterministic path forwarded through
        the same ppf path as ``rvs(u=)``."""
        fit = base_fit.fit
        h = 12
        u = jnp.linspace(0.02, 0.98, h)
        fc = fit.forecast(h=h, method="simulation", u=u)
        ref_path = fit.rvs(u=u, last_state=fit.terminal_state)
        np.testing.assert_allclose(
            np.asarray(fc["paths"]), np.asarray(ref_path), rtol=1e-6, atol=1e-6,
        )

    def test_forecast_u_deterministic(self, base_fit):
        """Two ``forecast(u=U)`` calls with the same ``U`` are identical
        (no internal randomness when ``u`` is supplied)."""
        fit = base_fit.fit
        u = jnp.asarray(
            np.random.default_rng(1).uniform(0.01, 0.99, size=(8, 6))
        )
        a = fit.forecast(h=6, method="simulation", u=u)
        b = fit.forecast(h=6, method="simulation", u=u)
        np.testing.assert_allclose(np.asarray(a["paths"]), np.asarray(b["paths"]))

    def test_forecast_no_u_no_paths_still_raises(self, base_fit):
        """``method='simulation'`` with neither ``u`` nor ``n_paths`` still
        raises the existing informative ``ValueError`` (no silent change)."""
        with pytest.raises(ValueError):
            base_fit.fit.forecast(h=5, method="simulation")

    def test_forecast_u_matches_internal_sampling_shapes(self, base_fit):
        """forecast(u=U) output dict has the same keys/shapes as the
        internally-sampled simulation forecast."""
        fit = base_fit.fit
        n_paths, h = 20, 7
        u = jnp.asarray(
            np.random.default_rng(2).uniform(0.01, 0.99, size=(n_paths, h))
        )
        fc_u = fit.forecast(h=h, method="simulation", u=u)
        fc_internal = fit.forecast(
            h=h, method="simulation", n_paths=n_paths, key=jax.random.PRNGKey(3),
        )
        assert set(fc_u.keys()) == set(fc_internal.keys())
        assert fc_u["mean"].shape == fc_internal["mean"].shape == (h,)
        assert fc_u["variance"].shape == fc_internal["variance"].shape == (h,)
        assert fc_u["paths"].shape == fc_internal["paths"].shape == (n_paths, h)

    # --- Variance-only base ---

    def test_variance_base_forecast_u_2d_matches_rvs(self):
        """Variance-only base: forecast(u=U) 2D parity with rvs(u=U)."""
        vf = _fit_standalone_garch(seed=0)
        n_paths, h = 16, 10
        u = jnp.asarray(
            np.random.default_rng(4).uniform(0.01, 0.99, size=(n_paths, h))
        )
        fc = vf.forecast(h=h, method="simulation", u=u)
        ref_paths = vf.rvs(u=u, last_state=vf.terminal_state)
        assert fc["paths"].shape == (n_paths, h)
        np.testing.assert_allclose(
            np.asarray(fc["paths"]), np.asarray(ref_paths), rtol=1e-6, atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(fc["mean"]),
            np.asarray(jnp.mean(ref_paths, axis=0)),
            rtol=1e-6, atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(fc["variance"]),
            np.asarray(jnp.var(ref_paths, axis=0)),
            rtol=1e-6, atol=1e-6,
        )

    def test_variance_base_forecast_u_1d_matches_rvs(self):
        """Variance-only base: forecast(u=U) 1D parity with rvs(u=U)."""
        vf = _fit_standalone_garch(seed=1)
        h = 12
        u = jnp.linspace(0.02, 0.98, h)
        fc = vf.forecast(h=h, method="simulation", u=u)
        ref_path = vf.rvs(u=u, last_state=vf.terminal_state)
        np.testing.assert_allclose(
            np.asarray(fc["paths"]), np.asarray(ref_path), rtol=1e-6, atol=1e-6,
        )

    def test_variance_base_no_u_no_paths_still_raises(self):
        """Variance-only base: simulation with neither u nor n_paths still
        raises the existing informative ValueError."""
        vf = _fit_standalone_garch(seed=2)
        with pytest.raises(ValueError):
            vf.forecast(h=5, method="simulation")

    def test_variance_base_analytical_unchanged(self):
        """Variance-only base: analytical forecast (no u) unchanged."""
        vf = _fit_standalone_garch(seed=3)
        fc = vf.forecast(h=10, method="analytical")
        assert fc["paths"] is None
        assert fc["variance"].shape == (10,)
        assert np.all(np.isfinite(np.asarray(fc["variance"])))


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
        fit = _cached_matrix_fit("arma11_igarch11_normal")
        persistence = (
            float(fit.params["alpha"][0]) + float(fit.params["beta"][0])
        )
        np.testing.assert_allclose(persistence, 1.0, atol=1e-6)

    def test_qgarch_positivity_invariant(self):
        fit = _cached_matrix_fit("arma11_qgarch11_normal")
        omega = float(fit.params["omega"])
        alpha = float(np.asarray(fit.params["alpha"]).reshape(-1)[0])
        psi = float(np.asarray(fit.params["psi"]).reshape(-1)[0])
        assert omega + 1e-9 >= psi * psi / (4.0 * alpha)

    def test_gjr_persistence_below_one(self):
        fit = _cached_matrix_fit("arma11_gjr11_normal")
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
# Fitted residual distribution (promotion contract)
# ---------------------------------------------------------------------------
class TestFittedResidualDist:
    def test_fit_promotes_residual_dist(self, matrix_fit):
        """``fit.residual_dist`` is the fitted standardised instance —
        ``.cdf`` works directly, enabling the PIT step ``u = F(z)``."""
        rd = matrix_fit.fit.residual_dist
        assert rd._stored_params is not None
        assert rd.name.endswith("-stdresid")
        u = np.asarray(rd.cdf(jnp.array([-1.0, 0.0, 1.0])))
        assert np.all(np.isfinite(u))
        assert np.all((u > 0.0) & (u < 1.0))


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
        # Opt into the full multi-start candidate set: init-mode invariance
        # (every mode returns the same argmax over the shared candidate set)
        # is a property of the multi-start path, not the single-start
        # default.  With the full set each mode ranks its own seed first but
        # explores the identical candidate union, so the fits agree.
        #
        # Routed through the module-scoped cache: the pairwise
        # parametrisation and test_each_mode_matches_rugarch request the
        # same (label, mode) fits repeatedly, so caching runs each distinct
        # computation exactly once.
        return _cached_init_mode_fit(label, mode, _N_STARTS_FULL, maxiter)

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
        # Every cold-start init mode assembles the same candidate set and
        # returns the same argmax, so each mode meets the D-08 one-sided
        # LL-dominance gate against rugarch's reference params.
        case = _build_case(label)
        fit = self._fit_with_init(label, mode)
        _assert_ll_dominance(fit, case, label=f"{label} mode={mode}")

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
        """D-08 Layer-2 gate: copulax's fit is at least as good as
        rugarch's params under our likelihood (one-sided LL dominance,
        DELTA-LL-equivalence on flat ARMA ridges), with the frozen
        same-optimum parameter caps asserted where both solvers converged
        to the same optimum."""
        _assert_ll_dominance(
            rugarch_fit.fit, rugarch_fit, label=rugarch_fit.label,
        )

    def test_loglikelihood_dominates_rugarch(self, rugarch_fit):
        """copulax and rugarch use different solvers (Adam projected
        gradient vs L-BFGS-B with restarts).  The D-08 gate is one-sided:
        our fit must be at least as good as the reference's params under
        OUR likelihood (beating a reference that under-converged is
        success, not failure) — never a two-sided equality against the
        reference's own reported LL.  Flat ARMA ridges use a measured
        DELTA-LL-equivalence bound instead of a dominance direction."""
        ll_ours = float(rugarch_fit.fit.loglikelihood())
        ll_ref = _ll_at_ref_params(rugarch_fit)
        margin = ll_ours - ll_ref
        if rugarch_fit.label in _HIGH_ORDER_ARMA:
            assert abs(margin) <= _FLAT_RIDGE_DELTA_LL, (
                f"{rugarch_fit.label}: flat-ridge DELTA-LL-equivalence "
                f"violated: ll_ours={ll_ours} ll_ref={ll_ref} "
                f"|margin|={abs(margin)}"
            )
        else:
            assert margin >= -_LL_DOMINANCE_EPS, (
                f"{rugarch_fit.label}: LL dominance violated: "
                f"ll_ours={ll_ours} < ll_ref={ll_ref} - {_LL_DOMINANCE_EPS}"
            )

    def test_aic_bic_match_rugarch(self, rugarch_fit):
        r"""AIC / BIC via (a) an exact internal identity and (b) one-sided
        dominance vs rugarch's params under OUR likelihood.

        The retired form asserted two-sided ``assert_allclose`` against
        rugarch's REPORTED AIC/BIC — the same two-sided-vs-a-fitted-
        reference shape D-08 retired for the log-likelihood (a reference
        that under-converged fails for the wrong reason).  This replaces it
        with the D-08 pattern already used for the LL:

        (a) Exact identity: ``AIC == 2k - 2*LL`` and ``BIC ==
            k*ln(n) - 2*LL`` against OUR reported log-likelihood and
            ``n_params`` — this exercises the CR-01 free-parameter count
            (an over/under-count in ``k`` breaks the identity), independent
            of any reference.
        (b) One-sided dominance downstream of the LL dominance gate: with
            ``ll_ours >= ll_ref - eps`` and identical ``k`` / ``n``,
            ``AIC = 2k - 2*LL`` gives ``AIC_ours <= AIC_ref(under ours)
            + 2*eps`` (BIC likewise; the ``k*ln(n)`` term cancels).  Flat
            ARMA ridges are DELTA-LL-equivalent, so their AIC/BIC use the
            symmetric ``2*_FLAT_RIDGE_DELTA_LL`` bound.
        """
        fit = rugarch_fit.fit
        k = int(fit.n_params)
        n = int(np.asarray(rugarch_fit.y).shape[0])
        ll_ours = float(fit.loglikelihood())
        aic_ours = float(fit.aic())
        bic_ours = float(fit.bic())

        # (a) Exact internal identity against our own LL and k (CR-01
        # count).  The tolerance only absorbs float last-bit recombination
        # between the stored scalar and this re-derivation (values ~5e3, so
        # rtol=1e-9 ~ atol 5e-6); a CR-01 miscount shifts AIC by >=2 (BIC by
        # >=ln(n)), five orders above the tolerance, so it is still caught.
        np.testing.assert_allclose(
            aic_ours, 2.0 * k - 2.0 * ll_ours, rtol=1e-9, atol=1e-6,
            err_msg=f"{rugarch_fit.label}: AIC != 2k - 2LL (k={k})",
        )
        np.testing.assert_allclose(
            bic_ours, k * np.log(n) - 2.0 * ll_ours, rtol=1e-9, atol=1e-6,
            err_msg=f"{rugarch_fit.label}: BIC != k*ln(n) - 2LL (k={k})",
        )

        # (b) One-sided dominance vs rugarch's params under OUR likelihood.
        # AIC_ref/BIC_ref are computed from ll_ref (reference params under
        # our recursion + likelihood) with the SAME k and n, so they are the
        # information criteria rugarch's optimum would score under our fit.
        ll_ref = _ll_at_ref_params(rugarch_fit)
        aic_ref = 2.0 * k - 2.0 * ll_ref
        bic_ref = k * np.log(n) - 2.0 * ll_ref
        if rugarch_fit.label in _HIGH_ORDER_ARMA:
            # Flat ridge: DELTA-LL-equivalent optima -> symmetric AIC/BIC
            # bound (2x the LL-equivalence bound; k*ln(n) cancels).
            assert abs(aic_ours - aic_ref) <= 2.0 * _FLAT_RIDGE_DELTA_LL, (
                f"{rugarch_fit.label}: AIC flat-ridge equivalence violated: "
                f"aic_ours={aic_ours} aic_ref={aic_ref}"
            )
            assert abs(bic_ours - bic_ref) <= 2.0 * _FLAT_RIDGE_DELTA_LL, (
                f"{rugarch_fit.label}: BIC flat-ridge equivalence violated: "
                f"bic_ours={bic_ours} bic_ref={bic_ref}"
            )
        else:
            # Lower IC is better; dominance in LL => AIC_ours <= AIC_ref +
            # 2*eps (and the same for BIC).
            assert aic_ours <= aic_ref + 2.0 * _LL_DOMINANCE_EPS, (
                f"{rugarch_fit.label}: AIC dominance violated: "
                f"aic_ours={aic_ours} > aic_ref={aic_ref} "
                f"+ {2.0 * _LL_DOMINANCE_EPS}"
            )
            assert bic_ours <= bic_ref + 2.0 * _LL_DOMINANCE_EPS, (
                f"{rugarch_fit.label}: BIC dominance violated: "
                f"bic_ours={bic_ours} > bic_ref={bic_ref} "
                f"+ {2.0 * _LL_DOMINANCE_EPS}"
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
        if rugarch_fit.label in _HIGH_ORDER_ARMA:
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
        if rugarch_fit.label in _HIGH_ORDER_ARMA:
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
      ~5-15% scale.  Skipped here for the same reason the D-08 gate
      switches to DELTA-LL-equivalence on those labels (see
      :data:`_HIGH_ORDER_ARMA`).
    * IGARCH's constrained simplex (α + β = 1) places the MLE at
      a slightly different point than rugarch's solver does — the
      Q-divergence on residuals is ~5%, and on squared residuals
      ~10%.  Within the ``rtol=0.10`` cross-library budget.
    """

    def test_ljung_box_matches_rugarch(self, rugarch_fit):
        if rugarch_fit.label in _HIGH_ORDER_ARMA:
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
        if rugarch_fit.label in _HIGH_ORDER_ARMA:
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

# Common-series ranking labels (rugarch fits all four on ONE shared
# series; see model_selection_reference_data.py / the .R regenerator).
_MODEL_RANK_LABELS = ("garch", "igarch", "gjr", "egarch")


#: Model-selection fits keyed by rank label.  The AIC and BIC ranking
#: tests fit the SAME four variants on the SAME ``MODEL_SELECTION_Y``
#: with identical settings and differ only in which information
#: criterion they read off the resulting fitted models, so the second
#: test's four fits were pure repetition.  Same idiom, same safety
#: argument as ``_INIT_MODE_FIT_CACHE``: frozen equinox PyTrees, read
#: only.
def _cached_model_selection_fit(label):
    """Return the REFERENCE-tier shared-series fit for a ranking ``label``.

    REFERENCE, like the matrix fits: this is an oracle comparison (the
    AIC/BIC ranking is checked against rugarch's ranking on the same
    series), and a like-for-like ranking against rugarch's converged
    fits needs each variant at its best optimum, not a single-start
    basin — so the tier's ``init="analytical"``, full multi-start
    candidate set and ``maxiter=1500`` are exactly what is required, and
    are frozen.
    """
    ref = MODEL_SELECTION_REFERENCE[label]
    cls = _VAR_MODEL_FROM_NAME[ref["var_model"]]
    return shared_fit(
        ArmaGarch(
            mean_order=ref["mean_order"], var_model=cls,
            var_order=ref["var_order"], residual_dist=normal,
        ),
        "model_selection_reference_y", tier=REFERENCE,
        y=jnp.asarray(MODEL_SELECTION_Y), tag="reference",
    )


def _fit_common_series_ic(ic_getter):
    """Fit copulax's four variants on the SHARED model-selection series
    and return {label: IC value} using ``ic_getter(fit)``.

    Both variants are fitted on ``MODEL_SELECTION_Y`` — the same series
    rugarch fit all four variants on in
    ``model_selection_reference_data.py`` — so the resulting ranking is a
    genuine common-series ranking directly comparable to rugarch's.
    """
    cx_ic = {}
    rg_ic = {}
    for label in _MODEL_RANK_LABELS:
        ref = MODEL_SELECTION_REFERENCE[label]
        fit = _cached_model_selection_fit(label)
        cx_ic[label] = float(ic_getter(fit))
        rg_ic[label] = float(ref[ic_getter.__ic_key__])
    return cx_ic, rg_ic


def _aic_getter(fit):
    return fit.aic()


_aic_getter.__ic_key__ = "aic"


def _bic_getter(fit):
    return fit.bic()


_bic_getter.__ic_key__ = "bic"


class TestModelSelectionConsistency:
    """AIC and BIC rankings across (GARCH, IGARCH, GJR, EGARCH) fitted on
    ONE shared series produce the same ordering in copulax as rugarch.

    The reference (``model_selection_reference_data.py``) is a
    common-series reference: rugarch fits all four variants on the SAME
    simulated ``arma11_garch11_normal``-style series, so both sides of
    the comparison rank the variants on identical data. This is the fix
    for J2 (HARD-05): the previous reference compared copulax's
    same-series ranking against rugarch numbers computed on four
    different per-variant series (var 18.14 vs ~1.7 across labels), whose
    ``igarch last`` agreement was a scale artifact rather than a
    like-for-like ranking. Catches a defect in the IC formula (e.g. the
    CR-01 dof overcount) without requiring exact agreement on absolute
    values."""

    def test_aic_ranking_matches_rugarch(self):
        cx_aics, rg_aics = _fit_common_series_ic(_aic_getter)
        cx_rank = sorted(cx_aics, key=lambda k: cx_aics[k])
        rg_rank = sorted(rg_aics, key=lambda k: rg_aics[k])
        assert cx_rank == rg_rank, (
            f"AIC rank mismatch: copulax={cx_rank} rugarch={rg_rank}"
        )

    def test_bic_ranking_matches_rugarch(self):
        cx_bics, rg_bics = _fit_common_series_ic(_bic_getter)
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
        """Same data + same init + same maxiter -> reproducible fit.

        Deliberately NOT routed through the shared registry: the point
        is that two INDEPENDENT fits land on the same optimum, which a
        shared instance would make vacuously true.
        """
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
        fit = shared_fit(
            ArmaGarch(
                mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
                residual_dist=normal,
            ),
            _matrix_series_name(base_fit.label), tier=STANDARD,
            y=y_short, tag="first_120",
        )
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
        truth = {
            "phi": (0.3,), "theta": (0.0,), "mu": 0.0,
            "omega": 0.001, "alpha": (0.05,), "beta": (0.94,),
        }
        # Frozen near-boundary AR(1)-GARCH(1,1) draw (persistence 0.99),
        # parametrised by truth rather than by rugarch reference data.
        # The recursion that produced it lives in the committed
        # regenerator, not here.
        name = "ar1garch11_nearboundary_n1500_s99"
        y_short = series(name)
        fit = shared_fit(
            ArmaGarch(
                mean_order=(1, 0), var_model=GARCH, var_order=(1, 1),
                residual_dist=normal,
            ),
            name, tier=PRECISION,
        )

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


# ---------------------------------------------------------------------------
# Shared-fit isolation guard
#
# Placed last so it collects after every class that consumes the matrix
# fixtures: by the time it runs, the caches have served their full
# workload and any mutation a consumer performed is already visible.
# ---------------------------------------------------------------------------

class TestSharedFitIsolation:
    """The shared fit registry hands out ONE frozen fitted model per key
    and a FRESH wrapper per request — across modules, not just here.

    Sharing a fitted model between fixtures is only sound while nothing
    writes to it.  This guard pins every leg of that argument:
    consumers receive the same fitted instance (so the fit really did
    run once), they receive distinct wrappers (so ``case.fit = ...``
    style writes cannot leak between fixtures), the shared instance
    still matches the snapshot taken when it was built (so nothing
    mutated it in flight), a differing key never collides with it, and a
    BEHAVIOURAL fit is never served from — nor written into — the cache.

    The cross-MODULE half of the contract is asserted from a second file
    (``test_timeseries_variance.py::TestSharedRegistryCrossModule``), so
    the identity holds whichever file pytest collects first.
    """

    def test_cached_fit_is_shared_wrapper_is_fresh_and_unmutated(
        self, base_fit,
    ):
        label = base_fit.label

        # (a) Identity: the wrapper each fixture builds — matrix_fit,
        # rugarch_fit and base_fit all call the same accessor — exposes
        # the one registry-held fitted model, so the label is fitted once.
        from_matrix = _matrix_case_view(label)
        from_rugarch = _matrix_case_view(label)
        assert from_matrix.fit is base_fit.fit
        assert from_rugarch.fit is base_fit.fit
        assert from_matrix.y is base_fit.y

        # (b) Distinct wrappers: attribute writes stay local.
        assert from_matrix is not from_rugarch
        assert from_matrix is not base_fit
        assert from_rugarch is not base_fit
        sentinel = object()
        from_matrix.fit = sentinel
        assert from_rugarch.fit is base_fit.fit
        assert _cached_matrix_fit(label) is base_fit.fit

        # (c) Mutation tripwire: the shared fitted model still equals
        # the snapshot the registry captured when it was first built.
        assert_snapshot_intact(base_fit.fit_key)

    def test_registry_key_separates_differing_fits(self, base_fit):
        """A fit that differs in ANY key component is a different entry.

        The registry key is ``(tier, model signature, series name, data
        tag, data digest, fit arguments)``.  Changing the tier, the
        model structure or an explicit fit argument must each produce a
        distinct fitted instance, never the REFERENCE one.
        """
        case = _cached_case(base_fit.label)
        name = _matrix_series_name(base_fit.label)
        tag = _matrix_tag(base_fit.label)

        # Different tier.
        other_tier = shared_fit(
            _matrix_model(case), name, tier=STANDARD, y=case.y, tag=tag,
        )
        assert other_tier is not base_fit.fit

        # Different model structure (residual law).
        other_model = shared_fit(
            ArmaGarch(
                mean_order=case.mean_order, var_model=case.var_model,
                var_order=case.var_order, residual_dist=student_t,
            ),
            name, tier=STANDARD, y=case.y, tag=tag,
        )
        assert other_model is not other_tier

        # Different explicit fit argument.
        other_args = shared_fit(
            _matrix_model(case), name, tier=STANDARD, y=case.y, tag=tag,
            maxiter=17,
        )
        assert other_args is not other_tier

        # Same key twice is the same instance (the sharing itself).
        assert shared_fit(
            _matrix_model(case), name, tier=STANDARD, y=case.y, tag=tag,
        ) is other_tier

    def test_behavioural_fits_are_never_shared(self, base_fit):
        """BEHAVIOURAL fits bypass the cache entirely, in both
        directions: two identical requests return DISTINCT instances,
        and neither is the shared REFERENCE fit."""
        case = _cached_case(base_fit.label)
        name = _matrix_series_name(base_fit.label)
        tag = _matrix_tag(base_fit.label)
        kwargs = dict(
            tier=BEHAVIOURAL, y=case.y, tag=tag,
            init="analytical", maxiter=2, lr=_FIT_LR,
        )
        first = shared_fit(_matrix_model(case), name, **kwargs)
        second = shared_fit(_matrix_model(case), name, **kwargs)
        assert first is not second
        assert first is not base_fit.fit
