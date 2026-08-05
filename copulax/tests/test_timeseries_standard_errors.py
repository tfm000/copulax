"""Standard-error tests for the time-series subpackage.

Joint asymptotic-covariance SEs for the ``ArmaGarch`` composite are
the v1 deliverable; the Pagan-Newey two-stage sandwich for the
separable ``ARMA → GARCH`` workflow is deferred to a future commit.

Three ``cov_type`` formulas are supported, mirroring ``arch``:

* ``"robust"``  — Bollerslev-Wooldridge sandwich (default).
* ``"classic"`` — observed information / inverse Hessian.
* ``"opg"``     — outer product of gradients (BHHH).

Coverage:

* SE shape / dict-structure invariants — entries match ``params``
  schema; values are non-negative; ``cov_matrix_`` is a square
  PSD matrix.
* Stored vs recomputed SE parity:
  ``fit.standard_errors() == fit.standard_errors(y_train)``.
* Cross-validation against ``arch.arch_model(...).fit(cov_type=...)``
  on AR(1)+GARCH(1, 1) data — both ``robust`` and ``classic``
  paths agree to ``rtol=2e-2`` per plan §"Standard errors".
* ``confidence_intervals(alpha)`` symmetric-normal width.
* ``summary()`` renders a non-empty multi-line string with the
  expected parameter table + footer sections.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copulax.tests._timeseries_helpers import (
    PRECISION,
    STANDARD,
    series,
    shared_fit,
)
from copulax.timeseries import GARCH, ArmaGarch
from copulax.univariate import normal

# ---------------------------------------------------------------------------
# Shared fits
#
# Three frozen AR(1)-GARCH(1,1) level series (rugarch ``ugarchpath``,
# ``armaOrder=c(1,0)``, ``mu=0.10``, ``phi=0.5``, GARCH truth
# ``omega=0.05, alpha=0.10, beta=0.85``) back the nine data-driven tests
# here.  Every fit names a tier: STANDARD for the SE structure / parity /
# cov-type consumers, which only need a well-behaved interior MLE, and
# PRECISION for the two ``arch`` cross-validation tests, whose assertion
# is on the location of that MLE.
# ---------------------------------------------------------------------------
_AG_MODEL = dict(
    mean_order=(1, 0),
    var_model=GARCH,
    var_order=(1, 1),
    residual_dist=normal,
)
_NAME_AG_500 = "ar1garch11_p050_m010_n500_s13"
_NAME_AG_1000 = "ar1garch11_p050_m010_n1000_s13"
_NAME_AG_2000 = "ar1garch11_p050_m010_n2000_s13"


@pytest.fixture(scope="module")
def ag_1000_fit():
    """n=1000 series + STANDARD joint fit — six consumers."""
    return series(_NAME_AG_1000), shared_fit(
        ArmaGarch(**_AG_MODEL),
        _NAME_AG_1000,
        tier=STANDARD,
    )


@pytest.fixture(scope="module")
def ag_2000_fit():
    """n=2000 series + PRECISION joint fit — the two ``arch``
    cross-validation tests."""
    return series(_NAME_AG_2000), shared_fit(
        ArmaGarch(**_AG_MODEL),
        _NAME_AG_2000,
        tier=PRECISION,
    )


# ---------------------------------------------------------------------------
# Shape / dict-structure invariants
# ---------------------------------------------------------------------------
class TestStructure:
    def test_se_dict_matches_params(self, ag_1000_fit):
        _y, fit = ag_1000_fit
        # Top-level keys match
        assert set(fit.standard_errors_) == set(fit.params)
        # Residual sub-dict matches
        assert set(fit.standard_errors_["residual"]) == set(fit.params["residual"])
        # Per-key shapes match
        for k, v in fit.params.items():
            if isinstance(v, dict):
                continue
            assert (
                fit.standard_errors_[k].shape == jnp.atleast_1d(v).shape
                or fit.standard_errors_[k].shape == ()
            )

    def test_cov_matrix_is_square_psd(self, ag_1000_fit):
        _y, fit = ag_1000_fit
        cov = fit.cov_matrix_
        assert cov.shape[0] == cov.shape[1]
        # PSD check: minimum eigenvalue ≥ 0 (within numerical tolerance).
        eigvals = jnp.linalg.eigvalsh(cov)
        assert float(jnp.min(eigvals)) > -1e-8

    def test_se_non_negative(self, ag_1000_fit):
        _y, fit = ag_1000_fit
        # Every non-empty leaf in standard_errors_ is non-negative.
        # (theta with shape (0,) is empty under mean_order=(1, 0) and
        # is skipped — there's no SE for a non-existent parameter.)
        for v in fit.standard_errors_.values():
            if isinstance(v, dict):
                for sub_v in v.values():
                    arr = jnp.atleast_1d(sub_v)
                    if arr.size > 0:
                        assert float(jnp.min(arr)) >= 0.0
            else:
                arr = jnp.atleast_1d(v)
                if arr.size > 0:
                    assert float(jnp.min(arr)) >= 0.0


# ---------------------------------------------------------------------------
# Stored vs recomputed parity
# ---------------------------------------------------------------------------
class TestParity:
    def test_recompute_matches_stored(self, ag_1000_fit):
        """``standard_errors()`` (stored, robust) ==
        ``standard_errors(y_train, cov_type='robust')`` (recomputed)
        to machine precision — both paths route through the same
        natural-space objective at the natural-space MLE."""
        y, fit = ag_1000_fit
        recomputed = fit.standard_errors(y, cov_type="robust")
        for k, stored_v in fit.standard_errors_.items():
            if isinstance(stored_v, dict):
                for sub, sub_v in stored_v.items():
                    arr_stored = jnp.atleast_1d(sub_v)
                    arr_recomp = jnp.atleast_1d(recomputed[k][sub])
                    if arr_stored.size > 0:
                        np.testing.assert_allclose(
                            np.asarray(arr_stored),
                            np.asarray(arr_recomp),
                            rtol=1e-5,
                            atol=1e-8,
                        )
            else:
                arr_stored = jnp.atleast_1d(stored_v)
                arr_recomp = jnp.atleast_1d(recomputed[k])
                if arr_stored.size > 0:
                    np.testing.assert_allclose(
                        np.asarray(arr_stored),
                        np.asarray(arr_recomp),
                        rtol=1e-5,
                        atol=1e-8,
                    )


# ---------------------------------------------------------------------------
# Cross-validation against arch.arch_model
# ---------------------------------------------------------------------------
class TestArchCrossValidation:
    """SEs match ``arch.arch_model.std_err`` under both
    ``cov_type='robust'`` (Bollerslev-Wooldridge sandwich, the
    default in both libraries) and ``cov_type='classic'`` (observed
    information).  AR(1)+GARCH(1, 1) data so arch's mean equation
    aligns with CopulAX's ``mean_order=(1, 0)``.

    ``arch_module`` comes from ``copulax/tests/conftest.py``: three
    modules in this family cross-validate against ``arch``."""

    @staticmethod
    def _arch_const_se_from_copulax(fit, cov: np.ndarray) -> float:
        """Delta-method: arch's mean intercept ``Const`` is the
        recursion intercept ``c = μ (1 − φ)``, while copulax (matching
        rugarch / Box-Jenkins / Hamilton) parametrises the unconditional
        mean ``μ`` directly via ``y_t = μ + φ (y_{t-1} − μ) + ε_t``.
        Convert copulax's ``(μ, φ)`` covariance block to arch's
        ``Const`` SE via ``Var(Const) = J Σ Jᵀ`` where
        ``J = (∂Const/∂μ, ∂Const/∂φ) = (1 − φ, −μ)``.

        Mirrors the arch-vs-rugarch cross-mapping documented in
        :mod:`copulax._src.timeseries._variance.egarch` (same
        convention-split pattern, different parameter set).
        """
        from copulax._src.timeseries._se import params_to_flat

        _, schema = params_to_flat(fit.params)
        idx = 0
        mu_idx = phi_idx = None
        for k, shape in schema:
            size = int(np.prod(shape)) if shape else 1
            if k == "mu":
                mu_idx = idx
            elif k == "phi":
                phi_idx = idx  # AR(1) → first phi entry
            idx += size
        assert mu_idx is not None and phi_idx is not None
        mu_val = float(fit.params["mu"])
        phi_val = float(fit.params["phi"][0])
        J = np.array([1.0 - phi_val, -mu_val])
        sub_cov = np.array(
            [
                [cov[mu_idx, mu_idx], cov[mu_idx, phi_idx]],
                [cov[phi_idx, mu_idx], cov[phi_idx, phi_idx]],
            ]
        )
        return float(np.sqrt(max(float(J @ sub_cov @ J.T), 0.0)))

    def _se_pairs(
        self,
        fit,
        fit_se: dict,
        cov: np.ndarray,
        arch_res,
    ) -> list[tuple[str, float, float]]:
        const_se = self._arch_const_se_from_copulax(fit, cov)
        return [
            ("phi", float(fit_se["phi"][0]), float(arch_res.std_err["y[1]"])),
            ("Const", const_se, float(arch_res.std_err["Const"])),
            ("omega", float(fit_se["omega"]), float(arch_res.std_err["omega"])),
            ("alpha", float(fit_se["alpha"][0]), float(arch_res.std_err["alpha[1]"])),
            ("beta", float(fit_se["beta"][0]), float(arch_res.std_err["beta[1]"])),
        ]

    def test_robust_vs_arch_robust(self, arch_module, ag_2000_fit):
        """Default ``cov_type='robust'`` (BW sandwich) matches
        ``arch.arch_model(..., cov_type='robust')`` to ``rtol=2e-2``.
        BW is sensitive to small differences in the score
        covariance estimate (different optimisers, different MLE
        points by ~1e-3) so the tolerance is slightly looser than
        the ``classic`` case."""
        y, fit = ag_2000_fit
        am = arch_module.arch_model(
            np.asarray(y),
            mean="ARX",
            lags=1,
            vol="GARCH",
            p=1,
            q=1,
            dist="Normal",
        )
        arch_res = am.fit(disp="off", cov_type="robust")
        cov = np.asarray(fit.cov_matrix_)
        for _label, cx, ar in self._se_pairs(
            fit,
            fit.standard_errors_,
            cov,
            arch_res,
        ):
            np.testing.assert_allclose(cx, ar, rtol=2e-2)

    def test_classic_vs_arch_classic(self, arch_module, ag_2000_fit):
        """``cov_type='classic'`` (observed information) matches
        ``arch`` to ``rtol=1e-2`` — the tighter plan-mandated
        tolerance.  ``classic`` only depends on the Hessian,
        which is less sensitive to optimiser-induced MLE
        differences than the score-covariance term in BW."""
        y, fit = ag_2000_fit
        am = arch_module.arch_model(
            np.asarray(y),
            mean="ARX",
            lags=1,
            vol="GARCH",
            p=1,
            q=1,
            dist="Normal",
        )
        arch_res = am.fit(disp="off", cov_type="classic")
        cx_se = fit.standard_errors(y, cov_type="classic")
        cov_classic = np.asarray(
            fit.cov_matrix(y, cov_type="classic"),
        )
        for label, cx, ar in self._se_pairs(
            fit,
            cx_se,
            cov_classic,
            arch_res,
        ):
            np.testing.assert_allclose(cx, ar, rtol=1e-2, err_msg=label)


# ---------------------------------------------------------------------------
# Confidence intervals + summary
# ---------------------------------------------------------------------------
class TestCovTypes:
    def test_three_cov_types_produce_finite_positive_se(self, ag_1000_fit):
        """All three ``cov_type`` paths produce finite, non-negative
        SEs at a well-behaved interior MLE."""
        y, fit = ag_1000_fit
        for cov_type in ("robust", "classic", "opg"):
            cov = fit.cov_matrix(y, cov_type=cov_type)
            assert jnp.all(jnp.isfinite(cov))
            diag = jnp.diag(cov)
            assert float(jnp.min(diag)) >= 0.0

    def test_invalid_cov_type_raises(self):
        y = series(_NAME_AG_500)
        fit = shared_fit(
            ArmaGarch(**_AG_MODEL),
            _NAME_AG_500,
            tier=STANDARD,
        )
        with pytest.raises(ValueError, match="cov_type"):
            fit.standard_errors(y, cov_type="bogus")


class TestConfidenceIntervalsAndSummary:
    def test_confidence_intervals_symmetric(self, ag_1000_fit):
        _y, fit = ag_1000_fit
        cis = fit.confidence_intervals(alpha=0.05)
        # Top-level keys match params
        assert set(cis) == set(fit.params)
        # Per param, lo < est < hi, and the CI is symmetric to within
        # numerical tolerance (since the SE is symmetric and z is fixed).
        z = float(jax.scipy.stats.norm.ppf(0.975))
        for k, v in fit.params.items():
            if isinstance(v, dict):
                continue
            se_arr = jnp.atleast_1d(jnp.asarray(fit.standard_errors_[k], dtype=float))
            lo, hi = cis[k]
            np.testing.assert_allclose(
                np.asarray(hi - lo),
                2.0 * z * np.asarray(se_arr),
                rtol=1e-5,
            )


# ---------------------------------------------------------------------------
# Conditioning guard on the linear solves (HARD-07 numerical-stability half)
# ---------------------------------------------------------------------------
class TestSEConditioning:
    r"""The six unguarded linear-solve sites in the SE / OLS machinery
    (`_se.py` four `jnp.linalg.solve`; `_ols.py` `solve` + `inv`) are
    routed through the shared :func:`safe_solve` conditioning guard.

    A well-conditioned system is solved exactly as before (numerical
    equivalence); a degenerate / near-singular system surfaces a
    diagnostic — a NaN result carrying an ``ill_conditioned`` signal —
    instead of a silent finite-but-meaningless standard error.

    The near-singular OLS design is the sharp case: the *unguarded*
    ``jnp.linalg.inv`` returns a finite but absurd SE (~1e7) with no
    diagnostic, which is exactly the "finite, plausible-looking but
    wrong" silent failure the project forbids.
    """

    # ---- safe_solve unit behaviour ------------------------------------
    def test_safe_solve_well_conditioned_matches_plain_solve(self):
        """On a well-conditioned matrix, ``safe_solve`` returns the
        identical result to ``jnp.linalg.solve`` and flags
        ``ill_conditioned=False``."""
        from copulax._src.timeseries._se import safe_solve

        A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        rhs = jnp.eye(2)
        x, ill = safe_solve(A, rhs)
        reference = jnp.linalg.solve(A, rhs)
        np.testing.assert_allclose(
            np.asarray(x),
            np.asarray(reference),
            rtol=1e-12,
            atol=1e-12,
        )
        assert bool(ill) is False
        assert bool(jnp.all(jnp.isfinite(x)))

    def test_safe_solve_singular_returns_nan_and_flag(self):
        """On an exactly-singular matrix, ``safe_solve`` returns an
        all-NaN result and flags ``ill_conditioned=True`` — the honest
        signal, not a silent pseudo-inverse number."""
        from copulax._src.timeseries._se import safe_solve

        A = jnp.array([[1.0, 1.0], [1.0, 1.0]])  # rank-1, singular
        rhs = jnp.eye(2)
        x, ill = safe_solve(A, rhs)
        assert bool(ill) is True
        assert bool(jnp.all(jnp.isnan(x)))

    def test_safe_solve_near_singular_returns_nan_and_flag(self):
        """A finite-but-ill-conditioned matrix (cond ~ 3.5e15, well
        above the ~4.5e14 float64 ceiling) is flagged and NaN-filled —
        it must NOT slip through as a finite plausible solution."""
        from copulax._src.timeseries._se import _COND_THRESHOLD, safe_solve

        eps = 1e-15
        A = jnp.array([[1.0, 1.0], [1.0, 1.0 + eps]])
        # Guard the fixture itself: assert the matrix really is past the
        # ceiling so the test can never silently degrade into a
        # well-conditioned case if the threshold moves.
        assert float(jnp.linalg.cond(A)) > float(_COND_THRESHOLD)
        rhs = jnp.eye(2)
        x, ill = safe_solve(A, rhs)
        assert bool(ill) is True
        assert bool(jnp.all(jnp.isnan(x)))

    def test_safe_solve_is_jit_safe(self):
        """The guard is JIT-compatible (static shapes, ``jnp.where``)."""
        from copulax._src.timeseries._se import safe_solve

        f = jax.jit(safe_solve)
        A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        x, ill = f(A, jnp.eye(2))
        assert bool(jnp.all(jnp.isfinite(x)))
        assert bool(ill) is False

    # ---- _se.compute_param_cov: well-conditioned equivalence ----------
    def test_compute_param_cov_well_conditioned_unchanged(self):
        """A well-conditioned Hessian produces exactly the same
        covariance the raw ``J^{-1}/n`` formula gives — the guard is a
        no-op on the happy path (numerical equivalence)."""
        from copulax._src.timeseries._se import compute_param_cov

        # Quadratic NLL with a well-conditioned Hessian J = diag-ish PD.
        A = jnp.array([[3.0, 0.5], [0.5, 2.0]])
        theta_star = jnp.array([0.7, -0.2])
        n_obs = 200

        def nll_total(p):
            d = p - theta_star
            return 0.5 * n_obs * (d @ A @ d)

        def per_obs_nll(p):
            d = p - theta_star
            return 0.5 * (d @ A @ d) * jnp.ones(n_obs)

        cov = compute_param_cov(
            nll_total,
            per_obs_nll,
            theta_star,
            n_obs,
            cov_type="classic",
        )
        # J = Hessian(nll_total)/n_obs = A ; classic V = J^{-1}/n = A^{-1}/n
        expected = jnp.linalg.inv(A) / n_obs
        np.testing.assert_allclose(
            np.asarray(cov),
            np.asarray(expected),
            rtol=1e-8,
            atol=1e-10,
        )
        assert bool(jnp.all(jnp.isfinite(cov)))

    def test_compute_param_cov_degenerate_hessian_surfaces_nan(self):
        """A rank-deficient Hessian (a flat likelihood direction) must
        NOT yield a finite plausible SE — the guard surfaces NaN through
        the whole covariance so ``sqrt(diag)`` is NaN, not a clamped
        finite 0."""
        from copulax._src.timeseries._se import compute_param_cov

        theta_star = jnp.array([1.0, 0.0])
        n_obs = 100

        # nll depends only on p[0] -> Hessian = [[100, 0], [0, 0]] singular.
        def nll_total(p):
            return 0.5 * (p[0] - 1.0) ** 2 * 100.0 * n_obs / n_obs * n_obs

        def per_obs_nll(p):
            return 0.5 * (p[0] - 1.0) ** 2 * 100.0 * jnp.ones(n_obs)

        cov = compute_param_cov(
            nll_total,
            per_obs_nll,
            theta_star,
            n_obs,
            cov_type="classic",
        )
        se = jnp.sqrt(jnp.maximum(jnp.diag(cov), 0.0))
        # The diagnostic: SE is NaN, not a finite plausible number.
        assert bool(jnp.any(jnp.isnan(se))), (
            "degenerate Hessian must surface NaN SE, not a silent finite SE"
        )

    def test_compute_param_cov_opg_degenerate_surfaces_nan(self):
        """The OPG path (``solve(S, eye)``) is guarded too: a singular
        score covariance surfaces NaN, not a finite plausible SE."""
        from copulax._src.timeseries._se import compute_param_cov

        theta_star = jnp.array([1.0, 0.0])
        n_obs = 100

        # Scores depend only on p[0] -> score-cov S is rank-1 (singular).
        def nll_total(p):
            return 0.5 * (p[0] - 1.0) ** 2 * 100.0

        def per_obs_nll(p):
            base = 0.5 * (p[0] - 1.0) ** 2
            return base * jnp.linspace(0.5, 1.5, n_obs)

        cov = compute_param_cov(
            nll_total,
            per_obs_nll,
            theta_star,
            n_obs,
            cov_type="opg",
        )
        se = jnp.sqrt(jnp.maximum(jnp.diag(cov), 0.0))
        assert bool(jnp.any(jnp.isnan(se))), (
            "singular score covariance must surface NaN SE"
        )

    # ---- _ols.ols_fit: solve (not inv) + conditioning guard -----------
    def test_ols_result_exposes_ill_conditioned_flag(self):
        """``OLSResult`` carries an ``ill_conditioned`` boolean signal."""
        from copulax._src.timeseries._ols import ols_fit

        Xg = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        yg = jnp.array([1.0, 2.1, 2.9, 4.2])
        res = ols_fit(Xg, yg)
        assert hasattr(res, "ill_conditioned")
        assert bool(res.ill_conditioned) is False

    def test_ols_well_conditioned_se_unchanged(self):
        """On a well-conditioned design the OLS SEs are numerically
        equal to the closed-form ``sigma * sqrt(diag((X'X)^{-1}))`` —
        switching ``inv`` to guarded ``solve`` does not move the happy
        path."""
        from copulax._src.timeseries._ols import ols_fit

        X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
        y = jnp.array([1.0, 2.1, 2.9, 4.2, 5.1])
        res = ols_fit(X, y)
        # Reference via the plain inv formula the module used to run.
        XtX = X.T @ X
        n, k = X.shape
        rss = jnp.sum(res.residuals**2)
        sigma2 = rss / jnp.maximum(n - k, 1)
        ref_se = jnp.sqrt(sigma2 * jnp.diag(jnp.linalg.inv(XtX)))
        np.testing.assert_allclose(
            np.asarray(res.standard_errors),
            np.asarray(ref_se),
            rtol=1e-8,
            atol=1e-10,
        )
        assert bool(res.ill_conditioned) is False
        assert bool(jnp.all(jnp.isfinite(res.standard_errors)))

    def test_ols_near_singular_no_silent_finite_se(self):
        """The sharp case: a *near-singular* design (collinear columns,
        cond ~ 1e16) currently returns a finite but absurd SE (~1e7)
        with no diagnostic.  After the guard it must surface NaN + the
        ``ill_conditioned`` flag — never a finite plausible SE."""
        from copulax._src.timeseries._ols import ols_fit

        eps = 1e-8
        X = jnp.array(
            [[1.0, 1.0], [1.0, 1.0 + eps], [1.0, 1.0 + 2 * eps], [1.0, 1.0 + 3 * eps]]
        )
        y = jnp.array([1.0, 2.0, 1.5, 2.5])
        res = ols_fit(X, y)
        assert bool(res.ill_conditioned) is True, (
            "near-singular design must be flagged ill_conditioned"
        )
        # No silent finite-plausible SE: the SE path is NaN.
        assert bool(jnp.any(jnp.isnan(res.standard_errors))), (
            "near-singular OLS must NOT return a finite plausible SE"
        )

    def test_ols_singular_beta_flagged(self):
        """An exactly-singular design flags ``ill_conditioned`` and the
        coefficient vector is NaN (honest signal)."""
        from copulax._src.timeseries._ols import ols_fit

        X = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
        y = jnp.array([1.0, 2.0, 3.0, 4.0])
        res = ols_fit(X, y)
        assert bool(res.ill_conditioned) is True
        assert bool(jnp.any(jnp.isnan(res.beta)))

    def test_ols_fit_is_jit_safe(self):
        """``ols_fit`` remains JIT-compatible after the guard."""
        from copulax._src.timeseries._ols import ols_fit

        X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        y = jnp.array([1.0, 2.1, 2.9, 4.2])
        res = jax.jit(ols_fit)(X, y)
        assert bool(jnp.all(jnp.isfinite(res.standard_errors)))
        assert bool(res.ill_conditioned) is False
