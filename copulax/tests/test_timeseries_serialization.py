"""Save/load round-trip tests for time-series models.

Round-trips every public model in :mod:`copulax.timeseries` —
mean models (AR / MA / ARMA), all six variance variants
(GARCH / IGARCH / GJR_GARCH / EGARCH / TGARCH / QGARCH /
GARCH_M), and the joint ``ArmaGarch`` composite under each
supported variance backend.

Each round-trip checks:

* The ``params`` dict is preserved exactly (parameter values match
  bit-for-bit after load).
* Fit-time diagnostics (``loglikelihood_``, ``aic_``, ``bic_``,
  ``n_train_``) survive.
* Standard errors and the covariance matrix (where stored)
  survive — relevant for ``ArmaGarch`` only.
* ``conditional_variance(y)`` / ``conditional_mean(y)`` / forecast
  outputs match between the original and the loaded instance,
  proving the recursion graph works post-load.
* The ``terminal_state`` is preserved across the round-trip.
* The static ``residual_dist`` and (for ``ArmaGarch``) the
  ``var_model`` class field are restored to the same singleton /
  class.

File-format invariants (``.cpx`` extension auto-append, metadata
dispatch fields, etc.) are also verified.
"""

from __future__ import annotations

import json
import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import copulax
from copulax.timeseries import (
    AR,
    ARMA,
    EGARCH,
    GARCH,
    GARCH_M,
    GJR_GARCH,
    IGARCH,
    MA,
    QGARCH,
    TGARCH,
    ArmaGarch,
)
from copulax.univariate import normal, student_t


# ---------------------------------------------------------------------------
# Shared simulated series
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def y_series():
    """Short simulated return series for level-input models."""
    key = jax.random.PRNGKey(7)
    return jax.random.normal(key, (400,)) * 0.1


@pytest.fixture(scope="module")
def eps_series():
    """Short simulated innovation series for variance-only models."""
    key = jax.random.PRNGKey(11)
    return jax.random.normal(key, (400,)) * 0.1


# ---------------------------------------------------------------------------
# Equality helpers
# ---------------------------------------------------------------------------
def _assert_array_equal(a, b, label=""):
    np.testing.assert_array_equal(
        np.asarray(a), np.asarray(b),
        err_msg=f"Mismatch on {label}",
    )


def _assert_params_equal(p1: dict, p2: dict, prefix: str = ""):
    """Walk a params dict and assert exact equality at every leaf."""
    assert set(p1.keys()) == set(p2.keys()), (
        f"Param keys differ at {prefix or '<root>'}: "
        f"{sorted(p1)} vs {sorted(p2)}"
    )
    for key, val in p1.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            _assert_params_equal(val, p2[key], prefix=full)
        else:
            _assert_array_equal(val, p2[key], label=full)


def _assert_diagnostics_match(orig, loaded):
    for attr in ("loglikelihood_", "aic_", "bic_"):
        v1 = getattr(orig, attr, None)
        v2 = getattr(loaded, attr, None)
        if v1 is not None or v2 is not None:
            _assert_array_equal(v1, v2, label=attr)
    assert orig.n_train_ == loaded.n_train_


# ---------------------------------------------------------------------------
# Mean models
# ---------------------------------------------------------------------------
MEAN_CONFIGS = [
    pytest.param(AR, {"p": 1}, normal, id="AR1-normal"),
    pytest.param(AR, {"p": 2}, student_t, id="AR2-student_t"),
    pytest.param(MA, {"q": 1}, normal, id="MA1-normal"),
    pytest.param(ARMA, {"p": 1, "q": 1}, normal, id="ARMA11-normal"),
    pytest.param(ARMA, {"p": 1, "q": 1}, student_t, id="ARMA11-student_t"),
]


class TestMeanModelRoundTrip:
    """Round-trip the AR / MA / ARMA mean models."""

    @pytest.mark.parametrize("cls,kwargs,resid", MEAN_CONFIGS)
    def test_round_trip_preserves_params_and_diagnostics(
        self, tmp_path, y_series, cls, kwargs, resid,
    ):
        fit = cls(residual_dist=resid, **kwargs).fit(y_series, maxiter=80)
        path = tmp_path / f"{cls.__name__}.cpx"
        fit.save(str(path))
        loaded = copulax.load(str(path))

        assert type(loaded) is type(fit)
        assert type(loaded.residual_dist) is type(resid)
        _assert_params_equal(fit.params, loaded.params)
        _assert_diagnostics_match(fit, loaded)
        # Conditional mean trajectory matches.
        np.testing.assert_array_equal(
            np.asarray(fit.conditional_mean(y_series)),
            np.asarray(loaded.conditional_mean(y_series)),
        )

    def test_terminal_state_preserved(self, tmp_path, y_series):
        fit = ARMA(p=1, q=1, residual_dist=normal).fit(y_series, maxiter=80)
        path = tmp_path / "arma_terminal.cpx"
        fit.save(str(path))
        loaded = copulax.load(str(path))
        _assert_array_equal(
            fit.terminal_state.y_lags, loaded.terminal_state.y_lags,
            label="terminal_state.y_lags",
        )
        _assert_array_equal(
            fit.terminal_state.eps_lags, loaded.terminal_state.eps_lags,
            label="terminal_state.eps_lags",
        )

    def test_forecast_matches_post_load(self, tmp_path, y_series):
        fit = AR(p=1, residual_dist=normal).fit(y_series, maxiter=80)
        path = tmp_path / "ar_forecast.cpx"
        fit.save(str(path))
        loaded = copulax.load(str(path))
        f1 = fit.forecast(5)
        f2 = loaded.forecast(5)
        np.testing.assert_array_equal(
            np.asarray(f1["mean"]), np.asarray(f2["mean"]),
        )
        np.testing.assert_array_equal(
            np.asarray(f1["variance"]), np.asarray(f2["variance"]),
        )


# ---------------------------------------------------------------------------
# Variance models (eps-input)
# ---------------------------------------------------------------------------
VARIANCE_CLASSES = [GARCH, IGARCH, GJR_GARCH, EGARCH, TGARCH, QGARCH]


class TestVarianceModelRoundTrip:
    """Round-trip every GARCH-family variance variant."""

    @pytest.mark.parametrize(
        "cls", VARIANCE_CLASSES, ids=[c.__name__ for c in VARIANCE_CLASSES],
    )
    def test_round_trip_preserves_params_and_recursion(
        self, tmp_path, eps_series, cls,
    ):
        fit = cls(p=1, q=1, residual_dist=normal).fit(eps_series, maxiter=80)
        path = tmp_path / f"{cls.__name__}.cpx"
        fit.save(str(path))
        loaded = copulax.load(str(path))

        assert type(loaded) is type(fit)
        _assert_params_equal(fit.params, loaded.params)
        _assert_diagnostics_match(fit, loaded)
        # Save-side and load-side residual_dist must both be the
        # promoted (fitted, parameterised) standardised instance.
        assert fit.residual_dist._stored_params is not None
        assert loaded.residual_dist._stored_params is not None
        # Conditional variance must match after load.
        np.testing.assert_array_equal(
            np.asarray(fit.conditional_variance(eps_series)),
            np.asarray(loaded.conditional_variance(eps_series)),
        )

    @pytest.mark.parametrize(
        "cls", VARIANCE_CLASSES, ids=[c.__name__ for c in VARIANCE_CLASSES],
    )
    def test_terminal_state_preserved(self, tmp_path, eps_series, cls):
        fit = cls(p=1, q=1, residual_dist=normal).fit(eps_series, maxiter=60)
        path = tmp_path / f"{cls.__name__}_terminal.cpx"
        fit.save(str(path))
        loaded = copulax.load(str(path))
        # Each variant has its own terminal-state subclass, but the
        # leaves are always arrays.
        leaves_orig = jax.tree_util.tree_leaves(fit.terminal_state)
        leaves_loaded = jax.tree_util.tree_leaves(loaded.terminal_state)
        assert len(leaves_orig) == len(leaves_loaded)
        for i, (a, b) in enumerate(zip(leaves_orig, leaves_loaded)):
            _assert_array_equal(a, b, label=f"terminal_state.leaf[{i}]")

    def test_garch_t_residual_round_trip(self, tmp_path, eps_series):
        """Student-T residual params (the ν shape parameter) survive."""
        fit = GARCH(p=1, q=1, residual_dist=student_t).fit(
            eps_series, maxiter=80,
        )
        path = tmp_path / "garch_studentt.cpx"
        fit.save(str(path))
        loaded = copulax.load(str(path))
        assert type(loaded.residual_dist) is type(student_t)
        _assert_array_equal(
            fit.residual_params["nu"], loaded.residual_params["nu"],
            label="residual_params.nu",
        )

    def test_garch_m_round_trip(self, tmp_path, y_series):
        """GARCH-M has its own mu/lambda_m mean kwargs."""
        fit = GARCH_M(p=1, q=1, residual_dist=normal).fit(
            y_series, maxiter=80,
        )
        path = tmp_path / "garch_m.cpx"
        fit.save(str(path))
        loaded = copulax.load(str(path))
        _assert_array_equal(fit.mu, loaded.mu, label="mu")
        _assert_array_equal(
            fit.lambda_m, loaded.lambda_m, label="lambda_m",
        )
        _assert_params_equal(fit.params, loaded.params)
        np.testing.assert_array_equal(
            np.asarray(fit.conditional_variance(y_series)),
            np.asarray(loaded.conditional_variance(y_series)),
        )


# ---------------------------------------------------------------------------
# ArmaGarch joint composite
# ---------------------------------------------------------------------------
ARMA_GARCH_VARIANTS = [GARCH, IGARCH, GJR_GARCH, EGARCH, TGARCH, QGARCH]


class TestArmaGarchRoundTrip:
    """Round-trip the joint composite under every supported variant."""

    @pytest.mark.parametrize(
        "var_cls", ARMA_GARCH_VARIANTS,
        ids=[c.__name__ for c in ARMA_GARCH_VARIANTS],
    )
    def test_round_trip(self, tmp_path, y_series, var_cls):
        fit = ArmaGarch(
            mean_order=(1, 0),
            var_model=var_cls,
            var_order=(1, 1),
            residual_dist=normal,
        ).fit(y_series, maxiter=80)
        path = tmp_path / f"AG_{var_cls.__name__}.cpx"
        fit.save(str(path))
        loaded = copulax.load(str(path))

        assert type(loaded) is type(fit)
        assert loaded.var_model is var_cls
        assert loaded.mean_order == fit.mean_order
        assert loaded.var_order == fit.var_order
        _assert_params_equal(fit.params, loaded.params)
        _assert_diagnostics_match(fit, loaded)
        np.testing.assert_array_equal(
            np.asarray(fit.conditional_variance(y_series)),
            np.asarray(loaded.conditional_variance(y_series)),
        )
        np.testing.assert_array_equal(
            np.asarray(fit.conditional_mean(y_series)),
            np.asarray(loaded.conditional_mean(y_series)),
        )

    def test_standard_errors_preserved(self, tmp_path, y_series):
        fit = ArmaGarch(
            mean_order=(1, 0),
            var_model=GARCH,
            var_order=(1, 1),
            residual_dist=normal,
        ).fit(y_series, maxiter=80)
        path = tmp_path / "AG_se.cpx"
        fit.save(str(path))
        loaded = copulax.load(str(path))
        _assert_array_equal(
            fit.cov_matrix_, loaded.cov_matrix_, label="cov_matrix_",
        )
        _assert_params_equal(
            fit.standard_errors_, loaded.standard_errors_,
        )

    def test_terminal_state_preserved(self, tmp_path, y_series):
        fit = ArmaGarch(
            mean_order=(1, 1),
            var_model=GJR_GARCH,
            var_order=(1, 1),
            residual_dist=normal,
        ).fit(y_series, maxiter=80)
        path = tmp_path / "AG_terminal.cpx"
        fit.save(str(path))
        loaded = copulax.load(str(path))
        _assert_array_equal(
            fit.terminal_state.y_lags, loaded.terminal_state.y_lags,
            label="terminal_state.y_lags",
        )
        _assert_array_equal(
            fit.terminal_state.eps_lags, loaded.terminal_state.eps_lags,
            label="terminal_state.eps_lags",
        )
        for i, (a, b) in enumerate(zip(
            fit.terminal_state.var_state, loaded.terminal_state.var_state,
        )):
            _assert_array_equal(a, b, label=f"terminal_state.var_state[{i}]")


# ---------------------------------------------------------------------------
# File-format invariants
# ---------------------------------------------------------------------------
class TestFileFormat:
    """Verify .cpx-format invariants for the timeseries dispatch."""

    def test_auto_appends_cpx_extension(self, tmp_path, eps_series):
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(
            eps_series, maxiter=40,
        )
        path = tmp_path / "no_ext"
        fit.save(str(path))
        assert (tmp_path / "no_ext.cpx").exists()
        loaded = copulax.load(str(tmp_path / "no_ext.cpx"))
        _assert_params_equal(fit.params, loaded.params)

    def test_metadata_dispatch_fields(self, tmp_path, eps_series):
        fit = GARCH(p=1, q=1, residual_dist=student_t).fit(
            eps_series, maxiter=40,
        )
        path = tmp_path / "meta.cpx"
        fit.save(str(path))
        with zipfile.ZipFile(path, "r") as zf:
            metadata = json.loads(zf.read("metadata.json"))
        assert metadata["dist_family"] == "timeseries"
        assert metadata["dist_class"] == "GARCH"
        assert metadata["residual_dist_class"] == "StudentT"
        assert metadata["p"] == 1
        assert metadata["q"] == 1

    def test_arma_garch_metadata_records_var_model(self, tmp_path, y_series):
        fit = ArmaGarch(
            mean_order=(1, 0),
            var_model=GJR_GARCH,
            var_order=(1, 1),
            residual_dist=normal,
        ).fit(y_series, maxiter=40)
        path = tmp_path / "AG_meta.cpx"
        fit.save(str(path))
        with zipfile.ZipFile(path, "r") as zf:
            metadata = json.loads(zf.read("metadata.json"))
        assert metadata["dist_family"] == "timeseries"
        assert metadata["dist_class"] == "ArmaGarch"
        assert metadata["var_model_class"] == "GJR_GARCH"
        assert metadata["mean_order"] == [1, 0]
        assert metadata["var_order"] == [1, 1]

    def test_save_unfitted_raises(self, tmp_path):
        unfitted = GARCH(p=1, q=1, residual_dist=normal)
        with pytest.raises(ValueError, match="unfitted"):
            unfitted.save(str(tmp_path / "unfitted.cpx"))

    def test_name_override_on_load(self, tmp_path, eps_series):
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(
            eps_series, maxiter=40, name="original",
        )
        path = tmp_path / "rename.cpx"
        fit.save(str(path))
        loaded = copulax.load(str(path), name="renamed")
        assert loaded.name == "renamed"
        _assert_params_equal(fit.params, loaded.params)


# ---------------------------------------------------------------------------
# WR-03 — diag_n_train_ serialisation prefix collision
# ---------------------------------------------------------------------------
#
# ``_serialise_traced`` writes ``diag_n_train_`` for every model with
# ``n_train_`` set, INCLUDING models with no diagnostics bundle.  On load
# ``_deserialise_residual_diagnostics`` uses a ``startswith("diag_")``
# prefix test to decide whether any diagnostic state was serialised; that
# test matches ``diag_n_train_``, so for a diagnostics-less model the
# function used to build and return an empty dict ``{}``.  The constructor
# then stored ``{}`` (it only checks ``is not None``), so every downstream
# ``residual_diagnostics_ is not None`` gate passed and the fast paths
# raised a bare ``KeyError`` / ``TypeError`` instead of the designed
# informative error.  Per 00-REVIEW.md WR-03 the fix returns ``None`` for
# a diagnostics-less model (``diag_n_train_`` excluded from the prefix
# match / ``return diag or None``), so the constructor stores ``None`` and
# the WR-04 guards below produce an informative ``ValueError``.


def _fitted_diagnosticsless_armagarch():
    """A *fitted* (params present) ArmaGarch(1,1)×GARCH(1,1) that carries
    ``n_train_`` but NO ``residual_diagnostics_`` / ``standard_errors_``
    bundle — the exact WR-03/WR-04 trigger state.
    """
    return ArmaGarch(
        mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
        residual_dist=normal,
        phi=jnp.array([0.3]), theta=jnp.array([0.2]), mu=jnp.array(0.0),
        var_params={
            "omega": jnp.array(0.1),
            "alpha": jnp.array([0.1]),
            "beta": jnp.array([0.8]),
        },
        residual_params={}, n_train_=400,
    )


class TestDiagNTrainCollision:
    """WR-03: a diagnostics-less model must NOT resurrect a ``{}`` bundle
    from the ``diag_n_train_`` serialisation key."""

    def test_deserialise_returns_none_when_only_n_train_present(self):
        """The unit-level collision: given an ``arrays`` dict containing
        only ``diag_n_train_`` (and no diagnostics metadata / arrays),
        ``_deserialise_residual_diagnostics`` returns ``None`` — not an
        empty ``{}`` that later masks the informative error."""
        from copulax._src.timeseries._base import (
            _deserialise_residual_diagnostics,
        )

        out = _deserialise_residual_diagnostics(
            {"diag_n_train_": np.asarray(400)}, {},
        )
        assert out is None

    def test_round_trip_diagnosticsless_model_has_none_diagnostics(
        self, tmp_path,
    ):
        """End-to-end: saving and loading a fitted-but-diagnostics-less
        ``ArmaGarch`` yields ``residual_diagnostics_ is None`` on the
        loaded instance (not ``{}``), while ``n_train_`` still survives."""
        fit = _fitted_diagnosticsless_armagarch()
        assert fit.residual_diagnostics_ is None  # precondition
        path = tmp_path / "diagless.cpx"
        fit.save(str(path))
        loaded = copulax.load(str(path))

        assert loaded.residual_diagnostics_ is None
        assert loaded.n_train_ == 400
        _assert_params_equal(fit.params, loaded.params)

    def test_fitted_model_with_diagnostics_still_round_trips(
        self, tmp_path, y_series,
    ):
        """Regression: a genuinely diagnostics-bearing fit is unaffected —
        its bundle survives the round-trip (the fix only changes the
        diagnostics-less case)."""
        fit = ArmaGarch(
            mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y_series, maxiter=40)
        assert fit.residual_diagnostics_ is not None  # precondition
        path = tmp_path / "withdiag.cpx"
        fit.save(str(path))
        loaded = copulax.load(str(path))
        assert loaded.residual_diagnostics_ is not None
        _assert_array_equal(
            fit.residual_diagnostics_["loglikelihood"],
            loaded.residual_diagnostics_["loglikelihood"],
            label="loglikelihood",
        )


# ---------------------------------------------------------------------------
# WR-04 — informative ValueError on unfitted / diagnostics-less fast paths
# ---------------------------------------------------------------------------
#
# ``ArmaGarch.loglikelihood()/aic()/bic()`` (the ``y=None`` fast paths)
# and ``ArmaGarch.summary()`` dereference ``residual_diagnostics_[...]``
# (and ``summary`` also ``standard_errors_[...]``) guarded only by
# ``_require_fitted()``, which checks PARAMS, not the diagnostics bundle.
# A fitted instance whose bundle is ``None`` therefore crashed with a bare
# ``TypeError: 'NoneType' object is not subscriptable``.  These must raise
# an informative ``ValueError`` instead (the house ``_require_fitted``
# wording style), consistent with the sibling accessors and with
# ``ARMABase.summary()`` / ``GARCHBase.summary()``.


class TestUnfittedFastPathRaises:
    """WR-04: the cached fast paths raise an informative ``ValueError``."""

    @pytest.mark.parametrize("method", ["loglikelihood", "aic", "bic"])
    def test_diagnosticsless_scalar_accessor_raises_valueerror(self, method):
        """``loglikelihood()/aic()/bic()`` on a fitted-but-diagnostics-less
        model raise ``ValueError`` (NOT a bare ``TypeError``)."""
        model = _fitted_diagnosticsless_armagarch()
        assert model.is_fitted  # params present; only the bundle is absent
        with pytest.raises(ValueError):
            getattr(model, method)()
        # And specifically NOT a bare TypeError.
        with pytest.raises(ValueError):
            getattr(model, method)()

    def test_diagnosticsless_summary_raises_valueerror(self):
        """``summary()`` on a fitted-but-diagnostics-less model raises
        ``ValueError`` (NOT a bare ``TypeError`` from a ``None`` subscript
        of ``residual_diagnostics_`` / ``standard_errors_``)."""
        model = _fitted_diagnosticsless_armagarch()
        with pytest.raises(ValueError):
            model.summary()

    def test_truly_unfitted_scalar_accessors_raise_valueerror(self):
        """An unfitted ``ArmaGarch`` (no params) raises the informative
        ``_require_fitted`` ``ValueError`` on the fast paths — the
        regression leg of the same guard surface."""
        unfitted = ArmaGarch(
            mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        )
        assert not unfitted.is_fitted
        for method in ("loglikelihood", "aic", "bic"):
            with pytest.raises(ValueError, match="not fitted"):
                getattr(unfitted, method)()
        with pytest.raises(ValueError, match="not fitted"):
            unfitted.summary()

    def test_recompute_paths_unaffected_by_guard(self, y_series):
        """A fitted model with diagnostics still returns finite cached
        scalars, and the ``y``-recompute path is unaffected — proving the
        new guard only fences the ``None``-bundle case."""
        fit = ArmaGarch(
            mean_order=(1, 1), var_model=GARCH, var_order=(1, 1),
            residual_dist=normal,
        ).fit(y_series, maxiter=40)
        assert np.isfinite(float(fit.loglikelihood()))
        assert np.isfinite(float(fit.aic()))
        assert np.isfinite(float(fit.bic()))
        # Recompute-on-series path (does not touch the cached bundle).
        assert np.isfinite(float(fit.loglikelihood(y_series)))
