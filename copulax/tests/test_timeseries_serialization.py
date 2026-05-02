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
