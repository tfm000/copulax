"""Tests for save/load (serialization) of fitted distributions.

Round-trip tests for univariate, multivariate, elliptical copula, and
Archimedean copula distributions.  Verifies parameter recovery, logpdf
consistency, error handling, and metadata integrity.
"""

import json
import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import copulax
from copulax.univariate import normal
from copulax._src.univariate.normal import Normal
from copulax._src.univariate.student_t import StudentT
from copulax._src.univariate.gamma import Gamma as GammaClass
from copulax._src.univariate.lognormal import LogNormal
from copulax._src.univariate.ig import IG
from copulax._src.univariate.gig import GIG
from copulax._src.univariate.gh import GH
from copulax._src.multivariate.mvt_normal import MvtNormal
from copulax._src.multivariate.mvt_student_t import MvtStudentT
from copulax.copulas import (
    gaussian_copula, student_t_copula, clayton_copula,
    frank_copula, gumbel_copula, joe_copula, independence_copula,
)


@pytest.fixture(autouse=True, scope="module")
def _enable_x64():
    jax.config.update("jax_enable_x64", True)
    yield


# ---------------------------------------------------------------------------
# Parametrization configs
# ---------------------------------------------------------------------------

UNIVARIATE_CONFIGS = [
    (Normal,     {"mu": 1.5, "sigma": 2.3}),
    (StudentT,   {"nu": 5.0, "mu": -1.0, "sigma": 0.5}),
    (GammaClass, {"alpha": 3.0, "beta": 2.0}),
    (LogNormal,  {"mu": 0.5, "sigma": 0.8}),
    (IG,         {"alpha": 4.0, "beta": 2.0}),
    (GIG,        {"lamb": 1.0, "chi": 2.0, "psi": 3.0}),
    (GH,         {"lamb": 1.0, "chi": 2.0, "psi": 3.0,
                  "mu": 0.5, "sigma": 1.0, "gamma": 0.0}),
]
UNIVARIATE_IDS = [cls.__name__ for cls, _ in UNIVARIATE_CONFIGS]

# Subset of univariate configs used for logpdf-consistency, each paired with
# a test-point range appropriate to its support.
UNIVARIATE_LOGPDF_CONFIGS = [
    (Normal,     {"mu": 0.0, "sigma": 1.0}, (-3.0, 3.0)),
    (GammaClass, {"alpha": 3.0, "beta": 2.0}, (0.1, 5.0)),
    (LogNormal,  {"mu": 0.5, "sigma": 0.8}, (0.1, 5.0)),
    (IG,         {"alpha": 4.0, "beta": 2.0}, (0.1, 5.0)),
]
UNIVARIATE_LOGPDF_IDS = [cls.__name__ for cls, _, _ in UNIVARIATE_LOGPDF_CONFIGS]

MULTIVARIATE_CONFIGS = [
    (MvtNormal,   {"mu": jnp.array([[1.0], [2.0], [3.0]]),
                   "sigma": jnp.eye(3) * 2.0}),
    (MvtStudentT, {"nu": 5.0,
                   "mu": jnp.array([[1.0], [2.0]]),
                   "sigma": jnp.eye(2) * 2.0}),
]
MULTIVARIATE_IDS = [cls.__name__ for cls, _ in MULTIVARIATE_CONFIGS]

ELLIPTICAL_COPULAS = [gaussian_copula, student_t_copula]
ELLIPTICAL_IDS = [c.name for c in ELLIPTICAL_COPULAS]

ARCHIMEDEAN_COPULAS = [
    clayton_copula, frank_copula, gumbel_copula,
    joe_copula, independence_copula,
]
ARCHIMEDEAN_IDS = [c.name for c in ARCHIMEDEAN_COPULAS]


# ---------------------------------------------------------------------------
# Univariate round-trip
# ---------------------------------------------------------------------------

class TestUnivariateRoundTrip:
    """Save/load round-trip for univariate distributions."""

    @pytest.mark.parametrize("cls,kwargs", UNIVARIATE_CONFIGS,
                             ids=UNIVARIATE_IDS)
    def test_round_trip(self, tmp_path, cls, kwargs):
        """Distribution survives save/load round-trip with params intact."""
        fitted = cls(name="test", **kwargs)
        path = tmp_path / f"{cls.__name__}.cpx"
        fitted.save(str(path))

        loaded = copulax.load(str(path))
        assert loaded == fitted
        assert loaded.name == "test"

    @pytest.mark.parametrize("cls,kwargs,x_range", UNIVARIATE_LOGPDF_CONFIGS,
                             ids=UNIVARIATE_LOGPDF_IDS)
    def test_logpdf_consistency(self, tmp_path, cls, kwargs, x_range):
        """Loaded distribution produces identical logpdf output."""
        fitted = cls(name="test", **kwargs)
        path = tmp_path / f"{cls.__name__}_lp.cpx"
        fitted.save(str(path))
        loaded = copulax.load(str(path))

        x = jnp.linspace(x_range[0], x_range[1], 20)
        np.testing.assert_array_equal(
            np.asarray(fitted.logpdf(x)), np.asarray(loaded.logpdf(x))
        )

    def test_rvs_runs(self, tmp_path):
        """Loaded distribution can generate samples without error."""
        fitted = Normal(name="test", mu=0.0, sigma=1.0)
        path = tmp_path / "normal_rvs.cpx"
        fitted.save(str(path))
        loaded = copulax.load(str(path))

        key = jax.random.PRNGKey(0)
        samples = loaded.rvs(size=10, key=key)
        assert samples.shape == (10,)
        assert np.all(np.isfinite(np.asarray(samples)))


# ---------------------------------------------------------------------------
# Multivariate round-trip
# ---------------------------------------------------------------------------

class TestMultivariateRoundTrip:
    """Save/load round-trip for multivariate distributions."""

    @pytest.mark.parametrize("cls,kwargs", MULTIVARIATE_CONFIGS,
                             ids=MULTIVARIATE_IDS)
    def test_round_trip(self, tmp_path, cls, kwargs):
        """Multivariate distribution survives save/load round-trip."""
        fitted = cls(name="test", **kwargs)
        path = tmp_path / f"{cls.__name__}.cpx"
        fitted.save(str(path))

        loaded = copulax.load(str(path))
        assert loaded == fitted
        assert loaded.name == "test"

    def test_logpdf_consistency(self, tmp_path):
        """Loaded multivariate produces identical logpdf."""
        mu = jnp.array([[0.0], [0.0]])
        sigma = jnp.eye(2)
        fitted = MvtNormal(name="test", mu=mu, sigma=sigma)
        path = tmp_path / "mvn_logpdf.cpx"
        fitted.save(str(path))
        loaded = copulax.load(str(path))

        x = jnp.array([[0.5, -0.5], [1.0, 1.0], [-1.0, 0.0]])
        np.testing.assert_array_equal(
            np.asarray(fitted.logpdf(x)), np.asarray(loaded.logpdf(x))
        )


# ---------------------------------------------------------------------------
# Elliptical copula round-trip
# ---------------------------------------------------------------------------

class TestEllipticalCopulaRoundTrip:
    """Save/load round-trip for elliptical copulas."""

    @pytest.mark.parametrize("copula", ELLIPTICAL_COPULAS, ids=ELLIPTICAL_IDS)
    def test_round_trip(self, tmp_path, copula):
        """Elliptical copula survives save/load round-trip."""
        params = copula.example_params(dim=3)
        fitted = copula._fitted_instance(params, name="test")
        path = tmp_path / f"{copula.name}.cpx"
        fitted.save(str(path))

        loaded = copulax.load(str(path))
        assert loaded == fitted
        assert loaded.name == "test"

    @pytest.mark.parametrize("copula", ELLIPTICAL_COPULAS, ids=ELLIPTICAL_IDS)
    def test_copula_logpdf_consistency(self, tmp_path, copula):
        """Loaded elliptical copula produces identical copula_logpdf."""
        params = copula.example_params(dim=3)
        fitted = copula._fitted_instance(params, name="test")
        path = tmp_path / f"{copula.name}_lp.cpx"
        fitted.save(str(path))
        loaded = copulax.load(str(path))

        u = jnp.array(np.random.uniform(0.01, 0.99, size=(10, 3)))
        np.testing.assert_array_equal(
            np.asarray(fitted.copula_logpdf(u)),
            np.asarray(loaded.copula_logpdf(u)),
        )


# ---------------------------------------------------------------------------
# Archimedean copula round-trip
# ---------------------------------------------------------------------------

class TestArchimedeanCopulaRoundTrip:
    """Save/load round-trip for Archimedean copulas."""

    @pytest.mark.parametrize("copula", ARCHIMEDEAN_COPULAS,
                             ids=ARCHIMEDEAN_IDS)
    def test_round_trip(self, tmp_path, copula):
        """Archimedean copula survives save/load round-trip."""
        params = copula.example_params(dim=3)
        fitted = copula._fitted_instance(params, name="test")
        path = tmp_path / f"{copula.name}.cpx"
        fitted.save(str(path))

        loaded = copulax.load(str(path))
        assert loaded == fitted
        assert loaded.name == "test"

    @pytest.mark.parametrize("copula", ARCHIMEDEAN_COPULAS,
                             ids=ARCHIMEDEAN_IDS)
    def test_copula_logpdf_consistency(self, tmp_path, copula):
        """Loaded Archimedean copula produces identical copula_logpdf."""
        params = copula.example_params(dim=3)
        fitted = copula._fitted_instance(params, name="test")
        path = tmp_path / f"{copula.name}_lp.cpx"
        fitted.save(str(path))
        loaded = copulax.load(str(path))

        u = jnp.array(np.random.uniform(0.01, 0.99, size=(10, 3)))
        np.testing.assert_array_equal(
            np.asarray(fitted.copula_logpdf(u)),
            np.asarray(loaded.copula_logpdf(u)),
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Error cases for save/load."""

    def test_save_unfitted_raises(self, tmp_path):
        """Saving an unfitted distribution raises ValueError."""
        path = tmp_path / "unfitted.cpx"
        with pytest.raises(ValueError, match="unfitted"):
            normal.save(str(path))

    def test_load_nonexistent_raises(self):
        """Loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            copulax.load("/nonexistent/path/model.cpx")


# ---------------------------------------------------------------------------
# File format and API details
# ---------------------------------------------------------------------------

class TestFileFormat:
    """Verify file format details."""

    def test_auto_append_cpx_extension(self, tmp_path):
        """The .cpx extension is auto-appended when missing."""
        fitted = Normal(name="test", mu=0.0, sigma=1.0)
        path = tmp_path / "model"
        fitted.save(str(path))
        assert (tmp_path / "model.cpx").exists()

        loaded = copulax.load(str(tmp_path / "model.cpx"))
        assert loaded == fitted

    def test_metadata_is_valid_json(self, tmp_path):
        """The metadata.json inside the .cpx is valid JSON."""
        fitted = Normal(name="test", mu=0.0, sigma=1.0)
        path = tmp_path / "meta_check.cpx"
        fitted.save(str(path))

        with zipfile.ZipFile(path, "r") as zf:
            metadata = json.loads(zf.read("metadata.json"))
            assert metadata["dist_family"] == "univariate"
            assert metadata["dist_class"] == "Normal"
            assert metadata["dist_name"] == "test"
            assert metadata["dist_dtype"] == "continuous"
            assert "mu" in metadata["params"]
            assert "sigma" in metadata["params"]

    def test_name_override_on_load(self, tmp_path):
        """The name kwarg on load() overrides the saved name."""
        fitted = Normal(name="original", mu=1.0, sigma=2.0)
        path = tmp_path / "name_override.cpx"
        fitted.save(str(path))

        loaded = copulax.load(str(path), name="renamed")
        assert loaded.name == "renamed"
        np.testing.assert_array_equal(
            np.asarray(loaded.mu), np.asarray(fitted.mu)
        )
        np.testing.assert_array_equal(
            np.asarray(loaded.sigma), np.asarray(fitted.sigma)
        )

    def test_copula_metadata_has_expected_keys(self, tmp_path):
        """Copula .cpx metadata has copula-specific keys."""
        params = gaussian_copula.example_params(dim=2)
        fitted = gaussian_copula._fitted_instance(params, name="test")
        path = tmp_path / "cop_meta.cpx"
        fitted.save(str(path))

        with zipfile.ZipFile(path, "r") as zf:
            metadata = json.loads(zf.read("metadata.json"))
            assert metadata["dist_family"] == "copula"
            assert metadata["copula_type"] == "elliptical"
            assert "copula_params" in metadata
            assert "marginals" in metadata
            assert len(metadata["marginals"]) == 2
