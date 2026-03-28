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
from copulax.univariate import normal, student_t
from copulax._src.univariate.normal import Normal
from copulax._src.univariate.student_t import StudentT
from copulax.multivariate import mvt_normal
from copulax._src.multivariate.mvt_normal import MvtNormal
from copulax.copulas import gaussian_copula, student_t_copula, clayton_copula


@pytest.fixture(autouse=True, scope="module")
def _enable_x64():
    jax.config.update("jax_enable_x64", True)
    yield


# ---------------------------------------------------------------------------
# Univariate round-trip
# ---------------------------------------------------------------------------

class TestUnivariateRoundTrip:
    """Save/load round-trip for univariate distributions."""

    def test_normal(self, tmp_path):
        """Normal (2 params) round-trips correctly."""
        fitted = Normal(name="my_normal", mu=1.5, sigma=2.3)
        path = tmp_path / "normal.cpx"
        fitted.save(str(path))

        loaded = copulax.load(str(path))
        assert loaded == fitted
        assert loaded.name == "my_normal"

    def test_student_t(self, tmp_path):
        """Student-T (3 params) round-trips correctly."""
        fitted = StudentT(name="my_t", nu=5.0, mu=-1.0, sigma=0.5)
        path = tmp_path / "student_t.cpx"
        fitted.save(str(path))

        loaded = copulax.load(str(path))
        assert loaded == fitted

    def test_logpdf_consistency(self, tmp_path):
        """Loaded distribution produces identical logpdf output."""
        fitted = Normal(name="test", mu=0.0, sigma=1.0)
        path = tmp_path / "normal_logpdf.cpx"
        fitted.save(str(path))
        loaded = copulax.load(str(path))

        x = jnp.linspace(-3.0, 3.0, 20)
        original_lp = fitted.logpdf(x)
        loaded_lp = loaded.logpdf(x)
        np.testing.assert_array_equal(
            np.asarray(original_lp), np.asarray(loaded_lp)
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

    def test_mvt_normal(self, tmp_path):
        """Multivariate Normal round-trips correctly."""
        mu = jnp.array([[1.0], [2.0], [3.0]])
        sigma = jnp.eye(3) * 2.0
        fitted = MvtNormal(name="my_mvn", mu=mu, sigma=sigma)
        path = tmp_path / "mvn.cpx"
        fitted.save(str(path))

        loaded = copulax.load(str(path))
        assert loaded == fitted
        assert loaded.name == "my_mvn"

    def test_logpdf_consistency(self, tmp_path):
        """Loaded multivariate produces identical logpdf."""
        mu = jnp.array([[0.0], [0.0]])
        sigma = jnp.eye(2)
        fitted = MvtNormal(name="test", mu=mu, sigma=sigma)
        path = tmp_path / "mvn_logpdf.cpx"
        fitted.save(str(path))
        loaded = copulax.load(str(path))

        x = jnp.array([[0.5, -0.5], [1.0, 1.0], [-1.0, 0.0]])
        original_lp = fitted.logpdf(x)
        loaded_lp = loaded.logpdf(x)
        np.testing.assert_array_equal(
            np.asarray(original_lp), np.asarray(loaded_lp)
        )


# ---------------------------------------------------------------------------
# Copula round-trip
# ---------------------------------------------------------------------------

class TestEllipticalCopulaRoundTrip:
    """Save/load round-trip for elliptical copulas."""

    def test_gaussian_copula(self, tmp_path):
        """Gaussian copula round-trips correctly."""
        params = gaussian_copula.example_params(dim=3)
        fitted = gaussian_copula._fitted_instance(params, name="my_gauss")
        path = tmp_path / "gauss_cop.cpx"
        fitted.save(str(path))

        loaded = copulax.load(str(path))
        assert loaded == fitted
        assert loaded.name == "my_gauss"

    def test_student_t_copula(self, tmp_path):
        """Student-T copula round-trips correctly."""
        params = student_t_copula.example_params(dim=3)
        fitted = student_t_copula._fitted_instance(params, name="my_t_cop")
        path = tmp_path / "t_cop.cpx"
        fitted.save(str(path))

        loaded = copulax.load(str(path))
        assert loaded == fitted

    def test_copula_logpdf_consistency(self, tmp_path):
        """Loaded copula produces identical copula_logpdf."""
        params = gaussian_copula.example_params(dim=3)
        fitted = gaussian_copula._fitted_instance(params, name="test")
        path = tmp_path / "gauss_logpdf.cpx"
        fitted.save(str(path))
        loaded = copulax.load(str(path))

        u = jnp.array(np.random.uniform(0.01, 0.99, size=(10, 3)))
        original_lp = fitted.copula_logpdf(u)
        loaded_lp = loaded.copula_logpdf(u)
        np.testing.assert_array_equal(
            np.asarray(original_lp), np.asarray(loaded_lp)
        )


class TestArchimedeanCopulaRoundTrip:
    """Save/load round-trip for Archimedean copulas."""

    def test_clayton_copula(self, tmp_path):
        """Clayton copula round-trips correctly."""
        params = clayton_copula.example_params(dim=3)
        fitted = clayton_copula._fitted_instance(params, name="my_clayton")
        path = tmp_path / "clayton.cpx"
        fitted.save(str(path))

        loaded = copulax.load(str(path))
        assert loaded == fitted
        assert loaded.name == "my_clayton"

    def test_copula_logpdf_consistency(self, tmp_path):
        """Loaded Archimedean copula produces identical copula_logpdf."""
        params = clayton_copula.example_params(dim=3)
        fitted = clayton_copula._fitted_instance(params, name="test")
        path = tmp_path / "clayton_logpdf.cpx"
        fitted.save(str(path))
        loaded = copulax.load(str(path))

        u = jnp.array(np.random.uniform(0.01, 0.99, size=(10, 3)))
        original_lp = fitted.copula_logpdf(u)
        loaded_lp = loaded.copula_logpdf(u)
        np.testing.assert_array_equal(
            np.asarray(original_lp), np.asarray(loaded_lp)
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
        # Parameters should still match
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
