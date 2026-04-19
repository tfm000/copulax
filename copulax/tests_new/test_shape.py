"""Rigorous tests for copulax/_src/multivariate/_shape.py.

Cross-validates correlation and covariance methods against scipy/numpy,
tests edge cases, round-trip consistency, and error handling.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy import stats as sp_stats

from copulax.multivariate import corr, cov, random_correlation, random_covariance
from copulax._src.multivariate._shape import _corr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def correlated_data():
    """Correlated multivariate normal data (n=200, d=4) from a guaranteed-PSD R.

    R is built as a normalised random Gram matrix (F F^T + 0.5 I, rescaled to
    unit diagonal) so that np.random.multivariate_normal samples from exactly R
    and not a silently-nearest-PSD approximation. Off-diagonals span
    (-0.72, 0.82) with mixed signs — non-trivial correlation structure.
    """
    rng = np.random.default_rng(12345)
    F = rng.standard_normal((4, 4))
    C_pd = F @ F.T + 0.5 * np.eye(4)
    s = 1.0 / np.sqrt(np.diag(C_pd))
    R = C_pd * np.outer(s, s)
    sigma = np.diag([2.0, 1.5, 3.0, 0.5])
    C = sigma @ R @ sigma
    return rng.multivariate_normal(np.zeros(4), C, size=200)


@pytest.fixture(scope="module")
def uncorrelated_data():
    """Uncorrelated standard normal data (n=200, d=3)."""
    np.random.seed(99999)
    return np.random.normal(size=(200, 3))


CORRELATION_METHODS = [
    "pearson", "spearman", "kendall", "pp_kendall",
    "rm_pearson", "rm_spearman", "rm_kendall", "rm_pp_kendall",
    "laloux_pearson", "laloux_spearman", "laloux_kendall", "laloux_pp_kendall",
]


# ---------------------------------------------------------------------------
# Scipy cross-validation
# ---------------------------------------------------------------------------

class TestCorrVsScipy:
    """Cross-validate correlation methods against scipy/numpy."""

    def test_pearson_vs_numpy(self, correlated_data):
        copulax_R = np.array(corr(correlated_data, method="pearson"))
        numpy_R = np.corrcoef(correlated_data, rowvar=False)
        np.testing.assert_allclose(copulax_R, numpy_R, atol=1e-14)

    def test_spearman_vs_scipy(self, correlated_data):
        copulax_R = np.array(corr(correlated_data, method="spearman"))
        scipy_R = sp_stats.spearmanr(correlated_data).statistic
        np.testing.assert_allclose(copulax_R, scipy_R, atol=1e-14)

    def test_kendall_vs_scipy(self, correlated_data):
        copulax_R = np.array(corr(correlated_data, method="kendall"))
        d = correlated_data.shape[1]
        scipy_R = np.eye(d)
        for i in range(d):
            for j in range(i + 1, d):
                tau, _ = sp_stats.kendalltau(
                    correlated_data[:, i], correlated_data[:, j]
                )
                scipy_R[i, j] = tau
                scipy_R[j, i] = tau
        np.testing.assert_allclose(copulax_R, scipy_R, atol=1e-14)


def _assert_valid_correlation(R, name):
    """Assert R is symmetric, unit-diagonal, PSD. Used by the 3 property tests
    that need this structural contract."""
    np.testing.assert_allclose(R, R.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-12)
    min_eig = float(np.linalg.eigvalsh(R).min())
    assert min_eig > -1e-10, (
        f"{name} produced non-PSD matrix, min eig = {min_eig}"
    )


class TestCorrVsReference:
    """Property-based tests for the 9 methods not covered by TestCorrVsScipy.

    Closes Gap 4 from .claude/test_audit/04_coverage_gaps_old_vs_new.md.

    Strategy:
      - pp_kendall: cross-check against scipy kendalltau + elliptical identity
        rho = sin(pi tau / 2). Independent implementation (scipy), so valid.
      - rm_*, laloux_*: no reimplementation of the algorithm. Test mathematical
        properties the output must satisfy by definition.
    """

    # -------------------- pp_kendall vs scipy kendalltau --------------------

    def test_pp_kendall_matches_scipy_identity(self, correlated_data):
        """pp_kendall must equal sin(pi tau / 2) where tau is scipy's Kendall tau."""
        d = correlated_data.shape[1]
        tau = np.eye(d)
        for i in range(d):
            for j in range(i + 1, d):
                t, _ = sp_stats.kendalltau(
                    correlated_data[:, i], correlated_data[:, j]
                )
                tau[i, j] = tau[j, i] = t
        expected = np.sin(0.5 * np.pi * tau)
        got = np.array(corr(correlated_data, method="pp_kendall"))
        np.testing.assert_allclose(got, expected, atol=1e-14)

    # -------------------- rm_*: structural + behavioural --------------------

    @pytest.mark.parametrize(
        "base", ["pearson", "spearman", "kendall", "pp_kendall"]
    )
    def test_rm_output_is_valid_correlation(self, correlated_data, base):
        R = np.array(corr(correlated_data, method=f"rm_{base}"))
        _assert_valid_correlation(R, f"rm_{base}")

    @pytest.mark.parametrize(
        "base", ["pearson", "spearman", "kendall", "pp_kendall"]
    )
    def test_rm_on_already_psd_is_near_noop(self, correlated_data, base):
        """If the base estimator is already PSD, RM should leave it nearly unchanged.

        n=200, d=4 is well-conditioned for all 4 base estimators. RM's only
        work on such input is the diagonal rescale, which should be a near no-op.
        """
        base_R = np.array(corr(correlated_data, method=base))
        assert np.linalg.eigvalsh(base_R).min() > 0, (
            f"test precondition: base={base} must be PSD on the fixture"
        )
        rm_R = np.array(corr(correlated_data, method=f"rm_{base}"))
        np.testing.assert_allclose(rm_R, base_R, atol=1e-10)

    def test_rm_clamps_negative_eigenvalues(self):
        """Given a constructed non-PSD matrix, RM must produce a PSD output."""
        rng = np.random.default_rng(0)
        Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        eigvals_bad = np.array([2.5, 0.8, 0.3, -0.1])
        A = Q @ np.diag(eigvals_bad) @ Q.T
        # Rescale to unit diagonal so it looks like a plausible raw correlation.
        s = 1.0 / np.sqrt(np.diag(A))
        A = A * np.outer(s, s)
        assert np.linalg.eigvalsh(A).min() < 0, (
            "test precondition: A must be non-PSD"
        )

        # Feed directly to the private _rm helper; public corr() starts from data.
        cleaned = np.array(_corr._rm(jnp.asarray(A), delta=1e-5))
        _assert_valid_correlation(cleaned, "rm(constructed non-PSD)")

    # -------------------- laloux_*: structural + MP cutoff --------------------

    @pytest.mark.parametrize(
        "base", ["pearson", "spearman", "kendall", "pp_kendall"]
    )
    def test_laloux_output_is_valid_correlation(self, correlated_data, base):
        R = np.array(corr(correlated_data, method=f"laloux_{base}"))
        _assert_valid_correlation(R, f"laloux_{base}")

    def test_laloux_collapses_bulk_vs_rm_baseline(self):
        """Laloux must compress bulk eigenvalue dispersion strictly more than RM.

        Compares Laloux's output to RM's output on the same input matrix.

        Mathematical argument:
        - Input A is constructed PSD (eigenvalues all positive) and unit-diagonal,
          so RM's eigenvalue-clamping and rescale are both near no-ops — RM
          output has essentially the same eigenvalues as A.
        - Laloux replaces the sub-cutoff eigenvalues with their arithmetic mean
          BEFORE the rescale step, collapsing their pre-rescale std to exactly 0.
          The final rescale can re-introduce some dispersion, but only through
          eigenvector-basis-dependent distortions — it can never re-introduce
          as much dispersion as was originally in the input.
        - Therefore std(bulk eigenvalues of laloux_out) < std(bulk eigenvalues
          of rm_out). Strict inequality, no tolerance tuning needed.

        Also verifies the signal eigenvalue is preserved — if the MP cutoff
        direction were wrong (n/d swapped with d/n), the cutoff at n=80, d=4
        would balloon to ~21, every eigenvalue would be classified as bulk
        including the 3.1 signal, and the signal assertion would fail loudly.
        """
        rng = np.random.default_rng(123)
        Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        # n=80, d=4 -> MP cutoff = (1 + sqrt(4/80))**2 ~ 1.49
        # One eigenvalue above the cutoff (signal), three clearly below (bulk).
        eigvals_true = np.array([3.1, 0.7, 0.3, 0.2])
        A_raw = Q @ np.diag(eigvals_true) @ Q.T
        s = 1.0 / np.sqrt(np.diag(A_raw))
        A = jnp.asarray(A_raw * np.outer(s, s))

        # Only the shape of x_stub matters for the MP cutoff ratio Q_mp = n/d.
        x_stub = jnp.asarray(rng.standard_normal((80, 4)))

        rm_out = np.array(_corr._rm(A, delta=1e-5))
        laloux_out = np.array(_corr._laloux(x_stub, A, delta=1e-5))

        _assert_valid_correlation(rm_out, "rm_baseline")
        _assert_valid_correlation(laloux_out, "laloux")

        rm_eigs = np.sort(np.linalg.eigvalsh(rm_out))
        laloux_eigs = np.sort(np.linalg.eigvalsh(laloux_out))

        # Signal preservation: rank-1 eigenvalue should still be clearly above
        # the MP cutoff. Guards the cutoff-direction bug.
        assert laloux_eigs[-1] > 2.0, (
            f"Signal eigenvalue lost: expected > 2.0, got {laloux_eigs[-1]}. "
            f"Laloux eigs: {laloux_eigs}"
        )

        # Core collapse property: bulk dispersion is strictly smaller after
        # Laloux than after RM (which preserves input eigenvalues). The d-k=3
        # smallest eigenvalues form the bulk for this input.
        rm_bulk_std = float(np.std(rm_eigs[:3]))
        laloux_bulk_std = float(np.std(laloux_eigs[:3]))
        assert laloux_bulk_std < rm_bulk_std, (
            f"Laloux did not compress bulk vs RM: "
            f"std_rm_bulk={rm_bulk_std:.4f}, std_laloux_bulk={laloux_bulk_std:.4f}. "
            f"RM eigs: {rm_eigs}, Laloux eigs: {laloux_eigs}"
        )


class TestCovVsScipy:
    """Cross-validate covariance against numpy."""

    def test_pearson_cov_vs_numpy(self, correlated_data):
        """Pearson covariance should match numpy.cov (ddof=1)."""
        copulax_C = np.array(cov(correlated_data, method="pearson"))
        numpy_C = np.cov(correlated_data, rowvar=False, ddof=1)
        np.testing.assert_allclose(copulax_C, numpy_C, atol=1e-12)

    def test_cov_diagonal_is_sample_variance(self, correlated_data):
        """Diagonal of Pearson covariance should equal sample variances (ddof=1)."""
        copulax_C = np.array(cov(correlated_data, method="pearson"))
        expected_vars = np.var(correlated_data, axis=0, ddof=1)
        np.testing.assert_allclose(np.diag(copulax_C), expected_vars, atol=1e-12)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test that invalid inputs produce clear errors."""

    def test_invalid_method_raises_valueerror(self, correlated_data):
        with pytest.raises(ValueError, match="Unknown correlation method"):
            corr(correlated_data, method="nonexistent")

    def test_invalid_method_in_cov_raises(self, correlated_data):
        """cov() delegates to corr(), so invalid method should also raise."""
        with pytest.raises(ValueError, match="Unknown correlation method"):
            cov(correlated_data, method="bogus_method")


# ---------------------------------------------------------------------------
# Round-trip and conversion consistency
# ---------------------------------------------------------------------------

class TestRoundTrips:
    """Verify cov <-> corr conversion consistency."""

    def test_cov_to_corr_to_cov(self, correlated_data):
        """cov -> _corr_from_cov -> _cov_from_vars should round-trip."""
        C = jnp.array(np.cov(correlated_data, rowvar=False))
        R = _corr._corr_from_cov(C)
        vars_orig = jnp.diag(C)
        C_reconstructed = _corr._cov_from_vars(vars_orig, R)
        np.testing.assert_allclose(
            np.array(C_reconstructed), np.array(C), atol=1e-12
        )

    def test_corr_from_cov_unit_diagonal(self, correlated_data):
        """_corr_from_cov should produce unit diagonal."""
        C = jnp.array(np.cov(correlated_data, rowvar=False))
        R = _corr._corr_from_cov(C)
        np.testing.assert_allclose(
            np.array(jnp.diag(R)), np.ones(C.shape[0]), atol=1e-14
        )

    def test_corr_from_cov_preserves_psd(self):
        """_corr_from_cov on a PSD matrix should produce a PSD result."""
        key = random.PRNGKey(7)
        W = random.normal(key, (5, 5))
        C = W @ W.T + 0.1 * jnp.eye(5)
        R = _corr._corr_from_cov(C)
        eigvals = np.array(jnp.linalg.eigvalsh(R))
        assert np.all(eigvals >= -1e-12), f"min eigenvalue: {eigvals.min()}"


# ---------------------------------------------------------------------------
# random_correlation and random_covariance
# ---------------------------------------------------------------------------

class TestRandomMatrices:
    """Test random matrix generation."""

    @pytest.mark.parametrize("n", [2, 3, 5, 10, 50, 100])
    def test_random_correlation_properties(self, n):
        R = np.array(random_correlation(n, key=random.PRNGKey(n)))
        assert R.shape == (n, n)
        _assert_valid_correlation(R, f"random_correlation(n={n})")
        assert np.all(np.abs(R) <= 1 + 1e-10), "entries out of [-1, 1]"

    @pytest.mark.parametrize("n", [2, 5, 10, 100])
    def test_random_covariance_diagonal_matches_input(self, n):
        """Diagonal of random_covariance should equal the input variances."""
        input_vars = jnp.array(np.random.uniform(0.5, 5.0, size=(n,)))
        rcov = np.array(random_covariance(input_vars, key=random.PRNGKey(n)))
        np.testing.assert_allclose(
            np.diag(rcov), np.array(input_vars), atol=1e-12
        )

    @pytest.mark.parametrize("n", [2, 5, 10, 100])
    def test_random_covariance_is_psd(self, n):
        input_vars = jnp.array(np.random.uniform(0.5, 5.0, size=(n,)))
        rcov = np.array(random_covariance(input_vars, key=random.PRNGKey(n)))
        eigvals = np.linalg.eigvalsh(rcov)
        assert eigvals.min() > -1e-10, f"not PSD: min eig = {eigvals.min()}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: minimal dimensions, rank-deficient, etc."""

    @pytest.mark.parametrize("method", CORRELATION_METHODS)
    def test_d2_minimal_dimension(self, method):
        """All methods should work for d=2."""
        x = np.array(random.normal(random.PRNGKey(0), (50, 2)))
        R = np.array(corr(x, method=method))
        assert R.shape == (2, 2)
        _assert_valid_correlation(R, f"{method} (d=2)")
        assert np.all(np.abs(R) <= 1 + 1e-10)

    @pytest.mark.parametrize("method", ["pearson", "rm_pearson", "laloux_pearson"])
    def test_n_less_than_d(self, method):
        """Should handle rank-deficient data (n < d) without crashing."""
        x = np.array(random.normal(random.PRNGKey(42), (5, 10)))
        R = np.array(corr(x, method=method))
        assert R.shape == (10, 10)
        _assert_valid_correlation(R, f"{method} (n<d)")
        assert np.all(np.abs(R) <= 1 + 1e-10)

    def test_uncorrelated_data_off_diagonals_near_zero(self, uncorrelated_data):
        """Pearson of uncorrelated data should have off-diagonals near 0."""
        R = np.array(corr(uncorrelated_data, method="pearson"))
        off_diag = R[~np.eye(R.shape[0], dtype=bool)]
        assert np.max(np.abs(off_diag)) < 0.25, (
            f"Off-diag too large for uncorrelated data: {np.max(np.abs(off_diag)):.4f}"
        )

    @pytest.mark.parametrize("method", CORRELATION_METHODS)
    def test_uncorrelated_data_structural(self, uncorrelated_data, method):
        """All 12 methods must produce a valid correlation matrix on uncorrelated
        data. Guards a second data regime: the base estimators return R ~ I,
        which exercises the PSD/rescale paths of RM and Laloux differently than
        the correlated-data fixture (where there's real signal structure)."""
        R = np.array(corr(uncorrelated_data, method=method))
        _assert_valid_correlation(R, method)
        assert np.all(np.abs(R) <= 1 + 1e-10), "entries out of [-1, 1]"

    @pytest.mark.parametrize("method", CORRELATION_METHODS)
    def test_corr_output_shape(self, correlated_data, method):
        """Output shape should be (d, d)."""
        R = corr(correlated_data, method=method)
        d = correlated_data.shape[1]
        assert R.shape == (d, d)

    @pytest.mark.parametrize("method", CORRELATION_METHODS)
    def test_cov_output_shape(self, correlated_data, method):
        """Output shape should be (d, d)."""
        C = cov(correlated_data, method=method)
        d = correlated_data.shape[1]
        assert C.shape == (d, d)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

class TestJIT:
    """Verify JIT compilation works for all public functions."""

    @pytest.mark.parametrize("method", CORRELATION_METHODS)
    def test_corr_jit(self, correlated_data, method):
        jit_corr = jax.jit(corr, static_argnames=("method",))
        R = np.array(jit_corr(correlated_data, method=method))
        R_eager = np.array(corr(correlated_data, method=method))
        np.testing.assert_allclose(R, R_eager, atol=1e-14)

    @pytest.mark.parametrize("method", CORRELATION_METHODS)
    def test_cov_jit(self, correlated_data, method):
        jit_cov = jax.jit(cov, static_argnames=("method",))
        C = np.array(jit_cov(correlated_data, method=method))
        C_eager = np.array(cov(correlated_data, method=method))
        np.testing.assert_allclose(C, C_eager, atol=1e-14)

    def test_random_correlation_jit(self):
        jit_rc = jax.jit(random_correlation, static_argnames=("size",))
        R = np.array(jit_rc(5, key=random.PRNGKey(0)))
        R_eager = np.array(random_correlation(5, key=random.PRNGKey(0)))
        np.testing.assert_allclose(R, R_eager, atol=1e-14)

    def test_random_covariance_jit(self):
        input_vars = jnp.array([0.5, 1.0, 2.0, 3.0, 4.5])
        jit_rcov = jax.jit(random_covariance)
        C = np.array(jit_rcov(input_vars, key=random.PRNGKey(0)))
        C_eager = np.array(random_covariance(input_vars, key=random.PRNGKey(0)))
        np.testing.assert_allclose(C, C_eager, atol=1e-14)
