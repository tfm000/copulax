"""Tests for copulax.preprocessing.DataScaler.

Covers the plan verification checklist end-to-end:

* round-trip identity for 1D / 2D / 3D input across all methods
* method-specific output statistics
* new-data stability and shape handling
* zero-variance safety
* JIT and autograd compatibility
* PyTree serialisation round-trip
* offset_only / scale_only flags
* pre_fns / post_fns forward/inverse function pairs
* integration smoke test with copulax.univariate.Normal
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from copulax._src.univariate.normal import Normal
from copulax.preprocessing import DataScaler


METHODS = ("zscore", "minmax", "robust", "maxabs")


def _make_data(rng, shape, positive: bool = False):
    """Draw reproducible test data of the requested shape.

    If ``positive`` is True, the output is strictly positive (useful for
    log/exp round-trip tests).
    """
    x = rng.standard_normal(shape)
    if positive:
        x = np.abs(x) + 0.1
    return jnp.asarray(x, dtype=float)


# ---------------------------------------------------------------------------
# 1. Round-trip identity across methods and dimensionalities
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "shape",
    [(64,), (50, 3), (40, 4, 2)],
    ids=["1D", "2D", "3D"],
)
def test_roundtrip_identity(method, shape):
    rng = np.random.default_rng(0)
    x = _make_data(rng, shape)
    scaler, z = DataScaler(method).fit_transform(x)

    assert scaler.is_fitted
    assert scaler.offset.shape == x.shape[1:]
    assert scaler.scale.shape == x.shape[1:]

    recovered = scaler.inverse_transform(z)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(x), atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Method-specific output statistics
# ---------------------------------------------------------------------------
def test_zscore_output_moments():
    rng = np.random.default_rng(1)
    x = _make_data(rng, (200, 4))
    _, z = DataScaler("zscore").fit_transform(x)
    np.testing.assert_allclose(np.asarray(z.mean(axis=0)), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(z.std(axis=0)), 1.0, atol=1e-6)


def test_minmax_output_range():
    rng = np.random.default_rng(2)
    x = _make_data(rng, (200, 4))
    _, z = DataScaler("minmax").fit_transform(x)
    np.testing.assert_allclose(np.asarray(z.min(axis=0)), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(z.max(axis=0)), 1.0, atol=1e-6)


def test_robust_output_iqr():
    rng = np.random.default_rng(3)
    x = _make_data(rng, (400, 3))
    _, z = DataScaler("robust").fit_transform(x)
    z_np = np.asarray(z)
    np.testing.assert_allclose(np.median(z_np, axis=0), 0.0, atol=1e-6)
    iqr = np.quantile(z_np, 0.75, axis=0) - np.quantile(z_np, 0.25, axis=0)
    np.testing.assert_allclose(iqr, 1.0, atol=1e-6)


def test_maxabs_output_range():
    rng = np.random.default_rng(4)
    x = _make_data(rng, (200, 4))
    _, z = DataScaler("maxabs").fit_transform(x)
    np.testing.assert_allclose(np.abs(z).max(axis=0), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. New-data stability — fit on train, transform on test
# ---------------------------------------------------------------------------
def test_new_data_transform_preserves_shape():
    rng = np.random.default_rng(5)
    x_train = _make_data(rng, (100, 3))
    x_test = _make_data(rng, (17, 3))
    scaler = DataScaler("zscore").fit(x_train)
    z_test = scaler.transform(x_test)
    assert z_test.shape == x_test.shape
    # The test moments are *not* expected to be (0, 1) — only the train moments are.
    # But inverse_transform(z_test) must recover x_test exactly.
    np.testing.assert_allclose(
        np.asarray(scaler.inverse_transform(z_test)), np.asarray(x_test), atol=1e-6
    )


# ---------------------------------------------------------------------------
# 4. Zero-variance safety
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_zero_variance_column_is_safe(method):
    """A constant feature must pass through without NaN / Inf."""
    rng = np.random.default_rng(6)
    x = _make_data(rng, (50, 3))
    x = x.at[:, 1].set(7.0)  # column 1 is constant
    scaler, z = DataScaler(method).fit_transform(x)

    assert jnp.all(jnp.isfinite(z))
    # column 1 after scaling should be a single constant value across rows
    col = np.asarray(z[:, 1])
    np.testing.assert_allclose(col, col[0], atol=1e-12)
    # inverse recovers the original constant exactly
    recovered = np.asarray(scaler.inverse_transform(z))
    np.testing.assert_allclose(recovered[:, 1], 7.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. JIT compatibility
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_jit_fit_and_transform(method):
    rng = np.random.default_rng(7)
    x = _make_data(rng, (80, 2))
    template = DataScaler(method)

    @jax.jit
    def jit_fit_transform(arr):
        fitted = template.fit(arr)
        return fitted.offset, fitted.scale, fitted.transform(arr)

    offset_jit, scale_jit, z_jit = jit_fit_transform(x)
    fitted_plain = template.fit(x)

    np.testing.assert_allclose(np.asarray(offset_jit), np.asarray(fitted_plain.offset), atol=1e-10)
    np.testing.assert_allclose(np.asarray(scale_jit), np.asarray(fitted_plain.scale), atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(z_jit), np.asarray(fitted_plain.transform(x)), atol=1e-10
    )


# ---------------------------------------------------------------------------
# 6. Autograd through transform
# ---------------------------------------------------------------------------
def test_grad_of_transform():
    rng = np.random.default_rng(8)
    x = _make_data(rng, (40, 3))
    scaler = DataScaler("zscore").fit(x)

    def loss(arr):
        return scaler.transform(arr).sum()

    grad = jax.grad(loss)(x)
    # d/dx_{i,j} sum((x - offset) / scale) = 1 / scale_j
    expected_row = 1.0 / scaler.scale
    np.testing.assert_allclose(np.asarray(grad), np.broadcast_to(expected_row, x.shape), atol=1e-10)
    assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# 7. PyTree / equinox serialisation round-trip
# ---------------------------------------------------------------------------
def test_eqx_tree_round_trip(tmp_path):
    rng = np.random.default_rng(9)
    x = _make_data(rng, (30, 2))
    scaler = DataScaler("robust", q_low=0.1, q_high=0.9).fit(x)

    path = tmp_path / "scaler.eqx"
    eqx.tree_serialise_leaves(str(path), scaler)

    template = DataScaler("robust", q_low=0.1, q_high=0.9)
    # Deserialisation needs a template with matching treedef — pass dummy arrays
    # of the expected shape so it knows how many leaves to expect.
    seeded = DataScaler(
        "robust",
        q_low=0.1,
        q_high=0.9,
        offset=jnp.zeros(x.shape[1:]),
        scale=jnp.ones(x.shape[1:]),
    )
    restored = eqx.tree_deserialise_leaves(str(path), seeded)

    assert restored.method == "robust"
    np.testing.assert_allclose(np.asarray(restored.offset), np.asarray(scaler.offset), atol=1e-12)
    np.testing.assert_allclose(np.asarray(restored.scale), np.asarray(scaler.scale), atol=1e-12)


# ---------------------------------------------------------------------------
# 8. offset_only / scale_only flags
# ---------------------------------------------------------------------------
def test_offset_only_centres_without_rescaling():
    rng = np.random.default_rng(10)
    x = _make_data(rng, (200, 3))
    scaler, z = DataScaler("zscore", offset_only=True).fit_transform(x)

    np.testing.assert_allclose(np.asarray(z.mean(axis=0)), 0.0, atol=1e-6)
    # variance is preserved (only the mean is removed)
    np.testing.assert_allclose(
        np.asarray(z.std(axis=0)), np.asarray(x.std(axis=0)), atol=1e-6
    )
    np.testing.assert_allclose(np.asarray(scaler.scale), 1.0, atol=1e-12)


def test_scale_only_rescales_without_centring():
    rng = np.random.default_rng(11)
    x = _make_data(rng, (200, 3)) + 5.0  # shift to make the mean non-zero
    scaler, z = DataScaler("zscore", scale_only=True).fit_transform(x)

    np.testing.assert_allclose(np.asarray(z.std(axis=0)), 1.0, atol=1e-6)
    # mean is *not* zeroed — it's shifted by the un-subtracted offset divided by scale.
    assert np.all(np.abs(np.asarray(z.mean(axis=0))) > 1e-3)
    np.testing.assert_allclose(np.asarray(scaler.offset), 0.0, atol=1e-12)


def test_offset_only_and_scale_only_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        DataScaler("zscore", offset_only=True, scale_only=True)


# ---------------------------------------------------------------------------
# 9. pre_fns / post_fns
# ---------------------------------------------------------------------------
def test_pre_fns_log_exp_roundtrip():
    rng = np.random.default_rng(12)
    x = _make_data(rng, (100, 2), positive=True)
    scaler, z = DataScaler("zscore", pre_fns=(jnp.log, jnp.exp)).fit_transform(x)

    # z-score statistics hold on *log-data*, not raw data:
    np.testing.assert_allclose(np.asarray(z.mean(axis=0)), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(z.std(axis=0)), 1.0, atol=1e-6)

    # Round-trip recovers original positive data.
    np.testing.assert_allclose(
        np.asarray(scaler.inverse_transform(z)), np.asarray(x), atol=1e-5
    )


def test_pre_fns_missing_inverse_partial_roundtrip():
    """If the inverse half is None, inverse_transform silently skips that step."""
    rng = np.random.default_rng(13)
    x = _make_data(rng, (60, 2), positive=True)
    scaler, z = DataScaler("zscore", pre_fns=(jnp.log, None)).fit_transform(x)

    # inverse_transform returns log(x) reconstructed — not x.
    recovered = scaler.inverse_transform(z)
    np.testing.assert_allclose(
        np.asarray(recovered), np.asarray(jnp.log(x)), atol=1e-6
    )
    assert jnp.all(jnp.isfinite(recovered))


def test_post_fns_tanh_arctanh_roundtrip():
    rng = np.random.default_rng(14)
    x = _make_data(rng, (80, 2))
    scaler, z = DataScaler(
        "zscore", post_fns=(jnp.tanh, jnp.arctanh)
    ).fit_transform(x)
    assert jnp.all(jnp.abs(z) < 1.0)
    np.testing.assert_allclose(
        np.asarray(scaler.inverse_transform(z)), np.asarray(x), atol=1e-5
    )


def test_pre_fns_non_tuple_raises():
    with pytest.raises(ValueError, match="pre_fns must be a"):
        DataScaler("zscore", pre_fns=jnp.log)


def test_post_fns_non_callable_entry_raises():
    with pytest.raises(TypeError, match="must be callable or None"):
        DataScaler("zscore", post_fns=(jnp.tanh, "not-a-callable"))


def test_jit_with_pre_fns():
    rng = np.random.default_rng(15)
    x = _make_data(rng, (50, 2), positive=True)
    template = DataScaler("zscore", pre_fns=(jnp.log, jnp.exp))

    @jax.jit
    def jit_pipeline(arr):
        fitted = template.fit(arr)
        return fitted.transform(arr), fitted.inverse_transform(fitted.transform(arr))

    z, recovered = jit_pipeline(x)
    np.testing.assert_allclose(np.asarray(z.mean(axis=0)), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(x), atol=1e-5)


# ---------------------------------------------------------------------------
# 10. Integration smoke test — fit a Normal on scaled data, then invert
# ---------------------------------------------------------------------------
def test_integration_with_normal_distribution():
    """Fit Normal on z-scored data, sample, invert — moments match raw data."""
    rng = np.random.default_rng(16)
    raw = _make_data(rng, (2000,)) * 4.0 + 10.0  # mean ≈ 10, std ≈ 4
    scaler, z = DataScaler("zscore").fit_transform(raw)

    fitted_normal = Normal().fit(z)
    assert abs(float(fitted_normal.mu)) < 0.2
    assert abs(float(fitted_normal.sigma) - 1.0) < 0.2

    key = jax.random.PRNGKey(17)
    z_samples = jax.random.normal(key, (5000,)) * fitted_normal.sigma + fitted_normal.mu
    raw_samples = scaler.inverse_transform(z_samples)
    # Sample moments should be close to the raw data moments.
    assert abs(float(raw_samples.mean()) - float(raw.mean())) < 1.0
    assert abs(float(raw_samples.std()) - float(raw.std())) < 1.0


# ---------------------------------------------------------------------------
# 11. Miscellaneous / API guarantees
# ---------------------------------------------------------------------------
def test_unfitted_transform_raises():
    scaler = DataScaler("zscore")
    assert not scaler.is_fitted
    with pytest.raises(ValueError, match="not fitted"):
        scaler.transform(jnp.zeros((3, 2)))
    with pytest.raises(ValueError, match="not fitted"):
        scaler.inverse_transform(jnp.zeros((3, 2)))


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        DataScaler("garbage")


def test_quantile_bounds_validated():
    with pytest.raises(ValueError, match="q_low"):
        DataScaler("robust", q_low=0.9, q_high=0.1)
    with pytest.raises(ValueError, match="q_low"):
        DataScaler("robust", q_low=-0.1, q_high=0.5)
    with pytest.raises(ValueError, match="q_low"):
        DataScaler("robust", q_low=0.5, q_high=1.5)


def test_fit_is_pure_functional():
    """fit returns a new instance; the original is untouched."""
    rng = np.random.default_rng(18)
    x = _make_data(rng, (30, 2))
    template = DataScaler("zscore")
    assert not template.is_fitted
    fitted = template.fit(x)
    assert fitted.is_fitted
    assert not template.is_fitted
    assert fitted is not template


def test_repr_reflects_fitted_state():
    rng = np.random.default_rng(19)
    x = _make_data(rng, (10, 2))
    unfitted = DataScaler("minmax")
    fitted = unfitted.fit(x)
    assert "unfitted" in repr(unfitted)
    assert "fitted" in repr(fitted) and "unfitted" not in repr(fitted)
    # method name is shown in both
    assert "minmax" in repr(unfitted)
    assert "minmax" in repr(fitted)


# ---------------------------------------------------------------------------
# 12. Direct construction with pre-fitted offset/scale
# ---------------------------------------------------------------------------
def test_direct_construction_with_offset_and_scale():
    """Pass offset/scale at construction time; scaler should be fitted immediately."""
    offset = jnp.asarray([1.0, -2.0])
    scale = jnp.asarray([0.5, 3.0])
    scaler = DataScaler("zscore", offset=offset, scale=scale)
    assert scaler.is_fitted
    np.testing.assert_allclose(np.asarray(scaler.offset), np.asarray(offset), atol=1e-12)
    np.testing.assert_allclose(np.asarray(scaler.scale), np.asarray(scale), atol=1e-12)

    # transform uses those stored values exactly
    x = jnp.asarray([[1.0, -2.0], [2.0, 1.0]])
    z = scaler.transform(x)
    expected = (np.asarray(x) - np.asarray(offset)) / np.asarray(scale)
    np.testing.assert_allclose(np.asarray(z), expected, atol=1e-12)


def test_direct_construction_partial_is_unfitted():
    """Providing only one of offset/scale leaves the scaler unfitted."""
    scaler = DataScaler("zscore", offset=jnp.zeros(2))
    assert not scaler.is_fitted
    scaler = DataScaler("zscore", scale=jnp.ones(2))
    assert not scaler.is_fitted


# ---------------------------------------------------------------------------
# 13. offset_only / scale_only across all four methods
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_offset_only_all_methods(method):
    rng = np.random.default_rng(20)
    x = _make_data(rng, (150, 3))
    scaler, z = DataScaler(method, offset_only=True).fit_transform(x)
    # scale is forced to 1
    np.testing.assert_allclose(np.asarray(scaler.scale), 1.0, atol=1e-12)
    # the offset depends on method — verify it matches what fit() would store
    plain = DataScaler(method).fit(x)
    np.testing.assert_allclose(np.asarray(scaler.offset), np.asarray(plain.offset), atol=1e-6)
    # transform equals x - offset
    np.testing.assert_allclose(
        np.asarray(z), np.asarray(x) - np.asarray(scaler.offset), atol=1e-10
    )


@pytest.mark.parametrize("method", METHODS)
def test_scale_only_all_methods(method):
    rng = np.random.default_rng(21)
    x = _make_data(rng, (150, 3))
    scaler, z = DataScaler(method, scale_only=True).fit_transform(x)
    np.testing.assert_allclose(np.asarray(scaler.offset), 0.0, atol=1e-12)
    plain = DataScaler(method).fit(x)
    np.testing.assert_allclose(np.asarray(scaler.scale), np.asarray(plain.scale), atol=1e-6)
    np.testing.assert_allclose(np.asarray(z), np.asarray(x) / np.asarray(scaler.scale), atol=1e-10)


# ---------------------------------------------------------------------------
# 14. Combined pre_fns + post_fns
# ---------------------------------------------------------------------------
def test_pre_and_post_fns_combined():
    """Stack log-transform before and tanh-transform after the affine step."""
    rng = np.random.default_rng(22)
    x = _make_data(rng, (100, 2), positive=True)
    scaler, z = DataScaler(
        "zscore",
        pre_fns=(jnp.log, jnp.exp),
        post_fns=(jnp.tanh, jnp.arctanh),
    ).fit_transform(x)

    # output is squashed into (-1, 1)
    assert jnp.all(jnp.abs(z) < 1.0)

    # round-trip recovers x
    np.testing.assert_allclose(
        np.asarray(scaler.inverse_transform(z)), np.asarray(x), atol=1e-5
    )

    # verify the pipeline is actually: tanh((log(x) - mean_log_x) / std_log_x)
    log_x = np.log(np.asarray(x))
    expected = np.tanh((log_x - log_x.mean(axis=0)) / log_x.std(axis=0))
    np.testing.assert_allclose(np.asarray(z), expected, atol=1e-5)


def test_pre_fns_none_forward_with_inverse():
    """forward=None means no change at fit/transform; inverse still applies to inverse_transform."""
    rng = np.random.default_rng(23)
    x = _make_data(rng, (40, 2))
    scaler, z = DataScaler(
        "zscore", pre_fns=(None, lambda x_: x_ + 100.0)
    ).fit_transform(x)

    # transform is exactly the standard zscore (forward is None)
    plain_z = DataScaler("zscore").fit(x).transform(x)
    np.testing.assert_allclose(np.asarray(z), np.asarray(plain_z), atol=1e-10)

    # inverse_transform adds 100 at the end (the provided inverse half)
    recovered = scaler.inverse_transform(z)
    np.testing.assert_allclose(
        np.asarray(recovered), np.asarray(x) + 100.0, atol=1e-6
    )


def test_post_fns_none_forward_with_inverse():
    rng = np.random.default_rng(24)
    x = _make_data(rng, (40, 2))
    scaler, z = DataScaler(
        "zscore", post_fns=(None, lambda z_: z_ * 0.0)
    ).fit_transform(x)
    # transform is standard zscore (post forward is None)
    plain_z = DataScaler("zscore").fit(x).transform(x)
    np.testing.assert_allclose(np.asarray(z), np.asarray(plain_z), atol=1e-10)
    # inverse_transform starts by applying the provided post inverse: z * 0
    # Then affine inverse: 0 * scale + offset = offset.
    recovered = scaler.inverse_transform(z)
    np.testing.assert_allclose(
        np.asarray(recovered),
        np.broadcast_to(np.asarray(scaler.offset), recovered.shape),
        atol=1e-10,
    )


def test_both_tuples_none_halves_pass_through():
    rng = np.random.default_rng(25)
    x = _make_data(rng, (30, 2))
    scaler, z = DataScaler(
        "zscore", pre_fns=(None, None), post_fns=(None, None)
    ).fit_transform(x)
    plain_scaler, plain_z = DataScaler("zscore").fit_transform(x)
    np.testing.assert_allclose(np.asarray(z), np.asarray(plain_z), atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(scaler.inverse_transform(z)), np.asarray(x), atol=1e-6
    )


# ---------------------------------------------------------------------------
# 15. Custom q_low / q_high for robust
# ---------------------------------------------------------------------------
def test_robust_custom_quantiles_behaviour():
    rng = np.random.default_rng(26)
    x = _make_data(rng, (500, 2))
    ql, qh = 0.1, 0.9
    scaler, z = DataScaler("robust", q_low=ql, q_high=qh).fit_transform(x)

    z_np = np.asarray(z)
    # median of scaled data is ~0 (median is the stored offset)
    np.testing.assert_allclose(np.median(z_np, axis=0), 0.0, atol=1e-6)
    # the gap between the 10th and 90th quantiles of scaled data is 1.0
    gap = np.quantile(z_np, qh, axis=0) - np.quantile(z_np, ql, axis=0)
    np.testing.assert_allclose(gap, 1.0, atol=1e-6)

    # scaler stored scale matches the raw-data 10/90 gap
    raw_gap = np.quantile(np.asarray(x), qh, axis=0) - np.quantile(np.asarray(x), ql, axis=0)
    np.testing.assert_allclose(np.asarray(scaler.scale), raw_gap, atol=1e-6)


# ---------------------------------------------------------------------------
# 16. Non-JAX array inputs (ArrayLike contract)
# ---------------------------------------------------------------------------
def test_accepts_numpy_array_input():
    x_np = np.random.default_rng(27).standard_normal((50, 3))
    scaler, z = DataScaler("zscore").fit_transform(x_np)
    assert isinstance(z, jnp.ndarray)
    np.testing.assert_allclose(np.asarray(z.mean(axis=0)), 0.0, atol=1e-6)


def test_accepts_python_list_input():
    x_list = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    scaler, z = DataScaler("minmax").fit_transform(x_list)
    np.testing.assert_allclose(np.asarray(z.min(axis=0)), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(z.max(axis=0)), 1.0, atol=1e-6)
    recovered = scaler.inverse_transform(z)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(x_list), atol=1e-6)


def test_accepts_integer_array_input():
    x_int = jnp.asarray([[1, 2], [3, 4], [5, 6]])
    scaler, z = DataScaler("zscore").fit_transform(x_int)
    # upcast to float happened internally
    assert z.dtype in (jnp.float32, jnp.float64)
    np.testing.assert_allclose(np.asarray(z.mean(axis=0)), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 17. vmap compatibility
# ---------------------------------------------------------------------------
def test_vmap_over_batch_of_datasets():
    """vmap a fit over a stack of independent datasets."""
    rng = np.random.default_rng(28)
    # batch of 5 independent (100, 3) datasets
    batch = jnp.asarray(rng.standard_normal((5, 100, 3)))
    template = DataScaler("zscore")

    fit_single = lambda data: template.fit(data)
    fitted_batch = jax.vmap(fit_single)(batch)
    # offset/scale now have leading batch dim
    assert fitted_batch.offset.shape == (5, 3)
    assert fitted_batch.scale.shape == (5, 3)

    # each batch element matches its own independent fit
    for i in range(5):
        indep = template.fit(batch[i])
        np.testing.assert_allclose(
            np.asarray(fitted_batch.offset[i]), np.asarray(indep.offset), atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(fitted_batch.scale[i]), np.asarray(indep.scale), atol=1e-10
        )


# ---------------------------------------------------------------------------
# 18. Re-fitting an already-fitted scaler
# ---------------------------------------------------------------------------
def test_refit_replaces_previous_state():
    rng = np.random.default_rng(29)
    x1 = _make_data(rng, (100, 2))
    x2 = _make_data(rng, (100, 2)) + 50.0
    scaler1 = DataScaler("zscore").fit(x1)
    scaler2 = scaler1.fit(x2)
    # scaler1 is untouched
    np.testing.assert_allclose(np.asarray(scaler1.offset), np.asarray(x1.mean(axis=0)), atol=1e-6)
    # scaler2 reflects x2, not a blend
    np.testing.assert_allclose(np.asarray(scaler2.offset), np.asarray(x2.mean(axis=0)), atol=1e-6)
    # confirm the second fit actually differs from the first
    assert float(jnp.abs(scaler1.offset - scaler2.offset).max()) > 10.0


# ---------------------------------------------------------------------------
# 19. Grad through inverse_transform
# ---------------------------------------------------------------------------
def test_grad_of_inverse_transform():
    rng = np.random.default_rng(30)
    x = _make_data(rng, (40, 2))
    scaler = DataScaler("zscore").fit(x)
    z = scaler.transform(x)

    def loss(arr):
        return scaler.inverse_transform(arr).sum()

    grad = jax.grad(loss)(z)
    # d/dz_{i,j} (z * scale + offset).sum() = scale_j
    expected = jnp.broadcast_to(scaler.scale, z.shape)
    np.testing.assert_allclose(np.asarray(grad), np.asarray(expected), atol=1e-10)


# ---------------------------------------------------------------------------
# 20. Shape mismatch between fit and transform trailing dims
# ---------------------------------------------------------------------------
def test_transform_with_incompatible_feature_dims_errors():
    rng = np.random.default_rng(31)
    x_train = _make_data(rng, (50, 3))
    scaler = DataScaler("zscore").fit(x_train)
    # transforming data with wrong trailing dim should broadcast-fail
    x_bad = _make_data(rng, (10, 5))
    with pytest.raises((TypeError, ValueError)):
        _ = scaler.transform(x_bad)
