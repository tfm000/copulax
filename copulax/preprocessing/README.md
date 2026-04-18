# Preprocessing

This directory contains jittable, autograd-compatible preprocessing objects built on top of `equinox.Module` PyTrees. Each preprocessor fits parameters to input data and exposes `transform` / `inverse_transform` for applying and undoing the same operation on later observations.

All objects in this package:

- store fitted parameters as traced JAX arrays
- compose with `jax.jit`, `jax.grad`, `jax.vmap`, and `equinox` serialisation
- return a new instance from `fit` — originals are never mutated (pure functional)

## DataScaler

`DataScaler` fits an affine rescaling of the form `z = (x - offset) / scale` under one of four methods and exposes `transform`, `inverse_transform`, and `fit_transform`.

| Method     | `offset`                | `scale`                |
| ---------- | ----------------------- | ---------------------- |
| `zscore`   | `x.mean(axis=0)`        | `x.std(axis=0)`        |
| `minmax`   | `x.min(axis=0)`         | `x.max - x.min`        |
| `robust`   | `median(x, axis=0)`     | `q_high - q_low` (IQR) |
| `maxabs`   | `0`                     | `abs(x).max(axis=0)`   |

**Axis 0 is always the sample axis.** For input of shape `(n, *feature_dims)`, the fitted `offset` and `scale` have shape `feature_dims`, and `transform` broadcasts naturally over any leading batch shape.

### Quick start

```python
import jax.numpy as jnp
from copulax.preprocessing import DataScaler

x = jnp.asarray([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]])

# z-score (default)
scaler, z = DataScaler("zscore").fit_transform(x)
assert jnp.allclose(z.mean(axis=0), 0.0)
assert jnp.allclose(z.std(axis=0), 1.0)

# round-trip
x_back = scaler.inverse_transform(z)
assert jnp.allclose(x_back, x)
```

### Additional options

- `offset_only=True` — fit the offset as usual but leave `scale = 1` (centring only).
- `scale_only=True` — fit the scale as usual but leave `offset = 0` (rescaling only).
- `q_low`, `q_high` — quantiles used by the `robust` method (default `0.25`, `0.75`).
- `pre_fns=(forward, inverse)` — optional JAX-compliant function pair applied to the data *before* the affine scaling. Either half may be `None` to skip that direction. Example: `pre_fns=(jnp.log, jnp.exp)` gives z-score over log-data with a faithful round-trip.
- `post_fns=(forward, inverse)` — optional function pair applied *after* the affine scaling during `transform`, inverted first during `inverse_transform`. Example: `post_fns=(jnp.tanh, jnp.arctanh)` squashes the scaled output into `(-1, 1)`.

### Zero-variance safety

If the fitted `scale` contains any zeros (constant feature, degenerate IQR, etc.), those entries are silently replaced with `1.0` so division does not break autograd or produce NaNs. The corresponding output features are pure offsets without rescaling.

### Saving and loading

Fitted `DataScaler` instances can be saved to a `.cpx` file (ZIP archive of a human-readable `metadata.json` plus NumPy `.npy` arrays) and loaded back via the top-level `copulax.load` function — the same entry point used for fitted distributions.

```python
import copulax
from copulax.preprocessing import DataScaler

scaler = DataScaler("zscore").fit(x_train)
scaler.save("my_scaler.cpx")

# ...later, possibly in a new session:
loaded = copulax.load("my_scaler.cpx")
loaded.transform(x_test)  # identical to the original scaler
```

The save path also preserves `pre_fns` / `post_fns` when they reference importable module-level functions. Callables are stored by their import qualname (`{module}.{qualname}`) — no `pickle` is used. Lambdas and locally-defined closures cannot be round-tripped this way and are rejected with a clear `ValueError` at save time. Functions defined at the top level of a script or notebook (`__module__ == "__main__"`) save successfully but will only reload cleanly in a session where an identically-named function is defined in `__main__` — a `UserWarning` is emitted to flag this.
