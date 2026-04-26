# Changelog

All notable changes to CopulAX are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and CopulAX follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] — 2026-04-25

A major release focused on API consistency, numerical stability, and serialisation. Several public surfaces were renamed or restructured; see the **Migration Guide** below for upgrade steps.

### Migration Guide (2.x → 3.0.0)

| 2.x | 3.0.0 |
| --- | --- |
| `from copulax import get_local_random_key` | `from copulax import get_random_key` |
| `dist.fit(data, method="MLE")` (or `"EM"`, `"MOM"`, `"LDMLE"`, `"MoM"`) | `dist.fit(data, method="mle")` (lowercase across the board) |
| `copula.fit_copula(u, method="ml")` | `copula.fit_copula(u, method="fc_mle")` |
| `copula.fit_copula(u, method="em")` | `copula.fit_copula(u, method="ecme")` |
| `copula.fit_copula(u, method="em2")` | `copula.fit_copula(u, method="ecme_double_gamma")` |
| `copula.fit_copula(u, method="em3")` | `copula.fit_copula(u, method="ecme_outer_gamma")` |
| `dist.ppf(q, params, cubic=True, num_points=100, maxiter=50)` | `dist.ppf(q, params, brent=False, nodes=100, maxiter=20)` |
| `copula.rvs(..., cubic=True)` | `copula.rvs(..., brent=False, nodes=100)` |
| `params = {"lambda": ..., "chi": ..., "psi": ...}` (GH / GIG / MvtGH / `gh_copula`) | `params = {"lamb": ..., "chi": ..., "psi": ...}` |

Unknown `method=` strings now raise `ValueError` instead of silently falling back, because every distribution class declares an explicit `_supported_methods` frozenset.

### Added

- **Univariate distributions**: `nig` (Normal-Inverse Gaussian), `wald` (Inverse Gaussian), `asym_gen_normal` (asymmetric generalised normal), `exponential` (rate-parameterised; closed-form `logpdf`/`cdf`/`ppf`/`rvs` and MLE `fit` with `λ̂ = 1/mean(x)`, NaN propagation for out-of-support `fit` inputs, scipy `expon` cross-validated).
- **Archimedean copula**: `amh_copula` (Ali-Mikhail-Haq).
- **Preprocessing sub-package** (`copulax.preprocessing`) with `DataScaler` — a jittable, autodiff-compatible affine rescaler supporting z-score, min-max, robust, and max-abs methods, plus optional user-supplied pre/post transform pairs.
- **Serialisation**: `Distribution.save(path)` and top-level `copulax.load(path)` for fitted distributions, copulas, and `DataScaler` instances. Cross-platform `.cpx` format; no `pickle` dependency for callable fields (saved by import qualname).
- **Special functions**: `digamma`, `trigamma`, and `log_kv` exposed via `copulax.special`. `log_kv` has a custom JVP for accurate, fast gradients.
- **Top-level utilities**: `get_random_key` (replaces removed `get_local_random_key`), `__version__` exposed at package root.
- **Multivariate Student-t MLE**: `mvt_student_t.fit(method="mle")`.
- **ECME fitting** extended to `mvt_skewed_t` (default `method="em"`, fallback `"ldmle"`).
- **NIG method-of-moments initialisation** for GH-family parameter optimisation, replacing weaker prior heuristics.
- **PPF cubic-spline path**: a Chebyshev-Lobatto cubic spline of the inverse CDF in quadax-transformed t-space, with an IFT custom VJP for exact gradients. Activated by the new `brent=False` default.
- **Adaptive CDF integration**: numerical CDFs now use adaptive breakpoints with a shorter-tail switch — large speed and tail-accuracy improvement for heavy-tailed distributions.
- **JAX export compatibility tests** for distributions and copulas (`b232258`).
- **Third-party cross-validation tests**: scipy for univariate / elliptical-copula manual Sklar decomposition, R `ghyp` for `mvt_gh` / `mvt_skewed_t`, R `copula` for Archimedean copulas. Replaces the previous v1.0.1 golden regression fixtures.
- **`Distribution.__init_subclass__` hook** that surfaces inherited docstrings on subclass overrides — `help()`, IPython `?`, and IDE hover tooltips now show the parent contract for overrides that omit a docstring (Python's `inspect.getdoc()` does not walk the MRO past such overrides on its own). Covers all univariate, multivariate, copula and Archimedean families through the single common base class.
- **Docstring-visibility regression test** (`tests/test_docstring_visibility.py`) — fails CI if any public method on any public distribution / preprocessing / stats / special object returns an empty `inspect.getdoc()`.
- **`Distribution._resolve_params` regression suite** (`tests/test_resolve_params.py`) — pins the resolve-params contract for every public method that takes `params: dict = None` across univariate, multivariate, Archimedean and mean-variance copula families: fitted instances must dispatch through stored params, and unfitted instances must raise `ValueError("No parameters provided. ...")` at Python trace time rather than crashing inside JAX.

### Changed

- **Default PPF solver**: numerical inverse-CDF distributions now default to a Chebyshev-Lobatto cubic spline; pass `brent=True` to opt back into per-quantile Brent root-finding. Closed-form distributions (`normal`, `gen_normal`, `uniform`, `ig`, `gamma`, `lognormal`) are unaffected — they always use the analytical `_ppf` override.
- **Mean-variance distribution module split**: previously consolidated `_mean_variance.py` is now `_normal_mixture.py` plus a separated base-class file. Resolves prior circular-import issues. Internal only — no public-API impact beyond what is already listed in the migration table.
- **Distribution registries moved under `_src/`**: each family's registry now lives in `copulax/_src/{univariate,multivariate,copulas}/_registry.py` (was `copulax/{univariate,multivariate,copulas}/distributions.py`). The public `distributions` modules thinly re-export from the new locations — user-facing imports are unchanged. Internal only: lets `univariate_fitter` and `_serialization` import the registry without booting the public package's `__init__`, which previously cycled back into the fitter.
- **Skewed-t log-PDF stability**: introduced `log_kv_plus_s_log_r` for continuous evaluation at γ = 0.
- **Outer iterations of copula fits** now run via `lax.scan` instead of Python loops (commit `87aa737`).
- **LDMLE feasibility reparametrisation** for mean-variance distributions; consistent with ECME initialisation.
- **`Distribution.__hash__` / `__eq__`** now hash by `id(self)` and compare by stored fitted parameters recursively (was `type + name`). Affects code using distributions as dict keys / set members or comparing via `==`.
- **Fitted-instance auto-naming** scheme changed from `Fitted{Cls}-{counter}` to `Fitted{Cls}-{id(params):x}`.
- **Correlation denoising** (`Correlation._rm`, `Correlation._laloux`, `_corr_from_cov`) switched from `inv(eigenvectors)` to `eigenvectors.T` and uses diagonal rescaling. `spearman` no longer calls `_ensure_valid`. Numerical outputs may shift at the floating-point level versus 2.x; downstream copula fits using these correlation matrices may show small differences.
- **`_dist_tree["common"]`**: structure fixed from a recursive self-copy to `{"continuous": {}, "discrete": {}}`.

### Removed

- **`copulax.get_local_random_key`** — use `copulax.get_random_key`.
- **`golden` and `local_only` pytest markers** — `golden` retired with the v1.0.1 regression fixtures; `local_only` no longer used.
- **Legacy v1.0.1 golden regression fixtures** (`tests/test_golden.py`, `tests/golden/*.npz`). Replaced with third-party cross-validation tests in `tests_new/`.
- **`pdf` overrides** on `Gamma`, `GH`, `GIG` — base-class `pdf` is now inherited (no behavioural change).
- **`Gamma.logpdf` and `Gamma.logcdf` overrides** — pure single-line `super()` passthroughs; the base class provides identical behaviour, and removing them lets `inspect.getdoc()` resolve cleanly.

### Fixed

- **Far-right CDF tail underflow** for heavy-tailed univariate distributions (commits `cc94c70`, `babffe3`) — previously CDFs could fail to reach 1 in extreme tails. Now passes the full `test_cdf_far_right_tail_is_one` parametrised suite.
- **`student_t_copula.fit_copula` JIT incompatibility**: latent `float(traced)` bug surfaced and fixed during JIT-compatibility test rollout.
- **Version retrieval logic** in `copulax/__init__.py` now uses `importlib.metadata.version` with `PackageNotFoundError` fallback.
- **`get_random_key` no longer hard-fails when `jax_enable_x64` is off** (the JAX default). Previously, `pure_callback` rejected the `jnp.int64` `ShapeDtypeStruct`, so any code path that lazily resolved a key — including the README quick-start — raised `ValueError` in a fresh notebook session. The seed dtype now adapts to the active x64 setting (int64 with x64, int32 without). Regression test in `tests/test_utils.py::test_works_with_x64_disabled` exercises the x64-off path in a subprocess (the conftest forces x64 on globally for the rest of the suite).
- **Kolmogorov-Smirnov small-λ guard restored.** The `_KS_LAM_MIN` lookup table in `_src/univariate/_gof.py` was silently broken: keys were dtype *classes* but `arr.dtype` returns an *instance* with a different hash, so the dict lookup always fell through to a fallback that itself underflowed to 0.0 when `jax_enable_x64` was off (its float64 `tiny` underflowed in float32 arithmetic at module load). Net effect with x64=off: the small-λ guard never fired, and `ks_test` p-values for very small statistics were the result of an alternating series that diverges in that regime. Replaced with `_ks_lam_min(d)` that derives the threshold from `jnp.finfo(d.dtype).tiny` at call time — dtype-aware, jit-safe, robust to x64 setting. New regression class `tests/test_gof.py::TestKSLamMin` pins the dtype-correct value, jit compatibility, and the small-d → p=1 clamp.
- **`stats.skew(bias=False)` and `stats.kurtosis(bias=False)` no longer emit `UserWarning` in x64-off mode.** The `jnp.float64(...)` wrapper around the small-sample correction denominator silently demoted to float32 (with a JAX warning on every call) and achieved nothing — surrounding arithmetic was already float32 in x64-off, and the Python scalar promotes naturally to float64 when x64 is on. Wrapper removed; computed values are bit-identical in both modes.
- **`AsymGenNormal.logpdf` stability at the support boundary**: `log1p(-κ·z)` and `log(α - κ·(x − ζ))` could be evaluated with non-positive arguments at and beyond the support edge (right-bounded for κ > 0, left-bounded for κ < 0), producing NaNs that contaminated autograd through `jnp.where`. The boundary branch now substitutes `x` with `ζ` (always strictly inside the support) so both log arguments stay valid, then masks the result back to `-inf` on the out-of-support region.

### Notes

- Status classifier remains `Development Status :: 4 - Beta`. The 3.0.0 bump reflects the breaking changes catalogued above; the API surface is expected to be stable through the 3.x line.
- The legacy `tests/` directory has been removed and the replacement `tests_new/` suite has been renamed to `tests/`. See `.claude/test_audit/` for the historical migration scoreboard.

[3.0.0]: https://github.com/tfm000/copulax/releases/tag/v3.0.0
