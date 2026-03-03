# Continuation Notes — Parameter Storage Refactor

## Goal

Refactor copulax to store distribution parameters as instance fields on `eqx.Module` subclasses, making the `params` dict argument optional in all public methods (logpdf, pdf, cdf, ppf, rvs, stats, etc.). Full backward compatibility maintained — passing `params` dict still works.

## Branch

`restructure/v2.1`

## Python Environment

`c:\Users\tyler\Repos\copulax-2\venv\Scripts\python.exe`

**Important:** Always use `-B` flag when running pytest to avoid stale bytecode cache issues:

```
c:\Users\tyler\Repos\copulax-2\venv\Scripts\python.exe -B -m pytest ...
```

---

## What Was Done

### Pattern Applied Everywhere

1. Added parameter fields with `None` defaults to each distribution class
2. Added/updated `__init__` to accept optional param kwargs and convert to jnp arrays
3. Added `_stored_params` property returning params dict (or `None` if any param is missing)
4. Made `params=None` in all public method signatures
5. Added `_resolve_params(params)` call at start of each method body

### Base Classes (DONE)

- **`copulax/_src/_distributions.py`**: Added `_stored_params` property (returns `None` by default) and `_resolve_params(params)` method to `Distribution` base class. Made `params=None` optional in all method signatures on `Univariate`, `Multivariate`, `GeneralMultivariate`, `NormalMixture`.

### Univariate Distributions (DONE — all 9)

All files in `copulax/_src/univariate/`:

- `normal.py` (mu, sigma)
- `student_t.py` (nu, mu, sigma)
- `gamma.py` (alpha, beta)
- `lognormal.py` (mu, sigma)
- `uniform.py` (a, b)
- `ig.py` (alpha, beta)
- `gig.py` (lamb, chi, psi)
- `gh.py` (lamb, chi, psi, mu, sigma, gamma)
- `skewed_t.py` (nu, mu, sigma, gamma)

### Multivariate Distributions (DONE — all 4)

All files in `copulax/_src/multivariate/`:

- `mvt_normal.py` (mu, sigma)
- `mvt_student_t.py` (nu, mu, sigma)
- `mvt_skewed_t.py` (nu, mu, gamma, sigma)
- `mvt_gh.py` (lamb, chi, psi, mu, gamma, sigma)

### Copula Distributions (DONE)

- `copulax/_src/copulas/_distributions.py` — `CopulaBase` has `_marginals`, `_copula_params` fields; `Copula` subclass `__init__` accepts and stores them.
- `copulax/_src/copulas/_archimedean.py` — `ArchimedeanCopula.__init__` accepts and stores `marginals`, `copula_params`.

### Test Files (DONE)

Updated tests to handle new `__dict__` attributes (parameter field names like mu, sigma, etc. now appear on instances). The `test_all_methods_implemented` checks were modified to filter to only callable attributes.

---

## Critical Bug Fixed (Univariate Only)

### The `_Missing` Problem

Equinox `eqx.Module` fields that are declared but not assigned in `__init__` get `equinox._module._flatten._Missing` sentinel value. JAX cannot trace `_Missing`, so passing a singleton (e.g., `normal = Normal("Normal")`) to any JIT-compiled function fails with:

```
TypeError: Error interpreting argument ... as an abstract array.
The problematic value is of type <class 'equinox._module._flatten._Missing'>
```

### Fix Applied (Univariate — DONE)

Changed from conditional assignment:

```python
# BAD — leaves field as _Missing when param is None
if mu is not None:
    self.mu = jnp.asarray(mu, dtype=float).reshape(())
```

To always-assign:

```python
# GOOD — field is always set (to None or array)
self.mu = jnp.asarray(mu, dtype=float).reshape(()) if mu is not None else None
```

**This fix was applied to all 9 univariate distributions.**

### Fix NOT YET Applied (Multivariate & Copula)

The same `_Missing` bug exists in all 4 multivariate distributions and 2 copula base classes. They still use the conditional `if ... is not None: self.x = ...` pattern. **This must be fixed before multivariate and copula tests will pass.**

Files needing the fix:

- `copulax/_src/multivariate/mvt_normal.py` — mu, sigma
- `copulax/_src/multivariate/mvt_student_t.py` — nu, mu, sigma
- `copulax/_src/multivariate/mvt_skewed_t.py` — nu, mu, gamma, sigma
- `copulax/_src/multivariate/mvt_gh.py` — lamb, chi, psi, mu, gamma, sigma
- `copulax/_src/copulas/_distributions.py` — \_marginals, \_copula_params (in `Copula.__init__`)
- `copulax/_src/copulas/_archimedean.py` — \_marginals, \_copula_params (in `ArchimedeanCopula.__init__`)

**Note:** Multivariate params like `mu` and `sigma` are arrays (not scalars), so they use `jnp.asarray(x, dtype=float)` without `.reshape(())`. The fix pattern for these is:

```python
self.mu = jnp.asarray(mu, dtype=float) if mu is not None else None
```

---

## Testing Status

### Tests Need to Be Run

Tests have **not** been fully run yet. Here's the status:

| Test Module                                          | Status         | Notes                            |
| ---------------------------------------------------- | -------------- | -------------------------------- |
| `copulax/tests/univariate/test_gof.py`               | PASSED (10/10) | All GOF tests pass               |
| `copulax/tests/univariate/test_univariate.py`        | NOT RUN        | 270+ tests, needs full run       |
| `copulax/tests/univariate/test_univariate_fitter.py` | NOT RUN        | Tests univariate_fitter function |
| `copulax/tests/multivariate/`                        | NOT RUN        | Fix `_Missing` bug first         |
| `copulax/tests/copulas/`                             | NOT RUN        | Fix `_Missing` bug first         |
| `copulax/tests/test_golden.py`                       | NOT RUN        | Golden value tests               |
| `copulax/tests/test_validation.py`                   | NOT RUN        |                                  |
| `copulax/tests/test_special.py`                      | NOT RUN        |                                  |
| `copulax/tests/test_utils.py`                        | NOT RUN        |                                  |

### Test Run Order (user requested)

1. Run univariate tests, fix bugs, re-run until clean
2. Run multivariate tests (after fixing `_Missing` bug), fix bugs, re-run
3. Run copula tests (after fixing `_Missing` bug), fix bugs, re-run
4. Run remaining tests (golden, validation, special, utils)
5. Run full suite

### Test Commands

```powershell
# Univariate
c:\Users\tyler\Repos\copulax-2\venv\Scripts\python.exe -B -m pytest copulax/tests/univariate/ -x --tb=short

# Multivariate
c:\Users\tyler\Repos\copulax-2\venv\Scripts\python.exe -B -m pytest copulax/tests/multivariate/ -x --tb=short

# Copulas
c:\Users\tyler\Repos\copulax-2\venv\Scripts\python.exe -B -m pytest copulax/tests/copulas/ -x --tb=short

# Remaining
c:\Users\tyler\Repos\copulax-2\venv\Scripts\python.exe -B -m pytest copulax/tests/test_golden.py copulax/tests/test_validation.py copulax/tests/test_special.py copulax/tests/test_utils.py -x --tb=short

# Full suite
c:\Users\tyler\Repos\copulax-2\venv\Scripts\python.exe -B -m pytest copulax/tests/ -x --tb=short
```

---

## Remaining Work Summary

1. **Fix `_Missing` bug in multivariate/copula files** (6 files listed above)
2. **Run univariate tests** — test_univariate.py and test_univariate_fitter.py not yet verified
3. **Run multivariate tests** — after fixing `_Missing`
4. **Run copula tests** — after fixing `_Missing`
5. **Run remaining test files** (golden, validation, special, utils)
6. **Run full test suite** to confirm everything passes together

---

## Architecture Notes

- Singleton instances exist at module bottom (e.g., `normal = Normal("Normal")`, `gamma = Gamma("Gamma")`) — these have `None` params and are used when params are passed explicitly.
- `univariate_fitter` in `copulax/_src/univariate/univariate_fitter.py` uses `_DIST_REGISTRY` of singletons and always passes params explicitly — should work unchanged.
- `_cdf.py`, `_ppf.py`, `_rvs.py`, `_gof.py` are internal utilities that always pass params explicitly — no changes needed.
- GIGBase/GHBase/SkewedTBase have static methods for `logpdf`/`pdf` — kept as-is for `_pdf_for_cdf`. Instance methods on GIG/GH/SkewedT override and add `_resolve_params`.
