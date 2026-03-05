# Tests

This directory contains the copulAX test suite, organized by distribution family.

## Structure

| Directory / File     | Description                                                                        |
| -------------------- | ---------------------------------------------------------------------------------- |
| `conftest.py`        | Top-level fixtures: sample generation, constants (`NUM_SAMPLES`, `NUM_ASSETS`)     |
| `helpers.py`         | Shared test utilities: parameter validation, shape checking, gradient verification |
| `test_golden.py`     | Golden regression tests — verify outputs match pre-computed baselines              |
| `test_special.py`    | Tests for special mathematical functions                                           |
| `test_stats.py`      | Tests for statistical functions                                                    |
| `test_utils.py`      | Tests for utility functions                                                        |
| `test_validation.py` | Input validation tests                                                             |
| `test_plot.py`       | Plotting tests                                                                     |
| `univariate/`        | Univariate distribution tests                                                      |
| `multivariate/`      | Multivariate distribution tests                                                    |
| `copulas/`           | Copula distribution tests (elliptical + Archimedean)                               |
| `golden/`            | Golden test fixture data (`.npz` files) and generation scripts                     |

## Running Tests

```bash
# Run all tests
pytest copulax/tests/

# Run a specific test file
pytest copulax/tests/copulas/test_archimedean.py -v

# Run a specific test function
pytest copulax/tests/copulas/test_archimedean.py::TestIndependenceCopula::test_copula_logpdf_is_zero -v

# Run tests matching a pattern
pytest copulax/tests/ -k "independence" -v

# Run golden regression tests
pytest copulax/tests/test_golden.py -v -m golden
```

## Golden Tests

Golden tests compare current outputs against pre-computed baselines stored in `golden/*.npz` files:

- `univariate.npz`, `multivariate.npz`, `copulas.npz` — baselines from copulax v1.0.1
- `archimedean.npz` — baselines from current code (no v1.0.1 equivalents)

To regenerate archimedean golden data:

```bash
python -m copulax.tests.golden.generate_golden
```

## Test Patterns

- **Parametrized tests**: Most tests are parametrized over distribution objects using `pytest.param`
- **Property-based tests**: Verify mathematical properties (Fréchet bounds, generator roundtrip, density positivity)
- **Gradient tests**: Verify JAX autodiff works correctly on distribution functions
- **JIT tests**: Verify functions are compatible with `jax.jit`
- **KS tests**: Verify random samples have correct marginal distributions
