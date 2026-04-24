<p align="center">
  <a href="https://copulax.readthedocs.io/en/latest/">
    <img src="docs/_static/logo.png" alt="CopulAX logo" width="600">
  </a>
</p>


<p align="center">
    <a href="https://www.python.org">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/copulax"></a> &nbsp;
    <a href="https://pypi.org/project/copulax/">
        <img alt="PyPI - Package Version" src="https://img.shields.io/pypi/v/copulax"></a> &nbsp;
    <a href="https://github.com/tfm000/copulax/blob/main/LICENSE.txt">
        <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg"
            alt="MIT license"></a> &nbsp;
    <a href="https://copulax.readthedocs.io/en/latest/?badge=latest">
        <img src="https://readthedocs.org/projects/copulax/badge/?version=latest"
            alt="build"></a> &nbsp;
    <a href="https://github.com/tfm000/copulax/actions/workflows/tests.yml">
        <img src="https://github.com/tfm000/copulax/actions/workflows/tests.yml/badge.svg?branch=main"
            alt="build"></a> &nbsp;
    <a href="https://codecov.io/github/tfm000/copulax">
        <img src="https://codecov.io/gh/tfm000/copulax/graph/badge.svg?token=OM89GVW36L"
            alt="coverage"></a> &nbsp;
    <a href="https://github.com/tfm000/copulax/">
        <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg"
            alt="maintained"></a>
</p>

<p align="center">
    <a href="https://github.com/tfm000/copulax/">
        <img src="https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=apple&logoColor=white"
            alt="mac os"></a>
    <a href="https://github.com/tfm000/copulax/">
        <img src="https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white"
            alt="windows"></a>
    <a href="https://github.com/tfm000/copulax/">
        <img src="https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white"
            alt="windows"></a>
</p>

CopulAX is an open-source library for probability distribution fitting, written in [JAX](https://github.com/jax-ml/jax/) with an emphasis on low-dimensional optimization. It is the spiritual successor to [SklarPy](https://github.com/tfm000/sklarpy/) and provides univariate, multivariate and copula distribution objects with JIT compilation and automatic differentiation support.

This library is designed for use cases ranging from machine learning to finance.

## Table of contents

- [Documentation](#documentation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Low-Dimensional Optimization](#low-dimensional-optimization)
- [Development Status](#development-status)
- [Implemented Distributions](#implemented-distributions)
- [Testing](#testing)
- [Examples](#examples)

## Documentation

- Read the Docs: <https://copulax.readthedocs.io/en/latest/>
- API reference: <https://copulax.readthedocs.io/en/latest/api/index.html>

## Installation

CopulAX is available on PyPI and can be installed by running:

```bash
pip install copulax
```

## Quick Start

```python
import jax.random as jr
from copulax.univariate import normal, univariate_fitter
from copulax.multivariate import mvt_normal
from copulax.copulas import gaussian_copula
from copulax.preprocessing import DataScaler

key = jr.PRNGKey(0)
k1, k2, k3 = jr.split(key, 3)

# Univariate fitting
x_uni = jr.normal(k1, shape=(500,))
fitted_uni = normal.fit(x_uni)
best_idx, candidates = univariate_fitter(x_uni)

# Multivariate fitting
x_mvt = jr.normal(k2, shape=(500, 3))
fitted_mvt = mvt_normal.fit(x_mvt)

# Copula fitting
x_cop = jr.normal(k3, shape=(500, 3))
fitted_cop = gaussian_copula.fit(x_cop)

# Preprocessing — jittable z-score / min-max / robust / max-abs scaler
scaler, x_scaled = DataScaler("zscore").fit_transform(x_mvt)
x_original = scaler.inverse_transform(x_scaled)
```

## Saving and Loading

Fitted distributions and fitted `DataScaler` preprocessors can be saved to disk and loaded back in a later session. All distribution types (univariate, multivariate, and copula) are supported, as are `DataScaler` instances. Files use the `.cpx` format and are cross-platform.

```python
import copulax

# Save a fitted distribution
fitted_uni.save("my_model.cpx")

# Load it back (in the same or a different session)
loaded = copulax.load("my_model.cpx")
loaded.logpdf(x_uni)  # identical to fitted_uni.logpdf(x_uni)

# DataScaler uses the same save/load entry points
scaler.save("my_scaler.cpx")
loaded_scaler = copulax.load("my_scaler.cpx")
```

## Low-Dimensional Optimization

In many settings, sample sizes are limited. Probabilistic modeling can help generate additional data with similar statistical structure, but multivariate and copula models often require shape/covariance/correlation parameters that grow as O($d^2$). CopulAX reduces this burden where possible by using analytical relationships between these matrices and other parameters. Estimating location and shape robustly outside the optimization loop can materially reduce the number of numerically optimized parameters.

## Development Status

CopulAX is under active development. Current coverage includes:

- Continuous univariate distributions.
- Multivariate normal-mixture families.
- Elliptical and Archimedean copulas.
- JIT/autodiff-compatible fitting workflows and utility functions.

Near-term roadmap:

- Additional univariate distributions (including discrete support).
- Additional multivariate and copula families.
- Broader CDF coverage for multivariate and elliptical copula objects.
- Empirical distribution support with multiple fitting methods.

## Implemented Distributions

A list of all implemented distributions can be found here:

- <a href="https://github.com/tfm000/copulax/blob/main/copulax/univariate/README.md">Univariate implemented distributions</a>
- <a href="https://github.com/tfm000/copulax/blob/main/copulax/multivariate/README.md">Multivariate implemented distributions</a>
- <a href="https://github.com/tfm000/copulax/blob/main/copulax/copulas/README.md">Copula implemented distributions</a>

## Testing

Tests are comprehensive, but some suites can be slow. A practical workflow is:

1. Run only affected tests first.
2. Run individual test functions while iterating.
3. Keep a timestamped test log while debugging.

```bash
# Default iteration: exclude slow tests (what CI runs)
pytest copulax/tests_new/ -v -m "not slow"

# Specific test file (e.g. elliptical copulas)
pytest copulax/tests_new/test_copulas_elliptical.py -v -m "not slow"

# Specific test function
pytest copulax/tests_new/test_copulas_elliptical.py::TestCopulaFitting::test_fit_returns_valid_params -v
```

```powershell
# Append test output to a running log (PowerShell)
pytest copulax/tests_new/test_copulas_elliptical.py -v -m "not slow" *>&1 `
  | Tee-Object -FilePath copula_test_results.txt -Append
```

## Examples

We have provided <a href="https://github.com/tfm000/copulax/tree/main/examples">jupyter notebooks</a> containing example code for using univariate, multivariate and copula distribution objects.
