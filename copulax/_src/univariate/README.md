# Univariate Distribution Implementations

This directory contains all univariate probability distribution implementations in pure JAX, plus fitting utilities and goodness-of-fit tests.

## Architecture

### Base Class (`_distributions.py` in parent)

The `Univariate` base class provides a standardized interface:

- `pdf()` / `logpdf()` — probability density
- `cdf()` / `logcdf()` — cumulative distribution function
- `ppf()` — percent-point function (inverse CDF), with optional cubic spline approximation
- `rvs()` / `sample()` — random variate sampling
- `fit()` — parameter estimation via maximum likelihood
- `stats()` — distribution statistics (mean, variance, etc.)
- `support()` — distribution support bounds
- `plot()` — visualization with optional sample overlay

### Support-Aware Behavior

The base `Univariate` implementation now enforces support consistently:

- `logpdf(x)` maps values outside support to `-inf`
- `cdf(x)` maps values below support to `0` and above support to `1`
- fitting uses a penalized objective that discourages parameter proposals
  implying out-of-support observations or non-finite log-density values

### Auto-Discovery

New distributions are automatically discovered by `copulax/univariate/distributions.py`. To add a distribution:

1. Create a module in this directory (e.g., `my_dist.py`)
2. Define a class extending `Univariate` and instantiate a singleton
3. It will be automatically available via `from copulax.univariate import my_dist`

### Fitting Utilities

- `univariate_fitter.py` — fits all distributions to data and selects the best by metric (AIC, BIC, etc.)
- `batch_univariate_fitter()` — applies `univariate_fitter` to multiple data columns simultaneously

### Goodness-of-Fit Tests (`_gof.py`)

- `ks_test` — Kolmogorov-Smirnov test with Marsaglia-Tsang-Wang p-value approximation
- `cvm_test` — Cramér-von Mises test with Csörgő-Faraway eigenvalue expansion

### Helper Modules

| Module              | Purpose                                          |
| ------------------- | ------------------------------------------------ |
| `_cdf.py`           | CDF computation via numerical integration of PDF |
| `_ppf.py`           | PPF computation via root-finding on CDF          |
| `_rvs.py`           | Common random variate sampling utilities         |
| `_normal_mixture.py` | Normal-mixture helpers: stats, feasibility reparam |
| `_utils.py`         | Shared internal utilities                        |
