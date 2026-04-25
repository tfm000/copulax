# Multivariate Distribution Implementations

This directory contains all multivariate probability distribution implementations in pure JAX, plus correlation and covariance matrix utilities.

## Architecture

### Base Class (`_distributions.py` in parent)

The `Multivariate` base class provides a standardized interface:

- `pdf()` / `logpdf()` — probability density
- `rvs()` / `sample()` — random variate sampling
- `fit()` — parameter estimation
- `support()` — distribution support bounds

### Implemented Distributions

| Module             | Distribution                        | Key Parameters                  |
| ------------------ | ----------------------------------- | ------------------------------- |
| `mvt_normal.py`    | Multivariate Normal                 | mean, covariance                |
| `mvt_student_t.py` | Multivariate Student's T            | mean, shape, degrees of freedom |
| `mvt_gh.py`        | Multivariate Generalized Hyperbolic | mean, shape, lamb, chi, psi     |
| `mvt_skewed_t.py`  | Multivariate Skewed Student's T     | mean, shape, gamma, df          |

### Correlation and Covariance (`_shape.py`)

- `corr()` — correlation matrix estimation (Pearson, Spearman, Kendall methods)
- `cov()` — covariance matrix estimation
- `random_correlation()` — uniformly sample random correlation matrices
- `random_covariance()` — sample random positive-definite covariance matrices

### Helper Modules

| Module      | Purpose                                                  |
| ----------- | -------------------------------------------------------- |
| `_utils.py` | Input validation and preprocessing for multivariate data |
