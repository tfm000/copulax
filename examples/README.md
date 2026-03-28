# Examples

Jupyter notebooks demonstrating CopulAX functionality.

## Notebooks

| Notebook                                                             | Description                                                                                                                   |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [univariate_example.ipynb](univariate_example.ipynb)                 | Univariate distributions: PDF, CDF, PPF, sampling, fitting, automatic distribution selection, goodness-of-fit tests, plotting, save/load |
| [multivariate_example.ipynb](multivariate_example.ipynb)             | Multivariate distributions: PDF, sampling, fitting, JIT compilation, save/load                                                            |
| [copula_example.ipynb](copula_example.ipynb)                         | Elliptical copulas: parameter specification, PDF evaluation, sampling, fitting, save/load                                                 |
| [archimedean_copula_example.ipynb](archimedean_copula_example.ipynb) | Archimedean copulas (Clayton, Frank, Gumbel, Joe, AMH, Independence): PDF, CDF, sampling, fitting, Kendall's tau, save/load              |
| [corr_cov_example.ipynb](corr_cov_example.ipynb)                     | Correlation and covariance matrix estimation and random sampling                                                                          |

## Prerequisites

```bash
pip install copulax jupyter matplotlib
```

## Running

```bash
jupyter notebook examples/
```

Or open the notebooks directly in VS Code with the Jupyter extension.
