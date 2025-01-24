# CopulAX

CopulAX is an open-source software for probability distribution fitting written in JAX, with an emphasis on low-dimensional optimization. The spirital successor to the eariler [SklarPy](https://github.com/tfm000/sklarpy/) project, this JAX implementation provides improved standardization and optimization performance across distribution objects, in addition to inbuilt automatic differentiation capabilities and greater speed via JIT compilation for both CPUs and GPUs.

We foresee this library having many different possible use cases, ranging from machine learning to finance.

<!-- ## Installation
CopulAX is available on PyPI and can be installed by running:

```bash
pip install copulax
``` -->

## Low-Dimensional Optimization
In many fields data remains limited, which can be one of the main motivators for using probabilistic software which can allow the generaton of additional data points with similar statistical properties. However, when dealing with multivariate data, even this can become challenging due the shape / covariance / correlation matrix arguments of many multivariate and copula distributions resulting in the number of parameters required to be estimated to be O(d^2). CopulAX aims to work around this constraint where possible, by using analytical relationships between the mean and covariance matrices and other parameters. Estimating the mean and covariance using techniques robust to low sample sizes, then allows for distribution fitting in such settings by removing a large number of the estimated parameters from the numerical optimization loop.

## Development Status
As CopulAX is still in its early stages, we have so far only released a limited number of continuous univariate and multivariate distributions, however in the near future we aim to implement the following:
- More univariate distributions, including for discrete variables.
- A UnivariateFitter object allowing for the automatic selection of univariate distributions based on a given metric.
- More multivariate distributions. Namely, the Student-T, Skewed-T and Generalized Hyperbolic.
- Copulas based on each of the aformentioned multivariate distributions.
- Cdf functions for multivariate and copula distributions. This will depend upon the progress of third party jax-based numerical integration libraries such as [quadax](https://github.com/f0uriest/quadax).
- Archimedean copulas.
- Empirical distributions. 

CopulAX is currently under active development and so bugs are to be expected. However we have extensive tests for each distribution and function, so we are aiming to limit there number.
