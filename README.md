# CopulAX

<p align="center">
    <a href="https://www.python.org">
        <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue"
            alt="Python 3.10, 3.11, 3.12"></a> &nbsp;
    <a href="https://github.com/tfm000/copulax/blob/main/LICENSE.txt">
        <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg"
            alt="MIT license"></a> &nbsp;
    <a href="https://github.com/tfm000/copulax/actions/workflows/tests.yml">
        <img src="https://github.com/tfm000/copulax/actions/workflows/tests.yml/badge.svg?branch=main"
            alt="build"></a> &nbsp;
    <a href="https://codecov.io/github/tfm000/copulax">
        <img src="https://codecov.io/github/tfm000/copulax/branch/main/graph/badge.svg?"
            alt="coverage"></a> &nbsp;
    <!-- <a href="https://sklarpy.readthedocs.io/en/latest/?badge=latest">
        <img src="https://readthedocs.org/projects/sklarpy/badge/?version=latest"
            alt="build"></a> &nbsp;
    <a href="https://pepy.tech/project/sklarpy">
        <img src="https://static.pepy.tech/personalized-badge/sklarpy?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads"
            alt="downloads"></a> &nbsp; -->
    <!-- <a href="https://pypi.org/project/sklarpy/"> -->
    <a href="https://github.com/tfm000/copulax/">
        <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg"
            alt="maintained"></a>
</p>

<p align="center">
    <!-- <a href="https://pypi.org/project/sklarpy/"> -->
    <a href="https://github.com/tfm000/copulax/">
        <img src="https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=apple&logoColor=white"
            alt="mac os"></a>
    <!-- <a href="https://pypi.org/project/sklarpy/"> -->
    <a href="https://github.com/tfm000/copulax/">
        <img src="https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white"
            alt="windows"></a>
    <!-- <a href="https://github.com/tfm000/copulax/">
        <img src="https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white"
            alt="windows"></a> -->
</p>

CopulAX is an open-source software for probability distribution fitting written in JAX, with an emphasis on low-dimensional optimization. The spirital successor to the eariler [SklarPy](https://github.com/tfm000/sklarpy/) project, this JAX implementation provides improved standardization and optimization performance across distribution objects, in addition to inbuilt automatic differentiation capabilities and greater speed via JIT compilation for both CPUs and GPUs.

We foresee this library having many different possible use cases, ranging from machine learning to finance.

## Table of contents

- [Table of contents](#table_of_contents)
- [Installation](#installation)
- [Low-Dimensional Optimization](#low-dimensional-optimization)
- [Implemented Distributions](#implemented-distributions)
- [Examples](#examples)

## Installation

CopulAX is available on PyPI and can be installed by running:

```bash
pip install copulax
```

## Low-Dimensional Optimization

In many fields data remains limited, which can be one of the main motivators for using probabilistic software which can allow the generaton of additional data points with similar statistical properties. However, when dealing with multivariate data, even this can become challenging, due the shape / covariance / correlation matrix arguments of many multivariate and copula distributions resulting in the number of parameters required to be estimated to be O($d^2$). CopulAX aims to work around this constraint where possible, by using analytical relationships between the mean and covariance matrices and other parameters; Estimating the mean and covariance using techniques robust to low sample sizes, then allows for distribution fitting in such settings by removing a large number of the estimated parameters from the numerical optimization loop.

## Development Status

As CopulAX is still in its early stages, we have so far only released a limited number of continuous univariate and multivariate distributions and their copulas, however in the near future we aim to implement the following:

- Many more univariate distributions, including for discrete variables.
- Incorporating goodness of fit tests into univariate_fitter.
- More multivariate distributions. Namely, the special and limiting cases of the generalized hyperbolic.
- Copulas based on each of the aformentioned multivariate distributions.
- Cdf functions for multivariate and copula distributions. This will depend upon the progress of third party jax-based numerical integration libraries such as [quadax](https://github.com/f0uriest/quadax).
- Archimedean copulas.
- Empirical distributions, with different fitting methods (smoothing splines vs 'as is'/ non-smoothed).

CopulAX is currently under active development and so bugs are to be expected. However we have extensive tests for each distribution and function, so we are aiming to limit there number.

## Implemented Distributions

A list of all implemented distributions can be found here:

- <a href="https://github.com/tfm000/copulax/blob/main/copulax/univariate/README.md">Univariate implemented distributions</a>
- <a href="https://github.com/tfm000/copulax/blob/main/copulax/multivariate/README.md">Multivariate implemented distributions</a>
- <a href="https://github.com/tfm000/copulax/blob/main/copulax/copulas/README.md">Copula implemented distributions</a>

## Examples

We have provided <a href="https://github.com/tfm000/copulax/tree/main/examples">jupyter notebooks</a> containing example code for using univariate, multivariate and copula distribution objects.
