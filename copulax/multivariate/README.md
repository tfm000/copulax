# Multivariate Probability Distributions

This directory contains all implemented CopulAX multivariate probability distributions, in addition to functions for estimating and randomly sampling covariance and correlation matrices.

## Implemented Multivariate Distributions

Currently the following multivariate distributions are implemented in copulAX:
| Object / Module | Distribution |
| --- | --- |
| mvt_normal | [Multivariate Normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)|
| mvt_student_t | [Multivariate Student's T](https://en.wikipedia.org/wiki/Multivariate_t-distribution)|
| mvt_gh | [Multivariate Generalized Hyperbolic](https://en.wikipedia.org/wiki/Generalised_hyperbolic_distribution)|
| mvt_skewed_t |Multivariate Skewed/Asymmetric Student's T|

## Correlation and Covariance Estimators

CopulAX provides several methods for estimating both the correlation and covariance matrices from sample data.
These are accessable through the `corr` and `cov` functions respectively.

## Correlation and Covariance Matrix sampling

CopulAX implements random uniform samplers for both correlation and covariance matrices through the `random_correlation` and `random_covariance` functions respectively.
