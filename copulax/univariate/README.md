# Univariate Probability Distributions

This directory contains several univariate probability distributions, in addition to the UnivariateFitter object.

## UnivariateFitter

The `univariate_fitter` function fits all / a subset of the probability distributions implemented in copulAX to the sample data, returning the 'best' distribution according to a given metric. The `batch_univariate_fitter` function applies this to multiple columns of data simultaneously.

Support handling is standardized across univariate distributions:

- `logpdf` returns `-inf` for values outside support
- `cdf` returns `0` below support and `1` above support
- fitting uses a support-aware penalized objective to reduce invalid
  parameter proposals during optimization

## Goodness of Fit Tests

CopulAX provides goodness-of-fit tests for univariate distributions:

| Function | Test                                                                                           |
| -------- | ---------------------------------------------------------------------------------------------- |
| ks_test  | [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)       |
| cvm_test | [Cramér-von Mises Test](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93von_Mises_criterion) |

## Implemented Univariate Distributions

Currently the following univariate distributions are implemented in copulAX:
| Object / Module | Distribution |
| --- | --- |
| gamma | [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution)|
| asym_gen_normal | [Asymmetric Generalized Normal](https://en.wikipedia.org/wiki/Generalized_normal_distribution)|
| gen_normal | [Generalized Normal](https://en.wikipedia.org/wiki/Generalized_normal_distribution)|
| gh | [Generalized Hyperbolic](https://en.wikipedia.org/wiki/Generalised_hyperbolic_distribution)|
| gig | [Generalized Inverse Gaussian](https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution)|
| ig | [Inverse-Gamma](https://en.wikipedia.org/wiki/Inverse-gamma_distribution)|
| lognormal | [Log-Normal](https://en.wikipedia.org/wiki/Log-normal_distribution)|
| nig | [Normal-Inverse Gaussian](https://en.wikipedia.org/wiki/Normal-inverse_Gaussian_distribution)|
| normal | [Normal/Gaussian](https://en.wikipedia.org/wiki/Normal_distribution)|
| skewed_t | Skewed/Asymmetric Student's T|
| student_t | [Student's T](https://en.wikipedia.org/wiki/Student%27s_t-distribution)|
| uniform | [Continuous Uniform](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)|
| wald | [Wald/Inverse Gaussian](https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution)|
