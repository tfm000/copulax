# Time-Series Models

This directory contains all implemented CopulAX time-series models, alongside the standalone diagnostics, unit-root tests, and two-stage standard-error machinery that support them.

All models are JIT-compatible, autograd-compatible, and built on the `equinox.Module` PyTree pattern shared with the rest of CopulAX. Each model exposes a uniform `fit` / `forecast` / `residuals` / `stats` / `summary` contract and supports warm-start fitting for fast rolling-window refits.

The `(p, q, residual_dist)` triple — and, for the joint composite, the `var_model` choice — is part of the model's **static** configuration: it parameterises the compiled fit graph and is fixed for the lifetime of the instance. Construct a new instance to fit a different specification.

## Mean-Equation Models

Autoregressive / moving-average mean models. Innovations are drawn from any standardised (mean = 0, var = 1) law on the residual whitelist.

| Object / Module | Model                                                                                                   |
| --------------- | ------------------------------------------------------------------------------------------------------- |
| AR              | [Autoregressive AR(p)](https://en.wikipedia.org/wiki/Autoregressive_model)                              |
| MA              | [Moving-Average MA(q)](https://en.wikipedia.org/wiki/Moving-average_model)                              |
| ARMA            | [Autoregressive Moving-Average ARMA(p, q)](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model) |

## Conditional-Variance Models

GARCH-family conditional-variance models. Each variant exposes a uniform recursion / pack-unpack / cold-start interface so the joint composite (`ArmaGarch`) can plug any of them in without duplication.

| Object / Module | Model                                                                                                                                        |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| GARCH           | [GARCH(p, q)](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity#GARCH) — Bollerslev (1986)                         |
| IGARCH          | Integrated GARCH(p, q) — Engle & Bollerslev (1986); persistence pinned to 1                                                                  |
| GJR_GARCH       | GJR-GARCH(p, q) — Glosten, Jagannathan & Runkle (1993); asymmetric / leverage σ²-form                                                         |
| EGARCH          | Exponential GARCH(p, q) — Nelson (1991); recursion on log-variance                                                                            |
| TGARCH          | Threshold GARCH(p, q) — Zakoian (1994); recursion on σ with sign-split innovations                                                            |
| QGARCH          | Quadratic ARCH(1, q) — Sentana (1995); linear-in-ε asymmetry term                                                                             |
| GARCH_M         | GARCH-in-Mean(p, q) — Engle, Lilien & Robins (1987); conditional variance enters the mean equation                                           |

## Joint ARMA-GARCH Composite

`ArmaGarch` fits the mean and variance equations under a single MLE objective. The variance equation can be any of the GARCH-family variants above. Standard errors are correct under joint MLE — the separable workflow (fit ARMA, then fit a GARCH on its residuals) is also supported, but should be paired with the two-stage standard-error correction below.

```python
from copulax.timeseries import ArmaGarch, GJR_GARCH
from copulax.univariate import skewed_t

fit = ArmaGarch(
    mean_order=(1, 1),
    var_model=GJR_GARCH,
    var_order=(1, 1),
    residual_dist=skewed_t,
).fit(y)
```

## Allowed Residual Distributions

The residual-law whitelist comprises every univariate distribution for which the moment-matching `_standardise_params` classmethod produces a (mean = 0, var = 1) form. These are the innovations the time-series subpackage allows for AR / MA / ARMA, every GARCH variant, and the joint composite.

| Object / Module | Distribution                                                                                       |
| --------------- | -------------------------------------------------------------------------------------------------- |
| normal          | [Normal/Gaussian](https://en.wikipedia.org/wiki/Normal_distribution)                               |
| student_t       | [Student's T](https://en.wikipedia.org/wiki/Student%27s_t-distribution)                            |
| gen_normal      | [Generalized Normal](https://en.wikipedia.org/wiki/Generalized_normal_distribution)                |
| nig             | [Normal-Inverse Gaussian](https://en.wikipedia.org/wiki/Normal-inverse_Gaussian_distribution)      |
| gh              | [Generalized Hyperbolic](https://en.wikipedia.org/wiki/Generalised_hyperbolic_distribution)        |
| skewed_t        | Skewed/Asymmetric Student's T                                                                      |

## Serial-Correlation and ARCH Diagnostics

CopulAX provides standalone diagnostics on a univariate series. All are JIT- / autograd-compatible. `acf` / `pacf` return JAX 1-D arrays; `ljung_box` / `arch_lm` return self-describing result dicts with the same numerical schema as the unit-root tests below — keys `statistic`, `p_value`, `used_lag`, `n_obs`, `dof` (the χ² reference degrees of freedom). H0 / H1 statements are in each function's docstring rather than the dict, so the result is a pure-JAX pytree and round-trips through `jax.jit`. The matplotlib-based plotting helpers drop out of JAX traces and consume the JAX output as numpy arrays for rendering.

| Function   | Test                                                                                                                       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------- |
| acf        | [Sample Autocorrelation Function](https://en.wikipedia.org/wiki/Autocorrelation) (biased ACVF estimator)                   |
| pacf       | [Partial Autocorrelation Function](https://en.wikipedia.org/wiki/Partial_autocorrelation_function) (Levinson-Durbin)       |
| ljung_box  | [Ljung-Box Q-Test](https://en.wikipedia.org/wiki/Ljung%E2%80%93Box_test) for serial correlation                            |
| arch_lm    | [Engle's LM Test](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity#Testing_for_ARCH) for ARCH effects |
| plot_acf   | matplotlib companion for `acf` with significance bands                                                                     |
| plot_pacf  | matplotlib companion for `pacf` with significance bands                                                                    |

## Unit-Root and Stationarity Tests

Confirmatory pre-fit checks that the input series is appropriate for an ARMA / ARMA-GARCH model. Both return the standardised result dict — keys `statistic`, `p_value`, `used_lag`, `n_obs`, `crit_values` (mapping `"1%"` / `"5%"` / `"10%"` to the tabulated critical values). H0 / H1 statements are in each function's docstring; for `adf` H1 adapts to the `regression` argument, for `kpss` H0 adapts.

| Function | Test                                                                                                                  | H₀                                  |
| -------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| adf      | [Augmented Dickey-Fuller](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test) — MacKinnon (1996) CVs | unit root (non-stationary)          |
| kpss     | [Kwiatkowski-Phillips-Schmidt-Shin](https://en.wikipedia.org/wiki/KPSS_test) — KPSS (1992) Table 1 CVs              | (level- or trend-) stationary       |

## Two-Stage Standard Errors

When the user fits ARMA first and then a GARCH variant on the ARMA residuals separately, naive plug-in standard errors on the second-stage parameters ignore the noise contributed by the first-stage estimate and are biased — typically too small. The Pagan-Newey (1988) sandwich corrects for this by augmenting the second-stage score with a first-stage influence-function term.

| Function                    | Output                                                                                                |
| --------------------------- | ----------------------------------------------------------------------------------------------------- |
| two_stage_cov               | Pagan-Newey corrected covariance matrix on the second-stage natural-parameter MLE                     |
| two_stage_standard_errors   | √diag of the corrected covariance, packed back into a parameter dict matching the GARCH `params` schema |
