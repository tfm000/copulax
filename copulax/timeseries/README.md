# Time-Series Models

This directory contains all implemented CopulAX time-series models, alongside the standalone diagnostics, unit-root tests, and two-stage standard-error machinery that support them.

All models are JIT-compatible, autograd-compatible, and built on the `equinox.Module` PyTree pattern shared with the rest of CopulAX. Each model exposes a uniform `fit` / `forecast` / `residuals` / `stats` / `summary` contract and supports warm-start fitting for fast rolling-window refits.

The `(p, q)` orders and (for the joint composite) the `var_model` class are **static** configuration — they parameterise the compiled fit graph and are fixed for the lifetime of the instance. The `residual_dist` is a traced PyTree leaf: pre-fit it stores the user-supplied template (defaults to `normal`), and post-fit it stores the fully-fitted standardised (mean = 0, var = 1) residual distribution, so `fit.residual_dist.params` / `fit.residual_dist.sample(...)` / `fit.residual_dist.logpdf(...)` work directly. Only the residual-law *type* drives JIT recompilation across fits — same-type-with-different-parameters reuses the compiled graph. Construct a new instance to fit a different specification.

## Mean-Equation Models

Autoregressive / moving-average mean models. Innovations are drawn from any standardised (mean = 0, var = 1) law on the residual whitelist.

| Object / Module | Model                                                                                                   |
| --------------- | ------------------------------------------------------------------------------------------------------- |
| AR              | [Autoregressive AR(p)](https://en.wikipedia.org/wiki/Autoregressive_model)                              |
| MA              | [Moving-Average MA(q)](https://en.wikipedia.org/wiki/Moving-average_model)                              |
| ARMA            | [Autoregressive Moving-Average ARMA(p, q)](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model) |

Fitted instances expose `.standard_errors()`, `.confidence_intervals()`, residual-diagnostic accessors (`.ljung_box()`, `.arch_lm()`, `.adf_residuals()`, `.kpss_residuals()`, `.acf()`, `.pacf()`), the model-fit scalars (`.loglikelihood()`, `.aic()`, `.bic()`), and `.summary()` — a printable parameter / diagnostics table with mean-equation, residual-distribution, and residual-diagnostics sections plus R-style significance codes. Standard errors come from the inverse observed Hessian (`cov_type="classic"`) at the constrained MLE; every diagnostic and the three model-fit scalars are computed at fit time on the standardised residuals and bundled into the single canonical `residual_diagnostics_` dict (keys `loglikelihood` / `aic` / `bic` / `acf` / `pacf` / `ljung_box` / `ljung_box_sq` / `arch_lm` / `adf` / `kpss`), so passing no series to any of these accessors returns the cached value, while passing a series recomputes against it. `.residuals(y)` returns the uniform `{"residuals": ε_t, "standardised_residuals": z_t}` dict — same schema across ARMA / GARCH / ArmaGarch.

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

Fitted instances expose the same inferential surface as the mean models — `.standard_errors()`, `.confidence_intervals()`, the four residual-diagnostic accessors, and `.summary()`. The summary table groups parameters under a `Variance equation — <ClassName>(p, q)` section header, then a residual-distribution section (suppressed for `normal`), then the cached residual-diagnostics block. SE flavour is again the inverse observed Hessian.

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

Fitted instances expose `.standard_errors()`, `.confidence_intervals()`, the four residual-diagnostic accessors, and `.summary()`. The composite's summary lays out three labelled param sections (mean equation, variance equation, residual distribution) plus the residual-diagnostics block. SE flavour defaults to the Bollerslev-Wooldridge robust sandwich (`cov_type="robust"` on `.standard_errors()` / `.cov_matrix()`) — the standalone variance-stage routes use the classic / observed-Hessian form; if you fit ARMA and then a GARCH on its residuals separately, pair the second stage with [`two_stage_cov`](#two-stage-standard-errors) to correct for first-stage noise.

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
