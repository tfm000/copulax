# Copula Implementations

This directory contains the core copula distribution implementations in pure JAX.

## Architecture

### `CopulaBase` (`_distributions.py`)

Universal abstract base class for every copula family, implementing Sklar's theorem:

$$\log f(\mathbf{x}) = \log c(F_1(x_1), \ldots, F_d(x_d)) + \sum_{i=1}^{d} \log f_i(x_i)$$

Key functionality:

- **Marginal fitting** via `fit_marginals()` — fits univariate distributions to each dimension
- **Joint density** via `logpdf()` / `pdf()` — combines copula density with marginal densities
- **Sampling** via `rvs()` — generates copula samples then applies inverse marginal CDFs
- **Grouped marginal application** — efficiently batches operations across dimensions sharing the same distribution type

### Mean-Variance Copulas (`_mv_copulas.py`)

Copulas derived from normal mixture distributions (McNeil, Frey, Embrechts 2005, §3.2).  The hierarchy splits by whether the underlying mixture is symmetric (γ=0, strictly elliptical) or asymmetric (γ≠0, non-elliptical):

```
MeanVarianceCopulaBase   # umbrella: fit_copula dispatcher + correlation machinery
├── EllipticalCopula     # normal variance mixtures, γ=0
│   ├── GaussianCopula
│   └── StudentTCopula
└── MeanVarianceCopula   # normal mean-variance mixtures, γ≠0
    ├── GHCopula
    └── SkewedTCopula
```

`MeanVarianceCopulaBase.fit_copula` validates the user-supplied `method` against the subclass's `_supported_methods` frozenset and rejects inapplicable kwargs against `_METHOD_KWARGS` — no silent dropping.  Supported fitting methods:

- `fc_mle` — Fixed-Correlation MLE (Σ held at Kendall-τ estimate; shape-only MLE).  Available on every concrete subclass.
- `mle` — full joint MLE over Σ and shape parameters.  Mean-variance subclasses only.
- `ecme` — inner EM on (Σ, γ); outer MLE on remaining shape params on the original copula log-likelihood.
- `ecme_double_gamma` — like `ecme` but γ is additionally re-optimised in the outer numerical M-step.
- `ecme_outer_gamma` — inner EM on Σ only; outer MLE on all shape parameters including γ.

### `ArchimedeanCopula` (`_archimedean.py`)

Base class for Archimedean copulas defined by a generator function φ and its inverse ψ:

$$C(u_1, \ldots, u_d) = \psi(\phi(u_1) + \cdots + \phi(u_d))$$

Key abstractions:

- `generator(t, theta)` — the φ function
- `generator_inv(s, theta)` — the ψ function (pseudo-inverse)
- `_tau_to_theta(tau)` — Kendall's tau inversion for parameter fitting
- `_rvs_frailty(key, theta, size)` — frailty distribution sampling for the Marshall-Olkin algorithm

Archimedean `fit_copula` currently supports `method="kendall"` (average-pairwise Kendall's tau inversion).  Like the mean-variance side, unknown methods and inapplicable kwargs raise `ValueError` at the dispatcher.

**Implemented Archimedean copulas:**

| Class              | Copula          | Generator φ(t)              | Frailty V                 |
| ------------------ | --------------- | --------------------------- | ------------------------- |
| ClaytonCopula      | Clayton         | t^{−θ} − 1                  | Gamma(1/θ, 1)             |
| FrankCopula        | Frank           | −ln((e^{−θt}−1)/(e^{−θ}−1)) | Logarithmic(1−e^{−\|θ\|}) |
| GumbelCopula       | Gumbel          | (−ln t)^θ                   | Stable(1/θ)               |
| JoeCopula          | Joe             | −ln(1−(1−t)^θ)              | Sibuya(1/θ)               |
| AMHCopula          | Ali-Mikhail-Haq | ln((1−θ(1−t))/t)            | Geometric(1−θ)            |
| IndependenceCopula | Independence    | −ln(t)                      | Point mass at 1           |
