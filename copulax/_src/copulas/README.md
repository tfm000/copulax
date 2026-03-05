# Copula Implementations

This directory contains the core copula distribution implementations in pure JAX.

## Architecture

### CopulaBase (`_distributions.py`)

Base class for all copula distributions, implementing Sklar's theorem:

$$\log f(\mathbf{x}) = \log c(F_1(x_1), \ldots, F_d(x_d)) + \sum_{i=1}^{d} \log f_i(x_i)$$

Key functionality:

- **Marginal fitting** via `fit_marginals()` — fits univariate distributions to each dimension
- **Joint density** via `logpdf()` / `pdf()` — combines copula density with marginal densities
- **Sampling** via `rvs()` — generates copula samples then applies inverse marginal CDFs
- **Grouped marginal application** — efficiently batches operations across dimensions sharing the same distribution type

### Elliptical Copulas (`_distributions.py`)

Copulas derived from elliptical multivariate distributions (Gaussian, Student-T, Generalized Hyperbolic, Skewed-T). These inherit the correlation structure from their parent multivariate distribution.

### ArchimedeanCopula (`_archimedean.py`)

Base class for Archimedean copulas defined by a generator function φ and its inverse ψ:

$$C(u_1, \ldots, u_d) = \psi(\phi(u_1) + \cdots + \phi(u_d))$$

Key abstractions:

- `generator(t, theta)` — the φ function
- `generator_inv(s, theta)` — the ψ function (pseudo-inverse)
- `_tau_to_theta(tau)` — Kendall's tau inversion for parameter fitting
- `_rvs_frailty(key, theta, size)` — frailty distribution sampling for the Marshall-Olkin algorithm

**Implemented Archimedean copulas:**

| Class              | Copula          | Generator φ(t)              | Frailty V                 |
| ------------------ | --------------- | --------------------------- | ------------------------- |
| ClaytonCopula      | Clayton         | t^{−θ} − 1                  | Gamma(1/θ, 1)             |
| FrankCopula        | Frank           | −ln((e^{−θt}−1)/(e^{−θ}−1)) | Logarithmic(1−e^{−\|θ\|}) |
| GumbelCopula       | Gumbel          | (−ln t)^θ                   | Stable(1/θ)               |
| JoeCopula          | Joe             | −ln(1−(1−t)^θ)              | Sibuya(1/θ)               |
| AMHCopula          | Ali-Mikhail-Haq | ln((1−θ(1−t))/t)            | Geometric(1−θ)            |
| IndependenceCopula | Independence    | −ln(t)                      | Point mass at 1           |
