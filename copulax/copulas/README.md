# Copula Distributions

This directory contains all implemented CopulAX copula distributions.

## Implemented Copula Distributions

### Elliptical Copulas
| Object / Module | Distribution |
| --- | --- |
| gaussian_copula | [Gaussian/Normal Copula](<https://en.wikipedia.org/wiki/Copula*(statistics)>)|
| student_t_copula | [Student's T Copula](https://en.wikipedia.org/wiki/Multivariate_t-distribution)|
| gh_copula | [Generalized Hyperbolic Copula](https://en.wikipedia.org/wiki/Generalised_hyperbolic_distribution)|
| skewed_t_copula |Skewed/Asymmetric Student's T Copula|

### Archimedean Copulas
| Object / Module | Distribution | θ range | Kendall's τ |
| --- | --- | --- | --- |
| clayton_copula | [Clayton Copula](https://en.wikipedia.org/wiki/Copula_(statistics)#Most_important_Archimedean_copulas) | (0, ∞) | θ/(θ+2) |
| frank_copula | [Frank Copula](https://en.wikipedia.org/wiki/Copula_(statistics)#Most_important_Archimedean_copulas) | ℝ \ {0} | 1 − 4/θ·(1−D₁(θ)) |
| gumbel_copula | [Gumbel Copula](https://en.wikipedia.org/wiki/Copula_(statistics)#Most_important_Archimedean_copulas) | [1, ∞) | 1 − 1/θ |
| joe_copula | [Joe Copula](https://en.wikipedia.org/wiki/Copula_(statistics)#Most_important_Archimedean_copulas) | [1, ∞) | series |
| amh_copula | [Ali-Mikhail-Haq Copula](https://en.wikipedia.org/wiki/Copula_(statistics)#Most_important_Archimedean_copulas) | [-1, 1) (d≤2) | analytical |
