r"""The skewed-t distribution is a generalisation of the continuous Student's 
t-distribution that allows for skewness. It can also be expressed as a limiting 
case of the Generalized Hyperbolic distribution when phi -> 0 in addition to 
lambda = -0.5*chi.

We use the 4 parameter McNeil et al (2005) specification of the distribution

Currently the following methods are implemented:
- support
- logpdf
- pdf
- logcdf
- cdf
- ppf
- inverse_cdf
- rvs
- sample
- fit
- stats
- loglikelihood
- aic
- bic
- dtype
- dist_type
- name
"""


from copulax._src.univariate.skewed_t import (
    support,
    logpdf, 
    pdf,
    cdf,
    logcdf,
    ppf,
    rvs,
    fit,
    stats,
    loglikelihood,
    aic,
    bic,
    )
from copulax._src.univariate.skewed_t import ppf as inverse_cdf
from copulax._src.univariate.skewed_t import rvs as sample
dtype = "continuous"
dist_type = "univariate"
name = "skewed_t"