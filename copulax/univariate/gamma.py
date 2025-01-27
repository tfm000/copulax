r"""The gamma distribution is a two-parameter family of continuous probability
distributions, which includes the exponential, Erlang and chi-squared 
distributions as special cases.

We use the rate parameterization of the gamma distribution specified by 
McNeil et al (2005).
https://en.wikipedia.org/wiki/Gamma_distribution

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
"""


from copulax._src.univariate.gamma import (
    support,
    logpdf,
    pdf,
    logcdf,
    cdf,
    ppf,
    fit,
    rvs,
    stats,
    )
from copulax._src.univariate.gamma import ppf as inverse_cdf
from copulax._src.univariate.gamma import rvs as sample