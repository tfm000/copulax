r"""The Generalized Hyperbolic distribution is a flexible family of continuous
distributions, which includes the normal, student-T, skewed-T, NIG and 
Generalized Laplace distributions as special and limiting cases.

We use the 6 parameter McNeil et al (2005) specification of the distribution

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


from copulax._src.univariate.gh import (
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
from copulax._src.univariate.gh import ppf as inverse_cdf
from copulax._src.univariate.gh import rvs as sample
