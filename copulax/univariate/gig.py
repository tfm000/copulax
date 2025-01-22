r"""The Generalized Inverse Gaussian distribution is a 3 parameter family of 
continuous distributions.

We use the McNeil et al (2005) specification of the distribution.

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


from copulax._src.univariate.gig import (
    support,
    logpdf,
    pdf,
    logcdf,
    cdf,
    ppf,
    rvs,
    fit,
    stats,
)
from copulax._src.univariate.gig import ppf as inverse_cdf
from copulax._src.univariate.gig import rvs as sample