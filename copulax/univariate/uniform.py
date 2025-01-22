r"""The continuous uniform distribution is a 2 parameter family of distributions 
where all intervals of the same length are equally probable.

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
from copulax._src.univariate.uniform import (
    support,
    logpdf,
    pdf,
    logcdf,
    cdf,
    ppf,
    rvs,
    fit,
    stats
)
from copulax._src.univariate.uniform import ppf as inverse_cdf
from copulax._src.univariate.uniform import rvs as sample
