r"""The normal / Gaussian distribution is a continuous 'bell shaped' 2 
parameter family.

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
from copulax._src.univariate.normal import (
    support,
    logpdf,
    pdf,
    logcdf,
    cdf,
    ppf,
    rvs,
    fit,
    stats,
    loglikelihood,
    aic,
    bic,
)
from copulax._src.univariate.normal import ppf as inverse_cdf
from copulax._src.univariate.normal import rvs as sample
dtype = "continuous"
dist_type = "univariate"
name = "normal"