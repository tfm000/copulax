r"""The log-normal distribution of X is one where Y = log(X) is normally 
distributed. It is a continuous 2 parameter family of distributions.

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


from copulax._src.univariate.lognormal import (
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
from copulax._src.univariate.lognormal import ppf as inverse_cdf
from copulax._src.univariate.lognormal import rvs as sample
dtype = "continuous"
dist_type = "univariate"
name = "lognormal"