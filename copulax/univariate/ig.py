r"""The inverse gamma distribution is a two-parameter family of continuous 
probability distributions which represents the reciprocal of gamma distributed 
random variables.

We use the rate parameterization of the inverse gamma distribution specified by
McNeil et al (2005).
https://en.wikipedia.org/wiki/Inverse-gamma_distribution

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


from copulax._src.univariate.ig import (
    support,
    logpdf,
    pdf,
    logcdf,
    cdf,
    ppf,
    fit,
    rvs,
    stats,
    loglikelihood,
    aic,
    bic,
    )
from copulax._src.univariate.ig import ppf as inverse_cdf
from copulax._src.univariate.ig import rvs as sample
dtype = "continuous"
dist_type = "univariate"
name = "ig"