r"""The student-T distribution is a 3 parameter family of continuous 
distributions which generalize the normal distribuion, allowing it to have 
heavier tails.

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
from copulax._src.univariate.student_t import (
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
from copulax._src.univariate.student_t import ppf as inverse_cdf
from copulax._src.univariate.student_t import rvs as sample
dtype = "continuous"
dist_type = "univariate"
name = "student_t"
