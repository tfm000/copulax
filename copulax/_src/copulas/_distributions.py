from copulax._src._distributions import Copula
from copulax._src.multivariate.mvt_normal import mvt_normal
from copulax._src.multivariate.mvt_student_t import mvt_student_t
from copulax._src.multivariate.mvt_gh import mvt_gh

from copulax._src.univariate.normal import normal
from copulax._src.univariate.student_t import student_t
from copulax._src.univariate.gh import gh

gaussian_copula = Copula('Gaussian-Copula', mvt_normal, normal)
student_t_copula = Copula('Student-T-Copula', mvt_student_t, student_t)
gh_copula = Copula('GH-Copula', mvt_gh, gh)