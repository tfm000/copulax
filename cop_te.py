import jax.numpy as jnp
from jax import jit

from copulax.copulas import gaussian_copula, student_t_copula, gh_copula
from copulax.univariate import normal, student_t, gamma
from copulax.multivariate import mvt_normal, mvt_gh
from copulax._src.multivariate._shape import _corr
from copulax.multivariate import corr, cov
from copulax import *
from copulax.univariate import gh, student_t, gamma

t_params={'nu': 3.0, 'mu': 0.0, 'sigma': 1.0}
g_params = {'alpha': 2.0, 'beta': 1.0}
rvs=gh.rvs(size=1000)
breakpoint()

mu = jnp.array([1.5, -6.3, 2.0])
sigma_incomplete=jnp.array([[4.3, 3.45, 0.35], 
                            [3.35, 1.0, 0.3], 
                            [0.35, 0.3, 1.0]])
sigma = _corr._rm_incomplete(sigma_incomplete, 1e-5)  # 1e-11 too small precision for float32 -> only 6-7 digits of precision
sample_raw = mvt_normal.rvs(size=1000, mu=mu, sigma=sigma)
sample = sample_raw.at[:, 2].set(jnp.abs(sample_raw[:, 2]))  # make sure the third variable is positive
sample_corr = jit(corr, static_argnums=(1,))(sample, 'laloux_pp_kendall')
# params = mvt_gh.fit(sample)
# breakpoint()
# params
marginals = (
    (normal, {"mu": 1.3, "sigma": 2.5}), 
    (student_t, {"nu": 5.0, "mu": -5.7, "sigma": 1.1}), 
    (gamma, {"alpha": 2.1, "beta": 1.04}),
    )

copula = {
    "mu": jnp.array([0.0, 0.0, 0.0]), 
    "sigma": jnp.array([[1.0, 0.5, 0.3], 
                        [0.5, 1.0, 0.2], 
                        [0.3, 0.2, 1.0]]),
          }

params = {"marginals": marginals, "copula": copula}

support = jit(gaussian_copula.support)(params)
print("Support:\n", support)

u_sample = jit(gaussian_copula.get_u)(sample, params)
print("u sample:\n", u_sample)

x_dash_sample = gaussian_copula.get_x_dash(u_sample, params)
print("x' sample:\n", x_dash_sample)

copula_logpdf = jit(gaussian_copula.copula_logpdf)(u_sample, params)
print("Copula logpdf:\n", copula_logpdf)

logpdf = jit(gaussian_copula.logpdf)(sample, params)
print("Logpdf:\n", logpdf)

copula_rvs = jit(gaussian_copula.copula_rvs, static_argnames=("size",))(size=1000, params=params)
print("Copula rvs:\n", copula_rvs)

rvs = jit(gaussian_copula.rvs, static_argnames=("size",))(size=1000, params=params)
print("Rvs:\n", rvs)

fitted_marginals = gaussian_copula.fit_marginals(sample)
print("Fitted marginals:\n", fitted_marginals)

fitted_copula = gaussian_copula.fit_copula(u_sample)
print("Fitted copula:\n", fitted_copula)

fitted_joint = gaussian_copula.fit(sample, corr_method='laloux_pp_kendall')
print("Fitted joint:\n", fitted_joint)

breakpoint()