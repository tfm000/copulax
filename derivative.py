# from jax import jit, random, grad

# from copulax.univariate import student_t, normal
# from copulax import DEFAULT_RANDOM_KEY


# n_params = {'mu': 1.8, 'sigma': 3.1}
t_params={'nu': 1.0, 'mu': 0.0, 'sigma': 1.0}
# # t_params = {'mu': -1.0, 'sigma': 2.5, 'nu': 5.0}

# dist = student_t
# dist_params = t_params

# func = lambda q, p: dist.ppf(q, p).sum()

# uvars = random.uniform(DEFAULT_RANDOM_KEY, shape=(1,))

# ppf_grad = grad(func, argnums=[0, 1])(uvars, dist_params)
# print(ppf_grad)


from jax import jit, random, grad
import numpy as np
import matplotlib.pyplot as plt

from copulax.univariate import student_t, normal, skewed_t, univariate_fitter, gamma
from copulax import DEFAULT_RANDOM_KEY

# gp = {'alpha': 1.0, 'beta': 1.0}
# xrange = np.linspace(0, 10, 100)
# pdf = gamma.pdf(xrange, gp)
# plt.plot(xrange, pdf, label='PDF')
# plt.grid(True)
# plt.show()

xvars = random.uniform(DEFAULT_RANDOM_KEY, shape=(10,), minval=0, maxval=10)
index, fitted=univariate_fitter(xvars)
breakpoint()


n_params = {'mu': 0.0, 'sigma': 2.5}
t_params = {'nu': 1.0, 'mu': 0.0, 'sigma': 1.0}
st_params = skewed_t.example_params()
cubic = True
func = lambda q, p: skewed_t.ppf(q, p, params=st_params, cubic=cubic).sum()
func2 = lambda x, p: skewed_t.cdf(x, p).sum()
# ppf_grad = jit(grad(func, argnums=[0, 1]))
# ppf_grad = grad(func, argnums=[0, 1])
cdf_grad = grad(func2, argnums=[0, 1])
eps = 1e-4
# uvars = random.uniform(DEFAULT_RANDOM_KEY, shape=(1000,), minval=eps, maxval=1-eps)
xvars = random.uniform(DEFAULT_RANDOM_KEY, shape=(10,), minval=-10, maxval=10)
# print(ppf_grad(uvars, st_params))
print(cdf_grad(xvars, st_params))
# uvar_x_val = student_t.ppf(uvars, t_params, cubic=cubic)
# uvar_x_val = student_t.ppf(uvars, t_params, lr = 1.0, maxiter=100, cubic=cubic)


# vars = normal.rvs(size=(100,), **n_params)
# func2 = lambda x, p : student_t.cdf(x, p).sum()
# cdf_grad = grad(func2, argnums=[0, 1])
# cdf_grad_vals = cdf_grad(uvar_x_val, t_params)
# print(cdf_grad_vals)
# ppf_x_grad = 1/cdf_grad_vals[0]
# print(ppf_x_grad)

# plt.hist(uvar_x_val, density=True, bins=50)
# xrange = np.linspace(uvar_x_val.min() - 1, uvar_x_val.max() + 1, 1000)
# plt.plot(xrange, student_t.pdf(xrange, t_params), label='PDF')
# plt.grid()
# plt.show()