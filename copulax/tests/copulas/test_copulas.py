# """Tests for copula distributions."""
# import pytest
# import jax.numpy as jnp
# from jax import grad, jit
# import numpy as np

# from copulax._src._distributions import Univariate
# from copulax.tests.helpers import *
# from copulax.tests.copulas.conftest import get_all_args, NUM_ASSETS
# from copulax._src.typing import Scalar

# # Helper functions for testing copula distributions
# @jit
# def jitable(data, dist, params: dict):
#     return dist.get_u(data, params=params)


# def gradients(func, params: dict, s, data):
#     """Calculate the gradients of the output."""
#     new_func = lambda x: func(x, params=params).sum()
#     grad_output = grad(new_func)(data)
#     assert no_nans(grad_output), f"{s} gradient contains NaNs"
#     assert is_finite(grad_output), f"{s} gradient contains non-finite values"


# # Test combinations
# all_args: dict = get_all_args()


# # Tests for copula distributions
# @pytest.mark.parametrize("dist, dist_args", all_args.items())
# def test_all_methods_implemented(dist, dist_args):
#     methods: set[str] = {
#         'dtype', 'dist_type', 'name', 'support', 'get_u', 'get_x_dash', 
#         'copula_logpdf', 'copula_pdf', 'logpdf', 'pdf', 'copula_rvs', 
#         'copula_sample', 'rvs', 'sample', 'fit_marginals', 'fit_copula', 
#         'fit', 'aic', 'bic', 'loglikelihood', 'stats',
#     }
#     # testing desired methods are implemented
#     for method in methods:
#         assert hasattr(dist, method), f"{dist} missing method {method}"

#     # testing no additional methods are implemented
#     pytree_methods: set[str] = {'tree_flatten', 'tree_unflatten'}
#     extra_methods = set(dist.__dict__.keys()) - methods - pytree_methods
#     extra_methods = {m for m in extra_methods if not m.startswith('_')}
#     assert not extra_methods, f"{dist} has extra methods: {extra_methods}"


# @pytest.mark.parametrize("dist, dist_args", all_args.items())
# def test_name(dist, dist_args):
#     assert isinstance(dist.name, str), f"{dist} name is not a string"
#     assert dist.name != "", f"{dist} name is an empty string"
#     assert dist.name == str(dist), f"{dist} name does not match its string representation"


# @pytest.mark.parametrize("dist, dist_args", all_args.items())
# def test_dist_object_jitable(dist, dist_args, continuous_sample):
#     params: dict = dist_args['params']
#     jitable(data=continuous_sample, dist=dist, params=params)


# @pytest.mark.parametrize("dist, dist_args", all_args.items())
# def test_dtype(dist, dist_args):
#     assert isinstance(dist.dtype, str), f"{dist} dtype is not a string"
#     assert dist.dtype != "", f"{dist} dtype is an empty string"
#     assert dist.dtype in ('continuous', 'discrete'), f"dtype is not 'continuous' or 'discrete' for {dist}"


# @pytest.mark.parametrize("dist, dist_args", all_args.items())
# def test_dist_type(dist, dist_args):
#     assert isinstance(dist.dist_type, str), f"{dist} dist_type is not a string"
#     assert dist.dist_type != "", f"{dist} dist_type is an empty string"
#     assert dist.dist_type == 'copula', f"dist_type is not 'copula' for {dist}"


# @pytest.mark.parametrize("dist, dist_args", all_args.items())
# def test_support(dist, dist_args):
#     params = dist_args['params']
#     support = dist.support(params)
#     # Checking properties
#     assert isinstance(support, jnp.ndarray), f"{dist} support is not a JAX array"
#     assert two_dim(support), f"{dist} support is not a 2D array"
#     assert np.all(support[:, 0] < support[:, 1]), f"{dist} support bounds are not in order."
#     assert no_nans(support), f"{dist} support contains NaNs."
#     # Check jit
#     jitted_support = jit(dist.support)(params)


# @pytest.mark.parametrize("dist, dist_args", all_args.items())
# def test_get_u(dist, dist_args, continuous_sample):
#     params = dist_args['params']
#     u = dist.get_u(continuous_sample, params)
#     # Checking properties
#     assert isinstance(u, jnp.ndarray), f"{dist} u is not a JAX array"
#     assert two_dim(u), f"{dist} u is not a 2D array"
#     assert np.all(u >= 0) and np.all(u <= 1), f"{dist} u is not in [0, 1]"
#     assert no_nans(u), f"{dist} u contains NaNs."
#     # Check jit
#     jitted_u = jit(dist.get_u)(continuous_sample, params)
#     # Check gradients
#     gradients(dist.get_u, params, f"{dist} u", continuous_sample)

# @pytest.mark.parametrize("dist, dist_args", all_args.items())
# def test_get_x_dash(dist, dist_args, u_sample):
#     params = dist_args['params']
#     x_dash = dist.get_x_dash(u_sample, params)
#     # Checking properties
#     assert isinstance(x_dash, jnp.ndarray), f"{dist} x_dash is not a JAX array"
#     assert two_dim(x_dash), f"{dist} x_dash is not a 2D array"
#     assert no_nans(x_dash), f"{dist} x_dash contains NaNs."
#     # Check jit
#     jitted_x_dash = jit(dist.get_x_dash)(u_sample, params)
#     # Check gradients
#     gradients(dist.get_x_dash, params, f"{dist} x_dash", u_sample)


# @pytest.mark.parametrize("dist, dist_args", all_args.items())
# def test_copula_logpdf(dist, dist_args, u_sample):
#     params = dist_args['params']
#     logpdf = dist.copula_logpdf(u_sample, params)
#     # Checking properties
#     assert correct_mvt_shape(u_sample, logpdf), f"{dist} copula_logpdf has incorrect shape."
#     assert no_nans(logpdf), f"{dist} copula_logpdf contains NaNs."
#     # Check jit
#     jitted_logpdf = jit(dist.copula_logpdf)(u_sample, params)
#     # Check gradients
#     gradients(dist.copula_logpdf, params, f"{dist} copula_logpdf", u_sample)


# @pytest.mark.parametrize("dist, dist_args", all_args.items())
# def test_copula_pdf(dist, dist_args, u_sample):
#     params = dist_args['params']
#     pdf = dist.copula_pdf(u_sample, params)
#     # Checking properties
#     assert correct_mvt_shape(u_sample, pdf), f"{dist} copula_pdf has incorrect shape."
#     assert is_positive(pdf), f"{dist} copula_pdf is not positive."
#     assert no_nans(pdf), f"{dist} copula_pdf contains NaNs."
#     assert is_finite(pdf), f"{dist} copula_pdf contains non-finite values."
#     # Check jit
#     jitted_pdf = jit(dist.copula_pdf)(u_sample, params)
#     # Check gradients
#     gradients(dist.copula_pdf, params, f"{dist} copula_pdf", u_sample)


# @pytest.mark.parametrize("dist, dist_args", all_args.items())
# def test_logpdf(dist, dist_args, continuous_sample):
#     params = dist_args['params']
#     logpdf = dist.logpdf(continuous_sample, params)
#     # Checking properties
#     assert correct_mvt_shape(continuous_sample, logpdf), f"{dist} logpdf has incorrect shape."
#     assert no_nans(logpdf), f"{dist} logpdf contains NaNs."
#     # Check jit
#     jitted_logpdf = jit(dist.logpdf)(continuous_sample, params)
#     # Check gradients
#     gradients(dist.logpdf, params, f"{dist} logpdf", continuous_sample)


# @pytest.mark.parametrize("dist, dist_args", all_args.items())
# def test_pdf(dist, dist_args, continuous_sample):
#     params = dist_args['params']
#     pdf = dist.pdf(continuous_sample, params)
#     # Checking properties
#     assert correct_mvt_shape(continuous_sample, pdf), f"{dist} pdf has incorrect shape."
#     assert is_positive(pdf), f"{dist} pdf contains negative values."
#     assert no_nans(pdf), f"{dist} pdf contains NaNs."
#     assert is_finite(pdf), f"{dist} pdf contains non-finite values."
#     # Check jit
#     jitted_pdf = jit(dist.pdf)(continuous_sample, params)
#     # Check gradients
#     gradients(dist.pdf, params, f"{dist} pdf", continuous_sample)


# SIZES = (0, 1, 2, 11)
# SIZE_COMBINATIONS = [(dist, size) for dist in all_args.keys() for size in SIZES]
# @pytest.mark.parametrize("dist, size", SIZE_COMBINATIONS)
# def test_copula_rvs(dist, size):
#     params = all_args[dist]['params']
#     rvs = dist.copula_rvs(size=size, params=params)
#     # Checking properties
#     assert two_dim(rvs), f"{dist} copula_rvs is not a 2D array"
#     expected_shape = (size, NUM_ASSETS)
#     assert rvs.shape == expected_shape, f"{dist} copula_rvs has incorrect shape. Expected {expected_shape}, got {rvs.shape}"
#     assert no_nans(rvs), f"{dist} copula_rvs contains NaNs for size {size}"
#     assert is_finite(rvs), f"{dist} copula_rvs contains infinite values for size {size}"
#     # Check jit
#     jitted_rvs = jit(dist.copula_rvs, static_argnums=0)(size=size, params=params)


# @pytest.mark.parametrize("dist, size", SIZE_COMBINATIONS)
# def test_rvs(dist, size):
#     params = all_args[dist]['params']
#     rvs = dist.rvs(size=size, params=params)
#     # Checking properties
#     assert two_dim(rvs), f"{dist} rvs is not a 2D array"
#     expected_shape = (size, NUM_ASSETS)
#     assert rvs.shape == expected_shape, f"{dist} rvs has incorrect shape. Expected {expected_shape}, got {rvs.shape}"
#     assert no_nans(rvs), f"{dist} rvs contains NaNs for size {size}"
#     assert is_finite(rvs), f"{dist} rvs contains infinite values for size {size}"
#     # Check jit
#     jitted_rvs = jit(dist.rvs, static_argnums=0)(size=size, params=params)


# def _check_fitted_marginals(dist, sample, fitted_marginals):
#     # Checking properties
#     assert isinstance(fitted_marginals, dict), f"{dist} fitted_marginals is not a dictionary."
#     assert len(fitted_marginals) == 1, f"{dist} fitted_marginals does not have length 1."
#     assert 'marginals' in fitted_marginals, f"{dist} fitted_marginals does not have 'marginals' key."
#     marginals: tuple = fitted_marginals['marginals']
#     assert isinstance(marginals, tuple), f"{dist} fitted_marginals subdict is not a tuple."
#     fitted_num_dims = len(marginals)
#     assert fitted_num_dims == NUM_ASSETS, f"{dist} fitted_marginals has not fitted the correct number of dimensions. Expected {num_dims}, got {fitted_num_dims}."
#     for marginal_tup in marginals:
#         assert isinstance(marginal_tup, tuple), f"{dist} underlying fitted_marginals are not in tuple form."
#         assert len(marginal_tup) == 2, f"{dist} underlying fitted_marginals tuples do not have length 2."
#         marginal_dist, marginal_params = marginal_tup
#         assert isinstance(marginal_dist, Univariate), f"{dist} underlying fitted_marginals distribution is not a Univariate object."
#         assert isinstance(marginal_params, dict), f"{dist} underlying fitted_marginals parameters are not in dictionary form."
#         # TODO: add check valid params


# @pytest.mark.parametrize("dist", all_args.keys())
# def test_fit_marginals(dist, continuous_sample):
#     fitted_marginals = dist.fit_marginals(continuous_sample)
#     # Checking properties
#     _check_fitted_marginals(dist, continuous_sample, fitted_marginals)


# def _check_fitted_copula(dist, fitted_copula):
#     # Checking properties
#     assert isinstance(fitted_copula, dict), f"{dist} fitted_copula is not a dictionary."
#     assert len(fitted_copula) == 1, f"{dist} fitted_copula does not have length 1."
#     assert 'copula' in fitted_copula, f"{dist} fitted_copula does not have 'copula' key."
#     copula_params = fitted_copula['copula']
#     assert isinstance(copula_params, dict), f"{dist} fitted_copula parameters are not in dictionary form."
#     # TODO: add check valid params
#     # TODO: add check for correct dims


# @pytest.mark.parametrize("dist", all_args.keys())
# def test_fit_copula(dist, u_sample):
#     fitted_copula = dist.fit_copula(u_sample)
#     # Checking properties
#     _check_fitted_copula(dist, fitted_copula)
#     # Check jit
#     jitted_fitted_copula = jit(dist.fit_copula)(u_sample)


# @pytest.mark.parametrize("dist", all_args.keys())
# def test_fit(dist, continuous_sample):
#     fitted_joint = dist.fit(continuous_sample)
#     # Checking properties
#     assert isinstance(fitted_joint, dict), f"{dist} fit does not return a dictionary"
#     assert len(fitted_joint) == 2, f"{dist} fit does not have length 2."
#     assert 'marginals' in fitted_joint and 'copula' in fitted_joint, f"{dist} fit does not have 'marginals' and 'copula' keys."
#     fitted_marginals = fitted_joint.copy()
#     fitted_marginals.pop('copula')
#     _check_fitted_marginals(dist, continuous_sample, fitted_marginals)
#     fitted_copula = fitted_joint.copy()
#     fitted_copula.pop('marginals')
#     _check_fitted_copula(dist, fitted_copula)


# # TODO: implement aic, bic, loglikelihood
# # @pytest.mark.parametrize("dist, dist_args", all_args.items())
# # def test_aic(dist, continuous_sample):
# #     params = all_args[dist]['params']
# #     aic = dist.aic(continuous_sample, params)
# #     assert isinstance(aic, Scalar), f"{dist} aic is not a float"
# #     assert is_scalar(aic), f"{dist} aic is not a scalar"
# #     assert no_nans(aic), f"{dist} aic contains NaNs"
# #     assert is_positive(aic), f"{dist} aic is negative"


# # @pytest.mark.parametrize("dist, dist_args", all_args.items())
# # def test_bic(dist, continuous_sample):
# #     params = all_args[dist]['params']
# #     bic = dist.bic(continuous_sample, params)
# #     assert isinstance(bic, Scalar), f"{dist} bic is not a float"
# #     assert is_scalar(bic), f"{dist} bic is not a scalar"
# #     assert no_nans(bic), f"{dist} bic contains NaNs"
# #     assert is_positive(bic), f"{dist} bic is negative"


# # @pytest.mark.parametrize("dist, dist_args", all_args.items())
# # def test_loglikelihood(dist, continuous_sample):
# #     params = all_args[dist]['params']
# #     loglikelihood = dist.loglikelihood(continuous_sample, params)
# #     assert isinstance(loglikelihood, Scalar), f"{dist} loglikelihood is not a float"
# #     assert is_scalar(loglikelihood), f"{dist} loglikelihood is not a scalar"
# #     assert no_nans(loglikelihood), f"{dist} loglikelihood contains NaNs"
# #     assert is_finite(loglikelihood), f"{dist} loglikelihood contains non-finite values"