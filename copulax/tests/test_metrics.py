# import numpy as np
# import jax
# import jax.numpy as jnp


# from copulax.metrics import kl_divergence
# from copulax.univariate import normal, student_t, uniform


# def test_kl_divergence():
#     # testing values and gradients
#     dists_tup: tuple = (normal, uniform, student_t)

#     for dist1 in dists_tup:
#         # getting sample from P
#         if dist1.dist_type == "univariate":
#             # univariate dist
#             sample = dist1.rvs(shape=(100,))
#         else:
#             # multivariate dist
#             sample = dist1.rvs(size=100)

#         # fitting distributions
#         params1: dict = dist1.fit(sample)

#         for dist2 in dists_tup:
#             if dist1.dtype != dist2.dtype:
#                 # only comparing univariate to univariate and multivariate to multivariate
#                 continue

#             if dist1.name == dist2.name:
#                 params2: dict = params1
#             else:
#                 params2: dict = dist2.fit(sample)

#             kl_div1: float = kl_divergence(logpdf_P=dist1.logpdf, logpdf_Q=dist2.logpdf, params_P=params1, params_Q=params2, x=sample)
#             kl_div2: float = kl_divergence(logpdf_Q=dist1.logpdf, logpdf_P=dist2.logpdf, params_Q=params1, params_P=params2, x=sample)

#             assert isinstance(kl_div1, jnp.ndarray) and isinstance(kl_div2, jnp.ndarray), f"kl_divergence not a array for {dist1.name} and {dist2.name}"
#             assert kl_div1.shape == () and kl_div2.shape == (), f"kl_divergence not scalar for {dist1.name} and {dist2.name}"
#             assert kl_div1 >= 0, f"kl_divergence not positive for {dist1.name} and {dist2.name}"
#             assert (not np.isnan(kl_div1)) and (not np.isnan(kl_div2)), f"kl_divergence contains NaNs for {dist1.name} and {dist2.name}"

#             if dist1 == dist2:
#                 assert kl_div1 == kl_div2 == 0, f"kl_divergence not zero for same distributions {dist1.name} and {dist2.name}"

            