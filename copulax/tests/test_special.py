# import numpy as np
# import jax
# import jax.numpy as jnp

# from copulax.special import kv



# def test_kv():
#     # checking values and gradients
#     kv_vg = jax.jit(jax.value_and_grad(kv, argnums=(0, 1)))
#     for v in np.linspace(-10, 10, 100):
#         for x in np.linspace(1e-10, 10, 100):
#             val, grad = kv_vg(v, x)

#             assert val >= 0, f"kv not positive for v={v}, x={x}"
#             assert not np.isnan(val), f"kv contains NaNs for v={v}, x={x}"
#             assert (not np.isnan(grad[0])) and (not np.isnan(grad[1])), f"kv gradient contains NaNs for v={v}, x={x}"

#     # checking shape
#     xs = [jnp.array([[1.0, 2.0, 3.0], [2.3, 3.4, 5.5]]), jnp.array([1.0, 2.0, 3.0]), jnp.array(1.0)]
#     for x in xs:
#         val = kv(0.5, x)
#         assert val.shape == x.shape, f"kv shape mismatch for x={x}"