Preprocessing
=============

CopulAX provides a small set of jittable, autograd-compatible
preprocessing objects that compose cleanly with the rest of the
library. All preprocessors are :class:`equinox.Module` PyTrees — their
fitted parameters are traced JAX arrays, so they can be passed through
``jax.jit``, ``jax.grad``, ``jax.vmap``, and ``equinox`` serialisation.

Data Scaling
------------

:class:`~copulax.preprocessing.DataScaler` fits an affine rescaling to
input data and exposes ``transform`` / ``inverse_transform`` for later
observations. Four methods are supported (z-score, min-max, robust,
max-abs), all reducing to a uniform ``(x - offset) / scale``
representation. Optional user-supplied pre-/post- transform function
pairs allow, for example, z-score normalisation over log-transformed
data with a faithful round-trip.

.. automodule:: copulax.preprocessing
   :members:
   :undoc-members:

.. automodule:: copulax._src.preprocessing.data_scaler
   :members:
   :undoc-members:
