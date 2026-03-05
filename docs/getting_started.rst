Getting Started
===============

Installation
------------

copulAX is available on PyPI:

.. code-block:: bash

   pip install copulax

To install with documentation build dependencies:

.. code-block:: bash

   pip install copulax[docs]

Requirements
------------

- Python >= 3.10
- JAX >= 0.4.38
- equinox >= 0.13.0
- optax >= 0.2.4
- quadax >= 0.2.8
- interpax >= 0.3.7
- matplotlib >= 3.9.2

Quick Start
-----------

Univariate fitting
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp
   from copulax.univariate import univariate_fitter, normal

   # Fit a normal distribution
   data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
   fitted = normal.fit(data)
   print(fitted.stats())

   # Automatic distribution selection
   best_idx, results = univariate_fitter(data)

Multivariate fitting
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp
   from copulax.multivariate import mvt_normal

   data = jnp.ones((100, 3))
   fitted = mvt_normal.fit(data)
   samples = fitted.rvs(size=50)

Copula fitting
^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp
   from copulax.copulas import gaussian_copula

   data = jnp.ones((100, 3))
   fitted = gaussian_copula.fit(data)
   samples = fitted.rvs(size=50)
