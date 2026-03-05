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

Archimedean copulas
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from copulax.copulas import clayton_copula, independence_copula
   from copulax import get_random_key

   key = get_random_key()

   # sample from a Clayton copula
   params = clayton_copula.example_params(dim=3)
   u = clayton_copula.copula_rvs(size=200, params=params, key=key)

   # evaluate copula CDF, PDF and log-PDF
   cdf = clayton_copula.copula_cdf(u, params=params)
   pdf = clayton_copula.copula_pdf(u, params=params)

   # fit a copula to uniform data
   fitted = clayton_copula.fit_copula(u, key=key)

   # model selection with AIC / BIC
   aic = clayton_copula.aic(u, params=params)
   bic = clayton_copula.bic(u, params=params)
