Getting Started
===============

Installation
------------

CopulAX is available on PyPI:

.. code-block:: bash

   pip install copulax

To install with documentation build dependencies:

.. code-block:: bash

   pip install copulax[docs]

Documentation
-------------

- Read the Docs: https://copulax.readthedocs.io/en/latest/
- API Reference: https://copulax.readthedocs.io/en/latest/api/index.html

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

   import jax.random as jr
   from copulax.univariate import univariate_fitter, normal

   key = jr.PRNGKey(0)

   # Fit a normal distribution
   data = jr.normal(key, shape=(500,))
   fitted = normal.fit(data)
   print(fitted.stats())

   # Automatic distribution selection
   best_idx, results = univariate_fitter(data)

Multivariate fitting
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax.random as jr
   from copulax.multivariate import mvt_normal

   key = jr.PRNGKey(1)
   data = jr.normal(key, shape=(500, 3))
   fitted = mvt_normal.fit(data)
   samples = fitted.rvs(size=50)

Copula fitting
^^^^^^^^^^^^^^

.. code-block:: python

   import jax.random as jr
   from copulax.copulas import gaussian_copula

   key = jr.PRNGKey(2)
   data = jr.normal(key, shape=(500, 3))
   fitted = gaussian_copula.fit(data)
   samples = fitted.rvs(size=50)

Archimedean copulas
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from copulax.copulas import clayton_copula
   from copulax import get_random_key

   key = get_random_key()

   # sample from a Clayton copula
   params = clayton_copula.example_params(dim=3)
   u = clayton_copula.copula_rvs(size=200, params=params, key=key)

   # evaluate copula CDF, PDF and log-PDF
   cdf = clayton_copula.copula_cdf(u, params=params)
   pdf = clayton_copula.copula_pdf(u, params=params)

   # fit a copula to uniform data
   fitted = clayton_copula.fit_copula(u)

   # model selection with AIC / BIC
   aic = clayton_copula.aic(u, params=params)
   bic = clayton_copula.bic(u, params=params)

Saving and loading distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fitted distributions can be saved to disk and loaded back in a later
session.  All distribution types — univariate, multivariate, and copula
— are supported.  Files use the ``.cpx`` format and are cross-platform
(Windows, macOS, Linux).

.. code-block:: python

   import copulax
   from copulax.univariate import normal

   # Fit and save
   fitted = normal.fit(data)
   fitted.save("my_model.cpx")

   # Load (same or different session)
   loaded = copulax.load("my_model.cpx")
   loaded.logpdf(data)  # identical output

The ``name`` keyword on ``load`` lets you rename the instance on load:

.. code-block:: python

   loaded = copulax.load("my_model.cpx", name="production_model")

Copulas (including their fitted marginals) are saved and loaded in
exactly the same way:

.. code-block:: python

   from copulax.copulas import gaussian_copula

   fitted_cop = gaussian_copula.fit(data)
   fitted_cop.save("copula_model.cpx")
   loaded_cop = copulax.load("copula_model.cpx")

Testing efficiently
-------------------

The full test suite can take time. During development, run only affected tests
and prefer single test functions while iterating.

.. code-block:: bash

   # specific test function
   pytest copulax/tests/copulas/test_copulas.py::TestFitting::test_fit -v

   # affected file only
   pytest copulax/tests/copulas/test_copulas.py -v

.. code-block:: powershell

   # keep an append-only log while iterating
   pytest copulax/tests/copulas/test_copulas.py::TestFitting::test_fit -v *>&1 `
     | Tee-Object -FilePath copula_test_results.txt -Append
