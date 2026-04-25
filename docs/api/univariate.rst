Univariate Distributions
========================

Support Handling
----------------

Univariate distributions enforce support-aware outputs:

- ``logpdf(x)`` returns ``-inf`` outside support.
- ``cdf(x)`` returns ``0`` below support and ``1`` above support.
- fitting uses a penalized objective that discourages invalid
  parameter regions and non-finite likelihood contributions.

.. automodule:: copulax.univariate
   :members:
   :undoc-members:

Distribution Objects
--------------------

.. automodule:: copulax._src.univariate.normal
   :members:
   :undoc-members:

.. automodule:: copulax._src.univariate.student_t
   :members:
   :undoc-members:

.. automodule:: copulax._src.univariate.uniform
   :members:
   :undoc-members:

.. automodule:: copulax._src.univariate.gamma
   :members:
   :undoc-members:

.. automodule:: copulax._src.univariate.exponential
   :members:
   :undoc-members:

.. automodule:: copulax._src.univariate.lognormal
   :members:
   :undoc-members:

.. automodule:: copulax._src.univariate.ig
   :members:
   :undoc-members:

.. automodule:: copulax._src.univariate.gig
   :members:
   :undoc-members:

.. automodule:: copulax._src.univariate.gen_normal
   :members:
   :undoc-members:

.. automodule:: copulax._src.univariate.asym_gen_normal
   :members:
   :undoc-members:

.. automodule:: copulax._src.univariate.skewed_t
   :members:
   :undoc-members:

.. automodule:: copulax._src.univariate.gh
   :members:
   :undoc-members:

.. automodule:: copulax._src.univariate.nig
   :members:
   :undoc-members:

.. automodule:: copulax._src.univariate.wald
   :members:
   :undoc-members:

Fitting Utilities
-----------------

.. automodule:: copulax._src.univariate.univariate_fitter
   :members:
   :undoc-members:

Goodness of Fit
---------------

.. automodule:: copulax._src.univariate._gof
   :members:
   :undoc-members:
