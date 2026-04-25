Copula Distributions
====================

.. automodule:: copulax.copulas
   :members:
   :undoc-members:

Copula Base
-----------

The universal abstract base class shared by every copula family
(Mean-Variance, Archimedean).

.. automodule:: copulax._src.copulas._distributions
   :members:
   :undoc-members:

Mean-Variance Copulas
---------------------

Copulas derived from normal mixture distributions (McNeil, Frey,
Embrechts 2005, §3.2).  ``MeanVarianceCopulaBase`` is the umbrella;
:class:`EllipticalCopula` covers normal *variance* mixtures
(Gaussian, Student-T; γ=0, strictly elliptical); :class:`MeanVarianceCopula`
covers normal *mean-variance* mixtures (GH, Skewed-T; γ≠0).

.. automodule:: copulax._src.copulas._mv_copulas
   :members:
   :undoc-members:

Archimedean Copulas
-------------------

Copulas defined by a generator function φ and its inverse ψ.

.. automodule:: copulax._src.copulas._archimedean
   :members:
   :undoc-members:
