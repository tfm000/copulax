"""ARMA(p, q) mean-equation model — full implementation singleton.

Concrete user-facing entry point for the autoregressive
moving-average mean model.  The full implementation lives in
:class:`copulax._src.timeseries._mean._arma_base.ARMABase`; this
module simply provides the public class and singleton with their
display name set so :meth:`fit` / :meth:`forecast` / etc. produce
clearly-named fitted instances.

Example:
    >>> from copulax.univariate import normal
    >>> from copulax.timeseries import arma
    >>> import jax.numpy as jnp, jax
    >>> y = jax.random.normal(jax.random.PRNGKey(0), (500,))
    >>> fit = arma.fit(y, p=1, q=1, residual_dist=normal)  # doctest: +SKIP
    >>> fit.params  # doctest: +SKIP
    {'phi': ..., 'theta': ..., 'c': ..., 'sigma_eps': ..., 'residual': ...}

Cross-validation: parameter estimates from this fit match
``statsmodels.tsa.arima.ARIMA(y, order=(p, 0, q))`` to the
documented tolerances under correctly-specified data.
"""

from __future__ import annotations

from copulax._src.timeseries._mean._arma_base import ARMABase


class ARMA(ARMABase):
    r"""ARMA(p, q) mean-equation model.

    Inherits all behaviour from :class:`ARMABase` — see that class
    for the full :meth:`fit` / :meth:`forecast` / :meth:`residuals`
    / :meth:`stats` contract.  This subclass exists purely to give
    the public singleton its own concrete type so ``isinstance``
    checks downstream remain ergonomic.

    .. math::

        y_t = c + \sum_{i=1}^p \phi_i\, y_{t-i}
                 + \sum_{j=1}^q \theta_j\, \varepsilon_{t-j}
                 + \varepsilon_t,
        \qquad
        \varepsilon_t = \sigma_\varepsilon\, z_t,
        \qquad
        z_t \sim f_z\,(\text{mean}=0, \mathrm{var}=1).
    """


#: Singleton entry point for ARMA fitting.  Construct fitted models
#: via :meth:`ARMA.fit`; the singleton itself carries no parameters.
arma = ARMA("ARMA")
