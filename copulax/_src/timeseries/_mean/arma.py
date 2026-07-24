"""ARMA(p, q) mean-equation model — full implementation.

Concrete user-facing entry point for the autoregressive
moving-average mean model.  All actual machinery lives in
:class:`copulax._src.timeseries._mean._arma_base.ARMABase`; this
module simply provides the public class.

Example:
    >>> from copulax.univariate import normal
    >>> from copulax.timeseries import ARMA
    >>> import jax.numpy as jnp, jax
    >>> y = jax.random.normal(jax.random.PRNGKey(0), (500,))
    >>> fit = ARMA(p=1, q=1, residual_dist=normal).fit(y)  # doctest: +SKIP
    >>> fit.params  # doctest: +SKIP
    {'phi': ..., 'theta': ..., 'mu': ..., 'sigma_eps': ..., 'residual': ...}

Cross-validation: parameter estimates from this fit match
``statsmodels.tsa.arima.ARIMA(y, order=(p, 0, q))`` to the
documented tolerances under correctly-specified data.
"""

from __future__ import annotations

from typing import Optional

from copulax._src._distributions import Univariate
from copulax._src.timeseries._mean._arma_base import ARMABase, ARMATerminalState


class ARMA(ARMABase):
    r"""ARMA(p, q) mean-equation model.

    Construct with the desired orders and residual law:

    .. code-block:: python

        from copulax.timeseries import ARMA
        from copulax.univariate import normal
        model = ARMA(p=1, q=1, residual_dist=normal)
        fit = model.fit(y)

    Inherits :meth:`fit` / :meth:`forecast` / :meth:`residuals` /
    :meth:`stats` etc. from :class:`ARMABase` — see that class for
    the full method contract.

    .. math::

        y_t = \mu + \sum_{i=1}^p \phi_i\, (y_{t-i} - \mu)
                  + \sum_{j=1}^q \theta_j\, \varepsilon_{t-j}
                  + \varepsilon_t,
        \qquad
        \varepsilon_t = \sigma_\varepsilon\, z_t,
        \qquad
        z_t \sim f_z\,(\text{mean}=0, \mathrm{var}=1).

    The intercept :math:`\mu` is the unconditional mean of the
    process (centred / Box-Jenkins / Hamilton convention; matches
    rugarch and ``statsmodels.tsa.arima.ARIMA``).

    References
    ----------
    .. [1] Box, G.E.P. & Jenkins, G.M. (1970). *Time Series Analysis:
       Forecasting and Control*. Holden-Day. (Centred / mean-adjusted
       ARMA(p, q) form; :math:`\mu` is the unconditional mean.)
    .. [2] Hamilton, J.D. (1994). *Time Series Analysis*, ch. 3-5.
       Princeton University Press. (ARMA(p, q) recursion sec. 3.4;
       conditional maximum-likelihood sec. 5.2; standard errors sec. 5.8;
       exact unconditional variance via Yule-Walker, e.g. ARMA(1,1)
       :math:`\sigma_\varepsilon^2 (1 + 2\phi\theta + \theta^2)/(1 - \phi^2)`.)
    """

    def __init__(
        self,
        p: int = 0,
        q: int = 0,
        *,
        residual_dist: Optional[Univariate] = None,
        name: str = "ARMA",
        phi=None,
        theta=None,
        mu=None,
        sigma_eps=None,
        residual_params=None,
        terminal_state: Optional[ARMATerminalState] = None,
        n_train_: Optional[int] = None,
        cov_matrix_=None,
        standard_errors_=None,
        residual_diagnostics_=None,
    ):
        super().__init__(
            name=name,
            p=p,
            q=q,
            residual_dist=residual_dist,
            phi=phi,
            theta=theta,
            mu=mu,
            sigma_eps=sigma_eps,
            residual_params=residual_params,
            terminal_state=terminal_state,
            n_train_=n_train_,
            cov_matrix_=cov_matrix_,
            standard_errors_=standard_errors_,
            residual_diagnostics_=residual_diagnostics_,
        )
