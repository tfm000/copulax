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
    {'phi': ..., 'theta': ..., 'c': ..., 'sigma_eps': ..., 'residual': ...}

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

        y_t = c + \sum_{i=1}^p \phi_i\, y_{t-i}
                 + \sum_{j=1}^q \theta_j\, \varepsilon_{t-j}
                 + \varepsilon_t,
        \qquad
        \varepsilon_t = \sigma_\varepsilon\, z_t,
        \qquad
        z_t \sim f_z\,(\text{mean}=0, \mathrm{var}=1).
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
        c=None,
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
            c=c,
            sigma_eps=sigma_eps,
            residual_params=residual_params,
            terminal_state=terminal_state,
            n_train_=n_train_,
            cov_matrix_=cov_matrix_,
            standard_errors_=standard_errors_,
            residual_diagnostics_=residual_diagnostics_,
        )
