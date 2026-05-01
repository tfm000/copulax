r"""GARCH(p, q) conditional-variance model — Bollerslev 1986.

Concrete user-facing entry point for the vanilla σ²-form GARCH
recursion.  All actual machinery lives in
:class:`copulax._src.timeseries._variance._garch_base.GARCHBase`.

Recursion:

.. math::

    \sigma^2_t = \omega
               + \sum_{i=1}^p \alpha_i\, \varepsilon^2_{t-i}
               + \sum_{j=1}^q \beta_j\, \sigma^2_{t-j}.

Cross-validation: parameter estimates and the maximum log-likelihood
match ``arch.arch_model(eps, mean='Zero', vol='GARCH', dist='Normal',
p=p, q=q).fit()`` to the documented tolerances under correctly-
specified data.

Reference:
    Bollerslev, T. (1986). *Generalized autoregressive conditional
    heteroskedasticity*.  Journal of Econometrics, 31(3), 307-327.
"""

from __future__ import annotations

from typing import Optional

from copulax._src._distributions import Univariate
from copulax._src.timeseries._variance._garch_base import (
    GARCHBase,
    GARCHTerminalState,
)


class GARCH(GARCHBase):
    r"""Vanilla GARCH(p, q) conditional-variance model.

    Construct with the desired orders and residual law:

    .. code-block:: python

        from copulax.timeseries import GARCH
        from copulax.univariate import normal
        fit = GARCH(p=1, q=1, residual_dist=normal).fit(eps)

    Inherits :meth:`fit` / :meth:`forecast` / :meth:`residuals` /
    :meth:`stats` etc. from :class:`GARCHBase`.
    """

    def __init__(
        self,
        p: int = 0,
        q: int = 0,
        *,
        residual_dist: Optional[Univariate] = None,
        name: str = "GARCH",
        omega=None,
        alpha=None,
        beta=None,
        residual_params=None,
        terminal_state: Optional[GARCHTerminalState] = None,
        loglikelihood_=None,
        aic_=None,
        bic_=None,
        n_train_: Optional[int] = None,
    ):
        super().__init__(
            name=name,
            p=p,
            q=q,
            residual_dist=residual_dist,
            omega=omega,
            alpha=alpha,
            beta=beta,
            residual_params=residual_params,
            terminal_state=terminal_state,
            loglikelihood_=loglikelihood_,
            aic_=aic_,
            bic_=bic_,
            n_train_=n_train_,
        )
