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

from jax.typing import ArrayLike

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

    References
    ----------
    .. [1] Bollerslev, T. (1986). *Generalized autoregressive conditional
       heteroskedasticity*. Journal of Econometrics, 31(3), 307-327,
       eq. (2) (the :math:`\sigma^2` recursion). Standard errors follow
       Bollerslev & Wooldridge (1992); see
       :mod:`copulax._src.timeseries._se`.
    """

    def __init__(
        self,
        p: int = 0,
        q: int = 0,
        *,
        residual_dist: Univariate | None = None,
        name: str = "GARCH",
        omega: ArrayLike | None = None,
        alpha: ArrayLike | None = None,
        beta: ArrayLike | None = None,
        residual_params: dict | None = None,
        terminal_state: GARCHTerminalState | None = None,
        n_train_: int | None = None,
        cov_matrix_: ArrayLike | None = None,
        standard_errors_: dict | None = None,
        residual_diagnostics_: dict | None = None,
        converged: ArrayLike | None = None,
        grad_norm: ArrayLike | None = None,
        n_iterations: ArrayLike | None = None,
        nan_encountered: ArrayLike | None = None,
        n_finite_candidates: ArrayLike | None = None,
        best_candidate: ArrayLike | None = None,
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
            n_train_=n_train_,
            cov_matrix_=cov_matrix_,
            standard_errors_=standard_errors_,
            residual_diagnostics_=residual_diagnostics_,
            converged=converged,
            grad_norm=grad_norm,
            n_iterations=n_iterations,
            nan_encountered=nan_encountered,
            n_finite_candidates=n_finite_candidates,
            best_candidate=best_candidate,
        )
