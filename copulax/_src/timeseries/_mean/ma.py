"""MA(q) mean-equation model — special case of ARMA(0, q).

Concrete user-facing MA class fixing ``p = 0``.  All actual
machinery lives in
:class:`copulax._src.timeseries._mean._arma_base.ARMABase`.

Example:
    >>> from copulax.univariate import normal
    >>> from copulax.timeseries import MA
    >>> import jax.random
    >>> y = jax.random.normal(jax.random.PRNGKey(0), (500,))
    >>> fit = MA(q=2, residual_dist=normal).fit(y)  # doctest: +SKIP

Cross-validation: matches ``statsmodels.tsa.arima.ARIMA(y,
order=(0, 0, q))`` to the documented tolerances under
correctly-specified data.
"""

from __future__ import annotations

from copulax._src._distributions import Univariate
from copulax._src.timeseries._mean._arma_base import ARMABase, ARMATerminalState


class MA(ARMABase):
    r"""MA(q) mean-equation model.

    Pure moving-average specialisation of :class:`ARMABase` with
    :math:`p = 0` pinned.

    Construct with the desired MA order and residual law:

    .. code-block:: python

        from copulax.timeseries import MA
        from copulax.univariate import normal
        fit = MA(q=2, residual_dist=normal).fit(y)

    .. math::

        y_t = \mu + \sum_{j=1}^q \theta_j\, \varepsilon_{t-j}
                  + \varepsilon_t.

    For pure-MA models :math:`\mu` is both the unconditional mean
    and the per-step intercept (the centred form collapses since
    there are no AR terms to rescale).

    Inherits :meth:`fit` / :meth:`forecast` / :meth:`residuals` /
    :meth:`stats` etc. from :class:`ARMABase`.

    References
    ----------
    .. [1] Box, G.E.P. & Jenkins, G.M. (1970). *Time Series Analysis:
       Forecasting and Control*. Holden-Day. (MA(q) form.)
    .. [2] Hamilton, J.D. (1994). *Time Series Analysis*, ch. 3-5.
       Princeton University Press. (MA(q) recursion sec. 3.3; conditional
       maximum-likelihood sec. 5.2; unconditional variance
       :math:`\sigma_\varepsilon^2 (1 + \sum_j \theta_j^2)` sec. 3.3.)
    """

    def __init__(
        self,
        q: int = 0,
        *,
        residual_dist: Univariate | None = None,
        name: str = "MA",
        # ``p`` is accepted as a kwarg only so the equinox PyTree
        # round-trip and the parent ``fit`` method see a uniform
        # signature.  Any non-zero value is rejected — use ARMA.
        p: int = 0,
        phi=None,
        theta=None,
        mu=None,
        sigma_eps=None,
        residual_params=None,
        terminal_state: ARMATerminalState | None = None,
        n_train_: int | None = None,
        cov_matrix_=None,
        standard_errors_=None,
        residual_diagnostics_=None,
        converged=None,
        grad_norm=None,
        n_iterations=None,
        nan_encountered=None,
        n_finite_candidates=None,
        best_candidate=None,
    ):
        if int(p) != 0:
            raise ValueError(f"MA requires p=0; got p={int(p)}.  Use ARMA for p > 0.")
        super().__init__(
            name=name,
            p=0,
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
            converged=converged,
            grad_norm=grad_norm,
            n_iterations=n_iterations,
            nan_encountered=nan_encountered,
            n_finite_candidates=n_finite_candidates,
            best_candidate=best_candidate,
        )

    # ------------------------------------------------------------------
    # Summary header overrides — drop the unused AR order since MA
    # pins p=0.
    # ------------------------------------------------------------------
    def _summary_header(self) -> str:
        from copulax._src.timeseries._summary import display_residual_name

        return (
            f"MA({self.q}) — {display_residual_name(self.residual_dist.name)} residuals"
        )

    def _mean_section_label(self) -> str:
        return f"Mean equation — MA({self.q})"
