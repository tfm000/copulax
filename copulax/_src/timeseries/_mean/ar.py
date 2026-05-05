"""AR(p) mean-equation model — special case of ARMA(p, 0).

Concrete user-facing AR class fixing ``q = 0``.  All actual
machinery lives in
:class:`copulax._src.timeseries._mean._arma_base.ARMABase`; this
module specialises the constructor to drop the ``q`` kwarg and to
produce fitted instances whose class name reads as ``AR`` rather
than ``ARMA``.

Example:
    >>> from copulax.univariate import normal
    >>> from copulax.timeseries import AR
    >>> import jax.random
    >>> y = jax.random.normal(jax.random.PRNGKey(0), (500,))
    >>> fit = AR(p=2, residual_dist=normal).fit(y)  # doctest: +SKIP

Cross-validation: matches ``statsmodels.tsa.arima.ARIMA(y,
order=(p, 0, 0))`` to the documented tolerances under
correctly-specified data.
"""

from __future__ import annotations

from typing import Optional

from copulax._src._distributions import Univariate
from copulax._src.timeseries._mean._arma_base import ARMABase, ARMATerminalState


class AR(ARMABase):
    r"""AR(p) mean-equation model.

    Pure autoregressive specialisation of :class:`ARMABase` with
    :math:`q = 0` pinned.

    Construct with the desired AR order and residual law:

    .. code-block:: python

        from copulax.timeseries import AR
        from copulax.univariate import normal
        fit = AR(p=2, residual_dist=normal).fit(y)

    .. math::

        y_t = c + \sum_{i=1}^p \phi_i\, y_{t-i} + \varepsilon_t.

    Inherits :meth:`fit` / :meth:`forecast` / :meth:`residuals` /
    :meth:`stats` etc. from :class:`ARMABase`.
    """

    def __init__(
        self,
        p: int = 0,
        *,
        residual_dist: Optional[Univariate] = None,
        name: str = "AR",
        # ``q`` is accepted as a kwarg only so the equinox PyTree
        # round-trip and the parent ``fit`` method (which always
        # constructs the fitted result via ``cls(p=..., q=..., ...)``)
        # see a uniform signature.  Any non-zero value is rejected —
        # use :class:`ARMA` for q > 0.
        q: int = 0,
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
        if int(q) != 0:
            raise ValueError(
                f"AR requires q=0; got q={int(q)}.  Use ARMA for q > 0."
            )
        super().__init__(
            name=name,
            p=p,
            q=0,
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

    # ------------------------------------------------------------------
    # Summary header overrides — drop the unused MA order from the
    # display label since AR pins q=0.
    # ------------------------------------------------------------------
    def _summary_header(self) -> str:
        from copulax._src.timeseries._summary import display_residual_name
        return (
            f"AR({self.p}) — "
            f"{display_residual_name(self.residual_dist.name)} residuals"
        )

    def _mean_section_label(self) -> str:
        return f"Mean equation — AR({self.p})"
