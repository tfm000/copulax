"""MA(q) mean-equation model — special case of ARMA(0, q).

Concrete singleton fixing ``p = 0``.  All actual machinery lives in
:class:`copulax._src.timeseries._mean._arma_base.ARMABase`; this
module specialises the public :meth:`fit` signature to drop the
``p`` kwarg.

Example:
    >>> from copulax.univariate import normal
    >>> from copulax.timeseries import ma
    >>> import jax.random
    >>> y = jax.random.normal(jax.random.PRNGKey(0), (500,))
    >>> fit = ma.fit(y, q=2, residual_dist=normal)  # doctest: +SKIP

Cross-validation: matches ``statsmodels.tsa.arima.ARIMA(y,
order=(0, 0, q))`` to the documented tolerances under
correctly-specified data.
"""

from __future__ import annotations

from typing import Optional

from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src.timeseries._mean._arma_base import ARMABase


class MA(ARMABase):
    r"""MA(q) mean-equation model.

    Pure moving-average specialisation of :class:`ARMABase` with
    :math:`p = 0` pinned.

    .. math::

        y_t = c + \sum_{j=1}^q \theta_j\, \varepsilon_{t-j}
                 + \varepsilon_t.

    Inherits the full base contract — see
    :class:`copulax._src.timeseries._mean._arma_base.ARMABase` for
    method documentation.
    """

    def fit(
        self,
        y: ArrayLike,
        *,
        q: Optional[int] = None,
        residual_dist: Optional[Univariate] = None,
        init: str = "analytical",
        init_params: Optional[dict] = None,
        backcast_length: Optional[int] = None,
        maxiter: int = 200,
        lr: float = 0.05,
        name: Optional[str] = None,
    ) -> "MA":
        r"""Fit the MA(q) model to a series.

        Identical contract to :meth:`ARMABase.fit` with ``p = 0``
        forced.  See that method for full argument documentation.
        """
        return super().fit(
            y, p=0, q=q, residual_dist=residual_dist,
            init=init, init_params=init_params,
            backcast_length=backcast_length, maxiter=maxiter, lr=lr,
            name=name,
        )


#: Singleton entry point for MA(q) fitting.
ma = MA("MA")
