"""AR(p) mean-equation model — special case of ARMA(p, 0).

Concrete singleton fixing ``q = 0``.  All actual machinery lives in
:class:`copulax._src.timeseries._mean._arma_base.ARMABase`; this
module specialises the public :meth:`fit` signature to drop the
``q`` kwarg (which would always be ignored) and to produce fitted
instances whose class name reads as ``AR`` rather than ``ARMA``.

Example:
    >>> from copulax.univariate import normal
    >>> from copulax.timeseries import ar
    >>> import jax.random
    >>> y = jax.random.normal(jax.random.PRNGKey(0), (500,))
    >>> fit = ar.fit(y, p=2, residual_dist=normal)  # doctest: +SKIP

Cross-validation: matches ``statsmodels.tsa.arima.ARIMA(y,
order=(p, 0, 0))`` to the documented tolerances under
correctly-specified data.
"""

from __future__ import annotations

from typing import Optional

from jax import Array
from jax.typing import ArrayLike

from copulax._src._distributions import Univariate
from copulax._src.timeseries._mean._arma_base import ARMABase


class AR(ARMABase):
    r"""AR(p) mean-equation model.

    Pure autoregressive specialisation of :class:`ARMABase` with
    :math:`q = 0` pinned.

    .. math::

        y_t = c + \sum_{i=1}^p \phi_i\, y_{t-i} + \varepsilon_t.

    Inherits the full base contract — see
    :class:`copulax._src.timeseries._mean._arma_base.ARMABase` for
    method documentation.
    """

    def fit(
        self,
        y: ArrayLike,
        *,
        p: Optional[int] = None,
        residual_dist: Optional[Univariate] = None,
        init: str = "analytical",
        init_params: Optional[dict] = None,
        backcast_length: Optional[int] = None,
        maxiter: int = 200,
        lr: float = 0.05,
        name: Optional[str] = None,
    ) -> "AR":
        r"""Fit the AR(p) model to a series.

        Identical contract to :meth:`ARMABase.fit` with ``q = 0``
        forced.  See that method for full argument documentation.

        Raises:
            ValueError: When ``init`` is not one of the supported
                strings, or when ``init_params`` is omitted under
                ``init="warm"``.
        """
        return super().fit(
            y, p=p, q=0, residual_dist=residual_dist,
            init=init, init_params=init_params,
            backcast_length=backcast_length, maxiter=maxiter, lr=lr,
            name=name,
        )


#: Singleton entry point for AR(p) fitting.
ar = AR("AR")
