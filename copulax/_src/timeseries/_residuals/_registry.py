"""Registry of allowed standardised innovation laws.

The whitelist comprises every univariate distribution for which the
moment-matching ``_standardise_params`` classmethod produces a
``(mean = 0, var = 1)`` form — these are the residual laws the
time-series subpackage allows as innovations for AR / MA / ARMA mean
models, GARCH-family conditional-variance models, and the joint
ARMA-GARCH composite.

To add a new distribution as an allowed residual:

1. Implement :meth:`_standardise_params` on the distribution class so
   it returns a parameter dict producing a ``(mean = 0, var = 1)``
   distribution given the chosen free shape parameters; cite the
   derivation in the docstring.
2. Append the class to :data:`_RESIDUAL_SHAPE_KEYS` below with the
   tuple of shape-parameter keys to expose as the free fitting
   variables.
3. Append the singleton to :data:`_ALLOWED_RESIDUAL_DISTS` so the
   :class:`copulax._src.timeseries._residuals._standardise.StandardisedResidual`
   wrapper accepts it.
4. Verify the distribution exposes :meth:`logpdf`, :meth:`cdf`,
   :meth:`ppf`, and :meth:`rvs` (already true for every
   :class:`Univariate` distribution in CopulAX).

Shape keys are *only* the parameters that flow through the time-series
fit objective.  ``mu`` and ``sigma`` (and any other location / scale
parameters) are derived inside ``_standardise_params`` from the
shape-only inputs — they must never appear here.
"""

from __future__ import annotations

from copulax._src.univariate.gen_normal import GenNormal, gen_normal
from copulax._src.univariate.gh import GH, gh
from copulax._src.univariate.nig import NIG, nig
from copulax._src.univariate.normal import Normal, normal
from copulax._src.univariate.skewed_t import SkewedT, skewed_t
from copulax._src.univariate.student_t import StudentT, student_t


#: Per-class tuple of free shape-parameter keys for the standardised
#: form.  The order of each tuple defines the parameter ordering in
#: the flat optimiser-state array consumed by
#: :meth:`StandardisedResidual.shape_params_to_array` /
#: :meth:`shape_params_from_array`.
_RESIDUAL_SHAPE_KEYS: dict[type, tuple[str, ...]] = {
    Normal:    (),
    StudentT:  ("nu",),
    GenNormal: ("beta",),
    NIG:       ("alpha", "beta"),
    GH:        ("lamb", "chi", "psi", "gamma"),
    SkewedT:   ("nu", "gamma"),
}


#: Feasible default shape-parameter values for the *standardised*
#: form.  The wrapper uses these (rather than projecting
#: :meth:`Univariate.example_params` onto :data:`_RESIDUAL_SHAPE_KEYS`)
#: because the existing per-distribution ``example_params`` are tuned
#: to be illustrative for the *unstandardised* density and do not all
#: satisfy the standardised-form feasibility bounds — most
#: prominently, ``Skewed-T`` ``example_params`` carries
#: :math:`\gamma = 1` which violates :math:`\gamma^2 < 1 /
#: \mathrm{Var}[W]` at any practical :math:`\nu`.  Defaults below are
#: chosen to be moment-matched feasible and statistically sensible
#: as cold-start parameter values for the residual half of a
#: time-series fit.
_RESIDUAL_DEFAULT_SHAPE_PARAMS: dict[type, dict] = {
    Normal:    {},
    StudentT:  {"nu": 5.0},
    GenNormal: {"beta": 2.0},
    NIG:       {"alpha": 2.0, "beta": 0.0},
    GH:        {"lamb": 0.0, "chi": 1.0, "psi": 1.0, "gamma": 0.0},
    SkewedT:   {"nu": 8.0, "gamma": 0.0},
}


#: Tuple of allowed residual-law singletons.  The
#: :class:`StandardisedResidual` wrapper rejects any base distribution
#: not in this collection — explicit `ValueError` rather than a silent
#: fall-through to a hand-rolled standardisation, per CLAUDE.md "no
#: silent failures".
_ALLOWED_RESIDUAL_DISTS: tuple = (
    normal,
    student_t,
    gen_normal,
    nig,
    gh,
    skewed_t,
)


__all__ = [
    "_RESIDUAL_SHAPE_KEYS",
    "_RESIDUAL_DEFAULT_SHAPE_PARAMS",
    "_ALLOWED_RESIDUAL_DISTS",
]
