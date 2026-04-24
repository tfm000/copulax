"""Univariate probability distributions and fitting utilities.

The distribution singletons (``normal``, ``gamma``, ``gh``, ...) and
their classes are loaded eagerly via ``from .distributions import *``.

Fitter utilities (``univariate_fitter``, ``batch_univariate_fitter``)
and the goodness-of-fit helpers (``ks_test``, ``cvm_test``) are exposed
through PEP 562 lazy ``__getattr__`` below.  The laziness matters:
``copulax._src.univariate.univariate_fitter`` itself imports from
``copulax.univariate.distributions``, so a direct
``from copulax._src.univariate.univariate_fitter import ...`` would
otherwise trigger *this* package's ``__init__`` while the fitter module
is still loading, yielding a partial-module ``ImportError`` on the
eager re-export.  Deferring the re-export to attribute-access time
lets the fitter module finish loading first.
"""

from copulax.univariate.distributions import *


_LAZY_ATTRS = {
    "univariate_fitter": (
        "copulax._src.univariate.univariate_fitter",
        "univariate_fitter",
    ),
    "batch_univariate_fitter": (
        "copulax._src.univariate.univariate_fitter",
        "batch_univariate_fitter",
    ),
    "ks_test": ("copulax._src.univariate._gof", "ks_test"),
    "cvm_test": ("copulax._src.univariate._gof", "cvm_test"),
}


def __getattr__(name: str):
    r"""PEP 562 lazy attribute resolver for re-exported fitter utilities."""
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    import importlib
    mod = importlib.import_module(target[0])
    return getattr(mod, target[1])


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY_ATTRS))
