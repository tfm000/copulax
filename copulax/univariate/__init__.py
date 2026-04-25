"""Univariate probability distributions and fitting utilities."""

from copulax._src.univariate._registry import *
from copulax._src.univariate.univariate_fitter import (
    batch_univariate_fitter,
    univariate_fitter,
)
from copulax._src.univariate._gof import cvm_test, ks_test
