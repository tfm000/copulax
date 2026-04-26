"""Regression tests for mvt_gh and mvt_skewed_t against R ghyp package.

The ghyp R package (v1.6.5) is the authoritative reference implementation
of the McNeil et al. (2005) generalised-hyperbolic and skewed-t families.
CopulAX uses the identical parametrisation via ghyp's (lambda, chi, psi,
mu, sigma, gamma) constructor.

Replaces the v1.0.1 self-referential golden fixtures for the two
multivariate distributions that have no scipy equivalent.

Reference data is hardcoded in `_r_reference/gh_reference_data.py`.
To regenerate:
    Rscript copulax/tests/_r_reference/generate_gh_reference.R \\
        > copulax/tests/_r_reference/gh_reference_data.py
"""

import jax.numpy as jnp
import numpy as np
import pytest

from copulax.multivariate import mvt_gh, mvt_skewed_t
from copulax.tests._r_reference.gh_reference_data import (
    GH_CASES, SKEWT_CASES,
)


LOGPDF_RTOL = 1e-8
LOGPDF_ATOL = 1e-10
PDF_RTOL = 1e-6
PDF_ATOL = 1e-10


def _build_params(dist, case):
    """Build a CopulAX params dict from a reference case entry.

    Dispatches on the case dict's key set: `"lamb"` → mvt_gh,
    otherwise → mvt_skewed_t.
    """
    mu = jnp.asarray(case["mu"]).reshape(-1, 1)
    gamma_vec = jnp.asarray(case["gamma"]).reshape(-1, 1)
    sigma = jnp.asarray(case["sigma"])
    if "lamb" in case:
        return dist._params_dict(
            lamb=case["lamb"], chi=case["chi"], psi=case["psi"],
            mu=mu, gamma=gamma_vec, sigma=sigma,
        )
    return dist._params_dict(
        nu=case["nu"], mu=mu, gamma=gamma_vec, sigma=sigma,
    )


ALL_CASES = (
    [(mvt_gh, name, case) for name, case in GH_CASES.items()]
    + [(mvt_skewed_t, name, case) for name, case in SKEWT_CASES.items()]
)
ALL_IDS = [name for _, name, _ in ALL_CASES]


class TestMvtAgainstR:
    """Cross-validate mvt_gh and mvt_skewed_t logpdf / pdf against R ghyp."""

    @pytest.mark.parametrize(
        "dist,case_name,case", ALL_CASES, ids=ALL_IDS,
    )
    def test_logpdf_matches_r(self, dist, case_name, case):
        params = _build_params(dist, case)
        x = jnp.asarray(case["x"])
        cx = np.asarray(dist.logpdf(x, params)).flatten()
        np.testing.assert_allclose(
            cx, np.asarray(case["logpdf"]),
            rtol=LOGPDF_RTOL, atol=LOGPDF_ATOL,
            err_msg=f"{dist.name} logpdf mismatch vs R ghyp at {case_name}",
        )

    @pytest.mark.parametrize(
        "dist,case_name,case", ALL_CASES, ids=ALL_IDS,
    )
    def test_pdf_matches_r(self, dist, case_name, case):
        params = _build_params(dist, case)
        x = jnp.asarray(case["x"])
        cx = np.asarray(dist.pdf(x, params)).flatten()
        np.testing.assert_allclose(
            cx, np.asarray(case["pdf"]),
            rtol=PDF_RTOL, atol=PDF_ATOL,
            err_msg=f"{dist.name} pdf mismatch vs R ghyp at {case_name}",
        )
