"""Generate golden regression test fixtures.

Univariate, multivariate, and elliptical-copula fixtures were generated
from copulax v1.0.1 (PyPI) — the pre-refactor baseline.  The generation
script (``gen_golden_v101.py``) was run from *outside* the workspace so
that the installed package was imported, not the local development code:

    python3 -m venv /tmp/copulax_golden_venv
    source /tmp/copulax_golden_venv/bin/activate
    pip install copulax          # installs 1.0.1
    cd /tmp
    python gen_golden_v101.py <golden_dir>

Archimedean copulas do not exist in v1.0.1 and therefore have no
pre-refactor baseline.  Their fixtures are generated from the current
(post-refactor) code by running this script:

    python -m copulax.tests.golden.generate_golden

These fixtures are immutable — any change to numerical outputs
should be deliberate and require re-running the appropriate script.
"""

import sys
import jax
import jax.numpy as jnp
import numpy as np
import os

# Deterministic key for reproducibility
KEY = jax.random.PRNGKey(42)


def _log(msg):
    print(msg, flush=True)


def _generate_archimedean_fixtures():
    """Generate golden data for Archimedean copula distributions.

    NOTE: Archimedean copulas are new in v2.0 — no v1.0.1 baseline exists.
    """
    from copulax.copulas import (
        clayton_copula,
        frank_copula,
        gumbel_copula,
        joe_copula,
        amh_copula,
        independence_copula,
    )

    dists_3d = [
        clayton_copula,
        frank_copula,
        gumbel_copula,
        joe_copula,
        independence_copula,
    ]
    fixtures = {}

    # Uniform marginal data (3D)
    key = jax.random.PRNGKey(789)
    u_3d = jax.random.uniform(key, shape=(10, 3), minval=0.01, maxval=0.99)
    # 2D for AMH
    u_2d = u_3d[:, :2]

    for dist in dists_3d:
        _log(f"  Processing {dist.name}...")
        params = dist.example_params(dim=3)
        cdf_vals = dist.copula_cdf(u_3d, params)
        logpdf_vals = dist.copula_logpdf(u_3d, params)
        pdf_vals = dist.copula_pdf(u_3d, params)

        name = dist.name.replace("-", "_").lower()
        fixtures[name] = {
            "u": np.asarray(u_3d),
            "copula_cdf": np.asarray(cdf_vals),
            "copula_logpdf": np.asarray(logpdf_vals),
            "copula_pdf": np.asarray(pdf_vals),
        }
        _log(f"  Generated: {dist.name}")

    # AMH (2D only)
    _log("  Processing AMH-Copula...")
    params = amh_copula.example_params(dim=2)
    cdf_vals = amh_copula.copula_cdf(u_2d, params)
    logpdf_vals = amh_copula.copula_logpdf(u_2d, params)
    pdf_vals = amh_copula.copula_pdf(u_2d, params)
    fixtures["amh_copula"] = {
        "u": np.asarray(u_2d),
        "copula_cdf": np.asarray(cdf_vals),
        "copula_logpdf": np.asarray(logpdf_vals),
        "copula_pdf": np.asarray(pdf_vals),
    }
    _log("  Generated: AMH-Copula")

    # Independence copula (3D) — copula_logpdf is always 0, copula_pdf is always 1
    _log("  Processing Independence-Copula...")
    params = independence_copula.example_params(dim=3)
    cdf_vals = independence_copula.copula_cdf(u_3d, params)
    logpdf_vals = independence_copula.copula_logpdf(u_3d, params)
    pdf_vals = independence_copula.copula_pdf(u_3d, params)
    fixtures["independence_copula"] = {
        "u": np.asarray(u_3d),
        "copula_cdf": np.asarray(cdf_vals),
        "copula_logpdf": np.asarray(logpdf_vals),
        "copula_pdf": np.asarray(pdf_vals),
    }
    _log("  Generated: Independence-Copula")

    return fixtures


def main():
    golden_dir = os.path.dirname(__file__)

    _log("Generating Archimedean copula golden fixtures (v2.0 — no v1.0.1 baseline)...")
    arch = _generate_archimedean_fixtures()
    np.savez(
        os.path.join(golden_dir, "archimedean.npz"),
        **{
            f"{name}/{key}": val
            for name, data in arch.items()
            for key, val in _flatten_dict(data).items()
        },
    )

    _log(f"Done. Archimedean golden fixtures saved to: {golden_dir}")
    _log("NOTE: univariate.npz, multivariate.npz, and copulas.npz were")
    _log("generated from copulax v1.0.1 (PyPI) — see docstring for details.")


def _flatten_dict(d, prefix=""):
    """Flatten a nested dict into dotted keys."""
    items = {}
    for k, v in d.items():
        new_key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items


if __name__ == "__main__":
    main()
