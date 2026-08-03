r"""Regenerate ``frozen_series_data.py`` — the frozen test-series corpus.

Run from the project root::

    python copulax/tests/_r_reference/generate_frozen_series_handrolled.py

This is the **driver** for the whole corpus. It has four sources:

1. ``generate_frozen_series.R`` (invoked here as a subprocess) — every
   GARCH-bearing series, drawn from a pinned ``rugarch`` spec via
   ``ugarchpath``. That covers the standalone GARCH(1, 1) residual series
   and the AR(1)-GARCH(1, 1) level series.
2. ``statsmodels.tsa.arima_process.arma_generate_sample`` (in this file)
   — the pure AR / MA / ARMA mean-model series, which have no variance
   dynamics and therefore no reason to go through a GARCH engine.
3. A **one-time port** of the hand-rolled matrix simulator (in this file)
   — the four ``test_timeseries_arma_garch.py`` matrix labels that no
   third-party engine can represent: TGARCH's ``sigma``-form recursion,
   QGARCH's asymmetric ``psi`` term, and the copulax ``gh`` / ``skewed_t``
   residual parameterisations. This recursion runs **only here, only at
   regeneration time** — never in the test run.
4. A **one-time port** of the module-local variance-variant simulators
   (in this file) — the IGARCH / GJR / EGARCH / TGARCH / QGARCH / GARCH-M
   residual series that ``test_timeseries_variance.py`` used to roll at
   collection time, plus the near-boundary AR(1)-GARCH(1, 1) series
   inlined in ``test_timeseries_arma_garch.py``. Like source 3 these have
   no third-party equivalent in the copulax parameterisation, and like
   source 3 they run **only here**.

Why the corpus exists
---------------------
Before it, every time-series test series was simulated at test runtime by
a copulax-authored recursion. That put a copulax formula on both sides of
every statistical assertion: the process the test fits and the estimator
it checks came from the same hand. Sourcing the data from third-party
engines and committing the result removes the hand-rolled DGP from the
test path entirely, and makes every series byte-identical on every
machine and every CI leg.

Determinism
-----------
Every source is seeded explicitly:

* rugarch — ``ugarchpath(..., rseed = <seed>)``.
* statsmodels — ``distrvs=numpy.random.Generator.standard_normal`` bound
  to a fresh ``numpy.random.default_rng(<seed>)``.
* the one-time port — ``jax.random.PRNGKey(<seed>)``, with the same
  ``_deterministic_seed`` (SHA-256 of the label) the test module uses.

Re-running this script therefore reproduces ``frozen_series_data.py``
byte for byte.

Precision
---------
The R half transfers its doubles as ``%.17g`` decimal literals **and** as
big-endian IEEE-754 hex. This script parses the decimals, re-encodes each
value, and asserts the result matches the hex for every element — a
self-contained proof that the R → Python transfer is lossless to the bit.
(R's own ``as.numeric`` is not correctly rounded and cannot verify this;
CPython's ``float()`` is.)

Values are written out with ``repr(float)``, which is the shortest decimal
string that round-trips a double exactly. The written module is loaded and
its per-series SHA-256 re-checked before the script exits.

Verifying the committed data
----------------------------
::

    python - <<'PY'
    import hashlib, numpy as np
    from copulax.tests._r_reference.frozen_series_data import FROZEN_SERIES
    for name, entry in FROZEN_SERIES.items():
        digest = hashlib.sha256(np.asarray(entry["y"]).tobytes()).hexdigest()
        assert digest == entry["provenance"]["sha256"], name
    print(len(FROZEN_SERIES), "series verified")
    PY

Note
----
``frozen_series_data.py`` imports **numpy only** — it is deliberately free
of any jax or copulax dependency so that loading a series costs nothing
but a module import. This script, by contrast, needs jax, copulax,
statsmodels and an R toolchain with ``rugarch``, because it is the
regeneration path, not the load path.
"""

from __future__ import annotations

import ast
import hashlib
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


_HERE = Path(__file__).resolve().parent
_R_SCRIPT = _HERE / "generate_frozen_series.R"
_OUT_PATH = _HERE / "frozen_series_data.py"

#: Burn-in discarded before the returned sample begins. Shared by all
#: three sources so the whole corpus has one mixing convention:
#: ``ugarchpath(n.start=)``, ``arma_generate_sample(burnin=)``, and the
#: ``_BURN_IN`` of the one-time hand-rolled port.
BURN_IN = 500

#: Values per line in the emitted ``np.array([...])`` literals. Keeps the
#: committed module diffable instead of one multi-megabyte line.
_VALUES_PER_LINE = 6


def _enable_x64() -> None:
    """Put jax in double precision, exactly as the test suite does.

    ``copulax/tests/conftest.py`` sets ``jax_enable_x64 = True`` at
    module level, so every runtime recursion this file ports ran in
    float64 inside the suite: the PRNG draws, the scan arithmetic and
    the resulting series were all doubles. Generating without x64 draws
    a float32 sample from the same key — a DIFFERENT realization of the
    same DGP, not a rounding difference — and the frozen values would
    then fail to reproduce the runtime path they replace.

    Must be called before the first jax array is created.
    """
    import jax

    jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Source 1: rugarch (subprocess)
# ---------------------------------------------------------------------------

def load_rugarch_series() -> dict[str, dict[str, Any]]:
    """Run the R generator and return its series, verified bit-exact.

    Returns
    -------
    dict
        ``{name: {"y": np.ndarray, "provenance": dict}}``.  The
        provenance dict carries ``generator``, ``engine``, ``spec``,
        ``seed`` and ``n``; ``sha256`` is added later by
        :func:`build_corpus`.

    Raises
    ------
    RuntimeError
        If ``Rscript`` is unavailable, the R script exits non-zero, or
        its output does not contain the expected assignment.
    AssertionError
        If any value fails to round-trip from ``%.17g`` to the exact
        double the R side held.
    """
    if not _R_SCRIPT.is_file():
        raise RuntimeError(f"missing R generator: {_R_SCRIPT}")

    try:
        proc = subprocess.run(
            ["Rscript", str(_R_SCRIPT)],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - toolchain guard
        raise RuntimeError(
            "Rscript not found on PATH. The rugarch half of the corpus "
            "cannot be regenerated without an R toolchain carrying the "
            "rugarch package."
        ) from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"{_R_SCRIPT.name} exited {proc.returncode}:\n{proc.stderr}"
        )

    tree = ast.parse(proc.stdout)
    assigns = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "FROZEN_SERIES_RUGARCH"
    ]
    if len(assigns) != 1:
        raise RuntimeError(
            f"{_R_SCRIPT.name} did not emit exactly one "
            f"FROZEN_SERIES_RUGARCH assignment (found {len(assigns)})."
        )
    raw = ast.literal_eval(assigns[0].value)

    out: dict[str, dict[str, Any]] = {}
    for name, entry in raw.items():
        y = np.asarray(entry["y"], dtype=np.float64)
        encoded = [struct.pack(">d", float(v)).hex() for v in y]
        assert encoded == entry["y_bits"], (
            f"{name}: R -> Python decimal transfer lost precision; the "
            f"%.17g literals do not re-encode to the bits R held."
        )
        provenance = dict(entry["provenance"])
        assert y.size == provenance["n"], name
        out[name] = {"y": y, "provenance": provenance}
    return out


# ---------------------------------------------------------------------------
# Source 2: statsmodels mean-model series
# ---------------------------------------------------------------------------

def simulate_arma(
    n: int,
    phi: Sequence[float],
    theta: Sequence[float],
    seed: int,
    *,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> tuple[np.ndarray, str]:
    r"""Draw one ARMA(p, q) path from statsmodels.

    Simulates the centred process

    .. math::
        y_t - \mu = \sum_{i=1}^{p} \phi_i (y_{t-i} - \mu)
                    + \varepsilon_t
                    + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j},
        \qquad \varepsilon_t \sim N(0, \sigma^2),

    which is exactly copulax's mean-equation convention.
    ``arma_generate_sample`` takes the AR and MA **lag polynomials**
    including the zero lag, so the AR coefficients enter negated
    (``ar = [1, -phi_1, ..., -phi_p]``) while the MA coefficients enter
    as-is (``ma = [1, theta_1, ..., theta_q]``).  The process is
    generated zero-mean at scale ``sigma`` and shifted by ``mu``.

    Parameters
    ----------
    n : int
        Number of observations to keep (after :data:`BURN_IN`).
    phi : Sequence[float]
        Autoregressive coefficients, lag 1 first.  Empty for a pure MA.
    theta : Sequence[float]
        Moving-average coefficients, lag 1 first.  Empty for a pure AR.
    seed : int
        Seed for ``numpy.random.default_rng``.
    mu : float, optional
        Unconditional mean.  Default ``0.0``.
    sigma : float, optional
        Innovation standard deviation.  Default ``1.0``.

    Returns
    -------
    tuple[numpy.ndarray, str]
        The simulated series (shape ``(n,)``, float64) and a
        human-readable provenance ``spec`` string.
    """
    from statsmodels.tsa.arima_process import arma_generate_sample

    ar_poly = np.r_[1.0, -np.asarray(phi, dtype=float)]
    ma_poly = np.r_[1.0, np.asarray(theta, dtype=float)]
    rng = np.random.default_rng(seed)
    y = arma_generate_sample(
        ar_poly, ma_poly, n, scale=sigma,
        distrvs=rng.standard_normal, burnin=BURN_IN,
    )
    y = np.asarray(y, dtype=np.float64) + mu
    assert y.shape == (n,), (y.shape, n)
    assert np.all(np.isfinite(y))

    spec = (
        f"ARMA(p={len(phi)}, q={len(theta)}) "
        f"phi={tuple(float(v) for v in phi)} "
        f"theta={tuple(float(v) for v in theta)} "
        f"mu={mu:g} sigma={sigma:g}; centred form "
        f"(y_t - mu) = sum phi_i (y_{{t-i}} - mu) + eps_t "
        f"+ sum theta_j eps_{{t-j}}, eps ~ N(0, sigma^2); "
        f"arma_generate_sample nsample={n} burnin={BURN_IN} "
        f"distrvs=default_rng({seed}).standard_normal"
    )
    return y, spec


#: Pure mean-model cases: one entry per distinct (DGP, length, seed)
#: tuple in the suite, per 01-15-AUDIT.md section "Task B inventory".
#: Seeds are the jax ``PRNGKey`` integers the replaced call sites used, so
#: every frozen series stays traceable to its origin.
#:
#: The helpers being replaced carried an ``init`` argument selecting the
#: pre-sample innovation convention ("pre_sample" vs "zero").
#: ``arma_generate_sample`` has a single convention (zero pre-sample
#: state, then ``burnin`` steps discarded), so that split disappears
#: here.  No two call sites collide as a result: every pair that differed
#: only in ``init`` also differs in seed or length.
STATSMODELS_CASES: tuple[dict[str, Any], ...] = (
    # --- AR(1) -------------------------------------------------------
    {"name": "ar1_p040_n500_s13",  "n":  500, "phi": (0.4,), "theta": (), "seed":  13},
    {"name": "ar1_p050_n250_s13",  "n":  250, "phi": (0.5,), "theta": (), "seed":  13},
    {"name": "ar1_p050_n500_s0",   "n":  500, "phi": (0.5,), "theta": (), "seed":   0},
    {"name": "ar1_p050_n500_s13",  "n":  500, "phi": (0.5,), "theta": (), "seed":  13},
    {"name": "ar1_p050_n500_s42",  "n":  500, "phi": (0.5,), "theta": (), "seed":  42},
    {"name": "ar1_p050_n500_s123", "n":  500, "phi": (0.5,), "theta": (), "seed": 123},
    {"name": "ar1_p050_n2000_s0",  "n": 2000, "phi": (0.5,), "theta": (), "seed":   0},
    {"name": "ar1_p050_n2000_s8",  "n": 2000, "phi": (0.5,), "theta": (), "seed":   8},
    {"name": "ar1_p060_n500_s13",  "n":  500, "phi": (0.6,), "theta": (), "seed":  13},
    {"name": "ar1_p060_n500_s42",  "n":  500, "phi": (0.6,), "theta": (), "seed":  42},
    {"name": "ar1_p060_n800_s7",   "n":  800, "phi": (0.6,), "theta": (), "seed":   7},
    {"name": "ar1_p060_n1000_s42", "n": 1000, "phi": (0.6,), "theta": (), "seed":  42},
    {"name": "ar1_p060_n3000_s20", "n": 3000, "phi": (0.6,), "theta": (), "seed":  20},
    {"name": "ar1_p060_m025_sd050_n500_s42",  "n":  500, "phi": (0.6,),
     "theta": (), "seed": 42, "mu": 0.25, "sigma": 0.5},
    {"name": "ar1_p060_m025_sd050_n2000_s42", "n": 2000, "phi": (0.6,),
     "theta": (), "seed": 42, "mu": 0.25, "sigma": 0.5},

    # --- AR(3) -------------------------------------------------------
    {"name": "ar3_n2000_s1", "n": 2000, "phi": (0.4, -0.2, 0.1),
     "theta": (), "seed": 1},

    # --- MA(1) -------------------------------------------------------
    {"name": "ma1_q040_n1500_s10", "n": 1500, "phi": (), "theta": (0.4,), "seed": 10},
    {"name": "ma1_q040_n2000_s2",  "n": 2000, "phi": (), "theta": (0.4,), "seed":  2},
    {"name": "ma1_q040_m010_sd050_n2000_s7", "n": 2000, "phi": (),
     "theta": (0.4,), "seed": 7, "mu": 0.1, "sigma": 0.5},

    # --- ARMA(1, 1) --------------------------------------------------
    {"name": "arma11_p050_qm030_n800_s101", "n": 800, "phi": (0.5,),
     "theta": (-0.3,), "seed": 101},
    {"name": "arma11_p050_qm030_n1500_s6", "n": 1500, "phi": (0.5,),
     "theta": (-0.3,), "seed": 6},
    {"name": "arma11_p050_qm030_n1500_s11", "n": 1500, "phi": (0.5,),
     "theta": (-0.3,), "seed": 11},
    {"name": "arma11_p050_qm030_n2000_s3", "n": 2000, "phi": (0.5,),
     "theta": (-0.3,), "seed": 3},
    {"name": "arma11_p050_qm030_n2000_s4", "n": 2000, "phi": (0.5,),
     "theta": (-0.3,), "seed": 4},
    {"name": "arma11_p050_qm030_n2000_s5", "n": 2000, "phi": (0.5,),
     "theta": (-0.3,), "seed": 5},
    {"name": "arma11_p050_qm030_n2000_s60", "n": 2000, "phi": (0.5,),
     "theta": (-0.3,), "seed": 60},
    {"name": "arma11_p060_q030_n1500_s99", "n": 1500, "phi": (0.6,),
     "theta": (0.3,), "seed": 99},
    {"name": "arma11_p050_q030_m020_sd050_n500_s13",  "n":  500, "phi": (0.5,),
     "theta": (0.3,), "seed": 13, "mu": 0.2, "sigma": 0.5},
    {"name": "arma11_p050_q030_m020_sd050_n1500_s13", "n": 1500, "phi": (0.5,),
     "theta": (0.3,), "seed": 13, "mu": 0.2, "sigma": 0.5},
    {"name": "arma11_p050_q030_m020_sd050_n2000_s13", "n": 2000, "phi": (0.5,),
     "theta": (0.3,), "seed": 13, "mu": 0.2, "sigma": 0.5},
)


def load_statsmodels_series() -> dict[str, dict[str, Any]]:
    """Generate every pure mean-model series in :data:`STATSMODELS_CASES`.

    Returns
    -------
    dict
        ``{name: {"y": np.ndarray, "provenance": dict}}``.
    """
    import statsmodels

    engine = f"statsmodels {statsmodels.__version__}"
    out: dict[str, dict[str, Any]] = {}
    for case in STATSMODELS_CASES:
        y, spec = simulate_arma(
            case["n"], case["phi"], case["theta"], case["seed"],
            mu=case.get("mu", 0.0), sigma=case.get("sigma", 1.0),
        )
        out[case["name"]] = {
            "y": y,
            "provenance": {
                "generator": "generate_frozen_series_handrolled.py",
                "engine": engine,
                "spec": spec,
                "seed": case["seed"],
                "n": case["n"],
            },
        }
    return out


# ---------------------------------------------------------------------------
# Source 3: one-time port of the hand-rolled matrix simulator
#
# Ported verbatim from test_timeseries_arma_garch.py::_simulate_handrolled
# so the frozen values are bit-identical to what that runtime path
# produced. The four labels have no third-party equivalent:
#   * TGARCH  — rugarch has no sigma-form (absolute-value) recursion with
#               separate positive/negative ARCH coefficients.
#   * QGARCH  — rugarch has no quadratic/asymmetric psi * eps_{t-1} term.
#   * gh      — copulax's generalised-hyperbolic parameterisation
#               (lamb, chi, psi, gamma) is not rugarch's ghyp.
#   * skewed_t— copulax's (nu, gamma) skewed-t is not rugarch's sstd.
# This recursion executes ONLY here, at regeneration time.
# ---------------------------------------------------------------------------

#: ARMA mean truth shared by all four matrix labels.
_MATRIX_MEAN_TRUTH = {"phi": 0.5, "theta": 0.3, "mu": 0.10}

#: Per-label variance truth, variant tag, residual law and shape truth.
MATRIX_CASES: tuple[dict[str, Any], ...] = (
    {
        "label": "arma11_tgarch11_normal",
        "variant": "tgarch",
        "residual": "normal",
        "residual_shape": {},
        "var_truth": {"omega": 0.02, "alpha_pos": 0.10,
                      "alpha_neg": 0.20, "beta": 0.70},
    },
    {
        "label": "arma11_qgarch11_normal",
        "variant": "qgarch",
        "residual": "normal",
        "residual_shape": {},
        "var_truth": {"omega": 0.05, "alpha": 0.10,
                      "psi": -0.05, "beta": 0.85},
    },
    {
        "label": "arma11_garch11_gh",
        "variant": "garch",
        "residual": "gh",
        "residual_shape": {"lamb": 0.0, "chi": 1.0, "psi": 1.0, "gamma": 0.0},
        "var_truth": {"omega": 0.05, "alpha": 0.10, "beta": 0.85},
    },
    {
        "label": "arma11_garch11_skewedt",
        "variant": "garch",
        "residual": "skewed_t",
        "residual_shape": {"nu": 6.0, "gamma": 0.2},
        "var_truth": {"omega": 0.05, "alpha": 0.10, "beta": 0.85},
    },
)


def matrix_seed(label: str) -> int:
    """Stable, process-independent PRNG seed for a matrix label.

    Identical to ``test_timeseries_arma_garch.py::_deterministic_seed``:
    the leading four bytes of ``sha256(label)`` as a big-endian integer,
    reduced modulo ``2 ** 31``.  ``hash()`` is randomised per process
    unless ``PYTHONHASHSEED`` is pinned, so a digest is used instead.

    Parameters
    ----------
    label : str
        The matrix label.

    Returns
    -------
    int
        The seed for :func:`jax.random.PRNGKey`.
    """
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


def _draw_standardised_z(residual: str, shape: dict, n: int, seed: int):
    """Draw ``n`` iid standardised residuals through the production wrapper.

    Parameters
    ----------
    residual : str
        Name of the copulax residual law (``"normal"``, ``"gh"``,
        ``"skewed_t"``).
    shape : dict
        Shape-parameter truth for that law.
    n : int
        Number of draws.
    seed : int
        Seed for :func:`jax.random.PRNGKey`.

    Returns
    -------
    numpy.ndarray
        The standardised draws.
    """
    _enable_x64()

    import jax
    from copulax.univariate import gh, normal, skewed_t
    from copulax._src.timeseries._residuals._standardise import (
        StandardisedResidual,
    )

    dist = {"normal": normal, "gh": gh, "skewed_t": skewed_t}[residual]
    z = StandardisedResidual(dist).rvs(
        size=(n,), shape_params=shape, key=jax.random.PRNGKey(seed),
    )
    return np.asarray(z)


def simulate_matrix_case(case: dict[str, Any], n: int = 2000) -> tuple[np.ndarray, str]:
    r"""Simulate one hand-rolled ARMA(1, 1)-variant matrix series.

    The mean equation is the centred ARMA(1, 1)

    .. math::
        y_t = \mu + \phi (y_{t-1} - \mu) + \theta \varepsilon_{t-1}
              + \varepsilon_t,

    driven by one of three variance recursions, each seeded at unit
    variance and floored exactly as the runtime simulator floored it:

    * ``tgarch`` — :math:`\sigma_t = \max(\omega
      + \alpha^{+} \max(\varepsilon_{t-1}, 0)
      + \alpha^{-} \max(-\varepsilon_{t-1}, 0)
      + \beta \sigma_{t-1},\ 10^{-6})`, a **standard-deviation**-form
      recursion.
    * ``qgarch`` — :math:`\sigma^2_t = \max(\omega
      + \alpha \varepsilon^2_{t-1} + \psi \varepsilon_{t-1}
      + \beta \sigma^2_{t-1},\ 10^{-12})`.
    * ``garch``  — :math:`\sigma^2_t = \max(\omega
      + \alpha \varepsilon^2_{t-1} + \beta \sigma^2_{t-1},\ 10^{-12})`.

    :data:`BURN_IN` leading observations are discarded.

    Parameters
    ----------
    case : dict
        One entry of :data:`MATRIX_CASES`.
    n : int, optional
        Number of observations to keep.  Default ``2000``.

    Returns
    -------
    tuple[numpy.ndarray, str]
        The simulated series (shape ``(n,)``, float64) and a
        human-readable provenance ``spec`` string.

    Note
    ----
    If you intend to jit wrap this function, ensure that ``n`` is a
    static argument.  (It is a regeneration-time utility and is never
    traced in practice.)
    """
    variant = case["variant"]
    truth = case["var_truth"]
    seed = matrix_seed(case["label"])
    total = n + BURN_IN

    z = _draw_standardised_z(
        case["residual"], case["residual_shape"], total, seed,
    )

    phi = float(_MATRIX_MEAN_TRUTH["phi"])
    theta = float(_MATRIX_MEAN_TRUTH["theta"])
    mu = float(_MATRIX_MEAN_TRUTH["mu"])
    omega = float(truth["omega"])

    eps_lag = 0.0
    # Centred-form ARMA: the unconditional mean of y_t IS mu (no AR
    # rescaling required), so the AR lag is seeded at the unconditional
    # mean.
    y_lag = mu

    if variant == "tgarch":
        alpha_pos = float(truth["alpha_pos"])
        alpha_neg = float(truth["alpha_neg"])
        beta = float(truth["beta"])
        sigma_lag = 1.0
    elif variant in ("qgarch", "garch"):
        alpha = float(truth["alpha"])
        beta = float(truth["beta"])
        psi = float(truth["psi"]) if variant == "qgarch" else 0.0
        eps_sq_lag = 1.0
        var_lag = 1.0
    else:  # pragma: no cover - guarded by MATRIX_CASES
        raise ValueError(f"unsupported matrix variant {variant!r}")

    y = np.zeros(total)
    for t in range(total):
        if variant == "tgarch":
            sigma_t = max(
                omega + alpha_pos * max(eps_lag, 0.0)
                + alpha_neg * max(-eps_lag, 0.0) + beta * sigma_lag,
                1e-6,
            )
            sigma2_t = sigma_t * sigma_t
        elif variant == "qgarch":
            sigma2_t = max(
                omega + alpha * eps_sq_lag + psi * eps_lag + beta * var_lag,
                1e-12,
            )
            sigma_t = float(np.sqrt(sigma2_t))
        else:
            sigma2_t = max(
                omega + alpha * eps_sq_lag + beta * var_lag, 1e-12,
            )
            sigma_t = float(np.sqrt(sigma2_t))

        eps_t = sigma_t * float(z[t])
        mu_t = mu + phi * (y_lag - mu) + theta * eps_lag
        y_t = mu_t + eps_t
        y[t] = y_t

        # Lag updates.
        if variant == "tgarch":
            sigma_lag = sigma_t
        else:
            eps_sq_lag = eps_t * eps_t
            var_lag = sigma2_t
        eps_lag = eps_t
        y_lag = y_t

    out = np.asarray(y[BURN_IN:], dtype=np.float64)
    assert out.shape == (n,), (out.shape, n)
    assert np.all(np.isfinite(out))

    var_desc = " ".join(f"{k}={v:g}" for k, v in truth.items())
    shape_desc = (
        " ".join(f"{k}={v:g}" for k, v in case["residual_shape"].items())
        or "(no shape params)"
    )
    spec = (
        f"ARMA(1,1) phi={phi:g} theta={theta:g} mu={mu:g} "
        f"+ {variant.upper()}(1,1) {var_desc}; "
        f"residual={case['residual']} {shape_desc} standardised to "
        f"zero mean / unit variance via StandardisedResidual; "
        f"variance lags seeded at 1.0; n.sim={total} burn_in={BURN_IN}; "
        f"z ~ StandardisedResidual.rvs(key=PRNGKey({seed}))"
    )
    return out, spec


def load_matrix_series() -> dict[str, dict[str, Any]]:
    """Generate the four one-time hand-rolled matrix series.

    Returns
    -------
    dict
        ``{name: {"y": np.ndarray, "provenance": dict}}``.
    """
    import copulax
    import jax

    engine = (
        f"one-time python port of "
        f"test_timeseries_arma_garch._simulate_handrolled "
        f"(copulax {copulax.__version__}, jax {jax.__version__})"
    )
    out: dict[str, dict[str, Any]] = {}
    for case in MATRIX_CASES:
        y, spec = simulate_matrix_case(case)
        out[f"matrix_{case['label']}_n{y.size}"] = {
            "y": y,
            "provenance": {
                "generator": "generate_frozen_series_handrolled.py",
                "engine": engine,
                "spec": spec,
                "seed": matrix_seed(case["label"]),
                "n": int(y.size),
            },
        }
    return out


# ---------------------------------------------------------------------------
# Source 4: one-time port of the module-local variance-variant simulators
#
# Ported verbatim from the ``_simulate_*`` helpers that used to live in
# test_timeseries_variance.py (IGARCH, GJR-GARCH, EGARCH, TGARCH, QGARCH,
# GARCH-M) and from the near-boundary ARMA-GARCH simulator inlined in
# test_timeseries_arma_garch.py::TestNearBoundaryStability.  None of the
# six variance recursions is reachable through rugarch in the copulax
# parameterisation:
#   * IGARCH   — the runtime helper seeds both variance lags at 1.0, not
#                at rugarch's backcast; the persistence-pinned series is
#                only meaningful under that seeding.
#   * GJR      — copulax's gamma multiplies the NEGATIVE indicator on the
#                previous eps; rugarch's gjrGARCH uses eta1 on a signed
#                news-impact term with a different pre-sample.
#   * EGARCH   — Nelson's (alpha on z, gamma on |z| - E|z|) split with the
#                log-variance lag seeded at omega / (1 - beta).
#   * TGARCH   — Zakoian sigma-form with separate positive/negative ARCH
#                coefficients; rugarch's fGARCH/TGARCH is not this
#                recursion (see TestTGARCHFGarchReference for the mapping
#                that IS cross-validated).
#   * QGARCH   — Sentana's psi * eps_{t-1} term has no rugarch analogue.
#   * GARCH-M  — variance-in-mean on the LEVEL, with mu_t + lambda *
#                sigma2_t, seeded at the unconditional variance.
# Freezing them verbatim keeps every downstream assertion bit-identical to
# the values the runtime path produced, so no assertion in those regions
# moves.  These recursions execute ONLY here, at regeneration time.
# ---------------------------------------------------------------------------

#: Burn-in of the near-boundary joint simulator, which discards its own
#: leading window rather than using :data:`BURN_IN` implicitly.
_NEAR_BOUNDARY_BURN_IN = 500


def _variance_variant_series(name: str, n: int, seed: int, **params):
    r"""Run one ported variance-variant recursion under jax.

    Every recursion below is the verbatim ``jax.lax.scan`` body of the
    ``test_timeseries_variance.py`` helper it replaces, including the
    pre-sample seeding.  Running it under jax (rather than re-deriving it
    in numpy) is what makes the frozen values bit-identical to the
    runtime path: the draws come from the same
    :func:`jax.random.normal` stream and the arithmetic runs at the same
    precision (double — see :func:`_enable_x64`).

    Parameters
    ----------
    name : str
        Variant tag — one of ``"igarch11"``, ``"gjr11"``, ``"egarch11"``,
        ``"tgarch11"``, ``"qgarch11"``, ``"garchm11"``.
    n : int
        Number of observations.
    seed : int
        Seed for :func:`jax.random.PRNGKey`.
    **params
        The variant's truth parameters.

    Returns
    -------
    tuple[numpy.ndarray, str]
        The float64 series and a human-readable spec string.

    Raises
    ------
    ValueError
        If ``name`` is not a known variant tag.
    """
    _enable_x64()

    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(seed)
    z = jax.random.normal(key, (n,))

    if name == "igarch11":
        omega, alpha, beta = (
            params["omega"], params["alpha"], params["beta"],
        )
        assert abs((alpha + beta) - 1.0) < 1e-10, "IGARCH requires alpha+beta=1"

        def step(carry, z_t):
            sigma2_prev, eps2_prev = carry
            sigma2_t = omega + alpha * eps2_prev + beta * sigma2_prev
            eps_t = jnp.sqrt(sigma2_t) * z_t
            return (sigma2_t, eps_t * eps_t), eps_t

        _, out = jax.lax.scan(step, (1.0, 1.0), z)
        spec = (
            f"IGARCH(1,1) residual series, omega={omega}, alpha={alpha}, "
            f"beta={beta} (alpha+beta=1), normal innovations, variance "
            f"lags seeded at 1.0"
        )

    elif name == "gjr11":
        omega, alpha, gamma, beta = (
            params["omega"], params["alpha"], params["gamma"], params["beta"],
        )
        sigma2_uncond = omega / (1.0 - alpha - 0.5 * gamma - beta)

        def step(carry, z_t):
            sigma2_prev, eps_prev = carry
            eps_sq_prev = eps_prev ** 2
            neg_eps_sq_prev = jnp.where(eps_prev < 0, eps_sq_prev, 0.0)
            sigma2_t = (
                omega
                + alpha * eps_sq_prev
                + gamma * neg_eps_sq_prev
                + beta * sigma2_prev
            )
            eps_t = jnp.sqrt(sigma2_t) * z_t
            return (sigma2_t, eps_t), eps_t

        _, out = jax.lax.scan(
            step, (sigma2_uncond, jnp.array(0.0)), z,
        )
        spec = (
            f"GJR-GARCH(1,1) residual series, omega={omega}, alpha={alpha}, "
            f"gamma={gamma}, beta={beta}, normal innovations, variance lag "
            f"seeded at the unconditional variance"
        )

    elif name == "egarch11":
        omega, alpha, gamma, beta = (
            params["omega"], params["alpha"], params["gamma"], params["beta"],
        )
        e_abs_z = (2.0 / jnp.pi) ** 0.5

        def step(carry, z_t):
            log_var_prev, z_prev = carry
            log_var_t = (
                omega
                + alpha * z_prev
                + gamma * (jnp.abs(z_prev) - e_abs_z)
                + beta * log_var_prev
            )
            sigma_t = jnp.exp(0.5 * log_var_t)
            eps_t = sigma_t * z_t
            return (log_var_t, z_t), eps_t

        log_var_init = omega / (1.0 - beta) if beta != 1 else 0.0
        _, out = jax.lax.scan(step, (log_var_init, jnp.array(0.0)), z)
        spec = (
            f"EGARCH(1,1) residual series (Nelson 1991), omega={omega}, "
            f"alpha={alpha} (leverage on z), gamma={gamma} (size on "
            f"|z|-E|z|), beta={beta}, normal innovations, log-variance lag "
            f"seeded at omega/(1-beta)"
        )

    elif name == "tgarch11":
        omega, alpha_pos, alpha_neg, beta = (
            params["omega"], params["alpha_pos"], params["alpha_neg"],
            params["beta"],
        )
        e_pos = (2.0 / jnp.pi) ** 0.5 / 2
        persistence = e_pos * alpha_pos + e_pos * alpha_neg + beta
        sigma_uncond = omega / (1.0 - persistence)

        def step(carry, z_t):
            sigma_prev, eps_prev = carry
            eps_pos_prev = jnp.maximum(eps_prev, 0.0)
            eps_neg_prev = jnp.maximum(-eps_prev, 0.0)
            sigma_t = (
                omega
                + alpha_pos * eps_pos_prev
                + alpha_neg * eps_neg_prev
                + beta * sigma_prev
            )
            eps_t = sigma_t * z_t
            return (sigma_t, eps_t), eps_t

        _, out = jax.lax.scan(step, (sigma_uncond, jnp.array(0.0)), z)
        spec = (
            f"TGARCH(1,1) residual series (Zakoian sigma-form), "
            f"omega={omega}, alpha_pos={alpha_pos}, alpha_neg={alpha_neg}, "
            f"beta={beta}, normal innovations, sigma lag seeded at the "
            f"unconditional sigma"
        )

    elif name == "qgarch11":
        omega, alpha, psi, beta = (
            params["omega"], params["alpha"], params["psi"], params["beta"],
        )
        sigma2_uncond = omega / (1.0 - alpha - beta)

        def step(carry, z_t):
            sigma2_prev, eps_prev = carry
            sigma2_t = (
                omega + alpha * eps_prev ** 2 + psi * eps_prev
                + beta * sigma2_prev
            )
            sigma2_t = jnp.maximum(sigma2_t, 1e-10)
            eps_t = jnp.sqrt(sigma2_t) * z_t
            return (sigma2_t, eps_t), eps_t

        _, out = jax.lax.scan(step, (sigma2_uncond, jnp.array(0.0)), z)
        spec = (
            f"QGARCH(1,1) residual series (Sentana 1995), omega={omega}, "
            f"alpha={alpha}, psi={psi}, beta={beta}, normal innovations, "
            f"variance floored at 1e-10, variance lag seeded at the "
            f"unconditional variance"
        )

    elif name == "garchm11":
        mu_t, lambda_m, omega, alpha, beta = (
            params["mu"], params["lambda_m"], params["omega"],
            params["alpha"], params["beta"],
        )
        sigma2_uncond = omega / (1.0 - alpha - beta)

        def step(carry, z_t):
            sigma2_prev, eps2_prev = carry
            sigma2_t = omega + alpha * eps2_prev + beta * sigma2_prev
            sigma_t = jnp.sqrt(sigma2_t)
            mu_at_t = mu_t + lambda_m * sigma2_t
            eps_t = sigma_t * z_t
            y_t = mu_at_t + eps_t
            return (sigma2_t, eps_t * eps_t), y_t

        _, out = jax.lax.scan(step, (sigma2_uncond, sigma2_uncond), z)
        spec = (
            f"GARCH-M(1,1) level series, mu={mu_t}, lambda={lambda_m}, "
            f"omega={omega}, alpha={alpha}, beta={beta}, normal "
            f"innovations, variance lags seeded at the unconditional "
            f"variance"
        )

    else:
        raise ValueError(f"unknown variance variant {name!r}")

    return np.asarray(out, dtype=np.float64), spec


#: ``(frozen name, variant tag, n, seed, truth parameters)`` for every
#: variance-variant series the test suite used to simulate at runtime.
VARIANCE_VARIANT_CASES: tuple[tuple[str, str, int, int, dict], ...] = (
    ("igarch11_n500_s2", "igarch11", 500, 2,
     {"omega": 0.05, "alpha": 0.10, "beta": 0.90}),
    ("igarch11_n2000_s2", "igarch11", 2000, 2,
     {"omega": 0.05, "alpha": 0.10, "beta": 0.90}),
    ("gjr11_n2000_s2", "gjr11", 2000, 2,
     {"omega": 0.05, "alpha": 0.05, "gamma": 0.10, "beta": 0.85}),
    ("egarch11_n500_s2", "egarch11", 500, 2,
     {"omega": -0.05, "alpha": -0.05, "gamma": 0.10, "beta": 0.95}),
    ("egarch11_n2000_s2", "egarch11", 2000, 2,
     {"omega": -0.05, "alpha": -0.05, "gamma": 0.10, "beta": 0.95}),
    ("tgarch11_n500_s2", "tgarch11", 500, 2,
     {"omega": 0.038, "alpha_pos": 0.10, "alpha_neg": 0.18, "beta": 0.85}),
    ("tgarch11_n2000_s2", "tgarch11", 2000, 2,
     {"omega": 0.038, "alpha_pos": 0.10, "alpha_neg": 0.18, "beta": 0.85}),
    ("qgarch11_n500_s2", "qgarch11", 500, 2,
     {"omega": 0.05, "alpha": 0.10, "psi": -0.05, "beta": 0.85}),
    ("qgarch11_n2000_s2", "qgarch11", 2000, 2,
     {"omega": 0.05, "alpha": 0.10, "psi": -0.05, "beta": 0.85}),
    ("garchm11_n500_s2", "garchm11", 500, 2,
     {"mu": 0.05, "lambda_m": 0.20, "omega": 0.05, "alpha": 0.10,
      "beta": 0.85}),
    ("garchm11_n2000_s2", "garchm11", 2000, 2,
     {"mu": 0.05, "lambda_m": 0.20, "omega": 0.05, "alpha": 0.10,
      "beta": 0.85}),
)


def simulate_near_boundary_joint(
    n: int = 1500, seed: int = 99,
) -> tuple[np.ndarray, str]:
    r"""Port of the near-boundary AR(1)-GARCH(1, 1) joint simulator.

    Verbatim from ``test_timeseries_arma_garch.py`` — a centred-form
    AR(1) mean driven by a near-integrated GARCH(1, 1)
    (:math:`\alpha + \beta = 0.99`), simulated in numpy float64 over a
    jax standard-normal draw and burnt in for
    :data:`_NEAR_BOUNDARY_BURN_IN` steps.

    Parameters
    ----------
    n : int, optional
        Observations kept after burn-in.  Default ``1500``.
    seed : int, optional
        Seed for :func:`jax.random.PRNGKey`.  Default ``99``.

    Returns
    -------
    tuple[numpy.ndarray, str]
        The float64 series and a human-readable spec string.
    """
    _enable_x64()

    import jax

    truth = {
        "phi": 0.3, "mu": 0.0,
        "omega": 0.001, "alpha": 0.05, "beta": 0.94,
    }
    total = n + _NEAR_BOUNDARY_BURN_IN
    z = np.asarray(jax.random.normal(jax.random.PRNGKey(seed), (total,)))

    eps_sq_lag = 1.0
    var_lag = 1.0
    y = np.zeros(total)
    y_lag = float(truth["mu"])
    for t in range(total):
        sigma2 = max(
            truth["omega"] + truth["alpha"] * eps_sq_lag
            + truth["beta"] * var_lag,
            1e-12,
        )
        sigma = float(np.sqrt(sigma2))
        eps = sigma * float(z[t])
        y_t = truth["mu"] + truth["phi"] * (y_lag - truth["mu"]) + eps
        y[t] = y_t
        eps_sq_lag = eps * eps
        var_lag = sigma2
        y_lag = y_t

    spec = (
        "near-boundary AR(1)-GARCH(1,1) level series, phi=0.3, mu=0.0, "
        "omega=0.001, alpha=0.05, beta=0.94 (persistence 0.99), normal "
        f"innovations, variance lags seeded at 1.0, burn-in "
        f"{_NEAR_BOUNDARY_BURN_IN}"
    )
    return y[_NEAR_BOUNDARY_BURN_IN:], spec


def load_variance_variant_series() -> dict[str, dict[str, Any]]:
    """Generate the one-time variance-variant and near-boundary series.

    Returns
    -------
    dict
        ``{name: {"y": np.ndarray, "provenance": dict}}``.
    """
    import copulax
    import jax

    engine = (
        f"one-time python port of the test-module variance-variant "
        f"simulators (copulax {copulax.__version__}, jax {jax.__version__})"
    )
    out: dict[str, dict[str, Any]] = {}
    for name, variant, n, seed, truth in VARIANCE_VARIANT_CASES:
        y, spec = _variance_variant_series(variant, n, seed, **truth)
        out[name] = {
            "y": y,
            "provenance": {
                "generator": "generate_frozen_series_handrolled.py",
                "engine": engine,
                "spec": spec,
                "seed": seed,
                "n": int(y.size),
            },
        }

    y, spec = simulate_near_boundary_joint()
    out["ar1garch11_nearboundary_n1500_s99"] = {
        "y": y,
        "provenance": {
            "generator": "generate_frozen_series_handrolled.py",
            "engine": engine,
            "spec": spec,
            "seed": 99,
            "n": int(y.size),
        },
    }
    return out


# ---------------------------------------------------------------------------
# Merge + emit
# ---------------------------------------------------------------------------

def build_corpus() -> dict[str, dict[str, Any]]:
    """Collect all four sources and attach a SHA-256 to each series.

    Returns
    -------
    dict
        ``{name: {"y": np.ndarray, "provenance": dict}}``, ordered
        rugarch first, then statsmodels, then the matrix series, then the
        variance-variant series.

    Raises
    ------
    RuntimeError
        If two sources produce the same series name.
    """
    corpus: dict[str, dict[str, Any]] = {}
    for loader in (load_rugarch_series, load_statsmodels_series,
                   load_matrix_series, load_variance_variant_series):
        for name, entry in loader().items():
            if name in corpus:
                raise RuntimeError(f"duplicate frozen-series name {name!r}")
            corpus[name] = entry

    for name, entry in corpus.items():
        y = entry["y"]
        assert y.dtype == np.float64, (name, y.dtype)
        entry["provenance"]["sha256"] = hashlib.sha256(
            np.asarray(y).tobytes()
        ).hexdigest()
    return corpus


def _format_array(y: np.ndarray, indent: int) -> str:
    """Render a float64 array as a wrapped ``np.array([...])`` literal.

    ``repr(float)`` is the shortest decimal string that round-trips a
    double exactly, so no precision is lost.
    """
    pad = " " * indent
    values = [repr(float(v)) for v in y]
    lines = [
        pad + ", ".join(values[i:i + _VALUES_PER_LINE]) + ","
        for i in range(0, len(values), _VALUES_PER_LINE)
    ]
    body = "\n".join(lines)
    return f"np.array([\n{body}\n{' ' * (indent - 4)}], dtype=float)"


def _format_provenance(provenance: dict[str, Any], indent: int) -> str:
    """Render a provenance dict as a Python literal, one key per line."""
    pad = " " * indent
    order = ("generator", "engine", "spec", "seed", "n", "sha256")
    assert set(order) == set(provenance), sorted(provenance)
    lines = [
        f"{pad}{key!r}: {provenance[key]!r},"
        for key in order
    ]
    body = "\n".join(lines)
    return "{\n" + body + "\n" + " " * (indent - 4) + "}"


_MODULE_DOCSTRING = '''"""Auto-generated frozen test series for the time-series test family.

DO NOT EDIT BY HAND. Regenerate with::

    python copulax/tests/_r_reference/generate_frozen_series_handrolled.py

which re-runs both committed regenerators —
``generate_frozen_series.R`` (rugarch) and
``generate_frozen_series_handrolled.py`` (statsmodels + the two one-time
hand-rolled ports) — and rewrites this module in full.

Every ``test_timeseries_*.py`` series lives here. Nothing in the test
suite simulates a process at runtime any more: the data is committed, so
it is identical on every machine, every CI leg and every rerun, and no
copulax-authored recursion sits on the input side of a copulax
assertion.

Layout
------
``FROZEN_SERIES`` maps a series name to::

    {{
        "y": np.ndarray,          # float64, shape (n,)
        "provenance": {{
            "generator": str,     # which committed script produced it
            "engine":    str,     # third-party engine + version
            "spec":      str,     # the DGP, in full
            "seed":      int,     # the seed that draws it
            "n":         int,     # len(y)
            "sha256":    str,     # sha256 of np.asarray(y).tobytes()
        }},
    }}

Series names encode DGP, length and seed, e.g. ``garch11_n2000_s2`` is a
GARCH(1, 1) residual series of length 2000 drawn with seed 2. Parameter
tags appear in the name only where they distinguish two series of the
same family (``ar1_p060_m025_sd050_n2000_s42``); the full parameter set is
always in ``provenance["spec"]``.

Engines
-------
{engine_summary}

Consumers hold float64 here and downcast at the call site (jax runs in
float32 unless x64 is enabled), exactly as the other reference modules in
this directory do.

Integrity
---------
Each entry carries the SHA-256 of its own bytes::

    import hashlib, numpy as np
    for name, entry in FROZEN_SERIES.items():
        digest = hashlib.sha256(np.asarray(entry["y"]).tobytes()).hexdigest()
        assert digest == entry["provenance"]["sha256"], name

This module imports numpy and nothing else — no jax, no copulax — so
loading a series costs one module import.

Corpus: {n_series} series, {n_obs} observations.
"""
'''


def write_module(corpus: dict[str, dict[str, Any]], path: Path) -> None:
    """Write ``frozen_series_data.py`` for ``corpus``.

    Parameters
    ----------
    corpus : dict
        Output of :func:`build_corpus`.
    path : pathlib.Path
        Destination file.
    """
    by_engine: dict[str, list[str]] = {}
    for entry in corpus.values():
        by_engine.setdefault(entry["provenance"]["engine"], []).append("")
    engine_summary = "\n".join(
        f"* {engine} — {len(names)} series"
        for engine, names in sorted(by_engine.items())
    )
    n_obs = sum(int(entry["y"].size) for entry in corpus.values())

    parts: list[str] = [
        _MODULE_DOCSTRING.format(
            engine_summary=engine_summary,
            n_series=len(corpus),
            n_obs=f"{n_obs:,}",
        ),
        "\nimport numpy as np\n\n\nFROZEN_SERIES = {\n",
    ]
    for name, entry in corpus.items():
        parts.append(f"    {name!r}: {{\n")
        parts.append(f"        'y': {_format_array(entry['y'], 12)},\n")
        parts.append(
            f"        'provenance': "
            f"{_format_provenance(entry['provenance'], 12)},\n"
        )
        parts.append("    },\n")
    parts.append("}\n")
    path.write_text("".join(parts), encoding="utf-8")


def verify_written_module(path: Path, corpus: dict[str, dict[str, Any]]) -> None:
    """Load the written module and re-check every series against ``corpus``.

    Confirms that the emitted decimal literals round-trip to the exact
    doubles that were generated, and that every recorded SHA-256 matches
    the bytes actually written.

    Raises
    ------
    AssertionError
        On any mismatch.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_frozen_series_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = module.FROZEN_SERIES

    assert set(loaded) == set(corpus), (
        sorted(set(loaded) ^ set(corpus))
    )
    for name, entry in loaded.items():
        y = np.asarray(entry["y"])
        assert y.dtype == np.float64, (name, y.dtype)
        assert y.tobytes() == corpus[name]["y"].tobytes(), (
            f"{name}: written literals do not round-trip to the "
            f"generated doubles"
        )
        digest = hashlib.sha256(y.tobytes()).hexdigest()
        assert digest == entry["provenance"]["sha256"], name
        assert y.size == entry["provenance"]["n"], name


def main() -> int:
    """Regenerate, write and verify ``frozen_series_data.py``.

    Returns
    -------
    int
        Process exit status (``0`` on success).
    """
    corpus = build_corpus()
    write_module(corpus, _OUT_PATH)
    verify_written_module(_OUT_PATH, corpus)

    n_obs = sum(int(entry["y"].size) for entry in corpus.values())
    by_engine: dict[str, int] = {}
    for entry in corpus.values():
        engine = entry["provenance"]["engine"]
        by_engine[engine] = by_engine.get(engine, 0) + 1

    print(f"wrote {_OUT_PATH}")
    print(f"  series:       {len(corpus)}")
    print(f"  observations: {n_obs:,}")
    print(f"  bytes:        {_OUT_PATH.stat().st_size:,}")
    for engine, count in sorted(by_engine.items()):
        print(f"  {count:3d} from {engine}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
