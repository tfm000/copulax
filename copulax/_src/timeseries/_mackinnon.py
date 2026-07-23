r"""JAX port of statsmodels' MacKinnon ADF p-value polynomial and
asymptotic critical-value response surface.

This module vendors numerical constants from
``statsmodels.tsa.adfvalues`` (BSD-3-Clause) so CopulAX can compute
ADF p-values and critical values without a runtime statsmodels
dependency.  The polynomial *control flow* is reimplemented in JAX
for JIT / autograd compatibility; only the literal numerical tables
are copied.

Provenance of the vendored numbers (see attribution block at the
bottom of this file for the full BSD-3 notice):

* The asymptotic response surface for the 1% / 5% / 10% critical
  values (``tau_2010s``) comes from MacKinnon (2010) — *Critical
  Values for Cointegration Tests*, Queen's University Working Paper
  1227 — for the ``c`` (constant) and ``ct`` (constant + trend)
  regressions.  The ``n`` (no constant) row is unchanged from
  MacKinnon (1996) since the 2010 paper deliberately excluded that
  case (statsmodels notes this in its ``mackinnoncrit`` docstring).

* The continuous p-value polynomial coefficients
  (``tau_*_smallp`` / ``tau_*_largep`` plus the
  ``tau_star_*`` / ``tau_min_*`` / ``tau_max_*`` cutoffs) come from
  MacKinnon (1994) — *Approximate Asymptotic Distribution Functions
  for Unit-Root and Cointegration Tests*, JBES 12.2.  MacKinnon (2010)
  re-simulated the discrete critical-value table but did not publish
  updated polynomial coefficients in a directly usable form, so
  statsmodels' ``mackinnonp`` still uses the 1994 polynomial — and so
  do we.

CopulAX only needs the ``N=1`` (univariate ADF, no cointegration)
case; everything below is the row-0 entry of statsmodels' tables.
The ``ctt`` regression and the Z-statistic data blocks in
statsmodels' source are not used here and were trimmed out.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm
from jax.typing import ArrayLike


# =====================================================================
# 1994 polynomial cutoffs — vendored verbatim from
# statsmodels.tsa.adfvalues lines 9-17 (n / c / ct rows; ctt dropped).
# Index [N-1]; for univariate ADF (N=1) we use [0].
# =====================================================================

tau_star_nc = [-1.04, -1.53, -2.68, -3.09, -3.07, -3.77]
tau_min_nc = [-19.04, -19.62, -21.21, -23.25, -21.63, -25.74]
tau_max_nc = [jnp.inf, 1.51, 0.86, 0.88, 1.05, 1.24]
tau_star_c = [-1.61, -2.62, -3.13, -3.47, -3.78, -3.93]
tau_min_c = [-18.83, -18.86, -23.48, -28.07, -25.96, -23.27]
tau_max_c = [2.74, 0.92, 0.55, 0.61, 0.79, 1]
tau_star_ct = [-2.89, -3.19, -3.50, -3.65, -3.80, -4.36]
tau_min_ct = [-16.18, -21.15, -25.37, -26.63, -26.53, -26.18]
tau_max_ct = [0.7, 0.63, 0.71, 0.93, 1.19, 1.42]

_tau_maxs = {"n": tau_max_nc, "c": tau_max_c, "ct": tau_max_ct}
_tau_mins = {"n": tau_min_nc, "c": tau_min_c, "ct": tau_min_ct}
_tau_stars = {"n": tau_star_nc, "c": tau_star_c, "ct": tau_star_ct}


# =====================================================================
# 1994 polynomial coefficients (small p) — vendored verbatim from
# statsmodels.tsa.adfvalues lines 42-67 (n / c / ct; ctt dropped).
# After the post-multiplication by ``small_scaling`` each ``tau_*_smallp``
# is a (6, 3) JAX array; row 0 is the N=1 case.
# =====================================================================

small_scaling = jnp.asarray([1, 1, 1e-2])
tau_nc_smallp = [
    [0.6344, 1.2378, 3.2496],
    [1.9129, 1.3857, 3.5322],
    [2.7648, 1.4502, 3.4186],
    [3.4336, 1.4835, 3.19],
    [4.0999, 1.5533, 3.59],
    [4.5388, 1.5344, 2.9807]]
tau_nc_smallp = jnp.asarray(tau_nc_smallp)*small_scaling

tau_c_smallp = [
    [2.1659, 1.4412, 3.8269],
    [2.92, 1.5012, 3.9796],
    [3.4699, 1.4856, 3.164],
    [3.9673, 1.4777, 2.6315],
    [4.5509, 1.5338, 2.9545],
    [5.1399, 1.6036, 3.4445]]
tau_c_smallp = jnp.asarray(tau_c_smallp)*small_scaling

tau_ct_smallp = [
    [3.2512, 1.6047, 4.9588],
    [3.6646, 1.5419, 3.6448],
    [4.0983, 1.5173, 2.9898],
    [4.5844, 1.5338, 2.8796],
    [5.0722, 1.5634, 2.9472],
    [5.53, 1.5914, 3.0392]]
tau_ct_smallp = jnp.asarray(tau_ct_smallp)*small_scaling

_tau_smallps = {"n": tau_nc_smallp, "c": tau_c_smallp, "ct": tau_ct_smallp}


# =====================================================================
# 1994 polynomial coefficients (large p) — vendored verbatim from
# statsmodels.tsa.adfvalues lines 87-110 (n / c / ct; ctt dropped).
# After the post-multiplication by ``large_scaling`` each ``tau_*_largep``
# is a (6, 4) JAX array; row 0 is the N=1 case.
# =====================================================================

large_scaling = jnp.asarray([1, 1e-1, 1e-1, 1e-2])
tau_nc_largep = [
    [0.4797, 9.3557, -0.6999, 3.3066],
    [1.5578, 8.558, -2.083, -3.3549],
    [2.2268, 6.8093, -3.2362, -5.4448],
    [2.7654, 6.4502, -3.0811, -4.4946],
    [3.2684, 6.8051, -2.6778, -3.4972],
    [3.7268, 7.167, -2.3648, -2.8288]]
tau_nc_largep = jnp.asarray(tau_nc_largep)*large_scaling

tau_c_largep = [
    [1.7339, 9.3202, -1.2745, -1.0368],
    [2.1945, 6.4695, -2.9198, -4.2377],
    [2.5893, 4.5168, -3.6529, -5.0074],
    [3.0387, 4.5452, -3.3666, -4.1921],
    [3.5049, 5.2098, -2.9158, -3.3468],
    [3.9489, 5.8933, -2.5359, -2.721]]
tau_c_largep = jnp.asarray(tau_c_largep)*large_scaling

tau_ct_largep = [
    [2.5261, 6.1654, -3.7956, -6.0285],
    [2.85, 5.272, -3.6622, -5.1695],
    [3.221, 5.255, -3.2685, -4.1501],
    [3.652, 5.9758, -2.7483, -3.2081],
    [4.0712, 6.6428, -2.3464, -2.546],
    [4.4735, 7.1757, -2.0681, -2.1196]]
tau_ct_largep = jnp.asarray(tau_ct_largep)*large_scaling

_tau_largeps = {"n": tau_nc_largep, "c": tau_c_largep, "ct": tau_ct_largep}


# =====================================================================
# 2010 critical-value response surface — vendored verbatim from
# statsmodels.tsa.adfvalues lines 318-446 (n / c / ct; ctt dropped).
#
# Each ``tau_*_2010`` array has shape (N_dim, 3, 4):
#   axis 0: number of cointegrating series (N=1, 2, ..., 12 for c/ct;
#           N=1 only for n)
#   axis 1: significance level — [1%, 5%, 10%]
#   axis 2: response-surface coefficients — [β_∞, β_1, β_2, β_3]
#           in 1/T:
#             τ_p(T) = β_∞ + β_1/T + β_2/T² + β_3/T³
#
# CopulAX uses ``[0, :, 0]`` (N=1 row, asymptotic β_∞ column) for the
# ``crit_values`` reporting array; the response-surface columns are
# preserved so a future finite-sample-corrected path can pick them up
# without re-vendoring.
# =====================================================================

tau_nc_2010 = [[
    [-2.56574, -2.2358, -3.627, 0],  # N = 1
    [-1.94100, -0.2686, -3.365, 31.223],
    [-1.61682, 0.2656, -2.714, 25.364]]]
tau_nc_2010 = jnp.asarray(tau_nc_2010)

tau_c_2010 = [
    [[-3.43035, -6.5393, -16.786, -79.433],  # N = 1, 1%
     [-2.86154, -2.8903, -4.234, -40.040],   # 5 %
     [-2.56677, -1.5384, -2.809, 0]],        # 10 %
    [[-3.89644, -10.9519, -33.527, 0],       # N = 2
     [-3.33613, -6.1101, -6.823, 0],
     [-3.04445, -4.2412, -2.720, 0]],
    [[-4.29374, -14.4354, -33.195, 47.433],  # N = 3
     [-3.74066, -8.5632, -10.852, 27.982],
     [-3.45218, -6.2143, -3.718, 0]],
    [[-4.64332, -18.1031, -37.972, 0],       # N = 4
     [-4.09600, -11.2349, -11.175, 0],
     [-3.81020, -8.3931, -4.137, 0]],
    [[-4.95756, -21.8883, -45.142, 0],       # N = 5
     [-4.41519, -14.0405, -12.575, 0],
     [-4.13157, -10.7417, -3.784, 0]],
    [[-5.24568, -25.6688, -57.737, 88.639],  # N = 6
     [-4.70693, -16.9178, -17.492, 60.007],
     [-4.42501, -13.1875, -5.104, 27.877]],
    [[-5.51233, -29.5760, -69.398, 164.295],  # N = 7
     [-4.97684, -19.9021, -22.045, 110.761],
     [-4.69648, -15.7315, -5.104, 27.877]],
    [[-5.76202, -33.5258, -82.189, 256.289],  # N = 8
     [-5.22924, -23.0023, -24.646, 144.479],
     [-4.95007, -18.3959, -7.344, 94.872]],
    [[-5.99742, -37.6572, -87.365, 248.316],  # N = 9
     [-5.46697, -26.2057, -26.627, 176.382],
     [-5.18897, -21.1377, -9.484, 172.704]],
    [[-6.22103, -41.7154, -102.680, 389.33],  # N = 10
     [-5.69244, -29.4521, -30.994, 251.016],
     [-5.41533, -24.0006, -7.514, 163.049]],
    [[-6.43377, -46.0084, -106.809, 352.752],  # N = 11
     [-5.90714, -32.8336, -30.275, 249.994],
     [-5.63086, -26.9693, -4.083, 151.427]],
    [[-6.63790, -50.2095, -124.156, 579.622],  # N = 12
     [-6.11279, -36.2681, -32.505, 314.802],
     [-5.83724, -29.9864, -2.686, 184.116]]]
tau_c_2010 = jnp.asarray(tau_c_2010)

tau_ct_2010 = [
    [[-3.95877, -9.0531, -28.428, -134.155],   # N = 1
     [-3.41049, -4.3904, -9.036, -45.374],
     [-3.12705, -2.5856, -3.925, -22.380]],
    [[-4.32762, -15.4387, -35.679, 0],         # N = 2
     [-3.78057, -9.5106, -12.074, 0],
     [-3.49631, -7.0815, -7.538, 21.892]],
    [[-4.66305, -18.7688, -49.793, 104.244],   # N = 3
     [-4.11890, -11.8922, -19.031, 77.332],
     [-3.83511, -9.0723, -8.504, 35.403]],
    [[-4.96940, -22.4694, -52.599, 51.314],    # N = 4
     [-4.42871, -14.5876, -18.228, 39.647],
     [-4.14633, -11.2500, -9.873, 54.109]],
    [[-5.25276, -26.2183, -59.631, 50.646],    # N = 5
     [-4.71537, -17.3569, -22.660, 91.359],
     [-4.43422, -13.6078, -10.238, 76.781]],
    [[-5.51727, -29.9760, -75.222, 202.253],   # N = 6
     [-4.98228, -20.3050, -25.224, 132.03],
     [-4.70233, -16.1253, -9.836, 94.272]],
    [[-5.76537, -33.9165, -84.312, 245.394],   # N = 7
     [-5.23299, -23.3328, -28.955, 182.342],
     [-4.95405, -18.7352, -10.168, 120.575]],
    [[-6.00003, -37.8892, -96.428, 335.92],    # N = 8
     [-5.46971, -26.4771, -31.034, 220.165],
     [-5.19183, -21.4328, -10.726, 157.955]],
    [[-6.22288, -41.9496, -109.881, 466.068],  # N = 9
     [-5.69447, -29.7152, -33.784, 273.002],
     [-5.41738, -24.2882, -8.584, 169.891]],
    [[-6.43551, -46.1151, -120.814, 566.823],  # N = 10
     [-5.90887, -33.0251, -37.208, 346.189],
     [-5.63255, -27.2042, -6.792, 177.666]],
    [[-6.63894, -50.4287, -128.997, 642.781],  # N = 11
     [-6.11404, -36.4610, -36.246, 348.554],
     [-5.83850, -30.1995, -5.163, 210.338]],
    [[-6.83488, -54.7119, -139.800, 736.376],  # N = 12
     [-6.31127, -39.9676, -37.021, 406.051],
     [-6.03650, -33.2381, -6.606, 317.776]]]
tau_ct_2010 = jnp.asarray(tau_ct_2010)

tau_2010s = {"n": tau_nc_2010, "c": tau_c_2010, "ct": tau_ct_2010}


# =====================================================================
# JAX reimplementation of statsmodels' ``mackinnonp`` (and asymptotic
# critical-value reporter) for N=1.
#
# Hot-path JAX arrays are pre-converted at module load so the JIT
# trace doesn't see numpy → jax conversions in the critical path.
# Keys are always Python strings; static under jax.jit via
# ``static_argnames=("regression",)``.
# =====================================================================

_TAU_STAR_N1 = {k: float(v[0]) for k, v in _tau_stars.items()}
_TAU_MIN_N1 = {k: float(v[0]) for k, v in _tau_mins.items()}
_TAU_MAX_N1 = {k: float(v[0]) for k, v in _tau_maxs.items()}
_TAU_SMALLP_N1 = {
    k: jnp.asarray(v[0], dtype=float) for k, v in _tau_smallps.items()
}
_TAU_LARGEP_N1 = {
    k: jnp.asarray(v[0], dtype=float) for k, v in _tau_largeps.items()
}
_TAU_2010_N1_ASYMPTOTIC = {
    k: jnp.asarray(v[0, :, 0], dtype=float) for k, v in tau_2010s.items()
}


def mackinnonp_jit(teststat: ArrayLike, regression: str) -> Array:
    r"""JAX port of statsmodels' MacKinnon (1994) ADF p-value polynomial
    for the univariate (``N=1``) case.

    Returns the asymptotic p-value of the Augmented Dickey-Fuller
    t-statistic ``teststat`` under H₀ (unit root).  The function is
    bit-equivalent to ``statsmodels.tsa.adfvalues.mackinnonp(teststat,
    regression=regression, N=1)`` to within float64 reordering — see
    the parametric cross-validation test at
    ``copulax/tests/test_timeseries_diagnostics.py`` for the contract.

    Args:
        teststat: ADF τ-statistic; scalar or ``(...,)``-shape JAX array.
        regression: One of ``"n"`` (no constant), ``"c"`` (constant
            only — default in :func:`copulax._src.timeseries._unit_root.adf`),
            ``"ct"`` (constant + linear trend).  Static under JIT —
            pass via ``static_argnames=("regression",)``.

    Returns:
        Scalar (or shape-matched) JAX ``Array`` p-value in ``[0, 1]``.
        Outside the polynomial's calibration range the function returns
        ``0.0`` (below ``tau_min``) or ``1.0`` (above ``tau_max``) —
        same hard saturation as statsmodels.  Saturation cutoffs (N=1):
        ``n``: ``[-19.04, +∞)``; ``c``: ``[-18.83, +2.74]``; ``ct``:
        ``[-16.18, +0.70]``.

    Raises:
        ValueError: When ``regression`` is not one of ``{"n", "c", "ct"}``.
    """
    if regression not in ("n", "c", "ct"):
        raise ValueError(
            f"regression must be 'n', 'c', or 'ct'; got {regression!r}."
        )
    tau_star = _TAU_STAR_N1[regression]
    tau_min = _TAU_MIN_N1[regression]
    tau_max = _TAU_MAX_N1[regression]
    smallp = _TAU_SMALLP_N1[regression]
    largep = _TAU_LARGEP_N1[regression]

    stat = jnp.asarray(teststat, dtype=float)
    # Both polynomial branches; ``jnp.polyval`` follows numpy's
    # high-to-low coefficient order, hence the reversal.
    z_small = jnp.polyval(smallp[::-1], stat)
    z_large = jnp.polyval(largep[::-1], stat)
    z = jnp.where(stat <= tau_star, z_small, z_large)
    p_interior = norm.cdf(z)

    # Hard saturation outside the polynomial's calibration range —
    # matches statsmodels' early-return behaviour at lines 301-304 of
    # ``adfvalues.py``.
    return jnp.where(
        stat > tau_max,
        1.0,
        jnp.where(stat < tau_min, 0.0, p_interior),
    )


def mackinnon_asymptotic_crit(regression: str) -> Array:
    r"""Asymptotic 1% / 5% / 10% critical values for the ADF τ-statistic.

    For ``c`` and ``ct`` regressions these are the MacKinnon (2010)
    response-surface intercepts (β_∞ column of ``tau_2010s``); for
    ``n`` they are the MacKinnon (1996) values that the 2010 paper
    deliberately did not update (statsmodels documents this in its
    ``mackinnoncrit`` docstring).

    Args:
        regression: One of ``"n"``, ``"c"``, ``"ct"``.  Static under JIT.

    Returns:
        Shape-``(3,)`` JAX ``Array`` ordered ``[1%, 5%, 10%]``.

    Raises:
        ValueError: When ``regression`` is not one of ``{"n", "c", "ct"}``.
    """
    if regression not in ("n", "c", "ct"):
        raise ValueError(
            f"regression must be 'n', 'c', or 'ct'; got {regression!r}."
        )
    return _TAU_2010_N1_ASYMPTOTIC[regression]


__all__ = ["mackinnonp_jit", "mackinnon_asymptotic_crit"]


# =====================================================================
# Attribution
# =====================================================================
# The numerical constants vendored above are copied verbatim from
# ``statsmodels.tsa.adfvalues``.  statsmodels is distributed under
# the BSD-3-Clause licence reproduced below; per clause (a) the
# notice is preserved here.
#
# Copyright (C) 2006, Jonathan E. Taylor
# All rights reserved.
#
# Copyright (c) 2006-2008 Scipy Developers.
# All rights reserved.
#
# Copyright (c) 2009-2018 statsmodels Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#   a. Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
#   c. Neither the name of statsmodels nor the names of its
#      contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
# STATSMODELS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# References for the vendored numerical values:
#   MacKinnon, J.G. (1994). "Approximate Asymptotic Distribution
#     Functions for Unit-Root and Cointegration Tests."  Journal of
#     Business & Economics Statistics, 12.2, 167-176.
#   MacKinnon, J.G. (2010). "Critical Values for Cointegration
#     Tests."  Queen's University, Department of Economics Working
#     Paper 1227.
