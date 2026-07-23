r"""Shared summary-table formatter for fitted time-series models.

Every fitted ``ARMABase`` / ``GARCHBase`` / ``ArmaGarch`` instance routes
its ``summary()`` through :func:`format_summary` so the rendered tables
share one visual contract:

* A 78-char-wide header line, equals-bordered.
* A 7-column param table — ``param | estimate | CI | std err | z |
  P>|z| | sig`` — with R-style significance codes.
* Param sections separated by inline-labelled dashed lines
  (``---- Mean equation — ARMA(1, 1) -----...``); empty sections
  are silently suppressed.
* A residual-diagnostics sub-table: rows for Ljung-Box on residuals,
  Ljung-Box on squared residuals, ARCH-LM, ADF, and KPSS — each
  carrying a ``✓`` / ``✗`` glyph reflecting whether the decision
  matches what we'd expect for a well-specified model.
* An R-style significance-code legend.
* A footer line with log-likelihood / AIC / BIC / n_train.

Numerical helpers (z-stat, p-value, CI bounds) come from
:mod:`jax.scipy.stats.norm`; everything else is pure Python.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from jax.scipy.stats import norm as _norm


###############################################################################
# Column / layout constants
###############################################################################
_TOTAL_WIDTH = 78

# Param-table column widths.  Sum + 6 single-space separators = 78.
_PARAM_WIDTH = 14
_EST_WIDTH = 10
_CI_WIDTH = 20
_SE_WIDTH = 10
_Z_WIDTH = 7
_P_WIDTH = 8
_SIG_WIDTH = 3

# Diagnostic-table column widths.  33 + 1 + 10 + 1 + 10 + 1 + 22 = 78.
_DIAG_LABEL_WIDTH = 33
_DIAG_STAT_WIDTH = 10
_DIAG_P_WIDTH = 10
_DIAG_DECISION_WIDTH = 22

_BLANK_NUMERIC = "--"


###############################################################################
# Public dataclasses
###############################################################################
@dataclass(frozen=True)
class ParamRow:
    r"""One row in a :class:`ParamSection`.

    Attributes:
        label: Row label rendered in the leftmost column (e.g. ``"phi[1]"``,
            ``"omega"``, ``"nu"``).  Bare — the section label already
            conveys ARMA / GARCH / residual-distribution context, so no
            ``"residual."`` prefix is added.
        estimate: Point estimate for the parameter.
        std_err: Asymptotic standard error.  Pass ``None`` (or a
            non-finite / zero value) to render ``--`` in the SE / z /
            p-value / sig cells; the CI cell will read ``[--, --]``.
    """

    label: str
    estimate: float
    std_err: Optional[float]


@dataclass(frozen=True)
class ParamSection:
    r"""A labelled group of :class:`ParamRow` entries.

    Sections are emitted in order separated by inline-labelled dashed
    lines.  Sections whose ``rows`` list is empty are silently skipped
    (so a standalone ``GARCH`` fit's mean-equation section, or an
    ``ARMA`` fit with ``normal`` residuals, simply doesn't appear).
    """

    label: str
    rows: list[ParamRow]


@dataclass(frozen=True)
class DiagnosticRow:
    r"""One row in the residual-diagnostics sub-table.

    Attributes:
        label: Test descriptor (e.g. ``"ljung_box(z, lags=10)"``).
        statistic: Test statistic.
        p_value: p-value under H0.
        h0_rejected: Decision at α=0.05 (``p_value < 0.05``).
        rejection_is_good: ``True`` for tests where rejecting H0 is the
            healthy outcome (only ADF among the standard diagnostics);
            ``False`` for the others (Ljung-Box, ARCH-LM, KPSS — where
            failing to reject H0 is what we want for a well-specified
            model).  Drives the ``✓`` / ``✗`` glyph appended to the
            decision text.
    """

    label: str
    statistic: float
    p_value: float
    h0_rejected: bool
    rejection_is_good: bool


###############################################################################
# Helpers
###############################################################################
def _is_finite(x: Optional[float]) -> bool:
    if x is None:
        return False
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(xf)


def _significance_code(p_value: float) -> str:
    r"""R ``summary.lm``-style significance code for a p-value.

    ===================  ====
    p-value range        code
    ===================  ====
    p < 0.001            ``***``
    0.001 ≤ p < 0.01     ``**``
    0.01  ≤ p < 0.05     ``*``
    0.05  ≤ p < 0.1      ``.``
    p ≥ 0.1 or non-finite blank
    ===================  ====
    """
    if not _is_finite(p_value):
        return ""
    p = float(p_value)
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def _section_separator(label: str) -> str:
    r"""``---- {label} ---...---`` of total length :data:`_TOTAL_WIDTH`."""
    prefix = f"---- {label} "
    pad = _TOTAL_WIDTH - len(prefix)
    if pad < 0:
        # Section label itself is wider than 78 chars — clip the dash
        # tail to a minimum of three.
        pad = 3
    return prefix + ("-" * pad)


def _plain_separator() -> str:
    return "-" * _TOTAL_WIDTH


def _double_separator() -> str:
    return "=" * _TOTAL_WIDTH


def _format_param_row(row: ParamRow, z_crit: float) -> str:
    r"""Render one :class:`ParamRow` as an aligned 78-char line.

    Rows whose ``std_err`` is ``None``, zero, or non-finite show
    ``[--, --]`` for the CI and ``--`` for the std-err / z / p-value
    cells; the sig column is left blank.
    """
    label = row.label
    est = float(row.estimate)
    se = row.std_err
    if _is_finite(se) and float(se) > 0.0:
        se_f = float(se)
        lo = est - z_crit * se_f
        hi = est + z_crit * se_f
        z_stat = est / se_f
        p_value = 2.0 * float(1.0 - _norm.cdf(abs(z_stat)))
        ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
        se_str = f"{se_f:.4f}"
        z_str = f"{z_stat:.2f}"
        p_str = f"{p_value:.4f}"
        sig = _significance_code(p_value)
    else:
        ci_str = "[--, --]"
        se_str = _BLANK_NUMERIC
        z_str = _BLANK_NUMERIC
        p_str = _BLANK_NUMERIC
        sig = ""
    line = (
        f"{label:<{_PARAM_WIDTH}} "
        f"{est:>{_EST_WIDTH}.4f} "
        f"{ci_str:^{_CI_WIDTH}} "
        f"{se_str:>{_SE_WIDTH}} "
        f"{z_str:>{_Z_WIDTH}} "
        f"{p_str:>{_P_WIDTH}} "
        f"{sig:<{_SIG_WIDTH}}"
    )
    return line.rstrip()


def _format_param_header() -> str:
    line = (
        f"{'param':<{_PARAM_WIDTH}} "
        f"{'estimate':>{_EST_WIDTH}} "
        f"{'CI':^{_CI_WIDTH}} "
        f"{'std err':>{_SE_WIDTH}} "
        f"{'z':>{_Z_WIDTH}} "
        f"{'P>|z|':>{_P_WIDTH}} "
        f"{'':<{_SIG_WIDTH}}"
    )
    return line.rstrip()


def _format_diagnostic_header() -> str:
    line = (
        f"{'test':<{_DIAG_LABEL_WIDTH}} "
        f"{'statistic':>{_DIAG_STAT_WIDTH}} "
        f"{'p-value':>{_DIAG_P_WIDTH}} "
        f"{'decision (α=0.05)':<{_DIAG_DECISION_WIDTH}}"
    )
    return line.rstrip()


def _format_diagnostic_row(row: DiagnosticRow) -> str:
    stat_str = (
        f"{float(row.statistic):.2f}"
        if _is_finite(row.statistic) else _BLANK_NUMERIC
    )
    p_str = (
        f"{float(row.p_value):.4f}"
        if _is_finite(row.p_value) else _BLANK_NUMERIC
    )
    if row.h0_rejected:
        decision_text = "reject H0"
    else:
        decision_text = "fail to reject H0"
    healthy = (row.h0_rejected == row.rejection_is_good)
    glyph = "✓" if healthy else "✗"
    decision = f"{decision_text} {glyph}"
    line = (
        f"{row.label:<{_DIAG_LABEL_WIDTH}} "
        f"{stat_str:>{_DIAG_STAT_WIDTH}} "
        f"{p_str:>{_DIAG_P_WIDTH}} "
        f"{decision:<{_DIAG_DECISION_WIDTH}}"
    )
    return line.rstrip()


###############################################################################
# Param-row builders shared across all model summaries
###############################################################################
def iter_param_rows(
    params_subset: dict,
    std_errs_subset: Optional[dict],
    *,
    vector_keys: tuple[str, ...] = (),
) -> list[ParamRow]:
    r"""Build :class:`ParamRow` instances from a params sub-dict.

    Iterates ``params_subset`` keys in insertion order.  For each key:

    * If the value is a 0-dim or length-1-or-less array AND the key
      is *not* listed in ``vector_keys``, render a single row with a
      bare ``"{key}"`` label.
    * Otherwise render one row per element with ``"{key}[{i}]"``
      labels (1-indexed).

    The ``vector_keys`` override is for parameters that are
    semantically vectors even when ``p`` or ``q`` happens to be 1
    (e.g. ``phi`` / ``theta`` / ``alpha`` / ``beta``) — keeps row
    labelling stable across model orders.

    Args:
        params_subset: Sub-dict of the model's ``params`` containing
            only the keys this section should render.
        std_errs_subset: Matching sub-dict of standard errors, or
            ``None`` if no SEs are available (every row will render
            blanks for SE / z / p / sig).
        vector_keys: Keys whose values should always render with
            ``[i]`` indexing even at length 1.

    Returns:
        List of :class:`ParamRow`.
    """
    rows: list[ParamRow] = []
    for key, value in params_subset.items():
        est_arr = _atleast_1d(value)
        if std_errs_subset is None:
            se_arr = None
        else:
            se_arr = _atleast_1d(std_errs_subset.get(key))
        treat_as_vector = (key in vector_keys) or len(est_arr) > 1
        for i, est in enumerate(est_arr):
            label = f"{key}[{i + 1}]" if treat_as_vector else key
            se = None if se_arr is None else se_arr[i]
            rows.append(
                ParamRow(label=label, estimate=float(est), std_err=se)
            )
    return rows


def residual_section(
    residual_params: dict,
    residual_std_errs: Optional[dict],
    *,
    dist_name: str,
) -> ParamSection:
    r"""Build the ``Residual distribution — <name>`` :class:`ParamSection`.

    Row labels are bare (``nu``, ``gamma``, ``alpha``, ``beta``, …) —
    the section label already says "Residual distribution", so the
    ``residual.`` prefix carried by the old ``ArmaGarch.summary`` is
    redundant and is dropped here.

    Returns an empty-rows :class:`ParamSection` (which the formatter
    suppresses) when ``residual_params`` is empty — covers the
    ``normal`` case where the standardised form has no free shape
    parameters.
    """
    rows: list[ParamRow] = []
    for key, value in residual_params.items():
        est_arr = _atleast_1d(value)
        if residual_std_errs is None:
            se_arr = None
        else:
            se_arr = _atleast_1d(residual_std_errs.get(key))
        for i, est in enumerate(est_arr):
            label = f"{key}[{i + 1}]" if len(est_arr) > 1 else key
            se = None if se_arr is None else se_arr[i]
            rows.append(
                ParamRow(label=label, estimate=float(est), std_err=se)
            )
    return ParamSection(
        label=f"Residual distribution — {dist_name}", rows=rows,
    )


def build_diagnostic_rows(residual_diagnostics: dict) -> list[DiagnosticRow]:
    r"""Convert the cached ``residual_diagnostics_`` dict into the
    list of :class:`DiagnosticRow` objects rendered in the
    diagnostics sub-table.

    Expects the five canonical keys ``"ljung_box"``, ``"ljung_box_sq"``,
    ``"arch_lm"``, ``"adf"``, ``"kpss"`` — each mapping to a result
    dict with at least ``"statistic"`` and ``"p_value"`` entries (see
    :mod:`copulax._src.timeseries._diagnostics` and
    :mod:`copulax._src.timeseries._unit_root` for the standardised
    schema).
    """
    spec = [
        ("ljung_box",    "ljung_box(z, lags=10)",   False),
        ("ljung_box_sq", "ljung_box(z², lags=10)", False),
        ("arch_lm",      "arch_lm(z, lags=5)",      False),
        ("adf",          'adf(z, regression="c")',  True),
        ("kpss",         'kpss(z, regression="c")', False),
    ]
    rows: list[DiagnosticRow] = []
    for key, label, rejection_is_good in spec:
        if key not in residual_diagnostics:
            continue
        d = residual_diagnostics[key]
        stat = float(d["statistic"])
        p_val = float(d["p_value"])
        h0_rejected = _is_finite(p_val) and p_val < 0.05
        rows.append(
            DiagnosticRow(
                label=label,
                statistic=stat,
                p_value=p_val,
                h0_rejected=h0_rejected,
                rejection_is_good=rejection_is_good,
            )
        )
    return rows


def _atleast_1d(value) -> list[float]:
    r"""Flatten ``value`` (scalar, 0-d / 1-d JAX or numpy array, list)
    to a Python list of floats.  Returns ``[]`` for ``None``."""
    if value is None:
        return []
    try:
        # JAX / numpy arrays expose ``.tolist()`` with sensible nested
        # behaviour for higher dims, but we only ever see 0-d or 1-d
        # leaves here so ``reshape(-1).tolist()`` is safe.
        flat = value.reshape(-1).tolist()
    except AttributeError:
        # Plain Python scalar or list.
        if isinstance(value, (list, tuple)):
            flat = list(value)
        else:
            flat = [value]
    return [float(x) for x in flat]


###############################################################################
# Main formatter entry point
###############################################################################
_SIG_LEGEND = (
    "Signif. codes:  ***  p<0.001    **  p<0.01    *  p<0.05    .  p<0.1"
)


def display_residual_name(name: str) -> str:
    r"""Strip the internal ``-stdresid`` suffix from a fitted
    residual-distribution name for display in summary headers /
    section labels.

    Fitted-time-series-model ``residual_dist`` instances carry the
    ``-stdresid`` suffix on their ``.name`` field to disambiguate
    them from the unfitted singleton in ``repr`` output (and to
    flag the standardised (mean=0, var=1) contract); the suffix is
    purely internal hygiene and should not bleed into the
    user-facing ``summary()`` table.
    """
    suffix = "-stdresid"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def format_summary(
    *,
    header: str,
    param_sections: list[ParamSection],
    diagnostic_rows: list[DiagnosticRow],
    loglikelihood: float,
    aic: float,
    bic: float,
    n_train: int,
    alpha: float = 0.05,
) -> str:
    r"""Render the full summary string.

    Layout:

    1. Header line + double-equals border.
    2. Param-table column header row.
    3. Each non-empty :class:`ParamSection`: inline-labelled dashed
       separator, then its rows.
    4. If ``diagnostic_rows`` is non-empty: a ``Residual diagnostics``
       inline-labelled separator, the diagnostic-table column header,
       its rows, and a plain dashed line.
    5. The R-style significance-code legend, sandwiched between two
       plain dashed lines (matches R's footer convention).
    6. Footer line — ``loglikelihood: …  AIC: …  BIC: …  n_train: …``
       — bracketed by double-equals borders.

    See module docstring for the visual contract.
    """
    z_crit = float(_norm.ppf(1.0 - alpha / 2.0))
    out: list[str] = [header, _double_separator(), _format_param_header()]

    for section in param_sections:
        if not section.rows:
            continue
        out.append(_section_separator(section.label))
        for row in section.rows:
            out.append(_format_param_row(row, z_crit=z_crit))

    if diagnostic_rows:
        out.append(_section_separator("Residual diagnostics"))
        out.append(_format_diagnostic_header())
        for diag in diagnostic_rows:
            out.append(_format_diagnostic_row(diag))

    out.append(_plain_separator())
    out.append(_SIG_LEGEND)
    out.append(_plain_separator())
    out.append(
        f"loglikelihood: {float(loglikelihood):.4f}  "
        f"AIC: {float(aic):.4f}  "
        f"BIC: {float(bic):.4f}  "
        f"n_train: {int(n_train)}"
    )
    out.append(_double_separator())
    return "\n".join(out)


__all__ = [
    "ParamRow",
    "ParamSection",
    "DiagnosticRow",
    "format_summary",
    "iter_param_rows",
    "residual_section",
    "build_diagnostic_rows",
]
