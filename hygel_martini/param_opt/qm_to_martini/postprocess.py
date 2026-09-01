"""Screening postprocess for Bartender ITP outputs (stage-03 final step).

This module owns the "screening" stage of the ``qm_to_martini`` workflow:
it re-parses every Bartender ``gmx_out.itp`` under the configured output
roots (including commented-out candidate lines), applies the configured
per-section screening rules (preferred function number, RMSD ceiling,
force-metric floor, bond-vs-constraint policy, overlap deduplication),
and writes both the complete candidate inventory and the screened result.

Callers: ``pipeline.run_pipeline`` / ``pipeline.run_postprocess_only``
invoke ``run_screening_postprocess`` when
``bartender_pipeline.postprocess.screening.enabled`` is true.

Inputs: the resolved config dict plus ``gmx_out.itp`` / ``case.json``
files under the postprocess roots. Outputs (per root, in the resolved
output directory): ``all_terms.json`` / ``all_terms.itp`` (inventory),
``screened_summary.json`` / ``screened_forcefield.itp`` (screened set),
per-section CSV tables and PDF diagnostic plots, and
``screening_report.json``.

Invariants:

- Parsing is purely textual: units are whatever GROMACS uses for each
  section (nm, kJ/mol/nm^2, deg, kJ/mol ...); no unit conversion happens.
- The "force metric" is a scalar screening proxy derived from the force
  constants of a line (see ``_force_values_and_metric``), not a physical
  observable.
- Screening never edits the source ITPs; all outputs are new files.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ITP section name -> number of atom-index columns before the funct column.
SECTION_INFO = {
    "bonds": 2,
    "constraints": 2,
    "angles": 3,
    "dihedrals": 4,
    "impropers": 4,
}
# Canonical output ordering for every per-section table/ITP block.
SECTION_ORDER = ("bonds", "constraints", "angles", "dihedrals", "impropers")
# Extracts the "rmsd: <float>" annotation Bartender writes into ITP comments.
RMSD_RE = re.compile(r"rmsd:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
# Maximum candidate points per PDF plot page (extra points paginate).
PLOT_MAX_POINTS = 10

# (section, funct) -> (human name, LaTeX equation, parameter note) used as
# plot annotations; unknown combinations fall back via _potential_title.
POTENTIAL_INFO = {
    ("bonds", 1): (
        "harmonic bond",
        r"$V(r)=\frac{1}{2}k_b(r-r_0)^2$",
        "params: r0, k_b",
    ),
    ("constraints", 1): (
        "distance constraint",
        r"$r=r_0;\ \mathrm{force\ metric}=k\ \mathrm{if\ available}$",
        "params: r0[, k]",
    ),
    ("angles", 1): (
        "harmonic angle",
        r"$V(\theta)=\frac{1}{2}k_{\theta}(\theta-\theta_0)^2$",
        "params: theta0, k_theta",
    ),
    ("angles", 2): (
        "G96 harmonic cosine angle",
        r"$V(\theta)=\frac{1}{2}k_{\theta}(\cos\theta-\cos\theta_0)^2$",
        "params: theta0, k",
    ),
    ("angles", 10): (
        "restricted bending angle (ReB)",
        r"$V(\theta)=\frac{1}{2}k_{\theta}\frac{(\cos\theta-\cos\theta_0)^2}{\sin^2\theta}$",
        "params: theta0, k",
    ),
    ("dihedrals", 1): (
        "proper periodic dihedral",
        r"$V(\phi)=k_{\phi}\left[1+\cos(n\phi-\phi_0)\right]$",
        "params: phi0, k_phi, n",
    ),
    ("dihedrals", 2): (
        "harmonic improper-style dihedral",
        r"$V(\phi)=\frac{1}{2}k_{\phi}(\phi-\phi_0)^2$",
        "params: phi0, k_phi",
    ),
    ("dihedrals", 3): (
        "Ryckaert-Bellemans dihedral",
        r"$V(\phi)=\sum_{i=0}^{5}C_i\cos^i(\phi)$",
        "params: C0..C5",
    ),
    ("dihedrals", 11): (
        "combined bending-torsion",
        r"$V=\sum_i C_i f_i(\theta_1,\theta_2,\phi)$",
        "params: C0..C5",
    ),
    ("impropers", 1): (
        "harmonic improper",
        r"$V(\xi)=\frac{1}{2}k_{\xi}(\xi-\xi_0)^2$",
        "params: xi0, k_xi",
    ),
    ("impropers", 2): (
        "periodic improper",
        r"$V(\xi)=k_{\xi}\left[1+\cos(n\xi-\xi_0)\right]$",
        "params: xi0, k_xi, n",
    ),
}


def _as_list(value: Any) -> List[Any]:
    """Coerce a scalar / list / tuple / None config value into a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _parse_float(value: str) -> Optional[float]:
    """Parse a float token, returning None for non-numeric input."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _canon_indices(indices: Sequence[int]) -> Tuple[int, ...]:
    """Return the order-independent canonical form of a bead-index tuple."""
    return tuple(sorted(int(value) for value in indices))


def _find_case_json(start: Path) -> Optional[Path]:
    """Locate the nearest ``case.json`` at or above ``start`` (max 8 levels).

    Args:
        start: directory to start from (usually an ITP's parent).

    Returns:
        Path of the first ``case.json`` found, or None.
    """
    current = start.resolve()
    for _ in range(8):
        candidate = current / "case.json"
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def _relative_to_or_name(path: Path, base: Optional[Path]) -> Path:
    """Return ``path`` relative to ``base``, or just its basename.

    The basename fallback is used when ``base`` is None or ``path`` lies
    outside it, so callers always get a usable (possibly flat) label.
    """
    if base is None:
        return Path(path.name)
    try:
        return path.resolve().relative_to(base.resolve())
    except ValueError:
        return Path(path.name)


def _format_number(value: Optional[float]) -> str:
    """Format a number compactly for plot labels ("NA" for None/non-finite).

    Uses scientific notation outside [0.01, 10000), one decimal above 100,
    and 3 significant digits otherwise.
    """
    if value is None or not math.isfinite(float(value)):
        return "NA"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) >= 10000 or abs(value) < 0.01:
        return f"{value:.2e}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3g}"


def _chunk_sizes(total: int, limit: int = PLOT_MAX_POINTS) -> List[int]:
    """Split ``total`` items into near-equal chunk sizes of at most ``limit``.

    The smaller chunks come first so the sizes differ by at most one.

    Args:
        total: number of items to split (<=0 yields an empty list).
        limit: maximum items per chunk.

    Returns:
        Chunk sizes summing to ``total``.
    """
    if total <= 0:
        return []
    chunk_count = math.ceil(total / limit)
    base = total // chunk_count
    remainder = total % chunk_count
    return [base] * (chunk_count - remainder) + [base + 1] * remainder


def _chunk_rows(rows: Sequence[Dict[str, Any]], limit: int = PLOT_MAX_POINTS) -> List[List[Dict[str, Any]]]:
    """Partition term rows into plot pages of at most ``limit`` rows each."""
    chunks: List[List[Dict[str, Any]]] = []
    start = 0
    for size in _chunk_sizes(len(rows), limit):
        chunks.append(list(rows[start : start + size]))
        start += size
    return chunks


def _potential_title(section: str, funct: int) -> Tuple[str, str, str]:
    """Look up plot annotations (name, equation, params) for a potential.

    Falls back to a generic placeholder for unannotated (section, funct)
    combinations instead of raising.
    """
    return POTENTIAL_INFO.get(
        (section, funct),
        (
            f"{section} potential",
            r"$\mathrm{equation\ not\ annotated}$",
            "params: see ITP line",
        ),
    )


def _axis_bounds(values: Sequence[float], threshold: Optional[float], *, zero_floor: bool = True) -> Tuple[float, float]:
    """Compute padded y-axis limits that include the data and the cutoff.

    Args:
        values: finite data values to cover.
        threshold: optional cutoff line to keep visible (ignored when
            non-finite or None).
        zero_floor: clamp the lower bound to 0 when all data is >= 0.

    Returns:
        (lo, hi) axis limits; (0, 1) when there is nothing to show.
    """
    finite = list(values)
    if threshold is not None and math.isfinite(float(threshold)):
        finite.append(float(threshold))
    if not finite:
        return (0.0, 1.0)
    lo = min(finite)
    hi = max(finite)
    if zero_floor and lo >= 0:
        lo = 0.0
    if math.isclose(lo, hi):
        pad = max(abs(hi) * 0.1, 1.0)
        lo -= pad
        hi += pad
        if zero_floor and lo < 0:
            lo = 0.0
    else:
        pad = (hi - lo) * 0.12
        lo -= pad
        hi += pad
        if zero_floor and lo < 0:
            lo = 0.0
    return lo, hi


def _selected_key(row: Dict[str, Any]) -> Tuple[str, str, Tuple[int, ...], int]:
    """Build the identity key used to match a plotted row to a screened term.

    Keyed on (source file, section, index tuple in original order, funct)
    so the same bonded term from different ITPs stays distinguishable.
    """
    return (
        str(row.get("source", "")),
        str(row.get("section", "")),
        tuple(row.get("indices", ())),
        int(row.get("funct", 0)),
    )


def _write_pdf_plot(
    path: Path,
    title: str,
    rows: Sequence[Dict[str, Any]],
    *,
    selected_keys: set[Tuple[str, str, Tuple[int, ...], int]],
    section: str,
    funct: int,
    force_threshold: Optional[float],
    rmsd_threshold: Optional[float],
    page_index: int = 1,
    page_count: int = 1,
    global_start_index: int = 0,
    total_count: Optional[int] = None,
) -> None:
    """Render one screening-diagnostic PDF page for a (section, funct) group.

    The page shows two aligned panels (force_metric on top, RMSD below)
    with candidate points ordered by plot index, selected/screened points
    highlighted, and the cutoff line plus shaded reject region drawn only
    when the page mixes passing and failing points.

    Args:
        path: output PDF path (parent directories are created).
        title: page headline, e.g. "bonds funct 1".
        rows: parsed term rows for this page (already chunked).
        selected_keys: identity keys (see ``_selected_key``) of terms that
            survived screening.
        section: ITP section name, used for potential annotation lookup.
        funct: GROMACS function number of this group.
        force_threshold: force-metric floor (pass when metric >= value).
        rmsd_threshold: RMSD ceiling (pass when RMSD <= value), or None.
        page_index: 1-based page number within the group.
        page_count: total pages for the group.
        global_start_index: 0-based offset of this page's first row in the
            full group ordering (for point numbering).
        total_count: total rows in the group (defaults to offset+len(rows)).

    Raises:
        RuntimeError: matplotlib is not installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.ticker import FuncFormatter, MaxNLocator
    except Exception as exc:  # pragma: no cover - depends on optional plotting dependency
        raise RuntimeError("PDF plot output requires matplotlib to be installed") from exc

    selected_color = "#006d77"
    parsed_color = "#8f9aa3"
    grid_color = "#e7edf1"
    reject_color = "#e7eaed"
    text_color = "#17212b"
    potential_name, equation, param_note = _potential_title(section, funct)
    total_label = total_count if total_count is not None else global_start_index + len(rows)
    page_label = f"part {page_index}/{page_count}" if page_count > 1 else ""
    selected_count = sum(1 for row in rows if _selected_key(row) in selected_keys)

    fig = plt.figure(figsize=(11.8, 8.6))
    gs = fig.add_gridspec(
        2,
        1,
        left=0.09,
        right=0.97,
        top=0.62,
        bottom=0.17,
        hspace=0.56,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]

    fig.text(0.09, 0.94, title, fontsize=20, fontweight="bold", color=text_color, ha="left", va="top")
    if page_label:
        fig.text(0.97, 0.94, page_label, fontsize=10, color="#66737c", ha="right", va="top")
    fig.text(0.09, 0.89, f"funct {funct}: {potential_name}", fontsize=12, color="#34434d", ha="left")
    fig.text(0.09, 0.82, equation, fontsize=15, color="#34434d", ha="left")
    fig.text(0.09, 0.765, param_note, fontsize=10, color="#66737c", ha="left")
    fig.text(
        0.97,
        0.825,
        f"points {global_start_index + 1}-{global_start_index + len(rows)} of {total_label} | "
        f"selected in this panel: {selected_count}",
        fontsize=10,
        color="#66737c",
        ha="right",
    )

    def finite_values(key: str) -> List[float]:
        """Collect the finite numeric values of ``key`` across the rows."""
        values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        return values

    def cutoff_state(values: List[float], threshold: Optional[float], mode: str) -> Tuple[str, bool, bool]:
        """Classify how the cutoff relates to the panel's values.

        Args:
            values: finite panel values.
            threshold: cutoff value, or None.
            mode: "min" (pass when value >= cutoff) or "max" (<= cutoff).

        Returns:
            (status label, mixed pass/fail, all rejected) — the cutoff
            line/shading is drawn only in the mixed case.
        """
        if threshold is None or not math.isfinite(float(threshold)) or not values:
            return "no finite cutoff", False, False
        if mode == "min":
            passes = [value >= float(threshold) for value in values]
        else:
            passes = [value <= float(threshold) for value in values]
        if all(passes):
            return "all pass cutoff", False, False
        if not any(passes):
            return "all reject by cutoff", False, True
        return f"cutoff = {_format_number(float(threshold))}", True, False

    def format_axis(value: float, _pos: int) -> str:
        """Matplotlib tick formatter delegating to ``_format_number``."""
        return _format_number(value)

    def draw_panel(ax: Any, *, key: str, label: str, threshold: Optional[float], pass_mode: str) -> None:
        """Draw one metric panel: grid, cutoff region, line, and points."""
        values = finite_values(key)
        status, mixed, all_reject = cutoff_state(values, threshold, pass_mode)
        threshold_for_axis = threshold if mixed else None
        lo, hi = _axis_bounds(values, threshold_for_axis)
        ax.set_ylim(lo, hi)
        ax.set_xlim(-0.45, max(len(rows) - 1, 0) + 0.45)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=grid_color, linewidth=0.9)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(FuncFormatter(format_axis))
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label} | {status}", loc="left", fontsize=11, fontweight="bold", color=text_color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if all_reject:
            ax.set_facecolor("#eef1f3")
        elif mixed and threshold is not None:
            threshold_value = float(threshold)
            if pass_mode == "min":
                ax.axhspan(lo, threshold_value, facecolor=reject_color, alpha=0.9, zorder=0)
            else:
                ax.axhspan(threshold_value, hi, facecolor=reject_color, alpha=0.9, zorder=0)
            ax.axhline(threshold_value, color="#7a8790", linewidth=1.2, linestyle=(0, (6, 4)))
            ax.text(
                0.995,
                threshold_value,
                f" {label} cutoff {_format_number(threshold_value)}",
                transform=ax.get_yaxis_transform(),
                fontsize=8.5,
                color="#5f6b73",
                va="bottom",
                ha="right",
            )

        xs: List[int] = []
        ys: List[float] = []
        for idx, row in enumerate(rows):
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                xs.append(idx)
                ys.append(float(value))
        if xs:
            ax.plot(xs, ys, color="#c7d0d6", linewidth=1.0, zorder=1)

        for selected in (False, True):
            point_x = []
            point_y = []
            for idx, row in enumerate(rows):
                value = row.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    continue
                if (_selected_key(row) in selected_keys) != selected:
                    continue
                point_x.append(idx)
                point_y.append(float(value))
            if point_x:
                ax.scatter(
                    point_x,
                    point_y,
                    s=54 if selected else 34,
                    color=selected_color if selected else parsed_color,
                    edgecolors="#073b43" if selected else "white",
                    linewidths=0.8,
                    zorder=3 if selected else 2,
                )

    draw_panel(axes[0], key="force_metric", label="force_metric", threshold=force_threshold, pass_mode="min")
    draw_panel(axes[1], key="rmsd", label="RMSD", threshold=rmsd_threshold, pass_mode="max")

    labels = []
    for idx, row in enumerate(rows):
        ordinal = global_start_index + idx + 1
        indices = "-".join(str(value) for value in row.get("indices", ()))
        labels.append(f"#{ordinal}\n{indices}")
    tick_positions = list(range(len(rows)))
    for ax in axes:
        ax.set_xticks(tick_positions)
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_xlabel("plot index / atom indices", fontsize=10)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=selected_color, markeredgecolor="#073b43", markersize=7, label="selected/screened"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=parsed_color, markeredgecolor="white", markersize=6, label="parsed candidate"),
        Patch(facecolor=reject_color, edgecolor="none", alpha=0.9, label="cutoff reject region"),
    ]
    fig.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.09, 0.035), ncol=3, frameon=False, fontsize=9)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


class ScreeningProcessor:
    """Parse Bartender ITP outputs and write full plus screened postprocess data.

    Configured once from ``bartender_pipeline.postprocess.screening`` and
    then applied to one or more output roots via ``process``. Screening
    keeps, per section: terms of the preferred function number (or any,
    for "bartender"), with RMSD <= ``rmsd_max``, force metric >= the
    per-section floor, respecting the bond/constraint policy, and at most
    one term per canonical index tuple (best RMSD wins, force metric
    breaks ties).
    """

    def __init__(self, cfg: Dict[str, Any]):
        """Resolve and normalize all screening settings from the config.

        Args:
            cfg: full resolved pipeline config; only ``paths`` and
                ``bartender_pipeline.postprocess.screening`` are used.
                A scalar ``thresholds.force_metric_min`` is broadcast to
                every section.

        Raises:
            ValueError: unsupported ``bond_constraint_mode`` or
                ``candidate_source`` value.
        """
        self.cfg = cfg
        self.paths_cfg = cfg.get("paths", {})
        self.post_cfg = cfg.get("bartender_pipeline", {}).get("postprocess", {})
        self.screen_cfg = self.post_cfg.get("screening", {})

        self.pref_potentials = self.screen_cfg.get("potentials", {})

        self.fc_min_cfg = self.screen_cfg.get("thresholds", {}).get("force_metric_min", 0.0)
        if isinstance(self.fc_min_cfg, (int, float)):
            val = float(self.fc_min_cfg)
            self.fc_min_cfg = {section: val for section in SECTION_ORDER}
        self.threshold_mode = str(
            self.screen_cfg.get("thresholds", {}).get("force_metric_min_mode", "absolute")
        ).strip().lower()
        self.rmsd_max = float(self.screen_cfg.get("thresholds", {}).get("rmsd_max", math.inf))

        self.multi_constant_metric = str(self.screen_cfg.get("multi_constant_metric", "max_abs")).strip().lower()
        self.bond_constraint_mode = self._normalize_bond_constraint_mode(
            self.screen_cfg.get("bond_constraint_mode", "bartender")
        )
        self.candidate_source = self._normalize_candidate_source(
            self.screen_cfg.get("candidate_source", "active")
        )
        self.show_all_info = bool(self.screen_cfg.get("show_all_info", True))
        self.write_plots = bool(self.screen_cfg.get("write_plots", True))

    @staticmethod
    def _normalize_bond_constraint_mode(raw: Any) -> str:
        """Map user aliases onto a canonical bond/constraint policy.

        Args:
            raw: user-supplied ``screening.bond_constraint_mode`` value.

        Returns:
            One of "bartender" (keep whichever section Bartender chose),
            "ignore_constraints" (drop [constraints] terms), or
            "ignore_bonds" (drop [bonds] terms).

        Raises:
            ValueError: value does not map to a supported mode.
        """
        mode = str(raw or "bartender").strip().lower()
        aliases = {
            "both": "bartender",
            "screened": "bartender",
            "bartender_selected": "bartender",
            "keep_bartender": "bartender",
            "ignore_constraint": "ignore_constraints",
            "bonds_only": "ignore_constraints",
            "bond_only": "ignore_constraints",
            "ignore_bond": "ignore_bonds",
            "constraints_only": "ignore_bonds",
            "constraint_only": "ignore_bonds",
        }
        mode = aliases.get(mode, mode)
        supported = {"ignore_constraints", "bartender", "ignore_bonds"}
        if mode not in supported:
            raise ValueError(f"Unsupported screening.bond_constraint_mode={raw!r}. Use one of {sorted(supported)}.")
        return mode

    @staticmethod
    def _normalize_candidate_source(raw: Any) -> str:
        """Map user aliases onto a canonical candidate-source policy.

        Args:
            raw: user-supplied ``screening.candidate_source`` value.

        Returns:
            "active" (only uncommented, Bartender-selected lines) or
            "all" (commented-out candidate lines included).

        Raises:
            ValueError: value does not map to a supported mode.
        """
        mode = str(raw or "active").strip().lower()
        aliases = {
            "bartender": "active",
            "bartender_active": "active",
            "active_only": "active",
            "selected": "active",
            "all_candidates": "all",
            "all_terms": "all",
            "include_commented": "all",
            "commented": "all",
        }
        mode = aliases.get(mode, mode)
        supported = {"active", "all"}
        if mode not in supported:
            raise ValueError(f"Unsupported screening.candidate_source={raw!r}. Use one of {sorted(supported)}.")
        return mode

    def _term_is_allowed_by_candidate_source(self, term: Dict[str, Any]) -> bool:
        """Check the term against the candidate-source policy ("all"/"active")."""
        if self.candidate_source == "all":
            return True
        return self._term_is_bartender_active(term)

    def _force_values_and_metric(
        self,
        section: str,
        funct: int,
        numeric_params: Sequence[float],
    ) -> tuple[List[float], Optional[float], str]:
        """Extract force-constant values and reduce them to a scalar metric.

        For bonds/constraints/angles and funct-1/2 dihedrals/impropers the
        force constant is the second numeric parameter (fallback: last),
        so the metric is a single value. For every other potential all
        numeric parameters count and ``multi_constant_metric`` decides the
        reduction: "l2", "mean_abs", "first", "none"/"disabled" (metric
        None), or the default "max_abs".

        Args:
            section: ITP section name.
            funct: GROMACS function number.
            numeric_params: numeric parameters after the funct column, in
                GROMACS units for that potential.

        Returns:
            (force values used, scalar metric or None, method label).
        """
        values: List[float] = []
        method = "single"
        if section in {"bonds", "constraints", "angles"}:
            if len(numeric_params) >= 2:
                values = [float(numeric_params[1])]
            elif numeric_params:
                values = [float(numeric_params[-1])]
        elif section in {"dihedrals", "impropers"} and funct in {1, 2}:
            if len(numeric_params) >= 2:
                values = [float(numeric_params[1])]
            elif numeric_params:
                values = [float(numeric_params[-1])]
        else:
            values = [float(value) for value in numeric_params]
            method = self.multi_constant_metric

        if not values:
            return [], None, method

        abs_values = [abs(value) for value in values]
        if method == "l2":
            metric = math.sqrt(sum(value * value for value in values))
        elif method == "mean_abs":
            metric = sum(abs_values) / len(abs_values)
        elif method == "first":
            metric = abs_values[0]
        elif method in {"none", "disabled"}:
            metric = None
        else:
            metric = max(abs_values)
            method = "max_abs" if len(values) > 1 else "single"
        return values, metric, method

    def _parse_itp_line(self, line: str, section: str, n_idx: int) -> Optional[Dict[str, Any]]:
        """Parse one ITP data line (commented or active) into a term record.

        Leading ``;`` markers are stripped but remembered as ``commented``
        so Bartender's rejected candidates remain analyzable. Lines whose
        content does not start with a digit (headers, prose comments) are
        skipped. The Bartender ``rmsd:`` annotation is pulled from the
        inline comment (fallback: anywhere in the raw line).

        Args:
            line: raw ITP line.
            section: current section name.
            n_idx: number of atom-index columns for this section.

        Returns:
            Term dict (indices, funct, params, force metric, rmsd,
            commented flag, raw line, section) or None for non-term lines.
        """
        raw = line.rstrip("\n")
        stripped = raw.strip()
        if not stripped:
            return None
        commented = stripped.startswith(";")
        content = stripped
        while content.startswith(";"):
            content = content[1:].strip()
        if not content or not content[0].isdigit():
            return None

        if ";" in content:
            main_part, inline_comment = content.split(";", 1)
        else:
            main_part, inline_comment = content, ""

        parts = main_part.split()
        if len(parts) < n_idx + 1:
            return None

        try:
            indices = tuple(int(p) for p in parts[:n_idx])
            funct = int(parts[n_idx])
        except (ValueError, IndexError):
            return None

        params = parts[n_idx + 1 :]
        numeric_params = [value for value in (_parse_float(param) for param in params) if value is not None]
        force_values, force_metric, force_metric_method = self._force_values_and_metric(section, funct, numeric_params)

        rmsd = None
        match = RMSD_RE.search(inline_comment) or RMSD_RE.search(raw)
        if match:
            rmsd = float(match.group(1))

        return {
            "indices": indices,
            "funct": funct,
            "params": params,
            "numeric_params": numeric_params,
            "force_values": force_values,
            "force_metric": force_metric,
            "force_metric_method": force_metric_method,
            "rmsd": rmsd,
            "commented": commented,
            "raw": raw,
            "section": section,
        }

    def _parse_itp(self, itp_path: Path, out_root: Path) -> Dict[str, List[Dict[str, Any]]]:
        """Parse a whole ``gmx_out.itp`` into per-section term lists.

        Each term is tagged with provenance: absolute source path, path
        relative to ``out_root``, a ``<sequence_stem>:<job_dir>`` tag, and
        the owning ``case.json`` (best effort; parse failures leave the
        case data empty rather than aborting).

        Args:
            itp_path: Bartender output ITP to parse.
            out_root: postprocess root used for relative source labels.

        Returns:
            Section name -> list of term dicts (only known sections).
        """
        parsed: Dict[str, List[Dict[str, Any]]] = {section: [] for section in SECTION_ORDER}
        current_section = None
        case_json = _find_case_json(itp_path.parent)
        case_data: Dict[str, Any] = {}
        if case_json is not None:
            try:
                case_data = json.loads(case_json.read_text(encoding="utf-8"))
            except Exception:
                case_data = {}
        sequence_stem = str(case_data.get("sequence_stem") or itp_path.parent.parent.name)
        relative_source = str(_relative_to_or_name(itp_path, out_root))
        source_tag = f"{sequence_stem}:{itp_path.parent.name}"

        for line in itp_path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                sec = stripped.strip("[]").strip().lower()
                current_section = sec if sec in SECTION_INFO else None
                continue
            if current_section is None:
                continue
            term = self._parse_itp_line(line, current_section, SECTION_INFO[current_section])
            if term is None:
                continue
            term["source"] = str(itp_path)
            term["relative_source"] = relative_source
            term["source_tag"] = source_tag
            term["case_json"] = str(case_json) if case_json else None
            parsed[current_section].append(term)
        return parsed

    def _get_overlap_key(self, term: Dict[str, Any]) -> Tuple[Any, ...]:
        """Return the (section, sorted indices) key used for overlap dedup."""
        return (term["section"], _canon_indices(tuple(term["indices"])))

    @staticmethod
    def _term_is_bartender_active(term: Dict[str, Any]) -> bool:
        """True when Bartender kept the line active (not commented out)."""
        return not bool(term["commented"])

    def _term_is_allowed_by_bond_constraint_mode(self, term: Dict[str, Any]) -> bool:
        """Apply the bond/constraint policy to one term's section."""
        section = term["section"]
        if section == "constraints" and self.bond_constraint_mode == "ignore_constraints":
            return False
        if section == "bonds" and self.bond_constraint_mode == "ignore_bonds":
            return False
        return True

    @staticmethod
    def _term_matches_preferred_potential(term: Dict[str, Any], preferred: Any) -> bool:
        """Return true when a term matches the configured funct preference.

        Integer values select that funct. The string "bartender" means do not
        filter by funct; the candidate line itself decides the funct. Commented
        line handling is controlled separately by screening.candidate_source.
        """
        if preferred is None:
            return True
        if isinstance(preferred, str):
            normalized = preferred.strip().lower()
            if normalized in {"", "bartender"}:
                return True
            try:
                preferred_funct = int(normalized)
            except ValueError as exc:
                raise ValueError(
                    "screening.potentials values must be function numbers or "
                    "'bartender' to use Bartender-active function numbers"
                ) from exc
        else:
            preferred_funct = int(preferred)
        return int(term["funct"]) == preferred_funct

    def _threshold_for(self, section: str, funct: int, terms: Sequence[Dict[str, Any]]) -> float:
        """Resolve the effective force-metric floor for a (section, funct).

        In "absolute" mode the configured per-section value is returned as
        is. In "relative(_to_section_max)" mode the configured value is a
        fraction of the largest finite metric among the given terms with
        the same section/funct; with no such metrics the floor is +inf
        (everything rejected).

        Args:
            section: ITP section name (fallback config key: "bonds").
            funct: GROMACS function number (relative mode only).
            terms: candidate pool the relative maximum is taken from.

        Returns:
            Minimum acceptable force metric.

        Raises:
            ValueError: unknown ``force_metric_min_mode``.
        """
        raw = self.fc_min_cfg.get(section, self.fc_min_cfg.get("bonds", 0.0))
        raw_value = float(raw)
        if self.threshold_mode in {"relative", "relative_to_max", "relative_to_section_max"}:
            metrics = [
                float(term["force_metric"])
                for term in terms
                if term["section"] == section
                and int(term["funct"]) == int(funct)
                and isinstance(term.get("force_metric"), (int, float))
            ]
            if not metrics:
                return math.inf
            return raw_value * max(metrics)
        if self.threshold_mode not in {"absolute", "abs"}:
            raise ValueError("screening.thresholds.force_metric_min_mode must be 'absolute' or 'relative_to_section_max'")
        return raw_value

    def _screen_terms(self, all_terms: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Apply the full screening pipeline per section.

        Filter order: bond/constraint policy -> candidate source ->
        preferred funct -> RMSD ceiling -> force-metric floor. Survivors
        are ranked by (RMSD asc, force metric desc) and the first term per
        canonical index tuple wins; the accepted set is finally re-sorted
        by (section, indices, funct) for stable output.

        Args:
            all_terms: parsed terms per section (all sources combined).

        Returns:
            Section name -> accepted term list.
        """
        screened_results: Dict[str, List[Dict[str, Any]]] = {section: [] for section in SECTION_ORDER}

        for section in SECTION_ORDER:
            terms = list(all_terms.get(section, []))
            if not terms:
                continue
            pref_funct = self.pref_potentials.get(section)
            candidate_terms = []
            for term in terms:
                if not self._term_is_allowed_by_bond_constraint_mode(term):
                    continue
                if not self._term_is_allowed_by_candidate_source(term):
                    continue
                if not self._term_matches_preferred_potential(term, pref_funct):
                    continue
                if term["rmsd"] is None or float(term["rmsd"]) > self.rmsd_max:
                    continue
                if term["force_metric"] is None:
                    continue
                candidate_terms.append(term)

            valid_terms = []
            for term in candidate_terms:
                threshold = self._threshold_for(section, int(term["funct"]), candidate_terms)
                if float(term["force_metric"]) < threshold:
                    continue
                valid_terms.append(term)

            valid_terms.sort(
                key=lambda item: (
                    float(item["rmsd"]) if item["rmsd"] is not None else math.inf,
                    -float(item["force_metric"]) if item["force_metric"] is not None else 0.0,
                )
            )
            accepted = []
            occupied = set()
            for term in valid_terms:
                okey = self._get_overlap_key(term)
                if okey in occupied:
                    continue
                accepted.append(term)
                occupied.add(okey)
            accepted.sort(key=lambda item: (item["section"], item["indices"], item["funct"]))
            screened_results[section] = accepted

        return screened_results

    def _info_terms(self, all_terms: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Select which terms go into the inspection outputs (json/itp/plots).

        With ``show_all_info`` every parsed term is included; otherwise
        only terms passing the policy/funct filters (but not the RMSD or
        force-metric thresholds) are kept.
        """
        if self.show_all_info:
            return all_terms
        info: Dict[str, List[Dict[str, Any]]] = {section: [] for section in SECTION_ORDER}
        for section in SECTION_ORDER:
            pref_funct = self.pref_potentials.get(section)
            for term in all_terms.get(section, []):
                if not self._term_is_allowed_by_bond_constraint_mode(term):
                    continue
                if not self._term_is_allowed_by_candidate_source(term):
                    continue
                if not self._term_matches_preferred_potential(term, pref_funct):
                    continue
                info[section].append(term)
        return info

    def _output_dir_for_root(self, out_root: Path) -> Path:
        """Resolve where the postprocess outputs for one root are written.

        When ``postprocess_output_root`` is configured, the input root's
        path relative to ``postprocess_mirror_root`` is mirrored under it
        (keeping multi-root runs separated). Otherwise ``output_dir``
        (default ``postprocessing_result``) is used, relative to the input
        root unless absolute.
        """
        output_root_raw = self.paths_cfg.get("postprocess_output_root") or self.screen_cfg.get("output_root")
        mirror_root_raw = self.paths_cfg.get("postprocess_mirror_root") or self.screen_cfg.get("mirror_root")
        if output_root_raw:
            output_root = Path(str(output_root_raw)).resolve()
            mirror_root = Path(str(mirror_root_raw)).resolve() if mirror_root_raw else None
            return output_root / _relative_to_or_name(out_root, mirror_root)

        output_dir = Path(str(self.screen_cfg.get("output_dir", "postprocessing_result")))
        if output_dir.is_absolute():
            return output_dir
        return out_root / output_dir

    @staticmethod
    def _json_terms(terms: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strip the raw ITP line from term dicts for JSON serialization."""
        rows = []
        for term in terms:
            rows.append({key: value for key, value in term.items() if key != "raw"})
        return rows

    def _write_all_terms_itp(self, path: Path, all_terms: Dict[str, List[Dict[str, Any]]]) -> None:
        """Write the inspection ITP: every kept term with provenance comments.

        Each term is echoed verbatim (original commented state preserved)
        preceded by a comment line recording source, metric, and RMSD; the
        file is documentation, not a runnable topology.
        """
        lines = [
            "; Bartender terms kept for postprocess inspection.",
            "; Original comment state is preserved in the line body.",
            "",
        ]
        for section in SECTION_ORDER:
            terms = all_terms.get(section, [])
            lines.append(f"[{section}]")
            for term in terms:
                metric = term.get("force_metric")
                metric_text = "NA" if metric is None else f"{float(metric):.6g}"
                rmsd = term.get("rmsd")
                rmsd_text = "NA" if rmsd is None else f"{float(rmsd):.6g}"
                lines.append(
                    f"; source={term.get('relative_source')} commented={term.get('commented')} "
                    f"force_metric={metric_text} metric_method={term.get('force_metric_method')} rmsd={rmsd_text}"
                )
                lines.append(str(term.get("raw", "")).rstrip())
            lines.append("")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _write_itp(self, path: Path, results: Dict[str, List[Dict[str, Any]]]) -> None:
        """Write the screened force field as an ITP with header settings.

        Active (uncommented) term lines are regenerated from the parsed
        fields, each annotated with its RMSD, force metric, and source tag.
        """
        lines = [
            "; Screened forcefield using Hygel Martini post-processor",
            f"; bond_constraint_mode = {self.bond_constraint_mode}",
            f"; force_metric_min_mode = {self.threshold_mode}",
            f"; multi_constant_metric = {self.multi_constant_metric}",
            "",
        ]
        for section in SECTION_ORDER:
            terms = results.get(section, [])
            lines.append(f"[{section}]")
            for term in terms:
                idx_str = " ".join(f"{i:>4}" for i in term["indices"])
                params_str = " ".join(f"{p:>10}" for p in [str(term["funct"])] + term["params"])
                rmsd_val = f"{term['rmsd']:.3f}" if term["rmsd"] is not None else "N/A"
                metric = term.get("force_metric")
                metric_val = f"{metric:.3g}" if isinstance(metric, (int, float)) else "N/A"
                lines.append(
                    f"{idx_str} {params_str} ; rmsd: {rmsd_val} | "
                    f"force_metric: {metric_val} | from {term.get('source_tag', '')}"
                )
            lines.append("")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _write_plots(
        self,
        out_dir: Path,
        all_terms: Dict[str, List[Dict[str, Any]]],
        screened: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Write per-(section, funct) CSV tables and paginated PDF plots.

        Rows are grouped by function number and ordered by (source tag,
        indices). The plotted cutoff uses only rows that would reach the
        threshold stage of screening (falling back to all rows when none
        qualify). Stale plot files for the same group are removed before
        new pages are written. No-op when ``write_plots`` is false.

        Args:
            out_dir: postprocess output directory (plots go to ``plots/``).
            all_terms: inspection term set to tabulate/plot.
            screened: screened result used to highlight selected points.
        """
        if not self.write_plots:
            return
        plot_dir = out_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        selected_keys = {_selected_key(term) for terms in screened.values() for term in terms}
        for section in SECTION_ORDER:
            grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for term in all_terms.get(section, []):
                grouped[int(term["funct"])].append(term)
            for funct, rows in sorted(grouped.items()):
                rows = sorted(rows, key=lambda item: (item.get("source_tag", ""), item.get("indices", ())))
                csv_path = plot_dir / f"{section}_funct_{funct}.csv"
                with csv_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            "plot_index",
                            "source_tag",
                            "section",
                            "indices",
                            "funct",
                            "commented",
                            "selected",
                            "force_metric",
                            "force_metric_method",
                            "rmsd",
                            "force_values",
                            "params",
                            "source",
                        ]
                    )
                    for row_index, row in enumerate(rows, start=1):
                        term_key = _selected_key(row)
                        writer.writerow(
                            [
                                row_index,
                                row.get("source_tag", ""),
                                row.get("section", ""),
                                "-".join(str(value) for value in row.get("indices", ())),
                                row.get("funct", ""),
                                row.get("commented", ""),
                                term_key in selected_keys,
                                row.get("force_metric", ""),
                                row.get("force_metric_method", ""),
                                row.get("rmsd", ""),
                                " ".join(str(value) for value in row.get("force_values", [])),
                                " ".join(str(value) for value in row.get("params", [])),
                                row.get("source", ""),
                            ]
                        )
                pref_funct = self.pref_potentials.get(section)
                threshold_rows = [
                    row
                    for row in rows
                    if self._term_is_allowed_by_bond_constraint_mode(row)
                    and self._term_is_bartender_active(row)
                    and self._term_matches_preferred_potential(row, pref_funct)
                    and row.get("rmsd") is not None
                    and float(row["rmsd"]) <= self.rmsd_max
                    and isinstance(row.get("force_metric"), (int, float))
                ]
                force_threshold = self._threshold_for(section, int(funct), threshold_rows or rows)
                rmsd_threshold = self.rmsd_max if math.isfinite(self.rmsd_max) else None
                chunks = _chunk_rows(rows)
                for old_svg in plot_dir.glob(f"{section}_funct_{funct}*.svg"):
                    old_svg.unlink()
                for old_pdf in plot_dir.glob(f"{section}_funct_{funct}*.pdf"):
                    old_pdf.unlink()
                start_index = 0
                for page_index, chunk in enumerate(chunks, start=1):
                    if len(chunks) == 1:
                        pdf_path = plot_dir / f"{section}_funct_{funct}.pdf"
                    else:
                        pdf_path = plot_dir / f"{section}_funct_{funct}_part_{page_index:02d}_of_{len(chunks):02d}.pdf"
                    _write_pdf_plot(
                        pdf_path,
                        f"{section} funct {funct}",
                        chunk,
                        selected_keys=selected_keys,
                        section=section,
                        funct=int(funct),
                        force_threshold=force_threshold,
                        rmsd_threshold=rmsd_threshold,
                        page_index=page_index,
                        page_count=len(chunks),
                        global_start_index=start_index,
                        total_count=len(rows),
                    )
                    start_index += len(chunk)

    def process(self, out_root: Path) -> Dict[str, Any]:
        """Run the full screening postprocess for one output root.

        Parses every ``gmx_out.itp`` under ``out_root``, screens the
        combined term pool, and writes all inventory/screened/plot outputs
        plus ``screening_report.json`` into the resolved output directory.

        Args:
            out_root: pipeline output root to scan.

        Returns:
            The report payload (settings, per-section counts, file paths).
        """
        out_root = out_root.resolve()
        all_terms: Dict[str, List[Dict[str, Any]]] = {section: [] for section in SECTION_ORDER}
        input_files = sorted(out_root.rglob("gmx_out.itp"))
        for itp_path in input_files:
            parsed = self._parse_itp(itp_path, out_root)
            for section, terms in parsed.items():
                all_terms[section].extend(terms)

        screened_results = self._screen_terms(all_terms)
        info_terms = self._info_terms(all_terms)
        final_output_dir = self._output_dir_for_root(out_root)
        final_output_dir.mkdir(parents=True, exist_ok=True)

        all_json_path = final_output_dir / "all_terms.json"
        all_json_path.write_text(
            json.dumps({section: self._json_terms(info_terms[section]) for section in SECTION_ORDER}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._write_all_terms_itp(final_output_dir / "all_terms.itp", info_terms)

        summary_path = final_output_dir / "screened_summary.json"
        summary_path.write_text(
            json.dumps(
                {section: self._json_terms(screened_results[section]) for section in SECTION_ORDER},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._write_itp(final_output_dir / "screened_forcefield.itp", screened_results)
        self._write_plots(final_output_dir, info_terms, screened_results)

        report = {
            "input_root": str(out_root),
            "output_dir": str(final_output_dir),
            "input_file_count": len(input_files),
            "settings": {
                "potentials": self.pref_potentials,
                "bond_constraint_mode": self.bond_constraint_mode,
                "candidate_source": self.candidate_source,
                "show_all_info": self.show_all_info,
                "force_metric_min": self.fc_min_cfg,
                "force_metric_min_mode": self.threshold_mode,
                "multi_constant_metric": self.multi_constant_metric,
                "rmsd_max": self.rmsd_max,
            },
            "parsed_counts": {section: len(all_terms[section]) for section in SECTION_ORDER},
            "all_counts": {section: len(info_terms[section]) for section in SECTION_ORDER},
            "screened_counts": {section: len(screened_results[section]) for section in SECTION_ORDER},
            "files": {
                "all_terms_json": str(all_json_path),
                "all_terms_itp": str(final_output_dir / "all_terms.itp"),
                "screened_summary_json": str(summary_path),
                "screened_forcefield_itp": str(final_output_dir / "screened_forcefield.itp"),
                "plots_dir": str(final_output_dir / "plots") if self.write_plots else None,
            },
        }
        (final_output_dir / "screening_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return report


def _resolve_postprocess_roots(cfg: Dict[str, Any]) -> List[Path]:
    """Collect the postprocess roots from ``paths`` config, deduplicated.

    Sources, in order: expanded ``out_root_glob`` patterns, then
    ``out_roots`` (list) or, only when that is absent, the single
    ``out_root``. Order is preserved; duplicates (by resolved path) are
    dropped.
    """
    paths_cfg = cfg.get("paths", {})
    roots: List[Path] = []
    for pattern in _as_list(paths_cfg.get("out_root_glob")):
        roots.extend(Path(path).resolve() for path in sorted(glob.glob(str(pattern))))
    if paths_cfg.get("out_roots") is not None:
        roots.extend(Path(str(path)).resolve() for path in _as_list(paths_cfg.get("out_roots")))
    elif paths_cfg.get("out_root") is not None:
        roots.append(Path(str(paths_cfg["out_root"])).resolve())

    deduped: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def run_screening_postprocess(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run the screening postprocess over every configured output root.

    Entry point used by ``pipeline``; one ``ScreeningProcessor`` is shared
    across roots so all roots see identical screening settings.

    Args:
        cfg: full resolved pipeline config.

    Returns:
        ``{"root_count", "outputs"}`` with one report per root.

    Raises:
        ValueError: no postprocess root is configured.
    """
    processor = ScreeningProcessor(cfg)
    roots = _resolve_postprocess_roots(cfg)
    if not roots:
        raise ValueError("No postprocess roots configured. Set paths.out_root, paths.out_roots, or paths.out_root_glob.")
    reports = [processor.process(root) for root in roots]
    return {
        "root_count": len(reports),
        "outputs": reports,
    }
