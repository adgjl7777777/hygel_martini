from __future__ import annotations

import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SECTION_INFO = {
    "bonds": 2,
    "constraints": 2,
    "angles": 3,
    "dihedrals": 4,
    "impropers": 4,
}
SECTION_ORDER = ("bonds", "constraints", "angles", "dihedrals", "impropers")
RMSD_RE = re.compile(r"rmsd:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
PLOT_MAX_POINTS = 10

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
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _canon_indices(indices: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(value) for value in indices))


def _find_case_json(start: Path) -> Optional[Path]:
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
    if base is None:
        return Path(path.name)
    try:
        return path.resolve().relative_to(base.resolve())
    except ValueError:
        return Path(path.name)


def _format_number(value: Optional[float]) -> str:
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
    if total <= 0:
        return []
    chunk_count = math.ceil(total / limit)
    base = total // chunk_count
    remainder = total % chunk_count
    return [base] * (chunk_count - remainder) + [base + 1] * remainder


def _chunk_rows(rows: Sequence[Dict[str, Any]], limit: int = PLOT_MAX_POINTS) -> List[List[Dict[str, Any]]]:
    chunks: List[List[Dict[str, Any]]] = []
    start = 0
    for size in _chunk_sizes(len(rows), limit):
        chunks.append(list(rows[start : start + size]))
        start += size
    return chunks


def _potential_title(section: str, funct: int) -> Tuple[str, str, str]:
    return POTENTIAL_INFO.get(
        (section, funct),
        (
            f"{section} potential",
            r"$\mathrm{equation\ not\ annotated}$",
            "params: see ITP line",
        ),
    )


def _axis_bounds(values: Sequence[float], threshold: Optional[float], *, zero_floor: bool = True) -> Tuple[float, float]:
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
        values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        return values

    def cutoff_state(values: List[float], threshold: Optional[float], mode: str) -> Tuple[str, bool, bool]:
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
        return _format_number(value)

    def draw_panel(ax: Any, *, key: str, label: str, threshold: Optional[float], pass_mode: str) -> None:
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
    """Parse Bartender ITP outputs and write full plus screened postprocess data."""

    def __init__(self, cfg: Dict[str, Any]):
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
        if self.candidate_source == "all":
            return True
        return self._term_is_bartender_active(term)

    def _force_values_and_metric(
        self,
        section: str,
        funct: int,
        numeric_params: Sequence[float],
    ) -> tuple[List[float], Optional[float], str]:
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
        return (term["section"], _canon_indices(tuple(term["indices"])))

    @staticmethod
    def _term_is_bartender_active(term: Dict[str, Any]) -> bool:
        return not bool(term["commented"])

    def _term_is_allowed_by_bond_constraint_mode(self, term: Dict[str, Any]) -> bool:
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
        rows = []
        for term in terms:
            rows.append({key: value for key, value in term.items() if key != "raw"})
        return rows

    def _write_all_terms_itp(self, path: Path, all_terms: Dict[str, List[Dict[str, Any]]]) -> None:
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
    processor = ScreeningProcessor(cfg)
    roots = _resolve_postprocess_roots(cfg)
    if not roots:
        raise ValueError("No postprocess roots configured. Set paths.out_root, paths.out_roots, or paths.out_root_glob.")
    reports = [processor.process(root) for root in roots]
    return {
        "root_count": len(reports),
        "outputs": reports,
    }
