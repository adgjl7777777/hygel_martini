#!/usr/bin/env python3
"""Summarize postprocess sweep outputs into case/variant CSV tables.

This script owns the tabulation step between a screening-threshold sweep
and its plots: it walks
``<result_dir>/outputs/<variant>/<label>/<mode>/screening_report.json``
(plus the sibling ``screened_summary.json``), builds one row per case
with accepted/parsed/info counts, screening settings, and RMSD ("rmse")
and force-metric statistics (n/min/max/mean/median/p90), and aggregates
the cases per variant.

Callers: run as a standalone CLI; its outputs
(``tables/case_summary.csv``, ``tables/variant_summary.csv``,
``tables/missing_outputs.csv``, ``tables/summary_overview.md``) are the
inputs of ``analysis.plotter``.

Conventions: the column prefix "rmse" carries the Bartender per-term
``rmsd`` fit values; "force" carries the screening force metric. An
optional ``expected_case_outputs.tsv`` in the result directory lets
missing case outputs be reported explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence


SECTIONS = ["bonds", "constraints", "angles", "dihedrals", "impropers"]
STAT_KEYS = ["n", "min", "max", "mean", "median", "p90"]


def as_float(value: Any) -> float | None:
    """Return the value as a finite float, or None for anything else."""
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Linearly interpolated percentile of the values.

    Args:
        values: sample values (order irrelevant); empty yields None.
        fraction: percentile as a fraction, e.g. 0.9 for P90.

    Returns:
        The interpolated percentile, or None for an empty input.
    """
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = fraction * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def stats(values: Iterable[Any]) -> Dict[str, Any]:
    """Compute the standard statistic block over the finite values.

    Non-numeric/non-finite entries are dropped first; with nothing left,
    every key maps to "" so CSV cells stay blank rather than zero.

    Returns:
        Dict with the ``STAT_KEYS`` (n, min, max, mean, median, p90).
    """
    numeric = [value for value in (as_float(item) for item in values) if value is not None]
    if not numeric:
        return {key: "" for key in STAT_KEYS}
    return {
        "n": len(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "mean": mean(numeric),
        "median": median(numeric),
        "p90": percentile(numeric, 0.9),
    }


def flatten_stats(prefix: str, values: Iterable[Any]) -> Dict[str, Any]:
    """Prefix the stat block keys, e.g. "all_rmse" -> "all_rmse_p90"."""
    return {f"{prefix}_{key}": value for key, value in stats(values).items()}


def load_json(path: Path) -> Dict[str, Any]:
    """Read and parse a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def safe_get(mapping: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Walk nested dict keys, returning ``default`` on any missing step."""
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def variant_from_report_path(report_path: Path, outputs_dir: Path) -> tuple[str, str, str]:
    """Extract (variant_id, label, mode) from a report's path.

    Relies on the fixed sweep layout
    ``outputs/<variant>/<label>/<mode>/screening_report.json``.

    Raises:
        ValueError: the path is too shallow to carry all three parts.
    """
    rel = report_path.relative_to(outputs_dir)
    parts = rel.parts
    if len(parts) < 4:
        raise ValueError(f"Unexpected report path under outputs: {report_path}")
    return parts[0], parts[1], parts[2]


def selected_terms(summary: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Normalize a ``screened_summary.json`` payload to lists per section."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    for section in SECTIONS:
        rows = summary.get(section, [])
        result[section] = rows if isinstance(rows, list) else []
    return result


def read_expected(result_dir: Path) -> set[tuple[str, str, str, Path]]:
    """Read the optional expected-output manifest for missing-case checks.

    Returns:
        Set of (variant_id, label, mode, expected report path); empty when
        ``expected_case_outputs.tsv`` does not exist.
    """
    expected_path = result_dir / "expected_case_outputs.tsv"
    if not expected_path.exists():
        return set()
    rows: set[tuple[str, str, str, Path]] = set()
    with expected_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            report = Path(row["expected_report"])
            rows.add((row["variant_id"], row["label"], row["mode"], report))
    return rows


def collect_case_rows(result_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build one summary row per (variant, label, mode) case.

    For each ``screening_report.json`` found under ``outputs/``: record
    the screening settings, per-section accepted/parsed/info counts and
    force-metric cutoffs, plus rmse/force stat blocks over the screened
    terms (all terms and per section). Cases named in the expected
    manifest whose report is genuinely absent become missing rows.

    Args:
        result_dir: sweep result directory.

    Returns:
        (case rows, missing-output rows).
    """
    outputs_dir = result_dir / "outputs"
    case_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    expected = read_expected(result_dir)
    seen: set[tuple[str, str, str, Path]] = set()

    reports = sorted(outputs_dir.glob("*/*/*/screening_report.json")) if outputs_dir.exists() else []
    for report_path in reports:
        variant_id, label, mode = variant_from_report_path(report_path, outputs_dir)
        seen.add((variant_id, label, mode, report_path))
        report = load_json(report_path)
        summary_path = report_path.with_name("screened_summary.json")
        summary = load_json(summary_path) if summary_path.exists() else {section: [] for section in SECTIONS}
        terms_by_section = selected_terms(summary)
        all_terms = [term for section in SECTIONS for term in terms_by_section[section]]

        settings = report.get("settings", {})
        counts = report.get("screened_counts", {})
        parsed_counts = report.get("parsed_counts", {})
        info_counts = report.get("all_counts", {})

        row: dict[str, Any] = {
            "variant_id": variant_id,
            "label": label,
            "mode": mode,
            "report_path": str(report_path),
            "angles_funct": safe_get(settings, "potentials", "angles"),
            "dihedrals_funct": safe_get(settings, "potentials", "dihedrals"),
            "impropers_funct": safe_get(settings, "potentials", "impropers"),
            "bond_constraint_mode": settings.get("bond_constraint_mode", ""),
            "candidate_source": settings.get("candidate_source", ""),
            "multi_constant_metric": settings.get("multi_constant_metric", ""),
            "force_metric_min_mode": settings.get("force_metric_min_mode", ""),
            "rmsd_max_cutoff": settings.get("rmsd_max", ""),
            "show_all_info": settings.get("show_all_info", ""),
            "accepted_total": sum(int(counts.get(section, 0) or 0) for section in SECTIONS),
            "parsed_total": sum(int(parsed_counts.get(section, 0) or 0) for section in SECTIONS),
            "info_total": sum(int(info_counts.get(section, 0) or 0) for section in SECTIONS),
        }
        for section in SECTIONS:
            row[f"accepted_{section}"] = int(counts.get(section, 0) or 0)
            row[f"parsed_{section}"] = int(parsed_counts.get(section, 0) or 0)
            row[f"info_{section}"] = int(info_counts.get(section, 0) or 0)
            row[f"force_min_cutoff_{section}"] = safe_get(settings, "force_metric_min", section)

        row.update(flatten_stats("all_rmse", (term.get("rmsd") for term in all_terms)))
        row.update(flatten_stats("all_force", (term.get("force_metric") for term in all_terms)))
        for section in SECTIONS:
            terms = terms_by_section[section]
            row.update(flatten_stats(f"{section}_rmse", (term.get("rmsd") for term in terms)))
            row.update(flatten_stats(f"{section}_force", (term.get("force_metric") for term in terms)))

        case_rows.append(row)

    for variant_id, label, mode, report in sorted(expected):
        if (variant_id, label, mode, report) not in seen and not report.exists():
            missing_rows.append(
                {
                    "variant_id": variant_id,
                    "label": label,
                    "mode": mode,
                    "expected_report": str(report),
                }
            )

    return case_rows, missing_rows


def aggregate_variant_rows(case_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate case rows into one row per variant.

    Settings columns are copied from the variant's first case (assumed
    identical across cases); counts are summed and ``zero_<section>_cases``
    counts cases with no accepted terms. The rmse/force stat blocks are
    recomputed over the pooled screened terms re-read from each case's
    ``screened_summary.json`` (not averaged from per-case stats).

    Args:
        case_rows: rows from ``collect_case_rows``.

    Returns:
        Variant rows sorted by variant id.
    """
    grouped: Dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[str(row["variant_id"])].append(row)

    variant_rows: list[dict[str, Any]] = []
    for variant_id, rows in sorted(grouped.items()):
        first = rows[0]
        total_cases = len(rows)
        variant_row: dict[str, Any] = {
            "variant_id": variant_id,
            "case_count": total_cases,
            "angles_funct": first.get("angles_funct", ""),
            "dihedrals_funct": first.get("dihedrals_funct", ""),
            "impropers_funct": first.get("impropers_funct", ""),
            "bond_constraint_mode": first.get("bond_constraint_mode", ""),
            "candidate_source": first.get("candidate_source", ""),
            "multi_constant_metric": first.get("multi_constant_metric", ""),
            "force_metric_min_mode": first.get("force_metric_min_mode", ""),
            "rmsd_max_cutoff": first.get("rmsd_max_cutoff", ""),
            "show_all_info": first.get("show_all_info", ""),
            "accepted_total": sum(int(row.get("accepted_total", 0) or 0) for row in rows),
            "parsed_total": sum(int(row.get("parsed_total", 0) or 0) for row in rows),
            "info_total": sum(int(row.get("info_total", 0) or 0) for row in rows),
        }
        for section in SECTIONS:
            variant_row[f"accepted_{section}"] = sum(int(row.get(f"accepted_{section}", 0) or 0) for row in rows)
            variant_row[f"zero_{section}_cases"] = sum(1 for row in rows if int(row.get(f"accepted_{section}", 0) or 0) == 0)
            variant_row[f"force_min_cutoff_{section}"] = first.get(f"force_min_cutoff_{section}", "")

        report_terms: list[dict[str, Any]] = []
        for row in rows:
            summary_path = Path(str(row["report_path"])).with_name("screened_summary.json")
            if not summary_path.exists():
                continue
            summary = selected_terms(load_json(summary_path))
            report_terms.extend(term for section in SECTIONS for term in summary[section])

        variant_row.update(flatten_stats("all_rmse", (term.get("rmsd") for term in report_terms)))
        variant_row.update(flatten_stats("all_force", (term.get("force_metric") for term in report_terms)))
        for section in SECTIONS:
            section_terms = [term for term in report_terms if term.get("section") == section]
            variant_row.update(flatten_stats(f"{section}_rmse", (term.get("rmsd") for term in section_terms)))
            variant_row.update(flatten_stats(f"{section}_force", (term.get("force_metric") for term in section_terms)))

        variant_rows.append(variant_row)
    return variant_rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write dict rows as CSV, unioning keys in first-seen order.

    An empty row list produces an empty file (no header), which the
    downstream plotter treats as absent data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_overview(path: Path, case_rows: Sequence[dict[str, Any]], variant_rows: Sequence[dict[str, Any]], missing: Sequence[dict[str, Any]]) -> None:
    """Write the Markdown overview: row counts plus suggested sort columns."""
    lines = [
        "# Sweep Summary Overview",
        "",
        f"- cases summarized: {len(case_rows)}",
        f"- variants summarized: {len(variant_rows)}",
        f"- missing expected outputs: {len(missing)}",
        "",
        "## Quick Filters",
        "",
        "Start by sorting `variant_summary.csv` with these columns:",
        "",
        "- `accepted_total`",
        "- `accepted_angles`, `accepted_dihedrals`, `accepted_impropers`",
        "- `zero_angles_cases`, `zero_dihedrals_cases`",
        "- `all_rmse_max`, `all_rmse_median`, `all_rmse_p90`",
        "- `angles_rmse_max`, `dihedrals_rmse_max`",
        "- `angles_force_min`, `dihedrals_force_min`",
        "",
        "Then inspect `case_summary.csv` for label/mode-specific failures.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """CLI entry point: collect case rows, aggregate, and write the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="Analyze result directory, e.g. analyze/results/01_summary_sweep")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for CSV outputs. Default: result_dir/tables")
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else result_dir / "tables"

    case_rows, missing_rows = collect_case_rows(result_dir)
    variant_rows = aggregate_variant_rows(case_rows)

    write_csv(out_dir / "case_summary.csv", case_rows)
    write_csv(out_dir / "variant_summary.csv", variant_rows)
    write_csv(out_dir / "missing_outputs.csv", missing_rows)
    write_overview(out_dir / "summary_overview.md", case_rows, variant_rows, missing_rows)

    print(f"[summary] cases={len(case_rows)} variants={len(variant_rows)} missing={len(missing_rows)}")
    print(f"[summary] wrote {out_dir}")


if __name__ == "__main__":
    main()
