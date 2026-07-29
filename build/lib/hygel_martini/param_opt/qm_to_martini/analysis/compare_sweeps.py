#!/usr/bin/env python3
"""Compare two postprocess sweep result directories (e.g., old trim vs new trim).

Loads tables/variant_summary.csv from both directories, merges on variant_id,
and produces side-by-side bar charts, a diff heatmap, and a diff CSV.

Usage:
    python compare_sweeps.py <old_result_dir> <new_result_dir> [options]

Example:
    python analyze/scripts/compare_sweeps.py \\
        analyze_old/results/01b_all_candidate_sweep \\
        analyze/results/01b_all_candidate_sweep \\
        --label-old old --label-new new
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPARE_COLS = [
    "accepted_total",
    "accepted_angles",
    "accepted_dihedrals",
    "accepted_impropers",
    "all_rmse_mean",
    "all_rmse_max",
    "angles_rmse_mean",
    "dihedrals_rmse_mean",
]


def load_variant_table(result_dir: Path) -> pd.DataFrame | None:
    path = result_dir / "tables" / "variant_summary.csv"
    if not path.exists():
        print(f"[compare_sweeps] WARN: not found: {path}")
        return None
    return pd.read_csv(path)


def _to_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def plot_bar_comparison(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    out_dir: Path,
    label_old: str,
    label_new: str,
) -> None:
    merged = old_df.merge(new_df, on="variant_id", suffixes=(f"_{label_old}", f"_{label_new}"))
    if merged.empty:
        print("[compare_sweeps] No matching variants to plot.")
        return

    variants = merged["variant_id"].tolist()
    n = len(variants)
    x = np.arange(n)
    width = 0.35

    present_cols = [
        c for c in COMPARE_COLS
        if f"{c}_{label_old}" in merged.columns and f"{c}_{label_new}" in merged.columns
    ]
    if not present_cols:
        return

    fig, axes = plt.subplots(len(present_cols), 1, figsize=(max(10, n * 0.5), 3.5 * len(present_cols)))
    if len(present_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, present_cols):
        old_vals = _to_float_series(merged[f"{col}_{label_old}"]).tolist()
        new_vals = _to_float_series(merged[f"{col}_{label_new}"]).tolist()
        ax.bar(x - width / 2, old_vals, width, label=label_old, color="#4c72b0", alpha=0.85)
        ax.bar(x + width / 2, new_vals, width, label=label_new, color="#55a868", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, fontsize=6, rotation=60, ha="right")
        ax.set_ylabel(col, fontsize=8)
        ax.legend(fontsize=8)

    fig.suptitle(f"Variant comparison: {label_old} vs {label_new}", fontsize=10)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"variant_comparison.{suffix}", dpi=150)
    plt.close(fig)


def plot_diff_heatmap(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    out_dir: Path,
    label_old: str,
    label_new: str,
) -> None:
    merged = old_df.merge(new_df, on="variant_id", suffixes=(f"_{label_old}", f"_{label_new}"))
    if merged.empty:
        return

    diff_cols = [
        c for c in ["accepted_total", "accepted_angles", "accepted_dihedrals", "accepted_impropers"]
        if f"{c}_{label_old}" in merged.columns and f"{c}_{label_new}" in merged.columns
    ]
    if not diff_cols:
        return

    variants = merged["variant_id"].tolist()
    diff_matrix = np.zeros((len(diff_cols), len(variants)))
    for j, col in enumerate(diff_cols):
        old_vals = _to_float_series(merged[f"{col}_{label_old}"]).to_numpy()
        new_vals = _to_float_series(merged[f"{col}_{label_new}"]).to_numpy()
        diff_matrix[j] = new_vals - old_vals

    vmax = max(float(np.abs(diff_matrix).max()), 1.0)
    fig, ax = plt.subplots(figsize=(max(8, len(variants) * 0.4), max(3, len(diff_cols) * 0.9)))
    im = ax.imshow(diff_matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, fontsize=5, rotation=75, ha="right")
    ax.set_yticks(range(len(diff_cols)))
    ax.set_yticklabels(diff_cols, fontsize=8)
    fig.colorbar(im, ax=ax, label=f"{label_new} − {label_old}")
    ax.set_title(f"Accepted term diff ({label_new} − {label_old})")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"diff_heatmap.{suffix}", dpi=150)
    plt.close(fig)


def write_diff_csv(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    out_dir: Path,
    label_old: str,
    label_new: str,
) -> None:
    merged = old_df.merge(new_df, on="variant_id", suffixes=(f"_{label_old}", f"_{label_new}"))
    if merged.empty:
        return

    present_cols = [
        c for c in COMPARE_COLS
        if f"{c}_{label_old}" in merged.columns and f"{c}_{label_new}" in merged.columns
    ]

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        diff_row: dict[str, Any] = {"variant_id": row["variant_id"]}
        for col in present_cols:
            old_val = row.get(f"{col}_{label_old}", "")
            new_val = row.get(f"{col}_{label_new}", "")
            diff_row[f"{col}_{label_old}"] = old_val
            diff_row[f"{col}_{label_new}"] = new_val
            try:
                diff_row[f"{col}_diff"] = round(float(new_val) - float(old_val), 4)
            except (TypeError, ValueError):
                diff_row[f"{col}_diff"] = ""
        rows.append(diff_row)

    if rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with (out_dir / "variant_diff.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old_result_dir", type=Path, help="Old sweep result directory (e.g., analyze_old/results/01b_all_candidate_sweep)")
    parser.add_argument("new_result_dir", type=Path, help="New sweep result directory (e.g., analyze/results/01b_all_candidate_sweep)")
    parser.add_argument("--label-old", default="old", help="Legend label for old dataset (default: old)")
    parser.add_argument("--label-new", default="new", help="Legend label for new dataset (default: new)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Default: <new_result_dir>/comparison/")
    args = parser.parse_args()

    old_df = load_variant_table(args.old_result_dir.resolve())
    new_df = load_variant_table(args.new_result_dir.resolve())

    if old_df is None or new_df is None:
        print("[compare_sweeps] ERROR: tables/variant_summary.csv missing in one or both directories.")
        print("[compare_sweeps] Run summarize_postprocess_sweep.py on each result dir first.")
        return

    out_dir = args.out_dir.resolve() if args.out_dir else args.new_result_dir.resolve() / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_bar_comparison(old_df, new_df, out_dir, args.label_old, args.label_new)
    plot_diff_heatmap(old_df, new_df, out_dir, args.label_old, args.label_new)
    write_diff_csv(old_df, new_df, out_dir, args.label_old, args.label_new)
    print(f"[compare_sweeps] wrote {out_dir}")


if __name__ == "__main__":
    main()
