#!/usr/bin/env python3
"""Summarize auto-trim results from xtb_traj_to_pdb runs.

Scans a compare_existing_terms root for *_trim_info.json files produced by
xtb_traj_to_pdb.py, and outputs trim_summary.csv + bar chart.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def scan_trim_info(root: Path) -> list[dict[str, Any]]:
    rows = []
    for info_path in sorted(root.rglob("*_trim_info.json")):
        try:
            data = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Expected path structure: <root>/<label>/<topology>/bartender_job/*_trim_info.json
        try:
            rel_parts = info_path.relative_to(root).parts
            label = rel_parts[0] if len(rel_parts) > 0 else "?"
            topology = rel_parts[1] if len(rel_parts) > 1 else "?"
        except ValueError:
            label, topology = "?", "?"
        rows.append({
            "label": label,
            "topology": topology,
            "total_frames": data.get("total_frames", ""),
            "t0": data.get("t0", ""),
            "skip_frames": data.get("skip_frames", ""),
            "start_index": data.get("start_index", ""),
            "written_frames": data.get("written_frames", ""),
            "trim_fraction": data.get("trim_fraction", ""),
            "path": str(info_path),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_trim(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    x_labels = [f"{r['label']}\n{r['topology']}" for r in rows]
    x = list(range(len(rows)))

    def to_float(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    total = [to_float(r["total_frames"]) for r in rows]
    written = [to_float(r["written_frames"]) for r in rows]
    trimmed = [t - w for t, w in zip(total, written)]
    trim_pct = [100.0 * (t - w) / t if t > 0 else 0.0 for t, w in zip(total, written)]

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(rows) * 0.9 + 2), 5))

    ax = axes[0]
    ax.bar(x, written, label="used", color="#4c72b0")
    ax.bar(x, trimmed, bottom=written, label="trimmed", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("frames")
    ax.set_title("Frame count per case")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.bar(x, trim_pct, color="#dd8452")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
    ax2.set_ylabel("trim %")
    ax2.set_title("Trim fraction per case")
    ax2.set_ylim(0, 100)

    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"trim_summary.{suffix}", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "compare_root",
        type=Path,
        help="compare_existing_terms root directory to scan for *_trim_info.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <compare_root>/trim_summary/",
    )
    args = parser.parse_args()

    root = args.compare_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else root / "trim_summary"

    rows = scan_trim_info(root)
    write_csv(out_dir / "trim_summary.csv", rows)
    plot_trim(rows, out_dir)
    print(f"[trim_summary] found={len(rows)} wrote={out_dir}")


if __name__ == "__main__":
    main()
