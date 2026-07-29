#!/usr/bin/env python3
"""Move flat sweep logs into logs/<variant>/<label>/<mode>.log."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def organize_result_dir(result_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    expected_path = result_dir / "expected_case_outputs.tsv"
    log_dir = result_dir / "logs"
    if not expected_path.exists() or not log_dir.exists():
        return 0, 0

    moved = 0
    skipped = 0
    with expected_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            variant_id = row["variant_id"]
            label = row["label"]
            mode = row["mode"]
            flat = log_dir / f"{variant_id}_{label}_{mode}.log"
            nested = log_dir / variant_id / label / f"{mode}.log"
            if not flat.exists():
                skipped += 1
                continue
            if nested.exists():
                if flat.read_bytes() == nested.read_bytes():
                    if not dry_run:
                        flat.unlink()
                    moved += 1
                else:
                    skipped += 1
                continue
            if not dry_run:
                nested.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(flat), str(nested))
            moved += 1
    return moved, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_dirs",
        nargs="*",
        type=Path,
        help="Result directories. Default: all analyze/results/* directories with logs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not move files.")
    args = parser.parse_args()

    if args.result_dirs:
        result_dirs = [path.resolve() for path in args.result_dirs]
    else:
        project_dir = Path(__file__).resolve().parents[2]
        result_dirs = sorted((project_dir / "analyze" / "results").glob("*"))

    for result_dir in result_dirs:
        moved, skipped = organize_result_dir(result_dir, dry_run=args.dry_run)
        print(f"[logs] {result_dir} moved={moved} skipped={skipped}")


if __name__ == "__main__":
    main()
