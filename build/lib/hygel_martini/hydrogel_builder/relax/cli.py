from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hygel-relax",
        description="Run the hydrogel_builder relaxation workflow from a maker YAML/JSON file.",
        epilog=(
            "Examples:\n"
            "  hygel-relax maker_soft_em.yaml\n"
            "  hygel-relax --config maker_soft_md.yaml\n"
            "  python -m hygel_martini.hydrogel_builder.relax maker_soft_em.yaml"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        help="Optional positional path to the relaxation config (.yaml/.yml/.json).",
    )
    parser.add_argument(
        "--config",
        help="Path to the relaxation config (.yaml/.yml/.json). Overrides the positional config path.",
    )
    args = parser.parse_args()

    config_value = args.config or args.config_path or "maker_soft_em.yaml"

    try:
        from .generator import run_relax_workflow

        run_relax_workflow(Path(config_value))
    except FileNotFoundError as exc:
        parser.exit(2, f"[ERROR] {exc}\n")
