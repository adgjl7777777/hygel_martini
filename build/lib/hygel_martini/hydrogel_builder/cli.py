from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hygel-builder",
        description="Run the hydrogel_builder workflow from a maker YAML/JSON file.",
        epilog=(
            "Examples:\n"
            "  hygel-builder maker.yaml\n"
            "  hygel-builder --config /path/to/maker.yaml\n"
            "  python -m hygel_martini.hydrogel_builder maker.yaml"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        help="Optional positional path to the maker config (.yaml/.yml/.json).",
    )
    parser.add_argument(
        "--config",
        help="Path to the maker config (.yaml/.yml/.json). Overrides the positional config path.",
    )
    args = parser.parse_args()

    config_value = args.config or args.config_path or "maker.yaml"

    try:
        from .generator import run_hydrogel_builder

        run_hydrogel_builder(Path(config_value))
    except FileNotFoundError as exc:
        parser.exit(2, f"[ERROR] {exc}\n")
