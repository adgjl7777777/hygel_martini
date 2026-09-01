"""Command-line entry point for the hydrogel_builder workflow.

Owns the ``hygel-builder`` console script (also reachable as
``python -m hygel_martini.hydrogel_builder``): it resolves the maker
config path (``--config`` beats the positional argument; default
``maker.yaml`` in the current directory) and delegates the whole build
to ``generator.run_hydrogel_builder``.  A missing config exits with
status 2 instead of a traceback.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Parse the config location and run the hydrogel builder workflow."""
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
