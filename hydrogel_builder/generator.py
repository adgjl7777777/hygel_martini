"""Thin workflow entry helper for hydrogel generation."""

from __future__ import annotations

from pathlib import Path

from .config_params.generator import run_hydrogel_example


def run_hydrogel_builder(config_path: str | Path) -> None:
    """Run the hydrogel workflow from a maker YAML/JSON file."""
    resolved_path = Path(config_path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config not found: {resolved_path}")
    run_hydrogel_example(str(resolved_path.resolve()))
