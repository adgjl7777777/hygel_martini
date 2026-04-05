"""Thin workflow entry helper for QM-to-OPLS preparation."""

from __future__ import annotations

from pathlib import Path

from ..core.config import load_config
from .defaults import DEFAULT_CONFIG
from .orca_runner import generate_orca_inputs


def run_qm_to_opls(config_path: str | Path) -> None:
    """Load a qm_to_opls maker file and generate ORCA preparation inputs."""
    cfg = load_config(Path(config_path), DEFAULT_CONFIG)
    generate_orca_inputs(cfg)
