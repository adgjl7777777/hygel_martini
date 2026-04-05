"""Thin workflow entry helper for OPLS-to-Martini case generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from ..core.config import apply_cli_overrides, load_config
from .builder import build_cases
from .defaults import DEFAULT_CONFIG


def run_opls_to_martini(
    config_path: str | Path,
    overrides: argparse.Namespace | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load an opls_to_martini maker file, apply optional overrides, and build cases."""
    cfg = load_config(Path(config_path), DEFAULT_CONFIG)
    if overrides is not None:
        cfg = apply_cli_overrides(cfg, overrides)
    if int(cfg["system"]["replicas"]) < 1:
        raise ValueError("replicas must be >= 1")
    if int(cfg["sampling"]["sample_nsteps"]) < 1:
        raise ValueError("sample_nsteps must be >= 1")
    result = build_cases(cfg)
    return cfg, result
