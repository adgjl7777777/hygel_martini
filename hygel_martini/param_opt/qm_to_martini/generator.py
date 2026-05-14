"""Thin workflow entry helper for QM-to-Martini/Bartender generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from hygel_martini.core.config import apply_cli_overrides, load_config
from .defaults import DEFAULT_CONFIG
from .config import check_configured_tools
from .pipeline import run_pipeline, run_postprocess_only


def run_qm_to_martini(
    config_path: str | Path,
    overrides: argparse.Namespace | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load a qm_to_martini maker file, apply optional overrides, and run the pipeline."""
    cfg = load_config(Path(config_path), DEFAULT_CONFIG)
    if overrides is not None:
        cfg = apply_cli_overrides(cfg, overrides)
    if overrides is not None and getattr(overrides, "check_tools", None):
        result = check_configured_tools(cfg, overrides.check_tools)
    elif overrides is not None and getattr(overrides, "postprocess_only", False):
        result = run_postprocess_only(cfg)
    else:
        result = run_pipeline(cfg)
    return cfg, result
