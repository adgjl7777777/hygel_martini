"""Thin workflow entry helper for OPLS-to-Martini case generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from hygel_martini.core.config import apply_cli_overrides, load_config
from .builder import build_cases
from .defaults import DEFAULT_CONFIG
from .fitting import check_existing_data_tools, run_existing_data_fit, run_postprocess_only


def run_opls_to_martini(
    config_path: str | Path,
    overrides: argparse.Namespace | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load an opls_to_martini maker file, apply optional overrides, and build cases."""
    cfg = load_config(Path(config_path), DEFAULT_CONFIG)
    if overrides is not None:
        cfg = apply_cli_overrides(cfg, overrides)

    if overrides is not None and getattr(overrides, "postprocess_only", False):
        return cfg, run_postprocess_only(cfg)

    check_tools = []
    if overrides is not None:
        if getattr(overrides, "check_gmx", False):
            check_tools.append("gmx")
        if getattr(overrides, "check_bartender", False):
            check_tools.append("bartender")
    if check_tools:
        return cfg, check_existing_data_tools(cfg, check_tools)

    workflow_cfg = cfg.get("workflow", {})
    if not isinstance(workflow_cfg, dict):
        workflow_cfg = {"mode": str(workflow_cfg)}
    workflow_mode = str(workflow_cfg.get("mode", "constructor")).strip().lower()
    if workflow_mode in {"existing_data_fit", "existing-data-fit", "fit_existing", "bartender_existing"}:
        return cfg, run_existing_data_fit(cfg)

    if int(cfg["system"]["replicas"]) < 1:
        raise ValueError("replicas must be >= 1")
    if int(cfg["sampling"]["sample_nsteps"]) < 1:
        raise ValueError("sample_nsteps must be >= 1")
    result = build_cases(cfg)
    return cfg, result
