"""Thin workflow entry helper for QM-to-OPLS preparation.

Owns the programmatic entry point of stage 01: it only loads the config
(merging it over ``DEFAULT_CONFIG``) and delegates the actual work to
``orca_runner.generate_orca_inputs``.  Called by ``cli.main`` and
re-exported from the package ``__init__``.
"""

from __future__ import annotations

from pathlib import Path

from hygel_martini.core.config import load_config
from .defaults import DEFAULT_CONFIG
from .orca_runner import generate_orca_inputs


def run_qm_to_opls(config_path: str | Path) -> None:
    """Load a qm_to_opls config file and generate ORCA preparation inputs.

    Args:
        config_path: Path to the user config; values are layered on top
            of ``DEFAULT_CONFIG`` by ``core.config.load_config``.
    """
    cfg = load_config(Path(config_path), DEFAULT_CONFIG)
    generate_orca_inputs(cfg)
