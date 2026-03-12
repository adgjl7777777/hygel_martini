from __future__ import annotations

import argparse
from pathlib import Path

from ..core.config import load_config
from .orca_runner import generate_orca_inputs

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OPLS parameters via DFT (ORCA) and LigParGen.")
    parser.add_argument("--config", required=True, help="Path to the config yaml (e.g. maker.yaml)")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    cfg = load_config(config_path)

    # 1. ORCA Input Generation
    generate_orca_inputs(cfg)

    # 2. LigParGen API (To be integrated next)
    # from .ligpargen_api import run_parameterization_flow
    # run_parameterization_flow(...)
