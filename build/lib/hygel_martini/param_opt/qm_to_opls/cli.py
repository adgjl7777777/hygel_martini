from __future__ import annotations

import argparse
import json
from pathlib import Path

from hygel_martini.core.config import add_config_args
from .defaults import DEFAULT_CONFIG
from .generator import run_qm_to_opls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="01 workflow: generate OPLS/ORCA preparation inputs from QM-side configs."
    )
    add_config_args(parser)
    args = parser.parse_args()

    config_path = Path(args.config)
    if args.dump_default_config:
        config_path.write_text(json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote default config: {config_path}")
        return

    run_qm_to_opls(config_path)
