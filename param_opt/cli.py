from __future__ import annotations

import argparse
import json
from pathlib import Path

from .simulation.builder import build_cases
from .core.config import add_cli_args, apply_cli_overrides, load_config
from .core.defaults import DEFAULT_CONFIG
from .simulation.writers import write_text


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate oligomer construction cases and GROMACS prep skeleton."
    )
    add_cli_args(parser)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None

    if args.dump_default_config:
        if config_path is None:
            raise ValueError("--dump-default-config needs --config path")
        write_text(config_path, json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False))
        print(f"Wrote default config: {config_path}")
        return

    cfg = load_config(config_path)
    cfg = apply_cli_overrides(cfg, args)

    if int(cfg["system"]["replicas"]) < 1:
        raise ValueError("replicas must be >= 1")
    if int(cfg["sampling"]["sample_nsteps"]) < 1:
        raise ValueError("sample_nsteps must be >= 1")

    result = build_cases(cfg)
    out_root = Path(cfg["paths"]["out_root"])
    print(f"Done. Generated {len(result['cases'])} case(s) at: {out_root}")
    print(f"Summary: {out_root / 'summary.json'}")
