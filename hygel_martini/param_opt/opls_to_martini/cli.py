from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.config import add_opls_to_martini_cli_args
from .defaults import DEFAULT_CONFIG
from .generator import run_opls_to_martini
from .writers import write_text


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="02 workflow: generate Martini constructor cases from OPLS-side inputs."
    )
    add_opls_to_martini_cli_args(parser)
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

    cfg, result = run_opls_to_martini(config_path, args)
    out_root = Path(cfg["paths"]["out_root"])
    print(f"Done. Generated {len(result['cases'])} case(s) at: {out_root}")
    print(f"Summary: {out_root / 'summary.json'}")
