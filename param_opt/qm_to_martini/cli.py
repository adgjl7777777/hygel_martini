from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.config import add_qm_to_martini_cli_args
from ..opls_to_martini.writers import write_text
from .defaults import DEFAULT_CONFIG
from .generator import run_qm_to_martini


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="03 workflow: generate Martini/Bartender cases from QM, ORCA, or xTB relaxation inputs."
    )
    add_qm_to_martini_cli_args(parser)
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

    cfg, result = run_qm_to_martini(config_path, args)
    out_root = Path(cfg["paths"]["out_root"])
    print(f"Done. Generated {len(result['cases'])} qm_to_martini case(s) at: {out_root}")
    print(f"Summary: {out_root / 'summary.json'}")
    if "collect" in result:
        collect_name = cfg["bartender_pipeline"]["postprocess"]["collect_json"]
        print(f"Collected Bartender summaries: {out_root / collect_name}")
    if "merge" in result:
        merged_name = cfg["bartender_pipeline"]["postprocess"]["merged_itp"]
        print(f"Merged forcefield: {out_root / merged_name}")
