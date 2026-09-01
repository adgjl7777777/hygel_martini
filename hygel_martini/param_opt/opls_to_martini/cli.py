"""Command-line entry point for the stage-02 (OPLS -> Martini) workflow.

Adds the workflow-specific flags (``--postprocess-only``, ``--check-gmx``,
``--check-bartender``) on top of the shared config arguments, then
dispatches to :func:`run_opls_to_martini` and prints a result summary
matching the workflow branch that ran (tool check, postprocess-only,
existing-data fit, or constructor case generation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hygel_martini.core.config import add_opls_to_martini_cli_args
from .defaults import DEFAULT_CONFIG
from .generator import run_opls_to_martini
from .writers import write_text


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse parser: shared config args plus 02-only flags."""
    parser = argparse.ArgumentParser(
        description="02 workflow: generate Martini constructor cases from OPLS-side inputs."
    )
    add_opls_to_martini_cli_args(parser)
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Skip fitting setup and run only Bartender screening postprocess on configured roots.",
    )
    parser.add_argument(
        "--check-gmx",
        action="store_true",
        help="Check the configured GROMACS command and exit.",
    )
    parser.add_argument(
        "--check-bartender",
        action="store_true",
        help="Check the configured Bartender binary and exit.",
    )
    return parser


def main() -> None:
    """Run the opls_to_martini CLI.

    Handles ``--dump-default-config`` locally (writes DEFAULT_CONFIG as
    JSON and returns); every other invocation is delegated to
    :func:`run_opls_to_martini`, and the returned result dict is printed
    according to which workflow branch produced it.  Tool checks exit
    with status 1 when a configured tool is missing.
    """
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
    if args.check_gmx or args.check_bartender:
        for tool in result["tools"]:
            status = "OK" if tool["exists"] else "MISSING"
            resolved = tool["resolved"] or "(not found)"
            print(f"[{status}] {tool['name']}: {tool['configured']} -> {resolved}")
        if not result["ok"]:
            raise SystemExit(1)
        return
    if args.postprocess_only:
        print("Done. Postprocessed existing OPLS-to-Martini Bartender results.")
        if result.get("summary_json"):
            print(f"Summary: {result['summary_json']}")
        for output in result.get("screening", {}).get("outputs", []):
            print(f"Output: {output.get('output_dir')}")
        return

    out_root = Path(cfg["paths"]["out_root"])
    if result.get("workflow") == "existing_data_fit":
        print(f"Done. Prepared {len(result['cases'])} existing-data fitting case(s) at: {out_root}")
        print(f"Run all: {result['run_all']}")
        print(f"Summary: {out_root / 'summary.json'}")
        return

    print(f"Done. Generated {len(result['cases'])} constructor case(s) at: {out_root}")
    print(f"Summary: {out_root / 'summary.json'}")
