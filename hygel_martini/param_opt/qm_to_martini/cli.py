"""Command-line entry point for the stage-03 (qm_to_martini) workflow.

This module owns argument parsing and console reporting for the
``hygel-qm-to-martini`` command. Shared config-related flags come from
``hygel_martini.core.config.add_qm_to_martini_cli_args``; the actual work
is delegated to ``generator.run_qm_to_martini`` (which dispatches to
``pipeline.run_pipeline`` / ``run_postprocess_only`` / tool checks).

Inputs: a YAML config path (``--config``) plus mode flags
(``--dump-default-config``, ``--postprocess-only``, ``--check-tools``).
Outputs: pipeline artifacts under the configured ``out_root`` and status
lines on stdout; ``--check-tools`` exits nonzero when a tool is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hygel_martini.core.config import add_qm_to_martini_cli_args
from ..opls_to_martini.writers import write_text
from .defaults import DEFAULT_CONFIG
from .generator import run_qm_to_martini


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse parser: shared config flags plus stage-03 modes."""
    parser = argparse.ArgumentParser(
        description="03 workflow: generate Martini/Bartender cases from QM, ORCA, or xTB relaxation inputs."
    )
    add_qm_to_martini_cli_args(parser)
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Skip case generation and run only screening postprocess on configured postprocess roots.",
    )
    parser.add_argument(
        "--check-tools",
        nargs="+",
        choices=("xtb", "orca", "bartender"),
        help="Validate configured tool paths from the YAML config and exit.",
    )
    return parser


def main() -> None:
    """Console-script entry point.

    Mutually exclusive-by-precedence modes: ``--dump-default-config``
    writes ``DEFAULT_CONFIG`` to the ``--config`` path and returns;
    ``--check-tools`` prints per-tool resolution status and exits 1 on any
    missing tool; ``--postprocess-only`` reports screening outputs; the
    default mode reports generated cases and the summary path.

    Raises:
        ValueError: ``--dump-default-config`` without ``--config``.
        SystemExit: a ``--check-tools`` failure (exit code 1).
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

    cfg, result = run_qm_to_martini(config_path, args)
    out_root = Path(str(cfg["paths"].get("out_root") or cfg["paths"].get("postprocess_output_root") or "."))
    if args.check_tools:
        print(f"Checked tool paths from config: {config_path}")
        for tool in result["tools"]:
            binary = tool["binary"]
            status = "OK" if binary["exists"] else "MISSING"
            resolved = binary["resolved"] or "(not found)"
            print(f"[{status}] {tool['name']} binary: {binary['configured']} -> {resolved}")
            for key in ("env_script", "root"):
                if key not in tool:
                    continue
                payload = tool[key]
                label = key.replace("_", " ")
                status = "OK" if payload["exists"] else "MISSING"
                resolved = payload["resolved"] or "(empty)"
                print(f"[{status}] {tool['name']} {label}: {payload['configured'] or '(empty)'} -> {resolved}")
        if not result["ok"]:
            raise SystemExit(1)
        return
    if args.postprocess_only:
        print("Done. Postprocessed existing qm_to_martini results.")
        if result.get("summary_json"):
            print(f"Summary: {result['summary_json']}")
        for output in result.get("screening", {}).get("outputs", []):
            print(f"Output: {output.get('output_dir')}")
    else:
        print(f"Done. Generated {len(result['cases'])} qm_to_martini case(s) at: {out_root}")
        print(f"Summary: {out_root / 'summary.json'}")
