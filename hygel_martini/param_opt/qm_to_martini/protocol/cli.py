"""Command-line interface for the sealed parameterization protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

from .engine import (
    ProtocolError,
    evaluate_evidence,
    initialize_project,
    new_iteration,
    project_status,
    refresh_checksums,
    seal_iteration,
    validate_project,
)
from .schema import ITERATION_CLASSES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hygel-parameter-protocol",
        description=(
            "Freeze and verify candidate-to-release bonded-parameter decisions "
            "with E0-E6 gates, checksums, data roles, and a hash-chained ledger."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="also write the command result to this JSON file",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create a non-overwriting project skeleton")
    init.add_argument("project_root", type=Path)
    init.add_argument("--project-id", required=True)
    init.add_argument("--title", required=True)
    init.add_argument("--claim-domain", required=True)

    validate = subparsers.add_parser("validate", help="verify schemas, files, seals, and ledger")
    validate.add_argument("project_root", type=Path)

    checksums = subparsers.add_parser(
        "hash-inputs",
        help="compute checksums for non-placeholder contract inputs",
    )
    checksums.add_argument("project_root", type=Path)
    checksums.add_argument("--iteration")
    checksums.add_argument(
        "--write",
        action="store_true",
        help="write observed checksums into the unsealed contract",
    )

    seal = subparsers.add_parser("seal", help="freeze the active prospective iteration")
    seal.add_argument("project_root", type=Path)
    seal.add_argument("--iteration")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate one evidence record against the next frozen gate",
    )
    evaluate.add_argument("project_root", type=Path)
    evaluate.add_argument("evidence", type=Path)
    evaluate.add_argument(
        "--commit",
        action="store_true",
        help="append the decision to the ledger (default is a read-only preview)",
    )

    status = subparsers.add_parser("status", help="show the current state and next legal gate")
    status.add_argument("project_root", type=Path)

    iteration = subparsers.add_parser(
        "new-iteration",
        help="fork a closed non-pass result without changing its terminal",
    )
    iteration.add_argument("project_root", type=Path)
    iteration.add_argument("--id", required=True, dest="new_iteration_id")
    iteration.add_argument("--from", dest="from_iteration_id")
    iteration.add_argument("--class", required=True, dest="iteration_class", choices=ITERATION_CLASSES)
    iteration.add_argument("--failure-mechanism", required=True)
    return parser


def _emit(payload: Dict[str, Any], output: Optional[Path]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    sys.stdout.write(rendered)
    if output is not None:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result = initialize_project(
                args.project_root,
                project_id=args.project_id,
                title=args.title,
                claim_domain=args.claim_domain,
            )
        elif args.command == "validate":
            result = validate_project(args.project_root)
        elif args.command == "hash-inputs":
            result = refresh_checksums(
                args.project_root,
                iteration_id=args.iteration,
                write=args.write,
            )
        elif args.command == "seal":
            result = seal_iteration(args.project_root, iteration_id=args.iteration)
        elif args.command == "evaluate":
            result = evaluate_evidence(args.project_root, args.evidence, commit=args.commit)
        elif args.command == "status":
            result = project_status(args.project_root)
        elif args.command == "new-iteration":
            result = new_iteration(
                args.project_root,
                new_iteration_id=args.new_iteration_id,
                iteration_class=args.iteration_class,
                failure_mechanism=args.failure_mechanism,
                from_iteration_id=args.from_iteration_id,
            )
        else:  # pragma: no cover - argparse enforces the command set
            parser.error(f"unknown command: {args.command}")
            return 2
    except (ProtocolError, OSError, ValueError, yaml.YAMLError) as error:
        _emit(
            {
                "decision": "ERROR",
                "error_type": type(error).__name__,
                "message": str(error),
            },
            args.json_out,
        )
        return 2
    _emit(result, args.json_out)
    if args.command == "validate" and result.get("decision") != "PASS":
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
