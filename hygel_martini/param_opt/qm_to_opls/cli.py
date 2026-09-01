"""Command-line entry point for the stage-01 (QM -> OPLS) workflow.

Parses the shared ``--config``/``--dump-default-config`` arguments (added
by ``hygel_martini.core.config.add_config_args``) and either writes the
package default config as JSON or runs :func:`run_qm_to_opls` on the
given config file.  Called by ``__main__`` and by the console script
wiring in the package ``__init__``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hygel_martini.core.config import add_config_args
from .defaults import DEFAULT_CONFIG
from .generator import run_qm_to_opls


def main() -> None:
    """Run the qm_to_opls CLI.

    With ``--dump-default-config`` the default config is written to the
    ``--config`` path as JSON and the program returns without running the
    workflow.  Otherwise the config file is loaded and ORCA preparation
    inputs are generated under the configured output root.
    """
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
