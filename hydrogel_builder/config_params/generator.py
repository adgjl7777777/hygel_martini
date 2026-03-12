"""Top-level entry helpers for hydrogel construction runs.

This module is intentionally thin: it loads the maker configuration, normalizes
input line endings, and then hands control to ``execute_mode``.
"""

import os
import sys

from hydrogel_builder.config_params.config import Config
from hydrogel_builder.config_params.read_json import execute_mode
from hydrogel_builder.core_utils.common.utility import run_dos2unix_on_inputs

def run_hydrogel_example(config_path):
    """Run a full hydrogel-generation job from a maker file."""
    print(f"\n--- 하이드로젤 생성 예시 실행 중 ({os.path.basename(config_path)}) ---")

    # Load the full maker configuration, including YAML includes.
    Config.load_config(config_path)
    
    # Normalize line endings before downstream parsers read structure templates.
    run_dos2unix_on_inputs(Config._data)

    # Delegate the actual workflow to the configured execution mode.
    execute_mode()

    print("\n--- 하이드로젤 생성 예시 완료 ---")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m hydrogel_builder.config_params.generator <maker.yaml>")
    config_to_run = sys.argv[1]
    run_hydrogel_example(config_to_run)
