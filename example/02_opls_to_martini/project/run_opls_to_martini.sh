#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"

REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
RUNNER="$REPO_ROOT/hygel_martini/bash_settings/param_opt/run_opls_to_martini.sh"

if [ ! -f "$RUNNER" ]; then
  RUNNER=$(python3 -c "from pathlib import Path; import hygel_martini; print(Path(hygel_martini.__file__).parent / 'bash_settings' / 'param_opt' / 'run_opls_to_martini.sh')" 2>/dev/null || true)
fi

if [ -z "$RUNNER" ] || [ ! -f "$RUNNER" ]; then
  echo "[ERROR] Cannot find hygel_martini bash_settings/param_opt/run_opls_to_martini.sh." >&2
  echo "[ERROR] Install hygel_martini in this Python environment, or run from a source checkout." >&2
  exit 1
fi

exec bash "$RUNNER" "$@"

