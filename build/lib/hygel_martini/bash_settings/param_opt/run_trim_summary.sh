#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
LAUNCHER_UTILS_PATH="$REPO_ROOT/hygel_martini/bash_settings/common/launcher_utils.sh"
if [ ! -f "$LAUNCHER_UTILS_PATH" ]; then
  echo "[ERROR] launcher_utils.sh not found at $LAUNCHER_UTILS_PATH" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$LAUNCHER_UTILS_PATH"
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
setup_hygel_env "$PROJECT_DIR"
require_python_module "hygel_martini.param_opt.qm_to_martini"
COMPARE_ROOT="${COMPARE_ROOT:-$PROJECT_DIR/parameter_determination/compare_trim_threshold_samet}"
OUT_DIR="${OUT_DIR:-$COMPARE_ROOT/trim_summary}"
"$PYTHON_BIN" -m hygel_martini.param_opt.qm_to_martini.analysis.trim_summary "$COMPARE_ROOT" --out-dir "$OUT_DIR" "$@"
