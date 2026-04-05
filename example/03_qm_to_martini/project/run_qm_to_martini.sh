#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Use central utilities
REPO_ROOT_LOCAL=$(cd "$SCRIPT_DIR/../../.." && pwd)
LAUNCHER_UTILS_PATH="$REPO_ROOT_LOCAL/launcher_utils.sh"
if [ -f "$LAUNCHER_UTILS_PATH" ]; then
  # shellcheck disable=SC1090
  source "$LAUNCHER_UTILS_PATH"
else
  echo "[ERROR] launcher_utils.sh not found at $LAUNCHER_UTILS_PATH" >&2
  echo "[ERROR] Run this launcher from a hygel_martini git clone after installing the package." >&2
  exit 1
fi

# Standard environment setup
setup_hygel_env "$SCRIPT_DIR"

usage() {
  cat <<EOF
Usage:
  bash run_qm_to_martini.sh config_common/common.yaml [qm_to_martini options...]
  bash run_qm_to_martini.sh config_common/postprocess.yaml --postprocess-only
  bash run_qm_to_martini.sh --check-xtb --check-bartender config_common/common.yaml
  bash run_qm_to_martini.sh --workflow-help
  bash run_qm_to_martini.sh --help

Shell environment:
  run_qm_to_martini.sh sources environment.sh if it exists.
  Override with ENVIRONMENT_FILE or PYTHON_BIN.
  The hygel_martini package must already be installed in that Python environment.
EOF
}

ORIGINAL_ARGC=$#
WORKFLOW_HELP=0
CHECK_XTB=0
CHECK_BARTENDER=0
POSTPROCESS_ONLY=0
PASSTHRU_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --workflow-help)
      WORKFLOW_HELP=1
      shift
      ;;
    --check-xtb)
      CHECK_XTB=1
      shift
      ;;
    --check-bartender)
      CHECK_BARTENDER=1
      shift
      ;;
    --postprocess-only)
      POSTPROCESS_ONLY=1
      PASSTHRU_ARGS+=("$1")
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

CONFIG_ARG=""
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
  CONFIG_ARG="$1"
  shift
fi

CONFIG_PATH=""
if [ "$WORKFLOW_HELP" -eq 0 ] && [ -z "$CONFIG_ARG" ]; then
  if [ "$ORIGINAL_ARGC" -eq 0 ]; then
    usage
    exit 0
  fi
  echo "[ERROR] Config path is required." >&2
  usage >&2
  exit 1
fi

if [ -n "$CONFIG_ARG" ]; then
  if [[ "$CONFIG_ARG" = /* ]]; then
    CONFIG_PATH="$CONFIG_ARG"
  else
    CONFIG_PATH="$SCRIPT_DIR/$CONFIG_ARG"
  fi
fi

if [ "$WORKFLOW_HELP" -eq 0 ] && [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] Config not found: $CONFIG_PATH" >&2
  exit 1
fi

cd "$SCRIPT_DIR"

if [ "$WORKFLOW_HELP" -eq 1 ]; then
  require_python_module "param_opt.qm_to_martini"
  "$PYTHON_BIN" -m param_opt.qm_to_martini --help
  exit 0
fi

CHECK_ARGS=()
if [ "$CHECK_XTB" -eq 1 ]; then
  CHECK_ARGS+=(xtb)
fi
if [ "$CHECK_BARTENDER" -eq 1 ]; then
  CHECK_ARGS+=(bartender)
fi
if [ "${#CHECK_ARGS[@]}" -gt 0 ]; then
  require_python_module "param_opt.qm_to_martini"
  "$PYTHON_BIN" -m param_opt.qm_to_martini --config "$CONFIG_PATH" --check-tools "${CHECK_ARGS[@]}" "${PASSTHRU_ARGS[@]}" "$@"
  exit 0
fi

require_python_module "param_opt.qm_to_martini"
"$PYTHON_BIN" -m param_opt.qm_to_martini --config "$CONFIG_PATH" "${PASSTHRU_ARGS[@]}" "$@"
