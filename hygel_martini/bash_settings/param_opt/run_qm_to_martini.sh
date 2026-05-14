#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
LAUNCHER_UTILS_PATH="$REPO_ROOT/hygel_martini/bash_settings/common/launcher_utils.sh"
if [ -f "$LAUNCHER_UTILS_PATH" ]; then
  # shellcheck disable=SC1090
  source "$LAUNCHER_UTILS_PATH"
else
  echo "[ERROR] launcher_utils.sh not found at $LAUNCHER_UTILS_PATH" >&2
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd)
setup_hygel_env "$PROJECT_DIR"

usage() {
  cat <<EOF
Usage:
  PROJECT_DIR=/path/to/project bash hygel_martini/bash_settings/param_opt/run_qm_to_martini.sh config_common/common.yaml [options...]
  bash hygel_martini/bash_settings/param_opt/run_qm_to_martini.sh /abs/path/config.yaml [options...]
  bash hygel_martini/bash_settings/param_opt/run_qm_to_martini.sh --workflow-help

Environment:
  PROJECT_DIR       Base directory for relative config paths. Default: current directory.
  ENVIRONMENT_FILE  Optional environment script. Default: <PROJECT_DIR>/environment.sh when present.
  PYTHON_BIN        Python executable. Default: python3.
EOF
}

ORIGINAL_ARGC=$#
WORKFLOW_HELP=0
CHECK_XTB=0
CHECK_BARTENDER=0
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
  elif [ -f "$PROJECT_DIR/$CONFIG_ARG" ]; then
    CONFIG_PATH="$PROJECT_DIR/$CONFIG_ARG"
  else
    CONFIG_PATH="$(pwd)/$CONFIG_ARG"
  fi
fi

if [ "$WORKFLOW_HELP" -eq 0 ] && [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] Config not found: $CONFIG_PATH" >&2
  exit 1
fi

cd "$PROJECT_DIR"

if [ "$WORKFLOW_HELP" -eq 1 ]; then
  require_python_module "hygel_martini.param_opt.qm_to_martini"
  "$PYTHON_BIN" -m hygel_martini.param_opt.qm_to_martini --help
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
  require_python_module "hygel_martini.param_opt.qm_to_martini"
  "$PYTHON_BIN" -m hygel_martini.param_opt.qm_to_martini --config "$CONFIG_PATH" --check-tools "${CHECK_ARGS[@]}" "${PASSTHRU_ARGS[@]}" "$@"
  exit 0
fi

require_python_module "hygel_martini.param_opt.qm_to_martini"
"$PYTHON_BIN" -m hygel_martini.param_opt.qm_to_martini --config "$CONFIG_PATH" "${PASSTHRU_ARGS[@]}" "$@"
