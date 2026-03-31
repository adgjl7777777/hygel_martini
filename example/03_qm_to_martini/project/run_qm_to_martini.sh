#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-$SCRIPT_DIR/environment.sh}"

HYGEL_REPO_ROOT="${HYGEL_REPO_ROOT:-}"
ADDITIONAL_BASH_PROFILE="${ADDITIONAL_BASH_PROFILE:-}"
ENV_NAME="${ENV_NAME:-}"
PYTHON_BIN="${PYTHON_BIN:-}"

source_optional_script() {
  local label="$1"
  local path="$2"
  if [ -z "$path" ]; then
    return 0
  fi
  if [ ! -f "$path" ]; then
    echo "[ERROR] $label not found: $path" >&2
    exit 1
  fi
  set +u
  # shellcheck disable=SC1090
  source "$path"
  set -u
}

activate_optional_env() {
  if [ -z "$ENV_NAME" ]; then
    return 0
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "[WARN] ENV_NAME is set but 'conda' is not available. Using the current shell environment." >&2
    return 0
  fi
  set +e
  conda activate "$ENV_NAME"
  local status=$?
  set -e
  if [ "$status" -ne 0 ]; then
    echo "[WARN] Failed to activate conda environment '$ENV_NAME'. Using the current shell environment." >&2
  fi
}

find_repo_root() {
  local current="$1"
  while [ -n "$current" ] && [ "$current" != "/" ]; do
    if [ -d "$current/param_opt" ] || [ -d "$current/hydrogel_builder" ]; then
      printf '%s\n' "$current"
      return 0
    fi
    current=$(dirname "$current")
  done
  if [ -d "/param_opt" ] || [ -d "/hydrogel_builder" ]; then
    printf '/\n'
    return 0
  fi
  return 1
}

if [ -f "$ENVIRONMENT_FILE" ]; then
  source_optional_script "ENVIRONMENT_FILE" "$ENVIRONMENT_FILE"
fi

if [ -n "$HYGEL_REPO_ROOT" ]; then
  REPO_ROOT="$HYGEL_REPO_ROOT"
elif REPO_ROOT=$(find_repo_root "$SCRIPT_DIR"); then
  :
else
  REPO_ROOT=""
fi

if [ -n "$REPO_ROOT" ]; then
  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

usage() {
  cat <<EOF
Usage:
  bash run_qm_to_martini.sh [config_common/common.yaml] [qm_to_martini options...]
  bash run_qm_to_martini.sh --help
  bash run_qm_to_martini.sh --workflow-help
  bash run_qm_to_martini.sh --check-xtb
  bash run_qm_to_martini.sh --check-bartender
  bash run_qm_to_martini.sh --postprocess-only [config_common/postprocess.yaml]

Default config:
  $SCRIPT_DIR/config_common/common.yaml

Examples:
  bash run_qm_to_martini.sh
  bash run_qm_to_martini.sh config_common/common.yaml
  bash run_qm_to_martini.sh --postprocess-only
  bash run_qm_to_martini.sh config_common/postprocess.yaml --postprocess-only
  bash run_qm_to_martini.sh --check-xtb --check-bartender
  bash run_qm_to_martini.sh --set bartender_pipeline.relaxation=orca
  bash run_qm_to_martini.sh --set 'system.sequences=[S,D,D,S]' --set paths.out_root=/tmp/qm_to_martini_test

Shell environment:
  run_qm_to_martini.sh sources $SCRIPT_DIR/environment.sh if it exists.
  Override with ENVIRONMENT_FILE=/path/to/environment.sh
  If this project is copied outside the repo, set HYGEL_REPO_ROOT=/path/to/hygel_martini
EOF
}

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

CONFIG_ARG="config_common/common.yaml"
if [ "$POSTPROCESS_ONLY" -eq 1 ]; then
  CONFIG_ARG="config_common/postprocess.yaml"
fi
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
  CONFIG_ARG="$1"
  shift
fi

if [[ "$CONFIG_ARG" = /* ]]; then
  CONFIG_PATH="$CONFIG_ARG"
else
  CONFIG_PATH="$SCRIPT_DIR/$CONFIG_ARG"
fi

if [ "$WORKFLOW_HELP" -eq 0 ] && [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] Config not found: $CONFIG_PATH" >&2
  exit 1
fi

source_optional_script "ADDITIONAL_BASH_PROFILE" "$ADDITIONAL_BASH_PROFILE"
activate_optional_env

RUN_DIR="$SCRIPT_DIR"
if [ -n "$REPO_ROOT" ]; then
  RUN_DIR="$REPO_ROOT"
fi
cd "$RUN_DIR"

if [ "$WORKFLOW_HELP" -eq 1 ]; then
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
  "$PYTHON_BIN" -m param_opt.qm_to_martini --config "$CONFIG_PATH" --check-tools "${CHECK_ARGS[@]}" "${PASSTHRU_ARGS[@]}" "$@"
  exit 0
fi

"$PYTHON_BIN" -m param_opt.qm_to_martini --config "$CONFIG_PATH" "${PASSTHRU_ARGS[@]}" "$@"
