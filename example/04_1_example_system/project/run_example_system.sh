#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-$SCRIPT_DIR/environment.sh}"

HYGEL_REPO_ROOT="${HYGEL_REPO_ROOT:-}"
ADDITIONAL_BASH_PROFILE="${ADDITIONAL_BASH_PROFILE:-}"
ENV_NAME="${ENV_NAME:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
GMXRC_PATH="${GMXRC_PATH:-}"
GMX_CMD="${GMX_CMD:-}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-}"
GMX_OPENMP_MAX_THREADS="${GMX_OPENMP_MAX_THREADS:-}"

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

GMXRC_PATH="${GMXRC_PATH:-/opt/gromacs/2026/bin/GMXRC}"
GMX_CMD="${GMX_CMD:-gmx_mpi}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
GMX_OPENMP_MAX_THREADS="${GMX_OPENMP_MAX_THREADS:-$OMP_NUM_THREADS}"
export OMP_NUM_THREADS
export GMX_OPENMP_MAX_THREADS

usage() {
  cat <<EOF
Usage:
  bash run_example_system.sh [--check-gmx] [maker.yaml]
  bash run_example_system.sh --help

Default config:
  $SCRIPT_DIR/maker.yaml

Examples:
  bash run_example_system.sh
  bash run_example_system.sh --check-gmx

Shell environment:
  run_example_system.sh sources $SCRIPT_DIR/environment.sh if it exists.
  Override with ENVIRONMENT_FILE=/path/to/environment.sh
  If this project is copied outside the repo, set HYGEL_REPO_ROOT=/path/to/hygel_martini
EOF
}

CHECK_GMX=0
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --check-gmx)
      CHECK_GMX=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

source_optional_script "ADDITIONAL_BASH_PROFILE" "$ADDITIONAL_BASH_PROFILE"
activate_optional_env
if [ -f "$GMXRC_PATH" ]; then
  source_optional_script "GMXRC_PATH" "$GMXRC_PATH"
else
  echo "[WARN] GMXRC_PATH not found: $GMXRC_PATH" >&2
fi

if [ "$CHECK_GMX" -eq 1 ]; then
  "$GMX_CMD" --version
  exit 0
fi

CONFIG_ARG="${1:-maker.yaml}"
if [[ "$CONFIG_ARG" = /* ]]; then
  CONFIG_PATH="$CONFIG_ARG"
else
  CONFIG_PATH="$SCRIPT_DIR/$CONFIG_ARG"
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] Config not found: $CONFIG_PATH" >&2
  exit 1
fi

RUN_DIR="$SCRIPT_DIR"
if [ -n "$REPO_ROOT" ]; then
  RUN_DIR="$REPO_ROOT"
fi
cd "$RUN_DIR"
"$PYTHON_BIN" -m hydrogel_builder --config "$CONFIG_PATH"
