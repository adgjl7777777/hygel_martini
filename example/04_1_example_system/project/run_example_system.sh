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

# GROMACS specific defaults
GMXRC_PATH="${GMXRC_PATH:-/opt/gromacs/2026/bin/GMXRC}"
GMX_CMD="${GMX_CMD:-gmx_mpi}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
GMX_OPENMP_MAX_THREADS="${GMX_OPENMP_MAX_THREADS:-$OMP_NUM_THREADS}"
export OMP_NUM_THREADS
export GMX_OPENMP_MAX_THREADS

if [ -f "$GMXRC_PATH" ]; then
  source_optional_script "GMXRC_PATH" "$GMXRC_PATH"
fi

usage() {
  cat <<EOF
Usage:
  bash run_example_system.sh maker.yaml
  bash run_example_system.sh --check-gmx
  bash run_example_system.sh --help

Shell environment:
  run_example_system.sh sources environment.sh if it exists.
  Override with ENVIRONMENT_FILE or PYTHON_BIN.
  The hygel_martini package must already be installed in that Python environment.
  GROMACS settings: GMXRC_PATH, GMX_CMD, OMP_NUM_THREADS.
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

if [ "$CHECK_GMX" -eq 0 ] && [ $# -eq 0 ]; then
  usage
  exit 0
fi

if [ "$CHECK_GMX" -eq 1 ]; then
  "$GMX_CMD" --version
  exit 0
fi

CONFIG_ARG="$1"
if [[ "$CONFIG_ARG" = /* ]]; then
  CONFIG_PATH="$CONFIG_ARG"
else
  CONFIG_PATH="$SCRIPT_DIR/$CONFIG_ARG"
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] Config not found: $CONFIG_PATH" >&2
  exit 1
fi

cd "$SCRIPT_DIR"
require_python_module "hydrogel_builder"
"$PYTHON_BIN" -m hydrogel_builder --config "$CONFIG_PATH"
