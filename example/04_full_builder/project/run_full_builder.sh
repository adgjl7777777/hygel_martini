#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Use central utilities (located via installed hygel_martini package)
LAUNCHER_UTILS_PATH=$(python3 -c "from pathlib import Path; import hygel_martini; print(Path(hygel_martini.__file__).parent / 'bash_settings' / 'launcher_utils.sh')" 2>/dev/null || true)
if [ -z "$LAUNCHER_UTILS_PATH" ] || [ ! -f "$LAUNCHER_UTILS_PATH" ]; then
  echo "[ERROR] Cannot find launcher_utils.sh — is hygel_martini installed? (pip install -e .)" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$LAUNCHER_UTILS_PATH"

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
  bash run_full_builder.sh maker.yaml
  bash run_full_builder.sh maker_anisotropy_x.yaml
  bash run_full_builder.sh --check-gmx
  bash run_full_builder.sh --help

Shell environment:
  run_full_builder.sh sources environment.sh if it exists.
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
require_python_module "hygel_martini.hydrogel_builder"
"$PYTHON_BIN" -m hygel_martini.hydrogel_builder --config "$CONFIG_PATH"
