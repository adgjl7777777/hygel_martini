#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Locate launcher_utils.sh: prefer source checkout, fall back to installed package.
REPO_ROOT_LOCAL=$(cd "$SCRIPT_DIR/../../.." && pwd)
LAUNCHER_UTILS_PATH="$REPO_ROOT_LOCAL/hygel_martini/bash_settings/common/launcher_utils.sh"
if [ ! -f "$LAUNCHER_UTILS_PATH" ]; then
  LAUNCHER_UTILS_PATH=$(python3 -c "from pathlib import Path; import hygel_martini; print(Path(hygel_martini.__file__).parent / 'bash_settings' / 'common' / 'launcher_utils.sh')" 2>/dev/null || true)
fi
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
# OMP_NUM_THREADS is NOT set here — it is controlled by omp_threads in simulation.yaml.
# geo_opt.py sets OMP_NUM_THREADS in the mdrun subprocess environment to match -ntomp.

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
  GROMACS settings: GMXRC_PATH, GMX_CMD.
  Thread count: set omp_threads in simulation.yaml (null=GROMACS auto-detect).
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
