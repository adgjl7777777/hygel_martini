#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONDA_PROFILE="${CONDA_PROFILE:-/nas_3/active/transcendence/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hygel}"
GMXRC_PATH="${GMXRC_PATH:-/opt/gromacs/2026/bin/GMXRC}"
GMX_CMD="${GMX_CMD:-gmx_mpi}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
GMX_OPENMP_MAX_THREADS="${GMX_OPENMP_MAX_THREADS:-$OMP_NUM_THREADS}"
export OMP_NUM_THREADS
export GMX_OPENMP_MAX_THREADS

usage() {
  cat <<EOF
Usage:
  bash hygel_run.sh [--check-gmx] [maker_soft_em.yaml|maker_soft_md.yaml]
  bash hygel_run.sh --workflow-help
  bash hygel_run.sh --help

Default config:
  $SCRIPT_DIR/maker_soft_em.yaml

Examples:
  bash hygel_run.sh
  bash hygel_run.sh maker_soft_md.yaml
  bash hygel_run.sh --check-gmx

Environment overrides:
  CONDA_PROFILE=$CONDA_PROFILE
  CONDA_ENV_NAME=$CONDA_ENV_NAME
  GMXRC_PATH=$GMXRC_PATH
  GMX_CMD=$GMX_CMD
  OMP_NUM_THREADS=$OMP_NUM_THREADS
  GMX_OPENMP_MAX_THREADS=$GMX_OPENMP_MAX_THREADS
EOF
}

CHECK_GMX=0
WORKFLOW_HELP=0
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
    --workflow-help)
      WORKFLOW_HELP=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [ ! -f "$CONDA_PROFILE" ]; then
  echo "[ERROR] CONDA_PROFILE not found: $CONDA_PROFILE" >&2
  exit 1
fi

source "$CONDA_PROFILE"
if [ -f "$GMXRC_PATH" ]; then
  set +u
  # shellcheck disable=SC1090
  source "$GMXRC_PATH"
  set -u
else
  echo "[WARN] GMXRC_PATH not found: $GMXRC_PATH" >&2
fi
conda activate "$CONDA_ENV_NAME"

if [ "$CHECK_GMX" -eq 1 ]; then
  "$GMX_CMD" --version
  exit 0
fi

if [ "$WORKFLOW_HELP" -eq 1 ]; then
  cd "$REPO_ROOT"
  python3 -m hydrogel_builder.relax --help
  exit 0
fi

CONFIG_ARG="${1:-maker_soft_em.yaml}"
if [[ "$CONFIG_ARG" = /* ]]; then
  CONFIG_PATH="$CONFIG_ARG"
else
  CONFIG_PATH="$SCRIPT_DIR/$CONFIG_ARG"
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] Config not found: $CONFIG_PATH" >&2
  exit 1
fi

cd "$REPO_ROOT"
python3 -m hydrogel_builder.relax --config "$CONFIG_PATH"
