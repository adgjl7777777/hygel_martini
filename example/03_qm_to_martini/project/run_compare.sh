#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

LABEL="${LABEL:-}"
BASE_CONFIG="${BASE_CONFIG:-}"
OUT_ROOT="${OUT_ROOT:-}"
MD_TRAJ="${MD_TRAJ:-}"
MODE_TAG="${MODE_TAG:-custom}"
TERM_MODE="${TERM_MODE:-}"
TERM_N="${TERM_N:-0}"
LOG_PATH="${LOG_PATH:-}"
CORE="${CORE:-${CORE_START:-0}}"
USE_SRUN="${USE_SRUN:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
BARTENDER_CPUS="${BARTENDER_CPUS:-1}"
SRUN_CPUS="${SRUN_CPUS:-1}"

usage() {
  cat <<EOF
Usage:
  LABEL=S \\
  BASE_CONFIG=config_common/common.yaml \\
  OUT_ROOT=compare_existing_terms/S/topology_n0 \\
  MD_TRAJ=md_S/S/relax_xtb_geoopt/xtb_traj.pdb \\
  TERM_MODE=topology_n TERM_N=0 MODE_TAG=topology_n0 \\
  bash run_compare.sh

Required environment:
  LABEL                 Case label/directory name inside OUT_ROOT.
  BASE_CONFIG           qm_to_martini base config for this label.
  OUT_ROOT              Output root for this one compare run.
  MD_TRAJ               Existing trajectory path, relative to project base_dir or absolute.
  TERM_MODE             Term-generation mode, e.g. init_only, all_unique, topology_n.

Optional environment:
  TERM_N=0
  MODE_TAG=custom       Human-readable tag used only for logs/messages.
  LOG_PATH=""           If set, stdout/stderr from qm_to_martini are written here.
  CORE=0                Logical core for taskset mode.
  USE_SRUN=auto         Empty/auto uses srun inside Slurm, taskset outside.
  BARTENDER_CPUS=1
  SRUN_CPUS=1
  SKIP_EXISTING=0
  DRY_RUN=0
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

cd "$SCRIPT_DIR"

truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_var() {
  local name="$1"
  local value="${!name:-}"
  if [ -z "$value" ]; then
    echo "[ERROR] $name is required." >&2
    usage >&2
    exit 1
  fi
}

require_var LABEL
require_var BASE_CONFIG
require_var OUT_ROOT
require_var MD_TRAJ
require_var TERM_MODE

if [[ "$BASE_CONFIG" = /* ]]; then
  base_config_path="$BASE_CONFIG"
else
  base_config_path="$SCRIPT_DIR/$BASE_CONFIG"
fi
if [ ! -f "$base_config_path" ]; then
  echo "[ERROR] BASE_CONFIG not found: $BASE_CONFIG" >&2
  exit 1
fi

if [ -z "$USE_SRUN" ] || [ "${USE_SRUN,,}" = "auto" ]; then
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    USE_SRUN=1
  else
    USE_SRUN=0
  fi
fi

use_srun=false
if truthy "$USE_SRUN"; then
  use_srun=true
fi

if "$use_srun"; then
  if ! command -v srun >/dev/null 2>&1; then
    echo "[ERROR] USE_SRUN is enabled but srun was not found" >&2
    exit 1
  fi
  launcher=(srun --exclusive --export=ALL -N 1 -n 1 -c "$SRUN_CPUS")
elif truthy "$DRY_RUN"; then
  launcher=(taskset -c "$CORE")
else
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "[WARN] Inside Slurm but USE_SRUN=false. Using taskset -c <core>." >&2
  fi
  if ! command -v taskset >/dev/null 2>&1; then
    echo "[ERROR] taskset not found" >&2
    exit 1
  fi
  launcher=(taskset -c "$CORE")
fi

result_itp="$OUT_ROOT/$LABEL/bartender_job/gmx_out.itp"
if ! truthy "$DRY_RUN" && truthy "$SKIP_EXISTING" && [ -f "$result_itp" ]; then
  echo "[skip] label=${LABEL} mode=${MODE_TAG} existing=${result_itp}"
  exit 0
fi

cmd=(
  "${launcher[@]}"
  bash "$SCRIPT_DIR/run_qm_to_martini.sh" "$BASE_CONFIG"
  --set "paths.out_root=${OUT_ROOT}"
  --set "bartender_pipeline.relaxation=off"
  --set "bartender_pipeline.md=existing"
  --set "bartender_pipeline.md_traj=${MD_TRAJ}"
  --set "bartender_pipeline.workdir_name=existing_traj_refit"
  --set "bartender_pipeline.term_generation.mode=${TERM_MODE}"
  --set "bartender_pipeline.term_generation.n=${TERM_N}"
  --set "bartender_pipeline.execution.run_relaxation=false"
  --set "bartender_pipeline.execution.run_bartender=true"
  --set "bartender_pipeline.execution.slurm=true"
  --set "bartender_pipeline.execution.use_srun=false"
  --set "bartender_pipeline.execution.shell=bash"
  --set "bartender_pipeline.bartender.cpus=${BARTENDER_CPUS}"
)

echo "[compare] core=${CORE} label=${LABEL} mode=${MODE_TAG} out_root=${OUT_ROOT}"
if truthy "$DRY_RUN"; then
  printf '  '
  printf '%q ' "${cmd[@]}"
  if [ -n "$LOG_PATH" ]; then
    printf '> %q 2>&1\n' "$LOG_PATH"
  else
    printf '\n'
  fi
  exit 0
fi

mkdir -p "$OUT_ROOT"
if [ -n "$LOG_PATH" ]; then
  mkdir -p "$(dirname "$LOG_PATH")"
  "${cmd[@]}" >"$LOG_PATH" 2>&1
else
  "${cmd[@]}"
fi
echo "[done] core=${CORE} label=${LABEL} mode=${MODE_TAG} out_root=${OUT_ROOT}"
