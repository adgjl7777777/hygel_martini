#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CORE_START="${CORE_START:-0}"
USE_SRUN="${USE_SRUN:-}"
if [ -z "$USE_SRUN" ]; then
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    USE_SRUN=1
  else
    USE_SRUN=0
  fi
fi

cd "$SCRIPT_DIR"

use_srun=false
case "${USE_SRUN,,}" in
  1|true|yes|on)
    use_srun=true
    ;;
esac

if "$use_srun"; then
  if ! command -v srun >/dev/null 2>&1; then
    echo "[ERROR] USE_SRUN is enabled but srun was not found" >&2
    exit 1
  fi
  echo "[INFO] USE_SRUN is enabled. Using srun --exclusive to launch jobs." >&2
else
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "[WARN] Inside Slurm but USE_SRUN=false. Using taskset -c <core> which may conflict with Slurm affinity." >&2
  else
    echo "[INFO] Using taskset -c <core> for local parallel launch." >&2
  fi
  if ! command -v taskset >/dev/null 2>&1; then
    echo "[ERROR] taskset not found" >&2
    exit 1
  fi
fi

declare -A BASE_CONFIG=(
  [C]="config_common/common_c.yaml"
  [D]="config_common/common_d.yaml"
  [S]="config_common/common.yaml"
)

declare -A MODE_NAME=(
  [init_only]="init_only"
  [topology_n0]="topology_n"
  [topology_n1]="topology_n"
  [topology_n2]="topology_n"
  [topology_swap_n0]="topology_swap_n"
  [topology_swap_n1]="topology_swap_n"
  [topology_swap_n2]="topology_swap_n"
)

declare -A MODE_N=(
  [init_only]=0
  [topology_n0]=0
  [topology_n1]=1
  [topology_n2]=2
  [topology_swap_n0]=0
  [topology_swap_n1]=1
  [topology_swap_n2]=2
)

labels=(C D S)
yamls=(init_only topology_n0 topology_n1 topology_n2 topology_swap_n0 topology_swap_n1 topology_swap_n2)

mkdir -p "$SCRIPT_DIR/compare_existing_terms/logs"

core="$CORE_START"
for label in "${labels[@]}"; do
  for yaml_tag in "${yamls[@]}"; do
    out_root="$SCRIPT_DIR/compare_existing_terms/${label}/${yaml_tag}"
    log_path="$SCRIPT_DIR/compare_existing_terms/logs/${label}_${yaml_tag}.log"
    mkdir -p "$out_root"
    echo "[launch] core=${core} label=${label} yaml=${yaml_tag} log=${log_path}"
    if "$use_srun"; then
      launcher=(srun --exclusive --export=ALL -N 1 -n 1 -c 1)
    else
      launcher=(taskset -c "$core")
    fi
    "${launcher[@]}" bash "$SCRIPT_DIR/run_qm_to_martini.sh" "${BASE_CONFIG[$label]}" \
      --set "paths.out_root=${out_root}" \
      --set "bartender_pipeline.relaxation=off" \
      --set "bartender_pipeline.md=existing" \
      --set "bartender_pipeline.md_traj=md_${label}/${label}/relax_xtb_geoopt/xtb_traj.pdb" \
      --set "bartender_pipeline.workdir_name=existing_traj_refit" \
      --set "bartender_pipeline.term_generation.mode=${MODE_NAME[$yaml_tag]}" \
      --set "bartender_pipeline.term_generation.n=${MODE_N[$yaml_tag]}" \
      --set "bartender_pipeline.execution.run_relaxation=false" \
      --set "bartender_pipeline.execution.run_bartender=true" \
      --set "bartender_pipeline.execution.slurm=true" \
      --set "bartender_pipeline.execution.use_srun=false" \
      --set "bartender_pipeline.execution.shell=bash" \
      --set "bartender_pipeline.bartender.cpus=1" \
      >"$log_path" 2>&1 &
    core=$((core + 1))
  done
done

echo "[info] launched $(( ${#labels[@]} * ${#yamls[@]} )) jobs starting at core ${CORE_START}"
echo "[info] wait with: wait"
echo "[info] logs under: $SCRIPT_DIR/compare_existing_terms/logs"
