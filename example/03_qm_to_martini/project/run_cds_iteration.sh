#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Edit this file directly for the current iteration set.
STAGE="${STAGE:-compare}"  # compare, postprocess, or both
CORE_START="${CORE_START:-0}"
CORE_COUNT="${CORE_COUNT:-24}"
MAX_JOBS="${MAX_JOBS:-24}"
DRY_RUN="${DRY_RUN:-0}"
USE_SRUN="${USE_SRUN:-}"

COMPARE_ROOT="${COMPARE_ROOT:-$SCRIPT_DIR/compare_existing_terms}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/postprocessing_result}"
POSTPROCESS_CONFIG="${POSTPROCESS_CONFIG:-config_common/postprocess.yaml}"
# Default to the raw xTB trajectory so existing .trj files can be auto-trimmed
# before Bartender refit. For a pre-converted trajectory, use:
#   TRAJ_REL=relax_xtb_geoopt/xtb_traj.pdb bash run_cds_iteration.sh
TRAJ_REL="${TRAJ_REL:-relax_xtb_geoopt/xtb.trj}"

labels=(C D S)
modes=(init_only all_unique topology_n0 topology_n1 topology_n2 topology_swap_n0 topology_swap_n1 topology_swap_n2)

base_config() {
  case "$1" in
    C) echo "config_common/common_c.yaml" ;;
    D) echo "config_common/common_d.yaml" ;;
    S) echo "config_common/common.yaml" ;;
    *) echo "[ERROR] Unknown label '$1'. Edit base_config() in run_cds_iteration.sh." >&2; return 1 ;;
  esac
}

term_mode() {
  case "$1" in
    init_only) echo "init_only 0" ;;
    all_unique) echo "all_unique 0" ;;
    topology_n0) echo "topology_n 0" ;;
    topology_n1) echo "topology_n 1" ;;
    topology_n2) echo "topology_n 2" ;;
    topology_swap_n0) echo "topology_swap_n 0" ;;
    topology_swap_n1) echo "topology_swap_n 1" ;;
    topology_swap_n2) echo "topology_swap_n 2" ;;
    *) echo "[ERROR] Unknown mode '$1'. Edit term_mode() in run_cds_iteration.sh." >&2; return 1 ;;
  esac
}

launched=0
running=0

launch_job() {
  local core
  core=$(( CORE_START + (launched % CORE_COUNT) ))
  launched=$(( launched + 1 ))
  running=$(( running + 1 ))
  CORE="$core" "$@" &
  if [ "$running" -ge "$MAX_JOBS" ]; then
    wait -n
    running=$(( running - 1 ))
  fi
}

run_compare_loop() {
  local label mode config out_root log_path md_traj mode_name mode_n
  for label in "${labels[@]}"; do
    config=$(base_config "$label")
    for mode in "${modes[@]}"; do
      read -r mode_name mode_n <<<"$(term_mode "$mode")"
      out_root="$COMPARE_ROOT/$label/$mode"
      log_path="$COMPARE_ROOT/logs/${label}_${mode}.log"
      md_traj="md_${label}/${label}/${TRAJ_REL}"
      mkdir -p "$(dirname "$log_path")"

      echo "[compare] core=$(( CORE_START + (launched % CORE_COUNT) )) label=${label} mode=${mode}"
      launch_job env \
        LABEL="$label" BASE_CONFIG="$config" OUT_ROOT="$out_root" MD_TRAJ="$md_traj" \
        MODE_TAG="$mode" TERM_MODE="$mode_name" TERM_N="$mode_n" \
        LOG_PATH="$log_path" USE_SRUN="$USE_SRUN" DRY_RUN="$DRY_RUN" \
        bash "$SCRIPT_DIR/run_compare.sh"
    done
  done
}

run_postprocess_loop() {
  local label mode input_root log_path
  for label in "${labels[@]}"; do
    for mode in "${modes[@]}"; do
      input_root="$COMPARE_ROOT/$label/$mode"
      log_path="$OUTPUT_ROOT/logs/${label}_${mode}_postprocess.log"
      mkdir -p "$(dirname "$log_path")"

      echo "[postprocess] core=$(( CORE_START + (launched % CORE_COUNT) )) label=${label} mode=${mode}"
      launch_job env \
        LABEL="$label" MODE_TAG="$mode" INPUT_ROOT="$input_root" \
        MIRROR_ROOT="$COMPARE_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
        POSTPROCESS_CONFIG="$POSTPROCESS_CONFIG" LOG_PATH="$log_path" \
        USE_SRUN="$USE_SRUN" DRY_RUN="$DRY_RUN" \
        bash "$SCRIPT_DIR/postprocess.sh"
    done
  done
}

case "$STAGE" in
  compare)
    run_compare_loop
    ;;
  postprocess)
    run_postprocess_loop
    ;;
  both)
    run_compare_loop
    wait
    running=0
    echo "[done] compare finished, starting postprocess"
    run_postprocess_loop
    ;;
  *)
    echo "[ERROR] STAGE must be compare, postprocess, or both." >&2
    exit 1
    ;;
esac

wait
echo "[done] stage=${STAGE} launched=${launched}"
