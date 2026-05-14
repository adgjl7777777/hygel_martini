#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

STAGE="${STAGE:-fit}"  # fit, postprocess, or both
CONFIG="${CONFIG:-config/opls_existing_data.yaml}"
MODE="${MODE:-setup}"
RUN_ROOT="${RUN_ROOT:-opls_bartender_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-postprocessing_result}"
POSTPROCESS_CONFIG="${POSTPROCESS_CONFIG:-config/postprocess.yaml}"
DRY_RUN="${DRY_RUN:-0}"

labels=(C D S)
modes=(init_only all_unique topology_n0 topology_n1 topology_n2 topology_swap_n0 topology_swap_n1 topology_swap_n2)

usage() {
  cat <<EOF
Usage:
  MODE=setup STAGE=fit bash run_cds_iteration.sh
  MODE=md STAGE=both bash run_cds_iteration.sh
  STAGE=postprocess bash run_cds_iteration.sh

This helper assumes config/opls_existing_data.yaml contains the cases/variants
for the labels and modes you want to process. Missing postprocess inputs are
skipped by default.

Environment:
  STAGE=fit|postprocess|both
  MODE=setup|md|md_notrim|trim|bartender|bartender_notrim|notrim_nobartender|off
  CONFIG=config/opls_existing_data.yaml
  RUN_ROOT=opls_bartender_runs
  OUTPUT_ROOT=postprocessing_result
  POSTPROCESS_CONFIG=config/postprocess.yaml
  DRY_RUN=0
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

run_fit() {
  CONFIG="$CONFIG" MODE="$MODE" OUT_ROOT="$RUN_ROOT" DRY_RUN="$DRY_RUN" \
    bash "$SCRIPT_DIR/run_existing_opls.sh"
}

run_postprocess() {
  local label mode input_root
  for label in "${labels[@]}"; do
    for mode in "${modes[@]}"; do
      input_root="$RUN_ROOT/$label/$mode"
      echo "[postprocess] label=${label} mode=${mode} input_root=${input_root}"
      INPUT_ROOT="$input_root" \
      MIRROR_ROOT="$RUN_ROOT" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      POSTPROCESS_CONFIG="$POSTPROCESS_CONFIG" \
      DRY_RUN="$DRY_RUN" \
        bash "$SCRIPT_DIR/postprocess.sh"
    done
  done
}

case "$STAGE" in
  fit)
    run_fit
    ;;
  postprocess)
    run_postprocess
    ;;
  both)
    run_fit
    run_postprocess
    ;;
  *)
    echo "[ERROR] STAGE must be fit, postprocess, or both." >&2
    usage >&2
    exit 1
    ;;
esac
