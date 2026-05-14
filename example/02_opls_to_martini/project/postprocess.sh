#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

POSTPROCESS_CONFIG="${POSTPROCESS_CONFIG:-config/postprocess.yaml}"
INPUT_ROOT="${INPUT_ROOT:-}"
MIRROR_ROOT="${MIRROR_ROOT:-opls_bartender_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-postprocessing_result}"
LOG_PATH="${LOG_PATH:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_MISSING="${SKIP_MISSING:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
BOND_CONSTRAINT_MODE="${BOND_CONSTRAINT_MODE:-}"
CANDIDATE_SOURCE="${CANDIDATE_SOURCE:-}"
SHOW_ALL_INFO="${SHOW_ALL_INFO:-}"
MULTI_CONSTANT_METRIC="${MULTI_CONSTANT_METRIC:-}"
WRITE_PLOTS="${WRITE_PLOTS:-}"
FORCE_METRIC_MIN_MODE="${FORCE_METRIC_MIN_MODE:-}"
FORCE_METRIC_MIN="${FORCE_METRIC_MIN:-}"
RMSD_MAX="${RMSD_MAX:-}"
POTENTIAL_ANGLES="${POTENTIAL_ANGLES:-}"
POTENTIAL_DIHEDRALS="${POTENTIAL_DIHEDRALS:-}"
POTENTIAL_IMPROPERS="${POTENTIAL_IMPROPERS:-}"

usage() {
  cat <<EOF
Usage:
  INPUT_ROOT=opls_bartender_runs/S/topology_n0 \\
  MIRROR_ROOT=opls_bartender_runs \\
  OUTPUT_ROOT=postprocessing_result \\
  bash postprocess.sh

Optional environment:
  POSTPROCESS_CONFIG=config/postprocess.yaml
  LOG_PATH=""
  DRY_RUN=0
  SKIP_MISSING=1
  SKIP_EXISTING=0
  BOND_CONSTRAINT_MODE=""
  CANDIDATE_SOURCE=""
  SHOW_ALL_INFO=""
  MULTI_CONSTANT_METRIC=""
  WRITE_PLOTS=""
  FORCE_METRIC_MIN_MODE=""
  FORCE_METRIC_MIN=""
  RMSD_MAX=""
  POTENTIAL_ANGLES=""
  POTENTIAL_DIHEDRALS=""
  POTENTIAL_IMPROPERS=""
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [ -z "$INPUT_ROOT" ]; then
  echo "[ERROR] INPUT_ROOT is required." >&2
  usage >&2
  exit 1
fi

if [ ! -f "$POSTPROCESS_CONFIG" ]; then
  echo "[ERROR] POSTPROCESS_CONFIG not found: $POSTPROCESS_CONFIG" >&2
  exit 1
fi

if ! truthy "$DRY_RUN" && truthy "$SKIP_MISSING" && ! find "$INPUT_ROOT" -path '*/bartender_job/gmx_out.itp' -type f -print -quit 2>/dev/null | grep -q .; then
  echo "[skip] missing gmx_out.itp under ${INPUT_ROOT}"
  exit 0
fi

report_path="$OUTPUT_ROOT/postprocess_summary.json"
if ! truthy "$DRY_RUN" && truthy "$SKIP_EXISTING" && [ -f "$report_path" ]; then
  echo "[skip] existing=${report_path}"
  exit 0
fi

cmd=(
  bash "$SCRIPT_DIR/run_opls_to_martini.sh" "$POSTPROCESS_CONFIG"
  --postprocess-only
  --set "paths.out_root=${INPUT_ROOT}"
  --set "paths.postprocess_mirror_root=${MIRROR_ROOT}"
  --set "paths.postprocess_output_root=${OUTPUT_ROOT}"
  --set "bartender_pipeline.postprocess.screening.enabled=true"
)

append_set() {
  local key="$1"
  local value="$2"
  if [ -n "$value" ]; then
    cmd+=(--set "${key}=${value}")
  fi
}

append_set "bartender_pipeline.postprocess.screening.bond_constraint_mode" "$BOND_CONSTRAINT_MODE"
append_set "bartender_pipeline.postprocess.screening.candidate_source" "$CANDIDATE_SOURCE"
append_set "bartender_pipeline.postprocess.screening.show_all_info" "$SHOW_ALL_INFO"
append_set "bartender_pipeline.postprocess.screening.multi_constant_metric" "$MULTI_CONSTANT_METRIC"
append_set "bartender_pipeline.postprocess.screening.write_plots" "$WRITE_PLOTS"
append_set "bartender_pipeline.postprocess.screening.thresholds.force_metric_min_mode" "$FORCE_METRIC_MIN_MODE"
append_set "bartender_pipeline.postprocess.screening.thresholds.force_metric_min" "$FORCE_METRIC_MIN"
append_set "bartender_pipeline.postprocess.screening.thresholds.rmsd_max" "$RMSD_MAX"
append_set "bartender_pipeline.postprocess.screening.potentials.angles" "$POTENTIAL_ANGLES"
append_set "bartender_pipeline.postprocess.screening.potentials.dihedrals" "$POTENTIAL_DIHEDRALS"
append_set "bartender_pipeline.postprocess.screening.potentials.impropers" "$POTENTIAL_IMPROPERS"

echo "[opls-postprocess] input_root=${INPUT_ROOT}"
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

if [ -n "$LOG_PATH" ]; then
  mkdir -p "$(dirname "$LOG_PATH")"
  "${cmd[@]}" >"$LOG_PATH" 2>&1
else
  "${cmd[@]}"
fi
