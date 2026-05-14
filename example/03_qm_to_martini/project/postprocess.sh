#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

POSTPROCESS_CONFIG="${POSTPROCESS_CONFIG:-}"
if [ -z "$POSTPROCESS_CONFIG" ]; then
  if [ -f "$SCRIPT_DIR/config_common/postprocess_test.yaml" ]; then
    POSTPROCESS_CONFIG="config_common/postprocess_test.yaml"
  else
    POSTPROCESS_CONFIG="config_common/postprocess.yaml"
  fi
fi

LABEL="${LABEL:-}"
MODE_TAG="${MODE_TAG:-}"
INPUT_ROOT="${INPUT_ROOT:-}"
MIRROR_ROOT="${MIRROR_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
LOG_PATH="${LOG_PATH:-}"
CORE="${CORE:-${CORE_START:-0}}"
USE_SRUN="${USE_SRUN:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_MISSING="${SKIP_MISSING:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
SRUN_CPUS="${SRUN_CPUS:-1}"
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
  LABEL=S MODE_TAG=topology_n0 \\
  INPUT_ROOT=compare_existing_terms/S/topology_n0 \\
  MIRROR_ROOT=compare_existing_terms \\
  OUTPUT_ROOT=postprocessing_result \\
  bash postprocess.sh

Required environment:
  LABEL                 Case label/directory name inside INPUT_ROOT.
  MODE_TAG              Mode tag used for output/log checks.
  INPUT_ROOT            One qm_to_martini output root to postprocess.
  MIRROR_ROOT           Root to mirror from, e.g. compare_existing_terms.
  OUTPUT_ROOT           Root for postprocessed output.

Optional environment:
  POSTPROCESS_CONFIG=auto
  LOG_PATH=""           If set, stdout/stderr from qm_to_martini are written here.
  CORE=0                Logical core for taskset mode.
  USE_SRUN=auto         Empty/auto uses srun inside Slurm, taskset outside.
  SKIP_MISSING=1
  SKIP_EXISTING=0
  DRY_RUN=0
  BOND_CONSTRAINT_MODE=""
  CANDIDATE_SOURCE=""   active uses only uncommented Bartender lines; all also screens commented candidates.
  SHOW_ALL_INFO=""      true keeps commented alternatives in all_terms/plots.
  MULTI_CONSTANT_METRIC=""
  WRITE_PLOTS=""
  FORCE_METRIC_MIN_MODE=""
  FORCE_METRIC_MIN=""   Scalar or JSON dict, e.g. '{"angles":20,"dihedrals":1}'.
  RMSD_MAX=""
  POTENTIAL_ANGLES=""    Preferred angle funct, e.g. 10 or bartender.
  POTENTIAL_DIHEDRALS=""
  POTENTIAL_IMPROPERS=""
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
require_var MODE_TAG
require_var INPUT_ROOT
require_var MIRROR_ROOT
require_var OUTPUT_ROOT

if [[ "$POSTPROCESS_CONFIG" = /* ]]; then
  postprocess_config_path="$POSTPROCESS_CONFIG"
else
  postprocess_config_path="$SCRIPT_DIR/$POSTPROCESS_CONFIG"
fi
if [ ! -f "$postprocess_config_path" ]; then
  echo "[ERROR] POSTPROCESS_CONFIG not found: $POSTPROCESS_CONFIG" >&2
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

result_itp="$INPUT_ROOT/$LABEL/bartender_job/gmx_out.itp"
report_path="$OUTPUT_ROOT/$LABEL/$MODE_TAG/screening_report.json"
if ! truthy "$DRY_RUN" && truthy "$SKIP_MISSING" && [ ! -f "$result_itp" ]; then
  echo "[skip] label=${LABEL} mode=${MODE_TAG} missing=${result_itp}"
  exit 0
fi
if ! truthy "$DRY_RUN" && truthy "$SKIP_EXISTING" && [ -f "$report_path" ]; then
  echo "[skip] label=${LABEL} mode=${MODE_TAG} existing=${report_path}"
  exit 0
fi

cmd=(
  "${launcher[@]}"
  bash "$SCRIPT_DIR/run_qm_to_martini.sh" "$POSTPROCESS_CONFIG"
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

echo "[postprocess] core=${CORE} label=${LABEL} mode=${MODE_TAG} input_root=${INPUT_ROOT}"
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
