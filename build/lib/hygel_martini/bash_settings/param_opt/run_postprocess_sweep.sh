#!/usr/bin/env bash
# Dispatch postprocess.sh across a (sweep config × targets) matrix.
#
# Each row in the matrix is launched as a child bash invocation of
# $PROJECT_DIR/postprocess.sh with the variant context exported as env vars.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ANALYZE_DIR=/path/to/analyze bash run_postprocess_sweep.sh configs/01_summary_sweep.yaml
  bash run_postprocess_sweep.sh /abs/path/to/sweep.yaml
  bash run_postprocess_sweep.sh -h

ANALYZE_DIR is required and is auto-detected when:
  - the first argument is a configs/*.yaml under an analyze directory, or
  - the script is invoked from a directory that contains configs/.

Useful environment:
  DRY_RUN=1              MAX_JOBS=24
  CORE_START=0           CORE_COUNT=24
  SKIP_EXISTING=1        SKIP_MISSING=1
  USE_SRUN=1             TARGETS_CONFIG=configs/00_targets.yaml
  ONLY_VARIANT=P1_a2_d3_i1_R7p5_F2_medium
  ONLY_LABEL=S           ONLY_MODE=topology_n1
  VARIANT_LIMIT=3
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LAUNCHER_UTILS_PATH="$SCRIPT_DIR/../common/launcher_utils.sh"
if [ ! -f "$LAUNCHER_UTILS_PATH" ]; then
  LAUNCHER_UTILS_PATH=$(python3 -c "from pathlib import Path; import hygel_martini; print(Path(hygel_martini.__file__).parent / 'bash_settings' / 'common' / 'launcher_utils.sh')" 2>/dev/null || true)
fi
if [ -z "$LAUNCHER_UTILS_PATH" ] || [ ! -f "$LAUNCHER_UTILS_PATH" ]; then
  echo "[ERROR] Cannot find launcher_utils.sh — is hygel_martini installed? (pip install -e .)" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$LAUNCHER_UTILS_PATH"

# Resolve ANALYZE_DIR (analyze/<group>/<dir>) and PROJECT_DIR (project root).
if [ -z "${ANALYZE_DIR:-}" ] && [ "$#" -gt 0 ]; then
  case "$1" in
    */configs/*.yaml) ANALYZE_DIR=$(cd "$(dirname "$1")/.." && pwd) ;;
    configs/*.yaml)   ANALYZE_DIR=$(pwd) ;;
  esac
fi
if [ -z "${ANALYZE_DIR:-}" ] && [ -d "$(pwd)/configs" ]; then
  ANALYZE_DIR=$(pwd)
fi
if [ -z "${ANALYZE_DIR:-}" ]; then
  echo "[ERROR] ANALYZE_DIR is required." >&2
  echo "        Run from an analyze directory, pass configs/<file>.yaml, or set ANALYZE_DIR=/path." >&2
  exit 1
fi
ANALYZE_DIR=$(cd "$ANALYZE_DIR" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd "$ANALYZE_DIR/../.." && pwd)}
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd)

setup_hygel_env "$PROJECT_DIR"

# Resolve sweep + targets configs (positional arg / env / convention).
if [ "$#" -gt 0 ]; then
  if [[ "$1" = /* ]]; then
    SWEEP_CONFIG="$1"
  elif [ -f "$PROJECT_DIR/$1" ]; then
    SWEEP_CONFIG="$PROJECT_DIR/$1"
  elif [ -f "$ANALYZE_DIR/$1" ]; then
    SWEEP_CONFIG="$ANALYZE_DIR/$1"
  else
    SWEEP_CONFIG="$1"
  fi
else
  SWEEP_CONFIG="$ANALYZE_DIR/configs/01_summary_sweep.yaml"
fi

TARGETS_CONFIG="${TARGETS_CONFIG:-$ANALYZE_DIR/configs/00_targets.yaml}"
if [[ "$TARGETS_CONFIG" != /* ]]; then
  if [ -f "$PROJECT_DIR/$TARGETS_CONFIG" ]; then
    TARGETS_CONFIG="$PROJECT_DIR/$TARGETS_CONFIG"
  elif [ -f "$ANALYZE_DIR/$TARGETS_CONFIG" ]; then
    TARGETS_CONFIG="$ANALYZE_DIR/$TARGETS_CONFIG"
  fi
fi

# Run knobs.
DRY_RUN="${DRY_RUN:-0}"
MAX_JOBS="${MAX_JOBS:-1}"
CORE_START="${CORE_START:-0}"
CORE_COUNT="${CORE_COUNT:-$MAX_JOBS}"
USE_SRUN="${USE_SRUN:-}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
SKIP_MISSING="${SKIP_MISSING:-1}"
ONLY_VARIANT="${ONLY_VARIANT:-}"
ONLY_LABEL="${ONLY_LABEL:-}"
ONLY_MODE="${ONLY_MODE:-}"
VARIANT_LIMIT="${VARIANT_LIMIT:-0}"

if [ "$MAX_JOBS" -lt 1 ]; then
  echo "[ERROR] MAX_JOBS must be >= 1 (got '$MAX_JOBS')" >&2
  exit 1
fi
if [ "$CORE_COUNT" -lt 1 ]; then
  echo "[ERROR] CORE_COUNT must be >= 1 (got '$CORE_COUNT')" >&2
  exit 1
fi

cd "$PROJECT_DIR"

# When the user Ctrl-Cs the launcher, kill outstanding postprocess workers so
# they don't keep eating CPU after we exit. Without this, MAX_JOBS workers can
# linger in the background.
cleanup() {
  trap - SIGINT SIGTERM EXIT
  local pids
  pids=$(jobs -pr 2>/dev/null || true)
  if [ -n "$pids" ]; then
    kill $pids 2>/dev/null || true
  fi
}
trap cleanup SIGINT SIGTERM EXIT

task_rows() {
  "$PYTHON_BIN" - "$SWEEP_CONFIG" "$TARGETS_CONFIG" "$ANALYZE_DIR" "$PROJECT_DIR" \
    "$ONLY_VARIANT" "$ONLY_LABEL" "$ONLY_MODE" "$VARIANT_LIMIT" <<'PY'
import itertools
import json
import os
import sys

import yaml

sweep_path, targets_path, analyze_dir, project_dir = sys.argv[1:5]
only_variant, only_label, only_mode, variant_limit_raw = sys.argv[5:9]
variant_limit = int(variant_limit_raw or "0")

with open(sweep_path, "r", encoding="utf-8") as handle:
    sweep = yaml.safe_load(handle) or {}
with open(targets_path, "r", encoding="utf-8") as handle:
    targets = yaml.safe_load(handle) or {}

def abs_from(base, value):
    value = str(value)
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(base, value))

def fmt_value(value):
    try:
        number = float(value)
    except Exception:
        return str(value).replace(".", "p").replace("-", "m")
    if number.is_integer():
        return str(int(number))
    return str(value).rstrip("0").rstrip(".").replace(".", "p").replace("-", "m")

def bool_text(value):
    return "true" if bool(value) else "false"

target_base = os.path.dirname(os.path.abspath(targets_path))
compare_root = abs_from(target_base, targets.get("compare_root", "../compare_existing_terms"))
mirror_root = abs_from(target_base, targets.get("mirror_root", targets.get("compare_root", "../compare_existing_terms")))
postprocess_config = abs_from(target_base, targets.get("postprocess_config", "../config_common/postprocess_test.yaml"))
output_base = abs_from(analyze_dir, sweep.get("output_root", f"results/{sweep.get('name', 'sweep')}/outputs"))
result_dir = abs_from(analyze_dir, sweep.get("result_dir", f"results/{sweep.get('name', 'sweep')}"))
log_dir = os.path.join(result_dir, "logs")

labels = list(targets.get("labels", []))
modes = list(targets.get("modes", []))
potential_sets = list(sweep.get("potential_sets", []))
rmsd_values = list(sweep.get("rmsd_max", []))
force_profiles = list(sweep.get("force_profiles", []))
metrics = list(sweep.get("multi_constant_metrics", ["max_abs"]))

variant_rows = []
for potential, metric, rmsd, force in itertools.product(potential_sets, metrics, rmsd_values, force_profiles):
    variant_id = f"{potential['id']}_R{fmt_value(rmsd)}_{force['id']}"
    if len(metrics) > 1:
        variant_id = f"{variant_id}_M{metric}"
    if only_variant and variant_id != only_variant:
        continue
    variant_rows.append((variant_id, potential, metric, rmsd, force))
    if variant_limit > 0 and len(variant_rows) >= variant_limit:
        break

for variant_id, potential, metric, rmsd, force in variant_rows:
    for label in labels:
        if only_label and label != only_label:
            continue
        for mode in modes:
            if only_mode and mode != only_mode:
                continue
            input_root = os.path.join(compare_root, label, mode)
            output_root = os.path.join(output_base, variant_id)
            log_path = os.path.join(log_dir, variant_id, label, f"{mode}.log")
            force_json = json.dumps(force["min"], separators=(",", ":"), sort_keys=True)
            row = [
                variant_id,
                label,
                mode,
                input_root,
                output_root,
                log_path,
                str(potential.get("angles", "bartender")),
                str(potential.get("dihedrals", "bartender")),
                str(potential.get("impropers", "bartender")),
                str(metric),
                str(force.get("mode", "absolute")),
                force_json,
                str(rmsd),
                bool_text(sweep.get("show_all_info", False)),
                bool_text(sweep.get("write_plots", False)),
                str(sweep.get("bond_constraint_mode", "bartender")),
                str(sweep.get("candidate_source", "active")),
                postprocess_config,
                mirror_root,
            ]
            print("\t".join(row))
PY
}

launched=0
running=0

while IFS=$'\t' read -r variant_id label mode input_root output_root log_path \
  angles dihedrals impropers metric force_mode force_json rmsd show_all write_plots \
  bond_constraint_mode candidate_source postprocess_config mirror_root; do

  [ -n "$variant_id" ] || continue
  core=$((CORE_START + (launched % CORE_COUNT)))
  mkdir -p "$(dirname "$log_path")"

  echo "[sweep] variant=${variant_id} label=${label} mode=${mode} core=${core}"
  LABEL="$label" \
  MODE_TAG="$mode" \
  INPUT_ROOT="$input_root" \
  MIRROR_ROOT="$mirror_root" \
  OUTPUT_ROOT="$output_root" \
  LOG_PATH="$log_path" \
  POSTPROCESS_CONFIG="$postprocess_config" \
  CORE="$core" \
  USE_SRUN="$USE_SRUN" \
  DRY_RUN="$DRY_RUN" \
  SKIP_MISSING="$SKIP_MISSING" \
  SKIP_EXISTING="$SKIP_EXISTING" \
  BOND_CONSTRAINT_MODE="$bond_constraint_mode" \
  CANDIDATE_SOURCE="$candidate_source" \
  SHOW_ALL_INFO="$show_all" \
  WRITE_PLOTS="$write_plots" \
  MULTI_CONSTANT_METRIC="$metric" \
  FORCE_METRIC_MIN_MODE="$force_mode" \
  FORCE_METRIC_MIN="$force_json" \
  RMSD_MAX="$rmsd" \
  POTENTIAL_ANGLES="$angles" \
  POTENTIAL_DIHEDRALS="$dihedrals" \
  POTENTIAL_IMPROPERS="$impropers" \
  bash "$PROJECT_DIR/postprocess.sh" &

  launched=$((launched + 1))
  running=$((running + 1))
  if [ "$running" -ge "$MAX_JOBS" ]; then
    wait -n
    running=$((running - 1))
  fi
done < <(task_rows)

wait
trap - SIGINT SIGTERM EXIT
echo "[done] launched=${launched} config=${SWEEP_CONFIG}"
