#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-${1:-config/opls_existing_data.yaml}}"
MODE="${MODE:-setup}"
OUT_ROOT="${OUT_ROOT:-}"
LOG_PATH="${LOG_PATH:-}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<EOF
Usage:
  MODE=setup bash run_existing_opls.sh [config/opls_existing_data.yaml]
  MODE=md bash run_existing_opls.sh
  MODE=md_notrim bash run_existing_opls.sh
  MODE=trim bash run_existing_opls.sh
  MODE=bartender bash run_existing_opls.sh

MODE presets:
  setup              create trim + Bartender scripts only
  md                 run trim/prepare, then run Bartender
  md_notrim          convert/use trajectory without auto-trim, then run Bartender
  trim               run trim/prepare only; no Bartender job
  bartender          run Bartender; prepare trajectory first only if needed
  bartender_notrim   same as bartender, but without auto-trim
  notrim_nobartender prepare trajectory without auto-trim; no Bartender job
  off                metadata scaffold only

Optional environment:
  CONFIG=config/opls_existing_data.yaml
  OUT_ROOT=""
  LOG_PATH=""
  DRY_RUN=0
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

cmd=(bash "$SCRIPT_DIR/run_opls_to_martini.sh" "$CONFIG" --set "opls_data.execution.mode=${MODE}")
if [ -n "$OUT_ROOT" ]; then
  cmd+=(--set "paths.out_root=${OUT_ROOT}")
fi

echo "[opls-existing] mode=${MODE} config=${CONFIG}"
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
