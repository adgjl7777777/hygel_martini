#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-hygel}"

usage() {
  cat <<EOF
Usage:
  bash hygel_run.sh [config_N/maker.yaml] [qm_to_martini options...]
  bash hygel_run.sh --help
  bash hygel_run.sh --workflow-help
  bash hygel_run.sh --check-xtb
  bash hygel_run.sh --check-bartender

Default config:
  $SCRIPT_DIR/config_1/maker.yaml

Examples:
  bash hygel_run.sh
  bash hygel_run.sh config_2/maker.yaml
  bash hygel_run.sh --check-xtb --check-bartender
  bash hygel_run.sh config_3/maker.yaml --symbols S,D --lengths 3,5

Environment overrides:
  CONDA_PROFILE=${CONDA_PROFILE:-/nas_3/active/transcendence/anaconda3/etc/profile.d/conda.sh}
  CONDA_ENV_NAME=${CONDA_ENV_NAME:-hygel}
  XTB_ENV_SCRIPT=${XTB_ENV_SCRIPT:-/opt/xtb-dist/share/xtb/config_env.bash}
  BTROOT=${BTROOT:-/opt/bartender-1.1.0}
EOF
}

WORKFLOW_HELP=0
CHECK_XTB=0
CHECK_BARTENDER=0
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --workflow-help)
      WORKFLOW_HELP=1
      shift
      ;;
    --check-xtb)
      CHECK_XTB=1
      shift
      ;;
    --check-bartender)
      CHECK_BARTENDER=1
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

CONFIG_ARG="config_1/maker.yaml"
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
  CONFIG_ARG="$1"
  shift
fi

if [[ "$CONFIG_ARG" = /* ]]; then
  CONFIG_PATH="$CONFIG_ARG"
else
  CONFIG_PATH="$SCRIPT_DIR/$CONFIG_ARG"
fi

if [ "$WORKFLOW_HELP" -eq 0 ] && [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] Config not found: $CONFIG_PATH" >&2
  exit 1
fi

CONDA_PROFILE="${CONDA_PROFILE:-/nas_3/active/transcendence/anaconda3/etc/profile.d/conda.sh}"
if [ ! -f "$CONDA_PROFILE" ]; then
  echo "[ERROR] conda.sh not found: $CONDA_PROFILE" >&2
  exit 1
fi

source "$CONDA_PROFILE"
conda activate "$CONDA_ENV_NAME"

XTB_ENV_SCRIPT="${XTB_ENV_SCRIPT:-/opt/xtb-dist/share/xtb/config_env.bash}"
if [ -f "$XTB_ENV_SCRIPT" ]; then
  set +u
  # shellcheck disable=SC1090
  source "$XTB_ENV_SCRIPT"
  set -u
fi

BTROOT="${BTROOT:-/opt/bartender-1.1.0}"
export BTROOT
BARTENDER_CONFIG_SCRIPT="${BARTENDER_CONFIG_SCRIPT:-$BTROOT/bartender_config.sh}"
if [ -f "$BARTENDER_CONFIG_SCRIPT" ]; then
  set +u
  # shellcheck disable=SC1090
  source "$BARTENDER_CONFIG_SCRIPT"
  set -u
fi

cd "$REPO_ROOT"

if [ "$CHECK_XTB" -eq 1 ]; then
  if command -v xtb >/dev/null 2>&1; then
    echo "[INFO] xtb: $(command -v xtb)"
  else
    echo "[ERROR] xtb not found in PATH" >&2
    exit 1
  fi
fi

if [ "$CHECK_BARTENDER" -eq 1 ]; then
  if command -v bartender >/dev/null 2>&1; then
    echo "[INFO] bartender: $(command -v bartender)"
  else
    echo "[ERROR] bartender not found in PATH" >&2
    exit 1
  fi
fi

if [ "$CHECK_XTB" -eq 1 ] || [ "$CHECK_BARTENDER" -eq 1 ]; then
  exit 0
fi

if [ "$WORKFLOW_HELP" -eq 1 ]; then
  python -m param_opt.qm_to_martini --help
  exit 0
fi

python -m param_opt.qm_to_martini --config "$CONFIG_PATH" "$@"
