#!/usr/bin/env bash

# Central utility functions for hygel_martini launchers.
# Source this file from run scripts to reduce duplication.

source_optional_script() {
  local label="$1"
  local path="$2"
  if [ -z "$path" ]; then
    return 0
  fi
  if [ ! -f "$path" ]; then
    echo "[ERROR] $label not found: $path" >&2
    exit 1
  fi
  set +u
  # shellcheck disable=SC1090
  source "$path"
  set -u
}

activate_optional_env() {
  # ENV_NAME should be set by sourced environment scripts
  if [ -z "${ENV_NAME:-}" ]; then
    return 0
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "[WARN] ENV_NAME is set but 'conda' is not available. Using the current shell environment." >&2
    return 0
  fi
  set +e
  conda activate "$ENV_NAME"
  local status=$?
  set -e
  if [ "$status" -ne 0 ]; then
    echo "[WARN] Failed to activate conda environment '$ENV_NAME'. Using the current shell environment." >&2
  fi
}

setup_hygel_env() {
  local script_dir="$1"

  # 1. Source environment.sh if it exists
  ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-$script_dir/environment.sh}"
  if [ -f "$ENVIRONMENT_FILE" ]; then
    source_optional_script "ENVIRONMENT_FILE" "$ENVIRONMENT_FILE"
  fi

  # 2. Resolve Python binary
  if [ -z "${PYTHON_BIN:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
      PYTHON_BIN="python3"
    else
      PYTHON_BIN="python"
    fi
  fi

  # 3. Source additional profile and activate conda if needed
  source_optional_script "ADDITIONAL_BASH_PROFILE" "${ADDITIONAL_BASH_PROFILE:-}"
  activate_optional_env

  export PYTHON_BIN
}

require_python_module() {
  local module_name="$1"

  if "$PYTHON_BIN" -c "import importlib; importlib.import_module('${module_name}')" >/dev/null 2>&1; then
    return 0
  fi

  local python_exe
  python_exe=$("$PYTHON_BIN" -c 'import sys; print(sys.executable)' 2>/dev/null || echo "$PYTHON_BIN")

  echo "[ERROR] Python module '${module_name}' is not importable." >&2
  echo "[ERROR]   Python used: ${python_exe}" >&2
  if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "[ERROR]   Active conda env: ${CONDA_DEFAULT_ENV}" >&2
  fi
  echo "[ERROR] Install hygel_martini into this environment:" >&2
  echo "[ERROR]   cd /path/to/hygel_martini && python -m pip install -e ." >&2
  exit 1
}
