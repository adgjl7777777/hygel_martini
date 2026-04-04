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

find_repo_root() {
  local current="$1"
  while [ -n "$current" ] && [ "$current" != "/" ]; do
    if [ -d "$current/param_opt" ] || [ -d "$current/hydrogel_builder" ]; then
      printf '%s\n' "$current"
      return 0
    fi
    current=$(dirname "$current")
  done
  if [ -d "/param_opt" ] || [ -d "/hydrogel_builder" ]; then
    printf '/\n'
    return 0
  fi
  return 1
}

setup_hygel_env() {
  local script_dir="$1"
  
  # 1. Source environment.sh if it exists
  ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-$script_dir/environment.sh}"
  if [ -f "$ENVIRONMENT_FILE" ]; then
    source_optional_script "ENVIRONMENT_FILE" "$ENVIRONMENT_FILE"
  fi

  # 2. Resolve REPO_ROOT
  if [ -n "${HYGEL_REPO_ROOT:-}" ]; then
    REPO_ROOT="$HYGEL_REPO_ROOT"
  elif REPO_ROOT=$(find_repo_root "$script_dir"); then
    :
  else
    REPO_ROOT=""
  fi

  # 3. Export PYTHONPATH
  if [ -n "$REPO_ROOT" ]; then
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  fi

  # 4. Resolve Python binary
  if [ -z "${PYTHON_BIN:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
      PYTHON_BIN="python3"
    else
      PYTHON_BIN="python"
    fi
  fi

  # 5. Source additional profile and activate conda if needed
  source_optional_script "ADDITIONAL_BASH_PROFILE" "${ADDITIONAL_BASH_PROFILE:-}"
  activate_optional_env
  
  export REPO_ROOT
  export PYTHON_BIN
}
