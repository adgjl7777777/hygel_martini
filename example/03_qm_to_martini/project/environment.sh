#!/usr/bin/env bash

# Optional shell environment for this launcher.
# Edit this file instead of editing run_qm_to_martini.sh.

# The hygel_martini package must already be installed in this Python environment.
ADDITIONAL_BASH_PROFILE="${ADDITIONAL_BASH_PROFILE:-/nas_3/active/transcendence/anaconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-hygel}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Keep xTB / ORCA / Bartender binaries and env scripts in config_common/common.yaml.
