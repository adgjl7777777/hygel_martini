#!/usr/bin/env bash

# Project-local environment overrides for 02_opls_to_martini.
# Leave empty when the active shell already has the right Python/GROMACS setup.

ADDITIONAL_BASH_PROFILE="${ADDITIONAL_BASH_PROFILE:-}"
ENV_NAME="${ENV_NAME:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

GMXRC_PATH="${GMXRC_PATH:-/opt/gromacs/2026/bin/GMXRC}"
GMX_CMD="${GMX_CMD:-gmx_mpi}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

export ADDITIONAL_BASH_PROFILE ENV_NAME PYTHON_BIN
export GMXRC_PATH GMX_CMD OMP_NUM_THREADS

