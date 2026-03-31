#!/usr/bin/env bash

# Optional shell environment for this launcher.
# Edit this file instead of editing run_full_builder.sh.

HYGEL_REPO_ROOT="${HYGEL_REPO_ROOT:-}"
ADDITIONAL_BASH_PROFILE="${ADDITIONAL_BASH_PROFILE:-/nas_3/active/transcendence/anaconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-hygel}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

GMXRC_PATH="${GMXRC_PATH:-/opt/gromacs/2026/bin/GMXRC}"
GMX_CMD="${GMX_CMD:-gmx_mpi}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
GMX_OPENMP_MAX_THREADS="${GMX_OPENMP_MAX_THREADS:-$OMP_NUM_THREADS}"
