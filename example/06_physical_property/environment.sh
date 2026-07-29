#!/usr/bin/env bash

# Optional shell environment for this launcher.
# Edit this file instead of editing run_property_extract.sh.

# hygel_martini 패키지가 이미 이 Python 환경에 설치되어 있어야 합니다.
ADDITIONAL_BASH_PROFILE="${ADDITIONAL_BASH_PROFILE:-}"
ENV_NAME="${ENV_NAME:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# GROMACS — 하드코딩 없음. 환경 변수 또는 PATH 자동 탐색.
# 특정 경로가 필요하면 여기서 지정:
#   GMXRC_PATH="/path/to/GMXRC"
#   GMX_CMD="/path/to/gmx_mpi"
GMXRC_PATH="${GMXRC_PATH:-}"
GMX_CMD="${GMX_CMD:-gmx_mpi}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
GMX_OPENMP_MAX_THREADS="${GMX_OPENMP_MAX_THREADS:-$OMP_NUM_THREADS}"
