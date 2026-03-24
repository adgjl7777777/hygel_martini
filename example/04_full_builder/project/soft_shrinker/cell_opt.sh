#!/bin/bash
#SBATCH -J sbma_cell_opt
#SBATCH -p gpupart
#SBATCH -N 1
#SBATCH -n 1            # ASE-GROMACS 인터페이스는 MPI 프로세스 1개가 관리하는 것이 안정적임
#SBATCH -c 64           # 실제 계산은 32개 코어 사용
#SBATCH --nodelist=nanode04
#SBATCH -o cell_opt.%j.out
#SBATCH -e cell_opt.%j.err

set -euo pipefail

# 가상환경 활성화 (ase가 설치된 환경)
# source activate hygel 

# GROMACS 실행을 위한 환경 변수
export OMP_NUM_THREADS=64
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
TOP="$PROJECT_DIR/output/system.top"
START="$PROJECT_DIR/output/final_optimized_system.gro"
python -u "$SCRIPT_DIR/cell_opt.py" \
  --top "$TOP" \
  --start-gro "$START" \
  --gmx gmx_mpi \
  --bonded-itp "$PROJECT_DIR/output/initial_hydrogel.itp"
