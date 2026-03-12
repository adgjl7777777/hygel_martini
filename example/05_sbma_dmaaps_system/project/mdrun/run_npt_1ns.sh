#!/bin/bash
#SBATCH -J sbma_npt_1ns
#SBATCH -p goldpart2
#SBATCH -N 1
#SBATCH -n 64
##SBATCH -t 02:00:00
#SBATCH --nodelist=node17
#SBATCH -o npt_1ns.%j.out
#SBATCH -e npt_1ns.%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

MDP="npt_1ns.mdp"
GRO="$PROJECT_DIR/soft_em/box_relax_loop_em/iter_300/em/em.gro"
TOP="$PROJECT_DIR/output/system.top"
TPR="npt_1ns.tpr"
DEFFNM="npt_1ns"

gmx_mpi grompp -f "${MDP}" -c "${GRO}" -p "${TOP}" -o "${TPR}" -maxwarn 1
gmx_mpi mdrun -deffnm "${DEFFNM}" -ntomp 64
