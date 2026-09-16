#!/usr/bin/env bash
set -o pipefail
source /opt/gromacs/2026/bin/GMXRC
E=/nas_1/transcendence/2026_2/omni_hygel_package/package/example/08_des_thiourethane_aa
W=/tmp/claude-1214/-home-transcendence/5aef5a06-1d06-4700-82e7-41c2ed176034/scratchpad/anneal_test
cd $W
gmx_mpi grompp -f $E/project/config_npt/anneal.mdp \
  -c $E/project/shrink_output_size_test_n3_r4_c32/final.gro \
  -p $E/project/output_size_test_n3_r4_c32/final_system_no_ions_geo_opt/system.top \
  -o anneal.tpr -po anneal.mdout.mdp -maxwarn 1 > grompp.log 2>&1 || { echo "GROMPP_FAILED"; tail -20 grompp.log; exit 1; }
echo "grompp ok"
gmx_mpi mdrun -deffnm anneal -ntomp 4 > mdrun.log 2>&1
echo "ANNEAL_EXIT=$?"
