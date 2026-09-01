#!/bin/bash
# Relax a tiled strand template in vacuo and write the relaxed coordinates back.
#
# tile_ppg.py places copies by an exact rigid screw, so bond lengths and the
# junction angles are already the parameterized ones -- but nothing has relaxed
# the side groups against each other along a 33-unit helix. Steepest descent in
# vacuo fixes that without coiling the chain (there is no thermal motion in an
# EM), which is what a rigidly placed strand template needs: locally clean,
# still extended.
#
# The two BCK attachment carbons are three-coordinate in the isolated molecule
# (their sulfur partners belong to the junction), so the ends relax slightly
# out of their bonded environment. That is accepted: the ends are two atoms of
# 373, and the builder re-forms those bonds anyway.
#
# Usage: bash relax_tiled.sh STR_n33
set -e
PREFIX="${1:-STR_n33}"
HERE="$(cd "$(dirname "$0")" && pwd)"
STRUCT="$HERE/../project/structure"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

NAME=$(awk '/\[ moleculetype \]/{getline; print $1; exit}' "$STRUCT/$PREFIX.itp")
cat > "$WORK/topol.top" <<TOP
#include "$STRUCT/forcefield.itp"
#include "$STRUCT/$PREFIX.itp"
[ system ]
tiled strand in vacuo
[ molecules ]
$NAME 1
TOP

# A box comfortably larger than the extended chain: vacuum, cut-off
# electrostatics (an isolated neutral molecule needs no lattice sum).
python3 - "$STRUCT/$PREFIX.gro" "$WORK/start.gro" <<'PY'
import sys
src = open(sys.argv[1]).read().splitlines()
count = int(src[1].split()[0])
xs = [(float(l[20:28]), float(l[28:36]), float(l[36:44])) for l in src[2:2+count]]
lo = [min(c[i] for c in xs) for i in range(3)]
box = [max(c[i] for c in xs) - lo[i] + 4.0 for i in range(3)]
out = [src[0], src[1]]
for line, c in zip(src[2:2+count], xs):
    out.append(line[:20] + "".join(f"{c[i]-lo[i]+2.0:8.3f}" for i in range(3)))
out.append("".join(f"{b:10.5f}" for b in box))
open(sys.argv[2], "w").write("\n".join(out) + "\n")
PY

cat > "$WORK/em.mdp" <<MDP
integrator  = steep
nsteps      = 20000
emtol       = 100.0
emstep      = 0.002
coulombtype = Cut-off
rcoulomb    = 1.5
vdw_type    = cutoff
rvdw        = 1.5
pbc         = xyz
MDP

cd "$WORK"
gmx_mpi grompp -f em.mdp -c start.gro -p topol.top -o em.tpr -maxwarn 2 > grompp.log 2>&1
OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} gmx_mpi mdrun -deffnm em -ntomp "${OMP_NUM_THREADS:-4}" -nb cpu > mdrun.log 2>&1
grep -E "Steepest Descents|Potential Energy|Maximum force" em.log | tail -3

python3 - "em.gro" "$STRUCT/$PREFIX.gro" <<'PY'
import sys, math
relaxed = open(sys.argv[1]).read().splitlines()
target = sys.argv[2]
original = open(target).read().splitlines()
count = int(relaxed[1].split()[0])
assert count == int(original[1].split()[0]), "atom count changed"
out = [original[0], original[1]]
coords = []
for src, dst in zip(relaxed[2:2+count], original[2:2+count]):
    coords.append((float(src[20:28]), float(src[28:36]), float(src[36:44])))
    out.append(dst[:20] + src[20:44])
out.append(original[2+count])
open(target, "w").write("\n".join(out) + "\n")
bck = [i for i, l in enumerate(original[2:2+count]) if l[10:15].strip().startswith("BCK")]
span = math.dist(coords[bck[0]], coords[bck[1]])
print(f"relaxed coordinates written back to {target}")
print(f"BCK1-BCK2 span after relaxation: {span:.3f} nm")
print(f"suggested cell_parameter ~= span + 2*(junction arm 0.772 + bond 0.1715) = {span + 2*(0.772+0.1715):.2f} nm")
PY
