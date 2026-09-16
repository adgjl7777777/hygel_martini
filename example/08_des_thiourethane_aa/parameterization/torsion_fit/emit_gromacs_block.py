#!/usr/bin/env python3
"""Emit the ready-to-paste [ dihedraltypes ] replacement for the fitted tau1 torsion.

Where it goes: project/structure/forcefield.itp, section [ dihedraltypes ].
The production topology (project/output/*_hydrogel.itp) writes the two proper
dihedrals across each builder-formed S-C(=O) bond WITHOUT parameters
    <N> <C> <S> <C>  3
    <O> <C> <S> <C>  3
so they are resolved by atom type from [ dihedraltypes ].  Replacing the type
rows changes all 384 junctions at once; no per-atom index has to be written.

The fit was made against phi = C-S-C(=O)-N.  phi(C-S-C(=O)-O) = phi + 180.000 deg
(mean over the 36 F3 scan geometries, sd 2.12), so the whole profile is carried
by the N-terminated row and the O-terminated row is set to zero.  Both rows must
remain present: grompp fails if a dihedral in the topology has no matching type.
"""
import json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
FF = HERE.parents[1] / 'project' / 'structure' / 'forcefield.itp'
C = json.load(open(HERE/'fit_coefficients.json'))['rb_C_kJ']
txt = FF.read_text().splitlines()
pat_N = re.compile(r'^\s*(hx\w+)\s+(lkS)\s+(st\w+)\s+(stN\w+)\s+3\s')
pat_O = re.compile(r'^\s*(hx\w+)\s+(lkS)\s+(st\w+)\s+(stO\w+)\s+3\s')
new, old = [], []
for l in txt:
    m = pat_N.match(l) or pat_O.match(l)
    if not m: continue
    old.append(l.strip())
    c = C if pat_N.match(l) else [0.0]*6
    new.append(f"  {m.group(1):<8s} {m.group(2):<8s} {m.group(3):<8s} {m.group(4):<8s} 3 "
               + " ".join(f"{v:9.4f}" for v in c))
out = HERE/'gromacs_dihedraltypes_block.txt'
out.write_text(
 "; --- fitted S-C(=O) (tau1) torsion, 2026-09-16 -------------------------------\n"
 "; Ryckaert-Bellemans, GROMACS funct 3, ALL COEFFICIENTS IN kJ/mol.\n"
 ";   V(psi) = sum_{n=0..5} C_n cos^n(psi),  psi = phi - 180 deg\n"
 ";   phi = C-S-C(=O)-N  (the row order below is exactly that)\n"
 "; Equivalent OPLS Fourier form, GROMACS funct 5, kJ/mol:\n"
 ";   V(phi) = 1/2[F1(1+cos phi) + F2(1-cos 2phi) + F3(1+cos 3phi) + F4(1-cos 4phi)]\n"
 ";   F1 = 8.5962   F2 = 17.9171   F3 = -3.2721   F4 = 0.0000  kJ/mol\n"
 "; Fitted to E_QM - E_MM(tau1 off) on soohki's F3 tau1 relaxed scan (35 of 37\n"
 "; points; scan.037 excluded upstream, scan.001 excluded here - see the report).\n"
 "; The O-terminated rows are zeroed: phi(O-C-S-C) = phi(N-C-S-C) + 180 deg, so the\n"
 "; N row already carries the whole profile.  They must stay present for grompp.\n"
 "; REPLACES these 24 rows in project/structure/forcefield.itp:\n"
 + "".join(f";   {o}\n" for o in old)
 + "; --- new rows ----------------------------------------------------------------\n"
 + "\n".join(new) + "\n")
print(f"{len(old)} rows replaced -> {out}")
print("\n".join(new[:4])); print("  ...")
