#!/usr/bin/env python3
"""Recompute the F3 tau1 MM single points with HIGH-PRECISION coordinates.

Why: the published residual (cowork/.../torsion_residual_F3_tau1.tsv) wrote the
GROMACS .gro with "%8.3f" in nm = 0.01 Angstrom coordinate rounding.  Stiff bond
and angle terms turn that into ~0.5 kcal/mol of scatter in E_MM, which shows up
as scatter in the residual.  Writing "%11.6f" instead (GROMACS reads variable
gro precision) removes it: the resulting residual is symmetric about phi=180 to
~0.03 kcal/mol, as it must be because the QM scan is mirror-symmetric to 1e-4
kcal/mol.

Read-only on the DFT scan directory.  Runs 2 x 36 gmx single points, 1 thread.
"""
import numpy as np, os, re, subprocess, sys
from pathlib import Path

SCAN = '/nas_3/active/soohki/27.des/dft/06_torsion/F3_thiouret_p__tau1_SC'
HERE = Path(__file__).resolve().parent
FULL_ITP = HERE.parent.parent / 'validation' / 'fragments' / 'F3.itp'
# torsion-off ITP (S-C(=O) proper RB rows 5-3-2-1 and 4-3-2-1 deleted)
OFF_ITP  = Path('/nas_1/transcendence/2026_2/omni_hygel_package/cowork/03_analysis/'
                'torsion_residual_F3_tau1_20260915/F3_tau1off.itp')
WORK = Path(sys.argv[1] if len(sys.argv) > 1 else './hiprec_work').resolve()
HART, KJ = 627.509474, 4.184

def main():
    WORK.mkdir(parents=True, exist_ok=True); os.chdir(WORK)
    # grompp's preprocessor choked on the long absolute include path -> copy locally
    for tag, src in (('full', FULL_ITP), ('off', OFF_ITP)):
        Path(f'{tag}.itp').write_text(Path(src).read_text())
        Path(f'sys_{tag}.top').write_text(
            f'[ defaults ]\n  1  3  yes  0.5  0.5\n#include "{tag}.itp"\n'
            f'[ system ]\nx\n[ molecules ]\nUNK 1\n')
    Path('sp.mdp').write_text(
        "integrator=md\nnsteps=0\ncutoff-scheme=Verlet\nnstlist=1\n"
        "coulombtype=Reaction-Field\nepsilon-rf=1\ncoulomb-modifier=none\n"
        "rcoulomb=3.5\nvdwtype=cutoff\nvdw-modifier=none\nrvdw=3.5\nrlist=3.5\npbc=xyz\n")
    names = [l.split()[4] for l in Path('full.itp').read_text().split('[ atoms ]')[1]
             .split('[ bonds ]')[0].splitlines() if l.strip() and not l.strip().startswith(';')]
    # ORCA scan xyz atom order == F3.itp atom order (element-by-element identity; checked)
    def read(f):
        L = Path(f).read_text().splitlines(); n = int(L[0])
        e = float(re.search(r'E\s+(-?\d+\.\d+)', L[1]).group(1))
        els = [l.split()[0] for l in L[2:2+n]]
        X = np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2+n]])
        return e, els, X
    def gro(X, name):
        g = [name, str(len(names))] + [
            f"{1:5d}{'MOL':<5s}{nm:>5s}{i:5d}" + "%11.6f%11.6f%11.6f" % tuple(c/10+4)
            for i, (nm, c) in enumerate(zip(names, X), 1)] + ["  8.00000   8.00000   8.00000"]
        Path(f'{name}.gro').write_text('\n'.join(g) + '\n')
    def energy(name, tag):
        subprocess.run(['gmx_mpi','grompp','-f','sp.mdp','-c',f'{name}.gro','-p',f'sys_{tag}.top',
                        '-o',f'{name}_{tag}.tpr','-po',f'{name}_{tag}.mdp','-maxwarn','2'],
                       capture_output=True, check=True)
        subprocess.run(['gmx_mpi','mdrun','-deffnm',f'{name}_{tag}','-ntomp','1','-nb','cpu','-pin','off'],
                       capture_output=True, check=True)
        t = Path(f'{name}_{tag}.log').read_text().split('Energies (kJ/mol)')[-1].splitlines()
        for i, l in enumerate(t):
            if 'Potential' in l:
                k = [x.strip() for x in re.split(r'\s{2,}', l.strip())]
                return dict(zip(k, [float(v) for v in t[i+1].split()]))['Potential']
    def dih(p0,p1,p2,p3):
        b0,b1,b2 = p0-p1, p2-p1, p3-p2; b1 = b1/np.linalg.norm(b1)
        v = b0-np.dot(b0,b1)*b1; w = b2-np.dot(b2,b1)*b1
        return np.degrees(np.arctan2(np.dot(np.cross(b1,v),w), np.dot(v,w))) % 360
    _, els0, X0 = read(f'{SCAN}/scan.001.xyz')
    assert els0 == [n[0] for n in names], "atom order mismatch"
    rows = []
    for s in range(1, 37):                     # scan.037 excluded upstream (bond graph differs)
        e, els, X = read(f'{SCAN}/scan.{s:03d}.xyz'); assert els == els0
        gro(X, f's{s:03d}')
        rows.append((s, dih(X[0],X[1],X[2],X[4]), e,
                     energy(f's{s:03d}','off'), energy(f's{s:03d}','full'),
                     dih(X[12],X[0],X[1],X[2])))
    k = min(range(len(rows)), key=lambda i: rows[i][2])
    out = HERE/'residual_F3_hiprec.tsv'
    with open(out,'w') as fh:
        fh.write("step\tphi_deg\tE_QM_Eh\tE_QM_rel_kcal\tE_MMfull_rel_hiprec_kcal\t"
                 "E_MMoff_rel_hiprec_kcal\tresidual_hiprec_kcal\tmethyl_HCSC_deg\n")
        for s, phi, e, eo, ef, me in rows:
            q = (e-rows[k][2])*HART; o = (eo-rows[k][3])/KJ; f_ = (ef-rows[k][4])/KJ
            fh.write(f"{s}\t{phi:.2f}\t{e:.9f}\t{q:.4f}\t{f_:.4f}\t{o:.4f}\t{q-o:.4f}\t{me:.2f}\n")
    print("wrote", out)

if __name__ == '__main__':
    main()
