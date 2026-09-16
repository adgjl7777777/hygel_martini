#!/usr/bin/env python3
"""Same high-precision-coordinate recompute for the 109 F3x tau1 geometries.
Only E_MM(torsion off) is needed: the validation compares the fitted torsion
against residual = E_QM_rel - E_MMoff_rel, branch by branch. Read-only on DFT."""
import sys, os, re, glob, subprocess, numpy as np
from pathlib import Path
sys.path.insert(0, '/nas_1/transcendence/2026_2/omni_hygel_package/package/example/08_des_thiourethane_aa/validation')
import ff_pair_benchmark as b
HERE = Path(__file__).resolve().parent
LNK = '/nas_1/transcendence/2026_2/omni_hygel_package/package/example/08_des_thiourethane_aa/parameterization/raw/LNK.itp'
OFF = '/nas_1/transcendence/2026_2/omni_hygel_package/cowork/03_analysis/torsion_residual_F3x_tau1_20260915/LNK_tau1off.itp'
D = '/nas_3/active/soohki/27.des/dft/06_torsion'
SERIES = {'orig':'F3x_thiouret_ext__tau1_SC','refwd':'F3x_thiouret_ext__tau1_SC_refwd','rerev':'F3x_thiouret_ext__tau1_SC_rerev'}
HART, KJ = 627.509474, 4.184
WORK = Path(sys.argv[1] if len(sys.argv)>1 else './f3x_work').resolve(); WORK.mkdir(parents=True, exist_ok=True)
at = b.read_atomtypes(LNK); atoms, bonds = b.read_itp(LNK, at)
os.chdir(WORK)
Path('off.itp').write_text(Path(OFF).read_text())
Path('sys_off.top').write_text('[ defaults ]\n  1  3  yes  0.5  0.5\n#include "off.itp"\n[ system ]\nLNK\n[ molecules ]\nLNK 1\n')
Path('sp.mdp').write_text("integrator=md\nnsteps=0\ncutoff-scheme=Verlet\nnstlist=1\ncoulombtype=Reaction-Field\nepsilon-rf=1\ncoulomb-modifier=none\nrcoulomb=3.5\nvdwtype=cutoff\nvdw-modifier=none\nrvdw=3.5\nrlist=3.5\npbc=xyz\n")
molname = [l.split()[0] for l in open(LNK) if l.strip() and not l.strip().startswith(('[',';')) and 'UNK' in l][0]
Path('sys_off.top').write_text(f'[ defaults ]\n  1  3  yes  0.5  0.5\n#include "off.itp"\n[ system ]\nLNK\n[ molecules ]\n{molname} 1\n')
def dih(p0,p1,p2,p3):
    b0,b1,b2 = p0-p1, p2-p1, p3-p2; b1 = b1/np.linalg.norm(b1)
    v = b0-np.dot(b0,b1)*b1; w = b2-np.dot(b2,b1)*b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1,v),w), np.dot(v,w))) % 360
ref = sorted(glob.glob(f'{D}/{SERIES["refwd"]}/scan.*.xyz'))[0]
els, xyz = b.read_xyz(ref)
m = b.match_graphs([a.element for a in atoms], bonds, els, b.bonds_from_geometry(els, xyz)); assert m
def energy(X, name):
    g = [f'LNK {name}', str(len(atoms))] + [
        f"{1:5d}{'LNK':<5s}{a.name:>5s}{i:5d}" + "%11.6f%11.6f%11.6f" % (c[0]/10+4, c[1]/10+4, c[2]/10+4)
        for i,(a,c) in enumerate(zip(atoms, X), 1)] + ["  8.00000   8.00000   8.00000"]
    Path(f'{name}.gro').write_text('\n'.join(g)+'\n')
    subprocess.run(['gmx_mpi','grompp','-f','sp.mdp','-c',f'{name}.gro','-p','sys_off.top','-o',f'{name}.tpr','-po',f'{name}.mdp','-maxwarn','2'], capture_output=True, check=True)
    subprocess.run(['gmx_mpi','mdrun','-deffnm',name,'-ntomp','1','-nb','cpu','-pin','off'], capture_output=True, check=True)
    t = Path(f'{name}.log').read_text().split('Energies (kJ/mol)')[-1].splitlines()
    for i,l in enumerate(t):
        if 'Potential' in l:
            k = [x.strip() for x in re.split(r'\s{2,}', l.strip())]
            return dict(zip(k, [float(v) for v in t[i+1].split()]))['Potential']
rows = []
for series, d in SERIES.items():
    for f in sorted(glob.glob(f'{D}/{d}/scan.*.xyz')):
        step = int(f[-7:-4]); L = open(f).read().splitlines()
        eqm = float(re.search(r'E\s+(-?\d+\.\d+)', L[1]).group(1))
        els2, xyz2 = b.read_xyz(f); assert els2 == els
        m2 = b.match_graphs([a.element for a in atoms], bonds, els2, b.bonds_from_geometry(els2, xyz2))
        assert m2 == m, f'{f}: bond graph changed'
        X = np.array(xyz2); phi = dih(X[5],X[6],X[7],X[9])
        rows.append((series, step, phi, eqm, energy([xyz2[k] for k in m], f'{series}_{step:03d}')))
out = HERE/'residual_F3x_hiprec.tsv'
with open(out,'w') as fh:
    fh.write("series\tstep\tphi_deg\tE_QM_Eh\tE_QM_rel_kcal\tE_MMoff_rel_hiprec_kcal\tresidual_hiprec_kcal\n")
    for s in SERIES:
        sub = [r for r in rows if r[0]==s]; k = min(range(len(sub)), key=lambda i: sub[i][3])
        for series, step, phi, eqm, eo in sub:
            q = (eqm-sub[k][3])*HART; o = (eo-sub[k][4])/KJ
            fh.write(f"{series}\t{step}\t{phi:.2f}\t{eqm:.9f}\t{q:.4f}\t{o:.4f}\t{q-o:.4f}\n")
print("wrote", out, len(rows), "points")
