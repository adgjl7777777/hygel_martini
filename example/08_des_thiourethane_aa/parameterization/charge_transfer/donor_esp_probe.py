#!/usr/bin/env python3
"""Turn the mono-functional-fragment error into kcal/mol at the donor, with no new QM.

Both the mono-functional fragments (F2 urethane, F3 thiourethane) and the full
2,4-TDI strand (parameterization/raw/STR.itp) carry LigParGen 1.14*CM1A-LBCC
charges, so the difference between them isolates the effect of the second ring
nitrogen inside ONE charge model.

Method.  Put a -1 point charge where a chloride would sit (on the N-H axis,
d A beyond H) and evaluate the Coulomb energy against point charges,
E = 332.06371 * sum_i q_i*(-1)/r_i  [kcal/mol, r in A], eps = 1.

Summing over the WHOLE molecule is not a fair comparison: the strand is 83 atoms
of extended, polar chain and its far field swamps the local difference (see the
'whole molecule' rows - the two thiourethane donors inside the same strand differ
by 35 kcal/mol purely from where the rest of the chain happens to lie).  So the
sum is restricted to the donor's own neighbourhood, the atoms within k bonds of
the donor nitrogen, with the SAME element composition checked on both sides.
The group's net charge is printed too: a non-zero net charge is not an artefact,
it is the charge the second nitrogen pulls out of the local group, and it is the
dominant part of the effect.
"""
from __future__ import annotations
import sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; PARAM = HERE.parent
sys.path.insert(0, str(PARAM)); import fragment_coverage as fc
KE = 332.06371
EX = PARAM.parent

def read_gro(p):
    L = Path(p).read_text().splitlines(); n = int(L[1])
    return np.array([[float(l[20+8*k:28+8*k])*10 for k in range(3)] for l in L[2:2+n]])

def donors(els, adj):
    out = []
    for i, e in enumerate(els):
        if e != 'N': continue
        hs = [j for j in adj[i] if els[j] == 'H']
        if not hs: continue
        for c in [j for j in adj[i] if els[j] == 'C']:
            o = [x for x in adj[c] if els[x] == 'O']; s = [x for x in adj[c] if els[x] == 'S']
            if s and o: out.append(('thiourethane', i, hs[0]))
            elif o and any(len(adj[x]) == 2 for x in o): out.append(('urethane', i, hs[0]))
    return out

def shell(i, adj, k):
    seen = {i}; front = {i}
    for _ in range(k):
        nxt = set().union(*[adj[a] for a in front]) - seen if front else set()
        seen |= nxt; front = nxt
    return sorted(seen)

def load(itp, gro):
    els, names, q, adj = fc.read_template(Path(itp))
    X = read_gro(gro); assert len(X) == len(els)
    return els, names, q, adj, X

SETS = {
 'F3 mono thiouret': (EX/'validation/fragments/F3.itp', EX/'validation/fragments/F3.gro'),
 'F2 mono urethane': (EX/'validation/fragments/F2.itp', EX/'validation/fragments/F2.gro'),
 'STR 2,4-TDI':      (PARAM/'raw/STR.itp', PARAM/'raw/STR.gro'),
}
D = 2.2
rows = defaultdict(list)
print(f"Cl- probe {D} A beyond H along N-H; eps=1; LigParGen 1.14*CM1A-LBCC charges\n")
print(f"{'system':18s} {'donor':18s} {'scope':10s} {'net q':>8s} {'E kcal/mol':>11s}")
for tag,(itp,gro) in SETS.items():
    els,names,q,adj,X = load(itp,gro)
    for kind,i,h in donors(els,adj):
        u = X[h]-X[i]; u/= np.linalg.norm(u); p = X[h]+D*u
        for scope,sel in [('whole',list(range(len(els))))]+[(f'k<={k}',shell(i,adj,k)) for k in (2,3,4)]:
            r = np.linalg.norm(X[sel]-p,axis=1); E = KE*float(np.sum(q[sel]*(-1.0)/r))
            comp = ''.join(f"{v}{k2}" for k2,v in sorted(Counter(els[j] for j in sel).items()))
            rows[(tag,kind,scope)].append((E,float(q[sel].sum()),comp,len(sel)))
            print(f"{tag:18s} {kind+' '+names[i]:18s} {scope:10s} {q[sel].sum():+8.4f} {E:11.3f}   {comp}")
    print()
print("--- thiourethane minus urethane preference (more negative E = stronger Cl- attraction) ---")
print(f"{'scope':10s} {'fragment set':>14s} {'2,4-TDI strand':>16s} {'shift':>10s}   (kcal/mol)")
for scope in ['whole','k<=2','k<=3','k<=4']:
    f_th = np.mean([e for e,_,_,_ in rows[('F3 mono thiouret','thiourethane',scope)]])
    f_ur = np.mean([e for e,_,_,_ in rows[('F2 mono urethane','urethane',scope)]])
    s_th = np.mean([e for e,_,_,_ in rows[('STR 2,4-TDI','thiourethane',scope)]])
    s_ur = np.mean([e for e,_,_,_ in rows[('STR 2,4-TDI','urethane',scope)]])
    print(f"{scope:10s} {f_th-f_ur:14.3f} {s_th-s_ur:16.3f} {(s_th-s_ur)-(f_th-f_ur):10.3f}")
