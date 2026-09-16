#!/usr/bin/env python3
"""What the ordering gate's quantity becomes if the charges are re-derived by a
pure radius-2 environment rule.

The committed gate (tests/test_resp_acceptance_gates.py) scores the FRAGMENT
charge sets against the FRAGMENT QM ESP, so a downstream transfer cannot move
it: the gate would pass unchanged and say nothing about the transfer.  To get a
number for the transfer itself we apply the radius-2 rule back onto the fragment
geometries and re-measure the same site preference with a common probe: a -1
point charge 2.2 A beyond H on the N-H axis, eps = 1, whole molecule.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; PARAM = HERE.parent
sys.path.insert(0, str(PARAM)); import fragment_coverage as fc
KE = 332.06371; D = 2.2; R = 2
idx, _ = fc.build_fragment_index(R, use_refit=True)
def probe(name, kind, qs, label):
    els, adj, _ = fc.read_fragment(name, None)
    X = np.array(fc.ffb.read_xyz(str(fc.DFT_ROOT/'05_resp'/name/'start.xyz'))[1])
    N = [i for i,e in enumerate(els) if e=='N' and any(els[j]=='H' for j in adj[i])
         and any(els[j]=='C' and any(els[x]=='O' for x in adj[j]) for j in adj[i])]
    i = N[0]; h = [j for j in adj[i] if els[j]=='H'][0]
    u = X[h]-X[i]; u/=np.linalg.norm(u); p = X[h]+D*u
    r = np.linalg.norm(X-p, axis=1)
    E = KE*float(np.sum(np.asarray(qs)*(-1.0)/r))
    print(f"  {label:26s} {name:18s} q(N) {qs[i]:+.4f} q(HN) {qs[h]:+.4f}  net {np.sum(qs):+.4f}  E = {E:8.3f} kcal/mol")
    return E
res = {}
for name, kind in (('F3_thiouret_p','thiourethane'), ('F2_urethane_p','urethane')):
    els, adj, qrefit = fc.read_fragment(name, fc.REFIT/f'{name}.qout')
    qship = np.array([float(v) for v in (fc.DFT_ROOT/'05_resp'/name/'qout_stage2').read_text().split()][:len(els)])
    qr2 = np.array([float(np.mean([q for _,_,q in idx.get(fc.fingerprint(i,els,adj,R),[]) if q is not None]
                                  or [qrefit[i]])) for i in range(len(els))])
    print(f"{kind}:")
    res[(kind,'shipped')] = probe(name, kind, qship, 'shipped RESP')
    res[(kind,'refit')]   = probe(name, kind, qrefit, 'Boltzmann refit')
    res[(kind,'r2')]      = probe(name, kind, qr2, 'radius-2 blind transfer')
    print()
print(f"{'charge set':26s} {'thiouret':>10s} {'urethane':>10s} {'F3-F2':>10s}  (kcal/mol, more negative = better donor)")
for tag,lab in (('shipped','shipped RESP'),('refit','Boltzmann refit'),('r2','radius-2 blind')):
    a,b = res[('thiourethane',tag)], res[('urethane',tag)]
    print(f"{lab:26s} {a:10.3f} {b:10.3f} {a-b:10.3f}")
print("\ngate reference: axial_delta_qm = -1.77, refit = -1.91, shipped = -0.55 kcal/mol "
      "(different probe, so read the SIGN and the RANK, not the magnitude)")
