#!/usr/bin/env python3
"""Isolate the CHARGE effect of the second ring nitrogen, at fixed geometry.

The whole-molecule probe mixes charge with conformation (the two thiourethane
donors of one STR strand differ by 35 kcal/mol purely from where the rest of the
chain lies).  So: keep ONE geometry - the mono-functional fragment's - and swap
only the charges of the donor's k<=2 group between

    (a) the fragment's own LigParGen charges, and
    (b) the corresponding group in the full 2,4-TDI strand (same charge model).

Whatever changes is the second nitrogen's effect on the charges, with geometry
held fixed.  The k<=2 group has identical element composition on both sides
(4C 1H 1N 1O 1S for thiourethane; 4C 1H 1N 2O for urethane); atoms are matched
by (bond distance from N, element, neighbour element multiset), and the two ring
ortho carbons, which that key cannot separate, are averaged.
"""
from __future__ import annotations
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; PARAM = HERE.parent
sys.path.insert(0, str(PARAM)); import fragment_coverage as fc
KE = 332.06371; EX = PARAM.parent; D = 2.2

def read_gro(p):
    L = Path(p).read_text().splitlines(); n = int(L[1])
    return np.array([[float(l[20+8*k:28+8*k])*10 for k in range(3)] for l in L[2:2+n]])
def load(itp,gro):
    els,names,q,adj = fc.read_template(Path(itp)); X = read_gro(gro)
    assert len(X)==len(els); return els,names,q,adj,X
def donors(els,adj):
    out=[]
    for i,e in enumerate(els):
        if e!='N': continue
        hs=[j for j in adj[i] if els[j]=='H']
        if not hs: continue
        for c in [j for j in adj[i] if els[j]=='C']:
            o=[x for x in adj[c] if els[x]=='O']; s=[x for x in adj[c] if els[x]=='S']
            if s and o: out.append(('thiourethane',i,hs[0]))
            elif o and any(len(adj[x])==2 for x in o): out.append(('urethane',i,hs[0]))
    return out
def group(i,adj,k=2):
    seen={i}; front={i}; dist={i:0}
    for d in range(1,k+1):
        nxt=set().union(*[adj[a] for a in front])-seen if front else set()
        for a in nxt: dist[a]=d
        seen|=nxt; front=nxt
    return dist
def key(a,dist,els,adj):
    return (dist[a], els[a], ''.join(sorted(els[j] for j in adj[a])))
def keyed(i,els,adj,q):
    dist=group(i,adj); out=defaultdict(list)
    for a in dist: out[key(a,dist,els,adj)].append(float(q[a]))
    return dist,out

SETS={'F3':(EX/'validation/fragments/F3.itp',EX/'validation/fragments/F3.gro'),
      'F2':(EX/'validation/fragments/F2.itp',EX/'validation/fragments/F2.gro'),
      'STR':(PARAM/'raw/STR.itp',PARAM/'raw/STR.gro')}
data={t:load(*SETS[t]) for t in SETS}
sels,snames,sq,sadj,sX = data['STR']
strand={}
for kind,i,h in donors(sels,sadj):
    _,kd = keyed(i,sels,sadj,sq)
    strand.setdefault(kind,[]).append(kd)
print(f"Cl- probe {D} A beyond H along N-H, eps=1, k<=2 group only, geometry = fragment\n")
out={}
for tag,kind in (('F3','thiourethane'),('F2','urethane')):
    els,names,q,adj,X = data[tag]
    i,h = [(i,h) for k2,i,h in donors(els,adj) if k2==kind][0]
    dist,kd = keyed(i,els,adj,q)
    sel=sorted(dist)
    u=X[h]-X[i]; u/=np.linalg.norm(u); p=X[h]+D*u
    r=np.linalg.norm(X[sel]-p,axis=1)
    qf=np.array([q[a] for a in sel])
    E_frag = KE*float(np.sum(qf*(-1.0)/r))
    print(f"--- {kind}: fragment {tag}, group {''.join(f'{v}{k3}' for k3,v in sorted(Counter(els[a] for a in sel).items()))}")
    print(f"{'role':34s} {'q_frag':>9s} {'q_strand':>9s} {'dq':>8s}")
    for donor_idx, sk in enumerate(strand[kind]):
        qs=[]
        for a in sel:
            k3=key(a,dist,els,adj)
            cand = sk.get(k3)
            if cand is None:
                # ring ortho carbons: relax the neighbour-multiset part of the key
                cand=[v for kk,vv in sk.items() if kk[0]==k3[0] and kk[1]==k3[1] for v in vv]
            qs.append(float(np.mean(cand)) if cand else float(q[a]))
        qs=np.array(qs)
        E_str = KE*float(np.sum(qs*(-1.0)/r))
        if donor_idx==0:
            for a,qa,qb in zip(sel,qf,qs):
                print(f"  d={dist[a]} {els[a]:2s} {names[a]:6s} nb={''.join(sorted(els[j] for j in adj[a])):8s} "
                      f"{qa:+9.4f} {qb:+9.4f} {qb-qa:+8.4f}")
        print(f"  strand donor #{donor_idx+1}: group net q {qf.sum():+.4f} -> {qs.sum():+.4f} ; "
              f"E {E_frag:+8.3f} -> {E_str:+8.3f} kcal/mol  (dE {E_str-E_frag:+.3f})")
        out.setdefault(kind,[]).append(E_str-E_frag)
    print()
dth=np.array(out['thiourethane']); dur=np.array(out['urethane'])
print(f"second-nitrogen charge effect on the Cl- attraction at the donor:")
print(f"  thiourethane  dE = {dth.mean():+.3f} kcal/mol  (individual {np.round(dth,3).tolist()})")
print(f"  urethane      dE = {dur.mean():+.3f} kcal/mol  (individual {np.round(dur,3).tolist()})")
print(f"  effect on the thiourethane-minus-urethane preference: {dth.mean()-dur.mean():+.3f} kcal/mol")
print(f"  spread among equivalent donors (LigParGen fitting noise): "
      f"thiourethane {dth.max()-dth.min():.3f}, urethane {dur.max()-dur.min():.3f} kcal/mol")
