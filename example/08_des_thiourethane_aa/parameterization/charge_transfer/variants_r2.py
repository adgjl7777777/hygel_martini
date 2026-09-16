#!/usr/bin/env python3
"""Three ways to do the radius-2 transfer, and what each costs in redistributed charge.

A  blind      every atom whose r=2 fingerprint is known takes the mean of all
              candidate fragment charges (what quantify_r2_transfer.py does)
B  unique     only atoms whose r=2 fingerprint is claimed by ONE fragment (or by
              several that agree to <0.02 e) are replaced; the rest keep LigParGen
C  curated    like B, plus the ambiguous donor atoms resolved by hand to the
              chemically intended fragment (F3 for thiourethane N/H, F2 for urethane)
"""
from __future__ import annotations
import sys, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; PARAM = HERE.parent
sys.path.insert(0, str(PARAM)); import fragment_coverage as fc
R = 2
idx, _ = fc.build_fragment_index(R, use_refit=True)
ITP = PARAM.parent / 'project' / 'output' / 'initial_hydrogel.itp'
els, names, q0, adj = fc.read_template(ITP); n = len(els)
fps = [fc.fingerprint(i, els, adj, R) for i in range(n)]
print(f"HYDROGEL {n} atoms, formal charge 0, LigParGen net {q0.sum():+.4f} e\n")
# donor fingerprints
def donor_sets():
    th_N=[];th_H=[];ur_N=[];ur_H=[]
    for i in range(n):
        if els[i]!='N': continue
        hs=[j for j in adj[i] if els[j]=='H']
        for c in [j for j in adj[i] if els[j]=='C']:
            o=[x for x in adj[c] if els[x]=='O']; s=[x for x in adj[c] if els[x]=='S']
            if s and o and hs: th_N.append(i); th_H.append(hs[0])
            elif o and any(len(adj[x])==2 for x in o) and hs: ur_N.append(i); ur_H.append(hs[0])
    return th_N,th_H,ur_N,ur_H
thN,thH,urN,urH = donor_sets()
for lab, S in (('thiourethane N',thN),('thiourethane H(N)',thH),('urethane N',urN),('urethane H(N)',urH)):
    fp = fps[S[0]]
    hits = idx.get(fp, [])
    print(f"{lab:20s} r=2 fingerprint {fp!r}")
    print(f"{'':20s} claimed by {sorted({f for f,_,_ in hits})}  charges "
          f"{[round(q,4) for _,_,q in hits]}")
print()
def report(tag, qnew):
    resid = float(qnew.sum())
    heavy = sum(1 for e in els if e != 'H')
    print(f"{tag:9s} net {qnew.sum():+.5f} e -> residual {resid:+.5f} e to redistribute; "
          f"uniform {resid/n:+.3e} e/atom, heavy {resid/heavy:+.3e} e/atom; "
          f"|dq| total {np.abs(qnew-q0).sum():.3f} e, n_changed {(np.abs(qnew-q0)>1e-9).sum()}")
    for lab, S in (('thiourethane',thN),('urethane',urN)):
        H = thH if lab=='thiourethane' else urH
        print(f"{'':9s} {lab:13s} q(N) {q0[S].mean():+.4f}->{qnew[S].mean():+.4f}   "
              f"q(HN) {q0[H].mean():+.4f}->{qnew[H].mean():+.4f}")
    return resid
# A blind
qA = q0.copy()
for i in range(n):
    qs=[q for _,_,q in idx.get(fps[i],[]) if q is not None]
    if qs: qA[i]=float(np.mean(qs))
report('A blind', qA)
# B unique-only
qB = q0.copy(); nb=0
for i in range(n):
    hits=idx.get(fps[i],[]); qs=[q for _,_,q in hits if q is not None]
    if not qs: continue
    if len({f for f,_,_ in hits})==1 or (max(qs)-min(qs))<0.02: qB[i]=float(np.mean(qs)); nb+=1
report('B unique', qB)
# C curated: B + donors resolved to intended fragment
frag_q = {}
for name in fc.FRAGMENTS:
    try: e,a,q = fc.read_fragment(name, fc.REFIT/f'{name}.qout')
    except FileNotFoundError: continue
    frag_q[name]=(e,a,q)
def pick(fragment, fp):
    e,a,q = frag_q[fragment]
    for i in range(len(e)):
        if fc.fingerprint(i,e,a,R)==fp: return float(q[i])
    return None
qC = qB.copy()
for S,frag in ((thN,'F3_thiouret_p'),(thH,'F3_thiouret_p'),(urN,'F2_urethane_p'),(urH,'F2_urethane_p')):
    v = pick(frag, fps[S[0]])
    if v is not None:
        for i in S: qC[i]=v
report('C curated', qC)
