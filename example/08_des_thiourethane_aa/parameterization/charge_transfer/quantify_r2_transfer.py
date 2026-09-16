#!/usr/bin/env python3
"""How much charge would a radius-2 transfer of the Boltzmann-refit RESP charges move?

Reuses fragment_coverage.py's fingerprint machinery, but instead of counting
coverage it actually performs the substitution and measures the consequences:

  * per moleculetype: net charge after substitution, i.e. the residual that
    apply_charges.py 'neutralize' would have to redistribute;
  * where that residual lands under the two rules apply_charges.py offers
    (uniform over all atoms, heavy over non-hydrogens);
  * the charge change at the atoms the project is actually measuring - the
    thiourethane and urethane N and H(N) donors.

Read-only on /nas_3 (fragment geometries only).  No new QM.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PARAM = HERE.parent
sys.path.insert(0, str(PARAM))
import fragment_coverage as fc   # noqa: E402

RADIUS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TARGETS = {
    'HYDROGEL': PARAM.parent / 'project' / 'output' / 'initial_hydrogel.itp',
    'ACC':      PARAM.parent / 'project' / 'structure' / 'ACC.itp',
    'CL':       PARAM.parent / 'project' / 'structure' / 'CL.itp',
    'PEG':      PARAM.parent / 'project' / 'structure' / 'PEG.itp',
}
FORMAL = {'HYDROGEL': 0, 'ACC': +1, 'CL': -1, 'PEG': 0}

idx, loaded = fc.build_fragment_index(RADIUS, use_refit=True)
print(f"radius {RADIUS}: {len(idx)} distinct fragment environments "
      f"over {sum(loaded.values())} atoms from {len(loaded)} fragments\n")

report = {'radius': RADIUS, 'targets': {}}
for tag, itp in TARGETS.items():
    if not itp.exists():
        print(f"{tag}: no ITP at {itp}"); continue
    els, names, q0, adj = fc.read_template(itp)
    n = len(els)
    qnew = q0.copy(); status = []
    for i in range(n):
        hits = idx.get(fc.fingerprint(i, els, adj, RADIUS), [])
        qs = [q for _, _, q in hits if q is not None]
        if not qs:
            status.append('orphan'); continue
        spread = max(qs) - min(qs)
        qnew[i] = float(np.mean(qs))
        status.append('unique' if (len({f for f, _, _ in hits}) == 1 or spread < 0.02) else 'ambiguous')
    cnt = Counter(status)
    net = float(qnew.sum()); resid = net - FORMAL[tag]
    heavy = [i for i in range(n) if els[i] != 'H']
    moved = float(np.abs(qnew - q0).sum())
    rep = dict(n_atoms=n, counts=dict(cnt),
               net_before=float(q0.sum()), net_after=net,
               residual_to_redistribute_e=resid,
               abs_charge_moved_e=moved,
               per_atom_uniform_e=resid / n,
               per_atom_heavy_e=resid / len(heavy) if heavy else None,
               n_heavy=len(heavy))
    report['targets'][tag] = rep
    print(f"{tag:9s} n={n:6d}  covered {cnt['unique']+cnt['ambiguous']:6d} "
          f"(uniq {cnt['unique']:6d}, ambig {cnt['ambiguous']:6d}), orphan {cnt['orphan']:6d}")
    print(f"{'':9s} net {float(q0.sum()):+.4f} -> {net:+.4f} e ; residual vs formal "
          f"{FORMAL[tag]:+d}: {resid:+.4f} e")
    print(f"{'':9s} |dq| summed over atoms = {moved:.3f} e ; "
          f"neutralisation smear: uniform {resid/n:+.2e} e/atom, "
          f"heavy {resid/len(heavy):+.2e} e/atom")
    # donors
    donors = []
    for i in range(n):
        if els[i] != 'N': continue
        hs = [j for j in adj[i] if els[j] == 'H']
        cs = [j for j in adj[i] if els[j] == 'C']
        kind = None
        for c in cs:
            o = [x for x in adj[c] if els[x] == 'O']; s = [x for x in adj[c] if els[x] == 'S']
            oo = [x for x in o if len(adj[x]) == 2]
            if s and o: kind = 'thiourethane'
            elif o and oo: kind = 'urethane'
        if kind and hs:
            donors.append((kind, names[i], float(q0[i]), float(qnew[i]),
                           names[hs[0]], float(q0[hs[0]]), float(qnew[hs[0]]),
                           status[i], status[hs[0]]))
    agg = defaultdict(list)
    for k, nm, a, b, hn, ha, hb, st, sth in donors: agg[k].append((a, b, ha, hb, st, sth))
    for k, v in agg.items():
        a = np.array([x[0] for x in v]); b = np.array([x[1] for x in v])
        ha = np.array([x[2] for x in v]); hb = np.array([x[3] for x in v])
        print(f"{'':9s} {k:13s} n={len(v):4d}  q(N) {a.mean():+.4f} -> {b.mean():+.4f} "
              f"(d={b.mean()-a.mean():+.4f})   q(HN) {ha.mean():+.4f} -> {hb.mean():+.4f} "
              f"(d={hb.mean()-ha.mean():+.4f})  status N/H = "
              f"{Counter(x[4] for x in v).most_common(1)[0][0]}/{Counter(x[5] for x in v).most_common(1)[0][0]}")
        rep.setdefault('donors', {})[k] = dict(n=len(v),
            qN_before=float(a.mean()), qN_after=float(b.mean()),
            qHN_before=float(ha.mean()), qHN_after=float(hb.mean()),
            status_N=Counter(x[4] for x in v).most_common(1)[0][0],
            status_H=Counter(x[5] for x in v).most_common(1)[0][0])
    print()
(HERE / f'r{RADIUS}_transfer_summary.json').write_text(json.dumps(report, indent=2) + '\n')
print('wrote', HERE / f'r{RADIUS}_transfer_summary.json')
