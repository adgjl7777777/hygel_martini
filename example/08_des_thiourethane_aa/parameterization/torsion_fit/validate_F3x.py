#!/usr/bin/env python3
"""Validate the F3-derived tau1 torsion on the 109 F3x geometries.

Both the residual (E_QM_rel - E_MMoff_rel) and a dihedral term are defined only up
to an additive constant, and each F3x series is referenced to its own QM minimum.
So the figure of merit is the SHAPE error: RMS of (residual - fit) after removing
its mean, computed separately inside each conformer branch.

Branches are maximal runs of consecutive scan steps with BOTH |dE_QM| < 3 kcal/mol
and |dE_MM(off)| < 3 kcal/mol between neighbours.  The QM-only criterion (the one in
the READMEs) misses the rerev breaks at step 15->16 and 30->31, where the ester arm
flips with little QM cost but a large MM cost.
"""
import numpy as np, json, os
from fit_tau1 import load_tsv, eval_opls, KCAL2KJ
HERE = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(HERE,'fit_coefficients.json')))['opls_F_kJ']
hdr, rows = load_tsv(os.path.join(HERE,'residual_F3x_hiprec.tsv'))
ser = {}
for r in rows:
    ser.setdefault(r['series'], []).append((int(r['step']), float(r['phi_deg']),
        float(r['E_QM_rel_kcal']), float(r['residual_hiprec_kcal'])*KCAL2KJ,
        float(r['E_MMoff_rel_hiprec_kcal'])))
lines = ["series\tbranch\tstep\tphi_deg\tresidual_kJ\tfit_kJ\tdiff_kJ\tdiff_minus_branchmean_kJ"]
summary = {}
for s, pts in ser.items():
    pts.sort()
    lab, cur = [], 0
    for i,(st,ph,q,y,o) in enumerate(pts):
        if i and (abs(q - pts[i-1][2]) > 3.0 or abs(o - pts[i-1][4]) > 3.0): cur += 1
        lab.append(cur)
    for L in sorted(set(lab)):
        sub = [(st,ph,y) for (st,ph,q,y,o),l in zip(pts,lab) if l==L]
        d = np.array([y - eval_opls(F,ph) for _,ph,y in sub]); d0 = d - d.mean()
        for (st,ph,y),dd,dd0 in zip(sub,d,d0):
            lines.append(f"{s}\t{L}\t{st}\t{ph:.2f}\t{y:.4f}\t{eval_opls(F,ph):.4f}\t{dd:.4f}\t{dd0:.4f}")
        if len(sub) >= 6:
            summary[f'{s}/branch{L}'] = dict(n=len(sub),
                phi_range=[min(p for _,p,_ in sub), max(p for _,p,_ in sub)],
                rms_shape_kJ=float(np.sqrt((d0**2).mean())), max_shape_kJ=float(np.abs(d0).max()),
                rms_shape_kcal=float(np.sqrt((d0**2).mean())/KCAL2KJ),
                max_shape_kcal=float(np.abs(d0).max()/KCAL2KJ))
open(os.path.join(HERE,'validation_F3x.tsv'),'w').write('\n'.join(lines)+'\n')
fw = {round(ph): y for st,ph,q,y,o in ser['refwd']}
rv = {round(ph): y for st,ph,q,y,o in ser['rerev']}
com = sorted(set(fw)&set(rv)); h = np.array([abs(fw[a]-rv[a]) for a in com])
print(f"fwd/rev hysteresis, hi-prec coords, {len(com)} common angles: "
      f"mean {h.mean():.2f} kJ/mol ({h.mean()/KCAL2KJ:.2f} kcal), "
      f"max {h.max():.2f} kJ/mol ({h.max()/KCAL2KJ:.2f} kcal) at phi={com[int(h.argmax())]}")
print(f"{'branch':18s} {'n':>3s} {'phi range':>14s}  shape RMS            max")
for k,v in summary.items():
    print(f"{k:18s} {v['n']:3d} {v['phi_range'][0]:6.0f}-{v['phi_range'][1]:6.0f}  "
          f"{v['rms_shape_kJ']:6.2f} kJ ({v['rms_shape_kcal']:5.2f} kcal)  "
          f"{v['max_shape_kJ']:6.2f} kJ ({v['max_shape_kcal']:5.2f} kcal)")
json.dump(dict(summary=summary, hysteresis_mean_kJ=float(h.mean()),
               hysteresis_max_kJ=float(h.max())),
          open(os.path.join(HERE,'validation_F3x.json'),'w'), indent=2)
