#!/usr/bin/env python3
"""Fit the F3 tau1 (C-S-C(=O)-N) torsion to the QM-minus-MM(torsion-off) residual.

Input  : residual_F3_hiprec.tsv   (produced by recompute_residual_hiprec.py)
Output : fit_coefficients.json / .md table, curve TSVs, validation against F3x.

Model  : E(phi) = a0 + sum_n a_n cos(n phi)   [sine terms checked and found ~0]
GROMACS OPLS (funct 5): V = 1/2[F1(1+cos p) + F2(1-cos 2p) + F3(1+cos 3p) + F4(1-cos 4p)]
  => F1 = 2 a1, F2 = -2 a2, F3 = 2 a3, F4 = -2 a4      (a_n in kJ/mol)
GROMACS RB  (funct 3): V = sum_n C_n cos^n(psi), psi = phi - 180
  C0 = F2 + (F1+F3)/2 ; C1 = (-F1+3F3)/2 ; C2 = -F2+4F4 ; C3 = -2F3 ; C4 = -4F4 ; C5 = 0
"""
import numpy as np, json, sys, os
KCAL2KJ = 4.184
HERE = os.path.dirname(os.path.abspath(__file__))

def design(phi_deg, nmax, sines=False):
    p = np.radians(phi_deg)
    cols = [np.ones_like(p)]
    for n in range(1, nmax+1):
        cols.append(np.cos(n*p))
        if sines: cols.append(np.sin(n*p))
    return np.column_stack(cols)

def fit(phi, y, nmax, sines=False):
    A = design(phi, nmax, sines)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ c
    return c, pred, y - pred

def opls_from_a(a):            # a = [a1..a4] in kJ/mol
    a = list(a) + [0.0]*(4-len(a))
    return [2*a[0], -2*a[1], 2*a[2], -2*a[3]]

def rb_from_opls(F):
    F1, F2, F3, F4 = F
    return [F2 + 0.5*(F1+F3), 0.5*(-F1 + 3*F3), -F2 + 4*F4, -2*F3, -4*F4, 0.0]

def eval_opls(F, phi_deg):
    p = np.radians(phi_deg)
    return 0.5*(F[0]*(1+np.cos(p)) + F[1]*(1-np.cos(2*p)) + F[2]*(1+np.cos(3*p)) + F[3]*(1-np.cos(4*p)))

def eval_rb(C, phi_deg):
    c = np.cos(np.radians(phi_deg) - np.pi)
    return sum(Cn*c**n for n, Cn in enumerate(C))

def load_tsv(path):
    rows = [l.split('\t') for l in open(path).read().strip().split('\n')]
    hdr = rows[0]
    return hdr, [dict(zip(hdr, r)) for r in rows[1:]]

if __name__ == '__main__':
    hdr, rows = load_tsv(os.path.join(HERE, 'residual_F3_hiprec.tsv'))
    keep = [r for r in rows if int(r['step']) != 1]          # step 1 = methyl-rotamer outlier
    phi = np.array([float(r['phi_deg']) for r in keep])
    res_kcal = np.array([float(r['residual_hiprec_kcal']) for r in keep])
    y = res_kcal * KCAL2KJ                                    # kJ/mol -- fit in kJ throughout

    print(f"{len(keep)} points used (step 1 / phi=180 excluded).")
    report = {}
    # --- sine check -------------------------------------------------------
    c_s, pred_s, r_s = fit(phi, y, 4, sines=True)
    sin_amp = [abs(c_s[2*n]) for n in range(1, 5)]
    print("sine amplitudes b1..b4 (kJ/mol):", " ".join(f"{v:.3f}" for v in sin_amp))
    report['sine_amplitudes_kJ'] = sin_amp

    # --- term-count scan --------------------------------------------------
    scan = []
    for nmax in range(1, 7):
        c, pred, r = fit(phi, y, nmax)
        rmse = float(np.sqrt(np.mean(r**2))); mx = float(np.max(np.abs(r)))
        scan.append(dict(nmax=nmax, rmse_kJ=rmse, max_kJ=mx,
                         rmse_kcal=rmse/KCAL2KJ, max_kcal=mx/KCAL2KJ,
                         a=[float(v) for v in c]))
        print(f"  n<={nmax}: RMSE {rmse:7.3f} kJ/mol ({rmse/KCAL2KJ:6.3f} kcal)  max|r| {mx:7.3f} kJ/mol")
    report['term_scan'] = scan

    NMAX = 3   # see report: n=4 does not improve max|resid|; n>=5 fits fragment-specific relaxation
    c, pred, r = fit(phi, y, NMAX)
    a0, a = float(c[0]), [float(v) for v in c[1:]]
    F = opls_from_a(a); C = rb_from_opls(F)
    # additive constant check: OPLS form carries its own constant
    const_opls = 0.5*sum(F)
    report.update(dict(nmax=NMAX, a0_kJ=a0, a_kJ=a, opls_F_kJ=F, rb_C_kJ=C,
                       opls_builtin_constant_kJ=const_opls,
                       constant_offset_kJ=a0 - const_opls,
                       rmse_kJ=float(np.sqrt(np.mean(r**2))),
                       max_resid_kJ=float(np.max(np.abs(r)))))
    print("\nOPLS F1..F4 (kJ/mol):", " ".join(f"{v:9.4f}" for v in F))
    print("RB   C0..C5 (kJ/mol):", " ".join(f"{v:9.4f}" for v in C))
    print("consistency OPLS vs RB max diff:",
          float(np.max(np.abs(eval_opls(F, phi) - eval_rb(C, phi)))), "kJ/mol")

    # --- curve TSV --------------------------------------------------------
    with open(os.path.join(HERE, 'fit_curve_F3.tsv'), 'w') as fh:
        fh.write("step\tphi_deg\tQM_rel_kcal\tMMoff_rel_kcal\tresidual_kcal\tresidual_kJ\t"
                 "fit_kJ\tfit_minus_residual_kJ\tused\n")
        for row in rows:
            p = float(row['phi_deg']); used = int(row['step']) != 1
            rk = float(row['residual_hiprec_kcal'])
            f_ = eval_opls(F, p) + (a0 - const_opls)
            fh.write(f"{row['step']}\t{p:.2f}\t{float(row['E_QM_rel_kcal']):.4f}\t"
                     f"{float(row['E_MMoff_rel_hiprec_kcal']):.4f}\t{rk:.4f}\t{rk*KCAL2KJ:.4f}\t"
                     f"{f_:.4f}\t{f_-rk*KCAL2KJ:.4f}\t{'yes' if used else 'no(methyl rotamer)'}\n")
    # dense curve
    with open(os.path.join(HERE, 'fit_curve_dense.tsv'), 'w') as fh:
        fh.write("phi_deg\tE_OPLS_kJ\tE_RB_kJ\n")
        for p in np.arange(0, 360.5, 2.0):
            fh.write(f"{p:.1f}\t{eval_opls(F,p):.4f}\t{eval_rb(C,p):.4f}\n")
    # alternative 4-term set, for completeness
    c4, p4, r4 = fit(phi, y, 4)
    F4set = opls_from_a([float(v) for v in c4[1:]])
    report['alt_nmax4'] = dict(opls_F_kJ=F4set, rb_C_kJ=rb_from_opls(F4set),
                               rmse_kJ=float(np.sqrt(np.mean(r4**2))),
                               max_resid_kJ=float(np.max(np.abs(r4))))
    print("alt n<=4 OPLS F1..F4 (kJ/mol):", " ".join(f"{v:9.4f}" for v in F4set))
    print("alt n<=4 RB   C0..C5 (kJ/mol):", " ".join(f"{v:9.4f}" for v in rb_from_opls(F4set)))
    json.dump(report, open(os.path.join(HERE, 'fit_coefficients.json'), 'w'), indent=2)
    print("\nwrote fit_curve_F3.tsv, fit_curve_dense.tsv, fit_coefficients.json")
