#!/usr/bin/env python3
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, json, os
from fit_tau1 import load_tsv, eval_opls, KCAL2KJ
HERE=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(os.path.join(HERE,'fit_coefficients.json'))); F=R['opls_F_kJ']
h,rows=load_tsv(os.path.join(HERE,'fit_curve_F3.tsv'))
phi=np.array([float(r['phi_deg']) for r in rows]); use=np.array([r['used']=='yes' for r in rows])
res=np.array([float(r['residual_kJ']) for r in rows]); qm=np.array([float(r['QM_rel_kcal']) for r in rows])*KCAL2KJ
mm=np.array([float(r['MMoff_rel_kcal']) for r in rows])*KCAL2KJ
d=np.arange(0,360.5,1.0); off=R['a0_kJ']-R['opls_builtin_constant_kJ']
fig,ax=plt.subplots(2,1,figsize=(7,7),sharex=True,height_ratios=[3,1])
o=np.argsort(phi)
ax[0].plot(phi[o],qm[o],'o-',ms=4,color='#1f77b4',label='QM (rel)')
ax[0].plot(phi[o],mm[o],'s-',ms=4,color='#888',label='MM, torsion off (hi-prec coords)')
ax[0].plot(phi[use],res[use],'ko',ms=5,label='residual = QM - MM$_{off}$ (fit target)')
ax[0].plot(phi[~use],res[~use],'rx',ms=9,mew=2,label='excluded: methyl rotamer (step 1)')
ax[0].plot(d,eval_opls(F,d)+off,'-',color='#d62728',lw=2,label='fit, OPLS 3-term')
lpg=9.623*(1+np.cos(np.radians(d)))+12.738*(1-np.cos(2*np.radians(d)))
ax[0].plot(d,lpg,'--',color='#2ca02c',lw=1.5,label='LigParGen original (removed)')
ax[0].set_ylabel('E (kJ/mol)'); ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3)
ax[0].set_title('F3 $\\tau_1$ C-S-C(=O)-N torsion fit')
ax[1].axhline(0,color='k',lw=.7)
ax[1].plot(phi[use],res[use]-(eval_opls(F,phi[use])+off),'ko-',ms=4)
ax[1].set_ylabel('fit error\n(kJ/mol)'); ax[1].set_xlabel('$\\phi$ (deg)'); ax[1].grid(alpha=.3)
ax[1].set_xticks(range(0,361,60))
plt.tight_layout(); plt.savefig(os.path.join(HERE,'fit_F3_tau1.png'),dpi=130)
# F3x validation figure
h2,v=load_tsv(os.path.join(HERE,'validation_F3x.tsv'))
fig,axs=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
for a,s in zip(axs,('orig','refwd','rerev')):
    sel=[r for r in v if r['series']==s]
    for L in sorted({r['branch'] for r in sel}):
        b=[r for r in sel if r['branch']==L]; b.sort(key=lambda r:float(r['phi_deg']))
        dm=np.mean([float(r['diff_kJ']) for r in b])
        a.plot([float(r['phi_deg']) for r in b],[float(r['residual_kJ'])-dm for r in b],'o',ms=4,label=f'branch {L} (n={len(b)})')
    a.plot(d,eval_opls(F,d),'-',color='#d62728',lw=2,label='F3 fit')
    a.set_title(f'F3x {s}'); a.set_xlabel('$\\phi$ (deg)'); a.grid(alpha=.3); a.legend(fontsize=7)
    a.set_xticks(range(0,361,90))
axs[0].set_ylabel('residual - branch mean offset (kJ/mol)')
plt.tight_layout(); plt.savefig(os.path.join(HERE,'validation_F3x.png'),dpi=130)
print("wrote fit_F3_tau1.png, validation_F3x.png")
