# Does the force field agree with the DFT it is meant to model?

The project's mechanism is an ordering. Chloride should prefer the new
thiourethane N–H over the urethane N–H over the residual thiol S–H, and every
MD observable we plan to measure — residence time, donor switching,
conductivity — is downstream of that ordering being right in the model.

The DFT side established it. The force field had never been asked.

`ff_pair_benchmark.py` asks. It is the first half of the independent review's
Major 4 gate (2026-09-05), and it needed no new DFT and no RESP release: the
complexes are optimized and their counterpoise interaction energies exist.

```
python3 ff_pair_benchmark.py            # uses fragments/ and raw/LNK.itp
```

## What it does

For each complex, the DFT-optimized geometry is used **exactly as it is**.
Our force field's atoms are placed on those coordinates by *molecular graph*
rather than by file order, so no re-parameterization or reordering can
misalign them, and the intermolecular non-bonded energy is summed in closed
form — two separate molecules share no exclusions and no 1–4 scaling, so
there is no cutoff, no PME and nothing to tune. That single point is
comparable to DFT's `ΔE_int(CP)`, the interaction at the complex geometry.
A rigid slide of Cl⁻ along the X–H···Cl axis then says where *this* force
field would put the contact.

**Verified against GROMACS.** Same system, same coordinates: LJ agrees to
0.03 kJ/mol. Coulomb appeared to differ by 19.85 kJ/mol until the cause was
identified — GROMACS's reaction-field **self-energy**, −½·f·q²/r_c, which for
q = −1 at r_c = 3.5 nm is exactly 19.848 kJ/mol. That term is an artifact of
the cutoff scheme, not part of a pair interaction. With it accounted for, the
two agree.

## The result

| complex | donor | MM @ DFT geom | DFT | q(H) | H···Cl (Å) | MM min (Å) |
|---|---|---:|---:|---:|---:|---:|
| C1 | thiol S–H | −5.20 | −18.34 | +0.1555 | 2.205 | 2.605 |
| C2 | urethane N–H | **−11.25** | −22.69 | +0.5082 | 2.108 | 2.358 |
| C3 | thiourethane N–H | **−10.41** | **−25.52** | +0.5041 | 2.046 | 2.296 |
| C3x | thiourethane N–H (ester arm, = `LNK`) | −10.73 | −27.40 | +0.4986 | 2.030 | 2.330 |

kcal/mol. Two things are wrong and one is right.

**Right:** the thiol is weakest, clearly, in both.

**Wrong, and it is the one that matters:** DFT puts the thiourethane N–H
**2.83 kcal/mol below** the urethane N–H. This force field puts it
**0.84 kcal/mol above** — the ordering the whole mechanism rests on is
inverted. Letting each donor relax to its own MM optimum does not rescue it
(−12.36 vs −12.80; the extended thiourethane only just edges ahead at
−12.92).

**Why:** 1.14\*CM1A-LBCC gives the two N–H hydrogens the *same* charge —
+0.5082 urethane, +0.5041 thiourethane, a difference of 0.004 e in the wrong
direction. The O→S substitution next to the carbonyl changes N–H acidity and
chloride affinity by ~3 kcal/mol in DFT and by nothing in the charge model.
Our shipped templates carry the same defect: `STR`'s urethane H is +0.5249
and `LNK`'s thiourethane H is +0.4986, so in the built network the *urethane*
is the preferred chloride site.

**Also wrong, but expected:** everything underbinds by 11–17 kcal/mol and
wants Cl⁻ 0.25–0.40 Å further out. A non-polarizable fixed-charge model has
no induction, and gas-phase ion–neutral binding is where that hurts most.
OPLS/CM1A was calibrated for condensed phases, so the absolute gap alone
would not be damning. The *relative* error is, because it is not uniform:
+11.4 for urethane against +15.1 for thiourethane.

## What follows

The pending RESP release stops being an improvement and becomes a
prerequisite. RESP is fit to the QM electrostatic potential, which contains
the O→S effect by construction, so it is the natural fix — but that has to be
demonstrated, not assumed. `tests/test_ff_pair_benchmark.py` pins the current
inversion precisely so that a new charge set makes it fail and forces the
comparison to be re-read.

Until then, no statement about **which donor holds chloride** may be made
from MD on this force field. Statements about construction, topology,
compression, stability and cost are unaffected — none of them depend on this
ordering.

## Scope

One geometry and one conformer per complex; gas phase (the DFT ranking does
survive CPCM screening to ε = 15, per the integrated report §10.2, but this
comparison is gas-phase); ΔE_int at fixed geometry, not free energy; charges
computed by LigParGen at its own optimized geometry and placed on the DFT
coordinates, which is what a fixed-charge model licenses and exactly how
every template in this example was made.

`fragments/` holds the three model-compound force fields and the exact PDBs
submitted, so this is reproducible without a network call. F3x needs no file:
that fragment **is** `parameterization/raw/LNK.itp`, atom for atom — so the
worst-performing case is the parameter set the builder actually uses for
every crosslink it forms.
