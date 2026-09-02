# 08 — All-atom thiourethane network (DES cowork chemistry)

The first **all-atom end-to-end** exercise of the builder, on the real
chemistry of the DES-bioelectronics collaboration: a hexafunctional thiol
crosslinker joined to isocyanate-terminated strands by thiourethane bonds.

| Component | Molecule | Template |
|---|---|---|
| Junction (f=6) | dipentaerythritol hexakis(3-mercaptopropionate) (Hexakis-SH), unreacted form + per-stub caps | `project/structure/HEXU.{itp,gro}` (93 atoms) |
| Strand | thiourethane–TDI–urethane–PPG(n=3)–urethane–TDI–thiourethane | `project/structure/STR.{itp,gro}` (73 atoms) |
| Builder bond | S–C(=O) (thiourethane), 0.1715 nm / 187443 kJ/mol/nm² | inline in `config/hydrogel.yaml` |

The experimental strand is a PPG–TDI prepolymer of Mn ≈ 2300 (PO n ≈ 33).
`maker.yaml` uses n = 3, which one LigParGen submission can parameterize
whole; `maker_n33.yaml` uses the experimental length, tiled from that
parameterization by `parameterization/tile_ppg.py` (see **Strand length**
below).

## How the templates were made (`parameterization/`)

1. `raw/` — untouched LigParGen server output (OPLS-AA, 1.14*CM1A-LBCC) for
   `HEX` (unreacted crosslinker), `STR` (CH₃S-capped strand model), and `LNK`
   (methyl 3-mercaptopropionate + p-tolyl isocyanate adduct: the source of
   every parameter across the S–C bond the builder forms).
   The junction template stays in its **unreacted** form (all six thiol
   hydrogens present): each hydrogen is a per-stub *cap atom*
   (`stub_caps:` in `config/hydrogel.yaml`, generated as
   `structure/hydrogel_stubs_snippet.yaml`), deleted — with the sulfur
   re-typed to `lkS` and its charge set to qS+qH, exactly neutral — only on
   arms that chemically react. An unreacted arm keeps real S–H.
2. `extend_conformer.py` (needs rdkit) — replaces the folded gas-phase STR
   conformer (BCK–BCK 0.44 nm) with an extended one (2.47 nm). Rigidly placed
   folded conformers overlap their neighbours; extended ones do not.
3. `tile_ppg.py --n 33` — replicates the PPG block's middle PO unit, with its
   parameters, up to the experimental chain length, then `relax_tiled.sh`
   relaxes the result in vacuo. See **Strand length**.
4. `build_templates.py` — reacts the arms (thiol H removed, charge folded
   into S, S re-typed to the thiourethane sulfur `lkS`, residue `BCK`),
   removes the strand's CH₃S caps (cap charge folded into the carbonyl
   carbon, renamed `BCK1`/`BCK2`), renames the two LigParGen 800-series type
   namespaces apart (`hx*`/`st*`), and assembles `forcefield.itp`:
   `[defaults]` + merged `[atomtypes]` + the `[angletypes]`/`[dihedraltypes]`
   entries that resolve every parameterless term the builder emits.

## Running

```bash
# full conversion (every SH reacted): one covalent network
PYTHONPATH=$REPO python -m hygel_martini.hydrogel_builder example/08_des_thiourethane_aa/project/maker.yaml

# experimental stoichiometry (SH conversion 1/6): sub-gel fragments
PYTHONPATH=$REPO python -m hygel_martini.hydrogel_builder example/08_des_thiourethane_aa/project/maker_partial.yaml
```

Measured on the shipped configuration (pcu 4×4×4, a = 4.36 nm):

* **Full** (n = 3): 19 584 atoms, 384 S–C crosslinks, post-EM bond lengths
  0.170–0.177 nm against b₀ = 0.1715, one connected component, every sulfur
  exactly 2-coordinated in reacted form (no thiol hydrogens survive full
  conversion), all EM stages converge to Fmax < 500 (Epot ≈ −8.7×10⁴
  kJ/mol; an earlier revision quoted −1.4×10⁵, which was the angle-less
  failed build one defect earlier — lower because a missing angle term
  costs nothing).
* **Partial 1/6**: 7 940 atoms, 56 crosslinks, and chemically faithful
  sulfur states: the 328 unreacted arms keep **real S–H** (post-EM S–H at
  0.134 nm against b₀ = 0.1336) with thiol-form charges and types, while
  the 56 reacted sulfurs carry the thiourethane form. 36 fragments. The
  fragmentation is physics, not a defect: Flory–Stockmayer for A₆+B₂ gels at
  p = 1/(f−1) = 0.2, and the experimental 1/6 sits *below* it — the real
  material's integrity comes from the DES hydrogen-bond network on top of the
  covalent pieces.
* **Full at the experimental length** (n = 33, `maker_n33.yaml`): 77 184 atoms
  in a 60.2 nm box, 384 S–C crosslinks at 0.1731 nm mean, one connected
  component, every sulfur 2-coordinated, all bonds 0.099–0.186 nm, net charge
  −0.026 e (LigParGen rounding), all EM stages converge to Fmax < 500. Build
  time ~1 min, which is why only the relaxed structure
  (`output_n33/final_system_no_ions_geo_opt/em.gro`) is tracked here rather
  than the whole 130 MB output tree.

## Strand length: n = 3 vs the experimental n ≈ 33

LigParGen has an atom-count ceiling, so the strand is parameterized at n = 3
and tiled. `tile_ppg.py` replicates the *middle* PO unit and refuses to run
unless the interior is verifiably regular: the units must be
parameter-identical position by position (element, mass, σ, ε, and the inline
bonded parameters — never the type *names*, since LigParGen mints a unique
name per atom), no bonded term may touch three units, and the per-unit term
counts must agree. Copies are placed by the rigid screw carrying the template
unit's three backbone atoms onto its successor's, which reproduces the
parameterized junction geometry exactly (168.2° rotation, 0.358 nm rise per
unit — a near-twofold, extended helix).

Measured for `STR_n33`: 373 atoms, net charge preserved to 1e-6 e, bond
lengths 0.101–0.154 nm, no nonbonded contact below 0.22 nm, BCK1–BCK2 span
13.152 nm after vacuum EM (converged, and the extension survives it — an EM
has no thermal motion to coil the chain with).

Two honest approximations, both printed when the script runs:

* **Charge transferability.** The same unit position differs by up to
  0.065 e between the first, middle and last PO unit — a monotonic
  1.14*CM1A end-effect. Copies take the middle unit's values (the most
  bulk-like of the three); the parameterized end units keep their own.
* **Charge neutrality.** The middle unit sums to +0.0196 e, so 30 verbatim
  copies would hand the molecule +0.59 e and the network +113 e. Each copy is
  neutralized by spreading −0.002 e over its ten atoms, well under the charge
  model's own uncertainty, and the molecule keeps its original net charge
  exactly.

### Density: why neither build is a melt, and what that implies

| build | strand span | cell_parameter | box | atoms | density |
|---|---|---|---|---|---|
| `maker.yaml` (n = 3) | 2.473 nm | 4.36 nm | 17.4 nm | 19 584 | 0.048 g/cm³ |
| `maker_n33.yaml` (n = 33) | 13.152 nm | 15.04 nm | 60.2 nm | 77 184 | 0.004 g/cm³ |

(Atom counts are `strands x strand atoms + junctions x 93 - removed caps`:
at full conversion all 6 x 64 cap hydrogens go, so n = 33 gives
192x373 + 64x93 - 384 = 77 184.)

Both are one to two orders of magnitude below a polymer melt (~1 g/cm³), and
that is by design rather than a defect. A construction box is sized by the
geometry it has to place: an ideal net puts one extended strand per lattice
edge, so `cell_parameter` follows the strand's end-to-end distance. Reaching
1.0 g/cm³ with the n = 33 strand needs `cell_parameter` ≈ 2.33 nm — a junction
spacing *one sixth* of the strand's 11.8 nm contour, i.e. coiled strands. A
rigidly placed conformer is not the thing that produces those coils, and it
does not have to be.

**Densification is a separate, existing stage**, not a missing feature:
staged minimization (`soft_em`), settling MD (`soft_md`) and guarded
shrink–minimization (`hard_em_shrink`) live in `hygel_martini/relax/` with
example 05 as their driver. The shrink contracts box and coordinates toward a
formulation-specific `target_box_nm` in small guarded steps, rolls back to the
last valid state on a failed guard, and is exactly the step that coils the
extended chains. The Series-01 systems were prepared along this path, over a
box contraction considerably larger than the ~6.5x in box length this one
needs, and hundreds of ns of NPT follow it before anything is measured.

So what these builds are: **correct topology and chemistry at construction
density**, the intended input to that workflow. Read no transport,
mechanical, or coordination number off them directly.

### Densification, run rather than argued (`maker_shrink.yaml`)

`maker_shrink.yaml` drives the guarded shrink on the n = 3 build, from its
17.44 nm construction box to a 6.34 nm target (≈1.0 g/cm³ for this
composition), 2 % of box length per step:

```bash
PYTHONPATH=$REPO python -m hygel_martini.hydrogel_builder.relax     example/08_des_thiourethane_aa/project/maker_shrink.yaml
```

Measured outcome — 51 steps, **all accepted, no guard rejection, no NVT
recovery**, ending exactly on target:

| | construction | post-shrink |
|---|---|---|
| box | 17.440 nm | 6.340 nm |
| density | 0.048 g/cm³ | **1.003 g/cm³** |
| bond lengths | 0.099–0.187 nm | 0.099–0.186 nm |
| crosslink S–C | mean 0.174 nm | mean 0.169 nm (b₀ = 0.1715) |
| covalent components | 1 | 1 |
| closest nonbonded | 0.193 nm | 0.137 nm |

A 21× compression in density leaves the bonded structure untouched: no bond
stretched or crushed, the crosslinks still sit on their equilibrium length,
and the network is still one covalent component. The one number that moves the
wrong way is the closest nonbonded contact (0.137 nm), which says exactly what
it should — this is an energy-minimized structure at melt density, not a
thermally equilibrated one. Long NPT is the next stage, and in the Series-01
systems it is also what erases the builder's lattice pattern.

Only `shrink_output/{final.gro,state.json,history.jsonl}` are tracked; the
per-step directories are reproducible.

## Known approximations (deliberate, documented)

* **Charges are rough.** 1.14*CM1A-LBCC validates construction; quantitative
  ion transport (the cowork's Q1–Q3) needs better charges (and likely charge
  scaling) via `param_opt`. LigParGen rounding leaves −0.0001 e per molecule
  (−0.026 e over the full system) under PME with a uniform background;
  `grompp_maxwarn: 2` covers the resulting warning.
* **The stub-sulfur angle parameters live in `forcefield.itp`.** The linker
  loader keeps stub atoms out of a template's own angle list, so the 18
  S-adjacent angles per junction are emitted parameterless and resolve from
  `[ angletypes ]`. `build_templates.py` generates those entries from the
  same HEXR rows, so they agree by construction — but if you edit one file,
  edit both: grompp resolves, it never compares.
* **Which arms react is chosen by the layout, not by geometry.** Reacted
  arm positions are drawn per junction from a stream derived from the
  conversion seed, before coordinates exist; the router then bonds only
  those arms, so an initially farther arm may take the bond and relax under
  EM. Physically this mimics reaction randomness rather than
  diffusion-controlled selectivity.
* **The carbonyl planarity improper (N–C(=O)–S=O) is generated by the
  builder**, since its four atoms span two molecules and neither template can
  declare it (in the strand that carbon is two-coordinate until the sulfur
  arrives). `junction_bonded_generation.impropers` emits one per centre the
  new bond completes, with the LNK model compound's own parameters. Measured
  at the 384 centres, post-EM out-of-plane deviation: **mean 8.69° / max
  49.66° without it, mean 2.44° / max 8.22° with it** (centres beyond 20°:
  35 → 0). The remaining approximation is that those parameters come from a
  model compound rather than from this exact environment.
* **No rewiring.** A rigid molecule has one length; rewired (heterogeneous)
  junction gaps cannot be spanned. The layout refuses the combination.
* AcChCl (the DES electrolyte) and the PEO plasticizer belong to the
  solvation stage, not the network build.
