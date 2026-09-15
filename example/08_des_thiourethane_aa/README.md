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

`maker_shrink_n33.yaml` does the same for the experimental-length network
(cut-off electrostatics during the shrink, since a 60 nm starting box would
need a ~512³ PME grid; PME belongs to the NPT that follows). Measured: **93
steps, all accepted, no rejection, no recovery**, 60.16 → 9.320 nm — a 6.5×
contraction in box length, 270× in volume — taking 77 184 atoms from 0.004 to
**1.002 g/cm³** with bonds 0.097–0.189 nm, crosslinks at 0.168 nm mean, one
covalent component, and the same 0.133 nm closest-contact signature. That the
larger contraction needed no guard intervention either is the point of the
guards: they are there to catch the run that does. Re-run after the crossing
impropers were enabled, it reproduced step for step, and the 384 thiourethane
carbonyl centres stay near planar through the compression (mean 4.62°, max
25.5°, 4 beyond 20°) — worse than the 2.44° at construction density, as a
270× compression should be, and far better than the 49.7° the term's absence
allowed before any compression at all.

Only `shrink_output/{final.gro,state.json,history.jsonl}` are tracked; the
per-step directories are reproducible.

## The DES itself (`maker_des.yaml`)

The formulation is Hexakis : AcChCl = 1 : 6, so a 64-junction network takes
**384 acetylcholine cations and 384 chlorides**. Two obstacles, both resolved
rather than worked around:

* **LigParGen refuses ions under LBCC.** 1.14*CM1A-LBCC is defined for neutral
  molecules; the server rejects any charged species with it and accepts them
  with plain `cm1a`. That one substitution is the whole difficulty — the same
  submission that failed as `cm1abcc` succeeds as `cm1a`.
* **A monatomic ion has no SMILES geometry to optimize**, so chloride cannot
  come from LigParGen at all. It is taken from the OPLS-AA force field GROMACS
  ships (`oplsaa.ff/ffnonbonded.itp`, `opls_401`: σ = 0.441724 nm,
  ε = 0.492833 kJ/mol), quoted with that provenance.

`parameterization/build_des_components.py` writes `ACC.{itp,gro}` and
`CL.{itp,gro}`, renames the cation's types (`ac*`) so LigParGen's
per-submission `opls_8xx` namespace cannot collide with the polymer
templates', corrects the cation to exactly +1 e (LigParGen leaves +0.9998, and
384 pairs would otherwise put −0.077 e on the system), and merges both
molecules' `[ atomtypes ]` into `forcefield.itp` inside a marked fence — that
file is the only correct home for them, since GROMACS wants every atomtype
before the first moleculetype. Run it *after* `build_templates.py`, which owns
that file.

**Insertion order is not a detail.** The DES goes in while the construction
box is still dilute, and the shrink then compresses network and solvent
together; at 1 g/cm³ there is no room left to add anything.
`config/add_des.yaml` places both species in one Packmol call through the
`add_molecule` stage, which now accepts a list. The ion stage cannot serve
here: `genion` inserts by *replacing* solvent molecules, and in a DES the ions
are the solvent.

Measured: 29 952 atoms in the 17.44 nm construction box — `HYDROGEL 1,
ACC 384, CL 384` in `[ molecules ]`, and 9 984 = 384 × 26 cation atoms plus
384 chlorides in the coordinates. All EM stages converge.

Charges are the provisional part. The collaboration's DFT project has RESP
charges for acetylcholine, and replacing the charge column is the intended
upgrade; nothing else about these templates depends on it.

### Densifying the DES system

`maker_shrink_des.yaml` compresses network and solvent together to a 7.19 nm
target (223.7 kamu at ~1.0 g/cm³ — larger than the dry network's 6.34 nm
because the DES adds mass). Measured: **44 steps, all accepted, no rejection,
no recovery**, 17.44 → 7.190 nm, density 0.070 → **0.999 g/cm³**, network
bonds 0.099–0.189 nm, crosslinks at 0.169 nm mean, one covalent component,
and the composition unchanged (STR 14 400 / HEX 5 184 / ACC 9 984 / CL 384).

**What this structure does *not* yet show.** A tempting first look is where
the chlorides sit: nearest network nitrogen 0.558 nm on average, nearest
sulfur 0.642 nm — closer to N, which is the ordering the DFT work predicts on
binding energy. That inference does not survive counting the sites. The
network has 768 N and 384 S, so nitrogen is at twice the number density, and
for randomly placed anions the nearest-neighbour distance alone scales as
n^(−1/3): the expected ratio is 2^(1/3) = 1.260, while the observed ratio is
1.151. The observation is *weaker* than random placement would give, so this
frame carries no site preference at all — which is what should be expected
from Packmol placement followed by minimization, with no thermal sampling in
between. Site preference is an NPT-trajectory question (RDFs, running
coordination numbers, hydrogen-bond occupancy and residence times), not a
single-frame one.

## Does the force field agree with the DFT? (`validation/`)

Not about the ordering the project rests on. `validation/ff_pair_benchmark.py`
puts our parameters on the DFT-optimized Cl⁻ complexes and sums the exact
intermolecular energy (verified against GROMACS once its reaction-field
self-energy is accounted for). DFT puts the thiourethane N–H **2.83 kcal/mol
below** the urethane N–H; this force field puts it **0.84 kcal/mol above**,
because 1.14\*CM1A-LBCC gives the two N–H hydrogens the same charge to within
0.004 e. The thiol is correctly weakest. Everything underbinds by 11–17
kcal/mol, which a non-polarizable model in the gas phase is expected to do —
the *relative* error is the problem.

So no statement about **which donor holds chloride** may be made from MD on
this force field; the pending RESP charges are a prerequisite, not an
upgrade. Construction, topology, compression, stability and cost results are
unaffected. See `validation/README.md`.

## Charges, releases and what comes after the shrink

Three more pieces sit beside the templates, each answering one of the
integrated report's hand-over conditions (§18B.5, §18C.4, §18C.5):

* **`parameterization/apply_charges.py`** is the only way a charge set enters
  these topologies. `neutralize` fixes a molecule's sum to its formal charge
  *exactly at the written precision* (the LigParGen tables miss zero by
  −1e-4 e per molecule, which PME hides behind a background charge and grompp
  warns about); `apply` swaps in a release CSV and refuses one that is
  incomplete or sums to the wrong integer; `convert` turns a DFT `.pc_resp`
  plus the DFT team's atom mapping into that CSV; `scale` writes the ion-only
  charge-scaling variants (`ACC_f080`, `CL_f069`, ...). Everything outside
  `[ atoms ]` is written back byte-for-byte, a provenance block and a
  `.charges.json` sidecar record what moved, and `--stub-caps` keeps the
  junction's reacted-sulfur overrides (q(S)+q(H)) consistent with the
  junction's charges.
* **`parameterization/release.py`** freezes the force-field files with
  sha256s and a status (`draft` / `candidate` / `production-approved`) under
  `parameterization/releases/`. Releases so far: `v0_ligpargen_draft` (what every build above was made with), `v1_ligpargen_neutral` (the same charges neutralized), and `v1b_ligpargen_neutral_peg` (v1 plus the PEG 200 plasticizer from `build_peg_component.py`), which the tree now matches.py current` says which one a checkout is.
* **`project/config_npt/`** is the protocol after the shrink: restrained
  heating (`heat_posres.mdp`, POSRES_FC 1000 → 200 → 0, X–H constrained),
  NPT equilibration (`npt_equil.mdp`, C-rescale, PME) and production
  (`production.mdp`), driven by `run_equilibration.sh`, which generates the
  heavy-atom restraints from the ITP masses into a topology copy, continues
  the second and third heating stages from the previous checkpoint (one
  heating, stepwise restraint release), and writes each stage's
  include-resolved topology (`grompp -pp`) and drawn seed to
  `stage_manifest.tsv`. When the screen says NOT YET because a term is still
  drifting, the `extend` stage continues the same NPT by `EXTEND_NS`
  nanoseconds (`convert-tpr -extend` plus `mdrun -cpi -append`), so the
  record stays one continuous energy file and the blocks cover the whole
  trajectory rather than two short runs. Nothing in it declares equilibrium:
  `check_convergence.py` requires a complete, finite record with enough
  frames after the skip, tests both the last block and the last half against
  earlier windows, and writes a `<edr>.plateau.json` bound to the energy
  file's sha256; the `production` stage refuses to start without a matching
  PLATEAU artifact. That artifact screens thermodynamic plateaus only --
  structural equilibrium is listed in it as not assessed. The mdps carry
  **300 K as a provisional temperature**; the experimental temperature has
  not been supplied. Exercised end to end on the v1b PEG pilot cell
  (28,502 atoms): heating 3 x 100 ps, then NPT at 41.7 ns/day on 8 threads.
  The first 10 ns came back **NOT YET** -- density, volume and potential were
  all still monotonic at the 0.5% tolerance -- so that cell is being extended
  rather than read. Treat 10 ns as too short for this system.

## Any size, on whatever node is free (`sizing/`)

The four makers above each fix a size. `sizing/` makes size an input instead:
`cell_sizes.py table` prints the size ladder (`pcu` takes even repeats of at
least 4, so 4, 6, 8, ... per axis, anisotropic allowed), `cell_sizes.py emit`
writes a matched build+shrink maker pair for one size, `node_resources.py`
recommends a thread count and GPU that leave the node usable, and
`run_size.sh` measures the node at launch, builds, recomputes the shrink
target from the realized topology, shrinks, and logs the cost. The arithmetic
is checked against this example: it reproduces all three shrink targets above
(6.34, 9.32 and 7.19 nm) to 0.01 nm. See `sizing/README.md`.

Worth reading off that ladder: the experiment-facing cell (n ≈ 33, molar-ratio
conversion, AcChCl) is **28 192 atoms** at the smallest legal size --
*smaller* than the n = 33 full-conversion demonstration here, because at 1/6
conversion five sixths of the net's edges hold no prepolymer. Size is not what
stands between this example and a production run; composition is.

## Known approximations (deliberate, documented)

* **Charges are rough, and measurably so.** 1.14*CM1A-LBCC validates
  construction and *inverts the DFT donor ranking* (`validation/`);
  quantitative ion transport (the cowork's Q1–Q3) needs better charges (and
  likely charge scaling). LigParGen rounding left −0.0001 e per molecule (−0.026 e over
  the full system, −0.0096 e over the count:32 cell) in every build above,
  made under release v0; release v1 neutralizes each template exactly, so
  builds from the current tree carry 0 e and `composition_audit.py` gates on
  it.
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
* **A shrunk cell is an EM state, not an equilibrated one.** The shrink ends
  after an energy minimization at the target box; nearest non-bonded contacts
  sit near 0.13 nm and no velocity has ever been assigned to most of it. No
  property -- density included, since 1.0 g/cm³ is the target that was *set*,
  not a measurement -- may be read before restrained heating and a proper NPT.
* **Partial-conversion cells need the recovery path, and it was untested.**
  The three shrinks above never triggered the guard's NVT recovery
  (`history.jsonl`: 0 recoveries in 51, 93 and 44 steps). The first
  sub-gel cell that did exposed an unconstrained-hydrogen instability in
  `config_shrink/nvt_recovery.mdp`; see defect #31.
* **A sparse cell's build-stage EM folds some urethane N-H onto its own
  carbonyl.** An OPLS-AA polar hydrogen has no LJ parameters, so its 1-4
  Coulomb attraction to the carbamate oxygen has no repulsive floor, and at
  T = 0 in a cell where nothing competes sterically the angle bends and the
  hydrogen falls in. Measured over every strand's `N-C(=O)` angle: full
  networks are clean (min 116-118 deg, 0 below 110), the 1/6 builds are not
  (min 76 deg, 10-13% below 110), and the guarded shrink removes it entirely
  (min 118.3, mean 123.5, none below 110). So `output_partial`'s geometry is a
  construction artifact; read geometry only after the shrink. Defect #32.
* AcChCl (the DES electrolyte) and the PEG 200 plasticizer belong to the
  solvation stage, not the network build. PEG 200 (Sigma P3015, Mn 200; the
  experimental team's "PEO") is HO(CH2CH2O)4H treated as unreacted; its OH
  ends can react with isocyanate -- a stated caveat, not a model.
