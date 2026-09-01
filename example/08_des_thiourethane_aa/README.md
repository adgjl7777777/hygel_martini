# 08 — All-atom thiourethane network (DES cowork chemistry)

The first **all-atom end-to-end** exercise of the builder, on the real
chemistry of the DES-bioelectronics collaboration: a hexafunctional thiol
crosslinker joined to isocyanate-terminated strands by thiourethane bonds.

| Component | Molecule | Template |
|---|---|---|
| Junction (f=6) | dipentaerythritol hexakis(3-mercaptopropionate) (Hexakis-SH), reacted form | `project/structure/HEXR.{itp,gro}` (87 atoms) |
| Strand | thiourethane–TDI–urethane–PPG(n=3)–urethane–TDI–thiourethane | `project/structure/STR.{itp,gro}` (73 atoms) |
| Builder bond | S–C(=O) (thiourethane), 0.1715 nm / 187443 kJ/mol/nm² | inline in `config/hydrogel.yaml` |

The experimental strand is a PPG–TDI prepolymer of Mn ≈ 2300 (PO n ≈ 33);
this example uses n = 3 so a single LigParGen submission parameterizes the
whole strand. Scaling to the experimental length means tiling interior PO
units — a topology-only edit of `STR.itp`/`STR.gro`.

## How the templates were made (`parameterization/`)

1. `raw/` — untouched LigParGen server output (OPLS-AA, 1.14*CM1A-LBCC) for
   `HEX` (unreacted crosslinker), `STR` (CH₃S-capped strand model), and `LNK`
   (methyl 3-mercaptopropionate + p-tolyl isocyanate adduct: the source of
   every parameter across the S–C bond the builder forms).
2. `extend_conformer.py` (needs rdkit) — replaces the folded gas-phase STR
   conformer (BCK–BCK 0.44 nm) with an extended one (2.47 nm). Rigidly placed
   folded conformers overlap their neighbours; extended ones do not.
3. `build_templates.py` — reacts the arms (thiol H removed, charge folded
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

* **Full**: 19 584 atoms, 384 S–C crosslinks, post-EM bond lengths
  0.170–0.177 nm against b₀ = 0.1715, one connected component, every sulfur
  exactly 2-coordinated, all EM stages converge to Fmax < 500
  (Epot ≈ −8.9×10⁴ kJ/mol; an earlier revision quoted −1.4×10⁵, which was
  the angle-less failed build one defect earlier — lower because a missing
  angle term costs nothing).
* **Partial 1/6**: 7 612 atoms, 56 crosslinks, sulfur coordination
  {1: 328, 2: 56} (85.4 % unreacted vs 5/6 expected), 36 fragments. The
  fragmentation is physics, not a defect: Flory–Stockmayer for A₆+B₂ gels at
  p = 1/(f−1) = 0.2, and the experimental 1/6 sits *below* it — the real
  material's integrity comes from the DES hydrogen-bond network on top of the
  covalent pieces.

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
* **Partial conversion leaves reacted-form sulfurs dangling.** The junction
  template is the fully reacted form, so unreacted arms have no thiol
  hydrogen. Topology/mechanics benchmark only; chemically faithful partial
  conversion needs per-stub cap atoms, which the builder does not have yet.
* **The carbonyl planarity improper (N–C(=O)–S=O) is lost** at every formed
  bond: its four atoms span two molecules and the builder generates only
  proper dihedrals across new bonds.
* **No rewiring.** A rigid molecule has one length; rewired (heterogeneous)
  junction gaps cannot be spanned. The layout refuses the combination.
* AcChCl (the DES electrolyte) and the PEO plasticizer belong to the
  solvation stage, not the network build.
