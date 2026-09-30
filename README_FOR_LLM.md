# README_FOR_LLM

Orientation for a language model that has just been given this repository and
has to do real work in it. Written in English because the other technical docs
here are (`docs/DEFECTS_FOUND_AND_FIXED.md`,
`docs/GENERAL_FUNCTIONALITY_NETWORKS.md`); the user-facing `README.md` and
`START_HERE_ko.md` are Korean and say the same things for humans.

Everything below is either checkable in the tree or labelled as a limit. If a
statement here disagrees with the code, the code is right and this file is a
bug — fix it.

---

## 0. Read this much, then stop reading

| You were asked to… | Read |
|---|---|
| understand the package at all | §1–§4 of this file |
| find a specific function | `docs/FUNCTION_REFERENCE.md` (generated, kept current by a test) |
| work out where a problem lives | §12 |
| build a network | §5 (config system), §6 (series), the target example's own README |
| change builder behaviour | §7 (invariants) **before** editing, then §8 (how to verify) |
| interpret a build's output | §9 (claim boundary) — this is the part most easily got wrong |
| know what is finished | §10 (current state) |
| quote a number | §13. Do not invent numbers; every number worth citing is in a file |

**To look up what a specific function does**, use
[`docs/FUNCTION_REFERENCE.md`](docs/FUNCTION_REFERENCE.md) — 164 modules, 77
classes, 950 functions and methods, with signature, docstring summary,
returns, raises, side-effect hints and call list for each. It is **generated
from source** (`tools/gen_function_reference.py --write`) and a test fails when
it drifts, so it is safe to trust in a way its predecessor was not: the
Series-01 version, `docs/archive/README_detailed_for_llm_series01.md`, was
generated once and ended up with zero mentions of every module this branch
added. That file is now history only, as is the rest of `docs/archive/`.

---

## 1. What this package is

A research pipeline for **coarse-grained and all-atom polymer networks**
(hydrogels, elastomers). It does three things, in this order:

1. **`param_opt`** — turn quantum chemistry into force-field parameters
   (xTB/ORCA → Bartender → screened Martini ITP; OPLS routes too).
2. **`hydrogel_builder`** — plan a periodic network **as a graph**, materialize
   it as coordinates and a GROMACS topology, then **audit that the topology it
   wrote is the graph it planned**. Includes post-build relaxation.
3. **`property_extract`** — measure structure, state, transport, mechanics from
   trajectories, behind requirement/manifest gates.

`core/` under all three: PBC/minimum image (triclinic), GRO reader, ITP parser,
physics constants.

**The design principle, and it is load-bearing:** *fail loudly rather than
succeed wrongly.* A silent success with a wrong mass, a dropped bonded term or
a rounded-away charge is treated as the worst possible outcome. Thirty-two such
defects have been found and are recorded with their fixing commits in
`docs/DEFECTS_FOUND_AND_FIXED.md`. **Read a few of those entries.** They teach
the failure modes of this codebase faster than the source does, and they are
the reason many guards look paranoid.

### What it is not

Not a force field, and not a validated predictor of material properties. It
constructs and audits models; whether a model's *energetics* are right is a
separate question the package can help ask (see §9) but does not answer by
building successfully.

---

## 2. The mental model: plan → materialize → audit

Most bugs in this domain come from conflating the three.

```
   graph plan                coordinates + topology            audit
   ────────────              ──────────────────────            ─────
   which junction            where atoms sit, which            did the written
   connects to which,        [bonds]/[angles]/[dihedrals]      topology realize
   via which arm             /[pairs] got written              the planned graph?
```

* The **plan** is combinatorial: a periodic net (`dia`, `pcu`), a transition
  system that pairs arms at each junction, optional span-constrained rewiring.
  No coordinates exist yet.
* **Materialization** places coordinates and emits bonded terms. Some terms come
  from templates, some are generated across builder-formed bonds.
* The **audit** compares the two. Endpoint audit, bonded-term completeness,
  component count, loop-order spectrum, charge sum.

When something is wrong, decide which layer first. A wrong loop-order histogram
is a plan problem; a missing angle is a materialization problem; a build that
"worked" but has 36 components is neither — it may be correct physics (see §7,
gel point).

---

## 3. Repository map

```
hygel_martini/
  core/                    pbc · gro · itp · physics · config (includes/merge)
  param_opt/
    qm_to_opls/            stage 01
    opls_to_martini/       stage 02  (reuses an existing OPLS trajectory)
    qm_to_martini/         stage 03  (+ protocol/ = E0–E6 decision protocol)
    bead_generator/        bead selector
  hydrogel_builder/
    config_params/         maker.yaml load · merge · VALIDATE · orchestration
                           read_json.py is the big one: workflow + validation
    core_utils/layout/     nets.py (dia/pcu definitions + repeat rules)
                           net_layout.py (plan on a net; conversion selection)
                           rewire.py (span-constrained rewiring)
                           local_matching.py (f-general transition system)
                           layout_executor.py · proto_* (coordinate emission)
    core_utils/runtime/    dynamic_crosslink.py (f-general crosslink router)
                           aa_bonded.py (crossing angles/dihedrals/impropers)
                           packer.py (Packmol) · geo_opt.py (GROMACS EM)
    core_utils/templates/  monomer · linker (N-stub, per-stub caps) ·
                           strand_loader (whole-molecule AA templates)
    core_utils/io/         writer.py (topology writer — charge precision lives here)
    main_components/       World · Hydrogel · Polymer · Universe (state objects)
    relax/                 soft_em · soft_md · hard_em_shrink (guarded compression)
  property_extract/        network_topology · cyclic_topology · transport · ...
  tools/                   audit_hydrogel_topology · xtb_traj_to_pdb
  bash_settings/           shared launchers, per workflow
example/                   00–08, the tracked series (§6)
docs/                      §11
tests/                     34 files, 288 tests. Run them.
tools/PoreBlazer           git submodule, separate program
```

Sibling directories of this repo (`../cowork/`, `../handoff/`, `../paper/`) are
**not part of the package**: collaboration reports, session handoffs and a
manuscript. They are often where the *reasoning* behind a recent change lives,
so read them when a change looks unmotivated — but never ship them as package
documentation.

---

## 4. Two things are called a "series". Keep them apart.

**(a) Research series** — a paper and the frozen code that produced it.

| Series | Branch | System | State |
|---|---|---|---|
| Series-01 | `master` | PEGDA/Pluronic, tetrafunctional diamond, coarse-grained Martini | **Frozen.** The manuscript cites commit hashes. Do not rewrite. Baseline `d02a821`, 48 tests. |
| current work | `omni/general-ff-and-f6` | arbitrary even functionality (f=6), net-driven layout, rewiring, all-atom OPLS route | Active. 288 tests. |

There is a **frozen copy of Series-01 on disk** at
`/nas_0/software_backup/hygel_martini`. It must never be modified. It is also a
trap: it is installed in some environments, so a bare `import hygel_martini`
can resolve to it instead of your working copy (§7.1).

**(b) Pipeline series / example stages 00–08** — the numbered directories in
`example/`. These are the workflow, in order, and each is runnable on its own.

---

## 5. The configuration system

Everything the builder does is driven by one YAML, conventionally `maker.yaml`,
which is almost always just an `includes:` list plus overrides.

```yaml
includes:
  - config/simulation.yaml   # runtime, GROMACS, mode switches
  - config/mdp.yaml          # EM parameters
  - config/hydrogel.yaml     # chemistry: monomers, linkers, backbones
  - config/network.yaml      # net, repeats, cell parameter
simulation_parameters:
  output_dir: ${CONFIG_DIR}/output      # this file overrides all of its includes
```

Rules you must know:

* **Merge order.** Includes are resolved recursively and merged **in order**;
  later includes override earlier ones; the including file overrides all of its
  includes. Circular includes are refused. (`core/config.py`)
* **`${CONFIG_DIR}` is the directory of the top-level maker file** — not of the
  file the token appears in. So an emitted or copied maker placed in a
  *subdirectory* silently re-points every `${CONFIG_DIR}/structure/...` path in
  every file it includes. Keep generated makers beside the hand-written ones.
* **`${REPO_ROOT}`** resolves to the repository root.
* **Unknown keys are refused, not ignored**, in the validated blocks:

  ```
  'network_layout' has unknown key(s) ['typo_key']; expected [...]
  ```

  A typo fails at load time rather than halfway through a build. If you add a
  config key you must also add it to the validator, or nobody can use it.
* **`${...}` tokens are only expanded in values whose *key name* looks like a
  path** — the key must end in `_path`, `_file`, `_dir`, `_gro`, `_itp` or
  `_root`, or be on the explicit allowlist (`output_dir_suffix` is explicitly
  excluded). A new key holding a path but named something else will keep its
  literal `${CONFIG_DIR}` text and fail later as a missing file. Name path keys
  with one of those suffixes.

Top-level blocks: `simulation_parameters`, `monomer_definitions`,
`hydrogel_components`, `add_series_parameters`, `additional_itp_files`,
`hydrogel_topology_connectivity_audit`, `bonds`. The two blocks that decide the
network live *inside* `simulation_parameters`:
`network_layout` (net, repeats, cell_parameter, rewiring, conversion) and
`junction_bonded_generation` (all-atom crossing terms).

Two reliability switches (ported from the Series-01 reliability copy on
2026-09-30, see `docs/DEFECTS_FOUND_AND_FIXED.md` #35–#36):

* `simulation_parameters.require_explicit_crosslink_plan: true` makes total
  loss of planner metadata an error instead of a silent nearest-end fallback,
  writes `planned_crosslinks.json` (one-based ITP pairs, stub first), re-reads
  each written HYDROGEL ITP and stops before the next external step unless
  every planned bond is present exactly once (`<itp>.plan_audit.json`). Off by
  default. Turn it on for any build whose connectivity you intend to report.
* `add_series_parameters.<stage>.enabled: false` actually skips `add_water`,
  `add_small_ion`, `add_molecule` (mapping form) or `add_polymer`. A block
  without the key keeps its old presence-means-run meaning; a string like
  `"false"` is refused.

Adding a `simulation_parameters` key is one line of YAML; adding a key to a
*validated* block (`network_layout`, `junction_bonded_generation`) also needs
the validator.

### Entry points

| Command | Module |
|---|---|
| `hygel-builder` | `hygel_martini.hydrogel_builder.cli` |
| `hygel-relax` | `hygel_martini.hydrogel_builder.relax.cli` |
| `hygel-property` | `hygel_martini.property_extract.__main__` |
| `hygel-qm-to-opls` / `hygel-opls-to-martini` / `hygel-qm-to-martini` | stages 01 / 02 / 03 |
| `hygel-parameter-protocol` | E0–E6 bonded-parameter protocol |
| `hygel-qm-reference-audit` | QM reference qualification gate |
| `hygel-bead-selector` | bead selection |
| `hygel-audit-topology` | bonded-graph audit of a built topology |
| `hygel-xtb-traj-to-pdb` | trajectory conversion |

Every one takes `--help`. Equivalent module form: `python -m
hygel_martini.hydrogel_builder <maker.yaml>`.

---

## 6. The series, and how to apply each

External programs are **not** installed by this package (licences differ):
GROMACS, Packmol, xTB, ORCA, Bartender, Martini force-field files.

| Stage | What it is | Runnable? | Apply it by |
|---|---|---|---|
| `00_bead_selector` | placeholder | no | — |
| `01_qm_to_opls` | placeholder | no | — |
| `02_opls_to_martini` | reuse an existing OPLS MD trajectory for Bartender refit | after you supply data | fill `project/config/opls_existing_data.yaml` paths, then `MODE=setup\|md\|md_notrim\|trim bash run_existing_opls.sh` |
| `03_qm_to_martini` | xTB/ORCA → Bartender → screened ITP | yes | `bash run_qm_to_martini.sh config_common/common.yaml`; `--check-xtb --check-bartender` first |
| `04_full_builder` | diamond (f=4) full CG build | yes (needs GROMACS+Packmol) | `bash .../run_full_builder.sh maker.yaml` |
| `04_1_example_system` | small CG example system | yes | `bash .../run_example_system.sh maker.yaml` |
| `05_hydrogel_relaxation` | staged minimization, settling MD, **guarded shrink** | yes | `bash .../run_hydrogel_relaxation.sh maker_soft_em.yaml` |
| `06_physical_property` | manifest-gated property extraction | yes | `bash run_property_extract.sh` |
| `07_hexafunctional` | **f=6 crosslinker on `pcu`, rewiring** | yes, end-to-end verified | `hygel-builder maker.yaml`; theory in `docs/GENERAL_FUNCTIONALITY_NETWORKS.md` |
| `08_des_thiourethane_aa` | **all-atom OPLS-AA thiourethane network** (Hexakis-SH f=6 + PPG–TDI whole-strand, AcChCl solvent, partial conversion) | yes, end-to-end verified | read `example/08_des_thiourethane_aa/README.md` first — it is the richest example and has four subsystems of its own |

Example 08 is where the newest machinery lives, and it has its own
sub-directories worth knowing:

* `parameterization/` — how the templates were made; `apply_charges.py` (the
  only sanctioned way a charge set enters a topology); `release.py` (freeze the
  force-field files with hashes and a `draft/candidate/production-approved`
  status).
* `sizing/` — make the **cell size an input**: `cell_sizes.py` (size ladder,
  maker generation, target box from a finished topology), `composition_audit.py`
  (does the build match its recipe?), `run_manifest.py` (freeze every input a
  run reads), `run_size.sh` (fail-closed driver).
* `project/config_npt/` — what happens after the shrink: restrained heating →
  NPT → production, with `project/config_npt/check_convergence.py` gating production on a plateau
  artifact bound to the energy file's hash.
* `validation/` — **does the force field agree with the DFT it models?** (§9).

**Two nets, and their repeat rules are not cosmetic**
(`hygel_martini/hydrogel_builder/core_utils/layout/nets.py`):

| net | f | girth | repeats allowed | why |
|---|---:|---:|---|---|
| `dia` | 4 | 6 | any ≥ 3 | bipartition is the A/B sublattice, survives odd supercells |
| `pcu` | 6 | 4 | **even, ≥ 4** | bipartition is a coordinate parity; an odd supercell manufactures odd cycles through the periodic boundary, and a repeat below 4 makes the measured girth a box artifact |

Anisotropic supercells are legal within those rules (`4 4 6`).

---

## 7. Invariants and traps

These are the things that break silently. Most correspond to a numbered defect.

**7.1 Import path.** `import hygel_martini` may resolve to the frozen Series-01
install rather than your working copy. Run from the repo root with
`PYTHONPATH=$PWD python3 ...`, and when a result is inexplicable check
`hygel_martini.__file__` **first**. Test counts that disagree with §10 are
usually this.

**7.2 Never modify `/nas_0/software_backup/hygel_martini`.** Frozen Series-01.

**7.3 Every `*.itp` under a project's include path is auto-included,
recursively.** Two files declaring the same `[moleculetype]` are refused; an
`add_molecule` ITP whose moleculetype an included file already declares is
refused when the content differs and deduplicated when byte-identical. So a
*variant* of a species (a charge-scaled ion, say) must carry its **own**
moleculetype name — `ACC_f080`, not `ACC`. Getting this wrong once meant a run
would have used unscaled ions under a scaled label.

**7.4 Whole-strand (all-atom) mode refuses rewiring.** A rigid molecule has one
length; heterogeneous junction gaps cannot be spanned. The layout refuses the
combination rather than bowing or scaling the molecule.

**7.5 Charges must sum to the formal integer at written precision.** The
topology writer prints six decimals and compares what it wrote against what the
world holds. LigParGen tables miss zero by ~1e-4 e per molecule, which scales
linearly with system size; `apply_charges.py neutralize` fixes it exactly.
`composition_audit.py` gates on the per-molecule integer, not on "rounds to zero".

**7.6 A build below the gel point produces fragments, and that is correct.**
For an ideal A6+B2 tree the gel point is `p_A p_B > 0.2`; the experimental
stoichiometry in example 08 is 1/6, i.e. sub-gel. The connectivity audit is
therefore configured to **report** rather than gate there. Do not "fix" it.

**7.7 A sparse cell's build-stage EM geometry is a construction artifact.**
OPLS-AA polar hydrogens have no LJ parameters, so a T=0 minimization can pull
an N–H onto its own carbonyl (measured: N–C=O angle down to 76° in 10–13% of
strands in the 1/6 builds, none in full networks). The guarded shrink removes
it. Read geometry only after the shrink. (Defect #32.)

**7.8 The shrink's NVT recovery needs `constraints = h-bonds`.** Unconstrained
X–H at 1 fs destroys the structure. This path went untested for a long time
because no committed shrink ever triggered it. (Defect #31.)

**7.9 A shrink target of 1.0 g/cm³ is a target that was *set*, not a density
that was measured.** The shrink ends in an EM state; no property may be read
from it before restrained heating and NPT.

**7.10 Wrappers are fail-closed on purpose.** `run_size.sh` stops on a refused
emit, a force field matching no frozen release, a failed build, a failed recipe
audit, drifted inputs or a failed retarget, and records which in the ledger.
`run_equilibration.sh production` refuses without a plateau artifact whose
sha256 matches the current energy file. If you add a stage, wire its exit
status; a stage whose failure does not stop the next one is a defect.

**7.11 `${CONFIG_DIR}` is resolved once, from the top-level maker.** The whole
merged tree is normalized with a single path context built from the file passed
on the command line, so a token inside an included file resolves against the
*maker's* directory, not its own. This is why §5 insists generated makers stay
beside the hand-written ones.

**7.12 This checkout lives on a NAS that reports every file as `777`.** With
git's default `core.fileMode=true` that surfaces as hundreds of phantom
`100644 => 100755` modifications with zero content change, and mode flips leak
into commits (209 files in HEAD carry the exec bit; 177 of them are not
scripts). `core.fileMode false` is set locally in this checkout to stop it.
If `git status` ever shows the whole tree modified, check
`git diff --stat` for `0 insertions(+), 0 deletions(-)` before believing it.

**7.13 Do not confuse a crash with a verdict.** An audit that raises is not an
audit that failed the science — it is an audit that could not judge. Both stop
the pipeline (correct), but they need different fixes.

---

## 8. How to verify you did not break anything

```bash
PYTHONPATH=$PWD python3 -m pytest -q -p no:cacheprovider   # expect 288 passed
```

Tests are written to pin *behaviour with a reason*, and their docstrings say
which defect or review finding they encode. If you change behaviour and a test
fails, the question is not "how do I make it pass" but "which of the two is
now wrong".

Beyond the suite, the build itself carries audits: endpoint, bonded-term
completeness, component count, loop-order spectrum, charge sum. For example 08
there are three more gates: `composition_audit.py` (build vs recipe),
`release.py current` (do the force-field files match a frozen release), and
`run_manifest.py verify` (has any input changed since the build).

When adding a guard, add the negative test with it. Several guards in this tree
were written correctly and never executed for weeks; two of them were wrong
(§7.8 and the plateau screen), and only running them found out.

---

## 9. Claim boundary — the part to get right

A successful build licenses a **construction** claim: *the intended graph was
materialized as coordinates and a topology, and the audits agree*. It does not
license anything about force-field accuracy, equilibrium swelling, mesh size,
rheology, or transport. Property claims must pass `property_extract`'s
requirement/observable/numerical/promotion gates separately. Matching a
loop-order distribution is a topology statement, not a mechanical one.

**And, specifically, as of this branch:** the all-atom example's force field
**inverts the DFT ordering of chloride binding sites**. DFT puts the
thiourethane N–H 2.83 kcal/mol below the urethane N–H; the force field puts it
0.84 kcal/mol above, because 1.14\*CM1A-LBCC gives the two N–H hydrogens the
same charge to within 0.004 e. Everything underbinds by 11–17 kcal/mol, which a
non-polarizable model in the gas phase is expected to do; the *non-uniformity*
is what breaks the ranking. Method, numbers and scope:
`example/08_des_thiourethane_aa/validation/README.md`.

Consequence: **no statement about which donor holds chloride may be made from MD
on this force field.** Construction, topology, compression, stability and cost
results are unaffected — none of them depend on that ordering. If you are asked
to interpret transport or site preference, say this rather than reporting the
numbers the trajectory would give.

---

## 10. Current state

| | State |
|---|---|
| CG f=4 (Series-01) route | frozen, published |
| f=6 / net-driven layout (07) | GROMACS end-to-end build, EM convergence, audits pass |
| all-atom OPLS-AA route (08) | end-to-end build, guarded shrink to melt density, heating + short NPT run |
| exact-count conversion, size profiles, run manifests, composition audit | implemented, tested |
| charges | **draft** (1.14\*CM1A-LBCC); does not resolve the thiourethane/urethane Cl⁻ preference (§9). RESP swap alone does not fix it; F3 RESP conformer refit requested from the DFT side |
| thiourethane torsions, carbonyl improper | provisional, from a model compound |
| NPT / production MD / transport / conductivity | **not done** |
| PEG 200 ("PEO") | parameterized as a free plasticizer (`PEG.itp`, release `v1b`); 1 wt% denominator still provisional |
| water | not in the base recipe; a separate comparison series once the experimental team quantifies it |
| exact experimental composition | unconfirmed — `1:0.5` basis, %NCO, conversion all pending |

Tests 288, defects recorded 32, branch `omni/general-ff-and-f6`.

---

## 11. Workflow call chains

What actually runs, for the four workflows that have one. Every name below
exists in `docs/FUNCTION_REFERENCE.md`; follow it there for the signature.

**Hydrogel build (examples 04, 04_1, 07, 08)**

```
cli.main
  → config_params.config.Config.load_config      YAML includes, ${...} tokens
  → config_params.read_json.execute_mode → _execute_all_mode
      → build_hydrogel.build_backbone_only       proto plan → layout → blueprint → World
           (net path) core_utils.layout.net_layout.generate_net_layout_plan
                        → nets.build_periodic_net        the periodic net
                        → rewire.span_constrained_rewire optional, span-limited
                        → local_matching.plan_single_circuit  f-general transition system
           (all-atom)  core_utils.templates.strand_loader.load_strand_template
      → _perform_dynamic_crosslinking            router joins linker stubs to strand ends
      → main_components.Hydrogel.construct_chemical_detail
                        / construct_angles / construct_dihedrals
           (all-atom)  core_utils.runtime.aa_bonded.generate_junction_bonded_terms
      → runtime: geo_opt (GROMACS EM) · packer (Packmol) · add_series · writer
```

**Post-build relaxation (example 05)**

```
relax.cli.main → relax.config.load_relax_config → relax.generator.run_relax_workflow
  → soft_em.run_soft_em | soft_md.run_soft_md | hard_em_shrink.run_hard_em_shrink
```

**Parameter preparation (examples 02, 03)**

```
param_opt.opls_to_martini.cli.main → generator.run_opls_to_martini
                                   → fitting.run_existing_data_fit   (existing trajectory)
param_opt.qm_to_martini.cli.main   → generator.run_qm_to_martini → pipeline.run_pipeline
                                   → run_screening_postprocess       (screened ITP)
```

**Property extraction and audit (example 06)**

```
property_extract.__main__.main → requirements / analysis_jobs gates → extractors/*
property_extract.cyclic_topology.cyclic_topology_report   vertex symbol, loop orders
tools.audit_hydrogel_topology.main                        bonded-graph audit
```

---

## 12. Where a problem probably lives

| Symptom | Look at, in this order |
|---|---|
| config key ignored or a path unresolved | `config_params/config.py` (`Config.load_config`, `_looks_like_path_key`), then §5 and §7.11 |
| "unknown key(s)" at load | the validator in `config_params/read_json.py` / `build_hydrogel.py` — the key must be registered |
| wrong loop-order spectrum, unexpected component count | plan layer: `layout/nets.py`, `layout/net_layout.py`, `layout/rewire.py`; audit with `property_extract/cyclic_topology.py`. Check §7.6 before treating fragments as a bug |
| a build refuses a supercell | `layout/nets.py::validate_repeats` and the table in §6 |
| missing angle/dihedral/pair, grompp complains about a bonded term | materialization: `layout/proto_populator.py`, `main_components/Hydrogel.py`, and for all-atom `runtime/aa_bonded.py` |
| template not loading, atom names rejected | `core_utils/templates/{monomer,linker,strand}_loader.py`, `core_utils/io/martini_parser.py` |
| net charge non-integer, charge drift | `core_utils/io/writer.py` (precision + drift check), then `example/08.../parameterization/apply_charges.py`; §7.5 |
| duplicate moleculetype, an ITP silently not included | `config_params/read_json.py::_admit_added_itp`, `_itp_moleculetypes`; §7.3 |
| GROMACS/Packmol invocation failing | `core_utils/runtime/{geo_opt,packer,topology_updater}.py`, `add_series/add_small_ion.py` |
| shrink stalls or a molecule breaks during it | `relax/hard_em_shrink.py`, the run's `history.jsonl`, and §7.7–7.8 |
| coordinates look wrong across the boundary | `core/pbc.py` — every distance must go through minimum image |
| results change between runs that should match | seeds: `network_layout.conversion.seed`, `rewire_seed`, `random_seed` (since 2026-09-30 it also seeds the compiled side-chain RNG, #33); and §7.1 (wrong copy imported) |
| did the written topology honour the crosslink plan? | `require_explicit_crosslink_plan: true`, then read `<itp>.plan_audit.json` (#36); the connectivity audit alone cannot tell a rewired pair from a planned one |
| the whole tree looks modified in git | §7.12 — check `git diff --stat` for zero insertions first |

---

## 13. Where the authoritative numbers live

Never invent a number. If you need one, it is in a file:

| Want | File |
|---|---|
| why a behaviour exists / what broke before | `docs/DEFECTS_FOUND_AND_FIXED.md` |
| f-general network theory, net constraints, rewiring, audits | `docs/GENERAL_FUNCTIONALITY_NETWORKS.md` |
| E0–E6 parameter decision protocol | `docs/PARAMETERIZATION_PROTOCOL.md` |
| Series-01 validation failures and why the design is what it is | `docs/VALIDATION_HISTORY_AND_DESIGN_RATIONALE.md` |
| all-atom example: atom counts, boxes, densities, approximations | `example/08_des_thiourethane_aa/README.md` |
| force field vs DFT | `example/08_des_thiourethane_aa/validation/README.md` |
| size ladder, profiles, fail-closed driver | `example/08_des_thiourethane_aa/sizing/README.md` |
| what a given run actually read | `example/08_des_thiourethane_aa/sizing/manifests/*.json` |
| measured run costs | `example/08_des_thiourethane_aa/sizing/run_ledger.tsv` |
| which force-field version a checkout matches | `parameterization/release.py current` |
| human run order | `START_HERE_ko.md` |
| what a function does, signature by signature | `docs/FUNCTION_REFERENCE.md` (generated; regenerate with `tools/gen_function_reference.py --write`) |

Numbers that appear in a manuscript are cited from a commit hash. Do not amend
a commit whose hash has been recorded in `docs/DEFECTS_FOUND_AND_FIXED.md` or a
paper — add a follow-up commit.
