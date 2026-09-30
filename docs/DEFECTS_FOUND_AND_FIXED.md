# Defects found and fixed

Record of every defect found while extending the builder toward general force
fields and general junction functionality, on branch `omni/general-ff-and-f6`.

Baseline: `d02a821` (Series-01 frozen tree). Tests at baseline: 48. Now: 192. First f=6 end-to-end GROMACS build: converged, all audits passing.

**The common shape.** Almost every defect below produced a *plausible but
wrong* result rather than an error. A build succeeded, a topology was written,
a number was reported — and the loss showed up, if at all, much later and
somewhere else. Where a fix was possible the failure was made loud; where the
old behaviour was load-bearing it was kept and labelled.

---

## Part 1 — Pre-existing defects in the frozen Series-01 tree

These reproduce at `d02a821`. All are fixed on this working branch only: the
frozen tree is submission provenance and is left untouched by standing
instruction. (#1 was briefly applied there and then reverted for that reason;
the fix remains available as `5dc494d` to cherry-pick at submission time.)

### 1. The first example in the start guide cannot load

`START_HERE_ko.md` names `example/04_full_builder` as the first thing to run.
It fails before building anything:

```
ValueError: 링커 'SS_linker'의 backbone_1에는 정확히 하나의
            backbone target이 필요합니다: ['BB1', 'BB2']
```

`_resolve_stub_target()` required a stub's entries to name exactly one target
backbone. The tracked configuration's own comments describe the list as one
entry per admissible partner (`target별 entry`), each with its own bond
parameters, and `backbone_1` legitimately lists both `BB1` and `BB2`. The
loader was wrong, not the example.

**Fix.** `_resolve_stub_targets()` returns the declared set; only an empty
declaration is an error.

**What the fix exposed.** A stub bead stands in for the backbone end it will
bond to and takes its mass from it, so several admissible partners give a
well-defined stub mass only when their masses agree. `_stub_mass_for_targets()`
now requires that and names the conflicting masses, instead of silently taking
whichever target sorted first.

*Fixed in `5dc494d` (this branch only).* The frozen Series-01 tree still
carries the defect, deliberately: it is submission provenance and stays at
`d02a821` by standing instruction. A fix was briefly committed there and
reverted; nothing was ever pushed.

> **Release note for Series-01.** The tracked example 04 cannot load at
> `d02a821`. If this is to be fixed before submission, cherry-pick `5dc494d`
> from this branch and then update the `d02a821` references in
> `submission_manifest.json` (5 places) and `main.tex:100,761` /
> `si.tex:96,97`. `linker_loader.py` is not in the manifest's hash records,
> so file-level provenance is unaffected either way.

### 2. An OPLS force field yields an empty mass table, silently

`read_atom_types()` reads the mass from column 2 of `[ atomtypes ]`, which is
the Martini layout. OPLS-AA puts a bonded type and an atomic number there and
the mass in column 4, so every row fails to parse — and
`except (ValueError, IndexError): pass` discarded them all. The map came back
empty and the failure surfaced much later as an unrelated *"mass for atom type
X could not be determined"* on some other molecule.

**Fix.** The layout mismatch is named where it happens. Replacing this with a
layout-aware reader is the first step of the force-field work.

*Fixed in `ca8bfc8`.*

### 3. A requested water fraction can produce a dry system

In `add_water()`, a zero dry-gel mass propagates as
`target_added_mass = (0 / gel_wt) - 0 = 0`, so zero water is added. The system
still builds and still runs. The neighbouring World-mass fallback swallowed
every exception with a bare `pass`.

**Fix.** Zero dry mass now raises; the fallback warns.

*Fixed in `ca8bfc8`.*

### 4. A duplicate bond with different parameters is dropped in silence

`Attributes.Bond` de-duplicates by `(i, j)`, which is intended. But it dropped
a duplicate carrying *different* `c0`/`c1` just as quietly, discarding a
bonded-topology decision. Template bonds and `bonded_topology_patch_file` rules
can both reach the same atom pair.

**Fix.** The first definition still wins; the conflict is now reported with
both values.

*Fixed in `ca8bfc8`.*

### 5. The system-charge estimate skips files it cannot read

`_compute_total_charge()` skipped any ITP that failed to parse, which
understates the system charge and therefore the neutralizing ion count.

This **compounds with #2**: an empty mass map makes every molecule unreadable,
so the charge estimate silently becomes `None`.

**Fix.** Warns per skipped file.

*Fixed in `ca8bfc8`.*

### 6. Dead code in the crosslink router

`_pick_stub_targets()` — 33 lines, never called, and shaped like part of the
assignment path.

**Fix.** Removed.

*Fixed in `ca8bfc8`.*

### 7. Configuration declarations that silently overwrite one another

An AST scan found 88 sites writing into a dictionary inside a loop. Most are
accumulators. The hazard is the subset whose key comes from user configuration
or a parsed file, where a repeated key discarded one declaration without a
word. Five classes:

| Collision | Consequence |
|---|---|
| duplicate monomer / linker / backbone `id` | one definition never appears |
| two backbones claiming one `residue_name` | monomer↔backbone matching ambiguous; one backbone never selected |
| one `between` pair given two bond rules | whichever sorted first wins (both the layout and polymer lookups) |
| one atom type given two masses | the later row wins |
| one molecule type in two ITPs | the later file wins |

**Fix.** A shared `collisions` helper with two policies — `require_unique` for
identities that may be claimed once, `require_consistent` for records that may
repeat only if they agree. All five sites now refuse by name.

The shipped Martini force-field files were checked for pre-existing duplicates
before enforcing; there are none.

*Fixed in `8dd4667`.*

### 8. The minimum-image convention was written seven times, six of them wrong

Two copies in `core_utils/common/utility.py`, three inline in
`layout/isotropic_builder.py`, one in `runtime/dynamic_crosslink.py`, one in
`property_extract/geometry.py`. Six applied
`delta -= box * round(delta / box)` — the *orthorhombic* convention —
unconditionally.

On the GROMACS-legal triclinic cell `[[4,0,0],[0,4,0],[2,2,3]]`:

```
separation of 1.04 nm reported as 2.96 nm
```

The crosslink router ranks candidate chain ends by exactly this distance, so a
triclinic box would have produced a different and wrong network.

The sharpest case was `dynamic_crosslink.normalize_box_vector()`, which
accepted a full 3×3 cell and reduced it with `np.diag` — discarding precisely
the off-diagonal terms that make a cell triclinic — then handed the result to
the orthorhombic formula.

**Fix.** One implementation in `hygel_martini/core/pbc.py`, orthorhombic as a
fast path. `property_extract` keeps its deliberately orthorhombic contract, and
validates it, but delegates the formula. The numba scalar-`L` helpers stay
hand-rolled for the inner overlap loop and are now labelled cubic-only.

This is not hypothetical: the `dia` seed added in this work uses FCC primitive
vectors, which are neither orthogonal nor lower-triangular.

*Fixed in `4188f10`.*

### 9. The GRO reader splits a fixed-column format on whitespace

GRO is `%5d%-5s%5s%5d%8.3f%8.3f%8.3f`. The builder's reader split the tail of
each record on whitespace, which fails on valid GROMACS output: a coordinate of
`-100.000` fills its eight columns exactly and abuts its neighbour.

```
    1BCK     C1    1-100.000-100.000-100.000   →   ValueError
```

Any box large enough for coordinates below −100 reaches it. Of the three GRO
readers in the package, only `network_topology`'s took the columns, and only it
parsed the nine-value triclinic box.

**Fix.** One reader in `hygel_martini/core/gro.py` with the union of what the
three did, inferring the coordinate field width rather than assuming three
decimals.

**Also found.** The tracked example structures under
`example/04_full_builder/project/structure/` shift the atom index one column
right of the standard `%5d` field. The old lenient reader had been quietly
accepting them. Fixed columns alone would reject the shipped examples;
whitespace alone rejects valid GROMACS output. The reader therefore tries the
format first and the observed deviation second — both exact where they apply,
neither guessing.

*Fixed in `fb7e479`.*

### 10. The ITP parser discards bonded entries that omit their parameters

`i j funct` is a **complete** GROMACS bond whose parameters come from
`[ bondtypes ]`, and that is the normal shape of an OPLS-AA topology. The
parser required four fields for a bond, five for an angle, six for a dihedral,
and dropped everything shorter without a word.

An all-atom input would have lost most of its bonded terms and still produced a
topology.

**How it was found.** The existing `test_network_topology` fixture writes bonds
as `1 2 1`. It had been passing only because that module carried its *own*
parser without the restriction. **The two parsers disagreed about what the
format is** — which is the argument for having one.

**Fix.** Correct minimum field counts; `params` is simply empty.

*Fixed in `c9f0609`.*

### 11. Connectivity could not be read without a mass table

`read_itp_definitions()` raised when a mass could not be resolved, so the
reduced-network audit — which needs bonds and nothing else — could not read a
topology unless an atom-type table was supplied.

**Fix.** `require_mass=False`.

*Fixed in `c9f0609`.*

---

## Part 2 — Defects in code written during this work

All caught by tests before being relied on. Listed because the failure modes
generalize.

### 12. A published formula applied outside its domain

Sen & Olsen's cycle-count expression weights each junction by `(f_j − 2)`. That
is zero for a two-connected node and **negative** for a one-connected one.
Ideal lattices satisfy `f ≥ 3` everywhere; partially converted networks — the
case the DES system requires — are full of such nodes.

**Fix.** `reduce_to_junctions()` peels dangling trees and contracts chain
continuations first, and a `loop_order_histogram_is_weighted_valid` flag makes
an unreduced graph fail loudly instead of returning an empty distribution.

*In `69addbd`.*

### 13. Girth read off a weighted histogram

Because of #12, a graph made only of two-connected nodes has an empty
histogram, so a square and a triangle both reported *no girth* despite plainly
having cycles.

**Fix.** Girth comes from the raw shortest rings.

*In `69addbd`.*

### 14. A fixed convergence tolerance cannot work

At a large rewiring cutoff nearly every proposal is accepted, so one sweep
decorrelates the configuration completely and successive loop-order
distributions differ by sampling noise forever. A sweep-to-sweep threshold
never fires however stationary the process is.

Measured: on a 256-strand `dia` cell the shift plateaus at `0.061`, identical
between sweeps 6–20 and 80–120 — so the `0.02` default was unreachable and
reported spurious non-convergence. The floor is finite-size:

| net | strands | noise floor | ratio |
|---|---|---|---|
| `dia` | 256 | 0.0557 | 1.00 |
| `dia` | 2048 | 0.0192 | 2.90 (√8 = 2.83) |
| `pcu` | 192 | 0.0338 | 1.00 |
| `pcu` | 1536 | 0.0125 | 2.70 (√8 = 2.83) |

**Fix.** The floor is estimated from consecutive snapshots in a rolling window
and drift is tested against it. Every cutoff then converges, and faster at high
acceptance — the correct direction.

*In `1b3c2e9`.*

### 15. A guard inside a broad exception handler is not a guard

The atom-type collision check (#7) was first written inside the
`except (ValueError, IndexError)` that skips malformed rows. `DuplicateDeclaration`
subclasses `ValueError`, so it was swallowed by the very clause it was meant to
escape. The test caught it.

*In `8dd4667`.*

### 16. A validity gate that assumed one convention rejected valid input

The first triclinic minimum-image implementation gated on the GROMACS
lower-triangular reduction condition. That rejected the FCC primitive basis of
the `dia` seed outright, because its diagonal contains zeros.

**Fix.** The search range is widened until the winning shift is strictly
interior, which assumes nothing about cell convention. A cell too skewed for
that is refused with a message saying to reduce the basis.

*In `4188f10`.*

---

## Part 3 — Claims corrected by implementing them

Not code defects; statements in the theory document that measurement changed.

| Claim as first written | Corrected by measurement |
|---|---|
| Bipartite seeds give even loop orders, full stop | Holds for the *net* graph. Partial conversion leaves two-connected nodes, and contracting them changes path parity — so the restriction does not survive into the reduced graph. `pcu` at 45 % conversion: bipartite before reduction, **not** after. |
| Odd repeat counts break bipartiteness | Per net. `pcu` is coloured by coordinate parity and loses it; `dia` is coloured by its A/B sublattice and every bond joins the two, so no repeat count can break it. Rejecting odd cells for both would have been wrong. Wrap length is per net too — one `pcu` step costs one bond, `dia` needs two. |
| The `f = 6` loop-order target is ≈ 5, from the `1/(f−1)` scaling | Measured `pcu`/`dia` ratio at matched cell size is 0.73 in the mean and 0.67 at the peak, against the 0.60 predicted. Target moved to ≈ 6, marked provisional. |
| Peak loop order is a property of the network | Box-limited below a minimum cell. `dia` reaches the literature peak-8 regime only at L = 6; at L = 4 it reports 6.3, which would have been read as a rewiring failure rather than a cell too small. |
| One table of "peak LO" values | Two different quantities were being mixed: *fundamental cycle size* of an ideal net (a property of the net) and *peak* of a generated distribution (a property of the algorithm). |
| The parity obstruction is a novel result | Elementary graph theory. Kept as a build requirement that is easy to miss, not as a claim. |

---

## Recurring patterns

1. **Silent degradation over loud failure.** #2, #3, #4, #5, #7, #9, #10, #12, #13.
   A build that succeeds with a wrong value is worse than one that stops.
2. **One format, two parsers, two interpretations.** #10 existed only because
   the same file format was read by two independent implementations that
   disagreed. #8 and #9 are the same shape.
3. **A guard is only a guard where it can be reached.** #15.
4. **A validity check that encodes one convention rejects valid input.** #16,
   and #1 in its original form.
5. **A formula outside its stated domain.** #12.
6. **Measurement artifacts read as physics.** #14 and the box-limited peak in
   Part 3. Both would have been reported as findings.

### 17. Junction functionality was assumed to be four throughout the router

The crosslink router required exactly two stubs in three places and its
caller derived the expected assignment count as `2 * targets_per_stub`. That
is not a parameter that happened to be two; it is the diamond convention
written into the runtime.

Generalizing it surfaced a distinction the diamond builder never had to make,
because it only ever had one case. A junction attaches
`stubs x targets_per_stub` backbone ends and the planner supplies
`2 x planned_edges` endpoints. *How* those agree decides what geometry is
still allowed to choose:

| regime | meaning | geometry chooses |
|---|---|---|
| one planned edge per stub | the stub is itself a two-way junction, as on the diamond linker | which stub takes which edge |
| one endpoint per stub | the stub is a single attachment, as on a six-arm crosslinker | which stub takes which endpoint |

In the second regime the planned pairing is a *traversal through* the junction,
not a grouping *of* stubs, so it cannot be reconstructed from stub groupings.
The exact planned/materialized edge-hash check therefore applies only to the
first regime; the second is verified by comparing the consumed endpoint set
against the planned set. Both are exact — the difference is what there is to
be exact about.

Anything matching neither regime is refused with both counts. The purely
geometric pairwise fallback genuinely pairs two stubs and now raises for a
multi-arm junction instead of skipping it, which would have left its arms
unbonded without a word.

*Fixed in `9830385`.*

### 18. A cross-project include silently re-pointed another example's paths

Found by a whole-tree YAML audit (syntax, in-file duplicate keys, include
resolution, resolved-path existence, ghost keys). Example 07's `maker.yaml`
included example 04's `simulation.yaml` for convenience. But `${CONFIG_DIR}`
resolves against the **top maker's** directory for every included file, so
04's `additional_itp_files: ${CONFIG_DIR}/structure/additional_ions.itp` came
to point inside example 07, where no such file exists — and 04's
`anisotropy: false` switched the isotropic diamond path on, which the
`network_layout` guard then (correctly) refused. The example shipped unable to
build, and the load-time checks did not catch it because path existence is not
checked at load.

**Fix.** Example 07 carries its own `simulation.yaml`; the semantics are
stated in it and in `maker.yaml`. The audit also confirmed: 0 syntax errors,
0 in-file duplicate keys, 0 unresolvable includes across all 43 YAML files,
and no configuration key that the code never reads (`example_metadata` is
intentional self-description).

*Fixed in `a4396ce`.*

### 19. Integration defects found by the first f=6 end-to-end build

Running example 07 through GROMACS -- previously listed as untested --
surfaced four defects in one afternoon, each invisible to the unit tests
because each lives between layers:

- **`external_bonds` reused with a different shape.** The per-stub emitter
  put a nested list under a key that three consumers (`proto_builder`,
  `proto_layout`, `isotropic_builder`) read as a flat list to sum bond
  lengths, crashing at plan time. The flat key keeps its old shape; the
  nested one is `external_bonds_by_stub`.
- **mdp overrides silently dropped.** `_create_mdp_file` accepted overrides
  into its defaults dict but only ever wrote six templated keys, so
  `periodic_molecules` -- the option an infinite covalent network *requires*
  (mdrun otherwise aborts with "inconsistent shifts over periodic
  boundaries") -- was configured and then never written. Overrides now pass
  through generically.
- **Straight-segment placement collides on a lattice.** Coordinates measured
  directly: parallel strands lie on one segment bead-for-bead, and a rewired
  strand longer than one lattice step is collinear with the lattice line,
  running through intermediate junctions and every shorter strand on it
  (whole-chain contact trains at ~0.006 nm; EM stuck at 1e21-1e24 kJ/mol,
  "converged to machine precision in 15 steps"). Every strand is now bowed
  off its line with a half-sine of ~0.5 nm real-space amplitude, ends fixed,
  azimuth spread per strand -- the placement-time deformation approach the
  original layout already used.
- **The coincidence resolver only caught exact duplicates.** It hashed atoms
  to cells of one threshold and compared atoms in the *same* cell, so a pair
  0.8 threshold apart in adjacent cells survived and produced Fmax = inf in
  single precision. It now searches the 27 neighbouring cells and pushes the
  pair apart along their actual separation direction.

After these: all EM stages converge (Fmax < 1000), the built network audits
as one component with all 64 junctions at degree exactly 6, planned and
materialized endpoint sets match exactly, and the loop spectrum is
non-bipartite with peak loop order 5 -- inside the provisional f=6 target.

*Fixed in `aa3fee6`.*

### 20. Two latent defects exposed by the first partial-conversion build

Partial conversion (strand-dilution model: each strand forms with probability
`conversion.fraction`) is the first feature that makes per-junction
expectations *vary*, and two pieces of code had silently assumed they never
would:

- **Truthiness as presence.** The router detected a planned stub with
  `bool(planned)`, so a fully unreacted crosslinker -- whose plan is the empty
  tuple, a legitimate outcome of conversion -- looked *unplanned* and tripped
  the partial-metadata guard. An empty plan is still a plan.
- **A loop variable leaking across loops.** Bond creation compared its
  success count against `expected_per_linker`, a variable left over from the
  last iteration of the *separate* audit loop above it. Every linker shared
  one expectation until conversion made it per-linker; the leak then failed
  correct builds. The audit loop owns the planned-versus-chosen comparison;
  the creation loop now checks only that every attempted bond was created.

With both fixed, the 0.5-conversion build runs end to end: 184 bonds for 92
formed strands exactly, all EM stages converged, and the connectivity audit
reports two components -- the giant cluster plus one free crosslinker
molecule, matching the degree histogram's single degree-0 junction. The
reduced-graph audit then shows the theory document's partial-conversion
mechanism live: a primary loop appears after contraction (girth 1) although
the coordinate layout placed none, because contracting degree-2 continuations
changes what a loop looks like.

*Fixed in `6d0aa34`.*

### 21. The net layout's periodic cell never reached the World box

Found by independent adversarial verification of the "built end to end"
claim: the delivered f=6 structure converged its EMs, passed every audit --
and carried ~0.9 MJ/mol of undisclosed bond pre-strain, with 155/384
crosslink bonds over 1 nm (max 6.0 nm against b0 = 0.47) held in balanced
tension. The audits measure topology and the EMs measure force balance;
neither measures strain, so nothing said a word.

Root cause, established by measuring the pre-EM structure: the net geometry
lives in its own periodic cell (12 nm here), but `World.box_vector` still
came from the diamond proto plan (~23 nm). A strand wrapping the true
boundary saw a box too large to fold it back, so its end sat 9-17 nm of
fictitious "distance" from its planned junction and was bonded across it.

**Fix.** The net cell is propagated into the World box (orthorhombic nets
only until the GRO writer emits nine-value boxes; `pcu` qualifies, the `dia`
FCC primitive cell does not). After the fix, pre-EM crosslink lengths max
1.24 nm, post-EM mean 0.555 / max 0.93 / none above 1 nm, and the converged
potential energy is negative rather than +8.8e5 kJ/mol.

*Fixed in `2662d1c`.*

### 22. The all-atom ownership rationale was contradicted by the pipeline

Also from the verification. `aa_bonded` refuses to regenerate
template-internal terms "because the template ITP owns them" -- but the
populator dropped template `[ pairs ]` outright and silently discarded any
template dihedral without inline parameters (the exact shape an OPLS template
carries). One rule assumed the other end existed; it did not.

**Fixes.** Template pairs are copied (index-remapped, deduplicated) into the
combined topology whenever `topology_nrexcl >= 2`, so Martini output is
unchanged; parameterless template dihedrals and impropers are registered
parameterless instead of dropped (populator and the monomer path both); the
junction walk unions its pairs with the template ones rather than
overwriting; and ownership is now per template *instance*, since two
molecules stamped from one template joined by a builder bond are different
owners -- identity comparison silently skipped exactly that case. Partial
conversion additionally requires a `seed`, since an unseeded selection made
builds silently irreproducible.

Recorded limits from the same review, documented rather than fixed: the
router's exhaustive slot assignment is factorial in f (fine through f=8;
switch to linear assignment beyond); the coincidence resolver is not
PBC-aware; unformed strands are removed from the system entirely rather than
kept as free chains, which changes composition and is now stated in the
partial-conversion docs; and "byte-compatible" writer output is
value-identical, not byte-identical (integer force constants print as `1250`,
not `1250.000000`).

*Fixed in `2662d1c`.*

### 23. The writer's nrexcl lookup referenced a name that was never imported

Found by the first all-atom end-to-end build (example 08): grompp reported
"Excluding 1 bonded neighbours" on a topology configured with
`topology_nrexcl: 3`. `write_combined_itp` resolved the setting via
`Config.get_param(...)` -- but the module never imports `Config`, so the line
raised `NameError`, the surrounding `except Exception` swallowed it, and
every configured nrexcl silently became 1. The test suite never caught it
because the tests pass `nrexcl` explicitly. Under nrexcl=1 with an OPLS
force field, 1-3 and 1-4 neighbours interact at full strength *and* the
`[ pairs ]` list double-counts the 1-4s -- all silently.

**Fix.** The import now lives inside the function, before the guarded
lookup. Same lesson as #12: a broad `except` around a lookup hides the
lookup not existing at all.

### 24. Pre-crosslink EM stages ran on an angle-less topology

Angles were constructed only in `finalize` (after crosslinking), because the
Martini path derives most of them from heuristics that want the final bond
graph. But the staged build minimizes structures *before* that -- and an
all-atom molecule held by bonds and dihedrals with no angles (and, under
nrexcl=3, no nonbonded repulsion between its 1-3 pairs either) collapses
onto itself. Measured directly: geminal hydrogens at 0.009 nm, EM pinned at
Fmax ≈ 3×10⁴ on the same strand atom in every instance, and one mdrun
segfault. The optional bonds-only soft-relax stage made it worse for the
same reason.

**Fixes.** Template-internal angles are registered by the blueprint
populator at populate time, so every EM stage sees them;
`construct_angles` skips triples that already exist instead of
double-registering; and the soft-relax stage is skipped (with a printed
reason) when whole-strand templates are in play -- its purpose, letting
Martini bead chains drift until their ends reach the stubs, does not apply
to ends placed at the junction gap by construction.

### 25. construct_dihedrals collapses template instances onto one map

The object-level walk processed each template *object* once and built its
`source_index -> atom_id` map over every atom of every instance -- so with
two or more instances the dict keeps whichever instance sorts last, and the
registered dihedrals mix atoms across molecules. Latent until now because no
Martini template on this branch declares dihedrals. It would also have
re-registered every term the blueprint populator had already mapped
per-instance.

**Fix.** The populator records the templates it has handled
(`World.template_dihedrals_done`); `construct_dihedrals` skips them. The
monomer/Polymer path, which sets no such record, is unchanged.

### 26. Ryckaert-Bellemans coefficients were truncated to three

The populator's dihedral mapping stored `c0..c2` only, silently discarding
coefficients four to six of a funct-3 (RB) dihedral -- the exact form OPLS
templates carry. Parameter lists longer than three are now stored whole via
`dihedral_params`, which both writers already emit verbatim.

*All four found (or made findable) by building example 08 -- the first
system to exercise the all-atom path end to end, which is precisely what the
example exists for. Fixed in `3dc09e5`.*

### 27. The atomtypes layout fix could still hand back an atomic number as a mass

Found by the second adversarial verification round, probing the very fix
entry 23's commit advertised. Two rows the previous heuristic got silently
wrong: `CT 6 12.011 0.0 A ...` (the standard Amber/CHARMM port layout,
atomic number but no bonded type) took the numeric-second-column fast path
and returned **6.0** as carbon's mass; and `opls_202 S 16 32.06 0.0 A ...`
(bonded type literally `S`, a real OPLS bonded type) produced two
particle-type candidates and was silently dropped. Example 08 dodged both by
luck of its renamed types, so the shipped outputs were unaffected -- the trap
was armed for the next force field.

**Fix.** The fast path is gone. The particle-type column is now identified
from the right as the A/S/V/D token followed only by numbers and preceded by
a numeric charge and mass, and the mass is read two columns before it. A
masquerading bonded type on the left can never win that scan. Both probe
rows are now regression tests.

The same round also caught the example README quoting Epot ≈ −1.4e5 for the
full build -- a number copied from the angle-less failed build one defect
earlier (entry 24), lower than the true −8.9e4 precisely because a missing
angle term costs nothing. Corrected in place; the commit message of
`3dc09e5` carries the wrong number permanently, which this entry supersedes.

*Fixed in `21d469f`.*

### 28. Linker body bonds were indexed positionally, not by bead map

Found by the first per-stub-cap build: withholding a reacted arm's thiol
hydrogen leaves a hole in the emitted atom list, and
``_create_linker_bonds``/``_mark_linker_terminals`` indexed the template's
bead-position bond rows *positionally* into the compacted list -- silently
shifting every bond after the hole. Measured as junctions fragmenting into
321 covalent components on a full-conversion build that should have one.
Harmless before caps existed (nothing was ever withheld), armed the moment
anything was.

**Fixes.** Both now index through the bead map (template position -> atom
id) and skip rows referencing withheld atoms. The same sweep replaced the
one-blueprint-atom-per-attachment-row stub emission -- which would have
duplicated any stub with two body bonds, exactly what an unreacted thiol
sulfur (CH2 and H) is -- with one atom per stub carrying all its attachment
rows. Regression tests pin both; Martini example 07 rebuilds bit-for-bit.

*Fixed in `1820be2`.*

### 29. The side-chain stage searched for hours with nothing to place

Found while building the experimental-length strand: with
``monomer_definitions.MONOMERS: []`` -- the normal configuration for every
template-driven build, since whole-strand and linker templates already carry
every atom -- ``construct_chemical_detail`` still ran its full per-atom
placement search. The per-atom guard does not catch it because an empty
``TemplateStrategyIterator`` is a truthy object, so each backbone atom paid a
candidate-vector sweep plus a neighbour scan to place one of zero templates.
Measured at ~25 atoms/s: 50 minutes of provably empty work on a 77k-atom
build, minutes on every smaller one.

**Fix.** The stage returns early, with a printed reason, when the monomer
library holds no records. Verified behaviour-preserving rather than argued:
example 08's n = 3 build drops from minutes to 40 s and its
``initial_hydrogel.itp`` is **bit-identical** to the pre-fix file.

*Fixed in `02c8f17`.*

### 30. The topology writer rounded charges away, four decimals at a time

Found by building the experimental-length strand: the network's written
topology carried a net charge of **-2.33 e** while every template it was built
from sums to -0.0001 e (predicted network total -0.0256 e). Nothing had
changed a charge; the writer printed them with four decimals. That is exactly
enough for a Martini bead (0, +1, -1) and silently lossy for anything else,
and the loss is systematic rather than random when thousands of atoms share a
value -- here the tiled strand's uniformly corrected copies, 57 600 atoms
each losing ~4e-5 e in the same direction.

A net charge that large is not cosmetic: under PME it changes the
compensating background, and it is precisely the kind of error that survives
every audit we had, because each audit compared topology against topology.

**Fix.** Charges are written with six decimals (both writers), and
``write_combined_itp`` now compares the sum of what it wrote against the sum
of what the world holds, printing the difference when it exceeds 1e-4 e --
so a future precision problem announces itself instead of being inferred
from a strange energy. Rebuilt: the n = 33 network now writes -0.0256 e,
matching the prediction to the last digit; example 07 (Martini) is unchanged
in value, and now prints charges as ``0.000000`` rather than ``0.0000``.

*Fixed in `02c8f17`.*

### 31. The shrink's NVT recovery had never run, and broke a molecule when it did

Found by shrinking the first cell whose compression actually needed the
guard's escape hatch. The guarded shrink runs an NVT recovery when a step's EM
fails, and `example/08`'s three committed shrinks (51, 93 and 44 steps) all
passed every step on EM alone -- `history.jsonl` shows `recovery: 0` in every
one of them. So `config_shrink/nvt_recovery.mdp` was configured, committed and
documented without ever having been executed on an all-atom system.

The first cell to trigger it (n = 3, `conversion.count: 32`, a sub-gel
fragmented network) stalled at step 11: steepest descents reported
"converged to machine precision" with `Fmax = 3.3e4` on one atom, at every
backoff scale down to 0.125% -- a localized pathology inherited from the
structure rather than caused by the compression. The atom was a strand's
urethane N-H hydrogen sitting **0.068 nm from its own carbonyl oxygen**, a
1-4 partner whose LJ repulsion carries `fudgeLJ = 0.5` and therefore offers no
wall to escape from.

The recovery mdp ran `constraints = none` at `dt = 1 fs` for 20 ps. An
unconstrained X-H stretch has a period near 11 fs, so that step samples it
about ten times per cycle; with generated velocities and a rescaling
thermostat the error accumulates until a hydrogen is driven somewhere no EM
can undo. The setting is harmless for the coarse-grained systems the shrink
workflow came from, which have no explicit hydrogens at all.

**Fix.** The recovery constrains X-H bonds (`constraints = h-bonds`, LINCS),
which removes exactly the motion a 1 fs step cannot resolve and leaves the
rest of the dynamics alone. The comment in the mdp records the observation
rather than the rule, so the next person can tell why it is there.

Two things this exposes beyond the one setting: a config path that no example
exercises is untested no matter how carefully it was written, and
`gen_seed = -1` makes each recovery attempt unreproducible -- which is
deliberate (a retry wants a different kick) but means a stalled shrink cannot
be replayed exactly.

Validated by re-running the same shrink: the compression that previously died
at step 11 (box 14.8 nm, `Fmax = 3.3e4`) now reaches the 4.819 nm target in 65
accepted steps, using the recovery three times along the way.

*Fixed in `734ac0e`.*

### 32. Energy minimization folds a urethane N-H onto its own carbonyl in sparse cells

Found while diagnosing #31, and the more interesting half of it. In OPLS-AA a
polar hydrogen carries **no Lennard-Jones parameters at all** (`st846`:
sigma = 0, epsilon = 0). Its 1-4 partner across the carbamate, the carbonyl
oxygen, carries -0.426 e against the hydrogen's +0.525 e. With no repulsive
term in the pair, that 1-4 Coulomb attraction is a well with no floor: the
only thing holding the hydrogen off the oxygen is the bonded geometry
(`N-C=O` angle, k = 669 kJ/mol/rad^2, and the `H-N-C=O` torsion).

At finite temperature that is enough. In a *minimization*, in a cell where
nothing else competes sterically, it is not -- the angle bends and the
hydrogen falls in. Measured across the `N-C(=O)` angle of every strand:

| build | strands | min angle | mean | below 110 deg |
|---|---:|---:|---:|---:|
| n=3 full (committed) | 192 | 116.2 | 123.7 | 0 |
| n=33 full (committed) | 192 | 117.5 | 124.5 | 0 |
| n=3 full + DES (committed) | 192 | 116.3 | 123.8 | 0 |
| n=3 partial 1/6 (committed) | 28 | **76.2** | 116.6 | **3 (10.7%)** |
| n=3 `count: 32` | 32 | **76.2** | 115.7 | **4 (12.5%)** |
| the same cell after the shrink | 32 | 118.3 | 123.5 | 0 |

So it is specifically the **partial-conversion** cells: at 1/6 conversion five
sixths of the net's edges are empty and a strand has room to fold onto itself,
where in a full network its neighbours are in the way. The worst cases reach
`H...O = 0.12 nm`, against 0.31 nm in the template and 0.26 nm for even a
fully *cis* planar carbamate -- and once there, a subsequent minimization
cannot climb back out, which is what starved the shrink in #31.

**Not fixed in the builder, and arguably not a builder bug**: the topology is
complete (every strand instance carries all 126 template angles, 182
dihedrals and 158 pairs; the parameters are the template's own), and the
force field is being applied correctly. It is minimization at T = 0 finding a
real, unphysical minimum of a fixed-charge model. The last row of the table is
the mitigation and it is already in the workflow: a short constrained NVT
removes every distortion.

**What follows from it.**

* No geometry may be read from a build-stage EM of a sparse cell. The
  committed `output_partial` evidence carries three folded urethanes; that is
  a construction artifact, not a conformational result.
* The guarded shrink (with #31's fix) is the earliest point at which the
  geometry is trustworthy, and a partial cell should always be run through it.
* If a build-stage structure is ever needed directly, the build needs a short
  restrained NVT of its own, or a minimization that guards polar-hydrogen
  contacts the way `min_distance_report` guards overlaps.

### 33. The run seed never reached the compiled geometry RNG

**Symptom.** Two CLI runs of example 04_1 with the same `random_seed: 2020`
produced the same 375-atom backbone and then different side chains: 1,363 of
1,746 coordinate lines in `initial_hydrogel.gro` differed, and everything
downstream (packed water, final EM) with them.

**Cause.** `random_normal_vector` is Numba-compiled. Numba keeps its own
per-thread random state, and `np.random.seed()` called from Python seeds NumPy,
not that. `_seed_random_generators` seeded `random` and `np.random` and
stopped. Measured directly: an unseeded compiled draw differed between two
fresh processes, and a Python-side `np.random.seed(2020)` left it different.

**Fix.** `seed_numba_random(seed)`, an `njit` wrapper around `np.random.seed`,
called from `_seed_random_generators` after the two existing seeds. It draws
nothing from Python or NumPy, so the first draws of every already-seeded run
are unchanged (pinned by `tests/test_rng_seeding.py`). After the fix the same
two-process comparison is byte-identical for all nine GRO and ITP files.

**Scope.** The serial geometry path the builder uses. Parallel worker streams
are not seeded. Ported from the Series-01 reliability copy 0.1.1.dev2.

### 34. The side-chain stage scanned the whole world before asking whether anything attaches

**Symptom.** `construct_chemical_detail` is O(N) per backbone atom, and on a
backbone with nothing to attach (PEG-only, or a whole-strand template that
already carries its atoms) the neighbour list it built was discarded unused.
The reliability copy measured a PEG N2/L56 build at 65.2 s falling to 17.1 s
once the order was fixed, with byte-identical outputs.

**Fix.** `iterator.next()` and its `None` check now precede the scan. This is
output-neutral by construction: the iterator does not touch `World`, and the
skipped scan draws no random numbers, so `random_normal_vector` sees the same
sequence. Checked here on example 04_1 (which does attach side chains):
the reordered build is byte-identical to the previous commit's build under the
same seed, nine of nine GRO/ITP files. Ported from 0.1.1.dev1.

### 35. `enabled: false` on a packing stage did not disable it

**Symptom.** Stage selection in all-mode tested for the presence of the
`add_water` / `add_small_ion` / `add_molecule` / `add_polymer` block, not for a
switch inside it. `add_water: {enabled: false, number_of_water: 10000}` added
ten thousand waters to a build that asked for none.

**Fix.** `_enabled_formulation_stages` drops a stage whose block says
`enabled: false`, refuses a non-boolean (`"false"` is a truthy string, not a
switch), and consumes the key so stage code never sees it. A block without the
key keeps the legacy presence semantics, so no existing input changes meaning;
no committed example carries the key. omni's list-valued `add_molecule` has no
switch and passes through. The `except Exception` that used to turn *any* error
in that block into "no packing" is now `except KeyError`. Ported from 0.1.1.dev0.

### 36. A lost crosslink plan fell back to distance, and nobody re-read the written ITP

**Symptom.** Two separate gaps. (a) When planner endpoint metadata was
partially present the run refused, but when it was entirely absent the same
call routed every stub to its nearest compatible end and finished. A build
whose plan had evaporated looked identical to one that honoured it. (b) The
memory bond registry was trusted as proof of the file: nothing re-read
`initial_backbone.itp` or `initial_hydrogel.itp` to check that the planned
stub-to-endpoint bonds were the ones written. The connectivity audit cannot
catch this class: a plan of a-b, c-d written as a-c, b-d stays one component,
and so does a ring with one bond deleted.

**Fix.** Opt-in `simulation_parameters.require_explicit_crosslink_plan: true`.
The planner then raises on total metadata loss instead of falling through;
`_perform_dynamic_crosslinking` records the expected one-based ITP pairs
(`planned_crosslinks.json`) and clears any pairs left by a previous in-process
build; and `_guard_written_crosslinks` re-reads each written ITP with an
independent parser and requires every planned pair exactly once with no other
bond between the listed stubs and endpoints, writing `<itp>.plan_audit.json`
and raising before the next external step. Default off; makers written before
the option behave as before.

**Why the indices are safe in omni.** The writer emits `atom.atom_id + 1` for
atoms and bonds alike, and per-stub cap atoms are removed at layout time,
before ids exist, so ids fixed at planning time are the ids in the file.
Exercised end to end: example 07 (pcu net, 384 crosslinks) and the DES
diagnostic cell (partial conversion, per-stub caps, 64 reacted arms across 32
components) both write `PASS` audits with planned == observed at both stages.
`tests/test_explicit_plan_guard.py` injects a rewire, a duplicate, a deleted
ring bond, a missing atom and a missing file, and checks that a corrupted
backbone write stops before any EM while a corrupted hydrogel write stops after
exactly the backbone EM. Ported from 0.1.1.dev0 (commit 50192af).

## Still open

- The f=6 path now builds end to end under GROMACS (example 07): all EM
  stages converge and every audit passes. Still untouched: NPT/production MD,
  solvation of the f=6 system, and the relaxation-stage (05) workflows on it.
- The straight-segment coordinate model cannot express a primary loop, so
  rewiring for a coordinate build forbids them by default and the layout
  refuses one rather than straightening it. Loop orders of two and above place
  normally. A layout that can place a loop excursion would lift this.
- The coordinate layout still uses the hard-coded diamond constants in
  `proto_layout.py`; `nets.py` and `rewire.py` are not yet wired into it.
- `read_atom_types()` still assumes the Martini column layout (#2 makes it
  loud, not general).
- The monomer template model represents a repeat unit as one backbone bead plus
  side beads, so head and tail attachment sites coincide — the structural
  obstacle to all-atom repeat units.
