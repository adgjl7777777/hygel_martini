# Defects found and fixed

Record of every defect found while extending the builder toward general force
fields and general junction functionality, on branch `omni/general-ff-and-f6`.

Baseline: `d02a821` (Series-01 frozen tree). Tests at baseline: 48. Now: 164.

**The common shape.** Almost every defect below produced a *plausible but
wrong* result rather than an error. A build succeeded, a topology was written,
a number was reported — and the loss showed up, if at all, much later and
somewhere else. Where a fix was possible the failure was made loud; where the
old behaviour was load-bearing it was kept and labelled.

---

## Part 1 — Pre-existing defects in the frozen Series-01 tree

These reproduce at `d02a821`. Only #1 was fixed there (`c24e7a0`); the rest are
fixed on the working branch only, because the frozen tree is submission
provenance.

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

*Fixed in both trees:* `5dc494d` (working), `c24e7a0` (Series-01).

> **Release note.** `linker_loader.py` is not in `submission_manifest.json`'s
> hash records, so file-level provenance is intact. But `d02a821` is cited as
> the public commit in the manifest (5 places) and in `main.tex:100,761` /
> `si.tex:96,97`. Series-01 HEAD is now `c24e7a0`.

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

## Still open

- The stub/junction model is still fixed at two stubs in the runtime router
  and its caller (`dynamic_crosslink.py:203,349,492`, `read_json.py:831`), so a
  six-arm crosslinker loads but cannot yet be materialized. The template loader
  is general (`97b0a0b`) and the diamond layout now refuses a
  multi-arm template rather than truncating it.
- The coordinate layout still uses the hard-coded diamond constants in
  `proto_layout.py`; `nets.py` and `rewire.py` are not yet wired into it.
- `read_atom_types()` still assumes the Martini column layout (#2 makes it
  loud, not general).
- The monomer template model represents a repeat unit as one backbone bead plus
  side beads, so head and tail attachment sites coincide — the structural
  obstacle to all-atom repeat units.
