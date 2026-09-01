# Networks of general junction functionality

The tetrafunctional builder places chains on a diamond lattice and joins them
with two-stub linkers. This document covers the extension to junctions of any
even functionality — in particular six-arm crosslinkers on a primitive-cubic
net — and the topology measurement that goes with it.

Companion documents:

- [`DEFECTS_FOUND_AND_FIXED.md`](DEFECTS_FOUND_AND_FIXED.md) — every defect
  found along the way, what it did, and how it was found.
- `../../paper/main.tex` — the reasoning, with sources. Read that before
  changing any of the criteria below; several of them are not arbitrary.

---

## Quick start

```bash
cd /path/to/package

# inspect the layout without running GROMACS
PYTHONPATH=$PWD python3 - <<'PY'
from hygel_martini.hydrogel_builder.core_utils.layout.net_layout import generate_net_layout_plan
class P: pass
result = generate_net_layout_plan(
    P(), [{"id": "BB1"}], [{"id": "HEX_linker"}],
    net="pcu", repeats=4, cell_parameter=3.0,
    max_span=6.0, rewire_seed=0, rewire_kwargs={"max_sweeps": 60},
)
for key, value in result.summary().items():
    print(f"{key}: {value}")
PY
```

A complete declared system is at `example/07_hexafunctional/project/`.

> **Environment note.** This machine's editable install of `hygel_martini`
> resolves to the frozen Series-01 tree, not to a working copy. Running a
> script from anywhere other than the package root silently imports the old
> code. Use `PYTHONPATH=$PWD` from the package root, and check
> `hygel_martini.__file__` if a result looks stale.

---

## Configuration

```yaml
simulation_parameters:
  network_layout:
    net: pcu                 # RCSR symbol: pcu (6-connected) or dia (4-connected)
    repeats: [4, 4, 4]       # supercell; validated per net, see below
    cell_parameter: 3.0      # nm between neighbouring junction sites

    rewiring:                # omit the whole block to keep the regular net
      max_span: 6.0          # nm; the physical knob, see below
      seed: 0
      max_sweeps: 60
      allow_primary_loops: false
```

Omitting `network_layout` selects the historical diamond path unchanged.

The block is validated before any structure is built, and unknown keys are
refused rather than ignored — a silently ignored key is a setting you believe
is in effect. Rewiring options given without `max_span` are refused for the
same reason: without a cutoff no rewiring runs, so the rest would do nothing
while appearing to.

### Crosslinker template

The attachment points are the `BCK` residue beads. Declare one entry per stub:

```yaml
LINKERS:
  - id: HEX_linker
    gro: ${CONFIG_DIR}/structure/HEX.gro
    itp: ${CONFIG_DIR}/structure/HEX.itp
    linker_residue_name: HEX
    backbone_residue_name: HB
    stubs:
      - [{between: BB1, bond_funct: 1, bond_c0: 0.47, bond_c1: 1250}]
      # ... one list per stub
```

`backbone_1` / `backbone_2` remains the two-stub spelling and is what the
diamond examples use. Mixing the two forms in one linker is refused.

A stub bead stands in for the backbone end it will bond to and takes that
backbone's mass. Listing several admissible partners is allowed; listing
partners whose *masses disagree* is refused, because the stub mass would then
be undefined and picking one silently would be wrong.

---

## Choosing `repeats` and `cell_parameter`

Two constraints, both **per net rather than universal**. The layout checks them
and refuses a cell that violates either.

**Parity.** `pcu` is two-coloured by site parity, so an odd repeat count
destroys bipartiteness through the periodic boundary and manufactures odd loop
orders that are box artifacts, not chemistry. `dia` is two-coloured by its A/B
sublattice and every bond joins the two, so no repeat count can break it —
rejecting odd cells for both would be wrong.

**Girth.** A cell can be small enough that a walk wraps the box in fewer bonds
than the net's own fundamental cycle, at which point the measured girth belongs
to the box. One step along a `pcu` axis costs one bond; `dia` needs two.

| net | coordination | fundamental cycle | odd repeats | smallest useful repeat |
|---|---|---|---|---|
| `dia` | 4 | 6 | allowed | 3 |
| `pcu` | 6 | 4 | refused | 4 |

Separately, cell size matters for *measurement*: `dia` reaches the
literature peak-loop-order regime only at L = 6. At L = 4 it reports a
truncated spectrum that reads like a construction failure but is a cell that is
too small.

`cell_parameter` is the junction-to-junction distance, so it should match the
strand's contour length rather than being chosen for convenience.

---

## Rewiring: what it is for

A regular net is not a network. Every junction has the same local environment,
the loop-order distribution is a single spike at the net's fundamental cycle
size, and because `dia` and `pcu` are both bipartite there are **no odd-order
loops at all** — no three-strand triangles, which real networks contain.

Rewiring moves the topology off the net by double-edge swaps accepted only when
both resulting strands fit within `max_span`. Junction functionality is
preserved exactly, so an f=6 network stays f=6.

`max_span` is the one physical knob. A strand can only bridge junctions it can
reach, so it is bounded above by the contour length, and it sets how local the
connections are. Small values keep connections short and bias toward short
loops; large values approach the unconstrained limit. Measured on `pcu` at
L = 6:

| `max_span` / bond | peak LO | mean LO | odd fraction | primary | secondary |
|---|---|---|---|---|---|
| seed net | 4 | 4.00 | 0.000 | 0.000 | 0.000 |
| 1.45 | 5.3 | 4.63 | 0.474 | 0.036 | 0.074 |
| 2.05 | 6.0 | 4.84 | 0.476 | 0.024 | 0.047 |
| 3.05 | 6.0 | 5.32 | 0.424 | 0.010 | 0.019 |
| unconstrained | 6.0 | 5.38 | 0.425 | 0.007 | 0.006 |

Small-loop content is *non-monotonic* in the cutoff: it peaks at intermediate
values and falls at both ends, because local rewiring closes short loops while
unconstrained rewiring spreads connections over the cell. You do not tune loop
orders separately — one parameter sets the whole spectrum.

Convergence is tested against a **measured** noise floor rather than a fixed
tolerance. At high acceptance one sweep decorrelates the configuration
completely, so successive distributions differ by sampling noise forever and
any fixed threshold below that floor can never fire. The floor scales as
`1/sqrt(strands)`, so no single constant serves every cell size.

---

## Auditing what you built

```bash
PYTHONPATH=$PWD python3 -m hygel_martini.property_extract.cyclic_topology \
    --itp output/initial_hydrogel.itp --junction-residue BCK
```

Reports vertex symbols, the loop-order histogram, girth, primary and secondary
loop fractions, and bipartiteness. Two fields deserve attention:

- **`bipartite: true`** is a warning, not a curiosity. It means the network
  cannot contain an odd-order loop, which for a net-seeded build means the loop
  spectrum belongs to the seed rather than to the chemistry.
- **`loop_order_histogram_is_weighted_valid: false`** means the graph was not
  reduced first. The cycle-count expression weights each junction by `(f − 2)`,
  which is zero at `f = 2` and negative at `f = 1`, so it is only meaningful
  once dangling trees are peeled and chain continuations contracted. Call
  `reduce_to_junctions()` first. Partially converted networks are full of such
  nodes.

Loop-order spectra measured before and after that reduction are different
quantities. Say which one you are reporting.

---

## Module map

| module | role |
|---|---|
| `core/pbc.py` | minimum-image convention, correct for triclinic cells |
| `core/gro.py` | GRO reader (column-based, orthorhombic and triclinic boxes) |
| `core/itp.py` | GROMACS topology parser (was `io/martini_parser.py`) |
| `layout/nets.py` | `dia` and `pcu` definitions with their published invariants |
| `layout/rewire.py` | span-constrained rewiring |
| `layout/local_matching.py` | transition systems for any even functionality |
| `layout/net_layout.py` | coordinate layout from a net |
| `common/collisions.py` | refuses declarations that would overwrite one another |
| `property_extract/cyclic_topology.py` | vertex symbols, loop orders, bipartiteness |

The transition-system planner uses an Eulerian circuit rather than a search:
every junction of even functionality has even strand degree, so each component
admits one, and an Eulerian circuit *is* a one-circuit transition system.
Hierholzer returns the optimum in linear time. Verified to give a single
circuit on `pcu` at L = 2, 3, 4 and on `dia` at L = 2, 3.

Circuit count is all the planner optimizes. It **cannot** change the loop-order
distribution of the reduced junction–strand graph, which is a property of the
seed net — that is what rewiring is for. The two act on different graphs.

---

## Limits

**One f = 6 system has been built end to end** (example 07): every EM stage
converges and every audit passes, including exact planned-versus-materialized
endpoint matching and a non-bipartite loop spectrum at peak order 5. Not yet
exercised: NPT/production MD, solvation, and the stage-05 relaxation
workflows on an f = 6 system.

**A primary loop cannot be placed.** The layout puts each strand on a straight
segment between two junction sites, which is well defined for every loop order
except the first: a strand returning to its own junction has no such segment.
It is refused rather than straightened into a zero-length chain or dropped —
either would leave coordinates and topology disagreeing, which is what the
construction audit exists to catch. Rewiring for a coordinate build therefore
forbids primary loops by default; orders two and above place normally.
Reproducing a target primary-loop fraction in coordinates needs a layout able
to place a loop excursion.

**The force field is still Martini.** `read_atom_types` reads the mass from the
second `[ atomtypes ]` column, which is the Martini layout; OPLS-AA puts it in
the fourth. The mismatch is now reported where it happens instead of yielding
an empty table in silence, but it is reported, not handled.

**The monomer model has one backbone bead per repeat unit**, so head and tail
attachment sites coincide. An all-atom repeat unit has distinct ones. This is
the structural obstacle to the all-atom path.

**The f = 6 loop-order target is provisional.** Measurement puts the `pcu`/`dia`
ratio at 0.73 in the mean and 0.67 at the peak, against 0.60 from the
`1/(f−1)` scaling, which moves the target from about 5 to about 6. Neither net
had saturated at the largest cell tested, and the comparison sits at the
unconstrained cutoff. Re-measure on the production cell rather than carrying
this number over.
