# 07 — Hexafunctional network on the `pcu` net

An `f = 6` crosslinker on the primitive-cubic net, as the counterpart to the
tetrafunctional diamond examples.

## What differs from `04_full_builder`

| | 04 (diamond) | 07 (this) |
|---|---|---|
| net | `dia`, four-connected | `pcu`, six-connected |
| crosslinker | two BCK stubs | six BCK stubs |
| stub declaration | `backbone_1` / `backbone_2` | `stubs:` list |
| matching states per junction | 3 (labelled x/y/z) | 15 |
| layout | hard-coded diamond sublattice | `network_layout:` block |

## Layout inspection without running GROMACS

```bash
cd /path/to/package
PYTHONPATH=$PWD python3 - <<'PY'
from hygel_martini.hydrogel_builder.core_utils.layout.net_layout import generate_net_layout_plan
class P: pass
r = generate_net_layout_plan(P(), [{"id": "BB1"}], [{"id": "HEX_linker"}],
                             net="pcu", repeats=4, cell_parameter=3.0,
                             max_span=6.0, rewire_seed=0,
                             rewire_kwargs={"max_sweeps": 60})
for k, v in r.summary().items():
    print(f"{k}: {v}")
PY
```

## Notes

`repeats` is validated per net, not by a single rule. `pcu` is two-coloured by
site parity, so an odd repeat count destroys bipartiteness through the periodic
boundary and manufactures odd loop orders that are box artifacts; `dia` is
two-coloured by its A/B sublattice and keeps it for any count. A cell can also
be small enough that a walk wraps the box in fewer bonds than the net's own
fundamental cycle, at which point the measured girth is an artifact.

`rewiring` moves the topology off the regular net. Without it the loop-order
distribution is a single spike at the net's fundamental cycle size with no
odd-order loops at all — a legitimate idealized benchmark, and a poor
representative network. `max_span` is bounded above by the strand contour
length and is the single knob controlling how local the connections are.

Primary loops are forbidden for a coordinate build: the layout places each
strand as a straight segment between two junction sites, and a strand
returning to its own junction has no such segment. Loop orders of two and
above are placed normally.
