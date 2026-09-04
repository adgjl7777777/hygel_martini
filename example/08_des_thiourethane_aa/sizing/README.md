# Sizing: the same chemistry at whatever size the machine allows

The four builds in this example each fix a size in their maker. That is fine
for demonstrating a feature and wrong for production, where the cell should be
as large as the node that happens to be free can carry, and the chemistry
should not have to be re-specified to get there.

These four files make size an input.

| file | what it does |
|---|---|
| `itp_inventory.py` | atom counts, masses and charges read out of ITP/TOP files; the shared composition primitive |
| `cell_sizes.py` | the size ladder, and the maker/shrink-maker generator for one size |
| `node_resources.py` | what the node has free, and a polite thread count and GPU |
| `run_size.sh` | emit for this size with this node's resources, build, retarget the shrink from the realized topology, shrink, log the cost |

## What actually changes with size

Everything is arithmetic on the net and the topologies, so it is computed, not
tabulated:

* `pcu` with repeats `(rx, ry, rz)` has `J = rx*ry*rz` junctions and `E = 3J`
  edges (six arms each, every edge shared by two);
* a formed strand consumes one arm at each end and each of those arms loses its
  thiol cap hydrogen, so `atoms = J*93 + S*N_strand - 2S`;
* AcChCl enters at the experimental 1:6 per junction;
* the construction box is `cell_parameter * repeats` and does **not** get
  denser with size -- the rigid strand sets the junction spacing, so the
  dilution is a constant of the construction method;
* the shrink target is the box holding the total mass at the target density.

The masses come from the project's own ITP files, so a re-parameterization
cannot leave this planner quoting stale numbers. The check that it is the same
model as the committed makers: it reproduces all three shrink targets already
in this example -- 6.34 nm (n=3 dry), 9.32 nm (n=33 dry) and 7.19 nm (n=3 +
DES) -- to 0.01 nm.

## The size ladder is not continuous

`pcu` accepts only **even repeats of at least 4**, because an odd supercell
closes odd walks through the periodic boundary and destroys the net's
bipartiteness, and a repeat below 4 makes the shortest cycle a box artifact
rather than the net's own girth (`nets.validate_repeats`). So sizes come in
steps of 2 per axis. Anisotropic supercells are legal and give finer
granularity: `4 4 6` is 96 junctions between `4 4 4`'s 64 and `4 6 6`'s 144,
and the shrink target is emitted as a three-value box so the proportions
survive.

```
$ python3 cell_sizes.py table --strand n33 --des --conversion fraction:0.166667 --max-atoms 500000
 strand   repeats  junct  strands     atoms  mass/kDa  build box  target box  shrink
    n33   4^3        64       32     28192     192.9      60.2       6.84    8.79
    n33   6^3       216      108     95148     651.1      90.2      10.26    8.79
    n33   8^3       512      256    225536    1543.4     120.3      13.68    8.79
    n33  10^3      1000      500    440500    3014.5     150.4      17.11    8.79
```

Note what that table says about the production target: the experiment-facing
cell (n≈33, molar-ratio conversion, AcChCl) is *smaller* than the n=33
full-conversion demonstration (77 184 atoms), because at 1/6 conversion five
sixths of the net's edges hold no prepolymer at all. Size is not the obstacle
to a production run; composition still is.

## Conversion: fraction or count

`--conversion fraction:F` forms each strand with probability `F`. That is the
right model for "each reactive pair had this chance", and its realized count
scatters: the committed partial example asked for 1/6 of 192 and got 28, not
32.

`--conversion count:N` forms exactly `N`. An experimental formulation supplies
a functional-equivalent ratio, which fixes how many prepolymer molecules are
*present*; a binomial draw around that number would be an artifact of the
model rather than the chemistry. Both need a seed, because which strands form
must be reproducible.

Which one the production cell should use depends on an experimental number
nobody has confirmed yet -- whether `Hexakis:PPG-TDI = 1:0.5` is a molar, an
equivalent or a mass ratio (see `handoff/DEV_PLAN_20260905.md` §2). All three
readings are one `--conversion` argument apart.

## Resources are measured, not declared

`node_resources.py` reports cores, load, memory and GPU state, and recommends
a thread count that leaves the node usable: at most half the cores, minus what
`loadavg` says is already busy, minus a two-core reserve -- and hard-capped at
4 on a node whose GPUs are busy, because someone else's GPU job needs CPU
cores to feed it and starving it to speed up our EM is a bad trade. A GPU is
claimed only when it is idle by both utilization and memory.

`run_size.sh` calls it at launch, so the same command is polite on a quiet
workstation and on a busy GPU node.

## One loop that has to be closed after the build

For `fraction` mode the strand count is only known once the build has drawn
it, so the emitted shrink target is an estimate. `run_size.sh` recomputes the
target from the finished topology (`cell_sizes.py target`) and rewrites the
shrink maker before shrinking. Run by hand, do the same:

```
python3 cell_sizes.py target --top ../project/output_size_<tag>/system.top --aspect 4 4 4
```

## Usage

```
# preview
python3 cell_sizes.py table --strand n33 --des --max-atoms 400000

# generate makers for one size (no run)
python3 cell_sizes.py emit --repeats 6 --strand n33 --des

# generate, build, retarget, shrink, and log what it cost
./run_size.sh --repeats 6 --strand n33 --des
```

`run_size.sh` needs `gmx_mpi` and `packmol` on PATH; it sources GROMACS's own
`GMXRC` if GROMACS is installed but unloaded, and reports (rather than
silently fixes) a missing Packmol, which lives in the `hygel` conda
environment here. It sets `PYTHONPATH` to this checkout explicitly, because an
unset one imports the frozen Series-01 install instead.

Costs land in `run_ledger.tsv` (wall time and peak RSS per size), so the next
size can be chosen from what runs actually cost on a given node rather than
from a guess.

## What this does not do

* **No chain-length distribution.** The whole-strand mode places one rigid
  template, so a cell is monodisperse in strand length. Reproducing an
  experimental Mn/Đ would need a different layout, not a different size
  (`handoff/DEV_PLAN_20260905.md` item 14).
* **No PEO or water.** `--extra NAME:COUNT` will place any species with an ITP
  in `project/structure/`, and refuses rather than inventing one; PEO and
  water are not parameterized yet.
* **No opinion about physics.** A cell that fits the node is not a cell that
  answers a question. The shrink ends in an EM state; nothing may be read from
  it before NPT.
