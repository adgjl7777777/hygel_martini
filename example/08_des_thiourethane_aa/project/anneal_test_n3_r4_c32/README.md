# Annealing test on the 8,224-atom diagnostic cell (2026-09-16)

Purpose: does `config_npt/anneal.mdp` actually erase the builder's seeded
lattice? On the 28.5k PEG pilot, 30 ns of NPT at 300 K left 84-96% of the
lattice ordering in place (`check_structure.py`, fraction retained
0.885 / 0.956 / 0.841). This run answers whether heating removes it.

Input: `shrink_output_size_test_n3_r4_c32/final.gro` (dry network, no ions,
box 4.819 nm) with `output_size_test_n3_r4_c32/.../system.top`.
Schedule: 300 -> 600 K over 250 ps, hold to 2250 ps, cool to 300 K by 4250 ps,
settle to 5000 ps. NPT, C-rescale, 4 OpenMP threads, ~2.5 h wall.

## Result

Lattice memory (resname HEX, 4 repeats per axis, 5504 scatterers):

    S(k) first frame :  14.03  10.39   5.41
    S(k) last frame  :   0.54   1.41   0.61
    fraction retained:  0.000  0.044  0.000   (gate: <= 0.25)  -> PASS

The lattice is gone. This is the first configuration in the project whose
junction arrangement is not the builder's.

## Caveats, all real

* **Density rose from 1000 to 1276 kg/m3.** The shrink target for this cell was
  1.0 g/cm3; after annealing and NPT cooling it sits at 1.276. Either the dry
  network is genuinely that dense under this force field, or the 2 ns cool is
  fast enough to over-compact. Not resolved. Do not carry this density anywhere.
* **The run started cold.** `anneal.mdp` has `continuation = yes`,
  `gen_vel = no`, written to follow a heating stage. Started here from a shrink
  structure with no velocities, so frame 0 is at T = 4.7 K. The ramp reached
  600 K by 250 ps regardless, so the lattice result stands, but a production
  use must either generate velocities or follow `heat_free`.
* **This cell predates the neutralised release.** Net charge -0.0096 e
  (grompp NOTE + Ewald WARNING, run with -maxwarn 1). Rounding-level, harmless
  for a protocol test, one more reason not to read properties from it.
* **No ions in this cell**, so mobility was not tested here. The PEG pilot's
  mobility finding (chloride alpha 0.377, rms 0.151 nm at 300 K) is untested
  at 600 K.

## Files

`anneal.xtc` (15.8 MB), `anneal.edr`, `anneal.tpr` and the `.log` files are on disk here but **not tracked by git** (the repo .gitignore excludes trajectories, energy files and logs); regenerate
with `run.sh` if missing. The README, final structure, processed mdp, run script and the structure verdict are tracked.
`anneal.xtc.structure.json` is the `check_structure.py` verdict, sha256-bound
to the trajectory.
