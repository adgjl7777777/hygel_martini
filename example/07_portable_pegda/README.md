# Portable dry PEGDA construction

This example builds a small network through the installed builder, verifies
the planned attachments in its saved files, and runs GROMACS preprocessing
with `-maxwarn 0`. It does not minimize, solvate, run MD, or reproduce the
paper's property results. The existing `test_mode: true` skips preparation
calculations; the topology generator, writers, and guards execute normally.

## Run from a new environment

Use a built HyGel **0.1.1.dev0** wheel and this entire example directory. The
example also ships in the source distribution. Obtain `martini_v3.0.0.itp`
separately from the Martini distribution under its applicable terms; it is
not included here. Install GROMACS and provide the executable name or path.
The Python runtime and Python dependencies must also be installed. No NAS
mount, source checkout, Packmol, GPU, or research trajectory is needed.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install /path/to/hygel_martini-0.1.1.dev0-py3-none-any.whl
.venv/bin/python run.py --gmx gmx --martini-dir /path/to/martini_v300 --output ./result
```

The recorded environment used Python 3.12.12 and GROMACS 2026.0. To request
the same Python dependency versions, add
`-c requirements-tested-py312.txt` to the wheel installation command. The file
is a tested constraint snapshot for Python 3.12, not a lock for every platform.

Run outside the source checkout so it cannot shadow the installed package.
`--output` must be a new path. The runner saves the resolved input and preserves
logs on failure; it does not overwrite earlier results. Each external command
has a 300-second timeout (adjustable using `--timeout`). CPU thread counts are
limited to one. The Python import location and the Martini file checksum are
recorded in `RESULT.json`.

## Expected result

- `RESULT.json`: `status: PASS`, atoms **176**, bonds **184**, one connected
  component, **32** planned attachments.
- `build/planned_crosslinks.json`: exact one-based stub/endpoint index pairs.
- `build/initial_backbone.itp.plan_audit.json` and
  `build/initial_hydrogel.itp.plan_audit.json`: both `PASS`.
- `network_audit.json`: additional independently parsed graph diagnostics.
- `smoke.tpr` and `grompp.log`: successful GROMACS preprocessing.

The count expectations follow from 16 strands of 10 beads, 16 linker beads,
144 internal strand bonds, 32 attachments, and 8 internal linker bonds. This
short-chain model is a software example, not a physical PEGDA state point.
Byte-for-byte agreement of all output files across GROMACS/library versions
is not assumed. Compare graph identities, counts, and acceptance decisions;
checksums establish the provenance of each individual run.

The molecular template and bonded definitions are copied from the Series-01
PEGDA configuration. The cell count and chain length are reduced, water and
ions are disabled, and explicit-plan enforcement is enabled. Force-field
parameters have not been fitted or tuned for this example.

## Fault checks

The package tests cover complete loss of plan metadata, incomplete metadata,
reused endpoints, and missing/duplicated/rewired written attachments. Pipeline
fault-injection tests verify interruption before the next preparation step.
Those tests and this dry build have different roles; neither establishes
equilibrium or experimental material properties.
