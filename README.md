# hygel_martini

[한국어 README](README_KO.md)

`hygel_martini` is a research Python package for building and analyzing coarse-grained Martini hydrogels. It covers graph-first network construction, parameter preparation from QM and OPLS data, post-build relaxation, topology audits, and property extraction.

**Authors:** Daehong Kim ([ORCID 0009-0007-1647-9270](https://orcid.org/0009-0007-1647-9270)), Nayeon Jung, Bomin Kim, Won Bo Lee

**Affiliations:** Department of Chemical and Biological Engineering, Institute of Chemical Processes, Seoul National University, 1 Gwanak-ro, Gwanak-gu, Seoul 08826, Republic of Korea (all authors); School of Transdisciplinary Innovations, Seoul National University, 1 Gwanak-ro, Gwanak-gu, Seoul 08826, Republic of Korea (W. B. Lee)

The package has four parts:

- `param_opt`: QM, OPLS, and xTB inputs and Martini parameter preparation
- `hydrogel_builder`: hydrogel network construction and post-build relaxation
- `property_extract`: state, structure, transport, and finite-rate mechanics analyses
- `tools`: trajectory conversion and bonded-topology audits

The repository provides the workflows and Python code. GROMACS, Packmol, xTB, ORCA, Bartender, and the Martini force-field files have their own licenses and installation requirements, so the `pip` package neither bundles nor installs them.

Start with these two documents:

- [Small PEGDA installation and replay example](example/07_portable_pegda/README.md): installs the released wheel, builds a structure, checks the saved bonds, and runs GROMACS preprocessing.
- [Required settings and file checks for plan-based crosslinking](docs/EXPLICIT_PLAN_RELIABILITY.md): how the builder stops a run when plan records are missing or the saved bonds differ from the plan.

## Installation

Python 3.10 or later is required. Continuous integration tests Python 3.11 and 3.12 and installs the built wheel in a fresh environment outside the source tree.

```bash
git clone --recurse-submodules https://github.com/adgjl7777777/hygel_martini.git
cd /path/to/hygel_martini
python -m pip install -e .
```

For an existing clone, register the external PoreBlazer source with:

```bash
git submodule update --init --recursive
```

PoreBlazer is an independently maintained program and is not part of the `hygel-martini` wheel. Its own source and usage terms apply.

To install the development and packaging test tools:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Command-line tools

| Command | Purpose | Equivalent module call |
|---|---|---|
| `hygel-builder` | hydrogel construction | `python -m hygel_martini.hydrogel_builder` |
| `hygel-relax` | post-build relaxation | `python -m hygel_martini.hydrogel_builder.relax` |
| `hygel-property` | property extraction | `python -m hygel_martini.property_extract` |
| `hygel-qm-to-opls` | Stage 01 | `python -m hygel_martini.param_opt.qm_to_opls` |
| `hygel-opls-to-martini` | Stage 02 | `python -m hygel_martini.param_opt.opls_to_martini` |
| `hygel-qm-to-martini` | Stage 03 | `python -m hygel_martini.param_opt.qm_to_martini` |
| `hygel-parameter-protocol` | E0 to E6 bonded-parameter decision protocol | `python -m hygel_martini.param_opt.qm_to_martini.protocol` |
| `hygel-qm-reference-audit` | check of xTB ensembles against higher-level references | `python -m hygel_martini.param_opt.qm_to_martini.analysis.reference_qualification` |
| `hygel-audit-topology` | bonded-graph audit | `python -m hygel_martini.tools.audit_hydrogel_topology` |

`COMMAND --help` lists the current options of each command. The example launchers assume that the package is installed in the active Python environment. Shared Bash launchers are in `hygel_martini/bash_settings/`.

## Workflow examples

### 02. Existing OPLS/GROMACS to Martini

Stage 02 reuses an existing OPLS/GROMACS production trajectory for Bartender fitting. The repository does not ship trajectories; replace the `data/...` paths in `example/02_opls_to_martini/project/config/opls_existing_data.yaml` with your own data.

```bash
cd /path/to/hygel_martini/example/02_opls_to_martini/project
MODE=setup bash run_existing_opls.sh
MODE=md bash run_existing_opls.sh
MODE=md_notrim bash run_existing_opls.sh
```

`MODE` sets whether the trajectory is trimmed, whether Bartender jobs are generated, and whether they run. The full mode table is in `example/02_opls_to_martini/project/README.md`.

### 03. QM to Martini

```bash
cd /path/to/hygel_martini/example/03_qm_to_martini/project
bash run_qm_to_martini.sh config_common/common.yaml
```

In the trajectory-based modes (`xtb`, `existing`), the workflow detects the statistically converged part of a trajectory and trims the unstable start. If an xTB run stops, the workflow finds `xtbrestart`, continues the run, and merges the trajectories. An existing `xtb_traj.pdb` or trimmed trajectory can be fed to Bartender again with `run_compare.sh`, and `postprocess.sh` writes the screened force field. A minimal example:

```bash
LABEL=S \
BASE_CONFIG=config_common/common.yaml \
OUT_ROOT=compare_existing_terms/S/topology_n0 \
MD_TRAJ=md_S/S/relax_xtb_geoopt/xtb_traj.pdb \
TERM_MODE=topology_n \
TERM_N=0 \
MODE_TAG=topology_n0 \
bash run_compare.sh

LABEL=S \
MODE_TAG=topology_n0 \
INPUT_ROOT=compare_existing_terms/S/topology_n0 \
MIRROR_ROOT=compare_existing_terms \
OUTPUT_ROOT=postprocessing_result \
bash postprocess.sh
```

For repeated C/D/S runs, see `example/03_qm_to_martini/project/run_cds_iteration.sh`.

#### From candidate terms to tested-domain parameters

`hygel-parameter-protocol` keeps candidate generation separate from the final choice of parameters. The mapping, bead and nonbonded parents, candidate functions, data roles, independent groupings, objective, and thresholds are frozen with checksums first. The protocol then evaluates the cumulative evidence stages in order:

```text
E0 provenance -> E1 analytic eligibility -> E2 grouped selection
-> E3 numerical realization -> E4 target + upstream non-regression
-> E5 unopened one-shot confirmation -> E6 transfer qualification
```

If one criterion fails, the later stages of that iteration stay closed. Only values that pass E5 become a sealed tested-domain release. Do not use E6 results (length, single chain, dilute solution, hydrogel) to retune the coefficients of the same version.

The complete synthetic example runs with one command:

```bash
bash example/03_qm_to_martini/protocol_project/run_demo.sh
```

For real studies, reuse the schema and the order of the stages, not the numbers in the example. The details are in [`docs/PARAMETERIZATION_PROTOCOL.md`](docs/PARAMETERIZATION_PROTOCOL.md).

#### Checking xTB ensembles against higher-level references

`hygel-qm-reference-audit` post-processes existing calculations and reports four kinds of evidence separately, with no combined score:

- `energy`: relative-energy minimum, pairwise ordering, and MAE, RMSE, and maximum error of xTB against DFT or another higher-level reference
- `gradient`: RMS and maximum component of single-point gradients at the higher level, and stationarity
- `endpoint`: structural integrity and RMSD of independently optimized endpoints, classified as `SINGLE_DFT_ENDPOINT_FAMILY` or `MULTIPLE_DFT_ENDPOINTS` by relative energy
- `overlap`: effective sample size and largest single weight of the `E_reference - E_xTB` importance weights

```bash
hygel-qm-reference-audit energy reference_energies.csv \
  --group-column chemistry \
  --output reference_energy_audit.json

hygel-qm-reference-audit overlap delta_energies.csv \
  --temperature-k 310 \
  --output overlap_audit.json
```

The input CSV schemas and the meaning of each result are in `hygel_martini/param_opt/qm_to_martini/analysis/README.md`. Passing this check means that sparse refinement can start. It does not make xTB a DFT ground truth or finalize the Martini parameters.

### 04. Full builder

```bash
cd /path/to/hygel_martini/example/04_full_builder/project
bash run_full_builder.sh maker.yaml
```

Anisotropy smoke test:

```bash
bash run_full_builder.sh maker_anisotropy_x.yaml
```

### 04_1. Example system

```bash
cd /path/to/hygel_martini/example/04_1_example_system/project
bash run_example_system.sh maker.yaml
```

### 05. Hydrogel relaxation

```bash
cd /path/to/hygel_martini/example/05_hydrogel_relaxation/project
bash run_hydrogel_relaxation.sh maker_soft_em.yaml
bash run_hydrogel_relaxation.sh maker_soft_md.yaml
```

### 06. Physical properties

```bash
cd /path/to/hygel_martini/example/06_physical_property
hygel-property requirements \
  --analysis analysis_jobs.yaml \
  --requirements md_requirements.yaml \
  --strict
hygel-property analyze \
  --analysis analysis_jobs.yaml \
  --requirements md_requirements.yaml \
  --manifest validation_manifest.yaml
```

This stage keeps three verdicts separate: calculation success, analyzability, and direct comparability with experiment. The configuration format is described in `hygel_martini/property_extract/README.md`.

The package includes the analysis routines used in the paper workflow: reduced-network topology audits, PBC-safe geometry and diffusion, periodic clearance, static structure factors, voxel and field correlations, contact aggregation, paired finite-rate mechanics, and realization-level statistics.

`docs/VALIDATION_HISTORY_AND_DESIGN_RATIONALE.md` (in Korean) records the validation failures and corrections behind these routines. Read its sections on fixed-water versus free-swelling states, pores versus clearance, and finite-rate versus equilibrium mechanics before interpreting results.

## Builder details

### Running the builder and relaxation

```bash
hygel-builder path/to/maker.yaml
hygel-builder --config path/to/maker.yaml
hygel-relax path/to/maker_soft_em.yaml
```

A builder run passes through:

1. `run_example_system.sh` or `run_full_builder.sh` in the `04_*` examples
2. `hygel-builder` or `python -m hygel_martini.hydrogel_builder`
3. `hygel_martini.hydrogel_builder.generator.run_hydrogel_builder`
4. `hygel_martini.hydrogel_builder.config_params.generator.run_hydrogel_example`
5. `hygel_martini.hydrogel_builder.config_params.read_json.execute_mode`

Post-build relaxation passes through:

1. `05_hydrogel_relaxation/project/run_hydrogel_relaxation.sh`
2. `hygel-relax` or `python -m hygel_martini.hydrogel_builder.relax`
3. `hygel_martini.hydrogel_builder.relax.generator.run_relax_workflow`
4. `hygel_martini.hydrogel_builder.relax.soft_em` or `hygel_martini.hydrogel_builder.relax.soft_md`

### BCK dynamic crosslinks

With `dynamic_crosslink_targets_per_stub: 2`, each BCK stub bonds to two nearby backbone ends. A linker with two BCK beads therefore makes four BCK-backbone bonds and two polymer junctions:

```text
BCK1: polymer A -- polymer B
BCK2: polymer C -- polymer D
```

The internal BCK1-BCK2 bond exists only if the linker template or topology defines it. In `connectivity_aware` mode, the exact endpoint-edge plan made by the layout planner is carried into the linker stub metadata, and the runtime crosslinker writes those edges as planned. The builder uses distance only to decide which planned edge goes to which physical BCK stub; the nearest-neighbor search does not reroute a planned endpoint to another chain. Layouts without plan metadata use geometry-only endpoint assignment with a global check that each endpoint is used once.

In `connectivity_aware` mode, the layout planner places BCK junctions by local matching before any bond is written. At each local vertex, the four positions `(000), (011), (101), (110)` have three perfect matchings, labeled x, y, and z:

```text
x: (000-011), (101-110)
y: (000-101), (011-110)
z: (000-110), (011-101)
```

The planner balances the x, y, and z choices across all vertices and makes the graph of backbone segments and BCK junction edges a single closed circuit where it can. The planner is a pure graph routine in `hygel_martini/hydrogel_builder/core_utils/layout/local_matching.py`. Small lattices are searched exhaustively over assignments whose x, y, and z counts differ by at most one. Large lattices use balanced Kotzig-style two-vertex swaps, Metropolis annealing, and sampled greedy refinement.

```yaml
simulation_parameters:
  # Turns on the layout graph planner, which chooses the x/y/z transitions.
  linker_orientation_strategy: connectivity_aware

  # Number of backbone ends each placed BCK stub bonds to.
  # 2 means one two-BCK linker = two polymer junctions = four BCK-backbone bonds.
  dynamic_crosslink_targets_per_stub: 2

  # Candidate width of the runtime endpoint search when no plan metadata exist.
  # It does not limit loops or components.
  dynamic_crosslink_candidate_limit: 64

# Safety check on the bonded graph of initial_hydrogel.gro/itp after the build.
# It does not control the x/y/z optimization or the number of loops.
hydrogel_topology_connectivity_audit:
  enabled: true
  min_largest_component_fraction: 0.95
  max_components: 1
  fail_on_violation: true
```

`connectivity_guard` is the old name of `hydrogel_topology_connectivity_audit`. New YAML files should use the new name; the old one is still read for compatibility.

## Configuration files

Builder examples usually include several YAML files from `maker.yaml`:

```yaml
includes:
  - config/simulation.yaml   # run environment, GPU, threads, GROMACS path
  - config/hydrogel.yaml     # backbone and linker types and ratios
  - config/mdp.yaml          # EM MDP overrides
  - config/add_series.yaml   # water and ion addition
  - config/backbone.yaml     # backbone bonded parameters
```

Two placeholders are common:

- `${CONFIG_DIR}`: the directory of the current maker file
- `${REPO_ROOT}`: the repository root

```yaml
simulation_parameters:
  output_dir: ${CONFIG_DIR}/output
  gromacs_include_path: ${REPO_ROOT}/martini_v300
```

### Main `simulation.yaml` parameters

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `gromacs_executable_path` | str | `gmx_mpi` | GROMACS executable |
| `gpu_id` | int or null | `null` | null runs on CPU only; N pins GPU N |
| `omp_threads` | int or null | `null` | mdrun threads for builder EM; null lets GROMACS decide |
| `segment_length` | int | none | CG beads per backbone |
| `number_of_cells` | int | 1 | repeats of the large cell |
| `mean_sep` | float | none | mean bead spacing (nm), also used for water packing |
| `anisotropy` | false or str | `false` | false for isotropic; x, y, or z for anisotropy along that axis |
| `gel_weight_fraction_mode` | str | `legacy` | how the water count is made an integer and whether ions count: `legacy` rounds down; `exclude_ions` and `include_ions` both round up |
| `grompp_maxwarn` | int | 1 | grompp warnings allowed in builder EM steps (ion addition uses a fixed 2) |
| `random_seed` | int | none | random seed for reproducible builds |
| `debug_mode` | bool | false | true writes a detailed log to output/debug.txt |
| `neutral_em_coulombtype` | str or null | `Cut-off` | Coulomb type for the neutral EM step |
| `packmol_pbc` | bool | false | adds an orthorhombic `pbc` line to Packmol input (Packmol 20.15 or later) |

## Running on shared servers

### GPU use

GROMACS 2026 `gmx_mpi mdrun` takes a GPU automatically unless it is given `-nb cpu`. To avoid colliding with other users' GPU jobs, set the device in `config/simulation.yaml`:

```yaml
simulation_parameters:
  # null -> CPU only (default; recommended on shared servers)
  # 0    -> GPU 0 only (CUDA_VISIBLE_DEVICES=0)
  # 1    -> GPU 1 only (CUDA_VISIBLE_DEVICES=1)
  gpu_id: null
```

For `05` soft_em and soft_md, use `runtime.gpu_id` in `config/common.yaml`.

### CPU threads

mdrun threads for the builder EM steps (`backbone_stage`, `initial_hydrogel`, and others):

```yaml
simulation_parameters:
  # null -> GROMACS chooses (it may take every core; be careful on shared servers)
  # N    -> mdrun -ntomp N
  omp_threads: 4
```

For `05` soft_em and soft_md, use `runtime.omp_threads` in `config/common.yaml` or `ntomp` in each section. Under SLURM, `SLURM_CPUS_PER_TASK` (or `SLURM_NTASKS`) is used when `omp_threads` is null.

## Environment variables

- `ADDITIONAL_BASH_PROFILE`: an extra bash profile to source, if needed
- `ENV_NAME`: a conda environment to activate, if needed
- `ENVIRONMENT_FILE`: the project-local shell settings each example launcher sources first
- `GMXRC_PATH`: the GMXRC used to set up GROMACS
- `GMX_CMD`: the GROMACS command as a shell-level override; `gromacs_executable_path` in YAML takes precedence
- `SLURM_CPUS_PER_TASK`, `SLURM_NTASKS`: used as mdrun `-ntomp` under SLURM when `omp_threads` is null
- `OPENMPI_HOME`, `ORCA_HOME`: ORCA paths

`OMP_NUM_THREADS` and `GMX_OPENMP_MAX_THREADS` do not need to be set. Builder EM threads come from `omp_threads` in YAML, and relaxation threads from `runtime.omp_threads`.

For `02_opls_to_martini`, set the GROMACS and Bartender paths in `config/opls_existing_data.yaml` or `project/environment.sh`. For `03_qm_to_martini`, edit the xTB, ORCA, and Bartender paths in `config_common/common.yaml` rather than in shell variables. In either case, set environment variables or YAML values instead of editing the scripts.

## Output files and debugging

### `hydrogel_builder`

Common outputs:

- `output/debug.txt`
- `output/dynamic_bonding_debug.log`
- `output/system.top`
- `output/final_system.gro`
- `output/final_optimized_system.gro`
- `output/*_geo_opt/grompp.log`
- `output/*_geo_opt/mdrun.log`
- `output/*_geo_opt/mdout.mdp` (the final parameter summary written by grompp)

If a build fails, check these in order: `debug.txt`, the failing step's `grompp.log` and `mdrun.log`, `dynamic_bonding_debug.log`, and `system.top`.

### `param_opt`

- `03`: `runs/.../summary.json`
- `02`: `opls_bartender_runs/.../summary.json`

### `hydrogel_builder.relax`

- `05`: `relax_output/soft_em/final.gro` and `relax_output/soft_md/soft_md.gro`

## Repository layout

- `hygel_martini/`
  - `core/`: physical constants, utilities, and the configuration loader (`physics.py`, `utils.py`, `config.py`)
  - `tools/`: general utilities such as `xtb_traj_to_pdb.py`
  - `bash_settings/`: Bash launchers for all workflows (`common/` for shared environment and Slurm templates; `param_opt/` for stages 02 and 03; `hydrogel_builder/` for stage 04; `relaxation/` for stage 05)
  - `param_opt/`: parameter preparation, stages 00 to 03 (`bead_generator` is planned for stage 00; `qm_to_opls` 01; `opls_to_martini` 02; `qm_to_martini` 03 with separate `workflow_logic` and `analysis`; `polymer_maker` is the shared polymer xyz generator)
  - `hydrogel_builder/`: the graph-first builder and relaxation, stages 04, 04_1, and 05
  - `property_extract/`: manifest-driven property analyses and comparison checks, stage 06
- `example/`: example projects included in a fresh clone (02 needs your own trajectories), with commented reference YAML files. If in doubt, look only at the `project/` directories under `example/`.
- `example_myrun/`: a local run directory listed in `.gitignore`; it is not in a fresh clone.
- `martini_v300/`: local Martini force-field files that you provide; they are not distributed.

Run the `param_opt` workflow modules directly, not the `hygel_martini.param_opt` root:

```bash
python -m hygel_martini.param_opt.bead_generator --help
python -m hygel_martini.param_opt.qm_to_opls --config ...
python -m hygel_martini.param_opt.opls_to_martini --config ...
python -m hygel_martini.param_opt.qm_to_martini --config ...
python -m hygel_martini.param_opt.qm_to_martini.analysis.reference_qualification --help
```

## Further documentation

Several of these documents are in Korean.

- `docs/VALIDATION_HISTORY_AND_DESIGN_RATIONALE.md`: validation failures, corrections, exclusion criteria, and how the current design follows from them
- `example/README_ko.md`: layout of the example projects (Korean)
- `hygel_martini/param_opt/README.md`: stages 00 to 03
- `hygel_martini/hydrogel_builder/README.md`: stages 04, 04_1, and 05
- `hygel_martini/property_extract/README.md`: stage 06 analysis jobs, validation manifests, and requirement checks

## Citation, funding, and license

Citation information is in [`CITATION.cff`](CITATION.cff). It will list the methodology paper next to the software release once the paper is published.

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-25424498).

When you publish results obtained with this software, please also cite the third-party tools your workflow used: GROMACS (Abraham et al., 2015), Martini 3 (Souza et al., 2021) or Martini 2 (Marrink et al., 2007), Packmol (Martínez et al., 2009), MDAnalysis (Michaud-Agrawal et al., 2011; Gowers et al., 2016), NumPy (Harris et al., 2020), SciPy (Virtanen et al., 2020), pandas (McKinney, 2010), Numba (Lam et al., 2015), Matplotlib (Hunter, 2007), ASE (Hjorth Larsen et al., 2017), CoolProp (Bell et al., 2014), pymbar/MBAR (Shirts and Chodera, 2008), and PoreBlazer (Sarkisov et al., 2020). The DOIs are listed under `references` in `CITATION.cff`.

Parts of the code, tests, and documentation were drafted or revised with AI coding assistants (Google Gemini, OpenAI Codex, and Anthropic Claude). The authors reviewed all changes and take responsibility for the software.

The source code is released under the MIT License ([`LICENSE`](LICENSE)). For the licenses of the Martini force-field files and third-party dependencies, see [`LICENSING.md`](LICENSING.md).

## Release notes

### 0.1.3

Documentation only. The README is now in English, with the Korean version in `README_KO.md`, and the list of AI coding assistants includes Google Gemini. The code is unchanged from 0.1.2 apart from the version number.

### 0.1.2

Fixes two side-chain (monomer) placement defects and the handling of a missing bonded-patch file, and adds optional Packmol PBC. Configurations without side chains (`MONOMERS: []`, such as PEGDA and Pluronic) write output byte-identical to 0.1.1.

- Linker stubs receive no side chains. A stub's `backbone_type` is used only to choose crosslink partners. Version 0.1.1 attached a second side chain to a stub when a matching monomer existed.
- If the builder finds no clash-free position for a side chain, the build stops with `RuntimeError` instead of warning. Version 0.1.1 skipped the site and wrote a structure without that side chain. In the free-polymer route, each new side chain is also checked against those already placed.
- If `bonded_topology_patch_file` is set but the file is missing, the build stops with `FileNotFoundError`. Version 0.1.1 printed a notice and skipped the file, which left the generic angle terms in place and wrote no backbone torsions. The default path (`config/backbone.yaml`) is still optional when the key is not set. The portable PEGDA example now registers the PEO-BCK-PEO angle rule explicitly; its output is unchanged.
- `simulation_parameters.packmol_pbc: true` adds an orthorhombic `pbc` line to the Packmol input (Packmol 20.15 or later) and stops if the fixed base lies outside the box `[0, L]`. The default is `false`, in which case the following minimization removes contacts across the box boundary.

### 0.1.1

First public release for the JCC methodology paper (Zenodo DOI 10.5281/zenodo.23208480). The software snapshot used for the earlier Series-01 work is `0.1.0` (`2472810c9430`).
