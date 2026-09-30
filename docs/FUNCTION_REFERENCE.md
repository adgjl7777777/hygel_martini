# Function reference

**Generated — do not edit.** Regenerate with
`tools/gen_function_reference.py --write`; `tests/test_function_reference.py`
fails when this file and the source disagree.

Covers 164 modules, 77 classes, 951 functions and methods under `hygel_martini`.

For orientation — architecture, the config system, how to apply each
example series, the invariants that break silently, and what a build does
and does not license you to claim — read [`../README_FOR_LLM.md`](../README_FOR_LLM.md)
first. This file is the index, not the map.

### How to read it, and what it cannot tell you

Everything below is static analysis over the syntax tree:

- **`calls`** lists distinct names called in the body, in first-appearance
  order, capped at twelve. Dynamic imports, `getattr` dispatch and
  runtime-registered callables are invisible here.
- **`effects`** are *hints* inferred from call names (`subprocess`,
  `filesystem`, `Config/runtime state`, `global registry`, `stdout`), not a
  proof. A function with no hint may still mutate state through a helper.
- **`returns`** shows up to three distinct returned expressions verbatim, so
  a function returning several shapes is visible as such.
- **`(no docstring)`** means exactly that. It is never replaced with
  generated prose, so this doubles as the list of undocumented functions.
- Line numbers are from the generation commit. If they are off, regenerate.

Two design facts worth knowing before reading any single function, both
detailed in `README_FOR_LLM.md`: `main_components/Universe.World` keeps
class-level registries, so constructing an `Atom` or `Bond` *is* a mutation;
and `config_params/config.Config` holds the loaded config, its path and
runtime state as class variables. Reading one function in isolation will
mislead you about both.

## Contents

- [`hygel_martini`](#hygel_martini)
- [`hygel_martini/core`](#hygel_martinicore)
- [`hygel_martini/hydrogel_builder`](#hygel_martinihydrogel_builder)
- [`hygel_martini/param_opt`](#hygel_martiniparam_opt)
- [`hygel_martini/property_extract`](#hygel_martiniproperty_extract)
- [`hygel_martini/tools`](#hygel_martinitools)

## `hygel_martini`

### `hygel_martini/__init__.py`

HyGel Martini public package namespace.

## `hygel_martini/core`

### `hygel_martini/core/__init__.py`

*(no module docstring)*

### `hygel_martini/core/config.py`

Shared maker-config loading for hygel_martini command-line workflows.

#### `deep_update(base: Dict[str, Any], override: Dict[str, Any])` — line 42
- Return a deep copy of ``base`` with ``override`` merged recursively.
- kind: function
- returns: `result`
- calls: `copy.deepcopy`, `override.items`, `deep_update`, `result.get`

#### `_load_yaml(path: Path)` — line 57
- Load one YAML file whose root must be a mapping (empty file -> {}).
- kind: function, internal
- returns: `data`; `{}`
- raises: `ValueError`, `RuntimeError`
- calls: `yaml.safe_load`, `path.read_text`, `ValueError`, `RuntimeError`

#### `_load_single_config(path: Path)` — line 79
- Load one .yaml/.yml/.json config file without include processing.
- kind: function, internal
- returns: `_load_yaml(path)`; `data`
- raises: `ValueError`
- calls: `path.suffix.lower`, `ValueError`, `_load_yaml`, `json.loads`, `path.read_text`

#### `_load_with_includes(path: Path, seen: List[Path] | None=None)` — line 96
- Load a config file, recursively merging its ``includes`` chain.
- kind: function, internal
- returns: `merged`
- raises: `ValueError`
- calls: `path.resolve`, `_load_single_config`, `data.pop`, `deep_update`, `join`, `ValueError`, `resolve`, `_load_with_includes`

#### `_resolve_path_value(value: str, config_dir: Path)` — line 138
- Expand one path string to an absolute, resolved filesystem path.
- kind: function, internal
- returns: `str(path_obj.resolve())`
- calls: `os.path.expanduser`, `resolved.replace`, `Path`, `os.path.expandvars`, `path_obj.is_absolute`, `path_obj.resolve`

#### `_normalize_paths(cfg: Dict[str, Any], config_path: Path | None)` — line 154
- Resolve path-like values under the top-level ``paths`` section.
- kind: function, internal
- returns: `result`; `cfg`
- calls: `copy.deepcopy`, `result.get`, `config_path.resolve`, `path_section.items`, `config_path.exists`, `key.endswith`, `_resolve_path_value`

#### `_parse_override_value(raw: str)` — line 182
- Coerce a ``--set`` value string to a Python scalar or container.
- kind: function, internal
- returns: `value`; `''`; `True` (+6 more)
- calls: `raw.strip`, `value.lower`, `_INT_RE.match`, `_FLOAT_RE.match`, `yaml.safe_load`, `json.loads`

#### `_apply_set_override(cfg: Dict[str, Any], expr: str)` — line 225
- Apply one ``key.path=value`` override to ``cfg`` in place.
- kind: function, internal
- raises: `ValueError`, `TypeError`
- calls: `expr.split`, `_parse_override_value`, `ValueError`, `part.strip`, `current.get`, `path_text.split`, `TypeError`, `join`

#### `load_config(config_path: Path | None, default_config: Dict[str, Any] | None=None)` — line 257
- Load a maker config merged over workflow defaults.
- kind: function
- returns: `_normalize_paths(cfg, config_path)`
- calls: `copy.deepcopy`, `_normalize_paths`, `config_path.exists`, `_load_with_includes`, `deep_update`

#### `apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace)` — line 276
- Overlay recognized argparse options onto a copy of the config.
- kind: function
- returns: `result`
- calls: `copy.deepcopy`, `result.setdefault`, `parse_csv_list`, `parse_semicolon_list`, `parse_int_csv`, `_apply_set_override`

#### `add_config_args(parser: argparse.ArgumentParser)` — line 341
- Register the ``--config`` / ``--dump-default-config`` options.
- kind: function
- calls: `parser.add_argument`

#### `add_sequence_override_args(parser: argparse.ArgumentParser)` — line 355
- Register the sequence-selection override options shared by stages.
- kind: function
- calls: `parser.add_argument`

#### `add_opls_to_martini_cli_args(parser: argparse.ArgumentParser)` — line 367
- Register the full option set of the 02 opls_to_martini CLI.
- kind: function
- calls: `add_config_args`, `add_sequence_override_args`, `parser.add_argument`

#### `add_qm_to_martini_cli_args(parser: argparse.ArgumentParser)` — line 397
- Register the option set of the 03 qm_to_martini CLI.
- kind: function
- calls: `add_config_args`, `add_sequence_override_args`, `parser.add_argument`

### `hygel_martini/core/gro.py`

One GRO reader for the whole package.

#### class `GroAtom` — line 40
One atom record from a GRO file.

#### class `GroFrame` — line 51
A parsed GRO file: title, atoms, and the periodic cell if present.

##### `__len__(self)` — line 59
- *(no docstring)*
- kind: method
- returns: `len(self.atoms)`

##### `positions(self)` — line 63
- (N, 3) coordinates in nm; empty (0, 3) array for an empty frame.
- kind: property
- returns: `np.array([atom.position for atom in self.atoms], dtype=float)`; `np.empty((0, 3), dtype=float)`
- calls: `np.array`, `np.empty`

##### `atom_names(self)` — line 70
- Atom names in file order.
- kind: property
- returns: `[atom.atom_name for atom in self.atoms]`

#### `_parse_tail(line: str, path: str, line_number: int)` — line 75
- Atom index and position from the part of a record after column 15.
- kind: function, internal
- returns: `(int(parts[0]), np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float))`; `(int(index_field), np.array([float(body[step * width:(step + 1) * width]) for step in range(3)], dtype=float))`
- raises: `ValueError`
- calls: `rstrip`, `index_field.strip`, `tail.split`, `ValueError`, `divmod`, `np.array`, `line.rstrip`

#### `read_gro(path: str | Path)` — line 123
- Parse a GRO file into title, atom records and periodic cell.
- kind: function
- returns: `GroFrame(title=title, atoms=atoms, box=box)`
- raises: `ValueError`
- calls: `read_text`, `text.splitlines`, `strip`, `_parse_box`, `GroFrame`, `ValueError`, `_parse_tail`, `atoms.append`, `Path`, `GroAtom`, `line.rstrip`

#### `_parse_box(line: str, path: str, line_number: int)` — line 174
- Parse a GRO box line, orthorhombic or triclinic.
- kind: function, internal
- returns: `box`; `None`
- raises: `ValueError`
- calls: `np.diag`, `ValueError`, `np.array`, `np.linalg.det`, `line.split`

#### `read_gro_atoms(path: str | Path)` — line 205
- Atom records only, for callers that ignore the periodic cell.
- kind: function
- returns: `read_gro(path).atoms`
- calls: `read_gro`

#### `read_gro_atom_names(path: str | Path)` — line 210
- Atom names in file order.
- kind: function
- returns: `read_gro(path).atom_names`
- calls: `read_gro`

### `hygel_martini/core/itp.py`

GROMACS topology (ITP) parsing for the whole package.

#### `read_atom_types(itp_file_path)` — line 32
- Map atom-type name -> {'mass': amu} from an ITP's [ atomtypes ].
- kind: function
- returns: `atom_types`; `{}`
- raises: `DuplicateDeclaration`
- effects: filesystem, stdout
- calls: `print`, `open`, `line.strip`, `re.match`, `line.startswith`, `rstrip`, `lower`, `line.split`, `atom_types.get`, `match.group`, `DuplicateDeclaration`

#### `read_itp_definitions(itp_file_path, atom_type_masses=None, prefer_explicit_masses=False, require_mass=True)` — line 134
- Parses a Martini .itp file and extracts molecule definitions.
- kind: function
- returns: `definitions`
- raises: `DuplicateDeclaration`, `ValueError`
- effects: filesystem
- calls: `sec_name.lower`, `sec_lower.startswith`, `current_molecule.setdefault`, `append`, `open`, `sec_lower.endswith`, `line.strip`, `re.match`, `other.setdefault`, `line.startswith`, `rstrip`, `lower`, +8 more

### `hygel_martini/core/pbc.py`

One minimum-image convention for the whole package.

#### `_shift_grid(reach: int)` — line 42
- All integer shifts within ``reach`` cells along each axis.
- kind: function, internal
- returns: `np.array([(i, j, k) for i in span for j in span for k in span], dtype=float)`
- calls: `np.array`

#### `normalize_cell(box)` — line 54
- Coerce a box specification to a 3x3 cell with rows as cell vectors.
- kind: function
- returns: `None`; `np.diag(cell)`; `cell`
- raises: `ValueError`
- calls: `np.asarray`, `ValueError`, `np.any`, `np.diag`, `np.linalg.det`, `cell.tolist`

#### `is_orthorhombic(cell: np.ndarray, tolerance: float=1e-09)` — line 73
- True when the cell is diagonal to within ``tolerance``.
- kind: function
- returns: `bool(np.all(np.abs(off_diagonal) <= tolerance))`
- calls: `np.diag`, `np.all`, `np.abs`

#### `nearest_image_reach(cell: np.ndarray, max_reach: int=_MAX_REACH)` — line 79
- Smallest shift range whose optimum is strictly interior.
- kind: function
- returns: `reach`
- raises: `ValueError`
- calls: `ValueError`, `_SHIFT_CACHE.setdefault`, `np.round`, `np.argmin`, `np.all`, `_shift_grid`, `np.linalg.solve`, `np.einsum`, `np.abs`, `cell.tolist`

#### `minimum_image(delta, cell)` — line 107
- Shortest periodic image of a displacement, or of a stack of them.
- kind: function
- returns: `result[0] if single else result.reshape(displacement.shape)`; `displacement`; `displacement - lengths * np.round(displacement / lengths)`
- raises: `ValueError`
- calls: `np.asarray`, `is_orthorhombic`, `nearest_image_reach`, `_SHIFT_CACHE.setdefault`, `np.round`, `np.argmin`, `ValueError`, `normalize_cell`, `np.diag`, `_shift_grid`, `displacement.reshape`, `np.linalg.solve`, +3 more

#### `minimum_image_distance(first, second, cell)` — line 144
- Shortest periodic distance between two positions.
- kind: function
- returns: `float(np.linalg.norm(minimum_image(delta, cell)))`
- calls: `np.asarray`, `np.linalg.norm`, `minimum_image`

#### `wrap_into_cell(positions, cell)` — line 150
- Wrap Cartesian positions into the primary cell.
- kind: function
- returns: `(fractional @ matrix).reshape(coordinates.shape)`; `coordinates`; `np.mod(coordinates, np.diag(matrix))`
- calls: `np.asarray`, `normalize_cell`, `is_orthorhombic`, `np.floor`, `reshape`, `np.mod`, `np.linalg.solve`, `np.diag`, `coordinates.reshape`

### `hygel_martini/core/physics.py`

Water-related physical property helpers for solvation box setup.

#### `water_density_g_cm3(temp_c: float)` — line 13
- Return water density in g/cm^3 at the target temperature. Prefer CoolProp. Fallback to Kell equation approximation (0-100C).
- kind: function
- returns: `rho_kg_m3 / 1000.0`
- calls: `PropsSI`

#### `estimate_water_molecules(box_ang: Sequence[float], density_g_cm3: float, molar_mass: float, avogadro: float)` — line 40
- Estimate how many water molecules fill a rectangular box.
- kind: function
- returns: `max(0, n_molecules)`

### `hygel_martini/core/utils.py`

Small text-parsing and box-sizing helpers shared across hygel_martini.

#### `parse_csv_list(text: str)` — line 13
- Split comma-separated text into stripped, non-empty tokens.
- kind: function
- returns: `[token.strip() for token in text.split(',') if token.strip()]`
- calls: `token.strip`, `text.split`

#### `parse_semicolon_list(text: str)` — line 18
- Split semicolon-separated text into stripped, non-empty tokens.
- kind: function
- returns: `[token.strip() for token in text.split(';') if token.strip()]`
- calls: `token.strip`, `text.split`

#### `parse_int_csv(text: str)` — line 23
- Parse a comma-separated list of integers.
- kind: function
- returns: `values`
- raises: `ValueError`
- calls: `parse_csv_list`, `values.append`, `ValueError`

#### `sequence_name(symbol: str, n_repeat: int)` — line 44
- Return the homopolymer sequence label (symbol repeated n_repeat times).
- kind: function
- returns: `symbol * n_repeat`

#### `ensure_min_box_nm(box_nm: Sequence[float], cutoff_nm: float, safety_nm: float)` — line 49
- Clamp each box edge to the minimum-image-safe minimum length.
- kind: function
- returns: `[max(value, min_len) for value in box_nm]`

## `hygel_martini/hydrogel_builder`

### `hygel_martini/hydrogel_builder/__init__.py`

Hydrogel construction package namespace.

#### `run_hydrogel_builder(*args, **kwargs)` — line 6
- *(no docstring)*
- kind: function
- returns: `_run_hydrogel_builder(*args, **kwargs)`
- calls: `_run_hydrogel_builder`

### `hygel_martini/hydrogel_builder/__main__.py`

*(no module docstring)*

### `hygel_martini/hydrogel_builder/add_series/__init__.py`

Post-build solvation stage: add water and small ions to a built hydrogel.

### `hygel_martini/hydrogel_builder/add_series/add_small_ion.py`

Ion insertion helpers used by the final hydrogel packing stage.

#### `_run_checked(cmd, label, cwd=None, env=None, input_text=None)` — line 49
- Run a subprocess through the shared logging wrapper.
- kind: function, internal
- returns: `proc`
- raises: `subprocess.CalledProcessError`
- effects: subprocess
- calls: `_run_with_logs`, `subprocess.CalledProcessError`

#### `_partition_ion_definitions(ion_list)` — line 72
- Split configured ions into primary and compensating pools.
- kind: function, internal
- returns: `(cations, anions, extra_cations, extra_anions, total_charge)`
- calls: `append`, `ion.get`

#### `_ensure_compensation_pool(primary_pool, compensation_pool, label)` — line 107
- Guarantee that a compensation pool contains at least one ion species.
- kind: function, internal
- returns: `None` (bare return)
- raises: `ValueError`
- calls: `compensation_pool.append`, `ValueError`, `primary_pool.pop`

#### `_apply_residual_charge(total_charge, compensation_pool, seed)` — line 129
- Increase compensation-ion counts so the staged system can be neutralized.
- kind: function, internal
- returns: `None` (bare return)
- raises: `ValueError`
- calls: `np.array`, `astype`, `np.dot`, `ValueError`, `itertools.product`, `random.Random`, `np.floor`, `rng.choice`, `valid_solutions.append`, `ion.get`

#### `resolve_effective_ion_plan(ion_params, seed=None)` — line 190
- Predict the effective ion counts after compensation adjustments.
- kind: function
- returns: `anion_list + cation_list`; `[]`
- effects: Config/runtime state
- calls: `copy.deepcopy`, `_partition_ion_definitions`, `_ensure_compensation_pool`, `additional_anion_list.reverse`, `additional_cation_list.reverse`, `anion_list.extend`, `cation_list.extend`, `get`, `_apply_residual_charge`, `Config.get_param`

#### `run_genion_for_neutralization(input_gro, output_gro, topology_file, sim_params, ion_params, solvent_name)` — line 238
- Run GROMACS genion to add ions and neutralize the system.
- kind: function
- returns: `{'output_gro': output_gro, 'ion_counts': ion_counts_summary}`
- effects: filesystem, Config/runtime state, stdout
- calls: `print`, `sim_params.get`, `os.path.basename`, `shutil.copy`, `os.environ.copy`, `_partition_ion_definitions`, `_ensure_compensation_pool`, `get`, `additional_anion_list.reverse`, `additional_cation_list.reverse`, `anion_list.extend`, `cation_list.extend`, +17 more

#### `_reorder_water_and_ions(gro_path, water_resname, ion_names)` — line 488
- Reorder a GRO file so water and ions follow topology ordering.
- kind: function, internal
- returns: `None` (bare return)
- effects: filesystem, stdout
- calls: `print`, `Config.debug_log`, `open`, `f.readlines`, `strip`, `reordered.extend`, `f.write`, `waters.append`, `ions_by_name.get`, `append`, `prefix.append`

### `hygel_martini/hydrogel_builder/add_series/add_water.py`

Helpers for estimating solvent content in hydrogel construction.

#### `get_weighted_average_mass(*args)` — line 59
- Compute a ratio-weighted mean molecular mass for configured components.
- kind: function
- returns: `total_mass / total_ratio if total_ratio > 0 else 0`; `0`
- effects: Config/runtime state, stdout
- calls: `Config.get_param`, `print`, `load_monomer_templates`, `component.get`, `bead.get`, `get`

#### `_safe_get_param(*keys, default=None)` — line 105
- Fetch a config value by key path, returning ``default`` when absent.
- kind: function, internal
- returns: `Config.get_param(*keys)`; `default`
- effects: Config/runtime state
- calls: `Config.get_param`

#### `_resolve_gel_weight_fraction_mode(sim_params)` — line 113
- Normalize and validate ``gel_weight_fraction_mode`` from the config.
- kind: function, internal
- returns: `mode`
- raises: `ValueError`
- calls: `lower`, `ValueError`, `strip`, `sim_params.get`

#### `_load_definition_lookup(itp_paths)` — line 136
- Build a molecule-name -> ITP definition map from candidate ITP files.
- kind: function, internal
- returns: `definitions`
- effects: Config/runtime state
- calls: `Config.get_runtime`, `definitions.update`, `read_itp_definitions`, `os.path.isfile`

#### `_estimate_ion_usage(sim_params)` — line 167
- Predict how many ions the later ion stage will insert, and their mass.
- kind: function, internal
- returns: `(total_ion_count, total_ion_mass)`; `(0, 0.0)`
- raises: `ValueError`
- calls: `resolve_effective_ion_plan`, `sim_params.get`, `candidate_itps.extend`, `_load_definition_lookup`, `_safe_get_param`, `ion_params.get`, `candidate_itps.append`, `ion.get`, `definitions.get`, `ValueError`, `os.path.join`, `missing_ions.add`, +3 more

#### `calculate_water_molecules(mode)` — line 229
- Estimate how many coarse-grained water beads should be inserted.
- kind: function
- returns: `n_water`
- raises: `ValueError`
- effects: Config/runtime state, global registry, stdout
- calls: `Config.get_param`, `add_water_params.get`, `_resolve_gel_weight_fraction_mode`, `get_weighted_average_mass`, `water_masses.get`, `print`, `ValueError`, `_estimate_ion_usage`, `format`, `math.ceil`, `water_masses.keys`, `World.Atoms.values`

### `hygel_martini/hydrogel_builder/cli.py`

Command-line entry point for the hydrogel_builder workflow.

#### `main()` — line 17
- Parse the config location and run the hydrogel builder workflow.
- kind: function, CLI entry
- calls: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `run_hydrogel_builder`, `Path`, `parser.exit`

### `hygel_martini/hydrogel_builder/config_params/__init__.py`

*(no module docstring)*

### `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py`

Backbone planning and materialization utilities for hydrogel generation.

#### `_debug_stage(message)` — line 46
- Emit a stage marker to the optional debug log.
- kind: function, internal
- calls: `Config.debug_log`

#### `_print_build_banner()` — line 54
- Print the standard banner used by backbone-construction stages.
- kind: function, internal
- effects: stdout
- calls: `print`

#### `_seed_random_generators(seed)` — line 61
- Seed Python, NumPy and the serial compiled-geometry RNG for this run.
- kind: function, internal
- returns: `None` (bare return)
- calls: `random.seed`, `np.random.seed`, `seed_numba_random`

#### `_compute_max_linker_span()` — line 79
- Return the largest linker span declared in the configuration.
- kind: function, internal
- returns: `max_span`; `0.0`
- effects: Config/runtime state
- calls: `Config.get_param`, `linker.get`, `definition.get`, `bond.get`, `ext.get`

#### `_gather_sorted_atoms()` — line 106
- Collect the current ``World`` atoms in deterministic atom-id order.
- kind: function, internal
- returns: `(atom_ids, atoms)`
- effects: global registry
- calls: `World.Atoms.keys`, `atoms.append`

#### `_log_min_distance_report(label)` — line 117
- Write a compact minimum-distance report for debugging.
- kind: function, internal
- returns: `None` (bare return)
- effects: filesystem, Config/runtime state, stdout
- calls: `Config.get_param`, `os.makedirs`, `_gather_sorted_atoms`, `np.array`, `find_minimum_distances`, `os.path.join`, `print`, `_debug_stage`, `open`, `log_f.write`

#### `apply_coordinates_from_gro(world, gro_path)` — line 160
- Project coordinates from a GRO file back into the current ``World``.
- kind: function
- returns: `None` (bare return)
- effects: filesystem, stdout
- calls: `_gather_sorted_atoms`, `os.path.exists`, `print`, `np.array`, `open`, `gro_f.readline`, `strip`, `coords.append`

#### `_reset_world_for_backbone(sim_params)` — line 194
- Reset the global world state and initialize box-scale parameters.
- kind: function, internal
- effects: global registry, stdout
- calls: `World.reset`, `Attributes.initialize`, `_seed_random_generators`, `print`, `initialize_world`, `sim_params.get`, `_compute_max_linker_span`

#### `_load_strand_templates(backbone_defs)` — line 207
- Attach whole-strand templates to backbone entries declaring one.
- kind: function, internal
- raises: `ValueError`
- effects: stdout
- calls: `entry.get`, `load_strand_template`, `print`, `ValueError`, `type`

#### `_load_backbone_context()` — line 246
- Load template libraries and sequence strategies for backbone planning.
- kind: function, internal
- returns: `{'backbone_cfg': backbone_cfg, 'backbone_defs': backbone_defs, 'backbone_strategy': backbone_strategy, 'linker_cfg': linker_cfg, 'linker_defs': linker_defs, 'linker_strategy': linker_strategy, 'linker_library': linker_library}`
- effects: Config/runtime state
- calls: `Config.get_param`, `_load_strand_templates`, `backbone_cfg.get`, `linker_cfg.get`, `Config.get_runtime`, `load_monomer_templates`, `Config.set_runtime`, `load_linker_templates`, `linker_definitions_from_library`

#### `_resolve_network_layout(sim_params)` — line 285
- Read the optional ``network_layout`` block.
- kind: function, internal
- returns: `{'net': str(net), 'repeats': repeats, 'cell_parameter': float(cell_parameter), 'max_span': None if max_span is None else float(max_span), 'rewire_seed': rewiring.get('seed'), 'rewire_kwargs': rewire_kwargs, 'conversion_fraction': fraction, 'conversion_count': count, 'conversion_seed': conversion.get('seed')}`; `None`
- raises: `ValueError`
- calls: `sim_params.get`, `raw.get`, `rewiring.get`, `conversion.get`, `ValueError`, `type`

#### `_resolve_isotropy_mode(sim_params)` — line 416
- Resolve whether the special isotropic builder path should be used.
- kind: function, internal
- returns: `bool(isotropy_cfg)`; `not anisotropy`; `anisotropy is None` (+3 more)
- calls: `sim_params.get`, `lower`, `isotropy_cfg.get`, `strip`

#### `_build_blueprint_summary(layout_plan, blueprint)` — line 437
- Print a compact summary of the generated layout and blueprint.
- kind: function, internal
- effects: stdout
- calls: `print`

#### `_plan_backbone_blueprint(sim_params, output_dir)` — line 448
- Build the proto plan and atom blueprint for the hydrogel backbone.
- kind: function, internal
- returns: `{**context, 'proto_plan': proto_plan, 'layout_plan': layout_plan, 'blueprint': blueprint, 'num_cells': num_cells, 'repeats': repeats, 'isotropy_mode': isotropy_mode, 'net_cell': net_cell, 'net_repeats': net_repeats}`
- raises: `ValueError`, `AssertionError`
- effects: stdout
- calls: `_load_backbone_context`, `prepare_proto_plan`, `print`, `_resolve_network_layout`, `_resolve_isotropy_mode`, `_build_blueprint_summary`, `sim_params.get`, `ValueError`, `AssertionError`, `generate_net_layout_plan`, `items`, `build_atom_blueprint`, +5 more

#### `_apply_materialization_box_settings(plan_context)` — line 572
- Copy proto-plan box data into ``World`` before object creation.
- kind: function, internal
- raises: `ValueError`
- effects: stdout
- calls: `plan_context.get`, `np.asarray`, `copy`, `print`, `is_orthorhombic`, `ValueError`, `np.max`, `np.maximum`, `np.array`, `proto_plan.box_vector`, `np.diag`

#### `build_backbone_only()` — line 612
- Construct only the backbone and linker skeleton of the hydrogel.
- kind: function
- returns: `(world, hd)`
- raises: `RuntimeError`
- effects: Config/runtime state, global registry, stdout
- calls: `Config.get_param`, `sim_params.get`, `_print_build_banner`, `_reset_world_for_backbone`, `_apply_materialization_box_settings`, `World`, `world.make_hydrogel`, `print`, `_debug_stage`, `populate_hydrogel_from_blueprint`, `_resolve_close_contacts`, `hd.construct_bonds`, +5 more

#### `finalize_hydrogel(world, hd)` — line 688
- Expand the backbone-only graph into a chemically detailed hydrogel.
- kind: function
- returns: `world`
- effects: stdout
- calls: `print`, `_debug_stage`, `hd.construct_chemical_detail`, `hd.construct_angles`, `hd.construct_dihedrals`, `hd.construct_impropers`, `world.update_hydrogel_attributes`, `_log_min_distance_report`

#### `main()` — line 714
- Run the standalone hydrogel builder entry point.
- kind: function, CLI entry
- returns: `world`
- calls: `build_backbone_only`, `finalize_hydrogel`

### `hygel_martini/hydrogel_builder/config_params/config.py`

Configuration loader with JSON/YAML include support.

#### class `Config` — line 13
Singleton-style access to configuration and runtime metadata.

##### `load_config(cls, file_path)` — line 30
- Load a JSON or YAML maker file into the global config cache.
- kind: classmethod
- returns: `cls._data`
- raises: `FileNotFoundError`, `ValueError`
- effects: filesystem
- calls: `os.path.abspath`, `cls._build_path_context`, `lower`, `cls._normalize_path_tree`, `cls._load_yaml_with_includes`, `os.path.splitext`, `open`, `json.load`, `FileNotFoundError`, `ValueError`

##### `get_param(cls, *keys, file_path=None)` — line 54
- Read a nested value from the loaded configuration tree.
- kind: classmethod
- returns: `current_level`
- raises: `ValueError`, `KeyError`
- calls: `cls.load_config`, `ValueError`, `KeyError`, `join`

##### `set_param(cls, value, *keys)` — line 77
- Write a nested value into the live configuration tree.
- kind: classmethod
- raises: `ValueError`
- calls: `ValueError`, `current_level.setdefault`

##### `set_runtime(cls, key, value)` — line 87
- Store ephemeral runtime state that should not live in the config.
- kind: classmethod

##### `get_runtime(cls, key, default=None)` — line 92
- Read ephemeral runtime state with an optional default.
- kind: classmethod
- returns: `cls._runtime_state.get(key, default)`
- calls: `cls._runtime_state.get`

##### `enable_debug_logging(cls, file_path)` — line 97
- Enable debug logging to a file (overwrite on enable).
- kind: classmethod
- effects: filesystem
- calls: `open`, `f.write`

##### `disable_debug_logging(cls)` — line 109
- Disable file-backed debug logging.
- kind: classmethod

##### `debug_log(cls, message)` — line 115
- Append a debug message with basic timestamp if debug logging is enabled.
- kind: classmethod
- returns: `None` (bare return)
- effects: filesystem
- calls: `strftime`, `open`, `f.write`, `datetime.now`

##### `_deep_merge(cls, base, incoming)` — line 129
- Recursively merge dict incoming into base (mutates base).
- kind: classmethod, internal
- returns: `base`
- calls: `incoming.items`, `cls._deep_merge`, `copy.deepcopy`, `base.get`

##### `_load_yaml_file(cls, path)` — line 139
- Read a single YAML file without processing includes.
- kind: classmethod, internal
- returns: `data if isinstance(data, dict) else {}`
- raises: `ImportError`, `FileNotFoundError`
- effects: filesystem
- calls: `ImportError`, `open`, `FileNotFoundError`, `yaml.safe_load`

##### `_load_yaml_with_includes(cls, path, seen=None)` — line 153
- Load a YAML file and recursively merge its ``includes`` chain.
- kind: classmethod, internal
- returns: `merged`
- raises: `ValueError`
- calls: `os.path.abspath`, `seen.add`, `cls._load_yaml_file`, `os.path.dirname`, `cls._deep_merge`, `ValueError`, `data.pop`, `cls._load_yaml_with_includes`, `os.path.isabs`, `os.path.join`

##### `_build_path_context(cls, file_path)` — line 174
- Return the ``${CONFIG_DIR}``/``${REPO_ROOT}`` substitution values.
- kind: classmethod, internal
- returns: `{'CONFIG_DIR': config_dir, 'REPO_ROOT': repo_root}`
- calls: `os.path.dirname`, `os.path.abspath`, `os.path.join`

##### `_looks_like_path_key(cls, key)` — line 185
- Decide whether a config key's string value should be path-resolved.
- kind: classmethod, internal
- returns: `key in {'gro', 'itp', 'molecule_gro', 'molecule_itp', 'polymer_pdb', 'water_pdb', 'packed_pdb', 'polymer_stage_pdb', 'audit_json'} or key.endswith(cls._PATH_SUFFIXES)`; `False`; `True`
- calls: `key.endswith`

##### `_resolve_path_value(cls, value, path_context)` — line 208
- Expand env vars, ``~``, and ``${CONFIG_DIR}``/``${REPO_ROOT}`` tokens.
- kind: classmethod, internal
- returns: `expanded`
- calls: `os.path.expanduser`, `path_context.items`, `os.path.expandvars`, `expanded.replace`, `os.path.isabs`, `os.path.abspath`, `os.path.join`

##### `_should_resolve_scalar_path(cls, key, value)` — line 222
- Return True when a path-like key's value should actually be resolved.
- kind: classmethod, internal
- returns: `True`; `value.startswith('.') or value.startswith('~') or '${' in value or ('/' in value) or ('\\' in value)`
- calls: `value.startswith`

##### `_normalize_path_tree(cls, node, path_context, parent_key=None)` — line 240
- Recursively rebuild the config tree with path values resolved.
- kind: classmethod, internal
- returns: `node`; `{key: cls._normalize_path_tree(value, path_context, key) for key, value in node.items()}`; `[cls._normalize_path_tree(item, path_context) for item in node]` (+2 more)
- calls: `cls._looks_like_path_key`, `cls._should_resolve_scalar_path`, `cls._resolve_path_value`, `cls._normalize_path_tree`, `node.items`

### `hygel_martini/hydrogel_builder/config_params/generator.py`

Top-level entry helpers for hydrogel construction runs.

#### `run_hydrogel_example(config_path)` — line 14
- Run a full hydrogel-generation job from a maker file.
- kind: function
- effects: Config/runtime state, stdout
- calls: `print`, `Config.load_config`, `run_dos2unix_on_inputs`, `execute_mode`, `os.path.basename`

### `hygel_martini/hydrogel_builder/config_params/make_polymer_only.py`

Batch driver for the standalone-polymer ("polymer only") build mode.

#### `generate_polymer_only_from_config(sim_params, poly_gen_params, polymer_config=None)` — line 12
- Generate one or more standalone polymer chains from config parameters.
- kind: function
- returns: `(generated_gro_paths, generated_itp_paths)`
- effects: filesystem, stdout
- calls: `print`, `os.makedirs`, `Polymer.configure`, `polymer_generator.generate_single_polymer_gro`, `generated_gro_paths.append`, `generated_itp_paths.append`, `os.path.splitext`, `os.path.join`

### `hygel_martini/hydrogel_builder/config_params/read_json.py`

Top-level workflow orchestration for hydrogel generation.

#### class `ProgressTracker` — line 64
Emit coarse percent-based progress updates into the debug log.

##### `__init__(self, total=100.0, run_id=None)` — line 73
- *(no docstring)*
- kind: method

##### `_emit(self, label=None)` — line 82
- Log every whole-percent step crossed since the last emission.
- kind: method, internal
- calls: `Config.debug_log`

##### `advance(self, delta, label=None)` — line 94
- Move the bar forward by ``delta`` percent (clamped to total).
- kind: method
- calls: `self._emit`

##### `start_stage(self, label, weight)` — line 99
- Open a named stage worth ``weight`` percent of the whole run.
- kind: method
- calls: `self._emit`

##### `stage_tick(self, fraction, label=None)` — line 106
- Set progress within the open stage to ``fraction`` (never backwards).
- kind: method
- returns: `None` (bare return)
- calls: `self._emit`

##### `end_stage(self, label=None)` — line 116
- Close the open stage, snapping progress to its full weight.
- kind: method
- returns: `None` (bare return)
- calls: `self._emit`

#### `_seed_all(sim_params)` — line 44
- Seed all RNGs used by the orchestration layer.
- kind: function, internal
- returns: `None` (bare return)
- calls: `sim_params.get`, `random.seed`, `np.random.seed`, `Config.debug_log`

#### `_coerce_bool(value, default=False)` — line 143
- Lenient bool from YAML-ish input ('true'/'1'/'yes'... else default).
- kind: function, internal
- returns: `default`; `value`; `bool(value)` (+2 more)
- calls: `lower`, `value.strip`

#### `resolve_block_copolymer_settings(sim_params)` — line 160
- Resolve generic block-copolymer controls with legacy alias support.
- kind: function
- returns: `{'respect_target_backbone': bool(respect_target_backbone), 'linker_terminal_compensation': terminal_comp, 'terminal_compensation_enabled': bool(terminal_enabled), 'chain_orientation_policy': chain_orientation_policy, 'legacy_policy': legacy_policy, 'warnings': warnings}`
- calls: `strip`, `_coerce_bool`, `sim_params.get`, `replace`, `legacy_policy.lower`, `warnings.append`, `terminal_comp.get`, `lower`

#### `_get_bonded_topology_patch_path(sim_params=None)` — line 239
- Return the configured bonded-topology patch YAML path.
- kind: function, internal
- returns: `os.path.join(os.path.dirname(Config._file_path), 'config', 'backbone.yaml')`; `patch_path`; `None`
- effects: Config/runtime state
- calls: `os.path.join`, `Config.get_param`, `sim_params.get`, `os.path.dirname`

#### `_load_base_parameters()` — line 261
- Loads base parameters like atom masses from the main ITP file. Also prepares a deduplicated list of ITP files for the topology.
- kind: function, internal
- raises: `FileNotFoundError`, `ValueError`
- effects: Config/runtime state, stdout
- calls: `print`, `Config.get_param`, `sim_params.get`, `read_atom_types`, `Config.set_runtime`, `final_itp_list.append`, `FileNotFoundError`, `ValueError`, `os.path.abspath`, `os.path.isdir`, `glob.glob`, `os.path.isfile`, +1 more

#### `_validate_config()` — line 307
- Validate the merged configuration before anything is built.
- kind: function, internal
- returns: `True`; `False`
- raises: `ValueError`
- effects: Config/runtime state, stdout
- calls: `print`, `Config.get_param`, `_load_strand_templates`, `ValueError`, `load_monomer_templates`, `Config.set_runtime`, `load_linker_templates`

#### `execute_mode()` — line 360
- Dispatch the configured top-level execution mode.
- kind: function
- returns: `None` (bare return)
- effects: filesystem, Config/runtime state, stdout
- calls: `print`, `strftime`, `Config.set_runtime`, `Config.debug_log`, `ProgressTracker`, `progress.advance`, `Config.get_param`, `sim_params_for_debug.get`, `_execute_pack_polymer_then_water_mode`, `_load_base_parameters`, `_validate_config`, `sys.exit`, +6 more

#### `_run_packing_step(step_name, base_structure_gro, molecules_to_add, final_output_gro, sim_params)` — line 413
- Run one Packmol stage and return the resulting GRO path.
- kind: function, internal
- returns: `(result_gro, success)`
- effects: stdout
- calls: `print`, `packer.pack_system_with_molecules`, `sim_params.get`

#### `_get_optional_config_section(*names)` — line 433
- First existing Config section among ``names``, else an empty dict.
- kind: function, internal
- returns: `{}`; `Config.get_param(name)`
- effects: Config/runtime state
- calls: `Config.get_param`

#### `_as_box_lengths_nm(job)` — line 443
- Normalize a two-stage-packmol job's box spec to [x, y, z] in nm.
- kind: function, internal
- returns: `[float(value)] * 3`; `[float(value[0])] * 3`; `[float(value[0]), float(value[1]), float(value[2])]`
- raises: `ValueError`
- calls: `ValueError`

#### `_has_packmol_route_md_outputs(output_dir)` — line 470
- Whether a previous packmol-route run left EM/NVT/NPT outputs here.
- kind: function, internal
- returns: `any((os.path.exists(os.path.join(output_dir, name)) for name in md_names))`
- calls: `os.path.exists`, `os.path.join`

#### `_merge_two_stage_job_defaults(defaults, job)` — line 490
- Job dict = shared two-stage defaults (minus 'jobs') overlaid by the job.
- kind: function, internal
- returns: `merged`
- calls: `merged.update`, `defaults.items`

#### `_itp_moleculetypes(path)` — line 501
- Names declared by ``[ moleculetype ]`` blocks in an ITP, or an empty set.
- kind: function, internal
- returns: `names`; `set()`
- effects: filesystem
- calls: `open`, `strip`, `stripped.startswith`, `names.add`, `line.split`, `stripped.split`

#### `_file_digest(path)` — line 526
- sha256 of a file's bytes, or None if unreadable.
- kind: function, internal
- returns: `hashlib.sha256(handle.read()).hexdigest()`; `None`
- effects: filesystem
- calls: `open`, `hexdigest`, `hashlib.sha256`, `handle.read`

#### `_admit_added_itp(itp_dest, itp_files_to_include)` — line 535
- Decide whether an ``add_molecule`` ITP joins the include list.
- kind: function, internal
- returns: `True`; `False`
- raises: `DuplicateDeclaration`
- effects: stdout
- calls: `_itp_moleculetypes`, `_file_digest`, `itp_files_to_include.append`, `DuplicateDeclaration`, `os.path.abspath`, `print`, `os.path.basename`

#### `_normalize_add_molecule_specs(add_series_params, sim_params)` — line 573
- Normalize ``add_series_parameters.add_molecule`` to a list of specs.
- kind: function, internal
- returns: `specs`; `[]`
- raises: `ValueError`
- effects: stdout
- calls: `add_series_params.get`, `entry.get`, `specs.append`, `ValueError`, `os.path.exists`, `sim_params.get`, `print`, `os.path.splitext`, `os.path.basename`, `type`

#### `_execute_pack_polymer_then_water_mode()` — line 638
- YAML-accessible route for polymer-first, fixed-polymer water packing.
- kind: function, internal
- raises: `KeyError`, `ValueError`, `RuntimeError`, `FileNotFoundError`
- effects: filesystem, Config/runtime state, stdout
- calls: `print`, `_get_optional_config_section`, `cfg.get`, `KeyError`, `Config.get_param`, `ValueError`, `sim_params.get`, `_merge_two_stage_job_defaults`, `os.makedirs`, `_as_box_lengths_nm`, `packer.pack_polymer_then_water_two_stage`, `results.append`, +7 more

#### `_compute_total_charge(itp_files_list, molecule_counts_dict)` — line 740
- Estimate the system charge from ITP definitions and molecule counts.
- kind: function, internal
- returns: `total_charge if found else None`; `None`
- raises: `DuplicateDeclaration`
- effects: Config/runtime state, stdout
- calls: `Config.get_runtime`, `molecule_counts_dict.items`, `definitions.update`, `definitions.get`, `read_itp_definitions`, `DuplicateDeclaration`, `print`, `bead.get`, `definition.get`

#### `_make_soft_bonds_itp(src_itp: str, soft_fc: float, dst_itp: str)` — line 783
- Write src_itp with ALL bond force constants replaced by soft_fc.
- kind: function, internal
- effects: filesystem
- calls: `open`, `fh.writelines`, `line.strip`, `stripped.startswith`, `out_lines.append`, `lower`, `stripped.split`, `strip`, `join`, `split`, `stripped.strip`

#### `_perform_geo_opt_step(step_name, base_gro_file, output_dir, itp_files_list, molecule_counts_dict, sim_params)` — line 814
- Run one GROMACS energy-minimization stage.
- kind: function, internal
- returns: `optimized_gro if optimized_gro else base_gro_file`; `base_gro_file`
- effects: filesystem, Config/runtime state, stdout
- calls: `print`, `sim_params.get`, `geo_opt_cfg.get`, `os.path.join`, `os.makedirs`, `topology_updater.create_system_topology`, `topology_updater.update_topology_molecules`, `_compute_total_charge`, `mdp_overrides.get`, `run_geo_opt`, `Config.get_param`, `final_itp_list.append`, +3 more

#### `_merge_world_and_itps(world, extra_itps, merged_itp_path, moleculetype_name='MERGED')` — line 921
- World 기반 구조와 추가 ITP 파일을 단일 ITP로 병합합니다. - World는 write_combined_itp로 기록 - 외부 ITP는 moleculetype 단위로 독립적이므로 인덱스 재배치 없이 그대로 이어붙임 (하나의 파일에 여러 moleculetype을 담는 목적)
- kind: function, internal
- returns: `merged_itp_path`
- effects: filesystem
- calls: `write_combined_itp`, `open`, `fout.write`, `os.remove`, `format`, `os.path.isfile`, `rstrip`, `f.read`

#### `_perform_dynamic_crosslinking(output_dir)` — line 950
- Connect linker stubs to true backbone ends rather than arbitrary beads.
- kind: function, internal
- returns: `default_params`; `merged`
- raises: `RuntimeError`
- effects: filesystem, Config/runtime state, global registry, stdout
- calls: `print`, `os.path.join`, `open`, `group_linker_stubs`, `collect_backbone_ends`, `debug_f.write`, `sim_params.get`, `lower`, `resolve_block_copolymer_settings`, `plan_dynamic_crosslinks`, `Config.get_param`, `format`, +21 more

#### `_get_hydrogel_topology_connectivity_audit_config()` — line 1205
- Return the post-build hydrogel topology audit config.
- kind: function, internal
- returns: `Config.get_param('hydrogel_topology_connectivity_audit')`; `legacy_cfg`; `None`
- effects: Config/runtime state, stdout
- calls: `Config.get_param`, `print`

#### `_audit_and_guard_connectivity(gro_path, itp_path, output_dir)` — line 1228
- Audit the generated hydrogel bonded topology and apply the optional guard.
- kind: function, internal
- returns: `x`
- raises: `RuntimeError`
- effects: filesystem, stdout
- calls: `print`, `_get_hydrogel_topology_connectivity_audit_config`, `UnionFind`, `defaultdict`, `os.path.join`, `get`, `os.path.exists`, `_handle_audit_error`, `append`, `components.values`, `audit_cfg.get`, `RuntimeError`, +19 more

#### `_execute_all_mode()` — line 1385
- Executes the full workflow with sequential packing and genion.
- kind: function, internal
- raises: `ValueError`
- effects: filesystem, Config/runtime state, stdout
- calls: `print`, `Config.get_runtime`, `Config.get_param`, `_seed_all`, `os.makedirs`, `build_hydrogel.build_backbone_only`, `_perform_dynamic_crosslinking`, `_get_bonded_topology_patch_path`, `os.path.join`, `write_to_gro`, `write_combined_itp`, `_perform_geo_opt_step`, +66 more

### `hygel_martini/hydrogel_builder/core_utils/__init__.py`

Compatibility layer for hydrogel builder helpers.

### `hygel_martini/hydrogel_builder/core_utils/common/__init__.py`

Shared math, selection, and small helper utilities for hydrogel builds.

### `hygel_martini/hydrogel_builder/core_utils/common/collisions.py`

Detect declarations that silently overwrite one another.

#### class `DuplicateDeclaration`(ValueError) — line 39
Two declarations claimed the same key.

#### `_format(value: Any, limit: int=120)` — line 43
- repr() truncated to ``limit`` characters for error messages.
- kind: function, internal
- returns: `text if len(text) <= limit else text[:limit - 3] + '...'`

#### `find_duplicates(keys: Iterable[Hashable])` — line 49
- Keys appearing more than once, with their counts.
- kind: function
- returns: `{key: count for key, count in counts.items() if count > 1}`
- calls: `counts.get`, `counts.items`

#### `require_unique(items: Iterable[Tuple[Hashable, Any]], what: str, key_name: str='identifier', source: str | None=None)` — line 57
- Build a lookup, refusing any key declared twice.
- kind: function
- returns: `lookup`
- raises: `DuplicateDeclaration`
- calls: `join`, `DuplicateDeclaration`, `duplicates.append`, `_format`

#### `require_consistent(items: Iterable[Tuple[Hashable, Any]], what: str, key_name: str='identifier', source: str | None=None, equal: Callable[[Any, Any], bool] | None=None)` — line 91
- Build a lookup, allowing repeats only when the values agree.
- kind: function
- returns: `lookup`
- raises: `DuplicateDeclaration`
- calls: `join`, `DuplicateDeclaration`, `same`, `conflicts.append`, `_format`

### `hygel_martini/hydrogel_builder/core_utils/common/sequence_strategy.py`

Shared helpers for applying random/alternating/block strategies to template selections.

#### class `StrategyRecord` — line 20
One selectable template plus its selection weight.

#### class `TemplateStrategyIterator` — line 37
Iterates over template records following the requested strategy. Supported strategies: random, alternating, block. Defaults to random.

##### `__init__(self, records: List[StrategyRecord], strategy_cfg: Optional[dict]=None)` — line 43
- Prepare the iterator state for the requested strategy.
- kind: method
- calls: `lower`, `random.Random`, `self._prepare_sequences`, `self.strategy_cfg.get`

##### `_prepare_sequences(self)` — line 64
- Precompute the repeating sequences used by non-random strategies.
- kind: method, internal
- returns: `None` (bare return)
- calls: `self.strategy_cfg.get`, `lookup.get`, `sequence.extend`, `block.get`

##### `next(self)` — line 99
- Return the next template according to the configured strategy.
- kind: method
- returns: `self.random_state.choices([rec.template for rec in self.records], weights=weights, k=1)[0]`; `None`; `self.records[0].template` (+1 more)
- calls: `self.random_state.choices`

### `hygel_martini/hydrogel_builder/core_utils/common/utility.py`

Numerical helpers, geometry utilities, and text-normalization helpers.

#### `interp3D(n, A, B)` — line 26
- 두 3D 점 A와 B 사이에 n개의 점을 등간격으로 보간하여 생성합니다. 반환되는 점들은 A와 B 사이의 선분을 n+1개의 구간으로 나눈 점들입니다.
- kind: function
- returns: `np.array([A + i * (B - A) / (n + 1) for i in range(1, n + 1)])`
- calls: `np.array`

#### `rij(position_i, position_j, L)` — line 43
- 주기 경계 조건(Periodic Boundary Conditions, PBC)을 고려하여 원자 i에서 원자 j로 향하는 벡터(r_ij)를 계산합니다. 가장 가까운 이미지(minimum image convention)를 사용합니다.
- kind: function
- returns: `r_ij`
- calls: `numba.jit`, `np.zeros`, `np.round`

#### `dij_sq(position_i, position_j, L)` — line 70
- PBC를 고려하여 두 원자 i와 j 사이의 거리의 제곱(d_ij^2)을 계산합니다. 제곱근 계산을 피하여 연산 속도를 높입니다.
- kind: function
- returns: `d_ij_sq`
- calls: `numba.jit`, `np.round`

#### `normal_to_3vectors(position_i, position_j, position_k, L)` — line 92
- 세 점 i, j, k가 이루는 평면의 법선 벡터를 계산합니다.
- kind: function
- returns: `normal_cross`
- calls: `numba.jit`, `rij`, `np.cross`, `np.sqrt`, `np.sum`, `np.square`

#### `normal_tetrahedral_vector(position_1, position_2, position_3, position_4, L)` — line 119
- 중심 원자(1)와 세 개의 이웃 원자(2, 3, 4)가 주어졌을 때, 사면체(tetrahedral) 구조에서 네 번째 결합이 향해야 할 방향 벡터를 계산합니다.
- kind: function
- returns: `r_tetra`
- calls: `numba.jit`, `rij`, `np.sqrt`, `np.sum`, `np.square`

#### `not_self(i, obj)` — line 145
- 결합(bond) 객체(obj)와 그 결합에 속한 원자(i) 하나를 입력받아, 그 결합에 속한 다른 원자를 반환하는 헬퍼 함수입니다.
- kind: function
- returns: `obj.bond_atom_2`; `obj.bond_atom_1`

#### `is_overlap(A, B, d, L)` — line 164
- 점 A가 점들의 배열 B에 있는 어떤 점과 거리 d 미만으로 겹치는지 확인합니다.
- kind: function
- returns: `True`; `False`
- calls: `numba.jit`, `dij_sq`

#### `seed_numba_random(seed)` — line 195
- Seed the calling thread's Numba RNG without consuming Python/NumPy draws.
- kind: function
- calls: `numba.njit`, `np.random.seed`

#### `random_normal_vector(A, B, C, r, L)` — line 211
- A-B-C로 연결된 구조에서 중심 원자 B에 대해, 두 결합(A-B, C-B)이 이루는 평면에 거의 수직인 방향으로 길이가 r인 무작위 벡터를 생성합니다. 곁사슬(side chain)을 생성할 때 사용됩니다.
- kind: function
- returns: `np.array([x1, y1, z1])`
- calls: `numba.jit`, `rij`, `np.linalg.norm`, `np.sqrt`, `np.array`, `np.random.random`

#### `find_minimum_distances(positions, box_length, top_n=10, cell_size=None)` — line 252
- Return the smallest inter-particle distances using a simple cell list search.
- kind: function
- returns: `results`; `[]`
- effects: Config/runtime state
- calls: `np.asarray`, `np.mod`, `astype`, `np.clip`, `defaultdict`, `cells.items`, `results.sort`, `append`, `dij_sq`, `Config.get_runtime`, `heapq.heappop`, `results.append`, +11 more

#### `run_dos2unix_on_inputs(config_data)` — line 356
- Normalize line endings for all configured input structure files.
- kind: function
- effects: subprocess, filesystem, stdout
- calls: `get`, `files_to_process.extend`, `shutil.which`, `config_data.get`, `replace`, `os.path.exists`, `print`, `open`, `src.read`, `files_to_process.append`, `subprocess.run`, `original.replace`, +2 more

### `hygel_martini/hydrogel_builder/core_utils/generators/__init__.py`

Standalone structure generators that build reusable intermediate systems.

### `hygel_martini/hydrogel_builder/core_utils/generators/polymer_generator.py`

Standalone single-polymer GRO/ITP generation.

#### `generate_single_polymer_gro(p_mon_num: int, output_filename: str, mean_sep: float, random_seed: int=2024, include_chemical_detail: bool=True, include_angles: bool=True, moleculetype_name: str='HDGEL', polymer_config: dict | None=None)` — line 24
- 단일 고분자 사슬의 .gro 파일을 생성합니다.
- kind: function
- effects: global registry, stdout
- calls: `print`, `World.reset`, `initialize_world`, `World`, `Attributes.initialize`, `world.make_polymer`, `pm.construct_atoms`, `writer.write_to_gro`, `writer.write_combined_itp`, `Polymer.configure`, `pm.construct_chemical_detail`, `pm.construct_angles`, +1 more

### `hygel_martini/hydrogel_builder/core_utils/io/__init__.py`

Low-level readers and writers for GRO, ITP, and topology text files.

### `hygel_martini/hydrogel_builder/core_utils/io/gro_parser.py`

GRO parsing for the builder.

### `hygel_martini/hydrogel_builder/core_utils/io/martini_parser.py`

GROMACS topology parsing for the builder.

### `hygel_martini/hydrogel_builder/core_utils/io/writer.py`

Topology and coordinate writers for World state.

#### `write_to_xyz(object, filename='xyz.xyz')` — line 31
- 시스템의 원자 좌표를 간단한 .xyz 파일 형식으로 저장합니다. 시각화 프로그램에서 구조를 빠르게 확인하는 데 유용합니다.
- kind: function
- effects: filesystem
- calls: `os.path.dirname`, `os.makedirs`, `open`, `f.write`, `format`, `np.random.randint`

#### `write_to_gro(object, filename='gromacs.gro')` — line 54
- 시스템 정보를 GROMACS .gro 파일 형식으로 저장합니다.
- kind: function
- returns: `1`
- effects: filesystem
- calls: `os.path.dirname`, `os.makedirs`, `open`, `f.write`, `np.array`, `format`, `np.any`

#### `write_to_itp(object, filename='gromacs.itp', moleculetype_name='HDGEL')` — line 90
- 시스템의 토폴로지 정보를 GROMACS .itp 파일 형식으로 저장합니다. 이 파일은 분자 내 상호작용(결합, 각도 등)을 정의합니다.
- kind: function
- returns: `1`
- effects: filesystem
- calls: `os.path.dirname`, `os.makedirs`, `open`, `f.write`, `extras.get`, `extras.items`, `vs_by_sec.items`, `format`, `join`, `vs.get`, `append`, `sec.endswith`, +7 more

#### `_bonded_params_text(*values)` — line 271
- Format optional bonded parameters, stopping at the first ``None``.
- kind: function, internal
- returns: `'  ' + ' '.join(parts) if parts else ''`
- calls: `parts.append`, `join`

#### `write_combined_itp(world, filename, moleculetype_name, nrexcl=None, extra_pairs=None)` — line 287
- 월드의 Atoms/Bonds/Angles/Dihedrals/Constraints/Exclusions 및 OtherSections에 저장된 추가 섹션을 모두 포함한 단일 ITP를 작성합니다. 외부 ITP를 건드리지 않고, 월드에서 생성된 분자에 대해서만 사용합니다.
- kind: function
- returns: `1`
- effects: filesystem, Config/runtime state, stdout
- calls: `os.path.dirname`, `print`, `os.makedirs`, `open`, `f.write`, `get`, `format`, `extras.items`, `Config.debug_log`, `validate_and_filter_other_sections`, `extras.get`, `vs_by_sec.items`, +8 more

### `hygel_martini/hydrogel_builder/core_utils/layout/__init__.py`

Proto-planning, layout construction, and blueprint population helpers.

### `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py`

Isotropic medium-cell builder with per-cell EM.

#### `_linear_index(ix: int, iy: int, iz: int, repeats: Tuple[int, int, int])` — line 72
- Flatten a 3D repeat-cell index into a row-major linear index.
- kind: function, internal
- returns: `ix * ny * nz + iy * nz + iz`

#### `_normalize(vec: np.ndarray)` — line 78
- Return ``vec`` scaled to unit length; near-zero vectors pass through.
- kind: function, internal
- returns: `vec / norm`; `vec`
- calls: `np.linalg.norm`

#### `_axis_rotation_matrix(axis: str)` — line 86
- Return the 90-degree rotation that maps the x axis onto ``axis``.
- kind: function, internal
- returns: `np.eye(3, dtype=float)`; `np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)`; `np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)`
- calls: `np.eye`, `np.array`

#### `_apply_rotation(vec: np.ndarray, rot: np.ndarray)` — line 107
- Apply rotation matrix ``rot`` to row vector(s) ``vec``.
- kind: function, internal
- returns: `vec @ rot.T`

#### `_get_anisotropy_axis()` — line 112
- Read ``simulation_parameters.anisotropy`` defensively.
- kind: function, internal
- returns: `axis`
- effects: Config/runtime state
- calls: `lower`, `get`, `Config.get_param`

#### `_normalize_linker_axes(linker_axes: Optional[Sequence[str]])` — line 129
- Coerce a linker-axis spec into exactly two valid axis names.
- kind: function, internal
- returns: `cleaned[:2]`
- calls: `lower`, `cleaned.append`

#### `_squeeze_positions_to_cube(positions: np.ndarray, cube_side: float, ghost_step: float=0.0)` — line 159
- Rescale all positions to fit inside a cube of side cube_side.
- kind: function, internal
- returns: `centered * scale`
- calls: `np.maximum`, `positions.min`, `positions.max`

#### `_linker_total_length(entry: Dict, fallback: float, override: float | None=None)` — line 191
- Resolve the end-to-end length (nm) of a linker definition.
- kind: function, internal
- returns: `float(total)`; `float(override)`
- calls: `entry.get`, `definition.get`, `bond.get`, `ext.get`

#### `_write_linker_debug(path: str, blueprint: LayoutBlueprint, positions_pre: np.ndarray, positions_post: np.ndarray, axes: List[str], medium_origin: np.ndarray, box_vector: np.ndarray, cube_side: float, small_edge: float, linker_len: float, base_size: np.ndarray, cell_vector: np.ndarray, scale: np.ndarray, mins: np.ndarray, maxs: np.ndarray)` — line 218
- Dump one medium cell's linker geometry to a JSON debug file.
- kind: function, internal
- effects: filesystem
- calls: `append`, `medium_origin.tolist`, `base_size.tolist`, `cell_vector.tolist`, `box_vector.tolist`, `scale.tolist`, `chain_atoms.get`, `chain.metadata.get`, `open`, `json.dump`, `tolist`, `mins.tolist`, +15 more

#### `_resolve_close_contacts(atoms: List, threshold: float=0.001, jitter: float=0.05, seed: int=0)` — line 359
- Detect atoms within *threshold* nm of each other and jitter one.
- kind: function, internal
- returns: `jittered`; `0`
- effects: stdout
- calls: `np.array`, `np.random.default_rng`, `append`, `Config.debug_log`, `print`, `tolist`, `copy`, `np.linalg.norm`, `rng.standard_normal`, `grid.setdefault`, `astype`, `grid.get`, +1 more

#### `_pick_boundary_atoms(positions: np.ndarray)` — line 449
- Pick up to four atoms nearest to alternating cube corners.
- kind: function, internal
- returns: `picked`
- calls: `positions.min`, `positions.max`, `np.maximum`, `np.array`, `np.linalg.norm`, `np.argmin`, `picked.append`

#### `_write_posre_itp(path: str, atom_indices: List[int], fc: float)` — line 482
- Write a GROMACS position-restraint ITP for the given atoms.
- kind: function, internal
- effects: filesystem
- calls: `open`, `f.write`

#### `_write_system_top(path: str, itp_path: str, posre_path: str, base_itp: str | None)` — line 497
- Write a minimal .top for one medium cell's EM run.
- kind: function, internal
- effects: filesystem
- calls: `open`, `f.write`, `os.path.abspath`

#### `_build_world_from_blueprint(blueprint: LayoutBlueprint, box_vector: np.ndarray, output_dir: str, mean_sep: float, construct_proto_bonds: bool=True)` — line 524
- Materialize a blueprint into a fresh global ``World``.
- kind: function, internal
- returns: `(world, hd)`
- effects: global registry, stdout
- calls: `World.reset`, `Attributes.initialize`, `World`, `world.make_hydrogel`, `populate_hydrogel_from_blueprint`, `np.array`, `hd.construct_bonds`, `np.max`

#### `_run_medium_cell_em(blueprint: LayoutBlueprint, box_vector: np.ndarray, fixed_atom_indices: List[int], out_dir: str, sim_params: Dict, temp_bonds: List[Tuple[int, int, Dict]] | None=None)` — line 564
- Energy-minimize one medium cell and return the relaxed positions.
- kind: function, internal
- returns: `[atom.position for atom in atoms]`
- raises: `FileNotFoundError`
- effects: filesystem, global registry, stdout
- calls: `os.makedirs`, `_build_world_from_blueprint`, `os.path.join`, `write_to_gro`, `write_combined_itp`, `_write_posre_itp`, `sim_params.get`, `_write_system_top`, `run_geo_opt`, `read_gro_atoms`, `World.Bonds.items`, `get`, +8 more

#### `pbc_diff(pos1, pos2, box)` — line 685
- Minimum-image displacement ``pos1 - pos2`` in an orthorhombic box.
- kind: function
- returns: `minimum_image(pos1 - pos2, box)`
- calls: `minimum_image`

#### `_optimize_linker_axes(repeats: Tuple[int, int, int], small_edge: float, cell_vector: np.ndarray, base_size: np.ndarray, seed: int | None=None)` — line 690
- Plan connectivity-aware, xyz-balanced linker axes for every cell.
- kind: function, internal
- returns: `{k: tuple(v) for k, v in cell_plans.items()}`
- raises: `ValueError`
- calls: `vertex_endpoints.items`, `plan_balanced_cycle_matchings`, `np.array`, `vertices.append`, `ValueError`, `LocalVertex`, `cell_plans.items`, `chain_edges.append`, `linkers.items`, `pbc_diff`, `np.linalg.norm`, `np.sign`, +1 more

#### `_offset_blueprint(blueprint: LayoutBlueprint, atom_offset: int, chain_offset: int)` — line 843
- Re-base a cell-local blueprint's atom and chain indices.
- kind: function, internal
- returns: `LayoutBlueprint(atoms=atoms, chains=chains)`
- effects: stdout
- calls: `LayoutBlueprint`, `atoms.append`, `chains.append`, `replace`

#### `build_isotropic_blueprint(proto_plan, backbone_defs: List[Dict], linker_defs: List[Dict], repeats: Tuple[int, int, int], backbone_strategy: Dict, linker_strategy: Dict, linker_library, output_dir: str, sim_params: Dict)` — line 872
- Build the full isotropic-mode coordinate blueprint.
- kind: function
- returns: `LayoutBlueprint(atoms=all_atoms, chains=all_chains)`; `minimum_image(vec, total_box)`; `res_name in target_names` (+3 more)
- effects: filesystem, stdout
- calls: `np.array`, `require_unique`, `sim_params.get`, `_resolve_close_contacts`, `LayoutBlueprint`, `np.sqrt`, `_optimize_linker_axes`, `minimum_image`, `os.path.join`, `os.makedirs`, `_build_world_from_blueprint`, `write_to_gro`, +47 more

### `hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py`

Materialize a LayoutPlan into concrete coordinates and atom blueprints.

#### class `InstantiatedChain` — line 33
One placed chain: absolute positions (nm) plus its definition/metadata.

#### class `InstantiatedLayout` — line 47
All placed chains, split by kind, in layout order.

#### class `AtomBlueprint` — line 55
Everything the populator needs to create one Atom.

#### class `ChainBlueprint` — line 82
Per-chain record: which blueprint atoms belong to it, plus metadata (planned ids, sequence, strand template, attachment positions...).

#### class `LayoutBlueprint` — line 95
The flat handoff consumed by ``proto_populator``.

#### `_center_positions(positions: np.ndarray)` — line 102
- Positions translated so their centroid sits at the origin.
- kind: function, internal
- returns: `positions - centroid`
- calls: `np.mean`

#### `_rotate_between_vectors(vectors: np.ndarray, source: np.ndarray, target: np.ndarray)` — line 108
- Rotate row vectors by the rotation carrying ``source`` onto ``target``.
- kind: function, internal
- returns: `vectors @ R.T`; `vectors`; `-vectors`
- calls: `np.linalg.norm`, `np.allclose`, `np.cross`, `np.arccos`, `np.array`, `np.clip`, `np.dot`, `np.eye`, `np.sin`, `np.cos`

#### `_rotate_from_xaxis(vectors: np.ndarray, target: np.ndarray)` — line 140
- Rotate row vectors from the +x axis onto ``target`` (Rodrigues).
- kind: function, internal
- returns: `vectors @ R.T`; `vectors`; `np.column_stack((-vectors[:, 0], vectors[:, 1], vectors[:, 2]))`
- calls: `np.linalg.norm`, `np.array`, `np.allclose`, `np.cross`, `np.arccos`, `np.column_stack`, `np.clip`, `np.dot`, `np.eye`, `np.sin`, `np.cos`

#### `_alignment_basis(axis: np.ndarray)` — line 168
- Right-handed orthonormal basis (columns) whose x axis is ``axis``.
- kind: function, internal
- returns: `np.column_stack((x_axis, y_axis, z_axis))`
- calls: `np.linalg.norm`, `np.array`, `np.cross`, `np.column_stack`, `np.dot`

#### `_axis_rotation(axis: np.ndarray, angle: float)` — line 203
- Rodrigues rotation matrix about a unit ``axis`` by ``angle`` radians.
- kind: function, internal
- returns: `np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)`
- calls: `np.array`, `np.eye`, `np.sin`, `np.cos`

#### `_rotation_between(source: np.ndarray, target: np.ndarray)` — line 210
- Rotation matrix carrying unit vector ``source`` onto unit ``target``.
- kind: function, internal
- returns: `_axis_rotation(cross / norm, np.arctan2(norm, dot))`; `_axis_rotation(perp, np.pi)`; `np.eye(3)`
- calls: `np.cross`, `_axis_rotation`, `np.linalg.norm`, `np.dot`, `np.array`, `np.arctan2`, `np.eye`

#### `instantiate_backbone(cell: LayoutCell, proto_positions: np.ndarray)` — line 234
- Place one strand: rigid whole-strand template, or scaled bead chain.
- kind: function
- returns: `InstantiatedChain(positions=positions, definition=cell.backbone_definition, metadata=metadata)`; `InstantiatedChain(positions=positions, definition=cell.backbone_definition, metadata=metadata, template=strand_template)`
- calls: `_center_positions`, `_rotate_between_vectors`, `InstantiatedChain`, `cell.metadata.get`, `_rotation_between`, `metadata.update`, `np.asarray`, `_axis_rotation`, `cell.metadata.items`

#### `instantiate_linker(layout_plan: LayoutPlan, link: LinkPlacement, proto_positions: np.ndarray)` — line 292
- Place one junction molecule's *body* beads (stubs are emitted later).
- kind: function
- returns: `InstantiatedChain(positions=positions, definition=definition, metadata=embed_metadata, template=template)`
- effects: stdout
- calls: `definition.get`, `defn_body.get`, `metadata.get`, `np.array`, `np.linalg.norm`, `embed_metadata.update`, `InstantiatedChain`, `link.metadata.copy`, `hasattr`, `library.lookup.get`, `print`, `_alignment_basis`, +4 more

#### `instantiate_layout(layout_plan: LayoutPlan)` — line 372
- Place every cell and link of the plan, in plan order.
- kind: function
- returns: `InstantiatedLayout(backbone_segments=backbone_segments, linker_segments=linker_segments)`
- calls: `InstantiatedLayout`, `np.zeros`, `backbone_segments.append`, `linker_segments.append`, `instantiate_backbone`, `instantiate_linker`

#### `_backbone_atom_params(component_entry: Dict[str, Any], bead_index: int)` — line 397
- Per-bead identity for a Martini backbone bead from its definition.
- kind: function, internal
- returns: `{'atom_name': atom_name, 'atom_type': atom_type, 'residue_name': residue_name, 'residue_number': residue_number, 'charge_group_number': cgnr, 'mass': mass, 'charge': charge}`
- calls: `component_entry.get`, `definition.get`

#### `_linker_atom_params(component_entry: Dict[str, Any], bead_index: int)` — line 431
- Per-bead identity for a linker body bead (stubs are emitted apart).
- kind: function, internal
- returns: `{'atom_name': atom_name, 'atom_type': atom_type, 'residue_name': residue_name, 'residue_number': residue_number, 'charge_group_number': cgnr, 'mass': mass, 'charge': charge}`
- calls: `component_entry.get`, `definition.get`, `bead_def.get`

#### `build_atom_blueprint(layout_plan: LayoutPlan, backbone_defs: List[Dict[str, Any]])` — line 464
- Flatten the instantiated layout into per-atom blueprints.
- kind: function
- returns: `LayoutBlueprint(atoms=atoms, chains=chains)`; `None`; `targets` (+1 more)
- effects: stdout
- calls: `instantiate_layout`, `LayoutBlueprint`, `component_entry.get`, `chains.append`, `np.array`, `np.linalg.norm`, `chain.metadata.get`, `_backbone_atom_params`, `get`, `raw_def.get`, `atoms.append`, `atom_indices.append`, +20 more

### `hygel_martini/hydrogel_builder/core_utils/layout/local_matching.py`

Local BCK matching planner for tetrahedral diamond vertices.

#### class `LocalVertex` — line 111
The polymer endpoints around one local crosslink vertex.

##### `functionality(self)` — line 125
- How many strand endpoints meet at this vertex (its degree f).
- kind: property
- returns: `len(self.endpoints)`

##### `is_tetrahedral(self)` — line 130
- True when this vertex uses the diamond local-coordinate labels.
- kind: property
- returns: `set(self.endpoints) == set(LOCAL_COORDS)`

##### `ordered_keys(self)` — line 134
- Endpoint labels in the fixed order the matching states index into.
- kind: method
- returns: `LOCAL_COORDS`; `tuple(sorted(self.endpoints))`; `tuple(sorted(self.endpoints, key=repr))`

##### `validate(self)` — line 143
- Refuse a vertex whose endpoint count admits no perfect matching.
- kind: method
- returns: `None` (bare return)
- raises: `ValueError`
- calls: `ValueError`, `self.endpoints.values`

#### class `VertexAxisChoice` — line 168
Chosen x/y/z matching state for one local vertex.

#### class `MatchingDiagnostics` — line 181
Connectivity and balance report for a local matching plan.

#### class `MatchingPlan` — line 196
Result of choosing local x/y/z states for a set of vertices.

##### `is_single_cycle(self)` — line 203
- True when the transition system closes into ONE Eulerian circuit.
- kind: property
- returns: `self.diagnostics.component_count == 1 and (not self.diagnostics.degree_violations)`

#### class `_UnionFind` — line 215
Path-compressing union-find over endpoint ids, for circuit counting.

##### `__init__(self, nodes: Iterable[EndpointId])` — line 218
- *(no docstring)*
- kind: method

##### `find(self, node: EndpointId)` — line 222
- Root of ``node``'s set, compressing the path on the way up.
- kind: method
- returns: `self.parent[node]`
- calls: `self.parent.setdefault`, `self.size.setdefault`, `self.find`

##### `union(self, first: EndpointId, second: EndpointId)` — line 230
- Merge the two sets, smaller root under the larger.
- kind: method
- returns: `None` (bare return)
- calls: `self.find`

#### class `_Traversal` — line 383
One directed pass of the Hierholzer walk across a strand edge.

#### `perfect_matchings(size: int)` — line 49
- All perfect matchings of ``range(size)``; there are ``(size-1)!!``.
- kind: function
- returns: `tuple(build(tuple(range(size))))`; `((),)`; `<generator>`
- raises: `ValueError`
- calls: `lru_cache`, `ValueError`, `build`

#### `matching_state_count(functionality: int)` — line 78
- ``(f-1)!!`` --- 3 states at f=4, 15 at f=6, 105 at f=8.
- kind: function
- returns: `len(perfect_matchings(functionality))`
- calls: `perfect_matchings`

#### `_normalize_state(state, functionality: int)` — line 88
- Accept either a general state index or a tetrafunctional axis label.
- kind: function, internal
- returns: `index`; `STATE_BY_AXIS[axis]`
- raises: `ValueError`
- calls: `matching_state_count`, `state.lower`, `ValueError`

#### `matching_edges_for_state(vertex: LocalVertex, state)` — line 242
- The ``f/2`` junction edges created by one matching state at one vertex.
- kind: function
- returns: `tuple(((vertex.endpoints[keys[left]], vertex.endpoints[keys[right]]) for left, right in matching))`
- calls: `vertex.validate`, `vertex.ordered_keys`, `_normalize_state`, `perfect_matchings`

#### `matching_edges_for_axis(vertex: LocalVertex, axis: str)` — line 259
- Tetrafunctional wrapper over :func:`matching_edges_for_state`.
- kind: function
- returns: `matching_edges_for_state(vertex, axis)`
- raises: `ValueError`
- calls: `lower`, `matching_edges_for_state`, `ValueError`

#### `_all_nodes(vertices: Sequence[LocalVertex], chain_edges: Sequence[Edge])` — line 272
- Every endpoint id, deduplicated, with each vertex validated first.
- kind: function, internal
- returns: `nodes`
- calls: `vertex.validate`, `vertex.endpoints.values`, `nodes.append`, `seen.add`

#### `evaluate_matching_plan(vertices: Sequence[LocalVertex], chain_edges: Sequence[Edge], axes_by_vertex: Mapping[Hashable, str])` — line 290
- Evaluate a full local matching assignment as a graph of polymer endpoints.
- kind: function
- returns: `MatchingPlan(tuple(choices), diagnostics)`
- calls: `Counter`, `_all_nodes`, `_UnionFind`, `MatchingDiagnostics`, `MatchingPlan`, `vertex.validate`, `_normalize_state`, `matching_edges_for_state`, `choices.append`, `local_edges.extend`, `uf.union`, `VertexAxisChoice`, +5 more

#### `_state_by_pairing(size: int)` — line 352
- Reverse lookup from a slot-index pairing to its matching-state index.
- kind: function, internal
- returns: `{frozenset((frozenset(pair) for pair in matching)): index for index, matching in enumerate(perfect_matchings(size))}`
- calls: `lru_cache`, `frozenset`, `perfect_matchings`

#### `state_for_pairing(vertex: LocalVertex, pairs: Sequence[Edge])` — line 360
- Which matching state realizes ``pairs`` at ``vertex``.
- kind: function
- returns: `lookup[wanted]`
- raises: `ValueError`
- calls: `vertex.validate`, `vertex.ordered_keys`, `_state_by_pairing`, `frozenset`, `ValueError`

#### `_strand_adjacency(vertices: Sequence[LocalVertex], chain_edges: Sequence[Edge])` — line 393
- Index the strand graph by junction, as directed traversals.
- kind: function, internal
- returns: `(adjacency, owner, unattached)`
- raises: `ValueError`
- calls: `defaultdict`, `vertex.validate`, `vertex.endpoints.values`, `append`, `_Traversal`, `ValueError`

#### `plan_single_circuit(vertices: Sequence[LocalVertex], chain_edges: Sequence[Edge])` — line 437
- Construct a transition system with one circuit per connected component.
- kind: function
- returns: `evaluate_matching_plan(vertices, chain_edges, states)`; `evaluate_matching_plan([], chain_edges, {})`
- raises: `ValueError`
- calls: `_strand_adjacency`, `defaultdict`, `evaluate_matching_plan`, `ValueError`, `walk.reverse`, `pairings.get`, `state_for_pairing`, `append`, `walk.append`, `used_edges.add`, `stack.append`, `stack.pop`, +1 more

#### `_balanced_axis_pool(count: int, rng: Random)` — line 532
- A shuffled pool of ``count`` axis labels with x/y/z counts within one.
- kind: function, internal
- returns: `pool`
- calls: `rng.shuffle`, `pool.extend`

#### `_axis_balance_penalty(axis_counts: Mapping[str, int])` — line 549
- Return zero only when x/y/z counts differ by at most one.
- kind: function, internal
- returns: `spread + sum(((3 * count - mean_num) ** 2 for count in counts))`; `0`
- calls: `axis_counts.get`

#### `_is_nearly_balanced(axis_counts: Mapping[str, int])` — line 561
- True when x/y/z state counts differ by at most one.
- kind: function, internal
- returns: `max(counts, default=0) - min(counts, default=0) <= 1`
- calls: `axis_counts.get`

#### `_score(plan: MatchingPlan)` — line 567
- Lexicographic plan quality: fewer circuits, then fewer degree violations, then better axis balance, then a larger main component.
- kind: function, internal
- returns: `(diag.component_count, len(diag.degree_violations), _axis_balance_penalty(diag.axis_counts), -diag.largest_component_size)`
- calls: `_axis_balance_penalty`

#### `_annealing_energy(plan: MatchingPlan)` — line 579
- Scalar score for Metropolis moves; lower is better.
- kind: function, internal
- returns: `1000000.0 * diag.component_count + 10000.0 * len(diag.degree_violations) + 100.0 * _axis_balance_penalty(diag.axis_counts) - float(diag.largest_component_size) / float(node_scale)`
- calls: `_axis_balance_penalty`

#### `_exact_balanced_search(vertices: Sequence[LocalVertex], chain_edges: Sequence[Edge], exact_limit: int)` — line 591
- Enumerate all nearly-balanced transition systems for small lattices.
- kind: function, internal
- returns: `best_plan`; `None`
- calls: `product`, `Counter`, `evaluate_matching_plan`, `_score`, `_is_nearly_balanced`

#### `_greedy_balanced_kotzig_descent(vertices: Sequence[LocalVertex], chain_edges: Sequence[Edge], axes_by_vertex: Dict[Hashable, str], max_passes: int, rng: Random, max_pair_checks: int, deadline: float | None=None)` — line 620
- Apply sampled best two-vertex transition swaps without breaking balance.
- kind: function, internal
- returns: `current`; `<generator>`
- calls: `evaluate_matching_plan`, `_score`, `_sample_pairs`, `time.monotonic`, `rng.randrange`, `seen_pairs.add`

#### `plan_balanced_cycle_matchings(vertices: Sequence[LocalVertex], chain_edges: Sequence[Edge], seed: int | None=None, attempts: int=256, swaps_per_attempt: int=512, exact_limit: int=12, greedy_passes: int=8, greedy_pair_checks: int=4096, time_budget_seconds: float | None=30.0)` — line 715
- Choose balanced local x/y/z transitions for one/few global cycles.
- kind: function
- returns: `best_plan`; `evaluate_matching_plan([], chain_edges, {})`; `exact_plan`
- calls: `_exact_balanced_search`, `Random`, `evaluate_matching_plan`, `_balanced_axis_pool`, `_score`, `_annealing_energy`, `_greedy_balanced_kotzig_descent`, `time.monotonic`, `rng.randrange`, `rng.random`, `math.exp`

### `hygel_martini/hydrogel_builder/core_utils/layout/net_layout.py`

Coordinate layout driven by a periodic net.

#### class `NetLayoutResult` — line 57
A layout plan plus the topology record that produced it.

##### `__init__(self, layout_plan, matching_plan, rewiring, net, repeats, cell, conversion=None)` — line 60
- *(no docstring)*
- kind: method

##### `summary(self)` — line 70
- Flat dict of construction facts for logs and audits.
- kind: method
- returns: `record`
- calls: `record.update`, `self.rewiring.summary`, `counts.get`, `degrees.get`, `degrees.items`

#### `_effective_span(length: float, retreat: float, index: int)` — line 116
- Chain span after both ends stop short of their junction sites.
- kind: function, internal
- returns: `effective`
- raises: `ValueError`
- calls: `ValueError`

#### `_junction_of(vertices)` — line 128
- Map every strand endpoint to the id of the junction that owns it.
- kind: function, internal
- returns: `owner`
- calls: `vertex.endpoints.values`

#### `generate_net_layout_plan(proto_plan, backbone_defs: List[Dict[str, Any]], linker_defs: List[Dict[str, Any]], net: NetDefinition | str, repeats: Sequence[int] | int, cell_parameter: float, linker_library=None, max_span: float | None=None, rewire_seed: int | None=None, rewire_kwargs: Dict[str, Any] | None=None, plan_seed: int | None=None, conversion_fraction: float | None=None, conversion_count: int | None=None, conversion_seed: int | None=None)` — line 137
- Build a :class:`LayoutPlan` on a periodic net.
- kind: function
- returns: `NetLayoutResult(layout_plan, matching_plan, rewiring, definition, counts, cell, conversion=conversion)`; `base + np.outer(np.sin(np.pi * t) * amplitude, lateral)`; `None`
- raises: `ValueError`
- effects: stdout
- calls: `build_periodic_net`, `normalize_cell`, `_junction_of`, `LayoutPlan`, `NetLayoutResult`, `get_net`, `rewire_kwargs.setdefault`, `span_constrained_rewire`, `ValueError`, `Random`, `plan_single_circuit`, `get`, +28 more

### `hygel_martini/hydrogel_builder/core_utils/layout/nets.py`

Periodic net definitions used as construction seeds.

#### class `NetDefinition` — line 65
A periodic net: lattice, basis, edges, and its published invariants.

##### `arms_of_site(self, site: int)` — line 90
- Stable arm labels for one basis site, in definition order.
- kind: method
- returns: `tuple(outgoing + incoming)`

##### `validate(self)` — line 96
- Check every basis site exposes exactly ``coordination`` arms.
- kind: method
- raises: `ValueError`
- calls: `self.arms_of_site`, `ValueError`

##### `bond_vectors(self)` — line 116
- Cartesian vector of each bond, for checking the embedding.
- kind: method
- returns: `[basis[j] + np.asarray(offset, dtype=float) @ cell - basis[i] for i, j, offset in self.bonds]`
- calls: `np.asarray`

#### `get_net(name: str)` — line 182
- Look up a net by RCSR symbol.
- kind: function
- returns: `net`
- raises: `ValueError`
- calls: `lower`, `net.validate`, `ValueError`, `strip`, `join`

#### `validate_repeats(net: NetDefinition, repeats: Sequence[int])` — line 194
- Reject supercells whose shortest cycles come from the box, not the net.
- kind: function
- raises: `ValueError`
- calls: `ValueError`, `join`

#### `build_periodic_net(net: NetDefinition | str, repeats: Sequence[int] | int, cell_parameter: float=1.0, check_repeats: bool=True)` — line 237
- Materialize a periodic net as planner input.
- kind: function
- returns: `(vertices, strands, positions)`
- calls: `definition.validate`, `get_net`, `validate_repeats`, `np.asarray`, `vertex.validate`, `vertices.append`, `strands.append`, `LocalVertex`, `definition.arms_of_site`

### `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py`

Prototype geometry: the reference strand and linker every placement copies.

#### class `ProtoChain` — line 77
A straight reference chain in its local frame.

#### class `ProtoPlan` — line 94
Everything the layout stages share: prototypes, sizes, sequence machinery.

##### `box_vector(self, repeats: Tuple[int, int, int])` — line 117
- Diamond-path box lengths (nm) for the given big-cell repeats.
- kind: method
- returns: `self.cell_vector * np.array(repeats, dtype=np.float64)`
- calls: `np.array`

#### class `BackboneSequenceFactory` — line 264
Draws one monomer sequence (and its geometry) per placed strand.

##### `__init__(self, segment_length: int, backbone_definitions: List[Dict[str, Any]], strategy: Optional[Dict[str, Any]], mean_sep: float, bond_lookup: Dict[Tuple[str, str], Dict[str, Any]], prototype_sequence: Optional[List[Dict[str, Any]]]=None)` — line 277
- *(no docstring)*
- kind: method
- calls: `copy`, `lower`, `self.strategy.get`, `entry.get`, `_resolve_block_pattern`

##### `_random_sequence(self)` — line 310
- One ratio-weighted random draw of ``num_proto_beads`` entries.
- kind: method, internal
- returns: `random.choices(self.definitions, weights=self.weights, k=self.num_proto_beads)`; `[]`
- calls: `random.choices`

##### `next_sequence(self)` — line 316
- The next sequence under the configured strategy (see class doc).
- kind: method
- returns: `self._random_sequence()`; `[]`; `sequence` (+1 more)
- calls: `self._random_sequence`

##### `instantiate(self, enforce_unique: bool=False, used_signatures: Optional[set]=None, max_attempts: int=50)` — line 338
- Draw a sequence and compute its geometry for one placement.
- kind: method
- returns: `(sequence, positions, raw_length, scaled_length)`
- calls: `_chain_geometry_from_sequence`, `self.next_sequence`, `used_signatures.add`, `entry.get`

##### `definition_count(self)` — line 371
- How many backbone chemistries exist (uniqueness is moot at 1).
- kind: property
- returns: `len(self.definitions)`

#### `_weighted_average(lengths: Sequence[float], weights: Sequence[float])` — line 32
- Ratio-weighted mean; plain mean when the weights sum to nothing.
- kind: function, internal
- returns: `float(sum((l * w for l, w in zip(lengths, weights))) / total_weight)`; `float(np.mean(lengths)) if lengths else 0.0`
- calls: `np.mean`

#### `_normalize_linker_axes(linker_axes: Optional[Sequence[str]])` — line 40
- Sanitize configured linker axes to exactly two of 'x'/'y'/'z'.
- kind: function, internal
- returns: `cleaned[:2]`
- calls: `lower`, `cleaned.append`

#### `_compute_linker_length(definition: Dict[str, Any])` — line 64
- Linker contour length (nm): sum of internal plus external bond lengths.
- kind: function, internal
- returns: `float(total)`
- calls: `definition.get`, `bond.get`, `ext.get`

#### `_build_bond_lookup(bond_rules: Optional[List[Dict[str, Any]]], fallback: float)` — line 122
- Index BONDS rules by their sorted ``between`` pair.
- kind: function, internal
- returns: `lookup`; `{}`
- calls: `require_unique`, `rule.get`

#### `_next_backbone_entry(strategy: Dict[str, Any], backbones: List[Dict[str, Any]])` — line 145
- Infinite generator of backbone entries under the sequence strategy.
- kind: function, internal
- returns: `<generator>`
- calls: `lower`, `get`, `require_unique`, `index.get`, `entry.get`, `block.get`, `sequence.extend`, `random.choices`

#### `_build_backbone_sequence(length: int, strategy: Dict[str, Any], backbones: List[Dict[str, Any]])` — line 179
- The first ``length`` entries of the strategy's infinite sequence.
- kind: function, internal
- returns: `[next(gen) for _ in range(length)]`; `[]`
- calls: `_next_backbone_entry`, `next`

#### `_build_backbone_positions(sequence: List[Dict[str, Any]], bond_lookup: Dict[Tuple[str, str], Dict[str, Any]], fallback: float)` — line 189
- Straight-chain bead positions along CHAIN_AXIS for a monomer sequence.
- kind: function, internal
- returns: `(np.array(positions, dtype=np.float64), total)`; `(np.zeros((0, 3), dtype=np.float64), 0.0)`
- calls: `np.zeros`, `positions.append`, `current.copy`, `get`, `bond_lookup.get`, `rule.get`, `np.array`

#### `_chain_geometry_from_sequence(sequence: List[Dict[str, Any]], segment_length: int, mean_sep: float, bond_lookup: Dict[Tuple[str, str], Dict[str, Any]])` — line 217
- Positions plus raw/scaled lengths for one drawn monomer sequence.
- kind: function, internal
- returns: `(positions, float(raw_length), float(scaled_length))`
- calls: `_build_backbone_positions`, `np.zeros`

#### `_resolve_block_pattern(strategy: Dict[str, Any], backbones: List[Dict[str, Any]])` — line 242
- Expand ``blocks: [{id, size}, ...]`` into the concrete entry pattern.
- kind: function, internal
- returns: `pattern or backbones`; `backbones`
- calls: `get`, `require_unique`, `index.get`, `pattern.extend`, `block.get`, `entry.get`

#### `build_proto_backbone(segment_length: int, backbone_definitions: List[Dict[str, Any]], mean_sep: float, strategy: Optional[Dict[str, Any]]=None, bond_rules: Optional[List[Dict[str, Any]]]=None, bond_lookup: Optional[Dict[Tuple[str, str], Dict[str, Any]]]=None)` — line 376
- Build the reference Martini backbone prototype.
- kind: function
- returns: `ProtoChain(positions=positions, types=types, length=float(scaled_length), raw_length=float(raw_length))`
- raises: `ValueError`
- calls: `_build_backbone_sequence`, `_chain_geometry_from_sequence`, `ProtoChain`, `ValueError`, `_build_bond_lookup`, `get`, `types.append`, `entry.get`

#### `build_proto_linker(linker_definitions: List[Dict[str, Any]], strategy: Optional[Dict[str, Any]]=None)` — line 410
- Build the reference linker prototype, or ``None`` without linkers.
- kind: function
- returns: `ProtoChain(positions=np.array(positions, dtype=np.float64), types=types, length=avg_length, raw_length=avg_length)`; `None`
- calls: `lower`, `selected.get`, `_compute_linker_length`, `definition.get`, `ProtoChain`, `positions.append`, `types.append`, `get`, `np.array`, `entry.get`, `random.choices`

#### `prepare_proto_plan(segment_length: int, mean_sep: float, backbone_defs: List[Dict[str, Any]], linker_defs: List[Dict[str, Any]], box_margin: float, backbone_strategy: Optional[Dict[str, Any]]=None, linker_strategy: Optional[Dict[str, Any]]=None, bond_rules: Optional[List[Dict[str, Any]]]=None, linker_library: LinkerTemplateLibrary | None=None, linker_axes: Optional[Sequence[str]]=None)` — line 452
- Assemble the shared ProtoPlan: prototypes, cell sizes, sequence factory.
- kind: function
- returns: `ProtoPlan(segment_length=segment_length, proto_backbone=proto_backbone, proto_linker=proto_linker, box_margin=margin, cell_vector=cell_vector, medium_size=medium_size, small_size=small_size, mean_sep=mean_sep, bond_lookup=bond_lookup, sequence_factory=sequence_factory, linker_library=linker_library, linker_span_lookup=linker_span_lookup)`
- raises: `ValueError`
- calls: `_build_bond_lookup`, `build_proto_linker`, `np.array`, `_normalize_linker_axes`, `require_unique`, `BackboneSequenceFactory`, `ProtoPlan`, `entry.get`, `_weighted_average`, `np.linspace`, `ProtoChain`, `build_proto_backbone`, +7 more

#### `describe_proto_summary(segment_length: int, mean_sep: float, backbone_defs: List[Dict[str, Any]], linker_defs: List[Dict[str, Any]], **kwargs)` — line 571
- Small dict of proto lengths/sizes, for logs and quick inspection.
- kind: function
- returns: `{'backbone_length': proto.proto_backbone.length, 'backbone_length_raw': proto.proto_backbone.raw_length, 'linker_length': proto.proto_linker.length if proto.proto_linker is not None else 0.0, 'num_backbone_beads': proto.proto_backbone.positions.shape[0], 'num_linker_beads': proto.proto_linker.positions.shape[0] if proto.proto_linker is not None else 0}`
- calls: `prepare_proto_plan`, `entry.get`, `definition.get`, `bond.get`, `ext.get`

### `hygel_martini/hydrogel_builder/core_utils/layout/proto_layout.py`

Proto-level layout planning for the historical diamond (f=4) network.

#### class `LayoutCell` — line 64
Placement of one backbone strand.

#### class `LinkPlacement` — line 82
Placement of one crosslinker molecule.

#### class `LayoutPlan` — line 100
Everything ``instantiate_layout`` needs: strand cells plus linkers.

#### `_linear_index(ix: int, iy: int, iz: int, repeats: Tuple[int, int, int])` — line 108
- Row-major linear index of big-cell ``(ix, iy, iz)``.
- kind: function, internal
- returns: `ix * ny * nz + iy * nz + iz`

#### `_normalize(vec: np.ndarray)` — line 114
- Unit vector of ``vec``; a near-zero vector is returned unchanged.
- kind: function, internal
- returns: `vec / norm`; `vec`
- calls: `np.linalg.norm`

#### `_get_anisotropy_axis()` — line 122
- The configured ``anisotropy`` axis ('x'/'y'/'z'), defaulting to 'x'.
- kind: function, internal
- returns: `axis`
- effects: Config/runtime state
- calls: `lower`, `get`, `Config.get_param`

#### `_axis_rotation_matrix(axis: str)` — line 140
- Rotation carrying the x-first construction onto ``axis``.
- kind: function, internal
- returns: `np.eye(3, dtype=float)`; `np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)`; `np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)`
- calls: `np.eye`, `np.array`

#### `_apply_rotation(vec: np.ndarray, rot: np.ndarray)` — line 155
- Apply rotation matrix ``rot`` to a row vector (or row array).
- kind: function, internal
- returns: `vec @ rot.T`

#### `_normalize_linker_axes(linker_axes: Optional[Sequence[str]])` — line 160
- Sanitize the configured linker axes to exactly two of 'x'/'y'/'z'.
- kind: function, internal
- returns: `cleaned[:2]`
- calls: `lower`, `cleaned.append`

#### `_backbone_target_length(entry: Dict[str, Any], proto_plan: ProtoPlan)` — line 185
- Target contour length (nm) of one strand of this backbone definition.
- kind: function, internal
- returns: `float(bond_len) * intervals`
- calls: `entry.get`, `definition.get`

#### `_linker_total_length(entry: Dict[str, Any], fallback: float, override: float | None=None)` — line 201
- Linker span (nm): an explicit override, else the sum of its bond lengths (internal + external), else ``fallback``.
- kind: function, internal
- returns: `float(total)`; `float(override)`
- calls: `entry.get`, `definition.get`, `bond.get`, `ext.get`

#### `generate_layout_plan(proto_plan: ProtoPlan, backbone_defs: List[Dict[str, Any]], linker_defs: List[Dict[str, Any]], repeats: Tuple[int, int, int], backbone_strategy: Dict[str, Any] | None=None, linker_strategy: Dict[str, Any] | None=None, linker_library: LinkerTemplateLibrary | None=None, linker_axes: Optional[Sequence[str]]=None)` — line 221
- Lay out the diamond network: 4 strands + 2 linkers per medium cell.
- kind: function
- returns: `LayoutPlan(proto_plan=proto_plan, cells=cells, links=links)`; `small_edge * (0.5 + idx)`; `linker_len * (1.0 + idx) + small_edge * (0.5 + idx)`
- effects: stdout
- calls: `require_unique`, `_get_anisotropy_axis`, `_axis_rotation_matrix`, `_normalize_linker_axes`, `print`, `LayoutPlan`, `entry.get`, `_linear_index`, `np.array`, `np.zeros`, `_normalize`, `cells.append`, +14 more

### `hygel_martini/hydrogel_builder/core_utils/layout/proto_populator.py`

Turn a LayoutBlueprint into World state: atoms, bonds, and rich terms.

#### `_ordered_chain_entries(chain_key: Tuple[str, int], chain_atom_map: Dict[Tuple[str, int], List[Tuple[int, int]]])` — line 44
- This chain's (bead_index, atom_id) pairs, sorted by bead order.
- kind: function, internal
- returns: `entries`
- calls: `chain_atom_map.get`, `entries.sort`

#### `_mark_backbone_terminals(hydrogel, atom_ids: List[int], metadata: Dict | None=None)` — line 57
- Mark a backbone chain's two attachment ends (``end_tag = 1``).
- kind: function, internal
- returns: `None` (bare return)
- calls: `get`, `append`

#### `_mark_linker_terminals(hydrogel, chain: ChainBlueprint, bead_map: Dict[int, int])` — line 89
- Mark linker beads that carry external bonds as stubs (``end_tag = 2``).
- kind: function, internal
- returns: `None` (bare return)
- calls: `chain.definition.get`, `ext.get`, `append`

#### `_create_backbone_bonds(chain: ChainBlueprint, atom_ids: List[int])` — line 117
- Bond consecutive Martini backbone beads along the chain.
- kind: function, internal
- returns: `None` (bare return)
- calls: `metadata.get`, `bond_lookup.get`, `params.get`, `Attributes.Bond`, `entry_a.get`, `entry_b.get`, `default_params.get`

#### `_create_linker_bonds(chain: ChainBlueprint, bead_map: Dict[int, int])` — line 148
- Create the linker body's internal bonds from its bead-indexed list.
- kind: function, internal
- returns: `None` (bare return)
- calls: `chain.definition.get`, `bond_def.get`, `bead_map.get`, `Attributes.Bond`, `bond_def.items`

#### `_finalize_counts(hydrogel)` — line 174
- Refresh the hydrogel's atom/bond counters from the class registries.
- kind: function, internal

#### `populate_hydrogel_from_blueprint(hydrogel, blueprint: LayoutBlueprint)` — line 180
- Materialize the blueprint into World atoms, bonds, and rich terms.
- kind: function
- raises: `ValueError`
- effects: Config/runtime state
- calls: `defaultdict`, `_finalize_counts`, `Attributes.Atom`, `np.array`, `chain_meta_by_key.get`, `append`, `_ordered_chain_entries`, `template_by_chain.get`, `orig_to_global_by_chain.get`, `_map_constraints`, `items`, `atom_bp.extra.get`, +38 more

### `hygel_martini/hydrogel_builder/core_utils/layout/rewire.py`

Span-constrained rewiring of a net seed toward a representative topology.

#### class `RewiringResult` — line 68
Outcome of a rewiring run, including why proposals were refused.

##### `acceptance_rate(self)` — line 86
- Accepted swaps over proposed swaps (0 when nothing was proposed).
- kind: property
- returns: `self.accepted / self.proposed if self.proposed else 0.0`

##### `summary(self)` — line 90
- Flat dict of the run's counters and the final loop-order snapshot.
- kind: method
- returns: `{'proposed': self.proposed, 'accepted': self.accepted, 'acceptance_rate': self.acceptance_rate, 'rejected_span': self.rejected_span, 'rejected_primary_loop': self.rejected_primary_loop, 'rejected_parallel': self.rejected_parallel, 'rejected_degenerate': self.rejected_degenerate, 'converged': self.converged, 'sweeps': self.sweeps, 'noise_floor': self.noise_floor}`

#### `_endpoint_owner(vertices)` — line 106
- Map every endpoint to its owning junction's id.
- kind: function, internal
- returns: `owner`
- calls: `vertex.validate`, `vertex.endpoints.values`

#### `reduced_edges(strands: Sequence[Strand], owner: Mapping[Endpoint, Hashable], node_index: Mapping[Hashable, int])` — line 116
- Collapse strands onto the junction--strand multigraph.
- kind: function
- returns: `[(node_index[owner[left]], node_index[owner[right]]) for left, right in strands]`

#### `loop_order_snapshot(n_nodes: int, edges: Sequence[Tuple[int, int]])` — line 128
- Normalized loop-order distribution, for convergence testing.
- kind: function
- returns: `{order: count / total for order, count in histogram.items()}`; `{}`
- calls: `loop_order_histogram`, `histogram.values`, `histogram.items`

#### `total_variation_distance(left: Mapping[int, float], right: Mapping[int, float])` — line 142
- Half the L1 distance between two normalized loop-order distributions.
- kind: function
- returns: `0.5 * sum((abs(left.get(order, 0.0) - right.get(order, 0.0)) for order in orders))`
- calls: `left.get`, `right.get`

#### `span_constrained_rewire(vertices, strands: Sequence[Strand], positions: Mapping[Hashable, np.ndarray], max_span: float, box: Sequence[float] | None=None, seed: int | None=None, max_sweeps: int=200, swaps_per_sweep: int | None=None, tolerance: float | None=None, patience: int=3, allow_primary_loops: bool=True, allow_parallel_strands: bool=True, record_history: bool=True)` — line 151
- Rewire ``strands`` under a span cutoff until the loop spectrum settles.
- kind: function
- returns: `result`; `RewiringResult(strands=working, converged=True)`; `float(np.linalg.norm(_minimum_image(delta, box_vector)))` (+1 more)
- raises: `ValueError`
- calls: `Random`, `_endpoint_owner`, `_normalize_box`, `RewiringResult`, `loop_order_snapshot`, `_assert_degree_preserved`, `ValueError`, `np.asarray`, `pair_key`, `reduced_edges`, `result.history.append`, `window.append`, +12 more

#### `_noise_floor(window: Sequence[Mapping[int, float]])` — line 351
- Sweep-to-sweep variation within the window halves.
- kind: function, internal
- returns: `sum(distances) / len(distances)`; `0.0`
- calls: `total_variation_distance`

#### `_mean_distribution(snapshots: Sequence[Mapping[int, float]])` — line 370
- Average several loop-order distributions order by order.
- kind: function, internal
- returns: `{order: sum((item.get(order, 0.0) for item in snapshots)) / count for order in orders}`; `{}`
- calls: `union`, `item.get`

#### `_assert_degree_preserved(vertices, original: Sequence[Strand], rewired: Sequence[Strand], owner: Mapping[Endpoint, Hashable])` — line 384
- A swap that changed a junction's functionality is a bug, not an outcome.
- kind: function, internal
- returns: `counts`
- raises: `AssertionError`
- calls: `AssertionError`, `degrees`, `changed.items`

### `hygel_martini/hydrogel_builder/core_utils/layout/template_placement.py`

Shared geometry helpers for template placement and side-chain tuning.

#### `build_alignment_basis(axis: np.ndarray | list[float] | tuple[float, ...])` — line 10
- Build an orthonormal basis whose x-axis follows ``axis``.
- kind: function
- returns: `np.column_stack((x_axis, y_axis, z_axis))`
- calls: `np.asarray`, `np.linalg.norm`, `np.array`, `np.cross`, `np.column_stack`, `np.dot`

#### `place_template_coords(coords: np.ndarray, origin: np.ndarray | list[float] | tuple[float, ...], axis_vector: np.ndarray | list[float] | tuple[float, ...])` — line 40
- Rotate and translate template coordinates onto an axis-aligned frame.
- kind: function
- returns: `np.asarray(origin, dtype=float) + np.asarray(coords, dtype=float) @ basis.T`
- calls: `build_alignment_basis`, `np.asarray`

#### `compute_template_positions(coords: np.ndarray, origin: np.ndarray | list[float] | tuple[float, ...], normal_vector: np.ndarray | list[float] | tuple[float, ...], tangent_vector: np.ndarray | list[float] | tuple[float, ...])` — line 50
- Build a local side-chain frame from normal and tangent vectors.
- kind: function
- returns: `np.asarray(origin, dtype=float) + np.asarray(coords, dtype=float) @ rotation.T`; `None`
- calls: `np.asarray`, `np.linalg.norm`, `np.cross`, `np.column_stack`

#### `resolve_sidechain_placement_tuning(sim_params: Dict[str, Any], atom_count: int)` — line 79
- Resolve side-chain search settings from config with large-system fallback.
- kind: function
- returns: `tuning`
- calls: `sim_params.get`

### `hygel_martini/hydrogel_builder/core_utils/runtime/__init__.py`

Runtime helpers for packmol, GROMACS, topology updates, and patching.

### `hygel_martini/hydrogel_builder/core_utils/runtime/aa_bonded.py`

Junction-crossing dihedrals and 1-4 pairs for all-atom force fields.

#### `_adjacency(world)` — line 59
- Undirected bond adjacency over every atom currently in ``world``.
- kind: function, internal
- returns: `adjacency`
- calls: `world.Bonds.values`, `add`

#### `_same_template(world, atom_ids)` — line 71
- Whether one template *instance* owns the whole path.
- kind: function, internal
- returns: `len(owners) == 1`; `False`
- calls: `owners.add`, `id`

#### `_builder_bonds(world, adjacency)` — line 94
- Bonds joining two different template instances.
- kind: function, internal
- returns: `crossing`
- calls: `world.Bonds.values`, `_same_template`, `crossing.append`

#### `generate_junction_bonded_terms(world, dihedral_funct: int=3, generate_dihedrals: bool=True, generate_pairs: bool=True, generate_impropers: bool=False, improper_funct: int=4, improper_params: List[float] | None=None)` — line 110
- Enumerate junction-crossing dihedrals and 1-4 pairs on ``world``.
- kind: function
- returns: `(dihedrals_added, sorted_pairs, impropers_added)`
- calls: `_adjacency`, `adjacency.items`, `existing.add`, `frozenset`, `_builder_bonds`, `Attributes.Dihedral`, `one_three.add`, `reversed`, `_same_template`, `centres.add`, `pairs.add`

### `hygel_martini/hydrogel_builder/core_utils/runtime/backbone_patcher.py`

Rule-based post-hoc patching of backbone bonded terms in ``World``.

#### `patch_backbone_topology(config_path, sections=('bonds', 'angles', 'dihedrals'))` — line 21
- Patches the World's Bonds, Angles and Dihedrals based on backbone.yaml rules.
- kind: function
- returns: `True`; `rule.get('residue_name', []).count('*') + rule.get('bead_type', []).count('*')`; `(funct, normalize_param(rule.get('c0')), normalize_param(rule.get('c1')), normalize_param(rule.get('multiplicity')))` (+9 more)
- effects: filesystem, Config/runtime state, global registry, stdout
- calls: `World.Bonds.items`, `print`, `os.path.exists`, `open`, `yaml.safe_load`, `backbone_res_names.intersection`, `append`, `adj.get`, `normalize_param`, `World.Angles.items`, `World.Angles.clear`, `existing_angles.items`, +27 more

### `hygel_martini/hydrogel_builder/core_utils/runtime/dynamic_crosslink.py`

Materialize planned linker edges or assign ends for unplanned layouts.

#### class `StubAssignment` — line 23
Chosen backbone end for a single linker stub.

#### `_jsonable_identifier(value)` — line 33
- Convert nested endpoint identifiers into deterministic JSON values.
- kind: function, internal
- returns: `repr(value)`; `[_jsonable_identifier(item) for item in value]`; `{str(key): _jsonable_identifier(value[key]) for key in sorted(value, key=str)}` (+1 more)
- calls: `_jsonable_identifier`

#### `_edge_plan_sha256(edges_by_linker)` — line 49
- Hash an unordered endpoint-edge set for each linker deterministically.
- kind: function, internal
- returns: `hashlib.sha256(encoded).hexdigest()`
- calls: `encode`, `hexdigest`, `canonical_edges.sort`, `payload.append`, `endpoints.sort`, `canonical_edges.append`, `json.dumps`, `hashlib.sha256`, `_jsonable_identifier`

#### `normalize_box_vector(box_vec)` — line 66
- Return a 3x3 cell (rows are cell vectors), or ``None`` without PBC.
- kind: function
- returns: `None`; `normalize_cell(box_vec)`
- calls: `normalize_cell`

#### `pbc_distance(first, second, box_size: np.ndarray | None)` — line 82
- Minimum-image distance between two coordinates.
- kind: function
- returns: `minimum_image_distance(first, second, box_size)`
- calls: `minimum_image_distance`

#### `group_linker_stubs(atoms: Iterable[object])` — line 87
- Group linker terminal stubs by linker chain index.
- kind: function
- returns: `grouped`
- calls: `append`, `sort`, `grouped.setdefault`, `stub_type.startswith`

#### `collect_backbone_ends(atoms: Iterable[object])` — line 110
- Collect true backbone end atoms keyed by backbone chain index.
- kind: function
- returns: `ends_by_chain`
- calls: `append`, `ends_by_chain.setdefault`

#### `_stub_target_backbone(stub: object)` — line 127
- The backbone id this stub is restricted to, or ``None`` for any.
- kind: function, internal
- returns: `None`; `target`; `fallback`

#### `_is_compatible_target(stub: object, backbone_atom: object)` — line 143
- Whether this backbone end satisfies the stub's target restriction.
- kind: function, internal
- returns: `getattr(backbone_atom, 'backbone_type', None) == target`; `True`
- calls: `_stub_target_backbone`

#### `_candidate_end_options(stub: object, backbone_ends: Dict[int, List[object]], box_size: np.ndarray | None, candidate_limit: int, respect_target_backbone: bool=True)` — line 151
- Return the nearest compatible end for each compatible backbone chain.
- kind: function, internal
- returns: `options`; `options[:candidate_limit]`
- calls: `backbone_ends.items`, `options.sort`, `options.append`, `pbc_distance`, `_is_compatible_target`

#### `_plan_explicit_graph_crosslinks(linker_stubs: Dict[int, List[object]], backbone_ends: Dict[int, List[object]], box_size: np.ndarray | None)` — line 182
- Materialize the endpoint edges selected by the layout graph planner.
- kind: function, internal
- returns: `(assignments, notes)`; `sum((pbc_distance(stub.position, endpoint_atom.position, box_size) for stub, group_index in zip(stubs, order) for _, endpoint_atom in groups[group_index]))`
- raises: `ValueError`
- calls: `backbone_ends.values`, `assignments.items`, `_edge_plan_sha256`, `notes.append`, `linker_stubs.items`, `ValueError`, `unpaired_planned.items`, `format`, `permutations`, `edge_regime_linkers.add`, `resolved_edges.append`, `planned_edges_by_linker.items`, +10 more

#### `plan_dynamic_crosslinks(linker_stubs: Dict[int, List[object]], backbone_ends: Dict[int, List[object]], box_vec, candidate_limit: int=8, targets_per_stub: int=1, respect_target_backbone_policy: bool=False)` — line 470
- Assign compatible backbone ends to each placed linker stub.
- kind: function
- returns: `(assignments, notes)`; `_plan_explicit_graph_crosslinks(linker_stubs, backbone_ends, box_size)`; `False` (+1 more)
- raises: `ValueError`
- calls: `normalize_box_vector`, `_plan_explicit_graph_crosslinks`, `notes.append`, `linker_stubs.items`, `_candidate_end_options`, `pairings.sort`, `pairing_options.append`, `used_end_atoms.add`, `ValueError`, `format`, `product`, `states.sort`, +18 more

### `hygel_martini/hydrogel_builder/core_utils/runtime/geo_opt.py`

Geometry optimization helpers (grompp/mdrun) with consistent stdout/stderr logging.

#### `mdrun_gpu_flags(gpu_id: Optional[int])` — line 20
- Return (extra_args, env_overrides) for GPU/CPU mdrun control (single-process).
- kind: function
- returns: `([], {'CUDA_VISIBLE_DEVICES': str(gpu_id)})`; `(['-nb', 'cpu', '-update', 'cpu'], {})`

#### `build_mdrun_cmd(gmx: str, base_args: List[str], gpu_id: Optional[str], mpi_np: Optional[int], mpi_args: List[str], extra: List[str])` — line 31
- Build the full mdrun command and required env overrides.
- kind: function
- returns: `(mdrun + extra, env_extra)`; `(['mpirun', '-np', str(mpi_np)] + mpi_args + mdrun + extra, env_extra)`

#### `_print_process_output(label: str, stdout: Optional[str], stderr: Optional[str])` — line 63
- Print stdout/stderr blocks for a subprocess result using a common format.
- kind: function, internal
- effects: stdout
- calls: `print`, `stdout_content.strip`, `stdout_content.rstrip`, `stderr_content.strip`, `stderr_content.rstrip`

#### `_run_with_logs(cmd, label, log_path=None, cwd=None, env=None, input_text=None)` — line 76
- Run a subprocess, print stdout/stderr in a consistent block, and optionally append to a log file. Debug logging goes to Config.debug_log when enabled.
- kind: function, internal
- returns: `proc`
- effects: subprocess, filesystem, stdout
- calls: `join`, `print`, `Config.debug_log`, `subprocess.run`, `_print_process_output`, `open`, `f.write`

#### `_create_mdp_file(directory: str, cell_opt: bool=False, em_tol: float=1000.0, nsteps: int=5000, mdp_overrides: Optional[Dict[str, Any]]=None)` — line 102
- Creates a gromacs .mdp file for energy minimization in the specified directory.
- kind: function, internal
- returns: `mdp_filepath`
- effects: filesystem
- calls: `os.path.join`, `mdp_overrides.items`, `passthrough.items`, `open`, `f.write`, `mdp_defaults.items`, `key.replace`

#### `run_geo_opt(structure_file: str, topology_file: str, output_dir: str, cell_opt: bool=False, gmx_executable: str='gmx', em_tol: float=1000.0, nsteps: int=5000, maxwarn: int=1, mdp_overrides: Optional[Dict[str, Any]]=None, deffnm_prefix: str='em', mdrun_extra_args: Optional[List[str]]=None, gpu_id: Optional[int]=None, ntomp: Optional[int]=None, mpi_np: Optional[int]=None, mpi_args: Optional[List[str]]=None)` — line 192
- Performs geometry optimization (energy minimization) for a given structure using Gromacs.
- kind: function
- returns: `None`; `optimized_structure`
- effects: filesystem, stdout
- calls: `os.path.abspath`, `_create_mdp_file`, `os.path.join`, `_run_with_logs`, `build_mdrun_cmd`, `os.environ.copy`, `run_env.update`, `os.path.exists`, `os.path.isdir`, `os.makedirs`, `grompp_cmd.extend`, `print`, +3 more

### `hygel_martini/hydrogel_builder/core_utils/runtime/packer.py`

Packmol wrappers and GRO/PDB/XYZ round-trip conversion helpers.

#### `_normalize_box_lengths(box_lengths_nm, fallback=None)` — line 28
- Returns a [lx, ly, lz] list in nm. Accepts scalar or iterable input.
- kind: function, internal
- returns: `[values[0], values[1], values[1]]`; `[val, val, val]`; `None` (+2 more)

#### `_gro_to_pdb_manual(gro_path, pdb_path)` — line 63
- Convert GRO to PDB by fixed-column parsing (no GROMACS needed).
- kind: function, internal
- returns: `pdb_path`
- effects: filesystem, stdout
- calls: `print`, `Config.debug_log`, `open`, `f.readlines`, `strip`, `out.write`, `upper`

#### `convert_gro_to_pdb(gro_path, pdb_path, gmx_path)` — line 97
- Converts a .gro file to a .pdb file using gmx editconf or a manual fallback.
- kind: function
- returns: `pdb_path`; `_gro_to_pdb_manual(gro_path, pdb_path)`
- effects: subprocess, stdout
- calls: `subprocess.run`, `print`, `Config.debug_log`, `_gro_to_pdb_manual`, `join`

#### `convert_pdb_to_gro(pdb_path, gro_path, gmx_path, box_lengths_nm=None)` — line 110
- Converts a .pdb file to a .gro file using gmx editconf or a manual fallback.
- kind: function
- returns: `gro_path`; `_pdb_to_gro_manual(pdb_path, gro_path, lengths_nm)`
- effects: subprocess, stdout
- calls: `_normalize_box_lengths`, `command.extend`, `subprocess.run`, `print`, `Config.debug_log`, `_pdb_to_gro_manual`, `join`

#### `_pdb_to_gro_manual(pdb_path, gro_path, box_lengths_nm=None)` — line 127
- Convert PDB to GRO by fixed-column parsing (no GROMACS needed).
- kind: function, internal
- returns: `gro_path`
- effects: filesystem, stdout
- calls: `_normalize_box_lengths`, `print`, `Config.debug_log`, `open`, `out.write`, `line.startswith`, `atoms.append`, `strip`

#### `convert_xyz_to_gro(xyz_path, gro_path, gmx_path, molecule_name='MOL')` — line 163
- Converts a .xyz file to a .gro file by manually parsing it.
- kind: function
- returns: `gro_path`
- effects: filesystem, stdout
- calls: `strip`, `print`, `open`, `f_xyz.readlines`, `f_gro.write`, `line.split`

#### `_read_gro_atom_names(gro_path)` — line 225
- Read the atom-name column of a GRO file; empty list on any failure.
- kind: function, internal
- returns: `read_gro_atom_names(gro_path)`; `[]`
- effects: stdout
- calls: `read_gro_atom_names`, `print`, `Config.debug_log`

#### `_restore_atom_names_from_sources(output_gro, base_gro, molecules_to_add)` — line 235
- Rewrite atom names in a packed GRO from the original source files.
- kind: function, internal
- returns: `False`; `True`
- effects: filesystem, stdout
- calls: `_read_gro_atom_names`, `expected_names.extend`, `Config.debug_log`, `mol.get`, `open`, `f.readlines`, `strip`, `print`, `updated.append`, `f.writelines`

#### `run_packmol(packmol_path, inp_filename, output_dir, sim_params=None)` — line 290
- Generates a Packmol input file and runs Packmol.
- kind: function
- returns: `None`
- raises: `subprocess.CalledProcessError`
- effects: subprocess, filesystem, stdout
- calls: `print`, `Config.debug_log`, `open`, `subprocess.run`, `subprocess.CalledProcessError`, `sim_params.get`

#### `_packmol_log_succeeded(log_path)` — line 349
- Return True when a Packmol log reports clean success.
- kind: function, internal
- returns: `'Success!' in text and 'ENDED WITHOUT PERFECT PACKING' not in text and ('STOP' not in text)`; `False`
- effects: filesystem
- calls: `os.path.exists`, `open`, `handle.read`

#### `_run_packmol_to_log(packmol_path, inp_filename, output_dir, log_filename, stage_name)` — line 366
- Run one Packmol stage, teeing stdout+stderr into a log file.
- kind: function, internal
- raises: `RuntimeError`
- effects: subprocess, filesystem, stdout
- calls: `print`, `open`, `subprocess.run`, `RuntimeError`, `_packmol_log_succeeded`

#### `_read_pdb_atoms_nm(pdb_path)` — line 388
- Parse ATOM/HETATM records into dicts with coordinates in nm.
- kind: function, internal
- returns: `atoms`
- effects: filesystem
- calls: `open`, `atoms.append`, `raw.startswith`, `strip`

#### `_normalize_pdb_atoms_to_box(atoms, box_lengths_nm)` — line 412
- Shift atoms to non-negative coordinates and grow the box to fit them.
- kind: function, internal
- returns: `(atoms, expanded)`; `(atoms, lengths)`
- calls: `_normalize_box_lengths`

#### `_write_gro_from_pdb_atoms(gro_path, atoms, box_lengths_nm, title)` — line 453
- Write atom dicts from ``_read_pdb_atoms_nm`` as a GRO file (nm units).
- kind: function, internal
- effects: filesystem
- calls: `_normalize_box_lengths`, `open`, `handle.write`

#### `_min_inter_polymer_molecule_distance(atoms, polymer_atom_count)` — line 470
- Minimum distance (nm) between atoms of different polymer molecules.
- kind: function, internal
- returns: `None if best == float('inf') else best`; `None`
- calls: `math.sqrt`

#### `pack_polymer_then_water_two_stage(*, output_dir, polymer_pdb, water_pdb, polymer_count, water_count, box_lengths_nm, packmol_path, final_output_gro, tolerance=2.0, seed=100000, polymer_nloop=500, water_nloop=250, polymer_stage_pdb='polymer_stage1.pdb', packed_pdb='packed.pdb', polymer_stage_prefix='packmol_polymer_stage', water_stage_prefix='packmol_water_stage', polymer_atom_count=None, audit_json=None, title='two-stage polymer then water Packmol')` — line 502
- Pack polymers first, then pack water around the fixed polymer stage.
- kind: function
- returns: `audit`
- raises: `ValueError`
- effects: filesystem
- calls: `os.makedirs`, `_normalize_box_lengths`, `os.path.abspath`, `os.path.join`, `_run_packmol_to_log`, `_read_pdb_atoms_nm`, `_normalize_pdb_atoms_to_box`, `_write_gro_from_pdb_atoms`, `Counter`, `ValueError`, `open`, `handle.write`, +4 more

#### `pack_system_with_molecules(step_name, base_structure_gro, molecules_to_add, final_output_gro, box_lengths_nm, sim_params)` — line 647
- Runs a full packing step: converts to PDB, generates packmol input, runs packmol, and converts back to GRO. Intermediate files are saved with step-specific names for debugging.
- kind: function
- returns: `(final_output_gro, True)`; `(final_output_gro, False)`
- raises: `ValueError`
- effects: filesystem, stdout
- calls: `print`, `_normalize_box_lengths`, `convert_gro_to_pdb`, `os.path.join`, `join`, `run_packmol`, `convert_pdb_to_gro`, `_restore_atom_names_from_sources`, `sim_params.get`, `ValueError`, `molecules_pdb_to_add.append`, `open`, +6 more

### `hygel_martini/hydrogel_builder/core_utils/runtime/topology_updater.py`

GROMACS ``.top`` file creation and in-place [ molecules ] maintenance.

#### `create_system_topology(output_dir, top_path, itp_files)` — line 17
- Write a fresh system ``.top`` with ordered includes and an empty molecules list.
- kind: function
- effects: filesystem, Config/runtime state, stdout
- calls: `Config.get_param`, `print`, `os.path.abspath`, `open`, `f.write`, `ordered_itps.append`

#### `update_topology_molecules(topology_file, molecule_counts, additional_itp_includes=None)` — line 54
- Updates the [ molecules ] section of a GROMACS topology file.
- kind: function
- returns: `None` (bare return)
- effects: filesystem, stdout
- calls: `os.path.exists`, `print`, `open`, `f.readlines`, `molecule_counts.items`, `startswith`, `re.match`, `lines.append`, `f.writelines`, `new_includes.append`, `new_molecule_lines.append`, `line.strip`, +2 more

### `hygel_martini/hydrogel_builder/core_utils/templates/__init__.py`

Template loaders and validators for monomers, linkers, and rich ITP data.

### `hygel_martini/hydrogel_builder/core_utils/templates/linker_loader.py`

Linker (junction/crosslinker) templates: N-stub molecules from GRO/ITP.

#### class `LinkerTemplate` — line 41
A parsed junction molecule in its local frame.

#### class `LinkerTemplateRecord` — line 105
A template plus its mixing ratio from the LINKERS entry.

#### class `LinkerTemplateLibrary` — line 113
All loaded linkers, in declaration order and by id.

#### `linker_definitions_from_library(library: LinkerTemplateLibrary)` — line 120
- Render the library back into config-shaped LINKERS definition dicts.
- kind: function
- returns: `definitions`; `[{'from_bead': idx, 'to_backbone': params.get('target'), 'to_backbones': params.get('targets'), 'stub_index': params.get('stub_index'), 'funct': params.get('funct', 1), 'length': params.get('c0'), 'fc': params.get('c1')} for idx, params in bonds]`
- calls: `definition.update`, `definitions.append`, `bead_defs.append`, `bonds.append`, `_external`, `params.get`

#### `_extract_definition(itp_path: str, molecule_name: str | None)` — line 227
- Return exactly one [ moleculetype ] from the linker ITP.
- kind: function, internal
- returns: `next(iter(definitions.values()))`; `definitions[molecule_name]`
- raises: `ValueError`
- effects: Config/runtime state
- calls: `Config.get_runtime`, `read_itp_definitions`, `next`, `ValueError`, `iter`, `definitions.values`

#### `_map_backbone_ids(beads: List[Dict], backbone_defs: List[Dict])` — line 258
- Map bead nr -> backbone id for beads whose residue names a backbone.
- kind: function, internal
- returns: `mapping`
- calls: `get`, `bead.get`, `residue_to_backbone.get`

#### `_convert_params(bond_def: Dict)` — line 283
- Normalize a parsed bond row to ``{funct, c0, c1}`` keyword params.
- kind: function, internal
- returns: `{'funct': bond_def.get('funct', 1), 'c0': length, 'c1': fc}`
- calls: `bond_def.get`

#### `_backbone_mass_lookup(backbone_defs: List[Dict])` — line 299
- Mass per backbone id, for the Martini stub-mass-override rule.
- kind: function, internal
- returns: `masses`
- calls: `backbone.get`, `definition.get`

#### `_stub_configuration(entry: Dict, linker_id: str)` — line 322
- Per-stub bond-parameter entries, in stub order.
- kind: function, internal
- returns: `[entry.get('backbone_1', []), entry.get('backbone_2', [])]`; `resolved`
- raises: `ValueError`
- calls: `entry.get`, `ValueError`, `resolved.append`

#### `_resolve_stub_targets(backbone_bonds: List[Dict], linker_id: str, side_name: str)` — line 357
- Backbone identifiers a stub may bond to.
- kind: function, internal
- returns: `tuple(targets)`
- raises: `ValueError`
- calls: `ValueError`, `bond.get`

#### `_stub_mass_for_targets(targets: Tuple[str, ...], backbone_masses: Dict[str, float], linker_id: str, side_name: str)` — line 385
- Mass of a stub that stands in for one of ``targets``.
- kind: function, internal
- returns: `next(iter(distinct))`; `None`
- raises: `ValueError`
- calls: `next`, `masses.values`, `ValueError`, `iter`, `masses.items`

#### `_resolve_stub_caps(entry: Dict, linker_id: str, functionality: int, beads: List[Dict], index_map: Dict[int, int], stub_indices: List[int], stub_bonds: List[List[Tuple[int, Dict]]])` — line 427
- Parse and validate the optional per-stub ``stub_caps`` list.
- kind: function, internal
- returns: `resolved`; `[{} for _ in range(functionality)]`
- raises: `ValueError`
- calls: `entry.get`, `ValueError`, `bead.get`, `items`, `resolved.append`, `spec.get`, `name_to_bead.get`, `cap_positions.append`, `cap_originals.append`, `type`

#### `_orthonormal_basis(span_vec: np.ndarray, ref: np.ndarray=np.array([0.0, 0.0, 1.0]))` — line 530
- Right-handed orthonormal basis whose x axis is ``span_vec``.
- kind: function, internal
- returns: `np.column_stack((x_axis, y_axis, z_axis))`
- raises: `ValueError`
- calls: `np.array`, `np.linalg.norm`, `ref.copy`, `np.cross`, `np.column_stack`, `ValueError`, `np.dot`

#### `_load_single_linker(entry: Dict, backbone_defs: List[Dict])` — line 564
- Load and validate one LINKERS entry into a :class:`LinkerTemplate`.
- kind: function, internal
- returns: `LinkerTemplate(id=linker_id, beads=bead_templates, coords=coords, internal_bonds=internal_bonds, internal_angles=internal_angles, internal_dihedrals=internal_dihedrals, internal_impropers=internal_impropers, dihedrals_full=definition.get('dihedrals', []), impropers_full=definition.get('impropers', []), constraints=constraints, pairs=pairs, exclusions=exclusions, virtual_sites=virtual_sites, restraints=restraints, cmaptypes=cmaptypes, polarization=polarization, other_sections=other_sections, stub_bonds=stub_bonds, stub_backbone_targets=tuple(stub_targets), arm_vectors=arm_vectors, functionality=functionality, stub_config_bonds=stub_config, stub_caps=stub_caps, stub_bonds_left=stub_bonds_left, stub_bonds_right=stub_bonds_right, backbone_ids=backbone_ids, span_vector=basis[:, 0] * span_length, span_length=span_length, linker_name=linker_name, backbone_name=backbone_name, stub_definitions=stub_definitions, backbone_1_bonds=backbone_1_bonds, backbone_2_bonds=backbone_2_bonds, stub_stub_bonds=stub_stub_bonds)`
- raises: `ValueError`, `FileNotFoundError`
- calls: `entry.get`, `_extract_definition`, `definition.get`, `_stub_configuration`, `_backbone_mass_lookup`, `stub_definitions.sort`, `read_gro_atoms`, `np.array`, `reshape`, `_resolve_stub_caps`, `LinkerTemplate`, `ValueError`, +35 more

#### `load_linker_templates(linker_entries: List[Dict], backbone_defs: List[Dict])` — line 882
- Load every LINKERS entry, refusing duplicate ids.
- kind: function
- returns: `LinkerTemplateLibrary(records=records, lookup=lookup)`
- raises: `DuplicateDeclaration`
- calls: `LinkerTemplateLibrary`, `_load_single_linker`, `LinkerTemplateRecord`, `records.append`, `DuplicateDeclaration`, `entry.get`

### `hygel_martini/hydrogel_builder/core_utils/templates/monomer_loader.py`

Monomer templates: one repeat unit per file, one backbone bead per unit.

#### class `BeadTemplate` — line 42
One template atom/bead, as parsed from the ITP with its GRO coordinate.

#### class `MonomerTemplate` — line 63
A parsed monomer: side beads plus every bonded term of the repeat unit.

#### class `TemplateRecord` — line 99
A template plus its mixing ratio from the MONOMERS entry.

#### class `MonomerTemplateLibrary` — line 107
All loaded monomers: in declaration order, by backbone id, and by id.

#### `_extract_single_definition(itp_path: str, molecule_name: str | None)` — line 115
- Return exactly one [ moleculetype ] definition from ``itp_path``.
- kind: function, internal
- returns: `next(iter(definitions.values()))`; `definitions[molecule_name]`
- raises: `ValueError`
- effects: Config/runtime state
- calls: `Config.get_runtime`, `read_itp_definitions`, `next`, `ValueError`, `iter`, `definitions.values`

#### `_find_backbone_bead_index(beads: List[Dict])` — line 147
- 1-based index of the first bead whose residue or atom name is BCK*.
- kind: function, internal
- returns: `None`; `bead['nr']`
- calls: `upper`, `residu.startswith`, `atom.startswith`, `bead.get`

#### `_match_backbone(beads: List[Dict], backbone_defs: List[Dict], override_id: str | None=None)` — line 157
- Decide which bead is the backbone bead and which backbone id owns it.
- kind: function, internal
- returns: `(fallback_idx, override_id)`; `(bead['nr'], backbone_id)`; `(fallback_idx, default_id)` (+1 more)
- raises: `ValueError`
- calls: `require_unique`, `_find_backbone_bead_index`, `ValueError`, `get`, `residue_pairs.extend`, `id_to_residue.get`, `residue_to_backbone.get`, `next`, `bead.get`, `iter`, `id_to_residue.keys`

#### `_convert_params(bond_def: Dict)` — line 217
- Normalize a parsed bond row to ``{funct, c0, c1}`` keyword params.
- kind: function, internal
- returns: `{'funct': bond_def.get('funct', 1), 'c0': length, 'c1': fc}`
- calls: `bond_def.get`

#### `_load_single(entry: Dict, backbone_defs: List[Dict])` — line 234
- Load one MONOMERS entry (``{id, gro, itp, [backbone_id], ...}``).
- kind: function, internal
- returns: `template`
- raises: `ValueError`, `FileNotFoundError`
- calls: `entry.get`, `_extract_single_definition`, `definition.get`, `_match_backbone`, `read_gro_atoms`, `np.array`, `MonomerTemplate`, `ValueError`, `os.path.isfile`, `FileNotFoundError`, `bead.get`, `coords_list.append`, +16 more

#### `load_monomer_templates(monomer_entries: List[Dict], backbone_defs: List[Dict])` — line 420
- Load every MONOMERS entry into a library, refusing duplicate ids.
- kind: function
- returns: `MonomerTemplateLibrary(records=records, by_backbone=by_backbone, lookup=lookup)`
- raises: `DuplicateDeclaration`
- calls: `MonomerTemplateLibrary`, `entry.get`, `_load_single`, `TemplateRecord`, `records.append`, `append`, `DuplicateDeclaration`, `by_backbone.setdefault`

### `hygel_martini/hydrogel_builder/core_utils/templates/rich_itp_validator.py`

Rich ITP section validator.

#### `_try_int(token: Any)` — line 15
- Return ``int(token)`` or None when the token is not an integer.
- kind: function, internal
- returns: `int(token)`; `None`

#### `_in_range(idx: int, atom_count: int)` — line 23
- Check a 1-based atom index against the molecule's atom count.
- kind: function, internal
- returns: `1 <= idx <= atom_count`

#### `validate_and_filter_other_sections(extras: Dict[str, List[Dict[str, Any]]], atom_count: int, strict: bool=False)` — line 28
- Validate rich sections stored in World.OtherSections.
- kind: function
- returns: `(filtered, warnings)`; `({}, [])`
- raises: `ValueError`
- calls: `extras.get`, `extras.items`, `row.get`, `out_excl.append`, `warnings.append`, `sec.lower`, `out_rows.append`, `ValueError`, `sec_lower.endswith`, `append`, `vs_by_sec.items`, `filtered.items`, +11 more

### `hygel_martini/hydrogel_builder/core_utils/templates/strand_loader.py`

Whole-strand templates: one molecule per network strand.

#### class `StrandTemplate` — line 43
One whole strand molecule, ready for rigid placement.

#### `load_strand_template(entry: Dict)` — line 88
- Load a whole-strand template from ``{id, gro, itp[, molecule_name]}``.
- kind: function
- returns: `StrandTemplate(id=strand_id, beads=template_beads, coords=coords, attachment_positions=(head_pos, tail_pos), attachment_axis=axis, span_length=span_length, total_mass=total_mass, internal_bonds=list(definition.get('bonds', [])), internal_angles=internal_angles, dihedrals_full=list(definition.get('dihedrals', [])), impropers_full=list(definition.get('impropers', [])), pairs=list(definition.get('pairs', [])), exclusions=list(definition.get('exclusions', [])), constraints=list(definition.get('constraints', [])), virtual_sites=list(definition.get('virtual_sites', [])), restraints=list(definition.get('restraints', [])), cmaptypes=list(definition.get('cmaptypes', [])), polarization=list(definition.get('polarization', [])), other_sections=dict(definition.get('other_sections', {})))`
- raises: `ValueError`, `FileNotFoundError`
- calls: `entry.get`, `_extract_single_definition`, `definition.get`, `read_gro_atoms`, `np.array`, `StrandTemplate`, `ValueError`, `os.path.isfile`, `FileNotFoundError`, `upper`, `name.startswith`, `np.linalg.norm`, +5 more

### `hygel_martini/hydrogel_builder/generator.py`

Thin workflow entry helper for hydrogel generation.

#### `run_hydrogel_builder(config_path: str | Path)` — line 10
- Run the hydrogel workflow from a maker YAML/JSON file.
- kind: function
- raises: `FileNotFoundError`
- calls: `expanduser`, `run_hydrogel_example`, `resolved_path.exists`, `FileNotFoundError`, `Path`, `resolved_path.resolve`

### `hygel_martini/hydrogel_builder/main_components/Attributes.py`

Primitive topology records used by the legacy builders.

#### class `Atom` — line 24
Representation of a single coarse-grained bead.

##### `__init__(self, source_template=None, source_index=None, source_residue_name=None)` — line 37
- *(no docstring)*
- kind: method
- calls: `np.array`, `append`

#### class `Bond` — line 118
Bond record linking two atoms and updating bond adjacency lists.

##### `__init__(self, i, j, **kwargs)` — line 129
- *(no docstring)*
- kind: method
- returns: `None` (bare return)
- effects: global registry, stdout
- calls: `World.Bonds.get`, `kwargs.get`, `bonded_atoms.append`, `append`, `join`, `print`

#### class `Network_bond` — line 199
Legacy record for network-only bonds distinct from normal bonds.

##### `__init__(self, i, j)` — line 206
- *(no docstring)*
- kind: method
- calls: `network_bonded_atoms.append`, `append`

#### class `Constraint` — line 240
Distance constraint between two atoms.

##### `__init__(self, i, j)` — line 247
- *(no docstring)*
- kind: method
- calls: `constrained_atoms.append`, `append`

#### class `Exclusion` — line 278
Exclusion entry removing a non-bonded interaction pair.

##### `__init__(self, i, j)` — line 285
- *(no docstring)*
- kind: method
- calls: `excluded_atoms.append`, `append`

#### class `Angle` — line 309
Angle interaction defined by three atoms.

##### `__init__(self, i, j, k)` — line 316
- *(no docstring)*
- kind: method
- calls: `angle_atoms.append`, `append`

#### class `Dihedral` — line 357
Dihedral or improper-like interaction defined by four atoms.

##### `__init__(self, i, j, m, n, c0=0)` — line 364
- *(no docstring)*
- kind: method
- calls: `append`

#### `initialize()` — line 13
- Reset global counters used to assign topology object identifiers.
- kind: function

### `hygel_martini/hydrogel_builder/main_components/Hydrogel.py`

Hydrogel network construction primitives.

#### class `Hydrogel` — line 30
Construct and enrich a hydrogel network stored in ``World``.

##### `__init__(self, x_number_of_repeat=6, y_number_of_repeat=6, z_number_of_repeat=6)` — line 44
- Initialize lattice repetition counts and terminal registries.
- kind: method

##### `make_lines(self, bx, by, bz)` — line 91
- Generate the line segments used to populate one lattice cell.
- kind: method
- returns: `(segment_xyz, link_xyz)`; `sign * magnitude * link_axis`
- effects: stdout
- calls: `print`, `n_segment.astype`, `np.array`, `lines.append`, `np.floor`, `segment_xyz.append`, `link_xyz.append`, `np.sqrt`, `axis_shift`, `interp3D`, `np.square`

##### `construct_atoms(self)` — line 182
- Materialize backbone and linker atoms from the geometric layout.
- kind: method
- returns: `create_generator`; `template_like.get(name, []) if isinstance(template_like, dict) else []`; `None` (+3 more)
- raises: `ValueError`
- effects: Config/runtime state, stdout
- calls: `pd`, `p.Config.get_param`, `backbone_definitions.get`, `get_backbone_generator_factory`, `block_settings.get`, `get_list`, `ValueError`, `p.resolve_block_copolymer_settings`, `print`, `sys.exit`, `append`, `idx_map.get`, +48 more

##### `_construct_proto_bonds(self, output_dir)` — line 642
- Attach linker stubs to the nearest allowed backbone terminals.
- kind: method, internal
- returns: `True`; `getattr(atom, 'pre_compress_position', atom.position)`; `best_atom`
- effects: stdout
- calls: `grouped.items`, `print`, `Config.debug_log`, `append`, `axis_index.get`, `np.max`, `np.any`, `hasattr`, `lower`, `_pick_backbone`, `best_atom.position.copy`, `Attributes.Bond`, +6 more

##### `construct_bonds(self, pbc, num_cell, output_dir)` — line 755
- Compatibility wrapper for proto-bond construction.
- kind: method
- returns: `self._construct_proto_bonds(output_dir)`
- calls: `self._construct_proto_bonds`

##### `construct_chemical_detail(self)` — line 765
- Expand backbone beads into detailed monomer side chains.
- kind: method
- returns: `None` (bare return)
- effects: Config/runtime state, stdout
- calls: `print`, `resolve_sidechain_placement_tuning`, `tqdm`, `monomer_counts.items`, `Config.debug_log`, `p.Config.get_param`, `load_monomer_templates`, `monomer_config.get`, `Config.get_runtime`, `sequence_generators.get`, `iterator.next`, `rij`, +39 more

##### `construct_angles(self)` — line 1077
- Generate angle terms from template metadata and fallback rules.
- kind: method
- effects: Config/runtime state, global registry, stdout
- calls: `World.Bonds.values`, `bonds_by_atom.items`, `print`, `p.Config.get_param`, `existing_angles.add`, `append`, `_atom.keys`, `get`, `Attributes.Angle`, `hasattr`, `reversed`

##### `construct_dihedrals(self)` — line 1230
- Generate internal dihedrals directly from template definitions.
- kind: method
- effects: global registry, stdout
- calls: `print`, `processed_templates.add`, `World.Atoms.values`, `id`, `Attributes.Dihedral`, `dihedral_def.get`

##### `construct_impropers(self)` — line 1284
- Placeholder for explicit improper handling.
- kind: method

### `hygel_martini/hydrogel_builder/main_components/Polymer.py`

Standalone polymer construction logic.

#### class `Polymer` — line 83
Construct a polymer chain and register it into ``World``.

##### `__init__(self, p_mon_num, p_length)` — line 126
- Store polymer dimensions and initialize the working box size.
- kind: method

##### `configure(cls, config: dict | None)` — line 153
- Cache template libraries and strategy iterators for polymer builds.
- kind: classmethod
- returns: `None` (bare return)
- effects: stdout
- calls: `config.get`, `require_unique`, `_build_polymer_bond_lookup`, `random.Random`, `StrategyRecord`, `TemplateStrategyIterator`, `cls._terminal_strategy.get`, `bb.get`, `load_monomer_templates`, `cls._sidechain_library.by_backbone.items`, `print`

##### `make_lines(self, random_seed)` — line 256
- Generate a reproducible straight-chain backbone path.
- kind: method
- returns: `interp3D(self.p_mon_num, pm_start_point, pm_last_point)`
- calls: `random`, `interp3D`, `np.array`, `Random`, `choice`

##### `construct_atoms(self, random_seed)` — line 297
- Dispatch to the template-driven or legacy atom-construction path.
- kind: method
- returns: `self._legacy_construct_atoms(random_seed)`; `self._construct_atoms_from_templates(random_seed)`
- calls: `self._legacy_construct_atoms`, `self._construct_atoms_from_templates`

##### `_legacy_construct_atoms(self, random_seed)` — line 312
- Construct a polymer using the historical single-backbone settings.
- kind: method, internal
- returns: `None` (bare return)
- effects: Config/runtime state, stdout
- calls: `self.make_lines`, `Attributes.Atom`, `p.Config.get_param`, `print`, `Attributes.Bond`

##### `_next_backbone_definition(self)` — line 366
- Return the next backbone template according to the configured strategy.
- kind: method, internal
- returns: `{}`; `template`; `self._backbone_defs[0]`
- calls: `self._backbone_iterator.next`

##### `_construct_atoms_from_templates(self, random_seed)` — line 381
- Construct backbone beads from template metadata.
- kind: method, internal
- calls: `self.make_lines`, `self._attach_terminals`, `self._next_backbone_definition`, `template.get`, `Attributes.Atom`, `definition.get`, `self._backbone_atom_ids.append`, `self._bond_lookup.get`, `Attributes.Bond`, `params.get`

##### `_select_terminal_templates(self)` — line 440
- Choose left and right terminal templates according to strategy.
- kind: method, internal
- returns: `(left_record.template if left_record else None, right_record.template if right_record else None)`; `(None, None)`; `self._terminal_random.choices(candidates, weights=weights, k=1)[0]`
- calls: `lower`, `pick`, `self._terminal_random.choices`, `self._terminal_strategy.get`

##### `_alignment_basis(self, axis)` — line 469
- Build an orthonormal basis whose x-axis follows ``axis``.
- kind: method, internal
- returns: `build_alignment_basis(axis)`
- calls: `build_alignment_basis`

##### `_place_template(self, template, origin, axis_vector)` — line 473
- Rotate and translate template coordinates onto an axis-aligned frame.
- kind: method, internal
- returns: `place_template_coords(template.coords, origin, axis_vector)`
- calls: `place_template_coords`

##### `_compute_template_positions(self, template, origin, normal_vector, tangent_vector)` — line 477
- Build a local side-chain frame from normal and tangent vectors.
- kind: method, internal
- returns: `compute_template_positions(template.coords, origin, normal_vector, tangent_vector)`
- calls: `compute_template_positions`

##### `_create_template_atoms(self, template, positions, residue_override=None)` — line 481
- Instantiate atoms for a placed template and return their IDs.
- kind: method, internal
- returns: `atom_ids`
- calls: `Attributes.Atom`, `atom_ids.append`

##### `_connect_template_bonds(self, template, created_atom_ids, backbone_atom_id)` — line 510
- Transfer template-local topology terms to global polymer indices.
- kind: method, internal
- returns: `None`; `0`; `depth + 1`
- raises: `ValueError`
- effects: stdout
- calls: `add`, `_add_template_edge`, `deque`, `append`, `Attributes.Bond`, `items`, `c.get`, `queue.popleft`, `template_graph.get`, `params.get`, `orig_to_global.get`, `_add_other`, +22 more

##### `_attach_terminals(self, World)` — line 726
- Attach terminal templates to the first and last backbone beads.
- kind: method, internal
- returns: `None` (bare return)
- calls: `self._select_terminal_templates`, `self._place_template`, `self._create_template_atoms`, `self._connect_template_bonds`, `np.array`

##### `_construct_sidechains_from_templates(self)` — line 762
- Attach polymer side-chain templates while avoiding local clashes.
- kind: method, internal
- returns: `None` (bare return)
- calls: `self._sidechain_iterators.get`, `iterator.next`, `not_self`, `rij`, `self._create_template_atoms`, `self._connect_template_bonds`, `bonded_atom_ids.add`, `np.linalg.norm`, `np.array`, `random_normal_vector`, `self._compute_template_positions`, `dij_sq`, +1 more

##### `construct_chemical_detail(self)` — line 856
- Expand the polymer backbone into a chemically detailed topology.
- kind: method
- returns: `self._construct_sidechains_from_templates()`
- effects: Config/runtime state, stdout
- calls: `p.Config.get_param`, `print`, `self._construct_sidechains_from_templates`, `not_self`, `Attributes.Atom`, `Attributes.Bond`, `normal_tetrahedral_vector`, `depth_2_atoms.remove`, `np.zeros`, `batom_positions.append`, `is_overlap`, `random_normal_vector`, +3 more

##### `construct_angles(self)` — line 1080
- Generate polymer angle terms from specific and default rules.
- kind: method
- effects: Config/runtime state
- calls: `angle_configs.get`, `np.array`, `pos.sort`, `near_cen_atom_ids.sort`, `p.Config.get_param`, `np.where`, `near_cen_atom_ids.append`, `Attributes.Angle`, `atom_types_in_angle.isdisjoint`

#### `_build_polymer_bond_lookup(bond_rules, fallback_length)` — line 44
- Create a fast lookup table for backbone-to-backbone bond parameters.
- kind: function, internal
- returns: `lookup`
- calls: `lookup.update`, `require_unique`, `rule.get`

### `hygel_martini/hydrogel_builder/main_components/Universe.py`

Global world-state container for hydrogel construction.

#### class `World` — line 62
Process-wide mutable container for topology and coordinate state.

##### `reset(cls)` — line 119
- Reset all geometry, counters, and topology registries.
- kind: classmethod
- effects: stdout
- calls: `np.array`, `collections.defaultdict`, `print`

##### `__init__(self)` — line 156
- *(no docstring)*
- kind: method
- effects: stdout
- calls: `print`

##### `make_hydrogel(self, fix_dna, nx=6, ny=6, nz=6)` — line 159
- Create and register a hydrogel object.
- kind: method
- effects: global registry
- calls: `World.hydrogels.append`, `World.Atoms.clear`, `World.Bonds.clear`, `World.Network_bonds.clear`, `World.Constraints.clear`, `World.Exclusions.clear`, `World.Angles.clear`, `World.Dihedrals.clear`, `Hydrogel`

##### `make_polymer(self, p_mon_num, p_length)` — line 190
- Create and register a standalone polymer object.
- kind: method
- effects: global registry
- calls: `World.polymers.append`, `Polymer`

##### `update_hydrogel_attributes(self, hydrogel)` — line 201
- Copy aggregate hydrogel counters onto the world container.
- kind: method
- effects: stdout
- calls: `print`

##### `update_polymer_attributes(self, polymer)` — line 209
- Copy aggregate polymer counters onto the world container.
- kind: method
- effects: stdout
- calls: `print`

#### `initialize_world(segment_length_from_config, mean_sep_from_config, max_linker_span_from_config=0.0)` — line 12
- Initialize global geometry parameters from configuration.
- kind: function
- returns: `None` (bare return)
- raises: `ValueError`
- effects: stdout
- calls: `np.array`, `np.roots`, `print`, `np.square`, `ValueError`, `np.isreal`

### `hygel_martini/hydrogel_builder/main_components/__init__.py`

*(no module docstring)*

### `hygel_martini/hydrogel_builder/relax/__init__.py`

Relaxation workflows that run after hydrogel_builder system construction.

#### `run_relax_workflow(*args, **kwargs)` — line 17
- Lazily import and invoke ``generator.run_relax_workflow``.
- kind: function
- returns: `_run_relax_workflow(*args, **kwargs)`
- calls: `_run_relax_workflow`

### `hygel_martini/hydrogel_builder/relax/__main__.py`

Module runner: ``python -m hygel_martini.hydrogel_builder.relax`` -> cli.main.

### `hygel_martini/hydrogel_builder/relax/cli.py`

Command-line entry point for the post-build relaxation workflow.

#### `main()` — line 17
- Parse the config location and run one relaxation stage.
- kind: function, CLI entry
- calls: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `run_relax_workflow`, `Path`, `parser.exit`

### `hygel_martini/hydrogel_builder/relax/config.py`

Config loading for the relax workflows (separate from core.config).

#### `_deep_merge(base: Dict[str, Any], incoming: Dict[str, Any])` — line 45
- Merge ``incoming`` into ``base`` in place (nested dicts recursively).
- kind: function, internal
- returns: `base`
- calls: `incoming.items`, `_deep_merge`, `copy.deepcopy`, `base.get`

#### `_load_yaml_file(path: Path)` — line 58
- Load one YAML file whose root must be a mapping (empty -> {}).
- kind: function, internal
- returns: `data`
- raises: `TypeError`, `ImportError`
- effects: filesystem
- calls: `path.open`, `TypeError`, `ImportError`, `yaml.safe_load`

#### `_load_with_includes(path: Path, seen: set[Path] | None=None)` — line 77
- Load a config file, expanding YAML ``includes:`` recursively.
- kind: function, internal
- returns: `data`; `merged`
- raises: `ValueError`, `TypeError`
- effects: filesystem
- calls: `path.resolve`, `seen.add`, `ValueError`, `path.suffix.lower`, `_load_yaml_file`, `_deep_merge`, `resolved.open`, `json.load`, `TypeError`, `data.pop`, `Path`, `inc_path.is_absolute`, +1 more

#### `_build_context(config_path: Path)` — line 120
- Substitution context: CONFIG_DIR (config file dir) and REPO_ROOT.
- kind: function, internal
- returns: `{'CONFIG_DIR': config_dir, 'REPO_ROOT': repo_root}`
- calls: `config_path.resolve`, `resolve`, `Path`

#### `_looks_like_path_key(key: str | None)` — line 131
- Decide by key name whether a string value should be path-resolved.
- kind: function, internal
- returns: `key.endswith(_PATH_SUFFIXES)`; `False`; `True`
- calls: `key.endswith`

#### `_resolve_path_value(value: str, context: Dict[str, str])` — line 140
- Expand env vars, ~, and ${TOKEN}s, then absolutize vs CONFIG_DIR.
- kind: function, internal
- returns: `expanded`
- calls: `os.path.expanduser`, `context.items`, `os.path.expandvars`, `expanded.replace`, `os.path.isabs`, `os.path.abspath`, `os.path.join`

#### `_normalize_tree(node: Any, context: Dict[str, str], parent_key: str | None=None)` — line 150
- Recursively normalize a config tree (see module docstring).
- kind: function, internal
- returns: `node`; `{key: _normalize_tree(value, context, key) for key, value in node.items()}`; `[_normalize_tree(item, context) for item in node]` (+2 more)
- calls: `_looks_like_path_key`, `_resolve_path_value`, `_normalize_tree`, `node.items`

#### `load_relax_config(config_path: str | Path)` — line 170
- Load, include-merge, and path-normalize a relax config file.
- kind: function
- returns: `_normalize_tree(data, context)`
- raises: `FileNotFoundError`
- calls: `resolve`, `_build_context`, `_load_with_includes`, `_normalize_tree`, `path.exists`, `FileNotFoundError`, `expanduser`, `Path`

### `hygel_martini/hydrogel_builder/relax/generator.py`

Thin workflow entry helper for post-build hydrogel relaxation runs.

#### `run_relax_workflow(config_path: str | Path)` — line 21
- Run one relaxation stage selected by ``workflow.mode`` in the config.
- kind: function
- returns: `result`
- raises: `FileNotFoundError`, `ValueError`
- effects: stdout
- calls: `expanduser`, `load_relax_config`, `cfg.get`, `lower`, `print`, `resolved_path.exists`, `FileNotFoundError`, `run_soft_em`, `Path`, `strip`, `run_soft_md`, `resolved_path.resolve`, +3 more

### `hygel_martini/hydrogel_builder/relax/hard_em_shrink.py`

Guarded, fixed-increment box compression for hydrated network structures.

#### `_finite_gro(path: Path)` — line 42
- Check a .gro file is structurally sane with finite numbers.
- kind: function, internal
- returns: `len(box) == 3 and all((math.isfinite(value) and value > 0.0 for value in box))`; `False`
- calls: `splitlines`, `strip`, `path.read_text`, `split`, `math.isfinite`

#### `_energy_is_finite(gmx: str, edr: Path, out_xvg: Path, env: Dict[str, str])` — line 64
- Extract the last Potential (kJ/mol) from an .edr and check finiteness.
- kind: function, internal
- returns: `(math.isfinite(potential), potential)`; `(False, None)`
- calls: `_extract_xvg`, `_read_xvg_rows`, `math.isfinite`

#### `_mdp_values(path: 'Path', keys=('dt', 'nsteps', 'ref_t', 'ref-t', 'constraints'))` — line 79
- The handful of mdp settings that describe a recovery run.
- kind: function, internal
- returns: `out`
- calls: `k.replace`, `splitlines`, `strip`, `text.partition`, `replace`, `read_text`, `value.strip`, `lower`, `Path`, `line.split`, `key.strip`, `value.lower`

#### `_run_nvt_recovery(*, gmx: str, mdp: Path, gro: Path, top: Path, outdir: Path, ntomp: int, maxwarn: int, env: Dict[str, str], gpu_id: Optional[str], mpi_np: Optional[int], mpi_args: List[str], mdrun_args: List[str])` — line 107
- Run a short NVT recovery MD to relax a structure that failed EM guards.
- kind: function, internal
- returns: `output`
- raises: `RuntimeError`
- calls: `_ensure_dir`, `_run`, `_build_mdrun_cmd`, `run_env.update`, `deffnm.with_suffix`, `RuntimeError`, `output.exists`, `_finite_gro`

#### `_target_box(config: Dict[str, Any])` — line 156
- Normalize target_box_nm (scalar or 3-list, in nm) to an (x, y, z) tuple.
- kind: function, internal
- returns: `(float(target), float(target), float(target))`; `tuple((float(value) for value in target))`
- raises: `ValueError`
- calls: `config.get`, `ValueError`

#### `run_hard_em_shrink(cfg: Dict[str, Any])` — line 170
- Compress the box stepwise to the target, guarding every step with EM.
- kind: function
- returns: `final`; `(valid, em_gro, fmax, potential, None)`; `(False, None, None, None, str(exc))`
- raises: `RuntimeError`, `ValueError`, `FileNotFoundError`
- effects: filesystem, stdout
- calls: `cfg.get`, `resolve`, `_mdp_values`, `_target_box`, `runtime.get`, `os.environ.copy`, `_parse_gro_box`, `RuntimeError`, `tools.get`, `shrink.get`, `ValueError`, `_finite_gro`, +27 more

### `hygel_martini/hydrogel_builder/relax/soft_em.py`

Soft-EM stage of the post-build relaxation (workflow.mode=soft_em).

#### `_run(cmd: List[str], *, cwd: Path | None=None, env: Dict[str, str] | None=None, input_str: str | None=None, check: bool=True)` — line 38
- Run a command with merged stdout/stderr captured as text.
- kind: function, internal
- returns: `process`
- raises: `RuntimeError`
- effects: subprocess
- calls: `subprocess.run`, `RuntimeError`, `join`

#### `_ensure_dir(path: Path)` — line 77
- Create the directory (and parents) if needed.
- kind: function, internal
- effects: filesystem
- calls: `path.mkdir`

#### `_clamp(value: float, low: float, high: float)` — line 82
- Clamp value into [low, high].
- kind: function, internal
- returns: `max(low, min(high, value))`

#### `_parse_gro_box(gro_path: Path)` — line 87
- Read the box diagonal (nm) from the last line of a .gro file.
- kind: function, internal
- returns: `tuple(map(float, last[:3]))`
- raises: `ValueError`
- calls: `split`, `ValueError`, `splitlines`, `strip`, `gro_path.read_text`

#### `_parse_em_fmax(log_path: Path)` — line 99
- Parse the final maximum force (kJ/mol/nm) from an EM log.
- kind: function, internal
- returns: `float(match.group(1))`
- raises: `RuntimeError`
- calls: `log_path.read_text`, `re.search`, `RuntimeError`, `match.group`

#### `_find_energy_indices(gmx_cmd: str, edr_file: Path, wanted_names: List[str], env: Dict[str, str])` — line 117
- Map energy-term names to their ``gmx energy`` menu indices.
- kind: function, internal
- returns: `[idx_map[name] for name in wanted_names]`
- raises: `RuntimeError`
- calls: `_run`, `probe.stdout.splitlines`, `re.finditer`, `join`, `RuntimeError`, `strip`, `match.group`, `re.search`

#### `_extract_xvg(gmx_cmd: str, edr_file: Path, out_xvg: Path, terms: List[str], env: Dict[str, str])` — line 157
- Extract the given energy terms from an .edr into a plain .xvg.
- kind: function, internal
- calls: `_find_energy_indices`, `_run`, `join`

#### `_read_xvg_rows(path: Path)` — line 179
- Parse numeric rows from an .xvg (skipping #/@ lines and bad rows).
- kind: function, internal
- returns: `rows`
- raises: `RuntimeError`
- calls: `splitlines`, `line.strip`, `RuntimeError`, `path.read_text`, `stripped.startswith`, `rows.append`, `stripped.split`

#### `_summarize_series(path: Path, column: int, mode: str)` — line 199
- Reduce one .xvg column to a scalar: mode "last" or mean (default).
- kind: function, internal
- returns: `mean((row[column] for row in rows))`; `rows[-1][column]`
- calls: `_read_xvg_rows`, `mean`

#### `_split_comment(line: str)` — line 207
- Split an .itp line into (code, ";comment") preserving the comment.
- kind: function, internal
- returns: `(line.rstrip(), '')`; `(code.rstrip(), ';' + comment)`
- calls: `line.split`, `line.rstrip`, `code.rstrip`

#### `_is_int_token(token: str)` — line 215
- True when the token parses as int.
- kind: function, internal
- returns: `True`; `False`

#### `_is_float_token(token: str)` — line 224
- True when the token parses as float.
- kind: function, internal
- returns: `True`; `False`

#### `scale_itp_bonded(in_itp: Path, out_itp: Path, factor: float)` — line 233
- Write a copy of an .itp with bonded force constants multiplied.
- kind: function
- effects: filesystem
- calls: `splitlines`, `out_itp.write_text`, `raw.rstrip`, `SECTION_RE.match`, `_split_comment`, `code.strip`, `stripped.split`, `group`, `join`, `in_itp.read_text`, `line.strip`, `lower`, +6 more

#### `patch_system_top(in_top: Path, out_top: Path, bonded_itp_basename: str, new_local_itp_name: str)` — line 313
- Copy a .top, redirecting the bonded include to a local scaled .itp.
- kind: function
- effects: filesystem
- calls: `re.compile`, `splitlines`, `out_top.write_text`, `raw.rstrip`, `pattern.match`, `output.append`, `join`, `in_top.read_text`, `line.strip`, `match.group`, `raw.endswith`, `Path`

#### `_build_mdrun_cmd(gmx: str, base_args: List[str], gpu_id: Optional[str], mpi_np: Optional[int], mpi_args: List[str], extra: List[str])` — line 336
- Build the full mdrun command and required env overrides.
- kind: function, internal
- returns: `(mdrun + extra, env_extra)`; `(['mpirun', '-np', str(mpi_np)] + mpi_args + mdrun + extra, env_extra)`

#### `_compute_box_deltas(pxx: float, pyy: float, pzz: float, box_x: float, box_y: float, box_z: float, p_target: float, scale_factor: float, max_dlen: float, box_mode: str, cubic_rate: float, cubic_max_dlen: float)` — line 368
- Compute (d_lx, d_ly, d_lz) — fractional length change for each axis.
- kind: function, internal
- returns: `(_clamp(scale_factor * (pxx - p_target), -max_dlen, max_dlen), _clamp(scale_factor * (pyy - p_target), -max_dlen, max_dlen), _clamp(scale_factor * (pzz - p_target), -max_dlen, max_dlen))`; `(d, d, d)`; `(_clamp(d_pressure + d_sx, -max_dlen, max_dlen), _clamp(d_pressure + d_sy, -max_dlen, max_dlen), _clamp(d_pressure + d_sz, -max_dlen, max_dlen))`
- calls: `_clamp`

#### `_wrap_pbc_gro(gro_path: Path, out_path: Path)` — line 425
- Wrap atom coordinates into the simulation box using modulo arithmetic.
- kind: function, internal
- effects: filesystem
- calls: `splitlines`, `out_lines.append`, `out_path.write_text`, `strip`, `gro_path.read_text`, `join`, `split`

#### `_grompp_and_run_em(gmx_cmd: str, mdp: Path, gro: Path, top: Path, outdir: Path, ntomp: int, maxwarn: int, env: Dict[str, str], gpu_id: Optional[str]=None, mpi_np: Optional[int]=None, mpi_args: Optional[List[str]]=None, mdrun_args: Optional[List[str]]=None)` — line 453
- Run one grompp + EM mdrun pass in ``outdir`` (deffnm "em").
- kind: function, internal
- returns: `(tpr, edr, log, gro_out)`
- raises: `RuntimeError`
- calls: `_ensure_dir`, `_build_mdrun_cmd`, `run_env.update`, `_run`, `path.exists`, `RuntimeError`

#### `run_soft_em(cfg: Dict[str, Any])` — line 499
- Run the iterative soft-EM loop until converged (see module docstring).
- kind: function
- returns: `final_path`
- raises: `RuntimeError`, `ValueError`, `FileNotFoundError`
- effects: filesystem, stdout
- calls: `cfg.get`, `resolve`, `runtime.get`, `os.environ.copy`, `print`, `RuntimeError`, `tools.get`, `soft_em.get`, `ValueError`, `workdir.exists`, `_ensure_dir`, `shutil.copy2`, +20 more

### `hygel_martini/hydrogel_builder/relax/soft_md.py`

Settling MD stage of the post-build relaxation (workflow.mode=soft_md).

#### `_run(cmd: List[str], *, cwd: Path, env: Dict[str, str])` — line 21
- Run a command with merged stdout/stderr; echo output on success.
- kind: function, internal
- raises: `RuntimeError`
- effects: subprocess, stdout
- calls: `subprocess.run`, `RuntimeError`, `print`, `process.stdout.rstrip`, `join`

#### `_string_list(value: Iterable[Any])` — line 46
- Stringify config list entries for use as CLI arguments.
- kind: function, internal
- returns: `[str(item) for item in value]`

#### `_build_mdrun_cmd(gmx: str, base_args: List[str], gpu_id: Optional[str], mpi_np: Optional[int], mpi_args: List[str], extra: List[str])` — line 51
- Build the full mdrun command and required env overrides.
- kind: function, internal
- returns: `(mdrun + extra, env_extra)`; `(['mpirun', '-np', str(mpi_np)] + mpi_args + mdrun + extra, env_extra)`

#### `run_soft_md(cfg: Dict[str, Any])` — line 83
- Run one settling MD (grompp + mdrun) in the configured workdir.
- kind: function
- returns: `workdir / f'{deffnm}.gro'`
- raises: `FileNotFoundError`
- effects: filesystem, stdout
- calls: `cfg.get`, `resolve`, `workdir.mkdir`, `runtime.get`, `os.environ.copy`, `grompp_cmd.extend`, `print`, `_run`, `_build_mdrun_cmd`, `env.update`, `tools.get`, `soft_md.get`, +5 more

## `hygel_martini/param_opt`

### `hygel_martini/param_opt/__init__.py`

Explicit workflow packages for 01/02/03 polymer parameter generation.

#### `main()` — line 4
- *(no docstring)*
- kind: function, CLI entry
- calls: `_main`

### `hygel_martini/param_opt/__main__.py`

*(no module docstring)*

### `hygel_martini/param_opt/bead_generator/__init__.py`

Planned workflow: bead generation / bead assignment.

### `hygel_martini/param_opt/bead_generator/__main__.py`

*(no module docstring)*

### `hygel_martini/param_opt/bead_generator/cli.py`

*(no module docstring)*

#### `main()` — line 6
- *(no docstring)*
- kind: function, CLI entry
- raises: `NotImplementedError`
- calls: `argparse.ArgumentParser`, `parser.parse_args`, `NotImplementedError`

### `hygel_martini/param_opt/cli.py`

*(no module docstring)*

#### `main()` — line 4
- *(no docstring)*
- kind: function, CLI entry
- raises: `SystemExit`
- calls: `SystemExit`

### `hygel_martini/param_opt/core/__init__.py`

*(no module docstring)*

### `hygel_martini/param_opt/opls_to_martini/__init__.py`

Stage 02 workflow package: OPLS/GROMACS data to Martini/Bartender fitting.

### `hygel_martini/param_opt/opls_to_martini/__main__.py`

Module runner: ``python -m param_opt.opls_to_martini`` delegates to cli.main.

### `hygel_martini/param_opt/opls_to_martini/builder.py`

Constructor-mode case builder for the stage-02 workflow.

#### `_sequence_stem(tokens: List[str])` — line 42
- Name a sequence: concatenate 1-letter tokens, else join with '_'.
- kind: function, internal
- returns: `'_'.join(tokens)`; `''.join(tokens)`
- calls: `join`

#### `_parse_sequence_entry(entry: Any, monomer_keys: set[str])` — line 49
- Normalize one ``system.sequences`` entry into a token list.
- kind: function, internal
- returns: `tokens`
- raises: `ValueError`, `TypeError`
- calls: `entry.strip`, `ValueError`, `parse_csv_list`, `TypeError`, `strip`, `text.split`, `type`

#### `_build_sequence_jobs(system_cfg: Dict[str, Any], monomer_keys: set[str])` — line 83
- Resolve the list of sequences (token lists) to build.
- kind: function, internal
- returns: `jobs`
- raises: `ValueError`
- calls: `system_cfg.get`, `ValueError`, `_parse_sequence_entry`, `jobs.append`

#### `build_cases(cfg: Dict[str, Any])` — line 112
- Generate all constructor cases and write out_root/summary.json.
- kind: function
- returns: `result`
- raises: `ValueError`, `KeyError`
- effects: filesystem
- calls: `Path`, `out_root.mkdir`, `resolve`, `load_monomer_library`, `_build_sequence_jobs`, `water_density_g_cm3`, `write_text`, `_sequence_stem`, `case_dir.mkdir`, `build_polymer`, `read`, `atoms.positions.min`, +21 more

### `hygel_martini/param_opt/opls_to_martini/cli.py`

Command-line entry point for the stage-02 (OPLS -> Martini) workflow.

#### `build_arg_parser()` — line 22
- Build the argparse parser: shared config args plus 02-only flags.
- kind: function
- returns: `parser`
- calls: `argparse.ArgumentParser`, `add_opls_to_martini_cli_args`, `parser.add_argument`

#### `main()` — line 46
- Run the opls_to_martini CLI.
- kind: function, CLI entry
- returns: `None` (bare return)
- raises: `ValueError`, `SystemExit`
- effects: filesystem, stdout
- calls: `build_arg_parser`, `parser.parse_args`, `run_opls_to_martini`, `Path`, `print`, `write_text`, `result.get`, `get`, `ValueError`, `json.dumps`, `SystemExit`, `output.get`

### `hygel_martini/param_opt/opls_to_martini/defaults.py`

Default configuration for the stage-02 (OPLS -> Martini) workflow.

### `hygel_martini/param_opt/opls_to_martini/fitting.py`

Existing-data fitting for stage 02: trim + Bartender refit job setup.

#### `_bool(value: Any, default: bool=False)` — line 33
- Coerce config values to bool ("1"/"true"/"yes"/"on" are truthy strings).
- kind: function, internal
- returns: `str(value).strip().lower() in {'1', 'true', 'yes', 'on'}`; `default`; `value` (+1 more)
- calls: `lower`, `strip`

#### `_as_path(base_dir: Path, value: Any, *, required: bool=True)` — line 44
- Resolve a config path against base_dir (absolute paths pass through).
- kind: function, internal
- returns: `(base_dir / path).resolve()`; `None`; `path`
- raises: `ValueError`
- calls: `Path`, `path.is_absolute`, `resolve`, `strip`, `ValueError`

#### `_q(value: str | Path)` — line 62
- Shell-quote a value for the generated bash scripts.
- kind: function, internal
- returns: `shlex.quote(str(value))`
- calls: `shlex.quote`

#### `_rel(path: Path, start: Path)` — line 67
- Relative path from start to path (scripts cd into their own dir).
- kind: function, internal
- returns: `os.path.relpath(str(path), start=str(start))`
- calls: `os.path.relpath`

#### `_repo_root()` — line 72
- Installable package root (3 levels up), exported as PYTHONPATH.
- kind: function, internal
- returns: `Path(__file__).resolve().parents[3]`
- calls: `resolve`, `Path`

#### `_merge_case_variant(case: Dict[str, Any], variant: Dict[str, Any])` — line 77
- Overlay one variant dict on its parent case (shallow; drops "variants").
- kind: function, internal
- returns: `merged`
- calls: `merged.update`, `merged.pop`

#### `_iter_case_variants(cases: Iterable[Dict[str, Any]])` — line 85
- Expand opls_data.cases: yield each case, or one merged dict per variant.
- kind: function, internal
- returns: `<generator>`
- raises: `TypeError`
- calls: `case.get`, `TypeError`, `_merge_case_variant`

#### `_mode_tag(case: Dict[str, Any])` — line 102
- Middle directory-name component: mode_tag > mode > name > "default".
- kind: function, internal
- returns: `str(case.get('mode_tag') or case.get('mode') or case.get('name') or 'default').strip()`
- calls: `strip`, `case.get`

#### `_label(case: Dict[str, Any])` — line 107
- Case label used for directory names: label > sequence > "CASE".
- kind: function, internal
- returns: `str(case.get('label') or case.get('sequence') or 'CASE').strip()`
- calls: `strip`, `case.get`

#### `_case_dir(out_root: Path, case: Dict[str, Any])` — line 112
- Case directory layout: <out_root>/<label>/<mode_tag>/<label>.
- kind: function, internal
- returns: `out_root / _label(case) / _mode_tag(case) / _label(case)`
- calls: `_label`, `_mode_tag`

#### `_resolve_md_mode(cfg: Dict[str, Any])` — line 117
- Normalize bartender_pipeline.md to one of md / md_notrim / trim / off.
- kind: function, internal
- returns: `mode`
- raises: `ValueError`
- calls: `cfg.get`, `lower`, `aliases.get`, `ValueError`, `strip`, `pipeline.get`

#### `_apply_execution_preset(cfg: Dict[str, Any])` — line 142
- Apply the opls_data.execution.mode preset onto the config in place.
- kind: function, internal
- returns: `raw_mode`; `''`
- raises: `TypeError`, `ValueError`
- calls: `cfg.setdefault`, `data_cfg.setdefault`, `replace`, `pipeline.setdefault`, `TypeError`, `join`, `ValueError`, `lower`, `strip`, `execution.get`

#### `_resolve_tool(cfg: Dict[str, Any], name: str, default: str)` — line 240
- Look up a tool command in cfg["tools"], falling back to default.
- kind: function, internal
- returns: `str(tools.get(name) or default)`
- calls: `cfg.get`, `tools.get`

#### `check_existing_data_tools(cfg: Dict[str, Any], tools: Iterable[str])` — line 248
- Probe the configured gmx / bartender commands without running them.
- kind: function
- returns: `{'ok': all((item['exists'] for item in checks)), 'tools': checks}`
- effects: filesystem
- calls: `resolve`, `cfg.get`, `_resolve_tool`, `pipeline.get`, `bartender_cfg.get`, `Path`, `checks.append`, `shutil.which`, `path.is_absolute`, `path.exists`, `exists`

#### `_prepare_script_lines(cfg: Dict[str, Any], case: Dict[str, Any], trim_dir: Path, base_dir: Path, md_mode: str)` — line 281
- Assemble the bash lines of trim/run_prepare_md.sh for one case.
- kind: function, internal
- returns: `(lines, output_pdb)`
- raises: `ValueError`
- calls: `cfg.get`, `_as_path`, `strip`, `_bool`, `trajectory.suffix.lower`, `lines.append`, `data_cfg.get`, `case.get`, `tools_cfg.get`, `trim_cfg.get`, `lines.extend`, `ValueError`, +4 more

#### `_write_prepare_md_job(cfg: Dict[str, Any], case: Dict[str, Any], case_dir: Path, base_dir: Path, md_mode: str)` — line 401
- Write trim/run_prepare_md.sh for one case (skipped when md_mode=off).
- kind: function, internal
- returns: `output_pdb`; `None`
- effects: filesystem
- calls: `trim_dir.mkdir`, `_prepare_script_lines`, `write_text`, `script.chmod`, `join`

#### `_write_bartender_job(cfg: Dict[str, Any], case: Dict[str, Any], case_dir: Path, base_dir: Path, md_mode: str, trajectory_pdb: Path | None)` — line 424
- Write the per-case Bartender refit job directory and script.
- kind: function, internal
- returns: `outdir`; `None`
- raises: `ValueError`
- effects: filesystem
- calls: `cfg.get`, `_as_path`, `outdir.mkdir`, `local_inp.write_text`, `strip`, `_rel`, `lines.append`, `write_text`, `script.chmod`, `pipeline.get`, `_bool`, `ValueError`, +7 more

#### `_run_script(path: Path)` — line 516
- Execute a generated bash script in its own directory (check=True).
- kind: function, internal
- effects: subprocess
- calls: `subprocess.run`

#### `run_existing_data_fit(cfg: Dict[str, Any])` — line 521
- Prepare (and optionally run) trim + Bartender jobs for every case.
- kind: function
- returns: `result`
- raises: `TypeError`, `ValueError`
- effects: filesystem, global registry
- calls: `resolve`, `out_root.mkdir`, `cfg.get`, `_apply_execution_preset`, `_resolve_md_mode`, `_bool`, `_iter_case_variants`, `write_text`, `run_all.chmod`, `TypeError`, `data_cfg.get`, `ValueError`, +16 more

#### `run_postprocess_only(cfg: Dict[str, Any])` — line 615
- Run only the Bartender screening postprocess on existing results.
- kind: function
- returns: `summary`
- raises: `ValueError`
- effects: filesystem
- calls: `get`, `resolve`, `summary_root.mkdir`, `write_text`, `ValueError`, `run_screening_postprocess`, `json.dumps`, `cfg.get`, `Path`, `post_cfg.get`

### `hygel_martini/param_opt/opls_to_martini/generator.py`

Thin workflow entry helper for OPLS-to-Martini case generation.

#### `run_opls_to_martini(config_path: str | Path, overrides: argparse.Namespace | None=None)` — line 29
- Load an opls_to_martini config, apply overrides, and run one branch.
- kind: function
- returns: `(cfg, result)`; `(cfg, run_postprocess_only(cfg))`; `(cfg, check_existing_data_tools(cfg, check_tools))` (+1 more)
- raises: `ValueError`
- calls: `load_config`, `cfg.get`, `lower`, `build_cases`, `Path`, `apply_cli_overrides`, `ValueError`, `run_postprocess_only`, `check_tools.append`, `check_existing_data_tools`, `strip`, `run_existing_data_fit`, +1 more

### `hygel_martini/param_opt/opls_to_martini/gromacs_traj_to_pdb.py`

Equilibration trimming of GROMACS multi-model PDB trajectories.

#### `_parse_xvg(path: Path)` — line 45
- Parse a GROMACS .xvg into (times, values) lists.
- kind: function, internal
- returns: `(times, values)`
- effects: filesystem
- calls: `path.exists`, `path.open`, `line.strip`, `stripped.split`, `stripped.startswith`, `times.append`, `values.append`

#### `_iter_pdb_frames(path: Path)` — line 72
- Yield PDB frames as raw line lists, split on MODEL/ENDMDL.
- kind: function, internal
- returns: `<generator>`
- effects: filesystem
- calls: `path.open`, `line.startswith`, `current.append`

#### `_detect_t0_pymbar(values: List[float], nskip: int, fast: bool)` — line 103
- Detect equilibration via pymbar timeseries.detect_equilibration.
- kind: function, internal
- returns: `(int(t0), float(g), float(neff))`; `(0, 1.0, float(len(values)))`
- calls: `np.asarray`, `timeseries.detect_equilibration`

#### `_detect_t0_energy_threshold(values: List[float], ref_fraction: float, threshold_sigma: float)` — line 118
- Detect t0 as the first frame whose tail mean matches the reference.
- kind: function, internal
- returns: `int(candidates[0]) if len(candidates) else 0`; `0`
- calls: `np.asarray`, `np.arange`, `np.mean`, `np.cumsum`, `np.where`, `np.std`, `np.abs`

#### `_write_plots(values: List[float], t0: int, start_index: int, out_pdb: Path)` — line 148
- Save an energy trace + cumulative-mean PNG next to the output PDB.
- kind: function, internal
- returns: `None` (bare return)
- calls: `np.asarray`, `np.arange`, `plt.subplots`, `plot`, `axvline`, `set_ylabel`, `legend`, `set_xlabel`, `plt.tight_layout`, `fig.savefig`, `plt.close`, `np.cumsum`, +1 more

#### `trim_pdb(input_pdb: Path, output_pdb: Path, *, energy_xvg: Path | None, auto_trim: bool, skip_frames: int, nskip: int, max_trim_fraction: float, trim_method: str, ref_fraction: float, threshold_sigma: float, fast: bool, write_plots: bool)` — line 178
- Trim leading equilibration frames from a multi-model PDB.
- kind: function
- returns: `info`
- effects: filesystem
- calls: `output_pdb.parent.mkdir`, `output_pdb.with_name`, `info_path.write_text`, `_iter_pdb_frames`, `_parse_xvg`, `output_pdb.open`, `json.dumps`, `_write_plots`, `_detect_t0_energy_threshold`, `_detect_t0_pymbar`, `handle.writelines`

#### `main()` — line 270
- CLI wrapper around :func:`trim_pdb`; prints trim info to stderr.
- kind: function, CLI entry
- effects: stdout
- calls: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `trim_pdb`, `print`, `Path`, `json.dumps`

### `hygel_martini/param_opt/opls_to_martini/writers.py`

File writers for stage-02 constructor cases.

#### `write_text(path: Path, text: str)` — line 16
- Write UTF-8 text to path, creating parent directories as needed.
- kind: function
- effects: filesystem
- calls: `path.parent.mkdir`, `path.write_text`

#### `write_packmol_input(path: Path, polymer_xyz: Path, output_xyz: str, box_ang: Sequence[float], n_waters: int, seed: int, cfg: Dict[str, Any])` — line 22
- Write a packmol input placing one fixed polymer plus n_waters waters.
- kind: function
- effects: filesystem
- calls: `write_text`, `Path`, `polymer_ref.as_posix`

#### `write_gromacs_mdp_templates(case_dir: Path, cfg: Dict[str, Any])` — line 66
- Write em/nvt/npt/md .mdp files under ``<case_dir>/mdp/``.
- kind: function
- effects: filesystem
- calls: `write_text`

#### `write_topol_stub(path: Path, cfg: Dict[str, Any])` — line 156
- Write a topol.top stub with includes and a [ molecules ] block.
- kind: function
- effects: filesystem
- calls: `top_cfg.get`, `include_lines.append`, `get`, `join`, `write_text`, `molecules_lines.append`, `cfg.get`

#### `write_pipeline_script(replica_dir: Path, box_nm: Sequence[float], cfg: Dict[str, Any])` — line 197
- Write the executable per-replica ``run_pipeline.sh`` (chmod 755).
- kind: function
- effects: filesystem
- calls: `write_text`, `chmod`

### `hygel_martini/param_opt/polymer_maker/__init__.py`

*(no module docstring)*

### `hygel_martini/param_opt/polymer_maker/maker.py`

*(no module docstring)*

#### `get_connection_info(atoms)` — line 15
- 모노머에서 연결 정보를 추출합니다. return: c0_idx: Head Carbon index (0) c1_idx: Tail Carbon index (1) bc_head_idx: C0에 연결된 BASE_CONNECTOR 인덱스 bc_tail_idx: C1에 연결된 BASE_CONNECTOR 인덱스
- kind: function
- returns: `(c0_idx, c1_idx, bc_head_idx, bc_tail_idx)`
- raises: `ValueError`
- calls: `ValueError`, `atoms.get_distance`, `atoms.info.get`

#### `cap_ends_with_hydrogen(atoms)` — line 47
- 양 끝단에 남아있는 BASE_CONNECTOR을 H로 치환하고 길이를 1.094A로 조정합니다.
- kind: function
- returns: `atoms`
- calls: `atoms.get_distance`, `np.linalg.norm`

#### `normalize_sequence(sequence)` — line 80
- 시퀀스 입력을 토큰 리스트로 정규화합니다. - ["S", "D", "B"] 형태를 권장 - "S,D,B" / "S D B" 문자열도 허용 - "SDB" 문자열은 한 글자 토큰으로 처리
- kind: function
- returns: `tokens`
- raises: `ValueError`
- calls: `sequence.strip`, `ValueError`, `strip`, `tok.strip`, `seq.split`

#### `load_monomer_library(monomer_files, base_dir=None)` — line 103
- monomer_files: {"S": "NEW_SBMA.xyz", "D": "NEW_DMAPS.xyz"} 또는 {"S": {"xyz": "NEW_SBMA.xyz"}, ...} 형태
- kind: function
- returns: `monomer_dict`
- raises: `ValueError`, `FileNotFoundError`
- calls: `monomer_files.items`, `Path`, `read`, `resolve`, `raw_entry.get`, `ValueError`, `path_obj.is_absolute`, `path_obj.exists`, `FileNotFoundError`

#### `_sequence_output_stem(sequence_tokens)` — line 132
- *(no docstring)*
- kind: function, internal
- returns: `'_'.join(sequence_tokens)`; `''.join(sequence_tokens)`
- calls: `join`

#### `build_polymer(sequence, monomer_dict, n_torsion, output_filename=None, output_dir='output')` — line 138
- Args: sequence: ["S", "D", "D", "S", "B"] 또는 "S,D,D,S,B" monomer_dict: {"S": ase.Atoms(...), "D": ase.Atoms(...)} 형태 n_torsion: 비틀림 각도를 결정하는 정수 (1, 2, 3, 4...) output_filename: 저장할 파일명 output_dir: 생성된 xyz를 저장할 디렉터리
- kind: function
- raises: `KeyError`
- effects: filesystem, stdout
- calls: `normalize_sequence`, `print`, `copy`, `get_connection_info`, `cap_ends_with_hydrogen`, `Path`, `output_root.mkdir`, `write`, `new_monomer.rotate`, `new_monomer.translate`, `KeyError`, `_sequence_output_stem`, +1 more

### `hygel_martini/param_opt/qm_to_martini/__init__.py`

*(no module docstring)*

### `hygel_martini/param_opt/qm_to_martini/__main__.py`

*(no module docstring)*

### `hygel_martini/param_opt/qm_to_martini/analysis/__init__.py`

Analysis helpers for qm_to_martini postprocessing.

#### `__getattr__(name: str)` — line 19
- *(no docstring)*
- kind: function
- returns: `module.main`
- raises: `AttributeError`
- calls: `import_module`, `AttributeError`

### `hygel_martini/param_opt/qm_to_martini/analysis/compare_sweeps.py`

Compare two postprocess sweep result directories (e.g., old trim vs new trim).

#### `load_variant_table(result_dir: Path)` — line 43
- *(no docstring)*
- kind: function
- returns: `pd.read_csv(path)`; `None`
- effects: stdout
- calls: `pd.read_csv`, `path.exists`, `print`

#### `_to_float_series(series: pd.Series)` — line 51
- *(no docstring)*
- kind: function, internal
- returns: `pd.to_numeric(series, errors='coerce').fillna(0.0)`
- calls: `fillna`, `pd.to_numeric`

#### `plot_bar_comparison(old_df: pd.DataFrame, new_df: pd.DataFrame, out_dir: Path, label_old: str, label_new: str)` — line 55
- *(no docstring)*
- kind: function
- returns: `None` (bare return)
- effects: stdout
- calls: `old_df.merge`, `tolist`, `np.arange`, `plt.subplots`, `fig.suptitle`, `fig.tight_layout`, `plt.close`, `print`, `ax.bar`, `ax.set_xticks`, `ax.set_xticklabels`, `ax.set_ylabel`, +3 more

#### `plot_diff_heatmap(old_df: pd.DataFrame, new_df: pd.DataFrame, out_dir: Path, label_old: str, label_new: str)` — line 100
- *(no docstring)*
- kind: function
- returns: `None` (bare return)
- calls: `old_df.merge`, `tolist`, `np.zeros`, `plt.subplots`, `ax.imshow`, `ax.set_xticks`, `ax.set_xticklabels`, `ax.set_yticks`, `ax.set_yticklabels`, `fig.colorbar`, `ax.set_title`, `fig.tight_layout`, +5 more

#### `write_diff_csv(old_df: pd.DataFrame, new_df: pd.DataFrame, out_dir: Path, label_old: str, label_new: str)` — line 140
- *(no docstring)*
- kind: function
- returns: `None` (bare return)
- effects: filesystem
- calls: `old_df.merge`, `merged.iterrows`, `rows.append`, `out_dir.mkdir`, `row.get`, `keys`, `open`, `csv.DictWriter`, `writer.writeheader`, `writer.writerows`

#### `main()` — line 179
- *(no docstring)*
- kind: function, CLI entry
- returns: `None` (bare return)
- effects: filesystem, stdout
- calls: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `load_variant_table`, `out_dir.mkdir`, `plot_bar_comparison`, `plot_diff_heatmap`, `write_diff_csv`, `print`, `args.old_result_dir.resolve`, `args.new_result_dir.resolve`, `args.out_dir.resolve`

### `hygel_martini/param_opt/qm_to_martini/analysis/log_manager.py`

Move flat sweep logs into logs/<variant>/<label>/<mode>.log.

#### `organize_result_dir(result_dir: Path, dry_run: bool=False)` — line 12
- *(no docstring)*
- kind: function
- returns: `(moved, skipped)`; `(0, 0)`
- effects: filesystem
- calls: `expected_path.open`, `csv.DictReader`, `expected_path.exists`, `log_dir.exists`, `nested.exists`, `flat.exists`, `nested.parent.mkdir`, `shutil.move`, `flat.read_bytes`, `nested.read_bytes`, `flat.unlink`

#### `main()` — line 46
- *(no docstring)*
- kind: function, CLI entry
- effects: stdout
- calls: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `organize_result_dir`, `print`, `path.resolve`, `glob`, `resolve`, `Path`

### `hygel_martini/param_opt/qm_to_martini/analysis/plotter.py`

Create visual diagnostics for a postprocess summary sweep.

#### `natural_key(text: str)` — line 86
- Sort key ordering "P2" before "P10"; unnumbered names sort last.
- kind: function
- returns: `(10000, str(text))`; `(int(match.group(1)), str(text))`
- calls: `re.match`, `match.group`

#### `clean_number(value: object)` — line 94
- Format a value compactly: NA, integers without .0, else %g.
- kind: function
- returns: `f'{number:g}'`; `'NA'`; `str(int(number))` (+1 more)
- calls: `pd.isna`, `math.isfinite`, `number.is_integer`

#### `rmsd_label(value: object)` — line 107
- Render an RMSD cutoff as the compact "R<value>" tag.
- kind: function
- returns: `f'R{clean_number(value)}'`
- calls: `clean_number`

#### `compact_force(value: str)` — line 112
- Humanize a force-profile id by replacing underscores with spaces.
- kind: function
- returns: `str(value).replace('_', ' ')`
- calls: `replace`

#### `compact_variant_label(row: pd.Series)` — line 117
- Two-line variant label: potential set over cutoff + force profile.
- kind: function
- returns: `f"{row['potential_set']}\n{rmsd_label(row['rmsd_max_cutoff'])} {compact_force(row['force_profile'])}"`
- calls: `rmsd_label`, `compact_force`

#### `one_line_variant_label(row: pd.Series)` — line 122
- Single-line variant label for horizontal bar-chart tick labels.
- kind: function
- returns: `f"{row['potential_set']} | {rmsd_label(row['rmsd_max_cutoff'])} {compact_force(row['force_profile'])}"`
- calls: `rmsd_label`, `compact_force`

#### `potential_label(row: pd.Series)` — line 127
- Two-line potential label with the angle/dihedral/improper functs.
- kind: function
- returns: `f'{potential}\nA{angle} D{dihedral} I{improper}'`; `f'{potential}\nBartender'`
- calls: `clean_number`, `row.get`

#### `enrich_variants(df: pd.DataFrame)` — line 142
- Derive the plotting columns from the raw variant summary table.
- kind: function
- returns: `out.copy()`
- calls: `items`, `pd.DataFrame.from_records`, `pd.concat`, `out.copy`, `VARIANT_RE.match`, `match.group`, `records.append`, `numeric_cols.append`, `replace`, `astype`, `profile.endswith`, `df.copy`, +2 more

#### `coverage_col(section: str, average: bool)` — line 224
- Return the count column name for a section (sum or per-case avg).
- kind: function
- returns: `f'{base}_avg_per_case' if average else base`

#### `count_label(average: bool)` — line 230
- Human label describing the count form used in axis/legend text.
- kind: function
- returns: `'average per label/mode case' if average else 'sum over label/mode cases'`

#### `count_stem_suffix(average: bool)` — line 235
- Filename suffix distinguishing the two count forms.
- kind: function
- returns: `'avg_per_case' if average else 'sum'`

#### `count_fmt(average: bool)` — line 240
- Annotation number format: 2 sig figs for averages, integers for sums.
- kind: function
- returns: `'.2g' if average else '.0f'`

#### `grid_shape(count: int)` — line 245
- Subplot grid shape: single axes, else two columns of ceil(n/2) rows.
- kind: function
- returns: `(math.ceil(count / 2), 2)`; `(1, 1)`
- calls: `math.ceil`

#### `potential_order(df: pd.DataFrame)` — line 252
- Distinct potential sets in natural (numbered) order.
- kind: function
- returns: `sorted(df['potential_set'].dropna().unique(), key=natural_key)`
- calls: `unique`, `dropna`

#### `force_order(df: pd.DataFrame)` — line 257
- Distinct force profiles in natural (numbered) order.
- kind: function
- returns: `sorted(df['force_profile'].dropna().unique(), key=natural_key)`
- calls: `unique`, `dropna`

#### `rmsd_order(df: pd.DataFrame)` — line 262
- Distinct RMSD cutoffs from loose (largest) to strict (smallest).
- kind: function
- returns: `sorted(values, reverse=True)`
- calls: `unique`, `dropna`, `pd.to_numeric`

#### `potential_label_map(df: pd.DataFrame)` — line 268
- Map each potential set to its display label (from its first row).
- kind: function
- returns: `labels`
- calls: `potential_order`, `potential_label`

#### `save_figure(fig: plt.Figure, out_dir: Path, stem: str)` — line 277
- Save a figure as both ``<stem>.pdf`` and ``<stem>.png`` and close it.
- kind: function
- effects: filesystem
- calls: `out_dir.mkdir`, `fig.savefig`, `plt.close`

#### `cleanup_stale_plot_outputs(out_dir: Path)` — line 288
- Delete outputs from older naming schemes and generated subfolders.
- kind: function
- effects: filesystem
- calls: `out_dir.glob`, `path.exists`, `path.is_file`, `subdir.exists`, `subdir.glob`, `path.unlink`

#### `annotate_heatmap(ax: plt.Axes, data: np.ndarray, fmt: str, threshold: float)` — line 337
- Write cell values onto a heatmap, skipping non-finite cells.
- kind: function
- calls: `ax.text`, `np.isfinite`, `format`

#### `annotation_box()` — line 356
- Shared bbox style for scatter-plot callout annotations.
- kind: function
- returns: `{'boxstyle': 'round,pad=0.22', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.78}`

#### `plot_heatmap_grid(df: pd.DataFrame, out_dir: Path, metric: str, title: str, colorbar_label: str, stem: str, cmap_name: str, fmt: str, log_scale: bool=False)` — line 366
- Render one metric as a grid of heatmaps, one panel per potential set.
- kind: function
- calls: `potential_order`, `force_order`, `rmsd_order`, `potential_label_map`, `pd.to_numeric`, `math.isclose`, `grid_shape`, `plt.subplots`, `copy`, `cmap.set_bad`, `fig.suptitle`, `fig.subplots_adjust`, +33 more

#### `plot_loose_screen_coverage(df: pd.DataFrame, out_dir: Path, average: bool=False)` — line 467
- Plot per-potential coverage under the loosest screen (plot 01).
- kind: function
- calls: `copy`, `sub.sort_values`, `plt.subplots`, `np.arange`, `np.zeros`, `ax.set_xticks`, `ax.set_xticklabels`, `ax.set_ylabel`, `ax.set_ylim`, `ax.set_title`, `ax.legend`, `set_visible`, +21 more

#### `color_map_for_potentials(df: pd.DataFrame)` — line 545
- Assign each potential set a palette color (cycled, stable order).
- kind: function
- returns: `{pot: POTENTIAL_PALETTE[i % len(POTENTIAL_PALETTE)] for i, pot in enumerate(potential_order(df))}`
- calls: `potential_order`

#### `plot_tradeoff_scatter(df: pd.DataFrame, out_dir: Path, average: bool=False)` — line 550
- Scatter accepted-angle coverage against P90 fit error (plot 05).
- kind: function
- calls: `plt.subplots`, `color_map_for_potentials`, `coverage_col`, `potential_order`, `ax.axhline`, `ax.text`, `annotations.items`, `ax.set_xlabel`, `ax.set_ylabel`, `ax.set_title`, `ax.grid`, `set_visible`, +14 more

#### `read_recommended_variants(result_dir: Path, df: pd.DataFrame)` — line 627
- Select the variants highlighted in "recommended candidates" plots.
- kind: function
- returns: `selected.sort_values(['_pot_order', '_rmsd_order', '_force_order'])`; `selected.sort_values('_order')`
- calls: `recommended.exists`, `potential_order`, `copy`, `selected.sort_values`, `line.strip`, `splitlines`, `isin`, `recommended.read_text`

#### `plot_recommended_candidates(df: pd.DataFrame, result_dir: Path, out_dir: Path, average: bool=False)` — line 665
- Bar-compare the recommended variants on coverage and RMSE (plot 06).
- kind: function
- returns: `None` (bare return)
- calls: `read_recommended_variants`, `np.arange`, `plt.subplots`, `set_yticks`, `set_yticklabels`, `invert_yaxis`, `title`, `fig.suptitle`, `fig.tight_layout`, `save_figure`, `compact_variant_label`, `fillna`, +15 more

#### `metric_available(df: pd.DataFrame, metric: str)` — line 719
- True when the column exists and has at least one numeric value.
- kind: function
- returns: `pd.to_numeric(df[metric], errors='coerce').notna().any()`; `False`
- calls: `notna`, `pd.to_numeric`

#### `plot_rmse_metric_heatmaps(df: pd.DataFrame, out_dir: Path)` — line 726
- Render the RMSE/RMSD heatmap set (plots 04*): all/angle/dihedral, P90 and max, skipping metrics absent from the table.
- kind: function
- calls: `plot_heatmap_grid`, `metric_available`

#### `plot_rmse_recommended_candidates(df: pd.DataFrame, result_dir: Path, out_dir: Path)` — line 788
- Bar-compare recommended variants on six RMSE/RMSD metrics (plot 06).
- kind: function
- returns: `None` (bare return)
- effects: filesystem
- calls: `read_recommended_variants`, `math.ceil`, `chunk_dir.exists`, `copy`, `chunk_dir.glob`, `np.arange`, `plt.subplots`, `set_yticks`, `set_yticklabels`, `invert_yaxis`, `fig.suptitle`, `fig.subplots_adjust`, +21 more

#### `plot_rmse_threshold_curves(df: pd.DataFrame, out_dir: Path, potential_set: str)` — line 862
- Plot RMSE/RMSD versus cutoff for one potential set (plot 07).
- kind: function
- returns: `None` (bare return)
- calls: `copy`, `math.ceil`, `plt.subplots`, `legend`, `fig.suptitle`, `fig.tight_layout`, `save_figure`, `axes.ravel`, `ax.invert_xaxis`, `ax.axhline`, `ax.set_xlabel`, `ax.set_title`, +10 more

#### `plot_rmse_metric_suite(df: pd.DataFrame, result_dir: Path, out_dir: Path)` — line 921
- Render the complete RMSE suite under ``<out_dir>/rmse_metrics``: heatmaps, tradeoff scatters (both count forms), candidate bars, and per-potential threshold curves.
- kind: function
- calls: `plot_rmse_metric_heatmaps`, `plot_rmse_recommended_candidates`, `potential_order`, `plot_tradeoff_scatter`, `plot_rmse_threshold_curves`

#### `plot_force_metric_heatmaps(df: pd.DataFrame, out_dir: Path)` — line 936
- Render the force-metric heatmap set (plots 09-11), log-scaled since force constants span orders of magnitude; absent metrics are skipped.
- kind: function
- calls: `plot_heatmap_grid`, `metric_available`

#### `plot_force_error_scatter(df: pd.DataFrame, out_dir: Path, section: str)` — line 999
- Scatter force-metric diagnostics for one section (plot 12).
- kind: function
- returns: `None` (bare return)
- calls: `df.copy`, `pd.to_numeric`, `color_map_for_potentials`, `plt.subplots`, `potential_order`, `ax.set_xscale`, `ax.axvline`, `ax.axhline`, `ax.text`, `ax.set_xlabel`, `ax.set_ylabel`, `ax.set_title`, +17 more

#### `plot_force_recommended_candidates(df: pd.DataFrame, result_dir: Path, out_dir: Path)` — line 1087
- Bar-compare recommended variants on force metrics (plot 13).
- kind: function
- returns: `None` (bare return)
- effects: filesystem
- calls: `read_recommended_variants`, `math.ceil`, `out_dir.glob`, `chunk_dir.exists`, `copy`, `stale.is_file`, `chunk_dir.glob`, `np.arange`, `plt.subplots`, `set_yticks`, `set_yticklabels`, `invert_yaxis`, +23 more

#### `plot_force_threshold_curves(df: pd.DataFrame, out_dir: Path, potential_set: str)` — line 1168
- Plot force metrics versus cutoff for one potential set (plot 14).
- kind: function
- returns: `None` (bare return)
- calls: `copy`, `math.ceil`, `plt.subplots`, `legend`, `fig.suptitle`, `fig.tight_layout`, `save_figure`, `axes.ravel`, `ax.invert_xaxis`, `ax.set_yscale`, `ax.set_xlabel`, `ax.set_title`, +12 more

#### `plot_force_metric_suite(df: pd.DataFrame, result_dir: Path, out_dir: Path)` — line 1228
- Render the complete force-metric suite under ``<out_dir>/force_metrics``: heatmaps, angle/dihedral scatters, candidate bars, and per-potential threshold curves.
- kind: function
- calls: `plot_force_metric_heatmaps`, `plot_force_recommended_candidates`, `potential_order`, `plot_force_error_scatter`, `plot_force_threshold_curves`

#### `plot_case_heatmap(case_df: pd.DataFrame, out_dir: Path, variant_id: str)` — line 1243
- Plot label x mode heatmaps for one variant's cases (plot 07_case).
- kind: function
- returns: `None` (bare return)
- calls: `copy`, `plt.subplots`, `axes.ravel`, `fig.suptitle`, `fig.tight_layout`, `save_figure`, `pd.to_numeric`, `selected.pivot_table`, `pivot.reindex`, `pivot.to_numpy`, `math.isclose`, `cmap.set_bad`, +15 more

#### `case_heatmap_variant_ids(result_dir: Path, variant_df: pd.DataFrame, requested_variant: str | None)` — line 1306
- Recommended variant ids (deduplicated, order kept) plus the explicitly requested one when it is not already included.
- kind: function
- returns: `ids`
- calls: `read_recommended_variants`, `dict.fromkeys`, `ids.append`, `tolist`

#### `plot_threshold_curves(df: pd.DataFrame, out_dir: Path, potential_set: str, average: bool=False)` — line 1316
- Plot coverage and P90 error versus cutoff for one potential (plot 08).
- kind: function
- returns: `None` (bare return)
- calls: `copy`, `plt.subplots`, `legend`, `fig.suptitle`, `fig.tight_layout`, `save_figure`, `ax.invert_xaxis`, `ax.set_xlabel`, `ax.set_title`, `ax.grid`, `set_visible`, `coverage_col`, +8 more

#### `write_plot_readme(out_dir: Path, result_dir: Path)` — line 1371
- Write ``README.md`` describing the plot outputs and reading order.
- kind: function
- effects: filesystem
- calls: `title`, `write_text`, `join`, `result_dir.name.replace`

#### `main()` — line 1401
- CLI entry point: load the sweep tables and render every plot group.
- kind: function, CLI entry
- raises: `SystemExit`
- effects: filesystem, stdout
- calls: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `args.result_dir.resolve`, `enrich_variants`, `pd.read_csv`, `out_dir.mkdir`, `cleanup_stale_plot_outputs`, `plot_rmse_metric_suite`, `plot_force_metric_suite`, `case_heatmap_variant_ids`, `write_plot_readme`, +16 more

### `hygel_martini/param_opt/qm_to_martini/analysis/reference_qualification.py`

Audit whether a sparse high-level reference can qualify an xTB ensemble.

#### `_finite_vector(values: Sequence[float], name: str, minimum_size: int=1)` — line 32
- *(no docstring)*
- kind: function, internal
- returns: `array`
- raises: `ValueError`
- calls: `reshape`, `ValueError`, `np.all`, `np.asarray`, `np.isfinite`

#### `_pairwise_sign(value: float, tolerance: float)` — line 41
- *(no docstring)*
- kind: function, internal
- returns: `0`; `1`; `-1`

#### `_coerce_bool(value: Any)` — line 49
- *(no docstring)*
- kind: function, internal
- returns: `bool(value)`; `True`; `False`
- raises: `ValueError`
- calls: `ValueError`, `lower`, `value.strip`

#### `audit_relative_energies(xtb_energy_kj_mol: Sequence[float], reference_energy_kj_mol: Sequence[float], *, max_abs_error_kj_mol: float=8.4, ordering_tolerance_kj_mol: float=1e-08)` — line 63
- Compare xTB and reference relative-energy landscapes.
- kind: function
- returns: `{'decision': 'PASS' if passed else 'FAIL', 'same_minimum': same_minimum, 'xtb_minimum_index': xtb_minimum_index, 'reference_minimum_index': reference_minimum_index, 'ordering_agreement_fraction': ordering_agreement, 'agreeing_pairs': agreeing_pairs, 'total_pairs': total_pairs, 'disagreeing_pairs': disagreeing_pairs, 'mae_kj_mol': float(np.mean(np.abs(error))), 'rmse_kj_mol': float(np.sqrt(np.mean(np.square(error)))), 'max_abs_error_kj_mol': max_abs_error, 'max_abs_error_gate_kj_mol': float(max_abs_error_kj_mol), 'xtb_relative_energy_kj_mol': xtb_relative.tolist(), 'reference_relative_energy_kj_mol': reference_relative.tolist(), 'relative_energy_error_kj_mol': error.tolist()}`
- raises: `ValueError`
- calls: `_finite_vector`, `ValueError`, `np.argmin`, `np.max`, `xtb_relative.tolist`, `reference_relative.tolist`, `error.tolist`, `np.min`, `_pairwise_sign`, `np.abs`, `np.mean`, `np.sqrt`, +2 more

#### `audit_gradient(gradient: Sequence[float] | Sequence[Sequence[float]], *, rms_threshold: float=3e-05, max_threshold: float=0.0001)` — line 141
- Audit reference-level stationarity from Cartesian gradient components.
- kind: function
- returns: `{'decision': 'STATIONARY' if stationary else 'NON_STATIONARY', 'stationary': stationary, 'component_count': int(components.size), 'rms_gradient': rms, 'max_abs_gradient': max_abs, 'rms_threshold': float(rms_threshold), 'max_threshold': float(max_threshold), 'rms_ratio_to_threshold': rms / rms_threshold if rms_threshold > 0 else math.inf, 'max_ratio_to_threshold': max_abs / max_threshold if max_threshold > 0 else math.inf}`
- raises: `ValueError`
- calls: `_finite_vector`, `np.asarray`, `ValueError`, `np.sqrt`, `np.max`, `np.mean`, `np.abs`, `np.square`

#### `audit_endpoint_family(entries: Iterable[Mapping[str, Any]], *, rmsd_threshold_nm: float=0.05, energy_threshold_kj_mol: float=2.0)` — line 173
- Classify optimized endpoints as a single or multiple DFT family.
- kind: function
- returns: `{'decision': decision, 'endpoint_count': len(normalized), 'rmsd_threshold_nm': float(rmsd_threshold_nm), 'energy_threshold_kj_mol': float(energy_threshold_kj_mol), 'integrity_failures': integrity_failures, 'rmsd_failures': rmsd_failures, 'energy_failures': energy_failures, 'endpoints': normalized}`
- raises: `ValueError`
- calls: `ValueError`, `_coerce_bool`, `normalized.append`, `entry.get`, `math.isfinite`

#### `audit_reweighting(delta_energy_kj_mol: Sequence[float], *, temperature_k: float, min_ess_fraction: float=0.2, max_normalized_weight: float=0.2)` — line 241
- Audit xTB-to-reference importance-weight overlap.
- kind: function
- returns: `{'decision': 'SUFFICIENT_OVERLAP' if sufficient_overlap else 'INSUFFICIENT_OVERLAP', 'sufficient_overlap': sufficient_overlap, 'sample_count': int(delta.size), 'temperature_k': float(temperature_k), 'effective_sample_size': ess, 'effective_sample_size_fraction': ess_fraction, 'minimum_ess_fraction': float(min_ess_fraction), 'maximum_normalized_weight': largest_weight, 'maximum_normalized_weight_gate': float(max_normalized_weight), 'normalized_weights': weights.tolist()}`
- raises: `ValueError`
- calls: `_finite_vector`, `np.exp`, `ValueError`, `np.sum`, `np.max`, `weights.tolist`, `np.square`

#### `_read_rows(path: Path)` — line 289
- *(no docstring)*
- kind: function, internal
- returns: `list(csv.DictReader(handle))`
- effects: filesystem
- calls: `path.open`, `csv.DictReader`

#### `_float_column(rows: Sequence[Mapping[str, str]], column: str)` — line 294
- *(no docstring)*
- kind: function, internal
- returns: `[float(row[column]) for row in rows]`
- raises: `ValueError`
- calls: `ValueError`

#### `_parse_bool(value: str)` — line 301
- *(no docstring)*
- kind: function, internal
- returns: `_coerce_bool(value)`
- calls: `_coerce_bool`

#### `_write_result(result: Mapping[str, Any], output: Path | None)` — line 305
- *(no docstring)*
- kind: function, internal
- returns: `None` (bare return)
- effects: filesystem, stdout
- calls: `json.dumps`, `output.parent.mkdir`, `output.write_text`, `print`

#### `_energy_command(args: argparse.Namespace)` — line 314
- *(no docstring)*
- kind: function, internal
- returns: `{'analysis': 'relative_energy_qualification', 'decision': 'PASS' if all((result['decision'] == 'PASS' for result in results.values())) else 'FAIL', 'groups': results}`
- raises: `ValueError`
- calls: `_read_rows`, `groups.items`, `ValueError`, `audit_relative_energies`, `append`, `_float_column`, `groups.setdefault`, `results.values`

#### `_gradient_command(args: argparse.Namespace)` — line 347
- *(no docstring)*
- kind: function, internal
- returns: `result`
- raises: `ValueError`
- calls: `_read_rows`, `audit_gradient`, `ValueError`

#### `_endpoint_command(args: argparse.Namespace)` — line 363
- *(no docstring)*
- kind: function, internal
- returns: `result`
- calls: `_read_rows`, `audit_endpoint_family`, `_parse_bool`

#### `_overlap_command(args: argparse.Namespace)` — line 383
- *(no docstring)*
- kind: function, internal
- returns: `result`
- calls: `_read_rows`, `audit_reweighting`, `_float_column`

#### `build_parser()` — line 395
- *(no docstring)*
- kind: function
- returns: `parser`
- calls: `argparse.ArgumentParser`, `parser.add_subparsers`, `subparsers.add_parser`, `energy.add_argument`, `energy.set_defaults`, `gradient.add_argument`, `gradient.set_defaults`, `endpoint.add_argument`, `endpoint.set_defaults`, `overlap.add_argument`, `overlap.set_defaults`

#### `main(argv: Sequence[str] | None=None)` — line 459
- *(no docstring)*
- kind: function, CLI entry
- returns: `0`
- calls: `build_parser`, `parser.parse_args`, `args.handler`, `_write_result`, `parser.error`

### `hygel_martini/param_opt/qm_to_martini/analysis/summarizer.py`

Summarize postprocess sweep outputs into case/variant CSV tables.

#### `as_float(value: Any)` — line 39
- Return the value as a finite float, or None for anything else.
- kind: function
- returns: `None`; `float(value)`
- calls: `math.isfinite`

#### `percentile(values: Sequence[float], fraction: float)` — line 46
- Linearly interpolated percentile of the values.
- kind: function
- returns: `sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight`; `None`; `sorted_values[0]` (+1 more)
- calls: `math.floor`, `math.ceil`

#### `stats(values: Iterable[Any])` — line 70
- Compute the standard statistic block over the finite values.
- kind: function
- returns: `{'n': len(numeric), 'min': min(numeric), 'max': max(numeric), 'mean': mean(numeric), 'median': median(numeric), 'p90': percentile(numeric, 0.9)}`; `{key: '' for key in STAT_KEYS}`
- calls: `mean`, `median`, `percentile`, `as_float`

#### `flatten_stats(prefix: str, values: Iterable[Any])` — line 92
- Prefix the stat block keys, e.g. "all_rmse" -> "all_rmse_p90".
- kind: function
- returns: `{f'{prefix}_{key}': value for key, value in stats(values).items()}`
- calls: `items`, `stats`

#### `load_json(path: Path)` — line 97
- Read and parse a UTF-8 JSON file.
- kind: function
- returns: `json.loads(path.read_text(encoding='utf-8'))`
- calls: `json.loads`, `path.read_text`

#### `safe_get(mapping: Dict[str, Any], *keys: str, default: Any='')` — line 102
- Walk nested dict keys, returning ``default`` on any missing step.
- kind: function
- returns: `current`; `default`

#### `variant_from_report_path(report_path: Path, outputs_dir: Path)` — line 112
- Extract (variant_id, label, mode) from a report's path.
- kind: function
- returns: `(parts[0], parts[1], parts[2])`
- raises: `ValueError`
- calls: `report_path.relative_to`, `ValueError`

#### `selected_terms(summary: Dict[str, Any])` — line 128
- Normalize a ``screened_summary.json`` payload to lists per section.
- kind: function
- returns: `result`
- calls: `summary.get`

#### `read_expected(result_dir: Path)` — line 137
- Read the optional expected-output manifest for missing-case checks.
- kind: function
- returns: `rows`; `set()`
- effects: filesystem
- calls: `expected_path.exists`, `expected_path.open`, `csv.DictReader`, `Path`, `rows.add`

#### `collect_case_rows(result_dir: Path)` — line 156
- Build one summary row per (variant, label, mode) case.
- kind: function
- returns: `(case_rows, missing_rows)`
- calls: `read_expected`, `outputs_dir.exists`, `variant_from_report_path`, `seen.add`, `load_json`, `report_path.with_name`, `selected_terms`, `report.get`, `row.update`, `case_rows.append`, `outputs_dir.glob`, `summary_path.exists`, +9 more

#### `aggregate_variant_rows(case_rows: Sequence[dict[str, Any]])` — line 240
- Aggregate case rows into one row per variant.
- kind: function
- returns: `variant_rows`
- calls: `defaultdict`, `append`, `grouped.items`, `variant_row.update`, `variant_rows.append`, `first.get`, `with_name`, `selected_terms`, `report_terms.extend`, `flatten_stats`, `summary_path.exists`, `load_json`, +3 more

#### `write_csv(path: Path, rows: Sequence[dict[str, Any]])` — line 303
- Write dict rows as CSV, unioning keys in first-seen order.
- kind: function
- returns: `None` (bare return)
- effects: filesystem
- calls: `path.parent.mkdir`, `path.write_text`, `path.open`, `csv.DictWriter`, `writer.writeheader`, `writer.writerows`, `fieldnames.append`

#### `write_overview(path: Path, case_rows: Sequence[dict[str, Any]], variant_rows: Sequence[dict[str, Any]], missing: Sequence[dict[str, Any]])` — line 324
- Write the Markdown overview: row counts plus suggested sort columns.
- kind: function
- effects: filesystem
- calls: `path.write_text`, `join`

#### `main()` — line 350
- CLI entry point: collect case rows, aggregate, and write the tables.
- kind: function, CLI entry
- effects: stdout
- calls: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `args.result_dir.resolve`, `collect_case_rows`, `aggregate_variant_rows`, `write_csv`, `write_overview`, `print`, `args.out_dir.resolve`

### `hygel_martini/param_opt/qm_to_martini/analysis/trim_sensitivity.py`

Energy-based trim sensitivity analysis for C/D/S xTB trajectories.

#### `read_energies(path: Path)` — line 83
- Read per-frame energies (Hartree) from an xTB XYZ trajectory.
- kind: function
- returns: `np.asarray(energies, dtype=float)`
- effects: filesystem
- calls: `np.asarray`, `path.open`, `handle.readline`, `ENERGY_RE.search`, `energies.append`, `line.strip`, `match.group`

#### `load_json(path: Path)` — line 118
- Read a JSON file, returning {} when it does not exist.
- kind: function
- returns: `json.loads(path.read_text(encoding='utf-8'))`; `{}`
- calls: `json.loads`, `path.exists`, `path.read_text`

#### `frame_to_ns(frame: int)` — line 125
- Convert a frame index/count to ns assuming DUMP_FS fs per frame.
- kind: function
- returns: `frame * DUMP_FS / 1000000.0`

#### `tail_reference(arr: np.ndarray, ref_fraction: float)` — line 130
- Compute the last-``ref_fraction`` reference statistics.
- kind: function
- returns: `(start, mean, std)`
- calls: `np.mean`, `np.std`

#### `threshold_t0(arr: np.ndarray, ref_fraction: float, sigma: float)` — line 151
- Detect the trim start via the tail-mean threshold criterion.
- kind: function
- returns: `(t0, ref_mean, ref_std, ref_start)`
- calls: `tail_reference`, `np.arange`, `np.abs`, `np.where`, `np.cumsum`

#### `rolling_stats(arr: np.ndarray, ref_mean: float, ref_std: float, window: int, sigma: float)` — line 174
- Diagnose when the rolling mean enters/stays in the reference band.
- kind: function
- returns: `{'first_in_band': first, 'stable_from': stable, 'fraction_in_band': float(np.mean(in_band)), 'max_abs_sigma': float(np.max(dev))}`; `{'first_in_band': None, 'stable_from': None, 'fraction_in_band': None}`
- calls: `np.convolve`, `np.abs`, `np.any`, `np.logical_and.accumulate`, `np.where`, `np.ones`, `np.mean`, `np.max`, `np.argmax`

#### `candidate_stats(arr: np.ndarray, start: int, ref_mean: float, ref_std: float)` — line 207
- Statistics of the energies retained by one candidate trim start.
- kind: function
- returns: `{'start': start, 'kept_frames': int(len(kept)), 'kept_ns': frame_to_ns(len(kept)), 'mean': mean, 'std': std, 'delta_ref_sigma': abs(mean - ref_mean) / ref_std, 'block_sem': block_sem}`; `{'start': start, 'kept_frames': 0, 'kept_ns': 0.0, 'mean': math.nan, 'std': math.nan, 'delta_ref_sigma': math.nan, 'block_sem': math.nan}`
- calls: `np.mean`, `np.array_split`, `np.asarray`, `frame_to_ns`, `np.std`, `math.sqrt`

#### `autocorrelation_summary(arr: np.ndarray, max_lag: int=50000)` — line 253
- Summarize the normalized energy autocorrelation function.
- kind: function
- returns: `values`
- calls: `np.fft.rfft`, `values.update`, `bit_length`, `np.fft.irfft`, `np.arange`, `np.zeros_like`, `np.any`, `np.where`, `np.mean`, `np.conjugate`, `np.argmax`, `np.sum`, +1 more

#### `write_csv(path: Path, rows: list[dict], fieldnames: list[str])` — line 306
- Write dict rows as CSV with the given fixed column order.
- kind: function
- effects: filesystem
- calls: `path.open`, `csv.DictWriter`, `writer.writeheader`, `writer.writerow`

#### `fmt_num(value, digits: int=3)` — line 315
- Format a value for the Markdown report ("-" for None/NaN/inf).
- kind: function
- returns: `str(value)`; `'-'`; `f'{value:.{digits}f}'`
- calls: `math.isnan`, `math.isinf`

#### `plot_energy(results: dict)` — line 326
- Plot per-label energy traces with rolling means and trim markers.
- kind: function
- returns: `None` (bare return)
- calls: `plt.subplots`, `fig.tight_layout`, `fig.savefig`, `plt.close`, `results.items`, `ax.plot`, `ax.axhline`, `ax.axhspan`, `items`, `ax.set_title`, `ax.set_xlabel`, `ax.set_ylabel`, +6 more

#### `plot_candidate_sensitivity(candidate_rows: list[dict])` — line 369
- Plot kept-mean deviation versus trim start for every candidate.
- kind: function
- returns: `None` (bare return)
- calls: `plt.subplots`, `ax.axhline`, `ax.set_xlabel`, `ax.set_ylabel`, `ax.set_title`, `ax.grid`, `ax.legend`, `fig.tight_layout`, `fig.savefig`, `plt.close`, `ax.plot`, `ax.text`

#### `configure_paths(project_dir: Path, out_dir: Path | None=None)` — line 400
- Rebind the module-level project/output/input path globals.
- kind: function
- calls: `project_dir.resolve`, `out_dir.resolve`

#### `main()` — line 430
- CLI entry point: run the full sensitivity study for C, D, and S.
- kind: function, CLI entry
- raises: `RuntimeError`
- effects: filesystem
- calls: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `configure_paths`, `OUTDIR.mkdir`, `TRAJECTORIES.items`, `write_csv`, `plot_energy`, `plot_candidate_sensitivity`, `report_lines.extend`, `write_text`, `read_energies`, +24 more

### `hygel_martini/param_opt/qm_to_martini/analysis/trim_summary.py`

Summarize auto-trim results from xtb_traj_to_pdb runs.

#### `scan_trim_info(root: Path)` — line 21
- *(no docstring)*
- kind: function
- returns: `rows`
- calls: `root.rglob`, `rows.append`, `json.loads`, `info_path.read_text`, `info_path.relative_to`, `data.get`

#### `write_csv(path: Path, rows: list[dict[str, Any]])` — line 49
- *(no docstring)*
- kind: function
- returns: `None` (bare return)
- effects: filesystem
- calls: `path.parent.mkdir`, `path.write_text`, `keys`, `path.open`, `csv.DictWriter`, `writer.writeheader`, `writer.writerows`

#### `plot_trim(rows: list[dict[str, Any]], out_dir: Path)` — line 61
- *(no docstring)*
- kind: function
- returns: `float(v)`; `0.0`
- effects: filesystem
- calls: `out_dir.mkdir`, `plt.subplots`, `ax.bar`, `ax.set_xticks`, `ax.set_xticklabels`, `ax.set_ylabel`, `ax.set_title`, `ax.legend`, `ax2.bar`, `ax2.set_xticks`, `ax2.set_xticklabels`, `ax2.set_ylabel`, +6 more

#### `main()` — line 105
- *(no docstring)*
- kind: function, CLI entry
- effects: stdout
- calls: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `args.compare_root.resolve`, `scan_trim_info`, `write_csv`, `plot_trim`, `print`, `args.out_dir.resolve`

### `hygel_martini/param_opt/qm_to_martini/analysis/trim_threshold_samet_analysis.py`

Analyze the trim_threshold_samet trajectory set.

#### `read_energies(path: Path)` — line 47
- *(no docstring)*
- kind: function
- returns: `np.asarray(energies, dtype=float)`
- effects: filesystem
- calls: `np.asarray`, `path.open`, `handle.readline`, `ENERGY_RE.search`, `energies.append`, `line.strip`, `match.group`

#### `load_trim_info(label: str)` — line 69
- *(no docstring)*
- kind: function
- returns: `json.loads(path.read_text(encoding='utf-8'))`
- calls: `json.loads`, `path.read_text`

#### `frame_to_ns(frame: int)` — line 74
- *(no docstring)*
- kind: function
- returns: `frame * DUMP_FS / 1000000.0`

#### `tail_reference(arr: np.ndarray, ref_fraction: float=0.2)` — line 78
- *(no docstring)*
- kind: function
- returns: `(start, mean, std)`
- calls: `np.mean`, `np.std`

#### `threshold_t0(arr: np.ndarray, ref_fraction: float=0.2, sigma: float=1.0)` — line 87
- *(no docstring)*
- kind: function
- returns: `int(hits[0]) if len(hits) else len(arr)`
- calls: `tail_reference`, `np.arange`, `np.abs`, `np.where`, `np.cumsum`

#### `rolling_stable_from(arr: np.ndarray, ref_mean: float, ref_std: float, window: int=500, sigma: float=1.0)` — line 95
- *(no docstring)*
- kind: function
- returns: `(first, stable)`; `(None, None)`
- calls: `np.convolve`, `np.abs`, `np.any`, `np.logical_and.accumulate`, `np.where`, `np.ones`, `np.argmax`

#### `acf_g(arr: np.ndarray, max_lag: int=50000)` — line 108
- *(no docstring)*
- kind: function
- returns: `(float(1.0 + 2.0 * np.sum(norm[1:cutoff])), first_nonpositive)`
- calls: `np.fft.rfft`, `bit_length`, `np.fft.irfft`, `np.arange`, `np.zeros`, `np.any`, `np.mean`, `np.conjugate`, `np.argmax`, `np.sum`

#### `write_csv(path: Path, rows: list[dict])` — line 122
- *(no docstring)*
- kind: function
- effects: filesystem
- calls: `path.open`, `csv.DictWriter`, `writer.writeheader`, `writer.writerows`, `keys`

#### `make_plots(results: dict[str, dict])` — line 129
- *(no docstring)*
- kind: function
- returns: `None` (bare return)
- calls: `plt.subplots`, `fig.tight_layout`, `fig.savefig`, `plt.close`, `ax.plot`, `ax.axhline`, `ax.axhspan`, `ax.axvline`, `ax.set_title`, `ax.set_xlabel`, `ax.set_ylabel`, `ax.grid`, +5 more

#### `configure_paths(project_dir: Path, out_dir: Path | None=None)` — line 185
- *(no docstring)*
- kind: function
- calls: `project_dir.resolve`, `out_dir.resolve`

#### `main()` — line 196
- *(no docstring)*
- kind: function, CLI entry
- raises: `RuntimeError`
- effects: filesystem
- calls: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `configure_paths`, `OUTDIR.mkdir`, `TRAJECTORIES.items`, `write_csv`, `write_text`, `make_plots`, `lines.extend`, `read_energies`, `load_trim_info`, +12 more

### `hygel_martini/param_opt/qm_to_martini/cli.py`

Command-line entry point for the stage-03 (qm_to_martini) workflow.

#### `build_arg_parser()` — line 27
- Build the argparse parser: shared config flags plus stage-03 modes.
- kind: function
- returns: `parser`
- calls: `argparse.ArgumentParser`, `add_qm_to_martini_cli_args`, `parser.add_argument`

#### `main()` — line 47
- Console-script entry point.
- kind: function, CLI entry
- returns: `None` (bare return)
- raises: `ValueError`, `SystemExit`
- effects: filesystem, stdout
- calls: `build_arg_parser`, `parser.parse_args`, `run_qm_to_martini`, `Path`, `write_text`, `print`, `result.get`, `get`, `ValueError`, `json.dumps`, `SystemExit`, `key.replace`, +1 more

### `hygel_martini/param_opt/qm_to_martini/config.py`

Configuration resolution and shared datatypes for the stage-03 pipeline.

#### class `ConnectionDetectionConfig` — line 49
How monomer connection (capping) atoms are located in an XYZ file.

#### class `TermGenerationConfig` — line 60
Resolved ``bartender_pipeline.term_generation`` settings.

#### class `WeightedAtomRef` — line 79
One atom's (possibly fractional) membership in a CG bead.

##### `weight(self)` — line 91
- Exact fractional weight (1/denominator) of this reference.
- kind: property
- returns: `Fraction(1, self.denominator)`
- calls: `Fraction`

##### `format(self)` — line 95
- Render the Bartender BEADS token, e.g. "12" or "12/2".
- kind: method
- returns: `f'{self.atom_index}/{self.denominator}'`; `str(self.atom_index)`

#### class `ValidationReport` — line 102
Accumulated problems/warnings from validating one template or input.

##### `ok(self)` — line 115
- True when no fatal problem was recorded (warnings are allowed).
- kind: property
- returns: `not self.problems`

##### `render(self)` — line 119
- Format the report as a human-readable multi-line status block.
- kind: method
- returns: `'\n'.join(lines) + '\n'`
- calls: `lines.append`, `lines.extend`, `join`

#### class `MonomerTemplate` — line 131
Parsed Bartender ``.inp`` mapping/topology template for one monomer.

##### `bead_count(self)` — line 155
- Number of CG beads defined by the template.
- kind: property
- returns: `len(self.beads)`

##### `atom_count(self)` — line 160
- Highest atom index referenced by any bead (0 if no beads).
- kind: property
- returns: `max((ref.atom_index for refs in self.beads.values() for ref in refs))`; `0`
- calls: `self.beads.values`

#### class `PolymerInputBundle` — line 170
Everything built for one polymer's Bartender input.

#### class `ParamLine` — line 198
One parsed bonded-parameter line from a Bartender ``gmx_out.itp``.

#### class `TypedRecord` — line 222
A ``ParamLine`` lifted to bead-type space for cross-case merging.

#### class `ConnectionMetadata` — line 260
Head/tail connection geometry inferred for one monomer.

#### class `MergedVariant` — line 283
One distinct parameter variant within a merged type group.

#### `write_text(path: Path, text: str)` — line 316
- Write UTF-8 text, creating parent directories as needed.
- kind: function
- effects: filesystem
- calls: `path.parent.mkdir`, `path.write_text`

#### `shell_assign(name: str, value: str)` — line 321
- Render a shell-safe ``name=value`` assignment line.
- kind: function
- returns: `f'{name}={shlex.quote(value)}'`
- calls: `shlex.quote`

#### `resolve_under_base(base_dir: Path, value: str | Path)` — line 325
- Resolve a possibly relative path against ``base_dir``.
- kind: function
- returns: `(base_dir / path).resolve()`; `path`
- calls: `Path`, `path.is_absolute`, `resolve`

#### `parse_bool(value: Any, default: bool=False)` — line 336
- Coerce a YAML/JSON scalar to bool.
- kind: function
- returns: `str(value).strip().lower() in {'1', 'true', 'yes', 'on'}`; `default`; `value` (+1 more)
- calls: `lower`, `strip`

#### `resolve_connection_detection_config(pipeline_cfg: Dict[str, Any])` — line 350
- Read connector-atom detection settings from the pipeline config.
- kind: function
- returns: `ConnectionDetectionConfig(indicator=indicator, cutoff=cutoff)`
- raises: `ValueError`
- calls: `ConnectionDetectionConfig`, `strip`, `pipeline_cfg.get`, `ValueError`

#### `resolve_term_generation_config(pipeline_cfg: Dict[str, Any])` — line 370
- Normalize ``bartender_pipeline.term_generation`` into a config object.
- kind: function
- returns: `TermGenerationConfig(mode=normalized_mode, n=budget, main_itp_dir=main_itp_dir, candidates_tsv_dir=candidates_tsv_dir)`
- raises: `ValueError`, `TypeError`
- calls: `pipeline_cfg.get`, `aliases.get`, `TermGenerationConfig`, `lower`, `ValueError`, `raw_cfg.get`, `TypeError`, `strip`

#### `default_workdir_name(relaxation: str, md: str)` — line 455
- Return the conventional relaxation workdir name for a mode pair.
- kind: function
- returns: `name`; `fallback`
- raises: `ValueError`
- calls: `_WORKDIR_NAMES.get`, `ValueError`

#### `_normalize_pipeline_mode(value: Any, default: str, field_name: str)` — line 479
- Lowercase a mode string; map None -> default and False -> "off".
- kind: function, internal
- returns: `str(value).strip().lower()`; `default`; `'off'`
- raises: `ValueError`
- calls: `lower`, `ValueError`, `strip`

#### `resolve_pipeline_modes(pipeline_cfg: Dict[str, Any])` — line 492
- Resolve the workflow "flow" triple: relaxation mode, md mode, workdir.
- kind: function
- returns: `{'relaxation': relaxation, 'md': md, 'workdir_name': workdir_name or default_workdir_name(relaxation, md)}`
- raises: `ValueError`
- calls: `pipeline_cfg.get`, `_normalize_pipeline_mode`, `strip`, `ValueError`, `default_workdir_name`, `mode_cfg.get`, `legacy_relax_cfg.get`, `join`, `legacy_bartender_cfg.get`

#### `resolve_spin_state(uhf_value: Any, multiplicity_value: Any, *, label: str)` — line 589
- Reconcile the (uhf, multiplicity) pair describing a spin state.
- kind: function
- returns: `(uhf, multiplicity)`; `(0, 1)`
- raises: `ValueError`
- calls: `ValueError`

#### `_normalize_index_list(raw: Any, *, label: str)` — line 633
- Convert user 0-based atom indices to a deduplicated 1-based list.
- kind: function, internal
- returns: `normalized`; `[]`
- raises: `TypeError`, `ValueError`
- calls: `TypeError`, `ValueError`, `normalized.append`, `seen.add`

#### `resolve_backbone_atom_config(raw: Any, *, label: str)` — line 663
- Normalize a monomer's ``backbone_atoms`` mapping.
- kind: function
- returns: `{'head': head, 'tail': tail, 'body': body}`; `{'head': [1], 'tail': [2], 'body': []}`
- raises: `TypeError`, `ValueError`
- calls: `_normalize_index_list`, `TypeError`, `raw.get`, `ValueError`

#### `export_backbone_atom_config(cfg: Dict[str, List[int]])` — line 691
- Convert an internal 1-based backbone config back to 0-based indices.
- kind: function
- returns: `{key: [int(value) - 1 for value in cfg.get(key, [])] for key in ('head', 'tail', 'body')}`
- calls: `cfg.get`

#### `normalize_monomer_configs(raw_monomers: Dict[str, Any], legacy_init_templates: Dict[str, Any])` — line 698
- Normalize the top-level ``monomers`` config section.
- kind: function
- returns: `normalized`
- raises: `ValueError`, `TypeError`
- calls: `raw_monomers.items`, `entry.get`, `resolve_spin_state`, `ValueError`, `resolve_backbone_atom_config`, `TypeError`, `legacy_init_templates.get`

#### `resolve_case_electronic_state(tokens: Sequence[str], monomer_cfg: Dict[str, Dict[str, Any]], pipeline_cfg: Dict[str, Any])` — line 750
- Determine the total charge and spin state of one polymer case.
- kind: function
- returns: `{'charge': charge, 'uhf': uhf, 'multiplicity': multiplicity, 'inferred_charge': inferred_charge, 'inferred_uhf': inferred_uhf}`
- raises: `TypeError`
- calls: `pipeline_cfg.get`, `state_cfg.get`, `TypeError`, `resolve_spin_state`

#### `resolve_optional_path(base_dir: Path, raw_value: Any)` — line 801
- Resolve an optional config path; empty/None becomes None.
- kind: function
- returns: `resolve_under_base(base_dir, value)`; `None`
- calls: `strip`, `resolve_under_base`

#### `resolve_xtb_settings(pipeline_cfg: Dict[str, Any])` — line 808
- Flatten ``bartender_pipeline.xtb`` (plus legacy fallbacks) to a dict.
- kind: function
- returns: `{'env_script': str(xtb_cfg.get('env_script', legacy_relax_cfg.get('xtb_env_script', ''))).strip(), 'binary': str(xtb_cfg.get('binary', legacy_relax_cfg.get('xtb_binary', 'xtb'))).strip(), 'gfn': int(xtb_cfg.get('gfn', 2)), 'parallel': int(xtb_cfg.get('parallel', legacy_relax_cfg.get('nprocs', 32))), 'opt_level': str(xtb_cfg.get('opt_level', 'normal')).strip(), 'opt_cycles': int(xtb_cfg.get('opt_cycles', 10000)), 'acc': float(xtb_cfg.get('acc', 1.0)), 'etemp': float(xtb_cfg.get('etemp', 300.0)), 'solvent_model': solvent_model, 'solvent': str(xtb_cfg.get('solvent', legacy_relax_cfg.get('solvent', 'water'))).strip(), 'solvent_reference': str(xtb_cfg.get('solvent_reference', '')).strip(), 'md_input_template_path': str(xtb_cfg.get('md_input_template_path', '')).strip(), 'md_temp_k': float(md_cfg.get('temp_k', legacy_relax_cfg.get('temp_k', 310.0))), 'md_time_ps': float(md_cfg.get('time_ps', legacy_relax_cfg.get('time_ps', 5000))), 'md_dump_fs': float(md_cfg.get('dump_fs', 50.0)), 'md_step_fs': float(md_cfg.get('step_fs', 4.0)), 'md_velo': parse_bool(md_cfg.get('velo', False)), 'md_hmass': int(md_cfg.get('hmass', 4)), 'md_shake': int(md_cfg.get('shake', 2)), 'md_sccacc': float(md_cfg.get('sccacc', 2.0)), 'md_restart': parse_bool(md_cfg.get('restart', False)), 'md_skip_frames': int(xtb_cfg.get('md_skip_frames', md_cfg.get('skip_frames', 0))), 'trim_nskip': int(xtb_cfg.get('trim_nskip', 1)), 'trim_max_fraction': float(xtb_cfg.get('trim_max_fraction', 1.0)), 'trim_detrend': parse_bool(xtb_cfg.get('trim_detrend', False), False), 'trim_fast': parse_bool(xtb_cfg.get('trim_fast', True), True), 'trim_method': str(xtb_cfg.get('trim_method', 'pymbar')), 'trim_ref_fraction': float(xtb_cfg.get('trim_ref_fraction', 0.2)), 'trim_threshold_sigma': float(xtb_cfg.get('trim_threshold_sigma', 1.0))}`
- raises: `TypeError`, `ValueError`
- calls: `pipeline_cfg.get`, `xtb_cfg.get`, `legacy_relax_cfg.get`, `TypeError`, `lower`, `ValueError`, `strip`, `parse_bool`, `md_cfg.get`

#### `resolve_orca_settings(pipeline_cfg: Dict[str, Any])` — line 877
- Flatten ``bartender_pipeline.orca`` (plus legacy fallbacks) to a dict.
- kind: function
- returns: `{'binary': str(orca_cfg.get('binary', legacy_relax_cfg.get('orca_binary', 'orca'))).strip(), 'nprocs': int(orca_cfg.get('nprocs', legacy_relax_cfg.get('nprocs', 32))), 'method_line': str(orca_cfg.get('method_line', f"{legacy_relax_cfg.get('orca_method', 'r2scan-3c')} CPCM({legacy_relax_cfg.get('solvent', 'water')}) Opt TightSCF")).strip(), 'max_iter': int(orca_cfg.get('max_iter', 300)), 'input_template_path': str(orca_cfg.get('input_template_path', '')).strip()}`
- raises: `TypeError`
- calls: `pipeline_cfg.get`, `legacy_relax_cfg.get`, `TypeError`, `strip`, `orca_cfg.get`

#### `_inspect_configured_executable(base_dir: Path, raw_value: Any)` — line 910
- Report how a configured executable resolves (path vs PATH lookup).
- kind: function, internal
- returns: `{'configured': configured, 'resolved': found, 'exists': found is not None, 'lookup': 'PATH'}`; `{'configured': configured, 'resolved': None, 'exists': False, 'lookup': 'missing'}`; `{'configured': configured, 'resolved': fallback or str(path), 'exists': path.exists() or fallback is not None, 'lookup': 'path->PATH' if fallback else 'path'}`
- effects: filesystem
- calls: `strip`, `shutil.which`, `configured.startswith`, `resolve_under_base`, `path.exists`

#### `resolve_executable_command(base_dir: Path, raw_value: Any)` — line 947
- Return the resolved executable path, or the raw string if unresolved.
- kind: function
- returns: `str(payload.get('configured') or '').strip()`; `resolved`
- calls: `_inspect_configured_executable`, `strip`, `payload.get`

#### `_inspect_optional_file(base_dir: Path, raw_value: Any)` — line 959
- Report existence of an optional file entry; empty counts as OK.
- kind: function, internal
- returns: `{'configured': configured, 'resolved': str(path), 'exists': path.exists(), 'lookup': 'path'}`; `{'configured': configured, 'resolved': None, 'exists': True, 'lookup': 'optional-empty'}`
- calls: `strip`, `resolve_under_base`, `path.exists`

#### `check_configured_tools(cfg: Dict[str, Any], requested: Optional[Sequence[str]]=None)` — line 977
- Verify that the configured external tool binaries can be found.
- kind: function
- returns: `{'ok': ok, 'base_dir': str(base_dir), 'tools': tools}`
- raises: `TypeError`
- calls: `resolve`, `cfg.get`, `resolve_xtb_settings`, `resolve_orca_settings`, `pipeline_cfg.get`, `lower`, `TypeError`, `tools.append`, `Path`, `strip`, `_inspect_configured_executable`, `_inspect_optional_file`, +3 more

#### `resolve_execution_settings(pipeline_cfg: Dict[str, Any])` — line 1043
- Flatten ``bartender_pipeline.execution`` into a settings dict.
- kind: function
- returns: `{'run_relaxation': parse_bool(exec_cfg.get('run_relaxation', False)), 'run_bartender': parse_bool(exec_cfg.get('run_bartender', bartender_cfg.get('execute', False))), 'shell': str(exec_cfg.get('shell', 'bash')).strip() or 'bash', 'slurm': slurm_enabled, 'use_srun': use_srun}`
- raises: `TypeError`
- calls: `pipeline_cfg.get`, `parse_bool`, `TypeError`, `exec_cfg.get`, `strip`, `bartender_cfg.get`

#### `_get_slurm_cpu_count()` — line 1077
- Return SLURM_CPUS_PER_TASK as an int >= 1, or 0 when unset/invalid.
- kind: function, internal
- returns: `0`; `max(1, int(val))`
- calls: `strip`, `os.environ.get`

#### `resolve_log_settings(pipeline_cfg: Dict[str, Any])` — line 1087
- Flatten ``bartender_pipeline.logs`` into a settings dict.
- kind: function
- returns: `{'enabled': parse_bool(log_cfg.get('enabled', True), True), 'dirname': str(log_cfg.get('dirname', 'logs')).strip() or 'logs', 'write_validation': parse_bool(log_cfg.get('write_validation', True), True), 'capture_runtime': parse_bool(log_cfg.get('capture_runtime', True), True)}`
- raises: `TypeError`
- calls: `pipeline_cfg.get`, `TypeError`, `parse_bool`, `log_cfg.get`, `strip`

#### `ensure_case_logs_dir(case_dir: Path, log_cfg: Dict[str, Any])` — line 1109
- Create and return the case logs directory, or None when disabled.
- kind: function
- returns: `logs_dir`; `None`
- effects: filesystem
- calls: `logs_dir.mkdir`, `log_cfg.get`

#### `execute_case_script(label: str, script_path: Path, cwd: Path, exec_cfg: Dict[str, Any], logs_dir: Optional[Path])` — line 1117
- Run one generated case script and optionally tee its output to logs.
- kind: function
- returns: `{'script': script_path.name, 'cwd': str(cwd), 'shell': str(exec_cfg.get('shell', 'bash')), 'slurm': slurm_enabled, 'use_srun': use_srun, 'command': command, 'returncode': returncode, 'stdout': stdout_name, 'stderr': stderr_name}`
- raises: `RuntimeError`
- effects: subprocess, filesystem
- calls: `parse_bool`, `exec_cfg.get`, `strip`, `subprocess.run`, `RuntimeError`, `shutil.which`, `srun_command.extend`, `open`, `subprocess.Popen`, `threading.Thread`, `t_out.start`, `t_err.start`, +10 more

#### `render_xtb_md_input(md_mode: str, xtb_cfg: Dict[str, Any], template_text: Optional[str])` — line 1214
- *(no docstring)*
- kind: function
- returns: `f"$md\n temp={xtb_cfg['md_temp_k']:.3f}\n time={xtb_cfg['md_time_ps']:.3f}\n dump={xtb_cfg['md_dump_fs']:.3f}\n step={xtb_cfg['md_step_fs']:.3f}\n velo={('true' if xtb_cfg['md_velo'] else 'false')}\n nvt=true\n hmass={xtb_cfg['md_hmass']}\n shake={xtb_cfg['md_shake']}\n sccacc={xtb_cfg['md_sccacc']:.3f}\n restart={('true' if xtb_cfg['md_restart'] else 'false')}\n$end\n"`; `template_text.rstrip() + '\n'`
- raises: `ValueError`
- calls: `ValueError`, `template_text.rstrip`

#### `render_orca_input(local_xyz_name: str, state: Dict[str, Any], orca_cfg: Dict[str, Any], template_text: Optional[str])` — line 1234
- *(no docstring)*
- kind: function
- returns: `f"{method_line}\n%pal nprocs {int(orca_cfg['nprocs'])} end\n\n%geom\n   MaxIter {int(orca_cfg['max_iter'])}\nend\n\n* xyzfile {int(state['charge'])} {int(state['multiplicity'])} {local_xyz_name}\n"`; `template_text.rstrip() + '\n\n' + f"* xyzfile {int(state['charge'])} {int(state.get('multiplicity', 1))} {local_xyz_name}\n"`
- calls: `strip`, `method_line.startswith`, `orca_cfg.get`, `re.search`, `re.sub`, `template_text.rstrip`, `state.get`

#### `normalize_sequence(sequence: Sequence[str] | str)` — line 1268
- *(no docstring)*
- kind: function
- returns: `tokens`
- raises: `ValueError`
- calls: `sequence.strip`, `ValueError`, `strip`, `token.strip`, `text.split`

#### `sequence_stem(tokens: Sequence[str])` — line 1285
- *(no docstring)*
- kind: function
- returns: `'_'.join(tokens)`; `''.join(tokens)`
- calls: `join`

#### `parse_sequence_entry(entry: Any, monomer_keys: set[str])` — line 1290
- *(no docstring)*
- kind: function
- returns: `tokens`
- raises: `ValueError`, `TypeError`
- calls: `entry.strip`, `ValueError`, `parse_csv_list`, `TypeError`, `strip`, `text.split`, `type`

#### `build_sequence_jobs(system_cfg: Dict[str, Any], monomer_keys: set[str])` — line 1312
- *(no docstring)*
- kind: function
- returns: `jobs`
- raises: `ValueError`
- calls: `system_cfg.get`, `ValueError`, `parse_sequence_entry`, `jobs.append`

#### `parse_xyz(path: Path)` — line 1330
- *(no docstring)*
- kind: function
- returns: `(symbols, coords)`
- raises: `ValueError`
- calls: `splitlines`, `ValueError`, `strip`, `line.split`, `symbols.append`, `coords.append`, `path.read_text`

### `hygel_martini/param_opt/qm_to_martini/defaults.py`

Built-in default configuration for the stage-03 (qm_to_martini) pipeline.

#### `_default_monomers()` — line 25
- Derive default per-monomer entries from the built-in monomer library.
- kind: function, internal
- returns: `monomers`
- calls: `DEFAULT_MONOMER_FILES.items`, `xyz_name.endswith`

### `hygel_martini/param_opt/qm_to_martini/generator.py`

Thin workflow entry helper for QM-to-Martini/Bartender generation.

#### `run_qm_to_martini(config_path: str | Path, overrides: argparse.Namespace | None=None)` — line 15
- Load a qm_to_martini maker file, apply optional overrides, and run the pipeline.
- kind: function
- returns: `(cfg, result)`
- calls: `load_config`, `Path`, `apply_cli_overrides`, `check_configured_tools`, `run_postprocess_only`, `run_pipeline`

### `hygel_martini/param_opt/qm_to_martini/pipeline.py`

Stage-03 pipeline orchestration: polymer build -> QM/xTB -> Bartender.

#### `_srun_reentry_lines(exec_cfg: Dict[str, Any], cpu_fallback_var: str)` — line 90
- Build shell lines that re-exec a run script under ``srun`` on SLURM.
- kind: function, internal
- returns: `['if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${SLURM_STEP_ID:-}" ]; then', '  if ! command -v srun >/dev/null 2>&1; then', '    echo "[ERROR] execution.use_srun=true but srun was not found" >&2', '    exit 1', '  fi', f'  exec srun --export=ALL --ntasks=1 --cpus-per-task "${{SLURM_CPUS_PER_TASK:-${cpu_fallback_var}}}" bash "$0" "$@"', 'fi']`; `[]`
- calls: `parse_bool`, `exec_cfg.get`

#### `_bartender_mode_args(flow: Dict[str, str], bartender_cfg: Dict[str, Any], bartender_charge: int, skip: int, trajectory: Optional[Path], outdir: Path)` — line 119
- md 모드에 따른 Bartender CLI 인자 목록 반환 (quoting 없이 raw 값).
- kind: function, internal
- returns: `args`
- raises: `ValueError`
- calls: `strip`, `os.path.relpath`, `ValueError`, `bartender_cfg.get`

#### `prepare_relaxation_job(case_dir: Path, case: Dict[str, Any], flow: Dict[str, str], pipeline_cfg: Dict[str, Any], base_dir: Path, exec_cfg: Dict[str, Any])` — line 173
- Write the per-case relaxation/MD work directory and run script.
- kind: function
- returns: `workdir`; `None`
- raises: `TypeError`, `ValueError`
- effects: filesystem
- calls: `workdir.mkdir`, `local_xyz.write_text`, `case.get`, `resolve_xtb_settings`, `resolve_orca_settings`, `parse_bool`, `resolve_executable_command`, `lines.extend`, `write_text`, `script_path.chmod`, `polymer_xyz.read_text`, `TypeError`, +16 more

#### `prepare_bartender_job(case_dir: Path, case: Dict[str, Any], flow: Dict[str, str], pipeline_cfg: Dict[str, Any], base_dir: Path, exec_cfg: Dict[str, Any])` — line 413
- Write the per-case Bartender (or trim-only) job directory and script.
- kind: function
- returns: `outdir`; `None`; `trim_dir`
- raises: `TypeError`, `FileNotFoundError`, `ValueError`
- effects: filesystem
- calls: `pipeline_cfg.get`, `case.get`, `outdir.mkdir`, `local_inp.write_text`, `os.path.relpath`, `resolve_executable_command`, `parse_bool`, `bartender_cfg.get`, `_bartender_mode_args`, `strip`, `script_lines.extend`, `shlex.quote`, +26 more

#### `find_case_json(start: Path)` — line 712
- Locate the nearest ``case.json`` at or above ``start``.
- kind: function
- returns: `None`; `candidate`
- calls: `start.resolve`, `candidate.exists`

#### `build_bead_maps(case: Dict[str, object], overrides: Dict[str, Dict[str, List[str]]])` — line 734
- Derive global bead label/type maps and backbone-bead set for a case.
- kind: function
- returns: `(label_map, type_map, backbone_beads)`
- raises: `ValueError`, `KeyError`
- calls: `case.get`, `ValueError`, `normalize_label_spec`, `KeyError`, `get`, `overrides.get`, `case_specs.get`, `backbone_beads.add`

#### `collect_results(root: Path, output: Path)` — line 799
- Summarize every Bartender ``gmx_out.itp`` under ``root`` into JSON.
- kind: function
- returns: `payload`
- effects: filesystem
- calls: `write_text`, `root.rglob`, `summarize_itp`, `find_case_json`, `records.append`, `json.dumps`, `json.loads`, `case_json.read_text`, `case.get`

#### `merge_results(root: Path, output_itp: Path, output_json: Path, label_map_path: Optional[Path])` — line 829
- Merge all Bartender ITPs under ``root`` into one typed force field.
- kind: function
- returns: `payload`
- effects: filesystem
- calls: `merge_records`, `write_merged_forcefield`, `merged_summary_payload`, `write_text`, `load_label_map`, `root.rglob`, `find_case_json`, `json.dumps`, `skipped.append`, `records.extend`, `typed_records_for_result`

#### `run_postprocess_only(cfg: Dict[str, Any])` — line 868
- Run only the screening postprocess stage on existing pipeline output.
- kind: function
- returns: `summary`
- raises: `ValueError`
- effects: filesystem
- calls: `resolve`, `summary_root.mkdir`, `write_text`, `get`, `ValueError`, `run_screening_postprocess`, `json.dumps`, `Path`, `post_cfg.get`

#### `run_pipeline(cfg: Dict[str, Any])` — line 905
- Run the full stage-03 preparation pipeline for every sequence.
- kind: function
- returns: `summary`
- raises: `KeyError`, `ValueError`
- effects: filesystem
- calls: `resolve`, `out_root.mkdir`, `resolve_pipeline_modes`, `resolve_execution_settings`, `resolve_log_settings`, `resolve_connection_detection_config`, `resolve_term_generation_config`, `pipeline_cfg.get`, `normalize_monomer_configs`, `load_monomer_library`, `build_sequence_jobs`, `get`, +29 more

### `hygel_martini/param_opt/qm_to_martini/postprocess.py`

Screening postprocess for Bartender ITP outputs (stage-03 final step).

#### class `ScreeningProcessor` — line 512
Parse Bartender ITP outputs and write full plus screened postprocess data.

##### `__init__(self, cfg: Dict[str, Any])` — line 524
- Resolve and normalize all screening settings from the config.
- kind: method
- calls: `cfg.get`, `get`, `self.post_cfg.get`, `self.screen_cfg.get`, `lower`, `self._normalize_bond_constraint_mode`, `self._normalize_candidate_source`, `strip`

##### `_normalize_bond_constraint_mode(raw: Any)` — line 564
- Map user aliases onto a canonical bond/constraint policy.
- kind: staticmethod, internal
- returns: `mode`
- raises: `ValueError`
- calls: `lower`, `aliases.get`, `ValueError`, `strip`

##### `_normalize_candidate_source(raw: Any)` — line 598
- Map user aliases onto a canonical candidate-source policy.
- kind: staticmethod, internal
- returns: `mode`
- raises: `ValueError`
- calls: `lower`, `aliases.get`, `ValueError`, `strip`

##### `_term_is_allowed_by_candidate_source(self, term: Dict[str, Any])` — line 628
- Check the term against the candidate-source policy ("all"/"active").
- kind: method, internal
- returns: `self._term_is_bartender_active(term)`; `True`
- calls: `self._term_is_bartender_active`

##### `_force_values_and_metric(self, section: str, funct: int, numeric_params: Sequence[float])` — line 634
- Extract force-constant values and reduce them to a scalar metric.
- kind: method, internal
- returns: `(values, metric, method)`; `([], None, method)`
- calls: `math.sqrt`

##### `_parse_itp_line(self, line: str, section: str, n_idx: int)` — line 691
- Parse one ITP data line (commented or active) into a term record.
- kind: method, internal
- returns: `{'indices': indices, 'funct': funct, 'params': params, 'numeric_params': numeric_params, 'force_values': force_values, 'force_metric': force_metric, 'force_metric_method': force_metric_method, 'rmsd': rmsd, 'commented': commented, 'raw': raw, 'section': section}`; `None`
- calls: `line.rstrip`, `raw.strip`, `stripped.startswith`, `content.startswith`, `main_part.split`, `self._force_values_and_metric`, `strip`, `content.split`, `RMSD_RE.search`, `isdigit`, `match.group`, `_parse_float`

##### `_parse_itp(self, itp_path: Path, out_root: Path)` — line 758
- Parse a whole ``gmx_out.itp`` into per-section term lists.
- kind: method, internal
- returns: `parsed`
- calls: `_find_case_json`, `splitlines`, `_relative_to_or_name`, `line.strip`, `self._parse_itp_line`, `append`, `json.loads`, `case_data.get`, `itp_path.read_text`, `stripped.startswith`, `stripped.endswith`, `lower`, +3 more

##### `_get_overlap_key(self, term: Dict[str, Any])` — line 804
- Return the (section, sorted indices) key used for overlap dedup.
- kind: method, internal
- returns: `(term['section'], _canon_indices(tuple(term['indices'])))`
- calls: `_canon_indices`

##### `_term_is_bartender_active(term: Dict[str, Any])` — line 809
- True when Bartender kept the line active (not commented out).
- kind: staticmethod, internal
- returns: `not bool(term['commented'])`

##### `_term_is_allowed_by_bond_constraint_mode(self, term: Dict[str, Any])` — line 813
- Apply the bond/constraint policy to one term's section.
- kind: method, internal
- returns: `True`; `False`

##### `_term_matches_preferred_potential(term: Dict[str, Any], preferred: Any)` — line 823
- Return true when a term matches the configured funct preference.
- kind: staticmethod, internal
- returns: `int(term['funct']) == preferred_funct`; `True`
- raises: `ValueError`
- calls: `lower`, `preferred.strip`, `ValueError`

##### `_threshold_for(self, section: str, funct: int, terms: Sequence[Dict[str, Any]])` — line 847
- Resolve the effective force-metric floor for a (section, funct).
- kind: method, internal
- returns: `raw_value`; `raw_value * max(metrics)`; `math.inf`
- raises: `ValueError`
- calls: `self.fc_min_cfg.get`, `ValueError`, `term.get`

##### `_screen_terms(self, all_terms: Dict[str, List[Dict[str, Any]]])` — line 884
- Apply the full screening pipeline per section.
- kind: method, internal
- returns: `screened_results`
- calls: `self.pref_potentials.get`, `valid_terms.sort`, `accepted.sort`, `all_terms.get`, `candidate_terms.append`, `self._threshold_for`, `valid_terms.append`, `self._get_overlap_key`, `accepted.append`, `occupied.add`, `self._term_is_allowed_by_bond_constraint_mode`, `self._term_is_allowed_by_candidate_source`, +1 more

##### `_info_terms(self, all_terms: Dict[str, List[Dict[str, Any]]])` — line 946
- Select which terms go into the inspection outputs (json/itp/plots).
- kind: method, internal
- returns: `info`; `all_terms`
- calls: `self.pref_potentials.get`, `all_terms.get`, `append`, `self._term_is_allowed_by_bond_constraint_mode`, `self._term_is_allowed_by_candidate_source`, `self._term_matches_preferred_potential`

##### `_output_dir_for_root(self, out_root: Path)` — line 968
- Resolve where the postprocess outputs for one root are written.
- kind: method, internal
- returns: `out_root / output_dir`; `output_root / _relative_to_or_name(out_root, mirror_root)`; `output_dir`
- calls: `Path`, `output_dir.is_absolute`, `self.paths_cfg.get`, `self.screen_cfg.get`, `resolve`, `_relative_to_or_name`

##### `_json_terms(terms: Iterable[Dict[str, Any]])` — line 990
- Strip the raw ITP line from term dicts for JSON serialization.
- kind: staticmethod, internal
- returns: `rows`
- calls: `rows.append`, `term.items`

##### `_write_all_terms_itp(self, path: Path, all_terms: Dict[str, List[Dict[str, Any]]])` — line 997
- Write the inspection ITP: every kept term with provenance comments.
- kind: method, internal
- effects: filesystem
- calls: `path.write_text`, `all_terms.get`, `lines.append`, `term.get`, `rstrip`, `join`

##### `_write_itp(self, path: Path, results: Dict[str, List[Dict[str, Any]]])` — line 1025
- Write the screened force field as an ITP with header settings.
- kind: method, internal
- effects: filesystem
- calls: `path.write_text`, `results.get`, `lines.append`, `join`, `term.get`, `rstrip`

##### `_write_plots(self, out_dir: Path, all_terms: Dict[str, List[Dict[str, Any]]], screened: Dict[str, List[Dict[str, Any]]])` — line 1054
- Write per-(section, funct) CSV tables and paginated PDF plots.
- kind: method, internal
- returns: `None` (bare return)
- effects: filesystem
- calls: `plot_dir.mkdir`, `_selected_key`, `defaultdict`, `all_terms.get`, `screened.values`, `append`, `grouped.items`, `self.pref_potentials.get`, `self._threshold_for`, `_chunk_rows`, `plot_dir.glob`, `csv_path.open`, +12 more

##### `process(self, out_root: Path)` — line 1164
- Run the full screening postprocess for one output root.
- kind: method
- returns: `report`
- effects: filesystem
- calls: `out_root.resolve`, `self._screen_terms`, `self._info_terms`, `self._output_dir_for_root`, `final_output_dir.mkdir`, `all_json_path.write_text`, `self._write_all_terms_itp`, `summary_path.write_text`, `self._write_itp`, `self._write_plots`, `write_text`, `out_root.rglob`, +5 more

#### `_as_list(value: Any)` — line 119
- Coerce a scalar / list / tuple / None config value into a list.
- kind: function, internal
- returns: `[value]`; `[]`; `list(value)`

#### `_parse_float(value: str)` — line 128
- Parse a float token, returning None for non-numeric input.
- kind: function, internal
- returns: `float(value)`; `None`

#### `_canon_indices(indices: Sequence[int])` — line 136
- Return the order-independent canonical form of a bead-index tuple.
- kind: function, internal
- returns: `tuple(sorted((int(value) for value in indices)))`

#### `_find_case_json(start: Path)` — line 141
- Locate the nearest ``case.json`` at or above ``start`` (max 8 levels).
- kind: function, internal
- returns: `None`; `candidate`
- calls: `start.resolve`, `candidate.exists`

#### `_relative_to_or_name(path: Path, base: Optional[Path])` — line 161
- Return ``path`` relative to ``base``, or just its basename.
- kind: function, internal
- returns: `Path(path.name)`; `path.resolve().relative_to(base.resolve())`
- calls: `Path`, `relative_to`, `base.resolve`, `path.resolve`

#### `_format_number(value: Optional[float])` — line 175
- Format a number compactly for plot labels ("NA" for None/non-finite).
- kind: function, internal
- returns: `f'{value:.3g}'`; `'NA'`; `'0'` (+2 more)
- calls: `math.isfinite`

#### `_chunk_sizes(total: int, limit: int=PLOT_MAX_POINTS)` — line 193
- Split ``total`` items into near-equal chunk sizes of at most ``limit``.
- kind: function, internal
- returns: `[base] * (chunk_count - remainder) + [base + 1] * remainder`; `[]`
- calls: `math.ceil`

#### `_chunk_rows(rows: Sequence[Dict[str, Any]], limit: int=PLOT_MAX_POINTS)` — line 213
- Partition term rows into plot pages of at most ``limit`` rows each.
- kind: function, internal
- returns: `chunks`
- calls: `_chunk_sizes`, `chunks.append`

#### `_potential_title(section: str, funct: int)` — line 223
- Look up plot annotations (name, equation, params) for a potential.
- kind: function, internal
- returns: `POTENTIAL_INFO.get((section, funct), (f'{section} potential', '$\\mathrm{equation\\ not\\ annotated}$', 'params: see ITP line'))`
- calls: `POTENTIAL_INFO.get`

#### `_axis_bounds(values: Sequence[float], threshold: Optional[float], *, zero_floor: bool=True)` — line 239
- Compute padded y-axis limits that include the data and the cutoff.
- kind: function, internal
- returns: `(lo, hi)`; `(0.0, 1.0)`
- calls: `math.isclose`, `math.isfinite`, `finite.append`

#### `_selected_key(row: Dict[str, Any])` — line 275
- Build the identity key used to match a plotted row to a screened term.
- kind: function, internal
- returns: `(str(row.get('source', '')), str(row.get('section', '')), tuple(row.get('indices', ())), int(row.get('funct', 0)))`
- calls: `row.get`

#### `_write_pdf_plot(path: Path, title: str, rows: Sequence[Dict[str, Any]], *, selected_keys: set[Tuple[str, str, Tuple[int, ...], int]], section: str, funct: int, force_threshold: Optional[float], rmsd_threshold: Optional[float], page_index: int=1, page_count: int=1, global_start_index: int=0, total_count: Optional[int]=None)` — line 289
- Render one screening-diagnostic PDF page for a (section, funct) group.
- kind: function, internal
- returns: `values`; `(f'cutoff = {_format_number(float(threshold))}', True, False)`; `_format_number(value)` (+3 more)
- raises: `RuntimeError`
- effects: filesystem
- calls: `_potential_title`, `plt.figure`, `fig.add_gridspec`, `fig.text`, `draw_panel`, `set_xticklabels`, `set_xlabel`, `fig.legend`, `path.parent.mkdir`, `fig.savefig`, `plt.close`, `matplotlib.use`, +37 more

#### `_resolve_postprocess_roots(cfg: Dict[str, Any])` — line 1241
- Collect the postprocess roots from ``paths`` config, deduplicated.
- kind: function, internal
- returns: `deduped`
- calls: `cfg.get`, `_as_list`, `paths_cfg.get`, `roots.extend`, `seen.add`, `deduped.append`, `roots.append`, `resolve`, `Path`, `glob.glob`

#### `run_screening_postprocess(cfg: Dict[str, Any])` — line 1269
- Run the screening postprocess over every configured output root.
- kind: function
- returns: `{'root_count': len(reports), 'outputs': reports}`
- raises: `ValueError`
- calls: `ScreeningProcessor`, `_resolve_postprocess_roots`, `ValueError`, `processor.process`

### `hygel_martini/param_opt/qm_to_martini/protocol/__init__.py`

Sealed, evidence-gated parameterization protocol.

### `hygel_martini/param_opt/qm_to_martini/protocol/__main__.py`

Module entry point for ``python -m ...protocol``.

### `hygel_martini/param_opt/qm_to_martini/protocol/cli.py`

Command-line interface for the sealed parameterization protocol.

#### `_parser()` — line 26
- *(no docstring)*
- kind: function, internal
- returns: `parser`
- calls: `argparse.ArgumentParser`, `parser.add_argument`, `parser.add_subparsers`, `subparsers.add_parser`, `init.add_argument`, `validate.add_argument`, `checksums.add_argument`, `seal.add_argument`, `evaluate.add_argument`, `status.add_argument`, `iteration.add_argument`

#### `_emit(payload: Dict[str, Any], output: Optional[Path])` — line 93
- *(no docstring)*
- kind: function, internal
- effects: filesystem
- calls: `sys.stdout.write`, `json.dumps`, `resolve`, `output.parent.mkdir`, `output.write_text`, `output.expanduser`

#### `main(argv: Optional[Sequence[str]]=None)` — line 102
- *(no docstring)*
- kind: function, CLI entry
- returns: `0`; `1`; `2`
- calls: `_parser`, `parser.parse_args`, `_emit`, `initialize_project`, `result.get`, `validate_project`, `refresh_checksums`, `seal_iteration`, `type`, `evaluate_evidence`, `project_status`, `new_iteration`, +1 more

### `hygel_martini/param_opt/qm_to_martini/protocol/engine.py`

State engine for a sealed, weakest-link parameterization protocol.

#### class `ProtocolError`(RuntimeError) — line 70
Raised when an operation would violate the frozen protocol.

#### `_root(path: Path)` — line 74
- Normalize a user-supplied project path (expanduser + resolve).
- kind: function, internal
- returns: `Path(path).expanduser().resolve()`
- calls: `resolve`, `expanduser`, `Path`

#### `_iteration_dir(root: Path, iteration_id: str)` — line 79
- Directory of one iteration under ``iterations/``.
- kind: function, internal
- returns: `root / 'iterations' / iteration_id`

#### `_seal_path(root: Path, iteration_id: str)` — line 84
- Path of an iteration's ``seal.json``.
- kind: function, internal
- returns: `_iteration_dir(root, iteration_id) / 'seal.json'`
- calls: `_iteration_dir`

#### `_contract_hash(contract: Mapping[str, Any])` — line 89
- SHA-256 of the whole contract in canonical JSON form.
- kind: function, internal
- returns: `sha256_bytes(canonical_json_bytes(contract))`
- calls: `sha256_bytes`, `canonical_json_bytes`

#### `_load_json(path: Path)` — line 94
- Read a JSON object file, wrapping failures in ``ProtocolError``.
- kind: function, internal
- returns: `payload`
- raises: `ProtocolError`
- calls: `json.loads`, `ProtocolError`, `path.read_text`

#### `_placeholder_text(name: str)` — line 105
- Content of a placeholder input file created by ``initialize_project``.
- kind: function, internal
- returns: `f'# PLACEHOLDER: replace {name} with the exact frozen input.\n# Then set placeholder: false in the contract and run hash-inputs.\n'`

#### `initialize_project(project_root: Path, *, project_id: str, title: str, claim_domain: str)` — line 113
- Create a non-overwriting project skeleton.
- kind: function
- returns: `{'decision': 'INITIALIZED', 'project_root': str(root), 'project_id': project_id, 'active_iteration': 'v001', 'ledger_event_hash': event['event_hash'], 'next_action': 'replace placeholders and freeze the prospective contract'}`
- raises: `ProtocolError`
- effects: filesystem
- calls: `_root`, `root.mkdir`, `protocol_template`, `contract_template`, `deepcopy`, `atomic_write_yaml`, `atomic_write_text`, `append_event`, `ID_RE.fullmatch`, `ProtocolError`, `root.exists`, `mkdir`, +6 more

#### `_artifact_rows(contract: Mapping[str, Any])` — line 184
- Materialize the contract's artifact specs (identity + data groups).
- kind: function, internal
- returns: `list(contract_artifacts(contract))`
- calls: `contract_artifacts`

#### `refresh_checksums(project_root: Path, *, iteration_id: Optional[str]=None, write: bool=False)` — line 189
- Compute checksums for explicitly accepted, project-local frozen inputs.
- kind: function
- returns: `{'decision': 'CHECKSUMS_WRITTEN' if write else 'CHECKSUMS_COMPUTED', 'iteration_id': iteration, 'artifact_count': len(results), 'artifacts': results}`
- raises: `ProtocolError`
- calls: `_root`, `load_yaml`, `load_project_documents`, `deepcopy`, `_artifact_rows`, `ProtocolError`, `exists`, `artifact.get`, `sha256_file`, `results.append`, `atomic_write_yaml`, `protocol.get`, +4 more

#### `_normalized_data_groups(contract: Mapping[str, Any])` — line 253
- Project each data group onto the fields that must match for TYPE_II.
- kind: function, internal
- returns: `normalized`
- calls: `contract.get`, `normalized.append`, `group.get`

#### `_validate_transition(root: Path, contract: Mapping[str, Any])` — line 274
- Enforce iteration-class boundaries against the parent contract.
- kind: function, internal
- returns: `errors`; `[f'parent contract does not exist: {parent_id!r}']`; `[f'parent contract is not a mapping: {parent_id!r}']`
- calls: `contract.get`, `load_yaml`, `parent_contract.get`, `parent.get`, `_iteration_dir`, `parent_contract_path.is_file`, `errors.append`, `_normalized_data_groups`

#### `_verify_seal(root: Path, iteration_id: str, contract: Mapping[str, Any], records: Optional[List[Mapping[str, Any]]]=None)` — line 320
- Verify an iteration's seal against the current contract and ledger.
- kind: function, internal
- returns: `(seal, errors)`; `(None, ['seal.json is absent'])`; `(None, [str(error)])`
- calls: `_seal_path`, `_contract_hash`, `scientific_identity_hash`, `seal.get`, `path.is_file`, `_load_json`, `errors.append`, `row.get`, `_artifact_rows`, `get`

#### `seal_iteration(project_root: Path, *, iteration_id: Optional[str]=None)` — line 379
- Freeze a complete prospective contract and its referenced artifacts.
- kind: function
- returns: `{**seal, 'decision': 'SEALED', 'ledger_event_hash': event['event_hash']}`; `{**dict(existing), 'decision': 'ALREADY_SEALED'}`
- raises: `ProtocolError`
- calls: `_root`, `load_yaml`, `load_project_documents`, `validate_protocol_document`, `validate_contract_document`, `_validate_transition`, `validate_ledger`, `_verify_seal`, `atomic_write_json`, `append_event`, `ProtocolError`, `errors.extend`, +11 more

#### `_decision_events(records: Sequence[Mapping[str, Any]], iteration_id: str)` — line 461
- Filter the ledger rows to one iteration's DECISION events, in order.
- kind: function, internal
- returns: `[row for row in records if row.get('iteration_id') == iteration_id and row.get('event_type') == 'DECISION']`
- calls: `row.get`

#### `_validate_evidence_artifacts(root: Path, artifacts: Any)` — line 470
- Validate and normalize the artifact list of one evidence file.
- kind: function, internal
- returns: `normalized`
- raises: `ProtocolError`
- calls: `ProtocolError`, `ids.add`, `sha256_file`, `normalized.append`, `artifact.get`, `SHA256_RE.fullmatch`, `safe_project_path`, `path.is_file`, `ID_RE.fullmatch`

#### `evaluate_evidence(project_root: Path, evidence_path: Path, *, commit: bool=False)` — line 516
- Evaluate one gate against a sealed contract; mutate only with ``commit``.
- kind: function
- returns: `decision`
- raises: `ProtocolError`
- calls: `_root`, `resolve`, `load_yaml`, `get`, `load_project_documents`, `validate_contract_document`, `validate_ledger`, `_verify_seal`, `_decision_events`, `evidence.get`, `rules.items`, `_validate_evidence_artifacts`, +18 more

#### `_validate_ledger_semantics(root: Path, records: List[Mapping[str, Any]], contracts: Mapping[str, Mapping[str, Any]])` — line 727
- Check ledger DECISION events against protocol semantics per iteration.
- kind: function, internal
- returns: `errors`
- calls: `contracts.items`, `events_for_iteration`, `errors.append`, `row.get`, `observed_gates.append`, `payload.get`, `_contract_hash`, `safe_project_path`, `artifact_path.is_file`, `evidence_path.is_file`, `sha256_file`, `artifact.get`

#### `validate_project(project_root: Path)` — line 798
- Validate schemas, artifacts, seals, ledger hashes, and state semantics.
- kind: function
- returns: `{'decision': 'PASS' if not errors else 'FAIL', 'project_root': str(root), 'project_id': protocol.get('project', {}).get('id'), 'active_iteration': protocol.get('active_iteration'), 'iterations': iteration_reports, 'ledger_rows': len(records), 'errors': errors, 'warnings': warnings}`; `{'decision': 'FAIL', 'project_root': str(root), 'errors': ['protocol.yaml is absent'], 'warnings': []}`; `{'decision': 'FAIL', 'project_root': str(root), 'errors': [str(error)], 'warnings': []}`
- calls: `_root`, `errors.extend`, `validate_ledger`, `protocol_path.is_file`, `load_yaml`, `validate_protocol_document`, `iteration_root.is_dir`, `errors.append`, `protocol.get`, `validate_contract_document`, `_validate_transition`, `warnings.extend`, +10 more

#### `project_status(project_root: Path)` — line 881
- Report the exact current gate and claim ceiling without changing state.
- kind: function
- returns: `{'decision': validation['decision'], 'project_id': protocol.get('project', {}).get('id'), 'active_iteration': iteration, 'state': state, 'next_gate': next_gate, 'terminal': terminal, 'completed_gates': [row['payload']['gate'] for row in decisions], 'opened_confirmation_group_ids': sorted(set(opened_confirmation)), 'claim_ceiling': contract.get('design', {}).get('claim_ceiling'), 'validation_error_count': len(validation['errors']), 'validation_warning_count': len(validation['warnings'])}`
- effects: filesystem
- calls: `_root`, `validate_project`, `load_yaml`, `validate_ledger`, `_decision_events`, `protocol.get`, `is_file`, `row.get`, `payload.get`, `get`, `_iteration_dir`, `opened_confirmation.extend`, +3 more

#### `new_iteration(project_root: Path, *, new_iteration_id: str, iteration_class: str, failure_mechanism: str, from_iteration_id: Optional[str]=None)` — line 952
- Fork a closed non-pass iteration without rewriting its terminal.
- kind: function
- returns: `{'decision': 'DRAFT_ITERATION_CREATED', 'iteration_id': new_iteration_id, 'parent_iteration_id': source_iteration, 'parent_terminal': last.get('action'), 'iteration_class': iteration_class, 'reclassified_opened_confirmation_ids': sorted(opened_confirmation), 'ledger_event_hash': event['event_hash'], 'next_action': 'edit the draft contract, refresh checksums, validate, and seal before opening evidence'}`
- raises: `ProtocolError`
- effects: filesystem
- calls: `_root`, `_iteration_dir`, `target_dir.exists`, `load_yaml`, `validate_ledger`, `_decision_events`, `deepcopy`, `contract.get`, `target_dir.mkdir`, `mkdir`, `atomic_write_yaml`, `append_event`, +14 more

### `hygel_martini/param_opt/qm_to_martini/protocol/io.py`

Small, deterministic I/O helpers used by the protocol engine.

#### `canonical_json_bytes(payload: Any)` — line 15
- Serialize a JSON-compatible object for stable hashing.
- kind: function
- returns: `json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')`
- calls: `encode`, `json.dumps`

#### `sha256_bytes(payload: bytes)` — line 27
- *(no docstring)*
- kind: function
- returns: `hashlib.sha256(payload).hexdigest()`
- calls: `hexdigest`, `hashlib.sha256`

#### `sha256_file(path: Path)` — line 31
- *(no docstring)*
- kind: function
- returns: `digest.hexdigest()`
- effects: filesystem
- calls: `hashlib.sha256`, `digest.hexdigest`, `path.open`, `iter`, `digest.update`, `handle.read`

#### `load_yaml(path: Path)` — line 39
- *(no docstring)*
- kind: function
- returns: `payload`
- raises: `ValueError`
- effects: filesystem
- calls: `path.open`, `yaml.safe_load`, `ValueError`

#### `atomic_write_text(path: Path, text: str)` — line 47
- Atomically replace one file without exposing a partial write.
- kind: function
- effects: filesystem
- calls: `path.parent.mkdir`, `tempfile.mkstemp`, `Path`, `os.replace`, `temporary.exists`, `os.fdopen`, `handle.write`, `handle.flush`, `os.fsync`, `temporary.unlink`, `handle.fileno`

#### `atomic_write_json(path: Path, payload: Any)` — line 66
- *(no docstring)*
- kind: function
- effects: filesystem
- calls: `atomic_write_text`, `json.dumps`

#### `atomic_write_yaml(path: Path, payload: Mapping[str, Any])` — line 73
- *(no docstring)*
- kind: function
- effects: filesystem
- calls: `yaml.safe_dump`, `atomic_write_text`

#### `safe_project_path(root: Path, relative: str)` — line 83
- Resolve a project-relative path and reject traversal outside the root.
- kind: function
- returns: `candidate`
- raises: `ValueError`
- calls: `resolve`, `root.resolve`, `ValueError`, `candidate.relative_to`, `relative.strip`

### `hygel_martini/param_opt/qm_to_martini/protocol/ledger.py`

Hash-chained, append-only verification ledger.

#### `_hash_event(event_without_hash: Mapping[str, Any])` — line 17
- *(no docstring)*
- kind: function, internal
- returns: `sha256_bytes(canonical_json_bytes(event_without_hash))`
- calls: `sha256_bytes`, `canonical_json_bytes`

#### `validate_records(records: List[Mapping[str, Any]])` — line 21
- *(no docstring)*
- kind: function
- returns: `errors`
- calls: `record.get`, `content.pop`, `_hash_event`, `errors.append`

#### `read_ledger(path: Path)` — line 43
- *(no docstring)*
- kind: function
- returns: `records`; `[]`
- raises: `ValueError`
- effects: filesystem
- calls: `path.exists`, `path.open`, `records.append`, `line.strip`, `json.loads`, `ValueError`

#### `validate_ledger(path: Path)` — line 61
- *(no docstring)*
- kind: function
- returns: `(records, validate_records(records))`; `([], [str(error)])`
- calls: `read_ledger`, `validate_records`

#### `append_event(path: Path, *, event_type: str, iteration_id: str, payload: Mapping[str, Any])` — line 69
- Append one event under an advisory lock and fsync it before returning.
- kind: function
- returns: `event`
- raises: `ValueError`
- effects: filesystem
- calls: `path.parent.mkdir`, `path.open`, `handle.seek`, `validate_records`, `_hash_event`, `handle.write`, `handle.flush`, `os.fsync`, `fcntl.flock`, `records.append`, `ValueError`, `isoformat`, +6 more

#### `events_for_iteration(records: List[Mapping[str, Any]], iteration_id: str)` — line 122
- *(no docstring)*
- kind: function
- returns: `[record for record in records if record.get('iteration_id') == iteration_id]`
- calls: `record.get`

#### `latest_decision(records: List[Mapping[str, Any]], iteration_id: str)` — line 128
- *(no docstring)*
- kind: function
- returns: `decisions[-1] if decisions else None`
- calls: `record.get`

### `hygel_martini/param_opt/qm_to_martini/protocol/schema.py`

Schema and frozen-criterion validation for parameterization contracts.

#### `_error(errors: List[str], location: str, message: str)` — line 57
- Append a "location: message" finding to the error list.
- kind: function, internal
- calls: `errors.append`

#### `_require_mapping(value: Any, errors: List[str], location: str)` — line 62
- Record an error and return {} unless the value is a mapping.
- kind: function, internal
- returns: `value`; `{}`
- calls: `_error`

#### `_require_nonempty_string(value: Any, errors: List[str], location: str)` — line 74
- Record an error and return "" unless the value is a non-blank str.
- kind: function, internal
- returns: `value.strip()`; `''`
- calls: `value.strip`, `_error`

#### `_validate_identifier(value: Any, errors: List[str], location: str)` — line 84
- Validate an id token against ``ID_RE`` (alnum start, [-._] allowed).
- kind: function, internal
- returns: `normalized`
- calls: `_require_nonempty_string`, `_error`, `ID_RE.fullmatch`

#### `_validate_artifact_spec(spec: Any, root: Path, errors: List[str], warnings: List[str], location: str, *, check_files: bool)` — line 92
- Validate one artifact spec (id, path, sha256, placeholder flag).
- kind: function, internal
- returns: `None` (bare return)
- calls: `_require_mapping`, `_require_nonempty_string`, `artifact.get`, `_error`, `warnings.append`, `SHA256_RE.fullmatch`, `safe_project_path`, `path.is_file`, `sha256_file`

#### `validate_protocol_document(payload: Any)` — line 144
- Validate the top-level ``protocol.yaml`` document.
- kind: function
- returns: `errors`
- calls: `_require_mapping`, `_validate_identifier`, `_require_nonempty_string`, `policy.get`, `document.get`, `_error`, `project.get`

#### `validate_contract_document(payload: Any, protocol: Mapping[str, Any], project_root: Path, expected_iteration: str, *, check_files: bool=True)` — line 189
- Validate one iteration's ``contract.yaml`` against the protocol.
- kind: function
- returns: `(errors, warnings)`
- calls: `_require_mapping`, `get`, `_validate_identifier`, `contract.get`, `_require_nonempty_string`, `design.get`, `_error`, `group_ids.add`, `item.get`, `_validate_artifact_spec`, `protocol.get`, `parent_map.get`, +5 more

#### `contract_artifacts(contract: Mapping[str, Any])` — line 402
- Yield every artifact spec a contract references.
- kind: function
- returns: `<generator>`
- calls: `contract.get`, `identity.get`

#### `scientific_identity_hash(contract: Mapping[str, Any])` — line 419
- SHA-256 of the canonical-JSON scientific-identity block.
- kind: function
- returns: `sha256_bytes(canonical_json_bytes(contract.get('scientific_identity', {})))`
- calls: `sha256_bytes`, `canonical_json_bytes`, `contract.get`

#### `evaluate_rule(rule: Mapping[str, Any], observed: Any)` — line 428
- Evaluate one frozen criterion without importing a post-result threshold.
- kind: function
- returns: `'PASS' if passed else 'FAIL'`; `'INCONCLUSIVE'`; `normalized`
- raises: `ValueError`
- calls: `rule.get`, `upper`, `ValueError`, `strip`, `math.isfinite`, `observed.strip`

#### `load_project_documents(project_root: Path, iteration_id: str)` — line 488
- Load a project's protocol.yaml and one iteration's contract.yaml.
- kind: function
- returns: `(protocol, contract)`
- raises: `ValueError`
- calls: `load_yaml`, `ValueError`

### `hygel_martini/param_opt/qm_to_martini/protocol/templates.py`

Built-in project skeletons for the evidence-gated protocol.

#### `protocol_template(project_id: str, title: str, claim_domain: str)` — line 32
- Build the initial ``protocol.yaml`` payload.
- kind: function
- returns: `{'schema_version': SCHEMA_VERSION, 'project': {'id': project_id, 'title': title, 'claim_domain': claim_domain}, 'active_iteration': 'v001', 'policy': {'gate_order': list(GATE_ORDER), 'strict_sequence': True, 'weakest_link': True, 'e6_parameter_feedback': 'prohibited', 'max_correction_iterations_per_mechanism': 2}}`

#### `_artifact(artifact_id: str, path: str)` — line 63
- Build a placeholder artifact spec (zero digest, placeholder true).
- kind: function, internal
- returns: `{'id': artifact_id, 'path': path, 'sha256': PLACEHOLDER_SHA256, 'placeholder': True}`

#### `_criterion(criterion_id: str, description: str, *, operator: str='truthy', expected: Any=None, on_fail: str, on_inconclusive: str='DATA_LIMITED')` — line 73
- Build one frozen-criterion mapping for the contract template.
- kind: function, internal
- returns: `criterion`

#### `contract_template(project_id: str)` — line 108
- Return a conservative contract that must be customized before sealing.
- kind: function
- returns: `{'schema_version': SCHEMA_VERSION, 'project_id': project_id, 'iteration_id': 'v001', 'parent': None, 'iteration_class': 'TYPE_I', 'failure_mechanism_addressed': 'INITIAL_REGISTERED_CANDIDATE_LADDER', 'scientific_identity': {'mapping': _artifact('mapping', 'inputs/mapping.yaml'), 'topology_graph': _artifact('topology_graph', 'inputs/topology_graph.yaml'), 'bead_model': _artifact('bead_model', 'inputs/bead_model.yaml'), 'nonbonded_parent': _artifact('nonbonded_parent', 'inputs/nonbonded_parent.yaml'), 'exclusions': _artifact('exclusions', 'inputs/exclusions.yaml')}, 'data_groups': [{**_artifact('development_groups', 'data/development_groups.tsv'), 'role': 'development', 'sealed': False}, {**_artifact('validation_groups', 'data/validation_groups.tsv'), 'role': 'validation', 'sealed': False}, {**_artifact('stress_groups', 'data/stress_groups.tsv'), 'role': 'stress', 'sealed': False}, {**_artifact('confirmation_groups', 'data/confirmation_groups.tsv'), 'role': 'confirmation', 'sealed': True}], 'design': {'coordinate': 'REPLACE_WITH_BOND_ANGLE_DIHEDRAL_OR_COMPLETE_TOPOLOGY', 'predecessor': 'EXPLICIT_OMISSION_OR_FROZEN_UPSTREAM_TOPOLOGY', 'candidate_ladder': ['omission', 'REPLACE_WITH_REGISTERED_FUNCTION'], 'maximum_complexity': 1, 'primary_objective': 'REPLACE_WITH_FAMILY_GROUPED_PRIMARY_OBJECTIVE', 'sensitivity_objectives': ['REPLACE_WITH_REGISTERED_SENSITIVITY'], 'grouping_unit': 'independent_start_family', 'stop_rule': 'weakest_link; at most two corrections for one unchanged mechanism', 'claim_ceiling': 'tested-domain bonded-topology release; no universal transfer claim'}, 'gates': deepcopy(gates), 'permitted_repairs': ['parser or sign correction with unchanged scientific identity', 'runtime repair with identical frozen inputs, seeds, and task manifest'], 'prohibited_after_seal': ['mapping, graph, bead, nonbonded, or exclusion change', 'candidate, function, grouping, objective, threshold, or stop-rule change', 'data-role reassignment or premature confirmation access', 'discarding an unfavorable required task or resample']}`
- calls: `deepcopy`, `_artifact`, `_criterion`

### `hygel_martini/param_opt/qm_to_martini/tools/__init__.py`

*(no module docstring)*

### `hygel_martini/param_opt/qm_to_martini/workflow_logic/__init__.py`

*(no module docstring)*

### `hygel_martini/param_opt/qm_to_martini/workflow_logic/builder.py`

*(no module docstring)*

#### `_parse_main_itp(itp_path: Path)` — line 22
- Parse {label}_topology_n0_main_bonds.itp into section → list of index tuples.
- kind: function, internal
- returns: `result`
- effects: filesystem
- calls: `open`, `strip`, `re.match`, `line.split`, `section_map.get`, `lower`, `append`, `raw.split`, `indices.append`, `m.group`, `_section_arity`

#### `_parse_candidates_tsv(tsv_path: Path)` — line 61
- Parse *_force_sorted_candidates.tsv into section → list of index tuples.
- kind: function, internal
- returns: `result`
- effects: filesystem
- calls: `open`, `csv.DictReader`, `lower`, `row.get`, `_section_arity`, `append`, `indices_str.split`

#### `_section_arity(section: str)` — line 83
- *(no docstring)*
- kind: function, internal
- returns: `{'bonds': 2, 'constraints': 2, 'angles': 3, 'dihedrals': 4, 'impropers': 4}.get(section, 2)`
- calls: `get`

#### `_map_terms_to_global(terms_by_section: Dict[str, List[Tuple[int, ...]]], n_monomers: int, bead_count_per_monomer: int)` — line 87
- Replicate monomer-local terms for each monomer using sequential bead offsets.
- kind: function, internal
- returns: `result`
- calls: `terms_by_section.items`, `append`

#### `_term_spans_connection(term: Sequence[int], connection_bond_set: Set[Tuple[int, int]])` — line 102
- Return True if at least one consecutive edge pair in term is a connection bond.
- kind: function, internal
- returns: `False`; `True`
- calls: `_sorted_pair`

#### `_connection_proxy_count(indices: Sequence[int], backbone_beads: set[int])` — line 114
- *(no docstring)*
- kind: function, internal
- returns: `len({int(index) for index in indices if int(index) in backbone_beads})`

#### `_filter_connection_proxy_terms(terms: Sequence[Tuple[int, ...]], backbone_beads: set[int], minimum_distinct: int=2)` — line 117
- *(no docstring)*
- kind: function, internal
- returns: `[tuple((int(value) for value in term)) for term in terms if _connection_proxy_count(term, backbone_beads) >= minimum_distinct]`
- calls: `_connection_proxy_count`

#### `_distance_cache(graph: Dict[int, set[int]])` — line 128
- *(no docstring)*
- kind: function, internal
- returns: `lookup`; `cache[key]`
- calls: `_sorted_pair`, `shortest_path_len`

#### `_topology_reference_cost(section: str, indices: Sequence[int], distance_lookup)` — line 140
- *(no docstring)*
- kind: function, internal
- returns: `total`; `None`
- raises: `ValueError`
- calls: `distance_lookup`, `ValueError`

#### `_changed_index_count(term: Sequence[int], reference: Sequence[int])` — line 166
- *(no docstring)*
- kind: function, internal
- returns: `max(0, changed - 1)`

#### `_topology_term_cost(section: str, term: Sequence[int], distance_lookup, *, allow_swaps: bool)` — line 170
- *(no docstring)*
- kind: function, internal
- returns: `best`; `direct_cost`
- calls: `_topology_reference_cost`, `permutations`, `_changed_index_count`

#### `_filter_topology_terms(section: str, terms: Sequence[Tuple[int, ...]], graph: Dict[int, set[int]], budget: int, *, allow_swaps: bool)` — line 192
- *(no docstring)*
- kind: function, internal
- returns: `filtered`
- calls: `_distance_cache`, `_topology_term_cost`, `filtered.append`

#### `_generate_all_linkage_bonds(inp_data: MonomerTemplate)` — line 208
- *(no docstring)*
- kind: function, internal
- returns: `[tuple((int(value) for value in candidate)) for candidate in _generate_all_reversible_combinations(bead_ids, 2, existing)]`
- calls: `inp_data.beads.keys`, `_sorted_pair`, `_generate_all_reversible_combinations`

#### `_generate_all_linkage_angles(inp_data: MonomerTemplate)` — line 213
- *(no docstring)*
- kind: function, internal
- returns: `[tuple((int(value) for value in candidate)) for candidate in _generate_all_reversible_combinations(bead_ids, 3, existing)]`
- calls: `inp_data.beads.keys`, `_canon_angle`, `_generate_all_reversible_combinations`

#### `_generate_all_linkage_dihedrals(inp_data: MonomerTemplate)` — line 221
- *(no docstring)*
- kind: function, internal
- returns: `[tuple((int(value) for value in candidate)) for candidate in _generate_all_reversible_combinations(bead_ids, 4, existing)]`
- calls: `inp_data.beads.keys`, `_canon_reversible`, `_generate_all_reversible_combinations`

#### `_generate_all_linkage_impropers(inp_data: MonomerTemplate)` — line 229
- *(no docstring)*
- kind: function, internal
- returns: `[tuple((int(value) for value in candidate)) for candidate in _generate_all_reversible_combinations(bead_ids, 4, existing)]`
- calls: `inp_data.beads.keys`, `_canon_reversible`, `_generate_all_reversible_combinations`

#### `_generate_augmented_terms(base: MonomerTemplate, term_cfg: TermGenerationConfig, backbone_beads: Sequence[int], connection_bond_set: Optional[Set[Tuple[int, int]]]=None)` — line 237
- *(no docstring)*
- kind: function, internal
- returns: `(new_bonds, new_angles, new_dihedrals, new_impropers)`; `([], [], [], [])`
- calls: `_generate_all_linkage_bonds`, `_generate_all_linkage_angles`, `_generate_all_linkage_dihedrals`, `_generate_all_linkage_impropers`, `_build_graph`, `_filter_topology_terms`, `_filter_connection_proxy_terms`, `_term_spans_connection`

#### `format_inp(template: MonomerTemplate)` — line 292
- *(no docstring)*
- kind: function
- returns: `'\n'.join(lines) + '\n'`
- calls: `lines.append`, `template.beads.items`, `lines.extend`, `join`, `ref.format`

#### `build_polymer_input(sequence: Sequence[str] | str, polymer_xyz_path: Path, templates: Dict[str, MonomerTemplate], metadata, term_cfg: TermGenerationConfig)` — line 317
- *(no docstring)*
- kind: function
- returns: `PolymerInputBundle(base=base, augmented=augmented, base_text=format_inp(base), augmented_text=format_inp(augmented), base_report=report, augmented_report=augmented_report, connection_bonds=connection_bonds, connection_beads=sorted(set(connection_beads)), backbone_beads=sorted(set(backbone_beads)))`
- raises: `KeyError`
- calls: `normalize_sequence`, `parse_xyz`, `OrderedDict`, `ValidationReport`, `MonomerTemplate`, `validate_generated_input`, `report.problems.extend`, `report.warnings.extend`, `_generate_augmented_terms`, `PolymerInputBundle`, `template.beads.items`, `bonds.extend`, +36 more

### `hygel_martini/param_opt/qm_to_martini/workflow_logic/core.py`

Small geometry/graph/combinatorics primitives for workflow_logic.

#### `_distance(a: Tuple[float, float, float], b: Tuple[float, float, float])` — line 26
- Euclidean distance between two points (units follow the input).
- kind: function, internal
- returns: `math.sqrt(sum(((x - y) ** 2 for x, y in zip(a, b))))`
- calls: `math.sqrt`

#### `_split_csv(raw: str)` — line 30
- Split a comma-separated string into stripped, non-empty tokens.
- kind: function, internal
- returns: `[token.strip() for token in re.split('\\s*,\\s*', raw.strip()) if token.strip()]`
- calls: `token.strip`, `re.split`, `raw.strip`

#### `_sorted_pair(a: int, b: int)` — line 34
- Canonical (ascending) form of an undirected bond pair.
- kind: function, internal
- returns: `(a, b) if a <= b else (b, a)`

#### `_canon_angle(i: int, j: int, k: int)` — line 38
- Canonical angle triple: outer beads ordered, center bead fixed.
- kind: function, internal
- returns: `(i, j, k) if i <= k else (k, j, i)`

#### `_canon_reversible(values: Sequence[int])` — line 42
- Canonical form of any reversal-symmetric index tuple.
- kind: function, internal
- returns: `forward if forward <= reverse else reverse`
- calls: `reversed`

#### `_build_graph(edges: Iterable[Tuple[int, int]])` — line 52
- Build an undirected adjacency map from (a, b) edge pairs.
- kind: function, internal
- returns: `graph`
- calls: `defaultdict`, `add`

#### `shortest_path_len(graph: Dict[int, set[int]], start: int, goal: int)` — line 60
- Return the BFS shortest-path edge count between two nodes.
- kind: function
- returns: `None`; `0`; `dist + 1`
- calls: `deque`, `queue.popleft`, `graph.get`, `seen.add`, `queue.append`

#### `_reversal_unique_permutations(values: Sequence[int])` — line 86
- Enumerate all orderings of ``values``, one per reversal class.
- kind: function, internal
- returns: `unique`
- calls: `permutations`, `unique.sort`, `_canon_reversible`, `seen.add`, `unique.append`

#### `_generate_all_reversible_combinations(bead_ids: Sequence[int], body_size: int, existing: set[Tuple[int, ...]])` — line 106
- Generate every new candidate term of ``body_size`` beads.
- kind: function, internal
- returns: `generated`
- calls: `combinations`, `_reversal_unique_permutations`, `generated.append`, `seen.add`

### `hygel_martini/param_opt/qm_to_martini/workflow_logic/loader.py`

Bartender ``.inp`` template parsing, validation, and connection inference.

#### `_parse_weighted_atom(token: str)` — line 57
- Parse one BEADS atom token, e.g. "12" or "12/2" (shared atom).
- kind: function, internal
- returns: `WeightedAtomRef(atom_index=int(match.group(1)), denominator=denominator)`
- raises: `ValueError`
- calls: `re.fullmatch`, `WeightedAtomRef`, `ValueError`, `match.group`

#### `_parse_section_ints(path: Path, line: str, expected: int)` — line 77
- Parse one comma-separated bonded-term line into exactly N bead ids.
- kind: function, internal
- returns: `values`
- raises: `ValueError`
- calls: `ValueError`, `_split_csv`

#### `parse_bartender_inp(path: Path)` — line 93
- Parse a Bartender ``.inp`` file into a ``MonomerTemplate``.
- kind: function
- returns: `MonomerTemplate(path=path, preamble=preamble, beads=beads, bonds=[(int(a), int(b)) for a, b in bonds], constraints=[(int(a), int(b)) for a, b in constraints], angles=[(int(a), int(b), int(c)) for a, b, c in angles], dihedrals=[(int(a), int(b), int(c), int(d)) for a, b, c, d in dihedrals], impropers=[(int(a), int(b), int(c), int(d)) for a, b, c, d in impropers])`
- raises: `ValueError`
- calls: `splitlines`, `OrderedDict`, `MonomerTemplate`, `raw.strip`, `stripped.upper`, `append`, `ValueError`, `re.match`, `path.read_text`, `preamble.append`, `stripped.startswith`, `match.group`, +3 more

#### `_weighted_atom_owners(template: MonomerTemplate)` — line 167
- Invert the bead mapping: atom index -> its references across beads.
- kind: function, internal
- returns: `owners`
- calls: `defaultdict`, `template.beads.values`, `append`

#### `_connector_indices(symbols: Sequence[str], indicator: str)` — line 179
- Return 1-based indices of atoms whose element symbol is ``indicator``.
- kind: function, internal
- returns: `[index for index, symbol in enumerate(symbols, start=1) if str(symbol).strip().upper() == marker]`
- calls: `upper`, `indicator.strip`, `strip`

#### `infer_backbone_beads(template: MonomerTemplate, xyz_path: Path, backbone_atom_cfg: Dict[str, List[int]])` — line 184
- Map configured backbone atoms onto the beads that contain them.
- kind: function
- returns: `tuple(backbone_beads)`; `()`
- raises: `ValueError`
- calls: `_weighted_atom_owners`, `template.beads.items`, `backbone_atom_cfg.get`, `ValueError`, `backbone_beads.append`, `seen_beads.add`

#### `validate_template(template: MonomerTemplate, xyz_path: Path, connection_cfg: ConnectionDetectionConfig)` — line 236
- Validate a monomer template against its XYZ file.
- kind: function
- returns: `report`
- calls: `parse_xyz`, `ValidationReport`, `_weighted_atom_owners`, `owners.items`, `_connector_indices`, `report.problems.append`, `template.beads.keys`, `add`, `next`, `deque`, `Fraction`, `join`, +7 more

#### `infer_connection_metadata(template: MonomerTemplate, xyz_path: Path, connection_cfg: ConnectionDetectionConfig, backbone_atom_cfg: Dict[str, List[int]])` — line 334
- Infer which connector atoms/beads are the monomer's head and tail.
- kind: function
- returns: `ConnectionMetadata(head_carbon=head_carbon, tail_carbon=tail_carbon, head_br=head_br, tail_br=tail_br, left_connection_bead=owner(head_br, 'head connector'), right_connection_bead=owner(tail_br, 'tail connector'), backbone_beads=infer_backbone_beads(template, xyz_path, backbone_atom_cfg))`; `min((_distance(coords[ref - 1], coords[connector_atom - 1]) for ref in refs))`; `owners[0]`
- raises: `ValueError`
- calls: `parse_xyz`, `_connector_indices`, `ConnectionMetadata`, `backbone_atom_cfg.get`, `ValueError`, `distance_to_refs`, `candidates.sort`, `owner`, `infer_backbone_beads`, `_distance`, `template.beads.items`

#### `default_bead_spec(token: str, bead_count: int)` — line 475
- Build the default bead labels/types for a monomer: token+1..N.
- kind: function
- returns: `{'labels': labels, 'types': list(labels)}`

#### `build_bead_maps(case: Dict[str, object], overrides: Dict[str, Dict[str, List[str]]])` — line 480
- Derive global bead label/type maps and backbone-bead set for a case.
- kind: function
- returns: `(label_map, type_map, backbone_beads)`
- raises: `ValueError`, `KeyError`
- calls: `case.get`, `ValueError`, `normalize_label_spec`, `KeyError`, `get`, `overrides.get`, `case_specs.get`, `backbone_beads.add`

#### `validate_generated_input(template: MonomerTemplate, xyz_path: Path, terminal_cap_indices: List[int])` — line 546
- Validate a generated polymer MonomerTemplate against its XYZ file.
- kind: function
- returns: `report`
- calls: `parse_xyz`, `ValidationReport`, `_weighted_atom_owners`, `owners.items`, `report.problems.append`, `template.beads.keys`, `Fraction`

### `hygel_martini/param_opt/qm_to_martini/workflow_logic/merger.py`

Bartender ITP parsing and cross-case merging into a typed force field.

#### `split_main_and_comment(raw: str)` — line 47
- Split an ITP line into (data part, inline comment).
- kind: function
- returns: `(stripped.strip(), '')`; `(main.strip(), comment.strip())`
- calls: `raw.lstrip`, `stripped.startswith`, `lstrip`, `stripped.split`, `stripped.strip`, `main.strip`, `comment.strip`

#### `parse_param_line(raw: str, section: str, n_idx: int)` — line 62
- Parse one ITP data line (commented or active) into a ``ParamLine``.
- kind: function
- returns: `ParamLine(section=section, indices=indices, tokens=tuple(parts[n_idx:]), commented=stripped.startswith(';'), inline_comment=comment, rmsd=rmsd, raw=raw.rstrip('\n'))`; `None`
- calls: `raw.strip`, `split_main_and_comment`, `main.split`, `ParamLine`, `RMSD_RE.search`, `isdigit`, `stripped.startswith`, `raw.rstrip`, `match.group`

#### `parse_gmx_out_itp(path: Path)` — line 111
- Parse a Bartender ITP into per-section ``ParamLine`` lists.
- kind: function
- returns: `parsed`
- calls: `splitlines`, `raw.strip`, `parse_param_line`, `path.read_text`, `stripped.startswith`, `stripped.endswith`, `lower`, `header_map.get`, `append`, `strip`, `stripped.strip`

#### `summarize_itp(path: Path)` — line 155
- Summarize one ITP as a JSON-friendly dict of per-section term lists.
- kind: function
- returns: `{'path': str(path), 'counts': {section: len(lines) for section, lines in parsed.items()}, 'bonds': [_payload(line) for line in parsed['bonds']], 'constraints': [_payload(line) for line in parsed['constraints']], 'angles': [_payload(line) for line in parsed['angles']], 'dihedrals': [_payload(line) for line in parsed['dihedrals']], 'impropers': [_payload(line) for line in parsed['impropers']]}`; `payload`
- calls: `parse_gmx_out_itp`, `_payload`, `parsed.items`

#### `choose_best_rmsd_uncomment(lines: List[ParamLine])` — line 190
- Keep only the best-RMSD line active per identical index tuple.
- kind: function
- returns: `updated`
- calls: `defaultdict`, `grouped.values`, `append`, `math.isinf`, `ParamLine`

#### `typed_records_for_result(itp_path: Path, case_path: Path, label_overrides: Dict[str, Dict[str, List[str]]])` — line 232
- Convert one case's ITP terms into bead-type keyed ``TypedRecord``s.
- kind: function
- returns: `records`; `'WITH_BACKBONE' if any((index in backbone_beads for index in indices)) else 'WITHOUT_BACKBONE'`; `(display, types)`
- raises: `KeyError`
- calls: `json.loads`, `parse_gmx_out_itp`, `build_bead_maps`, `_build_graph`, `choose_best_rmsd_uncomment`, `case_path.read_text`, `map_labels`, `shortest_path_len`, `records.append`, `case.get`, `TypedRecord`, `_sorted_pair`, +2 more

#### `merge_records(records: List[TypedRecord])` — line 371
- Group typed records across cases and rank the parameter variants.
- kind: function
- returns: `merged`; `(0 if not sample.commented else 1, 0, 0.0, sample.source_tag)`; `(0 if not sample.commented else 1, 0 if item['rmsd_values'] else 1, rmsd, sample.source_tag)`
- calls: `defaultdict`, `grouped.items`, `append`, `variants_by_signature.values`, `items.append`, `variants.append`, `MergedVariant`, `record.inline_comment.strip`

#### `_format_type_names(type_names: Tuple[str, ...], widths: Tuple[int, ...])` — line 444
- Left-pad each type name to its column width for aligned ITP output.
- kind: function, internal
- returns: `' '.join((f'{value:<{width}}' for value, width in zip(type_names, widths)))`
- calls: `join`

#### `line_from_variant(variant: MergedVariant)` — line 448
- Render one merged variant as an ITP line with provenance comments.
- kind: function
- returns: `main + (' ; ' + ' ; '.join(comment_parts) if comment_parts else '')`
- calls: `rstrip`, `comment_parts.append`, `join`, `_format_type_names`

#### `write_merged_forcefield(path: Path, merged: Dict[Tuple[str, str, str, Tuple[str, ...]], List[MergedVariant]], root: Path, label_map_path: Optional[Path])` — line 472
- Write the merged force field as a ``*types`` ITP file.
- kind: function
- effects: filesystem
- calls: `write_text`, `lines.append`, `rstrip`, `join`, `line_from_variant`

#### `merged_summary_payload(root: Path, merged: Dict[Tuple[str, str, str, Tuple[str, ...]], List[MergedVariant]], skipped: List[Dict[str, str]])` — line 528
- Build the JSON-friendly summary payload for a merged force field.
- kind: function
- returns: `{'root': str(root), 'group_count': len(groups), 'groups': groups, 'skipped': skipped}`
- calls: `merged.items`, `groups.append`, `next`

### `hygel_martini/param_opt/qm_to_opls/__init__.py`

Stage 01 workflow package: QM-side inputs to OPLS preparation.

### `hygel_martini/param_opt/qm_to_opls/__main__.py`

Module runner: ``python -m param_opt.qm_to_opls`` delegates to cli.main.

### `hygel_martini/param_opt/qm_to_opls/ase_utils.py`

ASE-based file format helpers for the qm_to_opls workflow.

#### `xyz_to_pdb(xyz_path: str | Path, pdb_path: str | Path)` — line 15
- Convert an XYZ file to a PDB file using ASE read/write.
- kind: function
- effects: stdout
- calls: `read`, `write`, `print`

### `hygel_martini/param_opt/qm_to_opls/cli.py`

Command-line entry point for the stage-01 (QM -> OPLS) workflow.

#### `main()` — line 21
- Run the qm_to_opls CLI.
- kind: function, CLI entry
- returns: `None` (bare return)
- effects: filesystem, stdout
- calls: `argparse.ArgumentParser`, `add_config_args`, `parser.parse_args`, `Path`, `run_qm_to_opls`, `config_path.write_text`, `print`, `json.dumps`

### `hygel_martini/param_opt/qm_to_opls/defaults.py`

Default configuration for the stage-01 (QM -> OPLS) workflow.

### `hygel_martini/param_opt/qm_to_opls/generator.py`

Thin workflow entry helper for QM-to-OPLS preparation.

#### `run_qm_to_opls(config_path: str | Path)` — line 18
- Load a qm_to_opls config file and generate ORCA preparation inputs.
- kind: function
- calls: `load_config`, `generate_orca_inputs`, `Path`

### `hygel_martini/param_opt/qm_to_opls/ligpargen_api.py`

LigParGen submission stubs for the qm_to_opls workflow.

#### `submit_to_ligpargen(pdb_path: str | Path, output_dir: str | Path, name: str='molecule')` — line 19
- Create placeholder OPLS-AA outputs for a PDB submission.
- kind: function
- returns: `str(itp_output)`
- effects: filesystem, stdout
- calls: `Path`, `output_dir.mkdir`, `print`, `itp_output.write_text`, `gro_output.write_text`

#### `run_parameterization_flow(xyz_path: str | Path, out_root: str | Path, symbol: str)` — line 48
- High-level flow: XYZ -> PDB -> LigParGen -> ITP/GRO.
- kind: function
- returns: `itp_path`
- effects: filesystem, stdout
- calls: `temp_dir.mkdir`, `xyz_to_pdb`, `submit_to_ligpargen`, `print`, `Path`

### `hygel_martini/param_opt/qm_to_opls/orca_runner.py`

ORCA input generation for stage 01 (QM -> OPLS preparation).

#### `generate_orca_inputs(cfg: Dict[str, Any])` — line 23
- Generate ORCA optimization inputs for all monomer combinations.
- kind: function
- effects: filesystem, stdout
- calls: `resolve`, `out_root.mkdir`, `load_monomer_library`, `cfg.get`, `monomer_dict.keys`, `itertools.product`, `Path`, `_sequence_output_stem`, `_build_atoms_for_dft`, `_write_orca_file`, `print`, `get`

#### `_build_atoms_for_dft(sequence, monomer_dict, n_torsion)` — line 91
- Build an H-capped N-mer geometry as ASE Atoms.
- kind: function, internal
- returns: `cap_ends_with_hydrogen(chain)`
- calls: `copy`, `get_connection_info`, `cap_ends_with_hydrogen`, `new_monomer.rotate`, `new_monomer.translate`, `np.linalg.norm`

#### `_write_orca_file(name, atoms, dft_cfg, out_root)` — line 150
- Write one ORCA optimization input and its capped XYZ.
- kind: function, internal
- effects: filesystem
- calls: `target_dir.mkdir`, `write_text`, `write`, `dft_cfg.get`

## `hygel_martini/property_extract`

### `hygel_martini/property_extract/__init__.py`

Public API of :mod:`hygel_martini.property_extract`.

### `hygel_martini/property_extract/__main__.py`

CLI entry point: ``python -m hygel_martini.property_extract``.

#### `_build_parser()` — line 30
- Build the argument parser with all subcommands and legacy flags.
- kind: function, internal
- returns: `parser`
- calls: `argparse.ArgumentParser`, `parser.add_argument`, `parser.add_subparsers`, `subparsers.add_parser`, `ana.add_argument`, `req.add_argument`, `man.add_argument`, `topology.add_argument`, `mechanics.add_argument`, `clearance.add_argument`

#### `_legacy_main(args, parser)` — line 124
- Handle the legacy no-subcommand path (Phase-A ``--config`` mode).
- kind: function, internal
- returns: `p if os.path.isabs(str(p)) else os.path.join(cfg_dir, str(p))`; `None`
- effects: filesystem, stdout
- calls: `print`, `os.path.dirname`, `HydrogelAnalyzer.from_config`, `cfg.get`, `resolve`, `analyzer.analyze`, `analyzer.report`, `results.get`, `run_gmx`, `parser.error`, `open`, `yaml.safe_load`, +7 more

#### `_analyze_main(args)` — line 195
- Run the ``analyze`` subcommand: execute jobs and print results.
- kind: function, internal
- effects: stdout
- calls: `run_analysis`, `print`, `results.items`, `load_manifest`, `find_target_properties_for_simulation_property`, `os.path.abspath`, `upper`, `pr.metadata.get`, `pr.status.replace`

#### `_requirements_main(args)` — line 268
- Run the ``requirements`` subcommand: report the MD gate per job.
- kind: function, internal
- effects: stdout
- calls: `load_analysis_jobs`, `print`, `os.path.join`, `job.resolved_inputs`, `check_requirements`, `sys.exit`, `os.path.abspath`, `join`

#### `_manifest_main(args)` — line 315
- Run the ``manifest`` subcommand: show target mappings for a property.
- kind: function, internal
- effects: stdout
- calls: `load_manifest`, `find_target_properties_for_simulation_property`, `print`, `sys.exit`

#### `_write_json_result(payload, output_path=None)` — line 341
- Serialize a payload as sorted, indented JSON to a file or stdout.
- kind: function, internal
- effects: filesystem, stdout
- calls: `json.dumps`, `os.path.abspath`, `os.makedirs`, `print`, `os.path.dirname`, `open`, `handle.write`

#### `_topology_main(args)` — line 356
- Run the ``topology`` subcommand via the reduced-network extractor.
- kind: function, internal
- raises: `SystemExit`
- calls: `compute`, `_write_json_result`, `result.to_dict`, `SystemExit`, `ReducedNetworkTopologyExtractor`

#### `_mechanics_step_main(args)` — line 379
- Run the ``mechanics-step`` subcommand: paired-step XVG summary.
- kind: function, internal
- calls: `paired_step_xvg_summary`, `_write_json_result`

#### `_clearance_frame_main(args)` — line 399
- Run the ``clearance-frame`` subcommand via the clearance extractor.
- kind: function, internal
- raises: `SystemExit`
- calls: `compute`, `_write_json_result`, `result.to_dict`, `SystemExit`, `PeriodicClearanceExtractor`

#### `main()` — line 423
- Parse arguments, dispatch the subcommand, and normalize errors.
- kind: function, CLI entry
- returns: `None` (bare return)
- effects: stdout
- calls: `_build_parser`, `parser.parse_args`, `_legacy_main`, `_analyze_main`, `_requirements_main`, `_manifest_main`, `_topology_main`, `_mechanics_step_main`, `_clearance_frame_main`, `print`, `sys.exit`

### `hygel_martini/property_extract/aggregation.py`

Contact-graph aggregation primitives with explicit cutoff provenance.

#### class `ContactGraphResult` — line 29
Connected-component decomposition of one chain contact graph.

##### `largest_component_size(self)` — line 48
- Chain count of the largest aggregate (0 for no chains).
- kind: property
- returns: `len(self.components[0]) if self.components else 0`

##### `n_components(self)` — line 53
- Number of aggregates, counting isolated chains.
- kind: property
- returns: `len(self.components)`

#### `chain_contact_graph(chain_site_positions: list[np.ndarray], box: np.ndarray, cutoff: float)` — line 58
- Build a chain graph from any inter-chain site pair within ``cutoff``.
- kind: function
- returns: `ContactGraphResult(tuple(components), counts, float(cutoff))`
- raises: `ValueError`
- calls: `orthorhombic_box_lengths`, `np.asarray`, `np.repeat`, `np.vstack`, `cKDTree`, `tree.query_pairs`, `components.sort`, `ContactGraphResult`, `ValueError`, `wrapped.append`, `np.arange`, `owner_pairs.sort`, +8 more

### `hygel_martini/property_extract/analysis_jobs.py`

Manifest-driven analysis job loading and execution.

#### class `AnalysisJob` — line 31
One entry of the ``analysis_jobs`` mapping in analysis_jobs.yaml.

##### `resolved_inputs(self, base_dir: str)` — line 62
- Resolve relative input paths against ``base_dir``.
- kind: method
- returns: `resolved`
- calls: `self.inputs.items`, `os.path.isabs`, `os.path.abspath`, `os.path.join`

#### `load_analysis_jobs(path: str, allow_template: bool=False)` — line 83
- Read analysis_jobs.yaml and return the job list plus its base dir.
- kind: function
- returns: `(jobs, os.path.dirname(path))`
- raises: `FileNotFoundError`, `ValueError`
- effects: filesystem
- calls: `os.path.abspath`, `jobs_spec.items`, `os.path.exists`, `FileNotFoundError`, `open`, `yaml.safe_load`, `ValueError`, `data.get`, `jobs.append`, `os.path.dirname`, `spec.get`, `AnalysisJob`

#### `run_analysis(analysis_path: str, md_requirements_path: str | None=None)` — line 157
- Run every job in analysis_jobs.yaml through its extractor.
- kind: function
- returns: `results`
- calls: `load_analysis_jobs`, `os.path.join`, `os.path.exists`, `job.resolved_inputs`, `EXTRACTOR_REGISTRY.get`, `extractor_cls`, `_write_job_output`, `check_requirements`, `PropertyResult.not_implemented`, `extractor.can_compute`, `extractor.missing_inputs_list`, `PropertyResult.missing`, +4 more

#### `_write_job_output(job: AnalysisJob, result: PropertyResult, base_dir: str)` — line 248
- Write ``output.report`` atomically when the analysis job requests it.
- kind: function, internal
- returns: `None` (bare return)
- effects: filesystem
- calls: `job.output.get`, `Path`, `report_path.parent.mkdir`, `report_path.with_suffix`, `temporary.write_text`, `temporary.replace`, `report_path.is_absolute`, `json.dumps`, `result.to_dict`

### `hygel_martini/property_extract/analyzer.py`

One-stop per-system analysis facade over the individual analyzers.

#### class `HydrogelAnalyzer` — line 27
Aggregate analyzer for one hydrogel system's files.

##### `__init__(self, top_file, itp_file, gro_file=None, energy_xvg=None, pore_selection_residues=None, polymer_residue_name=None, polymer_atom_name=None, solvent_molecule_names='W', polymer_bead_mass=45.0, solvent_bead_mass=72.0, polymer_bead_vol_nm3=0.065, pore_grid_spacing_nm=0.2, pore_bead_radius_nm=0.24, pore_bins=50, equilibration_threshold=0.01, equilibration_window=0.2, start_time_ps=0)` — line 36
- Store paths/settings and parse composition from top/itp.
- kind: method
- calls: `SwellingAnalyzer.from_files`

##### `from_config(cls, config_path)` — line 117
- Build a HydrogelAnalyzer from a YAML configuration file.
- kind: classmethod
- returns: `instance`; `p if os.path.isabs(str(p)) else os.path.join(cfg_dir, str(p))`; `None`
- raises: `ValueError`
- effects: filesystem
- calls: `os.path.dirname`, `cfg.get`, `resolve`, `gmx_cfg.get`, `get`, `mass_cfg.get`, `vf_cfg.get`, `pore_cfg.get`, `cls`, `open`, `yaml.safe_load`, `os.path.abspath`, +10 more

##### `extract_energy_from_edr(self, edr_file, output_xvg='energy.xvg', terms=None)` — line 262
- Extract energy terms from an .edr into an XVG via ``gmx energy``.
- kind: method
- returns: `output_xvg`
- raises: `RuntimeError`
- calls: `os.path.dirname`, `os.path.abspath`, `join`, `run_gmx`, `RuntimeError`

##### `analyze(self, start_time_ps=None)` — line 305
- Run every analysis the available files permit.
- kind: method
- returns: `results`
- calls: `self.swelling_analyzer.composition_summary`, `os.path.exists`, `PropertyResult.missing`, `self.swelling_analyzer.analyze_trajectory`, `parse_xvg`, `next`, `parse_gro_coords`, `PropertyResult.invalid_input`, `PropertyResult.analysis_failed`, `get_peak_pore_size`, `mask.sum`, `check_stability`, +1 more

##### `report(self, results: dict[str, PropertyResult], targets=None)` — line 441
- Print a console summary of results, optionally versus targets.
- kind: method
- effects: stdout
- calls: `print`, `results.items`, `_report_targets`, `pr.metadata.get`, `_format_status`

#### `_find_result_for_target(results: dict[str, PropertyResult], target_key: str)` — line 495
- Match a target key to a result directly or via ``target_aliases``.
- kind: function, internal
- returns: `(None, None)`; `(pr, target_key)`; `(result, result_key)`
- calls: `results.get`, `results.items`, `result.metadata.get`

#### `_format_status(status: str)` — line 515
- Render a status token for console display (underscores -> spaces).
- kind: function, internal
- returns: `status.replace('_', ' ').upper()`
- calls: `upper`, `status.replace`

#### `_report_targets(results: dict, targets: dict)` — line 520
- Compare each target spec against its matching PropertyResult.
- kind: function, internal
- effects: stdout
- calls: `targets.items`, `_find_result_for_target`, `print`, `pr.metadata.get`, `target_spec.get`

### `hygel_martini/property_extract/config.py`

One-shot loader for the three YAML files that drive an analysis run.

#### `load_all(analysis_path: str, manifest_path: str | None=None, requirements_path: str | None=None)` — line 18
- Load analysis jobs plus optional manifest and requirements YAMLs.
- kind: function
- returns: `(jobs, base_dir, manifest_targets, requirements_path)`
- calls: `load_analysis_jobs`, `os.path.join`, `os.path.exists`, `load_manifest`

### `hygel_martini/property_extract/cyclic_topology.py`

Cyclic-topology measurement on a reduced junction--strand multigraph.

#### `simple_adjacency(n_nodes: int, edges: list[tuple[int, int]])` — line 72
- Adjacency of the simple graph underlying a multigraph.
- kind: function
- returns: `adjacency`
- calls: `add`

#### `reduce_to_junctions(n_nodes: int, edges: list[tuple[int, int]])` — line 90
- Strip dangling trees and contract chain continuations.
- kind: function
- returns: `(len(order), reduced, stats)`; `sum((2 if live_edges[e][0] == live_edges[e][1] else 1 for e in incident[node] if e in live_edges))`
- calls: `defaultdict`, `deque`, `append`, `queue.popleft`, `alive.discard`, `live_edges.pop`, `live_edges.values`, `degree`, `queue.append`

#### `bipartite_check(n_nodes: int, edges: list[tuple[int, int]])` — line 191
- Two-colour the simple graph; return ``(is_bipartite, witness)``.
- kind: function
- returns: `(True, None)`; `(False, [left, left])`; `(False, _odd_walk(node, neighbour, parent))`
- calls: `simple_adjacency`, `deque`, `queue.popleft`, `queue.append`, `_odd_walk`

#### `_odd_walk(left: int, right: int, parent: dict[int, int | None])` — line 226
- Reconstruct a closed walk of odd length through the edge (left, right).
- kind: function, internal
- returns: `left_path + right_path[::-1]`; `path`; `left_path[:index + 1] + right_path[:right_index[node]][::-1]`
- calls: `to_root`, `path.append`

#### `shortest_ring_through_pair(adjacency: dict[int, set[int]], centre: int, first: int, second: int)` — line 250
- Size of the smallest cycle passing through ``first-centre-second``.
- kind: function
- returns: `None`; `distance[neighbour] + 2`
- calls: `deque`, `queue.popleft`, `queue.append`

#### `vertex_symbols(n_nodes: int, edges: list[tuple[int, int]])` — line 278
- Smallest ring size for every incident-strand pair, per junction.
- kind: function
- returns: `symbols`
- calls: `simple_adjacency`, `combinations`, `shortest_ring_through_pair`, `sizes.append`

#### `loop_order_histogram(n_nodes: int, edges: list[tuple[int, int]], symbols: dict[int, list[int | str]] | None=None)` — line 300
- Global cycle count per loop order, following Sen and Olsen Eq. (1).
- kind: function
- returns: `dict(sorted(counts.items()))`
- calls: `Counter`, `defaultdict`, `symbols.items`, `vertex_symbols`, `items`, `histogram.items`, `counts.get`, `counts.items`, `multiplicity.values`

#### `cyclic_topology_report(n_nodes: int, edges: list[tuple[int, int]])` — line 357
- Full cyclic-topology diagnostic for a junction--strand multigraph.
- kind: function
- returns: `{'junction_count': n_nodes, 'strand_count': n_strands, 'low_degree_junction_count': low_degree, 'loop_order_histogram_is_weighted_valid': low_degree == 0, 'mean_junction_degree': sum(degrees) / n_nodes if n_nodes else float('nan'), 'junction_degree_distribution': dict(sorted(Counter(degrees).items())), 'loop_order_histogram': histogram, 'loop_order_distribution': {order: count / n_strands for order, count in histogram.items()} if n_strands else {}, 'peak_loop_order': max(histogram, key=lambda order: (histogram[order], -order)) if histogram else None, 'girth': min(raw_ring_sizes) if raw_ring_sizes else None, 'odd_loop_order_count': odd_cycles, 'odd_loop_order_fraction': odd_cycles / total_cycles if total_cycles else 0.0, 'primary_loop_count': histogram.get(1, 0), 'secondary_loop_count': histogram.get(2, 0), 'primary_loop_fraction': histogram.get(1, 0) / n_strands if n_strands else 0.0, 'secondary_loop_fraction': histogram.get(2, 0) / n_strands if n_strands else 0.0, 'maximum_edge_multiplicity': max(multiplicity.values(), default=0), 'bipartite': is_bipartite, 'odd_cycle_witness': witness, 'distinct_vertex_symbol_count': len(symbol_frequency), 'vertex_symbol_frequency': dict(symbol_frequency.most_common())}`
- raises: `ValueError`
- calls: `vertex_symbols`, `loop_order_histogram`, `bipartite_check`, `Counter`, `ValueError`, `raw_ring_sizes.append`, `histogram.values`, `histogram.get`, `symbols.values`, `join`, `multiplicity.values`, `symbol_frequency.most_common`, +3 more

#### `report_from_itp(itp: str | Path, junction_residue: str='BCK')` — line 448
- Reduce an ITP to its junction--strand graph and measure its topology.
- kind: function
- returns: `report`
- calls: `audit_reduced_network`, `cyclic_topology_report`

#### `main(argv: list[str] | None=None)` — line 464
- CLI: audit an ITP's cyclic topology and print/save the report.
- kind: function, CLI entry
- returns: `0`
- effects: filesystem, stdout
- calls: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `report_from_itp`, `json.dumps`, `print`, `write_text`, `Path`

### `hygel_martini/property_extract/diffusion.py`

PBC-safe translational MSD primitives for within-model mobility analysis.

#### `unwrap_trajectory(positions: np.ndarray, boxes: np.ndarray)` — line 22
- Cumulatively unwrap particle positions for orthorhombic boxes.
- kind: function
- returns: `out`
- raises: `ValueError`
- calls: `np.asarray`, `np.empty_like`, `ValueError`, `np.repeat`, `minimum_image_displacement`, `orthorhombic_box_lengths`

#### `multi_origin_msd(unwrapped_positions: np.ndarray, frame_dt: float, max_lag_frames: int | None=None, origin_stride: int=1)` — line 65
- Return lag times, 3-D MSD, and number of time origins per lag.
- kind: function
- returns: `(np.arange(max_lag + 1, dtype=float) * frame_dt, msd, counts)`
- raises: `ValueError`
- calls: `np.asarray`, `np.empty`, `ValueError`, `np.arange`, `np.mean`, `np.sum`

#### `fit_diffusion_coefficient(lag_times: np.ndarray, msd: np.ndarray, fit_start: float, fit_end: float, dimensions: int=3)` — line 116
- Fit ``MSD = 2*d*D*t + intercept`` over an explicit lag window.
- kind: function
- returns: `{'diffusion_coefficient_coordinate2_per_time': coefficient, 'slope': float(slope), 'intercept': float(intercept), 'r_squared': float(r_squared), 'fit_start': float(fit_start), 'fit_end': float(fit_end), 'dimensions': int(dimensions), 'n_fit_points': int(np.count_nonzero(mask))}`
- raises: `ValueError`
- calls: `np.asarray`, `np.polyfit`, `ValueError`, `np.isfinite`, `np.count_nonzero`, `np.sum`, `np.mean`

### `hygel_martini/property_extract/equilibration.py`

Time-series stability gate for equilibration judgment.

#### `check_stability(data, threshold=0.01, window=0.2)` — line 16
- Judge stability from the linear drift of the trailing window.
- kind: function
- returns: `PropertyResult(property='volume_stability', value=is_stable, status='computed', direct_experiment_comparison_allowed=False, validation_role='', metadata={'mean': mean, 'std': std, 'drift': rel_drift, 'is_stable': is_stable, 'threshold': threshold, 'window_fraction': window})`; `PropertyResult.insufficient_data('volume_stability', reason='stability 판단에는 최소 10개 이상의 데이터 포인트가 필요합니다.', metadata={'mean': float(np.mean(data)) if len(data) > 0 else float('nan'), 'std': float(np.std(data)) if len(data) > 0 else float('nan'), 'drift': float('nan'), 'is_stable': False})`; `PropertyResult(property='volume_stability', value=True, status='computed', direct_experiment_comparison_allowed=False, validation_role='', metadata={'mean': mean, 'std': std, 'drift': 0.0, 'is_stable': True})`
- calls: `np.arange`, `np.polyfit`, `PropertyResult`, `PropertyResult.insufficient_data`, `np.mean`, `np.std`

#### `find_equilibration_time(times, data, threshold=0.01, window_size=100)` — line 88
- Placeholder: automatic equilibration-time detection.
- kind: function
- raises: `NotImplementedError`
- calls: `NotImplementedError`

### `hygel_martini/property_extract/experimental_mapping.py`

Scaling-law bridges from simulation observables to experiment.

#### `calculate_mesh_size(phi, strand_length_nm)` — line 12
- Estimate the mesh size xi from a semi-dilute scaling law.
- kind: function
- returns: `strand_length_nm * phi ** (-0.75)`; `0.0`

#### `estimate_elastic_modulus(phi, temp_k, crosslink_density, strand_molar_mass, polymer_density_g_cm3, functionality)` — line 35
- Placeholder: rubber-elasticity shear modulus estimate.
- kind: function
- raises: `NotImplementedError`
- calls: `NotImplementedError`

#### `calculate_water_uptake(loading_qm)` — line 53
- Convert the loading ratio Qm to a water-uptake percentage.
- kind: function
- returns: `(1.0 - 1.0 / loading_qm) * 100.0`

### `hygel_martini/property_extract/extractors/__init__.py`

Extractor plugin package: adapters from analyzers to the manifest runner.

### `hygel_martini/property_extract/extractors/_registry.py`

Plugin registry and base class for manifest-driven extractors.

#### class `BaseExtractor` — line 47
Adapter contract between the manifest runner and one analyzer.

##### `can_compute(self, inputs: dict[str, str | None])` — line 67
- Return True when every required input path exists on disk.
- kind: method
- returns: `all((inputs.get(k) and os.path.exists(str(inputs[k])) for k in self.required_inputs))`
- calls: `inputs.get`, `os.path.exists`

##### `missing_inputs_list(self, inputs: dict[str, str | None])` — line 79
- Return the required input keys that are absent or nonexistent.
- kind: method
- returns: `[k for k in self.required_inputs if not (inputs.get(k) and os.path.exists(str(inputs[k] or '')))]`
- calls: `inputs.get`, `os.path.exists`

##### `compute(self, inputs: dict[str, str | None], params: dict)` — line 90
- Run the analysis; subclasses must override.
- kind: method
- raises: `NotImplementedError`
- calls: `NotImplementedError`

#### `register_extractor(name: str)` — line 30
- Return a class decorator that registers an extractor under ``name``.
- kind: function
- returns: `decorator`; `cls`

### `hygel_martini/property_extract/extractors/clearance.py`

Extractor adapter: periodic local-clearance distribution on a frame.

#### class `PeriodicClearanceExtractor`(BaseExtractor) — line 21
Calculate definition-explicit local clearance on one periodic GRO frame.

##### `compute(self, inputs: dict, params: dict)` — line 27
- Compute the clearance distribution and report its median.
- kind: method
- returns: `PropertyResult(property='local_clearance_diameter_p50_nm', value=value, status='computed', direct_experiment_comparison_allowed=False, validation_role='proxy', metadata={**summary, 'obstacle_definition': {'selection_residues': list(residues), 'bead_radius_nm': float(params.get('bead_radius_nm', 0.24))}, 'claim_boundary': 'local clearance/probe-admissible volume; not a unique pore, pore-limiting diameter, or experimental mesh size'})`; `PropertyResult.invalid_input('local_clearance_diameter_p50_nm', reason='selection_residues selected zero obstacle beads', inputs=[str(inputs['gro'])], validation_role='proxy')`
- calls: `parse_gro_coords`, `calculate_periodic_clearance_distribution`, `PropertyResult`, `params.get`, `PropertyResult.invalid_input`

### `hygel_martini/property_extract/extractors/composition.py`

Extractor adapter: as-built composition from topology files alone.

#### class `CompositionExtractor`(BaseExtractor) — line 17
Compute the count-based loading ratio ``q_m`` from top/itp files.

##### `compute(self, inputs: dict, params: dict)` — line 27
- Build a SwellingAnalyzer from top/itp and return its summary.
- kind: method
- returns: `analyzer.composition_summary()`
- calls: `SwellingAnalyzer.from_files`, `analyzer.composition_summary`, `params.get`

### `hygel_martini/property_extract/extractors/mechanics.py`

Extractor adapter: paired-step apparent shear response from XVGs.

#### class `PairedStepXVGExtractor`(BaseExtractor) — line 20
Compute the registered apparent response from aligned +/- step XVGs.

##### `compute(self, inputs: dict, params: dict)` — line 26
- Summarize the paired step and wrap it as a PropertyResult.
- kind: method
- returns: `PropertyResult(property='paired_step_finite_rate_apparent_shear_response', value=summary['apparent_response_mean_mpa'], status='computed', direct_experiment_comparison_allowed=False, validation_role='finite_rate', metadata=summary)`; `PropertyResult.invalid_input('paired_step_finite_rate_apparent_shear_response', reason=f"missing mechanics parameters: {', '.join(missing)}", validation_role='finite_rate')`
- calls: `paired_step_xvg_summary`, `PropertyResult`, `PropertyResult.invalid_input`, `params.get`, `join`

### `hygel_martini/property_extract/extractors/pore_size.py`

Extractor adapter: single-frame grid pore size from a GRO file.

#### class `PoreSizeGridExtractor`(BaseExtractor) — line 19
Peak pore size of one .gro frame via a nearest-surface grid.

##### `compute(self, inputs: dict, params: dict)` — line 28
- Parse the frame, then histogram grid-to-surface distances.
- kind: method
- returns: `get_peak_pore_size(coords, box, grid_spacing=grid_spacing, bead_radius=bead_radius, bins=bins)`; `PropertyResult.invalid_input('pore_size_single_frame_grid', reason='polymer atom 0개 선택됨 — selection_residues 확인 필요', inputs=[str(inputs['gro'])], validation_role='proxy', metadata={'target_aliases': ['pore_diameter_nm']})`; `PropertyResult.invalid_input('pore_size_single_frame_grid', reason=str(e), inputs=[str(inputs['gro'])], validation_role='proxy', metadata={'target_aliases': ['pore_diameter_nm']})` (+1 more)
- calls: `params.get`, `get_peak_pore_size`, `parse_gro_coords`, `PropertyResult.invalid_input`, `next`, `PropertyResult.analysis_failed`, `iter`, `bead_radius_raw.values`

### `hygel_martini/property_extract/extractors/rheology_nemd.py`

Extractor adapter: viscosity vs shear rate from an NEMD series.

#### class `NEMDShearRateExtractor`(BaseExtractor) — line 22
Viscosity vs shear rate from a series of NEMD shear runs.

##### `can_compute(self, inputs: dict)` — line 33
- Gate on glob expansion: True only if the pattern matches.
- kind: method
- returns: `bool(matches)`; `False`
- calls: `inputs.get`, `_glob.glob`

##### `missing_inputs_list(self, inputs: dict)` — line 42
- Report the input key (unset) or the pattern (no matches).
- kind: method
- returns: `[]`; `['shear_dirs']`; `[str(shear_dirs)]`
- calls: `inputs.get`, `_glob.glob`

##### `compute(self, inputs: dict, params: dict)` — line 52
- Average per-directory NEMD viscosities into one result.
- kind: method
- returns: `PropertyResult.missing('viscosity_vs_shear_rate', missing_inputs=[shear_dirs_pattern], validation_role='trend_only')`; `PropertyResult.invalid_input('viscosity_vs_shear_rate', reason='parameters.shear_rates_ps_inv가 없습니다.', validation_role='trend_only')`; `PropertyResult.missing('viscosity_vs_shear_rate', missing_inputs=missing_xvg, validation_role='trend_only')` (+3 more)
- calls: `params.get`, `_glob.glob`, `PropertyResult.missing`, `PropertyResult.invalid_input`, `os.path.join`, `analyze_shear_rate_viscosity`, `PropertyResult`, `PropertyResult.analysis_failed`, `os.path.exists`, `np.mean`, `viscosities.tolist`

### `hygel_martini/property_extract/extractors/swelling.py`

Extractor adapter: polymer volume fraction from an energy XVG.

#### class `SwellingVolumeExtractor`(BaseExtractor) — line 19
Compute polymer_volume_fraction from the energy.xvg Volume column.

##### `compute(self, inputs: dict, params: dict)` — line 29
- Analyze the volume time series; refuse on bad parameters.
- kind: method
- returns: `PropertyResult.invalid_input('polymer_volume_fraction', reason=f'method={method!r}는 지원하지 않습니다. 현재 구현은 bead_volume만 가능합니다.', validation_role='direct')`; `PropertyResult.invalid_input('polymer_volume_fraction', reason='parameters.bead_volume_nm3가 없습니다.', validation_role='direct')`; `analyzer.analyze_trajectory(str(inputs['energy_xvg']), start_time_ps=start_time_ps)` (+2 more)
- calls: `params.get`, `SwellingAnalyzer.from_files`, `PropertyResult.invalid_input`, `analyzer.analyze_trajectory`, `PropertyResult.analysis_failed`

### `hygel_martini/property_extract/extractors/topology.py`

Extractor adapter: reduced junction-strand network audit gate.

#### class `ReducedNetworkTopologyExtractor`(BaseExtractor) — line 20
Audit a junction--strand graph without chemistry-specific defaults.

##### `compute(self, inputs: dict, params: dict)` — line 26
- Run the audit and evaluate the manifest's expected counts.
- kind: method
- returns: `PropertyResult(property='reduced_network_topology_audit', value=gate_pass, status='computed', direct_experiment_comparison_allowed=False, validation_role='structural_audit', metadata={'checks': checks, 'gate_pass': gate_pass, 'audit': audit, 'claim_boundary': 'bonded-graph construction audit; not force-field or equilibrium-mechanics validation'})`
- raises: `ValueError`
- calls: `audit_reduced_network`, `expected.items`, `params.get`, `PropertyResult`, `checks.values`, `inputs.get`, `ValueError`

### `hygel_martini/property_extract/geometry.py`

Periodic geometry primitives shared by PEG and Pluronic analyses.

#### `orthorhombic_box_lengths(box: np.ndarray)` — line 27
- Normalize a 3-vector or diagonal 3x3 box to positive lengths.
- kind: function
- returns: `lengths.astype(float, copy=True)`
- raises: `ValueError`
- calls: `np.asarray`, `np.any`, `lengths.astype`, `ValueError`, `np.diag`, `np.allclose`

#### `minimum_image_displacement(delta: np.ndarray, box: np.ndarray)` — line 56
- Apply the orthorhombic minimum-image convention to displacement(s).
- kind: function
- returns: `core_minimum_image(delta, lengths)`
- calls: `orthorhombic_box_lengths`, `core_minimum_image`

#### `wrap_positions(positions: np.ndarray, box: np.ndarray)` — line 76
- Wrap Cartesian positions into ``[0, L)`` for an orthorhombic box.
- kind: function
- returns: `np.mod(pos, lengths)`
- raises: `ValueError`
- calls: `orthorhombic_box_lengths`, `np.asarray`, `np.mod`, `ValueError`

#### `unwrap_ordered_chain(positions: np.ndarray, box: np.ndarray)` — line 97
- Unwrap a bonded/ordered chain using consecutive minimum images.
- kind: function
- returns: `out`
- raises: `ValueError`
- calls: `np.asarray`, `np.empty_like`, `ValueError`, `minimum_image_displacement`

#### `gyration_metrics(positions: np.ndarray)` — line 126
- Return Rg, end-to-end distance, eigenvalues, and relative anisotropy.
- kind: function
- returns: `{'radius_of_gyration': float(np.sqrt(max(total, 0.0))), 'end_to_end': float(np.linalg.norm(pos[-1] - pos[0])), 'gyration_eigenvalues': eigenvalues.tolist(), 'relative_shape_anisotropy': kappa2}`
- raises: `ValueError`
- calls: `np.asarray`, `np.linalg.eigvalsh`, `ValueError`, `np.mean`, `np.sum`, `eigenvalues.tolist`, `np.sqrt`, `np.linalg.norm`

#### `bond_orientation_metrics(vectors: np.ndarray)` — line 164
- Return second-rank orientational order from PBC-corrected bond vectors.
- kind: function
- returns: `{'n_vectors': int(len(unit)), 'second_moment': second_moment.tolist(), 'orientation_tensor': tensor.tolist(), 'eigenvalues': eigenvalues.tolist(), 'largest_eigenvalue': float(eigenvalues[-1]), 'principal_axis': principal.tolist()}`
- raises: `ValueError`
- calls: `np.asarray`, `np.linalg.norm`, `np.linalg.eigh`, `ValueError`, `np.any`, `second_moment.tolist`, `tensor.tolist`, `eigenvalues.tolist`, `principal.tolist`, `np.eye`, `np.isfinite`

### `hygel_martini/property_extract/gmx_utils.py`

Thin GROMACS process and XVG parsing helpers.

#### `run_gmx(cmd, input_text=None, cwd=None)` — line 18
- Run a GROMACS command and return its stdout.
- kind: function
- returns: `proc.stdout`
- raises: `RuntimeError`
- effects: subprocess, filesystem
- calls: `os.environ.get`, `RuntimeError`, `subprocess.run`, `shutil.which`, `cmd.insert`, `join`

#### `parse_xvg(xvg_file)` — line 69
- Parse a GROMACS .xvg file into a column dictionary.
- kind: function
- returns: `result`
- raises: `ValueError`
- effects: filesystem
- calls: `np.array`, `open`, `ValueError`, `data.reshape`, `line.startswith`, `re.search`, `line.split`, `labels.append`, `data.append`, `match.group`

### `hygel_martini/property_extract/mechanics.py`

Small-deformation mechanics primitives with explicit claim boundaries.

#### `classical_network_modulus_bounds(n_strands: int, n_junctions: int, volume_nm3: float, temperature_k: float)` — line 44
- Return affine and classical phantom-network shear-modulus estimates.
- kind: function
- returns: `{'affine_shear_modulus_kpa': float(affine_kpa), 'phantom_shear_modulus_kpa': float(phantom_kpa), 'mean_functionality': float(mean_functionality), 'phantom_to_affine_ratio': float(phantom_kpa / affine_kpa), 'strand_number_density_nm3': float(n_strands / volume_nm3)}`
- raises: `ValueError`
- calls: `ValueError`

#### `harmonic_strain_energy_kbt(modulus_kpa: float, strain: float, volume_nm3: float, temperature_k: float)` — line 103
- Return ``(1/2) * modulus * strain**2 * volume`` in units of ``kBT``.
- kind: function
- returns: `float(energy_j / (BOLTZMANN_J_PER_K * float(temperature_k)))`
- raises: `ValueError`
- calls: `ValueError`

#### `volume_preserving_uniaxial(coordinates: np.ndarray, stretch: float, axis: str='x')` — line 146
- Apply a homogeneous incompressible uniaxial deformation.
- kind: function
- returns: `xyz * factors`
- raises: `ValueError`
- calls: `np.asarray`, `np.full`, `ValueError`, `UNIAXIAL_AXES.index`

#### `volume_preserving_uniaxial_box(lengths: np.ndarray, stretch: float, axis: str='x')` — line 180
- Return row-wise box vectors after incompressible uniaxial deformation.
- kind: function
- returns: `np.diag(volume_preserving_uniaxial(box[np.newaxis, :], stretch, axis)[0])`
- raises: `ValueError`
- calls: `np.asarray`, `np.diag`, `np.any`, `ValueError`, `volume_preserving_uniaxial`

#### `uniaxial_nominal_stress_from_pressure(axial_pressure: np.ndarray, lateral_pressure_1: np.ndarray, lateral_pressure_2: np.ndarray, stretch: float, pressure_to_stress: float=0.1)` — line 205
- Convert GROMACS pressure components to nominal uniaxial stress.
- kind: function
- returns: `-(axial - lateral_mean) / float(stretch) * pressure_to_stress`
- raises: `ValueError`
- calls: `np.broadcast_arrays`, `ValueError`, `np.asarray`

#### `affine_simple_shear(coordinates: np.ndarray, gamma: float, plane: str='xy')` — line 247
- Return coordinates after a simple engineering-shear step.
- kind: function
- returns: `sheared`
- raises: `ValueError`
- calls: `np.asarray`, `xyz.copy`, `ValueError`

#### `simple_shear_box(lengths: np.ndarray, gamma: float, plane: str='xy')` — line 284
- Return three GROMACS box vectors as rows after volume-preserving shear.
- kind: function
- returns: `vectors`
- raises: `ValueError`
- calls: `np.asarray`, `np.diag`, `np.any`, `ValueError`

#### `gromacs_box_values(box_vectors: np.ndarray)` — line 321
- Convert three row-wise box vectors to the nine-value GRO ordering.
- kind: function
- returns: `np.array([v1[0], v2[1], v3[2], v1[1], v1[2], v2[0], v2[2], v3[0], v3[1]])`
- raises: `ValueError`
- calls: `np.asarray`, `np.array`, `ValueError`

#### `write_step_sheared_gro(source: str | Path, destination: str | Path, gamma: float, plane: str='xy', orthorhombic_tolerance: float=1e-08)` — line 338
- Write a sheared GRO while preserving atom fields and velocities.
- kind: function
- raises: `ValueError`
- effects: filesystem
- calls: `Path`, `splitlines`, `np.asarray`, `np.any`, `np.empty`, `affine_simple_shear`, `gromacs_box_values`, `output.append`, `destination_path.parent.mkdir`, `destination_path.write_text`, `ValueError`, `simple_shear_box`, +5 more

#### `write_uniaxially_deformed_gro(source: str | Path, destination: str | Path, stretch: float, axis: str='x', orthorhombic_tolerance: float=1e-08)` — line 411
- Write an affinely deformed constant-volume GRO file.
- kind: function
- raises: `ValueError`
- effects: filesystem
- calls: `Path`, `splitlines`, `np.asarray`, `np.any`, `np.empty`, `volume_preserving_uniaxial`, `volume_preserving_uniaxial_box`, `output.append`, `destination_path.parent.mkdir`, `destination_path.write_text`, `ValueError`, `join`, +5 more

#### `paired_step_shear_response(baseline_pressure: np.ndarray, positive_pressure: np.ndarray, negative_pressure: np.ndarray, gamma: float, pressure_to_modulus: float=0.1)` — line 486
- Decompose matched +/- pressure traces into odd and even responses.
- kind: function
- returns: `{'odd_pressure': odd_pressure, 'even_residual_pressure': even_residual, 'apparent_modulus': -odd_pressure / gamma * pressure_to_modulus, 'positive_apparent_modulus': -(pp - p0) / gamma * pressure_to_modulus, 'negative_apparent_modulus': (pm - p0) / gamma * pressure_to_modulus}`
- raises: `ValueError`
- calls: `np.broadcast_arrays`, `ValueError`, `np.asarray`

### `hygel_martini/property_extract/mechanics_analysis.py`

Reusable finite-rate mechanics analysis with explicit claim boundaries.

#### `read_labeled_xvg(path: str | Path)` — line 45
- Read a GROMACS XVG into named columns, requiring complete legends.
- kind: function
- returns: `{'Time': values[:, 0], **{legend: values[:, index + 1] for index, legend in enumerate(legends)}}`
- raises: `ValueError`
- calls: `read_xvg`, `ValueError`, `np.all`, `np.isfinite`

#### `paired_step_window_summary(time_ps: np.ndarray, baseline_pressure_bar: np.ndarray, positive_pressure_bar: np.ndarray, negative_pressure_bar: np.ndarray, *, gamma: float, window_start_ps: float, window_end_ps: float)` — line 77
- Summarize a matched +/- step-shear response over a registered window.
- kind: function
- returns: `{'observable': 'paired_step_finite_rate_apparent_shear_response', 'gamma': float(gamma), 'window_start_ps': float(window_start_ps), 'window_end_ps': float(window_end_ps), 'n_samples': int(np.count_nonzero(mask)), 'apparent_response_mean_mpa': float(np.mean(modulus)), 'apparent_response_sample_sd_mpa': float(np.std(modulus, ddof=1)) if modulus.size > 1 else 0.0, 'apparent_response_first_sample_mpa': float(modulus[0]), 'elastic_first_sample_sign_positive': bool(modulus[0] > 0), 'odd_pressure_abs_mean_bar': odd_scale, 'even_residual_mean_bar': float(np.mean(even)), 'even_to_odd_abs_ratio': float(np.mean(np.abs(even)) / odd_scale) if odd_scale > 0 else math.inf, 'claim_boundary': 'finite-rate apparent response; not equilibrium, plateau, storage, zero-frequency, or experimental modulus'}`
- raises: `ValueError`
- calls: `np.asarray`, `paired_step_shear_response`, `np.any`, `ValueError`, `np.count_nonzero`, `np.mean`, `np.abs`, `np.diff`, `response.values`, `np.std`

#### `paired_step_xvg_summary(baseline_xvg: str | Path, positive_xvg: str | Path, negative_xvg: str | Path, *, component: str, gamma: float, window_start_ps: float, window_end_ps: float)` — line 158
- Read three aligned labeled XVGs and summarize one pressure component.
- kind: function
- returns: `result`
- raises: `ValueError`
- calls: `paired_step_window_summary`, `result.update`, `read_labeled_xvg`, `ValueError`, `Path`, `np.allclose`

#### `analyze_paired_ramp(plus: Mapping[str, np.ndarray], minus: Mapping[str, np.ndarray], *, target_amplitude: float, ramp_ps: float, target_temperature_k: float, fraction_edges: Sequence[float]=DEFAULT_RAMP_FRACTION_EDGES)` — line 228
- Analyze aligned +/- finite ramps using block means along the ramp.
- kind: function
- returns: `(points, blocks, summary)`
- raises: `ValueError`
- calls: `np.asarray`, `np.any`, `np.concatenate`, `ValueError`, `math.isclose`, `blocks.append`, `np.mean`, `active.tolist`, `np.allclose`, `np.diff`, `np.std`, `math.sqrt`, +7 more

#### `analyze_cycle_blocks(g_prime_mpa: np.ndarray, g_double_prime_mpa: np.ndarray, *, block_size_cycles: int=5, minimum_blocks: int=6, bootstrap_samples: int=100000, bootstrap_seed: int=20260728)` — line 404
- Estimate periodic-response uncertainty from contiguous cycle blocks.
- kind: function
- returns: `(rows, summary)`
- raises: `ValueError`
- calls: `np.asarray`, `mean`, `np.random.default_rng`, `rng.integers`, `np.hypot`, `np.divide`, `complex`, `np.abs`, `ValueError`, `np.mean`, `math.hypot`, `np.percentile`, +7 more

#### `summarize_equal_realizations(realization_values: Mapping[str, Sequence[float]])` — line 561
- Give each realization equal weight after averaging within realization.
- kind: function
- returns: `{'statistical_unit': 'realization', 'n_realizations': int(len(network_means)), 'within_realization_counts': counts, 'realization_means': means, 'equal_weight_mean': float(np.mean(network_means)), 'realization_sample_sd': float(np.std(network_means, ddof=1)), 'realization_sem': float(np.std(network_means, ddof=1) / math.sqrt(len(network_means))), 'claim_boundary': 'within-realization samples are not counted as independent network realizations'}`
- raises: `ValueError`
- calls: `realization_values.items`, `np.asarray`, `ValueError`, `np.mean`, `means.values`, `np.std`, `np.all`, `math.sqrt`, `np.isfinite`

#### `holm_adjust(pvalues: Sequence[float])` — line 613
- Return Holm family-wise-error adjusted p-values.
- kind: function
- returns: `adjusted.tolist()`
- raises: `ValueError`
- calls: `np.asarray`, `np.argsort`, `np.empty`, `adjusted.tolist`, `ValueError`, `np.any`, `np.isfinite`

### `hygel_martini/property_extract/network_topology.py`

Graph-level audits for covalently crosslinked hydrogel networks.

#### `_read_itp_atoms_bonds(path: str | Path)` — line 44
- Atoms and bonds of a single-molecule ITP, via the shared parser.
- kind: function, internal
- returns: `(atoms, bonds)`
- raises: `ValueError`
- calls: `read_itp_definitions`, `next`, `ValueError`, `iter`, `definitions.values`, `definition.get`, `join`, `bond.get`

#### `_components(nodes: Iterable[int], adjacency: dict[int, set[int]])` — line 89
- Return connected components (as sets) via iterative traversal.
- kind: function, internal
- returns: `output`
- calls: `remaining.remove`, `output.append`, `queue.pop`, `component.add`, `queue.append`

#### `_reduced_components(n_nodes: int, edges: list[tuple[int, int]])` — line 113
- Connected components of the reduced junction graph (0..n-1 nodes).
- kind: function, internal
- returns: `_components(range(n_nodes), adjacency)`
- calls: `_components`, `add`

#### `_multigraph_degrees(n_nodes: int, edges: list[tuple[int, int]])` — line 124
- Per-node degrees counting multi-edges; a self-loop contributes 2.
- kind: function, internal
- returns: `degrees`

#### `_two_core(n_nodes: int, edges: list[tuple[int, int]])` — line 138
- Return the multigraph 2-core as (surviving nodes, surviving edge ids).
- kind: function, internal
- returns: `(active_nodes, active_edges)`; `value`
- calls: `deque`, `add`, `queue.popleft`, `active_nodes.remove`, `active_edges.difference_update`, `neighbors.update`, `degree`, `queue.append`

#### `_bridge_edges(n_nodes: int, edges: list[tuple[int, int]])` — line 182
- Return multigraph bridge IDs using edge-aware Tarjan traversal.
- kind: function, internal
- returns: `bridges`
- calls: `defaultdict`, `append`, `visit`, `bridges.add`

#### `_shortest_path(start: int, end: int, allowed: set[int], adjacency: dict[int, set[int]])` — line 221
- BFS shortest bonded path from start to end within ``allowed`` atoms.
- kind: function, internal
- returns: `[start]`; `list(reversed(path))`
- raises: `ValueError`
- calls: `deque`, `ValueError`, `queue.popleft`, `queue.append`, `path.append`, `reversed`

#### `_read_gro(path: str | Path)` — line 254
- Coordinates and periodic cell, via the shared reader.
- kind: function, internal
- returns: `(frame.positions, frame.box)`
- raises: `ValueError`
- calls: `read_gro`, `ValueError`

#### `_canonical_winding(vector: np.ndarray)` — line 262
- Canonicalize a winding vector's sign (v and -v describe one cycle).
- kind: function, internal
- returns: `values`; `tuple((-item for item in values))`

#### `_periodic_winding(coordinates: np.ndarray, box: np.ndarray, adjacency: dict[int, set[int]], junction_components: list[set[int]], strands: list[dict[str, object]], n_itp_atoms: int)` — line 277
- Audit periodic winding of the reduced network from GRO coordinates.
- kind: function, internal
- returns: `{'box_volume_nm3': float(np.linalg.det(box)), 'winding_rank': rank, 'spans_x': bool(spans[0]), 'spans_y': bool(spans[1]), 'spans_z': bool(spans[2]), 'winding_vectors': [list(vector) for vector in sorted(winding_vectors)]}`; `delta - np.round(delta)`
- raises: `ValueError`
- calls: `np.linalg.inv`, `defaultdict`, `np.asarray`, `ValueError`, `_shortest_path`, `astype`, `shifted_edges.append`, `append`, `np.zeros`, `deque`, `np.any`, `np.round`, +10 more

#### `audit_reduced_network(itp: str | Path, gro: str | Path | None=None, junction_residue: str='BCK')` — line 442
- Collapse an atomistic/CG ITP into a junction--strand multigraph.
- kind: function
- returns: `result`
- raises: `ValueError`
- effects: stdout
- calls: `Path`, `_read_itp_atoms_bonds`, `defaultdict`, `_components`, `junction_components.sort`, `strand_components.sort`, `_reduced_components`, `_multigraph_degrees`, `Counter`, `_two_core`, `_bridge_edges`, `join`, +17 more

#### `main(argv: list[str] | None=None)` — line 626
- CLI entry point: audit one ITP and emit the result as JSON.
- kind: function, CLI entry
- returns: `0`
- effects: filesystem, stdout
- calls: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `audit_reduced_network`, `json.dumps`, `arguments.output.parent.mkdir`, `arguments.output.write_text`, `print`

### `hygel_martini/property_extract/parametric.py`

Sweep-level analysis across state points (T, P, wt%).

#### class `ParametricAnalyzer` — line 50
Analyzes hydrogel properties across different state points (T, P, wt%).

##### `__init__(self, workspace_dir, top_filename='system.top', itp_filename='initial_hydrogel.itp', gro_filename='production.gro', edr_filename='production.edr', start_time_ps=100000)` — line 55
- Configure the sweep's file conventions and output location.
- kind: method

##### `add_point(self, temp, press, wt, dir_path)` — line 86
- Register one state point (T in K, P, wt%) and its directory.
- kind: method
- calls: `self.points.append`

##### `collect_properties(self)` — line 90
- Run the full analysis on every registered point.
- kind: method
- returns: `{'results': results, 'missing_files': missing_files, 'failed_points': failed_points}`
- calls: `os.path.join`, `os.path.exists`, `missing_files.append`, `HydrogelAnalyzer`, `_flatten_property_results`, `res.update`, `results.append`, `analyzer.extract_energy_from_edr`, `analyzer.analyze`, `failed_points.append`

##### `plot_temperature_sensitivity(self, results, target_property='loading_qm')` — line 140
- Plot one property against temperature and save the PNG.
- kind: method
- raises: `ValueError`
- calls: `plt.figure`, `plt.plot`, `plt.xlabel`, `plt.ylabel`, `plt.title`, `plt.savefig`, `_property_value`, `ValueError`, `os.path.join`, `np.isfinite`

##### `plot_phase_diagram(self, results, prop='loading_qm')` — line 178
- Placeholder: 2-D property heatmap over T and wt%.
- kind: method
- raises: `NotImplementedError`
- calls: `NotImplementedError`

#### `_property_value(value)` — line 22
- Return the scalar payload of a PropertyResult, else pass through.
- kind: function, internal
- returns: `value`; `value.value`

#### `_flatten_property_results(results)` — line 29
- Flatten a name->PropertyResult mapping into plain plot-ready values.
- kind: function, internal
- returns: `flat`
- calls: `results.items`, `result.to_dict`

### `hygel_martini/property_extract/polymer_stats.py`

Chain-level conformation statistics via MDAnalysis.

#### class `PolymerStats` — line 25
Conformation statistics for the selected polymer atoms.

##### `__init__(self, gro_file, traj_file=None, selection='resname PEO or resname HYDROGEL')` — line 33
- Load the system and select the polymer atoms.
- kind: method
- calls: `self.u.select_atoms`, `mda.Universe`

##### `calculate_rg(self)` — line 49
- Calculate the radius of gyration of the whole selection.
- kind: method
- returns: `np.array(rg_list)`
- calls: `hasattr`, `np.array`, `rg_list.append`, `self.polymer.atoms.radius_of_gyration`

##### `calculate_end_to_end(self)` — line 65
- Calculate the mean per-chain end-to-end distance per frame.
- kind: method
- returns: `np.array(results)`
- raises: `NoDataError`
- calls: `np.array`, `NoDataError`, `warnings.warn`, `self.polymer.split`, `np.linalg.norm`, `chain_dists.append`, `results.append`, `np.mean`

##### `estimate_persistence_length(self)` — line 107
- Estimate the persistence length from bond-bond correlation.
- kind: method
- returns: `np.mean(lps) if lps else 0.0`; `0.0`
- calls: `np.linalg.norm`, `np.mean`, `np.sum`, `warnings.warn`, `lps.append`, `np.log`

### `hygel_martini/property_extract/pore_size.py`

Periodic, definition-explicit grid clearance analysis.

#### `parse_gro_coords(gro_file, selection_residue=None, selection_residues=None)` — line 41
- Parse a .gro file into selected-residue coordinates and box vector.
- kind: function
- returns: `(np.array(coords), box)`
- raises: `ValueError`
- effects: filesystem
- calls: `open`, `f.readlines`, `np.array`, `strip`, `split`, `ValueError`, `coords.append`

#### `_validate_box(box_size: np.ndarray)` — line 106
- Coerce to a ``(3,)`` float box; require positive finite lengths.
- kind: function, internal
- returns: `box`
- raises: `ValueError`
- calls: `np.asarray`, `np.any`, `ValueError`, `np.all`, `np.isfinite`

#### `_grid_shape_and_spacing(box_size: np.ndarray, target_spacing: float)` — line 114
- Fit integer cell counts to the box; return shape and true spacing.
- kind: function, internal
- returns: `(shape, box_size / shape)`
- raises: `ValueError`
- calls: `np.maximum`, `ValueError`, `astype`, `np.isfinite`, `np.floor`

#### `periodic_clearance_grid(obstacle_groups: Sequence[tuple[np.ndarray, float]], box_size: np.ndarray, grid_spacing: float=0.2, chunk_size: int=_DEFAULT_CHUNK_SIZE)` — line 129
- Return nearest obstacle-surface clearance on a periodic cell-centred grid.
- kind: function
- returns: `(clearance.reshape(shape_tuple), spacing)`
- raises: `ValueError`
- calls: `_validate_box`, `_grid_shape_and_spacing`, `np.empty`, `ValueError`, `np.asarray`, `np.prod`, `warnings.warn`, `np.arange`, `np.column_stack`, `np.full`, `clearance.reshape`, `np.all`, +8 more

#### `periodic_component_summary(mask: np.ndarray)` — line 210
- Summarize 6-connected components after merging periodic face contacts.
- kind: function
- returns: `{'n_periodic_components': len(merged), 'largest_component_voxels': largest, 'largest_component_fraction_of_grid': float(largest / values.size), 'largest_component_fraction_of_admissible': float(largest / admissible)}`; `{'n_periodic_components': 0, 'largest_component_voxels': 0, 'largest_component_fraction_of_grid': 0.0, 'largest_component_fraction_of_admissible': 0.0}`; `label`
- raises: `ValueError`
- calls: `np.asarray`, `ndimage.generate_binary_structure`, `ndimage.label`, `np.arange`, `np.bincount`, `ValueError`, `find`, `np.take`, `labels.ravel`, `merged.values`, `np.count_nonzero`, `union`, +1 more

#### `summarize_periodic_clearance(clearance_nm: np.ndarray, probe_radius_nm: float=0.1657, bins: int=50)` — line 285
- Summarize local-clearance diameters and probe-admissible volume.
- kind: function
- returns: `(centres, hist, summary)`; `(np.array([]), np.array([]), summary)`
- raises: `ValueError`
- calls: `np.asarray`, `periodic_component_summary`, `np.histogram`, `summary.update`, `ValueError`, `np.all`, `np.isfinite`, `np.count_nonzero`, `np.mean`, `np.array`, `tolist`, `np.max`, +2 more

#### `calculate_periodic_clearance_distribution(obstacle_groups: Sequence[tuple[np.ndarray, float]], box_size: np.ndarray, grid_spacing: float=0.2, probe_radius: float=0.1657, bins: int=50, chunk_size: int=_DEFAULT_CHUNK_SIZE)` — line 362
- Calculate a mixed-radius periodic clearance distribution.
- kind: function
- returns: `(centres, hist, summary)`
- calls: `periodic_clearance_grid`, `summarize_periodic_clearance`, `summary.update`, `actual_spacing.tolist`, `np.asarray`

#### `calculate_pore_size_distribution(coords, box_size, grid_spacing=0.2, bead_radius=0.24, bins=50)` — line 412
- Compute a grid-based void radius distribution (legacy interface).
- kind: function
- returns: `(centres / 2.0, hist * 2.0, meta)`; `(np.array([]), np.array([]), meta)`
- calls: `np.asarray`, `calculate_periodic_clearance_distribution`, `meta.update`, `np.array`

#### `get_peak_pore_size(coords, box_size, grid_spacing=0.2, bead_radius=0.24, bins=50)` — line 463
- Return the peak pore diameter (nm) as a gated PropertyResult.
- kind: function
- returns: `PropertyResult(property='pore_size_single_frame_grid', value=peak_diameter, status='computed', direct_experiment_comparison_allowed=False, validation_role='proxy', metadata={**meta, 'target_aliases': ['pore_diameter_nm'], 'note': 'single-frame nearest-surface-grid; not comparable to Poreblazer trajectory pore diameter'})`; `PropertyResult.insufficient_data('pore_size_single_frame_grid', reason='void grid points가 0개입니다 — box가 polymer로 완전히 채워져 있거나 grid_spacing이 너무 큽니다.', validation_role='proxy', metadata={**meta, 'target_aliases': ['pore_diameter_nm']})`
- calls: `calculate_pore_size_distribution`, `PropertyResult`, `PropertyResult.insufficient_data`, `np.argmax`

### `hygel_martini/property_extract/requirements.py`

MD requirement gate driven by ``md_requirements.yaml``.

#### class `RequirementStatus` — line 29
Outcome of the requirement gate for one property.

##### `to_dict(self)` — line 60
- Serialize to a plain dict for JSON reports.
- kind: method
- returns: `{'property': self.property_name, 'md_required': self.md_required, 'satisfied': self.satisfied, 'validation_role': self.validation_role, 'required_md_jobs': self.required_md_jobs, 'missing_required_inputs': self.missing_required_inputs, 'missing_md_jobs': self.missing_md_jobs, 'invalid_inputs': self.invalid_inputs, 'notes': self.notes}`

#### `_resolve_glob(pattern: str, base_dir: str)` — line 75
- Expand a glob pattern relative to ``base_dir``.
- kind: function, internal
- returns: `matches`
- calls: `_glob.glob`, `os.path.isabs`, `os.path.join`

#### `_existing_input_by_basename(filename: str, available_files: dict[str, str])` — line 91
- Return the first available-file value whose basename matches.
- kind: function, internal
- returns: `None`; `str(value)`
- calls: `available_files.values`, `os.path.exists`, `os.path.basename`

#### `_existing_input_by_key(key: str, available_files: dict[str, str])` — line 108
- Return ``available_files[key]`` if it points to an existing file.
- kind: function, internal
- returns: `None`; `str(value)`
- calls: `available_files.get`, `os.path.exists`

#### `_candidate_output_matches(pattern: str, base_dir: str, available_files: dict[str, str])` — line 119
- Find on-disk artifacts satisfying one ``required_outputs`` entry.
- kind: function, internal
- returns: `[by_basename] if by_basename else []`; `matches`
- calls: `_resolve_glob`, `os.path.basename`, `_existing_input_by_basename`

#### `_validate_required_columns(paths: list[str], columns: list[str])` — line 145
- Check that every found .xvg artifact carries the required columns.
- kind: function, internal
- returns: `invalid`; `[]`
- calls: `endswith`, `parse_xvg`, `invalid.append`

#### `check_requirements(property_name: str, available_files: dict[str, str], md_requirements_path: str, base_dir: str | None=None)` — line 181
- Evaluate one property's MD requirement gate against the filesystem.
- kind: function
- returns: `RequirementStatus(property_name=property_name, md_required=md_required, satisfied=satisfied, validation_role=validation_role, required_md_jobs=required_jobs, missing_required_inputs=missing_inputs, missing_md_jobs=missing_jobs if not satisfied else [], invalid_inputs=invalid_inputs, notes=notes)`; `RequirementStatus(property_name=property_name, md_required=False, satisfied=False, validation_role='', required_md_jobs=[], missing_md_jobs=[], missing_required_inputs=[], invalid_inputs=[], notes=f"md_requirements.yaml에 '{property_name}' 항목이 없습니다.")`
- raises: `FileNotFoundError`
- effects: filesystem
- calls: `get`, `prop_reqs.get`, `available_files.values`, `invalid_inputs.extend`, `RequirementStatus`, `os.path.exists`, `FileNotFoundError`, `open`, `yaml.safe_load`, `req.get`, `os.getcwd`, `available_files.get`, +8 more

#### `check_all_requirements(properties: list[str], available_files: dict[str, str], md_requirements_path: str, base_dir: str | None=None)` — line 313
- Run :func:`check_requirements` for every property in ``properties``.
- kind: function
- returns: `{prop: check_requirements(prop, available_files, md_requirements_path, base_dir) for prop in properties}`
- calls: `check_requirements`

### `hygel_martini/property_extract/result.py`

Standard result container shared by every property extractor.

#### class `PropertyResult` — line 46
Standard return structure for every extractor.

##### `__post_init__(self)` — line 97
- Validate vocabularies and enforce the claim-boundary rule.
- kind: method
- raises: `ValueError`, `TypeError`
- calls: `join`, `ValueError`, `TypeError`

##### `to_dict(self)` — line 124
- Serialize to a plain dict for JSON output.
- kind: method
- returns: `{'property': self.property, 'value': self.value, 'status': self.status, 'calculability': self.status, 'direct_experiment_comparison_allowed': self.direct_experiment_comparison_allowed, 'validation_role': self.validation_role, 'missing_required_inputs': self.missing_required_inputs, 'metadata': self.metadata}`

##### `missing(property_name: str, missing_inputs: list[str], validation_role: str='', metadata: dict[str, Any] | None=None)` — line 142
- Build a ``missing_required_md`` result listing absent inputs.
- kind: staticmethod
- returns: `PropertyResult(property=property_name, value=None, status='missing_required_md', direct_experiment_comparison_allowed=False, validation_role=validation_role, missing_required_inputs=missing_inputs, metadata=metadata or {})`
- calls: `PropertyResult`

##### `invalid_input(property_name: str, reason: str, inputs: list[str] | None=None, validation_role: str='', metadata: dict[str, Any] | None=None)` — line 160
- Build an ``invalid_input`` result; ``reason`` goes to metadata.
- kind: staticmethod
- returns: `PropertyResult(property=property_name, value=None, status='invalid_input', direct_experiment_comparison_allowed=False, validation_role=validation_role, missing_required_inputs=inputs or [], metadata=meta)`
- calls: `PropertyResult`

##### `analysis_failed(property_name: str, error: str, inputs: list[str] | None=None, validation_role: str='', metadata: dict[str, Any] | None=None)` — line 181
- Build an ``analysis_failed`` result; ``error`` goes to metadata.
- kind: staticmethod
- returns: `PropertyResult(property=property_name, value=None, status='analysis_failed', direct_experiment_comparison_allowed=False, validation_role=validation_role, missing_required_inputs=inputs or [], metadata=meta)`
- calls: `PropertyResult`

##### `insufficient_data(property_name: str, reason: str, validation_role: str='', metadata: dict[str, Any] | None=None)` — line 202
- Build an ``insufficient_data`` result; ``reason`` to metadata.
- kind: staticmethod
- returns: `PropertyResult(property=property_name, value=None, status='insufficient_data', direct_experiment_comparison_allowed=False, validation_role=validation_role, metadata=meta)`
- calls: `PropertyResult`

##### `not_implemented(property_name: str, reason: str='')` — line 221
- Build a ``not_implemented`` result for an unsupported property.
- kind: staticmethod
- returns: `PropertyResult(property=property_name, value=None, status='not_implemented', direct_experiment_comparison_allowed=False, validation_role='', metadata={'reason': reason})`
- calls: `PropertyResult`

### `hygel_martini/property_extract/rheology.py`

Shear viscosity analysis from GROMACS energy XVGs.

#### `calculate_viscosity_green_kubo(energy_xvg, temperature, volume_nm3, dt_ps)` — line 16
- Placeholder: Green-Kubo shear viscosity.
- kind: function
- raises: `NotImplementedError`
- calls: `NotImplementedError`

#### `analyze_shear_rate_viscosity(energy_xvgs, shear_rates_ps_inv)` — line 32
- Compute NEMD steady-shear viscosities, one per shear-rate run.
- kind: function
- returns: `np.array(viscosities)`
- raises: `ValueError`
- calls: `np.array`, `parse_xvg`, `data.get`, `viscosities.append`, `ValueError`, `np.abs`, `np.mean`

### `hygel_martini/property_extract/simulation_protocols.py`

GROMACS MDP generation for the rheology/dynamics MD protocols.

#### class `MDProtocolGenerator` — line 18
GROMACS MDP file generator (draft status).

##### `__init__(self, base_mdp_path=None)` — line 41
- Store an optional base MDP path (currently unused by getters).
- kind: method

##### `get_shear_mdp(self, shear_rate, temperature=310.15, nsteps=10000000, dt=0.02, tau_t=1.0, tau_p=12.0, ref_p=1.013, compressibility=4.5e-05, nstenergy=1000)` — line 45
- Render the NEMD shear MDP text.
- kind: method
- returns: `f'integrator          = md\nnsteps              = {nsteps}\ndt                  = {dt}\ncomm-mode           = Linear\nnstxout             = 5000\nnstvout             = 5000\nnstfout             = 0\nnstenergy           = {nstenergy}\nnstlog              = 5000\n' + self._MARTINI_NONBONDED + f'\n; T-coupling\ntcoupl              = v-rescale\ntc-grps             = System\ntau_t               = {tau_t}\nref_t               = {temperature}\n\n; P-coupling with shear (초안 — 물리적 적합성 검증 필요)\npcoupl              = Parrinello-Rahman\npcoupltype          = anisotropic\ntau_p               = {tau_p}\ncompressibility     = {compr} {compr} {compr} 0 0 0\nref_p               = {ref_p} {ref_p} {ref_p} 0 0 0\ndeform              = 0 0 0 {shear_rate} 0 0  ; Shear in XY plane\n'`

##### `get_dynamics_mdp(self, temperature=310.15, nsteps=5000000, dt=0.02, tau_t=1.0, tau_p=12.0, ref_p=1.013, compressibility=4.5e-05, nstenergy=10)` — line 105
- Render an MDP for high-frequency energy output (Green-Kubo).
- kind: method
- returns: `f'integrator          = md\nnsteps              = {nsteps}\ndt                  = {dt}\nnstenergy           = {nstenergy}  ; ACF용 고빈도\nnstxout             = 5000\n' + self._MARTINI_NONBONDED + f'\ntcoupl              = v-rescale\ntc-grps             = System\ntau_t               = {tau_t}\nref_t               = {temperature}\n\npcoupl              = Parrinello-Rahman\npcoupltype          = isotropic\ntau_p               = {tau_p}\nref_p               = {ref_p}\ncompressibility     = {compr}\n'`

##### `create_shear_series(self, output_dir, rates=None, temperature=310.15, **mdp_kwargs)` — line 156
- Create per-shear-rate directories with their shear MDPs.
- kind: method
- returns: `created`
- effects: filesystem
- calls: `os.path.join`, `os.makedirs`, `created.append`, `open`, `f.write`, `self.get_shear_mdp`

### `hygel_martini/property_extract/spatial.py`

Resolution-explicit voxel heterogeneity analysis.

#### `voxel_counts(positions: np.ndarray, box: np.ndarray, target_spacing: float)` — line 25
- Count positions in a periodic grid and return counts and actual spacing.
- kind: function
- returns: `(counts, spacing)`
- raises: `ValueError`
- calls: `orthorhombic_box_lengths`, `np.maximum`, `wrap_positions`, `astype`, `np.minimum`, `np.zeros`, `np.add.at`, `ValueError`, `np.floor`

#### `summarize_voxel_counts(counts: np.ndarray)` — line 64
- Summarize count heterogeneity without converting it to a pore size.
- kind: function
- returns: `{'n_voxels': int(values.size), 'mean_count': mean, 'std_count': float(np.std(values)), 'coefficient_of_variation': float(np.std(values) / mean) if mean else float('nan'), 'empty_fraction': float(np.mean(values == 0)), 'percentiles_5_25_50_75_95': percentiles.tolist(), 'minimum': float(np.min(values)), 'maximum': float(np.max(values)), 'interpretation': 'resolution-dependent composition diagnostic; not a physical pore size'}`
- raises: `ValueError`
- calls: `ravel`, `np.percentile`, `ValueError`, `np.mean`, `percentiles.tolist`, `np.asarray`, `np.std`, `np.min`, `np.max`

#### `periodic_field_correlation(reference: np.ndarray, current: np.ndarray)` — line 98
- Correlate equal-shaped periodic fields before and after translation.
- kind: function
- returns: `{'zero_shift_correlation': zero_shift, 'translation_aligned_correlation': maximum, 'best_periodic_shift_cells': signed_shift}`
- raises: `ValueError`
- calls: `np.asarray`, `np.unravel_index`, `ValueError`, `np.mean`, `np.sqrt`, `np.fft.ifftn`, `np.isfinite`, `np.sum`, `np.argmax`, `np.conj`, `np.fft.fftn`

#### `phase_randomized_field(field: np.ndarray, rng: np.random.Generator)` — line 155
- Randomize spatial phase while preserving mean and Fourier amplitude.
- kind: function
- returns: `np.fft.ifftn(amplitude * unit_phase).real + mean`
- raises: `ValueError`
- calls: `np.asarray`, `np.abs`, `np.fft.fftn`, `np.divide`, `ValueError`, `np.all`, `np.mean`, `rng.normal`, `np.isfinite`, `np.ones_like`, `np.fft.ifftn`

### `hygel_martini/property_extract/structure_factor.py`

Static structure-factor primitives for periodic coarse-grained systems.

#### `_validate_positions(positions: np.ndarray)` — line 28
- Coerce to a float ``(n, 3)`` array; reject empty/non-finite input.
- kind: function, internal
- returns: `pos`
- raises: `ValueError`
- calls: `np.asarray`, `ValueError`, `np.all`, `np.isfinite`

#### `_validate_grid_shape(grid_shape: int | Sequence[int])` — line 38
- Expand an int to a cubic shape; require three integers >= 4.
- kind: function, internal
- returns: `shape`
- raises: `ValueError`
- calls: `ValueError`

#### `cic_density_grid(positions: np.ndarray, box: np.ndarray, grid_shape: int | Sequence[int])` — line 49
- Deposit particles on a periodic grid using cloud-in-cell assignment.
- kind: function
- returns: `density`
- raises: `RuntimeError`
- calls: `_validate_positions`, `orthorhombic_box_lengths`, `_validate_grid_shape`, `np.asarray`, `astype`, `np.zeros`, `np.mod`, `np.isclose`, `RuntimeError`, `np.floor`, `np.add.at`, `density.sum`

#### `fft_structure_factor(positions: np.ndarray, box: np.ndarray, grid_shape: int | Sequence[int]=64, *, q_max: float | None=None, deconvolve_cic: bool=True)` — line 103
- Return reciprocal-vector magnitudes and static ``S(q)`` values.
- kind: function
- returns: `(q_magnitude[keep], structure[keep])`
- raises: `ValueError`
- calls: `_validate_positions`, `orthorhombic_box_lengths`, `_validate_grid_shape`, `cic_density_grid`, `np.fft.rfftn`, `np.sqrt`, `np.fft.fftfreq`, `np.fft.rfftfreq`, `np.sinc`, `np.abs`, `ValueError`, `np.isfinite`

#### `radial_bin_structure_factor(q_magnitude: np.ndarray, structure_factor: np.ndarray, bin_edges: np.ndarray)` — line 173
- Radially average reciprocal modes into explicit ``q`` bins.
- kind: function
- returns: `{'q_lower': edges[:-1], 'q_upper': edges[1:], 'q_center': 0.5 * (edges[:-1] + edges[1:]), 'mean_structure_factor': mean, 'n_modes': count}`
- raises: `ValueError`
- calls: `ravel`, `np.asarray`, `astype`, `np.bincount`, `np.full`, `np.divide`, `ValueError`, `np.any`, `np.digitize`, `np.diff`, `np.isfinite`

#### `reciprocal_axis_structure_factor(positions: np.ndarray, box: np.ndarray, max_mode: int)` — line 220
- Calculate exact static structure factors along box ``x/y/z`` axes.
- kind: function
- returns: `output`
- raises: `ValueError`
- calls: `_validate_positions`, `orthorhombic_box_lengths`, `np.arange`, `ValueError`, `np.outer`, `np.exp`, `np.abs`

### `hygel_martini/property_extract/swelling.py`

Swelling/composition analysis from GROMACS topology and energy files.

#### class `SwellingAnalyzer` — line 28
Composition and volume-fraction calculator for one built system.

##### `__init__(self, n_polymer_beads, n_solvent_beads, polymer_bead_mass=45.0, solvent_bead_mass=72.0, polymer_bead_vol_nm3=0.065)` — line 37
- Store bead counts and per-bead mass/volume constants.
- kind: method

##### `composition_summary(self)` — line 63
- Summarize the as-built composition without any MD output.
- kind: method
- returns: `PropertyResult(property='loading_qm', value=loading_qm, status='computed', direct_experiment_comparison_allowed=False, validation_role='composition_check_only', metadata={'note': 'count-based initial composition ratio; not equilibrium swelling (fixed-water NPT)', 'n_polymer_beads': self.n_polymer_beads, 'n_solvent_beads': self.n_solvent_beads})`
- calls: `self.calculate_loading_qm`, `PropertyResult`

##### `calculate_loading_qm(self)` — line 93
- loading_qm = (m_polymer + m_solvent) / m_polymer
- kind: method
- returns: `mass_wet / mass_dry`

##### `calculate_phi(self, box_volume_nm3: float)` — line 99
- polymer volume fraction phi = V_polymer / V_box
- kind: method
- returns: `v_poly / box_volume_nm3`

##### `analyze_trajectory(self, energy_xvg, start_time_ps=0)` — line 104
- Compute polymer_volume_fraction from an energy XVG's Volume.
- kind: method
- returns: `PropertyResult(property='polymer_volume_fraction', value=phi, status='computed', direct_experiment_comparison_allowed=True, validation_role='direct', metadata={'phi_std': phi_std, 'method': f'bead_volume (vol_per_bead={self.polymer_bead_vol_nm3} nm3)', 'vol_avg_nm3': avg_vol, 'vol_std_nm3': std_vol, 'start_time_ps': start_time_ps, 'n_frames_used': int(mask.sum())})`
- raises: `ValueError`
- calls: `parse_xvg`, `next`, `np.mean`, `np.std`, `self.calculate_phi`, `PropertyResult`, `ValueError`, `mask.sum`

##### `from_files(cls, top_file, itp_file, polymer_bead_mass=45.0, solvent_bead_mass=72.0, polymer_bead_vol_nm3=0.065, polymer_residue_name=None, polymer_atom_name=None, solvent_molecule_names='W')` — line 162
- Build an analyzer by counting beads in top/itp files.
- kind: classmethod
- returns: `cls(n_polymer_beads=n_polymer, n_solvent_beads=n_solvent, polymer_bead_mass=polymer_bead_mass, solvent_bead_mass=solvent_bead_mass, polymer_bead_vol_nm3=polymer_bead_vol_nm3)`
- effects: filesystem
- calls: `os.path.exists`, `cls`, `open`, `line.strip`, `stripped.startswith`, `re.search`, `stripped.split`, `m.group`

### `hygel_martini/property_extract/timeseries.py`

Reusable time-series statistics for simulation observables.

#### `read_xvg(path: str | Path)` — line 19
- Read numeric XVG data and return ``(legends, array)``.
- kind: function
- returns: `(legends, np.asarray(rows, dtype=float))`
- raises: `ValueError`
- calls: `splitlines`, `raw.strip`, `line.startswith`, `rows.append`, `ValueError`, `np.asarray`, `read_text`, `legends.append`, `line.split`, `Path`, `rsplit`

#### `select_time_window(times: np.ndarray, values: np.ndarray, start: float | None=None, end: float | None=None)` — line 65
- Select an inclusive time window with shape and monotonicity checks.
- kind: function
- returns: `(t[mask], y[mask])`
- raises: `ValueError`
- calls: `np.asarray`, `np.ones`, `ValueError`, `np.any`, `np.diff`

#### `block_statistics(values: np.ndarray, n_blocks: int=5)` — line 104
- Return sample statistics and SEM across contiguous block means.
- kind: function
- returns: `{'n_samples': int(y.size), 'n_blocks': int(len(blocks)), 'mean': float(np.mean(y)), 'sample_std': float(np.std(y, ddof=1)), 'block_means': block_means.tolist(), 'block_sem': float(np.std(block_means, ddof=1) / np.sqrt(len(blocks)))}`
- raises: `ValueError`
- calls: `np.asarray`, `ValueError`, `block_means.tolist`, `np.array_split`, `np.mean`, `np.std`, `np.sqrt`

#### `linear_drift(times: np.ndarray, values: np.ndarray)` — line 140
- Fit a linear drift and report total/relative change over the window.
- kind: function
- returns: `{'slope_per_time': float(slope), 'intercept': float(intercept), 'window_duration': duration, 'fitted_change': change, 'relative_change': float(change / mean) if mean != 0 else float('nan')}`
- raises: `ValueError`
- calls: `np.asarray`, `np.polyfit`, `ValueError`, `np.mean`

### `hygel_martini/property_extract/validation_manifest.py`

Schema and loader for ``validation_manifest.yaml``.

#### class `ManifestProperty` — line 35
One experimental target property inside a manifest target.

##### `is_directly_comparable(self, simulation_property: str)` — line 71
- True if the manifest declares this simulation property comparable.
- kind: method
- returns: `simulation_property in self.comparable_to`

##### `is_not_comparable(self, simulation_property: str)` — line 75
- True if the manifest forbids comparison with this property.
- kind: method
- returns: `simulation_property in self.not_comparable_to`

#### class `ManifestTarget` — line 81
One experimental target: a reference plus its measured properties.

##### `get_property(self, name: str)` — line 98
- Return the named property, or ``None`` if absent.
- kind: method
- returns: `self.properties.get(name)`
- calls: `self.properties.get`

#### `_parse_property(name: str, spec: dict)` — line 103
- Parse and validate one property spec into a ManifestProperty.
- kind: function, internal
- returns: `ManifestProperty(name=name, unit=str(spec['unit']), value=float(spec['value']) if spec.get('value') is not None else None, min=float(spec['min']) if spec.get('min') is not None else None, max=float(spec['max']) if spec.get('max') is not None else None, uncertainty=float(spec['uncertainty']) if spec.get('uncertainty') is not None else None, tolerance=float(spec['tolerance']) if spec.get('tolerance') is not None else None, definition=str(spec.get('definition', '')), method=str(spec.get('method', '')), comparable_to=comparable_to, not_comparable_to=not_comparable_to, notes=str(spec.get('notes', '')), raw=spec)`
- raises: `ValueError`
- calls: `_as_string_list`, `ManifestProperty`, `ValueError`, `spec.get`

#### `_parse_target(target_id: str, spec: dict)` — line 137
- Parse and validate one target spec into a ManifestTarget.
- kind: function, internal
- returns: `ManifestTarget(target_id=target_id, reference=str(spec['reference']), formulation=dict(spec.get('formulation') or {}), properties=properties)`
- raises: `ValueError`
- calls: `properties_spec.items`, `ManifestTarget`, `ValueError`, `spec.get`, `_parse_property`, `spec.keys`

#### `_as_string_list(value: Any, field_name: str)` — line 164
- Coerce an optional YAML value into ``list[str]``.
- kind: function, internal
- returns: `list(value)`; `[]`
- raises: `ValueError`
- calls: `ValueError`

#### `_validate_references(references: dict[str, Any])` — line 181
- Validate the manifest's ``references`` map.
- kind: function, internal
- raises: `ValueError`
- calls: `references.items`, `ValueError`, `ref.get`, `ref.keys`

#### `_validate_property_value(target_id: str, prop: ManifestProperty)` — line 204
- Placeholder value check — intentionally accepts value-less entries.
- kind: function, internal
- returns: `None`

#### `_validate_targets_against_references(targets: list[ManifestTarget], references: dict[str, Any])` — line 212
- Ensure every target cites a declared reference.
- kind: function, internal
- raises: `ValueError`
- calls: `target.properties.values`, `ValueError`, `_validate_property_value`

#### `load_manifest(path: str)` — line 232
- Read validation_manifest.yaml into validated ManifestTargets.
- kind: function
- returns: `targets`
- raises: `FileNotFoundError`, `ValueError`
- effects: filesystem
- calls: `os.path.abspath`, `_validate_references`, `items`, `_validate_targets_against_references`, `os.path.exists`, `FileNotFoundError`, `open`, `yaml.safe_load`, `ValueError`, `data.get`, `targets.append`, `data.keys`, +1 more

#### `get_target_property(targets: list[ManifestTarget], property_name: str)` — line 276
- Return the first (target, property) match by exact property name.
- kind: function
- returns: `(None, None)`; `(target, prop)`
- calls: `target.get_property`

#### `find_target_properties_for_simulation_property(targets: list[ManifestTarget], simulation_property: str)` — line 293
- Find manifest target properties related to a simulation property.
- kind: function
- returns: `matches`
- calls: `target.properties.values`, `matches.append`

## `hygel_martini/tools`

### `hygel_martini/tools/__init__.py`

*(no module docstring)*

### `hygel_martini/tools/audit_hydrogel_topology.py`

Generic bonded-topology audit for hydrogel_builder outputs.

#### `_csv_tokens(value: str | None)` — line 29
- Split a comma-separated CLI value into a set of stripped tokens.
- kind: function, internal
- returns: `{token.strip() for token in value.split(',') if token.strip()}`; `set()`
- calls: `token.strip`, `value.split`

#### `_parse_pattern(value: str | None)` — line 36
- Parse "RES:COUNT,RES:COUNT,..." into an ordered (residue, count) list.
- kind: function, internal
- returns: `pattern`; `[]`
- raises: `ValueError`
- calls: `value.split`, `raw.strip`, `token.split`, `pattern.append`, `ValueError`, `residue.strip`, `count.strip`

#### `parse_itp(path: Path)` — line 56
- Parse the atoms/bonds/angles/dihedrals sections of a GROMACS .itp.
- kind: function
- returns: `(atoms, bonds, angles, dihedrals)`
- calls: `splitlines`, `strip`, `line.startswith`, `line.split`, `path.read_text`, `lower`, `isdigit`, `bonds.append`, `raw.split`, `line.strip`, `angles.append`, `dihedrals.append`, +1 more

#### `connected_components(atom_ids: Iterable[int], bonds: list[tuple[int, int, list[str]]])` — line 99
- Find connected components of the bond graph restricted to atom_ids.
- kind: function
- returns: `(sorted(components, key=len, reverse=True), adj)`
- calls: `deque`, `seen.add`, `components.append`, `append`, `queue.popleft`, `comp.append`, `queue.append`

#### `path_order(component: list[int], adj: dict[int, list[int]])` — line 135
- Walk a (near-)linear component from an endpoint, returning id order.
- kind: function
- returns: `ordered`
- calls: `ordered.append`, `degrees.items`, `adj.get`

#### `expected_sequence(pattern: list[tuple[str, int]])` — line 160
- Expand (residue, count) pattern pairs into a flat residue sequence.
- kind: function
- returns: `seq`
- calls: `seq.extend`

#### `audit(args)` — line 168
- Run every audit check on the parsed .itp and print the summary.
- kind: function
- returns: `1 if issues and args.fail_on_issue else 0`
- effects: stdout
- calls: `parse_itp`, `_csv_tokens`, `_parse_pattern`, `expected_sequence`, `connected_components`, `Counter`, `all_adj.items`, `atoms.keys`, `path_order`, `chain_reports.append`, `residues.get`, `issues.append`, +18 more

#### `main()` — line 386
- Parse CLI options and exit with the audit result code.
- kind: function, CLI entry
- raises: `SystemExit`
- calls: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `SystemExit`, `audit`

### `hygel_martini/tools/xtb_traj_to_pdb.py`

Convert an xTB XYZ trajectory to a multi-MODEL PDB with auto-trimming.

#### `extract_energy(comment: str)` — line 52
- Extract the "energy:" value (Hartree) from an xTB comment line.
- kind: function
- returns: `None`; `float(match.group(1))`
- calls: `ENERGY_RE.search`, `match.group`

#### `parse_frames_streaming(path: Path)` — line 60
- Yield (comment_line, atom_lines) for each frame in an XYZ trajectory.
- kind: function
- returns: `<generator>`
- effects: filesystem
- calls: `path.open`, `f.readline`, `strip`, `atom_lines.append`, `line.strip`, `atom_line.strip`

#### `parse_pdb_frame_count(path: Path)` — line 92
- Count MODEL records in a PDB file (proxy for frame count).
- kind: function
- returns: `count`
- effects: filesystem
- calls: `path.open`, `line.startswith`

#### `pdb_atom_line(atom_index: int, symbol: str, x: float, y: float, z: float)` — line 102
- Format one fixed-width PDB ATOM record (resname MOL, chain A, res 1).
- kind: function
- returns: `f'ATOM  {atom_index:5d} {atom_name:<4} MOL A{1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {symbol[:2].upper():>2}\n'`
- calls: `rjust`, `upper`

#### `detect_t0(path: Path, nskip: int=1, max_trim_fraction: float=1.0, detrend: bool=False, fast: bool=False)` — line 115
- Pass 1: Extract energies, detect equilibration start via pymbar.
- kind: function
- returns: `(0, total, [])`; `(0, total_frames, energies)`; `(int(t0), total_frames, energies)`
- effects: stdout
- calls: `parse_frames_streaming`, `print`, `np.array`, `_tqdm`, `extract_energy`, `np.arange`, `np.polyfit`, `timeseries.detect_equilibration`, `energies.append`

#### `detect_t0_energy_threshold(path: Path, ref_fraction: float=0.2, threshold_sigma: float=1.0, max_trim_fraction: float=1.0)` — line 187
- Pass 1: Extract energies, detect t0 by tail-mean convergence.
- kind: function
- returns: `(t0, total_frames, energies)`; `(0, total, [])`; `(0, total_frames, energies)`
- effects: stdout
- calls: `parse_frames_streaming`, `print`, `np.array`, `np.arange`, `_tqdm`, `extract_energy`, `np.mean`, `np.cumsum`, `np.abs`, `np.where`, `energies.append`, `np.std`

#### `_block_standard_error(data, max_block_size=None)` — line 277
- Compute block standard error vs block size for a convergence check.
- kind: function, internal
- returns: `(block_sizes, bse)`
- calls: `block_sizes.append`, `bse.append`, `np.mean`, `np.std`, `np.sqrt`

#### `save_convergence_plots(energies: list, t0: int, start_index: int, output_path: Path)` — line 302
- Save energy time-series and block-averaging plots alongside the PDB.
- kind: function
- returns: `None` (bare return)
- effects: stdout
- calls: `np.array`, `np.arange`, `plt.subplots`, `plot`, `set_ylabel`, `set_title`, `legend`, `axhline`, `set_xlabel`, `plt.tight_layout`, `output_path.with_name`, `fig.savefig`, +13 more

#### `main()` — line 375
- CLI driver: optional pass-1 t0 detection, then pass-2 PDB writing.
- kind: function, CLI entry
- returns: `None` (bare return)
- effects: filesystem, stdout
- calls: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `Path`, `print`, `output_path.with_name`, `trim_info_path.write_text`, `parse_pdb_frame_count`, `output_path.open`, `parse_frames_streaming`, `json.dumps`, `save_convergence_plots`, +6 more
