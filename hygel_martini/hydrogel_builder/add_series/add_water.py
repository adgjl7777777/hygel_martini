"""Helpers for estimating solvent content in hydrogel construction.

The water-addition stage uses an approximate mass-balance model rather than a
target density. That keeps the logic independent of the current box size and
lets the user specify the desired gel weight fraction directly in the config.
"""

import math
import os
import sys

from hygel_martini.hydrogel_builder.config_params.config import Config
from hygel_martini.hydrogel_builder.add_series.add_small_ion import resolve_effective_ion_plan
from hygel_martini.hydrogel_builder.core_utils.io.martini_parser import read_itp_definitions
from hygel_martini.hydrogel_builder.core_utils.templates.monomer_loader import load_monomer_templates


DEFAULT_WATER_MASSES = {
    "TW": 36.0,
    "SW": 54.0,
    "W": 72.0,
    "P4": 72.0,
}
LEGACY_GEL_WEIGHT_FRACTION_MODE = "legacy"
GEL_WEIGHT_FRACTION_MODES = {
    LEGACY_GEL_WEIGHT_FRACTION_MODE,
    "exclude_ions",
    "include_ions",
}

def get_weighted_average_mass(*args):
    """
    Compute a ratio-weighted mean mass for configured components.
    """
    total_mass = 0
    total_ratio = 0
    
    components = Config.get_param(*args)
    
    if not components:
        # Extract the last part of the args for a more informative message
        component_name = args[-1] if args else "components"
        print(f"Warning: No definitions found for {component_name}.")
        return 0

    first = components[0]
    if 'definition' in first:
        for component in components:
            ratio = component.get('ratio', 1)
            component_mass = sum(bead.get('mass', 0.0) for bead in component['definition'].get('beads', []))
            total_mass += component_mass * ratio
            total_ratio += ratio
    else:
        # GRO/ITP 기반 템플릿
        backbone_defs = Config.get_param('hydrogel_components', 'backbone_definitions', 'BACKBONES')
        library = load_monomer_templates(components, backbone_defs)
        for record in library.records:
            total_mass += record.template.total_mass * record.ratio
            total_ratio += record.ratio
        
    return total_mass / total_ratio if total_ratio > 0 else 0


def _safe_get_param(*keys, default=None):
    try:
        return Config.get_param(*keys)
    except KeyError:
        return default


def _resolve_gel_weight_fraction_mode(sim_params):
    mode = str(sim_params.get("gel_weight_fraction_mode", LEGACY_GEL_WEIGHT_FRACTION_MODE)).strip().lower()
    if mode not in GEL_WEIGHT_FRACTION_MODES:
        raise ValueError(
            "Invalid simulation_parameters.gel_weight_fraction_mode: "
            f"{mode}. Must be one of {sorted(GEL_WEIGHT_FRACTION_MODES)}"
        )
    return mode


def _load_definition_lookup(itp_paths):
    definitions = {}
    mass_map = Config.get_runtime("atom_type_masses", {})
    for itp_path in itp_paths:
        if not itp_path or not os.path.isfile(itp_path):
            continue
        try:
            defs = read_itp_definitions(
                itp_path,
                atom_type_masses=mass_map,
                prefer_explicit_masses=True,
            )
        except Exception:
            continue
        definitions.update(defs)
    return definitions


def _estimate_ion_usage(sim_params):
    ion_params = _safe_get_param("add_series_parameters", "add_small_ion", default={}) or {}
    if not ion_params.get("ions"):
        return 0, 0.0

    effective_ions = resolve_effective_ion_plan(
        ion_params,
        seed=sim_params.get("random_seed", 0),
    )
    total_ion_count = sum(int(ion.get("number", 0) or 0) for ion in effective_ions)
    if total_ion_count <= 0:
        return 0, 0.0

    candidate_itps = []
    include_dir = sim_params.get("gromacs_include_path")
    if include_dir:
        candidate_itps.append(os.path.join(include_dir, "martini_v3.0.0_ions_v1.itp"))
    candidate_itps.extend(_safe_get_param("additional_itp_files", default=[]) or [])
    candidate_itps.extend(ion_params.get("additional_ion_itp_files", []) or [])

    definitions = _load_definition_lookup(candidate_itps)
    total_ion_mass = 0.0
    missing_ions = set()

    for ion in effective_ions:
        ion_name = ion.get("ion_name")
        count = int(ion.get("number", 0) or 0)
        if not ion_name or count <= 0:
            continue
        definition = definitions.get(ion_name)
        if not definition:
            missing_ions.add(str(ion_name))
            continue
        mol_mass = sum(bead.get("mass", 0.0) for bead in definition.get("beads", []))
        total_ion_mass += mol_mass * count

    if missing_ions:
        raise ValueError(
            "Could not resolve masses for configured ions: "
            + ", ".join(sorted(missing_ions))
        )

    return total_ion_count, total_ion_mass

def calculate_water_molecules(mode):
    """
    Estimate how many coarse-grained water beads should be inserted.
    """
    sim_params = Config.get_param('simulation_parameters')
    add_water_params = Config.get_param('add_series_parameters', 'add_water')
    water_masses = add_water_params.get('water_masses', DEFAULT_WATER_MASSES)
    
    # --- Get parameters from config ---
    nmer = sim_params['segment_length']
    num_cell = sim_params['number_of_cells']
    gel_wt = add_water_params['gel_weight_fraction']
    water_bead_type = add_water_params.get('water_bead_type', 'W')
    gel_fraction_mode = _resolve_gel_weight_fraction_mode(sim_params)

    # Derive the dry gel mass from the configured topology composition.
    mass_monomer = get_weighted_average_mass('monomer_definitions', 'MONOMERS')
    mass_bis = get_weighted_average_mass('hydrogel_components', 'linker_definitions', 'LINKERS')
    mass_water = water_masses.get(water_bead_type)

    if not mass_water:
        raise ValueError(f"Invalid water_bead_type: {water_bead_type}. Must be one of {list(water_masses.keys())}")

    print(f"Calculation mode: {mode}")
    print(f"Gel weight fraction mode: {gel_fraction_mode}")
    print(f"Nmer: {nmer}, Gel weight fraction: {gel_wt}, NumCell: {num_cell}")
    print(f"Weighted avg. monomer mass: {mass_monomer:.2f}")
    print(f"Weighted avg. bis-linker mass: {mass_bis:.2f}")
    print(f"Selected water bead: {water_bead_type} (Mass: {mass_water:.2f})")

    unit_mer = nmer * 4 * 4
    tot_mer = unit_mer * (num_cell / 2)**3
    pol_mass = tot_mer * mass_monomer

    if mode == 'full':
        tot_bis_mass = 8 * (num_cell / 2)**3 * mass_bis
        total_gel_mass = pol_mass + tot_bis_mass
    else: # monomer_only
        total_gel_mass = pol_mass

    # Backbone-defined gels have MONOMERS: [] even when explicit linkers exist.
    # In that case the generated World is the authoritative dry-gel mass source.
    if mass_monomer == 0 or total_gel_mass == 0:
        try:
            from hygel_martini.hydrogel_builder.main_components.Universe import World
            world_mass = sum(float(atom[0].mass) for atom in World.Atoms.values())
            if world_mass > 0:
                total_gel_mass = world_mass
        except Exception as exc:
            print(
                f"[WARN] Could not read dry mass from the generated World: {exc}",
                file=sys.stderr,
            )

    if not (0 < gel_wt < 1):
        raise ValueError("Error: gel_weight_fraction must be between 0 and 1.")

    # A zero dry mass silently propagates to zero added water, i.e. a dry
    # system that still builds and still runs.  Fail here instead.
    if total_gel_mass <= 0:
        raise ValueError(
            "Dry gel mass resolved to {!r}, so the requested "
            "gel_weight_fraction cannot be applied. Neither the monomer "
            "definitions nor the generated World supplied a mass; check that "
            "the hydrogel was built before add_water and that monomer ITPs "
            "carry masses.".format(total_gel_mass)
        )

    target_added_mass = (total_gel_mass / gel_wt) - total_gel_mass
    water_mass = target_added_mass
    reserved_water_sites = 0
    planned_ion_count = 0
    planned_ion_mass = 0.0

    if gel_fraction_mode != LEGACY_GEL_WEIGHT_FRACTION_MODE:
        planned_ion_count, planned_ion_mass = _estimate_ion_usage(sim_params)
        reserved_water_sites = planned_ion_count
        if gel_fraction_mode == "include_ions":
            water_mass = max(target_added_mass - planned_ion_mass, 0.0)

    if gel_fraction_mode == LEGACY_GEL_WEIGHT_FRACTION_MODE:
        n_water = int(water_mass / mass_water)
    else:
        n_water = math.ceil(water_mass / mass_water) + reserved_water_sites

    print(f"Total gel mass: {total_gel_mass:.2f}")
    print(f"Target added mass from gel fraction: {target_added_mass:.2f}")
    if gel_fraction_mode != LEGACY_GEL_WEIGHT_FRACTION_MODE:
        print(f"Planned ion count: {planned_ion_count}")
        print(f"Planned ion mass: {planned_ion_mass:.2f}")
        print(f"Target final water mass: {water_mass:.2f}")
        print(f"Reserved water sites for ion replacement: {reserved_water_sites}")
        print(f"Estimated final water beads after ion insertion: {max(n_water - reserved_water_sites, 0)}")
    else:
        print(f"Required water mass: {water_mass:.2f}")
    print(f"Number of water molecules to add: {n_water}")

    return n_water
