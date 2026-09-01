"""Backbone planning and materialization utilities for hydrogel generation.

This module owns the transition between declarative configuration data and the
mutable global ``World`` object used by the legacy construction code.

The build pipeline intentionally runs in two conceptual phases:

1. Planning
   Build a proto-network, decide whether the run is anisotropic or isotropic,
   and generate an atom-level blueprint.
2. Materialization
   Reset the mutable world state, instantiate the ``Hydrogel`` object, populate
   it from the blueprint, and then derive bonds, side chains, angles, and other
   topology sections.

Keeping those phases separate is important for reproducibility because several
downstream routines mutate ``World`` class state in-place.
"""

import os
import random
import traceback

import numpy as np

from hygel_martini.hydrogel_builder.config_params.config import Config
from hygel_martini.hydrogel_builder.core_utils.common.utility import find_minimum_distances
from hygel_martini.hydrogel_builder.core_utils.layout.isotropic_builder import build_isotropic_blueprint
from hygel_martini.hydrogel_builder.core_utils.layout.layout_executor import build_atom_blueprint
from hygel_martini.hydrogel_builder.core_utils.layout.proto_builder import prepare_proto_plan
from hygel_martini.hydrogel_builder.core_utils.layout.net_layout import generate_net_layout_plan
from hygel_martini.hydrogel_builder.core_utils.layout.proto_layout import generate_layout_plan
from hygel_martini.hydrogel_builder.core_utils.layout.proto_populator import populate_hydrogel_from_blueprint
from hygel_martini.hydrogel_builder.core_utils.templates.linker_loader import (
    linker_definitions_from_library,
    load_linker_templates,
)
from hygel_martini.hydrogel_builder.core_utils.templates.monomer_loader import load_monomer_templates
from hygel_martini.hydrogel_builder.main_components import Attributes
from hygel_martini.hydrogel_builder.main_components.Universe import World, initialize_world


def _debug_stage(message):
    """Emit a stage marker to the optional debug log."""
    try:
        Config.debug_log(message)
    except Exception:
        pass


def _print_build_banner():
    """Print the standard banner used by backbone-construction stages."""
    print("=" * 50)
    print("하이드로젤 구조 생성을 시작합니다.")
    print("=" * 50)


def _seed_random_generators(seed):
    """Seed Python and NumPy RNGs when a deterministic run is requested."""
    if seed is None:
        return
    try:
        seed_val = int(seed)
    except Exception:
        return
    random.seed(seed_val)
    np.random.seed(seed_val)


def _compute_max_linker_span():
    """Return the largest linker span declared in the configuration."""
    try:
        linker_defs = Config.get_param(
            "hydrogel_components",
            "linker_definitions",
            "LINKERS",
        )
    except (KeyError, ValueError):
        return 0.0

    max_span = 0.0
    for linker in linker_defs:
        definition = linker.get("definition", {})
        span = definition.get("span_from_gro")
        if span:
            total_length = span
        else:
            bonds = definition.get("bonds", [])
            external_bonds = definition.get("external_bonds", [])
            total_internal = sum(bond.get("length", 0.0) for bond in bonds)
            total_external = sum(ext.get("length", 0.0) for ext in external_bonds)
            total_length = total_internal + total_external
        max_span = max(max_span, total_length)
    return max_span


def _gather_sorted_atoms():
    """Collect the current ``World`` atoms in deterministic atom-id order."""
    atom_ids = sorted(World.Atoms.keys())
    atoms = []
    for atom_id in atom_ids:
        if not World.Atoms[atom_id]:
            continue
        atoms.append(World.Atoms[atom_id][0])
    return atom_ids, atoms


def _log_min_distance_report(label):
    """Write a compact minimum-distance report for debugging."""
    output_dir = Config.get_param("simulation_parameters", "output_dir")
    os.makedirs(output_dir, exist_ok=True)

    _, atoms = _gather_sorted_atoms()
    if len(atoms) < 2:
        return
    try:
        max_atoms = Config.get_param(
            "simulation_parameters",
            "min_distance_report_max_atoms",
        )
    except Exception:
        max_atoms = 3000
    if max_atoms is not None and len(atoms) > int(max_atoms):
        msg = f"[min_distance] skipped: atoms={len(atoms)} > max_atoms={max_atoms}"
        print(msg)
        _debug_stage(msg)
        return

    positions = np.array([atom.position for atom in atoms], dtype=np.float64)
    distances = find_minimum_distances(
        positions,
        World.box_length if World.box_length else None,
        top_n=10,
    )
    if not distances:
        return

    log_path = os.path.join(output_dir, f"min_distance_{label}.log")
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write("distance_nm atom_i atom_j\n")
        for distance, idx_i, idx_j in distances:
            atom_i = atoms[idx_i]
            atom_j = atoms[idx_j]
            log_f.write(
                f"{distance:.4f} "
                f"{atom_i.residue_name}:{atom_i.atom_name}({atom_i.atom_id}) "
                f"{atom_j.residue_name}:{atom_j.atom_name}({atom_j.atom_id})\n"
            )


def apply_coordinates_from_gro(world, gro_path):
    """Project coordinates from a GRO file back into the current ``World``."""
    del world
    if not os.path.exists(gro_path):
        print(f"[경고] 지정된 GRO 파일을 찾을 수 없습니다: {gro_path}")
        return

    coords = []
    try:
        with open(gro_path, "r", encoding="utf-8") as gro_f:
            gro_f.readline()
            n_atoms = int(gro_f.readline().strip())
            for _ in range(n_atoms):
                line = gro_f.readline()
                if not line:
                    break
                x = float(line[20:28])
                y = float(line[28:36])
                z = float(line[36:44])
                coords.append((x, y, z))
    except (ValueError, OSError) as exc:
        print(f"[경고] GRO 파일을 읽는 도중 오류가 발생했습니다: {exc}")
        return

    _, atoms = _gather_sorted_atoms()
    if len(coords) != len(atoms):
        print(
            f"[경고] GRO 파일의 원자 수({len(coords)})와 World 원자 수({len(atoms)})가 "
            "다릅니다. 가능한 범위까지만 적용합니다."
        )
    for idx, coord in enumerate(coords[: len(atoms)]):
        atoms[idx].position = np.array(coord, dtype=np.float64)


def _reset_world_for_backbone(sim_params):
    """Reset the global world state and initialize box-scale parameters."""
    World.reset()
    Attributes.initialize()
    _seed_random_generators(sim_params.get("random_seed"))
    print("\n--- World 초기화 중... ---")
    initialize_world(
        sim_params.get("segment_length"),
        sim_params.get("mean_sep"),
        _compute_max_linker_span(),
    )


def _load_backbone_context():
    """Load template libraries and sequence strategies for backbone planning."""
    backbone_cfg = Config.get_param("hydrogel_components", "backbone_definitions")
    linker_cfg = Config.get_param("hydrogel_components", "linker_definitions")
    backbone_defs = backbone_cfg["BACKBONES"]
    backbone_strategy = backbone_cfg.get("SEQUENCE_STRATEGY", {"strategy": "random"})
    linker_strategy = linker_cfg.get("SEQUENCE_STRATEGY", {"strategy": "random"})

    monomer_library = Config.get_runtime("monomer_library")
    if monomer_library is None:
        monomer_library = load_monomer_templates(
            Config.get_param("monomer_definitions", "MONOMERS"),
            backbone_defs,
        )
        Config.set_runtime("monomer_library", monomer_library)

    linker_library = Config.get_runtime("linker_library")
    if linker_library is None:
        linker_library = load_linker_templates(linker_cfg["LINKERS"], backbone_defs)
        Config.set_runtime("linker_library", linker_library)

    linker_defs = (
        linker_definitions_from_library(linker_library)
        if linker_library
        else linker_cfg["LINKERS"]
    )

    return {
        "backbone_cfg": backbone_cfg,
        "backbone_defs": backbone_defs,
        "backbone_strategy": backbone_strategy,
        "linker_cfg": linker_cfg,
        "linker_defs": linker_defs,
        "linker_strategy": linker_strategy,
        "linker_library": linker_library,
    }


def _resolve_network_layout(sim_params):
    """Read the optional ``network_layout`` block.

    Returning ``None`` selects the historical diamond layout, so an existing
    configuration is unaffected by the key existing. The block is validated
    here rather than at use, so a typo is reported before any structure is
    built.
    """
    raw = sim_params.get("network_layout")
    if raw in (None, {}, False):
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            f"'network_layout' must be a mapping, got {type(raw).__name__}"
        )

    known = {"net", "repeats", "cell_parameter", "rewiring"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"'network_layout' has unknown key(s) {unknown}; expected {sorted(known)}"
        )

    net = raw.get("net")
    if not net:
        raise ValueError("'network_layout' needs a 'net' (for example 'pcu' or 'dia')")

    repeats = raw.get("repeats")
    if repeats is None:
        raise ValueError("'network_layout' needs 'repeats'")
    if isinstance(repeats, int):
        repeats = (repeats, repeats, repeats)
    else:
        repeats = tuple(int(value) for value in repeats)
        if len(repeats) != 3:
            raise ValueError(f"'network_layout.repeats' needs three values, got {repeats}")

    cell_parameter = raw.get("cell_parameter")
    if cell_parameter is None or float(cell_parameter) <= 0.0:
        raise ValueError(
            "'network_layout' needs a positive 'cell_parameter' (nm between "
            "neighbouring junction sites)"
        )

    rewiring = raw.get("rewiring") or {}
    if not isinstance(rewiring, dict):
        raise ValueError("'network_layout.rewiring' must be a mapping")
    rewire_known = {
        "max_span", "seed", "max_sweeps", "tolerance", "patience",
        "allow_primary_loops", "allow_parallel_strands",
    }
    rewire_unknown = sorted(set(rewiring) - rewire_known)
    if rewire_unknown:
        raise ValueError(
            f"'network_layout.rewiring' has unknown key(s) {rewire_unknown}; "
            f"expected {sorted(rewire_known)}"
        )
    max_span = rewiring.get("max_span")
    rewire_kwargs = {
        key: rewiring[key] for key in rewire_known - {"max_span", "seed"} if key in rewiring
    }
    if max_span is None and rewire_kwargs:
        raise ValueError(
            "'network_layout.rewiring' options were given without 'max_span', "
            "so no rewiring would run. Set 'max_span' or remove the block."
        )

    return {
        "net": str(net),
        "repeats": repeats,
        "cell_parameter": float(cell_parameter),
        "max_span": None if max_span is None else float(max_span),
        "rewire_seed": rewiring.get("seed"),
        "rewire_kwargs": rewire_kwargs,
    }


def _resolve_isotropy_mode(sim_params):
    """Resolve whether the special isotropic builder path should be used."""
    anisotropy = sim_params.get("anisotropy")
    if isinstance(anisotropy, bool):
        return not anisotropy
    if anisotropy is not None:
        anisotropy_str = str(anisotropy).strip().lower()
        if anisotropy_str in ("false", "none", "0", "no"):
            return True
        if anisotropy_str in ("x", "y", "z"):
            return False

    isotropy_cfg = sim_params.get("isotropy")
    if isinstance(isotropy_cfg, dict):
        enabled = isotropy_cfg.get("enabled")
        if enabled is not None:
            return bool(enabled)
        return anisotropy is None
    return bool(isotropy_cfg)


def _build_blueprint_summary(layout_plan, blueprint):
    """Print a compact summary of the generated layout and blueprint."""
    if layout_plan is not None:
        print(f"Layout plan: {len(layout_plan.cells)} cells, {len(layout_plan.links)} linkers")
    if blueprint is not None:
        print(f"Blueprint atoms: {len(blueprint.atoms)}")
        if blueprint.atoms:
            sample = blueprint.atoms[0]
            print(" Sample atom:", sample.chain_type, sample.position)


def _plan_backbone_blueprint(sim_params, output_dir):
    """Build the proto plan and atom blueprint for the hydrogel backbone."""
    context = _load_backbone_context()
    proto_plan = prepare_proto_plan(
        sim_params.get("segment_length"),
        sim_params.get("mean_sep"),
        context["backbone_defs"],
        context["linker_defs"],
        sim_params.get("box_margin", 0.5),
        context["backbone_strategy"],
        context["linker_strategy"],
        bond_rules=context["backbone_cfg"].get("BONDS"),
        linker_library=context["linker_library"],
    )

    bb_len = proto_plan.proto_backbone.length
    bb_raw = getattr(proto_plan.proto_backbone, "raw_length", bb_len)
    print(
        f"Proto backbone length: {bb_len:.3f} nm "
        f"(raw {bb_raw:.3f} nm, {proto_plan.proto_backbone.positions.shape[0]} beads)"
    )
    if proto_plan.proto_linker is not None:
        ln_len = proto_plan.proto_linker.length
        print(
            f"Proto linker length  : {ln_len:.3f} nm "
            f"({proto_plan.proto_linker.positions.shape[0]} beads)"
        )
    else:
        print("Proto linker         : none (backbone-only mode)")
    print(f"Cell vector (nm)     : {proto_plan.cell_vector}")

    net_layout_config = _resolve_network_layout(sim_params)

    num_cells = int(sim_params.get("number_of_cells"))
    if num_cells < 1:
        raise ValueError("number_of_cells must be >= 1")
    if net_layout_config is None and num_cells % 2 != 0 and num_cells != 1:
        # The even-cell requirement is the diamond path's. A net-driven layout
        # validates its own repeat counts per net, since the constraint differs
        # between nets rather than being universal.
        raise AssertionError("Diamond network must have even number of cells or debug value 1")

    repeats = (num_cells, num_cells, num_cells)
    isotropy_mode = _resolve_isotropy_mode(sim_params)

    if net_layout_config is not None:
        if isotropy_mode:
            raise ValueError(
                "network_layout and the isotropy path are alternative layouts; "
                "enable only one."
            )
        print(
            "[INFO] network_layout enabled: net={net} repeats={repeats} "
            "a={cell_parameter} nm".format(**net_layout_config)
        )
        net_result = generate_net_layout_plan(
            proto_plan,
            context["backbone_defs"],
            context["linker_defs"],
            net=net_layout_config["net"],
            repeats=net_layout_config["repeats"],
            cell_parameter=net_layout_config["cell_parameter"],
            linker_library=context["linker_library"],
            max_span=net_layout_config["max_span"],
            rewire_seed=net_layout_config["rewire_seed"],
            rewire_kwargs=net_layout_config["rewire_kwargs"],
        )
        for key, value in net_result.summary().items():
            print(f"  network_layout.{key}: {value}")
        layout_plan = net_result.layout_plan
        blueprint = build_atom_blueprint(layout_plan, context["backbone_defs"])
    elif isotropy_mode:
        print("[INFO] isotropy mode enabled: per-medium-cell EM")
        layout_plan = None
        blueprint = build_isotropic_blueprint(
            proto_plan,
            context["backbone_defs"],
            context["linker_defs"],
            repeats,
            context["backbone_strategy"],
            context["linker_strategy"],
            linker_library=context["linker_library"],
            output_dir=output_dir,
            sim_params=sim_params,
        )
    else:
        layout_plan = generate_layout_plan(
            proto_plan,
            context["backbone_defs"],
            context["linker_defs"],
            repeats,
            context["backbone_strategy"],
            context["linker_strategy"],
            linker_library=context["linker_library"],
        )
        blueprint = build_atom_blueprint(layout_plan, context["backbone_defs"])

    _build_blueprint_summary(layout_plan, blueprint)

    return {
        **context,
        "proto_plan": proto_plan,
        "layout_plan": layout_plan,
        "blueprint": blueprint,
        "num_cells": num_cells,
        "repeats": repeats,
        "isotropy_mode": isotropy_mode,
    }


def _apply_materialization_box_settings(plan_context):
    """Copy proto-plan box data into ``World`` before object creation."""
    proto_plan = plan_context["proto_plan"]
    repeats = plan_context["repeats"]

    if plan_context["isotropy_mode"]:
        medium_size = np.array(proto_plan.small_size, dtype=np.float64) * 2.0
        World.cell_vector = medium_size * 2.0
        World.box_vector = World.cell_vector * np.array(repeats, dtype=np.float64)
        World.box_length = float(np.max(World.box_vector))
        World.ubox_length = float(np.max(World.cell_vector))
    else:
        World.cell_vector = np.array(proto_plan.cell_vector, dtype=np.float64)
        World.box_vector = proto_plan.box_vector(repeats)
        World.box_length = float(np.max(World.box_vector))
        World.ubox_length = float(np.max(proto_plan.cell_vector))
        print(f"Updated World.box_vector: {World.box_vector}")


def build_backbone_only():
    """Construct only the backbone and linker skeleton of the hydrogel."""
    sim_params = Config.get_param("simulation_parameters")
    output_dir = sim_params.get("output_dir")

    _print_build_banner()
    _reset_world_for_backbone(sim_params)

    try:
        plan_context = _plan_backbone_blueprint(sim_params, output_dir)
    except Exception as exc:
        print(f"[경고] 레이아웃 플랜 생성 중 오류: {exc}")
        traceback.print_exc()
        raise RuntimeError("proto builder blueprint 생성에 실패했습니다.") from exc

    _print_build_banner()
    _reset_world_for_backbone(sim_params)
    _apply_materialization_box_settings(plan_context)

    world = World()
    world.make_hydrogel(
        False,
        nx=plan_context["num_cells"],
        ny=plan_context["num_cells"],
        nz=plan_context["num_cells"],
    )
    hd = world.hydrogels[0]

    print("[INFO] proto blueprint를 적용합니다.")
    _debug_stage("[stage] populate_hydrogel_from_blueprint start")
    populate_hydrogel_from_blueprint(hd, plan_context["blueprint"])
    _debug_stage("[stage] populate_hydrogel_from_blueprint done")

    _debug_stage("[stage] construct_bonds start")
    hd.construct_bonds(
        sim_params.get("pbc_true_or_false"),
        plan_context["num_cells"],
        output_dir,
    )
    _debug_stage("[stage] construct_bonds done")

    _debug_stage("[stage] update_hydrogel_attributes start")
    world.update_hydrogel_attributes(hd)
    _debug_stage("[stage] update_hydrogel_attributes done")

    _debug_stage("[stage] _log_min_distance_report backbone start")
    _log_min_distance_report("backbone")
    _debug_stage("[stage] _log_min_distance_report backbone done")

    print("Backbone/링커 단계 완료.")
    return world, hd


def finalize_hydrogel(world, hd):
    """Expand the backbone-only graph into a chemically detailed hydrogel."""
    print("화학적 상세 구조를 구성합니다...")
    _debug_stage("[stage] construct_chemical_detail start")
    hd.construct_chemical_detail()
    _debug_stage("[stage] construct_chemical_detail done")

    print("각도(angle)를 구성합니다...")
    _debug_stage("[stage] construct_angles start")
    hd.construct_angles()
    _debug_stage("[stage] construct_angles done")

    _debug_stage("[stage] construct_dihedrals start")
    hd.construct_dihedrals()
    _debug_stage("[stage] construct_dihedrals done")

    _debug_stage("[stage] construct_impropers start")
    hd.construct_impropers()
    _debug_stage("[stage] construct_impropers done")

    world.update_hydrogel_attributes(hd)
    _log_min_distance_report("final")
    print("하이드로젤 구성 완료.")
    return world


def main():
    """Run the standalone hydrogel builder entry point."""
    world, hd = build_backbone_only()
    finalize_hydrogel(world, hd)
    return world


if __name__ == "__main__":
    Attributes.initialize()
    main()
