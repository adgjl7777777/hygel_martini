"""Isotropic medium-cell builder with per-cell EM."""

from dataclasses import replace
from typing import Dict, List, Tuple, Optional, Sequence
import os

import json
import random
import numpy as np

from hydrogel_builder.core_utils.io.gro_parser import read_gro_atoms
from hydrogel_builder.core_utils.io.writer import write_to_gro, write_combined_itp
from hydrogel_builder.core_utils.layout.layout_executor import LayoutBlueprint, build_atom_blueprint
from hydrogel_builder.core_utils.layout.proto_layout import LayoutPlan, LayoutCell, LinkPlacement
from hydrogel_builder.core_utils.layout.proto_layout import MEDIUM_ACTIVE_INDICES, POLYMER_POSITIONS, ORIENTATION_MAP
from hydrogel_builder.core_utils.layout.proto_populator import populate_hydrogel_from_blueprint
from hydrogel_builder.core_utils.runtime.geo_opt import run_geo_opt
from hydrogel_builder.main_components.Universe import World
from hydrogel_builder.main_components import Attributes
from hydrogel_builder.config_params.config import Config


def _linear_index(ix: int, iy: int, iz: int, repeats: Tuple[int, int, int]) -> int:
    nx, ny, nz = repeats
    return ix * ny * nz + iy * nz + iz


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return vec
    return vec / norm


def _axis_rotation_matrix(axis: str) -> np.ndarray:
    if axis == "y":
        return np.array([[0.0, -1.0, 0.0],
                         [1.0,  0.0, 0.0],
                         [0.0,  0.0, 1.0]], dtype=float)
    if axis == "z":
        return np.array([[0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0],
                         [-1.0, 0.0, 0.0]], dtype=float)
    return np.eye(3, dtype=float)


def _apply_rotation(vec: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return vec @ rot.T


def _get_anisotropy_axis() -> str:
    axis = "x"
    try:
        axis = str(Config.get_param('simulation_parameters').get('anisotropy', 'x')).lower()
    except Exception:
        axis = "x"
    if axis not in ("x", "y", "z"):
        axis = "x"
    return axis


def _normalize_linker_axes(linker_axes: Optional[Sequence[str]]) -> List[str]:
    if linker_axes is None:
        axes = ["x", "x"]
    elif isinstance(linker_axes, str):
        axes = [linker_axes]
    else:
        axes = list(linker_axes)
    cleaned = []
    for axis in axes:
        axis_str = str(axis).lower()
        if axis_str in ("x", "y", "z"):
            cleaned.append(axis_str)
    if not cleaned:
        cleaned = ["x", "x"]
    if len(cleaned) == 1:
        cleaned = [cleaned[0], cleaned[0]]
    return cleaned[:2]


def _squeeze_positions_to_cube(positions: np.ndarray, cube_side: float) -> np.ndarray:
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-9)
    scale = cube_side / spans
    centered = positions - 0.5 * (mins + maxs)
    return centered * scale


def _linker_total_length(entry: Dict, fallback: float, override: float | None = None) -> float:
    if override is not None and override > 0:
        return float(override)
    definition = entry.get('definition', entry)
    total = 0.0
    for bond in definition.get('bonds', []):
        total += bond.get('length', 0.0)
    for ext in definition.get('external_bonds', []):
        total += ext.get('length', 0.0)
    if total <= 0:
        total = fallback
    return float(total)


def _write_linker_debug(path: str,
                        blueprint: LayoutBlueprint,
                        positions_pre: np.ndarray,
                        positions_post: np.ndarray,
                        axes: List[str],
                        medium_origin: np.ndarray,
                        box_vector: np.ndarray,
                        cube_side: float,
                        small_edge: float,
                        linker_len: float,
                        base_size: np.ndarray,
                        cell_vector: np.ndarray,
                        scale: np.ndarray,
                        mins: np.ndarray,
                        maxs: np.ndarray) -> None:
    chain_atoms: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}
    for idx, atom in enumerate(blueprint.atoms):
        chain_atoms.setdefault((atom.chain_type, atom.chain_index), []).append((atom.bead_index, idx))

    payload = {
        "axes": axes,
        "medium_origin": medium_origin.tolist(),
        "small_edge": float(small_edge),
        "linker_len": float(linker_len),
        "base_size": base_size.tolist(),
        "cell_vector": cell_vector.tolist(),
        "box_vector": box_vector.tolist(),
        "cube_side": float(cube_side),
        "center_pre_compress": positions_pre.mean(axis=0).tolist() if len(positions_pre) else [0.0, 0.0, 0.0],
        "bounds_pre_compress": {"min": mins.tolist(), "max": maxs.tolist()},
        "scale": scale.tolist(),
        "backbones": [],
        "linkers": []
    }

    for chain in blueprint.chains:
        chain_key = (chain.chain_type, chain.chain_index)
        bead_entries = chain_atoms.get(chain_key, [])
        bead_entries = sorted(bead_entries, key=lambda item: item[0])
        if chain.chain_type == "backbone":
            bead_indices = [bead_idx for bead_idx, _ in bead_entries if bead_idx >= 0]
            if not bead_indices:
                continue
            head_idx = min(bead_indices)
            tail_idx = max(bead_indices)
            head_atom = next(idx for bead_idx, idx in bead_entries if bead_idx == head_idx)
            tail_atom = next(idx for bead_idx, idx in bead_entries if bead_idx == tail_idx)
            payload["backbones"].append({
                "chain_index": chain.chain_index,
                "local_indices": (chain.metadata or {}).get("local_indices"),
                "center": (chain.metadata or {}).get("origin"),
                "direction": (chain.metadata or {}).get("direction"),
                "head_bead_index": head_idx,
                "tail_bead_index": tail_idx,
                "head_pre": positions_pre[head_atom].tolist(),
                "tail_pre": positions_pre[tail_atom].tolist(),
                "head_post": positions_post[head_atom].tolist(),
                "tail_post": positions_post[tail_atom].tolist()
            })

    for chain in blueprint.chains:
        if chain.chain_type != "linker":
            continue
        chain_key = (chain.chain_type, chain.chain_index)
        bead_entries = chain_atoms.get(chain_key, [])
        pos_by_bead_pre = {bead_idx: positions_pre[idx] for bead_idx, idx in bead_entries if bead_idx >= 0}
        pos_by_bead_post = {bead_idx: positions_post[idx] for bead_idx, idx in bead_entries if bead_idx >= 0}
        definition = chain.definition or {}
        ext1 = definition.get("external_bonds_1", []) or []
        ext2 = definition.get("external_bonds_2", []) or []
        stub_indices = [ext.get("from_bead") for ext in (ext1 + ext2) if ext.get("from_bead") is not None]
        stub_positions_pre = [pos_by_bead_pre.get(idx) for idx in stub_indices if idx in pos_by_bead_pre]
        stub_positions_post = [pos_by_bead_post.get(idx) for idx in stub_indices if idx in pos_by_bead_post]
        bck_vector = None
        bck_unit = None
        if len(stub_positions_pre) >= 2:
            vec = np.array(stub_positions_pre[1], dtype=float) - np.array(stub_positions_pre[0], dtype=float)
            bck_vector = vec.tolist()
            norm = float(np.linalg.norm(vec))
            if norm > 1e-9:
                bck_unit = (vec / norm).tolist()
        axis_dir = chain.metadata.get("axis_direction")
        axis_unit = None
        if axis_dir is not None:
            axis_vec = np.array(axis_dir, dtype=float)
            norm = float(np.linalg.norm(axis_vec))
            if norm > 1e-9:
                axis_unit = (axis_vec / norm).tolist()

        dot = None
        if bck_unit is not None and axis_unit is not None:
            dot = float(np.dot(np.array(bck_unit), np.array(axis_unit)))

        payload["linkers"].append({
            "chain_index": chain.chain_index,
            "local_linker_index": chain.metadata.get("local_linker_index"),
            "linker_axis": chain.metadata.get("linker_axis"),
            "axis_direction": axis_dir.tolist() if isinstance(axis_dir, np.ndarray) else axis_dir,
            "axis_unit": axis_unit,
            "anchor_position": (
                chain.metadata.get("anchor_position").tolist()
                if isinstance(chain.metadata.get("anchor_position"), np.ndarray)
                else chain.metadata.get("anchor_position")
            ),
            "stub_indices": stub_indices,
            "stub_positions_pre": [pos.tolist() for pos in stub_positions_pre],
            "stub_positions_post": [pos.tolist() for pos in stub_positions_post],
            "bck_vector": bck_vector,
            "bck_unit": bck_unit,
            "axis_dot_bck": dot
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _pick_boundary_atoms(positions: np.ndarray) -> List[int]:
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-9)
    normalized = (positions - mins) / spans
    corners = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    picked = []
    for corner in corners:
        dists = np.linalg.norm(normalized - corner, axis=1)
        idx = int(np.argmin(dists))
        if idx not in picked:
            picked.append(idx)
    return picked


def _write_posre_itp(path: str, atom_indices: List[int], fc: float) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("[ position_restraints ]\n")
        f.write("; atom  type  fx  fy  fz\n")
        for idx in atom_indices:
            f.write(f"{idx:6d}  1  {fc:.1f}  {fc:.1f}  {fc:.1f}\n")


def _write_system_top(path: str, itp_path: str, posre_path: str, base_itp: str | None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if base_itp:
            f.write(f'#include "{os.path.abspath(base_itp)}"\n')
        f.write(f'#include "{os.path.abspath(itp_path)}"\n')
        f.write("#ifdef POSRES\n")
        f.write(f'#include "{os.path.abspath(posre_path)}"\n')
        f.write("#endif\n\n")
        f.write("[ system ]\nMediumCell\n\n")
        f.write("[ molecules ]\n")
        f.write("; Compound        #mols\n")
        f.write("HYDROGEL        1\n")


def _build_world_from_blueprint(blueprint: LayoutBlueprint,
                                box_vector: np.ndarray,
                                output_dir: str,
                                mean_sep: float,
                                construct_proto_bonds: bool = True):
    World.reset()
    Attributes.initialize()
    World.mean_sep = float(mean_sep)
    world = World()
    world.make_hydrogel(False, nx=1, ny=1, nz=1)
    hd = world.hydrogels[0]
    populate_hydrogel_from_blueprint(hd, blueprint)
    if construct_proto_bonds:
        hd.construct_bonds(False, 1, output_dir)
    World.box_vector = np.array(box_vector, dtype=float)
    World.box_length = float(np.max(World.box_vector))
    world.box_vector = World.box_vector
    return world, hd


def _run_medium_cell_em(blueprint: LayoutBlueprint,
                        box_vector: np.ndarray,
                        fixed_atom_indices: List[int],
                        out_dir: str,
                        sim_params: Dict,
                        temp_bonds: List[Tuple[int, int, Dict]] | None = None) -> List[np.ndarray]:
    os.makedirs(out_dir, exist_ok=True)
    mean_sep = float(sim_params.get("mean_sep", 0.24))
    world, _ = _build_world_from_blueprint(
        blueprint,
        box_vector,
        out_dir,
        mean_sep,
        construct_proto_bonds=False
    )
    backbone_atom_ids = {i for i, atom in enumerate(blueprint.atoms) if atom.chain_type == "backbone"}
    backbone_bond_snapshot = {}
    if backbone_atom_ids:
        for (i, j), bond_list in World.Bonds.items():
            if i in backbone_atom_ids and j in backbone_atom_ids:
                for bond in bond_list:
                    backbone_bond_snapshot[id(bond)] = bond.bond_c1
                    bond.bond_c1 = float(bond.bond_c1) * 0.1
    temp_keys = []
    if temp_bonds:
        for idx_a, idx_b, params in temp_bonds:
            i = int(idx_a)
            j = int(idx_b)
            if i > j:
                i, j = j, i
            Attributes.Bond(i, j,
                            funct=params.get("bond_funct", params.get("funct", 1)),
                            c0=params.get("bond_c0", params.get("length", mean_sep)),
                            c1=params.get("bond_c1", params.get("fc", 15000)))
            temp_keys.append((i, j))

    gro_path = os.path.join(out_dir, "medium_cell.gro")
    itp_path = os.path.join(out_dir, "system.itp")
    top_path = os.path.join(out_dir, "system.top")
    posre_path = os.path.join(out_dir, "posre.itp")

    write_to_gro(world, filename=gro_path)
    write_combined_itp(world, filename=itp_path, moleculetype_name="HYDROGEL")

    fc = float(sim_params.get("isotropy", {}).get("boundary_fix_fc", 100000.0))
    _write_posre_itp(posre_path, fixed_atom_indices, fc)
    base_itp = sim_params.get("base_itp_file")
    _write_system_top(top_path, itp_path, posre_path, base_itp)

    mdp_overrides = {"define": "-DPOSRES"}
    mdrun_extra_args = ["-ntomp", "1", "-pin", "off", "-dd", "1", "1", "1", "-rdd", "0.5", "-dds", "0.5"]
    run_geo_opt(
        structure_file=gro_path,
        topology_file=top_path,
        output_dir=out_dir,
        gmx_executable=sim_params.get("gromacs_executable_path", "gmx"),
        maxwarn=int(sim_params.get("grompp_maxwarn", 1)),
        mdp_overrides=mdp_overrides,
        mdrun_extra_args=mdrun_extra_args,
    )

    em_path = os.path.join(out_dir, "em.gro")
    if not os.path.exists(em_path):
        raise FileNotFoundError(f"medium-cell EM output missing: {em_path}")
    for key in temp_keys:
        if key in World.Bonds:
            del World.Bonds[key]
    if backbone_bond_snapshot:
        for bond_list in World.Bonds.values():
            for bond in bond_list:
                saved = backbone_bond_snapshot.get(id(bond))
                if saved is not None:
                    bond.bond_c1 = saved
    atoms = read_gro_atoms(em_path)
    return [atom.position for atom in atoms]


def _offset_blueprint(blueprint: LayoutBlueprint, atom_offset: int, chain_offset: int) -> LayoutBlueprint:
    atoms = []
    for atom in blueprint.atoms:
        atoms.append(replace(atom, chain_index=atom.chain_index + chain_offset))
    chains = []
    for chain in blueprint.chains:
        chains.append(replace(
            chain,
            chain_index=chain.chain_index + chain_offset,
            atom_indices=[idx + atom_offset for idx in chain.atom_indices]
        ))
    return LayoutBlueprint(atoms=atoms, chains=chains)


def build_isotropic_blueprint(proto_plan,
                              backbone_defs: List[Dict],
                              linker_defs: List[Dict],
                              repeats: Tuple[int, int, int],
                              backbone_strategy: Dict,
                              linker_strategy: Dict,
                              linker_library,
                              output_dir: str,
                              sim_params: Dict) -> LayoutBlueprint:
    nx, ny, nz = repeats
    small_edge = float(proto_plan.small_size[0])
    linker_len = float(proto_plan.proto_linker.length)
    base_size = np.array([2.0 * small_edge, 2.0 * small_edge, 2.0 * small_edge], dtype=float)
    cell_vector = base_size * 2.0
    cube_side = float(base_size[0])

    backbone_weights = [entry.get('ratio', 1) for entry in backbone_defs] if backbone_defs else [1]
    linker_records = list(linker_library.records) if linker_library and linker_library.records else []
    if linker_records:
        linker_weights = [max(float(record.ratio), 0.0) for record in linker_records]
    else:
        linker_weights = [entry.get('ratio', 1) for entry in linker_defs] if linker_defs else [1]
    
    linker_def_lookup = {entry.get('id'): entry for entry in linker_defs}
    sequence_factory = getattr(proto_plan, 'sequence_factory', None)
    bond_lookup = getattr(proto_plan, 'bond_lookup', {})
    mean_sep = getattr(proto_plan, 'mean_sep', 0.24)
    proto_linker_length = max(proto_plan.proto_linker.length, 1e-9)

    all_atoms = []
    all_chains = []
    atom_offset = 0
    chain_offset = 0
    per_cell_records = []
    global_linkers = []

    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                big_origin = np.array([ix, iy, iz], dtype=float) * cell_vector
                for medium_offset in MEDIUM_ACTIVE_INDICES:
                    medium_idx = tuple(int(v) for v in medium_offset)
                    medium_origin = big_origin + np.array(medium_offset, dtype=float) * base_size
                    axes = random.choices(["x", "y", "z"], k=2)
                    axes = _normalize_linker_axes(axes)
                    primary_axis = axes[0]
                    secondary_axis = axes[1]
                    axis_index = {"x": 0, "y": 1, "z": 2}
                    first_vec = np.zeros(3, dtype=float)
                    first_vec[axis_index[primary_axis]] = linker_len * 0.5
                    second_vec = np.array([small_edge, small_edge, small_edge], dtype=float)
                    second_vec[axis_index[primary_axis]] += linker_len
                    second_vec[axis_index[secondary_axis]] += linker_len * 0.5
                    anchor_secondary = medium_origin + second_vec
                    Config.debug_log(f"[isotropy] medium_cell {idx} axes={axes} origin={medium_origin.tolist()}")

                    def _axis_center(idx_val: int) -> float:
                        return small_edge * (0.5 + idx_val)

                    used_sequences: set[Tuple[str, ...]] = set()
                    enforce_unique_sequences = (
                        sequence_factory is not None and getattr(sequence_factory, 'definition_count', 0) > 1
                    )

                    cells: List[LayoutCell] = []
                    links: List[LinkPlacement] = []

                    for local_coords in POLYMER_POSITIONS:
                        x_idx, y_idx, z_idx = local_coords
                        center_local = np.array([
                            _axis_center(x_idx),
                            _axis_center(y_idx),
                            _axis_center(z_idx)
                        ], dtype=float)
                        center_local[axis_index[primary_axis]] += linker_len
                        if secondary_axis == "x" and x_idx == 1:
                            center_local[0] += linker_len
                        elif secondary_axis == "y" and y_idx == 1:
                            center_local[1] += linker_len
                        elif secondary_axis == "z" and z_idx == 1:
                            center_local[2] += linker_len
                        center = medium_origin + center_local
                        orient = ORIENTATION_MAP[local_coords]
                        direction = _normalize(orient)
                        length_scale = 1.0
                        sequence = []
                        proto_positions = None
                        chain_length = proto_plan.proto_backbone.length
                        raw_chain_length = proto_plan.proto_backbone.raw_length
                        if sequence_factory is not None:
                            sequence, proto_positions, raw_chain_length, chain_length = sequence_factory.instantiate(
                                enforce_unique=enforce_unique_sequences,
                                used_signatures=used_sequences
                            )
                        definition = sequence[0] if sequence else (
                            random.choices(backbone_defs, weights=backbone_weights, k=1)[0] if backbone_defs else {}
                        )
                        metadata = {
                            'medium_index': medium_idx,
                            'length_scale': length_scale,
                            'sequence': sequence,
                            'bond_lookup': bond_lookup,
                            'mean_sep': mean_sep,
                            'chain_length': chain_length,
                            'chain_length_raw': raw_chain_length,
                            'local_indices': local_coords
                        }
                        if proto_positions is not None:
                            metadata['proto_positions'] = np.array(proto_positions, dtype=np.float64)
                        cells.append(LayoutCell(
                            origin=center,
                            direction=direction,
                            backbone_definition=definition,
                            cell_index=(ix, iy, iz),
                            metadata=metadata
                        ))

                    linker_local_specs = [first_vec, second_vec]

                    for spec_idx, local_anchor in enumerate(linker_local_specs):
                        axis_choice = primary_axis if spec_idx == 0 else secondary_axis
                        anchor = medium_origin + local_anchor
                        axis_dir = np.zeros(3, dtype=float)
                        axis_dir[axis_index[axis_choice]] = random.choice([-1.0, 1.0])
                        template_record = None
                        if linker_records:
                            template_record = random.choices(linker_records, weights=linker_weights, k=1)[0]
                            link_template = template_record.template
                            link_def = linker_def_lookup.get(link_template.id, {'id': link_template.id})
                            span_override = float(link_template.span_length)
                            span_vector = np.array(link_template.span_vector, dtype=np.float64)
                            backbone_ids = link_template.backbone_ids
                        else:
                            link_def = random.choices(linker_defs, weights=linker_weights, k=1)[0] if linker_defs else {}
                            span_override = None
                            span_vector = None
                            backbone_ids = None
                        total_length = _linker_total_length(link_def, proto_linker_length, span_override)
                        scale = total_length / proto_linker_length if proto_linker_length > 1e-9 else 1.0
                        backbone_names = link_def.get('backbone_residue_name')
                        if not backbone_names:
                            definition = link_def.get('definition', {}) if isinstance(link_def, dict) else {}
                            backbone_names = definition.get('backbone_name')
                        metadata = {
                            'medium_index': medium_idx,
                            'length_scale': scale,
                            'local_linker_index': spec_idx,
                            'linker_axis': axis_choice,
                            'backbone_residue_names': backbone_names or []
                        }
                        if template_record:
                            metadata.update({
                                'linker_template_id': template_record.template.id,
                                'span_length': span_override,
                                'span_vector': span_vector,
                                'backbone_ids': backbone_ids
                            })
                        links.append(LinkPlacement(
                            anchor_position=anchor,
                            axis_direction=axis_dir,
                            linker_definition=link_def,
                            connected_cells=(_linear_index(ix, iy, iz, repeats),) * 2,
                            metadata=metadata
                        ))

                    sub_plan = LayoutPlan(proto_plan=proto_plan, cells=cells, links=links)
                    sub_blueprint = build_atom_blueprint(sub_plan, backbone_defs)

                    positions = np.array([a.position for a in sub_blueprint.atoms], dtype=np.float64)
                    mins = positions.min(axis=0) if len(positions) else np.zeros(3)
                    maxs = positions.max(axis=0) if len(positions) else np.zeros(3)
                    spans = np.maximum(maxs - mins, 1e-9)
                    scale = np.array([cube_side, cube_side, cube_side], dtype=float) / spans
                    for i, atom in enumerate(sub_blueprint.atoms):
                        atom.extra = dict(atom.extra or {})
                        atom.extra["pre_compress_position"] = positions[i].copy()
                    positions -= medium_origin
                    positions_post = _squeeze_positions_to_cube(positions, cube_side)
                    positions_post += np.array([cube_side / 2.0, cube_side / 2.0, cube_side / 2.0])

                    secondary_linker_chains = {
                        c.chain_index for c in sub_blueprint.chains
                        if c.chain_type == "linker" and (c.metadata or {}).get("local_linker_index") == 1
                    }
                    anchor_post = None
                    if secondary_linker_chains:
                        linker_indices = [
                            i for i, atom in enumerate(sub_blueprint.atoms)
                            if atom.chain_type == "linker" and atom.chain_index in secondary_linker_chains
                        ]
                        if linker_indices:
                            anchor_post = np.mean(positions_post[linker_indices], axis=0)
                    target_map = {"x": ((0, 1, 1), 0), "y": ((1, 0, 1), 1), "z": ((1, 1, 0), 2)}
                    target_coords, axis_idx = target_map.get(primary_axis, ((0, 1, 1), 0))
                    for chain in sub_blueprint.chains:
                        if chain.chain_type != "linker":
                            continue
                        meta = chain.metadata or {}
                        axis = meta.get("linker_axis")
                        if axis is None or not chain.atom_indices:
                            continue
                        center = positions_post[chain.atom_indices].mean(axis=0) + medium_origin
                        global_linkers.append((center, axis))
                    per_cell_records.append({
                        "idx": idx,
                        "medium_origin": medium_origin,
                        "primary_axis": primary_axis,
                        "target_coords": target_coords,
                        "axis_idx": axis_idx,
                        "anchor_post": anchor_post,
                        "positions_post": positions_post,
                        "sub_blueprint": sub_blueprint,
                        "base_size": base_size
                    })
                    idx += 1

    total_box = cell_vector * np.array([nx, ny, nz], dtype=float)
    mean_sep = float(sim_params.get("mean_sep", 0.24))

    # Pass 1: apply primary-axis shift based on global linker directions.
    for record in per_cell_records:
        medium_origin = record["medium_origin"]
        primary_axis = record["primary_axis"]
        target_coords = record["target_coords"]
        axis_idx = record["axis_idx"]
        anchor_post = record["anchor_post"]
        positions_post = record["positions_post"]
        sub_blueprint = record["sub_blueprint"]

        if anchor_post is not None:
            for chain in sub_blueprint.chains:
                if chain.chain_type != "backbone":
                    continue
                local_coords = (chain.metadata or {}).get("local_indices")
                if local_coords != target_coords:
                    continue
                atom_indices = chain.atom_indices
                if not atom_indices or len(atom_indices) < 3:
                    continue
                end0 = positions_post[atom_indices[0]]
                end1 = positions_post[atom_indices[-1]]
                if np.linalg.norm(end0 - anchor_post) <= np.linalg.norm(end1 - anchor_post):
                    ordered = list(atom_indices)
                    far_end = end1
                else:
                    ordered = list(reversed(atom_indices))
                    far_end = end0
                far_end_global = far_end + medium_origin
                apply_shift = True
                if global_linkers:
                    def _pbc_distance_sq(center):
                        delta = far_end_global - center
                        delta -= total_box * np.round(delta / total_box)
                        return float(np.dot(delta, delta))
                    nearest_axis = min(
                        global_linkers,
                        key=lambda item: _pbc_distance_sq(item[0])
                    )[1]
                    if nearest_axis == primary_axis:
                        apply_shift = False
                if apply_shift:
                    for i, atom_idx in enumerate(ordered, start=1):
                        offset = (i / (len(ordered) + 2)) * linker_len
                        positions_post[atom_idx][axis_idx] -= offset

    # Pass 2: additional adjustment based on linker backbone beads (global, PBC-aware).
    axis_index = {"x": 0, "y": 1, "z": 2}
    global_backbone = []
    for record in per_cell_records:
        medium_origin = record["medium_origin"]
        positions_post = record["positions_post"]
        sub_blueprint = record["sub_blueprint"]
        for chain in sub_blueprint.chains:
            if chain.chain_type != "backbone":
                continue
            for atom_idx in chain.atom_indices:
                global_backbone.append({
                    "pos": positions_post[atom_idx] + medium_origin,
                    "record": record,
                    "chain": chain
                })

    def _pbc_delta(vec):
        delta = vec.copy()
        delta -= total_box * np.round(delta / total_box)
        return delta

    for record in per_cell_records:
        medium_origin = record["medium_origin"]
        positions_post = record["positions_post"]
        sub_blueprint = record["sub_blueprint"]
        for chain in sub_blueprint.chains:
            if chain.chain_type != "linker":
                continue
            meta = chain.metadata or {}
            axis = meta.get("linker_axis")
            if axis is None:
                continue
            names = meta.get("backbone_residue_names") or []
            if isinstance(names, str):
                names = [names]
            if not names:
                Config.debug_log(
                    f"[isotropy][lnk+offset] linker={chain.chain_index} axis={axis} "
                    f"backbone_residue_names=EMPTY"
                )
                continue
            axis_idx = axis_index[axis]
            
            def _matches_any(res_name, target_names):
                if isinstance(res_name, list):
                    return any(name in target_names for name in res_name)
                return res_name in target_names

            linker_atom_indices = [
                atom_idx for atom_idx in chain.atom_indices
                if _matches_any(sub_blueprint.atoms[atom_idx].residue_name, names)
            ]
            if not linker_atom_indices:
                # Use str() to handle potentially unhashable lists in the set
                chain_res = sorted({str(sub_blueprint.atoms[i].residue_name) for i in chain.atom_indices})
                Config.debug_log(
                    f"[isotropy][lnk+offset] linker={chain.chain_index} axis={axis} "
                    f"names={names} matched=0 chain_res={chain_res}"
                )
            
            # Use only the two ends (first and last found backbone bead)
            if len(linker_atom_indices) > 2:
                linker_atom_indices = [linker_atom_indices[0], linker_atom_indices[-1]]

            for atom_idx in linker_atom_indices:
                anchor_pos = positions_post[atom_idx] + medium_origin
                near_backbones = []
                for entry in global_backbone:
                    delta = _pbc_delta(entry["pos"] - anchor_pos)
                    if np.linalg.norm(delta) <= (linker_len * 0.5):
                        near_backbones.append(entry)
                if not near_backbones:
                    Config.debug_log(
                        f"[isotropy][lnk+offset] linker={chain.chain_index} anchor={atom_idx} "
                        f"axis={axis} near_chains=0"
                    )
                    continue
                unique_chains = {}
                for entry in near_backbones:
                    key = (id(entry["record"]), entry["chain"].chain_index)
                    if key not in unique_chains:
                        unique_chains[key] = entry
                if len(unique_chains) != 3:
                    Config.debug_log(
                        f"[isotropy][lnk+offset] linker={chain.chain_index} anchor={atom_idx} "
                        f"axis={axis} near_chains={len(unique_chains)}"
                    )
                    continue
                selected = random.choice(list(unique_chains.values()))
                sel_record = selected["record"]
                sel_chain = selected["chain"]
                sel_positions = sel_record["positions_post"]
                sel_origin = sel_record["medium_origin"]
                atom_indices = sel_chain.atom_indices
                if len(atom_indices) < 3:
                    continue
                end0 = sel_positions[atom_indices[0]] + sel_origin
                end1 = sel_positions[atom_indices[-1]] + sel_origin
                if np.linalg.norm(_pbc_delta(end0 - anchor_pos)) <= np.linalg.norm(_pbc_delta(end1 - anchor_pos)):
                    ordered = list(atom_indices)
                else:
                    ordered = list(reversed(atom_indices))
                Config.debug_log(
                    f"[isotropy][lnk+offset] APPLY linker={chain.chain_index} anchor={atom_idx} "
                    f"axis={axis} chain={sel_chain.chain_index} beads={len(ordered)}"
                )
                # ordered[0] is the end closest to the linker anchor.
                # ordered[-1] is the opposite end (farthest).
                # We want to shift the chain such that the offset is max at ordered[0] and decays to 0 at ordered[-1].
                for i, sel_idx in enumerate(ordered[:-1]):
                    # i goes from 0 to len(ordered)-2.
                    # factor should decrease as i increases.
                    factor = (len(ordered) - i) / (len(ordered) + 2)
                    offset = factor * linker_len
                    sel_positions[sel_idx][axis_idx] += offset

    # Pass 3: write per-cell and assemble large cell.
    for record in per_cell_records:
        idx = record["idx"]
        medium_origin = record["medium_origin"]
        positions_post = record["positions_post"]
        sub_blueprint = record["sub_blueprint"]
        base_size = record["base_size"]

        for i, atom in enumerate(sub_blueprint.atoms):
            atom.position = positions_post[i]

        medium_out = os.path.join(output_dir, "medium_cell", f"{idx:04d}")
        os.makedirs(medium_out, exist_ok=True)
        world, _ = _build_world_from_blueprint(
            sub_blueprint,
            base_size,
            medium_out,
            mean_sep,
            construct_proto_bonds=False
        )
        write_to_gro(world, filename=os.path.join(medium_out, "medium_cell.gro"))

        for i, atom in enumerate(sub_blueprint.atoms):
            atom.position = positions_post[i] + medium_origin

        offset_blueprint = _offset_blueprint(sub_blueprint, atom_offset, chain_offset)
        all_atoms.extend(offset_blueprint.atoms)
        all_chains.extend(offset_blueprint.chains)
        atom_offset += len(offset_blueprint.atoms)
        if offset_blueprint.chains:
            chain_offset = max(c.chain_index for c in offset_blueprint.chains) + 1

        Config.debug_log(f"[isotropy] medium_cell {idx} done atoms={len(sub_blueprint.atoms)}")

    return LayoutBlueprint(atoms=all_atoms, chains=all_chains)
