"""Isotropic medium-cell builder with per-cell EM."""

from dataclasses import replace
from typing import Dict, List, Tuple, Optional, Sequence
import os

import json
import random
import numpy as np

from hygel_martini.core.pbc import minimum_image
from hygel_martini.hydrogel_builder.core_utils.common.collisions import require_unique
from hygel_martini.hydrogel_builder.core_utils.io.gro_parser import read_gro_atoms
from hygel_martini.hydrogel_builder.core_utils.io.writer import write_to_gro, write_combined_itp
from hygel_martini.hydrogel_builder.core_utils.layout.layout_executor import LayoutBlueprint, build_atom_blueprint
from hygel_martini.hydrogel_builder.core_utils.layout.proto_layout import LayoutPlan, LayoutCell, LinkPlacement
from hygel_martini.hydrogel_builder.core_utils.layout.proto_layout import MEDIUM_ACTIVE_INDICES, POLYMER_POSITIONS, ORIENTATION_MAP
from hygel_martini.hydrogel_builder.core_utils.layout.local_matching import (
    LocalVertex,
    plan_balanced_cycle_matchings,
)
from hygel_martini.hydrogel_builder.core_utils.layout.proto_populator import populate_hydrogel_from_blueprint
from hygel_martini.hydrogel_builder.core_utils.runtime.geo_opt import run_geo_opt
from hygel_martini.hydrogel_builder.main_components.Universe import World
from hygel_martini.hydrogel_builder.main_components import Attributes
from hygel_martini.hydrogel_builder.config_params.config import Config


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


def _squeeze_positions_to_cube(
    positions: np.ndarray,
    cube_side: float,
    ghost_step: float = 0.0,
) -> np.ndarray:
    """Rescale all positions to fit inside a cube of side cube_side.

    ghost_step: extend the bounding box by this amount on each side before
    scaling.  Set to backbone_bond_c0/sqrt(3) to restore the inter-cell gap
    that the bare squeeze would cancel: the boundary atoms then sit one bond
    projection away from the cell face, matching the gap that existed in raw
    (pre-squeeze) space.  Pass 0 to reproduce the original behaviour.
    """
    mins = positions.min(axis=0) - ghost_step
    maxs = positions.max(axis=0) + ghost_step
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


def _resolve_close_contacts(
    atoms: List,
    threshold: float = 0.001,
    jitter: float = 0.05,
    seed: int = 0,
) -> int:
    """Detect atoms within *threshold* nm of each other and jitter one.

    Diamond-lattice backbone placement causes the last atom of one chain and
    the first atom of an adjacent chain to share the same position exactly.
    GROMACS LJ goes to infinity (Epot ~ 1e37) when two atoms overlap,
    crashing EM at Step 0.  This function resolves exact/near-exact
    coincidences by displacing one atom by *jitter* nm before any EM run.

    Uses spatial hashing (O(n)) so it's safe on large backbones.
    """
    if not atoms:
        return 0
    positions = np.array([a.position for a in atoms], dtype=np.float64)
    # NOTE: this used to test only for atoms quantising to the SAME hash key,
    # which detects exact duplicates but misses a pair 0.8*threshold apart in
    # adjacent cells -- observed directly as Fmax = inf surviving the resolve
    # pass. The 27 neighbouring cells are now searched, and the displaced atom
    # is pushed apart along the actual separation direction so the resulting
    # distance is at least *jitter*, rather than in a random direction that
    # can leave the pair nearly as close as before.
    grid: dict = {}
    jittered = 0
    rng = np.random.default_rng(seed)
    for i in range(len(positions)):
        key = tuple(np.floor(positions[i] / threshold).astype(np.int64).tolist())
        partner = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        if float(np.linalg.norm(positions[i] - positions[j])) < threshold:
                            partner = j
                            break
                    if partner is not None:
                        break
                if partner is not None:
                    break
            if partner is not None:
                break
        if partner is not None:
            separation = positions[i] - positions[partner]
            distance = float(np.linalg.norm(separation))
            if distance < 1e-9:
                direction = rng.standard_normal(3)
                direction /= max(float(np.linalg.norm(direction)), 1e-9)
            else:
                direction = separation / distance
            positions[i] = positions[partner] + direction * jitter
            atoms[i].position = positions[i].copy()
            jittered += 1
            key = tuple(np.floor(positions[i] / threshold).astype(np.int64).tolist())
        grid.setdefault(key, []).append(i)
    if jittered:
        Config.debug_log(f"[isotropic] _resolve_close_contacts: {jittered} overlap(s) jittered by {jitter} nm")
        print(f"[info] Resolved {jittered} close-contact atom pair(s) by jitter={jitter} nm")
    return jittered


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
    gpu_id_raw = sim_params.get("gpu_id")
    gpu_id = int(str(gpu_id_raw)) if gpu_id_raw is not None else None
    # Medium-cell EM runs one cell at a time; default to 1 thread to avoid
    # over-subscription when many cells run sequentially on a shared machine.
    omp_threads_raw = sim_params.get("omp_threads", 1)
    ntomp = int(omp_threads_raw) if omp_threads_raw is not None else 1
    # Medium-cell EM is sequential → mpirun not beneficial; ignore mpi_np here.
    mdrun_extra_args = ["-pin", "off", "-dd", "1", "1", "1", "-rdd", "0.5", "-dds", "0.5"]
    run_geo_opt(
        structure_file=gro_path,
        topology_file=top_path,
        output_dir=out_dir,
        gmx_executable=sim_params.get("gromacs_executable_path", "gmx"),
        maxwarn=int(sim_params.get("grompp_maxwarn", 1)),
        mdp_overrides=mdp_overrides,
        mdrun_extra_args=mdrun_extra_args,
        gpu_id=gpu_id,
        ntomp=ntomp,
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


def pbc_diff(pos1, pos2, box):
    return minimum_image(pos1 - pos2, box)


def _optimize_linker_axes(
    repeats: Tuple[int, int, int],
    small_edge: float,
    cell_vector: np.ndarray,
    base_size: np.ndarray,
    seed: int | None = None
) -> Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], Tuple[Dict, Dict]]:
    """Plan connectivity-aware xyz-balanced linker axes for each cell using local_matching."""
    nx, ny, nz = repeats
    total_box = cell_vector * np.array([nx, ny, nz], dtype=float)

    # 1. Collect all linkers and their positions
    linkers = {}
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                big_origin = np.array([ix, iy, iz], dtype=float) * cell_vector
                for medium_offset in MEDIUM_ACTIVE_INDICES:
                    medium_idx = tuple(int(v) for v in medium_offset)
                    medium_origin = big_origin + np.array(medium_offset, dtype=float) * base_size

                    # Primary linker
                    p_pos = medium_origin
                    linkers[((ix, iy, iz), medium_idx, 0)] = p_pos

                    # Secondary linker
                    s_pos = medium_origin + np.array([small_edge, small_edge, small_edge], dtype=float)
                    linkers[((ix, iy, iz), medium_idx, 1)] = s_pos

    # 2. Collect all chains, their centers, endpoints, and map them to linkers
    chain_edges = []
    vertex_endpoints = {vid: {} for vid in linkers}

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                big_origin = np.array([ix, iy, iz], dtype=float) * cell_vector
                for medium_offset in MEDIUM_ACTIVE_INDICES:
                    medium_idx = tuple(int(v) for v in medium_offset)
                    medium_origin = big_origin + np.array(medium_offset, dtype=float) * base_size

                    for local_coords in POLYMER_POSITIONS:
                        chain_id = ((ix, iy, iz), medium_idx, local_coords)

                        center_local = small_edge * (np.array(local_coords, dtype=float) + 0.5)
                        C_chain = center_local + medium_origin
                        d = ORIENTATION_MAP[local_coords]

                        end_0_pos = center_local - 0.5 * d * small_edge + medium_origin
                        end_1_pos = center_local + 0.5 * d * small_edge + medium_origin

                        ep0 = (chain_id, 0)
                        ep1 = (chain_id, 1)
                        chain_edges.append((ep0, ep1))

                        # Find closest linker for end_0_pos
                        best_dist_0 = float('inf')
                        best_vid_0 = None
                        for vid, pos in linkers.items():
                            diff = pbc_diff(end_0_pos, pos, total_box)
                            dist = np.linalg.norm(diff)
                            if dist < best_dist_0:
                                best_dist_0 = dist
                                best_vid_0 = vid

                        # Find closest linker for end_1_pos
                        best_dist_1 = float('inf')
                        best_vid_1 = None
                        for vid, pos in linkers.items():
                            diff = pbc_diff(end_1_pos, pos, total_box)
                            dist = np.linalg.norm(diff)
                            if dist < best_dist_1:
                                best_dist_1 = dist
                                best_vid_1 = vid

                        # Map endpoints to their local coords at respective vertices
                        for ep, vid, end_pos in [(ep0, best_vid_0, end_0_pos), (ep1, best_vid_1, end_1_pos)]:
                            v = pbc_diff(C_chain, linkers[vid], total_box)
                            s = np.sign(v)
                            s = np.where(s == 0, 1.0, s)

                            if s[0] == s[1] == s[2]:
                                lc = (0, 0, 0)
                            elif s[0] != s[1] and s[0] != s[2]:
                                lc = (0, 1, 1)
                            elif s[1] != s[0] and s[1] != s[2]:
                                lc = (1, 0, 1)
                            else:
                                lc = (1, 1, 0)

                            vertex_endpoints[vid][lc] = ep

    vertices = []
    for vid, endpoints in vertex_endpoints.items():
        if len(endpoints) != 4:
            raise ValueError(f"Vertex {vid} has {len(endpoints)} endpoints instead of 4: {endpoints}")
        vertices.append(LocalVertex(vertex_id=vid, endpoints=endpoints))

    plan = plan_balanced_cycle_matchings(
        vertices=vertices,
        chain_edges=chain_edges,
        seed=seed,
        attempts=256,
        swaps_per_attempt=512
    )

    cell_plans = {}
    for choice in plan.choices:
        vid = choice.vertex_id
        cell_idx, medium_idx, type_idx = vid

        cell_key = (cell_idx, medium_idx)
        if cell_key not in cell_plans:
            cell_plans[cell_key] = [None, None]
        cell_plans[cell_key][type_idx] = {
            "axis": choice.axis,
            "planned_endpoint_edges": choice.edges,
        }

    return {k: tuple(v) for k, v in cell_plans.items()}


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
    linker_len = float(proto_plan.proto_linker.length) if proto_plan.proto_linker is not None else 0.0
    base_size = np.array([2.0 * small_edge, 2.0 * small_edge, 2.0 * small_edge], dtype=float)
    cell_vector = base_size * 2.0
    cube_side = float(base_size[0])

    backbone_weights = [entry.get('ratio', 1) for entry in backbone_defs] if backbone_defs else [1]
    linker_records = list(linker_library.records) if linker_library and linker_library.records else []
    if linker_records:
        linker_weights = [max(float(record.ratio), 0.0) for record in linker_records]
    else:
        linker_weights = [entry.get('ratio', 1) for entry in linker_defs] if linker_defs else [1]
    
    linker_def_lookup = require_unique(
        ((entry.get('id'), entry) for entry in linker_defs),
        'linker', 'id', source='LINKERS',
    )
    sequence_factory = getattr(proto_plan, 'sequence_factory', None)
    bond_lookup = getattr(proto_plan, 'bond_lookup', {})
    mean_sep = getattr(proto_plan, 'mean_sep', 0.24)
    proto_linker_length = max(proto_plan.proto_linker.length, 1e-9) if proto_plan.proto_linker is not None else 0.0

    # Ghost-bead extension for squeeze: one backbone bond step projected onto
    # a single axis (bond_c0 / sqrt(3)).  Derived from the actual chain geometry
    # so it works correctly for heterogeneous backbone mixtures — raw_length is
    # the sum of all bond_c0 values along the prototype chain (n-3 bonds for
    # n-2 proto beads), so dividing gives the true per-bond average.
    _n = proto_plan.segment_length
    _raw = proto_plan.proto_backbone.raw_length
    _backbone_bond_c0 = _raw / max(_n - 3, 1)
    ghost_step = _backbone_bond_c0 / np.sqrt(3.0)

    all_atoms = []
    all_chains = []
    atom_offset = 0
    chain_offset = 0
    per_cell_records = []
    global_linkers = []

    strategy = sim_params.get("linker_orientation_strategy", "random")
    planned_linkers = None
    if strategy == "connectivity_aware":
        seed = sim_params.get("random_seed")
        if seed is not None:
            seed = int(seed)
        planned_linkers = _optimize_linker_axes(
            repeats=repeats,
            small_edge=small_edge,
            cell_vector=cell_vector,
            base_size=base_size,
            seed=seed
        )

    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                big_origin = np.array([ix, iy, iz], dtype=float) * cell_vector
                for medium_offset in MEDIUM_ACTIVE_INDICES:
                    medium_idx = tuple(int(v) for v in medium_offset)
                    medium_origin = big_origin + np.array(medium_offset, dtype=float) * base_size
                    planned_endpoint_edges = [None, None]
                    if planned_linkers is not None:
                        cell_key = ((ix, iy, iz), medium_idx)
                        primary_plan, secondary_plan = planned_linkers[cell_key]
                        primary_axis = primary_plan["axis"]
                        secondary_axis = secondary_plan["axis"]
                        planned_endpoint_edges = [
                            primary_plan["planned_endpoint_edges"],
                            secondary_plan["planned_endpoint_edges"],
                        ]
                        axes = [primary_axis, secondary_axis]
                    else:
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
                            'local_indices': local_coords,
                            'planned_chain_id': ((ix, iy, iz), medium_idx, local_coords),
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
                            'backbone_residue_names': backbone_names or [],
                        }
                        if planned_endpoint_edges[spec_idx] is not None:
                            metadata['planned_endpoint_edges'] = planned_endpoint_edges[spec_idx]
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
                    positions_post = _squeeze_positions_to_cube(positions, cube_side, ghost_step=ghost_step)
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
                        delta = minimum_image(far_end_global - center, total_box)
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
        return minimum_image(vec, total_box)

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

                # Group backbone atoms by unique chain and find the minimum distance to this stub
                chain_dists = {}
                for entry in global_backbone:
                    key = (id(entry["record"]), entry["chain"].chain_index)
                    delta = _pbc_delta(entry["pos"] - anchor_pos)
                    dist = np.linalg.norm(delta)
                    if key not in chain_dists or dist < chain_dists[key][0]:
                        chain_dists[key] = (dist, entry)

                # Sort unique chains by distance
                sorted_chains = sorted(chain_dists.values(), key=lambda x: x[0])
                near_entries = [entry for dist, entry in sorted_chains[:3]]

                Config.debug_log(
                    f"[isotropy][lnk+offset] linker={chain.chain_index} anchor={atom_idx} "
                    f"axis={axis} closest_dists={[dist for dist, entry in sorted_chains[:3]]}"
                )

                if len(near_entries) < 3:
                    continue

                selected = random.choice(near_entries)
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
                near_end_pos = sel_positions[ordered[0]] + sel_origin
                shift_delta = _pbc_delta(anchor_pos - near_end_pos)
                shift_sign = float(np.sign(shift_delta[axis_idx]))
                Config.debug_log(
                    f"[isotropy][lnk+offset] APPLY linker={chain.chain_index} anchor={atom_idx} "
                    f"axis={axis} chain={sel_chain.chain_index} beads={len(ordered)} "
                    f"shift_sign={shift_sign:.0f} axis_delta={shift_delta[axis_idx]:.4f}"
                )
                # ordered[0] is the end closest to the linker anchor.
                # ordered[-1] is the opposite end (farthest).
                # We want to shift the chain such that the offset is max at ordered[0] and decays to 0 at ordered[-1].
                for i, sel_idx in enumerate(ordered[:-1]):
                    # i goes from 0 to len(ordered)-2.
                    # factor should decrease as i increases.
                    factor = (len(ordered) - i) / (len(ordered) + 2)
                    offset = factor * linker_len
                    sel_positions[sel_idx][axis_idx] += offset * shift_sign

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

    # Diamond lattice geometry: endpoint of chain (0,0,0) and start of chain (0,1,1)
    # share the same (x, y, z) position exactly → LJ infinity → EM NaN at Step 0.
    # Resolve all such coincidences with a small jitter before any GROMACS run.
    _resolve_close_contacts(all_atoms, threshold=0.001, jitter=0.05)

    return LayoutBlueprint(atoms=all_atoms, chains=all_chains)
