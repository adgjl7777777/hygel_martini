"""Proto-level layout planning for the diamond network."""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Sequence

import random

import numpy as np

try:
    from hygel_martini.hydrogel_builder.config_params.config import Config
except Exception:  # pragma: no cover - config may be unavailable in some contexts
    Config = None

from hygel_martini.hydrogel_builder.core_utils.common.collisions import require_unique
from hygel_martini.hydrogel_builder.core_utils.layout.proto_builder import ProtoPlan
from hygel_martini.hydrogel_builder.core_utils.templates.linker_loader import LinkerTemplateLibrary


MEDIUM_ACTIVE_INDICES = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

POLYMER_POSITIONS = [
    (0, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
]

ORIENTATION_MAP = {
    (0, 0, 0): np.array([1.0, 1.0, 1.0]),
    (1, 1, 0): np.array([1.0, 1.0, -1.0]),
    (1, 0, 1): np.array([1.0, -1.0, 1.0]),
    (0, 1, 1): np.array([-1.0, 1.0, 1.0]),
}

LINKER_MEDIUM_INDEX = np.array([0.0, 0.0, 0.0], dtype=float)
LINKER_LOCAL_ANCHORS = [
    (np.array([0.0, 0.0, 0.0], dtype=float), np.array([1.0, 0.0, 0.0], dtype=float)),
    (np.array([0.5, 0.5, 0.5], dtype=float), np.array([-1.0, 0.0, 0.0], dtype=float)),
]


@dataclass
class LayoutCell:
    origin: np.ndarray
    direction: np.ndarray
    backbone_definition: Dict[str, Any]
    cell_index: Tuple[int, int, int]
    metadata: Dict[str, Any] | None = None


@dataclass
class LinkPlacement:
    anchor_position: np.ndarray
    axis_direction: np.ndarray
    linker_definition: Dict[str, Any]
    connected_cells: Tuple[int, int]
    metadata: Dict[str, Any] | None = None


@dataclass
class LayoutPlan:
    proto_plan: ProtoPlan
    cells: List[LayoutCell]
    links: List[LinkPlacement]


def _linear_index(ix: int, iy: int, iz: int, repeats: Tuple[int, int, int]) -> int:
    nx, ny, nz = repeats
    return ix * ny * nz + iy * nz + iz


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return vec
    return vec / norm


def _get_anisotropy_axis() -> str:
    axis = "x"
    if Config is not None:
        try:
            axis = str(Config.get_param('simulation_parameters').get('anisotropy', 'x')).lower()
        except Exception:
            axis = "x"
    if axis not in ("x", "y", "z"):
        axis = "x"
    return axis


def _axis_rotation_matrix(axis: str) -> np.ndarray:
    if axis == "y":
        # rotate +90 deg around z: x->y, y->-x
        return np.array([[0.0, -1.0, 0.0],
                         [1.0,  0.0, 0.0],
                         [0.0,  0.0, 1.0]], dtype=float)
    if axis == "z":
        # rotate +90 deg around y: x->z, z->-x
        return np.array([[0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0],
                         [-1.0, 0.0, 0.0]], dtype=float)
    return np.eye(3, dtype=float)


def _apply_rotation(vec: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return vec @ rot.T


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


def _backbone_target_length(entry: Dict[str, Any], proto_plan: ProtoPlan) -> float:
    definition = entry.get('definition', entry)
    bond_len = definition.get('bond_c0')
    if bond_len is None or bond_len <= 0:
        intervals = max(proto_plan.segment_length - 1, 1)
        avg = proto_plan.proto_backbone.length / intervals if intervals > 0 else proto_plan.proto_backbone.length
        bond_len = avg
    intervals = max(proto_plan.segment_length - 1, 1)
    return float(bond_len) * intervals


def _linker_total_length(entry: Dict[str, Any], fallback: float, override: float | None = None) -> float:
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


def generate_layout_plan(proto_plan: ProtoPlan,
                         backbone_defs: List[Dict[str, Any]],
                         linker_defs: List[Dict[str, Any]],
                         repeats: Tuple[int, int, int],
                         backbone_strategy: Dict[str, Any] | None = None,
                         linker_strategy: Dict[str, Any] | None = None,
                         linker_library: LinkerTemplateLibrary | None = None,
                         linker_axes: Optional[Sequence[str]] = None) -> LayoutPlan:
    nx, ny, nz = repeats
    cells: List[LayoutCell] = []
    links: List[LinkPlacement] = []
    medium_size = proto_plan.medium_size
    small_edge = proto_plan.small_size[0]
    proto_linker_length = max(proto_plan.proto_linker.length, 1e-9) if proto_plan.proto_linker is not None else 0.0
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
    linker_len = proto_linker_length
    sequence_factory = getattr(proto_plan, 'sequence_factory', None)
    bond_lookup = getattr(proto_plan, 'bond_lookup', {})
    mean_sep = getattr(proto_plan, 'mean_sep', 0.24)

    anisotropy_axis = _get_anisotropy_axis()
    rot = _axis_rotation_matrix(anisotropy_axis)
    linker_axes = _normalize_linker_axes(linker_axes)

    print("[DEBUG] generate_layout_plan start")
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                big_origin = np.array([ix, iy, iz], dtype=float) * proto_plan.cell_vector
                linear_idx = _linear_index(ix, iy, iz, repeats)

                for medium_offset_tuple in MEDIUM_ACTIVE_INDICES:
                    medium_idx = tuple(int(v) for v in medium_offset_tuple)
                    medium_origin = big_origin + np.array(medium_offset_tuple, dtype=float) * medium_size
                    used_sequences: set[Tuple[str, ...]] = set()
                    enforce_unique_sequences = (
                        sequence_factory is not None and getattr(sequence_factory, 'definition_count', 0) > 1
                    )

                    def _axis_center(idx: int) -> float:
                        return small_edge * (0.5 + idx)

                    def _x_center(idx: int) -> float:
                        return linker_len * (1.0 + idx) + small_edge * (0.5 + idx)

                    def _y_center(idx: int) -> float:
                        return linker_len * (1.0 + idx) + small_edge * (0.5 + idx)

                    def _z_center(idx: int) -> float:
                        return linker_len * (1.0 + idx) + small_edge * (0.5 + idx)

                    primary_axis = linker_axes[0] if linker_axes else "x"
                    for local_coords in POLYMER_POSITIONS:
                        x_idx, y_idx, z_idx = local_coords
                        if primary_axis == "y":
                            center_local = np.array([
                                _axis_center(x_idx),
                                _y_center(y_idx),
                                _axis_center(z_idx)
                            ], dtype=float)
                        elif primary_axis == "z":
                            center_local = np.array([
                                _axis_center(x_idx),
                                _axis_center(y_idx),
                                _z_center(z_idx)
                            ], dtype=float)
                        else:
                            center_local = np.array([
                                _x_center(x_idx),
                                _axis_center(y_idx),
                                _axis_center(z_idx)
                            ], dtype=float)
                        center = medium_origin + center_local
                        orient = ORIENTATION_MAP[local_coords]
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
                        direction = _normalize(orient)
                        if anisotropy_axis != "x":
                            center = _apply_rotation(center, rot)
                            direction = _normalize(_apply_rotation(direction, rot))
                        metadata = {
                            'medium_index': medium_idx,
                            'length_scale': 1.0,
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

                    primary_axis = linker_axes[0] if linker_axes else "x"
                    secondary_axis = linker_axes[1] if len(linker_axes) > 1 else primary_axis

                    axis_index = {"x": 0, "y": 1, "z": 2}
                    first_vec = np.zeros(3, dtype=float)
                    first_vec[axis_index[primary_axis]] = linker_len * 0.5

                    second_vec = np.array([small_edge, small_edge, small_edge], dtype=float)
                    second_vec[axis_index[primary_axis]] += linker_len
                    second_vec[axis_index[secondary_axis]] += linker_len * 0.5

                    linker_local_specs = [first_vec, second_vec]

                    if not linker_defs and not linker_records:
                        continue

                    for spec_idx, local_anchor in enumerate(linker_local_specs):
                        axis_choice = random.choice(linker_axes)
                        axis_rot = _axis_rotation_matrix(axis_choice)
                        rotated_anchor = _apply_rotation(local_anchor, axis_rot)
                        anchor = medium_origin + rotated_anchor
                        axis_dir = np.array([random.choice([-1.0, 1.0]), 0.0, 0.0], dtype=float)
                        axis_dir = _apply_rotation(axis_dir, axis_rot)
                        if anisotropy_axis != "x":
                            anchor = _apply_rotation(anchor, rot)
                            axis_dir = _normalize(_apply_rotation(axis_dir, rot))
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
                        metadata = {
                            'medium_index': medium_idx,
                            'length_scale': scale,
                            'local_linker_index': spec_idx,
                            'linker_axis': axis_choice
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
                            connected_cells=(linear_idx, linear_idx),
                            metadata=metadata
                        ))

    return LayoutPlan(proto_plan=proto_plan, cells=cells, links=links)
