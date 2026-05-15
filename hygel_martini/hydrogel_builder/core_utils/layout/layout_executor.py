"""Proto layout instantiation helpers."""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from hygel_martini.hydrogel_builder.core_utils.layout.proto_layout import LayoutPlan, LayoutCell, LinkPlacement

DEFAULT_BACKBONE_AXIS = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)


@dataclass
class InstantiatedChain:
    positions: np.ndarray
    definition: Dict[str, Any]
    metadata: Dict[str, Any]
    template: Any | None = None


@dataclass
class InstantiatedLayout:
    backbone_segments: List[InstantiatedChain]
    linker_segments: List[InstantiatedChain]


@dataclass
class AtomBlueprint:
    chain_type: str
    chain_index: int
    bead_index: int
    position: np.ndarray
    component_id: str
    atom_name: str
    atom_type: str
    residue_name: str
    residue_number: int
    charge_group_number: int
    mass: float
    charge: float
    backbone_type: str | None = None
    extra: Dict[str, Any] | None = None


@dataclass
class ChainBlueprint:
    chain_type: str
    chain_index: int
    component_id: str
    definition: Dict[str, Any]
    atom_indices: List[int]
    metadata: Dict[str, Any]


@dataclass
class LayoutBlueprint:
    atoms: List[AtomBlueprint]
    chains: List[ChainBlueprint]


def _center_positions(positions: np.ndarray) -> np.ndarray:
    centroid = np.mean(positions, axis=0)
    return positions - centroid


def _rotate_between_vectors(vectors: np.ndarray,
                            source: np.ndarray,
                            target: np.ndarray) -> np.ndarray:
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    if source_norm < 1e-9 or target_norm < 1e-9:
        return vectors
    s = source / source_norm
    t = target / target_norm
    if np.allclose(s, t):
        return vectors
    if np.allclose(s, -t):
        return -vectors
    axis = np.cross(s, t)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return vectors
    axis /= axis_norm
    angle = np.arccos(np.clip(np.dot(s, t), -1.0, 1.0))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return vectors @ R.T


def _rotate_from_xaxis(vectors: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_norm = np.linalg.norm(target)
    if target_norm < 1e-9:
        return vectors
    unit_target = target / target_norm
    basis = np.array([1.0, 0.0, 0.0])
    if np.allclose(unit_target, basis):
        return vectors
    if np.allclose(unit_target, -basis):
        return np.column_stack((-vectors[:, 0], vectors[:, 1], vectors[:, 2]))
    axis = np.cross(basis, unit_target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return vectors
    axis /= axis_norm
    angle = np.arccos(np.clip(np.dot(basis, unit_target), -1.0, 1.0))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return vectors @ R.T


def _alignment_basis(axis: np.ndarray) -> np.ndarray:
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        axis = np.array([1.0, 0.0, 0.0])
        axis_norm = 1.0
    x_axis = axis / axis_norm
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(x_axis, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    y_axis = ref - np.dot(ref, x_axis) * x_axis
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-9:
        y_axis = np.array([0.0, 1.0, 0.0])
        y_axis -= np.dot(y_axis, x_axis) * x_axis
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-9:
            y_axis = np.array([0.0, 1.0, 0.0])
            y_norm = np.linalg.norm(y_axis)
    y_axis /= y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-9:
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        z_axis /= z_norm
    return np.column_stack((x_axis, y_axis, z_axis))


def instantiate_backbone(cell: LayoutCell, proto_positions: np.ndarray) -> InstantiatedChain:
    custom_positions = None
    if cell.metadata:
        custom_positions = cell.metadata.get('proto_positions')
    base_positions = custom_positions if custom_positions is not None and len(custom_positions) > 0 else proto_positions
    centered = _center_positions(base_positions)
    rotated = _rotate_between_vectors(centered, DEFAULT_BACKBONE_AXIS, cell.direction)
    scale = 1.0
    if cell.metadata:
        scale = cell.metadata.get('length_scale', 1.0)
    positions = cell.origin + rotated * scale
    metadata = {'cell_index': cell.cell_index}
    if cell.metadata:
        filtered = {k: v for k, v in cell.metadata.items() if k != 'proto_positions'}
        metadata.update(filtered)
    return InstantiatedChain(
        positions=positions,
        definition=cell.backbone_definition,
        metadata=metadata
    )


def instantiate_linker(layout_plan: LayoutPlan,
                       link: LinkPlacement,
                       proto_positions: np.ndarray) -> InstantiatedChain:
    metadata = link.metadata.copy() if link.metadata else {}
    definition = link.linker_definition or {}
    defn_body = definition.get('definition', definition)
    bead_defs = defn_body.get('beads', [])
    library = getattr(layout_plan.proto_plan, 'linker_library', None)
    template = None
    template_id = metadata.get('linker_template_id')
    if template_id and library and hasattr(library, 'lookup'):
        template = library.lookup.get(template_id)
        if template is None:
            print(f"[경고] 링커 템플릿 '{template_id}'을(를) 찾지 못해 proto 좌표를 사용합니다.")
    axis_dir = np.array(link.axis_direction, dtype=np.float64)
    axis_norm = np.linalg.norm(axis_dir)
    if axis_norm < 1e-9:
        axis_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis_norm = 1.0
    axis_unit = axis_dir / axis_norm
    anchor = np.array(link.anchor_position, dtype=np.float64)
    positions = None

    if template is not None and template.coords.shape[0] == len(bead_defs):
        span_length = metadata.get('span_length') or template.span_length
        start_pos = anchor - axis_unit * (span_length / 2.0)
        basis = _alignment_basis(axis_unit)
        local_coords = np.array(template.coords, dtype=np.float64)
        oriented = local_coords @ basis.T
        positions = start_pos + oriented
    elif template is not None and template.coords.shape[0] != len(bead_defs):
        print(f"[경고] 템플릿 '{template_id}' bead 수({template.coords.shape[0]})가 "
              f"정의({len(bead_defs)})와 달라 proto 좌표를 사용합니다.")
        centered = _center_positions(proto_positions)
        rotated = _rotate_from_xaxis(centered, axis_dir)
        scale = metadata.get('length_scale', 1.0)
        positions = anchor + rotated * scale
    else:
        centered = _center_positions(proto_positions)
        rotated = _rotate_from_xaxis(centered, axis_dir)
        scale = metadata.get('length_scale', 1.0)
        positions = anchor + rotated * scale

    embed_metadata = {
        'connected_cells': link.connected_cells,
        'axis_direction': axis_dir,
        'anchor_position': anchor
    }
    embed_metadata.update(metadata)
    return InstantiatedChain(
        positions=positions,
        definition=definition,
        metadata=embed_metadata,
        template=template
    )


def instantiate_layout(layout_plan: LayoutPlan) -> InstantiatedLayout:
    backbone_segments: List[InstantiatedChain] = []
    linker_segments: List[InstantiatedChain] = []

    proto_backbone_positions = layout_plan.proto_plan.proto_backbone.positions
    proto_linker = layout_plan.proto_plan.proto_linker
    proto_linker_positions = proto_linker.positions if proto_linker is not None else np.zeros((0, 3), dtype=np.float64)

    for cell in layout_plan.cells:
        backbone_segments.append(instantiate_backbone(cell, proto_backbone_positions))

    for link in layout_plan.links:
        linker_segments.append(instantiate_linker(layout_plan, link, proto_linker_positions))

    return InstantiatedLayout(backbone_segments=backbone_segments,
                              linker_segments=linker_segments)


def _backbone_atom_params(component_entry: Dict[str, Any], bead_index: int) -> Dict[str, Any]:
    definition = component_entry.get('definition', component_entry)
    atom_name = definition.get('atom_name', f"BB{bead_index:02d}")
    atom_type = definition.get('atom_type', 'C1')
    raw_residue_name = definition.get('residue_name', 'BCK')
    
    # Handle list residue_name
    if isinstance(raw_residue_name, list):
        residue_name = raw_residue_name[0] # Default, will be overridden in builder if needed
    else:
        residue_name = raw_residue_name

    residue_number = definition.get('residue_number', 1)
    cgnr = definition.get('charge_group_number', 1)
    mass = float(definition.get('mass', 72.0))
    charge = float(definition.get('charge', 0.0))
    return {
        'atom_name': atom_name,
        'atom_type': atom_type,
        'residue_name': residue_name,
        'residue_number': residue_number,
        'charge_group_number': cgnr,
        'mass': mass,
        'charge': charge
    }


def _linker_atom_params(component_entry: Dict[str, Any], bead_index: int) -> Dict[str, Any]:
    definition = component_entry.get('definition', component_entry)
    raw_residue_name = definition.get('residue_name', 'LNK')
    
    if isinstance(raw_residue_name, list):
        residue_name = raw_residue_name[0]
    else:
        residue_name = raw_residue_name

    residue_number = definition.get('residue_number', 2)
    cgnr = definition.get('charge_group_number', 2)
    beads = definition.get('beads', [])
    bead_def = beads[bead_index] if bead_index < len(beads) else {}
    atom_name = bead_def.get('name', f"L{bead_index:02d}")
    atom_type = bead_def.get('type', definition.get('atom_type', 'P5'))
    mass = float(bead_def.get('mass', definition.get('mass', 72.0)))
    charge = float(bead_def.get('charge', definition.get('charge', 0.0)))
    return {
        'atom_name': atom_name,
        'atom_type': atom_type,
        'residue_name': residue_name,
        'residue_number': residue_number,
        'charge_group_number': cgnr,
        'mass': mass,
        'charge': charge
    }


def build_atom_blueprint(layout_plan: LayoutPlan,
                         backbone_defs: List[Dict[str, Any]]) -> LayoutBlueprint:
    inst = instantiate_layout(layout_plan)
    atoms: List[AtomBlueprint] = []
    chains: List[ChainBlueprint] = []

    for chain_idx, chain in enumerate(inst.backbone_segments):
        component_entry = chain.definition or {}
        component_id = component_entry.get('id', f"BACKBONE_{chain_idx}")
        sequence = chain.metadata.get('sequence', []) if chain.metadata else []
        atom_indices: List[int] = []
        for bead_idx, position in enumerate(chain.positions):
            entry = sequence[bead_idx] if bead_idx < len(sequence) else component_entry
            params = _backbone_atom_params(entry or component_entry, bead_idx)
            
            # Re-override residue_name based on chain_idx if it's a list
            raw_def = (entry or component_entry).get('definition', (entry or component_entry))
            raw_res_name = raw_def.get('residue_name')
            if isinstance(raw_res_name, list) and len(raw_res_name) > 0:
                params['residue_name'] = raw_res_name[chain_idx % len(raw_res_name)]

            bead_component_id = (entry or component_entry).get('id', component_id)
            atoms.append(AtomBlueprint(
                chain_type='backbone',
                chain_index=chain_idx,
                bead_index=bead_idx,
                position=np.array(position, dtype=np.float64),
                component_id=bead_component_id,
                atom_name=params['atom_name'],
                atom_type=params['atom_type'],
                residue_name=params['residue_name'],
                residue_number=params['residue_number'],
                charge_group_number=params['charge_group_number'],
                mass=params['mass'],
                charge=params['charge'],
                backbone_type=bead_component_id,
                extra=None
            ))
            atom_indices.append(len(atoms) - 1)

        chains.append(ChainBlueprint(
            chain_type='backbone',
            chain_index=chain_idx,
            component_id=component_id,
            definition=component_entry.get('definition', component_entry),
            atom_indices=atom_indices,
            metadata=chain.metadata or {}
        ))

    for chain_idx, chain in enumerate(inst.linker_segments):
        component_entry = chain.definition or {}
        component_id = component_entry.get('id', f"LINKER_{chain_idx}")
        atom_indices: List[int] = []
        for bead_idx, position in enumerate(chain.positions):
            params = _linker_atom_params(component_entry, bead_idx)
            template = chain.template
            original_index = None
            if template and bead_idx < len(getattr(template, "beads", [])):
                original_index = getattr(template.beads[bead_idx], "original_index", bead_idx + 1)
            atoms.append(AtomBlueprint(
                chain_type='linker',
                chain_index=chain_idx,
                bead_index=bead_idx,
                position=np.array(position, dtype=np.float64),
                component_id=component_id,
                atom_name=params['atom_name'],
                atom_type=params['atom_type'],
                residue_name=params['residue_name'],
                residue_number=params['residue_number'],
                charge_group_number=params['charge_group_number'],
                mass=params['mass'],
                charge=params['charge'],
                backbone_type=None,
                extra={'source_template': chain.template, 'original_index': original_index}
            ))
            atom_indices.append(len(atoms) - 1)

        definition = component_entry.get('definition', component_entry)
        axis_dir = np.array(chain.metadata.get('axis_direction', np.array([1.0, 0.0, 0.0])), dtype=np.float64)
        if np.linalg.norm(axis_dir) < 1e-9:
            axis_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis_dir /= np.linalg.norm(axis_dir)
        anchor = np.array(chain.metadata.get('anchor_position', np.zeros(3)), dtype=np.float64)
        
        # Process stubs for backbone_1 and backbone_2
        stub_loops = [
            (definition.get('external_bonds_1', []), 'backbone_1', 0),
            (definition.get('external_bonds_2', []), 'backbone_2', 1)
        ]

        for external_bonds, stub_type, stub_def_idx in stub_loops:
            stub_definitions = definition.get('stub_definitions', [])
            if not external_bonds or not stub_definitions or stub_def_idx >= len(stub_definitions):
                continue
            
            original_stub_def = stub_definitions[stub_def_idx]

            for stub_idx, ext in enumerate(external_bonds):
                bead_idx = int(ext.get('from_bead', 0))
                if bead_idx < 0 or bead_idx >= len(chain.positions):
                    continue
                
                target_bb = ext.get('to_backbone')
                if target_bb == 'dummy_id':
                    target_bb = None
                
                # Use original stub properties for naming, but backbone_residue_name for residue
                raw_backbone_name = definition.get('backbone_name', 'STUBRES')
                if isinstance(raw_backbone_name, list) and len(raw_backbone_name) > 0:
                    res_name = raw_backbone_name[stub_def_idx % len(raw_backbone_name)]
                else:
                    res_name = raw_backbone_name

                params = {
                    'atom_name': original_stub_def.get('atom', 'STUB'),
                    'atom_type': original_stub_def.get('type', 'P5'),
                    'residue_name': res_name,
                    'residue_number': original_stub_def.get('resnr', 1),
                    'charge_group_number': original_stub_def.get('cgnr', 1),
                    'mass': original_stub_def.get('mass', 72.0),
                    'charge': original_stub_def.get('charge', 0.0)
                }

                bead_pos = np.array(chain.positions[bead_idx], dtype=np.float64)
                proj = float(np.dot(bead_pos - anchor, axis_dir))
                sign = 1.0 if proj >= 0 else -1.0
                _proto_linker = layout_plan.proto_plan.proto_linker
                ext_length = float(ext.get('length', _proto_linker.length if _proto_linker is not None else 0.0))
                stub_pos = bead_pos + axis_dir * ext_length * sign
                
                extra = {
                    'stub_from_bead': bead_idx,
                    'target_backbone': target_bb,
                    'external_params': {k: v for k, v in ext.items() if k not in ('from_bead', 'to_backbone')},
                    'is_terminal_backbone': True,
                    'stub_type': stub_type,
                    'source_template': chain.template,
                    'original_index': original_stub_def.get('nr')
                }
                
                atoms.append(AtomBlueprint(
                    chain_type='linker',
                    chain_index=chain_idx,
                    bead_index=-(stub_idx + 10 * (stub_def_idx + 1)), # Ensure unique negative index
                    position=stub_pos,
                    component_id=target_bb or component_id,
                    atom_name=params['atom_name'],
                    atom_type=params['atom_type'],
                    residue_name=params['residue_name'],
                    residue_number=params['residue_number'],
                    charge_group_number=params['charge_group_number'],
                    mass=params['mass'],
                    charge=params['charge'],
                    backbone_type=target_bb,
                    extra=extra
                ))
                atom_indices.append(len(atoms) - 1)

        chains.append(ChainBlueprint(
            chain_type='linker',
            chain_index=chain_idx,
            component_id=component_id,
            definition=component_entry.get('definition', component_entry),
            atom_indices=atom_indices,
            metadata=chain.metadata or {}
        ))

    return LayoutBlueprint(atoms=atoms, chains=chains)
