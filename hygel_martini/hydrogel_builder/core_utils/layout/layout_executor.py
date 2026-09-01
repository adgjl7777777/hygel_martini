"""Materialize a LayoutPlan into concrete coordinates and atom blueprints.

Second stage of the layout pipeline: a :class:`LayoutPlan` (from the diamond
``proto_layout`` or the net-driven ``net_layout``) describes *where* each
strand and linker goes; this module computes actual per-atom positions
(:func:`instantiate_layout`) and flattens them, together with every per-atom
identity (type, residue, charge, mass, template provenance), into the
:class:`LayoutBlueprint` that ``proto_populator`` turns into World state.

Placement rules by chain kind:

* Martini backbone: centered proto bead chain, rotated from the default
  <111> axis onto the cell direction and scaled by ``length_scale``;
* whole-strand template (all-atom): the molecule placed rigidly -- attachment
  axis onto the segment, optional roll about it, never scaled;
* linker: template-local coordinates rotated via the alignment basis (f=2)
  or anchored at the centroid with stubs on the arm vectors (f>2), plus the
  stub atoms emitted as separate blueprint atoms carrying their planned
  endpoint metadata.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from hygel_martini.hydrogel_builder.core_utils.layout.proto_layout import LayoutPlan, LayoutCell, LinkPlacement

DEFAULT_BACKBONE_AXIS = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)


@dataclass
class InstantiatedChain:
    """One placed chain: absolute positions (nm) plus its definition/metadata.

    ``template`` is set for template-backed chains (linkers always; backbones
    only in whole-strand mode) so the blueprint can record provenance.
    """

    positions: np.ndarray
    definition: Dict[str, Any]
    metadata: Dict[str, Any]
    template: Any | None = None


@dataclass
class InstantiatedLayout:
    """All placed chains, split by kind, in layout order."""

    backbone_segments: List[InstantiatedChain]
    linker_segments: List[InstantiatedChain]


@dataclass
class AtomBlueprint:
    """Everything the populator needs to create one Atom.

    ``bead_index`` orders atoms within their chain; ``extra`` carries
    provenance and wiring: ``source_template``, ``original_index`` (1-based
    ITP number, the key of the populator's rich-section map), stub metadata
    (``stub_type``, ``stub_from_bead``, ``external_params``,
    ``target_backbone``) and planned-endpoint data.
    """

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
    """Per-chain record: which blueprint atoms belong to it, plus metadata
    (planned ids, sequence, strand template, attachment positions...)."""

    chain_type: str
    chain_index: int
    component_id: str
    definition: Dict[str, Any]
    atom_indices: List[int]
    metadata: Dict[str, Any]


@dataclass
class LayoutBlueprint:
    """The flat handoff consumed by ``proto_populator``."""

    atoms: List[AtomBlueprint]
    chains: List[ChainBlueprint]


def _center_positions(positions: np.ndarray) -> np.ndarray:
    """Positions translated so their centroid sits at the origin."""
    centroid = np.mean(positions, axis=0)
    return positions - centroid


def _rotate_between_vectors(vectors: np.ndarray,
                            source: np.ndarray,
                            target: np.ndarray) -> np.ndarray:
    """Rotate row vectors by the rotation carrying ``source`` onto ``target``.

    Degenerate inputs (zero-length source/target, parallel already) return
    the input unchanged; antiparallel returns the negation, which is exact
    for the centered, sign-symmetric proto chains this is used on.
    """
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
    """Rotate row vectors from the +x axis onto ``target`` (Rodrigues).

    The antiparallel case flips the x components only, preserving the
    template's local y/z handedness for two-stub linkers.
    """
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
    """Right-handed orthonormal basis (columns) whose x axis is ``axis``.

    Mirrors the loader's ``_orthonormal_basis`` (template coordinates are
    stored in such a frame) so instantiation is basis @ local. Degenerate
    axes fall back to a safe frame instead of raising: by the time this
    runs the loader has already validated real geometry.
    """
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


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit ``axis`` by ``angle`` radians."""
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation matrix carrying unit vector ``source`` onto unit ``target``.

    Unlike ``_rotate_between_vectors`` this returns the matrix itself (the
    whole-strand path composes it with a roll about ``target``), and the
    antiparallel case is an exact half-turn about a perpendicular axis
    rather than a coordinate negation, which would mirror the molecule.
    """
    cross = np.cross(source, target)
    norm = float(np.linalg.norm(cross))
    dot = float(np.dot(source, target))
    if norm < 1e-12:
        if dot > 0.0:
            return np.eye(3)
        # Antiparallel: rotate half a turn about any perpendicular axis.
        seed = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        perp = np.cross(source, seed)
        perp /= np.linalg.norm(perp)
        return _axis_rotation(perp, np.pi)
    return _axis_rotation(cross / norm, np.arctan2(norm, dot))


def instantiate_backbone(cell: LayoutCell, proto_positions: np.ndarray) -> InstantiatedChain:
    """Place one strand: rigid whole-strand template, or scaled bead chain.

    Whole-strand branch: template coords (already centered on the attachment
    midpoint) get the attachment axis rotated onto ``cell.direction``, an
    optional golden-angle ``roll`` about it, and a translation to
    ``cell.origin`` -- no scaling, all-atom bond lengths are not free.
    Martini branch: the (possibly bowed, per-cell ``proto_positions``
    metadata) prototype is centered, rotated from the default <111> axis
    onto the cell direction, scaled by ``length_scale`` and translated.
    """
    strand_template = cell.metadata.get('strand_template') if cell.metadata else None
    if strand_template is not None:
        # Whole-strand template: one rigid molecule. Its local coordinates are
        # centered on the attachment midpoint, so aligning the attachment axis
        # with the segment direction and translating to the segment center
        # (cell.origin) puts both attachment atoms on the junction line. No
        # scaling: all-atom bond lengths are not free parameters.
        rotation = _rotation_between(
            np.asarray(strand_template.attachment_axis, dtype=np.float64),
            np.asarray(cell.direction, dtype=np.float64),
        )
        roll = float(cell.metadata.get('roll', 0.0) or 0.0)
        if roll:
            rotation = _axis_rotation(
                np.asarray(cell.direction, dtype=np.float64), roll
            ) @ rotation
        positions = cell.origin + strand_template.coords @ rotation.T
        metadata = {'cell_index': cell.cell_index}
        metadata.update(cell.metadata)
        return InstantiatedChain(
            positions=positions,
            definition=cell.backbone_definition,
            metadata=metadata,
            template=strand_template,
        )

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
    """Place one junction molecule's *body* beads (stubs are emitted later).

    With a loaded template whose bead count matches the definition: two-stub
    templates rotate their local frame onto the link axis and scale to the
    planned span; multi-arm (f > 2) templates anchor their centroid at the
    link anchor without scaling. Without a usable template the proto linker
    bead chain is rotated/scaled instead (a bead-count mismatch is reported,
    then falls back the same way). The resolved template rides on the result
    so the blueprint can attach provenance and arm vectors.
    """
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
        local_coords = np.array(template.coords, dtype=np.float64)
        if getattr(template, 'functionality', 2) == 2:
            # Two-stub convention: the template's local frame puts its origin
            # on the first stub with x along the span, so the anchor is the
            # midpoint and the placement backs off half a span.
            span_length = metadata.get('span_length') or template.span_length
            start_pos = anchor - axis_unit * (span_length / 2.0)
            basis = _alignment_basis(axis_unit)
            positions = start_pos + local_coords @ basis.T
        else:
            # A multi-arm junction has no span to halve: its local frame is
            # already centred on the stub centroid, so the anchor is that
            # centroid and only an orientation is applied.
            orientation = metadata.get('orientation')
            if orientation is None:
                basis = np.eye(3, dtype=np.float64)
            else:
                basis = np.array(orientation, dtype=np.float64).reshape(3, 3)
            positions = anchor + local_coords @ basis.T
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
    """Place every cell and link of the plan, in plan order.

    Order is a contract: chain indices assigned downstream (blueprint,
    populator, planned-endpoint translation) all assume backbone segments
    appear in ``layout_plan.cells`` order and linkers in
    ``layout_plan.links`` order.
    """
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
    """Per-bead identity for a Martini backbone bead from its definition.

    Defaults are the Series-01 conventions (type C1, mass 72, residue BCK);
    a list-valued ``residue_name`` is resolved per chain by the caller.
    Whole-strand chains never reach this -- their identities come verbatim
    from the template.
    """
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
    """Per-bead identity for a linker body bead (stubs are emitted apart).

    Bead-level values win over definition-level fallbacks; the defaults are
    Martini-flavored (type P5, mass 72, residue LNK).
    """
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
    """Flatten the instantiated layout into per-atom blueprints.

    Backbones: whole-strand chains emit every template atom verbatim (with
    ``source_template``/``original_index`` provenance and the attachment
    positions in chain metadata); Martini chains emit per-bead identities
    from their definitions/sequence. Linkers: body beads first, then the
    stub atoms -- placed on the template arm vectors (f > 2) or at the span
    ends (two-stub), each carrying its stub bond parameters, admissible
    targets, and planned endpoint metadata for the crosslink router.

    Returns:
        The :class:`LayoutBlueprint` handed to
        ``proto_populator.populate_hydrogel_from_blueprint``.
    """
    inst = instantiate_layout(layout_plan)
    atoms: List[AtomBlueprint] = []
    chains: List[ChainBlueprint] = []

    for chain_idx, chain in enumerate(inst.backbone_segments):
        component_entry = chain.definition or {}
        component_id = component_entry.get('id', f"BACKBONE_{chain_idx}")

        strand_template = chain.metadata.get('strand_template') if chain.metadata else None
        if strand_template is not None:
            # Whole-strand template: every atom comes verbatim from the
            # template (type, charge, mass, residue), with its rigidly placed
            # coordinate. source_template/original_index make the populator's
            # rich-section mapping and the angle machinery see this chain
            # exactly like a linker template instance.
            atom_indices = []
            for bead_idx, bead in enumerate(strand_template.beads):
                atoms.append(AtomBlueprint(
                    chain_type='backbone',
                    chain_index=chain_idx,
                    bead_index=bead_idx,
                    position=np.array(chain.positions[bead_idx], dtype=np.float64),
                    component_id=component_id,
                    atom_name=bead.name,
                    atom_type=bead.atom_type,
                    residue_name=bead.residue_name,
                    residue_number=bead.residue_number,
                    charge_group_number=bead.cgnr,
                    mass=bead.mass,
                    charge=bead.charge,
                    backbone_type=component_id,
                    extra={
                        'source_template': strand_template,
                        'original_index': bead.original_index,
                    },
                ))
                atom_indices.append(len(atoms) - 1)
            metadata = dict(chain.metadata or {})
            metadata['attachment_positions'] = strand_template.attachment_positions
            chains.append(ChainBlueprint(
                chain_type='backbone',
                chain_index=chain_idx,
                component_id=component_id,
                definition=component_entry.get('definition', component_entry),
                atom_indices=atom_indices,
                metadata=metadata,
            ))
            continue

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

        # Per-stub caps (computed before the body loop, which must withhold
        # them): a cap atom (an unreacted arm's thiol hydrogen, say) is a real
        # template body atom that exists only while its stub is unreacted.
        # Reacted stub positions come from the layout metadata (chosen with
        # the conversion RNG for partial builds; every position otherwise). A
        # reacted arm's cap atoms are simply never emitted, so every bonded
        # term referencing them drops out of the original-index map naturally.
        _definition_early = component_entry.get('definition', component_entry)
        stub_caps_def = _definition_early.get('stub_caps') or []
        caps_configured = any(spec for spec in stub_caps_def)
        _stub_count = len(_definition_early.get('stub_definitions', []) or [])
        reacted_positions = set(
            chain.metadata.get('reacted_stub_positions', range(_stub_count))
            if chain.metadata else range(_stub_count)
        )
        withheld_body_positions = set()
        body_overrides = {}
        if caps_configured:
            for position in reacted_positions:
                if position < len(stub_caps_def) and stub_caps_def[position]:
                    spec = stub_caps_def[position]
                    withheld_body_positions.update(spec.get('cap_body_positions', ()))
                    # Overrides may also retouch body atoms of the reacted arm
                    # (keyed by original 1-based ITP index).
                    body_overrides.update(spec.get('overrides') or {})

        for bead_idx, position in enumerate(chain.positions):
            if bead_idx in withheld_body_positions:
                continue
            params = _linker_atom_params(component_entry, bead_idx)
            template = chain.template
            original_index = None
            if template and bead_idx < len(getattr(template, "beads", [])):
                original_index = getattr(template.beads[bead_idx], "original_index", bead_idx + 1)
            override = body_overrides.get(original_index)
            if override:
                for key, param_key in (('charge', 'charge'), ('type', 'atom_type'), ('mass', 'mass')):
                    if key in override:
                        params[param_key] = override[key]
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

        stub_definitions = definition.get('stub_definitions', []) or []
        stub_stub_bonds = definition.get('stub_stub_bonds', []) or []
        if len(chain.positions) == 0 and len(stub_definitions) == 2:
            span_length = float(chain.metadata.get('span_length') or getattr(chain.template, 'span_length', 0.0) or 0.0)
            stub_positions = [
                anchor - axis_dir * (span_length / 2.0),
                anchor + axis_dir * (span_length / 2.0),
            ]
            raw_backbone_name = definition.get('backbone_name') or definition.get('backbone_residue_name') or 'BCK'

            def _target_for_stub(stub_type):
                """Admissible backbone target(s) for the named stub slot."""
                key = 'backbone_1_bonds' if stub_type == 'backbone_1' else 'backbone_2_bonds'
                rows = definition.get(key, []) or []
                if not rows:
                    return None
                targets = rows[0].get('between')
                if isinstance(targets, str):
                    return targets
                if isinstance(targets, (list, tuple)) and len(targets) == 1:
                    return targets[0]
                return None

            for stub_def_idx, (original_stub_def, stub_pos) in enumerate(zip(stub_definitions, stub_positions)):
                if isinstance(raw_backbone_name, list) and len(raw_backbone_name) > 0:
                    res_name = raw_backbone_name[stub_def_idx % len(raw_backbone_name)]
                else:
                    res_name = raw_backbone_name
                stub_type = 'backbone_1' if stub_def_idx == 0 else 'backbone_2'
                target_bb = _target_for_stub(stub_type)
                atoms.append(AtomBlueprint(
                    chain_type='linker',
                    chain_index=chain_idx,
                    bead_index=stub_def_idx,
                    position=np.array(stub_pos, dtype=np.float64),
                    component_id=component_id,
                    atom_name=original_stub_def.get('atom', f'BCK{stub_def_idx + 1}'),
                    atom_type=original_stub_def.get('type', 'SN3r'),
                    residue_name=res_name,
                    residue_number=original_stub_def.get('resnr', stub_def_idx + 1),
                    charge_group_number=original_stub_def.get('cgnr', stub_def_idx + 1),
                    mass=float(original_stub_def.get('mass', 45.0)),
                    charge=float(original_stub_def.get('charge', 0.0)),
                    backbone_type=None,
                    extra={
                        'is_terminal_backbone': True,
                        'stub_type': stub_type,
                        'target_backbone': target_bb,
                        'source_template': chain.template,
                        'original_index': original_stub_def.get('nr'),
                    }
                ))
                atom_indices.append(len(atoms) - 1)
        
        # One entry per stub. The two-stub spelling keeps its historical
        # stub_type names because the diamond runtime matches on them; a
        # junction of any other functionality gets indexed names.
        by_stub = definition.get('external_bonds_by_stub') or []
        if by_stub:
            if len(by_stub) == 2:
                names = ('backbone_1', 'backbone_2')
            else:
                names = tuple(f'stub_{i}' for i in range(len(by_stub)))
            stub_loops = [
                (group, names[i], i) for i, group in enumerate(by_stub)
            ]
        else:
            stub_loops = [
                (definition.get('external_bonds_1', []), 'backbone_1', 0),
                (definition.get('external_bonds_2', []), 'backbone_2', 1),
            ]

        arm_vectors = getattr(chain.template, 'arm_vectors', None)
        orientation = chain.metadata.get('orientation') if chain.metadata else None
        multi_arm = len(stub_loops) != 2


        for external_bonds, stub_type, stub_def_idx in stub_loops:
            stub_definitions = definition.get('stub_definitions', [])
            if not external_bonds or not stub_definitions or stub_def_idx >= len(stub_definitions):
                continue

            original_stub_def = stub_definitions[stub_def_idx]

            # One blueprint atom PER STUB. The loop used to emit one atom per
            # body-attachment row, which duplicated the stub atom whenever a
            # stub had two body bonds (an unreacted thiol sulfur: CH2 and H).
            # All attachment rows now ride along in 'stub_body_bonds' and the
            # populator creates each bond; the first row keeps the legacy
            # single-bond keys so older consumers see what they always saw.
            first_ext = external_bonds[0]
            bead_idx = int(first_ext.get('from_bead', 0))
            if bead_idx < 0 or bead_idx >= len(chain.positions):
                continue

            target_bb = first_ext.get('to_backbone')
            if target_bb == 'dummy_id':
                target_bb = None

            # Use original stub properties for naming, but backbone_residue_name for residue
            raw_backbone_name = definition.get('backbone_name') or definition.get('backbone_residue_name') or 'STUBRES'
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
            # Per-stub cap: a reacted arm applies its declared overrides
            # (typically the sulfur's charge/type in reacted form).
            stub_caps = definition.get('stub_caps') or []
            cap_spec = stub_caps[stub_def_idx] if stub_def_idx < len(stub_caps) else None
            cap_reacted = stub_def_idx in reacted_positions if caps_configured else True
            if cap_spec and cap_reacted:
                override = (cap_spec.get('overrides') or {}).get(
                    original_stub_def.get('nr'), {}
                )
                for key, param_key in (('charge', 'charge'), ('type', 'atom_type'), ('mass', 'mass')):
                    if key in override:
                        params[param_key] = override[key]

            if multi_arm and arm_vectors is not None and stub_def_idx < len(arm_vectors):
                # A multi-arm junction is anchored on its stub centroid,
                # so each stub sits at its template arm position; the
                # two-stub axis projection below has no meaning for it.
                arm = np.asarray(arm_vectors[stub_def_idx], dtype=np.float64)
                if orientation is not None:
                    arm = np.asarray(orientation, dtype=np.float64).reshape(3, 3) @ arm
                stub_pos = anchor + arm
            else:
                bead_pos = np.array(chain.positions[bead_idx], dtype=np.float64)
                proj = float(np.dot(bead_pos - anchor, axis_dir))
                sign = 1.0 if proj >= 0 else -1.0
                _proto_linker = getattr(layout_plan.proto_plan, 'proto_linker', None)
                ext_length = float(first_ext.get('length', _proto_linker.length if _proto_linker is not None else 0.0))
                stub_pos = bead_pos + axis_dir * ext_length * sign

            extra = {
                'stub_from_bead': bead_idx,
                'target_backbone': target_bb,
                'external_params': {k: v for k, v in first_ext.items() if k not in ('from_bead', 'to_backbone')},
                'stub_body_bonds': [
                    (int(ext.get('from_bead', 0)),
                     {k: v for k, v in ext.items() if k not in ('from_bead', 'to_backbone')})
                    for ext in external_bonds
                ],
                'is_terminal_backbone': True,
                'stub_type': stub_type,
                'cap_reacted': cap_reacted,
                'source_template': chain.template,
                'original_index': original_stub_def.get('nr')
            }

            atoms.append(AtomBlueprint(
                chain_type='linker',
                chain_index=chain_idx,
                bead_index=-(10 * (stub_def_idx + 1)), # unique negative index per stub
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
