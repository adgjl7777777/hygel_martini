"""
Linker template loader from GRO/ITP pairs.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import os
import numpy as np

from hygel_martini.hydrogel_builder.core_utils.io.gro_parser import read_gro_atoms
from hygel_martini.hydrogel_builder.core_utils.io.martini_parser import read_itp_definitions
from hygel_martini.hydrogel_builder.core_utils.templates.monomer_loader import BeadTemplate
from hygel_martini.hydrogel_builder.config_params.config import Config
from hygel_martini.hydrogel_builder.core_utils.common.collisions import (
    DuplicateDeclaration,
    require_consistent,
    require_unique,
)


@dataclass
class LinkerTemplate:
    id: str
    beads: List[BeadTemplate]
    coords: np.ndarray
    internal_bonds: List[Tuple[int, int, Dict[str, float]]]
    internal_angles: List[Dict]
    internal_dihedrals: List[Dict]
    internal_impropers: List[Dict]
    dihedrals_full: List[Dict]
    impropers_full: List[Dict]
    constraints: List[Dict]
    pairs: List[Dict]
    exclusions: List[Dict]
    virtual_sites: List[Dict]
    restraints: List[Dict]
    cmaptypes: List
    polarization: List[Dict]
    other_sections: Dict
    # --- general N-stub description -------------------------------------
    #: Bonds from each stub to the linker body, indexed by stub.
    stub_bonds: List[List[Tuple[int, Dict[str, float]]]]
    #: Backbone identifiers each stub may bond to, indexed by stub.
    stub_backbone_targets: Tuple[Tuple[str, ...], ...]
    #: Stub position relative to the template origin, in the local frame.
    arm_vectors: np.ndarray
    #: Junction functionality: how many backbone ends this template can take.
    functionality: int
    #: Per-stub configuration entries, indexed by stub.
    stub_config_bonds: List[List[Dict]]

    # --- two-stub view, populated only when functionality == 2 -----------
    # Kept so the diamond layout, which is written around a left/right pair,
    # continues to work unchanged. A consumer that needs it must check
    # functionality first rather than assume.
    stub_bonds_left: List[Tuple[int, Dict[str, float]]]
    stub_bonds_right: List[Tuple[int, Dict[str, float]]]
    backbone_ids: Tuple[str, ...]
    span_vector: np.ndarray
    span_length: float
    linker_name: str
    backbone_name: str
    stub_definitions: List[Dict]
    backbone_1_bonds: List[Dict]
    backbone_2_bonds: List[Dict]
    stub_stub_bonds: List[Dict]


@dataclass
class LinkerTemplateRecord:
    template: LinkerTemplate
    ratio: float


@dataclass
class LinkerTemplateLibrary:
    records: List[LinkerTemplateRecord]
    lookup: Dict[str, LinkerTemplate]


def linker_definitions_from_library(library: LinkerTemplateLibrary) -> List[Dict]:
    definitions: List[Dict] = []
    for record in library.records:
        template = record.template
        bead_defs = []
        for bead in template.beads:
            bead_defs.append({
                "name": bead.name,
                "type": bead.atom_type,
                "charge": bead.charge,
                "mass": bead.mass
            })
        bonds = []
        for i, j, params in template.internal_bonds:
            bonds.append({
                "from": i,
                "to": j,
                "funct": params.get("funct", 1),
                "length": params.get("c0"),
                "fc": params.get("c1")
            })
        if not template.beads and template.stub_stub_bonds:
            for params in template.stub_stub_bonds:
                bonds.append({
                    "from": 0,
                    "to": 1,
                    "funct": params.get("funct", 1),
                    "length": params.get("c0"),
                    "fc": params.get("c1"),
                })
        
        def _external(bonds):
            return [
                {
                    "from_bead": idx,
                    "to_backbone": params.get("target"),
                    "to_backbones": params.get("targets"),
                    "stub_index": params.get("stub_index"),
                    "funct": params.get("funct", 1),
                    "length": params.get("c0"),
                    "fc": params.get("c1"),
                }
                for idx, params in bonds
            ]

        external_bonds_by_stub = [_external(group) for group in template.stub_bonds]
        # 'external_bonds' predates this work and is read as a FLAT list by
        # three consumers (proto_builder, proto_layout, isotropic_builder) that
        # sum bond lengths from it. Reusing that key for a per-stub nested list
        # broke them at runtime, so the flat spelling keeps its old shape and
        # the nested one gets its own name.
        external_bonds = [bond for group in external_bonds_by_stub for bond in group]
        # The two-stub spelling stays populated for the diamond layout, which
        # reads external_bonds_1/_2 directly.
        external_bonds_1 = external_bonds_by_stub[0] if template.functionality == 2 else []
        external_bonds_2 = external_bonds_by_stub[1] if template.functionality == 2 else []

        definition = {
            "linker_name": template.linker_name,
            "backbone_name": template.backbone_name,
            "residue_name": template.linker_name,
            "residue_number": template.beads[0].residue_number if template.beads else 1,
            "charge_group_number": template.beads[0].cgnr if template.beads else 1,
            "beads": bead_defs,
            "bonds": bonds,
            "external_bonds": external_bonds,
            "external_bonds_by_stub": external_bonds_by_stub,
            "functionality": template.functionality,
            "arm_vectors": template.arm_vectors,
            "external_bonds_1": external_bonds_1,
            "external_bonds_2": external_bonds_2,
            "stub_definitions": template.stub_definitions,
            "backbone_1_bonds": template.backbone_1_bonds,
            "backbone_2_bonds": template.backbone_2_bonds,
            "stub_stub_bonds": template.stub_stub_bonds,
            "span_from_gro": template.span_length
        }
        # Carry over rich sections so index integrity can be preserved downstream
        definition.update({
            "constraints": template.constraints,
            "pairs": template.pairs,
            "exclusions": template.exclusions,
            "virtual_sites": template.virtual_sites,
            "restraints": template.restraints,
            "cmaptypes": template.cmaptypes,
            "polarization": template.polarization,
            "other_sections": template.other_sections,
        })
        definitions.append({
            "id": template.id,
            "ratio": record.ratio,
            "definition": definition
        })
    return definitions


def _extract_definition(itp_path: str, molecule_name: str | None) -> Dict:
    mass_map = Config.get_runtime('atom_type_masses', {})
    definitions = read_itp_definitions(
        itp_path,
        atom_type_masses=mass_map,
        prefer_explicit_masses=True,
    )
    if not definitions:
        raise ValueError(f"링커 ITP '{itp_path}'에서 moleculetype을 찾을 수 없습니다.")
    if molecule_name:
        try:
            return definitions[molecule_name]
        except KeyError as exc:
            raise ValueError(f"링커 ITP '{itp_path}'에 '{molecule_name}' 정의가 없습니다.") from exc
    if len(definitions) > 1:
        raise ValueError(
            f"링커 ITP '{itp_path}'에 여러 moleculetype이 있습니다. 'molecule_name'을 지정해 주세요."
        )
    return next(iter(definitions.values()))


def _map_backbone_ids(beads: List[Dict], backbone_defs: List[Dict]) -> Dict[int, str]:
    residue_to_backbone = {}
    for bb in backbone_defs:
        res_name = bb["definition"].get("residue_name")
        bb_id = bb["id"]
        if isinstance(res_name, list):
            for name in res_name:
                residue_to_backbone[name] = bb_id
        else:
            residue_to_backbone[res_name] = bb_id

    mapping = {}
    for bead in beads:
        resname = bead.get("residue")
        bb_id = residue_to_backbone.get(resname)
        if bb_id:
            mapping[bead["nr"]] = bb_id
    return mapping


def _convert_params(bond_def: Dict) -> Dict[str, float]:
    params = bond_def.get("params", [])
    length = params[0] if params else bond_def.get("length")
    fc = params[1] if len(params) > 1 else bond_def.get("fc")
    return {
        "funct": bond_def.get("funct", 1),
        "c0": length,
        "c1": fc,
    }


def _backbone_mass_lookup(backbone_defs: List[Dict]) -> Dict[str, float]:
    masses: Dict[str, float] = {}
    for backbone in backbone_defs:
        backbone_id = backbone.get("id")
        definition = backbone.get("definition", {}) or {}
        mass = definition.get("mass")
        if backbone_id is not None and mass is not None:
            masses[str(backbone_id)] = float(mass)
    return masses


def _stub_configuration(entry: Dict, linker_id: str) -> List[List[Dict]]:
    """Per-stub bond-parameter entries, in stub order.

    The general form is ``stubs: [[...], [...], ...]`` with one list per stub.
    ``backbone_1``/``backbone_2`` remain accepted as the two-stub spelling,
    which is what every existing configuration uses; mixing the two forms is
    refused rather than silently resolved to one of them.
    """
    stubs = entry.get("stubs")
    legacy = [key for key in ("backbone_1", "backbone_2") if entry.get(key)]

    if stubs is not None and legacy:
        raise ValueError(
            f"링커 '{linker_id}'가 'stubs'와 {legacy}를 동시에 정의했습니다. "
            "둘 중 하나만 사용하십시오."
        )
    if stubs is not None:
        if not isinstance(stubs, list) or not stubs:
            raise ValueError(
                f"링커 '{linker_id}'의 'stubs'는 stub마다 하나씩의 리스트여야 합니다."
            )
        resolved = []
        for position, group in enumerate(stubs):
            if isinstance(group, dict):
                group = [group]
            if not isinstance(group, list) or not group:
                raise ValueError(
                    f"링커 '{linker_id}'의 stub {position}에 bond entry가 없습니다."
                )
            resolved.append(group)
        return resolved

    return [entry.get("backbone_1", []), entry.get("backbone_2", [])]


def _resolve_stub_targets(
    backbone_bonds: List[Dict],
    linker_id: str,
    side_name: str,
) -> Tuple[str, ...]:
    """Backbone identifiers a stub may bond to.

    The configuration lists one entry per admissible partner backbone, each
    carrying its own bond parameters, so a stub that can reach either of two
    backbone chemistries legitimately declares two entries.  Requiring exactly
    one target here rejected that, which made the tracked ``04_full_builder``
    example fail to load.  Only an empty declaration is an error.
    """
    targets = sorted(
        {
            str(bond.get("between"))
            for bond in backbone_bonds
            if bond.get("between") is not None
        }
    )
    if not targets:
        raise ValueError(
            f"링커 '{linker_id}'의 {side_name}에 backbone target이 하나도 없습니다. "
            f"'between' 값을 가진 entry가 최소 하나 필요합니다."
        )
    return tuple(targets)


def _stub_mass_for_targets(
    targets: Tuple[str, ...],
    backbone_masses: Dict[str, float],
    linker_id: str,
    side_name: str,
) -> float:
    """Mass of a stub that stands in for one of ``targets``.

    A stub bead is a placeholder for the backbone end it will bond to, so its
    mass comes from that backbone.  When several partners are admissible the
    mass is only well defined if they agree; disagreeing masses are refused
    rather than silently resolved to whichever target happens to sort first.
    """
    try:
        masses = {target: float(backbone_masses[target]) for target in targets}
    except KeyError as exc:
        raise ValueError(
            f"링커 '{linker_id}'의 {side_name} target '{exc.args[0]}'을 "
            f"backbone definition에서 찾을 수 없습니다."
        ) from exc
    distinct = set(masses.values())
    if len(distinct) > 1:
        raise ValueError(
            f"링커 '{linker_id}'의 {side_name}는 질량이 서로 다른 backbone "
            f"{masses}에 결합할 수 있습니다. stub 질량이 결정되지 않으므로 "
            f"stub을 분리하거나 backbone 질량을 맞추십시오."
        )
    return next(iter(distinct))


def _orthonormal_basis(span_vec: np.ndarray, ref: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    span_norm = np.linalg.norm(span_vec)
    if span_norm < 1e-8:
        raise ValueError("링커 stub 간 벡터의 길이가 0입니다.")
    x_axis = span_vec / span_norm
    ref_vec = ref.copy()
    if abs(np.dot(x_axis, ref_vec)) > 0.9:
        ref_vec = np.array([0.0, 1.0, 0.0])
    y_axis = ref_vec - np.dot(ref_vec, x_axis) * x_axis
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8:
        y_axis = np.array([0.0, 1.0, 0.0]) - x_axis * x_axis[1]
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-8:
            raise ValueError("링커 좌표에서 직교 기준을 찾을 수 없습니다.")
    y_axis /= y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        raise ValueError("링커 좌표에서 직교 기준을 찾을 수 없습니다.")
    z_axis /= z_norm
    return np.column_stack((x_axis, y_axis, z_axis))


def _load_single_linker(entry: Dict, backbone_defs: List[Dict]) -> LinkerTemplate:
    linker_id = entry.get("id")
    gro_path = entry.get("gro")
    itp_path = entry.get("itp")
    if not linker_id:
        raise ValueError("각 링커는 고유한 'id'가 필요합니다.")
    if not gro_path or not os.path.isfile(gro_path):
        raise FileNotFoundError(f"링커 '{linker_id}'의 GRO 파일을 찾을 수 없습니다: {gro_path}")
    if not itp_path or not os.path.isfile(itp_path):
        raise FileNotFoundError(f"링커 '{linker_id}'의 ITP 파일을 찾을 수 없습니다: {itp_path}")

    molecule_name = entry.get("molecule_name")
    definition = _extract_definition(itp_path, molecule_name)
    beads = definition.get("beads", [])
    if not beads:
        raise ValueError(f"링커 '{linker_id}'의 ITP에 bead 정보가 없습니다.")

    # Use new keys from user's JSON structure
    linker_name = entry.get("linker_residue_name")
    backbone_name = entry.get("backbone_residue_name")
    stub_config = _stub_configuration(entry, linker_id)
    backbone_1_bonds = stub_config[0] if len(stub_config) > 0 else []
    backbone_2_bonds = stub_config[1] if len(stub_config) == 2 else []

    if not linker_name or not backbone_name:
        raise ValueError(f"링커 '{linker_id}'는 maker.json에 'linker_residue_name'과 'backbone_residue_name'이 필요합니다.")

    stub_indices = sorted([bead['nr'] for bead in beads if bead['residue'] == 'BCK'])
    functionality = len(stub_indices)
    if functionality != len(stub_config):
        raise ValueError(
            f"링커 '{linker_id}' 템플릿의 'BCK' stub 원자는 {functionality}개인데 "
            f"설정은 {len(stub_config)}개 stub을 정의했습니다. 두 수가 같아야 합니다. "
            f"(f>2 junction은 'stubs' 리스트를 사용하십시오.)"
        )
    if functionality < 2:
        raise ValueError(
            f"링커 '{linker_id}' 템플릿에 'BCK' stub 원자가 {functionality}개뿐입니다. "
            "crosslinker는 최소 두 개의 backbone end를 이어야 합니다."
        )

    backbone_masses = _backbone_mass_lookup(backbone_defs)
    stub_targets: List[Tuple[str, ...]] = []
    stub_masses: Dict[int, float] = {}
    for position, (idx, group) in enumerate(zip(stub_indices, stub_config)):
        label = f"stub {position}" if len(stub_config) != 2 else f"backbone_{position + 1}"
        targets = _resolve_stub_targets(group, linker_id, label)
        stub_targets.append(targets)
        stub_masses[idx] = _stub_mass_for_targets(
            targets, backbone_masses, linker_id, label
        )
    is_pair = functionality == 2
    left_idx = stub_indices[0]
    right_idx = stub_indices[-1]
    left_backbone_id = stub_targets[0][0]
    right_backbone_id = stub_targets[-1][0]

    stub_definitions = []
    # Perform dynamic renaming and collect stub definitions
    definition['name'] = linker_name
    for bead in beads:
        if bead['nr'] in stub_indices:
            bead['mass'] = stub_masses[bead['nr']]
            stub_definitions.append(bead.copy()) # Save original stub definition
        if bead['residue'] == 'BCK':
            if isinstance(backbone_name, (list, tuple)):
                position = stub_indices.index(bead['nr'])
                if position >= len(backbone_name):
                    raise ValueError(
                        f"링커 '{linker_id}'의 backbone_residue_name은 이름 "
                        f"{len(backbone_name)}개를 주었지만 stub은 {functionality}개입니다."
                    )
                bead['residue'] = backbone_name[position]
            else:
                bead['residue'] = backbone_name
        elif bead['residue'] == 'LNK':
            bead['residue'] = linker_name
    
    # Ensure stub_definitions are sorted by index
    stub_definitions.sort(key=lambda x: x['nr'])

    backbone_map = {
        idx: targets[0] for idx, targets in zip(stub_indices, stub_targets)
    }
    stub_position = {idx: position for position, idx in enumerate(stub_indices)}

    gro_atoms = read_gro_atoms(gro_path)
    if len(gro_atoms) != len(beads):
        raise ValueError(f"링커 '{linker_id}'의 GRO/ITP 원자 수가 일치하지 않습니다.")

    stub_positions = np.array(
        [gro_atoms[idx - 1].position for idx in stub_indices], dtype=np.float64
    )

    if is_pair:
        # Two stubs define an axis, and the diamond layout is written around
        # it: origin at the first stub, x along the span. Kept exactly.
        left_pos = stub_positions[0]
        span_vector = stub_positions[1] - stub_positions[0]
        span_length = float(np.linalg.norm(span_vector))
        if span_length < 1e-8:
            raise ValueError(f"링커 '{linker_id}'의 backbone 간 거리가 0입니다.")
        basis = _orthonormal_basis(span_vector)
    else:
        # A junction with three or more arms has no distinguished axis, so the
        # origin is the stub centroid and the template keeps its own
        # orientation; the layout rotates it into place. At two stubs the
        # centroid is the midpoint and 2 * mean arm length is exactly the
        # stub-to-stub span, so the two branches agree in the limit.
        left_pos = stub_positions.mean(axis=0)
        arm_lengths = np.linalg.norm(stub_positions - left_pos, axis=1)
        if float(arm_lengths.min()) < 1e-8:
            raise ValueError(
                f"링커 '{linker_id}'의 stub 하나가 stub 중심과 겹칩니다; "
                "arm 방향을 정의할 수 없습니다."
            )
        span_length = float(2.0 * arm_lengths.mean())
        span_vector = np.zeros(3, dtype=np.float64)
        basis = np.eye(3, dtype=np.float64)

    arm_vectors = np.array(
        [basis.T @ (position - left_pos) for position in stub_positions],
        dtype=np.float64,
    )

    bead_templates: List[BeadTemplate] = []
    coords_local: List[np.ndarray] = []
    index_map: Dict[int, int] = {}
    total_mass = 0.0

    for bead in beads:
        idx = bead['nr']
        atom = gro_atoms[idx - 1]
        if idx in backbone_map:
            continue
        rel = atom.position - left_pos
        coord_local = basis.T @ rel
        coords_local.append(coord_local)
        total_mass += bead.get('mass', 0.0)
        index_map[idx] = len(bead_templates)
        bead_templates.append(
            BeadTemplate(
                name=atom.atom_name,
                atom_type=bead.get('type'),
                residue_name=bead.get('residue'),
                residue_number=bead.get('resnr', 1),
                original_index=idx,
                cgnr=bead.get('cgnr', 1),
                charge=bead.get('charge', 0.0),
                mass=bead.get('mass', 0.0),
                coord=coord_local,
            )
        )

    coords = np.array(coords_local, dtype=np.float64).reshape((-1, 3))
    constraints = definition.get("constraints", [])
    pairs = definition.get("pairs", [])
    exclusions = definition.get("exclusions", [])
    virtual_sites = definition.get("virtual_sites", [])
    restraints = definition.get("restraints", [])
    cmaptypes = definition.get("cmaptypes", [])
    polarization = definition.get("polarization", [])
    other_sections = definition.get("other_sections", {})

    internal_bonds: List[Tuple[int, int, Dict[str, float]]] = []
    stub_bonds: List[List[Tuple[int, Dict[str, float]]]] = [
        [] for _ in stub_indices
    ]
    stub_stub_bonds: List[Dict] = []

    for bond in definition.get('bonds', []):
        i = bond.get('from')
        j = bond.get('to')
        if i is None or j is None:
            continue
        params = _convert_params(bond)
        if i in backbone_map and j in backbone_map:
            stub_stub_bonds.append({
                "funct": params.get("funct", 1),
                "c0": params.get("c0"),
                "c1": params.get("c1"),
            })
            continue
        if i in backbone_map or j in backbone_map:
            other = j if i in backbone_map else i
            target = backbone_map[i] if i in backbone_map else backbone_map[j]
            other_idx = index_map.get(other)
            if other_idx is None:
                raise ValueError(f"링커 '{linker_id}'의 backbone 결합이 잘못된 bead를 참조합니다.")
            stub_atom = i if i in backbone_map else j
            position = stub_position[stub_atom]
            stub_bonds[position].append(
                (
                    other_idx,
                    {
                        "target": target,
                        "targets": stub_targets[position],
                        "stub_index": position,
                        **params,
                    },
                )
            )
        else:
            idx_i = index_map.get(i)
            idx_j = index_map.get(j)
            if idx_i is None or idx_j is None:
                continue
            internal_bonds.append((idx_i, idx_j, params))

    internal_angles = []
    for angle_def in definition.get("angles", []):
        atoms = [angle_def.get('from'), angle_def.get('center'), angle_def.get('to')]
        if any(a in backbone_map for a in atoms) or any(a is None for a in atoms):
            continue
        new_angle = {
            'from': index_map.get(atoms[0]),
            'center': index_map.get(atoms[1]),
            'to': index_map.get(atoms[2]),
            'funct': angle_def['funct'],
            'params': angle_def['params']
        }
        if any(v is None for v in new_angle.values()):
            continue
        internal_angles.append(new_angle)

    internal_dihedrals = []
    for dihedral_def in definition.get("dihedrals", []):
        atoms = [dihedral_def.get('i'), dihedral_def.get('j'), dihedral_def.get('k'), dihedral_def.get('l')]
        if any(a in backbone_map for a in atoms) or any(a is None for a in atoms):
            continue
        new_dihedral = {
            'i': index_map.get(atoms[0]),
            'j': index_map.get(atoms[1]),
            'k': index_map.get(atoms[2]),
            'l': index_map.get(atoms[3]),
            'funct': dihedral_def['funct'],
            'params': dihedral_def['params']
        }
        if any(v is None for v in new_dihedral.values()):
            continue
        internal_dihedrals.append(new_dihedral)

    internal_impropers = []
    for imp_def in definition.get("impropers", []):
        atoms = [imp_def.get('i'), imp_def.get('j'), imp_def.get('k'), imp_def.get('l')]
        if any(a in backbone_map for a in atoms) or any(a is None for a in atoms):
            continue
        new_imp = {
            'i': index_map.get(atoms[0]),
            'j': index_map.get(atoms[1]),
            'k': index_map.get(atoms[2]),
            'l': index_map.get(atoms[3]),
            'funct': imp_def['funct'],
            'params': imp_def['params']
        }
        if any(v is None for v in new_imp.values()):
            continue
        internal_impropers.append(new_imp)

    backbone_ids = tuple(targets[0] for targets in stub_targets)
    stub_bonds_left = stub_bonds[0] if is_pair else []
    stub_bonds_right = stub_bonds[1] if is_pair else []
    return LinkerTemplate(
        id=linker_id,
        beads=bead_templates,
        coords=coords,
        internal_bonds=internal_bonds,
        internal_angles=internal_angles,
        internal_dihedrals=internal_dihedrals,
        internal_impropers=internal_impropers,
        dihedrals_full=definition.get("dihedrals", []),
        impropers_full=definition.get("impropers", []),
        constraints=constraints,
        pairs=pairs,
        exclusions=exclusions,
        virtual_sites=virtual_sites,
        restraints=restraints,
        cmaptypes=cmaptypes,
        polarization=polarization,
        other_sections=other_sections,
        stub_bonds=stub_bonds,
        stub_backbone_targets=tuple(stub_targets),
        arm_vectors=arm_vectors,
        functionality=functionality,
        stub_config_bonds=stub_config,
        stub_bonds_left=stub_bonds_left,
        stub_bonds_right=stub_bonds_right,
        backbone_ids=backbone_ids,
        span_vector=basis[:, 0] * span_length,
        span_length=span_length,
        linker_name=linker_name,
        backbone_name=backbone_name,
        stub_definitions=stub_definitions,
        backbone_1_bonds=backbone_1_bonds,
        backbone_2_bonds=backbone_2_bonds,
        stub_stub_bonds=stub_stub_bonds,
    )


def load_linker_templates(linker_entries: List[Dict], backbone_defs: List[Dict]) -> LinkerTemplateLibrary:
    records: List[LinkerTemplateRecord] = []
    lookup: Dict[str, LinkerTemplate] = {}
    for entry in linker_entries:
        template = _load_single_linker(entry, backbone_defs)
        record = LinkerTemplateRecord(template=template, ratio=entry.get('ratio', 1.0))
        records.append(record)
        if template.id in lookup:
            raise DuplicateDeclaration(
                f"Duplicate linker id {template.id!r} in LINKERS; one of the "
                "definitions would be discarded silently."
            )
        lookup[template.id] = template
    return LinkerTemplateLibrary(records=records, lookup=lookup)
