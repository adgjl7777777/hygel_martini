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
    stub_bonds_left: List[Tuple[int, Dict[str, float]]]
    stub_bonds_right: List[Tuple[int, Dict[str, float]]]
    backbone_ids: Tuple[str, str]
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
        
        external_bonds_1 = []
        for idx, params in template.stub_bonds_left:
            external_bonds_1.append({
                "from_bead": idx,
                "to_backbone": params.get("target"),
                "funct": params.get("funct", 1),
                "length": params.get("c0"),
                "fc": params.get("c1")
            })
        external_bonds_2 = []
        for idx, params in template.stub_bonds_right:
            external_bonds_2.append({
                "from_bead": idx,
                "to_backbone": params.get("target"),
                "funct": params.get("funct", 1),
                "length": params.get("c0"),
                "fc": params.get("c1")
            })

        definition = {
            "linker_name": template.linker_name,
            "backbone_name": template.backbone_name,
            "residue_name": template.linker_name,
            "residue_number": template.beads[0].residue_number if template.beads else 1,
            "charge_group_number": template.beads[0].cgnr if template.beads else 1,
            "beads": bead_defs,
            "bonds": bonds,
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


def _resolve_stub_target(backbone_bonds: List[Dict], linker_id: str, side_name: str) -> str:
    targets = {
        str(bond.get("between"))
        for bond in backbone_bonds
        if bond.get("between") is not None
    }
    if len(targets) != 1:
        raise ValueError(
            f"링커 '{linker_id}'의 {side_name}에는 정확히 하나의 backbone target이 필요합니다: "
            f"{sorted(targets)}"
        )
    return next(iter(targets))


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
    backbone_1_bonds = entry.get("backbone_1", [])
    backbone_2_bonds = entry.get("backbone_2", [])

    if not linker_name or not backbone_name:
        raise ValueError(f"링커 '{linker_id}'는 maker.json에 'linker_residue_name'과 'backbone_residue_name'이 필요합니다.")

    stub_indices = sorted([bead['nr'] for bead in beads if bead['residue'] == 'BCK'])
    if len(stub_indices) != 2:
        raise ValueError(f"링커 '{linker_id}' 템플릿에는 stub 지점을 위해 정확히 2개의 'BCK' 잔기 원자가 있어야 합니다.")
    left_idx, right_idx = stub_indices
    backbone_masses = _backbone_mass_lookup(backbone_defs)
    left_backbone_id = _resolve_stub_target(backbone_1_bonds, linker_id, "backbone_1")
    right_backbone_id = _resolve_stub_target(backbone_2_bonds, linker_id, "backbone_2")
    try:
        stub_masses = {
            left_idx: backbone_masses[left_backbone_id],
            right_idx: backbone_masses[right_backbone_id],
        }
    except KeyError as exc:
        raise ValueError(
            f"링커 '{linker_id}'의 BCK stub mass를 backbone definition에서 찾을 수 없습니다: "
            f"{exc.args[0]}"
        ) from exc

    stub_definitions = []
    # Perform dynamic renaming and collect stub definitions
    definition['name'] = linker_name
    for bead in beads:
        if bead['nr'] in stub_indices:
            bead['mass'] = stub_masses[bead['nr']]
            stub_definitions.append(bead.copy()) # Save original stub definition
        if bead['residue'] == 'BCK':
            if isinstance(backbone_name, (list, tuple)):
                bead['residue'] = backbone_name[0] if bead['nr'] == left_idx else backbone_name[1]
            else:
                bead['residue'] = backbone_name
        elif bead['residue'] == 'LNK':
            bead['residue'] = linker_name
    
    # Ensure stub_definitions are sorted by index
    stub_definitions.sort(key=lambda x: x['nr'])

    backbone_map = {
        left_idx: left_backbone_id,
        right_idx: right_backbone_id,
    }

    gro_atoms = read_gro_atoms(gro_path)
    if len(gro_atoms) != len(beads):
        raise ValueError(f"링커 '{linker_id}'의 GRO/ITP 원자 수가 일치하지 않습니다.")

    left_atom = gro_atoms[left_idx - 1]
    right_atom = gro_atoms[right_idx - 1]
    left_pos = left_atom.position
    right_pos = right_atom.position

    span_vector = right_pos - left_pos
    span_length = np.linalg.norm(span_vector)
    if span_length < 1e-8:
        raise ValueError(f"링커 '{linker_id}'의 backbone 간 거리가 0입니다.")

    basis = _orthonormal_basis(span_vector)

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
    stub_bonds_left: List[Tuple[int, Dict[str, float]]] = []
    stub_bonds_right: List[Tuple[int, Dict[str, float]]] = []
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
            if (i in backbone_map and i == left_idx) or (j in backbone_map and j == left_idx):
                stub_bonds_left.append((other_idx, {"target": target, **params}))
            else:
                stub_bonds_right.append((other_idx, {"target": target, **params}))
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

    backbone_ids = (backbone_map[left_idx], backbone_map[right_idx])
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
        lookup[template.id] = template
    return LinkerTemplateLibrary(records=records, lookup=lookup)
