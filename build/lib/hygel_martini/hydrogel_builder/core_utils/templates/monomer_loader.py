"""
Helpers to ingest monomer templates from GRO/ITP pairs.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import os
import numpy as np

from hygel_martini.hydrogel_builder.core_utils.io.gro_parser import read_gro_atoms
from hygel_martini.hydrogel_builder.core_utils.io.martini_parser import read_itp_definitions
from hygel_martini.hydrogel_builder.config_params.config import Config


@dataclass
class BeadTemplate:
    name: str
    atom_type: str
    residue_name: str
    residue_number: int
    original_index: int
    cgnr: int
    charge: float
    mass: float
    coord: np.ndarray


@dataclass
class MonomerTemplate:
    id: str
    backbone_id: str
    backbone_original_index: int
    beads: List[BeadTemplate]
    coords: np.ndarray
    internal_bonds: List[Tuple[int, int, Dict[str, float]]]
    internal_angles: List[Dict]
    internal_dihedrals: List[Dict]
    internal_impropers: List[Dict]
    dihedrals_full: List[Dict]
    impropers_full: List[Dict]
    backbone_bonds: List[Tuple[int, Dict[str, float]]]
    constraints: List[Dict]
    pairs: List[Dict]
    exclusions: List[Dict]
    virtual_sites: List[Dict]
    restraints: List[Dict]
    cmaptypes: List
    polarization: List[Dict]
    other_sections: Dict
    total_mass: float


@dataclass
class TemplateRecord:
    template: MonomerTemplate
    ratio: float


@dataclass
class MonomerTemplateLibrary:
    records: List[TemplateRecord]
    by_backbone: Dict[str, List[TemplateRecord]]
    lookup: Dict[str, MonomerTemplate]


def _extract_single_definition(itp_path: str, molecule_name: str | None) -> Dict:
    mass_map = Config.get_runtime('atom_type_masses', {})
    definitions = read_itp_definitions(
        itp_path,
        atom_type_masses=mass_map,
        prefer_explicit_masses=True,
    )
    if not definitions:
        raise ValueError(f"ITP '{itp_path}'에는 유효한 [ moleculetype ] 정의가 없습니다.")
    if molecule_name:
        try:
            return definitions[molecule_name]
        except KeyError as exc:
            raise ValueError(f"ITP '{itp_path}'에 '{molecule_name}' 정의가 없습니다.") from exc
    if len(definitions) > 1:
        raise ValueError(
            f"ITP '{itp_path}'에 여러 moleculetype이 있습니다. 'molecule_name'을 지정해 주세요."
        )
    return next(iter(definitions.values()))


def _find_backbone_bead_index(beads: List[Dict]) -> int | None:
    for bead in beads:
        residu = (bead.get("residue") or "").upper()
        atom = (bead.get("atom") or "").upper()
        if residu.startswith("BCK") or atom.startswith("BCK"):
            return bead["nr"]
    return None


def _match_backbone(beads: List[Dict],
                    backbone_defs: List[Dict],
                    override_id: str | None = None) -> Tuple[int, str]:
    id_to_residue = {
        bb["id"]: bb["definition"].get("residue_name") for bb in backbone_defs
    }
    
    residue_to_backbone = {}
    for bb in backbone_defs:
        res_name = bb["definition"].get("residue_name")
        bb_id = bb["id"]
        if isinstance(res_name, list):
            for name in res_name:
                residue_to_backbone[name] = bb_id
        else:
            residue_to_backbone[res_name] = bb_id

    if override_id:
        residue_name = id_to_residue.get(override_id)
        if not residue_name:
            raise ValueError(f"Monomer에 정의된 backbone_id '{override_id}'가 BACKBONES 목록에 없습니다.")
        for bead in beads:
            if bead.get("residue") == residue_name:
                return bead["nr"], override_id
        fallback_idx = _find_backbone_bead_index(beads)
        if fallback_idx is not None:
            return fallback_idx, override_id
        raise ValueError(f"백본 ID '{override_id}'에 해당하는 residue '{residue_name}' bead를 ITP에서 찾을 수 없습니다.")

    for bead in beads:
        backbone_id = residue_to_backbone.get(bead.get("residue"))
        if backbone_id:
            return bead["nr"], backbone_id
    fallback_idx = _find_backbone_bead_index(beads)
    if fallback_idx is not None:
        default_id = next(iter(id_to_residue.keys()), None)
        if default_id:
            return fallback_idx, default_id
    raise ValueError("BCK(bead with backbone residue) 정보를 찾을 수 없습니다.")


def _convert_params(bond_def: Dict) -> Dict[str, float]:
    params = bond_def.get("params", [])
    length = params[0] if params else bond_def.get("length")
    fc = params[1] if len(params) > 1 else bond_def.get("fc")
    return {
        "funct": bond_def.get("funct", 1),
        "c0": length,
        "c1": fc,
    }


def _load_single(entry: Dict, backbone_defs: List[Dict]) -> MonomerTemplate:
    monomer_id = entry.get("id")
    if not monomer_id:
        raise ValueError("각 Monomer 항목에는 고유한 'id'가 반드시 필요합니다.")
    gro_path = entry.get("gro")
    itp_path = entry.get("itp")
    if not gro_path or not itp_path:
        raise ValueError(f"Monomer '{monomer_id}'에 'gro'와 'itp' 경로가 필요합니다.")
    if not os.path.isfile(gro_path):
        raise FileNotFoundError(f"GRO 파일을 찾을 수 없습니다: {gro_path}")
    if not os.path.isfile(itp_path):
        raise FileNotFoundError(f"ITP 파일을 찾을 수 없습니다: {itp_path}")

    molecule_name = entry.get("molecule_name")
    definition = _extract_single_definition(itp_path, molecule_name)
    beads = definition.get("beads", [])
    if not beads:
        raise ValueError(f"ITP '{itp_path}'에 bead 정보가 없습니다.")

    requested_backbone = entry.get("backbone_id")
    bck_index, backbone_id = _match_backbone(beads, backbone_defs, override_id=requested_backbone)
    backbone_map = {bck_index: backbone_id}
    gro_atoms = read_gro_atoms(gro_path)
    if len(gro_atoms) != len(beads):
        raise ValueError(
            f"GRO({gro_path}) 원자 수와 ITP({itp_path}) bead 수가 일치하지 않습니다."
        )

    bck_coord = gro_atoms[bck_index - 1].position
    bead_map: Dict[int, int] = {}
    template_beads: List[BeadTemplate] = []
    coords_list: List[np.ndarray] = []
    total_mass = 0.0

    for bead in beads:
        idx = bead["nr"]
        if idx == bck_index:
            continue
        atom_entry = gro_atoms[idx - 1]
        coord = atom_entry.position - bck_coord
        mass = bead.get("mass", 0.0)
        total_mass += mass
        bead_map[idx] = len(template_beads)
        coords_list.append(coord)
        template_beads.append(
            BeadTemplate(
                name=bead.get("atom"),
                atom_type=bead.get("type"),
                residue_name=bead.get("residue"),
                residue_number=bead.get("resnr", 1),
                original_index=idx,
                cgnr=bead.get("cgnr", 1),
                charge=bead.get("charge", 0.0),
                mass=mass,
                coord=coord,
            )
        )

    coords = np.array(coords_list, dtype=np.float64)
    internal_bonds: List[Tuple[int, int, Dict[str, float]]] = []
    backbone_bonds: List[Tuple[int, Dict[str, float]]] = []
    constraints = definition.get("constraints", [])
    pairs = definition.get("pairs", [])
    exclusions = definition.get("exclusions", [])
    virtual_sites = definition.get("virtual_sites", [])
    restraints = definition.get("restraints", [])
    cmaptypes = definition.get("cmaptypes", [])
    polarization = definition.get("polarization", [])
    other_sections = definition.get("other_sections", {})
    dihedrals_full = definition.get("dihedrals", [])
    impropers_full = definition.get("impropers", [])

    for bond_def in definition.get("bonds", []):
        i = bond_def.get("from")
        j = bond_def.get("to")
        if i is None or j is None:
            continue
        if i == bck_index and j == bck_index:
            continue
        params = _convert_params(bond_def)
        if i == bck_index or j == bck_index:
            other = j if i == bck_index else i
            other_idx = bead_map.get(other)
            if other_idx is None:
                raise ValueError("Backbone 결합이 side bead를 참조하지 않습니다.")
            params = {"target": backbone_id, **params}
            backbone_bonds.append((other_idx, params))
        else:
            idx_i = bead_map.get(i)
            idx_j = bead_map.get(j)
            if idx_i is None or idx_j is None:
                continue
            internal_bonds.append((idx_i, idx_j, params))

    # Process angles, dihedrals, etc. using the bead_map
    internal_angles = []
    for angle_def in definition.get("angles", []):
        atoms = [angle_def.get('from'), angle_def.get('center'), angle_def.get('to')]
        if any(a in backbone_map for a in atoms) or any(a is None for a in atoms):
            continue
        new_angle = {
            'from': bead_map.get(atoms[0]),
            'center': bead_map.get(atoms[1]),
            'to': bead_map.get(atoms[2]),
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
            'i': bead_map.get(atoms[0]),
            'j': bead_map.get(atoms[1]),
            'k': bead_map.get(atoms[2]),
            'l': bead_map.get(atoms[3]),
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
            'i': bead_map.get(atoms[0]),
            'j': bead_map.get(atoms[1]),
            'k': bead_map.get(atoms[2]),
            'l': bead_map.get(atoms[3]),
            'funct': imp_def['funct'],
            'params': imp_def['params']
        }
        if any(v is None for v in new_imp.values()):
            continue
        internal_impropers.append(new_imp)

    template = MonomerTemplate(
        id=monomer_id,
        backbone_id=backbone_id,
        backbone_original_index=bck_index,
        beads=template_beads,
        coords=coords,
        internal_bonds=internal_bonds,
        internal_angles=internal_angles,
        internal_dihedrals=internal_dihedrals,
        internal_impropers=internal_impropers,
        dihedrals_full=dihedrals_full,
        impropers_full=impropers_full,
        backbone_bonds=backbone_bonds,
        constraints=constraints,
        pairs=pairs,
        exclusions=exclusions,
        virtual_sites=virtual_sites,
        restraints=restraints,
        cmaptypes=cmaptypes,
        polarization=polarization,
        other_sections=other_sections,
        total_mass=total_mass,
    )

    return template


def load_monomer_templates(monomer_entries: List[Dict], backbone_defs: List[Dict]) -> MonomerTemplateLibrary:
    records: List[TemplateRecord] = []
    by_backbone: Dict[str, List[TemplateRecord]] = {}
    lookup: Dict[str, MonomerTemplate] = {}

    for entry in monomer_entries:
        ratio = entry.get("ratio", 1.0)
        template = _load_single(entry, backbone_defs)
        record = TemplateRecord(template=template, ratio=ratio)
        records.append(record)
        by_backbone.setdefault(template.backbone_id, []).append(record)
        lookup[template.id] = template

    return MonomerTemplateLibrary(records=records, by_backbone=by_backbone, lookup=lookup)
