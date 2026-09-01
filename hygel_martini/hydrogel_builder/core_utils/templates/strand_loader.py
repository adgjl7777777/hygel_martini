"""Whole-strand templates: one molecule per network strand.

The Martini path polymerizes a strand bead by bead, which presumes one
backbone bead per repeat unit. An all-atom strand is not built here at all --
a defined prepolymer such as PPG-TDI is a single molecule -- so the strand is
ingested whole from a GRO/ITP pair and placed rigidly between its two
junction sites.

The user marks exactly two attachment atoms by *atom name* ``BCK1`` and
``BCK2`` (the residue names stay chemical). These are real atoms -- for a
thiourethane strand, the two carbonyl carbons that bond to the crosslinker
sulfurs -- not placeholder beads, so nothing is stripped or renamed: every
atom, bond, angle, dihedral, pair and exclusion of the template survives into
the combined topology.

Rigidity is the load-bearing assumption. A molecule has one end-to-end
length, so it can only span junction gaps of essentially that length: the
layout refuses rewiring (which makes gap lengths heterogeneous) and warns
when the gap it must span disagrees with the template span. Absorbing a large
mismatch through the crosslink bonds would rebuild the silent pre-strain of
defect #21 on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import os

import numpy as np

from hygel_martini.hydrogel_builder.core_utils.io.gro_parser import read_gro_atoms
from hygel_martini.hydrogel_builder.core_utils.templates.monomer_loader import (
    BeadTemplate,
    _extract_single_definition,
)

__all__ = ["StrandTemplate", "load_strand_template"]


@dataclass
class StrandTemplate:
    id: str
    #: Every atom of the molecule, in ITP order. Nothing is skipped.
    beads: List[BeadTemplate]
    #: Local coordinates (nm), centered on the attachment midpoint.
    coords: np.ndarray
    #: 0-based bead positions of the two attachment atoms (head, tail).
    attachment_positions: tuple
    #: Unit vector head -> tail in the local frame.
    attachment_axis: np.ndarray
    #: Head-to-tail distance (nm): the gap length this strand can span.
    span_length: float
    total_mass: float
    #: [ bonds ] rows exactly as parsed (1-based 'from'/'to'), including the
    #: attachment atoms' own internal bonds.
    internal_bonds: List[Dict]
    #: Angle rows re-indexed to 0-based bead positions, in the shape
    #: construct_angles matches against source_index.
    internal_angles: List[Dict]
    #: Raw parsed rows (1-based indices) consumed via the populator's
    #: original-index map, like a linker template's.
    dihedrals_full: List[Dict]
    impropers_full: List[Dict]
    pairs: List[Dict]
    exclusions: List[Dict]
    constraints: List[Dict]
    virtual_sites: List[Dict]
    restraints: List[Dict]
    cmaptypes: List
    polarization: List[Dict]
    other_sections: Dict


def load_strand_template(entry: Dict) -> StrandTemplate:
    strand_id = entry.get("id")
    if not strand_id:
        raise ValueError("strand template 항목에는 'id'가 필요합니다.")
    gro_path = entry.get("gro")
    itp_path = entry.get("itp")
    if not gro_path or not itp_path:
        raise ValueError(f"Strand '{strand_id}'에 'gro'와 'itp' 경로가 필요합니다.")
    if not os.path.isfile(gro_path):
        raise FileNotFoundError(f"GRO 파일을 찾을 수 없습니다: {gro_path}")
    if not os.path.isfile(itp_path):
        raise FileNotFoundError(f"ITP 파일을 찾을 수 없습니다: {itp_path}")

    definition = _extract_single_definition(itp_path, entry.get("molecule_name"))
    beads = definition.get("beads", [])
    if not beads:
        raise ValueError(f"ITP '{itp_path}'에 atom 정보가 없습니다.")

    gro_atoms = read_gro_atoms(gro_path)
    if len(gro_atoms) != len(beads):
        raise ValueError(
            f"GRO({gro_path}) 원자 수({len(gro_atoms)})와 "
            f"ITP({itp_path}) atom 수({len(beads)})가 일치하지 않습니다."
        )

    # The two attachment atoms, by atom name. Exactly BCK1 and BCK2: a
    # different count is ambiguous about which ends bond, so it is refused
    # rather than guessed at.
    marked = {}
    for position, bead in enumerate(beads):
        name = str(bead.get("atom") or "").upper()
        if name.startswith("BCK"):
            marked[name] = position
    if set(marked) != {"BCK1", "BCK2"}:
        raise ValueError(
            f"Strand '{strand_id}' 템플릿에는 atom 이름 'BCK1'과 'BCK2'가 정확히 "
            f"하나씩 필요합니다 (현재: {sorted(marked) or '없음'}). 두 원자가 "
            "junction에 결합하는 실제 부착 원자입니다."
        )
    head_pos = marked["BCK1"]
    tail_pos = marked["BCK2"]

    raw_coords = np.array([atom.position for atom in gro_atoms], dtype=np.float64)
    head = raw_coords[head_pos]
    tail = raw_coords[tail_pos]
    span_vector = tail - head
    span_length = float(np.linalg.norm(span_vector))
    if span_length < 1e-6:
        raise ValueError(
            f"Strand '{strand_id}'의 BCK1/BCK2 좌표가 겹칩니다: 부착 축을 정의할 "
            "수 없습니다."
        )
    coords = raw_coords - 0.5 * (head + tail)
    axis = span_vector / span_length

    template_beads: List[BeadTemplate] = []
    total_mass = 0.0
    for position, bead in enumerate(beads):
        mass = bead.get("mass")
        if mass is None:
            raise ValueError(
                f"Strand '{strand_id}' ITP의 atom {bead.get('nr')}에 질량이 "
                "없습니다. 전 원자 템플릿은 질량을 명시해야 합니다."
            )
        total_mass += float(mass)
        template_beads.append(
            BeadTemplate(
                name=bead.get("atom"),
                atom_type=bead.get("type"),
                residue_name=bead.get("residue"),
                residue_number=bead.get("resnr", 1),
                original_index=bead["nr"],
                cgnr=bead.get("cgnr", 1),
                charge=bead.get("charge", 0.0),
                mass=float(mass),
                coord=coords[position],
            )
        )

    # 0-based position lookup for angle re-indexing. Atom numbering gaps in
    # the ITP would silently shift every term, so the identity is checked.
    position_of = {bead["nr"]: pos for pos, bead in enumerate(beads)}

    internal_angles = []
    for angle_def in definition.get("angles", []):
        atoms = [angle_def.get("from"), angle_def.get("center"), angle_def.get("to")]
        if any(a is None or a not in position_of for a in atoms):
            raise ValueError(
                f"Strand '{strand_id}'의 angle {atoms}가 존재하지 않는 atom을 "
                "참조합니다."
            )
        internal_angles.append(
            {
                "from": position_of[atoms[0]],
                "center": position_of[atoms[1]],
                "to": position_of[atoms[2]],
                "funct": angle_def["funct"],
                "params": angle_def["params"],
            }
        )

    return StrandTemplate(
        id=strand_id,
        beads=template_beads,
        coords=coords,
        attachment_positions=(head_pos, tail_pos),
        attachment_axis=axis,
        span_length=span_length,
        total_mass=total_mass,
        internal_bonds=list(definition.get("bonds", [])),
        internal_angles=internal_angles,
        dihedrals_full=list(definition.get("dihedrals", [])),
        impropers_full=list(definition.get("impropers", [])),
        pairs=list(definition.get("pairs", [])),
        exclusions=list(definition.get("exclusions", [])),
        constraints=list(definition.get("constraints", [])),
        virtual_sites=list(definition.get("virtual_sites", [])),
        restraints=list(definition.get("restraints", [])),
        cmaptypes=list(definition.get("cmaptypes", [])),
        polarization=list(definition.get("polarization", [])),
        other_sections=dict(definition.get("other_sections", {})),
    )
