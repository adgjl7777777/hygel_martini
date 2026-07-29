from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from collections import OrderedDict, defaultdict, deque
from fractions import Fraction

from ..config import (
    MonomerTemplate, 
    WeightedAtomRef, 
    ValidationReport, 
    ConnectionMetadata,
    ConnectionDetectionConfig,
    parse_xyz
)
from .core import _split_csv, _distance

SECTION_HEADERS = {
    "BEADS",
    "BONDS",
    "CONSTRAINTS",
    "ANGLES",
    "DIHEDRALS",
    "IMPROPERS",
}

def _parse_weighted_atom(token: str) -> WeightedAtomRef:
    match = re.fullmatch(r"(\d+)(?:/(\d+))?", token)
    if not match:
        raise ValueError(f"Malformed BEADS atom token: {token}")
    denominator = int(match.group(2)) if match.group(2) else 1
    if denominator < 1:
        raise ValueError(f"Invalid denominator in BEADS atom token: {token}")
    return WeightedAtomRef(atom_index=int(match.group(1)), denominator=denominator)

def _parse_section_ints(path: Path, line: str, expected: int) -> Tuple[int, ...]:
    values = tuple(int(token) for token in _split_csv(line))
    if len(values) != expected:
        raise ValueError(f"{path}: expected {expected} integers in line '{line}'")
    return values

def parse_bartender_inp(path: Path) -> MonomerTemplate:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    preamble: List[str] = []
    sections: Dict[str, List[str]] = {header: [] for header in SECTION_HEADERS}
    current: Optional[str] = None
    seen_header = False

    for raw in lines:
        stripped = raw.strip()
        if not seen_header:
            if stripped.upper() in SECTION_HEADERS:
                seen_header = True
                current = stripped.upper()
                continue
            preamble.append(raw)
            continue

        if not stripped or stripped.startswith("#"):
            continue

        header = stripped.upper()
        if header in SECTION_HEADERS:
            current = header
            continue
        if current is None:
            continue
        sections[current].append(stripped)

    if not sections["BEADS"]:
        raise ValueError(f"{path} has no BEADS section.")

    beads: "OrderedDict[int, List[WeightedAtomRef]]" = OrderedDict()
    for line in sections["BEADS"]:
        match = re.match(r"^(\d+)\s+(.+)$", line)
        if not match:
            raise ValueError(f"{path}: malformed BEADS line '{line}'")
        bead_id = int(match.group(1))
        refs = [_parse_weighted_atom(token) for token in _split_csv(match.group(2))]
        beads[bead_id] = refs

    bonds = [tuple(_parse_section_ints(path, line, 2)) for line in sections["BONDS"]]
    constraints = [tuple(_parse_section_ints(path, line, 2)) for line in sections["CONSTRAINTS"]]
    angles = [tuple(_parse_section_ints(path, line, 3)) for line in sections["ANGLES"]]
    dihedrals = [tuple(_parse_section_ints(path, line, 4)) for line in sections["DIHEDRALS"]]
    impropers = [tuple(_parse_section_ints(path, line, 4)) for line in sections["IMPROPERS"]]

    return MonomerTemplate(
        path=path,
        preamble=preamble,
        beads=beads,
        bonds=[(int(a), int(b)) for a, b in bonds],
        constraints=[(int(a), int(b)) for a, b in constraints],
        angles=[(int(a), int(b), int(c)) for a, b, c in angles],
        dihedrals=[(int(a), int(b), int(c), int(d)) for a, b, c, d in dihedrals],
        impropers=[(int(a), int(b), int(c), int(d)) for a, b, c, d in impropers],
    )

def _weighted_atom_owners(template: MonomerTemplate) -> Dict[int, List[WeightedAtomRef]]:
    owners: Dict[int, List[WeightedAtomRef]] = defaultdict(list)
    for refs in template.beads.values():
        for ref in refs:
            owners[ref.atom_index].append(ref)
    return owners

def _connector_indices(symbols: Sequence[str], indicator: str) -> List[int]:
    marker = indicator.strip().upper()
    return [index for index, symbol in enumerate(symbols, start=1) if str(symbol).strip().upper() == marker]

def infer_backbone_beads(
    template: MonomerTemplate,
    xyz_path: Path,
    backbone_atom_cfg: Dict[str, List[int]],
) -> Tuple[int, ...]:
    head_atoms = list(backbone_atom_cfg.get("head", []))
    tail_atoms = list(backbone_atom_cfg.get("tail", []))
    body_atoms = list(backbone_atom_cfg.get("body", []))
    if not head_atoms and not tail_atoms and not body_atoms:
        return ()

    tracked_atoms = set(head_atoms) | set(tail_atoms) | set(body_atoms)
    if template.atom_count and any(atom_index > template.atom_count for atom_index in tracked_atoms):
        raise ValueError(
            f"{xyz_path.name}: backbone atom indices {sorted(tracked_atoms)} exceed template atom count {template.atom_count}."
        )

    owners = _weighted_atom_owners(template)
    missing = sorted(atom_index for atom_index in tracked_atoms if atom_index not in owners)
    if missing:
        raise ValueError(f"{xyz_path.name}: backbone atoms {missing} are not assigned to any bead in the init template.")

    backbone_beads: List[int] = []
    seen_beads: set[int] = set()
    for bead_id, refs in template.beads.items():
        if any(ref.atom_index in tracked_atoms for ref in refs) and bead_id not in seen_beads:
            backbone_beads.append(int(bead_id))
            seen_beads.add(int(bead_id))

    if not backbone_beads:
        raise ValueError(f"{xyz_path.name}: no beads contain the configured backbone atoms {sorted(tracked_atoms)}.")

    return tuple(backbone_beads)

def validate_template(
    template: MonomerTemplate,
    xyz_path: Path,
    connection_cfg: ConnectionDetectionConfig,
) -> ValidationReport:
    symbols, _ = parse_xyz(xyz_path)
    natoms = len(symbols)
    report = ValidationReport(target=str(template.path))

    owners = _weighted_atom_owners(template)
    if template.atom_count != natoms:
        report.problems.append(
            f"Template max atom index is {template.atom_count}, but xyz atom count is {natoms}."
        )

    for atom_index, refs in owners.items():
        if atom_index < 1 or atom_index > natoms:
            report.problems.append(f"Atom index {atom_index} is out of range for {xyz_path.name}.")
            continue
        total_weight = sum((ref.weight for ref in refs), Fraction(0, 1))
        if total_weight != Fraction(1, 1):
            report.problems.append(f"Atom {atom_index} has total bead weight {total_weight} instead of 1.")
        if len(refs) == 1 and refs[0].denominator != 1:
            report.problems.append(
                f"Atom {atom_index} appears once but uses fractional token {refs[0].format()}."
            )
        if len(refs) > 1 and any(ref.denominator != len(refs) for ref in refs):
            tokens = ", ".join(ref.format() for ref in refs)
            report.problems.append(
                f"Atom {atom_index} is duplicated but tokens do not match the n-way fractional rule: {tokens}"
            )

    missing_atoms = [atom_index for atom_index in range(1, natoms + 1) if atom_index not in owners]
    if missing_atoms:
        report.problems.append(
            f"Template misses atom indices: {missing_atoms[:20]}{'...' if len(missing_atoms) > 20 else ''}"
        )

    connector_indices = _connector_indices(symbols, connection_cfg.indicator)
    if len(connector_indices) < 2:
        report.problems.append(
            f"{xyz_path.name} must contain at least two '{connection_cfg.indicator}' connector atoms, "
            f"found {len(connector_indices)}."
        )
    for connector_index in connector_indices:
        if connector_index not in owners:
            report.problems.append(
                f"Connector atom {connector_index} ('{connection_cfg.indicator}') is not assigned to any bead."
            )

    bead_ids = set(template.beads.keys())
    adjacency: Dict[int, set[int]] = {bead_id: set() for bead_id in bead_ids}
    for a, b in list(template.bonds) + list(template.constraints):
        if a not in bead_ids or b not in bead_ids:
            report.problems.append(f"Bond/constraint references unknown bead ids: {a},{b}")
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)

    isolated = sorted(bead_id for bead_id, neighbors in adjacency.items() if not neighbors)
    if isolated:
        report.problems.append(
            f"Each bead must participate in at least one bond/constraint. Isolated beads: {isolated}"
        )

    if bead_ids:
        start = next(iter(bead_ids))
        seen = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        disconnected = sorted(bead_id for bead_id in bead_ids if bead_id not in seen)
        if disconnected:
            report.problems.append(f"Bead graph is disconnected. Unreachable beads: {disconnected}")

    return report

def infer_connection_metadata(
    template: MonomerTemplate,
    xyz_path: Path,
    connection_cfg: ConnectionDetectionConfig,
    backbone_atom_cfg: Dict[str, List[int]],
) -> ConnectionMetadata:
    symbols, coords = parse_xyz(xyz_path)
    head_refs = list(backbone_atom_cfg.get("head", []))
    tail_refs = list(backbone_atom_cfg.get("tail", []))
    user_head_refs = [ref - 1 for ref in head_refs]
    user_tail_refs = [ref - 1 for ref in tail_refs]
    required_atoms = max(head_refs + tail_refs, default=0)
    if len(symbols) < required_atoms:
        raise ValueError(f"{xyz_path.name} must contain at least {required_atoms} atoms.")

    head_carbon = head_refs[0] if head_refs else 0
    tail_carbon = tail_refs[0] if tail_refs else 0
    head_br: Optional[int] = None
    tail_br: Optional[int] = None

    connector_indices = _connector_indices(symbols, connection_cfg.indicator)
    if len(connector_indices) < 2:
        raise ValueError(
            f"{xyz_path.name}: expected at least two '{connection_cfg.indicator}' connector atoms, found {len(connector_indices)}."
        )

    def distance_to_refs(connector_atom: int, refs: Sequence[int]) -> float:
        return min(_distance(coords[ref - 1], coords[connector_atom - 1]) for ref in refs)

    if head_refs and tail_refs:
        best_pair: Optional[Tuple[float, int, int]] = None
        for head_candidate in connector_indices:
            d_head = distance_to_refs(head_candidate, head_refs)
            if d_head > connection_cfg.cutoff:
                continue
            for tail_candidate in connector_indices:
                if tail_candidate == head_candidate:
                    continue
                d_tail = distance_to_refs(tail_candidate, tail_refs)
                if d_tail > connection_cfg.cutoff:
                    continue
                payload = (d_head + d_tail, head_candidate, tail_candidate)
                if best_pair is None or payload < best_pair:
                    best_pair = payload
        if best_pair is None:
            raise ValueError(
                f"{xyz_path.name}: could not infer distinct head/tail '{connection_cfg.indicator}' atoms "
                f"near backbone_atoms.head={user_head_refs} and backbone_atoms.tail={user_tail_refs} "
                f"with cutoff {connection_cfg.cutoff} A."
            )
        _, head_br, tail_br = best_pair
    elif head_refs:
        candidates = [
            (distance_to_refs(connector_atom, head_refs), connector_atom)
            for connector_atom in connector_indices
            if distance_to_refs(connector_atom, head_refs) <= connection_cfg.cutoff
        ]
        if not candidates:
            raise ValueError(
                f"{xyz_path.name}: could not infer head '{connection_cfg.indicator}' near backbone_atoms.head={user_head_refs} "
                f"with cutoff {connection_cfg.cutoff} A."
            )
        candidates.sort()
        head_br = candidates[0][1]
        remaining = [connector_atom for connector_atom in connector_indices if connector_atom != head_br]
        if len(remaining) != 1:
            raise ValueError(
                f"{xyz_path.name}: backbone_atoms defines only head, so exactly two connector atoms are required."
            )
        tail_br = remaining[0]
    elif tail_refs:
        candidates = [
            (distance_to_refs(connector_atom, tail_refs), connector_atom)
            for connector_atom in connector_indices
            if distance_to_refs(connector_atom, tail_refs) <= connection_cfg.cutoff
        ]
        if not candidates:
            raise ValueError(
                f"{xyz_path.name}: could not infer tail '{connection_cfg.indicator}' near backbone_atoms.tail={user_tail_refs} "
                f"with cutoff {connection_cfg.cutoff} A."
            )
        candidates.sort()
        tail_br = candidates[0][1]
        remaining = [connector_atom for connector_atom in connector_indices if connector_atom != tail_br]
        if len(remaining) != 1:
            raise ValueError(
                f"{xyz_path.name}: backbone_atoms defines only tail, so exactly two connector atoms are required."
            )
        head_br = remaining[0]
    else:
        raise ValueError(f"{xyz_path.name}: backbone_atoms must define at least one of head or tail.")

    def owner(atom_index: int, label: str) -> int:
        owners = [
            bead_id
            for bead_id, refs in template.beads.items()
            if any(ref.atom_index == atom_index for ref in refs)
        ]
        if len(owners) != 1:
            raise ValueError(
                f"{xyz_path.name}: expected exactly one bead for {label} atom {atom_index}, found {owners or 'none'}."
            )
        return owners[0]

    return ConnectionMetadata(
        head_carbon=head_carbon,
        tail_carbon=tail_carbon,
        head_br=head_br,
        tail_br=tail_br,
        left_connection_bead=owner(head_br, "head connector"),
        right_connection_bead=owner(tail_br, "tail connector"),
        backbone_beads=infer_backbone_beads(template, xyz_path, backbone_atom_cfg),
    )

def default_bead_spec(token: str, bead_count: int) -> dict[str, list[str]]:
    labels = [f"{token}{index}" for index in range(1, bead_count + 1)]
    return {"labels": labels, "types": list(labels)}

def build_bead_maps(
    case: Dict[str, object],
    overrides: Dict[str, Dict[str, List[str]]],
) -> tuple[Dict[int, str], Dict[int, str], set[int]]:
    from ..config import normalize_label_spec
    monomers = case.get("monomers")
    tokens = case.get("sequence_tokens")
    if not isinstance(monomers, dict) or not isinstance(tokens, list):
        raise ValueError("case.json must contain 'monomers' and 'sequence_tokens'.")

    case_specs = case.get("bead_specs", {})
    if not isinstance(case_specs, dict):
        case_specs = {}

    label_map: Dict[int, str] = {}
    type_map: Dict[int, str] = {}
    backbone_beads = set(int(value) for value in case.get("backbone_beads", []))
    use_case_backbone = bool(backbone_beads)
    if not backbone_beads:
        backbone_beads = set(int(value) for value in case.get("connection_beads", []))
        use_case_backbone = bool(backbone_beads)
    offset = 0
    for token in tokens:
        if token not in monomers:
            raise KeyError(f"Token {token} is not present in case['monomers'].")
        bead_count = int(monomers[token]["bead_count"])
        spec = normalize_label_spec(token, bead_count, overrides.get(token) or case_specs.get(token))
        for local_index in range(1, bead_count + 1):
            global_index = offset + local_index
            label_map[global_index] = spec["labels"][local_index - 1]
            type_map[global_index] = spec["types"][local_index - 1]
        if not use_case_backbone:
            local_backbone = monomers[token].get("backbone_beads", [])
            if isinstance(local_backbone, list) and local_backbone:
                for bead_id in local_backbone:
                    backbone_beads.add(offset + int(bead_id))
            else:
                backbone_beads.add(offset + int(monomers[token].get("left_connection_bead", 1)))
        offset += bead_count
    return label_map, type_map, backbone_beads


def validate_generated_input(
    template: MonomerTemplate,
    xyz_path: Path,
    terminal_cap_indices: List[int],
) -> ValidationReport:
    """Validate a generated polymer MonomerTemplate against its XYZ file.

    Unlike validate_template, this skips connector-atom checks and allows
    terminal_cap_indices (Br placeholder atoms) to be unassigned.
    """
    symbols, _ = parse_xyz(xyz_path)
    natoms = len(symbols)
    report = ValidationReport(target=str(template.path))
    cap_set = set(terminal_cap_indices)

    owners = _weighted_atom_owners(template)
    if template.atom_count != natoms:
        report.problems.append(
            f"Template max atom index is {template.atom_count}, but xyz atom count is {natoms}."
        )

    for atom_index, refs in owners.items():
        if atom_index < 1 or atom_index > natoms:
            report.problems.append(f"Atom index {atom_index} is out of range for {xyz_path.name}.")
            continue
        total_weight = sum((ref.weight for ref in refs), Fraction(0, 1))
        if total_weight != Fraction(1, 1):
            report.problems.append(f"Atom {atom_index} has total bead weight {total_weight} instead of 1.")

    missing = [i for i in range(1, natoms + 1) if i not in owners and i not in cap_set]
    if missing:
        report.problems.append(
            f"Template misses atom indices: {missing[:20]}{'...' if len(missing) > 20 else ''}"
        )

    bead_ids = set(template.beads.keys())
    for a, b in list(template.bonds) + list(template.constraints):
        if a not in bead_ids or b not in bead_ids:
            report.problems.append(f"Bond/constraint references unknown bead ids: {a},{b}")
    for a, b, c in template.angles:
        if any(x not in bead_ids for x in (a, b, c)):
            report.problems.append(f"Angle references unknown bead ids: {a},{b},{c}")
    for a, b, c, d in list(template.dihedrals) + list(template.impropers):
        if any(x not in bead_ids for x in (a, b, c, d)):
            report.problems.append(f"Dihedral/improper references unknown bead ids: {a},{b},{c},{d}")

    return report
