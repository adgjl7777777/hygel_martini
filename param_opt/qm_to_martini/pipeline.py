from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.utils import parse_csv_list
from ..polymer_maker.maker import build_polymer as build_polymer_structure
from ..polymer_maker.maker import load_monomer_library

from .config import (
    CONNECTION_CUTOFF,
    ConnectionDetectionConfig,
    MergedVariant,
    MonomerTemplate,
    ParamLine,
    PolymerInputBundle,
    TermGenerationConfig,
    TypedRecord,
    ValidationReport,
    WeightedAtomRef,
    _get_slurm_cpu_count,
    build_sequence_jobs,
    check_configured_tools,
    default_workdir_name,
    ensure_case_logs_dir,
    execute_case_script,
    export_backbone_atom_config,
    normalize_monomer_configs,
    normalize_sequence,
    parse_bool,
    parse_xyz,
    render_orca_input,
    render_xtb_md_input,
    resolve_backbone_atom_config,
    resolve_case_electronic_state,
    resolve_connection_detection_config,
    resolve_executable_command,
    resolve_execution_settings,
    resolve_log_settings,
    resolve_optional_path,
    resolve_orca_settings,
    resolve_spin_state,
    resolve_term_generation_config,
    resolve_under_base,
    resolve_xtb_settings,
    shell_assign,
    write_text,
)


SECTION_HEADERS = {
    "BEADS",
    "BONDS",
    "CONSTRAINTS",
    "ANGLES",
    "DIHEDRALS",
    "IMPROPERS",
}
CONNECTION_CUTOFF = 2.2
RMSD_RE = re.compile(r"rmsd:\s*([0-9]*\.?[0-9]+)")


@dataclass(frozen=True)
class ConnectionMetadata:
    head_carbon: int
    tail_carbon: int
    head_br: int
    tail_br: int
    left_connection_bead: int
    right_connection_bead: int
    backbone_beads: Tuple[int, ...] = ()


def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _split_csv(raw: str) -> List[str]:
    return [token.strip() for token in re.split(r"\s*,\s*", raw.strip()) if token.strip()]


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


def _sorted_pair(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _canon_angle(i: int, j: int, k: int) -> Tuple[int, int, int]:
    return (i, j, k) if i <= k else (k, j, i)


def _build_graph(edges: Iterable[Tuple[int, int]]) -> Dict[int, set[int]]:
    graph: Dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _canon_reversible(values: Sequence[int]) -> Tuple[int, ...]:
    forward = tuple(int(value) for value in values)
    reverse = tuple(reversed(forward))
    return forward if forward <= reverse else reverse


def _reversal_unique_permutations(values: Sequence[int]) -> List[Tuple[int, ...]]:
    unique: List[Tuple[int, ...]] = []
    seen: set[Tuple[int, ...]] = set()
    for perm in permutations(values):
        key = _canon_reversible(perm)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    unique.sort()
    return unique


def _generate_all_reversible_combinations(
    bead_ids: Sequence[int],
    body_size: int,
    existing: set[Tuple[int, ...]],
) -> List[Tuple[int, ...]]:
    generated: List[Tuple[int, ...]] = []
    seen: set[Tuple[int, ...]] = set()
    for combo in combinations(sorted(bead_ids), body_size):
        for candidate in _reversal_unique_permutations(combo):
            if candidate in existing or candidate in seen:
                continue
            generated.append(candidate)
            seen.add(candidate)
    return generated


def _generate_all_linkage_bonds(inp_data: MonomerTemplate) -> List[Tuple[int, int]]:
    bead_ids = sorted(inp_data.beads.keys())
    existing = {_sorted_pair(a, b) for a, b in inp_data.bonds} | {_sorted_pair(a, b) for a, b in inp_data.constraints}
    return [tuple(int(value) for value in candidate) for candidate in _generate_all_reversible_combinations(bead_ids, 2, existing)]


def _generate_all_linkage_angles(inp_data: MonomerTemplate) -> List[Tuple[int, int, int]]:
    bead_ids = sorted(inp_data.beads.keys())
    existing = {_canon_angle(a, b, c) for a, b, c in inp_data.angles}
    return [
        tuple(int(value) for value in candidate)
        for candidate in _generate_all_reversible_combinations(bead_ids, 3, existing)
    ]


def _generate_all_linkage_dihedrals(inp_data: MonomerTemplate) -> List[Tuple[int, int, int, int]]:
    bead_ids = sorted(inp_data.beads.keys())
    existing = {_canon_reversible((a, b, c, d)) for a, b, c, d in inp_data.dihedrals}
    return [
        tuple(int(value) for value in candidate)
        for candidate in _generate_all_reversible_combinations(bead_ids, 4, existing)
    ]


def _generate_all_linkage_impropers(inp_data: MonomerTemplate) -> List[Tuple[int, int, int, int]]:
    bead_ids = sorted(inp_data.beads.keys())
    existing = {_canon_reversible((a, b, c, d)) for a, b, c, d in inp_data.impropers}
    return [
        tuple(int(value) for value in candidate)
        for candidate in _generate_all_reversible_combinations(bead_ids, 4, existing)
    ]


def _connection_proxy_count(indices: Sequence[int], backbone_beads: set[int]) -> int:
    return len({int(index) for index in indices if int(index) in backbone_beads})


def _filter_connection_proxy_terms(
    terms: Sequence[Tuple[int, ...]],
    backbone_beads: set[int],
    minimum_distinct: int = 2,
) -> List[Tuple[int, ...]]:
    return [
        tuple(int(value) for value in term)
        for term in terms
        if _connection_proxy_count(term, backbone_beads) >= minimum_distinct
    ]


def _distance_cache(graph: Dict[int, set[int]]) -> Callable[[int, int], Optional[int]]:
    cache: Dict[Tuple[int, int], Optional[int]] = {}

    def lookup(a: int, b: int) -> Optional[int]:
        key = _sorted_pair(int(a), int(b))
        if key not in cache:
            cache[key] = shortest_path_len(graph, key[0], key[1])
        return cache[key]

    return lookup


def _topology_reference_cost(
    section: str,
    indices: Sequence[int],
    distance_lookup: Callable[[int, int], Optional[int]],
) -> Optional[int]:
    edges: List[Tuple[int, int]]
    if section == "bond":
        edges = [(indices[0], indices[1])]
    elif section == "angle":
        edges = [(indices[0], indices[1]), (indices[1], indices[2])]
    elif section == "dihedral":
        edges = [(indices[0], indices[1]), (indices[1], indices[2]), (indices[2], indices[3])]
    elif section == "improper":
        center = indices[0]
        edges = [(center, indices[1]), (center, indices[2]), (center, indices[3])]
    else:
        raise ValueError(f"Unsupported topology section: {section}")

    total = 0
    for left, right in edges:
        distance = distance_lookup(left, right)
        if distance is None:
            return None
        total += max(distance - 1, 0)
    return total


def _changed_index_count(term: Sequence[int], reference: Sequence[int]) -> int:
    changed = sum(1 for left, right in zip(term, reference) if int(left) != int(right))
    return max(0, changed - 1)


def _topology_term_cost(
    section: str,
    term: Sequence[int],
    distance_lookup: Callable[[int, int], Optional[int]],
    *,
    allow_swaps: bool,
) -> Optional[int]:
    direct_cost = _topology_reference_cost(section, term, distance_lookup)
    if not allow_swaps or len(term) <= 2:
        return direct_cost

    best = direct_cost
    for reference in permutations(term):
        ref_cost = _topology_reference_cost(section, reference, distance_lookup)
        if ref_cost is None:
            continue
        candidate_cost = ref_cost + _changed_index_count(term, reference)
        if best is None or candidate_cost < best:
            best = candidate_cost
    return best


def _filter_topology_terms(
    section: str,
    terms: Sequence[Tuple[int, ...]],
    graph: Dict[int, set[int]],
    budget: int,
    *,
    allow_swaps: bool,
) -> List[Tuple[int, ...]]:
    distance_lookup = _distance_cache(graph)
    filtered: List[Tuple[int, ...]] = []
    for term in terms:
        cost = _topology_term_cost(section, term, distance_lookup, allow_swaps=allow_swaps)
        if cost is not None and cost <= budget:
            filtered.append(tuple(int(value) for value in term))
    return filtered


def _generate_augmented_terms(
    base: MonomerTemplate,
    term_cfg: TermGenerationConfig,
    backbone_beads: Sequence[int],
) -> tuple[
    List[Tuple[int, int]],
    List[Tuple[int, int, int]],
    List[Tuple[int, int, int, int]],
    List[Tuple[int, int, int, int]],
]:
    if term_cfg.mode == "init_only":
        return [], [], [], []

    new_bonds = _generate_all_linkage_bonds(base)
    new_angles = _generate_all_linkage_angles(base)
    new_dihedrals = _generate_all_linkage_dihedrals(base)
    new_impropers = _generate_all_linkage_impropers(base)

    if term_cfg.mode in {"topology_n", "topology_swap_n"}:
        graph = _build_graph(list(base.bonds) + list(base.constraints))
        allow_swaps = term_cfg.mode == "topology_swap_n"
        new_bonds = _filter_topology_terms("bond", new_bonds, graph, term_cfg.n, allow_swaps=allow_swaps)
        new_angles = _filter_topology_terms("angle", new_angles, graph, term_cfg.n, allow_swaps=allow_swaps)
        new_dihedrals = _filter_topology_terms(
            "dihedral",
            new_dihedrals,
            graph,
            term_cfg.n,
            allow_swaps=allow_swaps,
        )
        new_impropers = _filter_topology_terms(
            "improper",
            new_impropers,
            graph,
            term_cfg.n,
            allow_swaps=allow_swaps,
        )

    if term_cfg.mode == "polymer_backbone":
        backbone_set = {int(value) for value in backbone_beads}
        new_bonds = [
            tuple(term)
            for term in _filter_connection_proxy_terms(new_bonds, backbone_set, minimum_distinct=2)
        ]
        new_angles = [
            tuple(term)
            for term in _filter_connection_proxy_terms(new_angles, backbone_set, minimum_distinct=2)
        ]
        new_dihedrals = [
            tuple(term)
            for term in _filter_connection_proxy_terms(new_dihedrals, backbone_set, minimum_distinct=2)
        ]
        new_impropers = [
            tuple(term)
            for term in _filter_connection_proxy_terms(new_impropers, backbone_set, minimum_distinct=2)
        ]

    return new_bonds, new_angles, new_dihedrals, new_impropers


def format_inp(template: MonomerTemplate) -> str:
    lines: List[str] = []
    if template.preamble:
        lines.extend(template.preamble)
    lines.append("BEADS")
    for bead_id, refs in template.beads.items():
        lines.append(f"{bead_id} " + ",".join(ref.format() for ref in refs))
    lines.append("BONDS")
    for a, b in template.bonds:
        lines.append(f"{a},{b}")
    if template.constraints:
        lines.append("CONSTRAINTS")
        for a, b in template.constraints:
            lines.append(f"{a},{b}")
    lines.append("ANGLES")
    for a, b, c in template.angles:
        lines.append(f"{a},{b},{c}")
    lines.append("DIHEDRALS")
    for a, b, c, d in template.dihedrals:
        lines.append(f"{a},{b},{c},{d}")
    lines.append("IMPROPERS")
    for a, b, c, d in template.impropers:
        lines.append(f"{a},{b},{c},{d}")
    return "\n".join(lines) + "\n"


def validate_generated_input(
    inp_data: MonomerTemplate,
    xyz_path: Path,
    terminal_cap_indices: Optional[Sequence[int]] = None,
) -> ValidationReport:
    symbols, _ = parse_xyz(xyz_path)
    natoms = len(symbols)
    report = ValidationReport(target=str(xyz_path))
    owners = _weighted_atom_owners(inp_data)

    missing_atoms = [atom_index for atom_index in range(1, natoms + 1) if atom_index not in owners]
    if missing_atoms:
        report.problems.append(
            f"Generated input misses atom indices: {missing_atoms[:20]}{'...' if len(missing_atoms) > 20 else ''}"
        )

    for atom_index, refs in owners.items():
        if atom_index < 1 or atom_index > natoms:
            report.problems.append(f"Generated input references atom {atom_index} outside 1..{natoms}.")
            continue
        total_weight = sum((ref.weight for ref in refs), Fraction(0, 1))
        if total_weight != Fraction(1, 1):
            report.problems.append(
                f"Generated input atom {atom_index} has total bead weight {total_weight} instead of 1."
            )

    if terminal_cap_indices:
        wrong_caps = [
            (atom_index, symbols[atom_index - 1])
            for atom_index in sorted(set(terminal_cap_indices))
            if 1 <= atom_index <= natoms and symbols[atom_index - 1] != "H"
        ]
        if wrong_caps:
            report.problems.append(f"Expected terminal cap atoms to be H, found {wrong_caps}")

    return report


def build_polymer_input(
    sequence: Sequence[str] | str,
    polymer_xyz_path: Path,
    templates: Dict[str, MonomerTemplate],
    metadata: Dict[str, ConnectionMetadata],
    term_cfg: TermGenerationConfig,
) -> PolymerInputBundle:
    tokens = normalize_sequence(sequence)
    symbols, _ = parse_xyz(polymer_xyz_path)
    natoms = len(symbols)

    preamble = [f"# Auto-generated polymer Bartender input for {sequence_stem(tokens)}"]
    beads: "OrderedDict[int, List[WeightedAtomRef]]" = OrderedDict()
    bonds: List[Tuple[int, int]] = []
    constraints: List[Tuple[int, int]] = []
    angles: List[Tuple[int, int, int]] = []
    dihedrals: List[Tuple[int, int, int, int]] = []
    impropers: List[Tuple[int, int, int, int]] = []

    report = ValidationReport(target=f"{sequence_stem(tokens)} base inp")
    atom_offset = 0
    bead_offset = 0
    connection_bonds: List[Tuple[int, int]] = []
    connection_beads: List[int] = []
    backbone_beads: List[int] = []
    terminal_cap_indices: List[int] = []
    previous_right_bead: Optional[int] = None
    expected_atoms = 0

    for block_index, token in enumerate(tokens):
        if token not in templates or token not in metadata:
            raise KeyError(f"Unknown monomer token '{token}'")
        template = templates[token]
        meta = metadata[token]

        removed = set()
        if len(tokens) > 1:
            if block_index > 0:
                removed.add(meta.head_br)
            if block_index < len(tokens) - 1:
                removed.add(meta.tail_br)

        local_to_global: Dict[int, int] = {}
        for local_atom_index in range(1, template.atom_count + 1):
            if local_atom_index in removed:
                continue
            local_to_global[local_atom_index] = atom_offset + len(local_to_global) + 1

        if meta.head_br in local_to_global:
            terminal_cap_indices.append(local_to_global[meta.head_br])
        if meta.tail_br in local_to_global:
            terminal_cap_indices.append(local_to_global[meta.tail_br])

        for bead_id, refs in template.beads.items():
            global_bead = bead_offset + bead_id
            mapped = [
                WeightedAtomRef(atom_index=local_to_global[ref.atom_index], denominator=ref.denominator)
                for ref in refs
                if ref.atom_index in local_to_global
            ]
            if not mapped:
                report.problems.append(
                    f"Block {block_index + 1} token {token} produced empty bead {global_bead} after connector removal."
                )
            beads[global_bead] = mapped

        bonds.extend([(bead_offset + a, bead_offset + b) for a, b in template.bonds])
        constraints.extend([(bead_offset + a, bead_offset + b) for a, b in template.constraints])
        angles.extend([(bead_offset + a, bead_offset + b, bead_offset + c) for a, b, c in template.angles])
        dihedrals.extend(
            [(bead_offset + a, bead_offset + b, bead_offset + c, bead_offset + d) for a, b, c, d in template.dihedrals]
        )
        impropers.extend(
            [(bead_offset + a, bead_offset + b, bead_offset + c, bead_offset + d) for a, b, c, d in template.impropers]
        )

        left_bead = bead_offset + meta.left_connection_bead
        right_bead = bead_offset + meta.right_connection_bead
        connection_beads.extend([left_bead, right_bead])
        local_backbone_beads = meta.backbone_beads or tuple(sorted({meta.left_connection_bead, meta.right_connection_bead}))
        backbone_beads.extend(bead_offset + int(bead_id) for bead_id in local_backbone_beads)
        if previous_right_bead is not None:
            bond = _sorted_pair(previous_right_bead, left_bead)
            bonds.append(bond)
            connection_bonds.append(bond)
        previous_right_bead = right_bead

        kept_atoms = template.atom_count - len(removed)
        expected_atoms += kept_atoms
        atom_offset += kept_atoms
        bead_offset += template.bead_count

    if natoms != expected_atoms:
        report.problems.append(f"Polymer xyz atom count is {natoms}, but template-based build expects {expected_atoms}.")

    base = MonomerTemplate(
        path=polymer_xyz_path.with_suffix(".inp"),
        preamble=preamble,
        beads=beads,
        bonds=[(int(a), int(b)) for a, b in bonds],
        constraints=[(int(a), int(b)) for a, b in constraints],
        angles=[(int(a), int(b), int(c)) for a, b, c in angles],
        dihedrals=[(int(a), int(b), int(c), int(d)) for a, b, c, d in dihedrals],
        impropers=[(int(a), int(b), int(c), int(d)) for a, b, c, d in impropers],
    )

    base_check = validate_generated_input(base, polymer_xyz_path, terminal_cap_indices)
    report.problems.extend(base_check.problems)
    report.warnings.extend(base_check.warnings)

    new_bonds, new_angles, new_dihedrals, new_impropers = _generate_augmented_terms(
        base,
        term_cfg,
        backbone_beads,
    )
    augmented = MonomerTemplate(
        path=polymer_xyz_path.with_suffix(".inp"),
        preamble=list(base.preamble),
        beads=base.beads,
        bonds=list(base.bonds) + new_bonds,
        constraints=list(base.constraints),
        angles=list(base.angles) + new_angles,
        dihedrals=list(base.dihedrals) + new_dihedrals,
        impropers=list(base.impropers) + new_impropers,
    )
    augmented_report = validate_generated_input(augmented, polymer_xyz_path, terminal_cap_indices)

    return PolymerInputBundle(
        base=base,
        augmented=augmented,
        base_text=format_inp(base),
        augmented_text=format_inp(augmented),
        base_report=report,
        augmented_report=augmented_report,
        connection_bonds=connection_bonds,
        connection_beads=sorted(set(connection_beads)),
        backbone_beads=sorted(set(backbone_beads)),
    )


def default_bead_spec(token: str, bead_count: int) -> dict[str, list[str]]:
    labels = [f"{token}{index}" for index in range(1, bead_count + 1)]
    return {"labels": labels, "types": list(labels)}


def split_main_and_comment(raw: str) -> Tuple[str, str]:
    stripped = raw.lstrip()
    while stripped.startswith(";"):
        stripped = stripped[1:].lstrip()
    if ";" in stripped:
        main, comment = stripped.split(";", 1)
        return main.strip(), comment.strip()
    return stripped.strip(), ""


def parse_param_line(raw: str, section: str, n_idx: int) -> Optional[ParamLine]:
    stripped = raw.strip()
    if not stripped:
        return None
    main, comment = split_main_and_comment(raw)
    if not main or not main[0].isdigit():
        return None
    parts = main.split()
    if len(parts) < n_idx + 1:
        return None

    rmsd = None
    if comment:
        match = RMSD_RE.search(comment)
        if match:
            try:
                rmsd = float(match.group(1))
            except ValueError:
                rmsd = None

    try:
        indices = tuple(int(parts[idx]) for idx in range(n_idx))
    except ValueError:
        return None

    return ParamLine(
        section=section,
        indices=indices,
        tokens=tuple(parts[n_idx:]),
        commented=stripped.startswith(";"),
        inline_comment=comment,
        rmsd=rmsd,
        raw=raw.rstrip("\n"),
    )


def parse_gmx_out_itp(path: Path) -> Dict[str, List[ParamLine]]:
    header_map = {
        "bonds": "bonds",
        "bondtypes": "bonds",
        "constraints": "constraints",
        "constrainttypes": "constraints",
        "angles": "angles",
        "angletypes": "angles",
        "dihedrals": "dihedrals",
        "dihedraltypes": "dihedrals",
        "impropers": "impropers",
        "impropertypes": "impropers",
    }
    parsed: Dict[str, List[ParamLine]] = {
        section: [] for section in ("bonds", "constraints", "angles", "dihedrals", "impropers")
    }
    current: Optional[str] = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            header = stripped.strip("[]").strip().lower()
            current = header_map.get(header)
            continue
        if current is None:
            continue
        n_idx = 2 if current in {"bonds", "constraints"} else 3 if current == "angles" else 4
        line = parse_param_line(raw, current, n_idx)
        if line is not None:
            parsed[current].append(line)
    return parsed


def summarize_itp(path: Path) -> Dict[str, object]:
    parsed = parse_gmx_out_itp(path)

    def _payload(line: ParamLine) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "indices": list(line.indices),
            "params": list(line.tokens),
            "commented": line.commented,
            "comment": line.inline_comment,
        }
        if line.rmsd is not None:
            payload["rmsd"] = line.rmsd
        return payload

    return {
        "path": str(path),
        "counts": {section: len(lines) for section, lines in parsed.items()},
        "bonds": [_payload(line) for line in parsed["bonds"]],
        "constraints": [_payload(line) for line in parsed["constraints"]],
        "angles": [_payload(line) for line in parsed["angles"]],
        "dihedrals": [_payload(line) for line in parsed["dihedrals"]],
        "impropers": [_payload(line) for line in parsed["impropers"]],
    }


def find_case_json(start: Path) -> Optional[Path]:
    current = start.resolve()
    for _ in range(6):
        candidate = current / "case.json"
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def resolve_case_artifact(case_dir: Path, case: Dict[str, object], key: str) -> Path:
    candidates: List[Path] = []
    artifacts = case.get("artifacts", {})
    if isinstance(artifacts, dict) and key in artifacts:
        candidates.append(case_dir / str(artifacts[key]))
    if key in case:
        raw = Path(str(case[key]))
        if raw.is_absolute():
            candidates.append(raw)
            candidates.append(case_dir / raw.name)
        else:
            candidates.append(case_dir / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve case artifact '{key}' from {case_dir}. Tried: {', '.join(str(path) for path in candidates)}"
    )


def normalize_label_spec(token: str, bead_count: int, raw_spec: Any) -> Dict[str, List[str]]:
    if raw_spec is None:
        return default_bead_spec(token, bead_count)
    if isinstance(raw_spec, list):
        labels = [str(value) for value in raw_spec]
        if len(labels) != bead_count:
            raise ValueError(f"Label override for token {token} has {len(labels)} entries, expected {bead_count}.")
        return {"labels": labels, "types": [label.split("(", 1)[0] if "(" in label else label for label in labels]}
    if isinstance(raw_spec, dict):
        labels = raw_spec.get("labels")
        types = raw_spec.get("types")
        if labels is None:
            raise ValueError(f"Label override for token {token} must contain 'labels'.")
        labels = [str(value) for value in labels]
        if len(labels) != bead_count:
            raise ValueError(f"Label override for token {token} has {len(labels)} labels, expected {bead_count}.")
        if types is None:
            types = [label.split("(", 1)[0] if "(" in label else label for label in labels]
        else:
            types = [str(value) for value in types]
        if len(types) != bead_count:
            raise ValueError(f"Type override for token {token} has {len(types)} entries, expected {bead_count}.")
        return {"labels": labels, "types": types}
    raise TypeError(f"Unsupported label specification for token {token}: {type(raw_spec)!r}")


def load_label_map(path: Optional[Path]) -> Dict[str, Dict[str, List[str]]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Label map JSON must be an object.")
    overrides: Dict[str, Dict[str, List[str]]] = {}
    for token, spec in data.items():
        if isinstance(spec, dict):
            entry = {"labels": [str(value) for value in spec.get("labels", [])]}
            if "types" in spec:
                entry["types"] = [str(value) for value in spec.get("types", [])]
            overrides[str(token)] = entry
        elif isinstance(spec, list):
            overrides[str(token)] = {"labels": [str(value) for value in spec], "types": []}
        else:
            raise ValueError(f"Unsupported label map entry for token {token}: {type(spec)!r}")
    return overrides


def build_bead_maps(
    case: Dict[str, object],
    overrides: Dict[str, Dict[str, List[str]]],
) -> tuple[Dict[int, str], Dict[int, str], set[int]]:
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


def shortest_path_len(graph: Dict[int, set[int]], start: int, goal: int) -> Optional[int]:
    if start == goal:
        return 0
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        node, dist = queue.popleft()
        for neighbor in graph.get(node, set()):
            if neighbor == goal:
                return dist + 1
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, dist + 1))
    return None


def choose_best_rmsd_uncomment(lines: List[ParamLine]) -> List[ParamLine]:
    grouped: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for index, line in enumerate(lines):
        grouped[line.indices].append(index)

    updated = list(lines)
    for positions in grouped.values():
        best_index = None
        best_value = math.inf
        for position in positions:
            value = updated[position].rmsd if updated[position].rmsd is not None else math.inf
            if value < best_value:
                best_value = value
                best_index = position
        if best_index is None or math.isinf(best_value):
            continue
        for position in positions:
            line = updated[position]
            updated[position] = ParamLine(
                section=line.section,
                indices=line.indices,
                tokens=line.tokens,
                commented=position != best_index,
                inline_comment=line.inline_comment,
                rmsd=line.rmsd,
                raw=line.raw,
            )
    return updated


def typed_records_for_result(
    itp_path: Path,
    case_path: Path,
    label_overrides: Dict[str, Dict[str, List[str]]],
) -> List[TypedRecord]:
    case = json.loads(case_path.read_text(encoding="utf-8"))
    parsed = parse_gmx_out_itp(itp_path)
    label_map, type_map, backbone_beads = build_bead_maps(case, label_overrides)

    graph = _build_graph(
        {_sorted_pair(*line.indices) for line in parsed["bonds"]}
        | {_sorted_pair(*line.indices) for line in parsed["constraints"]}
    )
    angle_lines = choose_best_rmsd_uncomment(parsed["angles"])
    source_tag = f"{case.get('sequence_stem', case_path.parent.name)}:{itp_path.parent.name}"

    def category(indices: Tuple[int, ...]) -> str:
        return "WITH_BACKBONE" if any(index in backbone_beads for index in indices) else "WITHOUT_BACKBONE"

    def map_labels(indices: Tuple[int, ...]) -> tuple[Tuple[str, ...], Tuple[str, ...]]:
        try:
            display = tuple(label_map[index] for index in indices)
            types = tuple(type_map[index] for index in indices)
        except KeyError as exc:
            raise KeyError(f"{itp_path}: bead index {exc.args[0]} is not present in the case bead map.") from exc
        return display, types

    section_map = {
        "bonds": "bondtypes",
        "constraints": "constrainttypes",
        "angles": "angletypes",
        "dihedrals": "dihedraltypes",
        "impropers": "impropertypes",
    }
    records: List[TypedRecord] = []

    for section_name in ("bonds", "constraints"):
        for line in parsed[section_name]:
            display, types = map_labels(line.indices)
            records.append(
                TypedRecord(
                    section=section_map[section_name],
                    category=category(line.indices),
                    angle_dist="",
                    type_names=types,
                    display_labels=display,
                    indices=line.indices,
                    tokens=line.tokens,
                    commented=line.commented,
                    inline_comment=line.inline_comment,
                    rmsd=line.rmsd,
                    source_tag=source_tag,
                    source_path=str(itp_path),
                )
            )

    for line in angle_lines:
        display, types = map_labels(line.indices)
        endpoint_dist = shortest_path_len(graph, line.indices[0], line.indices[2])
        records.append(
            TypedRecord(
                section="angletypes",
                category=category(line.indices),
                angle_dist="DIST_LE2" if endpoint_dist is not None and endpoint_dist <= 2 else "DIST_GE3",
                type_names=types,
                display_labels=display,
                indices=line.indices,
                tokens=line.tokens,
                commented=line.commented,
                inline_comment=line.inline_comment,
                rmsd=line.rmsd,
                source_tag=source_tag,
                source_path=str(itp_path),
            )
        )

    for line in parsed["dihedrals"]:
        display, types = map_labels(line.indices)
        records.append(
            TypedRecord(
                section="dihedraltypes",
                category=category(line.indices),
                angle_dist="",
                type_names=types,
                display_labels=display,
                indices=line.indices,
                tokens=line.tokens,
                commented=line.commented,
                inline_comment=line.inline_comment,
                rmsd=line.rmsd,
                source_tag=source_tag,
                source_path=str(itp_path),
            )
        )
    for line in parsed["impropers"]:
        display, types = map_labels(line.indices)
        records.append(
            TypedRecord(
                section="impropertypes",
                category=category(line.indices),
                angle_dist="",
                type_names=types,
                display_labels=display,
                indices=line.indices,
                tokens=line.tokens,
                commented=line.commented,
                inline_comment=line.inline_comment,
                rmsd=line.rmsd,
                source_tag=source_tag,
                source_path=str(itp_path),
            )
        )
    return records


def merge_records(records: List[TypedRecord]) -> Dict[Tuple[str, str, str, Tuple[str, ...]], List[MergedVariant]]:
    grouped: Dict[Tuple[str, str, str, Tuple[str, ...]], List[TypedRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.section, record.category, record.angle_dist, record.type_names)].append(record)

    merged: Dict[Tuple[str, str, str, Tuple[str, ...]], List[MergedVariant]] = {}
    for key, group in grouped.items():
        variants_by_signature: Dict[Tuple[Tuple[str, ...], bool, str], List[TypedRecord]] = defaultdict(list)
        for record in group:
            variants_by_signature[(record.tokens, record.commented, record.inline_comment.strip())].append(record)

        items = []
        for records_in_variant in variants_by_signature.values():
            sample = records_in_variant[0]
            items.append(
                {
                    "sample": sample,
                    "display_labels": sorted({record.display_labels for record in records_in_variant}),
                    "sources": sorted({record.source_tag for record in records_in_variant}),
                    "indices_examples": sorted({record.indices for record in records_in_variant}),
                    "inline_comments": sorted({record.inline_comment.strip() for record in records_in_variant if record.inline_comment.strip()}),
                    "rmsd_values": [record.rmsd for record in records_in_variant if record.rmsd is not None],
                }
            )

        def score(item: Dict[str, Any]) -> Tuple[float, int, float, str]:
            sample = item["sample"]
            if sample.section == "angletypes":
                rmsd = min(item["rmsd_values"]) if item["rmsd_values"] else math.inf
                return (0 if not sample.commented else 1, 0 if item["rmsd_values"] else 1, rmsd, sample.source_tag)
            return (0 if not sample.commented else 1, 0, 0.0, sample.source_tag)

        primary_item = min(items, key=score)
        variants: List[MergedVariant] = []
        for item in sorted(items, key=score):
            sample = item["sample"]
            variants.append(
                MergedVariant(
                    section=sample.section,
                    category=sample.category,
                    angle_dist=sample.angle_dist,
                    type_names=sample.type_names,
                    display_labels=item["display_labels"],
                    tokens=sample.tokens,
                    commented=sample.commented if item is primary_item else True,
                    sources=item["sources"],
                    indices_examples=item["indices_examples"],
                    inline_comments=item["inline_comments"],
                    rmsd_values=item["rmsd_values"],
                    primary=item is primary_item,
                )
            )
        merged[key] = variants

    return merged


def _format_type_names(type_names: Tuple[str, ...], widths: Tuple[int, ...]) -> str:
    return " ".join(f"{value:<{width}}" for value, width in zip(type_names, widths))


def line_from_variant(variant: MergedVariant) -> str:
    widths = (8, 8, 8, 8)
    prefix = _format_type_names(variant.type_names, widths[: len(variant.type_names)]).rstrip()
    main = f"{';' if variant.commented else ''}{prefix} {' '.join(variant.tokens)}".rstrip()
    comment_parts = []
    if variant.display_labels:
        comment_parts.append("labels=" + " | ".join(" ".join(entry) for entry in variant.display_labels))
    if variant.inline_comments:
        comment_parts.append("comments=" + " | ".join(variant.inline_comments))
    if variant.rmsd_values:
        comment_parts.append("rmsd=" + ",".join(f"{value:.3f}" for value in sorted(set(variant.rmsd_values))))
    if variant.sources:
        comment_parts.append("sources=" + ",".join(variant.sources))
    if variant.indices_examples:
        examples = " | ".join("-".join(str(value) for value in indices) for indices in variant.indices_examples[:5])
        comment_parts.append(f"indices={examples}")
    return main + (" ; " + " ; ".join(comment_parts) if comment_parts else "")


def write_merged_forcefield(
    path: Path,
    merged: Dict[Tuple[str, str, str, Tuple[str, ...]], List[MergedVariant]],
    root: Path,
    label_map_path: Optional[Path],
) -> None:
    lines = [
        "; Auto-generated merged Bartender forcefield summary",
        f"; root = {root}",
        f"; label_map = {label_map_path if label_map_path else '(default token-based labels)'}",
        "; The first uncommented line per type key is the selected representative.",
        "",
    ]

    section_order = ("bondtypes", "constrainttypes", "angletypes", "dihedraltypes", "impropertypes")
    category_order = ("WITH_BACKBONE", "WITHOUT_BACKBONE")

    for section in section_order:
        lines.append(f"[ {section} ]")
        for category in category_order:
            if section == "angletypes":
                for angle_dist in ("DIST_LE2", "DIST_GE3"):
                    keys = [key for key in merged if key[0] == section and key[1] == category and key[2] == angle_dist]
                    if not keys:
                        continue
                    lines.append(f"; {category} / {angle_dist}")
                    for key in sorted(keys, key=lambda item: item[3]):
                        for variant in merged[key]:
                            lines.append(line_from_variant(variant))
                    lines.append("")
            else:
                keys = [key for key in merged if key[0] == section and key[1] == category]
                if not keys:
                    continue
                lines.append(f"; {category}")
                for key in sorted(keys, key=lambda item: item[3]):
                    for variant in merged[key]:
                        lines.append(line_from_variant(variant))
                lines.append("")
        lines.append("")

    write_text(path, "\n".join(lines).rstrip() + "\n")


def merged_summary_payload(
    root: Path,
    merged: Dict[Tuple[str, str, str, Tuple[str, ...]], List[MergedVariant]],
    skipped: List[Dict[str, str]],
) -> Dict[str, object]:
    groups = []
    for key, variants in sorted(merged.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])):
        groups.append(
            {
                "section": key[0],
                "category": key[1],
                "angle_dist": key[2],
                "type_names": list(key[3]),
                "variant_count": len(variants),
                "selected_variant": next(index for index, variant in enumerate(variants) if variant.primary),
                "variants": [
                    {
                        "primary": variant.primary,
                        "commented": variant.commented,
                        "tokens": list(variant.tokens),
                        "display_labels": [list(entry) for entry in variant.display_labels],
                        "sources": list(variant.sources),
                        "indices_examples": [list(entry) for entry in variant.indices_examples],
                        "inline_comments": list(variant.inline_comments),
                        "rmsd_values": list(variant.rmsd_values),
                    }
                    for variant in variants
                ],
            }
        )

    return {"root": str(root), "group_count": len(groups), "groups": groups, "skipped": skipped}


_XTB_TRAJ_TO_PDB_SRC = Path(__file__).parent / "xtb_traj_to_pdb.py"


def _srun_reentry_lines(exec_cfg: Dict[str, Any], cpu_fallback_var: str) -> List[str]:
    if not (parse_bool(exec_cfg.get("slurm", False), False) and parse_bool(exec_cfg.get("use_srun", False), False)):
        return []
    return [
        "if [ -n \"${SLURM_JOB_ID:-}\" ] && [ -z \"${SLURM_STEP_ID:-}\" ]; then",
        "  if ! command -v srun >/dev/null 2>&1; then",
        "    echo \"[ERROR] execution.use_srun=true but srun was not found\" >&2",
        "    exit 1",
        "  fi",
        f"  exec srun --export=ALL --ntasks=1 --cpus-per-task \"${{SLURM_CPUS_PER_TASK:-${cpu_fallback_var}}}\" bash \"$0\" \"$@\"",
        "fi",
    ]


def _bartender_mode_args(
    flow: Dict[str, str],
    bartender_cfg: Dict[str, Any],
    bartender_charge: int,
    skip: int,
    trajectory: Optional[Path],
    outdir: Path,
) -> List[str]:
    """md 모드에 따른 Bartender CLI 인자 목록 반환 (quoting 없이 raw 값)."""
    args: List[str] = ["-charge", str(int(bartender_charge))]
    if flow["md"] == "bartender":
        args += ["-method", "gfn2",
                 "-time", str(int(bartender_cfg.get("time_ps", 5000))),
                 "-temperature", f"{float(bartender_cfg.get('temperature_k', 310.0)):.3f}"]
        solvent = str(bartender_cfg.get("solvent", "h2o")).strip()
        if solvent:
            args += ["-solvent", solvent]
        dcd_save = str(bartender_cfg.get("dcd_save", "")).strip()
        if dcd_save:
            args += ["-dcdSave", dcd_save]
        if skip > 1:
            args += ["-skip", str(skip)]
    elif flow["md"] in {"xtb", "existing"}:
        if trajectory is None:
            raise ValueError("Trajectory reuse mode requires a trajectory path.")
        trajectory_arg = os.path.relpath(str(trajectory), start=str(outdir))
        args += ["-owntraj", trajectory_arg, "-refit"]
        if skip > 1:
            args += ["-skip", str(skip)]
    else:
        raise ValueError(f"Unsupported md mode: {flow['md']}")
    return args


def prepare_relaxation_job(
    case_dir: Path,
    case: Dict[str, Any],
    flow: Dict[str, str],
    pipeline_cfg: Dict[str, Any],
    base_dir: Path,
    exec_cfg: Dict[str, Any],
) -> Optional[Path]:
    if flow["relaxation"] == "off" and flow["md"] not in {"xtb", "xtb_nobartender"}:
        case["relaxation"] = {
            "mode": flow["relaxation"],
            "md": flow["md"],
            "workdir": None,
            "run_script": None,
            "geometry_hint": str(case["artifacts"]["polymer_xyz"]),
            "trajectory_hint": None,
            "geometry_opt": False,
        }
        return None

    workdir_name = flow["workdir_name"]
    workdir = case_dir / workdir_name
    workdir.mkdir(parents=True, exist_ok=True)

    xyz_name = str(case["artifacts"]["polymer_xyz"])
    polymer_xyz = case_dir / xyz_name
    local_xyz = workdir / polymer_xyz.name
    local_xyz.write_text(polymer_xyz.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    state = case.get("electronic_state", {})
    if not isinstance(state, dict):
        raise TypeError("case.electronic_state must be a mapping")
    charge = int(state.get("charge", 0))
    uhf = int(state.get("uhf", 0))
    xtb_cfg = resolve_xtb_settings(pipeline_cfg)
    orca_cfg = resolve_orca_settings(pipeline_cfg)
    slurm_enabled = parse_bool(exec_cfg.get("slurm", False), False)
    slurm_cpu_count = _get_slurm_cpu_count() if slurm_enabled else 0
    resolved_xtb_bin = resolve_executable_command(base_dir, xtb_cfg["binary"])
    resolved_orca_bin = resolve_executable_command(base_dir, orca_cfg["binary"])
    xtb_env_script = xtb_cfg["env_script"]
    xtb_parallel = int(xtb_cfg["parallel"])
    orca_nprocs = max(1, int(orca_cfg["nprocs"]))
    if slurm_cpu_count > 0:
        xtb_parallel = slurm_cpu_count
        orca_nprocs = slurm_cpu_count
    xtb_parallel_expr = f"${{XTB_PARALLEL:-{xtb_parallel}}}"
    xtb_solvent_flags = ""
    if xtb_cfg["solvent_model"] != "off":
        xtb_solvent_flags = f" --{xtb_cfg['solvent_model']} {shlex.quote(xtb_cfg['solvent'])}"
        if xtb_cfg["solvent_reference"]:
            xtb_solvent_flags += f" {shlex.quote(xtb_cfg['solvent_reference'])}"
    xtb_common_flags = (
        f"--gfn {xtb_cfg['gfn']} "
        f"--chrg {charge} "
        f"--uhf {uhf} "
        f"--acc {xtb_cfg['acc']:.3f} "
        f"--etemp {xtb_cfg['etemp']:.3f}"
        f"{xtb_solvent_flags} "
        f"--parallel {xtb_parallel_expr}"
    )

    needs_xtb = flow["relaxation"] == "xtb" or flow["md"] in {"xtb", "xtb_nobartender"}
    needs_orca = flow["relaxation"] == "orca"
    geometry_hint = local_xyz.name
    trajectory_hint: Optional[str] = None
    geometry_opt = flow["relaxation"] == "xtb"
    thread_hint_default = max(
        xtb_parallel if needs_xtb else 1,
        orca_nprocs if needs_orca else 1,
    )

    thread_hint_expr = (
        f"${{HYGEL_THREAD_HINT:-${{SLURM_CPUS_PER_TASK:-{thread_hint_default}}}}}"
        if slurm_enabled
        else f"${{HYGEL_THREAD_HINT:-{thread_hint_default}}}"
    )

    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "cd \"$(dirname \"$0\")\"",
        f"export HYGEL_THREAD_HINT={thread_hint_expr}",
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$HYGEL_THREAD_HINT}",
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}",
        "export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}",
        "export OMP_STACKSIZE=1G",
    ]
    lines.extend(_srun_reentry_lines(exec_cfg, "HYGEL_THREAD_HINT"))
    if needs_xtb:
        lines.append("export XTB_PARALLEL=${XTB_PARALLEL:-$HYGEL_THREAD_HINT}")
    if xtb_env_script and needs_xtb:
        lines.extend(
            [
                "set +u",
                f"if [ -f {shlex.quote(xtb_env_script)} ]; then",
                f"  source {shlex.quote(xtb_env_script)}",
                "fi",
                "set -u",
            ]
        )
    if needs_orca:
        lines.append(shell_assign("ORCA_BIN", resolved_orca_bin))
    if needs_xtb:
        lines.append(shell_assign("XTB_BIN", resolved_xtb_bin))

    if flow["relaxation"] == "xtb":
        lines.append(
            f"$XTB_BIN {shlex.quote(local_xyz.name)} {xtb_common_flags} --opt {shlex.quote(xtb_cfg['opt_level'])} --cycles {xtb_cfg['opt_cycles']} > xtb_opt.out"
        )
        geometry_hint = "xtbopt.xyz"
    elif flow["relaxation"] == "orca":
        orca_template_path = resolve_optional_path(base_dir, orca_cfg["input_template_path"])
        orca_template_text = (
            orca_template_path.read_text(encoding="utf-8", errors="replace")
            if orca_template_path is not None
            else None
        )
        effective_orca_cfg = dict(orca_cfg)
        effective_orca_cfg["nprocs"] = orca_nprocs
        write_text(workdir / "relax.inp", render_orca_input(local_xyz.name, state, effective_orca_cfg, orca_template_text))
        lines.append("$ORCA_BIN relax.inp > relax.out")
        geometry_hint = "relax.xyz"
    elif flow["relaxation"] != "off":
        raise ValueError(f"Unsupported relaxation mode: {flow['relaxation']}")

    if flow["md"] in {"xtb", "xtb_nobartender"}:
        md_template_path = resolve_optional_path(base_dir, xtb_cfg["md_input_template_path"])
        md_template_text = (
            md_template_path.read_text(encoding="utf-8", errors="replace")
            if md_template_path is not None
            else None
        )
        write_text(workdir / "gochem.inp", render_xtb_md_input("nvt", xtb_cfg, md_template_text))
        shutil.copy(_XTB_TRAJ_TO_PDB_SRC, workdir / "xtb_traj_to_pdb.py")
        lines.append(
            f"$XTB_BIN {shlex.quote(geometry_hint)} {xtb_common_flags} --md --input gochem.inp > xtb_md.out"
        )
        lines.append("python3 xtb_traj_to_pdb.py xtb.trj xtb_traj.pdb")
        trajectory_hint = "xtb_traj.pdb"

    script_path = workdir / "run_relax.sh"
    write_text(script_path, "\n".join(lines) + "\n")
    script_path.chmod(0o755)
    case["relaxation"] = {
        "mode": flow["relaxation"],
        "md": flow["md"],
        "workdir": workdir_name,
        "run_script": script_path.name,
        "geometry_hint": geometry_hint,
        "trajectory_hint": trajectory_hint,
        "raw_trajectory_hint": "xtb.trj" if flow["md"] in {"xtb", "xtb_nobartender"} else None,
        "geometry_opt": geometry_opt,
    }
    return workdir


def prepare_bartender_job(
    case_dir: Path,
    case: Dict[str, Any],
    flow: Dict[str, str],
    pipeline_cfg: Dict[str, Any],
    base_dir: Path,
    exec_cfg: Dict[str, Any],
) -> Optional[Path]:
    bartender_cfg = pipeline_cfg.get("bartender", {})
    if not isinstance(bartender_cfg, dict):
        raise TypeError("bartender_pipeline.bartender must be a mapping")

    if flow["md"] in {"off", "xtb_nobartender"}:
        relax = case.get("relaxation", {})
        if not isinstance(relax, dict):
            relax = {}
        trajectory_path = None
        if flow["md"] == "xtb_nobartender":
            relax_workdir = relax.get("workdir")
            relax_trajectory = relax.get("trajectory_hint")
            if relax_workdir and relax_trajectory:
                trajectory_path = str(case_dir / str(relax_workdir) / str(relax_trajectory))
        case["bartender"] = {
            "mode": flow["md"],
            "workdir": None,
            "run_script": None,
            "geometry_source": None,
            "trajectory_source": "relaxation_output" if flow["md"] == "xtb_nobartender" else None,
            "trajectory_path": trajectory_path,
        }
        return None

    if not bartender_cfg.get("enabled", True):
        return None

    polymer_geometry = case_dir / str(case["artifacts"]["polymer_xyz"])
    relax = case.get("relaxation", {})
    if not isinstance(relax, dict):
        relax = {}
    if flow["relaxation"] == "off":
        geometry = polymer_geometry
        geometry_source = "polymer_xyz"
    else:
        relax_workdir = relax.get("workdir")
        relax_geometry = relax.get("geometry_hint")
        if not relax_workdir or not relax_geometry:
            raise ValueError("Relaxation metadata is incomplete; cannot determine Bartender geometry input.")
        geometry = case_dir / str(relax_workdir) / str(relax_geometry)
        geometry_source = "relaxation_output"

    trajectory: Optional[Path] = None
    if flow["md"] == "xtb":
        relax_workdir = relax.get("workdir")
        relax_trajectory = relax.get("trajectory_hint")
        if not relax_workdir or not relax_trajectory:
            raise ValueError("xTB reuse mode requires relaxation.trajectory_hint metadata.")
        trajectory = case_dir / str(relax_workdir) / str(relax_trajectory)
        trajectory_source = "relaxation_output"
    elif flow["md"] == "existing":
        md_traj = resolve_optional_path(base_dir, pipeline_cfg.get("md_traj"))
        if md_traj is None:
            raise ValueError("bartender_pipeline.md=existing requires bartender_pipeline.md_traj")
        trajectory = md_traj
        trajectory_source = "existing_md_traj"
    else:
        trajectory_source = None

    inp = case_dir / str(case["artifacts"]["bartender_inp"])
    if not inp.exists():
        raise FileNotFoundError(f"Bartender inp does not exist: {inp}")

    outdir = case_dir / str(bartender_cfg.get("output_dirname", "bartender_job"))
    outdir.mkdir(parents=True, exist_ok=True)
    local_inp = outdir / inp.name
    local_inp.write_text(inp.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    geometry_arg = os.path.relpath(str(geometry), start=str(outdir))
    resolved_bartender_bin = resolve_executable_command(base_dir, bartender_cfg.get("binary"))
    bartender_cpus = int(bartender_cfg.get("cpus", 1))
    slurm_enabled = parse_bool(exec_cfg.get("slurm", False), False)
    slurm_cpu_count = _get_slurm_cpu_count() if slurm_enabled else 0
    if slurm_cpu_count > 0:
        bartender_cpus = slurm_cpu_count
    state = case.get("electronic_state", {})
    if not isinstance(state, dict):
        state = {}
    bartender_charge = bartender_cfg.get("charge")
    if bartender_charge is None:
        bartender_charge = int(state.get("charge", 0))

    skip = int(bartender_cfg.get("skip", 1))
    mode_args = _bartender_mode_args(flow, bartender_cfg, bartender_charge, skip, trajectory, outdir)

    # JSON manifest: records the logical command (shell vars kept as literals)
    command = [resolved_bartender_bin, "-cpus", "$HYGEL_BARTENDER_CPUS"] + mode_args + [geometry_arg, local_inp.name]

    bt_root = str(bartender_cfg.get("root", "")).strip()
    bt_env_script = str(bartender_cfg.get("env_script", "")).strip()
    script_path = outdir / "run_bartender.sh"
    bartender_cpu_expr = (
        f"${{HYGEL_BARTENDER_CPUS:-${{SLURM_CPUS_PER_TASK:-{bartender_cpus}}}}}"
        if slurm_enabled
        else f"${{HYGEL_BARTENDER_CPUS:-{bartender_cpus}}}"
    )
    script_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "cd \"$(dirname \"$0\")\"",
        f"export HYGEL_BARTENDER_CPUS={bartender_cpu_expr}",
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$HYGEL_BARTENDER_CPUS}",
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}",
        "export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}",
        shell_assign("BARTENDER_BIN", resolved_bartender_bin),
    ]
    script_lines.extend(_srun_reentry_lines(exec_cfg, "HYGEL_BARTENDER_CPUS"))
    if bt_root:
        script_lines.append(f"export {shell_assign('BTROOT', bt_root)}")
    if bt_env_script:
        script_lines.extend(
            [
                "set +u",
                f"if [ -f {shlex.quote(bt_env_script)} ]; then",
                f"  source {shlex.quote(bt_env_script)}",
                "fi",
                "set -u",
            ]
        )
    # bash script: quoted args for safe shell expansion
    command_parts = (
        ["$BARTENDER_BIN", "-cpus", "\"$HYGEL_BARTENDER_CPUS\""]
        + [shlex.quote(a) for a in mode_args]
        + [shlex.quote(geometry_arg), shlex.quote(local_inp.name)]
    )
    script_lines.append(" ".join(command_parts))
    write_text(
        script_path,
        "\n".join(script_lines) + "\n",
    )
    script_path.chmod(0o755)

    geometry_exists = geometry.exists()
    trajectory_exists = trajectory.exists() if trajectory is not None else None
    manifest = {
        "mode": flow["md"],
        "geometry": geometry_arg,
        "inp": local_inp.name,
        "trajectory": os.path.relpath(str(trajectory), start=str(outdir)) if trajectory is not None else None,
        "command": command,
        "outdir": str(outdir),
        "geometry_source": geometry_source,
        "geometry_exists": geometry_exists,
        "trajectory_exists": trajectory_exists,
    }
    write_text(outdir / "bartender_job.json", json.dumps(manifest, indent=2))

    case.setdefault("bartender", {})
    case["bartender"]["job_dir"] = outdir.name
    case["bartender"]["run_script"] = script_path.name
    case["bartender"]["mode"] = flow["md"]
    case["bartender"]["geometry_source"] = geometry_source
    case["bartender"]["geometry_path"] = str(geometry)
    case["bartender"]["trajectory_source"] = trajectory_source
    case["bartender"]["trajectory_path"] = str(trajectory) if trajectory is not None else None

    return outdir


def collect_results(root: Path, output: Path) -> Dict[str, Any]:
    records = []
    for itp_path in sorted(root.rglob("gmx_out.itp")):
        summary = summarize_itp(itp_path)
        case_json = find_case_json(itp_path.parent)
        if case_json:
            case = json.loads(case_json.read_text(encoding="utf-8"))
            summary["case"] = {
                "sequence_stem": case.get("sequence_stem"),
                "sequence_tokens": case.get("sequence_tokens"),
                "case_json": str(case_json),
            }
        records.append(summary)
    payload = {"root": str(root), "count": len(records), "records": records}
    write_text(output, json.dumps(payload, indent=2))
    return payload


def merge_results(root: Path, output_itp: Path, output_json: Path, label_map_path: Optional[Path]) -> Dict[str, Any]:
    label_overrides = load_label_map(label_map_path) if label_map_path else {}
    records: List[TypedRecord] = []
    skipped: List[Dict[str, str]] = []
    for itp_path in sorted(root.rglob("gmx_out.itp")):
        case_path = find_case_json(itp_path.parent)
        if case_path is None:
            skipped.append({"path": str(itp_path), "reason": "case.json not found in parent chain"})
            continue
        try:
            records.extend(typed_records_for_result(itp_path, case_path, label_overrides))
        except Exception as exc:  # noqa: BLE001
            skipped.append({"path": str(itp_path), "reason": str(exc)})

    merged = merge_records(records)
    write_merged_forcefield(output_itp, merged, root=root, label_map_path=label_map_path)
    payload = merged_summary_payload(root, merged, skipped)
    write_text(output_json, json.dumps(payload, indent=2))
    return payload


def run_postprocess_only(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_dir = Path(cfg["paths"]["base_dir"]).resolve()
    out_root = Path(cfg["paths"]["out_root"]).resolve()
    if not out_root.exists():
        raise FileNotFoundError(f"Postprocess root does not exist: {out_root}")

    pipeline_cfg = cfg["bartender_pipeline"]
    post_cfg = pipeline_cfg["postprocess"]
    summary: Dict[str, Any] = {"settings": cfg, "postprocess_only": True, "out_root": str(out_root)}

    if post_cfg.get("collect", True):
        collect_path = resolve_under_base(out_root, str(post_cfg.get("collect_json", "bartender_summary.json")))
        summary["collect"] = collect_results(out_root, collect_path)
    if post_cfg.get("merge", False):
        label_map_value = post_cfg.get("label_map_path")
        label_map_path = resolve_under_base(base_dir, str(label_map_value)) if label_map_value else None
        output_itp = resolve_under_base(out_root, str(post_cfg.get("merged_itp", "merged_forcefield.itp")))
        output_json = resolve_under_base(out_root, str(post_cfg.get("merged_json", "merged_forcefield.json")))
        summary["merge"] = merge_results(out_root, output_itp, output_json, label_map_path)

    summary_name = str(post_cfg.get("summary_json", "postprocess_summary.json"))
    write_text(resolve_under_base(out_root, summary_name), json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_dir = Path(cfg["paths"]["base_dir"]).resolve()
    out_root = Path(cfg["paths"]["out_root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    pipeline_cfg = cfg["bartender_pipeline"]
    flow = resolve_pipeline_modes(pipeline_cfg)
    exec_cfg = resolve_execution_settings(pipeline_cfg)
    log_cfg = resolve_log_settings(pipeline_cfg)
    connection_cfg = resolve_connection_detection_config(pipeline_cfg)
    term_cfg = resolve_term_generation_config(pipeline_cfg)
    legacy_init_templates = pipeline_cfg.get("init_templates", {})
    if legacy_init_templates is None:
        legacy_init_templates = {}
    if not isinstance(legacy_init_templates, dict):
        raise TypeError("bartender_pipeline.init_templates must be a mapping when provided")

    monomer_cfg = normalize_monomer_configs(cfg["monomers"], legacy_init_templates)
    monomer_paths = {
        token: str(resolve_under_base(base_dir, entry["xyz"]))
        for token, entry in monomer_cfg.items()
    }
    library = load_monomer_library(monomer_paths, base_dir=base_dir)
    monomer_keys = set(library.keys())

    init_templates = {
        token: resolve_under_base(base_dir, entry["init_template"])
        for token, entry in monomer_cfg.items()
        if entry.get("init_template")
    }
    sequence_jobs = build_sequence_jobs(cfg["system"], monomer_keys)

    template_cache: Dict[str, MonomerTemplate] = {}
    metadata_cache: Dict[str, ConnectionMetadata] = {}
    validation_cache: Dict[str, ValidationReport] = {}

    cases: List[Dict[str, Any]] = []
    for tokens in sequence_jobs:
        for token in tokens:
            if token not in monomer_paths:
                raise KeyError(f"Unknown monomer token: {token}")
            if token not in init_templates:
                raise KeyError(f"Missing init template for monomer token: {token}")
            if token not in template_cache:
                template = parse_bartender_inp(init_templates[token])
                template_cache[token] = template
                validation_cache[token] = validate_template(template, Path(monomer_paths[token]), connection_cfg)
                metadata_cache[token] = infer_connection_metadata(
                    template,
                    Path(monomer_paths[token]),
                    connection_cfg,
                    monomer_cfg[token]["backbone_atoms"],
                )

        stem = sequence_stem(tokens)
        case_dir = out_root / stem
        case_dir.mkdir(parents=True, exist_ok=True)
        torsion_mode = str(cfg["system"].get("n_torsion_mode", "repeat"))
        torsion = len(tokens) if torsion_mode == "repeat" else max(1, len(tokens) - 1)

        builder_tmp = case_dir / "_builder_tmp"
        if builder_tmp.exists():
            shutil.rmtree(builder_tmp, ignore_errors=True)
        builder_tmp.mkdir(parents=True, exist_ok=True)
        build_polymer_structure(
            tokens,
            monomer_dict=library,
            n_torsion=torsion,
            output_filename=f"{stem}.xyz",
            output_dir=builder_tmp,
        )

        built_xyz = builder_tmp / f"{stem}.xyz"
        final_xyz = case_dir / f"{stem}.xyz"
        if not built_xyz.exists():
            raise FileNotFoundError(f"param_opt builder did not produce {built_xyz}")
        shutil.copyfile(built_xyz, final_xyz)
        if builder_tmp.exists():
            shutil.rmtree(builder_tmp, ignore_errors=True)

        bundle = build_polymer_input(tokens, final_xyz, template_cache, metadata_cache, term_cfg)
        base_inp = case_dir / f"{stem}_base.inp"
        final_inp = case_dir / f"{stem}_bartender.inp"
        write_text(base_inp, bundle.base_text)
        write_text(final_inp, bundle.augmented_text)
        logs_dir = ensure_case_logs_dir(case_dir, log_cfg)

        monomer_validation_text = []
        has_failure = False
        for token in sorted(set(tokens)):
            report = validation_cache[token]
            monomer_validation_text.append(f"[{token}] {report.target}")
            monomer_validation_text.append(report.render().strip())
            monomer_validation_text.append("")
            if not report.ok:
                has_failure = True

        if logs_dir is not None and log_cfg.get("write_validation", True):
            write_text(logs_dir / "monomer_validation.txt", "\n".join(monomer_validation_text).rstrip() + "\n")
            write_text(logs_dir / "polymer_base_validation.txt", bundle.base_report.render())
            write_text(logs_dir / "polymer_augmented_validation.txt", bundle.augmented_report.render())
        if not bundle.base_report.ok or not bundle.augmented_report.ok:
            has_failure = True

        electronic_state = resolve_case_electronic_state(tokens, monomer_cfg, pipeline_cfg)

        case: Dict[str, Any] = {
            "sequence_tokens": tokens,
            "sequence_stem": stem,
            "torsion": torsion,
            "workflow_mode": flow,
            "artifacts": {
                "polymer_xyz": final_xyz.name,
                "base_inp": base_inp.name,
                "bartender_inp": final_inp.name,
            },
            "polymer_xyz": str(final_xyz),
            "base_inp": str(base_inp),
            "bartender_inp": str(final_inp),
            "electronic_state": electronic_state,
            "term_generation": {
                "mode": term_cfg.mode,
                "n": term_cfg.n,
            },
            "connection_detection": {
                "indicator": connection_cfg.indicator,
                "cutoff": connection_cfg.cutoff,
            },
            "bead_specs": {
                token: default_bead_spec(token, template_cache[token].bead_count)
                for token in sorted(set(tokens))
            },
            "monomers": {
                token: {
                    "xyz": monomer_paths[token],
                    "init_inp": str(init_templates[token]),
                    "head_br": metadata_cache[token].head_br,
                    "tail_br": metadata_cache[token].tail_br,
                    "left_connection_bead": metadata_cache[token].left_connection_bead,
                    "right_connection_bead": metadata_cache[token].right_connection_bead,
                    "backbone_atoms": export_backbone_atom_config(monomer_cfg[token]["backbone_atoms"]),
                    "backbone_beads": list(metadata_cache[token].backbone_beads),
                    "bead_count": template_cache[token].bead_count,
                    "charge": monomer_cfg[token]["charge"],
                    "uhf": monomer_cfg[token]["uhf"],
                    "multiplicity": monomer_cfg[token]["multiplicity"],
                }
                for token in sorted(set(tokens))
            },
            "connection_bonds": bundle.connection_bonds,
            "connection_beads": bundle.connection_beads,
            "backbone_beads": bundle.backbone_beads,
            "reports": {
                "monomer_validation_ok": all(validation_cache[token].ok for token in sorted(set(tokens))),
                "base_validation_ok": bundle.base_report.ok,
                "augmented_validation_ok": bundle.augmented_report.ok,
            },
            "execution": {
                "run_relaxation": exec_cfg["run_relaxation"],
                "run_bartender": exec_cfg["run_bartender"],
                "shell": exec_cfg["shell"],
                "slurm": exec_cfg["slurm"],
                "use_srun": exec_cfg["use_srun"],
            },
            "logs": {
                "enabled": log_cfg["enabled"],
                "dir": str(logs_dir) if logs_dir is not None else None,
                "write_validation": log_cfg["write_validation"],
                "capture_runtime": log_cfg["capture_runtime"],
            },
        }

        if has_failure and not pipeline_cfg.get("allow_invalid", False):
            write_text(case_dir / "case.json", json.dumps(case, indent=2, ensure_ascii=False))
            raise ValueError(f"Validation failed for case {stem}. See {case_dir}.")

        relax_dir = prepare_relaxation_job(case_dir, case, flow, pipeline_cfg, base_dir, exec_cfg)
        bartender_dir = prepare_bartender_job(case_dir, case, flow, pipeline_cfg, base_dir, exec_cfg)
        write_text(case_dir / "case.json", json.dumps(case, indent=2, ensure_ascii=False))

        if exec_cfg["run_relaxation"] and relax_dir is not None:
            relax_script = relax_dir / str(case["relaxation"]["run_script"])
            case["execution"]["relaxation"] = execute_case_script(
                "relaxation",
                relax_script,
                relax_dir,
                {**exec_cfg, "capture_runtime": log_cfg["capture_runtime"]},
                logs_dir,
            )
            write_text(case_dir / "case.json", json.dumps(case, indent=2, ensure_ascii=False))

        if exec_cfg["run_bartender"] and bartender_dir is not None:
            bartender_meta = case.get("bartender", {})
            if not isinstance(bartender_meta, dict):
                bartender_meta = {}
            geometry_path = bartender_meta.get("geometry_path")
            trajectory_path = bartender_meta.get("trajectory_path")
            if geometry_path and not Path(str(geometry_path)).exists():
                raise FileNotFoundError(f"Bartender geometry source does not exist yet: {geometry_path}")
            if trajectory_path and not Path(str(trajectory_path)).exists():
                raise FileNotFoundError(f"Bartender owntraj source does not exist yet: {trajectory_path}")

            bartender_script = bartender_dir / str(case["bartender"]["run_script"])
            case["execution"]["bartender"] = execute_case_script(
                "bartender",
                bartender_script,
                bartender_dir,
                {**exec_cfg, "capture_runtime": log_cfg["capture_runtime"]},
                logs_dir,
            )
            gmx_out = bartender_dir / "gmx_out.itp"
            if gmx_out.exists() and logs_dir is not None:
                write_text(logs_dir / "gmx_out_summary.json", json.dumps(summarize_itp(gmx_out), indent=2))
            write_text(case_dir / "case.json", json.dumps(case, indent=2, ensure_ascii=False))

        cases.append(
            {
                "sequence_stem": stem,
                "sequence_tokens": tokens,
                "case_dir": str(case_dir),
                "polymer_xyz": str(final_xyz),
                "bartender_inp": str(final_inp),
                "relaxation_dir": str(relax_dir) if relax_dir else None,
                "bartender_dir": str(bartender_dir) if bartender_dir else None,
            }
        )

    summary: Dict[str, Any] = {"settings": cfg, "cases": cases}

    post_cfg = pipeline_cfg["postprocess"]
    if post_cfg.get("collect", True):
        collect_path = resolve_under_base(out_root, str(post_cfg.get("collect_json", "bartender_summary.json")))
        summary["collect"] = collect_results(out_root, collect_path)
    if post_cfg.get("merge", False):
        label_map_value = post_cfg.get("label_map_path")
        label_map_path = resolve_under_base(base_dir, str(label_map_value)) if label_map_value else None
        output_itp = resolve_under_base(out_root, str(post_cfg.get("merged_itp", "merged_forcefield.itp")))
        output_json = resolve_under_base(out_root, str(post_cfg.get("merged_json", "merged_forcefield.json")))
        summary["merge"] = merge_results(out_root, output_itp, output_json, label_map_path)

    write_text(out_root / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
