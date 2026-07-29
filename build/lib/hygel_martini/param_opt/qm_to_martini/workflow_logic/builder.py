from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from collections import OrderedDict

from ..config import (
    MonomerTemplate,
    PolymerInputBundle,
    TermGenerationConfig,
    ValidationReport,
    WeightedAtomRef,
    normalize_sequence,
    sequence_stem,
    parse_xyz
)
from .core import _sorted_pair, _canon_angle, _canon_reversible, _build_graph, _generate_all_reversible_combinations
from .loader import validate_generated_input

def _parse_main_itp(itp_path: Path) -> Dict[str, List[Tuple[int, ...]]]:
    """Parse {label}_topology_n0_main_bonds.itp into section → list of index tuples."""
    result: Dict[str, List[Tuple[int, ...]]] = {
        "bonds": [], "constraints": [], "angles": [], "dihedrals": [], "impropers": []
    }
    section_map = {
        "bonds": "bonds", "constraints": "constraints",
        "angles": "angles", "dihedrals": "dihedrals", "impropers": "impropers",
    }
    current: Optional[str] = None
    with open(itp_path) as f:
        for raw in f:
            line = raw.split(";")[0].strip()
            if not line:
                continue
            m = re.match(r"^\[\s*(\w+)\s*\]$", line)
            if m:
                current = section_map.get(m.group(1).lower())
                continue
            if current is None:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                # ITP lines: idx idx [funct] [params...]  — take only leading integer indices
                indices: List[int] = []
                for p in parts:
                    try:
                        indices.append(int(p))
                    except ValueError:
                        break
                if len(indices) >= 2:
                    result[current].append(tuple(indices[:_section_arity(current)]))
            except Exception:
                continue
    return result


def _parse_candidates_tsv(tsv_path: Path) -> Dict[str, List[Tuple[int, ...]]]:
    """Parse *_force_sorted_candidates.tsv into section → list of index tuples."""
    result: Dict[str, List[Tuple[int, ...]]] = {
        "bonds": [], "angles": [], "dihedrals": [], "impropers": []
    }
    section_col_map = {"bonds": 2, "angles": 2, "dihedrals": 2, "impropers": 2}
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sec = row.get("section", "").lower()
            if sec not in result:
                continue
            indices_str = row.get("indices", "")
            try:
                indices = tuple(int(x) for x in indices_str.split("-"))
                if len(indices) == _section_arity(sec):
                    result[sec].append(indices)
            except (ValueError, TypeError):
                continue
    return result


def _section_arity(section: str) -> int:
    return {"bonds": 2, "constraints": 2, "angles": 3, "dihedrals": 4, "impropers": 4}.get(section, 2)


def _map_terms_to_global(
    terms_by_section: Dict[str, List[Tuple[int, ...]]],
    n_monomers: int,
    bead_count_per_monomer: int,
) -> Dict[str, List[Tuple[int, ...]]]:
    """Replicate monomer-local terms for each monomer using sequential bead offsets."""
    result: Dict[str, List[Tuple[int, ...]]] = {sec: [] for sec in terms_by_section}
    for k in range(n_monomers):
        offset = k * bead_count_per_monomer
        for sec, terms in terms_by_section.items():
            for term in terms:
                result[sec].append(tuple(idx + offset for idx in term))
    return result


def _term_spans_connection(
    term: Sequence[int],
    connection_bond_set: Set[Tuple[int, int]],
) -> bool:
    """Return True if at least one consecutive edge pair in term is a connection bond."""
    for i in range(len(term) - 1):
        pair = _sorted_pair(int(term[i]), int(term[i + 1]))
        if pair in connection_bond_set:
            return True
    return False


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

def _distance_cache(graph: Dict[int, set[int]]):
    from .core import shortest_path_len
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
    distance_lookup,
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
    distance_lookup,
    *,
    allow_swaps: bool,
) -> Optional[int]:
    from itertools import permutations
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

def _generate_augmented_terms(
    base: MonomerTemplate,
    term_cfg: TermGenerationConfig,
    backbone_beads: Sequence[int],
    connection_bond_set: Optional[Set[Tuple[int, int]]] = None,
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
            "dihedral", new_dihedrals, graph, term_cfg.n, allow_swaps=allow_swaps,
        )
        new_impropers = _filter_topology_terms(
            "improper", new_impropers, graph, term_cfg.n, allow_swaps=allow_swaps,
        )

    if term_cfg.mode == "polymer_backbone":
        backbone_set = {int(value) for value in backbone_beads}
        new_bonds = [tuple(t) for t in _filter_connection_proxy_terms(new_bonds, backbone_set, minimum_distinct=2)]
        new_angles = [tuple(t) for t in _filter_connection_proxy_terms(new_angles, backbone_set, minimum_distinct=2)]
        new_dihedrals = [tuple(t) for t in _filter_connection_proxy_terms(new_dihedrals, backbone_set, minimum_distinct=2)]
        new_impropers = [tuple(t) for t in _filter_connection_proxy_terms(new_impropers, backbone_set, minimum_distinct=2)]

    if term_cfg.mode in {"polymer_n", "polymer_swap_n"}:
        cbs: Set[Tuple[int, int]] = connection_bond_set or set()
        graph = _build_graph(list(base.bonds) + list(base.constraints))
        allow_swaps = term_cfg.mode == "polymer_swap_n"
        # topology cost filter using full polymer graph
        new_bonds = _filter_topology_terms("bond", new_bonds, graph, term_cfg.n, allow_swaps=allow_swaps)
        new_angles = _filter_topology_terms("angle", new_angles, graph, term_cfg.n, allow_swaps=allow_swaps)
        new_dihedrals = _filter_topology_terms("dihedral", new_dihedrals, graph, term_cfg.n, allow_swaps=allow_swaps)
        new_impropers = _filter_topology_terms("improper", new_impropers, graph, term_cfg.n, allow_swaps=allow_swaps)
        # keep only terms that span at least one inter-monomer connection bond
        new_bonds = [t for t in new_bonds if _term_spans_connection(t, cbs)]
        new_angles = [t for t in new_angles if _term_spans_connection(t, cbs)]
        new_dihedrals = [t for t in new_dihedrals if _term_spans_connection(t, cbs)]
        new_impropers = [t for t in new_impropers if _term_spans_connection(t, cbs)]

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

def build_polymer_input(
    sequence: Sequence[str] | str,
    polymer_xyz_path: Path,
    templates: Dict[str, MonomerTemplate],
    metadata,
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

    connection_bond_set: Set[Tuple[int, int]] = {
        _sorted_pair(a, b) for a, b in connection_bonds
    }

    # For polymer_n / polymer_swap_n: inject main ITP + additional candidate terms
    # into the augmented base (globally indexed per monomer) before backbone exploration.
    polymer_extra_bonds: List[Tuple[int, int]] = []
    polymer_extra_angles: List[Tuple[int, int, int]] = []
    polymer_extra_dihedrals: List[Tuple[int, int, int, int]] = []
    polymer_extra_impropers: List[Tuple[int, int, int, int]] = []

    if term_cfg.mode in {"polymer_n", "polymer_swap_n"}:
        n_monomers = len(tokens)
        # Determine per-token bead count from the already-built base
        token_bead_counts: List[int] = [templates[tok].bead_count for tok in tokens]

        # Collect unique token types and load their ITP/TSV files once
        main_itp_dir = Path(term_cfg.main_itp_dir)  # type: ignore[arg-type]
        candidates_tsv_dir = Path(term_cfg.candidates_tsv_dir)  # type: ignore[arg-type]

        loaded_main: Dict[str, Dict[str, List[Tuple[int, ...]]]] = {}
        loaded_candidates: Dict[str, Dict[str, List[Tuple[int, ...]]]] = {}
        for tok in set(tokens):
            itp_path = main_itp_dir / f"{tok}_topology_n0_main_bonds.itp"
            tsv_path = candidates_tsv_dir / f"{tok}_swap_n2_force_sorted_candidates.tsv"
            if itp_path.exists():
                loaded_main[tok] = _parse_main_itp(itp_path)
            else:
                report.warnings.append(f"polymer_n: main ITP not found: {itp_path}")
                loaded_main[tok] = {}
            if tsv_path.exists():
                loaded_candidates[tok] = _parse_candidates_tsv(tsv_path)
            else:
                report.warnings.append(f"polymer_n: candidates TSV not found: {tsv_path}")
                loaded_candidates[tok] = {}

        # Map each monomer's local terms to global bead indices
        cumulative_offset = 0
        existing_bonds_set = {_sorted_pair(a, b) for a, b in base.bonds}
        existing_angles_set = {_canon_angle(a, b, c) for a, b, c in base.angles}
        existing_dihedrals_set = {_canon_reversible(t) for t in base.dihedrals}
        existing_impropers_set = {_canon_reversible(t) for t in base.impropers}

        for k, tok in enumerate(tokens):
            offset = cumulative_offset
            for src in (loaded_main.get(tok, {}), loaded_candidates.get(tok, {})):
                for b_local in src.get("bonds", []):
                    b_global = _sorted_pair(b_local[0] + offset, b_local[1] + offset)
                    if b_global not in existing_bonds_set:
                        polymer_extra_bonds.append(b_global)
                        existing_bonds_set.add(b_global)
                for b_local in src.get("constraints", []):
                    b_global = _sorted_pair(b_local[0] + offset, b_local[1] + offset)
                    if b_global not in existing_bonds_set:
                        polymer_extra_bonds.append(b_global)
                        existing_bonds_set.add(b_global)
                for a_local in src.get("angles", []):
                    a_global = (a_local[0] + offset, a_local[1] + offset, a_local[2] + offset)
                    canon = _canon_angle(*a_global)
                    if canon not in existing_angles_set:
                        polymer_extra_angles.append(a_global)
                        existing_angles_set.add(canon)
                for d_local in src.get("dihedrals", []):
                    d_global = tuple(x + offset for x in d_local)
                    canon = _canon_reversible(d_global)
                    if canon not in existing_dihedrals_set:
                        polymer_extra_dihedrals.append(d_global)  # type: ignore[arg-type]
                        existing_dihedrals_set.add(canon)
                for i_local in src.get("impropers", []):
                    i_global = tuple(x + offset for x in i_local)
                    canon = _canon_reversible(i_global)
                    if canon not in existing_impropers_set:
                        polymer_extra_impropers.append(i_global)  # type: ignore[arg-type]
                        existing_impropers_set.add(canon)
            cumulative_offset += token_bead_counts[k]

    new_bonds, new_angles, new_dihedrals, new_impropers = _generate_augmented_terms(
        base,
        term_cfg,
        backbone_beads,
        connection_bond_set=connection_bond_set,
    )

    augmented = MonomerTemplate(
        path=polymer_xyz_path.with_suffix(".inp"),
        preamble=list(base.preamble),
        beads=base.beads,
        bonds=list(base.bonds) + polymer_extra_bonds + new_bonds,
        constraints=list(base.constraints),
        angles=list(base.angles) + polymer_extra_angles + new_angles,
        dihedrals=list(base.dihedrals) + polymer_extra_dihedrals + new_dihedrals,
        impropers=list(base.impropers) + polymer_extra_impropers + new_impropers,
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
