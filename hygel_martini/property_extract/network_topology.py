"""Graph-level audits for covalently crosslinked hydrogel networks.

The builder's atom-level connectivity audit establishes that a polymer is one
connected molecule.  Mechanics additionally needs a reduced network graph:
crosslinker moieties are junctions and polymer components between them are
strands.  This module reports loops, parallel strands, dangling/bridge defects,
the two-core, and (when a GRO file is supplied) periodic winding.

The reduced graph is a structural diagnostic.  Calling a strand "in the
two-core" does not by itself prove that it contributes the classical ``kBT/V``
to an experimental or simulated equilibrium modulus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

from hygel_martini.core.gro import read_gro

import numpy as np


def _read_itp_atoms_bonds(
    path: str | Path,
) -> tuple[dict[int, dict[str, str]], list[tuple[int, int]]]:
    """Read the ``[ atoms ]`` and ``[ bonds ]`` sections of a standalone ITP."""
    atoms: dict[int, dict[str, str]] = {}
    bonds: list[tuple[int, int]] = []
    section = ""
    for line_number, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and "]" in line:
            section = line[1 : line.index("]")].strip().lower()
            continue
        fields = line.split()
        try:
            if section == "atoms":
                if len(fields) < 5:
                    raise ValueError("fewer than five atom fields")
                atom_id = int(fields[0])
                atoms[atom_id] = {
                    "type": fields[1],
                    "resnr": fields[2],
                    "residue": fields[3],
                    "name": fields[4],
                }
            elif section == "bonds":
                if len(fields) < 2:
                    raise ValueError("fewer than two bond fields")
                bonds.append((int(fields[0]), int(fields[1])))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: malformed {section} line") from exc
    if not atoms:
        raise ValueError(f"{path}: no [ atoms ] entries")
    if not bonds:
        raise ValueError(f"{path}: no [ bonds ] entries")
    unknown = {atom for bond in bonds for atom in bond if atom not in atoms}
    if unknown:
        raise ValueError(f"{path}: bonds reference unknown atom IDs {sorted(unknown)[:5]}")
    return atoms, bonds


def _components(nodes: Iterable[int], adjacency: dict[int, set[int]]) -> list[set[int]]:
    remaining = set(nodes)
    output: list[set[int]] = []
    while remaining:
        start = min(remaining)
        component = {start}
        queue = [start]
        remaining.remove(start)
        while queue:
            node = queue.pop()
            for neighbor in adjacency[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        output.append(component)
    return output


def _reduced_components(
    n_nodes: int, edges: list[tuple[int, int]]
) -> list[set[int]]:
    adjacency = {node: set() for node in range(n_nodes)}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return _components(range(n_nodes), adjacency)


def _multigraph_degrees(
    n_nodes: int, edges: list[tuple[int, int]]
) -> list[int]:
    degrees = [0] * n_nodes
    for left, right in edges:
        if left == right:
            degrees[left] += 2
        else:
            degrees[left] += 1
            degrees[right] += 1
    return degrees


def _two_core(
    n_nodes: int, edges: list[tuple[int, int]]
) -> tuple[set[int], set[int]]:
    incident: dict[int, set[int]] = {node: set() for node in range(n_nodes)}
    for edge_id, (left, right) in enumerate(edges):
        incident[left].add(edge_id)
        incident[right].add(edge_id)
    active_nodes = set(range(n_nodes))
    active_edges = set(range(len(edges)))

    def degree(node: int) -> int:
        value = 0
        for edge_id in incident[node] & active_edges:
            left, right = edges[edge_id]
            value += 2 if left == right == node else 1
        return value

    queue = deque(node for node in active_nodes if degree(node) < 2)
    while queue:
        node = queue.popleft()
        if node not in active_nodes or degree(node) >= 2:
            continue
        active_nodes.remove(node)
        removed = incident[node] & active_edges
        neighbors: set[int] = set()
        for edge_id in removed:
            left, right = edges[edge_id]
            neighbors.update((left, right))
        active_edges.difference_update(removed)
        for neighbor in neighbors:
            if neighbor in active_nodes and degree(neighbor) < 2:
                queue.append(neighbor)
    return active_nodes, active_edges


def _bridge_edges(n_nodes: int, edges: list[tuple[int, int]]) -> set[int]:
    """Return multigraph bridge IDs using edge-aware Tarjan traversal."""
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge_id, (left, right) in enumerate(edges):
        adjacency[left].append((right, edge_id))
        adjacency[right].append((left, edge_id))
    discovery = [-1] * n_nodes
    low = [-1] * n_nodes
    bridges: set[int] = set()
    clock = 0

    def visit(node: int, parent_edge: int | None) -> None:
        nonlocal clock
        discovery[node] = low[node] = clock
        clock += 1
        for neighbor, edge_id in adjacency[node]:
            if edge_id == parent_edge:
                continue
            if discovery[neighbor] < 0:
                visit(neighbor, edge_id)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovery[node]:
                    bridges.add(edge_id)
            else:
                low[node] = min(low[node], discovery[neighbor])

    for node in range(n_nodes):
        if discovery[node] < 0:
            visit(node, None)
    return bridges


def _shortest_path(
    start: int,
    end: int,
    allowed: set[int],
    adjacency: dict[int, set[int]],
) -> list[int]:
    if start == end:
        return [start]
    parents: dict[int, int | None] = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in allowed or neighbor in parents:
                continue
            parents[neighbor] = node
            if neighbor == end:
                path = [end]
                while path[-1] != start:
                    path.append(parents[path[-1]])  # type: ignore[arg-type]
                return list(reversed(path))
            queue.append(neighbor)
    raise ValueError(f"no bonded path between atoms {start} and {end}")


def _read_gro(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates and periodic cell, via the shared reader."""
    frame = read_gro(path)
    if frame.box is None:
        raise ValueError(f"{path}: GRO file has no box line")
    return frame.positions, frame.box


def _canonical_winding(vector: np.ndarray) -> tuple[int, int, int]:
    values = tuple(int(value) for value in vector)
    for value in values:
        if value < 0:
            return tuple(-item for item in values)
        if value > 0:
            return values
    return values


def _periodic_winding(
    coordinates: np.ndarray,
    box: np.ndarray,
    adjacency: dict[int, set[int]],
    junction_components: list[set[int]],
    strands: list[dict[str, object]],
    n_itp_atoms: int,
) -> dict[str, object]:
    if coordinates.shape[0] < n_itp_atoms:
        raise ValueError(
            "GRO contains fewer atoms than the audited ITP; atom ordering cannot match"
        )
    inverse_box = np.linalg.inv(box)
    representatives = [min(component) for component in junction_components]

    def minimum_image_fractional(left: int, right: int) -> np.ndarray:
        delta = (coordinates[right - 1] - coordinates[left - 1]) @ inverse_box
        return delta - np.round(delta)

    shifted_edges: list[tuple[int, int, np.ndarray]] = []
    for strand in strands:
        attachments = strand["attachments"]
        if not isinstance(attachments, list) or len(attachments) != 2:
            continue
        first, second = attachments
        left_junction, left_atom, left_strand_atom = first
        right_junction, right_atom, right_strand_atom = second
        if (left_junction, left_atom) > (right_junction, right_atom):
            (
                left_junction,
                left_atom,
                left_strand_atom,
                right_junction,
                right_atom,
                right_strand_atom,
            ) = (
                right_junction,
                right_atom,
                right_strand_atom,
                left_junction,
                left_atom,
                left_strand_atom,
            )
        left_rep = representatives[left_junction]
        right_rep = representatives[right_junction]
        left_path = _shortest_path(
            left_rep,
            left_atom,
            junction_components[left_junction],
            adjacency,
        )
        strand_path = _shortest_path(
            left_strand_atom,
            right_strand_atom,
            strand["atoms"],  # type: ignore[arg-type]
            adjacency,
        )
        right_path = _shortest_path(
            right_atom,
            right_rep,
            junction_components[right_junction],
            adjacency,
        )
        full_path = (
            left_path
            + [left_strand_atom]
            + strand_path[1:]
            + [right_atom]
            + right_path[1:]
        )
        unwrapped_delta = sum(
            (
                minimum_image_fractional(left, right)
                for left, right in zip(full_path, full_path[1:])
            ),
            np.zeros(3),
        )
        direct_delta = (
            coordinates[right_rep - 1] - coordinates[left_rep - 1]
        ) @ inverse_box
        shift = np.rint(unwrapped_delta - direct_delta).astype(int)
        residual = unwrapped_delta - direct_delta - shift
        if np.max(np.abs(residual)) > 1.0e-5:
            raise ValueError(
                "could not assign an integer periodic image shift; "
                f"maximum residual is {np.max(np.abs(residual)):.3g}"
            )
        shifted_edges.append((left_junction, right_junction, shift))

    graph: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for left, right, shift in shifted_edges:
        graph[left].append((right, shift))
        graph[right].append((left, -shift))
    potentials: dict[int, np.ndarray] = {}
    winding_vectors: set[tuple[int, int, int]] = set()
    for start in range(len(junction_components)):
        if start in potentials:
            continue
        potentials[start] = np.zeros(3, dtype=int)
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor, shift in graph[node]:
                candidate = potentials[node] + shift
                if neighbor not in potentials:
                    potentials[neighbor] = candidate
                    queue.append(neighbor)
                else:
                    winding = candidate - potentials[neighbor]
                    if np.any(winding):
                        winding_vectors.add(_canonical_winding(winding))
    winding_array = np.asarray(sorted(winding_vectors), dtype=float)
    rank = int(np.linalg.matrix_rank(winding_array)) if winding_vectors else 0
    spans = (
        np.any(winding_array != 0.0, axis=0)
        if winding_vectors
        else np.zeros(3, dtype=bool)
    )
    return {
        "box_volume_nm3": float(np.linalg.det(box)),
        "winding_rank": rank,
        "spans_x": bool(spans[0]),
        "spans_y": bool(spans[1]),
        "spans_z": bool(spans[2]),
        "winding_vectors": [list(vector) for vector in sorted(winding_vectors)],
    }


def audit_reduced_network(
    itp: str | Path,
    gro: str | Path | None = None,
    junction_residue: str = "BCK",
) -> dict[str, object]:
    """Collapse an atomistic/CG ITP into a junction--strand multigraph.

    A junction is a bonded component of atoms whose residue name equals
    ``junction_residue``.  Removing those atoms must leave polymer components
    with exactly two junction attachment bonds for them to count as valid
    strands.  This convention matches the PEGDA builder topology used by the
    Q-series and deliberately fails visibly for architectures requiring a
    different reduction rule.
    """
    itp_path = Path(itp)
    atoms, bonds = _read_itp_atoms_bonds(itp_path)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    junction_atoms = {
        atom_id
        for atom_id, atom in atoms.items()
        if atom["residue"] == junction_residue
    }
    if not junction_atoms:
        raise ValueError(
            f"{itp_path}: no atoms with junction residue {junction_residue!r}"
        )
    junction_adjacency = {
        atom: adjacency[atom] & junction_atoms for atom in junction_atoms
    }
    junction_components = _components(junction_atoms, junction_adjacency)
    junction_components.sort(key=min)
    atom_to_junction = {
        atom: junction_id
        for junction_id, component in enumerate(junction_components)
        for atom in component
    }

    nonjunction_atoms = set(atoms) - junction_atoms
    nonjunction_adjacency = {
        atom: adjacency[atom] & nonjunction_atoms for atom in nonjunction_atoms
    }
    strand_components = _components(nonjunction_atoms, nonjunction_adjacency)
    strand_components.sort(key=min)
    strands: list[dict[str, object]] = []
    malformed: list[dict[str, object]] = []
    reduced_edges: list[tuple[int, int]] = []
    strand_sizes: list[int] = []
    for component in strand_components:
        attachments = sorted(
            (
                atom_to_junction[junction_atom],
                junction_atom,
                strand_atom,
            )
            for strand_atom in component
            for junction_atom in adjacency[strand_atom] & junction_atoms
        )
        record: dict[str, object] = {
            "atoms": component,
            "attachments": attachments,
        }
        if len(attachments) != 2:
            malformed.append(
                {
                    "first_atom": min(component),
                    "atom_count": len(component),
                    "attachment_bond_count": len(attachments),
                    "junction_ids": sorted({item[0] for item in attachments}),
                }
            )
            continue
        left, right = attachments[0][0], attachments[1][0]
        reduced_edges.append((left, right))
        strand_sizes.append(len(component))
        strands.append(record)

    n_junctions = len(junction_components)
    components = _reduced_components(n_junctions, reduced_edges)
    degrees = _multigraph_degrees(n_junctions, reduced_edges)
    multiplicities = Counter(tuple(sorted(edge)) for edge in reduced_edges)
    two_core_nodes, two_core_edges = _two_core(n_junctions, reduced_edges)
    bridges = _bridge_edges(n_junctions, reduced_edges)
    canonical_edges = sorted(tuple(sorted(edge)) for edge in reduced_edges)
    fingerprint_text = "\n".join(f"{left},{right}" for left, right in canonical_edges)
    canonical_atom_bonds = sorted(tuple(sorted(bond)) for bond in bonds)
    atom_bond_text = "\n".join(
        f"{left},{right}" for left, right in canonical_atom_bonds
    )
    attachment_bonds = sorted(
        tuple(sorted((left, right)))
        for left, right in bonds
        if (left in junction_atoms) != (right in junction_atoms)
    )
    attachment_text = "\n".join(
        f"{left},{right}" for left, right in attachment_bonds
    )
    result: dict[str, object] = {
        "itp": str(itp_path),
        "junction_residue": junction_residue,
        "atom_count": len(atoms),
        "bond_count": len(bonds),
        "junction_atom_count": len(junction_atoms),
        "junction_count": n_junctions,
        "junction_atom_count_distribution": dict(
            sorted(Counter(map(len, junction_components)).items())
        ),
        "valid_strand_count": len(reduced_edges),
        "malformed_strand_component_count": len(malformed),
        "malformed_strand_components": malformed,
        "strand_atom_count_distribution": dict(
            sorted(Counter(strand_sizes).items())
        ),
        "junction_degree_distribution": dict(sorted(Counter(degrees).items())),
        "mean_junction_degree": (
            float(np.mean(degrees)) if degrees else float("nan")
        ),
        "self_loop_count": sum(left == right for left, right in reduced_edges),
        "parallel_junction_pair_count": sum(
            multiplicity > 1 for multiplicity in multiplicities.values()
        ),
        "parallel_strand_excess": sum(
            multiplicity - 1 for multiplicity in multiplicities.values()
        ),
        "maximum_edge_multiplicity": max(multiplicities.values(), default=0),
        "reduced_component_count": len(components),
        "largest_reduced_component_junctions": max(map(len, components), default=0),
        "two_core_junction_count": len(two_core_nodes),
        "two_core_strand_count": len(two_core_edges),
        "bridge_strand_count": len(bridges),
        "cycle_rank": len(reduced_edges) - n_junctions + len(components),
        "classical_phantom_coefficient_strands_minus_junctions": (
            len(reduced_edges) - n_junctions
        ),
        "junction_attachment_bond_count": len(attachment_bonds),
        "atom_bond_connectivity_sha256": hashlib.sha256(
            atom_bond_text.encode()
        ).hexdigest(),
        "junction_attachment_sha256": hashlib.sha256(
            attachment_text.encode()
        ).hexdigest(),
        "reduced_labeled_edge_sha256": hashlib.sha256(
            fingerprint_text.encode()
        ).hexdigest(),
        "reduced_edges": [list(edge) for edge in canonical_edges],
    }
    if gro is not None:
        coordinates, box = _read_gro(gro)
        result["gro"] = str(Path(gro))
        result["periodic"] = _periodic_winding(
            coordinates,
            box,
            adjacency,
            junction_components,
            strands,
            len(atoms),
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("itp", type=Path)
    parser.add_argument("--gro", type=Path)
    parser.add_argument("--junction-residue", default="BCK")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    result = audit_reduced_network(
        arguments.itp,
        arguments.gro,
        junction_residue=arguments.junction_residue,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
