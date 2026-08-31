"""Cyclic-topology measurement on a reduced junction--strand multigraph.

``network_topology.audit_reduced_network`` answers whether a network is
connected, what its junction degrees are, and whether it has self-loops or
parallel strands.  It does not say which *topological class* the network
occupies.  Two networks with identical component counts, degree distributions
and defect counts can still differ in the length of the cycles their strands
participate in, and that difference distinguishes a prescribed-lattice
construction from a physically representative one.

This module supplies that measurement.  It follows the 3D-net formalism of
Sen and Olsen (arXiv:2406.17883), which in turn borrows the vertex-symbol
notation of reticular chemistry (O'Keeffe et al., Acc. Chem. Res. 2008, 41,
1782):

* the **loop order** (LO) of a cycle is the number of strands needed to close
  it, so a strand returning to its own junction is LO 1, two strands joining
  the same junction pair are LO 2, and a junction triangle is LO 3;
* the **vertex symbol** of a junction of connectivity ``p`` records, for each
  of the ``p(p-1)/2`` pairs of incident strands, the size of the smallest ring
  closing that pair;
* the **global cycle length distribution** is assembled from the per-junction
  vertex symbols.

Two properties of this measurement matter for construction and are reported
explicitly rather than left implicit.

**Bipartiteness.**  The ``dia`` and ``pcu`` nets are bipartite, so a builder
that places strands on net edges produces strictly even loop orders and never
the odd-order cycles that real networks contain.  ``bipartite`` in the report
is therefore a build warning, not a curiosity.  Note that an odd number of
periodic repeats destroys bipartiteness through the boundary, which is a box
artifact rather than chemistry; see ``odd_cycle_witness``.

**Functionality dependence.**  Peak loop size scales roughly as
``1/(f-1)``, so a target peak LO measured on a tetrafunctional network does
not transfer unchanged to a hexafunctional one.  ``mean_junction_degree`` is
reported alongside the distribution so the two are never compared blind.

Scope: the ring search here finds, for each incident-strand pair, the shortest
cycle closing that pair.  That is the vertex-symbol construction, not a full
smallest-set-of-smallest-rings perception; cycles that are not shortest for
any pair are not enumerated.  Reporting a loop-order distribution is a
structural diagnostic and does not by itself establish any mechanical
property.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from itertools import combinations
from pathlib import Path

__all__ = [
    "simple_adjacency",
    "reduce_to_junctions",
    "bipartite_check",
    "shortest_ring_through_pair",
    "vertex_symbols",
    "loop_order_histogram",
    "cyclic_topology_report",
]

# A pair with no closing cycle is written '*' in reticular-chemistry vertex
# symbols; keep the same convention so symbols are comparable with published
# net data.
NO_RING = "*"


def simple_adjacency(
    n_nodes: int,
    edges: list[tuple[int, int]],
) -> dict[int, set[int]]:
    """Adjacency of the simple graph underlying a multigraph.

    Self-loops and edge multiplicity are dropped: they are counted separately
    as loop orders 1 and 2 and would otherwise corrupt the ring search.
    """
    adjacency: dict[int, set[int]] = {node: set() for node in range(n_nodes)}
    for left, right in edges:
        if left == right:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def reduce_to_junctions(
    n_nodes: int,
    edges: list[tuple[int, int]],
) -> tuple[int, list[tuple[int, int]], dict[str, int]]:
    """Strip dangling trees and contract chain continuations.

    Sen and Olsen Eq. (1) weights each junction by ``f_j - 2``.  That weight is
    zero for a two-connected node and *negative* for a one-connected node, so
    the expression is only meaningful once every remaining node is a genuine
    junction with three or more strands.  Partially converted networks --- the
    case this package must support --- are full of one- and two-connected
    nodes, so the reduction has to happen first rather than being assumed.

    Degree-0 and degree-1 nodes are removed iteratively, which deletes whole
    dangling trees.  Degree-2 nodes are then contracted, merging their two
    strands into one, which is the standard junction--strand reduction: a node
    that merely continues a chain is not a crosslink.  Self-loops surviving
    contraction are kept, since a chain that returns to its own junction is a
    primary loop however many two-connected nodes it passes through.

    Returns ``(n_nodes, edges, stats)`` with nodes relabelled to
    ``0..n_nodes-1``.
    """
    incident: dict[int, list[int]] = defaultdict(list)
    live_edges: dict[int, tuple[int, int]] = {}
    for edge_id, (left, right) in enumerate(edges):
        live_edges[edge_id] = (left, right)
        incident[left].append(edge_id)
        incident[right].append(edge_id)

    alive = set(range(n_nodes))

    def degree(node: int) -> int:
        return sum(
            2 if live_edges[e][0] == live_edges[e][1] else 1
            for e in incident[node]
            if e in live_edges
        )

    # 1. peel dangling trees
    queue = deque(node for node in alive if degree(node) <= 1)
    removed_dangling = 0
    while queue:
        node = queue.popleft()
        if node not in alive or degree(node) > 1:
            continue
        alive.discard(node)
        removed_dangling += 1
        for edge_id in list(incident[node]):
            if edge_id not in live_edges:
                continue
            left, right = live_edges.pop(edge_id)
            other = right if left == node else left
            if other in alive and degree(other) <= 1:
                queue.append(other)

    # 2. contract chain continuations
    contracted = 0
    changed = True
    while changed:
        changed = False
        for node in list(alive):
            ids = [e for e in incident[node] if e in live_edges]
            if len(ids) != 2 or degree(node) != 2:
                continue
            first, second = ids
            a_left, a_right = live_edges[first]
            b_left, b_right = live_edges[second]
            end_a = a_right if a_left == node else a_left
            end_b = b_right if b_left == node else b_left
            del live_edges[first]
            del live_edges[second]
            merged = max(live_edges) + 1 if live_edges else 0
            while merged in live_edges:
                merged += 1
            live_edges[merged] = (end_a, end_b)
            incident[end_a].append(merged)
            incident[end_b].append(merged)
            alive.discard(node)
            contracted += 1
            changed = True

    order = sorted(alive)
    relabel = {node: index for index, node in enumerate(order)}
    reduced = [
        (relabel[left], relabel[right])
        for left, right in live_edges.values()
        if left in relabel and right in relabel
    ]
    stats = {
        "input_junction_count": n_nodes,
        "input_strand_count": len(edges),
        "dangling_junctions_removed": removed_dangling,
        "continuation_junctions_contracted": contracted,
        "junction_count": len(order),
        "strand_count": len(reduced),
    }
    return len(order), reduced, stats


def bipartite_check(
    n_nodes: int,
    edges: list[tuple[int, int]],
) -> tuple[bool, list[int] | None]:
    """Two-colour the simple graph; return ``(is_bipartite, witness)``.

    The witness is a shortest odd closed walk through the first conflicting
    edge found, given as a node list.  It is returned so a caller can tell a
    chemical odd cycle from one that closes through the periodic boundary.
    """
    adjacency = simple_adjacency(n_nodes, edges)
    for left, right in edges:
        if left == right:
            return False, [left, left]

    colour: dict[int, int] = {}
    parent: dict[int, int | None] = {}
    for source in range(n_nodes):
        if source in colour:
            continue
        colour[source] = 0
        parent[source] = None
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbour in adjacency[node]:
                if neighbour not in colour:
                    colour[neighbour] = 1 - colour[node]
                    parent[neighbour] = node
                    queue.append(neighbour)
                elif colour[neighbour] == colour[node]:
                    return False, _odd_walk(node, neighbour, parent)
    return True, None


def _odd_walk(
    left: int,
    right: int,
    parent: dict[int, int | None],
) -> list[int]:
    """Reconstruct a closed walk of odd length through the edge (left, right)."""

    def to_root(node: int) -> list[int]:
        path = []
        while node is not None:
            path.append(node)
            node = parent[node]
        return path

    left_path = to_root(left)
    right_path = to_root(right)
    right_index = {node: index for index, node in enumerate(right_path)}
    for index, node in enumerate(left_path):
        if node in right_index:
            return left_path[: index + 1] + right_path[: right_index[node]][::-1]
    return left_path + right_path[::-1]


def shortest_ring_through_pair(
    adjacency: dict[int, set[int]],
    centre: int,
    first: int,
    second: int,
) -> int | None:
    """Size of the smallest cycle passing through ``first-centre-second``.

    Returns ``None`` when the pair is not closed by any cycle, which the
    vertex symbol writes as ``*``.
    """
    if first == second:
        return None
    distance = {first: 0}
    queue = deque([first])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if neighbour == centre or neighbour in distance:
                continue
            distance[neighbour] = distance[node] + 1
            if neighbour == second:
                # the two strands to the centre close the cycle
                return distance[neighbour] + 2
            queue.append(neighbour)
    return None


def vertex_symbols(
    n_nodes: int,
    edges: list[tuple[int, int]],
) -> dict[int, list[int | str]]:
    """Smallest ring size for every incident-strand pair, per junction.

    The returned lists are sorted so that two junctions with the same local
    topology compare equal; sizes are integers and unclosed pairs are
    ``NO_RING``.
    """
    adjacency = simple_adjacency(n_nodes, edges)
    symbols: dict[int, list[int | str]] = {}
    for node in range(n_nodes):
        neighbours = sorted(adjacency[node])
        sizes: list[int | str] = []
        for first, second in combinations(neighbours, 2):
            size = shortest_ring_through_pair(adjacency, node, first, second)
            sizes.append(NO_RING if size is None else size)
        symbols[node] = sorted(sizes, key=lambda item: (item == NO_RING, item))
    return symbols


def loop_order_histogram(
    n_nodes: int,
    edges: list[tuple[int, int]],
    symbols: dict[int, list[int | str]] | None = None,
) -> dict[int, int]:
    """Global cycle count per loop order, following Sen and Olsen Eq. (1).

    For ring size ``i``, with ``n_ij`` the number of size-``i`` rings in the
    vertex symbol of junction ``j``, ``m_j`` the total ring count in that
    symbol and ``f_j`` the number of strands at ``j``,

        N_i = nint( sum_j  n_ij * (f_j - 2) / (2 * m_j) )

    Loop orders 1 and 2 are counted directly from the multigraph rather than
    through this expression, because self-loops and parallel strands are
    excluded from the vertex symbols by construction.
    """
    if symbols is None:
        symbols = vertex_symbols(n_nodes, edges)

    degree: Counter[int] = Counter()
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1

    histogram: dict[int, float] = defaultdict(float)
    for node, sizes in symbols.items():
        ring_sizes = [size for size in sizes if size != NO_RING]
        total_rings = len(ring_sizes)
        if total_rings == 0:
            continue
        if degree[node] < 3:
            # (f_j - 2) is zero at f_j = 2 and negative at f_j = 1; such nodes
            # are chain continuations or dangling ends, not junctions.
            continue
        weight = (degree[node] - 2) / (2.0 * total_rings)
        for size in Counter(ring_sizes).items():
            histogram[size[0]] += size[1] * weight

    counts = {
        order: int(round(value))
        for order, value in histogram.items()
        if round(value) > 0
    }

    self_loops = sum(1 for left, right in edges if left == right)
    if self_loops:
        counts[1] = counts.get(1, 0) + self_loops
    multiplicity = Counter(
        tuple(sorted(edge)) for edge in edges if edge[0] != edge[1]
    )
    parallel = sum(count - 1 for count in multiplicity.values() if count > 1)
    if parallel:
        counts[2] = counts.get(2, 0) + parallel
    return dict(sorted(counts.items()))


def cyclic_topology_report(
    n_nodes: int,
    edges: list[tuple[int, int]],
) -> dict[str, object]:
    """Full cyclic-topology diagnostic for a junction--strand multigraph.

    ``edges`` is a list of ``(junction, junction)`` pairs, one per strand,
    with repeats for parallel strands and ``left == right`` for primary loops
    --- exactly the ``reduced_edges`` produced by
    ``network_topology.audit_reduced_network``.
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    for left, right in edges:
        if not (0 <= left < n_nodes and 0 <= right < n_nodes):
            raise ValueError(f"edge ({left}, {right}) outside 0..{n_nodes - 1}")

    symbols = vertex_symbols(n_nodes, edges)
    histogram = loop_order_histogram(n_nodes, edges, symbols)
    is_bipartite, witness = bipartite_check(n_nodes, edges)

    # Girth is read off the raw shortest rings, never off the weighted
    # histogram: Eq. (1) multiplies by (f_j - 2), so a graph made only of
    # two-connected nodes has an empty histogram while still having cycles.
    raw_ring_sizes = [
        size for sizes in symbols.values() for size in sizes if size != NO_RING
    ]
    if any(left == right for left, right in edges):
        raw_ring_sizes.append(1)
    multiplicity_all = Counter(
        tuple(sorted(edge)) for edge in edges if edge[0] != edge[1]
    )
    if any(count > 1 for count in multiplicity_all.values()):
        raw_ring_sizes.append(2)

    degree: Counter[int] = Counter({node: 0 for node in range(n_nodes)})
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
    degrees = [degree[node] for node in range(n_nodes)]

    total_cycles = sum(histogram.values())
    odd_cycles = sum(count for order, count in histogram.items() if order % 2)
    n_strands = len(edges)

    multiplicity = Counter(
        tuple(sorted(edge)) for edge in edges if edge[0] != edge[1]
    )

    # Vertex symbols are compared as multisets of ring sizes; a perfectly
    # regular net collapses to a single distinct symbol.
    symbol_frequency = Counter(
        ".".join(str(size) for size in sizes) for sizes in symbols.values()
    )

    low_degree = sum(1 for value in degrees if value < 3)

    return {
        "junction_count": n_nodes,
        "strand_count": n_strands,
        # Eq. (1) weights by (f_j - 2) and is only meaningful for f_j >= 3.
        # A nonzero count here means the graph was not reduced first and the
        # loop-order histogram understates (or, for f_j = 1, mis-signs) the
        # cycle counts; call reduce_to_junctions() before trusting it.
        "low_degree_junction_count": low_degree,
        "loop_order_histogram_is_weighted_valid": low_degree == 0,
        "mean_junction_degree": (sum(degrees) / n_nodes) if n_nodes else float("nan"),
        "junction_degree_distribution": dict(sorted(Counter(degrees).items())),
        "loop_order_histogram": histogram,
        "loop_order_distribution": {
            order: count / n_strands for order, count in histogram.items()
        } if n_strands else {},
        "peak_loop_order": (
            max(histogram, key=lambda order: (histogram[order], -order))
            if histogram else None
        ),
        "girth": min(raw_ring_sizes) if raw_ring_sizes else None,
        "odd_loop_order_count": odd_cycles,
        "odd_loop_order_fraction": (odd_cycles / total_cycles) if total_cycles else 0.0,
        "primary_loop_count": histogram.get(1, 0),
        "secondary_loop_count": histogram.get(2, 0),
        "primary_loop_fraction": (histogram.get(1, 0) / n_strands) if n_strands else 0.0,
        "secondary_loop_fraction": (histogram.get(2, 0) / n_strands) if n_strands else 0.0,
        "maximum_edge_multiplicity": max(multiplicity.values(), default=0),
        "bipartite": is_bipartite,
        "odd_cycle_witness": witness,
        "distinct_vertex_symbol_count": len(symbol_frequency),
        "vertex_symbol_frequency": dict(symbol_frequency.most_common()),
    }


def report_from_itp(
    itp: str | Path,
    junction_residue: str = "BCK",
) -> dict[str, object]:
    """Reduce an ITP to its junction--strand graph and measure its topology."""
    from hygel_martini.property_extract.network_topology import audit_reduced_network

    audit = audit_reduced_network(itp, junction_residue=junction_residue)
    n_nodes = int(audit["junction_count"])
    edges = [tuple(edge) for edge in audit["reduced_edges"]]
    report = cyclic_topology_report(n_nodes, edges)
    report["itp"] = str(itp)
    report["junction_residue"] = junction_residue
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the cyclic topology (vertex symbols, loop-order "
            "distribution, bipartiteness) of a built network."
        )
    )
    parser.add_argument("--itp", required=True, help="hydrogel topology ITP")
    parser.add_argument(
        "--junction-residue",
        default="BCK",
        help="residue name identifying crosslinker atoms (default: BCK)",
    )
    parser.add_argument("--output", help="write the report as JSON to this path")
    args = parser.parse_args(argv)

    report = report_from_itp(args.itp, junction_residue=args.junction_residue)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)

    if not report["bipartite"]:
        return 0
    print(
        "\n[NOTE] The reduced graph is bipartite: odd loop orders cannot occur. "
        "If this network was seeded on a dia or pcu net, its loop-order "
        "spectrum is a property of the seed, not of the chemistry.",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
