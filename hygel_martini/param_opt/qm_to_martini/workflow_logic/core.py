"""Small geometry/graph/combinatorics primitives for workflow_logic.

This module owns the dependency-free helpers shared by
``workflow_logic.loader`` and ``workflow_logic.builder``: Euclidean
distance, CSV token splitting, canonical (reversal-invariant) forms of
bonded-term index tuples, bead-graph construction, BFS path length, and
exhaustive generation of reversal-unique index combinations used for
candidate bonded terms.

Inputs/outputs are plain Python values; nothing here touches files.
Distances are in the caller's coordinate units (Angstrom for XYZ files).

Invariant: a bonded term and its reversal (i-j vs j-i, i-j-k vs k-j-i,
i-j-k-l vs l-k-j-i) are the same physical term, so every canonical form
here picks the lexicographically smaller of the two orientations.
"""

from __future__ import annotations

import math
import re
from collections import deque, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from itertools import combinations, permutations

def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Euclidean distance between two points (units follow the input)."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def _split_csv(raw: str) -> List[str]:
    """Split a comma-separated string into stripped, non-empty tokens."""
    return [token.strip() for token in re.split(r"\s*,\s*", raw.strip()) if token.strip()]

def _sorted_pair(a: int, b: int) -> Tuple[int, int]:
    """Canonical (ascending) form of an undirected bond pair."""
    return (a, b) if a <= b else (b, a)

def _canon_angle(i: int, j: int, k: int) -> Tuple[int, int, int]:
    """Canonical angle triple: outer beads ordered, center bead fixed."""
    return (i, j, k) if i <= k else (k, j, i)

def _canon_reversible(values: Sequence[int]) -> Tuple[int, ...]:
    """Canonical form of any reversal-symmetric index tuple.

    Returns whichever of the tuple and its reversal compares smaller, so
    e.g. a dihedral (1,2,3,4) and (4,3,2,1) map to the same key.
    """
    forward = tuple(int(value) for value in values)
    reverse = tuple(reversed(forward))
    return forward if forward <= reverse else reverse

def _build_graph(edges: Iterable[Tuple[int, int]]) -> Dict[int, set[int]]:
    """Build an undirected adjacency map from (a, b) edge pairs."""
    graph: Dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    return graph

def shortest_path_len(graph: Dict[int, set[int]], start: int, goal: int) -> Optional[int]:
    """Return the BFS shortest-path edge count between two nodes.

    Args:
        graph: undirected adjacency map (see ``_build_graph``).
        start: source node id.
        goal: target node id.

    Returns:
        Number of edges on the shortest path, 0 when start == goal, or
        None when the nodes are disconnected.
    """
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

def _reversal_unique_permutations(values: Sequence[int]) -> List[Tuple[int, ...]]:
    """Enumerate all orderings of ``values``, one per reversal class.

    Every permutation is reduced to its canonical reversible form and
    deduplicated, giving each distinct bonded-term ordering exactly once.

    Returns:
        Sorted list of canonical index tuples.
    """
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
    """Generate every new candidate term of ``body_size`` beads.

    For each ``body_size``-combination of ``bead_ids``, all reversal-unique
    orderings are produced; tuples already present in ``existing`` (assumed
    canonical) are skipped. Used by the exhaustive candidate-term modes.

    Args:
        bead_ids: available bead indices (1-based).
        body_size: number of beads per term (2=bond, 3=angle, 4=dihedral).
        existing: canonical tuples already covered by the template.

    Returns:
        New canonical index tuples, ordered by combination then
        permutation order.
    """
    generated: List[Tuple[int, ...]] = []
    seen: set[Tuple[int, ...]] = set()
    for combo in combinations(sorted(bead_ids), body_size):
        for candidate in _reversal_unique_permutations(combo):
            if candidate in existing or candidate in seen:
                continue
            generated.append(candidate)
            seen.add(candidate)
    return generated
