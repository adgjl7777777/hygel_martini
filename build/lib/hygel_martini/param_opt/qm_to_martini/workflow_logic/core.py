from __future__ import annotations

import math
import re
from collections import deque, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from itertools import combinations, permutations

def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def _split_csv(raw: str) -> List[str]:
    return [token.strip() for token in re.split(r"\s*,\s*", raw.strip()) if token.strip()]

def _sorted_pair(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)

def _canon_angle(i: int, j: int, k: int) -> Tuple[int, int, int]:
    return (i, j, k) if i <= k else (k, j, i)

def _canon_reversible(values: Sequence[int]) -> Tuple[int, ...]:
    forward = tuple(int(value) for value in values)
    reverse = tuple(reversed(forward))
    return forward if forward <= reverse else reverse

def _build_graph(edges: Iterable[Tuple[int, int]]) -> Dict[int, set[int]]:
    graph: Dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    return graph

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
