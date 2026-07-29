"""Local BCK matching planner for tetrahedral diamond vertices.

The planner models the user's intended BCK semantics directly:

* one linker object has two BCK stubs,
* each BCK stub creates one local two-chain polymer junction, and
* choosing x/y/z at a local vertex chooses one of the three perfect matchings
  of the four nearby polymer endpoints.

It intentionally does not inspect or mutate the runtime World.  Layout code can
use this module to choose local orientation states first; dynamic crosslinking
then only has to bond each placed BCK to the nearest compatible ends.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
import math
from random import Random
import time
from typing import Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple


LocalCoord = Tuple[int, int, int]
EndpointId = Hashable
Edge = Tuple[EndpointId, EndpointId]


LOCAL_COORDS: Tuple[LocalCoord, ...] = (
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
)

MATCHINGS_BY_AXIS: Mapping[str, Tuple[Tuple[LocalCoord, LocalCoord], ...]] = {
    "x": (((0, 0, 0), (0, 1, 1)), ((1, 0, 1), (1, 1, 0))),
    "y": (((0, 0, 0), (1, 0, 1)), ((0, 1, 1), (1, 1, 0))),
    "z": (((0, 0, 0), (1, 1, 0)), ((0, 1, 1), (1, 0, 1))),
}
AXES: Tuple[str, str, str] = ("x", "y", "z")


@dataclass(frozen=True)
class LocalVertex:
    """Four polymer endpoints around one local BCK-linker vertex."""

    vertex_id: Hashable
    endpoints: Mapping[LocalCoord, EndpointId]

    def validate(self) -> None:
        missing = [coord for coord in LOCAL_COORDS if coord not in self.endpoints]
        if missing:
            raise ValueError(f"LocalVertex {self.vertex_id!r} missing endpoints for {missing}")


@dataclass(frozen=True)
class VertexAxisChoice:
    """Chosen x/y/z matching state for one local vertex."""

    vertex_id: Hashable
    axis: str
    edges: Tuple[Edge, Edge]


@dataclass(frozen=True)
class MatchingDiagnostics:
    """Connectivity and balance report for a local matching plan."""

    component_count: int
    largest_component_size: int
    node_count: int
    axis_counts: Mapping[str, int]
    degree_violations: Mapping[EndpointId, int]


@dataclass(frozen=True)
class MatchingPlan:
    """Result of choosing local x/y/z states for a set of vertices."""

    choices: Tuple[VertexAxisChoice, ...]
    diagnostics: MatchingDiagnostics

    @property
    def is_single_cycle(self) -> bool:
        return (
            self.diagnostics.component_count == 1
            and not self.diagnostics.degree_violations
        )


class _UnionFind:
    def __init__(self, nodes: Iterable[EndpointId]):
        self.parent = {node: node for node in nodes}
        self.size = {node: 1 for node in nodes}

    def find(self, node: EndpointId) -> EndpointId:
        parent = self.parent.setdefault(node, node)
        if parent != node:
            self.parent[node] = self.find(parent)
        self.size.setdefault(node, 1)
        return self.parent[node]

    def union(self, first: EndpointId, second: EndpointId) -> None:
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]


def matching_edges_for_axis(vertex: LocalVertex, axis: str) -> Tuple[Edge, Edge]:
    """Return the two BCK junction edges created by one local axis choice."""
    axis = str(axis).lower()
    if axis not in MATCHINGS_BY_AXIS:
        raise ValueError(f"Unknown matching axis {axis!r}; expected x, y, or z")
    vertex.validate()
    return tuple(
        (vertex.endpoints[left], vertex.endpoints[right])
        for left, right in MATCHINGS_BY_AXIS[axis]
    )


def _all_nodes(vertices: Sequence[LocalVertex], chain_edges: Sequence[Edge]) -> List[EndpointId]:
    nodes = []
    seen = set()
    for vertex in vertices:
        vertex.validate()
        for endpoint in vertex.endpoints.values():
            if endpoint not in seen:
                nodes.append(endpoint)
                seen.add(endpoint)
    for first, second in chain_edges:
        for endpoint in (first, second):
            if endpoint not in seen:
                nodes.append(endpoint)
                seen.add(endpoint)
    return nodes


def evaluate_matching_plan(
    vertices: Sequence[LocalVertex],
    chain_edges: Sequence[Edge],
    axes_by_vertex: Mapping[Hashable, str],
) -> MatchingPlan:
    """Evaluate a full local matching assignment as a graph of polymer endpoints."""
    choices: List[VertexAxisChoice] = []
    local_edges: List[Edge] = []
    axis_counts = Counter({"x": 0, "y": 0, "z": 0})

    for vertex in vertices:
        axis = str(axes_by_vertex[vertex.vertex_id]).lower()
        edges = matching_edges_for_axis(vertex, axis)
        choices.append(VertexAxisChoice(vertex.vertex_id, axis, edges))
        local_edges.extend(edges)
        axis_counts[axis] += 1

    all_edges = list(chain_edges) + local_edges
    nodes = _all_nodes(vertices, chain_edges)
    uf = _UnionFind(nodes)
    degree = Counter({node: 0 for node in nodes})
    for first, second in all_edges:
        uf.union(first, second)
        degree[first] += 1
        degree[second] += 1

    component_sizes = Counter(uf.find(node) for node in nodes)
    degree_violations = {
        node: count
        for node, count in degree.items()
        if count != 2
    }
    diagnostics = MatchingDiagnostics(
        component_count=len(component_sizes),
        largest_component_size=max(component_sizes.values(), default=0),
        node_count=len(nodes),
        axis_counts={axis: axis_counts[axis] for axis in ("x", "y", "z")},
        degree_violations=degree_violations,
    )
    return MatchingPlan(tuple(choices), diagnostics)


def _balanced_axis_pool(count: int, rng: Random) -> List[str]:
    axes = list(AXES)
    rng.shuffle(axes)
    base = count // 3
    remainder = count % 3
    pool: List[str] = []
    for idx, axis in enumerate(axes):
        pool.extend([axis] * (base + (1 if idx < remainder else 0)))
    rng.shuffle(pool)
    return pool


def _axis_balance_penalty(axis_counts: Mapping[str, int]) -> int:
    """Return zero only when x/y/z counts differ by at most one."""
    counts = [int(axis_counts.get(axis, 0)) for axis in AXES]
    if not counts:
        return 0
    spread = max(counts) - min(counts)
    if spread <= 1:
        return 0
    mean_num = sum(counts)
    return spread + sum((3 * count - mean_num) ** 2 for count in counts)


def _is_nearly_balanced(axis_counts: Mapping[str, int]) -> bool:
    counts = [int(axis_counts.get(axis, 0)) for axis in AXES]
    return max(counts, default=0) - min(counts, default=0) <= 1


def _score(plan: MatchingPlan) -> Tuple[int, int, int, int]:
    diag = plan.diagnostics
    return (
        diag.component_count,
        len(diag.degree_violations),
        _axis_balance_penalty(diag.axis_counts),
        -diag.largest_component_size,
    )


def _annealing_energy(plan: MatchingPlan) -> float:
    """Scalar score for Metropolis moves; lower is better."""
    diag = plan.diagnostics
    node_scale = max(diag.node_count, 1)
    return (
        1000000.0 * diag.component_count
        + 10000.0 * len(diag.degree_violations)
        + 100.0 * _axis_balance_penalty(diag.axis_counts)
        - float(diag.largest_component_size) / float(node_scale)
    )


def _exact_balanced_search(
    vertices: Sequence[LocalVertex],
    chain_edges: Sequence[Edge],
    exact_limit: int,
) -> MatchingPlan | None:
    """Enumerate all nearly-balanced transition systems for small lattices."""
    if len(vertices) > exact_limit:
        return None

    vertex_ids = [vertex.vertex_id for vertex in vertices]
    best_plan = None
    best_score = None
    for axes in product(AXES, repeat=len(vertices)):
        counts = Counter(axes)
        if not _is_nearly_balanced(counts):
            continue

        axes_by_vertex = dict(zip(vertex_ids, axes))
        candidate = evaluate_matching_plan(vertices, chain_edges, axes_by_vertex)
        candidate_score = _score(candidate)
        if best_plan is None or candidate_score < best_score:
            best_plan = candidate
            best_score = candidate_score
            if candidate.is_single_cycle:
                break

    return best_plan


def _greedy_balanced_kotzig_descent(
    vertices: Sequence[LocalVertex],
    chain_edges: Sequence[Edge],
    axes_by_vertex: Dict[Hashable, str],
    max_passes: int,
    rng: Random,
    max_pair_checks: int,
    deadline: float | None = None,
) -> MatchingPlan:
    """Apply sampled best two-vertex transition swaps without breaking balance."""
    vertex_ids = [vertex.vertex_id for vertex in vertices]
    current = evaluate_matching_plan(vertices, chain_edges, axes_by_vertex)
    current_score = _score(current)
    max_pair_checks = max(int(max_pair_checks), 0)

    for _pass in range(max(int(max_passes), 0)):
        if deadline is not None and time.monotonic() >= deadline:
            break
        best_pair = None
        best_plan = current
        best_score = current_score

        pair_total = len(vertex_ids) * (len(vertex_ids) - 1) // 2
        exhaustive = max_pair_checks == 0 or max_pair_checks >= pair_total
        checked = 0

        if exhaustive:
            pair_iter = (
                (idx_a, idx_b)
                for idx_a in range(len(vertex_ids))
                for idx_b in range(idx_a + 1, len(vertex_ids))
            )
        else:
            seen_pairs = set()

            def _sample_pairs():
                attempts = 0
                max_attempts = max_pair_checks * 8
                while len(seen_pairs) < max_pair_checks and attempts < max_attempts:
                    attempts += 1
                    idx_a = rng.randrange(len(vertex_ids))
                    idx_b = rng.randrange(len(vertex_ids))
                    if idx_a == idx_b:
                        continue
                    if idx_a > idx_b:
                        idx_a, idx_b = idx_b, idx_a
                    pair = (idx_a, idx_b)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    yield pair

            pair_iter = _sample_pairs()

        for idx_a, idx_b in pair_iter:
            if deadline is not None and time.monotonic() >= deadline:
                break
            key_a = vertex_ids[idx_a]
            key_b = vertex_ids[idx_b]
            axis_a = axes_by_vertex[key_a]
            axis_b = axes_by_vertex[key_b]
            if axis_a == axis_b:
                continue

            axes_by_vertex[key_a], axes_by_vertex[key_b] = axis_b, axis_a
            candidate = evaluate_matching_plan(vertices, chain_edges, axes_by_vertex)
            candidate_score = _score(candidate)
            axes_by_vertex[key_a], axes_by_vertex[key_b] = axis_a, axis_b
            checked += 1

            if candidate_score < best_score:
                best_pair = (key_a, key_b)
                best_plan = candidate
                best_score = candidate_score

            if not exhaustive and checked >= max_pair_checks:
                break

        if best_pair is None:
            break

        key_a, key_b = best_pair
        axes_by_vertex[key_a], axes_by_vertex[key_b] = (
            axes_by_vertex[key_b],
            axes_by_vertex[key_a],
        )
        current = best_plan
        current_score = best_score
        if current.is_single_cycle:
            break

    return current


def plan_balanced_cycle_matchings(
    vertices: Sequence[LocalVertex],
    chain_edges: Sequence[Edge],
    seed: int | None = None,
    attempts: int = 256,
    swaps_per_attempt: int = 512,
    exact_limit: int = 12,
    greedy_passes: int = 8,
    greedy_pair_checks: int = 4096,
    time_budget_seconds: float | None = 30.0,
) -> MatchingPlan:
    """Choose balanced local x/y/z transitions for one/few global cycles.

    This is the transition-system/circuit-partition formulation from the graph
    literature: each vertex chooses one of the three local pairings, and the
    resulting 2-factor is judged by its circuit count.  Small lattices are
    solved by exact nearly-balanced enumeration.  Larger lattices use balanced
    Kotzig-style moves: swapping two vertices' transition labels preserves the
    global x/y/z quota while still changing two local pairings.
    """
    if not vertices:
        return evaluate_matching_plan([], chain_edges, {})

    exact_plan = _exact_balanced_search(vertices, chain_edges, max(int(exact_limit), 0))
    if exact_plan is not None:
        return exact_plan

    rng = Random(seed)
    attempts = max(int(attempts), 1)
    swaps_per_attempt = max(int(swaps_per_attempt), 0)
    vertex_ids = [vertex.vertex_id for vertex in vertices]
    deadline = None
    if time_budget_seconds is not None and float(time_budget_seconds) > 0.0:
        deadline = time.monotonic() + float(time_budget_seconds)

    best_plan = None
    best_score = None
    best_axes_by_vertex = None
    for _ in range(attempts):
        if deadline is not None and time.monotonic() >= deadline:
            break
        pool = _balanced_axis_pool(len(vertices), rng)
        axes_by_vertex: Dict[Hashable, str] = dict(zip(vertex_ids, pool))
        current = evaluate_matching_plan(vertices, chain_edges, axes_by_vertex)
        current_score = _score(current)
        current_energy = _annealing_energy(current)
        start_temperature = max(1.0, float(len(vertices)))

        for step in range(swaps_per_attempt):
            if deadline is not None and time.monotonic() >= deadline:
                break
            if current.is_single_cycle:
                break
            idx_a = rng.randrange(len(vertices))
            idx_b = rng.randrange(len(vertices))
            if idx_a == idx_b:
                continue
            key_a = vertex_ids[idx_a]
            key_b = vertex_ids[idx_b]
            if axes_by_vertex[key_a] == axes_by_vertex[key_b]:
                continue

            axes_by_vertex[key_a], axes_by_vertex[key_b] = (
                axes_by_vertex[key_b],
                axes_by_vertex[key_a],
            )
            candidate = evaluate_matching_plan(vertices, chain_edges, axes_by_vertex)
            candidate_score = _score(candidate)
            candidate_energy = _annealing_energy(candidate)
            temperature = start_temperature * (1.0 - (step / max(swaps_per_attempt, 1)))
            temperature = max(temperature, 0.05)
            delta = candidate_energy - current_energy
            accept = delta <= 0.0 or rng.random() < math.exp(-delta / temperature)
            if accept:
                current = candidate
                current_score = candidate_score
                current_energy = candidate_energy
            else:
                axes_by_vertex[key_a], axes_by_vertex[key_b] = (
                    axes_by_vertex[key_b],
                    axes_by_vertex[key_a],
                )

        current_score = _score(current)
        if best_plan is None or current_score < best_score:
            best_plan = current
            best_score = current_score
            best_axes_by_vertex = dict(axes_by_vertex)
        if best_plan.is_single_cycle:
            break

    if best_plan is not None and best_axes_by_vertex is not None and not best_plan.is_single_cycle:
        refined = _greedy_balanced_kotzig_descent(
            vertices,
            chain_edges,
            best_axes_by_vertex,
            max_passes=greedy_passes,
            rng=rng,
            max_pair_checks=greedy_pair_checks,
            deadline=deadline,
        )
        refined_score = _score(refined)
        if refined_score < best_score:
            best_plan = refined
            best_score = refined_score

    return best_plan
