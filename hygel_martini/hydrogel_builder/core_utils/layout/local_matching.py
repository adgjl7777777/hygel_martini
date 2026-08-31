"""Local BCK matching planner for tetrahedral diamond vertices.

The planner models the user's intended BCK semantics directly:

* one linker object has two BCK stubs,
* each BCK stub creates one local two-chain polymer junction, and
* choosing x/y/z at a local vertex chooses one of the three perfect matchings
  of the four nearby polymer endpoints.

It intentionally does not inspect or mutate the runtime World.  Layout code
uses this module to freeze local orientation states and exact endpoint-edge
pairs.  Dynamic crosslinking then materializes those pairs without endpoint
substitution.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
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


@lru_cache(maxsize=None)
def perfect_matchings(size: int) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    """All perfect matchings of ``range(size)``; there are ``(size-1)!!``.

    The recursion pairs the lowest unmatched index with each remaining index in
    turn, so the enumeration order is deterministic.  For ``size == 4`` it
    yields the x, y and z matchings of :data:`MATCHINGS_BY_AXIS` in that order,
    which is what lets the tetrafunctional axis vocabulary sit on top of the
    general one (see :data:`AXIS_BY_STATE`).
    """
    if size % 2:
        raise ValueError(f"odd endpoint count {size} has no perfect matching")
    if size == 0:
        return ((),)

    def build(items: Tuple[int, ...]):
        if not items:
            yield ()
            return
        first, rest = items[0], items[1:]
        for position in range(len(rest)):
            pair = (first, rest[position])
            remainder = rest[:position] + rest[position + 1:]
            for tail in build(remainder):
                yield (pair,) + tail

    return tuple(build(tuple(range(size))))


def matching_state_count(functionality: int) -> int:
    """``(f-1)!!`` --- 3 states at f=4, 15 at f=6, 105 at f=8."""
    return len(perfect_matchings(functionality))


# The tetrafunctional axis labels are the first three general states, in order.
AXIS_BY_STATE: Mapping[int, str] = {0: "x", 1: "y", 2: "z"}
STATE_BY_AXIS: Mapping[str, int] = {axis: state for state, axis in AXIS_BY_STATE.items()}


def _normalize_state(state, functionality: int) -> int:
    """Accept either a general state index or a tetrafunctional axis label."""
    if isinstance(state, str):
        axis = state.lower()
        if axis not in STATE_BY_AXIS:
            raise ValueError(f"Unknown matching axis {state!r}; expected x, y, or z")
        if functionality != 4:
            raise ValueError(
                f"Axis label {state!r} is only defined for functionality 4, "
                f"not {functionality}; use an integer state index instead"
            )
        return STATE_BY_AXIS[axis]
    index = int(state)
    total = matching_state_count(functionality)
    if not 0 <= index < total:
        raise ValueError(
            f"Matching state {index} out of range for functionality "
            f"{functionality} ({total} states)"
        )
    return index


@dataclass(frozen=True)
class LocalVertex:
    """The polymer endpoints around one local crosslink vertex.

    Tetrafunctional vertices key their endpoints by the diamond local
    coordinates of :data:`LOCAL_COORDS` and may use the x/y/z axis vocabulary.
    Vertices of any other even functionality key their endpoints by whatever
    hashable, sortable label the layout supplies and are addressed by integer
    matching-state index instead.
    """

    vertex_id: Hashable
    endpoints: Mapping[LocalCoord, EndpointId]

    @property
    def functionality(self) -> int:
        return len(self.endpoints)

    @property
    def is_tetrahedral(self) -> bool:
        """True when this vertex uses the diamond local-coordinate labels."""
        return set(self.endpoints) == set(LOCAL_COORDS)

    def ordered_keys(self) -> Tuple[LocalCoord, ...]:
        """Endpoint labels in the fixed order the matching states index into."""
        if self.is_tetrahedral:
            return LOCAL_COORDS
        try:
            return tuple(sorted(self.endpoints))
        except TypeError:
            return tuple(sorted(self.endpoints, key=repr))

    def validate(self) -> None:
        if self.is_tetrahedral:
            return
        count = len(self.endpoints)
        if count < 2 or count % 2:
            raise ValueError(
                f"LocalVertex {self.vertex_id!r} has {count} endpoints; a "
                "crosslink vertex needs an even number of at least two so its "
                "endpoints admit a perfect matching"
            )
        distinct = len(set(self.endpoints.values()))
        if distinct != count:
            raise ValueError(
                f"LocalVertex {self.vertex_id!r} repeats an endpoint identifier "
                f"({distinct} distinct for {count} slots)"
            )


@dataclass(frozen=True)
class VertexAxisChoice:
    """Chosen x/y/z matching state for one local vertex."""

    vertex_id: Hashable
    axis: str
    edges: Tuple[Edge, ...]
    # ``axis`` stays the tetrafunctional label ("x"/"y"/"z") and is the string
    # form of the index when the vertex is not tetrahedral.
    state: int = -1
    functionality: int = 4


@dataclass(frozen=True)
class MatchingDiagnostics:
    """Connectivity and balance report for a local matching plan."""

    component_count: int
    largest_component_size: int
    node_count: int
    axis_counts: Mapping[str, int]
    degree_violations: Mapping[EndpointId, int]
    # Census over general matching-state indices.  ``axis_counts`` stays the
    # tetrafunctional view and is empty when no vertex is tetrahedral.
    state_counts: Mapping[int, int] = field(default_factory=dict)
    functionality_counts: Mapping[int, int] = field(default_factory=dict)


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


def matching_edges_for_state(vertex: LocalVertex, state) -> Tuple[Edge, ...]:
    """The ``f/2`` junction edges created by one matching state at one vertex.

    ``state`` is an integer index into :func:`perfect_matchings`, or --- for a
    tetrafunctional vertex only --- one of the axis labels ``"x"``, ``"y"``,
    ``"z"``.
    """
    vertex.validate()
    keys = vertex.ordered_keys()
    index = _normalize_state(state, len(keys))
    matching = perfect_matchings(len(keys))[index]
    return tuple(
        (vertex.endpoints[keys[left]], vertex.endpoints[keys[right]])
        for left, right in matching
    )


def matching_edges_for_axis(vertex: LocalVertex, axis: str) -> Tuple[Edge, Edge]:
    """Tetrafunctional wrapper over :func:`matching_edges_for_state`."""
    axis = str(axis).lower()
    if axis not in MATCHINGS_BY_AXIS:
        raise ValueError(f"Unknown matching axis {axis!r}; expected x, y, or z")
    if not vertex.is_tetrahedral:
        raise ValueError(
            f"LocalVertex {vertex.vertex_id!r} is not tetrahedral, so the x/y/z "
            "vocabulary does not apply; use matching_edges_for_state()"
        )
    return matching_edges_for_state(vertex, axis)


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
    state_counts: Counter = Counter()
    functionality_counts: Counter = Counter()
    saw_tetrahedral = False

    for vertex in vertices:
        vertex.validate()
        functionality = vertex.functionality
        state = _normalize_state(axes_by_vertex[vertex.vertex_id], functionality)
        edges = matching_edges_for_state(vertex, state)
        if vertex.is_tetrahedral:
            saw_tetrahedral = True
            label = AXIS_BY_STATE[state]
            axis_counts[label] += 1
        else:
            label = str(state)
        choices.append(
            VertexAxisChoice(vertex.vertex_id, label, edges, state, functionality)
        )
        local_edges.extend(edges)
        state_counts[state] += 1
        functionality_counts[functionality] += 1

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
        axis_counts=(
            {axis: axis_counts[axis] for axis in ("x", "y", "z")}
            if saw_tetrahedral else {}
        ),
        degree_violations=degree_violations,
        state_counts=dict(sorted(state_counts.items())),
        functionality_counts=dict(sorted(functionality_counts.items())),
    )
    return MatchingPlan(tuple(choices), diagnostics)


@lru_cache(maxsize=None)
def _state_by_pairing(size: int) -> Mapping[frozenset, int]:
    """Reverse lookup from a slot-index pairing to its matching-state index."""
    return {
        frozenset(frozenset(pair) for pair in matching): index
        for index, matching in enumerate(perfect_matchings(size))
    }


def state_for_pairing(vertex: LocalVertex, pairs: Sequence[Edge]) -> int:
    """Which matching state realizes ``pairs`` at ``vertex``."""
    vertex.validate()
    keys = vertex.ordered_keys()
    slot = {vertex.endpoints[key]: index for index, key in enumerate(keys)}
    try:
        wanted = frozenset(
            frozenset((slot[left], slot[right])) for left, right in pairs
        )
    except KeyError as exc:
        raise ValueError(
            f"Endpoint {exc.args[0]!r} is not one of vertex {vertex.vertex_id!r}'s"
        ) from exc
    lookup = _state_by_pairing(len(keys))
    if wanted not in lookup:
        raise ValueError(
            f"Pairing {sorted(map(sorted, wanted))} is not a perfect matching of "
            f"vertex {vertex.vertex_id!r}"
        )
    return lookup[wanted]


@dataclass(frozen=True)
class _Traversal:
    edge_index: int
    from_vertex: Hashable
    from_endpoint: EndpointId
    to_vertex: Hashable
    to_endpoint: EndpointId


def _strand_adjacency(
    vertices: Sequence[LocalVertex],
    chain_edges: Sequence[Edge],
):
    """Index the strand graph by junction, as directed traversals."""
    owner: Dict[EndpointId, Hashable] = {}
    for vertex in vertices:
        vertex.validate()
        for endpoint in vertex.endpoints.values():
            if endpoint in owner:
                raise ValueError(
                    f"Endpoint {endpoint!r} belongs to both vertex "
                    f"{owner[endpoint]!r} and {vertex.vertex_id!r}"
                )
            owner[endpoint] = vertex.vertex_id

    adjacency: Dict[Hashable, List[_Traversal]] = defaultdict(list)
    seen: Dict[EndpointId, int] = {}
    for index, (left, right) in enumerate(chain_edges):
        for endpoint in (left, right):
            if endpoint not in owner:
                raise ValueError(
                    f"Strand {index} uses endpoint {endpoint!r}, which no vertex owns"
                )
            if endpoint in seen:
                raise ValueError(
                    f"Endpoint {endpoint!r} is used by strands {seen[endpoint]} "
                    f"and {index}; each endpoint carries exactly one strand"
                )
            seen[endpoint] = index
        left_vertex, right_vertex = owner[left], owner[right]
        adjacency[left_vertex].append(
            _Traversal(index, left_vertex, left, right_vertex, right)
        )
        adjacency[right_vertex].append(
            _Traversal(index, right_vertex, right, left_vertex, left)
        )

    unattached = [
        endpoint for endpoint in owner if endpoint not in seen
    ]
    return adjacency, owner, unattached


def plan_single_circuit(
    vertices: Sequence[LocalVertex],
    chain_edges: Sequence[Edge],
) -> MatchingPlan:
    """Construct a transition system with one circuit per connected component.

    Every junction of even functionality has even degree in the strand graph,
    so each component admits an Eulerian circuit.  An Eulerian circuit *is* a
    transition system: each of a junction's ``f/2`` visits enters by one
    endpoint and leaves by another, which pairs all ``f`` endpoints exactly
    once, and the circuit is a single closed walk.  Hierholzer's algorithm
    therefore returns the optimum --- one circuit per component --- in time
    linear in the number of strands, with no search.

    This is the seed for any even functionality.  It optimizes circuit count
    only; it says nothing about the loop-order distribution of the reduced
    junction--strand graph, which no transition system can change (see
    ``hygel_martini.property_extract.cyclic_topology``).

    Raises ``ValueError`` if an endpoint carries no strand or more than one, or
    if a junction ends up with odd degree, since no perfect matching exists
    then and a silently partial plan would be worse than a refusal.
    """
    if not vertices:
        return evaluate_matching_plan([], chain_edges, {})

    adjacency, _owner, unattached = _strand_adjacency(vertices, chain_edges)
    if unattached:
        raise ValueError(
            f"{len(unattached)} endpoint(s) carry no strand, e.g. "
            f"{sorted(map(repr, unattached))[:3]}; a transition system needs "
            "every endpoint matched"
        )
    odd = {
        vertex.vertex_id: len(adjacency[vertex.vertex_id])
        for vertex in vertices
        if len(adjacency[vertex.vertex_id]) % 2
    }
    if odd:
        raise ValueError(
            f"Odd strand degree at {len(odd)} junction(s), e.g. "
            f"{dict(list(odd.items())[:3])}; no perfect matching exists there"
        )

    used_edges: set = set()
    cursor: Dict[Hashable, int] = defaultdict(int)
    pairings: Dict[Hashable, List[Edge]] = defaultdict(list)

    for vertex in vertices:
        start = vertex.vertex_id
        if all(t.edge_index in used_edges for t in adjacency[start]):
            continue

        stack: List[Tuple[Hashable, _Traversal | None]] = [(start, None)]
        walk: List[Tuple[Hashable, _Traversal | None]] = []
        while stack:
            node, arrival = stack[-1]
            candidate = None
            options = adjacency[node]
            while cursor[node] < len(options):
                option = options[cursor[node]]
                cursor[node] += 1
                if option.edge_index not in used_edges:
                    candidate = option
                    break
            if candidate is None:
                walk.append(stack.pop())
            else:
                used_edges.add(candidate.edge_index)
                stack.append((candidate.to_vertex, candidate))
        walk.reverse()

        traversals = [entry[1] for entry in walk[1:]]
        if not traversals:
            continue
        for position, traversal in enumerate(traversals):
            following = traversals[(position + 1) % len(traversals)]
            pairings[traversal.to_vertex].append(
                (traversal.to_endpoint, following.from_endpoint)
            )

    states: Dict[Hashable, int] = {}
    for vertex in vertices:
        pairs = pairings.get(vertex.vertex_id, [])
        if len(pairs) * 2 != vertex.functionality:
            raise ValueError(
                f"Vertex {vertex.vertex_id!r} received {len(pairs)} pairing(s) "
                f"for functionality {vertex.functionality}; the Eulerian walk "
                "did not close consistently"
            )
        states[vertex.vertex_id] = state_for_pairing(vertex, pairs)

    return evaluate_matching_plan(vertices, chain_edges, states)


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
