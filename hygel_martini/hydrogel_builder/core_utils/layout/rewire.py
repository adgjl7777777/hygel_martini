"""Span-constrained rewiring of a net seed toward a representative topology.

A net seed (``nets.py``) is topologically regular: every junction has the same
local environment, the loop-order distribution is a single spike at the net's
fundamental cycle size, and --- because ``dia`` and ``pcu`` are bipartite ---
there are no odd-order loops at all.  Real networks have none of those
properties.

This module applies the lattice-to-random interpolation of Sen and Olsen
(arXiv:2406.17883), itself a constrained form of the Watts--Strogatz rewiring
(Nature 393, 440):

1. pick two strands at random;
2. exchange one junction attachment of each;
3. accept only if both resulting strands are shorter than a cutoff ``max_span``;
4. repeat until the loop-order distribution stops changing.

Step 3 carries the physics.  A strand can only bridge junctions it can
physically reach, so ``max_span`` is bounded above by the strand contour
length, and it is the single knob that sets what Sen and Olsen call the
topological proximity of crosslinkers at bond formation.  Small ``max_span``
keeps connections local and biases toward short loops; large ``max_span``
approaches the unconstrained kinetic-Monte-Carlo limit.

Two invariants hold by construction and are asserted rather than assumed:

**Junction degree is preserved exactly.**  A double-edge swap exchanges
endpoints between two strands, so every endpoint still carries exactly one
strand and every junction keeps its functionality.  Rewiring therefore cannot
turn an ``f=6`` network into something else, and it composes with the
transition-system planner, which requires even degree.

**The transition system is not preserved.**  Rewiring changes which junctions
are adjacent, so any previously computed matching plan is invalid afterwards.
Rewire first, then plan.

Primary loops (a strand returning to its own junction) and parallel strands
are permitted by default and counted, because real networks contain them and a
construction that forbids them is not more physical for doing so.  Set
``allow_primary_loops`` or ``allow_parallel_strands`` to ``False`` only when
building a deliberately idealized reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Dict, Hashable, List, Mapping, Sequence, Tuple

import numpy as np

__all__ = [
    "RewiringResult",
    "reduced_edges",
    "loop_order_snapshot",
    "total_variation_distance",
    "span_constrained_rewire",
]

Endpoint = Hashable
Strand = Tuple[Endpoint, Endpoint]


@dataclass
class RewiringResult:
    """Outcome of a rewiring run, including why proposals were refused."""

    strands: List[Strand]
    proposed: int = 0
    accepted: int = 0
    rejected_span: int = 0
    rejected_primary_loop: int = 0
    rejected_parallel: int = 0
    rejected_degenerate: int = 0
    converged: bool = False
    sweeps: int = 0
    #: Measured sweep-to-sweep variation at stationarity; the drift threshold
    #: when ``tolerance`` is left at ``None``.
    noise_floor: float | None = None
    history: List[Dict[str, object]] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    def summary(self) -> Dict[str, object]:
        return {
            "proposed": self.proposed,
            "accepted": self.accepted,
            "acceptance_rate": self.acceptance_rate,
            "rejected_span": self.rejected_span,
            "rejected_primary_loop": self.rejected_primary_loop,
            "rejected_parallel": self.rejected_parallel,
            "rejected_degenerate": self.rejected_degenerate,
            "converged": self.converged,
            "sweeps": self.sweeps,
            "noise_floor": self.noise_floor,
        }


def _endpoint_owner(vertices) -> Dict[Endpoint, Hashable]:
    owner: Dict[Endpoint, Hashable] = {}
    for vertex in vertices:
        vertex.validate()
        for endpoint in vertex.endpoints.values():
            owner[endpoint] = vertex.vertex_id
    return owner


def reduced_edges(
    strands: Sequence[Strand],
    owner: Mapping[Endpoint, Hashable],
    node_index: Mapping[Hashable, int],
) -> List[Tuple[int, int]]:
    """Collapse strands onto the junction--strand multigraph."""
    return [
        (node_index[owner[left]], node_index[owner[right]])
        for left, right in strands
    ]


def loop_order_snapshot(
    n_nodes: int,
    edges: Sequence[Tuple[int, int]],
) -> Dict[int, float]:
    """Normalized loop-order distribution, for convergence testing."""
    from hygel_martini.property_extract.cyclic_topology import loop_order_histogram

    histogram = loop_order_histogram(n_nodes, list(edges))
    total = sum(histogram.values())
    if not total:
        return {}
    return {order: count / total for order, count in histogram.items()}


def total_variation_distance(
    left: Mapping[int, float],
    right: Mapping[int, float],
) -> float:
    """Half the L1 distance between two normalized loop-order distributions."""
    orders = set(left) | set(right)
    return 0.5 * sum(abs(left.get(order, 0.0) - right.get(order, 0.0)) for order in orders)


_IMAGE_SHIFTS = np.array(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=float,
)


def _normalize_box(box) -> np.ndarray | None:
    """Accept a 3-vector (orthorhombic) or a 3x3 matrix of cell vectors."""
    if box is None:
        return None
    array = np.asarray(box, dtype=float)
    if array.shape == (3,):
        return np.diag(array)
    if array.shape == (3, 3):
        return array
    raise ValueError(f"box must be a 3-vector or a 3x3 matrix, got shape {array.shape}")


def _minimum_image(delta: np.ndarray, cell: np.ndarray | None) -> np.ndarray:
    """Shortest periodic image of ``delta``.

    Rows of ``cell`` are the supercell vectors.  Fractional rounding alone is
    not exact for a strongly non-orthogonal cell -- the diamond seed uses FCC
    primitive vectors at 60 degrees -- so the 27 neighbouring images are
    checked and the shortest taken.
    """
    if cell is None:
        return delta
    fractional = np.linalg.solve(cell.T, delta)
    fractional -= np.round(fractional)
    candidates = (fractional + _IMAGE_SHIFTS) @ cell
    return candidates[np.argmin(np.einsum("ij,ij->i", candidates, candidates))]


def span_constrained_rewire(
    vertices,
    strands: Sequence[Strand],
    positions: Mapping[Hashable, np.ndarray],
    max_span: float,
    box: Sequence[float] | None = None,
    seed: int | None = None,
    max_sweeps: int = 200,
    swaps_per_sweep: int | None = None,
    tolerance: float | None = None,
    patience: int = 3,
    allow_primary_loops: bool = True,
    allow_parallel_strands: bool = True,
    record_history: bool = True,
) -> RewiringResult:
    """Rewire ``strands`` under a span cutoff until the loop spectrum settles.

    ``max_span`` is a distance in the same units as ``positions``; a swap is
    accepted only when both new strands span no more than that under the
    minimum image convention of ``box``, which may be a 3-vector for an
    orthorhombic cell or a 3x3 matrix of cell vectors (pass ``None`` for a
    non-periodic cell).

    Convergence tests for *drift*, not for a quiet sweep.  At a large span
    cutoff nearly every proposal is accepted, so each sweep decorrelates the
    configuration completely and successive loop-order distributions differ by
    ordinary sampling noise forever; a sweep-to-sweep criterion would then
    never fire however stationary the process is.  Instead a rolling window of
    ``2 * patience`` sweeps is kept and the older half is compared with the
    newer half, each averaged first.  Averaging suppresses the per-sweep noise
    while leaving any systematic drift intact, so the test fires when the
    distribution has stopped moving rather than when it happens to sit still.

    The threshold is measured, not guessed.  The residual sweep-to-sweep
    variation of the loop-order distribution is a finite-size effect that
    scales as one over the square root of the strand count, so any fixed
    ``tolerance`` is simultaneously too loose for a large cell and unreachable
    for a small one --- on a 256-strand diamond cell the floor sits near 0.056,
    so a tolerance of 0.02 can never fire however stationary the process is.
    With ``tolerance=None`` (the default) the noise floor is therefore
    estimated from consecutive snapshots inside each half-window, and
    convergence is declared once the between-half drift falls to that level.
    Pass a float to override with a fixed threshold.

    A sweep defaults to one proposal per strand.  Returning with
    ``converged=False`` means the sweep budget ran out, which is reported
    rather than raised because a partially rewired network is still a valid
    input --- it simply sits closer to the seed than requested.
    """
    rng = Random(seed)
    working: List[Strand] = [tuple(strand) for strand in strands]
    if not working:
        return RewiringResult(strands=working, converged=True)

    owner = _endpoint_owner(vertices)
    missing = [
        endpoint
        for strand in working
        for endpoint in strand
        if endpoint not in owner
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} strand endpoint(s) belong to no vertex, e.g. "
            f"{sorted(map(repr, missing))[:3]}"
        )

    node_index = {vertex.vertex_id: index for index, vertex in enumerate(vertices)}
    coordinates = {
        key: np.asarray(value, dtype=float) for key, value in positions.items()
    }
    box_vector = _normalize_box(box)
    if swaps_per_sweep is None:
        swaps_per_sweep = len(working)

    def span(first: Endpoint, second: Endpoint) -> float:
        delta = coordinates[owner[first]] - coordinates[owner[second]]
        return float(np.linalg.norm(_minimum_image(delta, box_vector)))

    # Multiset of junction pairs, so parallel strands can be detected in O(1).
    def pair_key(strand: Strand) -> Tuple[int, int]:
        left, right = node_index[owner[strand[0]]], node_index[owner[strand[1]]]
        return (left, right) if left <= right else (right, left)

    pair_counts: Dict[Tuple[int, int], int] = {}
    for strand in working:
        key = pair_key(strand)
        pair_counts[key] = pair_counts.get(key, 0) + 1

    result = RewiringResult(strands=working)
    n_nodes = len(vertices)

    window_size = 2 * max(int(patience), 1)
    window: List[Mapping[int, float]] = []

    initial = loop_order_snapshot(
        n_nodes, reduced_edges(working, owner, node_index)
    )
    if record_history:
        result.history.append({"sweep": 0, "distribution": dict(initial), "shift": None})
    for sweep in range(1, int(max_sweeps) + 1):
        for _ in range(int(swaps_per_sweep)):
            result.proposed += 1
            if len(working) < 2:
                break
            first_index = rng.randrange(len(working))
            second_index = rng.randrange(len(working))
            if first_index == second_index:
                result.rejected_degenerate += 1
                continue

            a_left, a_right = working[first_index]
            b_left, b_right = working[second_index]
            if len({a_left, a_right, b_left, b_right}) < 4:
                result.rejected_degenerate += 1
                continue

            # Both reconnections preserve every junction's degree.
            if rng.random() < 0.5:
                new_first = (a_left, b_right)
                new_second = (b_left, a_right)
            else:
                new_first = (a_left, b_left)
                new_second = (a_right, b_right)

            if span(*new_first) > max_span or span(*new_second) > max_span:
                result.rejected_span += 1
                continue

            first_key = pair_key(new_first)
            second_key = pair_key(new_second)
            if not allow_primary_loops and (
                first_key[0] == first_key[1] or second_key[0] == second_key[1]
            ):
                result.rejected_primary_loop += 1
                continue

            old_first_key = pair_key(working[first_index])
            old_second_key = pair_key(working[second_index])
            if not allow_parallel_strands:
                projected = dict(pair_counts)
                for key in (old_first_key, old_second_key):
                    projected[key] -= 1
                for key in (first_key, second_key):
                    projected[key] = projected.get(key, 0) + 1
                if projected[first_key] > 1 or projected[second_key] > 1:
                    result.rejected_parallel += 1
                    continue

            for key in (old_first_key, old_second_key):
                pair_counts[key] -= 1
                if pair_counts[key] == 0:
                    del pair_counts[key]
            for key in (first_key, second_key):
                pair_counts[key] = pair_counts.get(key, 0) + 1

            working[first_index] = new_first
            working[second_index] = new_second
            result.accepted += 1

        result.sweeps = sweep
        current = loop_order_snapshot(
            n_nodes, reduced_edges(working, owner, node_index)
        )
        window.append(current)
        if len(window) > window_size:
            window.pop(0)

        shift = None
        noise = None
        if len(window) == window_size:
            half = window_size // 2
            shift = total_variation_distance(
                _mean_distribution(window[:half]),
                _mean_distribution(window[half:]),
            )
            noise = _noise_floor(window)
        if record_history:
            result.history.append(
                {
                    "sweep": sweep,
                    "distribution": dict(current),
                    "shift": shift,
                    "noise_floor": noise,
                }
            )

        if shift is not None:
            threshold = float(tolerance) if tolerance is not None else noise
            if threshold is not None and shift <= threshold:
                result.converged = True
                result.noise_floor = noise
                break

    _assert_degree_preserved(vertices, strands, result.strands, owner)
    return result


def _noise_floor(window: Sequence[Mapping[int, float]]) -> float:
    """Sweep-to-sweep variation within the window halves.

    Consecutive snapshots inside one half differ only by sampling noise, since
    any systematic motion is what the between-half comparison measures.  The
    mean of those consecutive distances is therefore an estimate of the floor
    the drift signal has to clear.
    """
    half = len(window) // 2
    distances = [
        total_variation_distance(block[index], block[index + 1])
        for block in (window[:half], window[half:])
        for index in range(len(block) - 1)
    ]
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def _mean_distribution(
    snapshots: Sequence[Mapping[int, float]],
) -> Dict[int, float]:
    """Average several loop-order distributions order by order."""
    if not snapshots:
        return {}
    orders = set().union(*(set(item) for item in snapshots))
    count = float(len(snapshots))
    return {
        order: sum(item.get(order, 0.0) for item in snapshots) / count
        for order in orders
    }


def _assert_degree_preserved(
    vertices,
    original: Sequence[Strand],
    rewired: Sequence[Strand],
    owner: Mapping[Endpoint, Hashable],
) -> None:
    """A swap that changed a junction's functionality is a bug, not an outcome."""
    if len(original) != len(rewired):
        raise AssertionError(
            f"Rewiring changed the strand count: {len(original)} -> {len(rewired)}"
        )

    def degrees(strands: Sequence[Strand]) -> Dict[Hashable, int]:
        counts: Dict[Hashable, int] = {vertex.vertex_id: 0 for vertex in vertices}
        for left, right in strands:
            counts[owner[left]] += 1
            counts[owner[right]] += 1
        return counts

    before, after = degrees(original), degrees(rewired)
    changed = {key: (before[key], after[key]) for key in before if before[key] != after[key]}
    if changed:
        raise AssertionError(
            f"Rewiring changed junction functionality at {len(changed)} junction(s), "
            f"e.g. {dict(list(changed.items())[:3])}"
        )

    used = [endpoint for strand in rewired for endpoint in strand]
    if len(used) != len(set(used)):
        raise AssertionError("Rewiring left an endpoint carrying more than one strand")
