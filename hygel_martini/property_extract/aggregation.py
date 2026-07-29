"""Contact-graph aggregation primitives with explicit cutoff provenance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geometry import orthorhombic_box_lengths, wrap_positions


@dataclass(frozen=True)
class ContactGraphResult:
    components: tuple[tuple[int, ...], ...]
    edge_contact_counts: dict[tuple[int, int], int]
    cutoff: float

    @property
    def largest_component_size(self) -> int:
        return len(self.components[0]) if self.components else 0

    @property
    def n_components(self) -> int:
        return len(self.components)


def chain_contact_graph(
    chain_site_positions: list[np.ndarray],
    box: np.ndarray,
    cutoff: float,
) -> ContactGraphResult:
    """Build a chain graph from any inter-chain site pair within ``cutoff``.

    ``chain_site_positions`` may contain PPO-only sites, all polymer sites, or
    another preregistered selection.  That selection and the cutoff remain part
    of the observable definition and must be reported with results.
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    lengths = orthorhombic_box_lengths(box)
    wrapped: list[np.ndarray] = []
    for positions in chain_site_positions:
        pos = np.asarray(positions, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) == 0:
            raise ValueError("each chain selection must have shape (n,3), n > 0")
        wrapped.append(wrap_positions(pos, lengths))

    # A single periodic tree scales much better than constructing/querying one
    # tree for every chain (important for 100+ chain micelle boxes).  Owner
    # labels recover the chain graph and contact multiplicities.
    counts_per_chain = np.asarray([len(pos) for pos in wrapped], dtype=int)
    owners = np.repeat(np.arange(len(wrapped), dtype=int), counts_per_chain)
    all_positions = np.vstack(wrapped)
    tree = cKDTree(all_positions, boxsize=lengths)
    site_pairs = tree.query_pairs(float(cutoff), output_type="ndarray")
    adjacency = [set() for _ in wrapped]
    counts: dict[tuple[int, int], int] = {}
    if len(site_pairs):
        owner_pairs = owners[site_pairs]
        owner_pairs.sort(axis=1)
        interchain = owner_pairs[owner_pairs[:, 0] != owner_pairs[:, 1]]
        if len(interchain):
            unique_pairs, pair_counts = np.unique(interchain, axis=0, return_counts=True)
            for pair, count in zip(unique_pairs, pair_counts):
                i, j = int(pair[0]), int(pair[1])
                adjacency[i].add(j)
                adjacency[j].add(i)
                counts[(i, j)] = int(count)

    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in range(len(wrapped)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    components.sort(key=lambda item: (-len(item), item))
    return ContactGraphResult(tuple(components), counts, float(cutoff))
