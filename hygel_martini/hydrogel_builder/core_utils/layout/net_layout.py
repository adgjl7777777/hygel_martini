"""Coordinate layout driven by a periodic net.

The diamond layout in :mod:`proto_layout` enumerates four hard-coded sublattice
offsets and three x/y/z orientation classes.  Neither generalizes: a
primitive-cubic seed has one site per cell, and a hexafunctional junction has
fifteen matching states rather than three.

This module builds the same :class:`LayoutPlan` from a
:class:`~hygel_martini.hydrogel_builder.core_utils.layout.nets.NetDefinition`
instead, so the net supplies the geometry and the coordination number:

1. materialize the net (junction sites and strand endpoints);
2. optionally rewire under a span cutoff, moving the topology off the regular
   net toward a representative loop spectrum;
3. plan a transition system --- for an even-coordination net this is an
   Eulerian circuit, so one circuit is obtained directly;
4. emit one backbone chain per strand and one junction per site, carrying the
   planned endpoint edges into the runtime as the diamond path does.

Order matters and is enforced: rewiring changes which junctions are adjacent,
so a transition system planned before it would no longer describe the network.

Strand geometry is computed under the minimum-image convention. A strand that
crosses the periodic boundary has its midpoint on the short side, not halfway
across the box, and getting that wrong would place chains through the cell
rather than across its face.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from hygel_martini.core.pbc import minimum_image, normalize_cell
from hygel_martini.hydrogel_builder.core_utils.layout.local_matching import (
    plan_single_circuit,
)
from hygel_martini.hydrogel_builder.core_utils.layout.nets import (
    NetDefinition,
    build_periodic_net,
    get_net,
)
from hygel_martini.hydrogel_builder.core_utils.layout.proto_layout import (
    LayoutCell,
    LayoutPlan,
    LinkPlacement,
)
from hygel_martini.hydrogel_builder.core_utils.layout.rewire import (
    span_constrained_rewire,
)

__all__ = ["NetLayoutResult", "generate_net_layout_plan"]


class NetLayoutResult:
    """A layout plan plus the topology record that produced it."""

    def __init__(self, layout_plan, matching_plan, rewiring, net, repeats, cell):
        self.layout_plan = layout_plan
        self.matching_plan = matching_plan
        self.rewiring = rewiring
        self.net = net
        self.repeats = repeats
        self.cell = cell

    def summary(self) -> Dict[str, Any]:
        diagnostics = self.matching_plan.diagnostics
        record: Dict[str, Any] = {
            "net": self.net.name,
            "coordination": self.net.coordination,
            "repeats": list(self.repeats),
            "junction_count": len(self.layout_plan.links),
            "strand_count": len(self.layout_plan.cells),
            "circuit_count": diagnostics.component_count,
            "single_circuit": self.matching_plan.is_single_cycle,
            "endpoint_degree_violations": len(diagnostics.degree_violations),
            "matching_state_counts": dict(diagnostics.state_counts),
            "functionality_counts": dict(diagnostics.functionality_counts),
            "rewired": self.rewiring is not None,
        }
        if self.rewiring is not None:
            record["rewiring"] = self.rewiring.summary()
        return record


def _effective_span(length: float, retreat: float, index: int) -> float:
    """Chain span after both ends stop short of their junction sites."""
    effective = length - 2.0 * retreat
    if effective <= 0.0:
        raise ValueError(
            f"Strand {index} spans {length:.3f} but the junction arms plus "
            f"external bonds already occupy {2.0 * retreat:.3f} of it; increase "
            "cell_parameter or shorten the crosslinker arms"
        )
    return effective


def _junction_of(vertices) -> Dict[Any, Any]:
    owner = {}
    for vertex in vertices:
        for endpoint in vertex.endpoints.values():
            owner[endpoint] = vertex.vertex_id
    return owner


def generate_net_layout_plan(
    proto_plan,
    backbone_defs: List[Dict[str, Any]],
    linker_defs: List[Dict[str, Any]],
    net: NetDefinition | str,
    repeats: Sequence[int] | int,
    cell_parameter: float,
    linker_library=None,
    max_span: float | None = None,
    rewire_seed: int | None = None,
    rewire_kwargs: Dict[str, Any] | None = None,
    plan_seed: int | None = None,
) -> NetLayoutResult:
    """Build a :class:`LayoutPlan` on a periodic net.

    ``max_span`` is the rewiring cutoff, in the same length units as
    ``cell_parameter``; pass ``None`` to keep the regular net, which leaves the
    loop-order spectrum a single spike at the net's fundamental cycle size and
    --- for the bipartite ``dia`` and ``pcu`` seeds --- with no odd-order loops
    at all.  That is a legitimate choice for an idealized benchmark and a poor
    one for a representative network, so it is a parameter rather than a
    default.
    """
    definition = net if isinstance(net, NetDefinition) else get_net(net)
    counts = (repeats, repeats, repeats) if isinstance(repeats, int) else tuple(
        int(value) for value in repeats
    )

    vertices, strands, positions = build_periodic_net(
        definition, counts, cell_parameter
    )
    cell = normalize_cell(
        np.asarray(definition.lattice_vectors, dtype=float)
        * float(cell_parameter)
        * np.asarray(counts, dtype=float)[:, None]
    )

    rewiring = None
    if max_span is not None:
        # Rewiring may close a strand back onto its own junction. That is
        # physical, and the topology layer counts such primary loops, but the
        # straight-segment coordinate model below cannot place one, so the
        # default here forbids them rather than failing later with the
        # geometry as the visible symptom.
        rewire_kwargs = dict(rewire_kwargs or {})
        rewire_kwargs.setdefault("allow_primary_loops", False)
        rewiring = span_constrained_rewire(
            vertices,
            strands,
            positions,
            max_span=float(max_span),
            box=cell,
            seed=rewire_seed,
            **rewire_kwargs,
        )
        strands = rewiring.strands

    # Planned after any rewiring: rewiring changes adjacency, so a transition
    # system planned before it would describe a different network.
    matching_plan = plan_single_circuit(vertices, strands)

    edges_by_vertex: Dict[Any, Tuple] = {
        choice.vertex_id: tuple(tuple(edge) for edge in choice.edges)
        for choice in matching_plan.choices
    }
    owner = _junction_of(vertices)

    # The populator names a chain's two ends (planned_chain_id, 0|1), head
    # first, and the runtime router looks planned endpoints up under exactly
    # those names. Net endpoint identifiers are therefore translated into that
    # convention here: the strand is laid head-to-tail along
    # left-junction -> right-junction, so left is end 0.
    endpoint_name: Dict[Any, Tuple[int, int]] = {}
    for index, (left, right) in enumerate(strands):
        endpoint_name[left] = (index, 0)
        endpoint_name[right] = (index, 1)

    proto_backbone = getattr(proto_plan, "proto_backbone", None)
    proto_length = float(getattr(proto_backbone, "length", 0.0) or 0.0)

    # A chain must not reach the junction centre: the junction's own beads sit
    # there, and six chain ends converging on one point give coincident
    # coordinates, whose r=0 bonded terms produce NaN forces at the first EM
    # step (observed directly on the first end-to-end run). Each end therefore
    # stops one arm length plus one external-bond length short of its site.
    retreat = 0.0
    if linker_library is not None and getattr(linker_library, "records", None):
        template = linker_library.records[0].template
        retreat = float(getattr(template, "span_length", 0.0) or 0.0) / 2.0
        external_lengths = [
            float(params.get("c0") or 0.0)
            for group in getattr(template, "stub_bonds", [])
            for _, params in group
        ]
        if external_lengths:
            retreat += sum(external_lengths) / len(external_lengths)

    # On a lattice, straight-segment placement collides in two ways that the
    # coincidence jitter cannot repair. Parallel strands (secondary loops from
    # rewiring) lie on one segment bead-for-bead; and any rewired strand
    # longer than one lattice step is collinear with the lattice line, so it
    # runs straight through the intermediate junctions and through every
    # shorter strand on that line -- measured directly as whole-chain contact
    # trains at ~0.006 nm and EM stuck at Epot ~ 1e21. Following the original
    # layout's approach of deforming chains at placement time, every strand is
    # bowed off its line with a half-sine (ends fixed at the junctions), with
    # a real-space amplitude that clears a junction bead and grows per
    # duplicate of the same junction pair, at a distinct azimuth per strand.
    pair_seen: Dict[Tuple, int] = {}
    proto_positions_base = getattr(proto_backbone, "positions", None)
    BOW_CLEARANCE_NM = 0.5

    def _bowed_proto(strand_index: int, duplicate_rank: int,
                     length_scale: float) -> np.ndarray | None:
        if proto_positions_base is None or len(proto_positions_base) < 3:
            return None
        base = np.array(proto_positions_base, dtype=np.float64)
        span_axis = base[-1] - base[0]
        norm = float(np.linalg.norm(span_axis))
        if norm < 1e-9 or length_scale < 1e-9:
            return None
        axis = span_axis / norm
        seed_vec = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(seed_vec, axis))) > 0.9:
            seed_vec = np.array([0.0, 1.0, 0.0])
        perp1 = seed_vec - float(np.dot(seed_vec, axis)) * axis
        perp1 /= float(np.linalg.norm(perp1))
        perp2 = np.cross(axis, perp1)
        azimuth = 2.399963229728653 * (strand_index + 7 * duplicate_rank)
        lateral = np.cos(azimuth) * perp1 + np.sin(azimuth) * perp2
        # instantiate_backbone multiplies proto coordinates by length_scale,
        # so a real-space amplitude has to be expressed in proto units here.
        amplitude = (BOW_CLEARANCE_NM * (1.0 + 0.5 * duplicate_rank)) / length_scale
        t = np.linspace(0.0, 1.0, len(base))
        return base + np.outer(np.sin(np.pi * t) * amplitude, lateral)

    cells: List[LayoutCell] = []
    for index, (left, right) in enumerate(strands):
        start = np.asarray(positions[owner[left]], dtype=float)
        delta = minimum_image(
            np.asarray(positions[owner[right]], dtype=float) - start, cell
        )
        length = float(np.linalg.norm(delta))
        if length < 1e-9:
            # A primary loop: both ends of one strand on the same junction.
            # It is a real network feature and the topology layer counts it,
            # but this coordinate model places a chain as a straight segment
            # between two sites, and a loop has no such segment. Refusing is
            # honest; silently straightening it into a zero-length chain, or
            # dropping it, would put the coordinates and the topology into
            # disagreement -- exactly what the construction audit exists to
            # catch.
            raise ValueError(
                f"Strand {index} returns to junction {owner[left]!r} (a primary "
                "loop). This layout places each strand as a straight segment "
                "between two junction sites and cannot express one. Either pass "
                "allow_primary_loops=False in rewire_kwargs, or use a layout "
                "that can place a loop excursion."
            )
        cells.append(
            LayoutCell(
                origin=start + 0.5 * delta,
                direction=delta / length,
                backbone_definition=(backbone_defs[0] if backbone_defs else {}),
                cell_index=(0, 0, 0),
                metadata={
                    "strand_index": index,
                    "planned_chain_id": index,
                    "endpoints": (left, right),
                    "junctions": (owner[left], owner[right]),
                    "strand_length": length,
                    # Stretch or compress the prototype chain to span the
                    # junction gap minus the retreat at each end; without this
                    # a chain keeps its prototype contour and either overlaps
                    # its junctions or never reaches them.
                    "length_scale": (
                        _effective_span(length, retreat, index) / proto_length
                        if proto_length > 1e-9 else 1.0
                    ),
                    "net": definition.name,
                },
            )
        )
        pair = tuple(sorted((owner[left], owner[right]), key=repr))
        duplicate_rank = pair_seen.get(pair, 0)
        pair_seen[pair] = duplicate_rank + 1
        bowed = _bowed_proto(index, duplicate_rank,
                             cells[-1].metadata["length_scale"])
        if bowed is not None:
            cells[-1].metadata["proto_positions"] = bowed
            cells[-1].metadata["parallel_rank"] = duplicate_rank

    link_definition = linker_defs[0] if linker_defs else {}
    template_id = None
    if linker_library is not None and getattr(linker_library, "records", None):
        template_id = linker_library.records[0].template.id

    links: List[LinkPlacement] = []
    for index, vertex in enumerate(vertices):
        metadata: Dict[str, Any] = {
            "junction_index": index,
            "junction_id": vertex.vertex_id,
            "functionality": vertex.functionality,
            "planned_endpoint_edges": tuple(
                tuple(endpoint_name[endpoint] for endpoint in edge)
                for edge in edges_by_vertex.get(vertex.vertex_id, ())
            ),
            "net": definition.name,
            # A multi-arm template is placed on its stub centroid, so the
            # anchor is the site itself and only an orientation applies.
            "orientation": np.eye(3, dtype=float),
        }
        if template_id is not None:
            metadata["linker_template_id"] = template_id
        links.append(
            LinkPlacement(
                anchor_position=np.asarray(positions[vertex.vertex_id], dtype=float),
                axis_direction=np.array([1.0, 0.0, 0.0]),
                linker_definition=link_definition,
                connected_cells=(index, index),
                metadata=metadata,
            )
        )

    layout_plan = LayoutPlan(proto_plan=proto_plan, cells=cells, links=links)
    return NetLayoutResult(
        layout_plan, matching_plan, rewiring, definition, counts, cell
    )
