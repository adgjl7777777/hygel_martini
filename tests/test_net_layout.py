"""Coordinate layout driven by a periodic net rather than by the diamond
sublattice constants."""

from __future__ import annotations

import numpy as np
import pytest

from hygel_martini.core.pbc import minimum_image
from hygel_martini.hydrogel_builder.core_utils.layout.net_layout import (
    generate_net_layout_plan,
)
from hygel_martini.property_extract.cyclic_topology import cyclic_topology_report

BACKBONES = [{"id": "BB1"}]
LINKERS = [{"id": "LNK"}]


class _Proto:
    """Stand-in for the proto plan, which LayoutPlan only carries through."""


def _layout(**kwargs):
    return generate_net_layout_plan(_Proto(), BACKBONES, LINKERS, **kwargs)


def _reduced(result):
    junctions = {}
    edges = []
    for cell in result.layout_plan.cells:
        left, right = cell.metadata["junctions"]
        for key in (left, right):
            junctions.setdefault(key, len(junctions))
        edges.append((junctions[left], junctions[right]))
    return len(junctions), edges


@pytest.mark.parametrize(
    "net,repeats,coordination,junctions,strands",
    [("pcu", 4, 6, 64, 192), ("dia", 4, 4, 128, 256)],
)
def test_the_net_supplies_the_counts(net, repeats, coordination, junctions, strands) -> None:
    result = _layout(net=net, repeats=repeats, cell_parameter=1.0)
    summary = result.summary()

    assert summary["coordination"] == coordination
    assert summary["junction_count"] == junctions
    assert summary["strand_count"] == strands
    # each junction carries f/2 planned edges
    for link in result.layout_plan.links:
        assert len(link.metadata["planned_endpoint_edges"]) == coordination // 2
        assert link.metadata["functionality"] == coordination


@pytest.mark.parametrize("net,repeats", [("pcu", 4), ("dia", 4)])
def test_an_unrewired_net_places_every_strand_at_the_same_length(net, repeats) -> None:
    result = _layout(net=net, repeats=repeats, cell_parameter=2.0)

    lengths = [cell.metadata["strand_length"] for cell in result.layout_plan.cells]
    assert lengths == pytest.approx([lengths[0]] * len(lengths))
    directions = np.array([cell.direction for cell in result.layout_plan.cells])
    assert np.linalg.norm(directions, axis=1) == pytest.approx(np.ones(len(directions)))


def test_strand_geometry_uses_the_minimum_image():
    # Every pcu strand joins nearest neighbours, so a midpoint computed without
    # the minimum image would land in the middle of the box for the strands
    # that wrap, and the length would be (L-1) lattice steps instead of one.
    parameter = 3.0
    result = _layout(net="pcu", repeats=4, cell_parameter=parameter)

    lengths = [cell.metadata["strand_length"] for cell in result.layout_plan.cells]
    assert max(lengths) == pytest.approx(parameter)

    box = np.diag(result.cell)
    origins = np.array([cell.origin for cell in result.layout_plan.cells])
    assert np.all(origins >= -1e-9)
    assert np.all(origins <= box + 1e-9)


def test_the_planner_runs_after_rewiring_not_before() -> None:
    # Rewiring changes which junctions are adjacent, so a plan made before it
    # would describe a different network. The planned edges must therefore
    # refer to endpoints that the final strand list actually uses.
    result = _layout(
        net="pcu", repeats=4, cell_parameter=3.0,
        max_span=6.0, rewire_seed=0, rewire_kwargs={"max_sweeps": 20},
    )

    assert result.rewiring is not None and result.rewiring.accepted > 0
    # Planned edges are translated into the populator's (chain_id, end)
    # convention, so together they must name both ends of every strand.
    expected = {
        (cell.metadata["planned_chain_id"], end)
        for cell in result.layout_plan.cells
        for end in (0, 1)
    }
    planned = {
        endpoint
        for link in result.layout_plan.links
        for edge in link.metadata["planned_endpoint_edges"]
        for endpoint in edge
    }
    assert planned == expected


def test_an_unrewired_seed_keeps_the_net_loop_spectrum() -> None:
    result = _layout(net="pcu", repeats=4, cell_parameter=1.0)
    report = cyclic_topology_report(*_reduced(result))

    assert report["girth"] == 4  # pcu fundamental cycle size
    assert report["bipartite"] is True
    assert report["odd_loop_order_count"] == 0
    assert report["distinct_vertex_symbol_count"] == 1


def test_rewiring_moves_the_layout_off_the_regular_net() -> None:
    result = _layout(
        net="pcu", repeats=4, cell_parameter=3.0,
        max_span=6.0, rewire_seed=0, rewire_kwargs={"max_sweeps": 30},
    )
    report = cyclic_topology_report(*_reduced(result))

    assert report["bipartite"] is False
    assert report["odd_loop_order_count"] > 0
    assert report["distinct_vertex_symbol_count"] > 1
    # and the network is still traceable as one circuit
    assert result.summary()["single_circuit"]


def test_rewiring_respects_the_span_cutoff_in_the_placed_geometry() -> None:
    cutoff = 6.0
    result = _layout(
        net="pcu", repeats=4, cell_parameter=3.0,
        max_span=cutoff, rewire_seed=1, rewire_kwargs={"max_sweeps": 30},
    )

    lengths = [cell.metadata["strand_length"] for cell in result.layout_plan.cells]
    assert max(lengths) <= cutoff + 1e-9
    assert min(lengths) > 0.0


def test_primary_loops_are_forbidden_by_default_when_placing_coordinates() -> None:
    # The straight-segment model cannot express a strand returning to its own
    # junction, so rewiring for a coordinate build refuses to create one.
    result = _layout(
        net="pcu", repeats=4, cell_parameter=3.0,
        max_span=6.0, rewire_seed=0, rewire_kwargs={"max_sweeps": 30},
    )

    assert result.rewiring.rejected_primary_loop > 0
    assert cyclic_topology_report(*_reduced(result))["primary_loop_count"] == 0


def test_a_primary_loop_is_refused_rather_than_straightened() -> None:
    # Explicitly allowing them must not produce a zero-length chain or a
    # dropped strand; either would leave coordinates and topology disagreeing.
    with pytest.raises(ValueError, match="primary loop"):
        _layout(
            net="pcu", repeats=4, cell_parameter=3.0,
            max_span=6.0, rewire_seed=0,
            rewire_kwargs={"max_sweeps": 30, "allow_primary_loops": True},
        )


def test_secondary_loops_are_placeable() -> None:
    # Two strands between the same pair of junctions are both straight
    # segments, so unlike a primary loop they need no special handling.
    result = _layout(
        net="pcu", repeats=4, cell_parameter=3.0,
        max_span=6.0, rewire_seed=0, rewire_kwargs={"max_sweeps": 30},
    )
    report = cyclic_topology_report(*_reduced(result))

    assert report["secondary_loop_count"] > 0
    assert all(cell.metadata["strand_length"] > 0 for cell in result.layout_plan.cells)
