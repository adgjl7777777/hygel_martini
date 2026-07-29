from __future__ import annotations

from hygel_martini.hydrogel_builder.core_utils.layout.local_matching import (
    LOCAL_COORDS,
    LocalVertex,
    evaluate_matching_plan,
    plan_balanced_cycle_matchings,
)


def _two_vertex_fixture():
    left = LocalVertex("left", {coord: ("left", coord) for coord in LOCAL_COORDS})
    right = LocalVertex(
        "right", {coord: ("right", coord) for coord in LOCAL_COORDS}
    )
    chain_edges = [
        (("left", coord), ("right", coord))
        for coord in LOCAL_COORDS
    ]
    return [left, right], chain_edges


def test_different_local_transitions_join_two_circuits_into_one_cycle() -> None:
    vertices, chain_edges = _two_vertex_fixture()

    same_axis = evaluate_matching_plan(
        vertices,
        chain_edges,
        {"left": "x", "right": "x"},
    )
    mixed_axes = evaluate_matching_plan(
        vertices,
        chain_edges,
        {"left": "x", "right": "y"},
    )

    assert same_axis.diagnostics.component_count == 2
    assert mixed_axes.diagnostics.component_count == 1
    assert mixed_axes.diagnostics.degree_violations == {}
    assert mixed_axes.is_single_cycle


def test_exact_planner_finds_balanced_single_cycle() -> None:
    vertices, chain_edges = _two_vertex_fixture()

    plan = plan_balanced_cycle_matchings(
        vertices,
        chain_edges,
        seed=101,
        exact_limit=12,
    )

    assert plan.is_single_cycle
    assert plan.diagnostics.component_count == 1
    assert plan.diagnostics.largest_component_size == 8
    assert plan.diagnostics.degree_violations == {}
    counts = list(plan.diagnostics.axis_counts.values())
    assert max(counts) - min(counts) <= 1
