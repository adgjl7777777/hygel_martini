"""General even-functionality transition systems.

The tetrafunctional x/y/z vocabulary of the diamond builder is one instance of
a matching state; these tests pin the generalization and the Eulerian seed that
replaces search at higher functionality.
"""

from __future__ import annotations

import pytest

from hygel_martini.hydrogel_builder.core_utils.layout.local_matching import (
    AXIS_BY_STATE,
    LOCAL_COORDS,
    MATCHINGS_BY_AXIS,
    LocalVertex,
    matching_edges_for_axis,
    matching_edges_for_state,
    matching_state_count,
    perfect_matchings,
    plan_single_circuit,
    state_for_pairing,
)

DIRECTIONS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}


def _pcu(repeats: int):
    """Periodic primitive-cubic net: hexafunctional junctions."""
    sites = [
        (i, j, k)
        for i in range(repeats)
        for j in range(repeats)
        for k in range(repeats)
    ]
    vertices = [
        LocalVertex(site, {arm: (site, arm) for arm in range(6)}) for site in sites
    ]
    strands = []
    for site in sites:
        for arm in (0, 2, 4):
            step = DIRECTIONS[arm]
            neighbour = tuple(
                (site[axis] + step[axis]) % repeats for axis in range(3)
            )
            strands.append(((site, arm), (neighbour, OPPOSITE[arm])))
    return vertices, strands


def _dia(repeats: int):
    """Periodic diamond net keyed by generic arm indices, not LOCAL_COORDS."""
    steps = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    cells = [
        (i, j, k)
        for i in range(repeats)
        for j in range(repeats)
        for k in range(repeats)
    ]
    vertices = []
    for cell in cells:
        for sublattice in (0, 1):
            node = cell + (sublattice,)
            vertices.append(LocalVertex(node, {arm: (node, arm) for arm in range(4)}))
    strands = []
    for cell in cells:
        origin = cell + (0,)
        for arm, step in enumerate(steps):
            target = tuple(
                (cell[axis] + step[axis]) % repeats for axis in range(3)
            ) + (1,)
            strands.append(((origin, arm), (target, arm)))
    return vertices, strands


def test_matching_state_counts_are_the_double_factorial() -> None:
    assert matching_state_count(4) == 3  # (4-1)!!
    assert matching_state_count(6) == 15  # (6-1)!!
    assert matching_state_count(8) == 105  # (8-1)!!
    assert len(perfect_matchings(6)) == 15


def test_odd_functionality_has_no_perfect_matching() -> None:
    with pytest.raises(ValueError, match="odd endpoint count"):
        perfect_matchings(5)

    with pytest.raises(ValueError, match="even number"):
        LocalVertex("v", {arm: ("v", arm) for arm in range(5)}).validate()


def test_axis_labels_are_the_first_three_general_states() -> None:
    # The tetrafunctional vocabulary must sit exactly on top of the general
    # enumeration, or existing diamond configurations change meaning.
    vertex = LocalVertex("v", {coord: ("v", coord) for coord in LOCAL_COORDS})
    for state, axis in AXIS_BY_STATE.items():
        from_state = {frozenset(edge) for edge in matching_edges_for_state(vertex, state)}
        from_axis = {
            frozenset((("v", left), ("v", right)))
            for left, right in MATCHINGS_BY_AXIS[axis]
        }
        assert from_state == from_axis
        assert from_state == {
            frozenset(edge) for edge in matching_edges_for_axis(vertex, axis)
        }


def test_axis_vocabulary_is_refused_for_non_tetrahedral_vertices() -> None:
    hexafunctional = LocalVertex("v", {arm: ("v", arm) for arm in range(6)})

    with pytest.raises(ValueError, match="not tetrahedral"):
        matching_edges_for_axis(hexafunctional, "x")

    with pytest.raises(ValueError, match="only defined for functionality 4"):
        matching_edges_for_state(hexafunctional, "x")


def test_state_for_pairing_round_trips() -> None:
    vertex = LocalVertex("v", {arm: ("v", arm) for arm in range(6)})
    for state in range(matching_state_count(6)):
        edges = matching_edges_for_state(vertex, state)
        assert state_for_pairing(vertex, edges) == state


@pytest.mark.parametrize("repeats", [2, 3, 4])
def test_eulerian_seed_gives_one_circuit_on_the_hexafunctional_net(repeats) -> None:
    # At f=6 a single circuit needs no search: every junction has even degree,
    # so an Eulerian circuit exists and is itself a one-circuit transition
    # system.  This is the property that replaces the f=4 annealing.
    vertices, strands = _pcu(repeats)
    plan = plan_single_circuit(vertices, strands)

    assert plan.diagnostics.component_count == 1
    assert plan.diagnostics.degree_violations == {}
    assert plan.is_single_cycle
    assert plan.diagnostics.functionality_counts == {6: len(vertices)}


@pytest.mark.parametrize("repeats", [2, 3])
def test_eulerian_seed_gives_one_circuit_on_the_tetrafunctional_net(repeats) -> None:
    vertices, strands = _dia(repeats)
    plan = plan_single_circuit(vertices, strands)

    assert plan.diagnostics.component_count == 1
    assert plan.is_single_cycle
    assert plan.diagnostics.functionality_counts == {4: len(vertices)}


def test_disconnected_input_yields_one_circuit_per_component() -> None:
    # Two independent two-functional junction pairs.
    vertices = [
        LocalVertex(name, {arm: (name, arm) for arm in range(2)})
        for name in ("a", "b", "c", "d")
    ]
    strands = [
        (("a", 0), ("b", 0)),
        (("a", 1), ("b", 1)),
        (("c", 0), ("d", 0)),
        (("c", 1), ("d", 1)),
    ]
    plan = plan_single_circuit(vertices, strands)

    assert plan.diagnostics.component_count == 2
    assert not plan.is_single_cycle


def test_unmatched_endpoint_is_refused_rather_than_partially_planned() -> None:
    vertices, strands = _pcu(2)
    with pytest.raises(ValueError, match="carry no strand"):
        plan_single_circuit(vertices, strands[:-1])


def test_reused_endpoint_is_refused() -> None:
    vertices, strands = _pcu(2)
    duplicated = list(strands) + [strands[0]]
    with pytest.raises(ValueError, match="each endpoint carries exactly one strand"):
        plan_single_circuit(vertices, duplicated)


def test_odd_strand_degree_is_refused() -> None:
    # A junction declaring six arms but wired with five strands cannot be
    # matched; a silently partial plan would be worse than a refusal.
    vertices = [
        LocalVertex("hub", {arm: ("hub", arm) for arm in range(6)}),
        LocalVertex("rim", {arm: ("rim", arm) for arm in range(6)}),
    ]
    strands = [(("hub", arm), ("rim", arm)) for arm in range(5)]
    with pytest.raises(ValueError, match="carry no strand"):
        plan_single_circuit(vertices, strands)
