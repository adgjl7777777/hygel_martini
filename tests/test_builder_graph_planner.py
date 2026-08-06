from __future__ import annotations

from hygel_martini.hydrogel_builder.core_utils.layout.local_matching import (
    LOCAL_COORDS,
    LocalVertex,
    evaluate_matching_plan,
    plan_balanced_cycle_matchings,
)
from hygel_martini.hydrogel_builder.core_utils.runtime.dynamic_crosslink import (
    plan_dynamic_crosslinks,
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


class _Atom:
    def __init__(self, atom_id, position, chain_index=None):
        self.atom_id = atom_id
        self.position = position
        self.chain_index = chain_index
        self.end_tag = 1
        self.chain_type = "backbone"
        self.linker_chain_index = None
        self.planned_endpoint_id = None
        self.planned_endpoint_edges = None
        self.stub_type = None


def test_runtime_materializes_planned_edges_instead_of_nearest_rewiring() -> None:
    ends = {}
    positions = {
        "a": (0.0, 0.0, 0.0),
        "b": (9.0, 0.0, 0.0),
        "c": (1.0, 0.0, 0.0),
        "d": (10.0, 0.0, 0.0),
    }
    for chain_index, name in enumerate(("a", "b", "c", "d")):
        atom = _Atom(chain_index, positions[name], chain_index)
        atom.planned_endpoint_id = name
        ends[chain_index] = [atom]

    left_stub = _Atom(10, (0.2, 0.0, 0.0))
    right_stub = _Atom(11, (9.8, 0.0, 0.0))
    for stub, stub_type in ((left_stub, "backbone_1"), (right_stub, "backbone_2")):
        stub.end_tag = 2
        stub.chain_type = "linker"
        stub.linker_chain_index = 7
        stub.stub_type = stub_type
        # Nearest-only rewiring would prefer (a,c) and (b,d).  The planner
        # instead selected cross-pairs, which runtime must preserve exactly.
        stub.planned_endpoint_edges = (("a", "b"), ("c", "d"))

    assignments, notes = plan_dynamic_crosslinks(
        {7: [left_stub, right_stub]},
        ends,
        box_vec=None,
        candidate_limit=64,
        targets_per_stub=2,
    )

    actual_edges = {
        frozenset(item.backbone_atom.planned_endpoint_id for item in chosen)
        for chosen in (
            assignments[7][:2],
            assignments[7][2:],
        )
    }
    assert actual_edges == {frozenset(("a", "b")), frozenset(("c", "d"))}
    assert any("explicit layout-planner endpoint edges" in note for note in notes)
    hash_note = next(note for note in notes if note.startswith("planned_edge_sha256="))
    fields = dict(field.split("=", 1) for field in hash_note.split())
    assert fields["planned_edge_sha256"] == fields["materialized_edge_sha256"]
    assert fields["exact_match"] == "true"
