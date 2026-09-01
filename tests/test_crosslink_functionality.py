"""Materializing a planned junction of any even functionality.

Two regimes, which the diamond builder never had to distinguish because it only
ever had one:

*one planned edge per stub* -- a stub is itself a two-way junction, as on the
diamond linker, so it realizes a whole planned edge and geometry only chooses
which stub takes which edge;

*one planned endpoint per stub* -- a stub is a single attachment, as on a
six-arm crosslinker, so the planned pairing describes a traversal through the
junction rather than a stub grouping.
"""

from __future__ import annotations

import pytest

from hygel_martini.hydrogel_builder.core_utils.runtime.dynamic_crosslink import (
    plan_dynamic_crosslinks,
)


class _Stub:
    def __init__(self, atom_id, position, planned_edges):
        self.atom_id = atom_id
        self.position = position
        self.planned_endpoint_edges = planned_edges
        self.stub_type = "backbone_1"
        self.target_backbone = None
        self.backbone_type = None


class _End:
    def __init__(self, atom_id, position, endpoint_id, chain_index):
        self.atom_id = atom_id
        self.position = position
        self.planned_endpoint_id = endpoint_id
        self.chain_index = chain_index
        self.end_tag = 1
        self.chain_type = "backbone"
        self.linker_chain_index = None
        self.backbone_type = None


def _ends(specs):
    """specs: {endpoint_id: (position, chain_index)}"""
    by_chain = {}
    for atom_id, (endpoint_id, (position, chain)) in enumerate(specs.items()):
        by_chain.setdefault(chain, []).append(
            _End(atom_id, position, endpoint_id, chain)
        )
    return by_chain


def test_a_two_stub_linker_realizes_one_planned_edge_each() -> None:
    # The diamond convention: two stubs, two planned edges, two ends per stub.
    edges = (("a", "b"), ("c", "d"))
    stubs = [_Stub(100, (0.0, 0.0, 0.0), edges), _Stub(101, (10.0, 0.0, 0.0), edges)]
    ends = _ends(
        {
            "a": ((0.0, 1.0, 0.0), 0),
            "b": ((0.0, -1.0, 0.0), 1),
            "c": ((10.0, 1.0, 0.0), 2),
            "d": ((10.0, -1.0, 0.0), 3),
        }
    )

    assignments, notes = plan_dynamic_crosslinks({0: stubs}, ends, None)

    chosen = assignments[0]
    assert len(chosen) == 4
    by_stub = {}
    for item in chosen:
        by_stub.setdefault(item.stub_atom.atom_id, set()).add(
            item.backbone_atom.planned_endpoint_id
        )
    # each stub took a whole planned edge, and the nearer one at that
    assert by_stub == {100: {"a", "b"}, 101: {"c", "d"}}
    assert any("edge_regime_linkers=1" in note for note in notes)


def test_a_six_arm_junction_takes_one_endpoint_per_stub() -> None:
    # Six stubs around an octahedral junction, three planned edges.
    edges = (("a", "b"), ("c", "d"), ("e", "f"))
    positions = [
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    ]
    stubs = [_Stub(200 + i, p, edges) for i, p in enumerate(positions)]
    ends = _ends(
        {
            "a": ((2.0, 0.0, 0.0), 0),
            "b": ((-2.0, 0.0, 0.0), 1),
            "c": ((0.0, 2.0, 0.0), 2),
            "d": ((0.0, -2.0, 0.0), 3),
            "e": ((0.0, 0.0, 2.0), 4),
            "f": ((0.0, 0.0, -2.0), 5),
        }
    )

    assignments, notes = plan_dynamic_crosslinks({0: stubs}, ends, None)

    chosen = assignments[0]
    assert len(chosen) == 6
    # every stub took exactly one endpoint, and every planned endpoint was used
    per_stub = {}
    for item in chosen:
        per_stub.setdefault(item.stub_atom.atom_id, []).append(item)
    assert sorted(len(v) for v in per_stub.values()) == [1] * 6
    assert {item.backbone_atom.planned_endpoint_id for item in chosen} == set("abcdef")
    assert any("endpoint_regime_linkers=1" in note for note in notes)


def test_each_arm_takes_the_end_it_actually_reaches() -> None:
    # Geometry chooses the stub-to-endpoint assignment, so the arm along +x
    # must take the end sitting along +x rather than an arbitrary one.
    edges = (("a", "b"), ("c", "d"), ("e", "f"))
    positions = [
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    ]
    stubs = [_Stub(200 + i, p, edges) for i, p in enumerate(positions)]
    ends = _ends(
        {
            "a": ((3.0, 0.0, 0.0), 0),
            "b": ((-3.0, 0.0, 0.0), 1),
            "c": ((0.0, 3.0, 0.0), 2),
            "d": ((0.0, -3.0, 0.0), 3),
            "e": ((0.0, 0.0, 3.0), 4),
            "f": ((0.0, 0.0, -3.0), 5),
        }
    )

    assignments, _ = plan_dynamic_crosslinks({0: stubs}, ends, None)

    taken = {
        item.stub_atom.atom_id: item.backbone_atom.planned_endpoint_id
        for item in assignments[0]
    }
    assert taken == {200: "a", 201: "b", 202: "c", 203: "d", 204: "e", 205: "f"}


def test_a_stub_count_that_matches_neither_regime_is_refused() -> None:
    # Three stubs against two planned edges: not one edge per stub (3 != 2) and
    # not one endpoint per stub (3 != 4).
    edges = (("a", "b"), ("c", "d"))
    stubs = [_Stub(300 + i, (float(i), 0.0, 0.0), edges) for i in range(3)]
    ends = _ends(
        {
            "a": ((0.0, 1.0, 0.0), 0),
            "b": ((1.0, 1.0, 0.0), 1),
            "c": ((2.0, 1.0, 0.0), 2),
            "d": ((3.0, 1.0, 0.0), 3),
        }
    )

    with pytest.raises(ValueError, match="cannot take 4 planned endpoints"):
        plan_dynamic_crosslinks({0: stubs}, ends, None)


def test_inconsistent_planned_metadata_across_stubs_is_refused() -> None:
    stubs = [
        _Stub(400, (0.0, 0.0, 0.0), (("a", "b"), ("c", "d"))),
        _Stub(401, (1.0, 0.0, 0.0), (("a", "c"), ("b", "d"))),
    ]
    ends = _ends(
        {
            "a": ((0.0, 1.0, 0.0), 0),
            "b": ((0.0, -1.0, 0.0), 1),
            "c": ((1.0, 1.0, 0.0), 2),
            "d": ((1.0, -1.0, 0.0), 3),
        }
    )

    with pytest.raises(ValueError, match="inconsistent planned endpoint edges"):
        plan_dynamic_crosslinks({0: stubs}, ends, None)


def test_the_pairwise_fallback_refuses_a_multi_arm_junction() -> None:
    # No planner metadata and one target per stub: this branch pairs two stubs
    # directly and has no meaning for six arms. Skipping would leave them
    # unbonded without saying so.
    class _Plain(_Stub):
        def __init__(self, atom_id, position):
            super().__init__(atom_id, position, None)
            self.planned_endpoint_edges = None

    stubs = [_Plain(500 + i, (float(i), 0.0, 0.0)) for i in range(6)]
    ends = _ends({f"e{i}": ((float(i), 1.0, 0.0), i) for i in range(6)})

    with pytest.raises(ValueError, match="needs exactly two stubs"):
        plan_dynamic_crosslinks({0: stubs}, ends, None, targets_per_stub=1)
