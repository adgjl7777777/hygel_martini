"""Periodic net seeds must reproduce their own published invariants.

RCSR fixes what ``dia`` and ``pcu`` mean.  If a seed built here disagrees with
the published coordination number, fundamental cycle size or bipartiteness,
that is a defect in the definition, so these tests assert the published values
rather than whatever the code happens to produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from hygel_martini.hydrogel_builder.core_utils.layout.local_matching import (
    plan_single_circuit,
)
from hygel_martini.hydrogel_builder.core_utils.layout.nets import (
    DIA,
    NETS,
    PCU,
    build_periodic_net,
    get_net,
    validate_repeats,
)
from hygel_martini.property_extract.cyclic_topology import cyclic_topology_report


def _reduced_graph(vertices, strands):
    """Collapse planner input to the junction--strand graph for auditing."""
    node_of = {vertex.vertex_id: index for index, vertex in enumerate(vertices)}
    owner = {
        endpoint: node_of[vertex.vertex_id]
        for vertex in vertices
        for endpoint in vertex.endpoints.values()
    }
    return len(vertices), [(owner[left], owner[right]) for left, right in strands]


@pytest.mark.parametrize("name,expected", [("dia", 4), ("pcu", 6)])
def test_published_coordination_numbers(name, expected) -> None:
    assert get_net(name).coordination == expected


def test_net_lookup_rejects_unknown_symbols() -> None:
    with pytest.raises(ValueError, match="Unknown net"):
        get_net("srs")


@pytest.mark.parametrize("net", [DIA, PCU])
def test_every_bond_has_the_same_length(net) -> None:
    # A four-connected graph is not diamond unless the embedding makes all four
    # bonds equivalent; this catches a topologically right but geometrically
    # wrong basis.
    lengths = [float(np.linalg.norm(vector)) for vector in net.bond_vectors()]
    assert len(lengths) == len(net.bonds)
    assert lengths == pytest.approx([lengths[0]] * len(lengths))


def test_diamond_bond_length_matches_the_ideal_tetrahedral_value() -> None:
    # With FCC primitive vectors of unit cube parameter, the C-C bond is
    # sqrt(3)/4 of the cube edge.
    lengths = [float(np.linalg.norm(vector)) for vector in DIA.bond_vectors()]
    assert lengths[0] == pytest.approx(np.sqrt(3.0) / 4.0)


def test_primitive_cubic_bond_length_is_the_cell_parameter() -> None:
    lengths = [float(np.linalg.norm(vector)) for vector in PCU.bond_vectors()]
    assert lengths[0] == pytest.approx(1.0)


@pytest.mark.parametrize("name,repeats", [("dia", 3), ("dia", 4), ("pcu", 4), ("pcu", 6)])
def test_built_net_reproduces_its_published_girth_and_coordination(name, repeats) -> None:
    net = get_net(name)
    vertices, strands, positions = build_periodic_net(net, repeats)

    n_nodes, reduced = _reduced_graph(vertices, strands)
    report = cyclic_topology_report(n_nodes, reduced)

    assert report["girth"] == net.fundamental_cycle_size
    assert report["mean_junction_degree"] == pytest.approx(net.coordination)
    assert report["bipartite"] is net.bipartite
    assert report["odd_loop_order_count"] == 0
    # A topologically regular net has one local environment everywhere.
    assert report["distinct_vertex_symbol_count"] == 1
    assert len(positions) == n_nodes


@pytest.mark.parametrize("name,repeats", [("dia", 4), ("pcu", 4)])
def test_a_net_seed_admits_a_single_circuit(name, repeats) -> None:
    vertices, strands, _ = build_periodic_net(name, repeats)
    plan = plan_single_circuit(vertices, strands)

    assert plan.is_single_cycle
    assert plan.diagnostics.component_count == 1


def test_odd_repeats_are_rejected_only_where_they_break_the_two_colouring() -> None:
    # pcu is two-coloured by site parity, which an odd supercell destroys.
    with pytest.raises(ValueError, match="odd along"):
        validate_repeats(PCU, (3, 3, 3))

    # dia is two-coloured by sublattice, which no repeat count can break, so an
    # odd cell is fine there as long as it is large enough.
    validate_repeats(DIA, (3, 3, 3))


def test_cells_too_small_for_their_own_girth_are_rejected() -> None:
    # One step along a pcu axis is a single bond, so L=2 wraps in 2 bonds and
    # undercuts the net's fundamental cycle size of 4.
    with pytest.raises(ValueError, match="wrap-around cycle of length 2"):
        validate_repeats(PCU, (2, 2, 2))

    # dia needs two bonds per lattice step, so L=1 wraps in 2 against a
    # fundamental cycle size of 6.
    with pytest.raises(ValueError, match="wrap-around cycle"):
        validate_repeats(DIA, (1, 1, 1))


def test_an_undersized_cell_really_would_have_reported_a_false_girth() -> None:
    # Confirms the guard is not merely conservative: bypass it and the measured
    # girth genuinely comes from the box rather than the net.
    vertices, strands, _ = build_periodic_net(PCU, 3, check_repeats=False)
    n_nodes, reduced = _reduced_graph(vertices, strands)
    report = cyclic_topology_report(n_nodes, reduced)

    assert report["girth"] == 3 < PCU.fundamental_cycle_size
    assert report["bipartite"] is False
    assert report["odd_loop_order_count"] > 0


def test_definitions_are_internally_consistent() -> None:
    for name, net in NETS.items():
        net.validate()  # arm count per site matches the coordination number
        assert net.name == name
        assert net.coordination % 2 == 0
        for site in range(len(net.basis)):
            assert len(net.arms_of_site(site)) == net.coordination


def test_endpoints_are_unique_and_each_carries_one_strand() -> None:
    vertices, strands, _ = build_periodic_net(PCU, 4)

    declared = [
        endpoint for vertex in vertices for endpoint in vertex.endpoints.values()
    ]
    used = [endpoint for strand in strands for endpoint in strand]

    assert len(declared) == len(set(declared))
    assert sorted(map(repr, used)) == sorted(map(repr, declared))
