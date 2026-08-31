from __future__ import annotations

import random

import pytest

from hygel_martini.property_extract.cyclic_topology import (
    bipartite_check,
    cyclic_topology_report,
    reduce_to_junctions,
    vertex_symbols,
)


def _pcu(repeats: int) -> tuple[int, list[tuple[int, int]]]:
    """Periodic primitive-cubic net: six-connected, fundamental cycle size 4."""
    index = {}
    sites = []
    for i in range(repeats):
        for j in range(repeats):
            for k in range(repeats):
                index[(i, j, k)] = len(sites)
                sites.append((i, j, k))
    edges = []
    for site in sites:
        for step in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            neighbour = tuple(
                (site[axis] + step[axis]) % repeats for axis in range(3)
            )
            edges.append((index[site], index[neighbour]))
    return len(sites), edges


def _dia(repeats: int) -> tuple[int, list[tuple[int, int]]]:
    """Periodic diamond net: four-connected, fundamental cycle size 6."""
    index = {}
    count = 0
    for i in range(repeats):
        for j in range(repeats):
            for k in range(repeats):
                for sub in (0, 1):
                    index[(i, j, k, sub)] = count
                    count += 1
    edges = []
    for i in range(repeats):
        for j in range(repeats):
            for k in range(repeats):
                origin = index[(i, j, k, 0)]
                for step in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    edges.append(
                        (
                            origin,
                            index[
                                (
                                    (i + step[0]) % repeats,
                                    (j + step[1]) % repeats,
                                    (k + step[2]) % repeats,
                                    1,
                                )
                            ],
                        )
                    )
    return count, edges


def test_diamond_net_reproduces_its_published_vertex_symbol() -> None:
    n_nodes, edges = _dia(3)
    symbols = vertex_symbols(n_nodes, edges)

    # RCSR gives dia the vertex symbol 6.6.6.6.6.6: four-connected, and each
    # of the six incident-strand pairs closed by a six-ring.
    assert {tuple(sizes) for sizes in symbols.values()} == {(6,) * 6}

    report = cyclic_topology_report(n_nodes, edges)
    assert report["girth"] == 6
    assert report["peak_loop_order"] == 6
    assert report["mean_junction_degree"] == pytest.approx(4.0)
    assert report["distinct_vertex_symbol_count"] == 1


def test_primitive_cubic_net_is_six_connected_with_cycle_size_four() -> None:
    n_nodes, edges = _pcu(4)
    symbols = vertex_symbols(n_nodes, edges)

    # six-connected -> 6*5/2 = 15 incident-strand pairs per junction
    assert {len(sizes) for sizes in symbols.values()} == {15}
    assert {tuple(sizes) for sizes in symbols.values()} == {(4,) * 15}

    report = cyclic_topology_report(n_nodes, edges)
    assert report["girth"] == 4
    assert report["mean_junction_degree"] == pytest.approx(6.0)


@pytest.mark.parametrize("net", ["dia", "pcu"])
def test_ideal_nets_are_bipartite_and_carry_no_odd_loop_orders(net: str) -> None:
    n_nodes, edges = _dia(4) if net == "dia" else _pcu(4)
    report = cyclic_topology_report(n_nodes, edges)

    assert report["bipartite"] is True
    assert report["odd_cycle_witness"] is None
    assert report["odd_loop_order_count"] == 0
    assert all(order % 2 == 0 for order in report["loop_order_histogram"])


def test_odd_repeat_count_breaks_bipartiteness_through_the_boundary() -> None:
    # A walk of odd length along one axis returns to its origin, so an odd
    # supercell manufactures odd cycles that are box artifacts, not chemistry.
    even_nodes, even_edges = _pcu(4)
    odd_nodes, odd_edges = _pcu(3)

    assert bipartite_check(even_nodes, even_edges)[0] is True

    is_bipartite, witness = bipartite_check(odd_nodes, odd_edges)
    assert is_bipartite is False
    assert witness is not None
    assert cyclic_topology_report(odd_nodes, odd_edges)["girth"] == 3


def test_girth_is_reported_even_when_every_node_is_two_connected() -> None:
    # Equation (1) weights junctions by (f - 2), so a bare cycle contributes
    # nothing to the histogram.  Girth must not be read off that histogram.
    for n_nodes, edges, expected in (
        (4, [(0, 1), (1, 2), (2, 3), (3, 0)], 4),
        (3, [(0, 1), (1, 2), (2, 0)], 3),
    ):
        report = cyclic_topology_report(n_nodes, edges)
        assert report["girth"] == expected
        assert report["loop_order_histogram"] == {}
        assert report["loop_order_histogram_is_weighted_valid"] is False
        assert report["low_degree_junction_count"] == n_nodes


def test_primary_and_secondary_loops_are_counted_from_the_multigraph() -> None:
    edges = [(0, 0), (0, 1), (0, 1), (1, 1)]
    report = cyclic_topology_report(2, edges)

    assert report["primary_loop_count"] == 2
    assert report["secondary_loop_count"] == 1
    assert report["maximum_edge_multiplicity"] == 2
    assert report["bipartite"] is False


def test_reduction_strips_dangling_trees_and_contracts_continuations() -> None:
    #   0=1 is a two-cycle; 1-2-3 is a dangling tail; 4 merely continues 0-4-1
    n_nodes, edges, stats = reduce_to_junctions(
        5, [(0, 1), (0, 1), (1, 2), (2, 3), (0, 4), (4, 1)]
    )

    assert stats["dangling_junctions_removed"] == 2  # nodes 3 then 2
    assert stats["continuation_junctions_contracted"] == 1  # node 4
    assert n_nodes == 2
    assert sorted(tuple(sorted(edge)) for edge in edges) == [(0, 1)] * 3


def test_reduction_makes_the_weighted_histogram_valid_after_partial_conversion() -> None:
    # A partially converted lattice is full of one- and two-connected nodes,
    # which Eq. (1) cannot weight.  Reduction is a precondition, not a tidy-up.
    n_nodes, edges = _pcu(4)
    rng = random.Random(0)
    kept = [edge for edge in edges if rng.random() < 0.45]

    raw = cyclic_topology_report(n_nodes, kept)
    assert raw["low_degree_junction_count"] > 0
    assert raw["loop_order_histogram_is_weighted_valid"] is False

    reduced_nodes, reduced_edges, _ = reduce_to_junctions(n_nodes, kept)
    reduced = cyclic_topology_report(reduced_nodes, reduced_edges)
    assert reduced["low_degree_junction_count"] == 0
    assert reduced["loop_order_histogram_is_weighted_valid"] is True


def test_contracting_continuations_can_break_lattice_parity() -> None:
    # The even-loop-order restriction of a bipartite seed is a property of the
    # net graph.  Once partial conversion leaves two-connected nodes and those
    # are contracted, path parity changes and odd loop orders appear, so the
    # restriction does not survive into the reduced graph.
    n_nodes, edges = _pcu(4)
    rng = random.Random(0)
    kept = [edge for edge in edges if rng.random() < 0.45]

    assert bipartite_check(n_nodes, kept)[0] is True

    reduced_nodes, reduced_edges, _ = reduce_to_junctions(n_nodes, kept)
    reduced = cyclic_topology_report(reduced_nodes, reduced_edges)
    assert reduced["bipartite"] is False
    assert reduced["odd_loop_order_count"] > 0
