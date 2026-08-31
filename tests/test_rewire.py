"""Span-constrained rewiring: invariants, the span constraint, convergence."""

from __future__ import annotations

import numpy as np
import pytest

from hygel_martini.hydrogel_builder.core_utils.layout.nets import (
    build_periodic_net,
    get_net,
)
from hygel_martini.hydrogel_builder.core_utils.layout.rewire import (
    _endpoint_owner,
    _minimum_image,
    _normalize_box,
    reduced_edges,
    span_constrained_rewire,
    total_variation_distance,
)
from hygel_martini.property_extract.cyclic_topology import cyclic_topology_report


def _seed(name, repeats):
    net = get_net(name)
    vertices, strands, positions = build_periodic_net(net, repeats)
    cell = np.asarray(net.lattice_vectors, dtype=float) * repeats
    nearest = float(np.linalg.norm(net.bond_vectors()[0]))
    return net, vertices, strands, positions, cell, nearest


def _report(vertices, strands):
    owner = _endpoint_owner(vertices)
    node_index = {vertex.vertex_id: i for i, vertex in enumerate(vertices)}
    return cyclic_topology_report(
        len(vertices), reduced_edges(strands, owner, node_index)
    )


def test_minimum_image_is_exact_for_a_non_orthogonal_cell() -> None:
    # FCC primitive vectors sit at 60 degrees; fractional rounding alone is not
    # enough there, so the neighbouring images have to be searched.
    cell = _normalize_box([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    shortened = _minimum_image(np.array([0.9, 0.0, 0.0]), cell)
    assert float(np.linalg.norm(shortened)) == pytest.approx(0.1)

    orthorhombic = _normalize_box([1.0, 1.0, 1.0])
    assert _minimum_image(np.array([0.9, 0.0, 0.0]), orthorhombic) == pytest.approx(
        [-0.1, 0.0, 0.0]
    )


def test_box_shape_is_validated() -> None:
    with pytest.raises(ValueError, match="3-vector or a 3x3 matrix"):
        _normalize_box([1.0, 2.0])


def test_rewiring_preserves_every_junction_functionality() -> None:
    # The whole point of a double-edge swap: an f=6 network must stay f=6, or
    # it no longer composes with the transition-system planner.
    _, vertices, strands, positions, cell, nearest = _seed("pcu", 4)
    before = _report(vertices, strands)

    result = span_constrained_rewire(
        vertices, strands, positions, max_span=3 * nearest,
        box=cell, seed=0, max_sweeps=10,
    )
    after = _report(vertices, result.strands)

    assert after["junction_degree_distribution"] == before["junction_degree_distribution"]
    assert after["mean_junction_degree"] == pytest.approx(6.0)
    assert after["strand_count"] == before["strand_count"]


def test_every_endpoint_still_carries_exactly_one_strand() -> None:
    _, vertices, strands, positions, cell, nearest = _seed("pcu", 4)
    result = span_constrained_rewire(
        vertices, strands, positions, max_span=3 * nearest,
        box=cell, seed=1, max_sweeps=10,
    )

    used = [endpoint for strand in result.strands for endpoint in strand]
    declared = [ep for vertex in vertices for ep in vertex.endpoints.values()]
    assert len(used) == len(set(used))
    assert set(used) == set(declared)


def test_the_span_constraint_is_actually_enforced() -> None:
    net, vertices, strands, positions, cell, nearest = _seed("pcu", 4)
    cutoff = 2.0 * nearest
    owner = _endpoint_owner(vertices)

    result = span_constrained_rewire(
        vertices, strands, positions, max_span=cutoff,
        box=cell, seed=2, max_sweeps=20,
    )

    box = _normalize_box(cell)
    for left, right in result.strands:
        delta = np.asarray(positions[owner[left]]) - np.asarray(positions[owner[right]])
        span = float(np.linalg.norm(_minimum_image(delta, box)))
        assert span <= cutoff + 1e-9


def test_rewiring_is_deterministic_for_a_fixed_seed() -> None:
    _, vertices, strands, positions, cell, nearest = _seed("pcu", 4)
    kwargs = dict(max_span=2 * nearest, box=cell, max_sweeps=8)

    first = span_constrained_rewire(vertices, strands, positions, seed=7, **kwargs)
    second = span_constrained_rewire(vertices, strands, positions, seed=7, **kwargs)
    other = span_constrained_rewire(vertices, strands, positions, seed=8, **kwargs)

    assert first.strands == second.strands
    assert first.summary() == second.summary()
    assert other.strands != first.strands


def test_rewiring_breaks_the_bipartite_parity_of_the_seed() -> None:
    # The seed's even-only loop spectrum is an artifact of the net; rewiring
    # exists to remove it.
    _, vertices, strands, positions, cell, nearest = _seed("pcu", 4)
    assert _report(vertices, strands)["odd_loop_order_count"] == 0

    result = span_constrained_rewire(
        vertices, strands, positions, max_span=2 * nearest,
        box=cell, seed=0, max_sweeps=30,
    )
    after = _report(vertices, result.strands)

    assert after["bipartite"] is False
    assert after["odd_loop_order_count"] > 0
    assert after["distinct_vertex_symbol_count"] > 1


def test_primary_loops_can_be_forbidden_when_building_an_idealized_reference() -> None:
    _, vertices, strands, positions, cell, nearest = _seed("pcu", 4)

    permissive = span_constrained_rewire(
        vertices, strands, positions, max_span=2 * nearest,
        box=cell, seed=3, max_sweeps=40, allow_primary_loops=True,
    )
    strict = span_constrained_rewire(
        vertices, strands, positions, max_span=2 * nearest,
        box=cell, seed=3, max_sweeps=40, allow_primary_loops=False,
    )

    assert _report(vertices, strict.strands)["primary_loop_count"] == 0
    assert strict.rejected_primary_loop > 0
    assert _report(vertices, permissive.strands)["primary_loop_count"] > 0


def test_a_cutoff_below_the_bond_length_admits_nothing() -> None:
    _, vertices, strands, positions, cell, nearest = _seed("pcu", 4)
    result = span_constrained_rewire(
        vertices, strands, positions, max_span=0.5 * nearest,
        box=cell, seed=0, max_sweeps=5,
    )

    assert result.accepted == 0
    assert result.rejected_span > 0
    assert result.strands == [tuple(strand) for strand in strands]


def test_convergence_threshold_is_measured_rather_than_fixed() -> None:
    # The residual sweep-to-sweep variation is a finite-size effect, so a fixed
    # tolerance below it can never fire however stationary the process is.
    _, vertices, strands, positions, cell, nearest = _seed("dia", 4)
    kwargs = dict(max_span=1.7 * nearest, box=cell, seed=0, max_sweeps=40)

    measured = span_constrained_rewire(vertices, strands, positions, tolerance=None, **kwargs)
    unreachable = span_constrained_rewire(vertices, strands, positions, tolerance=1e-6, **kwargs)

    assert measured.converged
    assert measured.noise_floor is not None and measured.noise_floor > 1e-6
    assert not unreachable.converged
    assert unreachable.sweeps == 40


def test_the_noise_floor_falls_with_system_size() -> None:
    # It scales as one over the square root of the strand count, which is why a
    # single fixed tolerance cannot serve every cell size.
    floors = {}
    for repeats in (4, 6):
        _, vertices, strands, positions, cell, _ = _seed("pcu", repeats)
        result = span_constrained_rewire(
            vertices, strands, positions, max_span=1e6,
            box=cell, seed=0, max_sweeps=20, tolerance=None,
        )
        assert result.converged
        floors[len(strands)] = result.noise_floor

    small, large = sorted(floors)
    assert floors[large] < floors[small]


def test_total_variation_distance_is_a_metric_on_distributions() -> None:
    assert total_variation_distance({}, {}) == 0.0
    assert total_variation_distance({4: 1.0}, {4: 1.0}) == 0.0
    assert total_variation_distance({4: 1.0}, {6: 1.0}) == pytest.approx(1.0)
    assert total_variation_distance({4: 0.5, 6: 0.5}, {4: 1.0}) == pytest.approx(0.5)
