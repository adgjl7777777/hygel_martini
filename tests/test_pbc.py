"""One minimum-image convention, correct for triclinic cells as well.

The package previously carried seven copies of this primitive, six of which
applied the orthorhombic rounding formula unconditionally. On a triclinic cell
that does not find the nearest image at all, and the resulting distance decides
which chain ends a crosslinker can reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from hygel_martini.core.pbc import (
    is_orthorhombic,
    minimum_image,
    minimum_image_distance,
    nearest_image_reach,
    normalize_cell,
    wrap_into_cell,
)

CUBIC = [4.0, 4.0, 4.0]
# GROMACS-legal triclinic cell: off-diagonals at exactly half the diagonal.
TRICLINIC = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [2.0, 2.0, 3.0]]
# FCC primitive vectors, the diamond seed's cell: not lower-triangular, and its
# diagonal contains zeros.
FCC = [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]


def test_a_three_vector_is_shorthand_for_a_diagonal_cell() -> None:
    assert normalize_cell([2.0, 3.0, 4.0]) == pytest.approx(np.diag([2.0, 3.0, 4.0]))
    assert normalize_cell(None) is None


def test_degenerate_and_malformed_cells_are_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        normalize_cell([1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="zero volume"):
        normalize_cell([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="3-vector or a 3x3"):
        normalize_cell([1.0, 2.0])


def test_orthorhombic_detection() -> None:
    assert is_orthorhombic(normalize_cell(CUBIC))
    assert not is_orthorhombic(normalize_cell(TRICLINIC))
    assert not is_orthorhombic(normalize_cell(FCC))


def test_orthorhombic_case_matches_the_rounding_formula() -> None:
    cell = normalize_cell(CUBIC)
    lengths = np.diag(cell)
    rng = np.random.default_rng(0)
    for _ in range(200):
        delta = rng.uniform(-9.0, 9.0, 3)
        expected = delta - lengths * np.round(delta / lengths)
        assert minimum_image(delta, cell) == pytest.approx(expected)


@pytest.mark.parametrize("box", [CUBIC, TRICLINIC, FCC])
def test_the_result_is_a_genuine_lattice_translate_and_is_shortest(box) -> None:
    # Brute-force check: no lattice translate within a wide range is shorter,
    # and the returned vector really differs from the input by a lattice vector.
    cell = normalize_cell(box)
    span = range(-3, 4)
    lattice = np.array([(i, j, k) for i in span for j in span for k in span]) @ cell
    rng = np.random.default_rng(1)

    for _ in range(120):
        delta = rng.uniform(-2.5, 2.5, 3) @ cell
        got = minimum_image(delta, cell)

        images = delta + lattice
        shortest = float(np.min(np.linalg.norm(images, axis=1)))
        assert float(np.linalg.norm(got)) == pytest.approx(shortest, abs=1e-9)

        offset = np.linalg.solve(cell.T, got - delta)
        assert offset == pytest.approx(np.round(offset), abs=1e-9)


def test_the_search_range_is_verified_not_assumed() -> None:
    # An earlier version tested the GROMACS lower-triangular condition, which
    # rejected the FCC primitive basis outright because its diagonal is zero.
    for box in (TRICLINIC, FCC):
        assert nearest_image_reach(normalize_cell(box)) >= 1

    with pytest.raises(ValueError, match="too skewed"):
        nearest_image_reach(normalize_cell([[1.0, 0.0, 0.0], [40.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))


def test_the_orthorhombic_formula_is_wrong_on_a_triclinic_cell() -> None:
    # This is the defect the unification removes, kept as a test so it cannot
    # quietly return.
    cell = normalize_cell(TRICLINIC)
    lengths = np.diag(cell)
    delta = np.array([-1.959, 1.954, 1.964])

    naive = delta - lengths * np.round(delta / lengths)
    correct = minimum_image(delta, cell)

    assert float(np.linalg.norm(correct)) == pytest.approx(1.038, abs=5e-3)
    assert float(np.linalg.norm(naive)) > 2.9


def test_stacked_displacements_match_one_at_a_time() -> None:
    cell = normalize_cell(TRICLINIC)
    rng = np.random.default_rng(2)
    deltas = rng.uniform(-6.0, 6.0, (25, 3))

    stacked = minimum_image(deltas, cell)
    assert stacked.shape == deltas.shape
    for row, single in zip(deltas, stacked):
        assert minimum_image(row, cell) == pytest.approx(single)


def test_distance_helper_is_symmetric_and_uses_the_cell() -> None:
    cell = normalize_cell(TRICLINIC)
    first = np.array([0.1, 0.2, 0.3])
    second = np.array([3.9, 3.8, 2.7])

    assert minimum_image_distance(first, second, cell) == pytest.approx(
        minimum_image_distance(second, first, cell)
    )
    assert minimum_image_distance(first, second, None) == pytest.approx(
        float(np.linalg.norm(first - second))
    )


@pytest.mark.parametrize("box", [CUBIC, TRICLINIC, FCC])
def test_wrapping_keeps_positions_inside_the_cell(box) -> None:
    cell = normalize_cell(box)
    rng = np.random.default_rng(3)
    positions = rng.uniform(-3.0, 3.0, (40, 3)) @ cell

    wrapped = wrap_into_cell(positions, cell)
    fractional = np.linalg.solve(cell.T, wrapped.T).T
    assert np.all(fractional > -1e-9)
    assert np.all(fractional < 1.0 + 1e-9)

    # wrapping moves every position by a lattice vector, nothing else
    offsets = np.linalg.solve(cell.T, (wrapped - positions).T).T
    assert offsets == pytest.approx(np.round(offsets), abs=1e-9)


def test_property_extract_helper_still_enforces_its_orthorhombic_contract() -> None:
    from hygel_martini.property_extract.geometry import minimum_image_displacement

    delta = np.array([3.5, 0.0, 0.0])
    assert minimum_image_displacement(delta, np.array(CUBIC)) == pytest.approx(
        [-0.5, 0.0, 0.0]
    )
