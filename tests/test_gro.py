"""One GRO reader, column-based, tolerant of the off-spec files we ship."""

from __future__ import annotations

import numpy as np
import pytest

from hygel_martini.core.gro import read_gro, read_gro_atom_names, read_gro_atoms

STANDARD = (
    "title\n"
    "    2\n"
    "    1BCK     C1    1   1.000   2.000   3.000\n"
    "    1LNK     C2    2  -1.500   0.250  -0.750\n"
    "   5.00000   6.00000   7.00000\n"
)

# Valid GROMACS output that whitespace splitting cannot read: -100.000 fills
# its eight columns exactly, so the fields abut.
ABUTTING = (
    "big box\n"
    "    2\n"
    "    1BCK     C1    1-100.000-100.000-100.000\n"
    "    1LNK     C2    2  12.345  -6.789   1.234\n"
    " 500.00000 500.00000 500.00000\n"
)

# The tracked example structures shift the atom index one column right.
OFF_BY_ONE = (
    "Martini PE tetramer\n"
    "    2\n"
    "    1BCK     C1     1   1.000   1.000   1.000\n"
    "    1LNK     C2     2   1.470   1.000   1.000\n"
    "   5.00000   5.00000   5.00000\n"
)

TRICLINIC_BOX = (
    "tri\n"
    "    1\n"
    "    1BCK     C1    1   1.000   1.000   1.000\n"
    "   4.00000   4.00000   3.00000   0.00000   0.00000   2.00000"
    "   0.00000   2.00000   0.00000\n"
)


def _write(tmp_path, text, name="test.gro"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_standard_record_fields(tmp_path) -> None:
    atoms = read_gro_atoms(_write(tmp_path, STANDARD))

    assert [a.index for a in atoms] == [1, 2]
    assert [a.residue_name for a in atoms] == ["BCK", "LNK"]
    assert [a.atom_name for a in atoms] == ["C1", "C2"]
    assert atoms[1].position == pytest.approx([-1.5, 0.25, -0.75])


def test_abutting_coordinate_fields_are_read(tmp_path) -> None:
    # The previous reader raised here; the fields are valid, just not
    # whitespace-separated.
    atoms = read_gro_atoms(_write(tmp_path, ABUTTING))

    assert atoms[0].position == pytest.approx([-100.0, -100.0, -100.0])
    assert atoms[0].index == 1
    assert atoms[1].position == pytest.approx([12.345, -6.789, 1.234])


def test_records_shifted_off_the_standard_columns_are_still_read(tmp_path) -> None:
    atoms = read_gro_atoms(_write(tmp_path, OFF_BY_ONE))

    assert [a.index for a in atoms] == [1, 2]
    assert atoms[1].position == pytest.approx([1.47, 1.0, 1.0])


def test_orthorhombic_box_becomes_a_diagonal_cell(tmp_path) -> None:
    frame = read_gro(_write(tmp_path, STANDARD))
    assert frame.box == pytest.approx(np.diag([5.0, 6.0, 7.0]))
    assert frame.title == "title"
    assert len(frame) == 2


def test_nine_value_box_is_read_as_a_triclinic_cell(tmp_path) -> None:
    frame = read_gro(_write(tmp_path, TRICLINIC_BOX))
    # GROMACS writes v1x v2y v3z v1y v1z v2x v2z v3x v3y, so with the values
    # 4 4 3 0 0 2 0 2 0 the cell vectors are v1=(4,0,0), v2=(2,4,0), v3=(2,0,3).
    assert frame.box == pytest.approx(
        np.array([[4.0, 0.0, 0.0], [2.0, 4.0, 0.0], [2.0, 0.0, 3.0]])
    )
    assert float(np.linalg.det(frame.box)) == pytest.approx(48.0)


def test_a_declared_count_larger_than_the_file_is_refused(tmp_path) -> None:
    text = "t\n    5\n    1BCK     C1    1   1.000   1.000   1.000\n   1.0 1.0 1.0\n"
    with pytest.raises(ValueError, match="declares 5 atoms"):
        read_gro(_write(tmp_path, text))


def test_a_malformed_record_names_its_line(tmp_path) -> None:
    text = (
        "t\n    1\n"
        "    1BCK     C1    x   nope    nope    nope\n"
        "   1.0 1.0 1.0\n"
    )
    with pytest.raises(ValueError, match=r":3:"):
        read_gro(_write(tmp_path, text))


def test_a_box_with_the_wrong_number_of_values_is_refused(tmp_path) -> None:
    text = (
        "t\n    1\n"
        "    1BCK     C1    1   1.000   1.000   1.000\n"
        "   1.0 2.0\n"
    )
    with pytest.raises(ValueError, match="must hold 3 or 9 values"):
        read_gro(_write(tmp_path, text))


def test_a_degenerate_box_is_refused(tmp_path) -> None:
    text = (
        "t\n    1\n"
        "    1BCK     C1    1   1.000   1.000   1.000\n"
        "   1.0 0.0 1.0\n"
    )
    with pytest.raises(ValueError, match="non-positive volume"):
        read_gro(_write(tmp_path, text))


def test_atom_name_helper_matches_the_full_read(tmp_path) -> None:
    path = _write(tmp_path, STANDARD)
    assert read_gro_atom_names(path) == [a.atom_name for a in read_gro_atoms(path)]


def test_the_builder_module_re_exports_the_shared_reader() -> None:
    from hygel_martini.core.gro import read_gro_atoms as core_reader
    from hygel_martini.hydrogel_builder.core_utils.io.gro_parser import (
        read_gro_atoms as builder_reader,
    )

    assert builder_reader is core_reader
