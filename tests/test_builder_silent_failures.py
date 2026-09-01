"""Regression tests for builder paths that used to fail quietly.

Each case here previously produced a plausible-looking build whose topology or
composition was wrong, with nothing on stderr to say so.
"""

from __future__ import annotations

import pytest

from hygel_martini.hydrogel_builder.core_utils.io.martini_parser import read_atom_types

MARTINI_ATOMTYPES = """\
[ atomtypes ]
P6 72.0 0.000 A 0.0 0.0
C1 72.0 0.000 A 0.0 0.0
"""

# OPLS-AA ffnonbonded.itp: name, bonded type, atomic number, mass, ...
OPLS_ATOMTYPES = """\
[ atomtypes ]
; name  bond_type  at.num   mass    charge  ptype  sigma   epsilon
 opls_001   C   6   12.01100   0.500   A    3.75000e-01  4.39320e-01
 opls_002   O   8   15.99940  -0.500   A    2.96000e-01  8.78640e-01
"""


def test_martini_atomtype_layout_is_read(tmp_path) -> None:
    path = tmp_path / "martini.itp"
    path.write_text(MARTINI_ATOMTYPES)

    assert read_atom_types(str(path)) == {
        "P6": {"mass": 72.0},
        "C1": {"mass": 72.0},
    }


def test_opls_atomtype_layout_is_read(tmp_path) -> None:
    # OPLS-style rows carry bonded-type and atomic-number columns before the
    # mass. The mass sits two columns before the particle-type letter in
    # every GROMACS layout, which is how the parser locates it: guessing
    # "first numeric column" would read the atomic number (6, 8) as a mass.
    path = tmp_path / "opls.itp"
    path.write_text(OPLS_ATOMTYPES)

    assert read_atom_types(str(path)) == {
        "opls_001": {"mass": 12.011},
        "opls_002": {"mass": 15.9994},
    }


def test_opls_layout_without_atomic_number_is_read(tmp_path) -> None:
    # LigParGen writes name/btype/mass/charge/ptype/sigma/epsilon.
    path = tmp_path / "lpg.itp"
    path.write_text(
        "[ atomtypes ]\n"
        "  opls_800  C800  12.0110  0.000  A  3.50000E-01  2.76144E-01\n"
    )

    assert read_atom_types(str(path)) == {"opls_800": {"mass": 12.011}}


def test_atomic_number_without_bonded_type_is_not_read_as_the_mass(tmp_path) -> None:
    # name/at.num/mass/charge/ptype -- the standard Amber/CHARMM port layout.
    # A first-numeric-column guess returns the atomic number (6) as the mass,
    # silently; found by adversarial probing of the previous heuristic.
    path = tmp_path / "atnum.itp"
    path.write_text(
        "[ atomtypes ]\n"
        "  CT  6  12.0110  0.000  A  3.39967e-01  4.57730e-01\n"
    )

    assert read_atom_types(str(path)) == {"CT": {"mass": 12.011}}


def test_a_bonded_type_that_looks_like_a_ptype_letter_is_not_fatal(tmp_path) -> None:
    # 'S' is a real OPLS bonded type. The previous heuristic saw two ptype
    # candidates and silently dropped the row; the ptype is the candidate
    # followed only by numbers, scanned from the right.
    path = tmp_path / "sulfur.itp"
    path.write_text(
        "[ atomtypes ]\n"
        "  opls_202  S  16  32.0600  0.000  A  3.55000e-01  1.04600e+00\n"
    )

    assert read_atom_types(str(path)) == {"opls_202": {"mass": 32.06}}


def test_atomtype_file_without_the_section_does_not_warn(tmp_path, capsys) -> None:
    path = tmp_path / "plain.itp"
    path.write_text("[ moleculetype ]\nFOO 1\n")

    assert read_atom_types(str(path)) == {}
    assert capsys.readouterr().err == ""


def _fresh_world():
    from hygel_martini.hydrogel_builder.main_components import Attributes
    from hygel_martini.hydrogel_builder.main_components.Universe import World

    World.Atoms.clear()
    World.Bonds.clear()
    Attributes.initialize()
    return Attributes, World


def test_duplicate_bond_with_identical_parameters_is_silent(capsys) -> None:
    Attributes, _ = _fresh_world()
    Attributes.Atom()
    Attributes.Atom()

    Attributes.Bond(0, 1, funct=1, c0=0.47, c1=1250.0)
    capsys.readouterr()
    Attributes.Bond(1, 0, funct=1, c0=0.47, c1=1250.0)

    assert capsys.readouterr().err == ""


def test_duplicate_bond_with_conflicting_parameters_warns(capsys) -> None:
    # Template bonds and patch rules can both reach the same atom pair.  The
    # first definition wins by design; silently discarding the second hid a
    # bonded-topology decision.
    Attributes, _ = _fresh_world()
    Attributes.Atom()
    Attributes.Atom()

    Attributes.Bond(0, 1, funct=1, c0=0.47, c1=1250.0)
    capsys.readouterr()
    Attributes.Bond(0, 1, funct=1, c0=0.33, c1=5000.0)

    warning = capsys.readouterr().err
    assert "redefined with different parameters" in warning
    assert "0.47" in warning and "0.33" in warning
    assert "1250" in warning and "5000" in warning


def test_conflicting_bond_keeps_the_first_definition(capsys) -> None:
    Attributes, World = _fresh_world()
    Attributes.Atom()
    Attributes.Atom()

    Attributes.Bond(0, 1, funct=1, c0=0.47, c1=1250.0)
    Attributes.Bond(0, 1, funct=1, c0=0.33, c1=5000.0)
    capsys.readouterr()

    bonds = World.Bonds[(0, 1)]
    assert len(bonds) == 1
    assert bonds[0].bond_c0 == pytest.approx(0.47)
    assert bonds[0].bond_c1 == pytest.approx(1250.0)
