"""Normalizing the add_molecule stage's species list.

A deep eutectic solvent needs two species inserted at once: a molecular
cation Packmol must place and a monatomic anion. The ion stage cannot help --
genion inserts by REPLACING solvent, and in a DES the ions *are* the solvent --
so both go through add_molecule, which therefore accepts a list. The single
mapping stays valid because every existing configuration uses it.
"""

from __future__ import annotations

import pytest

from hygel_martini.hydrogel_builder.config_params.read_json import (
    _normalize_add_molecule_specs,
)


@pytest.fixture
def gro(tmp_path):
    path = tmp_path / "ACCH.gro"
    path.write_text("acch\n1\n    1ACC    N01    1   0.000   0.000   0.000\n 1 1 1\n")
    return str(path)


@pytest.fixture
def other_gro(tmp_path):
    path = tmp_path / "CL.gro"
    path.write_text("cl\n1\n    1CL      CL    1   0.000   0.000   0.000\n 1 1 1\n")
    return str(path)


def test_a_single_mapping_still_works(gro) -> None:
    specs = _normalize_add_molecule_specs(
        {"add_molecule": {"molecule_gro": gro, "num_molecules": 10}}, {}
    )
    assert len(specs) == 1
    assert specs[0]["num_molecules"] == 10
    # Name defaults to the file stem, as it always did.
    assert specs[0]["molecule_name"] == "ACCH"


def test_several_species_are_returned_in_order(gro, other_gro) -> None:
    specs = _normalize_add_molecule_specs(
        {"add_molecule": [
            {"molecule_gro": gro, "num_molecules": 384, "molecule_name": "ACC"},
            {"molecule_gro": other_gro, "num_molecules": 384, "molecule_name": "CL"},
        ]},
        {},
    )
    assert [s["molecule_name"] for s in specs] == ["ACC", "CL"]
    assert [s["num_molecules"] for s in specs] == [384, 384]


def test_zero_count_species_are_dropped(gro, other_gro) -> None:
    specs = _normalize_add_molecule_specs(
        {"add_molecule": [
            {"molecule_gro": gro, "num_molecules": 0},
            {"molecule_gro": other_gro, "num_molecules": 5, "molecule_name": "CL"},
        ]},
        {},
    )
    assert [s["molecule_name"] for s in specs] == ["CL"]


def test_duplicate_names_are_refused(gro, other_gro) -> None:
    # Two species under one name would merge into a single [molecules] entry,
    # so the topology would claim one species where two were requested.
    with pytest.raises(ValueError, match="share the"):
        _normalize_add_molecule_specs(
            {"add_molecule": [
                {"molecule_gro": gro, "num_molecules": 3, "molecule_name": "SAME"},
                {"molecule_gro": other_gro, "num_molecules": 3, "molecule_name": "SAME"},
            ]},
            {},
        )


def test_a_missing_file_is_an_error_outside_test_mode(tmp_path) -> None:
    with pytest.raises(ValueError, match="not found"):
        _normalize_add_molecule_specs(
            {"add_molecule": {"molecule_gro": str(tmp_path / "nope.gro"),
                              "num_molecules": 1}},
            {},
        )


def test_a_missing_file_is_skipped_in_test_mode(tmp_path) -> None:
    specs = _normalize_add_molecule_specs(
        {"add_molecule": {"molecule_gro": str(tmp_path / "nope.gro"),
                          "num_molecules": 1}},
        {"test_mode": True},
    )
    assert specs == []


def test_an_unconfigured_stage_yields_nothing() -> None:
    assert _normalize_add_molecule_specs({}, {}) == []
    assert _normalize_add_molecule_specs({"add_molecule": None}, {}) == []
