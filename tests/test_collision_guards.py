"""Declarations that would silently overwrite one another are refused.

The builder indexes user-supplied identifiers -- component ids, residue names,
molecule names, atom types, bond rules -- into plain dictionaries. Declaring a
key twice discarded one declaration without a word, and the build then
succeeded with a wrong mass, a wrong bond parameter, or a component that never
appeared. Each site below now names the collision instead.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from hygel_martini.hydrogel_builder.config_params.config import Config
from hygel_martini.hydrogel_builder.core_utils.common.collisions import (
    DuplicateDeclaration,
    find_duplicates,
    require_consistent,
    require_unique,
)
from hygel_martini.hydrogel_builder.core_utils.io.martini_parser import (
    read_atom_types,
    read_itp_definitions,
)
from hygel_martini.hydrogel_builder.core_utils.layout.proto_builder import (
    _build_bond_lookup,
)
from hygel_martini.hydrogel_builder.core_utils.templates.linker_loader import (
    load_linker_templates,
)
from hygel_martini.hydrogel_builder.core_utils.templates.monomer_loader import (
    load_monomer_templates,
)
from hygel_martini.hydrogel_builder.main_components.Polymer import (
    _build_polymer_bond_lookup,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MAKER = REPO_ROOT / "example" / "04_full_builder" / "project" / "maker.yaml"


# --------------------------------------------------------------------------
# the primitives
# --------------------------------------------------------------------------

def test_require_unique_builds_a_lookup_and_names_a_collision() -> None:
    assert require_unique([("a", 1), ("b", 2)], "widget") == {"a": 1, "b": 2}

    with pytest.raises(DuplicateDeclaration) as excinfo:
        require_unique([("a", 1), ("a", 2)], "widget", "id", source="WIDGETS")
    message = str(excinfo.value)
    assert "Duplicate widget id in WIDGETS" in message
    assert "'a'" in message and "1" in message and "2" in message


def test_require_consistent_tolerates_agreeing_repeats() -> None:
    assert require_consistent([("a", 1), ("a", 1)], "widget") == {"a": 1}

    with pytest.raises(DuplicateDeclaration, match="Conflicting"):
        require_consistent([("a", 1), ("a", 2)], "widget")


def test_find_duplicates_reports_counts() -> None:
    assert find_duplicates(["a", "b", "a", "a"]) == {"a": 3}
    assert find_duplicates(["a", "b"]) == {}


# --------------------------------------------------------------------------
# bond rules
# --------------------------------------------------------------------------

def test_a_backbone_pair_declared_twice_is_refused() -> None:
    rules = [
        {"between": ["BB1", "BB2"], "bond_c0": 0.47, "bond_c1": 1250},
        {"between": ["BB2", "BB1"], "bond_c0": 0.33, "bond_c1": 5000},
    ]
    # The pair is the same after sorting, so one parameter set was being lost.
    for builder in (
        lambda r: _build_bond_lookup(r, 0.47),
        lambda r: _build_polymer_bond_lookup(r, 0.47),
    ):
        with pytest.raises(DuplicateDeclaration, match="between"):
            builder(rules)


def test_distinct_backbone_pairs_are_fine() -> None:
    rules = [
        {"between": ["BB1", "BB1"], "bond_c0": 0.47, "bond_c1": 1250},
        {"between": ["BB1", "BB2"], "bond_c0": 0.47, "bond_c1": 1250},
        {"between": ["BB2", "BB2"], "bond_c0": 0.47, "bond_c1": 1250},
    ]
    assert len(_build_bond_lookup(rules, 0.47)) == 3
    assert len(_build_polymer_bond_lookup(rules, 0.47)) == 3


# --------------------------------------------------------------------------
# ITP parsing
# --------------------------------------------------------------------------

def test_an_atom_type_declared_with_two_masses_is_refused(tmp_path) -> None:
    path = tmp_path / "ff.itp"
    path.write_text("[ atomtypes ]\nP4 72.0 0.0 A 0.0 0.0\nP4 45.0 0.0 A 0.0 0.0\n")

    with pytest.raises(DuplicateDeclaration, match="mass 72.0 and again with 45.0"):
        read_atom_types(str(path))


def test_an_atom_type_repeated_identically_is_accepted(tmp_path) -> None:
    path = tmp_path / "ff.itp"
    path.write_text("[ atomtypes ]\nP4 72.0 0.0 A 0.0 0.0\nP4 72.0 0.0 A 0.0 0.0\n")

    assert read_atom_types(str(path)) == {"P4": {"mass": 72.0}}


def test_a_molecule_type_defined_twice_in_one_file_is_refused(tmp_path) -> None:
    path = tmp_path / "mol.itp"
    path.write_text(
        "[ moleculetype ]\nFOO 1\n[ atoms ]\n1 P4 1 FOO A 1 0.0 72.0\n"
        "[ moleculetype ]\nFOO 1\n[ atoms ]\n1 P4 1 FOO A 1 0.0 72.0\n"
    )

    with pytest.raises(DuplicateDeclaration, match="defined twice"):
        read_itp_definitions(str(path), prefer_explicit_masses=True)


# --------------------------------------------------------------------------
# component libraries, against the tracked example
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def example_components():
    if not EXAMPLE_MAKER.is_file():
        pytest.skip("tracked example 04 not present")
    Config.load_config(str(EXAMPLE_MAKER))
    base_itp = Config.get_param("simulation_parameters")["base_itp_file"]
    if not Path(base_itp).is_file():
        pytest.skip("Martini force field not present")
    Config.set_runtime("atom_type_masses", read_atom_types(base_itp))
    components = Config.get_param("hydrogel_components")
    return {
        "backbones": components["backbone_definitions"]["BACKBONES"],
        "linkers": components["linker_definitions"]["LINKERS"],
        "monomers": Config.get_param("monomer_definitions")["MONOMERS"],
    }


def test_the_tracked_example_loads_with_every_guard_active(example_components) -> None:
    backbones = example_components["backbones"]
    load_linker_templates(example_components["linkers"], backbones)
    load_monomer_templates(example_components["monomers"], backbones)


def test_a_duplicate_monomer_id_is_refused(example_components) -> None:
    monomers = example_components["monomers"]
    with pytest.raises(DuplicateDeclaration, match="Duplicate monomer id"):
        load_monomer_templates(
            monomers + [dict(monomers[0])], example_components["backbones"]
        )


def test_a_duplicate_linker_id_is_refused(example_components) -> None:
    linkers = example_components["linkers"]
    with pytest.raises(DuplicateDeclaration, match="Duplicate linker id"):
        load_linker_templates(
            linkers + [dict(linkers[0])], example_components["backbones"]
        )


def test_two_backbones_claiming_one_residue_name_are_refused(example_components) -> None:
    # Monomer-to-backbone matching goes by residue name, so a shared name means
    # one backbone is never selected.
    backbones = copy.deepcopy(example_components["backbones"])
    backbones[1]["definition"]["residue_name"] = backbones[0]["definition"][
        "residue_name"
    ]

    with pytest.raises(DuplicateDeclaration, match="residue_name"):
        load_monomer_templates(example_components["monomers"], backbones)
