"""Crosslinker templates with more than two attachment points.

The diamond builder's junction is a two-stub linker, and the loader enforced
exactly two BCK stubs. A six-arm thiol such as dipentaerythritol
hexakis(3-mercaptopropionate) could not be loaded at all. These tests pin the
general N-stub form and, just as importantly, that the two-stub path is
unchanged -- the whole tetrafunctional layout is written around it.
"""

from __future__ import annotations

import numpy as np
import pytest

from hygel_martini.hydrogel_builder.config_params.config import Config
from hygel_martini.hydrogel_builder.core_utils.templates.linker_loader import (
    linker_definitions_from_library,
    load_linker_templates,
)

# Octahedral junction: a central bead with six arms along +-x, +-y, +-z.
ARMS = [
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
]
ARM_LENGTH = 0.35


def _octahedral_files(tmp_path, arms=ARMS, arm_length=ARM_LENGTH):
    """Write a GRO/ITP pair for a junction with one bead per arm."""
    count = len(arms) + 1
    centre = np.array([2.0, 2.0, 2.0])

    gro = [f"octahedral junction (f={len(arms)})", f"{count:5d}"]
    gro.append(
        "%5d%-5s%5s%5d%8.3f%8.3f%8.3f" % (1, "LNK", "C0", 1, *centre)
    )
    for position, direction in enumerate(arms, start=2):
        coordinate = centre + np.array(direction) * arm_length
        gro.append(
            "%5d%-5s%5s%5d%8.3f%8.3f%8.3f"
            % (1, "BCK", f"S{position - 1}", position, *coordinate)
        )
    gro.append("%10.5f%10.5f%10.5f" % (4.0, 4.0, 4.0))

    itp = ["[ moleculetype ]", "HEX 1", "", "[ atoms ]", "1 C1 1 LNK C0 1 0.0 72.0"]
    for position in range(2, count + 1):
        itp.append(f"{position} C1 1 BCK S{position - 1} {position} 0.0 72.0")
    itp += ["", "[ bonds ]"]
    for position in range(2, count + 1):
        itp.append(f"1 {position} 1 {arm_length:.3f} 1250")

    gro_path = tmp_path / "hex.gro"
    itp_path = tmp_path / "hex.itp"
    gro_path.write_text("\n".join(gro) + "\n")
    itp_path.write_text("\n".join(itp) + "\n")
    return gro_path, itp_path


BACKBONES = [
    {
        "id": "BB1",
        "ratio": 1,
        "definition": {
            "atom_type": "C1",
            "residue_name": "BCK1",
            "atom_name": "B1",
            "charge_group_number": 1,
            "mass": 56,
            "charge": 0.0,
        },
    }
]


def _entry(gro_path, itp_path, stub_count, **extra):
    entry = {
        "id": "HEX_linker",
        "ratio": 1,
        "gro": str(gro_path),
        "itp": str(itp_path),
        "linker_residue_name": "HEX",
        "backbone_residue_name": "HB",
        "stubs": [
            [{"between": "BB1", "bond_funct": 1, "bond_c0": 0.47, "bond_c1": 1250}]
            for _ in range(stub_count)
        ],
    }
    entry.update(extra)
    return entry


@pytest.fixture(autouse=True)
def _atom_masses():
    Config.set_runtime("atom_type_masses", {"C1": {"mass": 72.0}})


def test_a_six_arm_crosslinker_loads(tmp_path) -> None:
    gro_path, itp_path = _octahedral_files(tmp_path)
    library = load_linker_templates([_entry(gro_path, itp_path, 6)], BACKBONES)
    template = library.lookup["HEX_linker"]

    assert template.functionality == 6
    assert len(template.stub_bonds) == 6
    # every arm carries exactly one bond back to the central bead
    assert [len(group) for group in template.stub_bonds] == [1] * 6
    assert template.stub_backbone_targets == (("BB1",),) * 6
    # A stub bead stands in for the backbone end it will bond to, so its mass
    # comes from the backbone definition (56) and not from the ITP column (72).
    assert len(template.stub_definitions) == 6
    assert all(
        bead["mass"] == pytest.approx(56.0) for bead in template.stub_definitions
    )


def test_arm_vectors_are_measured_from_the_stub_centroid(tmp_path) -> None:
    gro_path, itp_path = _octahedral_files(tmp_path)
    library = load_linker_templates([_entry(gro_path, itp_path, 6)], BACKBONES)
    template = library.lookup["HEX_linker"]

    # The octahedron's stub centroid is its centre, so the arms come back as
    # the six axis directions at the arm length.
    assert template.arm_vectors.shape == (6, 3)
    assert np.linalg.norm(template.arm_vectors, axis=1) == pytest.approx(
        [ARM_LENGTH] * 6
    )
    assert template.arm_vectors.sum(axis=0) == pytest.approx([0.0, 0.0, 0.0])
    # span_length generalizes the stub-to-stub distance as twice the mean arm
    assert template.span_length == pytest.approx(2 * ARM_LENGTH)


def test_the_two_stub_view_is_left_empty_for_a_multi_arm_junction(tmp_path) -> None:
    # A consumer written around a left/right pair must not silently receive
    # half of a six-arm junction.
    gro_path, itp_path = _octahedral_files(tmp_path)
    template = load_linker_templates(
        [_entry(gro_path, itp_path, 6)], BACKBONES
    ).lookup["HEX_linker"]

    assert template.stub_bonds_left == []
    assert template.stub_bonds_right == []
    assert len(template.backbone_ids) == 6


def test_a_two_stub_template_still_fills_the_pair_view(tmp_path) -> None:
    gro_path, itp_path = _octahedral_files(tmp_path, arms=ARMS[:2])
    template = load_linker_templates(
        [_entry(gro_path, itp_path, 2)], BACKBONES
    ).lookup["HEX_linker"]

    assert template.functionality == 2
    assert len(template.stub_bonds_left) == 1
    assert len(template.stub_bonds_right) == 1
    assert template.stub_bonds_left == template.stub_bonds[0]
    assert template.stub_bonds_right == template.stub_bonds[1]
    # two stubs at +-arm_length along x: the span is the full separation, and
    # 2 * mean arm length agrees with it
    assert template.span_length == pytest.approx(2 * ARM_LENGTH)
    assert float(np.linalg.norm(template.span_vector)) == pytest.approx(2 * ARM_LENGTH)


def test_the_legacy_pair_spelling_still_works(tmp_path) -> None:
    gro_path, itp_path = _octahedral_files(tmp_path, arms=ARMS[:2])
    entry = _entry(gro_path, itp_path, 2)
    bond = entry.pop("stubs")[0]
    entry["backbone_1"] = bond
    entry["backbone_2"] = bond

    template = load_linker_templates([entry], BACKBONES).lookup["HEX_linker"]
    assert template.functionality == 2


def test_mixing_the_two_spellings_is_refused(tmp_path) -> None:
    gro_path, itp_path = _octahedral_files(tmp_path, arms=ARMS[:2])
    entry = _entry(gro_path, itp_path, 2)
    entry["backbone_1"] = entry["stubs"][0]

    with pytest.raises(ValueError, match="동시에 정의"):
        load_linker_templates([entry], BACKBONES)


def test_a_stub_count_mismatch_is_named(tmp_path) -> None:
    # Six BCK atoms in the template, four stubs declared.
    gro_path, itp_path = _octahedral_files(tmp_path)
    with pytest.raises(ValueError, match="stub 원자는 6개인데.*4개"):
        load_linker_templates([_entry(gro_path, itp_path, 4)], BACKBONES)


def test_definitions_expose_every_arm(tmp_path) -> None:
    gro_path, itp_path = _octahedral_files(tmp_path)
    library = load_linker_templates([_entry(gro_path, itp_path, 6)], BACKBONES)

    definition = linker_definitions_from_library(library)[0]["definition"]

    assert definition["functionality"] == 6
    assert len(definition["external_bonds"]) == 6
    assert all(len(group) == 1 for group in definition["external_bonds"])
    assert [group[0]["stub_index"] for group in definition["external_bonds"]] == list(
        range(6)
    )
    # the pair spelling is empty rather than a truncated view
    assert definition["external_bonds_1"] == []
    assert definition["external_bonds_2"] == []
