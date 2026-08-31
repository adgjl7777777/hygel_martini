"""A linker stub may declare several admissible partner backbones.

The configuration lists one entry per admissible partner, each with its own
bond parameters. Requiring exactly one target rejected that shape and made the
tracked ``04_full_builder`` example --- the first thing the start guide tells a
new user to run --- fail before any structure was built.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hygel_martini.hydrogel_builder.config_params.config import Config
from hygel_martini.hydrogel_builder.core_utils.templates.linker_loader import (
    _resolve_stub_targets,
    _stub_mass_for_targets,
    load_linker_templates,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MAKER = REPO_ROOT / "example" / "04_full_builder" / "project" / "maker.yaml"


def test_multiple_admissible_targets_are_accepted() -> None:
    targets = _resolve_stub_targets(
        [{"between": "BB2"}, {"between": "BB1"}], "LNK", "backbone_1"
    )
    assert targets == ("BB1", "BB2")  # sorted, so the result is stable


def test_a_single_target_still_resolves() -> None:
    assert _resolve_stub_targets([{"between": "BB1"}], "LNK", "backbone_1") == ("BB1",)


def test_declaring_no_target_is_still_an_error() -> None:
    with pytest.raises(ValueError, match="backbone target이 하나도 없습니다"):
        _resolve_stub_targets([], "LNK", "backbone_1")

    with pytest.raises(ValueError, match="backbone target이 하나도 없습니다"):
        _resolve_stub_targets([{"bond_c0": 0.47}], "LNK", "backbone_1")


def test_stub_mass_follows_its_partners_when_they_agree() -> None:
    mass = _stub_mass_for_targets(
        ("BB1", "BB2"), {"BB1": 56.0, "BB2": 56.0}, "LNK", "backbone_1"
    )
    assert mass == pytest.approx(56.0)


def test_disagreeing_partner_masses_are_refused_not_silently_resolved() -> None:
    # Picking whichever target sorts first would give a stub the wrong mass.
    with pytest.raises(ValueError, match="질량이 서로 다른"):
        _stub_mass_for_targets(
            ("BB1", "BB2"), {"BB1": 56.0, "BB2": 72.0}, "LNK", "backbone_1"
        )


def test_unknown_partner_backbone_is_named() -> None:
    with pytest.raises(ValueError, match="BBX"):
        _stub_mass_for_targets(("BBX",), {"BB1": 56.0}, "LNK", "backbone_1")


@pytest.mark.skipif(
    not EXAMPLE_MAKER.is_file(), reason="tracked example 04 not present"
)
def test_the_tracked_example_linker_configuration_loads() -> None:
    Config.load_config(str(EXAMPLE_MAKER))
    Config.set_runtime(
        "atom_type_masses", {"C1": {"mass": 72.0}, "C2": {"mass": 72.0}}
    )
    components = Config.get_param("hydrogel_components")

    library = load_linker_templates(
        components["linker_definitions"]["LINKERS"],
        components["backbone_definitions"]["BACKBONES"],
    )

    assert set(library.lookup) == {"SS_linker", "DD_linker", "SD_linker"}
    for template in library.lookup.values():
        assert template.span_length > 0
        assert template.stub_bonds_left and template.stub_bonds_right
