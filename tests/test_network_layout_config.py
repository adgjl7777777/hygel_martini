"""The ``network_layout`` block selects a net-driven layout.

Absent, the historical diamond path runs unchanged, so existing configurations
are unaffected by the key existing. Present, it is validated before any
structure is built, because a typo discovered halfway through a build is a typo
discovered too late.
"""

from __future__ import annotations

import pytest

from hygel_martini.hydrogel_builder.config_params.build_hydrogel import (
    _resolve_network_layout,
)


def _block(**overrides):
    block = {"net": "pcu", "repeats": 4, "cell_parameter": 3.0}
    block.update(overrides)
    return {"network_layout": block}


def test_absent_or_disabled_selects_the_diamond_path() -> None:
    assert _resolve_network_layout({}) is None
    assert _resolve_network_layout({"network_layout": None}) is None
    assert _resolve_network_layout({"network_layout": {}}) is None
    assert _resolve_network_layout({"network_layout": False}) is None


def test_a_minimal_block_resolves() -> None:
    resolved = _resolve_network_layout(_block())

    assert resolved["net"] == "pcu"
    assert resolved["repeats"] == (4, 4, 4)
    assert resolved["cell_parameter"] == pytest.approx(3.0)
    assert resolved["max_span"] is None  # no rewiring unless asked for
    assert resolved["rewire_kwargs"] == {}


def test_repeats_may_be_a_scalar_or_a_triple() -> None:
    assert _resolve_network_layout(_block(repeats=4))["repeats"] == (4, 4, 4)
    assert _resolve_network_layout(_block(repeats=[4, 6, 8]))["repeats"] == (4, 6, 8)

    with pytest.raises(ValueError, match="three values"):
        _resolve_network_layout(_block(repeats=[4, 4]))


def test_missing_required_keys_are_named() -> None:
    for missing, pattern in (
        ("net", "needs a 'net'"),
        ("repeats", "needs 'repeats'"),
        ("cell_parameter", "positive 'cell_parameter'"),
    ):
        block = _block()
        block["network_layout"].pop(missing)
        with pytest.raises(ValueError, match=pattern):
            _resolve_network_layout(block)


def test_a_non_positive_cell_parameter_is_refused() -> None:
    with pytest.raises(ValueError, match="positive 'cell_parameter'"):
        _resolve_network_layout(_block(cell_parameter=0.0))


def test_an_unknown_key_is_reported_rather_than_ignored() -> None:
    # A silently ignored key is a setting the user believes is in effect.
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['nett'\]"):
        _resolve_network_layout({"network_layout": {"nett": "pcu"}})

    with pytest.raises(ValueError, match=r"unknown key\(s\) \['max_spam'\]"):
        _resolve_network_layout(_block(rewiring={"max_spam": 6.0}))


def test_a_non_mapping_block_is_refused() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        _resolve_network_layout({"network_layout": "pcu"})

    with pytest.raises(ValueError, match="rewiring' must be a mapping"):
        _resolve_network_layout(_block(rewiring=6.0))


def test_rewiring_options_resolve() -> None:
    resolved = _resolve_network_layout(
        _block(
            rewiring={
                "max_span": 6.0,
                "seed": 7,
                "max_sweeps": 30,
                "allow_parallel_strands": False,
            }
        )
    )

    assert resolved["max_span"] == pytest.approx(6.0)
    assert resolved["rewire_seed"] == 7
    assert resolved["rewire_kwargs"] == {
        "max_sweeps": 30,
        "allow_parallel_strands": False,
    }


def test_rewiring_options_without_a_cutoff_are_refused() -> None:
    # Without max_span no rewiring runs at all, so the other options would
    # silently do nothing.
    with pytest.raises(ValueError, match="without 'max_span'"):
        _resolve_network_layout(_block(rewiring={"max_sweeps": 30}))


# --------------------------------------------------------------------------
# the tracked example
# --------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
HEX_MAKER = REPO_ROOT / "example" / "07_hexafunctional" / "project" / "maker.yaml"


@pytest.mark.skipif(not HEX_MAKER.is_file(), reason="example 07 not present")
def test_the_hexafunctional_example_declares_a_loadable_f6_system() -> None:
    from hygel_martini.hydrogel_builder.config_params.config import Config
    from hygel_martini.hydrogel_builder.core_utils.io.martini_parser import (
        read_atom_types,
    )
    from hygel_martini.hydrogel_builder.core_utils.templates.linker_loader import (
        load_linker_templates,
    )

    Config.load_config(str(HEX_MAKER))
    parameters = Config.get_param("simulation_parameters")
    if not Path(parameters["base_itp_file"]).is_file():
        pytest.skip("Martini force field not present")
    Config.set_runtime("atom_type_masses", read_atom_types(parameters["base_itp_file"]))

    resolved = _resolve_network_layout(parameters)
    assert resolved["net"] == "pcu"
    assert resolved["repeats"] == (4, 4, 4)
    assert resolved["max_span"] == pytest.approx(6.0)
    # a coordinate build cannot place a primary loop, so the example says so
    assert resolved["rewire_kwargs"]["allow_primary_loops"] is False

    components = Config.get_param("hydrogel_components")
    library = load_linker_templates(
        components["linker_definitions"]["LINKERS"],
        components["backbone_definitions"]["BACKBONES"],
    )
    template = library.lookup["HEX_linker"]

    # the crosslinker's functionality must match the net's coordination, or the
    # junction cannot take the ends the net gives it
    from hygel_martini.hydrogel_builder.core_utils.layout.nets import get_net

    assert template.functionality == get_net(resolved["net"]).coordination == 6
