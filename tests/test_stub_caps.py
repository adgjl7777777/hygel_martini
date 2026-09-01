"""Per-stub cap atoms: chemistry-faithful partial conversion.

A cap atom (an unreacted arm's thiol hydrogen) is a real template body atom
that exists only while its stub is unreacted; a reacted arm loses it and
takes per-atom overrides (the sulfur's reacted-form charge/type). The layout
decides which arms react, the blueprint withholds the caps, index maps do the
rest -- and everything downstream must index through maps, never positionally,
because a withheld atom leaves a hole.
"""

from __future__ import annotations

import numpy as np
import pytest

from hygel_martini.hydrogel_builder.core_utils.layout.proto_populator import (
    _create_linker_bonds,
)
from hygel_martini.hydrogel_builder.core_utils.layout.layout_executor import (
    ChainBlueprint,
)
from hygel_martini.hydrogel_builder.core_utils.runtime.dynamic_crosslink import (
    plan_dynamic_crosslinks,
)
from hygel_martini.hydrogel_builder.core_utils.templates.linker_loader import (
    load_linker_templates,
)
from hygel_martini.hydrogel_builder.main_components import Attributes
from hygel_martini.hydrogel_builder.main_components.Universe import World

# A two-arm thiol junction: body carbon C0, two BCK sulfurs, each with its
# thiol hydrogen. Exactly the shape the cap machinery exists for.
ITP = """\
[ moleculetype ]
  CAP   1
[ atoms ]
   1  hxC  1  LNK  C00  1  -0.10  12.011
   2  hxS  1  BCK  S01  2  -0.30  32.060
   3  hxH  1  LNK  H02  3   0.20   1.008
   4  hxS  1  BCK  S03  4  -0.30  32.060
   5  hxH  1  LNK  H04  5   0.20   1.008
[ bonds ]
  1 2 1 0.18 1000.0
  2 3 1 0.134 1200.0
  1 4 1 0.18 1000.0
  4 5 1 0.134 1200.0
"""

_COORDS = [
    ("C00", 0.0, 0.0, 0.0),
    ("S01", 0.18, 0.0, 0.0),
    ("H02", 0.25, 0.10, 0.0),
    ("S03", -0.18, 0.0, 0.0),
    ("H04", -0.25, 0.10, 0.0),
]
GRO = "cap linker\n5\n" + "".join(
    f"{1:5d}{'LNK':<5s}{name:>5s}{i + 1:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n"
    for i, (name, x, y, z) in enumerate(_COORDS)
) + "   5.00000   5.00000   5.00000\n"

BACKBONES = [{"id": "BB1", "definition": {"mass": 72.0, "residue_name": "BCK"}}]


def _entry(tmp_path, stub_caps):
    itp = tmp_path / "CAP.itp"
    gro = tmp_path / "CAP.gro"
    itp.write_text(ITP)
    gro.write_text(GRO)
    entry = {
        "id": "CAP_linker",
        "gro": str(gro),
        "itp": str(itp),
        "linker_residue_name": "LNK",
        "backbone_residue_name": "HB",
        "backbone_1": [{"between": "BB1", "bond_funct": 1, "bond_c0": 0.17, "bond_c1": 1000}],
        "backbone_2": [{"between": "BB1", "bond_funct": 1, "bond_c0": 0.17, "bond_c1": 1000}],
    }
    if stub_caps is not None:
        entry["stub_caps"] = stub_caps
    return entry


def test_caps_resolve_to_body_positions_and_overrides(tmp_path) -> None:
    caps = [
        {"cap_atoms": ["H02"], "reacted": {"S01": {"charge": -0.1, "type": "lkS"}}},
        {"cap_atoms": ["H04"], "reacted": {"S03": {"charge": -0.1, "type": "lkS"}}},
    ]
    library = load_linker_templates([_entry(tmp_path, caps)], BACKBONES)
    template = library.records[0].template

    # Body positions: C00=0, H02=1, H04=2 (stubs are split out of the body).
    assert template.stub_caps[0]["cap_body_positions"] == [1]
    assert template.stub_caps[0]["cap_original_indices"] == [3]
    assert template.stub_caps[1]["cap_body_positions"] == [2]
    # Overrides are keyed by the original 1-based index of the target atom.
    assert template.stub_caps[0]["overrides"] == {2: {"charge": -0.1, "type": "lkS"}}


def test_a_cap_list_of_the_wrong_length_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="stub 수"):
        load_linker_templates(
            [_entry(tmp_path, [{"cap_atoms": ["H02"]}])], BACKBONES
        )


def test_a_cap_naming_a_stub_atom_is_refused(tmp_path) -> None:
    caps = [{"cap_atoms": ["S01"]}, {}]
    with pytest.raises(ValueError, match="stub 원자"):
        load_linker_templates([_entry(tmp_path, caps)], BACKBONES)


def test_a_cap_on_another_arm_is_refused(tmp_path) -> None:
    # H04 belongs to the second arm; declaring it as the FIRST stub's cap
    # would delete another arm's atom whenever the first reacts.
    caps = [{"cap_atoms": ["H04"]}, {}]
    with pytest.raises(ValueError, match="결합해 있지"):
        load_linker_templates([_entry(tmp_path, caps)], BACKBONES)


def test_an_unknown_cap_name_is_refused(tmp_path) -> None:
    caps = [{"cap_atoms": ["H99"]}, {}]
    with pytest.raises(ValueError, match="H99"):
        load_linker_templates([_entry(tmp_path, caps)], BACKBONES)


def test_linker_bonds_index_through_the_bead_map_not_positionally() -> None:
    # A withheld cap atom leaves a hole in the bead map. Positional indexing
    # into the compacted atom list silently shifted every later bond across
    # the hole -- measured as junctions fragmenting into pieces on the first
    # capped build.
    World.Atoms.clear()
    World.Bonds.clear()
    Attributes.initialize()
    ids = [Attributes.Atom().atom_id for _ in range(3)]
    # Template body positions 0,1,2,3 where position 1 (the cap) is withheld.
    bead_map = {0: ids[0], 2: ids[1], 3: ids[2]}
    chain = ChainBlueprint(
        chain_type="linker",
        chain_index=0,
        component_id="CAP",
        definition={"bonds": [
            {"from": 0, "to": 2, "funct": 1, "length": 0.2, "fc": 100.0},
            {"from": 0, "to": 1, "funct": 1, "length": 0.15, "fc": 100.0},  # cap bond
            {"from": 2, "to": 3, "funct": 1, "length": 0.3, "fc": 100.0},
        ]},
        atom_indices=[],
        metadata={},
    )

    _create_linker_bonds(chain, bead_map)

    keys = set(World.Bonds)
    assert (min(ids[0], ids[1]), max(ids[0], ids[1])) in keys  # 0-2 mapped
    assert (min(ids[1], ids[2]), max(ids[1], ids[2])) in keys  # 2-3 mapped
    assert len(keys) == 2  # the cap bond dropped, nothing shifted


class _Stub:
    def __init__(self, atom_id, position, planned, reacted):
        self.atom_id = atom_id
        self.position = position
        self.planned_endpoint_edges = None
        self.planned_endpoints = planned
        self.stub_type = "stub_0"
        self.target_backbone = None
        self.backbone_type = None
        self.cap_reacted = reacted


class _End:
    def __init__(self, atom_id, position, endpoint_id, chain_index):
        self.atom_id = atom_id
        self.position = position
        self.planned_endpoint_id = endpoint_id
        self.chain_index = chain_index
        self.end_tag = 1
        self.chain_type = "backbone"
        self.linker_chain_index = None
        self.backbone_type = None


def test_router_bonds_only_reacted_stubs() -> None:
    # Six arms, one planned end. The NEAREST arm is unreacted (kept its cap);
    # the plan says a farther arm reacted, and the router must honor that --
    # bonding the unreacted arm would put a bond on an atom that still has
    # its thiol hydrogen.
    planned = ("a",)
    positions = [
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    ]
    reacted_flags = [False, False, True, False, False, False]
    stubs = [
        _Stub(600 + i, p, planned, reacted_flags[i])
        for i, p in enumerate(positions)
    ]
    ends = {0: [_End(0, (3.0, 0.0, 0.0), "a", 0)]}  # nearest to arm 0 (unreacted)

    assignments, notes = plan_dynamic_crosslinks({0: stubs}, ends, None)

    chosen = assignments[0]
    assert len(chosen) == 1
    assert chosen[0].stub_atom.atom_id == 602  # the reacted arm, not the nearest


def test_too_few_reacted_stubs_is_refused() -> None:
    planned = ("a", "b")
    stubs = [
        _Stub(700, (1.0, 0.0, 0.0), planned, True),
        _Stub(701, (-1.0, 0.0, 0.0), planned, False),
    ]
    ends = {
        0: [_End(0, (2.0, 0.0, 0.0), "a", 0)],
        1: [_End(1, (-2.0, 0.0, 0.0), "b", 1)],
    }

    with pytest.raises(ValueError, match="reacted"):
        plan_dynamic_crosslinks({0: stubs}, ends, None)
