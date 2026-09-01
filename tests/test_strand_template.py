"""Whole-strand templates: one rigid molecule per network strand.

The all-atom path ingests a strand as a single molecule (BCK1/BCK2 attachment
atoms) and places it rigidly between its two junction sites. Rigidity is the
contract: no scaling, no bowing, and no rewiring -- a molecule has one length,
so heterogeneous junction gaps cannot be spanned honestly.
"""

from __future__ import annotations

import numpy as np
import pytest

from hygel_martini.hydrogel_builder.core_utils.layout.layout_executor import (
    instantiate_backbone,
)
from hygel_martini.hydrogel_builder.core_utils.layout.proto_layout import LayoutCell
from hygel_martini.hydrogel_builder.core_utils.templates.strand_loader import (
    load_strand_template,
)

ITP = """\
[ moleculetype ]
  STR   3
[ atoms ]
;  nr type resnr residue atom cgnr charge mass
   1  st1  1  STR  BCK1  1   0.10  12.011
   2  st2  1  STR  C02   2  -0.20  12.011
   3  st3  1  STR  BCK2  3   0.10  12.011
   4  st4  1  STR  H03   4   0.00   1.008
[ bonds ]
  1 2 1 0.15 1000.0
  2 3 1 0.15 1000.0
  2 4 1 0.11 1200.0
[ angles ]
  1 2 3 1 111.0 400.0
[ pairs ]
  1 3 1
"""

_COORDS = [
    ("BCK1", 0.000, 0.000, 0.000),
    ("C02", 0.150, 0.100, 0.000),
    ("BCK2", 0.300, 0.000, 0.000),
    ("H03", 0.150, 0.210, 0.000),
]
GRO = "strand\n4\n" + "".join(
    f"{1:5d}{'STR':<5s}{name:>5s}{i + 1:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n"
    for i, (name, x, y, z) in enumerate(_COORDS)
) + "   5.00000   5.00000   5.00000\n"


def _write(tmp_path, itp=ITP, gro=GRO):
    itp_path = tmp_path / "STR.itp"
    gro_path = tmp_path / "STR.gro"
    itp_path.write_text(itp)
    gro_path.write_text(gro)
    return {"id": "STR1", "itp": str(itp_path), "gro": str(gro_path)}


def test_loader_finds_the_two_attachment_atoms(tmp_path) -> None:
    template = load_strand_template(_write(tmp_path))

    assert template.attachment_positions == (0, 2)
    assert template.span_length == pytest.approx(0.3)
    # Local frame is centered on the attachment midpoint.
    mid = 0.5 * (template.coords[0] + template.coords[2])
    assert np.allclose(mid, 0.0, atol=1e-12)
    assert np.allclose(template.attachment_axis, [1.0, 0.0, 0.0])
    # Every atom survives, with its ITP identity.
    assert [b.name for b in template.beads] == ["BCK1", "C02", "BCK2", "H03"]
    assert template.total_mass == pytest.approx(12.011 * 3 + 1.008)
    # Angle rows are re-indexed to 0-based bead positions for the
    # source_index lookup in construct_angles.
    assert template.internal_angles == [
        {"from": 0, "center": 1, "to": 2, "funct": 1, "params": [111.0, 400.0]}
    ]
    assert len(template.internal_bonds) == 3
    assert len(template.pairs) == 1


def test_a_wrong_number_of_bck_atoms_is_refused(tmp_path) -> None:
    single = ITP.replace("BCK2  3   0.10", "C0X   3   0.10")
    with pytest.raises(ValueError, match="BCK1.*BCK2"):
        load_strand_template(_write(tmp_path, itp=single))


def test_missing_mass_is_refused(tmp_path) -> None:
    broken = ITP.replace("   4  st4  1  STR  H03   4   0.00   1.008\n", "")
    with pytest.raises(ValueError):
        load_strand_template(_write(tmp_path, itp=broken))


def test_rigid_placement_puts_the_attachment_atoms_on_the_segment(tmp_path) -> None:
    template = load_strand_template(_write(tmp_path))
    origin = np.array([2.0, 3.0, 4.0])
    direction = np.array([0.0, 0.0, 1.0])
    cell = LayoutCell(
        origin=origin,
        direction=direction,
        backbone_definition={"id": "STR1"},
        cell_index=(0, 0, 0),
        metadata={"strand_template": template, "roll": 0.7},
    )

    chain = instantiate_backbone(cell, proto_positions=np.zeros((3, 3)))

    head = chain.positions[0]
    tail = chain.positions[2]
    # Ends sit at origin -/+ span/2 along the segment: rotated, never scaled.
    assert np.allclose(head, origin - 0.15 * direction, atol=1e-9)
    assert np.allclose(tail, origin + 0.15 * direction, atol=1e-9)
    # Internal geometry is untouched (rigid): all pairwise distances match.
    ref = template.coords
    for a in range(4):
        for b in range(a + 1, 4):
            assert np.linalg.norm(chain.positions[a] - chain.positions[b]) == (
                pytest.approx(float(np.linalg.norm(ref[a] - ref[b])), abs=1e-9)
            )


def test_net_layout_refuses_rewiring_with_a_strand_template(tmp_path) -> None:
    from hygel_martini.hydrogel_builder.core_utils.layout.net_layout import (
        generate_net_layout_plan,
    )

    template = load_strand_template(_write(tmp_path))

    class _Proto:
        proto_backbone = None

    with pytest.raises(ValueError, match="rewiring"):
        generate_net_layout_plan(
            _Proto(),
            [{"id": "STR1", "strand_template": template}],
            [{"id": "HEX"}],
            net="pcu",
            repeats=4,
            cell_parameter=3.0,
            max_span=6.0,
            rewire_seed=0,
        )
