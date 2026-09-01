"""Tiling a parameterized repeat unit up to a real chain length.

The experimental PPG-TDI strand is far past LigParGen's atom ceiling, so the
long strand is built by replicating the parameterized block's middle repeat
unit (``example/08_des_thiourethane_aa/parameterization/tile_ppg.py``). The
script carries its own verification and refuses to run on an irregular
interior; these tests pin the properties that make the replication defensible,
on the shipped templates, so a change to either the templates or the script
cannot quietly break them.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "example", "08_des_thiourethane_aa")
SCRIPT = os.path.join(EXAMPLE, "parameterization", "tile_ppg.py")
STRUCTURE = os.path.join(EXAMPLE, "project", "structure")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(STRUCTURE, "STR.itp")),
    reason="example 08 templates are not present",
)


@pytest.fixture(scope="module")
def tiler():
    spec = importlib.util.spec_from_file_location("tile_ppg", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def strand(tiler):
    return tiler.Strand(
        os.path.join(STRUCTURE, "STR.itp"), os.path.join(STRUCTURE, "STR.gro")
    )


def test_the_parameterized_interior_is_regular(tiler, strand) -> None:
    # The whole approach rests on this: units interchangeable in parameters
    # (not in type names, which LigParGen mints per atom), no term touching
    # three units, matching per-unit counts. check_regularity raises otherwise.
    spans = strand.check_regularity()
    assert len(strand.units) == 3
    assert all(len(unit) == 10 for unit in strand.units)
    for section, expected in (
        ("bonds", 9), ("angles", 15), ("dihedrals", 12), ("pairs", 12)
    ):
        # Every unit carries the same intra-unit pattern...
        for unit_index in range(len(strand.units)):
            assert spans[section][(unit_index,)] == expected, (section, unit_index)
        # ...and both unit boundaries carry the same spanning pattern, which is
        # what lets one boundary's terms stand in for every inserted one.
        assert spans[section][(0, 1)] == spans[section][(1, 2)] > 0, section


def test_tiling_preserves_net_charge_and_scales_terms(tiler, strand) -> None:
    atoms, terms, coords, _ = tiler.tile(strand, 33, quiet=True)
    report = tiler.verify(strand, atoms, terms, coords, 33)

    assert report["atoms"] == 73 + 30 * 10
    # Neutrality is not approximate: each copy is corrected to exactly zero,
    # so the molecule keeps the parameterization's own net charge.
    original = sum(a["q"] for a in strand.atoms.values())
    assert report["charge"] == pytest.approx(original, abs=1e-6)
    # Physical bonds and no clashes in the starting geometry.
    assert 0.09 < report["bond_min"] and report["bond_max"] < 0.20
    assert report["min_nonbonded"] > 0.15
    # An extended chain: 30 more units at ~0.36 nm rise.
    assert report["span"] > 12.0


def test_tiling_below_the_parameterized_length_is_refused(tiler, strand) -> None:
    with pytest.raises(SystemExit, match="below the parameterized"):
        tiler.tile(strand, 2, quiet=True)


def test_an_irregular_interior_is_refused(tiler, strand) -> None:
    # Perturb one interior atom's mass: the units stop being interchangeable,
    # and tiling must refuse rather than average over the difference.
    victim = strand.units[1][0]
    original = strand.atoms[victim]["m"]
    strand.atoms[victim]["m"] = original + 1.0
    try:
        with pytest.raises(SystemExit, match="differs from unit 0"):
            strand.check_regularity()
    finally:
        strand.atoms[victim]["m"] = original


def test_the_shipped_long_template_matches_the_script(tiler, strand) -> None:
    itp = os.path.join(STRUCTURE, "STR_n33.itp")
    if not os.path.isfile(itp):
        pytest.skip("STR_n33 has not been generated")
    shipped = tiler.parse_itp(itp)
    atoms, terms, _, _ = tiler.tile(strand, 33, quiet=True)
    assert len(shipped["atoms"]) == len(atoms)
    for section, width in tiler.BONDED_WIDTH.items():
        assert len(shipped[section]) == len(terms[section]), section
    charge = sum(float(row[6]) for row in shipped["atoms"])
    assert charge == pytest.approx(
        sum(a["q"] for a in strand.atoms.values()), abs=1e-5
    )
