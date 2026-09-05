"""The force field is asked whether it agrees with the DFT it is meant to model.

The project's mechanism is an ordering: chloride prefers the thiourethane N-H
over the urethane N-H over the residual thiol S-H. These tests pin the two
things that make the benchmark trustworthy -- the energy expression and the
graph matching that puts parameters on DFT coordinates -- and record the
result it currently returns, so a change to the charges (the pending RESP
release, for one) shows up as a test that has to be re-examined rather than a
number nobody re-read.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "example", "08_des_thiourethane_aa")
VALIDATION = os.path.join(EXAMPLE, "validation")
SCRIPT = os.path.join(VALIDATION, "ff_pair_benchmark.py")
DFT = "/nas_3/active/soohki/27.des/dft/03_complexes"

pytestmark = pytest.mark.skipif(not os.path.isfile(SCRIPT), reason="validation/ not present")


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("ff_pair_benchmark", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ff_pair_benchmark"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_two_point_charges_reproduce_coulombs_law(bench):
    """The energy expression, checked against arithmetic anyone can redo."""
    a = bench.Atom("Na", 0.0, 0.0, +1.0)
    b = bench.Atom("Cl", 0.0, 0.0, -1.0)
    total, lj, qq = bench.interaction_energy([a], [(0.0, 0.0, 0.0)], [b], [(3.0, 0.0, 0.0)])
    assert lj == 0.0
    assert qq == pytest.approx(-bench.COULOMB / 0.3)      # 3 Angstrom = 0.3 nm
    assert total == pytest.approx(qq)


def test_lennard_jones_vanishes_at_sigma_and_bottoms_at_the_minimum(bench):
    a = bench.Atom("X", 0.3, 1.0, 0.0)
    b = bench.Atom("Y", 0.3, 1.0, 0.0)
    at_sigma = bench.interaction_energy([a], [(0.0, 0.0, 0.0)], [b], [(3.0, 0.0, 0.0)])[1]
    assert at_sigma == pytest.approx(0.0, abs=1e-9)
    rmin = 0.3 * 2 ** (1 / 6) * 10          # Angstrom
    at_min = bench.interaction_energy([a], [(0.0, 0.0, 0.0)], [b], [(rmin, 0.0, 0.0)])[1]
    assert at_min == pytest.approx(-1.0, abs=1e-9)


def test_graph_matching_survives_a_shuffled_atom_order(bench):
    """Parameters must follow chemistry, not file order."""
    elements = ["C", "H", "H", "H", "O"]
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4)]
    shuffled = ["O", "H", "C", "H", "H"]
    shuffled_bonds = [(2, 1), (2, 3), (2, 4), (2, 0)]
    mapping = bench.match_graphs(elements, bonds, shuffled, shuffled_bonds)
    assert mapping is not None
    assert shuffled[mapping[0]] == "C" and shuffled[mapping[4]] == "O"


def test_a_different_molecule_is_refused_rather_than_matched(bench):
    assert bench.match_graphs(["C", "H"], [(0, 1)], ["C", "O"], [(0, 1)]) is None


@pytest.mark.skipif(not os.path.isdir(DFT), reason="DFT complexes not reachable")
def test_the_force_field_inverts_the_thiourethane_urethane_ranking(bench):
    """The finding, pinned. If this test changes, the mechanism story changed.

    DFT puts thiourethane N-H 2.8 kcal/mol below urethane N-H. This force
    field puts them within a kcal of each other in the wrong order, because
    1.14*CM1A-LBCC gives the two N-H hydrogens the same charge to within
    0.005 e. A charge set that fixes this -- RESP, when it is released --
    should make this test fail.
    """
    rows = {name: bench.benchmark(name, DFT, path)
            for name, path in bench.DEFAULT_ITPS.items() if os.path.exists(path)}
    urethane = rows["C2_urethane_Cl"]
    thiourethane = rows["C3_thiouret_Cl"]
    thiol = rows["C1_thiol_Cl"]

    # DFT reference ordering, as recomputed from the raw ORCA outputs.
    assert thiourethane["dft_int"] < urethane["dft_int"] < thiol["dft_int"]
    # The force field agrees about the thiol and disagrees about the rest.
    assert thiol["mm_int"] > urethane["mm_int"]
    assert thiourethane["mm_int"] > urethane["mm_int"], "ranking no longer inverted -- re-read the report"
    # And the reason: the donor hydrogens carry the same charge.
    assert abs(thiourethane["q_h"] - urethane["q_h"]) < 0.01
    # Everything underbinds; the model is not polarizable.
    for row in rows.values():
        assert row["mm_int"] - row["dft_int"] > 8.0
