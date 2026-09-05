"""The charge pipeline changes exactly one column, and the file adds up.

``apply_charges.py`` is the only sanctioned way a charge set enters this
example's topologies (integrated report §18B.2, §18C.4). These tests pin what
makes it safe to trust: nothing but charges moves, the written file sums to
its formal charge exactly, a release that does not fit the molecule is
refused rather than repaired, and the junction's reacted-sulfur overrides are
kept consistent with the junction's charges.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import shutil
from decimal import Decimal

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "example", "08_des_thiourethane_aa")
SCRIPT = os.path.join(EXAMPLE, "parameterization", "apply_charges.py")
STRUCTURE = os.path.join(EXAMPLE, "project", "structure")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(STRUCTURE, "HEXU.itp")),
    reason="example 08 templates are not present",
)


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("apply_charges", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def work(tmp_path):
    for name in ("HEXU.itp", "HEXU.gro", "STR.itp", "ACC.itp", "CL.itp"):
        shutil.copy(os.path.join(STRUCTURE, name), tmp_path / name)
    shutil.copy(os.path.join(EXAMPLE, "project", "config", "hydrogel.yaml"), tmp_path / "hydrogel.yaml")
    return tmp_path


def _sum_and_rows(path):
    total = Decimal(0)
    rows = []
    other = []
    section = None
    started = False
    for line in open(path):
        text = line.split(";", 1)[0].strip()
        if text.startswith("["):
            section = text.strip("[] ").lower()
            started = True
        if not started:
            continue
        if section == "atoms" and text and not text.startswith("["):
            parts = text.split()
            total += Decimal(parts[6])
            rows.append(parts)
        else:
            other.append(line.rstrip("\n"))
    return total, rows, other


def _perturb(path, delta="-0.000100"):
    """Recreate the upstream defect: knock one atom's charge off so the sum misses zero.

    The shipped templates are already neutralized (release v1), so the test
    manufactures the -1e-4 e that LigParGen tables carry rather than relying on
    the tree being in any particular state.
    """
    lines = open(path).read().split("\n")
    section = None
    for i, line in enumerate(lines):
        text = line.split(";", 1)[0].strip()
        if text.startswith("["):
            section = text.strip("[] ").lower()
            continue
        if section == "atoms" and text:
            parts = line.split()
            parts[6] = str(Decimal(parts[6]) + Decimal(delta))
            lines[i] = "  ".join(parts)
            break
    open(path, "w").write("\n".join(lines))


def test_neutralize_makes_the_sum_exactly_zero_and_touches_nothing_else(tool, work):
    _perturb(work / "HEXU.itp")
    before_total, before_rows, before_other = _sum_and_rows(work / "HEXU.itp")
    assert before_total == Decimal("-0.000100")
    tool.main(["neutralize", str(work / "HEXU.itp"), "--out", str(work / "out.itp"), "--rule", "heavy"])
    total, rows, other = _sum_and_rows(work / "out.itp")
    assert total == Decimal(0)
    assert other == before_other
    assert len(rows) == len(before_rows)
    for old, new in zip(before_rows, rows):
        assert old[:6] == new[:6] and old[7:] == new[7:]
        assert abs(Decimal(new[6]) - Decimal(old[6])) <= Decimal("0.00001")
    # hydrogens were not the ones adjusted
    hydrogens = [r for r, o in zip(rows, before_rows) if float(r[7]) < 1.5 and r[6] != o[6]]
    assert hydrogens == []
    assert os.path.exists(str(work / "out.itp") + ".charges.json")


def test_neutralize_is_idempotent(tool, work):
    tool.main(["neutralize", str(work / "STR.itp"), "--out", str(work / "a.itp")])
    tool.main(["neutralize", str(work / "a.itp"), "--out", str(work / "b.itp")])
    _, rows_a, _ = _sum_and_rows(work / "a.itp")
    _, rows_b, _ = _sum_and_rows(work / "b.itp")
    assert [r[6] for r in rows_a] == [r[6] for r in rows_b]


def test_stub_caps_follow_the_junction_charges(tool, work):
    tool.main(["neutralize", str(work / "HEXU.itp"), "--out", str(work / "out.itp"),
               "--rule", "heavy", "--stub-caps", str(work / "hydrogel.yaml")])
    itp = tool.Itp(str(work / "out.itp"))
    q = dict(zip(itp.names, itp.charges()))
    text = (work / "hydrogel.yaml").read_text()
    found = tool.STUB_CAP_RE.findall(text)
    assert len(found) == 6
    for match in tool.STUB_CAP_RE.finditer(text):
        cap, sulfur = match.group("cap"), match.group("sulfur")
        assert Decimal(match.group("q")) == (q[sulfur] + q[cap]).quantize(tool.QUANTUM)


def test_partial_or_mismatched_release_is_refused(tool, work):
    itp = tool.Itp(str(work / "STR.itp"))
    names = itp.names
    rel = work / "rel.csv"
    with open(rel, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["molecule", "atom_name", "charge"])
        for n in names[:-1]:                         # one atom missing
            w.writerow(["STR", n, "0.0"])
    with pytest.raises(ValueError, match="without a charge"):
        tool.cmd_apply(itp, str(work / "x.itp"), "uniform", 0, str(rel))

    with open(rel, "w", newline="") as handle:       # complete, but sums to +1
        w = csv.writer(handle)
        w.writerow(["molecule", "atom_name", "charge"])
        for i, n in enumerate(names):
            w.writerow(["STR", n, "1.0" if i == 0 else "0.0"])
    with pytest.raises(ValueError, match="fitting or mapping error"):
        tool.cmd_apply(tool.Itp(str(work / "STR.itp")), str(work / "x.itp"), "uniform", 0, str(rel))


def test_apply_round_trips_its_own_charges(tool, work):
    itp = tool.Itp(str(work / "STR.itp"))
    rel = work / "rel.csv"
    with open(rel, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["molecule", "atom_name", "charge"])
        for n, q in zip(itp.names, itp.charges()):
            w.writerow(["STR", n, str(q)])
    tool.main(["apply", str(work / "STR.itp"), "--release", str(rel), "--out", str(work / "y.itp")])
    total, rows, _ = _sum_and_rows(work / "y.itp")
    assert total == Decimal(0)
    for name, q, row in zip(itp.names, itp.charges(), rows):
        assert row[4] == name
        assert abs(Decimal(row[6]) - q) <= Decimal("0.00001")


def test_scale_hits_the_scaled_formal_charge_exactly_and_refuses_neutrals(tool, work):
    tool.main(["scale", str(work / "ACC.itp"), "--out", str(work / "ACC_f080.itp"),
               "--factor", "0.8", "--total", "1", "--rule", "heavy"])
    total, _, _ = _sum_and_rows(work / "ACC_f080.itp")
    assert total == Decimal("0.800000")
    tool.main(["scale", str(work / "CL.itp"), "--out", str(work / "CL_f069.itp"),
               "--factor", "0.69", "--total", "-1"])
    total, _, _ = _sum_and_rows(work / "CL_f069.itp")
    assert total == Decimal("-0.690000")
    with pytest.raises(ValueError, match="neutral molecule"):
        tool.cmd_scale(tool.Itp(str(work / "STR.itp")), str(work / "z.itp"), "0.8", 0, "uniform")


def test_scale_copies_coordinates_beside_the_variant(tool, work):
    # HEXU has a .gro beside it; a variant written under a new stem gets one too.
    tool.main(["neutralize", str(work / "HEXU.itp"), "--out", str(work / "HEXU_v.itp")])
    assert (work / "HEXU_v.gro").exists()


def test_convert_needs_a_complete_one_to_one_mapping(tool, tmp_path):
    pc = tmp_path / "x.pc_resp"
    pc.write_text("3\n# comment\nQ 0 0 0 0.5\nQ 1 0 0 -0.25\nQ 0 1 0 -0.25\n")
    good = tmp_path / "map.csv"
    good.write_text("dft_index,atom_name\n1,A\n2,B\n3,C\n")
    out = tmp_path / "rel.csv"
    assert tool.convert_resp(str(pc), str(good), "MOL", str(out)) == 3
    rows = list(csv.DictReader(open(out)))
    assert [r["atom_name"] for r in rows] == ["A", "B", "C"]
    assert sum(Decimal(r["charge"]) for r in rows) == 0

    bad = tmp_path / "bad.csv"
    bad.write_text("dft_index,atom_name\n1,A\n2,B\n")
    with pytest.raises(ValueError, match="exactly once"):
        tool.convert_resp(str(pc), str(bad), "MOL", str(out))
    dup = tmp_path / "dup.csv"
    dup.write_text("dft_index,atom_name\n1,A\n2,A\n3,C\n")
    with pytest.raises(ValueError, match="same atom name"):
        tool.convert_resp(str(pc), str(dup), "MOL", str(out))
