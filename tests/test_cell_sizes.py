"""The size planner is the same model as the builds it plans.

The check that matters is not internal consistency but agreement with what
was actually built and shrunk: the planner must reproduce the atom counts of
the committed builds and the shrink targets those builds were compressed to.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "example", "08_des_thiourethane_aa")
SIZING = os.path.join(EXAMPLE, "sizing")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(EXAMPLE, "project", "structure", "STR_n33.itp")),
    reason="example 08 templates (incl. tiled strand) are not present",
)


@pytest.fixture(scope="module")
def cs():
    sys.path.insert(0, SIZING)
    spec = importlib.util.spec_from_file_location("cell_sizes", os.path.join(SIZING, "cell_sizes.py"))
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules[__module__];
    # a module loaded from a path is not registered there by default.
    sys.modules["cell_sizes"] = module
    spec.loader.exec_module(module)
    return module


def _plan(cs, strand, repeats, conversion="full", count=None, des=False):
    strands = cs.discover_strands()
    s = strands[strand]
    j = repeats[0] * repeats[1] * repeats[2]
    extras = {"ACC": 6 * j, "CL": 6 * j} if des else {}
    types = cs._load_types(s, list(extras))
    return cs.plan_cell(repeats, s, types, conversion=conversion, conversion_count=count, extras=extras)


@pytest.mark.parametrize("strand,des,atoms,target", [
    ("n3", False, 19584, 6.34),    # maker.yaml / maker_shrink.yaml
    ("n33", False, 77184, 9.32),   # maker_n33.yaml / maker_shrink_n33.yaml
    ("n3", True, 29952, 7.19),     # maker_des.yaml / maker_shrink_des.yaml
])
def test_reproduces_the_committed_builds_and_shrink_targets(cs, strand, des, atoms, target):
    plan = _plan(cs, strand, (4, 4, 4), des=des)
    assert plan.atom_count == atoms
    assert plan.target_box[0] == pytest.approx(target, abs=0.011)


def test_exact_count_cell_matches_the_verified_build(cs):
    plan = _plan(cs, "n3", (4, 4, 4), conversion="count", count=32)
    assert plan.atom_count == 8224          # measured on the built topology
    assert plan.formed_strands == 32
    assert plan.free_thiols == 320
    assert plan.conversion == pytest.approx(1 / 6)


def test_pcu_size_rules_are_enforced(cs):
    with pytest.raises(ValueError, match="odd"):
        cs.validate_repeats((4, 4, 5))
    with pytest.raises(ValueError, match="below 4"):
        cs.validate_repeats((2, 4, 4))
    assert cs.validate_repeats((4, 4, 6)) == (4, 4, 6)


def test_anisotropic_target_keeps_the_proportions(cs):
    plan = _plan(cs, "n3", (4, 4, 6))
    x, y, z = plan.target_box
    assert x == pytest.approx(y)
    assert z / x == pytest.approx(1.5)


def test_profiles_scale_the_count_with_the_supercell(cs):
    class A:  # a minimal argparse stand-in
        profile = "target_molar"; strand = "n33"; repeats = None; des = False; conversion = "full"
    a = A()
    cs._apply_profile(a)
    assert a.repeats == [4, 4, 4] and a.conversion == "count:32" and a.des is True
    b = A(); b.repeats = [6, 6, 6]
    cs._apply_profile(b)
    assert b.conversion == "count:108"   # 0.5 per junction * 216
    c = A(); c.profile = "target_equiv"
    cs._apply_profile(c)
    assert c.conversion == "count:96"
