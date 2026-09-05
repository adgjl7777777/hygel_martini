"""The composition audit judges the cell; it must not crash on a name mismatch.

The tiled strand lives in ``STR_n33.itp`` but declares moleculetype ``STR33``.
The first pilot's audit keyed the network components by moleculetype name and
died with KeyError instead of reporting -- which the wrapper, correctly,
treated as a failed audit and refused to shrink. Recipe keys are file stems.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "example", "08_des_thiourethane_aa")
SIZING = os.path.join(EXAMPLE, "sizing")
PROJECT = os.path.join(EXAMPLE, "project")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(PROJECT, "structure", "STR_n33.itp")),
    reason="example 08 templates are not present",
)


@pytest.fixture(scope="module")
def audit():
    sys.path.insert(0, SIZING)
    spec = importlib.util.spec_from_file_location("composition_audit",
                                                  os.path.join(SIZING, "composition_audit.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["composition_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_expected_network_uses_file_stems_not_moleculetype_names(audit):
    recipe = yaml.safe_load(open(os.path.join(PROJECT, "recipes", "target_molar_r4.yaml")))
    exp = audit.expected_network(recipe, os.path.join(PROJECT, "structure"),
                                 os.path.join(PROJECT, "config", "hydrogel.yaml"))
    # 64 x 93 + 32 x 373 - 64 cap hydrogens
    assert exp.atom_count == 64 * 93 + 32 * 373 - 64 == 17824


def test_expected_network_matches_the_diagnostic_recipe(audit):
    recipe = yaml.safe_load(open(os.path.join(PROJECT, "recipes", "diagnostic_n3_r4_c32.yaml")))
    exp = audit.expected_network(recipe, os.path.join(PROJECT, "structure"),
                                 os.path.join(PROJECT, "config", "hydrogel.yaml"))
    assert exp.atom_count == 8224
