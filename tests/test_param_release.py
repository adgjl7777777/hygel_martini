"""A frozen release notices when the force-field files drift."""

from __future__ import annotations

import importlib.util
import os
import shutil

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "example", "08_des_thiourethane_aa")
SCRIPT = os.path.join(EXAMPLE, "parameterization", "release.py")
PROJECT = os.path.join(EXAMPLE, "project")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(PROJECT, "structure", "HEXU.itp")),
    reason="example 08 templates are not present",
)


@pytest.fixture(scope="module")
def rel():
    spec = importlib.util.spec_from_file_location("release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project_copy(tmp_path):
    for sub in ("structure", "config"):
        os.makedirs(tmp_path / sub)
    for relpath in ("structure/forcefield.itp", "structure/HEXU.itp", "structure/STR.itp",
                    "structure/ACC.itp", "structure/CL.itp", "config/hydrogel.yaml"):
        shutil.copy(os.path.join(PROJECT, relpath), tmp_path / relpath)
    return tmp_path


def test_freeze_verify_and_drift(rel, project_copy, tmp_path):
    releases = tmp_path / "releases"
    rel.freeze("v_test", "draft", ["note"], project=str(project_copy), releases=str(releases))
    result = rel.verify("v_test", project=str(project_copy), releases=str(releases))
    assert set(result.values()) == {"ok"}
    assert rel.current(project=str(project_copy), releases=str(releases)) == ["v_test"]

    with open(project_copy / "structure" / "STR.itp", "a") as handle:
        handle.write("; drift\n")
    result = rel.verify("v_test", project=str(project_copy), releases=str(releases))
    assert result["structure/STR.itp"] == "changed"
    assert rel.current(project=str(project_copy), releases=str(releases)) == []

    os.remove(project_copy / "structure" / "CL.itp")
    result = rel.verify("v_test", project=str(project_copy), releases=str(releases))
    assert result["structure/CL.itp"] == "missing"


def test_a_release_is_immutable_unless_forced(rel, project_copy, tmp_path):
    releases = tmp_path / "releases"
    rel.freeze("v_test", "draft", [], project=str(project_copy), releases=str(releases))
    with pytest.raises(FileExistsError):
        rel.freeze("v_test", "draft", [], project=str(project_copy), releases=str(releases))
    rel.freeze("v_test", "candidate", [], project=str(project_copy), releases=str(releases), force=True)
    with pytest.raises(ValueError):
        rel.freeze("v_bad", "approved", [], project=str(project_copy), releases=str(releases))


def test_the_shipped_tree_matches_a_frozen_release(rel):
    """Whatever is committed must be a release someone can name."""
    names = rel.current()
    assert names, "the committed force field matches no frozen release; run release.py freeze"
