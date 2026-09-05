"""The plateau screen fails closed on records that cannot support a verdict.

These are the independent review's three probes (2026-09-05, Major 2) plus a
stationary control: only a complete, finite, sufficiently long record with a
flat tail may return PLATEAU, and the two-window drift test must both pass.
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "example", "08_des_thiourethane_aa", "project", "config_npt",
                      "check_convergence.py")

pytestmark = pytest.mark.skipif(not os.path.isfile(SCRIPT), reason="example 08 not present")


@pytest.fixture
def conv(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("check_convergence", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    edr = tmp_path / "npt.edr"
    edr.write_bytes(b"not a real edr")
    mod._edr = str(edr)
    return mod


def _series(mod, terms, n=100, values=None):
    times = [float(i) for i in range(n)]
    out = {"_grids": {}}
    for t in terms:
        out[t] = list(values) if values is not None else [300.0] * n
        out["_grids"][t] = times
    return out


def test_missing_required_terms_are_exit_2(conv, monkeypatch):
    monkeypatch.setattr(conv, "energy_series", lambda *a, **k: _series(conv, ["Temperature"]))
    assert conv.main([conv._edr]) == 2


def test_skip_past_the_record_is_exit_2(conv, monkeypatch):
    monkeypatch.setattr(conv, "energy_series", lambda *a, **k: _series(conv, conv.REQUIRED))
    assert conv.main([conv._edr, "--skip-ps", "1000"]) == 2


def test_nan_is_exit_2(conv, monkeypatch):
    vals = [300.0] * 99 + [float("nan")]
    monkeypatch.setattr(conv, "energy_series",
                        lambda *a, **k: _series(conv, conv.REQUIRED, values=vals))
    assert conv.main([conv._edr]) == 2


def test_strong_earlier_drift_with_flat_last_two_blocks_is_not_a_plateau(conv, monkeypatch):
    vals = [100.0] * 20 + [200.0] * 20 + [300.0] * 20 + [400.0] * 40
    monkeypatch.setattr(conv, "energy_series",
                        lambda *a, **k: _series(conv, conv.REQUIRED, values=vals))
    assert conv.main([conv._edr]) == 1


def test_a_stationary_complete_record_passes_and_writes_a_bound_artifact(conv, monkeypatch):
    monkeypatch.setattr(conv, "energy_series", lambda *a, **k: _series(conv, conv.REQUIRED))
    assert conv.main([conv._edr]) == 0
    art = json.load(open(conv._edr + ".plateau.json"))
    assert art["verdict"] == "PLATEAU"
    assert len(art["edr_sha256"]) == 64
    assert "RDF/CN" in art["not_assessed_here"]
