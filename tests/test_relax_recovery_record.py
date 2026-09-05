"""The shrink's history says what its NVT recovery ran with."""

from __future__ import annotations

from pathlib import Path

from hygel_martini.hydrogel_builder.relax.hard_em_shrink import _mdp_values


def test_recovery_settings_are_read_from_the_mdp(tmp_path: Path) -> None:
    mdp = tmp_path / "nvt.mdp"
    mdp.write_text("integrator = md\ndt = 0.001 ; ps\nnsteps = 20000\n"
                   "constraints = h-bonds\nref-t = 300\npcoupl = no\n")
    got = _mdp_values(mdp)
    assert got == {"dt": 0.001, "nsteps": 20000, "constraints": "h-bonds", "ref_t": 300}


def test_a_missing_mdp_gives_an_empty_record_not_an_error(tmp_path: Path) -> None:
    assert _mdp_values(tmp_path / "nope.mdp") == {}
