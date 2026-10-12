"""Side-chain placement, bonded-patch and Packmol PBC guards added in 0.1.2."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hygel_martini.hydrogel_builder.core_utils.runtime import packer
from hygel_martini.hydrogel_builder.main_components import Attributes
from hygel_martini.hydrogel_builder.main_components import Hydrogel as hydrogel_module
from hygel_martini.hydrogel_builder.main_components.Hydrogel import Hydrogel
from hygel_martini.hydrogel_builder.main_components.Polymer import Polymer
from hygel_martini.hydrogel_builder.main_components.Universe import World


def _bead():
    return SimpleNamespace(
        residue_name="SID", atom_type="SC1", name="S1", mass=72.0, charge=0.0,
        residue_number=1, cgnr=1, original_index=2,
    )


def _template():
    return SimpleNamespace(
        id="SIDE", coords=[[0.0, 0.0, 0.0]], beads=[_bead()],
        internal_bonds=[], backbone_bonds=[(0, {})],
        constraints=[], pairs=[], exclusions=[], virtual_sites={},
        cmaptypes=[], polarization=[], other_sections={},
    )


def _atom(position, chain_type="backbone", backbone_type="BB"):
    atom = Attributes.Atom()
    atom.position = np.array(position, dtype=float)
    atom.chain_type = chain_type
    atom.backbone_type = backbone_type
    return atom


@pytest.fixture
def world():
    World.reset()
    World.box_length = 20.0
    World.box_vector = np.array([20.0, 20.0, 20.0])
    World.mean_sep = 0.24
    yield World
    World.reset()


def _patch_hydrogel_inputs(monkeypatch, template, place):
    record = SimpleNamespace(template=template, ratio=1)
    library = SimpleNamespace(records=[record], by_backbone={"BB": [record]}, lookup={template.id: template})
    params = {
        ("simulation_parameters",): {},
        ("monomer_definitions",): {"MONOMERS": ["SIDE"], "SEQUENCE_STRATEGY": {"strategy": "random", "seed": 1}},
        ("hydrogel_components", "backbone_definitions", "BACKBONES"): [{"id": "BB"}],
    }
    monkeypatch.setattr(hydrogel_module.p.Config, "get_param", classmethod(lambda cls, *keys, **kw: params[keys]))
    monkeypatch.setattr(hydrogel_module, "load_monomer_templates", lambda *args: library)
    monkeypatch.setattr(hydrogel_module, "compute_template_positions", place)


def test_linker_stub_receives_no_side_chain(world, monkeypatch):
    chain = [_atom((5.0 + i, 5.0, 5.0)) for i in range(3)]
    stub = _atom((8.0, 5.0, 5.0), chain_type="linker")
    for a, b in zip(chain, chain[1:] + [stub]):
        Attributes.Bond(a.atom_id, b.atom_id)
    _patch_hydrogel_inputs(monkeypatch, _template(), lambda coords, origin, vec, tangent: [origin + 0.5 * vec])

    Hydrogel.__new__(Hydrogel).construct_chemical_detail()

    side_atoms = [World.Atoms[k][0] for k in World.Atoms if World.Atoms[k][0].atom_type == "SC1"]
    assert len(side_atoms) == 3
    assert stub.number_of_bonds == 1


def test_failed_side_chain_placement_stops_construction(world, monkeypatch):
    a, b = _atom((5.0, 5.0, 5.0)), _atom((6.0, 5.0, 5.0))
    Attributes.Bond(a.atom_id, b.atom_id)
    blocker = _atom((5.0, 7.0, 5.0), backbone_type=None)
    _patch_hydrogel_inputs(monkeypatch, _template(), lambda *args: [blocker.position.copy()])

    with pytest.raises(RuntimeError, match="No clash-free placement"):
        Hydrogel.__new__(Hydrogel).construct_chemical_detail()


def test_polymer_side_chains_are_checked_against_each_other(world, monkeypatch):
    a, b = _atom((5.0, 5.0, 5.0)), _atom((6.0, 5.0, 5.0))
    Attributes.Bond(a.atom_id, b.atom_id)
    template = _template()
    iterator = SimpleNamespace(next=lambda: template)
    monkeypatch.setattr(Polymer, "_sidechain_library", object())
    monkeypatch.setattr(Polymer, "_sidechain_iterators", {"BB": iterator})
    shared_site = np.array([5.5, 6.0, 5.0])
    monkeypatch.setattr(Polymer, "_compute_template_positions", lambda self, *args: [shared_site.copy()])

    with pytest.raises(RuntimeError, match="No clash-free placement"):
        Polymer.__new__(Polymer)._construct_sidechains_from_templates()


def _packmol_input(tmp_path, monkeypatch, base_xyz_nm, **extra):
    def fake_gro_to_pdb(gro, pdb, gmx):
        x, y, z = (10.0 * v for v in base_xyz_nm)
        with open(pdb, "w", encoding="utf-8") as handle:
            handle.write(f"ATOM      1  B1  BAS     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
        return pdb

    monkeypatch.setattr(packer, "convert_gro_to_pdb", fake_gro_to_pdb)
    monkeypatch.setattr(packer, "run_packmol", lambda *args, **kwargs: None)
    sim = {"output_dir": str(tmp_path), "packmol_path": "packmol", "packmol_threshold": 2.0, "test_mode": True, **extra}
    (tmp_path / "base.gro").write_text("unused\n")
    packer.pack_system_with_molecules(
        "Add_Water", str(tmp_path / "base.gro"), [{"file": str(tmp_path / "W.gro"), "number": 3}],
        str(tmp_path / "out.gro"), [3.0, 3.0, 3.0], sim,
    )
    return (tmp_path / "Add_Water_packmol.inp").read_text().splitlines()


def test_packmol_pbc_is_opt_in(tmp_path, monkeypatch):
    default = _packmol_input(tmp_path, monkeypatch, (1.0, 1.0, 1.0))
    assert not any(line.startswith("pbc") for line in default)
    periodic = _packmol_input(tmp_path, monkeypatch, (1.0, 1.0, 1.0), packmol_pbc=True)
    assert periodic[3] == "pbc 30.0000 30.0000 30.0000"
    assert [line for line in periodic if not line.startswith("pbc")] == default


def test_packmol_pbc_rejects_base_outside_box(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="outside"):
        _packmol_input(tmp_path, monkeypatch, (1.0, 3.2, 1.0), packmol_pbc=True)


def test_missing_configured_bonded_patch_file_is_an_error(tmp_path):
    from hygel_martini.hydrogel_builder.config_params.read_json import _get_bonded_topology_patch_path

    with pytest.raises(FileNotFoundError, match="bonded_topology_patch_file"):
        _get_bonded_topology_patch_path({"bonded_topology_patch_file": str(tmp_path / "missing.yaml")})
    present = tmp_path / "present.yaml"
    present.write_text("bonds: []\n")
    assert _get_bonded_topology_patch_path({"bonded_topology_patch_file": str(present)}) == str(present)
