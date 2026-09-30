"""Opt-in explicit-plan construction: refuse silent fallbacks, verify the file.

Three behaviours ported from the Series-01 reliability copy (0.1.1.dev0,
commit 50192af), adapted to omni:

* ``require_explicit_crosslink_plan: true`` turns total loss of planner
  metadata from a silent distance-based assignment into an error.
* Each written HYDROGEL ITP is re-read and its stub-to-endpoint bonds are
  compared with the plan, so a rewired or deleted attachment stops the build
  before the next external step even when the graph stays connected.
* ``enabled: false`` on a packing stage actually disables it; a non-boolean
  is an error. omni's list-valued ``add_molecule`` passes through untouched.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_builder_contract_failures import _valid_planned_fixture, _write_topology_fixture
from hygel_martini.hydrogel_builder.config_params.config import Config
from hygel_martini.hydrogel_builder.core_utils.runtime.dynamic_crosslink import plan_dynamic_crosslinks
from hygel_martini.hydrogel_builder.core_utils.runtime.persisted_plan import guard_persisted_plan

PAIRS = [[1, 5], [3, 6], [2, 7], [4, 8]]


# --- stage switches ---------------------------------------------------------
def test_disabled_formulation_stages_override_positive_counts_and_legacy_remains():
    from hygel_martini.hydrogel_builder.config_params.read_json import _enabled_formulation_stages
    disabled = {
        'add_water': {'enabled': False, 'number_of_water': 10000},
        'add_small_ion': {'enabled': False, 'ions': [{'number': 10}]},
        'add_molecule': {'enabled': False, 'num_molecules': 5},
        'add_polymer': {'enabled': False, 'num_polymers': 5},
    }
    assert _enabled_formulation_stages(disabled) == {}
    legacy = {'add_water': {'number_of_water': 5}}
    assert _enabled_formulation_stages(legacy) == legacy
    # an explicit `enabled: true` is consumed, not handed to the stage code
    assert _enabled_formulation_stages({'add_water': {'enabled': True, 'number_of_water': 5}}) \
        == {'add_water': {'number_of_water': 5}}
    with pytest.raises(ValueError, match='YAML boolean'):
        _enabled_formulation_stages({'add_water': {'enabled': 'false'}})


def test_list_valued_add_molecule_has_no_switch_and_passes_through():
    """omni's add_molecule may be a list of species (one Packmol pass)."""
    from hygel_martini.hydrogel_builder.config_params.read_json import _enabled_formulation_stages
    stages = {'add_molecule': [{'molecule_gro': 'a.gro'}, {'molecule_gro': 'b.gro'}],
              'add_water': {'enabled': False}}
    assert _enabled_formulation_stages(stages) == {'add_molecule': stages['add_molecule']}


# --- explicit plan requirement --------------------------------------------
def test_total_metadata_loss_is_rejected_only_in_required_mode():
    ends, stubs = _valid_planned_fixture()
    for stub in stubs:
        stub.planned_endpoint_edges = None
    # The same geometric problem stays supported for callers that did not opt in.
    assignments, _ = plan_dynamic_crosslinks({7: stubs}, ends, None, targets_per_stub=2)
    assert len(assignments[7]) == 4
    with pytest.raises(ValueError, match="Explicit crosslink plan required"):
        plan_dynamic_crosslinks({7: stubs}, ends, None, targets_per_stub=2, require_explicit_plan=True)


@pytest.mark.parametrize("linkers,ends", [({}, {}), ({7: []}, {})])
def test_empty_required_plan_is_rejected(linkers, ends):
    with pytest.raises(ValueError, match="Explicit crosslink plan required"):
        plan_dynamic_crosslinks(linkers, ends, None, require_explicit_plan=True)


def test_string_boolean_is_not_silently_accepted():
    with pytest.raises(ValueError, match="must be a boolean"):
        plan_dynamic_crosslinks({}, {}, None, require_explicit_plan="false")


def test_endpoint_identity_survives_reordering_translation_and_periodic_images():
    ends, stubs = _valid_planned_fixture()
    for i, atoms in ends.items():
        atom = atoms[0]
        atom.atom_id = 100 - i
        x, y, z = atom.position
        atom.position = (x + 13 + (i % 2) * 20, y + 4, z + 3)
    for stub in stubs:
        x, y, z = stub.position
        stub.position = (x + 13, y + 4, z + 3)
    assignments, _ = plan_dynamic_crosslinks(
        {7: stubs[::-1]}, dict(reversed(list(ends.items()))), (20, 20, 20),
        targets_per_stub=2, require_explicit_plan=True,
    )
    groups = {}
    for item in assignments[7]:
        groups.setdefault(item.stub_atom.atom_id, set()).add(item.backbone_atom.planned_endpoint_id)
    assert {frozenset(v) for v in groups.values()} == {frozenset('ab'), frozenset('cd')}


# --- the written-file guard -----------------------------------------------
def test_written_cycle_deletion_fails_even_though_graph_is_still_connected(tmp_path):
    """Why a connectivity audit is not enough: one ring bond gone, still one component."""
    from hygel_martini.tools.audit_hydrogel_topology import parse_itp, connected_components
    valid, _ = _write_topology_fixture(tmp_path, False)
    mutated, _ = _write_topology_fixture(tmp_path, True)
    assert guard_persisted_plan(valid, PAIRS, tmp_path / 'valid.json')['status'] == 'PASS'
    atoms, bonds, _, _ = parse_itp(mutated)
    assert len(connected_components(atoms, bonds)[0]) == 1
    with pytest.raises(RuntimeError, match="Written crosslink plan mismatch"):
        guard_persisted_plan(mutated, PAIRS, tmp_path / 'failed.json')
    report = json.loads((tmp_path / 'failed.json').read_text())
    assert report['status'] == 'FAIL' and report['missing'] == [[4, 8]]


@pytest.mark.parametrize("corruption", ['duplicate', 'rewire', 'missing_atom', 'missing_file', 'missing_plan'])
def test_persisted_guard_rejects_corruption(tmp_path, corruption):
    itp, _ = _write_topology_fixture(tmp_path, False)
    text = itp.read_text()
    pairs = PAIRS
    if corruption == 'duplicate':
        itp.write_text(text + '1 5 1\n')
    elif corruption == 'rewire':
        itp.write_text(text.replace('1 5 1', '1 6 1'))
    elif corruption == 'missing_atom':
        itp.write_text(text.replace('5 T 3 PEO E1 3 0 1\n', ''))
    elif corruption == 'missing_file':
        itp.unlink()
    else:
        pairs = None
    with pytest.raises(RuntimeError):
        guard_persisted_plan(itp, pairs, tmp_path / 'report.json')
    assert json.loads((tmp_path / 'report.json').read_text())['status'] == 'FAIL'


def test_runtime_reads_required_plan_setting_and_clears_old_witness(tmp_path, monkeypatch):
    from hygel_martini.hydrogel_builder.config_params import read_json as workflow
    ends, stubs = _valid_planned_fixture()
    for stub in stubs:
        stub.planned_endpoint_edges = None
    atoms = [a for group in ends.values() for a in group] + stubs
    monkeypatch.setattr(workflow.World, 'Atoms', {a.atom_id: [a] for a in atoms})
    monkeypatch.setattr(Config, '_data', {'simulation_parameters': {
        'require_explicit_crosslink_plan': True, 'dynamic_crosslink_targets_per_stub': 2,
        'pbc_true_or_false': False}})
    monkeypatch.setattr(Config, '_runtime_state', {'expected_crosslink_pairs': PAIRS})
    with pytest.raises(ValueError, match="Explicit crosslink plan required"):
        workflow._perform_dynamic_crosslinking(str(tmp_path))
    # a stale witness from a previous build must not survive into this one
    assert Config.get_runtime('expected_crosslink_pairs') is None


def test_guard_is_inert_unless_opted_in(tmp_path, monkeypatch):
    from hygel_martini.hydrogel_builder.config_params import read_json as workflow
    monkeypatch.setattr(Config, '_data', {'simulation_parameters': {}})
    monkeypatch.setattr(Config, '_runtime_state', {})
    assert workflow._guard_written_crosslinks(str(tmp_path / 'absent.itp'), str(tmp_path)) is None
    assert not (tmp_path / 'absent.itp.plan_audit.json').exists()


@pytest.mark.parametrize('stage', ['backbone', 'hydrogel'])
def test_all_mode_stops_at_corrupted_write_before_next_external_step(tmp_path, monkeypatch, stage):
    """Fault injection on the real stage ordering of ``_execute_all_mode``.

    The writers are replaced by fixtures and the external steps by spies; this
    is not a molecular build. It pins two things the unit tests above cannot:
    the guard runs immediately after each ITP write, and a failure at the
    backbone write happens before any energy minimisation is launched, while
    a failure at the hydrogel write happens after exactly the backbone EM.
    """
    from hygel_martini.hydrogel_builder.config_params import read_json as workflow, build_hydrogel
    valid, gro = _write_topology_fixture(tmp_path, False)
    invalid, _ = _write_topology_fixture(tmp_path, True)
    monkeypatch.setattr(Config, '_data', {
        'simulation_parameters': {'output_dir': str(tmp_path), 'require_explicit_crosslink_plan': True},
        # omni's all-mode reads the component definitions between the two
        # writes to decide whether strand templates are in play; empty is fine.
        'hydrogel_components': {'backbone_definitions': {'BACKBONES': {}},
                                'linker_definitions': {'LINKERS': {}}}})
    monkeypatch.setattr(Config, '_runtime_state', {'final_itp_files': []})
    world = SimpleNamespace(OtherSections={}, Constraints=[], Exclusions=[], Dihedrals=[])
    monkeypatch.setattr(build_hydrogel, 'build_backbone_only', lambda: (world, object()))
    monkeypatch.setattr(build_hydrogel, 'finalize_hydrogel', lambda w, obj: w)
    monkeypatch.setattr(workflow, '_get_bonded_topology_patch_path', lambda cfg: None)
    monkeypatch.setattr(workflow, '_perform_dynamic_crosslinking',
                        lambda output: Config.set_runtime('expected_crosslink_pairs', PAIRS))
    monkeypatch.setattr(workflow, 'write_to_gro', lambda w, filename: Path(filename).write_text(gro.read_text()))

    def writer(w, filename, **kwargs):
        corrupt = ('backbone' in filename) == (stage == 'backbone')
        Path(filename).write_text((invalid if corrupt else valid).read_text())
    monkeypatch.setattr(workflow, 'write_combined_itp', writer)
    from hygel_martini.hydrogel_builder.core_utils.io import writer as writer_module
    monkeypatch.setattr(writer_module, 'write_combined_itp', writer)
    calls = []
    monkeypatch.setattr(workflow, '_perform_geo_opt_step', lambda *a, **k: calls.append(a[0]))

    with pytest.raises(RuntimeError, match="Written crosslink plan mismatch"):
        workflow._execute_all_mode()
    assert calls == ([] if stage == 'backbone' else ['backbone_stage'])
