"""Regression and pipeline-stop checks for opt-in explicit-plan construction."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_builder_contract_failures import _valid_planned_fixture, _write_topology_fixture
from hygel_martini.hydrogel_builder.config_params.config import Config
from hygel_martini.hydrogel_builder.core_utils.runtime.dynamic_crosslink import plan_dynamic_crosslinks
from hygel_martini.hydrogel_builder.core_utils.runtime.persisted_plan import guard_persisted_plan

PAIRS = [[1, 5], [3, 6], [2, 7], [4, 8]]


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
    with pytest.raises(ValueError, match='YAML boolean'):
        _enabled_formulation_stages({'add_water': {'enabled': 'false'}})


def test_total_metadata_loss_is_rejected_only_in_required_mode():
    ends, stubs = _valid_planned_fixture()
    for stub in stubs:
        stub.planned_endpoint_edges = None
    # Same geometric problem remains supported for explicitly legacy callers.
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


def test_written_cycle_deletion_fails_even_though_graph_is_still_connected(tmp_path):
    from hygel_martini.tools.audit_hydrogel_topology import parse_itp, connected_components
    valid, _ = _write_topology_fixture(tmp_path, False)
    mutated, _ = _write_topology_fixture(tmp_path, True)
    assert guard_persisted_plan(valid, PAIRS, tmp_path/'valid.json')['status'] == 'PASS'
    atoms, bonds, _, _ = parse_itp(mutated)
    assert len(connected_components(atoms, bonds)[0]) == 1
    with pytest.raises(RuntimeError, match="Written crosslink plan mismatch"):
        guard_persisted_plan(mutated, PAIRS, tmp_path/'failed.json')
    report = json.loads((tmp_path/'failed.json').read_text())
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
        guard_persisted_plan(itp, pairs, tmp_path/'report.json')
    assert json.loads((tmp_path/'report.json').read_text())['status'] == 'FAIL'


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
    assert Config.get_runtime('expected_crosslink_pairs') is None


@pytest.mark.parametrize('stage', ['backbone', 'hydrogel'])
def test_all_mode_stops_at_corrupted_write_before_next_external_step(tmp_path, monkeypatch, stage):
    """Exercise actual stage ordering with fixture writers and external-call spies.

    This is an integration fault injection, not a full molecular build. The
    separate wheel example exercises the real builder and GROMACS.
    """
    from hygel_martini.hydrogel_builder.config_params import read_json as workflow, build_hydrogel
    valid, gro = _write_topology_fixture(tmp_path, False)
    invalid, _ = _write_topology_fixture(tmp_path, True)
    monkeypatch.setattr(Config, '_data', {'simulation_parameters': {
        'output_dir': str(tmp_path), 'require_explicit_crosslink_plan': True}})
    monkeypatch.setattr(Config, '_runtime_state', {'final_itp_files': []})
    world = SimpleNamespace()
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
    def geometry(*args, **kwargs):
        calls.append(args[0])
        return None
    monkeypatch.setattr(workflow, '_perform_geo_opt_step', geometry)
    with pytest.raises(RuntimeError, match="Written crosslink plan mismatch"):
        workflow._execute_all_mode()
    assert calls == ([] if stage == 'backbone' else ['backbone_stage'])
