import copy
import pytest
from test_builder_contract_failures import _valid_planned_fixture
from hygel_martini.hydrogel_builder.core_utils.runtime.planner_witness import capture_planner_input,guard_written_planner_input
from hygel_martini.hydrogel_builder.core_utils.templates.template_contract import require_supported_template

def write_itp(path,witness,swap=False,wrong=False):
 rows=witness['linkers'];ids=set(witness['endpoint_atom_map'].values())|{s for r in rows for s in r['stubs']}
 lines=['[ atoms ]']+[f'{i} T 1 R A 1 0 1' for i in sorted(ids)]+['[ bonds ]']
 pairs=[]
 for r in rows:
  ss=r['stubs'][::-1] if swap else r['stubs']
  for s,p in zip(ss,r['logical_pairs']):
   pairs.extend((s,witness['endpoint_atom_map'][k]) for k in p)
 if wrong:
  a,b=pairs[0];c,d=pairs[-1];pairs[0]=(a,d);pairs[-1]=(c,b)
 lines += [f'{a} {b} 1' for a,b in pairs];path.write_text('\n'.join(lines)+'\n')

@pytest.mark.parametrize('swap',[False,True])
def test_original_plan_accepts_normal_and_equivalent_stub_swap(tmp_path,swap):
 ends,stubs=_valid_planned_fixture();w=capture_planner_input({7:stubs},ends)
 path=tmp_path/'run.itp';write_itp(path,w,swap=swap)
 assert guard_written_planner_input(path,w,tmp_path/'audit.json')['status']=='PASS'

def test_self_consistent_wrong_router_output_fails_against_original_plan(tmp_path):
 ends,stubs=_valid_planned_fixture();w=capture_planner_input({7:stubs},ends)
 path=tmp_path/'wrong.itp';write_itp(path,w,wrong=True)
 # An assignment-derived witness could agree with this output; the pre-router plan does not.
 with pytest.raises(RuntimeError,match='differs from pre-router'):
  guard_written_planner_input(path,w,tmp_path/'audit.json')

def test_duplicate_endpoint_mapping_rejected():
 ends,stubs=_valid_planned_fixture()
 atoms=[a for group in ends.values() for a in group];atoms[1].atom_id=atoms[0].atom_id
 with pytest.raises(ValueError,match='Non-injective'):capture_planner_input({7:stubs},ends)

@pytest.mark.parametrize('definition,pattern',[
 ({'name':'DMAPS','nrexcl':2},'nrexcl'),
 ({'bonds':[{'from':1,'to':2},{'from':2,'to':3}],'angles':[{'from':1,'center':2,'to':4}]},'non-path angle'),
 ({'exclusions':[{'atom':1,'exclude':[3]}]},'per-instance exclusions')])
def test_unsupported_template_is_rejected_instead_of_silent_hamiltonian_change(definition,pattern):
 with pytest.raises(ValueError,match=pattern):require_supported_template(definition)

def test_backbone_angle_and_simple_supported_template():
 d={'nrexcl':1,'bonds':[{'from':1,'to':2},{'from':2,'to':3}], 'angles':[{'from':1,'center':2,'to':3}]}
 require_supported_template(d)
 with pytest.raises(ValueError,match='replaced backbone'):require_supported_template(d,[1])

def test_explicit_two_bead_junction_convention_does_not_relax_long_template_gate():
 d={'name':'LNK','nrexcl':2,'beads':[{'nr':1},{'nr':2}], 'bonds':[{'from':1,'to':2}]}
 with pytest.raises(ValueError,match='nrexcl'):require_supported_template(d)
 require_supported_template(d,exclusion_policy='network_nrexcl_one')
 d['beads'].append({'nr':3})
 with pytest.raises(ValueError,match='nrexcl'):require_supported_template(d,exclusion_policy='network_nrexcl_one')

def test_pipeline_rejects_loss_of_captured_original_witness(tmp_path, monkeypatch):
 from hygel_martini.hydrogel_builder.config_params.config import Config
 from hygel_martini.hydrogel_builder.config_params.read_json import _guard_written_crosslinks
 monkeypatch.setattr(Config,'_data',{'simulation_parameters':{'require_explicit_crosslink_plan':True}})
 monkeypatch.setattr(Config,'_runtime_state',{'planner_input_required':True, 'planner_input_contract':None})
 with pytest.raises(RuntimeError,match='Missing pre-router'):
  _guard_written_crosslinks(tmp_path/'missing.itp',str(tmp_path))

@pytest.mark.parametrize('section',['dihedrals','impropers'])
def test_template_torsions_without_instance_mapping_are_rejected(section):
 d={'nrexcl':1,section:[{'i':1,'j':2,'k':3,'l':4,'funct':1,'params':[0,1]}]}
 with pytest.raises(ValueError,match='per-instance'):
  require_supported_template(d)
