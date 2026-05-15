"""Top-level workflow orchestration for hydrogel generation.

Historically this file grew into the central coordinator for the full pipeline.
It now contains the stage ordering logic that glues together:

1. template/config validation,
2. backbone construction,
3. staged geometry optimization,
4. optional add-series packing steps, and
5. final topology bookkeeping.

The lower-level geometry and IO details live elsewhere; this module should be
read as the execution plan for a complete run.
"""

import glob
import os
import random
import shutil
import sys

import numpy as np

from hygel_martini.hydrogel_builder.add_series import add_small_ion
from hygel_martini.hydrogel_builder.core_utils.io.martini_parser import read_atom_types, read_itp_definitions
from hygel_martini.hydrogel_builder.core_utils.io.writer import write_to_gro, write_combined_itp
from hygel_martini.hydrogel_builder.core_utils.runtime import packer, topology_updater
from hygel_martini.hydrogel_builder.core_utils.runtime.backbone_patcher import patch_backbone_topology
from hygel_martini.hydrogel_builder.core_utils.runtime.dynamic_crosslink import (
    collect_backbone_ends,
    group_linker_stubs,
    plan_dynamic_crosslinks,
)
from hygel_martini.hydrogel_builder.core_utils.runtime.geo_opt import run_geo_opt
from hygel_martini.hydrogel_builder.core_utils.templates.linker_loader import load_linker_templates
from hygel_martini.hydrogel_builder.core_utils.templates.monomer_loader import load_monomer_templates
from hygel_martini.hydrogel_builder.main_components.Universe import World
from hygel_martini.hydrogel_builder.main_components.Attributes import Bond


def _seed_all(sim_params):
    """Seed all RNGs used by the orchestration layer."""
    seed = sim_params.get("random_seed")
    if seed is None:
        return
    try:
        seed_val = int(seed)
    except Exception:
        return
    random.seed(seed_val)
    np.random.seed(seed_val)
    try:
        Config.debug_log(f"[seed] random_seed={seed_val}")
    except Exception:
        pass

from datetime import datetime
from hygel_martini.hydrogel_builder.config_params.config import Config


class ProgressTracker:
    """Emit coarse percent-based progress updates into the debug log.

    The pipeline is long and partly delegated to external executables, so the
    tracker intentionally uses weighted stage buckets rather than exact task
    counts. This keeps log output stable across refactors while still making it
    easy to see where a run stalled.
    """

    def __init__(self, total=100.0, run_id=None):
        self.total = float(total)
        self.current = 0.0
        self.last_logged = -1
        self.stage_base = 0.0
        self.stage_weight = 0.0
        self.stage_label = None
        self.run_id = run_id

    def _emit(self, label=None):
        target = int(min(self.total, max(0.0, self.current)))
        while self.last_logged < target:
            self.last_logged += 1
            msg = f"[progress] {self.last_logged}%"
            if self.run_id:
                msg += f" run={self.run_id}"
            if label:
                msg += f" stage={label}"
            Config.debug_log(msg)

    def advance(self, delta, label=None):
        self.current = min(self.total, self.current + float(delta))
        self._emit(label)

    def start_stage(self, label, weight):
        self.stage_base = self.current
        self.stage_weight = float(weight)
        self.stage_label = label
        self._emit(label)

    def stage_tick(self, fraction, label=None):
        if self.stage_weight <= 0:
            return
        frac = max(0.0, min(1.0, float(fraction)))
        target = self.stage_base + self.stage_weight * frac
        if target > self.current:
            self.current = min(self.total, target)
            self._emit(label or self.stage_label)

    def end_stage(self, label=None):
        if self.stage_weight <= 0:
            return
        target = self.stage_base + self.stage_weight
        if target > self.current:
            self.current = min(self.total, target)
            self._emit(label or self.stage_label)
        self.stage_weight = 0.0

WATER_GRO_TEMPLATES = {
    "W": "Single water molecule\n1\n    1{resname:<5}{atomname:<5}    1   0.000   0.000   0.000\n5.0 5.0 5.0\n",
    "SW": "Single water molecule\n1\n    1{resname:<5}{atomname:<5}    1   0.000   0.000   0.000\n5.0 5.0 5.0\n",
    "TW": "Single water molecule\n1\n    1{resname:<5}{atomname:<5}    1   0.000   0.000   0.000\n5.0 5.0 5.0\n"
}
WATER_ITP_TEMPLATES = {
    "W": ";Gromacs.itp file\n[ moleculetype ]\n; name  nrexcl\n{moleculetype}         1\n\n[ atoms ]\n;   nr    type    resnr   residu    atom    cgnr  charge  mass\n1      W   1   {resname}   W   1     0.0000  72.0000\n",
    "SW": ";Gromacs.itp file\n[ moleculetype ]\n; name  nrexcl\n{moleculetype}         1\n\n[ atoms ]\n;   nr    type    resnr   residu    atom    cgnr  charge  mass\n1      SW   1   {resname}   SW   1     0.0000  54.0000\n",
    "TW": ";Gromacs.itp file\n[ moleculetype ]\n; name  nrexcl\n{moleculetype}         1\n\n[ atoms ]\n;   nr    type    resnr   residu    atom    cgnr  charge  mass\n1      TW   1   {resname}   TW   1     0.0000  36.0000\n"
}

def _load_base_parameters():
    """
    Loads base parameters like atom masses from the main ITP file.
    Also prepares a deduplicated list of ITP files for the topology.
    """
    print("\n--- 기본 파라미터 로딩 ---")
    sim_params = Config.get_param('simulation_parameters')
    base_itp = sim_params.get('base_itp_file')
    
    if not base_itp or not os.path.isfile(base_itp):
        raise FileNotFoundError(f"'base_itp_file'에 지정된 '{base_itp}' 파일을 찾을 수 없습니다.")

    # Load atom masses
    atom_type_masses = read_atom_types(base_itp)
    if not atom_type_masses:
        raise ValueError(f"'{base_itp}' 파일에서 원자 타입(atomtypes)을 로드할 수 없습니다.")
    Config.set_runtime('atom_type_masses', atom_type_masses)
    print(f"'{base_itp}'에서 {len(atom_type_masses)}개의 원자 타입 질량 정보 로드 완료")

    # Prepare a deduplicated list of ITP files
    final_itp_list = []
    
    # 1. Add base_itp_file first
    final_itp_list.append(os.path.abspath(base_itp))

    # 2. Add files from gromacs_include_path
    default_itp_dir = sim_params.get('gromacs_include_path')
    if default_itp_dir and os.path.isdir(default_itp_dir):
        for itp_path in glob.glob(os.path.join(default_itp_dir, "**", "*.itp"), recursive=True):
            abs_path = os.path.abspath(itp_path)
            if abs_path not in final_itp_list:
                final_itp_list.append(abs_path)

    # 3. Add files from additional_itp_files
    try:
        additional_files = Config.get_param('additional_itp_files')
    except KeyError:
        additional_files = []
    for itp_path in additional_files:
        abs_path = os.path.abspath(itp_path)
        if abs_path not in final_itp_list:
            final_itp_list.append(abs_path)
            
    Config.set_runtime('final_itp_files', final_itp_list)
    print(f"최종 토폴로지에 포함될 ITP 파일 목록(중복 제거): {len(final_itp_list)}개")

def _validate_config():
    print("\n--- 설정 파일 유효성 검사 중 ---")
    try:
        # 1. Backbone ID 수집
        backbone_defs = Config.get_param('hydrogel_components', 'backbone_definitions', 'BACKBONES')
        if not backbone_defs:
            raise ValueError("'hydrogel_components.backbone_definitions.BACKBONES'가 비어있습니다.")
        backbone_ids = {bb['id'] for bb in backbone_defs}
        print(f"정의된 Backbone ID: {backbone_ids}")

        # 2. Monomer 템플릿 검증
        monomer_defs = Config.get_param('monomer_definitions', 'MONOMERS')
        monomer_library = None
        if monomer_defs:
            try:
                monomer_library = load_monomer_templates(monomer_defs, backbone_defs)
                Config.set_runtime('monomer_library', monomer_library)
            except Exception as exc:
                raise ValueError(f"Monomer 템플릿 검증 실패: {exc}") from exc
        
        # 3. Linker 연결 유효성 검사
        linker_defs = Config.get_param('hydrogel_components', 'linker_definitions', 'LINKERS')
        linker_library = None
        if linker_defs:
            try:
                linker_library = load_linker_templates(linker_defs, backbone_defs)
                Config.set_runtime('linker_library', linker_library)
            except Exception as exc:
                raise ValueError(f"Linker 템플릿 검증 실패: {exc}") from exc
        
        print("--- 유효성 검사 통과 ---")
        return True
    except (KeyError, ValueError) as e:
        print(f"!!! 설정 파일 유효성 검사 실패: {e}", file=sys.stderr)
        return False

def execute_mode():
    """Dispatch the configured top-level execution mode."""
    # mode is optional for YAML; default to 'all'
    try:
        mode = Config.get_param('mode')
    except KeyError:
        mode = 'all'
    print(f"\n--- 실행 모드: {mode} ---")

    # Debug 모드 설정 및 파일 초기화
    try:
        sim_params_for_debug = Config.get_param('simulation_parameters')
        debug_mode = sim_params_for_debug.get('debug_mode', False)
        if debug_mode:
            output_dir = sim_params_for_debug.get('output_dir') or '.'
            os.makedirs(output_dir, exist_ok=True)
            debug_file = os.path.join(output_dir, 'debug.txt')
            Config.enable_debug_logging(debug_file)
            Config.debug_log(f"Debug logging enabled. Mode={mode}")
        else:
            Config.disable_debug_logging()
    except Exception:
        Config.disable_debug_logging()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    Config.set_runtime("run_id", run_id)
    Config.debug_log(f"[run] id={run_id}")
    progress = ProgressTracker(total=100.0, run_id=run_id)
    Config.set_runtime("progress_tracker", progress)
    progress.advance(1, "start")

    try:
        _load_base_parameters()
    except (FileNotFoundError, ValueError) as e:
        print(f"!!! 기본 파라미터 로딩 실패: {e}", file=sys.stderr)
        sys.exit(1)

    if not _validate_config():
        sys.exit(1)

    if mode == 'all':
        _execute_all_mode()
    else:
        print(f"알 수 없는 모드 또는 이 스크립트에서 직접 실행 미지원: {mode}")

def _run_packing_step(step_name, base_structure_gro, molecules_to_add, final_output_gro, sim_params):
    """
    Run one Packmol stage and return the resulting GRO path.
    """
    print(f"\n--- Packmol 실행 단계: {step_name} ---")
    
    # Use the centralized packer function
    result_gro, success = packer.pack_system_with_molecules(
        step_name=step_name,
        base_structure_gro=base_structure_gro,
        molecules_to_add=molecules_to_add,
        final_output_gro=final_output_gro,
        box_lengths_nm=sim_params.get('box_lengths_nm'),
        sim_params=sim_params
    )
    
    print(f"단계 '{step_name}' 완료. 결과 파일: {final_output_gro}")
    return result_gro, success

def _compute_total_charge(itp_files_list, molecule_counts_dict):
    """Estimate the system charge from ITP definitions and molecule counts."""
    if not itp_files_list or not molecule_counts_dict:
        return None

    definitions = {}
    mass_map = Config.get_runtime('atom_type_masses', {})
    for itp_path in itp_files_list:
        try:
            defs = read_itp_definitions(itp_path, atom_type_masses=mass_map)
        except Exception:
            continue
        definitions.update(defs)

    total_charge = 0.0
    found = False
    for name, count in molecule_counts_dict.items():
        if count <= 0:
            continue
        definition = definitions.get(name)
        if not definition:
            continue
        found = True
        mol_charge = sum(bead.get('charge', 0.0) for bead in definition.get('beads', []))
        total_charge += mol_charge * count

    return total_charge if found else None

def _perform_geo_opt_step(step_name, base_gro_file, output_dir, itp_files_list, molecule_counts_dict, sim_params):
    """
    Run one GROMACS energy-minimization stage.

    The helper chooses a Coulomb model conservatively:

    - explicit config overrides always win,
    - early neutral stages can force ``Cut-off`` for speed, and
    - otherwise the function falls back to a charge-aware heuristic.
    """

    print(f"\n--- Geometry Optimization Step: {step_name} ---")
    if sim_params.get('test_mode'):
        print(f"[TEST_MODE] Geometry optimization for '{step_name}' is skipped.")
        return base_gro_file

    geo_opt_cfg = sim_params.get('geo_opt', {})
    output_suffix = geo_opt_cfg.get('output_dir_suffix', '_geo_opt')
    opt_output_dir = os.path.join(output_dir, f"{step_name}{output_suffix}")
    temp_top_name = geo_opt_cfg.get('temp_top_name', 'system.top')

    os.makedirs(opt_output_dir, exist_ok=True)



    # 1. Create a temporary topology for this specific step

    temp_top_path = os.path.join(opt_output_dir, temp_top_name)

    

    final_itp_list = []

    for itp in itp_files_list:

        if itp and itp not in final_itp_list:

            final_itp_list.append(itp)



    topology_updater.create_system_topology(opt_output_dir, temp_top_path, final_itp_list)

    topology_updater.update_topology_molecules(temp_top_path, molecule_counts_dict)



    # 2. Run the geometry optimization

    mdp_overrides = dict(geo_opt_cfg.get('mdp', {}))
    total_charge = _compute_total_charge(itp_files_list, molecule_counts_dict)
    explicit_ion_names = set()
    try:
        ion_cfg = Config.get_param('add_series_parameters', 'add_small_ion', 'ions')
        explicit_ion_names = {
            str(entry.get('ion_name')).strip()
            for entry in ion_cfg
            if entry.get('ion_name')
        }
    except Exception:
        explicit_ion_names = set()
    has_explicit_ions = any(molecule_counts_dict.get(name, 0) > 0 for name in explicit_ion_names)

    neutral_override = sim_params.get('neutral_em_coulombtype')
    neutral_rcoulomb = sim_params.get('neutral_em_rcoulomb')
    neutral_rvdw = sim_params.get('neutral_em_rvdw')
    if total_charge is not None and abs(total_charge) < 1e-6 and not has_explicit_ions:
        if neutral_override:
            mdp_overrides['coulombtype'] = neutral_override
            print(f"[INFO] System charge ~0 for '{step_name}'; using coulombtype={neutral_override}.")
        elif 'coulombtype' not in mdp_overrides:
            mdp_overrides['coulombtype'] = 'Cut-off'
            print(f"[INFO] System charge ~0 for '{step_name}'; using coulombtype=Cut-off.")
        if neutral_rcoulomb is not None:
            mdp_overrides['rcoulomb'] = neutral_rcoulomb
        if neutral_rvdw is not None:
            mdp_overrides['rvdw'] = neutral_rvdw
    em_tol_value = mdp_overrides.get('emtol', geo_opt_cfg.get('emtol', 1000.0))
    nsteps_value = mdp_overrides.get('nsteps', geo_opt_cfg.get('nsteps', 5000))
    deffnm_prefix = geo_opt_cfg.get('deffnm_prefix', 'em')

    optimized_gro = run_geo_opt(

        structure_file=base_gro_file,

        topology_file=temp_top_path,

        output_dir=opt_output_dir,

        cell_opt=False,

        gmx_executable=sim_params.get('gromacs_executable_path', 'gmx'),

        maxwarn=sim_params.get('grompp_maxwarn', 1),
        em_tol=em_tol_value,
        nsteps=nsteps_value,
        mdp_overrides=mdp_overrides,
        deffnm_prefix=deffnm_prefix

    )

    return optimized_gro if optimized_gro else base_gro_file

def _merge_world_and_itps(world, extra_itps, merged_itp_path, moleculetype_name="MERGED"):
    """
    World 기반 구조와 추가 ITP 파일을 단일 ITP로 병합합니다.
    - World는 write_combined_itp로 기록
    - 외부 ITP는 moleculetype 단위로 독립적이므로 인덱스 재배치 없이 그대로 이어붙임
      (하나의 파일에 여러 moleculetype을 담는 목적)
    """
    tmp_world_itp = None
    if world is not None:
        tmp_world_itp = merged_itp_path + ".world.tmp"
        write_combined_itp(world, filename=tmp_world_itp, moleculetype_name=moleculetype_name)

    with open(merged_itp_path, "w") as fout:
        if tmp_world_itp:
            with open(tmp_world_itp, "r") as f:
                fout.write(f.read().rstrip() + "\n")
        for itp in extra_itps:
            if not itp or not os.path.isfile(itp):
                continue
            fout.write("\n; ---- merged from: {} ----\n".format(itp))
            with open(itp, "r") as f:
                fout.write(f.read().rstrip() + "\n")

    if tmp_world_itp:
        try:
            os.remove(tmp_world_itp)
        except Exception:
            pass
    return merged_itp_path
def _perform_dynamic_crosslinking(output_dir):
    """
    Connect linker stubs to true backbone ends rather than arbitrary beads.
    """
    print("\n--- 동적 가교 결합 생성 시작 (Backbone-End Matching) ---")
    debug_log_path = os.path.join(output_dir, "dynamic_bonding_debug.log")

    with open(debug_log_path, "w") as debug_f:
        try:
            sim_params = Config.get_param('simulation_parameters')
            pbc = sim_params.get('pbc_true_or_false', True)
        except KeyError as e:
            debug_f.write(f"[ERROR] Could not read simulation_parameters: {e}\n")
            print(f"[ERROR] 설정 파일 오류: {e}")
            return

        all_atoms = [atom[0] for atom in World.Atoms.values()]
        linkers = group_linker_stubs(all_atoms)
        backbone_ends = collect_backbone_ends(all_atoms)

        debug_f.write(f"Found {len(linkers)} linkers, {len(backbone_ends)} backbone chains.\n")

        if not linkers or not backbone_ends:
            print("[WARNING] 링커 또는 백본 끝단이 없습니다.")
            return

        candidate_limit = sim_params.get("dynamic_crosslink_candidate_limit", 8)
        box_vec = World.box_vector if pbc else None
        assignments, notes = plan_dynamic_crosslinks(
            linkers,
            backbone_ends,
            box_vec,
            candidate_limit=candidate_limit,
        )
        for note in notes:
            debug_f.write(f"{note}\n")

        bonds_created = 0

        def _resolve_bond_params(stub, backbone_atom):
            bond_table = []
            if getattr(stub, 'source_template', None) and getattr(stub, 'stub_type', None):
                if stub.stub_type == 'backbone_1':
                    bond_table = getattr(stub.source_template, 'backbone_1_bonds', [])
                elif stub.stub_type == 'backbone_2':
                    bond_table = getattr(stub.source_template, 'backbone_2_bonds', [])

            target_type = getattr(backbone_atom, 'backbone_type', None)
            for entry in bond_table or []:
                targets = entry.get('between')
                if isinstance(targets, str):
                    targets = [targets]
                if targets and target_type in targets:
                    return entry

            if getattr(stub, 'stub_bond_params', None):
                return stub.stub_bond_params

            return sim_params.get(
                'default_dynamic_crosslink_bond',
                {'bond_funct': 1, 'bond_c0': 0.25, 'bond_c1': 5000},
            )

        for linker_index in sorted(linkers):
            chosen = assignments.get(linker_index)
            if not chosen:
                continue

            success = []
            for assignment in chosen:
                stub = assignment.stub_atom
                best_end = assignment.backbone_atom
                bond_params = _resolve_bond_params(stub, best_end)
                try:
                    Bond(
                        stub.atom_id,
                        best_end.atom_id,
                        funct=bond_params.get('bond_funct', bond_params.get('funct', 1)),
                        c0=bond_params.get(
                            'bond_c0',
                            bond_params.get(
                                'length',
                                getattr(stub, 'external_bond_length', World.mean_sep),
                            ),
                        ),
                        c1=bond_params.get('bond_c1', bond_params.get('fc', 1250)),
                    )
                    debug_f.write(
                        "Linker {}: Stub {} ({}) -> Backbone {} (chain {}, type {}) dist={:.4f}\n".format(
                            linker_index,
                            stub.atom_id,
                            getattr(stub, 'stub_type', None),
                            best_end.atom_id,
                            assignment.chain_index,
                            getattr(best_end, 'backbone_type', None),
                            assignment.distance,
                        )
                    )
                    success.append(True)
                except Exception as exc:
                    debug_f.write(
                        f"Linker {linker_index}: Bond creation failed for stub {stub.atom_id}: {exc}\n"
                    )
                    success.append(False)

            if len(success) == 2 and all(success):
                bonds_created += 2
                debug_f.write(f"Linker {linker_index}: Connected successfully.\n")
            else:
                debug_f.write(f"Linker {linker_index}: Partial failure ({success}).\n")

    print(f"[INFO] 동적 가교 결합 완료. 생성된 결합 수: {bonds_created}")



def _execute_all_mode():

    """

    Executes the full workflow with sequential packing and genion.

    """

    from hygel_martini.hydrogel_builder.config_params import build_hydrogel, make_polymer_only

    

    print("\n--- 하이드로젤 구성 및 추가 물질 삽입 시작 ---")
    progress = Config.get_runtime("progress_tracker")
    if progress:
        progress.advance(2, "prepare")

    sim_params = Config.get_param('simulation_parameters')
    _seed_all(sim_params)
    output_dir = sim_params['output_dir']

    # Tabulated potential 테이블 파일을 출력 폴더로 복사(옵션)
    tab_tables = sim_params.get("additional_tabulated_tables", []) or []
    if tab_tables:
        os.makedirs(output_dir, exist_ok=True)
        for tbl in tab_tables:
            try:
                dest = os.path.join(output_dir, os.path.basename(tbl))
                if os.path.abspath(tbl) != os.path.abspath(dest):
                    shutil.copy(tbl, dest)
                print(f"[정보] tabulated table '{tbl}' 를 '{dest}'로 복사했습니다. grompp/mdrun 호출 시 수동으로 테이블 옵션(-table 등)을 추가해야 합니다.")
            except Exception as e:
                print(f"[경고] tabulated table '{tbl}' 복사 실패: {e}")

    os.makedirs(output_dir, exist_ok=True)



    # 1. Initial hydrogel generation (now always staged)
    if progress:
        progress.start_stage("build_backbone_only", 15)
    hydrogel_world, hydrogel_obj = build_hydrogel.build_backbone_only()
    if progress:
        progress.end_stage("build_backbone_only")

    # Perform crosslinking before any geometry optimization to ensure bonded topology
    _perform_dynamic_crosslinking(output_dir)
    if progress:
        progress.advance(2, "dynamic_crosslink")

    # Apply backbone bond patches before writing initial_backbone.itp so the backbone EM uses correct k values.
    # bonds only: angles/dihedrals don't exist yet (construct_chemical_detail hasn't run),
    # so patching them here would create duplicates that persist into initial_hydrogel.itp.
    if Config._file_path:
        _bb_patch_path = os.path.join(os.path.dirname(Config._file_path), 'config', 'backbone.yaml')
        patch_backbone_topology(_bb_patch_path, sections=('bonds',))

    backbone_gro = os.path.join(output_dir, "initial_backbone.gro")

    backbone_itp = os.path.join(output_dir, "initial_backbone.itp")

    write_to_gro(hydrogel_world, filename=backbone_gro)

    # Combined ITP (rich sections) for hydrogel
    from hygel_martini.hydrogel_builder.core_utils.io.writer import write_combined_itp
    try:
        raw_other = {k: len(v) for k, v in getattr(hydrogel_world, "OtherSections", {}).items() if v}
        Config.debug_log(
            f"[stage:backbone-only] OtherSections counts={raw_other} "
            f"WorldConstraints={len(hydrogel_world.Constraints)} "
            f"WorldExclusions={len(hydrogel_world.Exclusions)} "
            f"WorldDihedrals={len(hydrogel_world.Dihedrals)}"
        )
    except Exception:
        pass
    write_combined_itp(hydrogel_world, filename=backbone_itp, moleculetype_name="HYDROGEL")

    print(f"백본 단계 파일 생성: {backbone_gro}, {backbone_itp}")



    # Use the centrally managed ITP list for the optimization step

    itp_files_for_opt = list(Config.get_runtime('final_itp_files'))

    if backbone_itp not in itp_files_for_opt:

        itp_files_for_opt.append(backbone_itp)



    optimized_backbone = _perform_geo_opt_step(

        "backbone_stage",

        backbone_gro,

        output_dir,

        itp_files_for_opt,

        {"HYDROGEL": 1},

        sim_params

    )

    if optimized_backbone:

        build_hydrogel.apply_coordinates_from_gro(hydrogel_world, optimized_backbone)
    if progress:
        progress.advance(5, "geo_opt_backbone")



    if progress:
        progress.start_stage("finalize_hydrogel", 25)
    hydrogel_world = build_hydrogel.finalize_hydrogel(hydrogel_world, hydrogel_obj)

    # Apply backbone topology patches (angles, dihedrals) if backbone.yaml exists
    if Config._file_path:
        backbone_config_path = os.path.join(os.path.dirname(Config._file_path), 'config', 'backbone.yaml')
        patch_backbone_topology(backbone_config_path)

    if progress:
        progress.end_stage("finalize_hydrogel")



    current_gro_file = os.path.join(output_dir, "initial_hydrogel.gro")

    initial_itp = os.path.join(output_dir, "initial_hydrogel.itp")

    write_to_gro(hydrogel_world, filename=current_gro_file)
    try:
        raw_other = {k: len(v) for k, v in getattr(hydrogel_world, "OtherSections", {}).items() if v}
        Config.debug_log(
            f"[stage:initial-hydrogel] OtherSections counts={raw_other} "
            f"WorldConstraints={len(hydrogel_world.Constraints)} "
            f"WorldExclusions={len(hydrogel_world.Exclusions)} "
            f"WorldDihedrals={len(hydrogel_world.Dihedrals)}"
        )
    except Exception:
        pass

    write_combined_itp(hydrogel_world, filename=initial_itp, moleculetype_name="HYDROGEL")

    print(f"성공적으로 초기 하이드로젤 파일 생성: {current_gro_file}, {initial_itp}")



    # The main list of ITPs is now managed centrally

    itp_files_to_include = list(Config.get_runtime('final_itp_files'))

    if initial_itp not in itp_files_to_include:

        itp_files_to_include.append(initial_itp)



    molecule_counts_for_top = {"HYDROGEL": 1}



    box_size_nm = hydrogel_world.box_length
    box_vector = getattr(hydrogel_world, 'box_vector', None)
    if box_vector is None or not np.any(box_vector):
        try:
            from hygel_martini.hydrogel_builder.main_components.Universe import World as _WorldRef
            box_vector = getattr(_WorldRef, 'box_vector', None)
        except ImportError:
            box_vector = None
    if box_vector is not None and np.any(box_vector):
        box_lengths_nm = [float(x) for x in np.asarray(box_vector).ravel().tolist()]
    else:
        box_lengths_nm = [box_size_nm, box_size_nm, box_size_nm]
    sim_params['box_size_nm'] = box_size_nm  # Save scalar for backward compatibility
    sim_params['box_lengths_nm'] = box_lengths_nm

    

    

    # --- GEO OPT 1: Initial Hydrogel ---

    current_gro_file = _perform_geo_opt_step(

        "initial_hydrogel", current_gro_file, output_dir, itp_files_to_include, molecule_counts_for_top, sim_params

    )
    if progress:
        progress.advance(5, "geo_opt_initial")

    

    try:

        add_series_params = Config.get_param('add_series_parameters')

    except Exception:

        add_series_params = {}

    watername = None



    # --- Sequential Packing Steps ---



    # 2. Add Polymer

    if 'add_polymer' in add_series_params and add_series_params['add_polymer'].get('num_polymers', 0) > 0:
        if progress:
            progress.start_stage("add_polymer", 10)

        poly_params = add_series_params['add_polymer']
        should_generate = poly_params.get('generation_mode') == 'generate' or 'monomer_definitions' in poly_params
        if 'polymer_output_gro_filename' not in poly_params:
            base_name = poly_params.get('molecule_name', 'POLY')
            poly_params['polymer_output_gro_filename'] = f"{base_name}.gro"
        if 'polymer_output_itp_filename' not in poly_params:
            base_name = poly_params.get('molecule_name', 'POLY')
            poly_params['polymer_output_itp_filename'] = f"{base_name}.itp"

        if should_generate:

            print("\n--- 고분자 사전 생성 시작 ---")

            polymer_definitions = poly_params.get('monomer_definitions', Config.get_param('monomer_definitions'))
            generated_gro_paths, generated_itp_paths = make_polymer_only.generate_polymer_only_from_config(sim_params, poly_params, polymer_definitions)

            

            # --- GEO OPT 2: Polymer Only ---

            optimized_polymers = []

            for i, (poly_gro, poly_itp) in enumerate(zip(generated_gro_paths, generated_itp_paths)):

                poly_mol_name = f"POLYMER_{i}"
                try:
                    with open(poly_itp, 'r') as f:
                        in_moleculetype_section = False
                        for line in f:
                            stripped_line = line.strip()
                            if stripped_line == '[ moleculetype ]':
                                in_moleculetype_section = True
                                continue
                            if in_moleculetype_section and stripped_line and not stripped_line.startswith(';'):
                                parts = stripped_line.split()
                                if parts:
                                    poly_mol_name = parts[0]
                                    break
                except (IOError, IndexError) as e:
                    print(f"Warning: Could not parse molecule name from {poly_itp}. Using default '{poly_mol_name}'. Error: {e}")
                
                # Corrected indentation starts here
                itp_files_to_include_for_poly_opt = list(Config.get_runtime('final_itp_files'))
                if poly_itp not in itp_files_to_include_for_poly_opt:
                    itp_files_to_include_for_poly_opt.append(poly_itp)

                optimized_poly_gro = _perform_geo_opt_step(
                    f"polymer_{i}", poly_gro, output_dir, itp_files_to_include_for_poly_opt, {poly_mol_name: 1}, sim_params
                )
                optimized_polymers.append(optimized_poly_gro)



            poly_params['polymer_source_file'] = optimized_polymers[0] if optimized_polymers else generated_gro_paths[0]
            for gen_itp in generated_itp_paths:
                if gen_itp not in itp_files_to_include:
                    itp_files_to_include.extend(generated_itp_paths)



        poly_source_gro = poly_params.get('polymer_source_file')
        if not poly_source_gro:
            raise ValueError("polymer_source_file이 지정되지 않았습니다. generate 모드이거나 monomer_definitions가 있으면 자동 생성되어야 합니다.")

        poly_dest_gro = os.path.join(output_dir, os.path.basename(poly_source_gro))

        if os.path.abspath(poly_source_gro) != os.path.abspath(poly_dest_gro):

            shutil.copy(poly_source_gro, poly_dest_gro)

        

        molecules_to_add = [{"file": poly_dest_gro, "number": poly_params['num_polymers']}]

        packed_after_poly_gro = os.path.join(output_dir, "packed_after_polymer.gro")

        

        current_gro_file, pack_success = _run_packing_step(
            "Add_Polymer", current_gro_file, molecules_to_add, packed_after_poly_gro, sim_params
        )

        if pack_success:
            molecule_counts_for_top[poly_params['molecule_name']] = molecule_counts_for_top.get(poly_params['molecule_name'], 0) + poly_params['num_polymers']
        else:
            print("[TEST_MODE] Packmol skipped; polymer count not added to topology.")

        # 병합 ITP 생성 (HYDROGEL + polymer ITP들) — 단일 ITP 파일로 정리
        extra_poly_itps = []
        if should_generate:
            extra_poly_itps = list(generated_itp_paths)
        else:
            guessed_itp = os.path.splitext(poly_source_gro)[0] + ".itp"
            if os.path.isfile(guessed_itp):
                extra_poly_itps.append(guessed_itp)
        if extra_poly_itps:
            merged_path = os.path.join(output_dir, "merged_after_polymer.itp")
            _merge_world_and_itps(hydrogel_world, extra_poly_itps, merged_path, moleculetype_name="HYDROGEL")
            # merged ITP는 편의용으로 생성만 하고, 실제 topology include는 기존 방식 유지
            Config.set_runtime("merged_itp_path", merged_path)



        # --- GEO OPT 3: Hydrogel + Polymer ---

        current_gro_file = _perform_geo_opt_step(

            "add_polymer", current_gro_file, output_dir, itp_files_to_include, molecule_counts_for_top, sim_params

        )
        if progress:
            progress.end_stage("add_polymer")



    # 3. Add Molecule

    if 'add_molecule' in add_series_params and add_series_params['add_molecule'].get('num_molecules', 0) > 0:
        if progress:
            progress.start_stage("add_molecule", 10)

        mol_params = add_series_params['add_molecule']

        mol_source = mol_params['molecule_gro']
        if not os.path.exists(mol_source) and sim_params.get('test_mode'):
            print(f"[TEST_MODE] Molecule source '{mol_source}' not found. Skipping add_molecule step.")
            mol_params['num_molecules'] = 0
            add_series_params['add_molecule']['num_molecules'] = 0
            mol_source = None
        if not mol_source:
            # skip the rest of add_molecule
            mol_source_gro = None
        else:
            mol_name = mol_params.get('molecule_name')
            if not mol_name:
                mol_name = os.path.splitext(os.path.basename(mol_source))[0]
                mol_params['molecule_name'] = mol_name

            if mol_source.endswith('.xyz'):
                gro_filename = f"{os.path.splitext(os.path.basename(mol_source))[0]}.gro"
                mol_source_gro = packer.convert_xyz_to_gro(
                    mol_source,
                    os.path.join(output_dir, gro_filename),
                    sim_params.get('gromacs_executable_path') or 'gmx_mpi',
                    molecule_name=mol_params['molecule_name']
                )
            else:
                mol_source_gro = mol_source

        if mol_source_gro is None:
            mol_params['num_molecules'] = 0
            add_series_params['add_molecule']['num_molecules'] = 0
            mol_source = None
        else:
            mol_dest_gro = os.path.join(output_dir, os.path.basename(mol_source_gro))

            if os.path.abspath(mol_source_gro) != os.path.abspath(mol_dest_gro):

                shutil.copy(mol_source_gro, mol_dest_gro)



            itp_path_to_add = None

            if 'molecule_itp' in mol_params:

                itp_path_to_add = mol_params['molecule_itp']

                print(f"설정 파일에서 분자 ITP 경로를 사용합니다: {itp_path_to_add}")

            else:

                base_path, _ = os.path.splitext(mol_source)

                potential_itp_path = base_path + ".itp"

                if os.path.exists(potential_itp_path):

                    itp_path_to_add = potential_itp_path

                    print(f"자동으로 분자 ITP 파일을 감지했습니다: {itp_path_to_add}")



            if itp_path_to_add:

                itp_dest = os.path.join(output_dir, os.path.basename(itp_path_to_add))

                if os.path.abspath(itp_path_to_add) != os.path.abspath(itp_dest):

                    shutil.copy(itp_path_to_add, itp_dest)
                if itp_dest not in itp_files_to_include:
                    itp_files_to_include.append(itp_dest)

                print(f"분자 ITP 파일 추가: {itp_dest}")
                # 기존 merged ITP가 있으면 외부 분자 ITP를 붙여 새 merged 생성
                prev_merged = Config.get_runtime("merged_itp_path")
                if prev_merged and os.path.isfile(prev_merged):
                    merged_path = os.path.join(output_dir, "merged_after_molecule.itp")
                    _merge_world_and_itps(None, [prev_merged, itp_dest], merged_path, moleculetype_name="HYDROGEL")
                    # merged ITP는 편의용으로 생성만 하고, 실제 topology include는 기존 방식 유지
                    Config.set_runtime("merged_itp_path", merged_path)

            else:

                print(f"경고: 분자 '{mol_params['molecule_name']}'의 ITP 파일을 찾을 수 없습니다. grompp 단계에서 오류가 발생할 수 있습니다.")



        if mol_source_gro is not None:

            molecules_to_add = [{"file": mol_dest_gro, "number": mol_params['num_molecules']}]

            packed_after_mol_gro = os.path.join(output_dir, "packed_after_molecule.gro")



            current_gro_file, pack_success = _run_packing_step(
                "Add_Molecule", current_gro_file, molecules_to_add, packed_after_mol_gro, sim_params
            )

            if pack_success:
                molecule_counts_for_top[mol_params['molecule_name']] = molecule_counts_for_top.get(mol_params['molecule_name'], 0) + mol_params['num_molecules']
            else:
                print("[TEST_MODE] Packmol skipped; molecule count not added to topology.")

        

        # --- GEO OPT 4: Hydrogel + Molecule ---
        current_gro_file = _perform_geo_opt_step(
            "add_molecule", current_gro_file, output_dir, itp_files_to_include, molecule_counts_for_top, sim_params
        )
        if progress:
            progress.end_stage("add_molecule")
        # add_molecule 단계 후 OtherSections를 보존한 ITP를 다시 쓰지 않음 (모든 섹션은 별도 ITP에 존재한다고 가정)



    

    # 5. Create topology for genion AND final topology

    # 4. Add Water (independent of add_molecule)
    if 'add_water' in add_series_params:
        if progress:
            progress.start_stage("add_water", 10)
        water_params = add_series_params['add_water']
        water_bead_type = water_params.get('water_bead_type', 'W')
        watername = water_params.get('molecule_name', 'W')

        try:
            from hygel_martini.hydrogel_builder.add_series.add_water import calculate_water_molecules
            n_water = calculate_water_molecules(water_params.get('mode', 'full'))
        except (ImportError, KeyError, ValueError) as e:
            print(f"Could not calculate water molecules due to an error: {e}. Using fallback value.")
            n_water = water_params.get('number_of_water', 10000)  # Fallback

        gro_template = WATER_GRO_TEMPLATES.get(water_bead_type)
        itp_template = WATER_ITP_TEMPLATES.get(water_bead_type)
        if not gro_template or not itp_template:
            raise ValueError(f"Invalid 'water_bead_type': {water_bead_type}. No templates found.")

        water_dest_gro = os.path.join(output_dir, f'{watername}.gro')
        water_dest_itp = os.path.join(output_dir, f'{watername}.itp')

        with open(water_dest_gro, 'w') as f:
            f.write(gro_template.format(resname=watername, atomname=water_bead_type))
        with open(water_dest_itp, 'w') as f:
            f.write(itp_template.format(moleculetype=watername, resname=watername))

        print(f"Generated {watername}.gro and {watername}.itp for bead type {water_bead_type}")

        molecules_to_add = [{"file": water_dest_gro, "number": n_water}]
        packed_after_water_gro = os.path.join(output_dir, "packed_after_water.gro")
        current_gro_file, pack_success = _run_packing_step(
            "Add_Water", current_gro_file, molecules_to_add, packed_after_water_gro, sim_params
        )

        if pack_success:
            molecule_counts_for_top[watername] = molecule_counts_for_top.get(watername, 0) + n_water
        else:
            print("[TEST_MODE] Packmol skipped; water count not added to topology.")

        default_solvents_itp = os.path.abspath(
            os.path.join(sim_params.get('gromacs_include_path'), 'martini_v3.0.0_solvents_v1.itp')
        )
        if default_solvents_itp in itp_files_to_include:
            itp_files_to_include.remove(default_solvents_itp)
            print(f"Removed default solvents ITP: {default_solvents_itp} to avoid redefinition of '{watername}'.")

        if water_dest_itp not in itp_files_to_include:
            itp_files_to_include.append(water_dest_itp)

        current_gro_file = _perform_geo_opt_step(
            "add_water", current_gro_file, output_dir, itp_files_to_include, molecule_counts_for_top, sim_params
        )
        if progress:
            progress.end_stage("add_water")

    if 'add_small_ion' in add_series_params and add_series_params['add_small_ion'].get('additional_ion_itp_files'):

        itp_ion_add_list = add_series_params['add_small_ion'].get('additional_ion_itp_files')

        if len(itp_ion_add_list) > 0:

            for ion_itp in itp_ion_add_list:

                if ion_itp not in itp_files_to_include:

                    itp_files_to_include.append(ion_itp)




    final_top_path = os.path.join(output_dir, "system.top")

    print(f"\n--- 최종 토폴로지 파일 생성 중: {final_top_path} ---")

    print(f"ITP files to include in topology: {itp_files_to_include}")

    

    topology_updater.create_system_topology(output_dir, final_top_path, itp_files_to_include)

    

    topology_updater.update_topology_molecules(final_top_path, molecule_counts_for_top)

    print("이온 추가 전 토폴로지 업데이트 완료.")



    # 6. Add Ions using GROMACS genion



    ions_defined = 'add_small_ion' in add_series_params and add_series_params['add_small_ion'].get('ions')

    if ions_defined and not sim_params.get('test_mode'):
        if progress:
            progress.start_stage("add_ions", 15)

        ion_config = add_series_params['add_small_ion']
        ion_params_for_function = ion_config.copy()
        final_gro_path = os.path.join(output_dir, "final_system.gro")

        ion_result = add_small_ion.run_genion_for_neutralization(
            input_gro=current_gro_file,
            output_gro=final_gro_path,
            topology_file=final_top_path,
            sim_params=sim_params,
            ion_params=ion_params_for_function,
            solvent_name=watername or 'W'
        )

        current_gro_file = final_gro_path

        ion_counts_summary = {}
        if isinstance(ion_result, dict):
            ion_counts_summary = ion_result.get("ion_counts", {}) or {}

        if ion_counts_summary:
            total_ions_added = sum(ion_counts_summary.values())
            water_entry = add_series_params.get('add_water')
            water_molecule_name = water_entry.get('molecule_name', 'W') if water_entry else None
            if water_molecule_name:
                prev_count = molecule_counts_for_top.get(water_molecule_name, 0)
                molecule_counts_for_top[water_molecule_name] = max(prev_count - total_ions_added, 0)
            for ion_name, count in ion_counts_summary.items():
                molecule_counts_for_top[ion_name] = molecule_counts_for_top.get(ion_name, 0) + count
            topology_updater.update_topology_molecules(final_top_path, molecule_counts_for_top)

        # --- GEO OPT 6: Final System with Ions ---
        current_gro_file = _perform_geo_opt_step(
            "final_system_with_ions", current_gro_file, output_dir, itp_files_to_include, molecule_counts_for_top, sim_params
        )
        if progress:
            progress.end_stage("add_ions")
        shutil.copy(current_gro_file, os.path.join(output_dir, "final_optimized_system.gro"))
    else:

        if ions_defined and sim_params.get('test_mode'):

            print("[TEST_MODE] Add_Small_Ion 단계가 스킵됩니다.")



        # If no ions are added, the last .gro file is the final one.

        shutil.copy(current_gro_file, os.path.join(output_dir, "final_system.gro"))

        # --- GEO OPT 6: Final System without Ions ---

        current_gro_file = _perform_geo_opt_step(



            "final_system_no_ions", current_gro_file, output_dir, itp_files_to_include, molecule_counts_for_top, sim_params

        )
        if progress:
            progress.advance(15, "final_no_ions")

        shutil.copy(current_gro_file, os.path.join(output_dir, "final_optimized_system.gro"))





    if progress:
        progress.advance(100.0 - progress.current, "done")
    print("\n--- 모든 작업 완료 ---")
