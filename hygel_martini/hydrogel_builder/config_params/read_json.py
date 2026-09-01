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
from hygel_martini.hydrogel_builder.core_utils.common.collisions import DuplicateDeclaration
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
    "TW": "Single water molecule\n1\n    1{resname:<5}{atomname:<5}    1   0.000   0.000   0.000\n5.0 5.0 5.0\n",
    "P4": "Single water molecule\n1\n    1{resname:<5}{atomname:<5}    1   0.000   0.000   0.000\n5.0 5.0 5.0\n"
}
WATER_ITP_TEMPLATES = {
    "W": ";Gromacs.itp file\n[ moleculetype ]\n; name  nrexcl\n{moleculetype}         1\n\n[ atoms ]\n;   nr    type    resnr   residu    atom    cgnr  charge  mass\n1      W   1   {resname}   W   1     0.0000  72.0000\n",
    "SW": ";Gromacs.itp file\n[ moleculetype ]\n; name  nrexcl\n{moleculetype}         1\n\n[ atoms ]\n;   nr    type    resnr   residu    atom    cgnr  charge  mass\n1      SW   1   {resname}   SW   1     0.0000  54.0000\n",
    "TW": ";Gromacs.itp file\n[ moleculetype ]\n; name  nrexcl\n{moleculetype}         1\n\n[ atoms ]\n;   nr    type    resnr   residu    atom    cgnr  charge  mass\n1      TW   1   {resname}   TW   1     0.0000  36.0000\n",
    "P4": ";Gromacs.itp file\n[ moleculetype ]\n; name  nrexcl\n{moleculetype}         1\n\n[ atoms ]\n;   nr    type    resnr   residu    atom    cgnr  charge  mass\n1      P4   1   {resname}   W   1     0.0000  72.0000\n"
}

LEGACY_TERMINAL_POLICY = "lnk_peo_terminal_compensated"
VALID_CHAIN_ORIENTATION_POLICIES = {"random", "one_direction", "graph_directed"}


def _coerce_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y", "on"}:
            return True
        if token in {"false", "0", "no", "n", "off", "none", ""}:
            return False
    return default


def resolve_block_copolymer_settings(sim_params):
    """Resolve generic block-copolymer controls with legacy alias support.

    The old ``block_copolymer_direction_policy: lnk_peo_terminal_compensated``
    string bundled target filtering, linker-terminal compensation, and chain
    direction into one chemistry-specific value.  New configs should set those
    concerns independently.  The legacy value is kept as a compatibility alias
    and mapped onto the new fields when explicit new fields are absent.
    """
    sim_params = sim_params or {}
    warnings = []

    legacy_policy = str(sim_params.get("block_copolymer_direction_policy", "") or "").strip()
    legacy_active = legacy_policy.lower() not in {"", "none", "false", "0"}
    legacy_known = legacy_policy == LEGACY_TERMINAL_POLICY

    respect_explicit = "respect_target_backbone" in sim_params
    respect_target_backbone = _coerce_bool(
        sim_params.get("respect_target_backbone"),
        default=False,
    )

    terminal_comp = sim_params.get("linker_terminal_compensation", {}) or {}
    if not isinstance(terminal_comp, dict):
        warnings.append(
            "[WARN] linker_terminal_compensation must be a mapping; ignoring invalid value."
        )
        terminal_comp = {}
    terminal_enabled = _coerce_bool(terminal_comp.get("enabled"), default=False)

    orientation_raw = sim_params.get("chain_orientation_policy")

    if legacy_active:
        if legacy_known:
            warnings.append(
                "[WARN] block_copolymer_direction_policy=lnk_peo_terminal_compensated "
                "is deprecated; use respect_target_backbone, "
                "linker_terminal_compensation, and chain_orientation_policy."
            )
            if not respect_explicit:
                respect_target_backbone = True
            if not terminal_comp:
                terminal_comp = {
                    "enabled": True,
                    "target_backbone": "PEO",
                    "compensation_beads": 1,
                }
                terminal_enabled = True
            if orientation_raw is None:
                orientation_raw = "one_direction"
        else:
            warnings.append(
                f"[WARN] Unknown block_copolymer_direction_policy={legacy_policy!r}; "
                "ignoring legacy policy."
            )

    if terminal_enabled and not respect_explicit:
        respect_target_backbone = True

    if orientation_raw is None:
        orientation_raw = "random"
    chain_orientation_policy = str(orientation_raw).strip().lower().replace("-", "_")
    if chain_orientation_policy not in VALID_CHAIN_ORIENTATION_POLICIES:
        warnings.append(
            f"[WARN] Unsupported chain_orientation_policy={orientation_raw!r}; "
            "falling back to 'random'."
        )
        chain_orientation_policy = "random"

    return {
        "respect_target_backbone": bool(respect_target_backbone),
        "linker_terminal_compensation": terminal_comp,
        "terminal_compensation_enabled": bool(terminal_enabled),
        "chain_orientation_policy": chain_orientation_policy,
        "legacy_policy": legacy_policy,
        "warnings": warnings,
    }


def _get_bonded_topology_patch_path(sim_params=None):
    """Return the configured bonded-topology patch YAML path.

    Historically the builder always looked for ``config/backbone.yaml`` next
    to the maker file.  Validation cases can now opt into a model-specific
    patch file while the default behavior remains unchanged.
    """
    if sim_params is None:
        sim_params = Config.get_param('simulation_parameters')

    patch_path = (
        sim_params.get('bonded_topology_patch_file')
        or sim_params.get('backbone_patch_file')
    )
    if patch_path:
        return patch_path

    if not Config._file_path:
        return None
    return os.path.join(os.path.dirname(Config._file_path), 'config', 'backbone.yaml')


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

    standalone_modes = {
        "pack_polymer_then_water",
        "two_stage_packmol",
        "pack_polymer_then_water_two_stage",
    }
    if mode in standalone_modes:
        _execute_pack_polymer_then_water_mode()
        return

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


def _get_optional_config_section(*names):
    for name in names:
        try:
            return Config.get_param(name)
        except KeyError:
            continue
    return {}


def _as_box_lengths_nm(job):
    if "box_lengths_nm" in job:
        value = job["box_lengths_nm"]
    elif "box_nm" in job:
        value = job["box_nm"]
    elif "packmol_box_nm" in job:
        value = job["packmol_box_nm"]
    else:
        raise ValueError("two_stage_packmol job requires box_lengths_nm, box_nm, or packmol_box_nm")

    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return [float(value[0])] * 3
        if len(value) >= 3:
            return [float(value[0]), float(value[1]), float(value[2])]
        raise ValueError("box_lengths_nm must have one or three values")
    return [float(value)] * 3


def _has_packmol_route_md_outputs(output_dir):
    md_names = (
        "em.gro",
        "em.cpt",
        "nvt.gro",
        "nvt.cpt",
        "npt.gro",
        "npt.cpt",
        "nvt_paper.gro",
        "nvt_paper.cpt",
        "npt_paper.gro",
        "npt_paper.cpt",
    )
    return any(os.path.exists(os.path.join(output_dir, name)) for name in md_names)


def _merge_two_stage_job_defaults(defaults, job):
    merged = {
        key: value
        for key, value in defaults.items()
        if key not in {"jobs"}
    }
    merged.update(job)
    return merged


def _execute_pack_polymer_then_water_mode():
    """YAML-accessible route for polymer-first, fixed-polymer water packing."""
    print("\n--- YAML mode: pack_polymer_then_water ---")
    cfg = _get_optional_config_section(
        "two_stage_packmol",
        "pack_polymer_then_water",
        "pack_polymer_then_water_two_stage",
    )
    if not cfg:
        raise KeyError(
            "mode pack_polymer_then_water requires a two_stage_packmol "
            "or pack_polymer_then_water section"
        )

    try:
        sim_params = Config.get_param("simulation_parameters")
    except KeyError:
        sim_params = {}

    jobs = cfg.get("jobs")
    if jobs is None:
        jobs = [cfg]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("two_stage_packmol.jobs must be a non-empty list")

    packmol_path = cfg.get("packmol_path") or sim_params.get("packmol_path") or "packmol"
    force_default = bool(cfg.get("force", False))
    allow_md_overwrite = bool(cfg.get("allow_md_overwrite", False))
    results = []

    for index, raw_job in enumerate(jobs, start=1):
        if not isinstance(raw_job, dict):
            raise ValueError(f"two_stage_packmol job #{index} must be a mapping")
        job = _merge_two_stage_job_defaults(cfg, raw_job)
        label = job.get("name") or job.get("label") or f"job_{index}"
        output_dir = job.get("output_dir") or sim_params.get("output_dir")
        if not output_dir:
            raise ValueError(f"{label}: output_dir is required")
        os.makedirs(output_dir, exist_ok=True)

        final_output_gro = job.get("final_output_gro") or os.path.join(output_dir, "packed.gro")
        if os.path.exists(final_output_gro) and not bool(job.get("force", force_default)):
            print(f"[SKIP] {label}: final_output_gro exists: {final_output_gro}")
            continue
        if bool(job.get("force", force_default)) and _has_packmol_route_md_outputs(output_dir) and not allow_md_overwrite:
            raise RuntimeError(
                f"{label}: refusing to overwrite packing because MD outputs exist in {output_dir}. "
                "Set allow_md_overwrite: true only if this is intentional."
            )

        polymer_pdb = job.get("polymer_pdb") or os.path.join(output_dir, "polymer_scaffold.pdb")
        water_pdb = job.get("water_pdb") or os.path.join(output_dir, "solvent_bead.pdb")
        polymer_count = int(job.get("polymer_count", job.get("chain_count", 0)))
        water_count = int(job.get("water_count", job.get("water_bead_count", 0)))
        if polymer_count <= 0:
            raise ValueError(f"{label}: polymer_count/chain_count must be positive")
        if water_count < 0:
            raise ValueError(f"{label}: water_count/water_bead_count must be non-negative")
        if not os.path.isfile(polymer_pdb):
            raise FileNotFoundError(f"{label}: polymer_pdb not found: {polymer_pdb}")
        if water_count > 0 and not os.path.isfile(water_pdb):
            raise FileNotFoundError(f"{label}: water_pdb not found: {water_pdb}")

        box_lengths_nm = _as_box_lengths_nm(job)
        seed = int(job.get("seed", job.get("packmol_seed", 100000 + index * 101)))
        chain_count_for_loop = polymer_count
        polymer_nloop = int(
            job.get(
                "polymer_nloop",
                max(int(job.get("packmol_nloop", 250)), 500 if chain_count_for_loop > 1 else 100),
            )
        )
        water_nloop = int(job.get("water_nloop", max(int(job.get("packmol_nloop", 250)), 250)))
        audit_json = job.get("audit_json") or os.path.join(output_dir, "packed_audit.json")

        print(
            f"[TWO_STAGE_YAML] {label} polymer={polymer_count} water={water_count} "
            f"box={box_lengths_nm} output={final_output_gro}",
            flush=True,
        )
        audit = packer.pack_polymer_then_water_two_stage(
            output_dir=output_dir,
            polymer_pdb=polymer_pdb,
            water_pdb=water_pdb,
            polymer_count=polymer_count,
            water_count=water_count,
            box_lengths_nm=box_lengths_nm,
            packmol_path=packmol_path,
            final_output_gro=final_output_gro,
            tolerance=float(job.get("tolerance", job.get("packmol_tolerance", 2.0))),
            seed=seed,
            polymer_nloop=polymer_nloop,
            water_nloop=water_nloop,
            polymer_atom_count=job.get("polymer_atom_count"),
            audit_json=audit_json,
            title=job.get("title", f"{label} two-stage Packmol"),
        )
        results.append((label, audit))
        print(f"[PACKED_TWO_STAGE_YAML] {label} atoms={audit['atom_count']} gro={final_output_gro}")

    print(f"--- YAML two-stage Packmol complete: {len(results)} job(s) packed, GROMACS was not run. ---")

def _compute_total_charge(itp_files_list, molecule_counts_dict):
    """Estimate the system charge from ITP definitions and molecule counts."""
    if not itp_files_list or not molecule_counts_dict:
        return None

    definitions = {}
    mass_map = Config.get_runtime('atom_type_masses', {})
    for itp_path in itp_files_list:
        try:
            defs = read_itp_definitions(itp_path, atom_type_masses=mass_map)
        except Exception as exc:
            # Skipping a file here quietly understates the system charge, which
            # then understates the neutralizing ion count.
            print(
                f"[WARN] System-charge estimate ignores '{itp_path}': {exc}",
                file=sys.stderr,
            )
            continue
        clash = set(defs) & set(definitions)
        if clash:
            raise DuplicateDeclaration(
                f"Molecule type(s) {sorted(clash)} are defined in more than one "
                f"topology file, most recently '{itp_path}'. The later file "
                "would silently win and the system charge would be computed "
                "from whichever definition happened to load last."
            )
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


def _make_soft_bonds_itp(src_itp: str, soft_fc: float, dst_itp: str) -> None:
    """Write src_itp with ALL bond force constants replaced by soft_fc.

    Softening all bonds is equivalent to softening only the stretched crosslink
    bonds: equilibrium bonds (PEO-PEO, BCK-BCK) already have F≈0 so their fc
    does not affect the EM trajectory.  This avoids any dependency on atom-type
    or attribute heuristics to identify which bonds to target.
    """
    in_bonds = False
    out_lines = []
    with open(src_itp) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith('['):
                section = stripped.strip('[] ').split(';')[0].strip().lower()
                in_bonds = (section == 'bonds')
                out_lines.append(line)
                continue
            if in_bonds and stripped and not stripped.startswith(';'):
                tokens = stripped.split()
                if len(tokens) >= 5:
                    try:
                        tokens[4] = str(soft_fc)
                        line = '  '.join(tokens) + '\n'
                    except (ValueError, IndexError):
                        pass
            out_lines.append(line)
    with open(dst_itp, 'w') as fh:
        fh.writelines(out_lines)


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

    gpu_id_raw = sim_params.get('gpu_id')
    gpu_id = int(str(gpu_id_raw)) if gpu_id_raw is not None else None
    omp_threads_raw = sim_params.get('omp_threads')
    ntomp = int(omp_threads_raw) if omp_threads_raw is not None else None
    mpi_np_raw = sim_params.get('mpi_np')
    mpi_np = int(mpi_np_raw) if mpi_np_raw is not None else None
    mpi_args = [str(a) for a in sim_params.get('mpi_args', [])]
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
        deffnm_prefix=deffnm_prefix,
        gpu_id=gpu_id,
        ntomp=ntomp,
        mpi_np=mpi_np,
        mpi_args=mpi_args,
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
        targets_per_stub = sim_params.get("dynamic_crosslink_targets_per_stub", 1)
        box_vec = World.box_vector if pbc else None

        strategy = str(sim_params.get("dynamic_crosslink_assignment_strategy", "nearest")).lower()
        if strategy not in ("nearest", "geometry_nearest"):
            debug_f.write(
                "[WARN] dynamic_crosslink_assignment_strategy={} is no longer handled "
                "by runtime endpoint reassignment. Connectivity-aware BCK selection is "
                "a layout local-matching problem; falling back to placed-BCK nearest "
                "compatible endpoint bonding.\n".format(strategy)
            )
        block_settings = resolve_block_copolymer_settings(sim_params)
        for warning in block_settings["warnings"]:
            debug_f.write(f"{warning}\n")
        debug_f.write(
            "block_copolymer_settings: respect_target_backbone={} "
            "chain_orientation_policy={} terminal_compensation_enabled={}\n".format(
                block_settings["respect_target_backbone"],
                block_settings["chain_orientation_policy"],
                block_settings["terminal_compensation_enabled"],
            )
        )
        assignments, notes = plan_dynamic_crosslinks(
            linkers,
            backbone_ends,
            box_vec,
            candidate_limit=candidate_limit,
            targets_per_stub=targets_per_stub,
            respect_target_backbone_policy=block_settings["respect_target_backbone"],
        )
        for note in notes:
            debug_f.write(f"{note}\n")

        targets_per_stub = max(int(targets_per_stub), 1)
        assignment_issues = []
        used_end_ids = {}
        for linker_index, stubs in sorted(linkers.items()):
            chosen = list(assignments.get(linker_index, ()))
            # How many backbone ends a junction should consume follows from its
            # own stub count, not from an assumed pair: a two-stub linker taking
            # two ends each and a six-arm crosslinker taking one each both bond
            # six ends only if the count is derived rather than hard-coded.
            if len(stubs) < 2:
                assignment_issues.append(
                    f"Linker {linker_index}: found {len(stubs)} BCK stub(s); a "
                    "crosslinker needs at least two"
                )
            expected_per_linker = len(stubs) * targets_per_stub
            if len(chosen) != expected_per_linker:
                assignment_issues.append(
                    f"Linker {linker_index}: expected {expected_per_linker} backbone-end "
                    f"assignments ({len(stubs)} stubs x {targets_per_stub} targets), "
                    f"found {len(chosen)}"
                )
            chain_ids = [assignment.chain_index for assignment in chosen]
            if len(set(chain_ids)) != len(chain_ids):
                assignment_issues.append(f"Linker {linker_index}: repeated backbone chain selection {chain_ids}")
            stub_counts = {}
            for assignment in chosen:
                stub_id = getattr(assignment.stub_atom, "atom_id", None)
                end_id = getattr(assignment.backbone_atom, "atom_id", None)
                stub_counts[stub_id] = stub_counts.get(stub_id, 0) + 1
                used_end_ids.setdefault(end_id, []).append(linker_index)
            expected_stub_ids = {getattr(stub, "atom_id", None) for stub in stubs}
            for stub_id in expected_stub_ids:
                if stub_counts.get(stub_id, 0) != targets_per_stub:
                    assignment_issues.append(
                        f"Linker {linker_index}: stub {stub_id} has {stub_counts.get(stub_id, 0)} backbone assignments; expected {targets_per_stub}"
                    )
        duplicate_end_ids = {
            end_id: owners
            for end_id, owners in used_end_ids.items()
            if end_id is None or len(owners) != 1
        }
        if duplicate_end_ids:
            assignment_issues.append(f"duplicate backbone end assignments: {duplicate_end_ids}")
        if assignment_issues:
            for issue in assignment_issues:
                debug_f.write(f"[BUILD FAIL] {issue}\n")
            # Detailed stub neighbor analysis
            debug_f.write("\n=== DETAILED STUB NEIGHBOR ANALYSIS ===\n")
            from hygel_martini.hydrogel_builder.core_utils.runtime.dynamic_crosslink import pbc_distance
            for linker_index, stubs in sorted(linkers.items()):
                debug_f.write(f"Linker {linker_index}:\n")
                for stub in stubs:
                    stub_id = getattr(stub, "atom_id", None)
                    debug_f.write(f"  Stub {stub_id} ({getattr(stub, 'residue_name', '')} {getattr(stub, 'atom_name', '')}) at {getattr(stub, 'position', None)}:\n")
                    all_ends_dist = []
                    for c_idx, ends_list in backbone_ends.items():
                        for end_atom in ends_list:
                            d = pbc_distance(stub.position, end_atom.position, box_vec)
                            all_ends_dist.append((d, c_idx, end_atom))
                    all_ends_dist.sort(key=lambda x: x[0])
                    for d, c_idx, end_atom in all_ends_dist[:10]:
                        end_id = getattr(end_atom, 'atom_id', None)
                        debug_f.write(f"    -> dist={d:.4f} nm to end_id={end_id} (chain={c_idx} res={getattr(end_atom, 'residue_name', '')}) type={getattr(end_atom, 'backbone_type', '')}\n")
            raise RuntimeError(
                "dynamic_crosslink: assignment validation failed before bond creation. "
                "See dynamic_bonding_debug.log for details."
            )

        bonds_created = 0

        def _resolve_bond_params(stub, backbone_atom):
            default_params = sim_params.get(
                'default_dynamic_crosslink_bond',
                {'bond_funct': 1, 'bond_c0': 0.25, 'bond_c1': 5000},
            )

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
                    merged = dict(default_params)
                    merged.update(entry)
                    return merged

            if getattr(stub, 'stub_bond_params', None):
                merged = dict(default_params)
                merged.update(stub.stub_bond_params)
                return merged

            return default_params

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

            if len(success) == expected_per_linker and all(success):
                bonds_created += len(success)
                debug_f.write(f"Linker {linker_index}: Connected successfully.\n")
            else:
                bonds_created += sum(1 for item in success if item)
                debug_f.write(f"Linker {linker_index}: Partial failure ({success}).\n")
                raise RuntimeError(
                    f"dynamic_crosslink: Linker {linker_index} created an incomplete bond set. "
                    "See dynamic_bonding_debug.log for details."
                )

    print(f"[INFO] 동적 가교 결합 완료. 생성된 결합 수: {bonds_created}")


def _get_hydrogel_topology_connectivity_audit_config():
    """Return the post-build hydrogel topology audit config.

    ``connectivity_guard`` is kept as a legacy alias so older YAML files still
    behave the same way.  New configs should use the more explicit
    ``hydrogel_topology_connectivity_audit`` name.
    """
    try:
        return Config.get_param('hydrogel_topology_connectivity_audit')
    except (KeyError, ValueError):
        pass
    try:
        legacy_cfg = Config.get_param('connectivity_guard')
        if legacy_cfg:
            print(
                "[WARNING] 'connectivity_guard' is deprecated; use "
                "'hydrogel_topology_connectivity_audit' for hydrogel bonded-graph checks."
            )
        return legacy_cfg
    except (KeyError, ValueError):
        return None


def _audit_and_guard_connectivity(gro_path, itp_path, output_dir):
    """
    Audit the generated hydrogel bonded topology and apply the optional guard.
    """
    from collections import Counter, defaultdict
    print("\n--- 하이드로젤 네트워크 가교 연결성 검사 (Topology Connectivity Audit) ---")
    audit_cfg = _get_hydrogel_topology_connectivity_audit_config()
    audit_enabled = bool(audit_cfg and audit_cfg.get('enabled', False))
    fail_on_violation = bool((audit_cfg or {}).get('fail_on_violation', True))

    def _handle_audit_error(message):
        print(f"[ERROR] {message}")
        if audit_enabled and fail_on_violation:
            raise RuntimeError(message)

    # 1. Parse ITP bonds
    bonds = []
    max_atom = 0
    section = None
    try:
        with open(itp_path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith(";"):
                    continue
                if line.startswith("["):
                    section = line.split(";", 1)[0].strip().strip("[] ").lower()
                    continue
                if section != "bonds":
                    continue
                fields = line.split()
                if len(fields) < 2 or not fields[0].isdigit() or not fields[1].isdigit():
                    continue
                a, b = int(fields[0]), int(fields[1])
                bonds.append((a, b))
                max_atom = max(max_atom, a, b)
    except Exception as e:
        _handle_audit_error(f"ITP bond parsing failed during connectivity audit: {e}")
        return

    # 2. Parse GRO to get atom count and labels
    n_atoms = 0
    labels = {}
    try:
        if os.path.exists(gro_path):
            with open(gro_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            n_atoms = int(lines[1].strip())
            for atom_id, line in enumerate(lines[2 : 2 + n_atoms], start=1):
                labels[atom_id] = {
                    "residue": line[5:10].strip(),
                    "atom": line[10:15].strip(),
                    "serial": line[15:20].strip(),
                }
    except Exception as e:
        print(f"[WARNING] GRO 파일 파싱 오류: {e}")

    if n_atoms == 0:
        n_atoms = max_atom
    if n_atoms == 0:
        _handle_audit_error("Connectivity audit found zero atoms; cannot evaluate hydrogel graph.")
        return

    # 3. Union-Find to find components
    class UnionFind:
        def __init__(self, n: int):
            self.parent = list(range(n + 1))
            self.size = [1] * (n + 1)

        def find(self, x: int) -> int:
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def union(self, a: int, b: int) -> None:
            ra, rb = self.find(a), self.find(b)
            if ra == rb:
                return
            if self.size[ra] < self.size[rb]:
                ra, rb = rb, ra
            self.parent[rb] = ra
            self.size[ra] += self.size[rb]

    uf = UnionFind(n_atoms)
    for a, b in bonds:
        if a <= n_atoms and b <= n_atoms:
            uf.union(a, b)

    components = defaultdict(list)
    for atom_id in range(1, n_atoms + 1):
        components[uf.find(atom_id)].append(atom_id)
    comp_list = sorted(components.values(), key=len, reverse=True)

    num_components = len(comp_list)
    largest_comp_size = len(comp_list[0]) if comp_list else 0
    largest_comp_fraction = largest_comp_size / n_atoms if n_atoms > 0 else 0.0

    # Write connectivity audit log
    audit_log_path = os.path.join(output_dir, "connectivity_audit.log")
    try:
        with open(audit_log_path, "w", encoding="utf-8") as f:
            f.write(f"itp: {itp_path}\n")
            f.write(f"gro: {gro_path}\n")
            f.write(f"atoms: {n_atoms}\n")
            f.write(f"bonds: {len(bonds)}\n")
            f.write(f"components: {num_components}\n")
            f.write(f"largest_component_atoms: {largest_comp_size}\n")
            f.write(f"largest_component_fraction: {largest_comp_fraction:.6f}\n")
            sizes_str = " ".join(str(len(c)) for c in comp_list[:12])
            f.write(f"component_sizes: {sizes_str}\n")

            if labels:
                f.write("component_composition:\n")
                for idx, component in enumerate(comp_list[:12], start=1):
                    residues = Counter(labels.get(atom_id, {}).get("residue", "?") for atom_id in component)
                    atoms_counter = Counter(labels.get(atom_id, {}).get("atom", "?") for atom_id in component)
                    residue_s = ", ".join(f"{key}:{value}" for key, value in sorted(residues.items()))
                    atom_s = ", ".join(f"{key}:{value}" for key, value in atoms_counter.most_common(8))
                    f.write(f"  {idx}: size={len(component)} residues=({residue_s}) atoms=({atom_s})\n")
        print(f"[INFO] Connectivity diagnostic written to {audit_log_path}")
    except Exception as e:
        print(f"[ERROR] Connectivity diagnostic writing failed: {e}")

    # 4. Check hydrogel_topology_connectivity_audit
    if audit_enabled:
        min_fraction = audit_cfg.get('min_largest_component_fraction')
        max_comps = audit_cfg.get('max_components')

        violation_msgs = []
        if min_fraction is not None:
            if largest_comp_fraction < float(min_fraction):
                violation_msgs.append(
                    f"Largest component fraction {largest_comp_fraction:.6f} is less than required {min_fraction}"
                )
        if max_comps is not None:
            if num_components > int(max_comps):
                violation_msgs.append(
                    f"Component count {num_components} exceeds maximum allowed {max_comps}"
                )

        if violation_msgs:
            error_msg = "Connectivity guard violation(s):\n" + "\n".join(violation_msgs)
            print(f"[ERROR] {error_msg}")
            if fail_on_violation:
                raise RuntimeError(error_msg)
            else:
                print("[WARNING] Proceeding because fail_on_violation is false.")
    else:
        print("[INFO] Connectivity guard is disabled or not configured in YAML.")


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
    _bb_patch_path = _get_bonded_topology_patch_path(sim_params)
    if _bb_patch_path:
        Config.debug_log(f"[topology-patch] bonds file={_bb_patch_path}")
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

    # Optional pre-relax for freshly created dynamic crosslinks. This is off
    # unless the YAML explicitly enables it, because it changes the staged build
    # path for every system that uses all-mode.
    if sim_params.get('dynamic_crosslink_relax_enabled', False):
        _soft_fc = float(sim_params.get('dynamic_crosslink_relax_fc', 50.0))
        _soft_itp = backbone_itp + '.soft_relax.tmp'
        _make_soft_bonds_itp(backbone_itp, _soft_fc, _soft_itp)
        _soft_itp_list = [_soft_itp if itp == backbone_itp else itp for itp in itp_files_for_opt]
        _relaxed_gro = _perform_geo_opt_step(
            "post_crosslink_relax",
            backbone_gro,
            output_dir,
            _soft_itp_list,
            {"HYDROGEL": 1},
            sim_params,
        )
        try:
            os.remove(_soft_itp)
        except OSError:
            pass
        if _relaxed_gro and os.path.isfile(_relaxed_gro):
            backbone_gro = _relaxed_gro
            build_hydrogel.apply_coordinates_from_gro(hydrogel_world, backbone_gro)

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
    backbone_config_path = _get_bonded_topology_patch_path(sim_params)
    if backbone_config_path:
        Config.debug_log(f"[topology-patch] full file={backbone_config_path}")
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

    # Run the connectivity audit and guard
    _audit_and_guard_connectivity(current_gro_file, initial_itp, output_dir)



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
