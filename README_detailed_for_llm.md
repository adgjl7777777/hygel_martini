# README detailed for LLM

이 문서는 `/nas_0/software_backup/hygel_martini` 저장소를 나중에 LLM이 빠르게 다시 파악하도록 만든 상세 코드 지도입니다.
코드 원문을 복붙한 문서가 아니라, 함수/메서드별 역할, 위치, 입출력 단서, 부작용, 주요 호출 관계를 정리한 참조 문서입니다.

- 생성일: 2026-04-17 17:03:14 (local time)
- 분석 루트: `/nas_0/software_backup/hygel_martini`
- Python 분석 파일: 92개, class: 46개, function/method/nested helper: 448개
- Shell function 분석: 8개

## 범위와 주의

- 포함: `hygel_martini/` 패키지, `setup.py`, tracked example launcher/validator Python 및 shell 함수, 현재 작업트리에 존재하는 source-like untracked Python 파일.
- 제외: `.git/`, `__pycache__/`, `hygel_martini.egg-info/`, `example_myrun/` 실행 산출물, 대량 Martini force-field data 파일의 함수 없는 내용.
- `martini_v300/`은 force-field resource입니다. 함수 reference 대상은 아니지만 builder 설정의 `gromacs_include_path`와 topology include에 중요합니다.
- `World`와 `Config`는 전역 mutable state를 많이 씁니다. 함수 하나만 읽을 때도 registry mutation과 runtime state를 같이 봐야 합니다.
- 현재 작업트리는 깨끗하지 않았습니다. 아래 status는 문서 생성 전후 판단에 남겨둡니다.

```text
M hygel_martini/param_opt/qm_to_martini/pipeline.py
?? hygel_martini/param_opt/qm_to_martini/pipeline.py.bak
?? hygel_martini/param_opt/qm_to_martini/postprocess.py
?? hygel_martini/param_opt/qm_to_martini/run_postprocess_patch.py
?? hygel_martini/param_opt/qm_to_martini/test_write
```

### 파싱 실패/주의 파일

- `hygel_martini/param_opt/qm_to_martini/run_postprocess_patch.py`: unexpected indent (line 1, col 4). 이 파일은 Python module로 import 가능한 완전한 파일이 아니라 patch fragment처럼 보입니다.
- `hygel_martini/param_opt/qm_to_martini/pipeline.py.bak`은 `.py.bak` 백업 파일이라 live import 대상은 아니며 함수 reference에는 넣지 않았습니다.
- `hygel_martini/param_opt/qm_to_martini/test_write`는 0 byte marker 파일입니다.

## 전체 구조

`hygel_martini`는 크게 두 축입니다.

- `param_opt`: builder 전에 필요한 QM/xTB/ORCA/Bartender/OPLS/Martini 파라미터 준비 workflow입니다.
  - `qm_to_martini/workflow_logic/`: 핵심 오케스트레이션 로직 모듈들
  - `qm_to_martini/analysis/`: 전문 분석 및 시각화 도구
- `hydrogel_builder`: 실제 hydrogel network를 만들고 GROMACS/Packmol 기반 packing, ion/water 추가, relaxation을 수행합니다.
- `core/`: 프로젝트 전역 공용 유틸리티 및 설정 모델
- `bash_settings/`: 통합된 Bash 런처 시스템
  - `common/`: 공통 환경 설정 (`environment.sh`) 및 Slurm 제출 도구

주요 진입점은 다음입니다.

- `python -m hygel_martini.param_opt.opls_to_martini --config ...`: 02 existing OPLS/GROMACS trajectory -> Martini/Bartender fitting workflow
- `python -m hygel_martini.param_opt.qm_to_martini --config ...`: 03 QM/xTB -> Martini workflow
- `python -m hygel_martini.hydrogel_builder --config maker.yaml`: 04/04_1 hydrogel builder
- `python -m hygel_martini.hydrogel_builder.relax --config maker_soft_em.yaml`: 05 post-build relaxation
- 예제 shell launcher는 package 안의 `hygel_martini/bash_settings/launcher_utils.sh`를 찾아 source한 뒤 위 Python module을 호출합니다.

## 핵심 워크플로

### 02 Existing OPLS/GROMACS -> Martini

1. CLI: `hygel_martini.param_opt.opls_to_martini.cli.main`
2. Config: `hygel_martini.core.config.load_config` 및 `apply_cli_overrides`
3. Generator: `opls_to_martini.generator.run_opls_to_martini`
4. Existing-data path: `opls_to_martini.fitting.run_existing_data_fit`
5. Case 입력: `opls_data.cases[]`의 `geometry`, `bartender_inp`, `trajectory`, optional `tpr`, optional `edr`
6. Mode preset: `opls_data.execution.mode`가 `bartender_pipeline.md`, `bartender_pipeline.bartender.enabled`, `run_trim`, `run_bartender`를 함께 결정
7. Generated jobs: `trim/run_prepare_md.sh`, `bartender_job/run_bartender.sh`, root `run_all.sh`
8. Postprocess: 03의 `run_screening_postprocess`를 공유해 `gmx_out.itp`를 screening

예제 진입점은 `example/02_opls_to_martini/project`입니다. 일반 실행은 `MODE=setup|md|md_notrim|trim|bartender bash run_existing_opls.sh` 형태를 사용합니다.

### 03 QM/xTB -> Martini

1. CLI: `hygel_martini.param_opt.qm_to_martini.cli.main`
2. Config: `param_opt.core.config.load_config` 및 `apply_cli_overrides`
3. Generator: `qm_to_martini.generator.run_qm_to_martini`
4. Pipeline: `qm_to_martini.pipeline.run_pipeline` 또는 `run_postprocess_only`
5. Case 생성: monomer XYZ와 init template를 읽고 polymer XYZ, `*_base.inp`, `*_bartender.inp`, `case.json` 생성
6. Optional execution: `run_relax.sh`가 xTB/ORCA를, `run_bartender.sh`가 Bartender를 호출
7. Postprocess: `run_screening_postprocess` is the active postprocess entry point.

### 04/04_1 Hydrogel Builder

1. CLI: `hygel_martini.hydrogel_builder.cli.main`
2. Config: `hydrogel_builder.config_params.config.Config.load_config`가 YAML include와 path placeholder를 해결
3. Orchestrator: `config_params.read_json.execute_mode` -> `_execute_all_mode`
4. Backbone: `build_hydrogel.build_backbone_only`가 proto plan/layout/blueprint를 World에 materialize
5. Dynamic crosslink: `_perform_dynamic_crosslinking`가 linker stub와 backbone end를 연결
6. Chemical detail: `Hydrogel.construct_chemical_detail`, `construct_angles`, `construct_dihedrals`
7. Runtime stages: GROMACS EM, Packmol packing, water/ion 추가, final topology 작성

### 05 Relaxation

1. CLI: `hydrogel_builder.relax.cli.main`
2. Config: `hydrogel_builder.relax.config.load_relax_config`
3. Generator: `hydrogel_builder.relax.generator.run_relax_workflow`
4. Mode dispatch: `soft_em.run_soft_em` 또는 `soft_md.run_soft_md`

## 설계상 중요한 상태

- `hydrogel_builder.main_components.Universe.World`: class-level dict registry (`Atoms`, `Bonds`, `Angles`, `Dihedrals`, `OtherSections`)를 사용합니다. `Attributes.Atom()`/`Bond()` 생성 자체가 registry mutation입니다.
- `hydrogel_builder.config_params.config.Config`: loaded config, file path, runtime state, debug log 설정을 class variable로 보관합니다.
- `param_opt.qm_to_martini.config`의 dataclass들은 pure data model에 가깝지만, 같은 파일의 `execute_case_script` 등은 subprocess와 파일 로그를 직접 다룹니다.
- 외부 tool 의존성: GROMACS (`gmx`/`gmx_mpi`), Packmol, xTB, ORCA, Bartender, conda/srun/taskset 등이 설정에 따라 호출됩니다.

## Python module function reference

각 항목의 `주요 호출`은 정적 AST 기준이라 동적 import/callable은 일부 빠질 수 있습니다. line number는 이 문서 생성 시점의 파일 기준입니다.

### `example/04_full_builder/project/structure/validate_structure_itps.py`

Quick validator for real_test/structure/*.itp Checks: - indices within atom count per molecule - duplicate virtual site IDs - self-pairs Run: python real_test/structure/validate_structure_itps.py
- 주요 import: `glob, os, from collections import defaultdict, from hygel_martini.core_utils.io.martini_parser import read_itp_definitions`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `main()`
- 위치: `example/04_full_builder/project/structure/validate_structure_itps.py:20`
- 종류: function, CLI entry
- 역할: `main` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 주요 호출: `d.get, defaultdict, defs.items, glob.glob, os.path.basename, os.path.dirname, os.path.join, read_itp_definitions, row.get, seen.add, vs.get, vs.get.split`

### `hygel_martini/hydrogel_builder/__init__.py`

Hydrogel construction package namespace.
- 주요 import: `from __future__ import annotations`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `run_hydrogel_builder(*args, **kwargs)`
- 위치: `hygel_martini/hydrogel_builder/__init__.py:6`
- 종류: function
- 역할: `run hydrogel builder` 실행 helper입니다. workflow 단계나 외부 command/script를 실행 또는 위임합니다.
- 반환: 명시적 return 1개. 예: `_run_hydrogel_builder(*args, **kwargs)`
- 주요 호출: `_run_hydrogel_builder`

### `hygel_martini/hydrogel_builder/add_series/add_small_ion.py`

GROMACS genion을 여러 ion species와 residual charge compensation에 맞춰 순차 실행합니다.
- 주요 import: `itertools, os, random, shutil, subprocess, sys, copy, numpy, from hygel_martini.hydrogel_builder.config_params.config import Config, from hygel_martini.hydrogel_builder.core_utils.runtime.geo_opt import _run_with_logs`
- class 수: 0, 함수/메서드 수: 7

#### Functions and methods

##### `_run_checked(cmd, label, cwd=None, env=None, input_text=None)`
- 위치: `hygel_martini/hydrogel_builder/add_series/add_small_ion.py:29`
- 종류: function, private/internal
- 역할: Run a subprocess through the shared logging wrapper.
- 반환: 명시적 return 1개. 예: `proc`
- 예외/검증: `subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 외부 command/subprocess 실행 가능
- 주요 호출: `_run_with_logs, subprocess.CalledProcessError`

##### `_partition_ion_definitions(ion_list)`
- 위치: `hygel_martini/hydrogel_builder/add_series/add_small_ion.py:37`
- 종류: function, private/internal
- 역할: Split configured ions into primary and compensating pools.
- 반환: 명시적 return 1개. 예: `(cations, anions, extra_cations, extra_anions, total_charge)`
- 주요 호출: `append, ion.get`

##### `_ensure_compensation_pool(primary_pool, compensation_pool, label)`
- 위치: `hygel_martini/hydrogel_builder/add_series/add_small_ion.py:58`
- 종류: function, private/internal
- 역할: Guarantee that a compensation pool contains at least one ion species.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 예외/검증: `ValueError(f'No {label} ions are available for neutralization.')`
- 주요 호출: `ValueError, compensation_pool.append, primary_pool.pop`

##### `_apply_residual_charge(total_charge, compensation_pool, seed)`
- 위치: `hygel_martini/hydrogel_builder/add_series/add_small_ion.py:67`
- 종류: function, private/internal
- 역할: Increase compensation-ion counts so the staged system can be neutralized.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 예외/검증: `ValueError('Compensation-ion pool has zero effective charge.') ; ValueError('Could not represent the residual charge with the configured compensation ions.')`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `compensation_pool[idx]['number']`
- 주요 호출: `ValueError, ion.get, itertools.product, np.array, np.dot, np.floor, np.floor.astype, random.Random, rng.choice, valid_solutions.append`

##### `resolve_effective_ion_plan(ion_params, seed=None)`
- 위치: `hygel_martini/hydrogel_builder/add_series/add_small_ion.py:110`
- 종류: function
- 역할: Predict the effective ion counts after compensation adjustments.
- 반환: 명시적 return 2개. 예: `anion_list + cation_list ; []`
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.get_param, Config.get_param.get, _apply_residual_charge, _ensure_compensation_pool, _partition_ion_definitions, additional_anion_list.reverse, additional_cation_list.reverse, anion_list.extend, cation_list.extend, copy.deepcopy, get`

##### `run_genion_for_neutralization(input_gro, output_gro, topology_file, sim_params, ion_params, solvent_name)`
- 위치: `hygel_martini/hydrogel_builder/add_series/add_small_ion.py:142`
- 종류: function
- 역할: Runs the GROMACS genion tool to add ions and neutralize the system. Args: input_gro (str): Path to the input .gro file. output_gro (str): Path for the final output .gro file. topology_file (str): Path to the topology file (.top). sim_params (dict): Simulation parameters from config. ion_params (dict): Ion parameters from config.
- 반환: 명시적 return 1개. 예: `{'output_gro': output_gro, 'ion_counts': ion_counts_summary}`
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기, 외부 command/subprocess 실행 가능, 객체/class/global attribute 갱신
- 주요 대입: `env['GMX_INCLUDE'], ion_counts_summary[ion_name]`
- 주요 호출: `Config.debug_log, Config.get_param, Config.get_param.get, _apply_residual_charge, _ensure_compensation_pool, _partition_ion_definitions, _reorder_water_and_ions, _run_checked, additional_anion_list.reverse, additional_cation_list.reverse, anion_list.extend, cation_list.extend, f.readlines, f.write, f.writelines, ion.get, ion_counts_summary.get, ion_params.get, new_lines.append, os.environ.copy, os.path.basename, os.path.exists, os.path.join, os.path.splitext, ... (+3)`

##### `_reorder_water_and_ions(gro_path, water_resname, ion_names)`
- 위치: `hygel_martini/hydrogel_builder/add_series/add_small_ion.py:363`
- 종류: function, private/internal
- 역할: Reorder a GRO file so water and ions follow topology ordering.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Config.debug_log, f.readlines, f.write, ions_by_name.append, ions_by_name.get, line.strip, lines.strip, prefix.append, reordered.extend, waters.append`

### `hygel_martini/hydrogel_builder/add_series/add_water.py`

gel weight fraction 설정에서 물 bead 수를 추정합니다. polymer/molecule/ion 질량까지 반영할 수 있습니다.
- 주요 import: `math, os, from hygel_martini.hydrogel_builder.config_params.config import Config, from hygel_martini.hydrogel_builder.add_series.add_small_ion import resolve_effective_ion_plan, from hygel_martini.hydrogel_builder.core_utils.io.martini_parser import read_itp_definitions, from hygel_martini.hydrogel_builder.core_utils.templates.monomer_loader import load_monomer_templates`
- class 수: 0, 함수/메서드 수: 6

#### Functions and methods

##### `get_weighted_average_mass(*args)`
- 위치: `hygel_martini/hydrogel_builder/add_series/add_water.py:29`
- 종류: function
- 역할: Compute a ratio-weighted mean mass for configured components.
- 반환: 명시적 return 2개. 예: `total_mass / total_ratio if total_ratio > 0 else 0 ; 0`
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.get_param, bead.get, component.get, load_monomer_templates`

##### `_safe_get_param(*keys, default=None)`
- 위치: `hygel_martini/hydrogel_builder/add_series/add_water.py:62`
- 종류: function, private/internal
- 역할: 설정 객체를 읽거나 runtime state를 갱신하는 helper입니다.
- 반환: 명시적 return 2개. 예: `Config.get_param(*keys) ; default`
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.get_param`

##### `_resolve_gel_weight_fraction_mode(sim_params)`
- 위치: `hygel_martini/hydrogel_builder/add_series/add_water.py:69`
- 종류: function, private/internal
- 역할: `resolve gel weight fraction mode` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `mode`
- 예외/검증: `ValueError(f'Invalid simulation_parameters.gel_weight_fraction_mode: {mode}. Must be one of {sorted(GEL_WEIGHT_FRACTION_MODES)}')`
- 주요 호출: `ValueError, sim_params.get`

##### `_load_definition_lookup(itp_paths)`
- 위치: `hygel_martini/hydrogel_builder/add_series/add_water.py:79`
- 종류: function, private/internal
- 역할: `load definition lookup` 계열 loader입니다. 설정/파일/템플릿을 읽어 후속 builder가 사용할 dict/dataclass 구조로 정규화합니다.
- 반환: 명시적 return 1개. 예: `definitions`
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.get_runtime, definitions.update, os.path.isfile, read_itp_definitions`

##### `_estimate_ion_usage(sim_params)`
- 위치: `hygel_martini/hydrogel_builder/add_series/add_water.py:93`
- 종류: function, private/internal
- 역할: `estimate ion usage` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 3개. 예: `(total_ion_count, total_ion_mass) ; (0, 0.0)`
- 예외/검증: `ValueError('Could not resolve masses for configured ions: ' + ', '.join(sorted(missing_ions)))`
- 주요 호출: `ValueError, _load_definition_lookup, _safe_get_param, bead.get, candidate_itps.append, candidate_itps.extend, definition.get, definitions.get, ion.get, ion_params.get, join, missing_ions.add, os.path.join, resolve_effective_ion_plan, sim_params.get`

##### `calculate_water_molecules(mode)`
- 위치: `hygel_martini/hydrogel_builder/add_series/add_water.py:137`
- 종류: function
- 역할: Estimate how many coarse-grained water beads should be inserted.
- 반환: 명시적 return 1개. 예: `n_water`
- 예외/검증: `ValueError('Error: gel_weight_fraction must be between 0 and 1.') ; ValueError(f'Invalid water_bead_type: {water_bead_type}. Must be one of {list(water_masses.keys())}')`
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.get_param, ValueError, _estimate_ion_usage, _resolve_gel_weight_fraction_mode, add_water_params.get, get_weighted_average_mass, math.ceil, water_masses.get, water_masses.keys`

### `hygel_martini/hydrogel_builder/cli.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `from __future__ import annotations, argparse, from pathlib import Path`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `main()`
- 위치: `hygel_martini/hydrogel_builder/cli.py:7`
- 종류: function, CLI entry
- 역할: `main` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 주요 호출: `Path, argparse.ArgumentParser, parser.add_argument, parser.exit, parser.parse_args, run_hydrogel_builder`

### `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py`

hydrogel backbone-only 단계와 chemical-detail 확장 단계를 분리해 실행합니다. proto plan/layout/blueprint를 만들고 World 상태에 물질화합니다.
- 주요 import: `os, random, traceback, numpy, from hygel_martini.hydrogel_builder.config_params.config import Config, from hygel_martini.hydrogel_builder.core_utils.common.utility import find_minimum_distances, from hygel_martini.hydrogel_builder.core_utils.layout.isotropic_builder import build_isotropic_blueprint, from hygel_martini.hydrogel_builder.core_utils.layout.layout_executor import build_atom_blueprint, from hygel_martini.hydrogel_builder.core_utils.layout.proto_builder import prepare_proto_plan, from hygel_martini.hydrogel_builder.core_utils.layout.proto_layout import generate_layout_plan, from hygel_martini.hydrogel_builder.core_utils.layout.proto_populator import populate_hydrogel_from_blueprint, from hygel_martini.hydrogel_builder.core_utils.templates.linker_loader import linker_definitions_from_library, load_linker_templates, ...`
- class 수: 0, 함수/메서드 수: 16

#### Functions and methods

##### `_debug_stage(message)`
- 위치: `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py:42`
- 종류: function, private/internal
- 역할: Emit a stage marker to the optional debug log.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.debug_log`

##### `_print_build_banner()`
- 위치: `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py:50`
- 종류: function, private/internal
- 역할: Print the standard banner used by backbone-construction stages.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.

##### `_seed_random_generators(seed)`
- 위치: `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py:57`
- 종류: function, private/internal
- 역할: Seed Python and NumPy RNGs when a deterministic run is requested.
- 반환: 명시적 return 2개이지만 값 없는 return 경로가 중심입니다.
- 주요 호출: `np.random.seed, random.seed`

##### `_compute_max_linker_span()`
- 위치: `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py:69`
- 종류: function, private/internal
- 역할: Return the largest linker span declared in the configuration.
- 반환: 명시적 return 2개. 예: `max_span ; 0.0`
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.get_param, bond.get, definition.get, ext.get, linker.get`

##### `_gather_sorted_atoms()`
- 위치: `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py:96`
- 종류: function, private/internal
- 역할: Collect the current ``World`` atoms in deterministic atom-id order.
- 반환: 명시적 return 1개. 예: `(atom_ids, atoms)`
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation
- 주요 호출: `World.Atoms.keys, atoms.append`

##### `_log_min_distance_report(label)`
- 위치: `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py:107`
- 종류: function, private/internal
- 역할: Write a compact minimum-distance report for debugging.
- 반환: 명시적 return 3개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Config.get_param, _debug_stage, _gather_sorted_atoms, find_minimum_distances, log_f.write, np.array, os.makedirs, os.path.join`

##### `apply_coordinates_from_gro(world, gro_path)`
- 위치: `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py:150`
- 종류: function
- 역할: Project coordinates from a GRO file back into the current ``World``.
- 반환: 명시적 return 2개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `atoms[idx].position`
- 주요 호출: `_gather_sorted_atoms, coords.append, gro_f.readline, gro_f.readline.strip, np.array, os.path.exists`

##### `_reset_world_for_backbone(sim_params)`
- 위치: `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py:184`
- 종류: function, private/internal
- 역할: Reset the global world state and initialize box-scale parameters.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation
- 주요 호출: `Attributes.initialize, World.reset, _compute_max_linker_span, _seed_random_generators, initialize_world, sim_params.get`

##### `_load_backbone_context()`
- 위치: `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py:197`
- 종류: function, private/internal
- 역할: Load template libraries and sequence strategies for backbone planning.
- 반환: 명시적 return 1개. 예: `{'backbone_cfg': backbone_cfg, 'backbone_defs': backbone_defs, 'backbone_strategy': backbone_strategy, 'linker_cfg': linker_cfg, 'linker_defs': linker_defs, 'linker_strategy': l...`
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.get_param, Config.get_runtime, Config.set_runtime, backbone_cfg.get, linker_cfg.get, linker_definitions_from_library, load_linker_templates, load_monomer_templates`

##### `_resolve_isotropy_mode(sim_params)`
- 위치: `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py:235`
- 종류: function, private/internal
- 역할: Resolve whether the special isotropic builder path should be used.
- 반환: 명시적 return 6개. 예: `bool(isotropy_cfg) ; not anisotropy ; anisotropy is None ; True ; False`
- 주요 호출: `isotropy_cfg.get, sim_params.get`

##### `_build_blueprint_summary(layout_plan, blueprint)`
- 위치: `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py:256`
- 종류: function, private/internal
- 역할: Print a compact summary of the generated layout and blueprint.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.

##### `_plan_backbone_blueprint(sim_params, output_dir)`
- 위치: `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py:267`
- 종류: function, private/internal
- 역할: Build the proto plan and atom blueprint for the hydrogel backbone.
- 반환: 명시적 return 1개. 예: `{**context, 'proto_plan': proto_plan, 'layout_plan': layout_plan, 'blueprint': blueprint, 'num_cells': num_cells, 'repeats': repeats, 'isotropy_mode': isotropy_mode}`
- 예외/검증: `AssertionError('Diamond network must have even number of cells or debug value 1') ; ValueError('number_of_cells must be >= 1')`
- 주요 호출: `AssertionError, ValueError, _build_blueprint_summary, _load_backbone_context, _resolve_isotropy_mode, build_atom_blueprint, build_isotropic_blueprint, context.get, generate_layout_plan, getattr, prepare_proto_plan, sim_params.get`

##### `_apply_materialization_box_settings(plan_context)`
- 위치: `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py:343`
- 종류: function, private/internal
- 역할: Copy proto-plan box data into ``World`` before object creation.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `World.cell_vector, World.box_vector, World.box_length, World.ubox_length`
- 주요 호출: `np.array, np.max, proto_plan.box_vector`

##### `build_backbone_only()`
- 위치: `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py:362`
- 종류: function
- 역할: World를 초기화하고 proto/layout/blueprint를 만든 뒤 Hydrogel backbone/linker skeleton만 생성합니다. chemical sidechain 확장 전 단계입니다.
- 반환: 명시적 return 1개. 예: `(world, hd)`
- 예외/검증: `RuntimeError('proto builder blueprint 생성에 실패했습니다.')`
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.get_param, RuntimeError, World, _apply_materialization_box_settings, _debug_stage, _log_min_distance_report, _plan_backbone_blueprint, _print_build_banner, _reset_world_for_backbone, hd.construct_bonds, populate_hydrogel_from_blueprint, sim_params.get, traceback.print_exc, world.make_hydrogel, world.update_hydrogel_attributes`

##### `finalize_hydrogel(world, hd)`
- 위치: `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py:415`
- 종류: function
- 역할: backbone-only World에 chemical detail, angles, dihedrals, impropers를 추가하고 최종 hydrogel counters를 갱신합니다.
- 반환: 명시적 return 1개. 예: `world`
- 주요 호출: `_debug_stage, _log_min_distance_report, hd.construct_angles, hd.construct_chemical_detail, hd.construct_dihedrals, hd.construct_impropers, world.update_hydrogel_attributes`

##### `main()`
- 위치: `hygel_martini/hydrogel_builder/config_params/build_hydrogel.py:441`
- 종류: function, CLI entry
- 역할: Run the standalone hydrogel builder entry point.
- 반환: 명시적 return 1개. 예: `world`
- 주요 호출: `build_backbone_only, finalize_hydrogel`

### `hygel_martini/hydrogel_builder/config_params/config.py`

builder용 전역 Config 캐시입니다. YAML include 병합, `${CONFIG_DIR}`/`${REPO_ROOT}` 경로 치환, runtime state와 debug log를 관리합니다.
- 주요 import: `copy, json, os`
- class 수: 1, 함수/메서드 수: 16

#### Classes

- `Config` (hygel_martini/hydrogel_builder/config_params/config.py:13)
  - 역할: Singleton-style access to configuration and runtime metadata.
  - 주요 field/class var: `_data = None, _file_path = None, _runtime_state = {}, _debug_file = None, _debug_enabled = False, _PATH_LIST_KEYS = {'additional_itps', 'additional_itp_files', 'additional_tabulated_tables'}, _LITERAL_COMMAND_KEYS = {'gromacs_executable_path', 'packmol_path'}, _NON_PATH_KEYS = {'output_dir_suffix'}, _PATH_SUFFIXES = ('_path', '_file', '_dir', '_gro', '_itp', '_root')`

#### Functions and methods

##### `Config.load_config(cls, file_path)`
- 위치: `hygel_martini/hydrogel_builder/config_params/config.py:26`
- 종류: method, classmethod
- decorators: `classmethod`
- 역할: Load a JSON or YAML maker file into the global config cache.
- 반환: 명시적 return 1개. 예: `cls._data`
- 예외/검증: `FileNotFoundError(f'Configuration file not found at {file_path}') ; ValueError(f'Error decoding JSON from {file_path}')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `cls._data, cls._runtime_state['config_dir'], cls._runtime_state['repo_root'], cls._file_path`
- 주요 호출: `FileNotFoundError, ValueError, cls._build_path_context, cls._load_yaml_with_includes, cls._normalize_path_tree, json.load, os.path.abspath, os.path.splitext, os.path.splitext.lower`

##### `Config.get_param(cls, *keys, file_path=None)`
- 위치: `hygel_martini/hydrogel_builder/config_params/config.py:50`
- 종류: method, classmethod
- decorators: `classmethod`
- 역할: Read a nested value from the loaded configuration tree.
- 반환: 명시적 return 1개. 예: `current_level`
- 예외/검증: `KeyError(f"Key '{key}' not found in configuration at path {'.'.join(str_keys)}") ; ValueError('Configuration not loaded. Call load_config(file_path) first.')`
- 주요 호출: `KeyError, ValueError, cls.load_config, join`

##### `Config.set_param(cls, value, *keys)`
- 위치: `hygel_martini/hydrogel_builder/config_params/config.py:73`
- 종류: method, classmethod
- decorators: `classmethod`
- 역할: Write a nested value into the live configuration tree.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 예외/검증: `ValueError('Configuration not loaded. Call load_config(file_path) first.')`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `current_level[keys[-1]]`
- 주요 호출: `ValueError, current_level.setdefault`

##### `Config.set_runtime(cls, key, value)`
- 위치: `hygel_martini/hydrogel_builder/config_params/config.py:83`
- 종류: method, classmethod
- decorators: `classmethod`
- 역할: Store ephemeral runtime state that should not live in the config.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `cls._runtime_state[key]`

##### `Config.get_runtime(cls, key, default=None)`
- 위치: `hygel_martini/hydrogel_builder/config_params/config.py:88`
- 종류: method, classmethod
- decorators: `classmethod`
- 역할: Read ephemeral runtime state with an optional default.
- 반환: 명시적 return 1개. 예: `cls._runtime_state.get(key, default)`
- 주요 호출: `cls._runtime_state.get`

##### `Config.enable_debug_logging(cls, file_path)`
- 위치: `hygel_martini/hydrogel_builder/config_params/config.py:93`
- 종류: method, classmethod
- decorators: `classmethod`
- 역할: Enable debug logging to a file (overwrite on enable).
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `cls._debug_enabled, cls._debug_file`
- 주요 호출: `f.write`

##### `Config.disable_debug_logging(cls)`
- 위치: `hygel_martini/hydrogel_builder/config_params/config.py:105`
- 종류: method, classmethod
- decorators: `classmethod`
- 역할: Disable file-backed debug logging.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `cls._debug_enabled, cls._debug_file`

##### `Config.debug_log(cls, message)`
- 위치: `hygel_martini/hydrogel_builder/config_params/config.py:111`
- 종류: method, classmethod
- decorators: `classmethod`
- 역할: Append a debug message with basic timestamp if debug logging is enabled.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `datetime.now, datetime.now.strftime, f.write`

##### `Config._deep_merge(cls, base, incoming)`
- 위치: `hygel_martini/hydrogel_builder/config_params/config.py:125`
- 종류: method, private/internal, classmethod
- decorators: `classmethod`
- 역할: Recursively merge dict incoming into base (mutates base).
- 반환: 명시적 return 1개. 예: `base`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `base[k]`
- 주요 호출: `base.get, cls._deep_merge, copy.deepcopy, incoming.items`

##### `Config._load_yaml_file(cls, path)`
- 위치: `hygel_martini/hydrogel_builder/config_params/config.py:135`
- 종류: method, private/internal, classmethod
- decorators: `classmethod`
- 역할: Read a single YAML file without processing includes.
- 반환: 명시적 return 1개. 예: `data if isinstance(data, dict) else {}`
- 예외/검증: `FileNotFoundError(f'Configuration file not found at {path}') ; ImportError('PyYAML이 필요합니다. `pip install pyyaml` 후 다시 시도하세요.')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `FileNotFoundError, ImportError, yaml.safe_load`

##### `Config._load_yaml_with_includes(cls, path, seen=None)`
- 위치: `hygel_martini/hydrogel_builder/config_params/config.py:149`
- 종류: method, private/internal, classmethod
- decorators: `classmethod`
- 역할: Load a YAML file and recursively merge its ``includes`` chain.
- 반환: 명시적 return 1개. 예: `merged`
- 예외/검증: `ValueError(f'Cyclic include detected for {path}')`
- 주요 호출: `ValueError, cls._deep_merge, cls._load_yaml_file, cls._load_yaml_with_includes, data.pop, os.path.abspath, os.path.dirname, os.path.isabs, os.path.join, seen.add`

##### `Config._build_path_context(cls, file_path)`
- 위치: `hygel_martini/hydrogel_builder/config_params/config.py:170`
- 종류: method, private/internal, classmethod
- decorators: `classmethod`
- 역할: `build path context` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 1개. 예: `{'CONFIG_DIR': config_dir, 'REPO_ROOT': repo_root}`
- 주요 호출: `os.path.abspath, os.path.dirname, os.path.join`

##### `Config._looks_like_path_key(cls, key)`
- 위치: `hygel_martini/hydrogel_builder/config_params/config.py:176`
- 종류: method, private/internal, classmethod
- decorators: `classmethod`
- 역할: `looks like path key` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 3개. 예: `key in {'gro', 'itp', 'molecule_gro', 'molecule_itp'} or key.endswith(cls._PATH_SUFFIXES) ; False ; True`
- 주요 호출: `key.endswith`

##### `Config._resolve_path_value(cls, value, path_context)`
- 위치: `hygel_martini/hydrogel_builder/config_params/config.py:184`
- 종류: method, private/internal, classmethod
- decorators: `classmethod`
- 역할: `resolve path value` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `expanded`
- 주요 호출: `expanded.replace, os.path.abspath, os.path.expanduser, os.path.expandvars, os.path.isabs, os.path.join, path_context.items`

##### `Config._should_resolve_scalar_path(cls, key, value)`
- 위치: `hygel_martini/hydrogel_builder/config_params/config.py:193`
- 종류: method, private/internal, classmethod
- decorators: `classmethod`
- 역할: `should resolve scalar path` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `True ; value.startswith('.') or value.startswith('~') or '${' in value or ('/' in value) or ('\\' in value)`
- 주요 호출: `value.startswith`

##### `Config._normalize_path_tree(cls, node, path_context, parent_key=None)`
- 위치: `hygel_martini/hydrogel_builder/config_params/config.py:205`
- 종류: method, private/internal, classmethod
- decorators: `classmethod`
- 역할: `normalize path tree` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 5개. 예: `node ; {key: cls._normalize_path_tree(value, path_context, key) for key, value in node.items()} ; [cls._normalize_path_tree(item, path_context) for item in node] ; cls._resolve_path_value(node, path_context) ; [cls._resolve_path_value(item, path_context) if isinstance(item, str) else item for item in node]`
- 주요 호출: `cls._looks_like_path_key, cls._normalize_path_tree, cls._resolve_path_value, cls._should_resolve_scalar_path, node.items`

### `hygel_martini/hydrogel_builder/config_params/generator.py`

Top-level entry helpers for hydrogel construction runs. This module is intentionally thin: it loads the maker configuration, normalizes input line endings, and then hands control to ``execute_mode``.
- 주요 import: `os, sys, from hygel_martini.hydrogel_builder.config_params.config import Config, from hygel_martini.hydrogel_builder.config_params.read_json import execute_mode, from hygel_martini.hydrogel_builder.core_utils.common.utility import run_dos2unix_on_inputs`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `run_hydrogel_example(config_path)`
- 위치: `hygel_martini/hydrogel_builder/config_params/generator.py:14`
- 종류: function
- 역할: Run a full hydrogel-generation job from a maker file.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 주요 호출: `Config.load_config, execute_mode, os.path.basename, run_dos2unix_on_inputs`

### `hygel_martini/hydrogel_builder/config_params/make_polymer_only.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `os, from hygel_martini.hydrogel_builder.core_utils.generators import polymer_generator, from hygel_martini.hydrogel_builder.main_components.Polymer import Polymer`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `generate_polymer_only_from_config(sim_params, poly_gen_params, polymer_config=None)`
- 위치: `hygel_martini/hydrogel_builder/config_params/make_polymer_only.py:5`
- 종류: function
- 역할: `generate polymer only from config` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `(generated_gro_paths, generated_itp_paths)`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Polymer.configure, generated_gro_paths.append, generated_itp_paths.append, os.makedirs, os.path.join, os.path.splitext, polymer_generator.generate_single_polymer_gro`

### `hygel_martini/hydrogel_builder/config_params/read_json.py`

hydrogel builder의 전체 실행 순서를 담당합니다. 설정 로드 후 backbone 생성, 동적 가교, geometry optimization, polymer/molecule/water/ion 추가, 최종 topology 작성 순서를 조립합니다.
- 주요 import: `glob, os, random, shutil, sys, numpy, from hygel_martini.hydrogel_builder.add_series import add_small_ion, from hygel_martini.hydrogel_builder.core_utils.io.martini_parser import read_atom_types, read_itp_definitions, from hygel_martini.hydrogel_builder.core_utils.io.writer import write_to_gro, write_to_itp, write_combined_itp, from hygel_martini.hydrogel_builder.core_utils.runtime import packer, topology_updater, from hygel_martini.hydrogel_builder.core_utils.runtime.backbone_patcher import patch_backbone_topology, from hygel_martini.hydrogel_builder.core_utils.runtime.dynamic_crosslink import collect_backbone_ends, group_linker_stubs, plan_dynamic_crosslinks, ...`
- class 수: 1, 함수/메서드 수: 16

#### Classes

- `ProgressTracker` (hygel_martini/hydrogel_builder/config_params/read_json.py:61)
  - 역할: Emit coarse percent-based progress updates into the debug log. The pipeline is long and partly delegated to external executables, so the tracker intentionally uses weighted stage buckets rather than exact task counts. This keeps log output stable across refactors while still making it easy to see where a run stalled.

#### Functions and methods

##### `_seed_all(sim_params)`
- 위치: `hygel_martini/hydrogel_builder/config_params/read_json.py:41`
- 종류: function, private/internal
- 역할: Seed all RNGs used by the orchestration layer.
- 반환: 명시적 return 2개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.debug_log, np.random.seed, random.seed, sim_params.get`

##### `ProgressTracker.__init__(self, total=100.0, run_id=None)`
- 위치: `hygel_martini/hydrogel_builder/config_params/read_json.py:70`
- 종류: method, private/internal
- 역할: `init` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `self.total, self.current, self.last_logged, self.stage_base, self.stage_weight, self.stage_label, self.run_id`

##### `ProgressTracker._emit(self, label=None)`
- 위치: `hygel_martini/hydrogel_builder/config_params/read_json.py:79`
- 종류: method, private/internal
- 역할: 설정 객체를 읽거나 runtime state를 갱신하는 helper입니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: Config/runtime state 접근, 객체/class/global attribute 갱신
- 주요 대입: `self.last_logged`
- 주요 호출: `Config.debug_log`

##### `ProgressTracker.advance(self, delta, label=None)`
- 위치: `hygel_martini/hydrogel_builder/config_params/read_json.py:90`
- 종류: method
- 역할: `advance` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `self.current`
- 주요 호출: `self._emit`

##### `ProgressTracker.start_stage(self, label, weight)`
- 위치: `hygel_martini/hydrogel_builder/config_params/read_json.py:94`
- 종류: method
- 역할: `start stage` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `self.stage_base, self.stage_weight, self.stage_label`
- 주요 호출: `self._emit`

##### `ProgressTracker.stage_tick(self, fraction, label=None)`
- 위치: `hygel_martini/hydrogel_builder/config_params/read_json.py:100`
- 종류: method
- 역할: `stage tick` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `self.current`
- 주요 호출: `self._emit`

##### `ProgressTracker.end_stage(self, label=None)`
- 위치: `hygel_martini/hydrogel_builder/config_params/read_json.py:109`
- 종류: method
- 역할: `end stage` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `self.stage_weight, self.current`
- 주요 호출: `self._emit`

##### `_load_base_parameters()`
- 위치: `hygel_martini/hydrogel_builder/config_params/read_json.py:129`
- 종류: function, private/internal
- 역할: Loads base parameters like atom masses from the main ITP file. Also prepares a deduplicated list of ITP files for the topology.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 예외/검증: `FileNotFoundError(f"'base_itp_file'에 지정된 '{base_itp}' 파일을 찾을 수 없습니다.") ; ValueError(f"'{base_itp}' 파일에서 원자 타입(atomtypes)을 로드할 수 없습니다.")`
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.get_param, Config.set_runtime, FileNotFoundError, ValueError, final_itp_list.append, glob.glob, os.path.abspath, os.path.isdir, os.path.isfile, os.path.join, read_atom_types, sim_params.get`

##### `_validate_config()`
- 위치: `hygel_martini/hydrogel_builder/config_params/read_json.py:175`
- 종류: function, private/internal
- 역할: `validate config` 검증 helper입니다. 입력 일관성, tool availability, template/topology 조건을 확인합니다.
- 반환: 명시적 return 2개. 예: `True ; False`
- 예외/검증: `ValueError("'hydrogel_components.backbone_definitions.BACKBONES'가 비어있습니다.") ; ValueError(f'Linker 템플릿 검증 실패: {exc}') ; ValueError(f'Monomer 템플릿 검증 실패: {exc}')`
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.get_param, Config.set_runtime, ValueError, load_linker_templates, load_monomer_templates`

##### `execute_mode()`
- 위치: `hygel_martini/hydrogel_builder/config_params/read_json.py:211`
- 종류: function
- 역할: Dispatch the configured top-level execution mode.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Config.debug_log, Config.disable_debug_logging, Config.enable_debug_logging, Config.get_param, Config.set_runtime, ProgressTracker, _execute_all_mode, _load_base_parameters, _validate_config, datetime.now, datetime.now.strftime, os.makedirs, os.path.join, progress.advance, sim_params_for_debug.get, sys.exit`

##### `_run_packing_step(step_name, base_structure_gro, molecules_to_add, final_output_gro, sim_params)`
- 위치: `hygel_martini/hydrogel_builder/config_params/read_json.py:255`
- 종류: function, private/internal
- 역할: Run one Packmol stage and return the resulting GRO path.
- 반환: 명시적 return 1개. 예: `(result_gro, success)`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 외부 command/subprocess 실행 가능
- 주요 호출: `packer.pack_system_with_molecules, sim_params.get`

##### `_compute_total_charge(itp_files_list, molecule_counts_dict)`
- 위치: `hygel_martini/hydrogel_builder/config_params/read_json.py:274`
- 종류: function, private/internal
- 역할: Estimate the system charge from ITP definitions and molecule counts.
- 반환: 명시적 return 2개. 예: `total_charge if found else None ; None`
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.get_runtime, bead.get, definition.get, definitions.get, definitions.update, molecule_counts_dict.items, read_itp_definitions`

##### `_perform_geo_opt_step(step_name, base_gro_file, output_dir, itp_files_list, molecule_counts_dict, sim_params)`
- 위치: `hygel_martini/hydrogel_builder/config_params/read_json.py:302`
- 종류: function, private/internal
- 역할: Run one GROMACS energy-minimization stage. The helper chooses a Coulomb model conservatively: - explicit config overrides always win, - early neutral stages can force ``Cut-off`` for speed, and - otherwise the function falls back to a charge-aware heuristic.
- 반환: 명시적 return 2개. 예: `optimized_gro if optimized_gro else base_gro_file ; base_gro_file`
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기, 외부 command/subprocess 실행 가능, 객체/class/global attribute 갱신
- 주요 대입: `mdp_overrides['coulombtype'], mdp_overrides['rcoulomb'], mdp_overrides['rvdw']`
- 주요 호출: `Config.get_param, _compute_total_charge, entry.get, final_itp_list.append, geo_opt_cfg.get, mdp_overrides.get, molecule_counts_dict.get, os.makedirs, os.path.join, run_geo_opt, sim_params.get, topology_updater.create_system_topology, topology_updater.update_topology_molecules`

##### `_merge_world_and_itps(world, extra_itps, merged_itp_path, moleculetype_name='MERGED')`
- 위치: `hygel_martini/hydrogel_builder/config_params/read_json.py:405`
- 종류: function, private/internal
- 역할: World 기반 구조와 추가 ITP 파일을 단일 ITP로 병합합니다. - World는 write_combined_itp로 기록 - 외부 ITP는 moleculetype 단위로 독립적이므로 인덱스 재배치 없이 그대로 이어붙임 (하나의 파일에 여러 moleculetype을 담는 목적)
- 반환: 명시적 return 1개. 예: `merged_itp_path`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `f.read, f.read.rstrip, format, fout.write, os.path.isfile, os.remove, write_combined_itp`

##### `_perform_dynamic_crosslinking(output_dir)`
- 위치: `hygel_martini/hydrogel_builder/config_params/read_json.py:434`
- 종류: function, private/internal
- 역할: Connect linker stubs to true backbone ends rather than arbitrary beads.
- 반환: 명시적 return 5개. 예: `sim_params.get('default_dynamic_crosslink_bond', {'bond_funct': 1, 'bond_c0': 0.25, 'bond_c1': 5000}) ; stub.stub_bond_params ; entry`
- 부작용 단서: Config/runtime state 접근, World/Attributes topology registry 접근 또는 mutation, 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Bond, Config.get_param, World.Atoms.values, _resolve_bond_params, assignments.get, bond_params.get, collect_backbone_ends, debug_f.write, entry.get, format, getattr, group_linker_stubs, os.path.join, plan_dynamic_crosslinks, sim_params.get, success.append`

##### `_execute_all_mode()`
- 위치: `hygel_martini/hydrogel_builder/config_params/read_json.py:549`
- 종류: function, private/internal
- 역할: builder의 full mode 본체입니다. backbone-only 생성, dynamic crosslink, hydrogel detail 확장, 단계별 geo-opt, add_polymer/add_molecule/add_water/add_small_ion, 최종 topology/GRO 작성을 순차 처리합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 예외/검증: `ValueError('polymer_source_file이 지정되지 않았습니다. generate 모드이거나 monomer_definitions가 있으면 자동 생성되어야 합니다.') ; ValueError(f"Invalid 'water_bead_type': {water_bead_type}. No templates found.")`
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `sim_params['box_size_nm'], sim_params['box_lengths_nm'], poly_params['polymer_output_gro_filename'], poly_params['polymer_output_itp_filename'], poly_params['polymer_source_file'], molecule_counts_for_top[poly_params['molecule_name']], mol_params['num_molecules'], add_series_params['add_molecule']['num_molecules'], molecule_counts_for_top[watername], mol_params['molecule_name']` ...
- 주요 호출: `Config.debug_log, Config.get_param, Config.get_runtime, Config.set_runtime, ValueError, WATER_GRO_TEMPLATES.get, WATER_ITP_TEMPLATES.get, _merge_world_and_itps, _perform_dynamic_crosslinking, _perform_geo_opt_step, _run_packing_step, _seed_all, add_series_params.get, add_small_ion.run_genion_for_neutralization, build_hydrogel.apply_coordinates_from_gro, build_hydrogel.build_backbone_only, build_hydrogel.finalize_hydrogel, calculate_water_molecules, extra_poly_itps.append, f.write, getattr, getattr.items, gro_template.format, ion_config.copy, ... (+43)`

### `hygel_martini/hydrogel_builder/core_utils/common/sequence_strategy.py`

Shared helpers for applying random/alternating/block strategies to template selections.
- 주요 import: `from dataclasses import dataclass, from typing import List, Optional, random`
- class 수: 2, 함수/메서드 수: 3

#### Classes

- `StrategyRecord` (hygel_martini/hydrogel_builder/core_utils/common/sequence_strategy.py:10)
  - decorators: `dataclass`
  - 주요 field/class var: `template: object, ratio: float, template_id: Optional[str]`
- `TemplateStrategyIterator` (hygel_martini/hydrogel_builder/core_utils/common/sequence_strategy.py:16)
  - 역할: Iterates over template records following the requested strategy. Supported strategies: random, alternating, block. Defaults to random.

#### Functions and methods

##### `TemplateStrategyIterator.__init__(self, records: List[StrategyRecord], strategy_cfg: Optional[dict]=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/common/sequence_strategy.py:22`
- 종류: method, private/internal
- 역할: `init` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `self.records, self.strategy_cfg, self.strategy, self.random_state, self._alternating_sequence, self._block_sequence, self._iterator_index`
- 주요 호출: `lower, random.Random, self._prepare_sequences, self.strategy_cfg.get`

##### `TemplateStrategyIterator._prepare_sequences(self)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/common/sequence_strategy.py:34`
- 종류: method, private/internal
- 역할: `prepare sequences` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `self.strategy, self._alternating_sequence, self._block_sequence`
- 주요 호출: `block.get, getattr, lookup.get, self.strategy_cfg.get, sequence.extend`

##### `TemplateStrategyIterator.next(self)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/common/sequence_strategy.py:61`
- 종류: method
- 역할: `next` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 6개. 예: `self.random_state.choices([rec.template for rec in self.records], weights=weights, k=1)[0] ; None ; self.records[0].template ; template.template`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `self._iterator_index`
- 주요 호출: `self.random_state.choices`

### `hygel_martini/hydrogel_builder/core_utils/common/utility.py`

Numerical helpers, geometry utilities, and text-normalization helpers.
- 주요 import: `from collections import defaultdict, heapq, os, shutil, subprocess, numba, numpy`
- class 수: 0, 함수/메서드 수: 11

#### Functions and methods

##### `interp3D(n, A, B)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/common/utility.py:16`
- 종류: function
- 역할: 두 3D 점 A와 B 사이에 n개의 점을 등간격으로 보간하여 생성합니다. 반환되는 점들은 A와 B 사이의 선분을 n+1개의 구간으로 나눈 점들입니다. Args: n (int): 생성할 점의 개수 A (np.array): 시작점 좌표 [x, y, z] B (np.array): 끝점 좌표 [x, y, z] Returns: np.array: n개의 보간된 점들을 담은 (n, 3) 크기의 배열
- 반환: 명시적 return 1개. 예: `np.array([A + i * (B - A) / (n + 1) for i in range(1, n + 1)])`
- 주요 호출: `np.array`

##### `rij(position_i, position_j, L)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/common/utility.py:33`
- 종류: function
- decorators: `numba.jit(fastmath=True, cache=True, nogil=True)`
- 역할: 주기 경계 조건(Periodic Boundary Conditions, PBC)을 고려하여 원자 i에서 원자 j로 향하는 벡터(r_ij)를 계산합니다. 가장 가까운 이미지(minimum image convention)를 사용합니다. Args: position_i (np.array): 원자 i의 좌표 position_j (np.array): 원자 j의 좌표 L (float): 시뮬레이션 박스의 한 변 길이 Returns: np.array: PBC를 고려한 i에서 j로의 벡터
- 반환: 명시적 return 1개. 예: `r_ij`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `r_ij[t]`
- 주요 호출: `np.round, np.zeros, numba.jit`

##### `dij_sq(position_i, position_j, L)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/common/utility.py:57`
- 종류: function
- decorators: `numba.jit(fastmath=True, cache=True, nogil=True)`
- 역할: PBC를 고려하여 두 원자 i와 j 사이의 거리의 제곱(d_ij^2)을 계산합니다. 제곱근 계산을 피하여 연산 속도를 높입니다. Args: position_i (np.array): 원자 i의 좌표 position_j (np.array): 원자 j의 좌표 L (float): 시뮬레이션 박스의 한 변 길이 Returns: float: PBC를 고려한 두 원자 사이의 거리의 제곱
- 반환: 명시적 return 1개. 예: `d_ij_sq`
- 주요 호출: `np.round, numba.jit`

##### `normal_to_3vectors(position_i, position_j, position_k, L)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/common/utility.py:79`
- 종류: function
- decorators: `numba.jit(fastmath=True, cache=True, nogil=True)`
- 역할: 세 점 i, j, k가 이루는 평면의 법선 벡터를 계산합니다. Args: position_i (np.array): 기준점 i의 좌표 position_j (np.array): 점 j의 좌표 position_k (np.array): 점 k의 좌표 L (float): 시뮬레이션 박스 길이 Returns: np.array: 평면의 단위 법선 벡터
- 반환: 명시적 return 1개. 예: `normal_cross`
- 주요 호출: `np.cross, np.sqrt, np.square, np.sum, numba.jit, rij`

##### `normal_tetrahedral_vector(position_1, position_2, position_3, position_4, L)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/common/utility.py:106`
- 종류: function
- decorators: `numba.jit(fastmath=True, cache=True, nogil=True)`
- 역할: 중심 원자(1)와 세 개의 이웃 원자(2, 3, 4)가 주어졌을 때, 사면체(tetrahedral) 구조에서 네 번째 결합이 향해야 할 방향 벡터를 계산합니다. Args: position_1 (np.array): 중심 원자의 좌표 position_2,3,4 (np.array): 이웃 원자들의 좌표 L (float): 시뮬레이션 박스 길이 Returns: np.array: 정사면체의 중심에서 꼭짓점으로 향하는 단위 벡터
- 반환: 명시적 return 1개. 예: `r_tetra`
- 주요 호출: `np.sqrt, np.square, np.sum, numba.jit, rij`

##### `not_self(i, obj)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/common/utility.py:132`
- 종류: function
- 역할: 결합(bond) 객체(obj)와 그 결합에 속한 원자(i) 하나를 입력받아, 그 결합에 속한 다른 원자를 반환하는 헬퍼 함수입니다. Args: i (Atom): 원자 객체 obj (Bond): 결합 객체 Returns: Atom: 결합의 상대방 원자 객체
- 반환: 명시적 return 2개. 예: `obj.bond_atom_2 ; obj.bond_atom_1`

##### `is_overlap(A, B, d, L)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/common/utility.py:151`
- 종류: function
- decorators: `numba.jit(fastmath=True, cache=True, nogil=True)`
- 역할: 점 A가 점들의 배열 B에 있는 어떤 점과 거리 d 미만으로 겹치는지 확인합니다. Args: A (np.array): 확인할 점의 좌표 B (np.array): 다른 점들의 좌표 배열 (N, 3) d (float): 겹침을 판단할 기준 거리 L (float): 시뮬레이션 박스 길이 Returns: bool: 겹치면 True, 겹치지 않으면 False
- 반환: 명시적 return 2개. 예: `True ; False`
- 주요 호출: `dij_sq, numba.jit`

##### `random_normal_vector(A, B, C, r, L)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/common/utility.py:182`
- 종류: function
- decorators: `numba.jit(fastmath=True, cache=True, nogil=True)`
- 역할: A-B-C로 연결된 구조에서 중심 원자 B에 대해, 두 결합(A-B, C-B)이 이루는 평면에 거의 수직인 방향으로 길이가 r인 무작위 벡터를 생성합니다. 곁사슬(side chain)을 생성할 때 사용됩니다. Args: A, B, C (np.array): 원자 A, B(중심), C의 좌표 r (float): 생성할 벡터의 길이 L (float): 시뮬레이션 박스 길이 Returns: np.array: 무작위 방향 벡터
- 반환: 명시적 return 1개. 예: `np.array([x1, y1, z1])`
- 주요 호출: `np.array, np.linalg.norm, np.random.random, np.sqrt, numba.jit, rij`

##### `find_minimum_distances(positions, box_length, top_n=10, cell_size=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/common/utility.py:223`
- 종류: function
- 역할: Return the smallest inter-particle distances using a simple cell list search. Args: positions (np.ndarray): (N, 3) 좌표 배열. box_length (float): 시뮬레이션 박스 길이 (정육면체 가정). top_n (int): 리포트할 최소 거리 쌍의 개수. cell_size (float, optional): 셀 리스트의 크기. 기본은 box_length/10. Returns: list[tuple]: (거리[nm], index_i, index_j) 튜플 리스트.
- 반환: 명시적 return 4개. 예: `results ; []`
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.get_runtime, cells.append, cells.get, cells.items, consider_pair, defaultdict, dij_sq, heapq.heappop, heapq.heappush, heapq.heapreplace, neighbor_offsets.append, np.asarray, np.ceil, np.clip, np.floor, np.floor.astype, np.max, np.min, np.mod, np.sqrt, progress.stage_tick, results.append, results.sort`

##### `find_minimum_distances.consider_pair(i, j)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/common/utility.py:270`
- 종류: nested helper
- 역할: `consider pair` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 주요 호출: `dij_sq, heapq.heappush, heapq.heapreplace`

##### `run_dos2unix_on_inputs(config_data)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/common/utility.py:322`
- 종류: function
- 역할: Normalize line endings for all configured input structure files. Args: config_data (dict): Fully merged configuration dictionary.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 외부 command/subprocess 실행 가능
- 주요 호출: `config_data.get, config_data.get.get, config_data.get.get.get, dst.write, f.startswith, files_to_process.append, files_to_process.extend, original.replace, original.replace.replace, os.path.exists, shutil.which, src.read, subprocess.run`

### `hygel_martini/hydrogel_builder/core_utils/generators/polymer_generator.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `os, from hygel_martini.hydrogel_builder.main_components.Universe import World, initialize_world, from hygel_martini.hydrogel_builder.main_components import Attributes, from hygel_martini.hydrogel_builder.core_utils.io import writer, from hygel_martini.hydrogel_builder.main_components.Polymer import Polymer`
- class 수: 1, 함수/메서드 수: 2

#### Classes

- `MockDNAsys` (hygel_martini/hydrogel_builder/core_utils/generators/polymer_generator.py:54)

#### Functions and methods

##### `generate_single_polymer_gro(p_mon_num: int, output_filename: str, mean_sep: float, random_seed: int=2024, include_chemical_detail: bool=True, include_angles: bool=True, moleculetype_name: str='HDGEL', polymer_config: dict | None=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/generators/polymer_generator.py:8`
- 종류: function
- 역할: 단일 고분자 사슬의 .gro 파일을 생성합니다. Args: p_mon_num (int): 고분자 사슬을 구성하는 단량체의 총 개수 (길이). output_filename (str): 생성될 .gro 파일의 이름. mean_sep (float): 비드 간의 평균 거리. random_seed (int): 고분자 구조 생성을 위한 무작위 시드. include_chemical_detail (bool): 곁사슬을 포함할지 여부. include_angles (bool): 각도 정보를 포함할지 여부.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation, 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `self.dna_atoms_list`
- 주요 호출: `Attributes.initialize, Polymer.configure, World, World.reset, initialize_world, os.path.splitext, pm.construct_angles, pm.construct_atoms, pm.construct_chemical_detail, world.make_polymer, writer.write_combined_itp, writer.write_to_gro`

##### `MockDNAsys.__init__(self)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/generators/polymer_generator.py:55`
- 종류: method, private/internal
- 역할: `init` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `self.dna_atoms_list`

### `hygel_martini/hydrogel_builder/core_utils/io/gro_parser.py`

Gro file parsing utilities.
- 주요 import: `from dataclasses import dataclass, from typing import List, numpy`
- class 수: 1, 함수/메서드 수: 1

#### Classes

- `GroAtom` (hygel_martini/hydrogel_builder/core_utils/io/gro_parser.py:12)
  - decorators: `dataclass`
  - 주요 field/class var: `index: int, residue_number: int, residue_name: str, atom_name: str, position: np.ndarray`

#### Functions and methods

##### `read_gro_atoms(path: str)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/io/gro_parser.py:20`
- 종류: function
- 역할: Parse a .gro file and return atom entries (box vector is ignored).
- 반환: 명시적 return 1개. 예: `atoms`
- 예외/검증: `ValueError(f"GRO line '{line.rstrip()}'가 예상되는 필드 수보다 적습니다.") ; ValueError(f"GRO line '{line.rstrip()}'에서 residue number를 파싱할 수 없습니다.") ; ValueError(f'GRO file {path} ended prematurely at atom {idx + 1}')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `GroAtom, ValueError, atoms.append, handle.readline, handle.readline.strip, line.rstrip, line.split, line.strip, np.array`

### `hygel_martini/hydrogel_builder/core_utils/io/martini_parser.py`

Martini/GROMACS ITP parser입니다. moleculetype 내부의 atoms/bonds/constraints/pairs/exclusions/virtual_sites/restraints/dihedrals/impropers 등을 dict로 추출합니다.
- 주요 import: `re`
- class 수: 0, 함수/메서드 수: 3

#### Functions and methods

##### `read_atom_types(itp_file_path)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/io/martini_parser.py:4`
- 종류: function
- 역할: Parses an .itp file and extracts only the [ atomtypes ] section. Returns a dictionary mapping atom type names to their mass.
- 반환: 명시적 return 2개. 예: `atom_types ; {}`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `atom_types[atom_type_name]`
- 주요 호출: `line.split, line.split.rstrip, line.startswith, line.strip, match.group, match.group.lower, re.match`

##### `read_itp_definitions(itp_file_path, atom_type_masses=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/io/martini_parser.py:41`
- 종류: function
- 역할: ITP의 moleculetype 단위 정의를 파싱해 beads, bonds, constraints, angles, dihedrals, impropers, pairs, exclusions, virtual sites 등 builder가 사용하는 통합 dict로 반환합니다.
- 반환: 명시적 return 5개. 예: `definitions`
- 예외/검증: `ValueError(f"Mass for atom type '{atom_type}' in molecule '{current_molecule['name']}' from file '{itp_file_path}' could not be determined. Please ensure it is defined in the ba...`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `definitions[molecule_name]`
- 주요 호출: `ValueError, current_molecule.append, current_molecule.setdefault, line.split, line.split.rstrip, line.startswith, line.strip, match.group, match.group.lower, other.setdefault, other.setdefault.append, p.lower, re.match, sec_lower.endswith, sec_lower.startswith, sec_name.lower, section.startswith, stash_raw, x.lower`

##### `read_itp_definitions.stash_raw(sec_name, line)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/io/martini_parser.py:67`
- 종류: nested helper
- 역할: `stash raw` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 4개이지만 값 없는 return 경로가 중심입니다.
- 주요 호출: `current_molecule.setdefault, other.setdefault, other.setdefault.append, sec_lower.endswith, sec_lower.startswith, sec_name.lower`

### `hygel_martini/hydrogel_builder/core_utils/io/writer.py`

World registry를 `.xyz`, `.gro`, `.itp` 및 rich section을 포함한 combined ITP로 직렬화합니다.
- 주요 import: `numpy, os`
- class 수: 0, 함수/메서드 수: 4

#### Functions and methods

##### `write_to_xyz(object, filename='xyz.xyz')`
- 위치: `hygel_martini/hydrogel_builder/core_utils/io/writer.py:11`
- 종류: function
- 역할: 시스템의 원자 좌표를 간단한 .xyz 파일 형식으로 저장합니다. 시각화 프로그램에서 구조를 빠르게 확인하는 데 유용합니다. Args: object (World): 원자 정보를 포함하는 World 객체 filename (str): 저장할 파일 이름
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `f.write, format, np.random.randint, os.makedirs, os.path.dirname`

##### `write_to_gro(object, filename='gromacs.gro')`
- 위치: `hygel_martini/hydrogel_builder/core_utils/io/writer.py:34`
- 종류: function
- 역할: 시스템 정보를 GROMACS .gro 파일 형식으로 저장합니다. Args: object (World): 시스템 정보를 포함하는 World 객체 DNA (DNAimport): DNA 정보 객체 (여기서는 사용되지 않음) filename (str): 저장할 파일 이름
- 반환: 명시적 return 1개. 예: `1`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `f.write, format, getattr, np.any, np.array, os.makedirs, os.path.dirname`

##### `write_to_itp(object, filename='gromacs.itp', moleculetype_name='HDGEL')`
- 위치: `hygel_martini/hydrogel_builder/core_utils/io/writer.py:70`
- 종류: function
- 역할: 시스템의 토폴로지 정보를 GROMACS .itp 파일 형식으로 저장합니다. 이 파일은 분자 내 상호작용(결합, 각도 등)을 정의합니다.
- 반환: 명시적 return 2개. 예: `1`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `c.get, cm.get, ex.get, extras.get, extras.items, f.write, format, getattr, join, os.makedirs, os.path.dirname, p_def.get, pol.get, r.get, sec.endswith, vs.get, vs_by_sec.items, vs_by_sec.setdefault, vs_by_sec.setdefault.append`

##### `write_combined_itp(world, filename, moleculetype_name)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/io/writer.py:245`
- 종류: function
- 역할: World.Atoms/Bonds/Angles/Dihedrals/OtherSections를 하나의 GROMACS ITP moleculetype로 작성합니다.
- 반환: 명시적 return 2개. 예: `1`
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Config.debug_log, Config.get_param, Config.get_param.get, _write_section, extras.get, extras.items, f.write, format, getattr, join, os.makedirs, os.path.dirname, r.get, validate_and_filter_other_sections, vs.get, vs_by_sec.items, vs_by_sec.setdefault, vs_by_sec.setdefault.append`

### `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py`

isotropic/anisotropic mode에서 중간 cell EM과 blueprint tiling을 수행하는 특수 builder입니다.
- 주요 import: `from dataclasses import replace, from typing import Dict, List, Tuple, Optional, Sequence, os, json, random, numpy, from hygel_martini.hydrogel_builder.core_utils.io.gro_parser import read_gro_atoms, from hygel_martini.hydrogel_builder.core_utils.io.writer import write_to_gro, write_combined_itp, from hygel_martini.hydrogel_builder.core_utils.layout.layout_executor import LayoutBlueprint, build_atom_blueprint, from hygel_martini.hydrogel_builder.core_utils.layout.proto_layout import LayoutPlan, LayoutCell, LinkPlacement, from hygel_martini.hydrogel_builder.core_utils.layout.proto_layout import MEDIUM_ACTIVE_INDICES, POLYMER_POSITIONS, ORIENTATION_MAP, from hygel_martini.hydrogel_builder.core_utils.layout.proto_populator import populate_hydrogel_from_blueprint, ...`
- class 수: 0, 함수/메서드 수: 17

#### Functions and methods

##### `_linear_index(ix: int, iy: int, iz: int, repeats: Tuple[int, int, int])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:23`
- 종류: function, private/internal
- 역할: `linear index` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `ix * ny * nz + iy * nz + iz`

##### `_normalize(vec: np.ndarray)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:28`
- 종류: function, private/internal
- 역할: `normalize` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 2개. 예: `vec / norm ; vec`
- 주요 호출: `np.linalg.norm`

##### `_axis_rotation_matrix(axis: str)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:35`
- 종류: function, private/internal
- 역할: `axis rotation matrix` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 3개. 예: `np.eye(3, dtype=float) ; np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float) ; np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)`
- 주요 호출: `np.array, np.eye`

##### `_apply_rotation(vec: np.ndarray, rot: np.ndarray)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:47`
- 종류: function, private/internal
- 역할: `apply rotation` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `vec @ rot.T`

##### `_get_anisotropy_axis()`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:51`
- 종류: function, private/internal
- 역할: 설정 객체를 읽거나 runtime state를 갱신하는 helper입니다.
- 반환: 명시적 return 1개. 예: `axis`
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.get_param, Config.get_param.get`

##### `_normalize_linker_axes(linker_axes: Optional[Sequence[str]])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:62`
- 종류: function, private/internal
- 역할: `normalize linker axes` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `cleaned[:2]`
- 주요 호출: `cleaned.append`

##### `_squeeze_positions_to_cube(positions: np.ndarray, cube_side: float)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:81`
- 종류: function, private/internal
- 역할: `squeeze positions to cube` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `centered * scale`
- 주요 호출: `np.maximum, positions.max, positions.min`

##### `_linker_total_length(entry: Dict, fallback: float, override: float | None=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:90`
- 종류: function, private/internal
- 역할: `linker total length` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `float(total) ; float(override)`
- 주요 호출: `bond.get, definition.get, entry.get, ext.get`

##### `_write_linker_debug(path: str, blueprint: LayoutBlueprint, positions_pre: np.ndarray, positions_post: np.ndarray, axes: List[str], medium_origin: np.ndarray, box_vector: np.ndarray, cube_side: float, small_edge: float, linker_len: float, base_size: np.ndarray, cell_vector: np.ndarray, scale: np.ndarray, mins: np.ndarray, maxs: np.ndarray)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:104`
- 종류: function, private/internal
- 역할: `write linker debug` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `axis_dir.tolist, base_size.tolist, box_vector.tolist, cell_vector.tolist, chain.metadata.get, chain.metadata.get.tolist, chain_atoms.get, chain_atoms.setdefault, chain_atoms.setdefault.append, definition.get, ext.get, get, json.dump, maxs.tolist, medium_origin.tolist, mins.tolist, np.array, np.dot, np.linalg.norm, payload.append, pos.tolist, pos_by_bead_post.get, pos_by_bead_pre.get, positions_post.tolist, ... (+6)`

##### `_pick_boundary_atoms(positions: np.ndarray)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:220`
- 종류: function, private/internal
- 역할: `pick boundary atoms` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `picked`
- 주요 호출: `np.argmin, np.array, np.linalg.norm, np.maximum, picked.append, positions.max, positions.min`

##### `_write_posre_itp(path: str, atom_indices: List[int], fc: float)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:240`
- 종류: function, private/internal
- 역할: `write posre itp` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `f.write`

##### `_write_system_top(path: str, itp_path: str, posre_path: str, base_itp: str | None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:248`
- 종류: function, private/internal
- 역할: `write system top` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `f.write, os.path.abspath`

##### `_build_world_from_blueprint(blueprint: LayoutBlueprint, box_vector: np.ndarray, output_dir: str, mean_sep: float, construct_proto_bonds: bool=True)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:262`
- 종류: function, private/internal
- 역할: `build world from blueprint` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 1개. 예: `(world, hd)`
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `World.mean_sep, World.box_vector, World.box_length`
- 주요 호출: `Attributes.initialize, World, World.reset, hd.construct_bonds, np.array, np.max, populate_hydrogel_from_blueprint, world.make_hydrogel`

##### `_run_medium_cell_em(blueprint: LayoutBlueprint, box_vector: np.ndarray, fixed_atom_indices: List[int], out_dir: str, sim_params: Dict, temp_bonds: List[Tuple[int, int, Dict]] | None=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:282`
- 종류: function, private/internal
- 역할: `run medium cell em` 실행 helper입니다. workflow 단계나 외부 command/script를 실행 또는 위임합니다.
- 반환: 명시적 return 1개. 예: `[atom.position for atom in atoms]`
- 예외/검증: `FileNotFoundError(f'medium-cell EM output missing: {em_path}')`
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation, 파일/디렉터리/topology 출력 또는 읽기, 외부 command/subprocess 실행 가능, 객체/class/global attribute 갱신
- 주요 대입: `backbone_bond_snapshot[id(bond)]`
- 주요 호출: `Attributes.Bond, FileNotFoundError, World.Bonds.items, World.Bonds.values, _build_world_from_blueprint, _write_posre_itp, _write_system_top, backbone_bond_snapshot.get, id, os.makedirs, os.path.exists, os.path.join, params.get, read_gro_atoms, run_geo_opt, sim_params.get, sim_params.get.get, temp_keys.append, write_combined_itp, write_to_gro`

##### `_offset_blueprint(blueprint: LayoutBlueprint, atom_offset: int, chain_offset: int)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:359`
- 종류: function, private/internal
- 역할: `offset blueprint` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `LayoutBlueprint(atoms=atoms, chains=chains)`
- 주요 호출: `LayoutBlueprint, atoms.append, chains.append, replace`

##### `build_isotropic_blueprint(proto_plan, backbone_defs: List[Dict], linker_defs: List[Dict], repeats: Tuple[int, int, int], backbone_strategy: Dict, linker_strategy: Dict, linker_library, output_dir: str, sim_params: Dict)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:373`
- 종류: function
- 역할: `build isotropic blueprint` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 6개. 예: `LayoutBlueprint(atoms=all_atoms, chains=all_chains) ; delta ; res_name in target_names ; any((name in target_names for name in res_name)) ; small_edge * (0.5 + idx_val)`
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `first_vec[axis_index[primary_axis]], second_vec[axis_index[primary_axis]], second_vec[axis_index[secondary_axis]], sel_positions[sel_idx][axis_idx], center_local[axis_index[primary_axis]], axis_dir[axis_index[axis_choice]], atom.extra['pre_compress_position'], positions_post[atom_idx][axis_idx], unique_chains[key], center_local[0]` ...
- 주요 호출: `Config.debug_log, LayoutBlueprint, LayoutCell, LayoutPlan, LinkPlacement, _axis_center, _build_world_from_blueprint, _linear_index, _linker_total_length, _matches_any, _normalize, _normalize_linker_axes, _offset_blueprint, _pbc_delta, _pbc_distance_sq, _squeeze_positions_to_cube, all_atoms.extend, all_chains.extend, build_atom_blueprint, cells.append, definition.get, entry.get, get, getattr, ... (+33)`

##### `build_isotropic_blueprint._pbc_delta(vec)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/isotropic_builder.py:656`
- 종류: nested helper, private/internal
- 역할: `pbc delta` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `delta`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `np.round, vec.copy`

### `hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py`

layout plan을 실제 atom blueprint로 변환하는 좌표 배치/회전 유틸리티입니다.
- 주요 import: `from dataclasses import dataclass, from typing import Any, Dict, List, numpy, from hygel_martini.hydrogel_builder.core_utils.layout.proto_layout import LayoutPlan, LayoutCell, LinkPlacement`
- class 수: 5, 함수/메서드 수: 10

#### Classes

- `InstantiatedChain` (hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py:14)
  - decorators: `dataclass`
  - 주요 field/class var: `positions: np.ndarray, definition: Dict[str, Any], metadata: Dict[str, Any], template: Any | None`
- `InstantiatedLayout` (hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py:22)
  - decorators: `dataclass`
  - 주요 field/class var: `backbone_segments: List[InstantiatedChain], linker_segments: List[InstantiatedChain]`
- `AtomBlueprint` (hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py:28)
  - decorators: `dataclass`
  - 주요 field/class var: `chain_type: str, chain_index: int, bead_index: int, position: np.ndarray, component_id: str, atom_name: str, atom_type: str, residue_name: str, residue_number: int, charge_group_number: int, mass: float, charge: float` ...
- `ChainBlueprint` (hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py:46)
  - decorators: `dataclass`
  - 주요 field/class var: `chain_type: str, chain_index: int, component_id: str, definition: Dict[str, Any], atom_indices: List[int], metadata: Dict[str, Any]`
- `LayoutBlueprint` (hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py:56)
  - decorators: `dataclass`
  - 주요 field/class var: `atoms: List[AtomBlueprint], chains: List[ChainBlueprint]`

#### Functions and methods

##### `_center_positions(positions: np.ndarray)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py:61`
- 종류: function, private/internal
- 역할: `center positions` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `positions - centroid`
- 주요 호출: `np.mean`

##### `_rotate_between_vectors(vectors: np.ndarray, source: np.ndarray, target: np.ndarray)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py:66`
- 종류: function, private/internal
- 역할: `rotate between vectors` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 5개. 예: `vectors @ R.T ; vectors ; -vectors`
- 주요 호출: `np.allclose, np.arccos, np.array, np.clip, np.cos, np.cross, np.dot, np.eye, np.linalg.norm, np.sin`

##### `_rotate_from_xaxis(vectors: np.ndarray, target: np.ndarray)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py:92`
- 종류: function, private/internal
- 역할: `rotate from xaxis` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 5개. 예: `vectors @ R.T ; vectors ; np.column_stack((-vectors[:, 0], vectors[:, 1], vectors[:, 2]))`
- 주요 호출: `np.allclose, np.arccos, np.array, np.clip, np.column_stack, np.cos, np.cross, np.dot, np.eye, np.linalg.norm, np.sin`

##### `_alignment_basis(axis: np.ndarray)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py:115`
- 종류: function, private/internal
- 역할: `alignment basis` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `np.column_stack((x_axis, y_axis, z_axis))`
- 주요 호출: `np.array, np.column_stack, np.cross, np.dot, np.linalg.norm`

##### `instantiate_backbone(cell: LayoutCell, proto_positions: np.ndarray)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py:143`
- 종류: function
- 역할: `instantiate backbone` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `InstantiatedChain(positions=positions, definition=cell.backbone_definition, metadata=metadata)`
- 주요 호출: `InstantiatedChain, _center_positions, _rotate_between_vectors, cell.metadata.get, cell.metadata.items, metadata.update`

##### `instantiate_linker(layout_plan: LayoutPlan, link: LinkPlacement, proto_positions: np.ndarray)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py:165`
- 종류: function
- 역할: `instantiate linker` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `InstantiatedChain(positions=positions, definition=definition, metadata=embed_metadata, template=template)`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `InstantiatedChain, _alignment_basis, _center_positions, _rotate_from_xaxis, definition.get, defn_body.get, embed_metadata.update, getattr, hasattr, library.lookup.get, link.metadata.copy, metadata.get, np.array, np.linalg.norm`

##### `instantiate_layout(layout_plan: LayoutPlan)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py:222`
- 종류: function
- 역할: `instantiate layout` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `InstantiatedLayout(backbone_segments=backbone_segments, linker_segments=linker_segments)`
- 주요 호출: `InstantiatedLayout, backbone_segments.append, instantiate_backbone, instantiate_linker, linker_segments.append`

##### `_backbone_atom_params(component_entry: Dict[str, Any], bead_index: int)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py:239`
- 종류: function, private/internal
- 역할: `backbone atom params` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `{'atom_name': atom_name, 'atom_type': atom_type, 'residue_name': residue_name, 'residue_number': residue_number, 'charge_group_number': cgnr, 'mass': mass, 'charge': charge}`
- 주요 호출: `component_entry.get, definition.get`

##### `_linker_atom_params(component_entry: Dict[str, Any], bead_index: int)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py:266`
- 종류: function, private/internal
- 역할: `linker atom params` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `{'atom_name': atom_name, 'atom_type': atom_type, 'residue_name': residue_name, 'residue_number': residue_number, 'charge_group_number': cgnr, 'mass': mass, 'charge': charge}`
- 주요 호출: `bead_def.get, component_entry.get, definition.get`

##### `build_atom_blueprint(layout_plan: LayoutPlan, backbone_defs: List[Dict[str, Any]])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/layout_executor.py:294`
- 종류: function
- 역할: `build atom blueprint` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 1개. 예: `LayoutBlueprint(atoms=atoms, chains=chains)`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `params['residue_name']`
- 주요 호출: `AtomBlueprint, ChainBlueprint, LayoutBlueprint, _backbone_atom_params, _linker_atom_params, atom_indices.append, atoms.append, chain.metadata.get, chains.append, component_entry.get, definition.get, entry.get, ext.get, ext.items, get, getattr, instantiate_layout, np.array, np.dot, np.linalg.norm, np.zeros, original_stub_def.get, raw_def.get`

### `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py`

backbone/linker 정의와 sequence strategy에서 proto unit cell plan을 만듭니다.
- 주요 import: `from dataclasses import dataclass, from typing import List, Dict, Any, Sequence, Tuple, Optional, random, numpy, from hygel_martini.hydrogel_builder.core_utils.templates.linker_loader import LinkerTemplateLibrary`
- class 수: 3, 함수/메서드 수: 19

#### Classes

- `ProtoChain` (hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:52)
  - decorators: `dataclass`
  - 주요 field/class var: `positions: np.ndarray, types: List[Tuple[str, str]], length: float, raw_length: float`
- `ProtoPlan` (hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:60)
  - decorators: `dataclass`
  - 주요 field/class var: `segment_length: int, proto_backbone: ProtoChain, proto_linker: ProtoChain, box_margin: float, cell_vector: np.ndarray, medium_size: np.ndarray, small_size: np.ndarray, mean_sep: float, bond_lookup: Dict[Tuple[str, str], Dict[str, Any]], sequence_factory: 'BackboneSequenceFactory', linker_library: LinkerTemplateLibrary | None, linker_span_lookup: Dict[str, float] | None`
- `BackboneSequenceFactory` (hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:183)
  - 역할: Generates backbone sequences/rescaled coordinates per placement, honoring the requested strategy.

#### Functions and methods

##### `_weighted_average(lengths: Sequence[float], weights: Sequence[float])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:14`
- 종류: function, private/internal
- 역할: `weighted average` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `float(sum((l * w for l, w in zip(lengths, weights))) / total_weight) ; float(np.mean(lengths)) if lengths else 0.0`
- 주요 호출: `np.mean`

##### `_normalize_linker_axes(linker_axes: Optional[Sequence[str]])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:21`
- 종류: function, private/internal
- 역할: `normalize linker axes` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `cleaned[:2]`
- 주요 호출: `cleaned.append`

##### `_compute_linker_length(definition: Dict[str, Any])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:40`
- 종류: function, private/internal
- 역할: `compute linker length` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `float(total)`
- 주요 호출: `bond.get, definition.get, ext.get`

##### `ProtoPlan.box_vector(self, repeats: Tuple[int, int, int])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:74`
- 종류: method
- 역할: `box vector` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `self.cell_vector * np.array(repeats, dtype=np.float64)`
- 주요 호출: `np.array`

##### `_build_bond_lookup(bond_rules: Optional[List[Dict[str, Any]]], fallback: float)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:78`
- 종류: function, private/internal
- 역할: `build bond lookup` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 2개. 예: `lookup ; {}`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `lookup[key]`
- 주요 호출: `rule.get`

##### `_next_backbone_entry(strategy: Dict[str, Any], backbones: List[Dict[str, Any]])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:92`
- 종류: function, private/internal
- 역할: `next backbone entry` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 주요 호출: `block.get, entry.get, get, get.lower, index.get, random.choices, sequence.extend`

##### `_build_backbone_sequence(length: int, strategy: Dict[str, Any], backbones: List[Dict[str, Any]])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:118`
- 종류: function, private/internal
- 역할: `build backbone sequence` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 2개. 예: `[next(gen) for _ in range(length)] ; []`
- 주요 호출: `_next_backbone_entry`

##### `_build_backbone_positions(sequence: List[Dict[str, Any]], bond_lookup: Dict[Tuple[str, str], Dict[str, Any]], fallback: float)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:127`
- 종류: function, private/internal
- 역할: `build backbone positions` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 2개. 예: `(np.array(positions, dtype=np.float64), total) ; (np.zeros((0, 3), dtype=np.float64), 0.0)`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `bond_lookup.get, current.copy, np.array, np.zeros, positions.append, rule.get, sequence.get`

##### `_chain_geometry_from_sequence(sequence: List[Dict[str, Any]], segment_length: int, mean_sep: float, bond_lookup: Dict[Tuple[str, str], Dict[str, Any]])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:150`
- 종류: function, private/internal
- 역할: `chain geometry from sequence` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `(positions, float(raw_length), float(scaled_length))`
- 주요 호출: `_build_backbone_positions, np.zeros`

##### `_resolve_block_pattern(strategy: Dict[str, Any], backbones: List[Dict[str, Any]])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:168`
- 종류: function, private/internal
- 역할: `resolve block pattern` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 2개. 예: `pattern or backbones ; backbones`
- 주요 호출: `block.get, entry.get, get, index.get, pattern.extend`

##### `BackboneSequenceFactory.__init__(self, segment_length: int, backbone_definitions: List[Dict[str, Any]], strategy: Optional[Dict[str, Any]], mean_sep: float, bond_lookup: Dict[Tuple[str, str], Dict[str, Any]], prototype_sequence: Optional[List[Dict[str, Any]]]=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:188`
- 종류: method, private/internal
- 역할: `init` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `self.segment_length, self.definitions, self.mean_sep, self.bond_lookup, self.num_proto_beads, self.strategy, self.strategy_name, self.weights, self.pattern, self.pattern_offset` ...
- 주요 호출: `_resolve_block_pattern, copy, entry.get, self.strategy.get, self.strategy.get.lower`

##### `BackboneSequenceFactory._random_sequence(self)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:214`
- 종류: method, private/internal
- 역할: `random sequence` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `random.choices(self.definitions, weights=self.weights, k=self.num_proto_beads) ; []`
- 주요 호출: `random.choices`

##### `BackboneSequenceFactory.next_sequence(self)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:219`
- 종류: method
- 역할: `next sequence` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 5개. 예: `self._random_sequence() ; [] ; sequence ; list(self._long_random_cache)`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `self.pattern_offset, self._long_random_cache`
- 주요 호출: `self._random_sequence`

##### `BackboneSequenceFactory.instantiate(self, enforce_unique: bool=False, used_signatures: Optional[set]=None, max_attempts: int=50)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:239`
- 종류: method
- 역할: `instantiate` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `(sequence, positions, raw_length, scaled_length)`
- 주요 호출: `_chain_geometry_from_sequence, entry.get, self.next_sequence, used_signatures.add`

##### `BackboneSequenceFactory.definition_count(self)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:262`
- 종류: method, property
- decorators: `property`
- 역할: `definition count` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `len(self.definitions)`

##### `build_proto_backbone(segment_length: int, backbone_definitions: List[Dict[str, Any]], mean_sep: float, strategy: Optional[Dict[str, Any]]=None, bond_rules: Optional[List[Dict[str, Any]]]=None, bond_lookup: Optional[Dict[Tuple[str, str], Dict[str, Any]]]=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:266`
- 종류: function
- 역할: `build proto backbone` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 1개. 예: `ProtoChain(positions=positions, types=types, length=float(scaled_length), raw_length=float(raw_length))`
- 예외/검증: `ValueError('segment_length must be >= 2')`
- 주요 호출: `ProtoChain, ValueError, _build_backbone_sequence, _build_bond_lookup, _chain_geometry_from_sequence, entry.get, entry.get.get, types.append`

##### `build_proto_linker(linker_definitions: List[Dict[str, Any]], strategy: Optional[Dict[str, Any]]=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:292`
- 종류: function
- 역할: `build proto linker` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 1개. 예: `ProtoChain(positions=np.array(positions, dtype=np.float64), types=types, length=avg_length, raw_length=avg_length)`
- 예외/검증: `ValueError('linker_definitions must not be empty')`
- 주요 호출: `ProtoChain, ValueError, _compute_linker_length, beads.get, definition.get, entry.get, get, get.lower, np.array, positions.append, random.choices, selected.get, types.append`

##### `prepare_proto_plan(segment_length: int, mean_sep: float, backbone_defs: List[Dict[str, Any]], linker_defs: List[Dict[str, Any]], box_margin: float, backbone_strategy: Optional[Dict[str, Any]]=None, linker_strategy: Optional[Dict[str, Any]]=None, bond_rules: Optional[List[Dict[str, Any]]]=None, linker_library: LinkerTemplateLibrary | None=None, linker_axes: Optional[Sequence[str]]=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:327`
- 종류: function
- 역할: `prepare proto plan` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 1개. 예: `ProtoPlan(segment_length=segment_length, proto_backbone=proto_backbone, proto_linker=proto_linker, box_margin=margin, cell_vector=cell_vector, medium_size=medium_size, small_siz...`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `linker_span_lookup[record.template.id], base_size[0], base_size[1], base_size[2]`
- 주요 호출: `BackboneSequenceFactory, ProtoChain, ProtoPlan, _build_bond_lookup, _normalize_linker_axes, _weighted_average, build_proto_backbone, build_proto_linker, definition.get, entry.get, id_lookup.get, linker_span_values.append, np.array, np.sqrt, proto_sequence.append, span_weights.append`

##### `describe_proto_summary(segment_length: int, mean_sep: float, backbone_defs: List[Dict[str, Any]], linker_defs: List[Dict[str, Any]], **kwargs)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_builder.py:408`
- 종류: function
- 역할: `describe proto summary` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `{'backbone_length': proto.proto_backbone.length, 'backbone_length_raw': proto.proto_backbone.raw_length, 'linker_length': proto.proto_linker.length, 'num_backbone_beads': proto....`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `linker_spans[entry.get('id')]`
- 주요 호출: `bond.get, definition.get, entry.get, ext.get, prepare_proto_plan`

### `hygel_martini/hydrogel_builder/core_utils/layout/proto_layout.py`

proto plan을 반복 cell layout으로 확장하고 backbone/linker placement를 계산합니다.
- 주요 import: `from dataclasses import dataclass, from typing import Dict, Any, List, Tuple, Optional, Sequence, random, numpy, from hygel_martini.hydrogel_builder.core_utils.layout.proto_builder import ProtoPlan, from hygel_martini.hydrogel_builder.core_utils.templates.linker_loader import LinkerTemplateLibrary`
- class 수: 3, 함수/메서드 수: 9

#### Classes

- `LayoutCell` (hygel_martini/hydrogel_builder/core_utils/layout/proto_layout.py:48)
  - decorators: `dataclass`
  - 주요 field/class var: `origin: np.ndarray, direction: np.ndarray, backbone_definition: Dict[str, Any], cell_index: Tuple[int, int, int], metadata: Dict[str, Any] | None`
- `LinkPlacement` (hygel_martini/hydrogel_builder/core_utils/layout/proto_layout.py:57)
  - decorators: `dataclass`
  - 주요 field/class var: `anchor_position: np.ndarray, axis_direction: np.ndarray, linker_definition: Dict[str, Any], connected_cells: Tuple[int, int], metadata: Dict[str, Any] | None`
- `LayoutPlan` (hygel_martini/hydrogel_builder/core_utils/layout/proto_layout.py:66)
  - decorators: `dataclass`
  - 주요 field/class var: `proto_plan: ProtoPlan, cells: List[LayoutCell], links: List[LinkPlacement]`

#### Functions and methods

##### `_linear_index(ix: int, iy: int, iz: int, repeats: Tuple[int, int, int])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_layout.py:72`
- 종류: function, private/internal
- 역할: `linear index` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `ix * ny * nz + iy * nz + iz`

##### `_normalize(vec: np.ndarray)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_layout.py:77`
- 종류: function, private/internal
- 역할: `normalize` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 2개. 예: `vec / norm ; vec`
- 주요 호출: `np.linalg.norm`

##### `_get_anisotropy_axis()`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_layout.py:84`
- 종류: function, private/internal
- 역할: 설정 객체를 읽거나 runtime state를 갱신하는 helper입니다.
- 반환: 명시적 return 1개. 예: `axis`
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.get_param, Config.get_param.get`

##### `_axis_rotation_matrix(axis: str)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_layout.py:96`
- 종류: function, private/internal
- 역할: `axis rotation matrix` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 3개. 예: `np.eye(3, dtype=float) ; np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float) ; np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)`
- 주요 호출: `np.array, np.eye`

##### `_apply_rotation(vec: np.ndarray, rot: np.ndarray)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_layout.py:110`
- 종류: function, private/internal
- 역할: `apply rotation` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `vec @ rot.T`

##### `_normalize_linker_axes(linker_axes: Optional[Sequence[str]])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_layout.py:114`
- 종류: function, private/internal
- 역할: `normalize linker axes` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `cleaned[:2]`
- 주요 호출: `cleaned.append`

##### `_backbone_target_length(entry: Dict[str, Any], proto_plan: ProtoPlan)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_layout.py:133`
- 종류: function, private/internal
- 역할: `backbone target length` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `float(bond_len) * intervals`
- 주요 호출: `definition.get, entry.get`

##### `_linker_total_length(entry: Dict[str, Any], fallback: float, override: float | None=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_layout.py:144`
- 종류: function, private/internal
- 역할: `linker total length` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `float(total) ; float(override)`
- 주요 호출: `bond.get, definition.get, entry.get, ext.get`

##### `generate_layout_plan(proto_plan: ProtoPlan, backbone_defs: List[Dict[str, Any]], linker_defs: List[Dict[str, Any]], repeats: Tuple[int, int, int], backbone_strategy: Dict[str, Any] | None=None, linker_strategy: Dict[str, Any] | None=None, linker_library: LinkerTemplateLibrary | None=None, linker_axes: Optional[Sequence[str]]=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_layout.py:158`
- 종류: function
- 역할: `generate layout plan` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 5개. 예: `LayoutPlan(proto_plan=proto_plan, cells=cells, links=links) ; small_edge * (0.5 + idx) ; linker_len * (1.0 + idx) + small_edge * (0.5 + idx)`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `first_vec[axis_index[primary_axis]], second_vec[axis_index[primary_axis]], second_vec[axis_index[secondary_axis]], metadata['proto_positions']`
- 주요 호출: `LayoutCell, LayoutPlan, LinkPlacement, _apply_rotation, _axis_center, _axis_rotation_matrix, _get_anisotropy_axis, _linear_index, _linker_total_length, _normalize, _normalize_linker_axes, _x_center, _y_center, _z_center, cells.append, entry.get, getattr, linker_def_lookup.get, links.append, metadata.update, np.array, np.sqrt, np.zeros, random.choice, ... (+2)`

### `hygel_martini/hydrogel_builder/core_utils/layout/proto_populator.py`

LayoutBlueprint를 Hydrogel/World registry에 실제 Atom/Bond로 채우는 materialization 단계입니다.
- 주요 import: `from collections import defaultdict, from typing import Dict, List, Tuple, numpy, from hygel_martini.hydrogel_builder.core_utils.layout.layout_executor import LayoutBlueprint, ChainBlueprint, from hygel_martini.hydrogel_builder.main_components import Attributes, from hygel_martini.hydrogel_builder.main_components.Universe import World`
- class 수: 0, 함수/메서드 수: 8

#### Functions and methods

##### `_ordered_chain_entries(chain_key: Tuple[str, int], chain_atom_map: Dict[Tuple[str, int], List[Tuple[int, int]]])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_populator.py:13`
- 종류: function, private/internal
- 역할: `ordered chain entries` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `entries`
- 주요 호출: `chain_atom_map.get, entries.sort`

##### `_mark_backbone_terminals(hydrogel, atom_ids: List[int])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_populator.py:20`
- 종류: function, private/internal
- 역할: `mark backbone terminals` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 주요 호출: `hydrogel.terminals.append`

##### `_mark_linker_terminals(hydrogel, chain: ChainBlueprint, bead_atom_ids: List[int])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_populator.py:32`
- 종류: function, private/internal
- 역할: `mark linker terminals` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 주요 호출: `chain.definition.get, ext.get, hydrogel.terminals.append`

##### `_create_backbone_bonds(chain: ChainBlueprint, atom_ids: List[int])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_populator.py:51`
- 종류: function, private/internal
- 역할: World topology registry를 읽거나 mutate하는 builder helper입니다.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation
- 주요 호출: `Attributes.Bond, bond_lookup.get, default_params.get, entry_a.get, entry_b.get, metadata.get, params.get`

##### `_create_linker_bonds(chain: ChainBlueprint, atom_ids: List[int])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_populator.py:75`
- 종류: function, private/internal
- 역할: World topology registry를 읽거나 mutate하는 builder helper입니다.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation
- 주요 호출: `Attributes.Bond, bond_def.get, bond_def.items, chain.definition.get`

##### `_finalize_counts(hydrogel)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_populator.py:90`
- 종류: function, private/internal
- 역할: `finalize counts` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.

##### `populate_hydrogel_from_blueprint(hydrogel, blueprint: LayoutBlueprint)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_populator.py:95`
- 종류: function
- 역할: Construct Atom objects and backbone/linker internal bonds directly from a proto blueprint.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: Config/runtime state 접근, World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `skipped_counts['pairs'], skipped_counts['virtual_sites'], skipped_counts[rst.get('section', 'restraints')], skipped_counts['cmaptypes'], skipped_counts['polarization'], mapped_counts['dihedrals'], mapped_counts['impropers'], orig_to_global_by_chain[chain_key][orig_idx], mapped_counts['constraints'], skipped_counts['exclusions']` ...
- 주요 호출: `Attributes.Atom, Attributes.Bond, Attributes.Constraint, Attributes.Dihedral, Attributes.Exclusion, Config.debug_log, World.OtherSections.append, _add_other, _create_backbone_bonds, _create_linker_bonds, _finalize_counts, _map_constraints, _mark_backbone_terminals, _mark_linker_terminals, _ordered_chain_entries, atom_bp.extra.get, bead.get, bead_map.get, chain_atom_map.setdefault, chain_atom_map.setdefault.append, chain_meta.get, chain_meta_by_key.get, defaultdict, dih.get, ... (+15)`

##### `populate_hydrogel_from_blueprint._add_other(sec: str, payload: Dict)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/proto_populator.py:189`
- 종류: nested helper, private/internal
- 역할: World topology registry를 읽거나 mutate하는 builder helper입니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation
- 주요 호출: `World.OtherSections.append`

### `hygel_martini/hydrogel_builder/core_utils/layout/template_placement.py`

Shared geometry helpers for template placement and side-chain tuning.
- 주요 import: `from __future__ import annotations, from typing import Any, Dict, numpy`
- class 수: 0, 함수/메서드 수: 4

#### Functions and methods

##### `build_alignment_basis(axis: np.ndarray | list[float] | tuple[float, ...])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/template_placement.py:10`
- 종류: function
- 역할: Build an orthonormal basis whose x-axis follows ``axis``.
- 반환: 명시적 return 1개. 예: `np.column_stack((x_axis, y_axis, z_axis))`
- 주요 호출: `np.array, np.asarray, np.column_stack, np.cross, np.dot, np.linalg.norm`

##### `place_template_coords(coords: np.ndarray, origin: np.ndarray | list[float] | tuple[float, ...], axis_vector: np.ndarray | list[float] | tuple[float, ...])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/template_placement.py:40`
- 종류: function
- 역할: Rotate and translate template coordinates onto an axis-aligned frame.
- 반환: 명시적 return 1개. 예: `np.asarray(origin, dtype=float) + np.asarray(coords, dtype=float) @ basis.T`
- 주요 호출: `build_alignment_basis, np.asarray`

##### `compute_template_positions(coords: np.ndarray, origin: np.ndarray | list[float] | tuple[float, ...], normal_vector: np.ndarray | list[float] | tuple[float, ...], tangent_vector: np.ndarray | list[float] | tuple[float, ...])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/template_placement.py:50`
- 종류: function
- 역할: Build a local side-chain frame from normal and tangent vectors.
- 반환: 명시적 return 4개. 예: `np.asarray(origin, dtype=float) + np.asarray(coords, dtype=float) @ rotation.T ; None`
- 주요 호출: `np.asarray, np.column_stack, np.cross, np.linalg.norm`

##### `resolve_sidechain_placement_tuning(sim_params: Dict[str, Any], atom_count: int)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/layout/template_placement.py:79`
- 종류: function
- 역할: Resolve side-chain search settings from config with large-system fallback.
- 반환: 명시적 return 1개. 예: `tuning`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `tuning['num_candidate_vectors'], tuning['search_radius_factor'], tuning['nearby_atom_limit']`
- 주요 호출: `sim_params.get`

### `hygel_martini/hydrogel_builder/core_utils/runtime/backbone_patcher.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `yaml, os, from hygel_martini.hydrogel_builder.main_components.Universe import World, from hygel_martini.hydrogel_builder.main_components.Attributes import Angle, Dihedral, Bond`
- class 수: 0, 함수/메서드 수: 4

#### Functions and methods

##### `patch_backbone_topology(config_path)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/backbone_patcher.py:6`
- 종류: function
- 역할: Patches the World's Bonds, Angles and Dihedrals based on backbone.yaml rules.
- 반환: 명시적 return 10개. 예: `True ; rule.get('residue_name', []).count('*') + rule.get('bead_type', []).count('*') ; False`
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation, 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Angle, Dihedral, World.Angles.items, World.Bonds.items, World.Dihedrals.append, World.Dihedrals.items, World.Dihedrals.remove, adj.get, adj.setdefault, adj.setdefault.append, backbone_res_names.add, config.get, existing_angles.setdefault, existing_angles.setdefault.append, existing_angles.setdefault.extend, existing_dihedrals.setdefault, existing_dihedrals.setdefault.append, existing_dihedrals.setdefault.extend, find_paths, matches, os.path.exists, rule.get, rule.get.count, yaml.safe_load`

##### `patch_backbone_topology.matches(atom, res_name_rule, bead_type_rule)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/backbone_patcher.py:32`
- 종류: nested helper
- 역할: `matches` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 7개. 예: `True ; False`

##### `patch_backbone_topology.find_paths(current_path, res_rules, type_rules)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/backbone_patcher.py:71`
- 종류: nested helper
- 역할: `find paths` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 주요 호출: `adj.get, find_paths, matches`

##### `patch_backbone_topology.count_wildcards(rule)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/backbone_patcher.py:87`
- 종류: nested helper
- 역할: `count wildcards` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `rule.get('residue_name', []).count('*') + rule.get('bead_type', []).count('*')`
- 주요 호출: `rule.get, rule.get.count`

### `hygel_martini/hydrogel_builder/core_utils/runtime/dynamic_crosslink.py`

linker stub와 backbone end 사이의 후보를 거리/호환성 기준으로 고르고 동적 crosslink 계획을 만듭니다.
- 주요 import: `from __future__ import annotations, from dataclasses import dataclass, from typing import Dict, Iterable, List, Tuple, numpy`
- class 수: 1, 함수/메서드 수: 8

#### Classes

- `StubAssignment` (hygel_martini/hydrogel_builder/core_utils/runtime/dynamic_crosslink.py:12)
  - 역할: Chosen backbone end for a single linker stub.
  - decorators: `dataclass(frozen=True)`
  - 주요 field/class var: `linker_index: int, stub_atom: object, backbone_atom: object, chain_index: int, distance: float`

#### Functions and methods

##### `normalize_box_vector(box_vec)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/dynamic_crosslink.py:22`
- 종류: function
- 역할: Return a 3-vector box size or ``None`` when PBC is disabled.
- 반환: 명시적 return 5개. 예: `None ; arr ; np.diag(arr)`
- 주요 호출: `np.asarray, np.diag`

##### `pbc_distance(first, second, box_size: np.ndarray | None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/dynamic_crosslink.py:37`
- 종류: function
- 역할: Compute the minimum-image distance between two coordinates.
- 반환: 명시적 return 1개. 예: `float(np.linalg.norm(delta))`
- 주요 호출: `np.asarray, np.linalg.norm, np.round`

##### `group_linker_stubs(atoms: Iterable[object])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/dynamic_crosslink.py:45`
- 종류: function
- 역할: Group linker terminal stubs by linker chain index.
- 반환: 명시적 return 1개. 예: `grouped`
- 주요 호출: `getattr, grouped.setdefault, grouped.setdefault.append, grouped.sort`

##### `collect_backbone_ends(atoms: Iterable[object])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/dynamic_crosslink.py:65`
- 종류: function
- 역할: Collect true backbone end atoms keyed by backbone chain index.
- 반환: 명시적 return 1개. 예: `ends_by_chain`
- 주요 호출: `ends_by_chain.setdefault, ends_by_chain.setdefault.append, getattr`

##### `_stub_target_backbone(stub: object)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/dynamic_crosslink.py:82`
- 종류: function, private/internal
- 역할: `stub target backbone` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 3개. 예: `None ; target ; fallback`
- 주요 호출: `getattr`

##### `_is_compatible_target(stub: object, backbone_atom: object)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/dynamic_crosslink.py:93`
- 종류: function, private/internal
- 역할: `is compatible target` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `getattr(backbone_atom, 'backbone_type', None) == target ; True`
- 주요 호출: `_stub_target_backbone, getattr`

##### `_candidate_end_options(stub: object, backbone_ends: Dict[int, List[object]], box_size: np.ndarray | None, candidate_limit: int)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/dynamic_crosslink.py:100`
- 종류: function, private/internal
- 역할: `candidate end options` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 3개. 예: `options ; options[:candidate_limit]`
- 주요 호출: `_is_compatible_target, backbone_ends.items, getattr, options.append, options.sort, pbc_distance`

##### `plan_dynamic_crosslinks(linker_stubs: Dict[int, List[object]], backbone_ends: Dict[int, List[object]], box_vec, candidate_limit: int=8)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/dynamic_crosslink.py:129`
- 종류: function
- 역할: Assign two compatible backbone ends to each linker. The matcher operates on true backbone end atoms rather than arbitrary backbone beads. It also avoids reusing the same end atom across multiple linkers whenever a unique assignment is available.
- 반환: 명시적 return 1개. 예: `(assignments, notes)`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `assignments[linker_index]`
- 주요 호출: `StubAssignment, _candidate_end_options, format, getattr, linker_stubs.items, normalize_box_vector, notes.append, pairing_options.append, pairings.append, pairings.sort, used_end_atoms.add`

### `hygel_martini/hydrogel_builder/core_utils/runtime/geo_opt.py`

GROMACS grompp/mdrun 기반 energy minimization helper입니다. mdp 생성, subprocess 실행, 로그 저장을 담당합니다.
- 주요 import: `os, subprocess, from typing import Any, Dict, List, Optional, from hygel_martini.hydrogel_builder.config_params.config import Config`
- class 수: 0, 함수/메서드 수: 4

#### Functions and methods

##### `_print_process_output(label: str, stdout: Optional[str], stderr: Optional[str])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/geo_opt.py:10`
- 종류: function, private/internal
- 역할: Print stdout/stderr blocks for a subprocess result using a common format.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 주요 호출: `stderr_content.rstrip, stderr_content.strip, stdout_content.rstrip, stdout_content.strip`

##### `_run_with_logs(cmd, label, log_path=None, cwd=None, env=None, input_text=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/geo_opt.py:23`
- 종류: function, private/internal
- 역할: Run a subprocess, print stdout/stderr in a consistent block, and optionally append to a log file. Debug logging goes to Config.debug_log when enabled.
- 반환: 명시적 return 1개. 예: `proc`
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기, 외부 command/subprocess 실행 가능
- 주요 호출: `Config.debug_log, _print_process_output, f.write, join, map, subprocess.run`

##### `_create_mdp_file(directory: str, cell_opt: bool=False, em_tol: float=1000.0, nsteps: int=5000, mdp_overrides: Optional[Dict[str, Any]]=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/geo_opt.py:49`
- 종류: function, private/internal
- 역할: Creates a gromacs .mdp file for energy minimization in the specified directory.
- 반환: 명시적 return 1개. 예: `mdp_filepath`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `mdp_defaults[key]`
- 주요 호출: `f.write, mdp_overrides.items, os.path.join`

##### `run_geo_opt(structure_file: str, topology_file: str, output_dir: str, cell_opt: bool=False, gmx_executable: str='gmx', em_tol: float=1000.0, nsteps: int=5000, maxwarn: int=1, mdp_overrides: Optional[Dict[str, Any]]=None, deffnm_prefix: str='em', mdrun_extra_args: Optional[List[str]]=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/geo_opt.py:106`
- 종류: function
- 역할: Performs geometry optimization (energy minimization) for a given structure using Gromacs. Args: structure_file (str): Absolute path to the input structure file (.gro, .pdb). topology_file (str): Absolute path to the input topology file (.top). output_dir (str): Absolute path to the directory where optimization will be run and results stored. cell_opt (bool): If True, allows for cell optimization. (Note: For energy minimization with 'steep', this has no effect. Cell optimization typically requires an NPT simulation). gmx_executable (str): The command to run gromacs (e.g., 'gmx' or 'gmx_mpi')...
- 반환: 명시적 return 5개. 예: `None ; optimized_structure`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 외부 command/subprocess 실행 가능
- 주요 호출: `_create_mdp_file, _run_with_logs, grompp_cmd.extend, mdp_overrides.get, mdrun_cmd.extend, os.environ.get, os.makedirs, os.path.abspath, os.path.exists, os.path.isdir, os.path.join`

### `hygel_martini/hydrogel_builder/core_utils/runtime/packer.py`

Packmol 및 GRO/PDB/XYZ 변환을 담당합니다. GROMACS `editconf`가 있으면 사용하고 없으면 수동 변환으로 fallback합니다.
- 주요 import: `subprocess, os, sys, shutil, numpy, from hygel_martini.hydrogel_builder.config_params.config import Config`
- class 수: 0, 함수/메서드 수: 10

#### Functions and methods

##### `_normalize_box_lengths(box_lengths_nm, fallback=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/packer.py:9`
- 종류: function, private/internal
- 역할: Returns a [lx, ly, lz] list in nm. Accepts scalar or iterable input.
- 반환: 명시적 return 8개. 예: `[values[0], values[1], values[1]] ; [val, val, val] ; None ; [values[0], values[0], values[0]] ; [values[0], values[1], values[2]]`

##### `_gro_to_pdb_manual(gro_path, pdb_path)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/packer.py:41`
- 종류: function, private/internal
- 역할: 설정 객체를 읽거나 runtime state를 갱신하는 helper입니다.
- 반환: 명시적 return 1개. 예: `pdb_path`
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Config.debug_log, atom_name.upper, f.readlines, line.strip, lines.strip, out.write`

##### `convert_gro_to_pdb(gro_path, pdb_path, gmx_path)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/packer.py:69`
- 종류: function
- 역할: Converts a .gro file to a .pdb file using gmx editconf or a manual fallback.
- 반환: 명시적 return 2개. 예: `pdb_path ; _gro_to_pdb_manual(gro_path, pdb_path)`
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기, 외부 command/subprocess 실행 가능
- 주요 호출: `Config.debug_log, _gro_to_pdb_manual, join, subprocess.run`

##### `convert_pdb_to_gro(pdb_path, gro_path, gmx_path, box_lengths_nm=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/packer.py:87`
- 종류: function
- 역할: Converts a .pdb file to a .gro file using gmx editconf or a manual fallback.
- 반환: 명시적 return 2개. 예: `gro_path ; _pdb_to_gro_manual(pdb_path, gro_path, lengths_nm)`
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기, 외부 command/subprocess 실행 가능
- 주요 호출: `Config.debug_log, _normalize_box_lengths, _pdb_to_gro_manual, command.extend, join, subprocess.run`

##### `_pdb_to_gro_manual(pdb_path, gro_path, box_lengths_nm=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/packer.py:109`
- 종류: function, private/internal
- 역할: 설정 객체를 읽거나 runtime state를 갱신하는 helper입니다.
- 반환: 명시적 return 1개. 예: `gro_path`
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Config.debug_log, _normalize_box_lengths, atoms.append, line.startswith, line.strip, out.write`

##### `convert_xyz_to_gro(xyz_path, gro_path, gmx_path, molecule_name='MOL')`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/packer.py:139`
- 종류: function
- 역할: Converts a .xyz file to a .gro file by manually parsing it.
- 반환: 명시적 return 1개. 예: `gro_path`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `f_gro.write, f_xyz.readlines, line.split, lines.strip`

##### `_read_gro_atom_names(gro_path)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/packer.py:201`
- 종류: function, private/internal
- 역할: 설정 객체를 읽거나 runtime state를 갱신하는 helper입니다.
- 반환: 명시적 return 2개. 예: `names ; []`
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Config.debug_log, f.readlines, line.strip, lines.strip`

##### `_restore_atom_names_from_sources(output_gro, base_gro, molecules_to_add)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/packer.py:214`
- 종류: function, private/internal
- 역할: 설정 객체를 읽거나 runtime state를 갱신하는 helper입니다.
- 반환: 명시적 return 5개. 예: `False ; True`
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `lines[2:2 + atom_count]`
- 주요 호출: `Config.debug_log, _read_gro_atom_names, expected_names.extend, f.readlines, f.writelines, lines.strip, mol.get, updated.append`

##### `run_packmol(packmol_path, inp_filename, output_dir, sim_params=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/packer.py:258`
- 종류: function
- 역할: Generates a Packmol input file and runs Packmol.
- 반환: 명시적 return 1개. 예: `None`
- 예외/검증: `subprocess.CalledProcessError(1, command, stdout=result.stdout, stderr=result.stderr)`
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기, 외부 command/subprocess 실행 가능
- 주요 호출: `Config.debug_log, sim_params.get, subprocess.CalledProcessError, subprocess.run`

##### `pack_system_with_molecules(step_name, base_structure_gro, molecules_to_add, final_output_gro, box_lengths_nm, sim_params)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/packer.py:300`
- 종류: function
- 역할: Runs a full packing step: converts to PDB, generates packmol input, runs packmol, and converts back to GRO. Intermediate files are saved with step-specific names for debugging. Args: step_name (str): The name of the current packing step (e.g., "Add_Polymer"). base_structure_gro (str): Path to the base .gro file (e.g., initial hydrogel). molecules_to_add (list): A list of dictionaries, where each dict contains 'file' (path to .gro) and 'number'. final_output_gro (str): Path for the final, packed .gro file. box_lengths_nm (Sequence[float]): The box lengths (nm) along x/y/z. sim_params (dict):...
- 반환: 명시적 return 2개. 예: `(final_output_gro, True) ; (final_output_gro, False)`
- 예외/검증: `ValueError('Box lengths could not be determined for packmol packing step.')`
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기, 외부 command/subprocess 실행 가능
- 주요 호출: `Config.debug_log, ValueError, _normalize_box_lengths, _restore_atom_names_from_sources, arr.mean, convert_gro_to_pdb, convert_pdb_to_gro, coords.append, f.write, join, line.startswith, lines.extend, molecules_pdb_to_add.append, np.array, os.path.abspath, os.path.basename, os.path.join, os.path.splitext, run_packmol, shutil.copy, sim_params.get`

### `hygel_martini/hydrogel_builder/core_utils/runtime/topology_updater.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `re, os, from hygel_martini.hydrogel_builder.config_params.config import Config`
- class 수: 0, 함수/메서드 수: 2

#### Functions and methods

##### `create_system_topology(output_dir, top_path, itp_files)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/topology_updater.py:5`
- 종류: function
- 역할: 설정 객체를 읽거나 runtime state를 갱신하는 helper입니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: Config/runtime state 접근, 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Config.get_param, f.write, ordered_itps.append, os.path.abspath`

##### `update_topology_molecules(topology_file, molecule_counts, additional_itp_includes=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/runtime/topology_updater.py:34`
- 종류: function
- 역할: Updates the [ molecules ] section of a GROMACS topology file. Args: topology_file (str): Path to the .top file. molecule_counts (dict): A dictionary where keys are molecule names and values are their counts. additional_itp_includes (list, optional): A list of paths to .itp files to include.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `lines[molecules_section_start + 1:molecules_section_start + 1], lines[insert_index:insert_index]`
- 주요 호출: `f.readlines, f.writelines, include_line.strip, l.strip, line.strip, line.strip.startswith, lines.append, molecule_counts.items, new_includes.append, new_molecule_lines.append, os.path.exists, re.match`

### `hygel_martini/hydrogel_builder/core_utils/templates/linker_loader.py`

GRO/ITP pair에서 linker template를 읽고 두 BCK stub, 내부 bond, 외부 backbone bond, rich topology section을 구조화합니다.
- 주요 import: `from dataclasses import dataclass, from typing import Dict, List, Tuple, os, numpy, from hygel_martini.hydrogel_builder.core_utils.io.gro_parser import read_gro_atoms, from hygel_martini.hydrogel_builder.core_utils.io.martini_parser import read_itp_definitions, from hygel_martini.hydrogel_builder.core_utils.templates.monomer_loader import BeadTemplate, from hygel_martini.hydrogel_builder.config_params.config import Config`
- class 수: 3, 함수/메서드 수: 7

#### Classes

- `LinkerTemplate` (hygel_martini/hydrogel_builder/core_utils/templates/linker_loader.py:18)
  - decorators: `dataclass`
  - 주요 field/class var: `id: str, beads: List[BeadTemplate], coords: np.ndarray, internal_bonds: List[Tuple[int, int, Dict[str, float]]], internal_angles: List[Dict], internal_dihedrals: List[Dict], internal_impropers: List[Dict], dihedrals_full: List[Dict], impropers_full: List[Dict], constraints: List[Dict], pairs: List[Dict], exclusions: List[Dict]` ...
- `LinkerTemplateRecord` (hygel_martini/hydrogel_builder/core_utils/templates/linker_loader.py:49)
  - decorators: `dataclass`
  - 주요 field/class var: `template: LinkerTemplate, ratio: float`
- `LinkerTemplateLibrary` (hygel_martini/hydrogel_builder/core_utils/templates/linker_loader.py:55)
  - decorators: `dataclass`
  - 주요 field/class var: `records: List[LinkerTemplateRecord], lookup: Dict[str, LinkerTemplate]`

#### Functions and methods

##### `linker_definitions_from_library(library: LinkerTemplateLibrary)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/templates/linker_loader.py:60`
- 종류: function
- 역할: `linker definitions from library` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `definitions`
- 주요 호출: `bead_defs.append, bonds.append, definition.update, definitions.append, external_bonds_1.append, external_bonds_2.append, params.get`

##### `_extract_definition(itp_path: str, molecule_name: str | None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/templates/linker_loader.py:135`
- 종류: function, private/internal
- 역할: 설정 객체를 읽거나 runtime state를 갱신하는 helper입니다.
- 반환: 명시적 return 2개. 예: `next(iter(definitions.values())) ; definitions[molecule_name]`
- 예외/검증: `ValueError(f"링커 ITP '{itp_path}'에 '{molecule_name}' 정의가 없습니다.") ; ValueError(f"링커 ITP '{itp_path}'에 여러 moleculetype이 있습니다. 'molecule_name'을 지정해 주세요.") ; ValueError(f"링커 ITP '{itp_path}'에서 moleculetype을 찾을 수 없습니다.")`
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.get_runtime, ValueError, definitions.values, read_itp_definitions`

##### `_map_backbone_ids(beads: List[Dict], backbone_defs: List[Dict])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/templates/linker_loader.py:152`
- 종류: function, private/internal
- 역할: `map backbone ids` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `mapping`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `residue_to_backbone[res_name], mapping[bead['nr']], residue_to_backbone[name]`
- 주요 호출: `bb.get, bead.get, residue_to_backbone.get`

##### `_convert_params(bond_def: Dict)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/templates/linker_loader.py:172`
- 종류: function, private/internal
- 역할: `convert params` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `{'funct': bond_def.get('funct', 1), 'c0': length, 'c1': fc}`
- 주요 호출: `bond_def.get`

##### `_orthonormal_basis(span_vec: np.ndarray, ref: np.ndarray=np.array([0.0, 0.0, 1.0]))`
- 위치: `hygel_martini/hydrogel_builder/core_utils/templates/linker_loader.py:183`
- 종류: function, private/internal
- 역할: `orthonormal basis` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `np.column_stack((x_axis, y_axis, z_axis))`
- 예외/검증: `ValueError('링커 stub 간 벡터의 길이가 0입니다.') ; ValueError('링커 좌표에서 직교 기준을 찾을 수 없습니다.')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `ValueError, np.array, np.column_stack, np.cross, np.dot, np.linalg.norm, ref.copy`

##### `_load_single_linker(entry: Dict, backbone_defs: List[Dict])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/templates/linker_loader.py:207`
- 종류: function, private/internal
- 역할: linker GRO/ITP를 읽어 두 BCK stub를 기준으로 linker local coordinate frame을 만들고 내부/외부 bond 및 rich topology metadata를 LinkerTemplate로 변환합니다.
- 반환: 명시적 return 1개. 예: `LinkerTemplate(id=linker_id, beads=bead_templates, coords=coords, internal_bonds=internal_bonds, internal_angles=internal_angles, internal_dihedrals=internal_dihedrals, internal...`
- 예외/검증: `FileNotFoundError(f"링커 '{linker_id}'의 GRO 파일을 찾을 수 없습니다: {gro_path}") ; FileNotFoundError(f"링커 '{linker_id}'의 ITP 파일을 찾을 수 없습니다: {itp_path}") ; ValueError("각 링커는 고유한 'id'가 필요합니다.") ; ValueError(f"링커 '{linker_id}' 템플릿에는 stub 지점을 위해 정확히 2개의 'BCK' 잔기 원자가 있어야 합니다.") ; ValueError(f"링커 '{linker_id}'는 maker.json에 'linker_residue_name'과 'backbone_residue_name'이 필요합니다.") ; ValueError(f"링커 '{linker_id}'의 GRO/ITP 원자 수가 일치하지 않습니다.")`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `definition['name'], index_map[idx], bead['residue']`
- 주요 호출: `BeadTemplate, FileNotFoundError, LinkerTemplate, ValueError, _convert_params, _extract_definition, _orthonormal_basis, angle_def.get, bead.copy, bead.get, bead_templates.append, bond.get, coords_local.append, definition.get, dihedral_def.get, entry.get, imp_def.get, index_map.get, index_map.items, internal_angles.append, internal_bonds.append, internal_dihedrals.append, internal_impropers.append, new_angle.values, ... (+10)`

##### `load_linker_templates(linker_entries: List[Dict], backbone_defs: List[Dict])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/templates/linker_loader.py:423`
- 종류: function
- 역할: `load linker templates` 계열 loader입니다. 설정/파일/템플릿을 읽어 후속 builder가 사용할 dict/dataclass 구조로 정규화합니다.
- 반환: 명시적 return 1개. 예: `LinkerTemplateLibrary(records=records, lookup=lookup)`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `lookup[template.id]`
- 주요 호출: `LinkerTemplateLibrary, LinkerTemplateRecord, _load_single_linker, entry.get, records.append`

### `hygel_martini/hydrogel_builder/core_utils/templates/monomer_loader.py`

GRO/ITP pair에서 monomer sidechain template를 읽고 backbone bead를 제외한 상대 좌표와 rich topology metadata를 보존합니다.
- 주요 import: `from dataclasses import dataclass, from typing import Dict, List, Tuple, os, numpy, from hygel_martini.hydrogel_builder.core_utils.io.gro_parser import read_gro_atoms, from hygel_martini.hydrogel_builder.core_utils.io.martini_parser import read_itp_definitions, from hygel_martini.hydrogel_builder.config_params.config import Config`
- class 수: 4, 함수/메서드 수: 6

#### Classes

- `BeadTemplate` (hygel_martini/hydrogel_builder/core_utils/templates/monomer_loader.py:17)
  - decorators: `dataclass`
  - 주요 field/class var: `name: str, atom_type: str, residue_name: str, residue_number: int, original_index: int, cgnr: int, charge: float, mass: float, coord: np.ndarray`
- `MonomerTemplate` (hygel_martini/hydrogel_builder/core_utils/templates/monomer_loader.py:30)
  - decorators: `dataclass`
  - 주요 field/class var: `id: str, backbone_id: str, backbone_original_index: int, beads: List[BeadTemplate], coords: np.ndarray, internal_bonds: List[Tuple[int, int, Dict[str, float]]], internal_angles: List[Dict], internal_dihedrals: List[Dict], internal_impropers: List[Dict], dihedrals_full: List[Dict], impropers_full: List[Dict], backbone_bonds: List[Tuple[int, Dict[str, float]]]` ...
- `TemplateRecord` (hygel_martini/hydrogel_builder/core_utils/templates/monomer_loader.py:55)
  - decorators: `dataclass`
  - 주요 field/class var: `template: MonomerTemplate, ratio: float`
- `MonomerTemplateLibrary` (hygel_martini/hydrogel_builder/core_utils/templates/monomer_loader.py:61)
  - decorators: `dataclass`
  - 주요 field/class var: `records: List[TemplateRecord], by_backbone: Dict[str, List[TemplateRecord]], lookup: Dict[str, MonomerTemplate]`

#### Functions and methods

##### `_extract_single_definition(itp_path: str, molecule_name: str | None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/templates/monomer_loader.py:67`
- 종류: function, private/internal
- 역할: 설정 객체를 읽거나 runtime state를 갱신하는 helper입니다.
- 반환: 명시적 return 2개. 예: `next(iter(definitions.values())) ; definitions[molecule_name]`
- 예외/검증: `ValueError(f"ITP '{itp_path}'에 '{molecule_name}' 정의가 없습니다.") ; ValueError(f"ITP '{itp_path}'에 여러 moleculetype이 있습니다. 'molecule_name'을 지정해 주세요.") ; ValueError(f"ITP '{itp_path}'에는 유효한 [ moleculetype ] 정의가 없습니다.")`
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `Config.get_runtime, ValueError, definitions.values, read_itp_definitions`

##### `_find_backbone_bead_index(beads: List[Dict])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/templates/monomer_loader.py:84`
- 종류: function, private/internal
- 역할: `find backbone bead index` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `None ; bead['nr']`
- 주요 호출: `atom.startswith, bead.get, residu.startswith, upper`

##### `_match_backbone(beads: List[Dict], backbone_defs: List[Dict], override_id: str | None=None)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/templates/monomer_loader.py:93`
- 종류: function, private/internal
- 역할: `match backbone` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 4개. 예: `(fallback_idx, override_id) ; (bead['nr'], backbone_id) ; (fallback_idx, default_id) ; (bead['nr'], override_id)`
- 예외/검증: `ValueError('BCK(bead with backbone residue) 정보를 찾을 수 없습니다.') ; ValueError(f"Monomer에 정의된 backbone_id '{override_id}'가 BACKBONES 목록에 없습니다.") ; ValueError(f"백본 ID '{override_id}'에 해당하는 residue '{residue_name}' bead를 ITP에서 찾을 수 없습니다.")`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `residue_to_backbone[res_name], residue_to_backbone[name]`
- 주요 호출: `ValueError, _find_backbone_bead_index, bb.get, bead.get, id_to_residue.get, id_to_residue.keys, residue_to_backbone.get`

##### `_convert_params(bond_def: Dict)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/templates/monomer_loader.py:134`
- 종류: function, private/internal
- 역할: `convert params` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `{'funct': bond_def.get('funct', 1), 'c0': length, 'c1': fc}`
- 주요 호출: `bond_def.get`

##### `_load_single(entry: Dict, backbone_defs: List[Dict])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/templates/monomer_loader.py:145`
- 종류: function, private/internal
- 역할: monomer GRO/ITP에서 backbone bead를 찾고 sidechain bead들의 상대 좌표, 내부 bond/angle/dihedral, backbone bond, rich topology section을 MonomerTemplate로 변환합니다.
- 반환: 명시적 return 1개. 예: `template`
- 예외/검증: `FileNotFoundError(f'GRO 파일을 찾을 수 없습니다: {gro_path}') ; FileNotFoundError(f'ITP 파일을 찾을 수 없습니다: {itp_path}') ; ValueError("각 Monomer 항목에는 고유한 'id'가 반드시 필요합니다.") ; ValueError('Backbone 결합이 side bead를 참조하지 않습니다.') ; ValueError(f"ITP '{itp_path}'에 bead 정보가 없습니다.") ; ValueError(f"Monomer '{monomer_id}'에 'gro'와 'itp' 경로가 필요합니다.")`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `bead_map[idx]`
- 주요 호출: `BeadTemplate, FileNotFoundError, MonomerTemplate, ValueError, _convert_params, _extract_single_definition, _match_backbone, angle_def.get, backbone_bonds.append, bead.get, bead_map.get, bond_def.get, coords_list.append, definition.get, dihedral_def.get, entry.get, imp_def.get, internal_angles.append, internal_bonds.append, internal_dihedrals.append, internal_impropers.append, new_angle.values, new_dihedral.values, new_imp.values, ... (+4)`

##### `load_monomer_templates(monomer_entries: List[Dict], backbone_defs: List[Dict])`
- 위치: `hygel_martini/hydrogel_builder/core_utils/templates/monomer_loader.py:317`
- 종류: function
- 역할: `load monomer templates` 계열 loader입니다. 설정/파일/템플릿을 읽어 후속 builder가 사용할 dict/dataclass 구조로 정규화합니다.
- 반환: 명시적 return 1개. 예: `MonomerTemplateLibrary(records=records, by_backbone=by_backbone, lookup=lookup)`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `lookup[template.id]`
- 주요 호출: `MonomerTemplateLibrary, TemplateRecord, _load_single, by_backbone.setdefault, by_backbone.setdefault.append, entry.get, records.append`

### `hygel_martini/hydrogel_builder/core_utils/templates/rich_itp_validator.py`

Rich ITP section validator. When users enable `simulation_parameters.emit_rich_itp_sections=true`, invalid test entries (duplicate virtual sites, out-of-range indices, bad funct) can make grompp fail. This module validates and filters those sections before they are written out.
- 주요 import: `from __future__ import annotations, from typing import Any, Dict, List, Tuple`
- class 수: 0, 함수/메서드 수: 3

#### Functions and methods

##### `_try_int(token: Any)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/templates/rich_itp_validator.py:15`
- 종류: function, private/internal
- 역할: `try int` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `int(token) ; None`

##### `_in_range(idx: int, atom_count: int)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/templates/rich_itp_validator.py:22`
- 종류: function, private/internal
- 역할: `in range` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `1 <= idx <= atom_count`

##### `validate_and_filter_other_sections(extras: Dict[str, List[Dict[str, Any]]], atom_count: int, strict: bool=False)`
- 위치: `hygel_martini/hydrogel_builder/core_utils/templates/rich_itp_validator.py:26`
- 종류: function
- 역할: Validate rich sections stored in World.OtherSections. Returns: (filtered_extras, warnings)
- 반환: 명시적 return 2개. 예: `(filtered, warnings) ; ({}, [])`
- 예외/검증: `ValueError(msg)`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `filtered['exclusions'], filtered[sec], filtered[sec_lower], filtered['virtual_sites']`
- 주요 호출: `ValueError, _in_range, _try_int, extras.get, extras.items, filtered.extend, filtered.items, filtered.setdefault, filtered.setdefault.extend, int_tokens.append, mapped_excl.append, out_excl.append, out_rows.append, row.get, row.get.split, sec.lower, sec_lower.endswith, sec_name.startswith, seen_sites.add, seen_sites.setdefault, vs_by_sec.items, vs_by_sec.setdefault, vs_by_sec.setdefault.append, warnings.append`

### `hygel_martini/hydrogel_builder/generator.py`

Thin workflow entry helper for hydrogel generation.
- 주요 import: `from __future__ import annotations, from pathlib import Path, from .config_params.generator import run_hydrogel_example`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `run_hydrogel_builder(config_path: str | Path)`
- 위치: `hygel_martini/hydrogel_builder/generator.py:10`
- 종류: function
- 역할: Run the hydrogel workflow from a maker YAML/JSON file.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 예외/검증: `FileNotFoundError(f'Config not found: {resolved_path}')`
- 주요 호출: `FileNotFoundError, Path, Path.expanduser, resolved_path.exists, resolved_path.resolve, run_hydrogel_example`

### `hygel_martini/hydrogel_builder/main_components/Attributes.py`

Atom, Bond, Angle, Dihedral 등 GROMACS topology primitive입니다. 생성자가 World registry를 직접 mutate한다는 점이 가장 중요합니다.
- 주요 import: `numpy`
- class 수: 7, 함수/메서드 수: 8

#### Classes

- `Atom` (hygel_martini/hydrogel_builder/main_components/Attributes.py:22)
  - 역할: Representation of a single coarse-grained bead. Args: source_template: Template object from which this atom originated. source_index: Zero-based bead index inside the source template. source_residue_name: Original residue name stored for traceability when the builder later remaps residue names in output files.
  - 주요 field/class var: `num_atoms = 0`
- `Bond` (hygel_martini/hydrogel_builder/main_components/Attributes.py:113)
  - 역할: Bond record linking two atoms and updating bond adjacency lists. Duplicate bonds are ignored by design: if the canonicalized ``(i, j)`` key already exists in ``World.Bonds`` the constructor returns immediately without creating a second object.
  - 주요 field/class var: `num_bonds = 0`
- `Network_bond` (hygel_martini/hydrogel_builder/main_components/Attributes.py:167)
  - 역할: Legacy record for network-only bonds distinct from normal bonds.
  - 주요 field/class var: `num_network_bonds = 0`
- `Constraint` (hygel_martini/hydrogel_builder/main_components/Attributes.py:208)
  - 역할: Distance constraint between two atoms.
  - 주요 field/class var: `num_constraints = 0`
- `Exclusion` (hygel_martini/hydrogel_builder/main_components/Attributes.py:246)
  - 역할: Exclusion entry removing a non-bonded interaction pair.
  - 주요 field/class var: `num_exclustions = 0`
- `Angle` (hygel_martini/hydrogel_builder/main_components/Attributes.py:277)
  - 역할: Angle interaction defined by three atoms.
  - 주요 field/class var: `num_angles = 0`
- `Dihedral` (hygel_martini/hydrogel_builder/main_components/Attributes.py:325)
  - 역할: Dihedral or improper-like interaction defined by four atoms.
  - 주요 field/class var: `num_dihedrals = 0`

#### Functions and methods

##### `initialize()`
- 위치: `hygel_martini/hydrogel_builder/main_components/Attributes.py:11`
- 종류: function
- 역할: Reset global counters used to assign topology object identifiers.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.

##### `Atom.__init__(self, source_template=None, source_index=None, source_residue_name=None)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Attributes.py:35`
- 종류: method, private/internal
- 역할: World topology registry를 읽거나 mutate하는 builder helper입니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `self.atom_id, self.atom_type, self.residue_number, self.residue_name, self.atom_name, self.cgnr, self.mass, self.charge, self.position, self.bonded_atoms` ...
- 주요 호출: `World.Atoms.append, np.array`

##### `Bond.__init__(self, i, j, **kwargs)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Attributes.py:124`
- 종류: method, private/internal
- 역할: World topology registry를 읽거나 mutate하는 builder helper입니다.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `self.bond_id, self.bond_funct, self.bond_atom_1, World.Atoms[i][0].number_of_bonds, self.bond_atom_2, World.Atoms[j][0].number_of_bonds, self.bond_c0, self.bond_c1`
- 주요 호출: `World.Atoms.bonded_atoms.append, World.Bonds.append, World.Bonds.get, kwargs.get`

##### `Network_bond.__init__(self, i, j)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Attributes.py:174`
- 종류: method, private/internal
- 역할: World topology registry를 읽거나 mutate하는 builder helper입니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `self.network_bond_id, self.network_bond_funct, self.network_bond_atom_1, World.Atoms[i][0].number_of_network_bonds, self.network_bond_atom_2, World.Atoms[j][0].number_of_network_bonds, self.network_bond_c0, self.network_bond_c1`
- 주요 호출: `World.Atoms.network_bonded_atoms.append, World.Network_bonds.append`

##### `Constraint.__init__(self, i, j)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Attributes.py:215`
- 종류: method, private/internal
- 역할: World topology registry를 읽거나 mutate하는 builder helper입니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `self.constraint_id, self.constraint_atom_1, self.constraint_atom_2, self.constraint_funct, self.constraint_c0`
- 주요 호출: `World.Atoms.constrained_atoms.append, World.Constraints.append`

##### `Exclusion.__init__(self, i, j)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Attributes.py:253`
- 종류: method, private/internal
- 역할: World topology registry를 읽거나 mutate하는 builder helper입니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `self.exclustion_id, self.exclusion_atom_1, self.exclusion_atom_2`
- 주요 호출: `World.Atoms.excluded_atoms.append, World.Exclusions.append`

##### `Angle.__init__(self, i, j, k)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Attributes.py:284`
- 종류: method, private/internal
- 역할: World topology registry를 읽거나 mutate하는 builder helper입니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `self.angle_id, self.angle_funct, self.angle_atom_1, World.Atoms[i][0].number_of_angles, self.angle_atom_2, World.Atoms[j][0].number_of_angles, self.angle_atom_3, World.Atoms[k][0].number_of_angles, self.angle_c0, self.angle_c1`
- 주요 호출: `World.Angles.append, World.Atoms.angle_atoms.append`

##### `Dihedral.__init__(self, i, j, m, n, c0)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Attributes.py:332`
- 종류: method, private/internal
- 역할: World topology registry를 읽거나 mutate하는 builder helper입니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `self.dihedral_id, self.dihedral_funct, self.dihedral_atom_1, self.dihedral_atom_2, self.dihedral_atom_3, self.dihedral_atom_4, self.dihedral_c0, self.dihedral_c1, self.dihedral_c2`
- 주요 호출: `World.Dihedrals.append`

### `hygel_martini/hydrogel_builder/main_components/Hydrogel.py`

lattice/proto blueprint에서 hydrogel network를 만들고, monomer/linker template 기반으로 sidechain과 rich topology section을 붙입니다.
- 주요 import: `numpy, sys, from typing import List, from hygel_martini.hydrogel_builder.main_components import Attributes, from itertools import product, from hygel_martini.hydrogel_builder.core_utils.common.utility import interp3D, dij_sq, rij, not_self, random_normal_vector, from hygel_martini.hydrogel_builder.core_utils.common.sequence_strategy import TemplateStrategyIterator, StrategyRecord, from hygel_martini.hydrogel_builder.core_utils.layout.template_placement import compute_template_positions, resolve_sidechain_placement_tuning, from hygel_martini.hydrogel_builder.core_utils.templates.monomer_loader import load_monomer_templates, random, from hygel_martini.hydrogel_builder.config_params import read_json, itertools, ...`
- class 수: 1, 함수/메서드 수: 13

#### Classes

- `Hydrogel` (hygel_martini/hydrogel_builder/main_components/Hydrogel.py:31)
  - 역할: Construct and enrich a hydrogel network stored in ``World``. The class is responsible for translating a geometric lattice plan into concrete atoms and topology terms. The implementation mixes newer template-driven logic with older fallback behavior, so detailed docstrings are used to make the stage boundaries explicit.
  - 주요 field/class var: `num_HDG_atoms = 0, num_HDG_bonds = 0, num_HDG_angles = 0, num_HDG_dihedrals = 0`

#### Functions and methods

##### `Hydrogel.__init__(self, x_number_of_repeat=6, y_number_of_repeat=6, z_number_of_repeat=6)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Hydrogel.py:45`
- 종류: method, private/internal
- 역할: Initialize lattice repetition counts and terminal registries. Args: x_number_of_repeat: Number of unit-cell repetitions along the x axis. y_number_of_repeat: Number of unit-cell repetitions along the y axis. z_number_of_repeat: Number of unit-cell repetitions along the z axis.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `self.x_number_of_repeat, self.y_number_of_repeat, self.z_number_of_repeat, World.box_length, self.terminals, self.terminals[1], self.terminals[2], self.terminals[3], self.terminals[4], self.num_Bonds_44` ...

##### `Hydrogel.make_lines(self, bx, by, bz)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Hydrogel.py:92`
- 종류: method
- 역할: Generate the line segments used to populate one lattice cell. Args: bx: Unit-cell x index. by: Unit-cell y index. bz: Unit-cell z index. Returns: tuple[list[np.ndarray], list[np.ndarray]]: Interpolated backbone coordinates and linker coordinates for the selected cell.
- 반환: 명시적 return 2개. 예: `(segment_xyz, link_xyz) ; sign * magnitude * link_axis`
- 주요 호출: `axis_shift, getattr, interp3D, lines.append, link_xyz.append, n_segment.astype, np.array, np.floor, np.sqrt, np.square, segment_xyz.append`

##### `Hydrogel.axis_shift(box_point, magnitude)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Hydrogel.py:129`
- 종류: method
- 역할: `axis shift` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `sign * magnitude * link_axis`

##### `Hydrogel.construct_atoms(self)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Hydrogel.py:182`
- 종류: method
- 역할: lattice cell을 순회하며 backbone bead와 linker bead를 만들고 template-local rich topology section을 World.OtherSections에 remap합니다.
- 반환: 명시적 return 6개. 예: `template_like.get(name, []) if isinstance(template_like, dict) else [] ; random_generator() ; itertools.cycle(backbones) ; None ; itertools.cycle(templates)`
- 예외/검증: `ValueError('BACKBONES 리스트가 비어있습니다.')`
- 부작용 단서: Config/runtime state 접근, World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `self._linker_library`
- 주요 호출: `Attributes.Atom, Attributes.Bond, Config.debug_log, ValueError, World.OtherSections.append, _add, _map_extra_sections, b.get, backbone_definitions.get, c.get, chosen_bb_def.get, chosen_linker.other_sections.items, ex_def.get, get_backbone_generator, get_linker_generator, get_list, getattr, idx_map.get, imp.get, itertools.cycle, join, linker_atoms.append, linker_map_entries.append, load_linker_templates, ... (+28)`

##### `Hydrogel._map_extra_sections(template_like, idx_map)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Hydrogel.py:226`
- 종류: method, private/internal
- 역할: Map template-local optional topology sections into ``World``. Args: template_like: Linker/backbone template metadata dictionary. idx_map: Mapping from template-local atom indices to global ``World`` atom identifiers.
- 반환: 명시적 return 1개. 예: `template_like.get(name, []) if isinstance(template_like, dict) else []`
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation
- 주요 호출: `World.OtherSections.append, _add, c.get, ex_def.get, get_list, idx_map.get, imp.get, join, mapped.append, mapped_parts.append, p_def.get, pol.get, rst.get, template_like.get, template_like.get.items, vals.extend, vs.get, vs.get.split`

##### `Hydrogel._add(sec, payload)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Hydrogel.py:235`
- 종류: method, private/internal
- 역할: World topology registry를 읽거나 mutate하는 builder helper입니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation
- 주요 호출: `World.OtherSections.append`

##### `Hydrogel.get_list(name)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Hydrogel.py:237`
- 종류: method
- 역할: `get list` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `template_like.get(name, []) if isinstance(template_like, dict) else []`
- 주요 호출: `template_like.get`

##### `Hydrogel._construct_proto_bonds(self, output_dir)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Hydrogel.py:603`
- 종류: method, private/internal
- 역할: Attach linker stubs to the nearest allowed backbone terminals. Args: output_dir: Output directory kept for interface compatibility with older callers. The current implementation only uses runtime state already stored in ``World``. Returns: bool: ``True`` when the proto-bond pass finishes without a fatal configuration error.
- 반환: 명시적 return 5개. 예: `True ; getattr(atom, 'pre_compress_position', atom.position) ; best_atom`
- 부작용 단서: Config/runtime state 접근, World/Attributes topology registry 접근 또는 mutation, 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `self.num_Bonds_12`
- 주요 호출: `Attributes.Bond, Config.debug_log, _pick_backbone, _pos, axis_index.get, best_atom.position.copy, dij_sq, getattr, grouped.items, grouped.setdefault, grouped.setdefault.append, hasattr, np.any, np.max, params.get, used_backbones.add, used_local.add`

##### `Hydrogel.construct_bonds(self, pbc, num_cell, output_dir)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Hydrogel.py:706`
- 종류: method
- 역할: Compatibility wrapper for proto-bond construction. Args: pbc: Unused legacy flag kept for call-site compatibility. num_cell: Unused legacy cell count kept for compatibility. output_dir: Output directory associated with the current run.
- 반환: 명시적 return 1개. 예: `self._construct_proto_bonds(output_dir)`
- 주요 호출: `self._construct_proto_bonds`

##### `Hydrogel.construct_chemical_detail(self)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Hydrogel.py:716`
- 종류: method
- 역할: backbone bead마다 monomer sidechain template를 선택하고 여러 candidate normal vector를 샘플링하여 overlap이 가장 낮은 배치를 선택합니다.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: Config/runtime state 접근, World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `self.num_HDG_atoms, sequence_generators[bb_id], monomer_counts[chosen_template.id]`
- 주요 호출: `Attributes.Atom, Attributes.Bond, Config.debug_log, Config.get_runtime, StrategyRecord, TemplateStrategyIterator, W.OtherSections.append, _add_other, all_atoms.append, bonded_atom_ids.add, c.get, compute_template_positions, dij_sq, ex_def.get, getattr, hasattr, idx_map.get, iterator.next, join, load_monomer_templates, mapped_by_template.items, mapped_by_template.setdefault, mapped_by_template.setdefault.append, mapped_parts.append, ... (+27)`

##### `Hydrogel.construct_angles(self)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Hydrogel.py:1013`
- 종류: method
- 역할: Generate angle terms from template metadata and fallback rules. The priority order is: 1. internal template angles explicitly defined in the source ITP, 2. coarse structural heuristics based on backbone/sidechain role, 3. configured default angle parameters.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: Config/runtime state 접근, World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `self.num_HDG_angles`
- 주요 호출: `Attributes.Angle, World.Bonds.values, _atom.keys, bonds_by_atom.append, bonds_by_atom.items, hasattr, p.Config.get_param`

##### `Hydrogel.construct_dihedrals(self)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Hydrogel.py:1126`
- 종류: method
- 역할: Generate internal dihedrals directly from template definitions.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `self.num_HDG_dihedrals`
- 주요 호출: `Attributes.Dihedral, World.Atoms.values, id, processed_templates.add`

##### `Hydrogel.construct_impropers(self)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Hydrogel.py:1166`
- 종류: method
- 역할: Placeholder for explicit improper handling. Improper-like records are currently carried through the shared dihedral machinery and auxiliary topology sections. A dedicated ``Improper`` object could replace this in a future cleanup pass.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.

### `hygel_martini/hydrogel_builder/main_components/Polymer.py`

단독 polymer 생성 모듈입니다. legacy straight-chain 방식과 template-driven backbone/sidechain/terminal 방식이 같이 있습니다.
- 주요 import: `numpy, random, from random import Random, from collections import deque, from hygel_martini.hydrogel_builder.main_components import Attributes, from hygel_martini.hydrogel_builder.core_utils.common.utility import interp3D, dij_sq, normal_tetrahedral_vector, not_self, is_overlap, random_normal_vector, rij, from hygel_martini.hydrogel_builder.config_params import read_json, from hygel_martini.hydrogel_builder.core_utils.common.sequence_strategy import TemplateStrategyIterator, StrategyRecord, from hygel_martini.hydrogel_builder.core_utils.layout.template_placement import build_alignment_basis, compute_template_positions, place_template_coords, from hygel_martini.hydrogel_builder.core_utils.templates.monomer_loader import load_monomer_templates`
- class 수: 1, 함수/메서드 수: 22

#### Classes

- `Polymer` (hygel_martini/hydrogel_builder/main_components/Polymer.py:37)
  - 역할: Construct a polymer chain and register it into ``World``. The class exposes a template-first workflow but still retains a legacy fallback for historical input sets that only define a single generic backbone bead type.
  - 주요 field/class var: `num_PLM_atoms = 0, num_PLM_bonds = 0, num_PLM_angles = 0, num_PLM_dihedrals = 0, _polymer_config = None, _backbone_iterator: TemplateStrategyIterator | None, _backbone_defs = [], _backbone_lookup = {}, _bond_lookup = {}, _sidechain_library = None, _sidechain_iterators = {}, _sidechain_strategy_cfg = {}` ...

#### Functions and methods

##### `_build_polymer_bond_lookup(bond_rules, fallback_length)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:20`
- 종류: function, private/internal
- 역할: Create a fast lookup table for backbone-to-backbone bond parameters.
- 반환: 명시적 return 2개. 예: `lookup`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `lookup[key]`
- 주요 호출: `rule.get`

##### `Polymer.__init__(self, p_mon_num, p_length)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:62`
- 종류: method, private/internal
- 역할: Store polymer dimensions and initialize the working box size. Args: p_mon_num: Number of backbone monomers. p_length: End-to-end length used for the initial straight-chain coordinate interpolation.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `self.p_length, self.p_mon_num, self._backbone_atom_ids, World.box_length`

##### `Polymer.configure(cls, config: dict | None)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:85`
- 종류: method, classmethod
- decorators: `classmethod`
- 역할: Cache template libraries and strategy iterators for polymer builds.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `cls._polymer_config, cls._backbone_defs, cls._backbone_lookup, cls._backbone_iterator, cls._bond_lookup, cls._sidechain_library, cls._sidechain_iterators, cls._terminal_records, cls._terminal_strategy, cls._terminal_random` ...
- 주요 호출: `StrategyRecord, TemplateStrategyIterator, _build_polymer_bond_lookup, bb.get, cls._sidechain_library.by_backbone.items, cls._terminal_strategy.get, config.get, getattr, load_monomer_templates, random.Random`

##### `Polymer.make_lines(self, random_seed)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:162`
- 종류: method
- 역할: Generate a reproducible straight-chain backbone path. Args: random_seed: Seed used to choose the chain center and orientation. Returns: np.ndarray: Interpolated backbone coordinates.
- 반환: 명시적 return 1개. 예: `interp3D(self.p_mon_num, pm_start_point, pm_last_point)`
- 주요 호출: `Random, Random.choice, Random.random, interp3D, np.array`

##### `Polymer.construct_atoms(self, random_seed)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:199`
- 종류: method
- 역할: Dispatch to the template-driven or legacy atom-construction path.
- 반환: 명시적 return 2개. 예: `self._legacy_construct_atoms(random_seed) ; self._construct_atoms_from_templates(random_seed)`
- 주요 호출: `self._construct_atoms_from_templates, self._legacy_construct_atoms`

##### `Polymer._legacy_construct_atoms(self, random_seed)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:207`
- 종류: method, private/internal
- 역할: Construct a polymer using the historical single-backbone settings.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: Config/runtime state 접근, World/Attributes topology registry 접근 또는 mutation
- 주요 호출: `Attributes.Atom, Attributes.Bond, p.Config.get_param, self.make_lines`

##### `Polymer._next_backbone_definition(self)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:252`
- 종류: method, private/internal
- 역할: Return the next backbone template according to the configured strategy.
- 반환: 명시적 return 3개. 예: `{} ; template ; self._backbone_defs[0]`
- 주요 호출: `self._backbone_iterator.next`

##### `Polymer._construct_atoms_from_templates(self, random_seed)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:261`
- 종류: method, private/internal
- 역할: Construct backbone beads from template metadata. Each interpolated backbone position receives a template-selected bead. Consecutive bead pairs are then connected using chemistry-specific bond parameters when available, falling back to the configured default backbone bond otherwise.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `self._backbone_atom_ids, self.num_PLM_atoms, self.num_PLM_bonds`
- 주요 호출: `Attributes.Atom, Attributes.Bond, definition.get, params.get, self._attach_terminals, self._backbone_atom_ids.append, self._bond_lookup.get, self._next_backbone_definition, self.make_lines, template.get`

##### `Polymer._select_terminal_templates(self)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:312`
- 종류: method, private/internal
- 역할: Choose left and right terminal templates according to strategy.
- 반환: 명시적 return 3개. 예: `(left_record.template if left_record else None, right_record.template if right_record else None) ; (None, None) ; self._terminal_random.choices(candidates, weights=weights, k=1)[0]`
- 주요 호출: `lower, pick, self._terminal_random.choices, self._terminal_strategy.get`

##### `Polymer.pick(exclude_id=None)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:318`
- 종류: method
- 역할: `pick` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `self._terminal_random.choices(candidates, weights=weights, k=1)[0]`
- 주요 호출: `self._terminal_random.choices`

##### `Polymer._alignment_basis(self, axis)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:331`
- 종류: method, private/internal
- 역할: Build an orthonormal basis whose x-axis follows ``axis``.
- 반환: 명시적 return 1개. 예: `build_alignment_basis(axis)`
- 주요 호출: `build_alignment_basis`

##### `Polymer._place_template(self, template, origin, axis_vector)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:335`
- 종류: method, private/internal
- 역할: Rotate and translate template coordinates onto an axis-aligned frame.
- 반환: 명시적 return 1개. 예: `place_template_coords(template.coords, origin, axis_vector)`
- 주요 호출: `place_template_coords`

##### `Polymer._compute_template_positions(self, template, origin, normal_vector, tangent_vector)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:339`
- 종류: method, private/internal
- 역할: Build a local side-chain frame from normal and tangent vectors.
- 반환: 명시적 return 1개. 예: `compute_template_positions(template.coords, origin, normal_vector, tangent_vector)`
- 주요 호출: `compute_template_positions`

##### `Polymer._create_template_atoms(self, template, positions, residue_override=None)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:343`
- 종류: method, private/internal
- 역할: Instantiate atoms for a placed template and return their IDs.
- 반환: 명시적 return 1개. 예: `atom_ids`
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation
- 주요 호출: `Attributes.Atom, atom_ids.append`

##### `Polymer._connect_template_bonds(self, template, created_atom_ids, backbone_atom_id)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:359`
- 종류: method, private/internal
- 역할: Transfer template-local topology terms to global polymer indices.
- 반환: 명시적 return 4개. 예: `None ; 0 ; depth + 1`
- 예외/검증: `ValueError(f'Dihedral params 부족: {dih}') ; ValueError(f'Improper params 부족: {imp}') ; ValueError(f'Proper dihedral params 부족: {dih}') ; ValueError(f'Proper improper-dihedral params 부족: {imp}')`
- 부작용 단서: Config/runtime state 접근, World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `orig_to_global[bck_orig], orig_to_global[orig_idx]`
- 주요 호출: `Attributes.Bond, Attributes.Constraint, Attributes.Dihedral, Attributes.Exclusion, Config.debug_log, ValueError, World.OtherSections.append, _add_other, _add_template_edge, _template_path_length, c.get, deque, dih.get, ex_def.get, getattr, getattr.items, imp.get, join, mapped.append, mapped_parts.append, orig_to_global.get, params.get, queue.append, queue.popleft, ... (+9)`

##### `Polymer._add_template_edge(i, j)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:375`
- 종류: method, private/internal
- 역할: `add template edge` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 주요 호출: `template_graph.setdefault, template_graph.setdefault.add`

##### `Polymer._template_path_length(start_idx, end_idx)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:393`
- 종류: method, private/internal
- 역할: `template path length` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 3개. 예: `None ; 0 ; depth + 1`
- 주요 호출: `deque, queue.append, queue.popleft, seen.add, template_graph.get`

##### `Polymer._add_other(sec, payload)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:409`
- 종류: method, private/internal
- 역할: World topology registry를 읽거나 mutate하는 builder helper입니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation
- 주요 호출: `World.OtherSections.append`

##### `Polymer._attach_terminals(self, World)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:550`
- 종류: method, private/internal
- 역할: Attach terminal templates to the first and last backbone beads.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 주요 호출: `getattr, np.array, self._connect_template_bonds, self._create_template_atoms, self._place_template, self._select_terminal_templates`

##### `Polymer._construct_sidechains_from_templates(self)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:575`
- 종류: method, private/internal
- 역할: Attach polymer side-chain templates while avoiding local clashes.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: Config/runtime state 접근
- 주요 호출: `bonded_atom_ids.add, dij_sq, getattr, iterator.next, nearby_atoms.append, not_self, np.array, np.linalg.norm, p.Config.get_param, random_normal_vector, rij, self._compute_template_positions, self._connect_template_bonds, self._create_template_atoms, self._sidechain_iterators.get`

##### `Polymer.construct_chemical_detail(self)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:653`
- 종류: method
- 역할: template sidechain iterator가 있으면 template 기반 sidechain 배치를 수행하고, 없으면 legacy geometric sidechain 생성으로 fallback합니다.
- 반환: 명시적 return 1개. 예: `self._construct_sidechains_from_templates()`
- 부작용 단서: Config/runtime state 접근, World/Attributes topology registry 접근 또는 mutation, 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `self.num_PLM_atoms, self.num_PLM_bonds, position_testers[i, :]`
- 주요 호출: `Attributes.Atom, Attributes.Bond, batom_positions.append, depth_1_atoms.append, depth_1_atoms.remove, depth_2_atoms.append, depth_2_atoms.remove, is_overlap, normal_tetrahedral_vector, not_self, np.zeros, p.Config.get_param, random_normal_vector, self._construct_sidechains_from_templates`

##### `Polymer.construct_angles(self)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Polymer.py:869`
- 종류: method
- 역할: Generate polymer angle terms from specific and default rules. Angle assignment first checks chemistry-specific overrides declared in the polymer configuration. If no override matches the atom-type set, the method applies the configured default angle parameters.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: Config/runtime state 접근, World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `self.num_PLM_angles`
- 주요 호출: `Attributes.Angle, angle_configs.get, atom_types_in_angle.isdisjoint, near_cen_atom_ids.append, near_cen_atom_ids.sort, np.array, np.where, p.Config.get_param, pos.sort`

### `hygel_martini/hydrogel_builder/main_components/Universe.py`

전역 mutable World 컨테이너입니다. Atom/Bond 등 모든 topology record가 World class-level registry에 등록됩니다.
- 주요 import: `collections, numpy`
- class 수: 1, 함수/메서드 수: 7

#### Classes

- `World` (hygel_martini/hydrogel_builder/main_components/Universe.py:62)
  - 역할: Process-wide mutable container for topology and coordinate state. The current pipeline builds structures incrementally and stores all atoms, bonds, exclusions, and auxiliary topology sections in class-level dictionaries. Resetting ``World`` between planning and materialization is therefore essential to avoid cross-stage contamination.
  - 주요 field/class var: `mean_sep = 0.24, ubox_length = 0.0, segment_length = 0, max_linker_span = 0.0, cell_vector = np.array([0.0, 0.0, 0.0], dtype=np.float64), box_vector = np.array([0.0, 0.0, 0.0], dtype=np.float64), box_length = 0.0, number_of_hydrogels = 0, number_of_polymers = 0, hydrogels = [], polymers = [], number_of_hydrogel_atoms = 0` ...

#### Functions and methods

##### `initialize_world(segment_length_from_config, mean_sep_from_config, max_linker_span_from_config=0.0)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Universe.py:12`
- 종류: function
- 역할: Initialize global geometry parameters from configuration. Args: segment_length_from_config: Number of backbone beads per segment. mean_sep_from_config: Target equilibrium spacing between consecutive coarse-grained backbone beads. max_linker_span_from_config: Maximum linker reach inferred from the linker template library. This acts as a lower bound on the unit cell size so that linkers are not forced into immediate overlap. Raises: ValueError: If the quadratic unit-cell estimate does not yield a valid positive real root.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 예외/검증: `ValueError('방정식에서 유효한(양의 실수) 박스 길이를 찾을 수 없습니다.')`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `World.segment_length, World.mean_sep, World.max_linker_span, World.cell_vector, World.box_vector, World.ubox_length`
- 주요 호출: `ValueError, np.array, np.isreal, np.roots, np.square`

##### `World.reset(cls)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Universe.py:119`
- 종류: method, classmethod
- decorators: `classmethod`
- 역할: Reset all geometry, counters, and topology registries.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `cls.mean_sep, cls.ubox_length, cls.segment_length, cls.max_linker_span, cls.cell_vector, cls.box_vector, cls.box_length, cls.number_of_hydrogels, cls.number_of_polymers, cls.hydrogels` ...
- 주요 호출: `collections.defaultdict, np.array`

##### `World.__init__(self)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Universe.py:151`
- 종류: method, private/internal
- 역할: `init` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.

##### `World.make_hydrogel(self, fix_dna, nx=6, ny=6, nz=6)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Universe.py:154`
- 종류: method
- 역할: Create and register a hydrogel object. Args: fix_dna: Legacy flag that forces hydrogel-related state to be cleared before constructing a fresh hydrogel instance. nx: Number of unit-cell repetitions along the x axis. ny: Number of unit-cell repetitions along the y axis. nz: Number of unit-cell repetitions along the z axis.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `World.number_of_hydrogels, World.hydrogels, World.number_of_hydrogel_atoms, World.number_of_hydrogel_bonds, World.number_of_hydrogel_angles, World.number_of_hydrogel_dihedrals, World.number_of_atoms`
- 주요 호출: `Hydrogel, World.Angles.clear, World.Atoms.clear, World.Bonds.clear, World.Constraints.clear, World.Dihedrals.clear, World.Exclusions.clear, World.Network_bonds.clear, World.hydrogels.append`

##### `World.make_polymer(self, p_mon_num, p_length)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Universe.py:185`
- 종류: method
- 역할: Create and register a standalone polymer object. Args: p_mon_num: Number of backbone monomers in the polymer chain. p_length: End-to-end span used to initialize the polymer box.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: World/Attributes topology registry 접근 또는 mutation, 객체/class/global attribute 갱신
- 주요 대입: `World.number_of_polymers`
- 주요 호출: `Polymer, World.polymers.append`

##### `World.update_hydrogel_attributes(self, hydrogel)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Universe.py:196`
- 종류: method
- 역할: Copy aggregate hydrogel counters onto the world container.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `self.number_of_hydrogel_atoms, self.number_of_hydrogel_bonds, self.number_of_hydrogel_angles, self.number_of_hydrogel_dihedrals`

##### `World.update_polymer_attributes(self, polymer)`
- 위치: `hygel_martini/hydrogel_builder/main_components/Universe.py:204`
- 종류: method
- 역할: Copy aggregate polymer counters onto the world container.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `self.number_of_polymer_atoms, self.number_of_polymer_bonds, self.number_of_polymer_angles, self.number_of_polymer_dihedrals`

### `hygel_martini/hydrogel_builder/relax/__init__.py`

Relaxation workflows that run after hydrogel_builder system construction.
- 주요 import: `from __future__ import annotations`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `run_relax_workflow(*args, **kwargs)`
- 위치: `hygel_martini/hydrogel_builder/relax/__init__.py:6`
- 종류: function
- 역할: `run relax workflow` 실행 helper입니다. workflow 단계나 외부 command/script를 실행 또는 위임합니다.
- 반환: 명시적 return 1개. 예: `_run_relax_workflow(*args, **kwargs)`
- 주요 호출: `_run_relax_workflow`

### `hygel_martini/hydrogel_builder/relax/cli.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `from __future__ import annotations, argparse, from pathlib import Path`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `main()`
- 위치: `hygel_martini/hydrogel_builder/relax/cli.py:7`
- 종류: function, CLI entry
- 역할: `main` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 주요 호출: `Path, argparse.ArgumentParser, parser.add_argument, parser.exit, parser.parse_args, run_relax_workflow`

### `hygel_martini/hydrogel_builder/relax/config.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `from __future__ import annotations, copy, json, os, from pathlib import Path, from typing import Any, Dict`
- class 수: 0, 함수/메서드 수: 8

#### Functions and methods

##### `_deep_merge(base: Dict[str, Any], incoming: Dict[str, Any])`
- 위치: `hygel_martini/hydrogel_builder/relax/config.py:24`
- 종류: function, private/internal
- 역할: `deep merge` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `base`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `base[key]`
- 주요 호출: `_deep_merge, base.get, copy.deepcopy, incoming.items`

##### `_load_yaml_file(path: Path)`
- 위치: `hygel_martini/hydrogel_builder/relax/config.py:33`
- 종류: function, private/internal
- 역할: `load yaml file` 계열 loader입니다. 설정/파일/템플릿을 읽어 후속 builder가 사용할 dict/dataclass 구조로 정규화합니다.
- 반환: 명시적 return 1개. 예: `data`
- 예외/검증: `ImportError('PyYAML is required for hydrogel_builder.relax.') ; TypeError(f'Relax config root must be a mapping: {path}')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `ImportError, TypeError, path.open, yaml.safe_load`

##### `_load_with_includes(path: Path, seen: set[Path] | None=None)`
- 위치: `hygel_martini/hydrogel_builder/relax/config.py:46`
- 종류: function, private/internal
- 역할: `load with includes` 계열 loader입니다. 설정/파일/템플릿을 읽어 후속 builder가 사용할 dict/dataclass 구조로 정규화합니다.
- 반환: 명시적 return 2개. 예: `data ; merged`
- 예외/검증: `TypeError(f'Relax config root must be a mapping: {resolved}') ; ValueError(f"'includes' is not supported in JSON config files: {resolved}\nConvert to .yaml/.yml to use includes.") ; ValueError(f'Cyclic include detected in relax config: {resolved}')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Path, TypeError, ValueError, _deep_merge, _load_with_includes, _load_yaml_file, data.pop, inc_path.is_absolute, json.load, path.resolve, path.suffix.lower, resolved.open, seen.add`

##### `_build_context(config_path: Path)`
- 위치: `hygel_martini/hydrogel_builder/relax/config.py:78`
- 종류: function, private/internal
- 역할: `build context` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 1개. 예: `{'CONFIG_DIR': config_dir, 'REPO_ROOT': repo_root}`
- 주요 호출: `Path, Path.resolve, config_path.resolve`

##### `_looks_like_path_key(key: str | None)`
- 위치: `hygel_martini/hydrogel_builder/relax/config.py:84`
- 종류: function, private/internal
- 역할: `looks like path key` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 3개. 예: `key.endswith(_PATH_SUFFIXES) ; False ; True`
- 주요 호출: `key.endswith`

##### `_resolve_path_value(value: str, context: Dict[str, str])`
- 위치: `hygel_martini/hydrogel_builder/relax/config.py:92`
- 종류: function, private/internal
- 역할: `resolve path value` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `expanded`
- 주요 호출: `context.items, expanded.replace, os.path.abspath, os.path.expanduser, os.path.expandvars, os.path.isabs, os.path.join`

##### `_normalize_tree(node: Any, context: Dict[str, str], parent_key: str | None=None)`
- 위치: `hygel_martini/hydrogel_builder/relax/config.py:101`
- 종류: function, private/internal
- 역할: `normalize tree` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 5개. 예: `node ; {key: _normalize_tree(value, context, key) for key, value in node.items()} ; [_normalize_tree(item, context) for item in node] ; _resolve_path_value(node, context) ; [str(item) for item in node]`
- 주요 호출: `_looks_like_path_key, _normalize_tree, _resolve_path_value, node.items`

##### `load_relax_config(config_path: str | Path)`
- 위치: `hygel_martini/hydrogel_builder/relax/config.py:116`
- 종류: function
- 역할: `load relax config` 계열 loader입니다. 설정/파일/템플릿을 읽어 후속 builder가 사용할 dict/dataclass 구조로 정규화합니다.
- 반환: 명시적 return 1개. 예: `_normalize_tree(data, context)`
- 예외/검증: `FileNotFoundError(f'Relax config not found: {path}')`
- 주요 호출: `FileNotFoundError, Path, Path.expanduser, Path.expanduser.resolve, _build_context, _load_with_includes, _normalize_tree, path.exists`

### `hygel_martini/hydrogel_builder/relax/generator.py`

Thin workflow entry helper for post-build hydrogel relaxation runs.
- 주요 import: `from __future__ import annotations, from pathlib import Path, from typing import Any, Dict, from .config import load_relax_config, from .soft_em import run_soft_em, from .soft_md import run_soft_md`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `run_relax_workflow(config_path: str | Path)`
- 위치: `hygel_martini/hydrogel_builder/relax/generator.py:13`
- 종류: function
- 역할: `run relax workflow` 실행 helper입니다. workflow 단계나 외부 command/script를 실행 또는 위임합니다.
- 반환: 명시적 return 1개. 예: `result`
- 예외/검증: `FileNotFoundError(f'Config not found: {resolved_path}') ; ValueError('workflow.mode must be one of: soft_em, soft_md')`
- 주요 호출: `FileNotFoundError, Path, Path.expanduser, ValueError, cfg.get, load_relax_config, resolved_path.exists, resolved_path.resolve, run_soft_em, run_soft_md, workflow.get`

### `hygel_martini/hydrogel_builder/relax/soft_em.py`

post-build soft energy minimization workflow입니다. bonded force constant scaling, topology patch, EM 반복, energy 추출/요약을 수행합니다.
- 주요 import: `from __future__ import annotations, os, re, shlex, shutil, subprocess, from pathlib import Path, from statistics import mean, from typing import Any, Dict, List, Tuple`
- class 수: 0, 함수/메서드 수: 16

#### Functions and methods

##### `_run(cmd: List[str], *, cwd: Path | None=None, env: Dict[str, str] | None=None, input_str: str | None=None, check: bool=True)`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_em.py:16`
- 종류: function, private/internal
- 역할: `run` 실행 helper입니다. workflow 단계나 외부 command/script를 실행 또는 위임합니다.
- 반환: 명시적 return 1개. 예: `process`
- 예외/검증: `RuntimeError(f"Command failed (rc={process.returncode}): {' '.join(map(shlex.quote, cmd))}\n--- output ---\n{process.stdout}")`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 외부 command/subprocess 실행 가능
- 주요 호출: `RuntimeError, join, map, subprocess.run`

##### `_ensure_dir(path: Path)`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_em.py:42`
- 종류: function, private/internal
- 역할: `ensure dir` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `path.mkdir`

##### `_clamp(value: float, low: float, high: float)`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_em.py:46`
- 종류: function, private/internal
- 역할: `clamp` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `max(low, min(high, value))`

##### `_parse_gro_box(gro_path: Path)`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_em.py:50`
- 종류: function, private/internal
- 역할: `parse gro box` 계열 parser입니다. 문자열/파일 내용을 내부 자료구조로 바꾸며, 입력 형식이 맞지 않으면 예외 또는 None 경로를 사용합니다.
- 반환: 명시적 return 1개. 예: `tuple(map(float, last[:3]))`
- 예외/검증: `ValueError(f'Cannot parse box from {gro_path}')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `ValueError, gro_path.read_text, gro_path.read_text.strip, gro_path.read_text.strip.splitlines, gro_path.read_text.strip.splitlines.split, map`

##### `_parse_em_fmax(log_path: Path)`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_em.py:57`
- 종류: function, private/internal
- 역할: `parse em fmax` 계열 parser입니다. 문자열/파일 내용을 내부 자료구조로 바꾸며, 입력 형식이 맞지 않으면 예외 또는 None 경로를 사용합니다.
- 반환: 명시적 return 2개. 예: `float(match.group(1))`
- 예외/검증: `RuntimeError(f'Could not parse Fmax from {log_path}')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `RuntimeError, log_path.read_text, match.group, re.search`

##### `_find_energy_indices(gmx_cmd: str, edr_file: Path, wanted_names: List[str], env: Dict[str, str])`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_em.py:68`
- 종류: function, private/internal
- 역할: `find energy indices` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `[idx_map[name] for name in wanted_names]`
- 예외/검증: `RuntimeError(f'Could not find energy terms in {edr_file}: {missing}\n--- energy menu excerpt ---\n{excerpt}')`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `idx_map[name]`
- 주요 호출: `RuntimeError, _run, join, match.group, match.group.strip, probe.stdout.splitlines, re.finditer, re.search`

##### `_extract_xvg(gmx_cmd: str, edr_file: Path, out_xvg: Path, terms: List[str], env: Dict[str, str])`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_em.py:97`
- 종류: function, private/internal
- 역할: `extract xvg` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 주요 호출: `_find_energy_indices, _run, join`

##### `_read_xvg_rows(path: Path)`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_em.py:114`
- 종류: function, private/internal
- 역할: `read xvg rows` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `rows`
- 예외/검증: `RuntimeError(f'No numeric data found in {path}')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `RuntimeError, line.strip, path.read_text, path.read_text.splitlines, rows.append, stripped.split, stripped.startswith`

##### `_summarize_series(path: Path, column: int, mode: str)`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_em.py:129`
- 종류: function, private/internal
- 역할: `summarize series` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `mean((row[column] for row in rows)) ; rows[-1][column]`
- 주요 호출: `_read_xvg_rows, mean`

##### `_split_comment(line: str)`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_em.py:136`
- 종류: function, private/internal
- 역할: `split comment` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `(line.rstrip(), '') ; (code.rstrip(), ';' + comment)`
- 주요 호출: `code.rstrip, line.rstrip, line.split`

##### `_is_int_token(token: str)`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_em.py:143`
- 종류: function, private/internal
- 역할: `is int token` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `True ; False`

##### `_is_float_token(token: str)`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_em.py:151`
- 종류: function, private/internal
- 역할: `is float token` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `True ; False`

##### `scale_itp_bonded(in_itp: Path, out_itp: Path, factor: float)`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_em.py:159`
- 종류: function
- 역할: `scale itp bonded` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `tokens[idx], tokens[6]`
- 주요 호출: `SECTION_RE.match, _is_float_token, _is_int_token, _split_comment, code.strip, in_itp.read_text, in_itp.read_text.splitlines, join, line.strip, match.group, match.group.lower, out_itp.write_text, output.append, raw.rstrip, re.match, re.match.group, stripped.split, stripped.startswith`

##### `patch_system_top(in_top: Path, out_top: Path, bonded_itp_basename: str, new_local_itp_name: str)`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_em.py:224`
- 종류: function
- 역할: `patch system top` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Path, in_top.read_text, in_top.read_text.splitlines, join, line.strip, match.group, out_top.write_text, output.append, pattern.match, raw.endswith, raw.rstrip, re.compile`

##### `_grompp_and_run_em(gmx_cmd: str, mdp: Path, gro: Path, top: Path, outdir: Path, ntomp: int, maxwarn: int, env: Dict[str, str])`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_em.py:241`
- 종류: function, private/internal
- 역할: `grompp and run em` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `(tpr, edr, log, gro_out)`
- 예외/검증: `RuntimeError(f'Missing EM output file: {path}')`
- 주요 호출: `RuntimeError, _ensure_dir, _run, path.exists`

##### `run_soft_em(cfg: Dict[str, Any])`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_em.py:265`
- 종류: function
- 역할: `run soft em` 실행 helper입니다. workflow 단계나 외부 command/script를 실행 또는 위임합니다.
- 반환: 명시적 return 1개. 예: `final_path`
- 예외/검증: `FileNotFoundError(path) ; RuntimeError(f'soft_em did not converge within max_iter={max_iter}. Last structure: {current}')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `env['OMP_NUM_THREADS'], env['GMX_OPENMP_MAX_THREADS']`
- 주요 호출: `FileNotFoundError, Path, Path.resolve, RuntimeError, _clamp, _ensure_dir, _extract_xvg, _grompp_and_run_em, _parse_em_fmax, _parse_gro_box, _run, _summarize_series, backup.exists, cfg.get, os.environ.copy, patch_system_top, path.exists, runtime.get, scale_itp_bonded, shutil.copy2, shutil.move, shutil.rmtree, soft_em.get, tools.get, ... (+2)`

### `hygel_martini/hydrogel_builder/relax/soft_md.py`

post-build soft MD wrapper입니다. grompp/mdrun을 실행하고 soft_md 결과 gro를 반환합니다.
- 주요 import: `from __future__ import annotations, os, shlex, subprocess, from pathlib import Path, from typing import Any, Dict, Iterable, List`
- class 수: 0, 함수/메서드 수: 3

#### Functions and methods

##### `_run(cmd: List[str], *, cwd: Path, env: Dict[str, str])`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_md.py:10`
- 종류: function, private/internal
- 역할: `run` 실행 helper입니다. workflow 단계나 외부 command/script를 실행 또는 위임합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 예외/검증: `RuntimeError(f"Command failed (rc={process.returncode}): {' '.join(map(shlex.quote, cmd))}\n--- output ---\n{process.stdout}")`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 외부 command/subprocess 실행 가능
- 주요 호출: `RuntimeError, join, map, process.stdout.rstrip, subprocess.run`

##### `_string_list(value: Iterable[Any])`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_md.py:29`
- 종류: function, private/internal
- 역할: `string list` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `[str(item) for item in value]`

##### `run_soft_md(cfg: Dict[str, Any])`
- 위치: `hygel_martini/hydrogel_builder/relax/soft_md.py:33`
- 종류: function
- 역할: `run soft md` 실행 helper입니다. workflow 단계나 외부 command/script를 실행 또는 위임합니다.
- 반환: 명시적 return 1개. 예: `workdir / f'{deffnm}.gro'`
- 예외/검증: `FileNotFoundError(path)`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `env['OMP_NUM_THREADS'], env['GMX_OPENMP_MAX_THREADS']`
- 주요 호출: `FileNotFoundError, Path, Path.resolve, _run, _string_list, cfg.get, grompp_cmd.extend, join, map, mdrun_cmd.extend, os.environ.copy, path.exists, runtime.get, soft_md.get, tools.get, workdir.mkdir`

### `hygel_martini/param_opt/__init__.py`

Explicit workflow packages for 01/02/03 polymer parameter generation.
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `main()`
- 위치: `hygel_martini/param_opt/__init__.py:4`
- 종류: function, CLI entry
- 역할: `main` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 주요 호출: `_main`

### `hygel_martini/param_opt/bead_generator/cli.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `from __future__ import annotations, argparse`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `main()`
- 위치: `hygel_martini/param_opt/bead_generator/cli.py:6`
- 종류: function, CLI entry
- 역할: `main` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 예외/검증: `NotImplementedError('param_opt.bead_generator is the planned bead-assignment workflow, but it is not implemented yet.')`
- 주요 호출: `NotImplementedError, argparse.ArgumentParser, parser.parse_args`

### `hygel_martini/param_opt/cli.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `from __future__ import annotations`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `main()`
- 위치: `hygel_martini/param_opt/cli.py:4`
- 종류: function, CLI entry
- 역할: `main` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 예외/검증: `SystemExit('Use an explicit workflow module instead of `python -m param_opt`:\n python -m param_opt.qm_to_opls --config ...\n python -m param_opt.opls_to_martini --config ...\n...`
- 주요 호출: `SystemExit`

### `hygel_martini/param_opt/core/config.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `from __future__ import annotations, argparse, copy, json, os, re, from pathlib import Path, from typing import Any, Dict, List, from .utils import parse_csv_list, parse_int_csv, parse_semicolon_list`
- class 수: 0, 함수/메서드 수: 14

#### Functions and methods

##### `deep_update(base: Dict[str, Any], override: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/core/config.py:19`
- 종류: function
- 역할: `deep update` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `result`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `result[key]`
- 주요 호출: `copy.deepcopy, deep_update, override.items, result.get`

##### `_load_yaml(path: Path)`
- 위치: `hygel_martini/param_opt/core/config.py:29`
- 종류: function, private/internal
- 역할: `load yaml` 계열 loader입니다. 설정/파일/템플릿을 읽어 후속 builder가 사용할 dict/dataclass 구조로 정규화합니다.
- 반환: 명시적 return 2개. 예: `data ; {}`
- 예외/검증: `RuntimeError('PyYAML is required to load .yaml/.yml config files. Install with: pip install pyyaml') ; ValueError(f'Config root must be a mapping: {path}')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `RuntimeError, ValueError, path.read_text, yaml.safe_load`

##### `_load_single_config(path: Path)`
- 위치: `hygel_martini/param_opt/core/config.py:45`
- 종류: function, private/internal
- 역할: `load single config` 계열 loader입니다. 설정/파일/템플릿을 읽어 후속 builder가 사용할 dict/dataclass 구조로 정규화합니다.
- 반환: 명시적 return 2개. 예: `_load_yaml(path) ; data`
- 예외/검증: `ValueError(f'Config root must be a mapping: {path}') ; ValueError(f'Unsupported config extension: {path}. Use .yaml/.yml or .json')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `ValueError, _load_yaml, json.loads, path.read_text, path.suffix.lower`

##### `_load_with_includes(path: Path, seen: List[Path] | None=None)`
- 위치: `hygel_martini/param_opt/core/config.py:57`
- 종류: function, private/internal
- 역할: `load with includes` 계열 loader입니다. 설정/파일/템플릿을 읽어 후속 builder가 사용할 dict/dataclass 구조로 정규화합니다.
- 반환: 명시적 return 1개. 예: `merged`
- 예외/검증: `ValueError(f"'includes' must be a list: {rpath}") ; ValueError(f'Circular includes detected: {chain}') ; ValueError(f'Include path must be a string in {rpath}: {item}')`
- 주요 호출: `ValueError, _load_single_config, _load_with_includes, data.pop, deep_update, join, path.resolve, resolve`

##### `_resolve_path_value(value: str, config_dir: Path)`
- 위치: `hygel_martini/param_opt/core/config.py:85`
- 종류: function, private/internal
- 역할: `resolve path value` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `str(path_obj.resolve())`
- 주요 호출: `Path, os.path.expanduser, os.path.expandvars, path_obj.is_absolute, path_obj.resolve, resolved.replace`

##### `_normalize_paths(cfg: Dict[str, Any], config_path: Path | None)`
- 위치: `hygel_martini/param_opt/core/config.py:95`
- 종류: function, private/internal
- 역할: `normalize paths` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 3개. 예: `result ; cfg`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `path_section[key]`
- 주요 호출: `_resolve_path_value, config_path.exists, config_path.resolve, copy.deepcopy, key.endswith, path_section.items, result.get`

##### `_parse_override_value(raw: str)`
- 위치: `hygel_martini/param_opt/core/config.py:111`
- 종류: function, private/internal
- 역할: `parse override value` 계열 parser입니다. 문자열/파일 내용을 내부 자료구조로 바꾸며, 입력 형식이 맞지 않으면 예외 또는 None 경로를 사용합니다.
- 반환: 명시적 return 10개. 예: `value ; '' ; True ; False ; None`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `_FLOAT_RE.match, _INT_RE.match, json.loads, raw.strip, value.lower, yaml.safe_load`

##### `_apply_set_override(cfg: Dict[str, Any], expr: str)`
- 위치: `hygel_martini/param_opt/core/config.py:148`
- 종류: function, private/internal
- 역할: `apply set override` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 예외/검증: `TypeError(f"Cannot apply --set {expr!r}: {'.'.join(keys[:-1])!r} is not a mapping") ; ValueError(f'Invalid --set override path: {expr!r}') ; ValueError(f'Invalid --set override: {expr!r}. Expected key.path=value')`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `current[keys[-1]], current[key]`
- 주요 호출: `TypeError, ValueError, _parse_override_value, current.get, expr.split, join, part.strip, path_text.split`

##### `load_config(config_path: Path | None, default_config: Dict[str, Any] | None=None)`
- 위치: `hygel_martini/param_opt/core/config.py:172`
- 종류: function
- 역할: `load config` 계열 loader입니다. 설정/파일/템플릿을 읽어 후속 builder가 사용할 dict/dataclass 구조로 정규화합니다.
- 반환: 명시적 return 1개. 예: `_normalize_paths(cfg, config_path)`
- 주요 호출: `_load_with_includes, _normalize_paths, config_path.exists, copy.deepcopy, deep_update`

##### `apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace)`
- 위치: `hygel_martini/param_opt/core/config.py:180`
- 종류: function
- 역할: `apply cli overrides` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `result`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `result['system']['symbols'], result['system']['sequences'], result['system']['lengths'], result['system']['replicas'], result['system']['cutoff_nm'], result['system']['min_box_safety_nm'], result['system']['temperature_c'], result['paths']['out_root'], result['system']['solvate_tool'], result['system']['n_torsion_mode']` ...
- 주요 호출: `_apply_set_override, copy.deepcopy, getattr, parse_csv_list, parse_int_csv, parse_semicolon_list, result.setdefault`

##### `add_config_args(parser: argparse.ArgumentParser)`
- 위치: `hygel_martini/param_opt/core/config.py:238`
- 종류: function
- 역할: `add config args` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 주요 호출: `parser.add_argument`

##### `add_sequence_override_args(parser: argparse.ArgumentParser)`
- 위치: `hygel_martini/param_opt/core/config.py:251`
- 종류: function
- 역할: `add sequence override args` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 주요 호출: `parser.add_argument`

##### `add_opls_to_martini_cli_args(parser: argparse.ArgumentParser)`
- 위치: `hygel_martini/param_opt/core/config.py:262`
- 종류: function
- 역할: `add opls to martini cli args` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 주요 호출: `add_config_args, add_sequence_override_args, parser.add_argument`

##### `add_qm_to_martini_cli_args(parser: argparse.ArgumentParser)`
- 위치: `hygel_martini/param_opt/core/config.py:280`
- 종류: function
- 역할: `add qm to martini cli args` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 주요 호출: `add_config_args, add_sequence_override_args, parser.add_argument`

### `hygel_martini/param_opt/core/physics.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `from __future__ import annotations, from typing import Sequence`
- class 수: 0, 함수/메서드 수: 2

#### Functions and methods

##### `water_density_g_cm3(temp_c: float)`
- 위치: `hygel_martini/param_opt/core/physics.py:6`
- 종류: function
- 역할: Return water density in g/cm^3 at the target temperature. Prefer CoolProp. Fallback to Kell equation approximation (0-100C).
- 반환: 명시적 return 2개. 예: `rho_kg_m3 / 1000.0`
- 주요 호출: `PropsSI`

##### `estimate_water_molecules(box_ang: Sequence[float], density_g_cm3: float, molar_mass: float, avogadro: float)`
- 위치: `hygel_martini/param_opt/core/physics.py:25`
- 종류: function
- 역할: `estimate water molecules` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `max(0, n_molecules)`

### `hygel_martini/param_opt/core/utils.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `from __future__ import annotations, from typing import List, Sequence`
- class 수: 0, 함수/메서드 수: 5

#### Functions and methods

##### `parse_csv_list(text: str)`
- 위치: `hygel_martini/param_opt/core/utils.py:6`
- 종류: function
- 역할: `parse csv list` 계열 parser입니다. 문자열/파일 내용을 내부 자료구조로 바꾸며, 입력 형식이 맞지 않으면 예외 또는 None 경로를 사용합니다.
- 반환: 명시적 return 1개. 예: `[token.strip() for token in text.split(',') if token.strip()]`
- 주요 호출: `text.split, token.strip`

##### `parse_semicolon_list(text: str)`
- 위치: `hygel_martini/param_opt/core/utils.py:10`
- 종류: function
- 역할: `parse semicolon list` 계열 parser입니다. 문자열/파일 내용을 내부 자료구조로 바꾸며, 입력 형식이 맞지 않으면 예외 또는 None 경로를 사용합니다.
- 반환: 명시적 return 1개. 예: `[token.strip() for token in text.split(';') if token.strip()]`
- 주요 호출: `text.split, token.strip`

##### `parse_int_csv(text: str)`
- 위치: `hygel_martini/param_opt/core/utils.py:14`
- 종류: function
- 역할: `parse int csv` 계열 parser입니다. 문자열/파일 내용을 내부 자료구조로 바꾸며, 입력 형식이 맞지 않으면 예외 또는 None 경로를 사용합니다.
- 반환: 명시적 return 1개. 예: `values`
- 예외/검증: `ValueError('lengths is empty')`
- 주요 호출: `ValueError, parse_csv_list, values.append`

##### `sequence_name(symbol: str, n_repeat: int)`
- 위치: `hygel_martini/param_opt/core/utils.py:23`
- 종류: function
- 역할: `sequence name` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `symbol * n_repeat`

##### `ensure_min_box_nm(box_nm: Sequence[float], cutoff_nm: float, safety_nm: float)`
- 위치: `hygel_martini/param_opt/core/utils.py:27`
- 종류: function
- 역할: `ensure min box nm` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `[max(value, min_len) for value in box_nm]`

### `hygel_martini/param_opt/opls_to_martini/builder.py`

02 OPLS -> Martini case directory/job 생성 로직입니다.
- 주요 import: `from __future__ import annotations, copy, json, shutil, from pathlib import Path, from typing import Any, Dict, List, from ase.io import read, write, from ..polymer_maker.maker import build_polymer, load_monomer_library, from ..core.physics import estimate_water_molecules, water_density_g_cm3, from ..core.utils import ensure_min_box_nm, parse_csv_list, from .defaults import NM_TO_ANGSTROM, from .writers import write_gromacs_mdp_templates, write_packmol_input, write_pipeline_script, write_text, write_topol_stub`
- class 수: 0, 함수/메서드 수: 4

#### Functions and methods

##### `_sequence_stem(tokens: List[str])`
- 위치: `hygel_martini/param_opt/opls_to_martini/builder.py:25`
- 종류: function, private/internal
- 역할: `sequence stem` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `'_'.join(tokens) ; ''.join(tokens)`
- 주요 호출: `join`

##### `_parse_sequence_entry(entry: Any, monomer_keys: set[str])`
- 위치: `hygel_martini/param_opt/opls_to_martini/builder.py:31`
- 종류: function, private/internal
- 역할: `parse sequence entry` 계열 parser입니다. 문자열/파일 내용을 내부 자료구조로 바꾸며, 입력 형식이 맞지 않으면 예외 또는 None 경로를 사용합니다.
- 반환: 명시적 return 1개. 예: `tokens`
- 예외/검증: `TypeError(f'Unsupported sequence entry type: {type(entry)!r}') ; ValueError('Empty sequence entry is not allowed') ; ValueError('Sequence entry produced no tokens')`
- 주요 호출: `TypeError, ValueError, entry.strip, parse_csv_list, text.split`

##### `_build_sequence_jobs(system_cfg: Dict[str, Any], monomer_keys: set[str])`
- 위치: `hygel_martini/param_opt/opls_to_martini/builder.py:55`
- 종류: function, private/internal
- 역할: `build sequence jobs` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 2개. 예: `jobs`
- 예외/검증: `ValueError('system.sequences is empty') ; ValueError('system.sequences must be a list when provided')`
- 주요 호출: `ValueError, _parse_sequence_entry, jobs.append, system_cfg.get`

##### `build_cases(cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/opls_to_martini/builder.py:74`
- 종류: function
- 역할: `build cases` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 1개. 예: `result`
- 예외/검증: `KeyError(f'Unknown monomer token: {token}. Available: {sorted(library.keys())}') ; ValueError('topology.polymer_itp must be a paths.base_dir-relative path, not an absolute path')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `case_cfg['topology']['polymer_itp']`
- 주요 호출: `KeyError, Path, Path.resolve, ValueError, _build_sequence_jobs, _sequence_stem, all_cases.append, atoms.positions.max, atoms.positions.min, build_polymer, case_dir.mkdir, cfg.get, copy.deepcopy, ensure_min_box_nm, estimate_water_molecules, json.dumps, library.keys, load_monomer_library, out_root.mkdir, polymer_itp_src.exists, polymer_itp_src.is_absolute, read, rep_dir.mkdir, replicas_info.append, ... (+10)`

### `hygel_martini/param_opt/opls_to_martini/cli.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `from __future__ import annotations, argparse, json, from pathlib import Path, from ..core.config import add_opls_to_martini_cli_args, from .defaults import DEFAULT_CONFIG, from .generator import run_opls_to_martini, from .writers import write_text`
- class 수: 0, 함수/메서드 수: 2

#### Functions and methods

##### `build_arg_parser()`
- 위치: `hygel_martini/param_opt/opls_to_martini/cli.py:13`
- 종류: function
- 역할: `build arg parser` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 1개. 예: `parser`
- 주요 호출: `add_opls_to_martini_cli_args, argparse.ArgumentParser`

##### `main()`
- 위치: `hygel_martini/param_opt/opls_to_martini/cli.py:21`
- 종류: function, CLI entry
- 역할: `main` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 예외/검증: `ValueError('--dump-default-config needs --config path')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Path, ValueError, build_arg_parser, json.dumps, parser.parse_args, run_opls_to_martini, write_text`

### `hygel_martini/param_opt/opls_to_martini/generator.py`

Thin workflow entry helper for OPLS-to-Martini case generation.
- 주요 import: `from __future__ import annotations, argparse, from pathlib import Path, from typing import Any, Dict, from ..core.config import apply_cli_overrides, load_config, from .builder import build_cases, from .defaults import DEFAULT_CONFIG`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `run_opls_to_martini(config_path: str | Path, overrides: argparse.Namespace | None=None)`
- 위치: `hygel_martini/param_opt/opls_to_martini/generator.py:14`
- 종류: function
- 역할: Load an opls_to_martini maker file, apply optional overrides, and build cases.
- 반환: 명시적 return 1개. 예: `(cfg, result)`
- 예외/검증: `ValueError('replicas must be >= 1') ; ValueError('sample_nsteps must be >= 1')`
- 주요 호출: `Path, ValueError, apply_cli_overrides, build_cases, load_config`

### `hygel_martini/param_opt/opls_to_martini/writers.py`

Packmol, GROMACS MDP, topology stub, shell pipeline script를 파일로 렌더링합니다.
- 주요 import: `from __future__ import annotations, from pathlib import Path, from typing import Any, Dict, Sequence`
- class 수: 0, 함수/메서드 수: 5

#### Functions and methods

##### `write_text(path: Path, text: str)`
- 위치: `hygel_martini/param_opt/opls_to_martini/writers.py:7`
- 종류: function
- 역할: `write text` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `path.parent.mkdir, path.write_text`

##### `write_packmol_input(path: Path, polymer_xyz: Path, output_xyz: str, box_ang: Sequence[float], n_waters: int, seed: int, cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/opls_to_martini/writers.py:12`
- 종류: function
- 역할: `write packmol input` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Path, polymer_ref.as_posix, write_text`

##### `write_gromacs_mdp_templates(case_dir: Path, cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/opls_to_martini/writers.py:41`
- 종류: function
- 역할: `write gromacs mdp templates` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `write_text`

##### `write_topol_stub(path: Path, cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/opls_to_martini/writers.py:122`
- 종류: function
- 역할: `write topol stub` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `cfg.get, cfg.get.get, include_lines.append, join, molecules_lines.append, top_cfg.get, write_text`

##### `write_pipeline_script(replica_dir: Path, box_nm: Sequence[float], cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/opls_to_martini/writers.py:156`
- 종류: function
- 역할: `write pipeline script` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `chmod, write_text`

### `hygel_martini/param_opt/polymer_maker/maker.py`

ASE 기반 monomer 연결 polymer XYZ 생성기입니다. Br connector를 사용해 monomer를 이어붙이고 terminal을 H로 cap합니다.
- 주요 import: `from pathlib import Path, numpy, from ase.io import read, write`
- class 수: 0, 함수/메서드 수: 6

#### Functions and methods

##### `get_connection_info(atoms)`
- 위치: `hygel_martini/param_opt/polymer_maker/maker.py:15`
- 종류: function
- 역할: 모노머에서 연결 정보를 추출합니다. return: c0_idx: Head Carbon index (0) c1_idx: Tail Carbon index (1) bc_head_idx: C0에 연결된 BASE_CONNECTOR 인덱스 bc_tail_idx: C1에 연결된 BASE_CONNECTOR 인덱스
- 반환: 명시적 return 1개. 예: `(c0_idx, c1_idx, bc_head_idx, bc_tail_idx)`
- 예외/검증: `ValueError(f"모노머 {atoms.info.get('name', '')}에서 연결용 {BASE_CONNECTOR}을 찾을 수 없습니다.")`
- 주요 호출: `ValueError, atoms.get_distance, atoms.info.get`

##### `cap_ends_with_hydrogen(atoms)`
- 위치: `hygel_martini/param_opt/polymer_maker/maker.py:47`
- 종류: function
- 역할: 양 끝단에 남아있는 BASE_CONNECTOR을 H로 치환하고 길이를 1.094A로 조정합니다.
- 반환: 명시적 return 1개. 예: `atoms`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `atoms[bc_idx].symbol, atoms.positions[bc_idx]`
- 주요 호출: `atoms.get_distance, np.linalg.norm`

##### `normalize_sequence(sequence)`
- 위치: `hygel_martini/param_opt/polymer_maker/maker.py:80`
- 종류: function
- 역할: 시퀀스 입력을 토큰 리스트로 정규화합니다. - ["S", "D", "B"] 형태를 권장 - "S,D,B" / "S D B" 문자열도 허용 - "SDB" 문자열은 한 글자 토큰으로 처리
- 반환: 명시적 return 1개. 예: `tokens`
- 예외/검증: `ValueError('시퀀스가 비어 있습니다.')`
- 주요 호출: `ValueError, seq.split, sequence.strip, tok.strip`

##### `load_monomer_library(monomer_files, base_dir=None)`
- 위치: `hygel_martini/param_opt/polymer_maker/maker.py:103`
- 종류: function
- 역할: monomer_files: {"S": "NEW_SBMA.xyz", "D": "NEW_DMAPS.xyz"} 또는 {"S": {"xyz": "NEW_SBMA.xyz"}, ...} 형태
- 반환: 명시적 return 1개. 예: `monomer_dict`
- 예외/검증: `FileNotFoundError(f'모노머 파일이 없습니다: {path_obj}') ; ValueError(f"모노머 '{symbol}'의 xyz 경로가 비어 있습니다.")`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `atoms.info['name'], monomer_dict[symbol]`
- 주요 호출: `FileNotFoundError, Path, Path.resolve, ValueError, monomer_files.items, path_obj.exists, path_obj.is_absolute, raw_entry.get, read`

##### `_sequence_output_stem(sequence_tokens)`
- 위치: `hygel_martini/param_opt/polymer_maker/maker.py:132`
- 종류: function, private/internal
- 역할: `sequence output stem` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `'_'.join(sequence_tokens) ; ''.join(sequence_tokens)`
- 주요 호출: `join`

##### `build_polymer(sequence, monomer_dict, n_torsion, output_filename=None, output_dir='output')`
- 위치: `hygel_martini/param_opt/polymer_maker/maker.py:138`
- 종류: function
- 역할: Args: sequence: ["S", "D", "D", "S", "B"] 또는 "S,D,D,S,B" monomer_dict: {"S": ase.Atoms(...), "D": ase.Atoms(...)} 형태 n_torsion: 비틀림 각도를 결정하는 정수 (1, 2, 3, 4...) output_filename: 저장할 파일명 output_dir: 생성된 xyz를 저장할 디렉터리
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 예외/검증: `KeyError(f"시퀀스 심볼 '{symbol}'이 monomer_dict에 없습니다.")`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `KeyError, Path, _sequence_output_stem, cap_ends_with_hydrogen, get_connection_info, monomer_dict.copy, new_monomer.rotate, new_monomer.translate, normalize_sequence, np.linalg.norm, output_root.mkdir, write`

### `hygel_martini/param_opt/qm_to_martini/cli.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `from __future__ import annotations, argparse, json, from pathlib import Path, from ..core.config import add_qm_to_martini_cli_args, from ..opls_to_martini.writers import write_text, from .defaults import DEFAULT_CONFIG, from .generator import run_qm_to_martini`
- class 수: 0, 함수/메서드 수: 2

#### Functions and methods

##### `build_arg_parser()`
- 위치: `hygel_martini/param_opt/qm_to_martini/cli.py:13`
- 종류: function
- 역할: `build arg parser` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 1개. 예: `parser`
- 주요 호출: `add_qm_to_martini_cli_args, argparse.ArgumentParser, parser.add_argument`

##### `main()`
- 위치: `hygel_martini/param_opt/qm_to_martini/cli.py:32`
- 종류: function, CLI entry
- 역할: `main` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개이지만 값 없는 return 경로가 중심입니다.
- 예외/검증: `SystemExit(1) ; ValueError('--dump-default-config needs --config path')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Path, SystemExit, ValueError, build_arg_parser, cfg.get, json.dumps, key.replace, parser.parse_args, run_qm_to_martini, write_text`

### `hygel_martini/param_opt/qm_to_martini/config.py`

03 QM/xTB -> Martini workflow의 설정 정규화, dataclass 모델, 실행 환경 검사, 스크립트 렌더링 유틸리티가 모여 있습니다. 이 파일은 “설정을 신뢰 가능한 내부 dict로 바꾸는 층”입니다.
- 주요 import: `from __future__ import annotations, os, re, shlex, shutil, subprocess, from dataclasses import dataclass, field, from pathlib import Path, from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, from fractions import Fraction, from ..core.utils import parse_csv_list`
- class 수: 9, 함수/메서드 수: 40

#### Classes

- `ConnectionDetectionConfig` (hygel_martini/param_opt/qm_to_martini/config.py:18)
  - decorators: `dataclass(frozen=True)`
  - 주요 field/class var: `indicator: str, cutoff: float`
- `TermGenerationConfig` (hygel_martini/param_opt/qm_to_martini/config.py:23)
  - decorators: `dataclass(frozen=True)`
  - 주요 field/class var: `mode: str, n: int`
- `WeightedAtomRef` (hygel_martini/param_opt/qm_to_martini/config.py:28)
  - decorators: `dataclass(frozen=True)`
  - 주요 field/class var: `atom_index: int, denominator: int`
- `ValidationReport` (hygel_martini/param_opt/qm_to_martini/config.py:42)
  - decorators: `dataclass`
  - 주요 field/class var: `target: str, problems: List[str], warnings: List[str]`
- `MonomerTemplate` (hygel_martini/param_opt/qm_to_martini/config.py:62)
  - decorators: `dataclass`
  - 주요 field/class var: `path: Path, preamble: List[str], beads: Dict[int, List[WeightedAtomRef]], bonds: List[Tuple[int, int]], constraints: List[Tuple[int, int]], angles: List[Tuple[int, int, int]], dihedrals: List[Tuple[int, int, int, int]], impropers: List[Tuple[int, int, int, int]]`
- `PolymerInputBundle` (hygel_martini/param_opt/qm_to_martini/config.py:83)
  - decorators: `dataclass`
  - 주요 field/class var: `base: MonomerTemplate, augmented: MonomerTemplate, base_text: str, augmented_text: str, base_report: ValidationReport, augmented_report: ValidationReport, connection_bonds: List[Tuple[int, int]], connection_beads: List[int], backbone_beads: List[int]`
- `ParamLine` (hygel_martini/param_opt/qm_to_martini/config.py:95)
  - decorators: `dataclass(frozen=True)`
  - 주요 field/class var: `section: str, indices: Tuple[int, ...], tokens: Tuple[str, ...], commented: bool, inline_comment: str, rmsd: Optional[float], raw: str`
- `TypedRecord` (hygel_martini/param_opt/qm_to_martini/config.py:105)
  - decorators: `dataclass(frozen=True)`
  - 주요 field/class var: `section: str, category: str, angle_dist: str, type_names: Tuple[str, ...], display_labels: Tuple[str, ...], indices: Tuple[int, ...], tokens: Tuple[str, ...], commented: bool, inline_comment: str, rmsd: Optional[float], source_tag: str, source_path: str`
- `MergedVariant` (hygel_martini/param_opt/qm_to_martini/config.py:120)
  - decorators: `dataclass`
  - 주요 field/class var: `section: str, category: str, angle_dist: str, type_names: Tuple[str, ...], display_labels: List[Tuple[str, ...]], tokens: Tuple[str, ...], commented: bool, sources: List[str], indices_examples: List[Tuple[int, ...]], inline_comments: List[str], rmsd_values: List[float], primary: bool`

#### Functions and methods

##### `WeightedAtomRef.weight(self)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:33`
- 종류: method, property
- decorators: `property`
- 역할: `weight` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `Fraction(1, self.denominator)`
- 주요 호출: `Fraction`

##### `WeightedAtomRef.format(self)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:36`
- 종류: method
- 역할: `format` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 2개. 예: `f'{self.atom_index}/{self.denominator}' ; str(self.atom_index)`

##### `ValidationReport.ok(self)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:48`
- 종류: method, property
- decorators: `property`
- 역할: `ok` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `not self.problems`

##### `ValidationReport.render(self)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:51`
- 종류: method
- 역할: `render` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 1개. 예: `'\n'.join(lines) + '\n'`
- 주요 호출: `join, lines.append, lines.extend`

##### `MonomerTemplate.bead_count(self)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:73`
- 종류: method, property
- decorators: `property`
- 역할: `bead count` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `len(self.beads)`

##### `MonomerTemplate.atom_count(self)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:77`
- 종류: method, property
- decorators: `property`
- 역할: `atom count` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `max((ref.atom_index for refs in self.beads.values() for ref in refs)) ; 0`
- 주요 호출: `self.beads.values`

##### `write_text(path: Path, text: str)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:134`
- 종류: function
- 역할: `write text` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `path.parent.mkdir, path.write_text`

##### `shell_assign(name: str, value: str)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:138`
- 종류: function
- 역할: `shell assign` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `f'{name}={shlex.quote(value)}'`
- 주요 호출: `shlex.quote`

##### `resolve_under_base(base_dir: Path, value: str | Path)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:141`
- 종류: function
- 역할: `resolve under base` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 2개. 예: `(base_dir / path).resolve() ; path`
- 주요 호출: `Path, path.is_absolute, resolve`

##### `parse_bool(value: Any, default: bool=False)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:147`
- 종류: function
- 역할: `parse bool` 계열 parser입니다. 문자열/파일 내용을 내부 자료구조로 바꾸며, 입력 형식이 맞지 않으면 예외 또는 None 경로를 사용합니다.
- 반환: 명시적 return 4개. 예: `str(value).strip().lower() in {'1', 'true', 'yes', 'on'} ; default ; value ; bool(value)`

##### `resolve_connection_detection_config(pipeline_cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:156`
- 종류: function
- 역할: `resolve connection detection config` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `ConnectionDetectionConfig(indicator=indicator, cutoff=cutoff)`
- 예외/검증: `ValueError('bartender_pipeline.connection_cutoff must be > 0')`
- 주요 호출: `ConnectionDetectionConfig, ValueError, pipeline_cfg.get`

##### `resolve_term_generation_config(pipeline_cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:166`
- 종류: function
- 역할: `resolve term generation config` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `TermGenerationConfig(mode=normalized_mode, n=budget)`
- 예외/검증: `TypeError('bartender_pipeline.term_generation must be a mapping or string when provided') ; ValueError('bartender_pipeline.term_generation.n must be >= 0') ; ValueError(f'bartender_pipeline.term_generation.mode must be one of {sorted(supported)} (or alias exhaustive/original), got {mode!r}')`
- 주요 호출: `TermGenerationConfig, TypeError, ValueError, aliases.get, pipeline_cfg.get, raw_cfg.get`

##### `default_workdir_name(relaxation: str, md: str)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:216`
- 종류: function
- 역할: `default workdir name` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `name ; fallback`
- 예외/검증: `ValueError(f'Unsupported md mode: {md}')`
- 주요 호출: `ValueError, _WORKDIR_NAMES.get`

##### `_normalize_pipeline_mode(value: Any, default: str, field_name: str)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:227`
- 종류: function, private/internal
- 역할: `normalize pipeline mode` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 3개. 예: `str(value).strip().lower() ; default ; 'off'`
- 예외/검증: `ValueError(f'{field_name} must be one of the documented string modes, not boolean true')`
- 주요 호출: `ValueError`

##### `resolve_pipeline_modes(pipeline_cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:236`
- 종류: function
- 역할: `resolve pipeline modes` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `{'relaxation': relaxation, 'md': md, 'workdir_name': workdir_name or default_workdir_name(relaxation, md)}`
- 예외/검증: `ValueError('bartender_pipeline.md must be one of: bartender, xtb, existing, xtb_nobartender, off') ; ValueError('bartender_pipeline.relaxation must be one of: xtb, orca, off') ; ValueError(f'Unsupported legacy relaxation backend: {backend}')`
- 주요 호출: `ValueError, _normalize_pipeline_mode, default_workdir_name, legacy_bartender_cfg.get, legacy_relax_cfg.get, mode_cfg.get, pipeline_cfg.get`

##### `resolve_spin_state(uhf_value: Any, multiplicity_value: Any, *, label: str)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:313`
- 종류: function
- 역할: `resolve spin state` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 2개. 예: `(uhf, multiplicity) ; (0, 1)`
- 예외/검증: `ValueError(f'{label}: multiplicity ({multiplicity}) must equal uhf + 1 ({uhf + 1})') ; ValueError(f'{label}: multiplicity must be >= 1') ; ValueError(f'{label}: uhf must be >= 0')`
- 주요 호출: `ValueError`

##### `_normalize_index_list(raw: Any, *, label: str)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:339`
- 종류: function, private/internal
- 역할: `normalize index list` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 2개. 예: `normalized ; []`
- 예외/검증: `TypeError(f'{label} must be an integer or a list of integers') ; ValueError(f'{label} must contain 0-based atom indices')`
- 주요 호출: `TypeError, ValueError, normalized.append, seen.add`

##### `resolve_backbone_atom_config(raw: Any, *, label: str)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:361`
- 종류: function
- 역할: `resolve backbone atom config` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 2개. 예: `{'head': head, 'tail': tail, 'body': body} ; {'head': [1], 'tail': [2], 'body': []}`
- 예외/검증: `TypeError(f'{label} must be a mapping with optional head/body/tail atom lists') ; ValueError(f'{label} must define at least one of head or tail')`
- 주요 호출: `TypeError, ValueError, _normalize_index_list, raw.get`

##### `export_backbone_atom_config(cfg: Dict[str, List[int]])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:374`
- 종류: function
- 역할: `export backbone atom config` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `{key: [int(value) - 1 for value in cfg.get(key, [])] for key in ('head', 'tail', 'body')}`
- 주요 호출: `cfg.get`

##### `normalize_monomer_configs(raw_monomers: Dict[str, Any], legacy_init_templates: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:380`
- 종류: function
- 역할: `normalize monomer configs` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `normalized`
- 예외/검증: `TypeError(f'monomers.{token} must be a string or mapping') ; ValueError(f'monomers.{token}.xyz is required')`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `normalized[token]`
- 주요 호출: `TypeError, ValueError, entry.get, legacy_init_templates.get, raw_monomers.items, resolve_backbone_atom_config, resolve_spin_state`

##### `resolve_case_electronic_state(tokens: Sequence[str], monomer_cfg: Dict[str, Dict[str, Any]], pipeline_cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:414`
- 종류: function
- 역할: `resolve case electronic state` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `{'charge': charge, 'uhf': uhf, 'multiplicity': multiplicity, 'inferred_charge': inferred_charge, 'inferred_uhf': inferred_uhf}`
- 예외/검증: `TypeError('bartender_pipeline.electronic_state must be a mapping')`
- 주요 호출: `TypeError, pipeline_cfg.get, resolve_spin_state, state_cfg.get`

##### `resolve_optional_path(base_dir: Path, raw_value: Any)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:450`
- 종류: function
- 역할: `resolve optional path` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 2개. 예: `resolve_under_base(base_dir, value) ; None`
- 주요 호출: `resolve_under_base`

##### `resolve_xtb_settings(pipeline_cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:456`
- 종류: function
- 역할: `resolve xtb settings` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `{'env_script': str(xtb_cfg.get('env_script', legacy_relax_cfg.get('xtb_env_script', ''))).strip(), 'binary': str(xtb_cfg.get('binary', legacy_relax_cfg.get('xtb_binary', 'xtb'))...`
- 예외/검증: `TypeError('bartender_pipeline.xtb must be a mapping') ; TypeError('bartender_pipeline.xtb.md must be a mapping') ; ValueError('bartender_pipeline.xtb.solvent_model must be one of: off, alpb, gbsa')`
- 주요 호출: `TypeError, ValueError, legacy_relax_cfg.get, md_cfg.get, parse_bool, pipeline_cfg.get, xtb_cfg.get`

##### `resolve_orca_settings(pipeline_cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:500`
- 종류: function
- 역할: `resolve orca settings` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `{'binary': str(orca_cfg.get('binary', legacy_relax_cfg.get('orca_binary', 'orca'))).strip(), 'nprocs': int(orca_cfg.get('nprocs', legacy_relax_cfg.get('nprocs', 32))), 'method_l...`
- 예외/검증: `TypeError('bartender_pipeline.orca must be a mapping')`
- 주요 호출: `TypeError, legacy_relax_cfg.get, orca_cfg.get, pipeline_cfg.get`

##### `_inspect_configured_executable(base_dir: Path, raw_value: Any)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:523`
- 종류: function, private/internal
- 역할: `inspect configured executable` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 3개. 예: `{'configured': configured, 'resolved': found, 'exists': found is not None, 'lookup': 'PATH'} ; {'configured': configured, 'resolved': None, 'exists': False, 'lookup': 'missing'} ; {'configured': configured, 'resolved': fallback or str(path), 'exists': path.exists() or fallback is not None, 'lookup': 'path->PATH' if fallback else 'path'}`
- 주요 호출: `configured.startswith, path.exists, resolve_under_base, shutil.which`

##### `resolve_executable_command(base_dir: Path, raw_value: Any)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:551`
- 종류: function
- 역할: `resolve executable command` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 2개. 예: `str(payload.get('configured') or '').strip() ; resolved`
- 주요 호출: `_inspect_configured_executable, payload.get`

##### `_inspect_optional_file(base_dir: Path, raw_value: Any)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:558`
- 종류: function, private/internal
- 역할: `inspect optional file` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `{'configured': configured, 'resolved': str(path), 'exists': path.exists(), 'lookup': 'path'} ; {'configured': configured, 'resolved': None, 'exists': True, 'lookup': 'optional-empty'}`
- 주요 호출: `path.exists, resolve_under_base`

##### `check_configured_tools(cfg: Dict[str, Any], requested: Optional[Sequence[str]]=None)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:575`
- 종류: function
- 역할: `check configured tools` 검증 helper입니다. 입력 일관성, tool availability, template/topology 조건을 확인합니다.
- 반환: 명시적 return 1개. 예: `{'ok': ok, 'base_dir': str(base_dir), 'tools': tools}`
- 예외/검증: `TypeError('bartender_pipeline must be a mapping') ; TypeError('bartender_pipeline.bartender must be a mapping')`
- 주요 호출: `Path, Path.resolve, TypeError, _inspect_configured_executable, _inspect_optional_file, bartender_cfg.get, cfg.get, orca_cfg.get, pipeline_cfg.get, resolve_orca_settings, resolve_xtb_settings, tools.append, xtb_cfg.get`

##### `resolve_execution_settings(pipeline_cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:623`
- 종류: function
- 역할: `resolve execution settings` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `{'run_relaxation': parse_bool(exec_cfg.get('run_relaxation', False)), 'run_bartender': parse_bool(exec_cfg.get('run_bartender', bartender_cfg.get('execute', False))), 'shell': s...`
- 예외/검증: `TypeError('bartender_pipeline.execution must be a mapping')`
- 주요 호출: `TypeError, bartender_cfg.get, exec_cfg.get, parse_bool, pipeline_cfg.get`

##### `_get_slurm_cpu_count()`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:647`
- 종류: function, private/internal
- 역할: `get slurm cpu count` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 3개. 예: `0 ; max(1, int(val))`
- 주요 호출: `os.environ.get`

##### `resolve_log_settings(pipeline_cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:656`
- 종류: function
- 역할: `resolve log settings` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `{'enabled': parse_bool(log_cfg.get('enabled', True), True), 'dirname': str(log_cfg.get('dirname', 'logs')).strip() or 'logs', 'write_validation': parse_bool(log_cfg.get('write_v...`
- 예외/검증: `TypeError('bartender_pipeline.logs must be a mapping')`
- 주요 호출: `TypeError, log_cfg.get, parse_bool, pipeline_cfg.get`

##### `ensure_case_logs_dir(case_dir: Path, log_cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:670`
- 종류: function
- 역할: `ensure case logs dir` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `logs_dir ; None`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `log_cfg.get, logs_dir.mkdir`

##### `execute_case_script(label: str, script_path: Path, cwd: Path, exec_cfg: Dict[str, Any], logs_dir: Optional[Path])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:677`
- 종류: function
- 역할: `execute case script` 실행 helper입니다. workflow 단계나 외부 command/script를 실행 또는 위임합니다.
- 반환: 명시적 return 1개. 예: `{'script': script_path.name, 'cwd': str(cwd), 'shell': str(exec_cfg.get('shell', 'bash')), 'slurm': slurm_enabled, 'use_srun': use_srun, 'command': command, 'returncode': result...`
- 예외/검증: `RuntimeError("execution.use_srun=true but 'srun' was not found in PATH") ; RuntimeError(result.stderr.strip() if capture_runtime else f'{label} failed with exit code {result.returncode}')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 외부 command/subprocess 실행 가능
- 주요 호출: `RuntimeError, exec_cfg.get, os.environ.get, parse_bool, result.stderr.strip, shutil.which, srun_command.extend, subprocess.run, write_text`

##### `render_xtb_md_input(md_mode: str, xtb_cfg: Dict[str, Any], template_text: Optional[str])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:727`
- 종류: function
- 역할: `render xtb md input` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 2개. 예: `f"$md\n temp={xtb_cfg['md_temp_k']:.3f}\n time={xtb_cfg['md_time_ps']:.3f}\n dump={xtb_cfg['md_dump_fs']:.3f}\n step={xtb_cfg['md_step_fs']:.3f}\n velo={('true' if xtb_cfg['md_v... ; template_text.rstrip() + '\n'`
- 예외/검증: `ValueError("xTB MD input generation only supports md_mode 'nvt'")`
- 주요 호출: `ValueError, template_text.rstrip`

##### `render_orca_input(local_xyz_name: str, state: Dict[str, Any], orca_cfg: Dict[str, Any], template_text: Optional[str])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:747`
- 종류: function
- 역할: `render orca input` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 2개. 예: `f"{method_line}\n%pal nprocs {int(orca_cfg['nprocs'])} end\n\n%geom\n MaxIter {int(orca_cfg['max_iter'])}\nend\n\n* xyzfile {int(state['charge'])} {int(state['multiplicity'])} {... ; template_text.rstrip() + '\n\n' + f"* xyzfile {int(state['charge'])} {int(state.get('multiplicity', 1))} {local_xyz_name}\n"`
- 주요 호출: `method_line.startswith, orca_cfg.get, orca_cfg.strip, re.search, re.sub, state.get, template_text.rstrip`

##### `normalize_sequence(sequence: Sequence[str] | str)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:781`
- 종류: function
- 역할: `normalize sequence` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `tokens`
- 예외/검증: `ValueError('Sequence is empty.') ; ValueError('Sequence produced no tokens.')`
- 주요 호출: `ValueError, sequence.strip, text.split, token.strip`

##### `sequence_stem(tokens: Sequence[str])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:798`
- 종류: function
- 역할: `sequence stem` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `'_'.join(tokens) ; ''.join(tokens)`
- 주요 호출: `join`

##### `parse_sequence_entry(entry: Any, monomer_keys: set[str])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:803`
- 종류: function
- 역할: `parse sequence entry` 계열 parser입니다. 문자열/파일 내용을 내부 자료구조로 바꾸며, 입력 형식이 맞지 않으면 예외 또는 None 경로를 사용합니다.
- 반환: 명시적 return 1개. 예: `tokens`
- 예외/검증: `TypeError(f'Unsupported sequence entry type: {type(entry)!r}') ; ValueError('Empty sequence entry is not allowed') ; ValueError('Sequence entry produced no tokens')`
- 주요 호출: `TypeError, ValueError, entry.strip, parse_csv_list, text.split`

##### `build_sequence_jobs(system_cfg: Dict[str, Any], monomer_keys: set[str])`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:825`
- 종류: function
- 역할: `build sequence jobs` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 2개. 예: `jobs`
- 예외/검증: `ValueError('system.sequences is empty') ; ValueError('system.sequences must be a list when provided')`
- 주요 호출: `ValueError, jobs.append, parse_sequence_entry, system_cfg.get`

##### `parse_xyz(path: Path)`
- 위치: `hygel_martini/param_opt/qm_to_martini/config.py:843`
- 종류: function
- 역할: `parse xyz` 계열 parser입니다. 문자열/파일 내용을 내부 자료구조로 바꾸며, 입력 형식이 맞지 않으면 예외 또는 None 경로를 사용합니다.
- 반환: 명시적 return 1개. 예: `(symbols, coords)`
- 예외/검증: `ValueError(f'Empty xyz: {path}') ; ValueError(f'XYZ {path} declares {natoms} atoms but contains {len(atom_lines)} coordinates.')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `ValueError, coords.append, line.split, lines.strip, path.read_text, path.read_text.splitlines, symbols.append`

### `hygel_martini/param_opt/qm_to_martini/defaults.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `from __future__ import annotations, from typing import Any, Dict, from ..polymer_maker.maker import DEFAULT_MONOMER_FILES`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `_default_monomers()`
- 위치: `hygel_martini/param_opt/qm_to_martini/defaults.py:8`
- 종류: function, private/internal
- 역할: `default monomers` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `monomers`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `monomers[symbol]`
- 주요 호출: `DEFAULT_MONOMER_FILES.items, xyz_name.endswith`

### `hygel_martini/param_opt/qm_to_martini/generator.py`

Thin workflow entry helper for QM-to-Martini/Bartender generation.
- 주요 import: `from __future__ import annotations, argparse, from pathlib import Path, from typing import Any, Dict, from ..core.config import apply_cli_overrides, load_config, from .defaults import DEFAULT_CONFIG, from .pipeline import check_configured_tools, run_pipeline, run_postprocess_only`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `run_qm_to_martini(config_path: str | Path, overrides: argparse.Namespace | None=None)`
- 위치: `hygel_martini/param_opt/qm_to_martini/generator.py:14`
- 종류: function
- 역할: Load a qm_to_martini maker file, apply optional overrides, and run the pipeline.
- 반환: 명시적 return 1개. 예: `(cfg, result)`
- 주요 호출: `Path, apply_cli_overrides, check_configured_tools, getattr, load_config, run_pipeline, run_postprocess_only`

### `hygel_martini/param_opt/qm_to_martini/pipeline.py`

03 workflow의 중심 오케스트레이터입니다. monomer init template 검증, polymer Bartender input 생성, xTB/ORCA/Bartender job script 작성, 실행, 결과 수집/병합/postprocess를 모두 연결합니다.
- 주요 import: `from __future__ import annotations, json, math, os, re, shlex, shutil, subprocess, from collections import OrderedDict, defaultdict, deque, from dataclasses import dataclass, field, from fractions import Fraction, from itertools import combinations, permutations, ...`
- class 수: 1, 함수/메서드 수: 63

#### Classes

- `ConnectionMetadata` (hygel_martini/param_opt/qm_to_martini/pipeline.py:78)
  - decorators: `dataclass(frozen=True)`
  - 주요 field/class var: `head_carbon: int, tail_carbon: int, head_br: int, tail_br: int, left_connection_bead: int, right_connection_bead: int, backbone_beads: Tuple[int, ...]`

#### Functions and methods

##### `_distance(a: Tuple[float, float, float], b: Tuple[float, float, float])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:88`
- 종류: function, private/internal
- 역할: `distance` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `math.sqrt(sum(((x - y) ** 2 for x, y in zip(a, b))))`
- 주요 호출: `math.sqrt`

##### `_split_csv(raw: str)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:92`
- 종류: function, private/internal
- 역할: `split csv` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `[token.strip() for token in re.split('\\s*,\\s*', raw.strip()) if token.strip()]`
- 주요 호출: `raw.strip, re.split, token.strip`

##### `_parse_weighted_atom(token: str)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:96`
- 종류: function, private/internal
- 역할: `parse weighted atom` 계열 parser입니다. 문자열/파일 내용을 내부 자료구조로 바꾸며, 입력 형식이 맞지 않으면 예외 또는 None 경로를 사용합니다.
- 반환: 명시적 return 1개. 예: `WeightedAtomRef(atom_index=int(match.group(1)), denominator=denominator)`
- 예외/검증: `ValueError(f'Invalid denominator in BEADS atom token: {token}') ; ValueError(f'Malformed BEADS atom token: {token}')`
- 주요 호출: `ValueError, WeightedAtomRef, match.group, re.fullmatch`

##### `_parse_section_ints(path: Path, line: str, expected: int)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:106`
- 종류: function, private/internal
- 역할: `parse section ints` 계열 parser입니다. 문자열/파일 내용을 내부 자료구조로 바꾸며, 입력 형식이 맞지 않으면 예외 또는 None 경로를 사용합니다.
- 반환: 명시적 return 1개. 예: `values`
- 예외/검증: `ValueError(f"{path}: expected {expected} integers in line '{line}'")`
- 주요 호출: `ValueError, _split_csv`

##### `parse_bartender_inp(path: Path)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:113`
- 종류: function
- 역할: `parse bartender inp` 계열 parser입니다. 문자열/파일 내용을 내부 자료구조로 바꾸며, 입력 형식이 맞지 않으면 예외 또는 None 경로를 사용합니다.
- 반환: 명시적 return 1개. 예: `MonomerTemplate(path=path, preamble=preamble, beads=beads, bonds=[(int(a), int(b)) for a, b in bonds], constraints=[(int(a), int(b)) for a, b in constraints], angles=[(int(a), i...`
- 예외/검증: `ValueError(f"{path}: malformed BEADS line '{line}'") ; ValueError(f'{path} has no BEADS section.')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `beads[bead_id]`
- 주요 호출: `MonomerTemplate, OrderedDict, ValueError, _parse_section_ints, _parse_weighted_atom, _split_csv, match.group, path.read_text, path.read_text.splitlines, preamble.append, raw.strip, re.match, sections.append, stripped.startswith, stripped.upper`

##### `_weighted_atom_owners(template: MonomerTemplate)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:172`
- 종류: function, private/internal
- 역할: `weighted atom owners` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `owners`
- 주요 호출: `defaultdict, owners.append, template.beads.values`

##### `_connector_indices(symbols: Sequence[str], indicator: str)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:180`
- 종류: function, private/internal
- 역할: `connector indices` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `[index for index, symbol in enumerate(symbols, start=1) if str(symbol).strip().upper() == marker]`
- 주요 호출: `indicator.strip, indicator.strip.upper`

##### `infer_backbone_beads(template: MonomerTemplate, xyz_path: Path, backbone_atom_cfg: Dict[str, List[int]])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:185`
- 종류: function
- 역할: `infer backbone beads` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `tuple(backbone_beads) ; ()`
- 예외/검증: `ValueError(f'{xyz_path.name}: backbone atom indices {sorted(tracked_atoms)} exceed template atom count {template.atom_count}.') ; ValueError(f'{xyz_path.name}: backbone atoms {missing} are not assigned to any bead in the init template.') ; ValueError(f'{xyz_path.name}: no beads contain the configured backbone atoms {sorted(tracked_atoms)}.')`
- 주요 호출: `ValueError, _weighted_atom_owners, backbone_atom_cfg.get, backbone_beads.append, seen_beads.add, template.beads.items`

##### `validate_template(template: MonomerTemplate, xyz_path: Path, connection_cfg: ConnectionDetectionConfig)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:220`
- 종류: function
- 역할: `validate template` 검증 helper입니다. 입력 일관성, tool availability, template/topology 조건을 확인합니다.
- 반환: 명시적 return 1개. 예: `report`
- 주요 호출: `Fraction, ValidationReport, _connector_indices, _weighted_atom_owners, adjacency.add, adjacency.items, deque, join, owners.items, parse_xyz, queue.append, queue.popleft, ref.format, refs.format, report.problems.append, seen.add, template.beads.keys`

##### `infer_connection_metadata(template: MonomerTemplate, xyz_path: Path, connection_cfg: ConnectionDetectionConfig, backbone_atom_cfg: Dict[str, List[int]])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:302`
- 종류: function
- 역할: `infer connection metadata` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 3개. 예: `ConnectionMetadata(head_carbon=head_carbon, tail_carbon=tail_carbon, head_br=head_br, tail_br=tail_br, left_connection_bead=owner(head_br, 'head connector'), right_connection_be... ; min((_distance(coords[ref - 1], coords[connector_atom - 1]) for ref in refs)) ; owners[0]`
- 예외/검증: `ValueError(f"{xyz_path.name}: could not infer distinct head/tail '{connection_cfg.indicator}' atoms near backbone_atoms.head={user_head_refs} and backbone_atoms.tail={user_tail_... ; ValueError(f"{xyz_path.name}: could not infer head '{connection_cfg.indicator}' near backbone_atoms.head={user_head_refs} with cutoff {connection_cfg.cutoff} A.") ; ValueError(f"{xyz_path.name}: could not infer tail '{connection_cfg.indicator}' near backbone_atoms.tail={user_tail_refs} with cutoff {connection_cfg.cutoff} A.") ; ValueError(f"{xyz_path.name}: expected at least two '{connection_cfg.indicator}' connector atoms, found {len(connector_indices)}.") ; ValueError(f"{xyz_path.name}: expected exactly one bead for {label} atom {atom_index}, found {owners or 'none'}.") ; ValueError(f'{xyz_path.name} must contain at least {required_atoms} atoms.')`
- 주요 호출: `ConnectionMetadata, ValueError, _connector_indices, _distance, backbone_atom_cfg.get, candidates.sort, distance_to_refs, infer_backbone_beads, owner, parse_xyz, template.beads.items`

##### `infer_connection_metadata.distance_to_refs(connector_atom: int, refs: Sequence[int])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:328`
- 종류: nested helper
- 역할: `distance to refs` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `min((_distance(coords[ref - 1], coords[connector_atom - 1]) for ref in refs))`
- 주요 호출: `_distance`

##### `infer_connection_metadata.owner(atom_index: int, label: str)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:394`
- 종류: nested helper
- 역할: `owner` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `owners[0]`
- 예외/검증: `ValueError(f"{xyz_path.name}: expected exactly one bead for {label} atom {atom_index}, found {owners or 'none'}.")`
- 주요 호출: `ValueError, template.beads.items`

##### `_sorted_pair(a: int, b: int)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:417`
- 종류: function, private/internal
- 역할: `sorted pair` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `(a, b) if a <= b else (b, a)`

##### `_canon_angle(i: int, j: int, k: int)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:421`
- 종류: function, private/internal
- 역할: `canon angle` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `(i, j, k) if i <= k else (k, j, i)`

##### `_build_graph(edges: Iterable[Tuple[int, int]])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:425`
- 종류: function, private/internal
- 역할: `build graph` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 1개. 예: `graph`
- 주요 호출: `defaultdict, graph.add`

##### `_canon_reversible(values: Sequence[int])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:433`
- 종류: function, private/internal
- 역할: `canon reversible` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `forward if forward <= reverse else reverse`
- 주요 호출: `reversed`

##### `_reversal_unique_permutations(values: Sequence[int])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:439`
- 종류: function, private/internal
- 역할: `reversal unique permutations` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `unique`
- 주요 호출: `_canon_reversible, permutations, seen.add, unique.append, unique.sort`

##### `_generate_all_reversible_combinations(bead_ids: Sequence[int], body_size: int, existing: set[Tuple[int, ...]])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:452`
- 종류: function, private/internal
- 역할: `generate all reversible combinations` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `generated`
- 주요 호출: `_reversal_unique_permutations, combinations, generated.append, seen.add`

##### `_generate_all_linkage_bonds(inp_data: MonomerTemplate)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:468`
- 종류: function, private/internal
- 역할: `generate all linkage bonds` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `[tuple((int(value) for value in candidate)) for candidate in _generate_all_reversible_combinations(bead_ids, 2, existing)]`
- 주요 호출: `_generate_all_reversible_combinations, _sorted_pair, inp_data.beads.keys`

##### `_generate_all_linkage_angles(inp_data: MonomerTemplate)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:474`
- 종류: function, private/internal
- 역할: `generate all linkage angles` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `[tuple((int(value) for value in candidate)) for candidate in _generate_all_reversible_combinations(bead_ids, 3, existing)]`
- 주요 호출: `_canon_angle, _generate_all_reversible_combinations, inp_data.beads.keys`

##### `_generate_all_linkage_dihedrals(inp_data: MonomerTemplate)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:483`
- 종류: function, private/internal
- 역할: `generate all linkage dihedrals` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `[tuple((int(value) for value in candidate)) for candidate in _generate_all_reversible_combinations(bead_ids, 4, existing)]`
- 주요 호출: `_canon_reversible, _generate_all_reversible_combinations, inp_data.beads.keys`

##### `_generate_all_linkage_impropers(inp_data: MonomerTemplate)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:492`
- 종류: function, private/internal
- 역할: `generate all linkage impropers` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `[tuple((int(value) for value in candidate)) for candidate in _generate_all_reversible_combinations(bead_ids, 4, existing)]`
- 주요 호출: `_canon_reversible, _generate_all_reversible_combinations, inp_data.beads.keys`

##### `_connection_proxy_count(indices: Sequence[int], backbone_beads: set[int])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:501`
- 종류: function, private/internal
- 역할: `connection proxy count` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `len({int(index) for index in indices if int(index) in backbone_beads})`

##### `_filter_connection_proxy_terms(terms: Sequence[Tuple[int, ...]], backbone_beads: set[int], minimum_distinct: int=2)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:505`
- 종류: function, private/internal
- 역할: `filter connection proxy terms` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `[tuple((int(value) for value in term)) for term in terms if _connection_proxy_count(term, backbone_beads) >= minimum_distinct]`
- 주요 호출: `_connection_proxy_count`

##### `_distance_cache(graph: Dict[int, set[int]])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:517`
- 종류: function, private/internal
- 역할: `distance cache` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `lookup ; cache[key]`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `cache[key]`
- 주요 호출: `_sorted_pair, shortest_path_len`

##### `_distance_cache.lookup(a: int, b: int)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:520`
- 종류: nested helper
- 역할: `lookup` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `cache[key]`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `cache[key]`
- 주요 호출: `_sorted_pair, shortest_path_len`

##### `_topology_reference_cost(section: str, indices: Sequence[int], distance_lookup: Callable[[int, int], Optional[int]])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:529`
- 종류: function, private/internal
- 역할: `topology reference cost` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `total ; None`
- 예외/검증: `ValueError(f'Unsupported topology section: {section}')`
- 주요 호출: `ValueError, distance_lookup`

##### `_changed_index_count(term: Sequence[int], reference: Sequence[int])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:556`
- 종류: function, private/internal
- 역할: `changed index count` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `max(0, changed - 1)`

##### `_topology_term_cost(section: str, term: Sequence[int], distance_lookup: Callable[[int, int], Optional[int]], *, allow_swaps: bool)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:561`
- 종류: function, private/internal
- 역할: `topology term cost` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `best ; direct_cost`
- 주요 호출: `_changed_index_count, _topology_reference_cost, permutations`

##### `_filter_topology_terms(section: str, terms: Sequence[Tuple[int, ...]], graph: Dict[int, set[int]], budget: int, *, allow_swaps: bool)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:583`
- 종류: function, private/internal
- 역할: `filter topology terms` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `filtered`
- 주요 호출: `_distance_cache, _topology_term_cost, filtered.append`

##### `_generate_augmented_terms(base: MonomerTemplate, term_cfg: TermGenerationConfig, backbone_beads: Sequence[int])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:600`
- 종류: function, private/internal
- 역할: `generate augmented terms` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `(new_bonds, new_angles, new_dihedrals, new_impropers) ; ([], [], [], [])`
- 주요 호출: `_build_graph, _filter_connection_proxy_terms, _filter_topology_terms, _generate_all_linkage_angles, _generate_all_linkage_bonds, _generate_all_linkage_dihedrals, _generate_all_linkage_impropers`

##### `format_inp(template: MonomerTemplate)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:660`
- 종류: function
- 역할: `format inp` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 1개. 예: `'\n'.join(lines) + '\n'`
- 주요 호출: `join, lines.append, lines.extend, ref.format, template.beads.items`

##### `validate_generated_input(inp_data: MonomerTemplate, xyz_path: Path, terminal_cap_indices: Optional[Sequence[int]]=None)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:686`
- 종류: function
- 역할: `validate generated input` 검증 helper입니다. 입력 일관성, tool availability, template/topology 조건을 확인합니다.
- 반환: 명시적 return 1개. 예: `report`
- 주요 호출: `Fraction, ValidationReport, _weighted_atom_owners, owners.items, parse_xyz, report.problems.append`

##### `build_polymer_input(sequence: Sequence[str] | str, polymer_xyz_path: Path, templates: Dict[str, MonomerTemplate], metadata: Dict[str, ConnectionMetadata], term_cfg: TermGenerationConfig)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:724`
- 종류: function
- 역할: monomer별 Bartender init template를 sequence 순서대로 offset해 polymer-level Bartender input을 만듭니다. 내부 connector atom 제거, inter-monomer bond 추가, term_generation mode에 따른 augmented term 추가까지 수행합니다.
- 반환: 명시적 return 1개. 예: `PolymerInputBundle(base=base, augmented=augmented, base_text=format_inp(base), augmented_text=format_inp(augmented), base_report=report, augmented_report=augmented_report, conne...`
- 예외/검증: `KeyError(f"Unknown monomer token '{token}'")`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `local_to_global[local_atom_index], beads[global_bead]`
- 주요 호출: `KeyError, MonomerTemplate, OrderedDict, PolymerInputBundle, ValidationReport, WeightedAtomRef, _generate_augmented_terms, _sorted_pair, angles.extend, backbone_beads.extend, bonds.append, bonds.extend, connection_beads.extend, connection_bonds.append, constraints.extend, dihedrals.extend, format_inp, impropers.extend, normalize_sequence, parse_xyz, polymer_xyz_path.with_suffix, removed.add, report.problems.append, report.problems.extend, ... (+5)`

##### `default_bead_spec(token: str, bead_count: int)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:864`
- 종류: function
- 역할: `default bead spec` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `{'labels': labels, 'types': list(labels)}`

##### `split_main_and_comment(raw: str)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:869`
- 종류: function
- 역할: `split main and comment` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `(stripped.strip(), '') ; (main.strip(), comment.strip())`
- 주요 호출: `comment.strip, main.strip, raw.lstrip, stripped.lstrip, stripped.split, stripped.startswith, stripped.strip`

##### `parse_param_line(raw: str, section: str, n_idx: int)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:879`
- 종류: function
- 역할: `parse param line` 계열 parser입니다. 문자열/파일 내용을 내부 자료구조로 바꾸며, 입력 형식이 맞지 않으면 예외 또는 None 경로를 사용합니다.
- 반환: 명시적 return 5개. 예: `ParamLine(section=section, indices=indices, tokens=tuple(parts[n_idx:]), commented=stripped.startswith(';'), inline_comment=comment, rmsd=rmsd, raw=raw.rstrip('\n')) ; None`
- 주요 호출: `ParamLine, RMSD_RE.search, main.isdigit, main.split, match.group, raw.rstrip, raw.strip, split_main_and_comment, stripped.startswith`

##### `parse_gmx_out_itp(path: Path)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:915`
- 종류: function
- 역할: `parse gmx out itp` 계열 parser입니다. 문자열/파일 내용을 내부 자료구조로 바꾸며, 입력 형식이 맞지 않으면 예외 또는 None 경로를 사용합니다.
- 반환: 명시적 return 1개. 예: `parsed`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `header_map.get, parse_param_line, parsed.append, path.read_text, path.read_text.splitlines, raw.strip, stripped.endswith, stripped.startswith, stripped.strip, stripped.strip.strip, stripped.strip.strip.lower`

##### `summarize_itp(path: Path)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:947`
- 종류: function
- 역할: `summarize itp` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `{'path': str(path), 'counts': {section: len(lines) for section, lines in parsed.items()}, 'bonds': [_payload(line) for line in parsed['bonds']], 'constraints': [_payload(line) f... ; payload`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `payload['rmsd']`
- 주요 호출: `_payload, parse_gmx_out_itp, parsed.items`

##### `summarize_itp._payload(line: ParamLine)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:950`
- 종류: nested helper, private/internal
- 역할: `payload` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `payload`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `payload['rmsd']`

##### `find_case_json(start: Path)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:972`
- 종류: function
- 역할: `find case json` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `None ; candidate`
- 주요 호출: `candidate.exists, start.resolve`

##### `resolve_case_artifact(case_dir: Path, case: Dict[str, object], key: str)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:984`
- 종류: function
- 역할: `resolve case artifact` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 1개. 예: `candidate`
- 예외/검증: `FileNotFoundError(f"Could not resolve case artifact '{key}' from {case_dir}. Tried: {', '.join((str(path) for path in candidates))}")`
- 주요 호출: `FileNotFoundError, Path, candidate.exists, candidates.append, case.get, join, raw.is_absolute`

##### `normalize_label_spec(token: str, bead_count: int, raw_spec: Any)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1004`
- 종류: function
- 역할: `normalize label spec` 계열 normalizer입니다. 사용자 설정값과 legacy alias를 검증 가능한 내부 표현으로 변환합니다.
- 반환: 명시적 return 3개. 예: `default_bead_spec(token, bead_count) ; {'labels': labels, 'types': [label.split('(', 1)[0] if '(' in label else label for label in labels]} ; {'labels': labels, 'types': types}`
- 예외/검증: `TypeError(f'Unsupported label specification for token {token}: {type(raw_spec)!r}') ; ValueError(f"Label override for token {token} must contain 'labels'.") ; ValueError(f'Label override for token {token} has {len(labels)} entries, expected {bead_count}.') ; ValueError(f'Label override for token {token} has {len(labels)} labels, expected {bead_count}.') ; ValueError(f'Type override for token {token} has {len(types)} entries, expected {bead_count}.')`
- 주요 호출: `TypeError, ValueError, default_bead_spec, label.split, raw_spec.get`

##### `load_label_map(path: Optional[Path])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1030`
- 종류: function
- 역할: `load label map` 계열 loader입니다. 설정/파일/템플릿을 읽어 후속 builder가 사용할 dict/dataclass 구조로 정규화합니다.
- 반환: 명시적 return 2개. 예: `overrides ; {}`
- 예외/검증: `ValueError('Label map JSON must be an object.') ; ValueError(f'Unsupported label map entry for token {token}: {type(spec)!r}')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `overrides[str(token)], entry['types']`
- 주요 호출: `ValueError, data.items, json.loads, path.read_text, spec.get`

##### `build_bead_maps(case: Dict[str, object], overrides: Dict[str, Dict[str, List[str]]])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1050`
- 종류: function
- 역할: `build bead maps` 계열 builder/helper입니다. 여러 입력 설정을 조합해 중간 계획, job, topology 또는 출력용 구조를 만듭니다.
- 반환: 명시적 return 1개. 예: `(label_map, type_map, backbone_beads)`
- 예외/검증: `KeyError(f"Token {token} is not present in case['monomers'].") ; ValueError("case.json must contain 'monomers' and 'sequence_tokens'.")`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `label_map[global_index], type_map[global_index]`
- 주요 호출: `KeyError, ValueError, backbone_beads.add, case.get, case_specs.get, monomers.get, normalize_label_spec, overrides.get`

##### `shortest_path_len(graph: Dict[int, set[int]], start: int, goal: int)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1091`
- 종류: function
- 역할: `shortest path len` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 3개. 예: `None ; 0 ; dist + 1`
- 주요 호출: `deque, graph.get, queue.append, queue.popleft, seen.add`

##### `choose_best_rmsd_uncomment(lines: List[ParamLine])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1107`
- 종류: function
- 역할: `choose best rmsd uncomment` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `updated`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `updated[position]`
- 주요 호출: `ParamLine, defaultdict, grouped.append, grouped.values, math.isinf`

##### `typed_records_for_result(itp_path: Path, case_path: Path, label_overrides: Dict[str, Dict[str, List[str]]])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1137`
- 종류: function
- 역할: `typed records for result` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 3개. 예: `records ; 'WITH_BACKBONE' if any((index in backbone_beads for index in indices)) else 'WITHOUT_BACKBONE' ; (display, types)`
- 예외/검증: `KeyError(f'{itp_path}: bead index {exc.args[0]} is not present in the case bead map.')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `KeyError, TypedRecord, _build_graph, _sorted_pair, build_bead_maps, case.get, case_path.read_text, category, choose_best_rmsd_uncomment, json.loads, map_labels, parse_gmx_out_itp, records.append, shortest_path_len`

##### `typed_records_for_result.category(indices: Tuple[int, ...])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1153`
- 종류: nested helper
- 역할: `category` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `'WITH_BACKBONE' if any((index in backbone_beads for index in indices)) else 'WITHOUT_BACKBONE'`

##### `typed_records_for_result.map_labels(indices: Tuple[int, ...])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1156`
- 종류: nested helper
- 역할: `map labels` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `(display, types)`
- 예외/검증: `KeyError(f'{itp_path}: bead index {exc.args[0]} is not present in the case bead map.')`
- 주요 호출: `KeyError`

##### `merge_records(records: List[TypedRecord])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1252`
- 종류: function
- 역할: `merge records` 결과 집계 helper입니다. 여러 case/result 파일을 모아 summary 또는 merged force-field 자료구조로 변환합니다.
- 반환: 명시적 return 3개. 예: `merged ; (0 if not sample.commented else 1, 0, 0.0, sample.source_tag) ; (0 if not sample.commented else 1, 0 if item['rmsd_values'] else 1, rmsd, sample.source_tag)`
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `merged[key]`
- 주요 호출: `MergedVariant, defaultdict, grouped.append, grouped.items, items.append, record.inline_comment.strip, variants.append, variants_by_signature.append, variants_by_signature.values`

##### `_format_type_names(type_names: Tuple[str, ...], widths: Tuple[int, ...])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1309`
- 종류: function, private/internal
- 역할: `format type names` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 1개. 예: `' '.join((f'{value:<{width}}' for value, width in zip(type_names, widths)))`
- 주요 호출: `join`

##### `line_from_variant(variant: MergedVariant)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1313`
- 종류: function
- 역할: `line from variant` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `main + (' ; ' + ' ; '.join(comment_parts) if comment_parts else '')`
- 주요 호출: `_format_type_names, _format_type_names.rstrip, comment_parts.append, join, rstrip`

##### `write_merged_forcefield(path: Path, merged: Dict[Tuple[str, str, str, Tuple[str, ...]], List[MergedVariant]], root: Path, label_map_path: Optional[Path])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1332`
- 종류: function
- 역할: `write merged forcefield` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `join, join.rstrip, line_from_variant, lines.append, write_text`

##### `merged_summary_payload(root: Path, merged: Dict[Tuple[str, str, str, Tuple[str, ...]], List[MergedVariant]], skipped: List[Dict[str, str]])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1376`
- 종류: function
- 역할: `merged summary payload` 결과 집계 helper입니다. 여러 case/result 파일을 모아 summary 또는 merged force-field 자료구조로 변환합니다.
- 반환: 명시적 return 1개. 예: `{'root': str(root), 'group_count': len(groups), 'groups': groups, 'skipped': skipped}`
- 주요 호출: `groups.append, merged.items`

##### `_srun_reentry_lines(exec_cfg: Dict[str, Any], cpu_fallback_var: str)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1413`
- 종류: function, private/internal
- 역할: `srun reentry lines` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `['if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${SLURM_STEP_ID:-}" ]; then', ' if ! command -v srun >/dev/null 2>&1; then', ' echo "[ERROR] execution.use_srun=true but srun was not fo... ; []`
- 주요 호출: `exec_cfg.get, parse_bool`

##### `_bartender_mode_args(flow: Dict[str, str], bartender_cfg: Dict[str, Any], bartender_charge: int, skip: int, trajectory: Optional[Path], outdir: Path)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1427`
- 종류: function, private/internal
- 역할: md 모드에 따른 Bartender CLI 인자 목록 반환 (quoting 없이 raw 값).
- 반환: 명시적 return 1개. 예: `args`
- 예외/검증: `ValueError('Trajectory reuse mode requires a trajectory path.') ; ValueError(f"Unsupported md mode: {flow['md']}")`
- 주요 호출: `ValueError, bartender_cfg.get, os.path.relpath`

##### `prepare_relaxation_job(case_dir: Path, case: Dict[str, Any], flow: Dict[str, str], pipeline_cfg: Dict[str, Any], base_dir: Path, exec_cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1461`
- 종류: function
- 역할: case별 xTB/ORCA geometry optimization 및 optional xTB MD를 실행할 `run_relax.sh`를 생성하고 case metadata에 relaxation artifact hint를 기록합니다.
- 반환: 명시적 return 2개. 예: `workdir ; None`
- 예외/검증: `TypeError('case.electronic_state must be a mapping') ; ValueError(f"Unsupported relaxation mode: {flow['relaxation']}")`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `case['relaxation'], effective_orca_cfg['nprocs']`
- 주요 호출: `TypeError, ValueError, _get_slurm_cpu_count, _srun_reentry_lines, case.get, exec_cfg.get, join, lines.append, lines.extend, local_xyz.write_text, md_template_path.read_text, orca_template_path.read_text, parse_bool, polymer_xyz.read_text, render_orca_input, render_xtb_md_input, resolve_executable_command, resolve_optional_path, resolve_orca_settings, resolve_xtb_settings, script_path.chmod, shell_assign, shlex.quote, shutil.copy, ... (+3)`

##### `prepare_bartender_job(case_dir: Path, case: Dict[str, Any], flow: Dict[str, str], pipeline_cfg: Dict[str, Any], base_dir: Path, exec_cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1618`
- 종류: function
- 역할: Bartender 실행 디렉터리를 만들고 input/geometry/trajectory를 배치한 뒤 `run_bartender.sh`와 `bartender_job.json` manifest를 생성합니다.
- 반환: 명시적 return 3개. 예: `outdir ; None`
- 예외/검증: `FileNotFoundError(f'Bartender inp does not exist: {inp}') ; TypeError('bartender_pipeline.bartender must be a mapping') ; ValueError('Relaxation metadata is incomplete; cannot determine Bartender geometry input.') ; ValueError('bartender_pipeline.md=existing requires bartender_pipeline.md_traj') ; ValueError('xTB reuse mode requires relaxation.trajectory_hint metadata.')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `case['bartender']['job_dir'], case['bartender']['run_script'], case['bartender']['mode'], case['bartender']['geometry_source'], case['bartender']['geometry_path'], case['bartender']['trajectory_source'], case['bartender']['trajectory_path'], case['bartender']`
- 주요 호출: `FileNotFoundError, TypeError, ValueError, _bartender_mode_args, _get_slurm_cpu_count, _srun_reentry_lines, bartender_cfg.get, case.get, case.setdefault, exec_cfg.get, geometry.exists, inp.exists, inp.read_text, join, json.dumps, local_inp.write_text, os.path.relpath, outdir.mkdir, parse_bool, pipeline_cfg.get, relax.get, resolve_executable_command, resolve_optional_path, script_lines.append, ... (+7)`

##### `collect_results(root: Path, output: Path)`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1785`
- 종류: function
- 역할: `collect results` 결과 집계 helper입니다. 여러 case/result 파일을 모아 summary 또는 merged force-field 자료구조로 변환합니다.
- 반환: 명시적 return 1개. 예: `payload`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `summary['case']`
- 주요 호출: `case.get, case_json.read_text, find_case_json, json.dumps, json.loads, records.append, root.rglob, summarize_itp, write_text`

##### `merge_results(root: Path, output_itp: Path, output_json: Path, label_map_path: Optional[Path])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1803`
- 종류: function
- 역할: `merge results` 결과 집계 helper입니다. 여러 case/result 파일을 모아 summary 또는 merged force-field 자료구조로 변환합니다.
- 반환: 명시적 return 1개. 예: `payload`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `find_case_json, json.dumps, load_label_map, merge_records, merged_summary_payload, records.extend, root.rglob, skipped.append, typed_records_for_result, write_merged_forcefield, write_text`

##### `run_postprocess_only(cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1824`
- 종류: function
- 역할: 이미 존재하는 postprocess root에서 screening postprocess만 수행합니다. 새 polymer/job 생성은 하지 않습니다. `collect`/`merge` config option은 제거되었고, `bartender_pipeline.postprocess.screening.enabled=true`가 필요합니다.
- 반환: 명시적 return 1개. 예: `summary`
- 예외/검증: screening이 꺼져 있으면 `ValueError`를 냅니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신
- 주요 대입: `summary['screening'], summary['summary_json']`
- 주요 호출: `Path, json.dumps, run_screening_postprocess, write_text`

##### `run_pipeline(cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_martini/pipeline.py:1852`
- 종류: function
- 역할: 03 workflow의 최상위 실행 함수입니다. config를 해석해 monomer library와 sequence job을 만들고, 각 case별 polymer XYZ/inp/case.json/job script를 생성한 뒤 옵션에 따라 relaxation/Bartender 실행과 postprocess를 수행합니다.
- 반환: 명시적 return 1개. 예: `summary`
- 예외/검증: `FileNotFoundError(f'Bartender geometry source does not exist yet: {geometry_path}') ; FileNotFoundError(f'Bartender owntraj source does not exist yet: {trajectory_path}') ; FileNotFoundError(f'param_opt builder did not produce {built_xyz}') ; KeyError(f'Missing init template for monomer token: {token}') ; KeyError(f'Unknown monomer token: {token}') ; TypeError('bartender_pipeline.init_templates must be a mapping when provided')`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 외부 command/subprocess 실행 가능, 객체/class/global attribute 갱신
- 주요 대입: `summary['screening'], case['execution']['relaxation'], case['execution']['bartender'], template_cache[token], validation_cache[token], metadata_cache[token]`
- 주요 호출: `FileNotFoundError, KeyError, Path, Path.exists, Path.resolve, TypeError, ValueError, bartender_meta.get, build_polymer_input, build_polymer_structure, build_sequence_jobs, builder_tmp.exists, builder_tmp.mkdir, built_xyz.exists, bundle.augmented_report.render, bundle.base_report.render, case.get, case_dir.mkdir, cases.append, cfg.get, default_bead_spec, ensure_case_logs_dir, entry.get, ...`

### `hygel_martini/param_opt/qm_to_martini/postprocess.py`

Bartender 결과 ITP들을 screening rule에 따라 파싱하고, 전체 후보와 screening 결과를 별도로 저장하는 후처리 모듈입니다. bond/constraint 동시 존재, multi-constant potential, RMSD/force metric threshold, root별 output mirror, CSV/PDF plot 생성을 담당합니다. 현재 git에는 untracked 상태로 존재합니다.
- 주요 import: `json, math, re, from collections import defaultdict, from pathlib import Path, from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple`
- class 수: 1, 함수/메서드 수: 24
- 핵심 설정:
  - `screening.potentials`: section별 funct preference입니다. 숫자이면 해당 funct만 허용하고, `"bartender"`이면 funct 번호 filtering을 끄고 Bartender가 active로 남긴 line의 funct를 그대로 따릅니다.
  - `screening.bond_constraint_mode`: `ignore_constraints`, `bartender`, `ignore_bonds` 중 하나입니다. `bartender`는 Bartender가 comment로 걸러낸 active bond/constraint 선택을 그대로 따릅니다.
  - `screening.show_all_info`: `true`이면 commented alternative까지 `all_terms`와 plots에 보존합니다. Screening 선택 자체는 항상 Bartender active line만 사용합니다.
  - `screening.multi_constant_metric`: 여러 force-like constant를 가진 potential의 대표값 계산 방식입니다. 기본값 `max_abs`, 선택지는 `l2`, `mean_abs`, `first`, `none`입니다.
  - `screening.thresholds.force_metric_min_mode`: `absolute`이면 threshold 값 자체를 씁니다. `relative_to_section_max`이면 section/funct 그룹 내 최대 force metric에 대한 비율입니다.
  - `paths.postprocess_output_root`: screening 산출물을 저장할 루트입니다. 각 input root의 상대 구조를 유지해 `S/topology_n0` 같은 하위 디렉터리를 만듭니다.
- 주요 산출물:
  - `all_terms.json`, `all_terms.itp`: postprocess inspection용 후보입니다. `show_all_info=true`이면 commented alternative까지 포함하고, `false`이면 Bartender active 후보 중심으로 저장합니다.
  - `screened_summary.json`, `screened_forcefield.itp`, `screening_report.json`: 최종 선택 결과와 요약 report입니다.
  - `plots/*.csv`, `plots/*.pdf`: section/funct별 force metric 및 RMSD plot입니다. PDF는 x축 index/atom indices, potential 이름과 LaTeX/mathtext 식, force/RMSD cutoff 상태를 함께 표시합니다. 한 그림에 10개를 넘는 point가 있으면 `*_part_01_of_N.pdf` 식으로 균등 분할합니다.

#### Classes

- `ScreeningProcessor` (hygel_martini/param_opt/qm_to_martini/postprocess.py:141)

#### Functions and methods

##### `_as_list(value: Any) -> List[Any]`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:24`
- 종류: function, private/internal
- 역할: scalar/tuple/list/None 설정값을 list로 정규화합니다. `out_roots`, potential preference 등 사용자 config 입력이 단일값과 배열을 모두 허용할 때 사용합니다.

##### `_parse_float(value: str) -> Optional[float]`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:32`
- 종류: function, private/internal
- 역할: 문자열 token을 float로 변환하고 실패하면 `None`을 반환합니다. ITP parameter token 중 숫자만 골라내기 위한 작은 helper입니다.

##### `_canon_indices(indices: Sequence[int]) -> Tuple[int, ...]`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:39`
- 종류: function, private/internal
- 역할: bond/angle/dihedral/improper index tuple을 reverse equivalent까지 고려해 canonical key로 만듭니다. 같은 원자열이 반대 방향으로 나온 중복을 겹치는 항으로 판단할 때 사용합니다.

##### `_find_case_json(start: Path) -> Optional[Path]`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:43`
- 종류: function, private/internal
- 역할: 입력 ITP 주변 directory에서 `case.json`을 위쪽으로 탐색합니다. 산출 JSON에 sequence/tag 같은 provenance를 넣기 위한 helper입니다.

##### `_relative_to_or_name(path: Path, base: Optional[Path]) -> Path`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:55`
- 종류: function, private/internal
- 역할: `path.relative_to(base)`가 가능하면 상대경로를, 아니면 파일명만 반환합니다. report/source 표시를 짧고 안정적으로 만들 때 사용합니다.

##### `_write_pdf_plot(...)`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py`
- 종류: function, private/internal
- 역할: matplotlib PDF backend으로 section/funct별 screening plot을 생성합니다. `force_metric`와 `rmsd`를 두 panel에 그리고, x축에는 `#index`와 atom indices를 표시합니다. potential funct의 이름/LaTeX 식/parameter 설명, cutoff dashed line, reject 영역 음영, selected/candidate legend를 함께 넣습니다. cutoff 기준으로 전부 pass 또는 전부 reject이면 dashed line은 생략하고 상태 text/배경으로만 표시합니다.
- 부작용 단서: PDF 파일 출력

##### `ScreeningProcessor.__init__(self, cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:144`
- 종류: method, private/internal
- 역할: config에서 postprocess/screening 설정을 읽어 내부 옵션을 구성합니다. potential preference, bond/constraint mode, comment policy, threshold mode, RMSD 상한, plot 여부, output root/mirror root 등을 모두 여기서 정규화합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 객체/class/global attribute 갱신
- 주요 대입: `self.cfg, self.post_cfg, self.screen_cfg, self.pref_potentials, self.rmsd_max, self.bond_constraint_mode, self.show_all_info, self.multi_constant_metric, self.write_plots`

##### `ScreeningProcessor._normalize_bond_constraint_mode(raw: Any) -> str`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:182`
- 종류: static method, private/internal
- 역할: `constraints_only`, `bonds_only`, `both`, `preserve_bartender` 같은 alias를 세 가지 canonical mode로 정규화합니다. 잘못된 값은 `ValueError`로 알려줍니다.

##### `ScreeningProcessor._force_values_and_metric(self, section, funct, numeric_params)`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:202`
- 종류: method, private/internal
- 역할: ITP parameter에서 force-like constant 후보들을 추출하고 대표 `force_metric`을 계산합니다. bond/constraint/angle은 보통 두 번째 numeric parameter, proper dihedral funct 1/2는 `kd`, RB/combined dihedral처럼 여러 constant가 있는 potential은 `multi_constant_metric`으로 대표값을 계산합니다.
- 반환: `(force_values, force_metric, metric_method)` tuple

##### `ScreeningProcessor._parse_itp_line(self, line: str, section: str, n_idx: int) -> Optional[Dict[str, Any]]`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:241`
- 종류: method, private/internal
- 역할: `[ bonds ]`, `[ constraints ]`, `[ angles ]`, `[ dihedrals ]`, `[ impropers ]` line 하나를 dict로 파싱합니다. comment 여부, raw line, indices, funct, params, numeric_params, force_values, force_metric, rmsd를 보존합니다.

##### `ScreeningProcessor._parse_itp(self, itp_path: Path, out_root: Path) -> Dict[str, List[Dict[str, Any]]]`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:292`
- 종류: method, private/internal
- 역할: ITP 파일 전체를 section별 term list로 파싱합니다. section header를 따라가며 `_parse_itp_line`을 호출하고, source path/case_json/source_tag metadata를 각 term에 붙입니다.
- 부작용 단서: ITP 파일 읽기

##### `ScreeningProcessor._get_overlap_key(self, term: Dict[str, Any]) -> Tuple[Any, ...]`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:324`
- 종류: method, private/internal
- 역할: screening 중 이미 선택된 term과 겹치는지를 판단할 key를 만듭니다. section과 canonical indices를 사용하므로, 같은 atom tuple에 대해 여러 funct가 살아 있으면 RMSD/force sorting 뒤 하나만 남습니다.

##### `ScreeningProcessor._term_is_bartender_active(term: Dict[str, Any]) -> bool`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:330`
- 종류: static method, private/internal
- 역할: ITP line이 commented line인지 확인해 Bartender active line만 screening 후보로 통과시킵니다.

##### `ScreeningProcessor._term_is_allowed_by_bond_constraint_mode(self, term: Dict[str, Any]) -> bool`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:344`
- 종류: method, private/internal
- 역할: bond/constraint 전용 mode를 적용합니다. `ignore_constraints`는 constraint를 버리고, `ignore_bonds`는 bond를 버리며, `bartender`는 둘 다 허용합니다. 실제 screening은 별도 Bartender active check를 통과한 line만 사용합니다.

##### `ScreeningProcessor._term_matches_preferred_potential(term: Dict[str, Any], preferred: Any) -> bool`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py`
- 종류: static method, private/internal
- 역할: `screening.potentials.<section>` 값을 적용합니다. 숫자이면 `term["funct"]`가 같은 term만 허용하고, `"bartender"`이면 funct 번호 filtering을 하지 않습니다. 이 경우 Bartender active line의 funct가 그대로 쓰입니다.

##### `ScreeningProcessor._threshold_for(self, section: str, funct: int, terms: Sequence[Dict[str, Any]]) -> float`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:352`
- 종류: method, private/internal
- 역할: section/funct별 force metric threshold를 계산합니다. `absolute` mode는 설정값 그대로, `relative_to_section_max` mode는 같은 그룹 내 최대 `force_metric`에 대한 비율로 바꿉니다.

##### `ScreeningProcessor._screen_terms(self, all_terms: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:370`
- 종류: method, private/internal
- 역할: 전체 후보에서 Bartender active 여부, bond/constraint mode, potential funct preference, force metric threshold, RMSD threshold, overlap rule을 적용해 최종 term을 고릅니다. `potentials.<section>: bartender`이면 funct 번호 filtering은 건너뛰고 Bartender active line을 그대로 사용합니다.

##### `ScreeningProcessor._info_terms(self, all_terms: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py`
- 종류: method, private/internal
- 역할: `show_all_info`에 따라 `all_terms.json`, `all_terms.itp`, plot에 남길 term 범위를 정합니다. `show_all_info=true`이면 commented alternative까지 포함하고, `false`이면 Bartender active 및 potential preference를 통과한 term만 inspection output에 남깁니다.

##### `ScreeningProcessor._output_dir_for_root(self, out_root: Path) -> Path`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:414`
- 종류: method, private/internal
- 역할: 입력 root가 `compare_existing_terms/S/topology_n0`이면 output root 아래 `S/topology_n0` 구조로 mirror될 수 있도록 산출 directory를 계산합니다.

##### `ScreeningProcessor._json_terms(terms: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:428`
- 종류: static method, private/internal
- 역할: `Path` 객체를 문자열로 바꿔 JSON 직렬화 가능한 term list를 만듭니다.

##### `ScreeningProcessor._write_all_terms_itp(self, path: Path, all_terms: Dict[str, List[Dict[str, Any]]]) -> None`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:434`
- 종류: method, private/internal
- 역할: screening 전 전체 후보를 ITP 형식으로 저장합니다. 원본 comment 상태와 raw line을 최대한 유지합니다.
- 부작용 단서: ITP 파일 출력

##### `ScreeningProcessor._write_itp(self, path: Path, results: Dict[str, List[Dict[str, Any]]]) -> None`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:456`
- 종류: method, private/internal
- 역할: screening된 결과를 GROMACS ITP 형식으로 저장합니다. 각 line 뒤에 `source`, `rmsd`, `force_metric`, `force_values` metadata comment를 붙여 추적 가능하게 합니다.
- 부작용 단서: ITP 파일 출력

##### `ScreeningProcessor._write_plots(self, out_dir: Path, all_terms, screened)`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py`
- 종류: method, private/internal
- 역할: section/funct별 CSV와 PDF plot을 생성합니다. CSV 각 row에는 `plot_index`, source, indices, funct, selected, commented, rmsd, force_metric, force_values, params가 들어갑니다. PDF는 최대 10개 point 단위로 나뉘며, 11개는 5/6, 15개는 7/8, 22개는 7/7/8처럼 가능한 균등하게 분할됩니다. 이전 실행에서 남은 같은 section/funct의 SVG/PDF는 새로 쓰기 전에 지웁니다.
- 부작용 단서: CSV/PDF 파일 출력

##### `ScreeningProcessor.process(self, out_root: Path)`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:551`
- 종류: method
- 역할: 한 input root 아래의 `gmx_out.itp`를 모두 찾아 파싱, screening, 전체/선별 결과 저장, plot 생성, report 저장까지 수행합니다.
- 반환: output directory, input file 수, all/screened counts, 산출 파일 경로, 설정 요약을 담은 dict
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기, 객체/class/global attribute 갱신

##### `_resolve_postprocess_roots(cfg: Dict[str, Any]) -> List[Path]`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:615`
- 종류: function, private/internal
- 역할: `paths.out_root`, `paths.out_roots`, `paths.out_root_glob`을 모두 해석해 postprocess 대상 root 목록을 만듭니다. 여러 label/mode root를 하나의 config에서 처리할 때 사용합니다.

##### `run_screening_postprocess(cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_martini/postprocess.py:636`
- 종류: function
- 역할: screening postprocess의 public entry point입니다. 대상 root가 하나면 단일 report dict를, 여러 개면 `root_count`와 `results`를 가진 aggregate dict를 반환합니다.
- 주요 호출: `ScreeningProcessor, _resolve_postprocess_roots, processor.process`

### `hygel_martini/param_opt/qm_to_martini/xtb_traj_to_pdb.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `from __future__ import annotations, sys, from pathlib import Path`
- class 수: 0, 함수/메서드 수: 3

#### Functions and methods

##### `parse_frames_streaming(path: Path)`
- 위치: `hygel_martini/param_opt/qm_to_martini/xtb_traj_to_pdb.py:7`
- 종류: function
- 역할: `parse frames streaming` 계열 parser입니다. 문자열/파일 내용을 내부 자료구조로 바꾸며, 입력 형식이 맞지 않으면 예외 또는 None 경로를 사용합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `atom_line.strip, atom_lines.append, f.readline, line.strip, path.open`

##### `pdb_atom_line(atom_index: int, symbol: str, x: float, y: float, z: float)`
- 위치: `hygel_martini/param_opt/qm_to_martini/xtb_traj_to_pdb.py:32`
- 종류: function
- 역할: `pdb atom line` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개. 예: `f'ATOM {atom_index:5d} {atom_name:<4} MOL A{1:4d} {x:8.3f}{y:8.3f}{z:8.3f} 1.00 0.00 {symbol[:2].upper():>2}\n'`
- 주요 호출: `symbol.upper, symbol.upper.rjust`

##### `main(argv: list[str])`
- 위치: `hygel_martini/param_opt/qm_to_martini/xtb_traj_to_pdb.py:40`
- 종류: function, CLI entry
- 역할: `main` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 2개. 예: `0 ; 1`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Path, out_f.write, output_path.open, parse_frames_streaming, pdb_atom_line, raw.split`

### `hygel_martini/param_opt/qm_to_opls/ase_utils.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `from __future__ import annotations, from pathlib import Path, from ase.io import read, write`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `xyz_to_pdb(xyz_path: str | Path, pdb_path: str | Path)`
- 위치: `hygel_martini/param_opt/qm_to_opls/ase_utils.py:8`
- 종류: function
- 역할: Converts an XYZ file to a PDB file using ASE.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `read, write`

### `hygel_martini/param_opt/qm_to_opls/cli.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `from __future__ import annotations, argparse, json, from pathlib import Path, from ..core.config import add_config_args, from .defaults import DEFAULT_CONFIG, from .generator import run_qm_to_opls`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `main()`
- 위치: `hygel_martini/param_opt/qm_to_opls/cli.py:12`
- 종류: function, CLI entry
- 역할: `main` 함수입니다. 아래 정적 분석 항목이 실제 입력/출력/부작용을 요약합니다.
- 반환: 명시적 return 1개이지만 값 없는 return 경로가 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Path, add_config_args, argparse.ArgumentParser, config_path.write_text, json.dumps, parser.parse_args, run_qm_to_opls`

### `hygel_martini/param_opt/qm_to_opls/generator.py`

Thin workflow entry helper for QM-to-OPLS preparation.
- 주요 import: `from __future__ import annotations, from pathlib import Path, from ..core.config import load_config, from .defaults import DEFAULT_CONFIG, from .orca_runner import generate_orca_inputs`
- class 수: 0, 함수/메서드 수: 1

#### Functions and methods

##### `run_qm_to_opls(config_path: str | Path)`
- 위치: `hygel_martini/param_opt/qm_to_opls/generator.py:12`
- 종류: function
- 역할: Load a qm_to_opls maker file and generate ORCA preparation inputs.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 주요 호출: `Path, generate_orca_inputs, load_config`

### `hygel_martini/param_opt/qm_to_opls/ligpargen_api.py`

모듈 docstring은 없지만 아래 함수/클래스가 workflow에서 사용됩니다.
- 주요 import: `from __future__ import annotations, from pathlib import Path`
- class 수: 0, 함수/메서드 수: 2

#### Functions and methods

##### `submit_to_ligpargen(pdb_path: str | Path, output_dir: str | Path, name: str='molecule')`
- 위치: `hygel_martini/param_opt/qm_to_opls/ligpargen_api.py:9`
- 종류: function
- 역할: Submits a PDB file to LigParGen and stores placeholder outputs.
- 반환: 명시적 return 1개. 예: `str(itp_output)`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Path, gro_output.write_text, itp_output.write_text, output_dir.mkdir`

##### `run_parameterization_flow(xyz_path: str | Path, out_root: str | Path, symbol: str)`
- 위치: `hygel_martini/param_opt/qm_to_opls/ligpargen_api.py:25`
- 종류: function
- 역할: High-level flow: XYZ -> PDB -> LigParGen -> ITP/GRO.
- 반환: 명시적 return 1개. 예: `itp_path`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Path, submit_to_ligpargen, temp_dir.mkdir, xyz_to_pdb`

### `hygel_martini/param_opt/qm_to_opls/orca_runner.py`

01 QM -> OPLS 준비 단계로 ORCA input을 생성합니다. polymer_maker와 ASE Atoms를 이용합니다.
- 주요 import: `from __future__ import annotations, from pathlib import Path, from typing import Any, Dict, List, itertools, from ase.io import read, write, from ..polymer_maker.maker import _sequence_output_stem, load_monomer_library`
- class 수: 0, 함수/메서드 수: 3

#### Functions and methods

##### `generate_orca_inputs(cfg: Dict[str, Any])`
- 위치: `hygel_martini/param_opt/qm_to_opls/orca_runner.py:10`
- 종류: function
- 역할: Generates ORCA input files for N-mers (up to 200 atoms). Automatically caps Br with H using existing maker.py logic.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `Path, Path.resolve, _build_atoms_for_dft, _sequence_output_stem, _write_orca_file, cfg.get, cfg.get.get, itertools.product, load_monomer_library, monomer_dict.keys, out_root.mkdir`

##### `_build_atoms_for_dft(sequence, monomer_dict, n_torsion)`
- 위치: `hygel_martini/param_opt/qm_to_opls/orca_runner.py:67`
- 종류: function, private/internal
- 역할: Internal helper that mirrors maker.build_polymer but returns ASE Atoms. Used for Phase 1 DFT input generation.
- 반환: 명시적 return 1개. 예: `cap_ends_with_hydrogen(chain)`
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `cap_ends_with_hydrogen, get_connection_info, monomer_dict.copy, new_monomer.rotate, new_monomer.translate, np.linalg.norm`

##### `_write_orca_file(name, atoms, dft_cfg, out_root)`
- 위치: `hygel_martini/param_opt/qm_to_opls/orca_runner.py:111`
- 종류: function, private/internal
- 역할: `write orca file` 출력 helper입니다. 내부 구조를 파일/문자열 형식으로 직렬화합니다.
- 반환: 명시적 return 없음. 일반적으로 `None` 또는 예외/side effect 중심입니다.
- 부작용 단서: 파일/디렉터리/topology 출력 또는 읽기
- 주요 호출: `dft_cfg.get, target_dir.mkdir, write, write_text`

### `setup.py`

패키지 설치 메타데이터입니다. `find_packages()`로 `hygel_martini` 패키지를 설치하고, `xtb_traj_to_pdb.py`와 launcher shell utility를 package_data로 포함합니다.
- 주요 import: `from setuptools import setup, find_packages`
- class 수: 0, 함수/메서드 수: 0

## Shell launcher function reference

### `example/03_qm_to_martini/project/run_qm_to_martini.sh`

- `usage()` (example/03_qm_to_martini/project/run_qm_to_martini.sh:19)
  - 역할: 해당 launcher의 사용법을 heredoc으로 출력합니다.
  - 주요 command 단서: `bash, cd, exit, echo`

### `example/04_1_example_system/project/run_example_system.sh`

- `usage()` (example/04_1_example_system/project/run_example_system.sh:30)
  - 역할: 해당 launcher의 사용법을 heredoc으로 출력합니다.
  - 주요 command 단서: `bash, cd, exit, echo`

### `example/04_full_builder/project/run_full_builder.sh`

- `usage()` (example/04_full_builder/project/run_full_builder.sh:30)
  - 역할: 해당 launcher의 사용법을 heredoc으로 출력합니다.
  - 주요 command 단서: `bash, cd, exit, echo`

### `example/05_hydrogel_relaxation/project/run_hydrogel_relaxation.sh`

- `usage()` (example/05_hydrogel_relaxation/project/run_hydrogel_relaxation.sh:30)
  - 역할: 해당 launcher의 사용법을 heredoc으로 출력합니다.
  - 주요 command 단서: `bash, cd, exit, echo`

### `hygel_martini/bash_settings/launcher_utils.sh`

- `source_optional_script()` (hygel_martini/bash_settings/launcher_utils.sh:6)
  - 역할: 경로가 비어 있으면 건너뛰고, 파일이 있으면 `set +u` 상태에서 source합니다. 없으면 에러 종료합니다.
  - 주요 command 단서: `source, exit, echo`
- `activate_optional_env()` (hygel_martini/bash_settings/launcher_utils.sh:22)
  - 역할: `ENV_NAME`이 설정된 경우 conda 환경 활성화를 시도합니다. 실패해도 경고 후 현재 환경을 유지합니다.
  - 주요 command 단서: `conda, echo`
- `setup_hygel_env()` (hygel_martini/bash_settings/launcher_utils.sh:40)
  - 역할: project-local `environment.sh`, 추가 profile, conda env를 순서대로 적용하고 `PYTHON_BIN`을 결정합니다.
  - 주요 command 단서: `conda, python3`
- `require_python_module()` (hygel_martini/bash_settings/launcher_utils.sh:65)
  - 역할: 현재 `PYTHON_BIN`으로 특정 Python module import 가능 여부를 확인하고, 실패 시 설치 안내와 함께 종료합니다.
  - 주요 command 단서: `conda, python, cd, exit, echo`

## Data/resource map

- `martini_v300/`: Martini 3.0 force-field ITP/FF/MAP resource. 함수는 없지만 `base_itp_file`, `gromacs_include_path`, topology include에 핵심입니다.
- `example/*/project/config/*.yaml`: maker/config 예제. 코드 path placeholder `${CONFIG_DIR}`, `${REPO_ROOT}`의 실제 사용 예가 들어 있습니다.
- `example/*/project/structure/*.gro|*.itp`: builder template 입력 예제입니다. monomer/linker loader가 읽는 형식의 실제 샘플입니다.
- `hygel_martini/hydrogel_builder/add_series/water.gro|water.itp`: water insertion용 기본 template resource입니다.

## LLM이 다시 볼 때 추천 순서

1. 실행 목적이 03이면 `hygel_martini/param_opt/qm_to_martini/config.py` -> `pipeline.py` -> 예제 `example/03_qm_to_martini/project/config_common/common.yaml` 순서로 봅니다.
2. 실행 목적이 hydrogel build이면 `hydrogel_builder/config_params/config.py` -> `read_json.py` -> `build_hydrogel.py` -> `main_components/{Universe,Attributes,Hydrogel}.py` 순서가 빠릅니다.
3. template parsing 문제면 `core_utils/io/martini_parser.py`, `templates/monomer_loader.py`, `templates/linker_loader.py`, `core_utils/templates/rich_itp_validator.py`를 같이 봅니다.
4. 좌표/layout 문제면 `core_utils/layout/proto_builder.py`, `proto_layout.py`, `layout_executor.py`, `proto_populator.py`, 필요 시 `isotropic_builder.py`를 봅니다.
5. GROMACS/Packmol 실행 문제면 `core_utils/runtime/{geo_opt,packer,topology_updater}.py`, `add_series/add_small_ion.py`, `relax/soft_em.py`를 우선 확인합니다.
