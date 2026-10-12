# hygel_martini

[English README](README.md)

`hygel_martini`는 coarse-grained Martini hydrogel을 만들고 분석하는 연구용 Python 패키지입니다. Graph-first network construction, QM/OPLS 기반 파라미터 준비, post-build relaxation, topology audit, 물성 추출을 다룹니다.

**Authors:** Daehong Kim ([ORCID 0009-0007-1647-9270](https://orcid.org/0009-0007-1647-9270)), Nayeon Jung, Bomin Kim, Won Bo Lee

**Affiliations:** Department of Chemical and Biological Engineering, Institute of Chemical Processes, Seoul National University, 1 Gwanak-ro, Gwanak-gu, Seoul 08826, Republic of Korea (all authors); School of Transdisciplinary Innovations, Seoul National University, 1 Gwanak-ro, Gwanak-gu, Seoul 08826, Republic of Korea (W. B. Lee)

패키지는 네 부분으로 나뉩니다.

- `param_opt`: QM, OPLS, xTB 입력과 Martini 파라미터 준비
- `hydrogel_builder`: hydrogel network 생성과 post-build relaxation
- `property_extract`: state, structure, transport, finite-rate mechanics 분석
- `tools`: trajectory 변환과 bonded-topology audit

이 저장소는 workflow와 Python 코드를 제공합니다. GROMACS, Packmol, xTB, ORCA, Bartender, Martini force-field 파일은 라이선스와 설치 환경이 서로 달라서 `pip` 패키지에 넣거나 자동으로 설치하지 않습니다.

먼저 다음 두 문서를 보면 됩니다.

- [portable PEGDA 설치·재현 예제](example/07_portable_pegda/README.md): 배포 wheel을 설치해 구조 생성, 저장된 결합 검사, GROMACS 전처리까지 확인합니다.
- [계획 기반 연결의 필수 설정과 파일 검증](docs/EXPLICIT_PLAN_RELIABILITY.md): 계획 기록이 없거나 저장된 결합이 계획과 다르면 builder가 어떻게 멈추는지 설명합니다.

## 설치

Python 3.10 이상이 필요합니다. CI는 Python 3.11과 3.12에서 wheel을 빌드해 소스 트리 밖의 새 환경에 설치한 뒤 시험합니다.

```bash
git clone --recurse-submodules https://github.com/adgjl7777777/hygel_martini.git
cd /path/to/hygel_martini
python -m pip install -e .
```

이미 받은 저장소에서는 외부 PoreBlazer 소스를 다음 명령으로 등록합니다.

```bash
git submodule update --init --recursive
```

PoreBlazer는 따로 관리되는 외부 프로그램이며 `hygel-martini` wheel에 들어 있지 않습니다. PoreBlazer에는 PoreBlazer 자체의 소스와 사용 조건이 적용됩니다.

개발과 packaging 시험 도구까지 설치하려면 다음을 씁니다.

```bash
python -m pip install -e ".[dev]"
pytest
```

## 명령어

| 명령 | 역할 | 같은 module 실행 |
|---|---|---|
| `hygel-builder` | hydrogel construction | `python -m hygel_martini.hydrogel_builder` |
| `hygel-relax` | post-build relaxation | `python -m hygel_martini.hydrogel_builder.relax` |
| `hygel-property` | property extraction | `python -m hygel_martini.property_extract` |
| `hygel-qm-to-opls` | Stage 01 | `python -m hygel_martini.param_opt.qm_to_opls` |
| `hygel-opls-to-martini` | Stage 02 | `python -m hygel_martini.param_opt.opls_to_martini` |
| `hygel-qm-to-martini` | Stage 03 | `python -m hygel_martini.param_opt.qm_to_martini` |
| `hygel-parameter-protocol` | E0부터 E6까지의 bonded-parameter 결정 절차 | `python -m hygel_martini.param_opt.qm_to_martini.protocol` |
| `hygel-qm-reference-audit` | xTB ensemble을 고수준 참조와 비교하는 검사 | `python -m hygel_martini.param_opt.qm_to_martini.analysis.reference_qualification` |
| `hygel-audit-topology` | bonded graph audit | `python -m hygel_martini.tools.audit_hydrogel_topology` |

각 명령의 현재 옵션은 `COMMAND --help`로 확인합니다. Example launcher는 이 패키지가 현재 Python 환경에 설치되어 있다고 가정합니다. 공용 Bash launcher는 `hygel_martini/bash_settings/`에 있습니다.

## Workflow 예제

### 02. Existing OPLS/GROMACS -> Martini

02는 이미 있는 OPLS/GROMACS production trajectory를 Bartender fitting에 다시 씁니다. 저장소에는 trajectory가 없으므로 `example/02_opls_to_martini/project/config/opls_existing_data.yaml`의 `data/...` 경로를 자신의 데이터로 바꿉니다.

```bash
cd /path/to/hygel_martini/example/02_opls_to_martini/project
MODE=setup bash run_existing_opls.sh
MODE=md bash run_existing_opls.sh
MODE=md_notrim bash run_existing_opls.sh
```

`MODE` 하나로 trajectory를 자를지, Bartender job을 만들지, 실제로 돌릴지 정합니다. 전체 mode 표는 `example/02_opls_to_martini/project/README.md`에 있습니다.

### 03. QM -> Martini

```bash
cd /path/to/hygel_martini/example/03_qm_to_martini/project
bash run_qm_to_martini.sh config_common/common.yaml
```

Trajectory 기반 모드(`xtb`, `existing`)에서는 통계적으로 수렴한 구간을 찾아 앞쪽의 불안정한 구간을 잘라냅니다. xTB 계산이 중간에 멈추면 `xtbrestart`를 찾아 이어서 돌리고 trajectory를 합칩니다. 이미 있는 `xtb_traj.pdb`나 trimmed trajectory는 `run_compare.sh`로 Bartender에 다시 넣을 수 있고, `postprocess.sh`가 screened force field를 만듭니다. 최소 예시는 다음과 같습니다.

```bash
LABEL=S \
BASE_CONFIG=config_common/common.yaml \
OUT_ROOT=compare_existing_terms/S/topology_n0 \
MD_TRAJ=md_S/S/relax_xtb_geoopt/xtb_traj.pdb \
TERM_MODE=topology_n \
TERM_N=0 \
MODE_TAG=topology_n0 \
bash run_compare.sh

LABEL=S \
MODE_TAG=topology_n0 \
INPUT_ROOT=compare_existing_terms/S/topology_n0 \
MIRROR_ROOT=compare_existing_terms \
OUTPUT_ROOT=postprocessing_result \
bash postprocess.sh
```

C/D/S 반복 실행 방법은 `example/03_qm_to_martini/project/run_cds_iteration.sh`를 봅니다.

#### 후보에서 tested-domain parameter까지

`hygel-parameter-protocol`은 후보 생성과 최종 파라미터 확정을 나눕니다. mapping, bead/nonbonded parent, 후보 함수, 데이터 역할, 독립 grouping, 목적함수와 threshold를 먼저 checksum으로 동결한 뒤 다음 누적 증거 단계를 차례로 판정합니다.

```text
E0 provenance -> E1 analytic eligibility -> E2 grouped selection
-> E3 numerical realization -> E4 target + upstream non-regression
-> E5 unopened one-shot confirmation -> E6 transfer qualification
```

한 criterion이라도 통과하지 못하면 그 iteration의 뒤 단계는 열리지 않습니다. E5까지 통과한 값만 sealed tested-domain release가 됩니다. E6의 length, single-chain, dilute-solution, hydrogel 결과로 같은 version의 계수를 다시 튜닝하지 않습니다.

합성 데이터 예제 전체를 명령 한 줄로 실행할 수 있습니다.

```bash
bash example/03_qm_to_martini/protocol_project/run_demo.sh
```

실제 연구에서는 예제의 숫자가 아니라 schema와 단계 순서를 재사용합니다. 자세한 설명은 [`docs/PARAMETERIZATION_PROTOCOL.md`](docs/PARAMETERIZATION_PROTOCOL.md)에 있습니다.

#### xTB ensemble을 고수준 참조와 비교하는 검사

`hygel-qm-reference-audit`는 계산을 돌리지 않는 후처리 검사입니다. 서로 다른 증거를 한 점수로 섞지 않고 네 항목을 따로 판정합니다.

- `energy`: xTB와 DFT 등 고수준 참조의 상대에너지 최소점, 쌍별 순서, MAE/RMSE/최대 오차
- `gradient`: 고수준 단일점 gradient의 RMS와 최대 성분, stationarity
- `endpoint`: 독립 최적화 endpoint의 구조 무결성과 RMSD, 상대에너지에 따른 `SINGLE_DFT_ENDPOINT_FAMILY` 또는 `MULTIPLE_DFT_ENDPOINTS`
- `overlap`: `E_reference - E_xTB` 중요도 가중치의 effective sample size와 최대 단일 가중치

```bash
hygel-qm-reference-audit energy reference_energies.csv \
  --group-column chemistry \
  --output reference_energy_audit.json

hygel-qm-reference-audit overlap delta_energies.csv \
  --temperature-k 310 \
  --output overlap_audit.json
```

입력 CSV schema와 판정의 의미는 `hygel_martini/param_opt/qm_to_martini/analysis/README.md`에 있습니다. 이 검사를 통과하면 sparse refinement를 시작할 수 있습니다. xTB가 DFT ground truth가 되었다거나 최종 Martini 파라미터가 정해졌다는 뜻은 아닙니다.

### 04. Full builder

```bash
cd /path/to/hygel_martini/example/04_full_builder/project
bash run_full_builder.sh maker.yaml
```

Anisotropy smoke test:

```bash
bash run_full_builder.sh maker_anisotropy_x.yaml
```

### 04_1. Example system

```bash
cd /path/to/hygel_martini/example/04_1_example_system/project
bash run_example_system.sh maker.yaml
```

### 05. Hydrogel relaxation

```bash
cd /path/to/hygel_martini/example/05_hydrogel_relaxation/project
bash run_hydrogel_relaxation.sh maker_soft_em.yaml
bash run_hydrogel_relaxation.sh maker_soft_md.yaml
```

### 06. Physical properties

```bash
cd /path/to/hygel_martini/example/06_physical_property
hygel-property requirements \
  --analysis analysis_jobs.yaml \
  --requirements md_requirements.yaml \
  --strict
hygel-property analyze \
  --analysis analysis_jobs.yaml \
  --requirements md_requirements.yaml \
  --manifest validation_manifest.yaml
```

이 단계는 계산이 성공했는지, 결과를 분석할 수 있는지, 실험과 바로 비교할 수 있는지를 따로 판정합니다. 설정 형식은 `hygel_martini/property_extract/README.md`에 있습니다.

패키지에는 논문 workflow에서 쓴 다음 분석 기능이 들어 있습니다. reduced-network topology audit, PBC-safe geometry/diffusion, periodic clearance, static structure factor, voxel/field correlation, contact aggregation, paired finite-rate mechanics, realization-level 통계.

`docs/VALIDATION_HISTORY_AND_DESIGN_RATIONALE.md`에는 이 기능들을 채택하기까지 거친 validation 실패와 교정이 기록되어 있습니다. 결과를 해석하기 전에 fixed-water와 free-swelling, pore와 clearance, finite-rate와 equilibrium mechanics의 경계를 다룬 부분을 먼저 보기를 권합니다.

## Builder 상세

### Builder와 relaxation 실행

```bash
hygel-builder path/to/maker.yaml
hygel-builder --config path/to/maker.yaml
hygel-relax path/to/maker_soft_em.yaml
```

Builder 실행 흐름:

1. `04_*` 예제의 `run_example_system.sh` 또는 `run_full_builder.sh`
2. `hygel-builder` 또는 `python -m hygel_martini.hydrogel_builder`
3. `hygel_martini.hydrogel_builder.generator.run_hydrogel_builder`
4. `hygel_martini.hydrogel_builder.config_params.generator.run_hydrogel_example`
5. `hygel_martini.hydrogel_builder.config_params.read_json.execute_mode`

후처리 relaxation 흐름:

1. `05_hydrogel_relaxation/project/run_hydrogel_relaxation.sh`
2. `hygel-relax` 또는 `python -m hygel_martini.hydrogel_builder.relax`
3. `hygel_martini.hydrogel_builder.relax.generator.run_relax_workflow`
4. `hygel_martini.hydrogel_builder.relax.soft_em` 또는 `hygel_martini.hydrogel_builder.relax.soft_md`

### BCK dynamic crosslink

`dynamic_crosslink_targets_per_stub: 2`이면 BCK stub 하나가 가까운 backbone end 두 개에 결합합니다. BCK 두 개를 가진 linker 하나는 BCK-backbone bond 네 개와 polymer junction 두 개를 만듭니다.

```text
BCK1: polymer A -- polymer B
BCK2: polymer C -- polymer D
```

내부 BCK1-BCK2 bond는 linker template이나 topology에 정의되어 있을 때만 생깁니다. `connectivity_aware` 모드에서는 layout planner가 만든 exact endpoint-edge plan이 linker stub metadata로 넘어가고, runtime dynamic crosslink가 그 edge를 계획대로 결합으로 기록합니다. 거리로 정하는 것은 두 planned edge 중 어느 것을 어느 physical BCK stub에 연결할지뿐이며, nearest search가 planner endpoint를 다른 chain으로 바꾸지 않습니다. Plan metadata가 없는 layout에서만 geometry-only endpoint assignment와 global endpoint uniqueness 검사를 씁니다.

`connectivity_aware` 모드에서는 결합을 쓰기 전에 layout planner가 local matching으로 BCK junction 위치를 정합니다. 한 local vertex의 네 위치 `(000), (011), (101), (110)`의 perfect matching은 세 가지이고, 이를 x, y, z라고 부릅니다.

```text
x: (000-011), (101-110)
y: (000-101), (011-110)
z: (000-110), (011-101)
```

Planner는 x/y/z 선택이 vertex 전체에 고르게 분포하도록 하면서, backbone segment edge와 BCK junction edge를 합친 graph가 가능한 한 하나의 닫힌 circuit이 되도록 합니다. 이 planner는 `hygel_martini/hydrogel_builder/core_utils/layout/local_matching.py`에 순수 graph 루틴으로 분리되어 있습니다. 작은 lattice는 x/y/z 개수 차이가 1 이하인 assignment를 모두 평가하고, 큰 lattice는 balanced Kotzig-style two-vertex swap, Metropolis annealing, sampled greedy refinement를 씁니다.

```yaml
simulation_parameters:
  # Layout graph planner를 켭니다. x/y/z transition 선택은 여기서 정해집니다.
  linker_orientation_strategy: connectivity_aware

  # 배치된 BCK stub 하나가 결합할 backbone end 수입니다.
  # 2이면 two-BCK linker 하나 = polymer junction 두 개 = BCK-backbone bond 네 개입니다.
  dynamic_crosslink_targets_per_stub: 2

  # Plan metadata가 없을 때 runtime endpoint search의 후보 폭입니다.
  # loop나 component 수를 제한하지 않습니다.
  dynamic_crosslink_candidate_limit: 64

# 빌드 뒤 initial_hydrogel.gro/itp의 bonded graph를 검사하는 안전장치입니다.
# x/y/z 최적화나 loop 수를 직접 조절하지 않습니다.
hydrogel_topology_connectivity_audit:
  enabled: true
  min_largest_component_fraction: 0.95
  max_components: 1
  fail_on_violation: true
```

`connectivity_guard`는 `hydrogel_topology_connectivity_audit`의 옛 이름입니다. 새 YAML에는 새 이름을 쓰고, 옛 이름은 호환을 위해서만 읽습니다.

## 설정 파일

Builder 예제는 보통 `maker.yaml`에서 여러 YAML을 include합니다.

```yaml
includes:
  - config/simulation.yaml   # 실행 환경, GPU, 스레드, GROMACS 경로
  - config/hydrogel.yaml     # backbone/linker 종류와 비율
  - config/mdp.yaml          # EM MDP 파라미터 override
  - config/add_series.yaml   # 물/이온 추가 설정
  - config/backbone.yaml     # backbone 결합 파라미터
```

자주 쓰는 placeholder는 두 개입니다.

- `${CONFIG_DIR}`: 현재 maker 파일이 있는 디렉터리
- `${REPO_ROOT}`: 저장소 루트

```yaml
simulation_parameters:
  output_dir: ${CONFIG_DIR}/output
  gromacs_include_path: ${REPO_ROOT}/martini_v300
```

### 주요 `simulation.yaml` 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `gromacs_executable_path` | str | `gmx_mpi` | GROMACS 실행 파일 |
| `gpu_id` | int 또는 null | `null` | null은 CPU 전용, N은 GPU N번 고정 |
| `omp_threads` | int 또는 null | `null` | builder EM의 mdrun 스레드 수, null은 GROMACS가 결정 |
| `segment_length` | int | 없음 | backbone당 CG bead 수 |
| `number_of_cells` | int | 1 | 큰 셀 반복 횟수 |
| `mean_sep` | float | 없음 | bead 간 평균 간격(nm), water packing에도 사용 |
| `anisotropy` | false 또는 str | `false` | false는 isotropic, x/y/z는 해당 축 anisotropy |
| `gel_weight_fraction_mode` | str | `legacy` | 물 개수를 정수로 만드는 방식과 이온 포함 여부: `legacy`는 버림, `exclude_ions`·`include_ions`는 둘 다 올림 |
| `grompp_maxwarn` | int | 1 | builder EM 단계에서 허용하는 grompp 경고 수 (이온 추가 단계는 2로 고정) |
| `random_seed` | int | 없음 | 재현성을 위한 난수 시드 |
| `debug_mode` | bool | false | true이면 output/debug.txt에 상세 로그 |
| `neutral_em_coulombtype` | str 또는 null | `Cut-off` | 중성 EM 단계의 Coulomb 타입 |
| `packmol_pbc` | bool | false | Packmol 입력에 직교 `pbc`를 넣음(Packmol 20.15 이상) |

## 공용 서버에서 쓸 때

### GPU

GROMACS 2026의 `gmx_mpi mdrun`은 `-nb cpu`가 없으면 GPU를 자동으로 잡습니다. 다른 사용자의 GPU 작업과 겹치지 않도록 `config/simulation.yaml`에서 사용할 장치를 지정합니다.

```yaml
simulation_parameters:
  # null -> CPU 전용 (기본값, 공용 서버 권장)
  # 0    -> GPU 0번 전용 (CUDA_VISIBLE_DEVICES=0)
  # 1    -> GPU 1번 전용 (CUDA_VISIBLE_DEVICES=1)
  gpu_id: null
```

`05` soft_em과 soft_md는 `config/common.yaml`의 `runtime.gpu_id`를 씁니다.

### CPU 스레드

builder EM 단계(`backbone_stage`, `initial_hydrogel` 등)의 mdrun 스레드 수:

```yaml
simulation_parameters:
  # null -> GROMACS가 결정 (전체 코어를 잡을 수 있으니 공용 서버에서 주의)
  # N    -> mdrun -ntomp N
  omp_threads: 4
```

`05` soft_em과 soft_md는 `config/common.yaml`의 `runtime.omp_threads`나 각 섹션의 `ntomp`를 씁니다. SLURM에서는 `omp_threads`가 null일 때 `SLURM_CPUS_PER_TASK`(없으면 `SLURM_NTASKS`)를 씁니다.

## 환경 변수

- `ADDITIONAL_BASH_PROFILE`: 필요할 때만 추가로 source할 bash profile
- `ENV_NAME`: 필요할 때만 activate할 conda 환경 이름
- `ENVIRONMENT_FILE`: 각 example launcher가 먼저 source하는 project-local shell 설정 파일
- `GMXRC_PATH`: GROMACS 환경을 설정할 GMXRC 경로
- `GMX_CMD`: shell 수준의 GROMACS 명령 override (YAML의 `gromacs_executable_path`가 우선)
- `SLURM_CPUS_PER_TASK`, `SLURM_NTASKS`: SLURM에서 `omp_threads`가 null이면 mdrun `-ntomp` 값으로 사용
- `OPENMPI_HOME`, `ORCA_HOME`: ORCA 실행 경로

`OMP_NUM_THREADS`와 `GMX_OPENMP_MAX_THREADS`는 따로 설정하지 않아도 됩니다. builder EM의 스레드 수는 YAML의 `omp_threads`, relaxation의 스레드 수는 `runtime.omp_threads`로 정합니다.

`02_opls_to_martini`의 GROMACS/Bartender 경로는 `config/opls_existing_data.yaml`이나 `project/environment.sh`에서 맞춥니다. `03_qm_to_martini`의 xTB/ORCA/Bartender 경로는 shell 환경 변수보다 `config_common/common.yaml`에서 고칩니다. 어느 경우든 스크립트를 직접 고치기보다 환경 변수나 YAML로 맞추는 편이 낫습니다.

## 출력물과 디버깅

### `hydrogel_builder`

자주 보는 출력:

- `output/debug.txt`
- `output/dynamic_bonding_debug.log`
- `output/system.top`
- `output/final_system.gro`
- `output/final_optimized_system.gro`
- `output/*_geo_opt/grompp.log`
- `output/*_geo_opt/mdrun.log`
- `output/*_geo_opt/mdout.mdp` (grompp가 기록한 최종 파라미터 요약)

build가 막히면 `debug.txt`, 해당 단계의 `grompp.log`와 `mdrun.log`, `dynamic_bonding_debug.log`, `system.top` 순서로 봅니다.

### `param_opt`

- `03`: `runs/.../summary.json`
- `02`: `opls_bartender_runs/.../summary.json`

### `hydrogel_builder.relax`

- `05`: `relax_output/soft_em/final.gro`, `relax_output/soft_md/soft_md.gro`

## 저장소 구조

- `hygel_martini/`
  - `core/`: 물리 상수, 유틸리티, 설정 로더 (`physics.py`, `utils.py`, `config.py`)
  - `tools/`: `xtb_traj_to_pdb.py` 같은 범용 도구
  - `bash_settings/`: 모든 workflow의 Bash launcher (`common/`은 공통 환경과 Slurm 템플릿, `param_opt/`는 02·03 단계, `hydrogel_builder/`는 04 단계, `relaxation/`은 05 단계)
  - `param_opt/`: 파라미터 준비, 00~03 단계 (`bead_generator`는 00 단계 예정 위치, `qm_to_opls` 01, `opls_to_martini` 02, `qm_to_martini` 03이며 `workflow_logic`과 `analysis`를 나눔, `polymer_maker`는 공용 polymer xyz 생성기)
  - `hydrogel_builder/`: graph-first builder와 relaxation, 04·04_1·05 단계
  - `property_extract/`: manifest 기반 물성 분석과 비교 검사, 06 단계
- `example/`: fresh clone에 들어 있는 예제 project (파라미터 주석이 달린 참고 YAML 포함, 02는 직접 준비한 trajectory가 필요). 헷갈리면 `example/` 아래 각 `project/` 디렉터리만 보면 됩니다.
- `example_myrun/`: `.gitignore`에 들어 있는 local 실행 공간이며 fresh clone에는 없습니다.
- `martini_v300/`: 사용자가 따로 준비하는 local force-field 파일이며 배포하지 않습니다.

`param_opt` 기능은 루트 `hygel_martini.param_opt`가 아니라 각 workflow module로 직접 실행합니다.

```bash
python -m hygel_martini.param_opt.bead_generator --help
python -m hygel_martini.param_opt.qm_to_opls --config ...
python -m hygel_martini.param_opt.opls_to_martini --config ...
python -m hygel_martini.param_opt.qm_to_martini --config ...
python -m hygel_martini.param_opt.qm_to_martini.analysis.reference_qualification --help
```

## 추가 문서

- `docs/VALIDATION_HISTORY_AND_DESIGN_RATIONALE.md`: validation 실패, 교정, 폐기 기준과 현재 설계의 근거
- `example/README_ko.md`: 예제 project 구성
- `hygel_martini/param_opt/README.md`: 00~03 단계
- `hygel_martini/hydrogel_builder/README.md`: 04, 04_1, 05 단계
- `hygel_martini/property_extract/README.md`: 06 분석 job, validation manifest, requirement 검사

## 인용, 연구비, 라이선스

소프트웨어 인용 정보는 [`CITATION.cff`](CITATION.cff)에 있습니다. 방법론 논문이 출판되면 software release와 논문을 함께 인용하도록 이 파일을 갱신합니다.

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-25424498).

이 소프트웨어로 얻은 결과를 발표할 때는 workflow에서 실제로 쓴 제3자 도구도 함께 인용해 주십시오: GROMACS (Abraham et al., 2015), Martini 3 (Souza et al., 2021) 또는 Martini 2 (Marrink et al., 2007), Packmol (Martínez et al., 2009), MDAnalysis (Michaud-Agrawal et al., 2011; Gowers et al., 2016), NumPy (Harris et al., 2020), SciPy (Virtanen et al., 2020), pandas (McKinney, 2010), Numba (Lam et al., 2015), Matplotlib (Hunter, 2007), ASE (Hjorth Larsen et al., 2017), CoolProp (Bell et al., 2014), pymbar/MBAR (Shirts and Chodera, 2008), PoreBlazer (Sarkisov et al., 2020). DOI 목록은 `CITATION.cff`의 `references`에 있습니다.

저자들은 코드, 테스트, 문서의 일부를 AI 코딩 도구(Google Gemini, OpenAI Codex, Anthropic Claude)의 도움을 받아 작성하거나 고쳤습니다. 모든 변경은 저자가 검토했고, 소프트웨어의 책임은 저자에게 있습니다.

소스 코드는 MIT License로 배포합니다([`LICENSE`](LICENSE)). Martini force-field 파일과 제3자 의존성의 라이선스는 [`LICENSING.md`](LICENSING.md)를 참고하십시오.

## 변경 내역

### 0.1.3

문서만 바뀌었습니다. README를 영어로 쓰고 한국어판을 `README_KO.md`로 옮겼으며, AI 코딩 도구 목록에 Google Gemini를 넣었습니다. 코드는 버전 번호 외에는 0.1.2와 같습니다.

### 0.1.2

곁사슬(monomer) 배치의 결함 두 가지와 bonded patch 파일 누락 처리를 고치고, 선택형 Packmol PBC를 추가했습니다. 곁사슬이 없는 설정(`MONOMERS: []`, 예: PEGDA, Pluronic)은 0.1.1과 바이트 단위로 같은 출력을 씁니다.

- linker stub에는 곁사슬을 붙이지 않습니다. stub의 `backbone_type`은 crosslink 상대를 고르는 데만 씁니다. 0.1.1은 일치하는 monomer가 있으면 stub에도 곁사슬을 한 번 더 붙였습니다.
- 충돌 없이 곁사슬을 놓을 자리를 찾지 못하면 경고 대신 `RuntimeError`로 멈춥니다. 0.1.1은 그 위치를 건너뛰고 곁사슬이 빠진 구조를 그대로 썼습니다. 자유 고분자 경로에서는 새로 붙인 곁사슬끼리도 충돌 검사를 합니다.
- `bonded_topology_patch_file`을 지정했는데 파일이 없으면 `FileNotFoundError`로 멈춥니다. 0.1.1은 안내 메시지만 찍고 건너뛰어, 일반 각도 항이 그대로 남고 backbone torsion이 빠졌습니다. 키를 지정하지 않았을 때의 기본 경로(`config/backbone.yaml`)는 예전처럼 없어도 됩니다. portable PEGDA 예제는 PEO-BCK-PEO 각도 규칙을 명시합니다(출력은 같음).
- `simulation_parameters.packmol_pbc: true`를 주면 Packmol 입력에 직교 `pbc`를 넣고(Packmol 20.15 이상), 고정 base 좌표가 상자 `[0, L]` 밖에 있으면 멈춥니다. 기본값은 `false`이며, 이때는 뒤따르는 minimization이 상자 경계 너머의 겹침을 풉니다.

### 0.1.1

JCC 방법론 논문을 위한 첫 공개 release입니다(Zenodo DOI 10.5281/zenodo.23208480). 이전 Series-01 작업의 소프트웨어 스냅샷은 `0.1.0`(`2472810c9430`)입니다.
