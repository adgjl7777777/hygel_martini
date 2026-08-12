# hygel_martini

`hygel_martini`는 coarse-grained Martini hydrogel을 구성하고 분석하기
위한 연구용 Python 패키지입니다. Graph-first network construction부터
QM/OPLS 기반 파라미터 준비, post-build relaxation, topology audit,
물성 추출까지 하나의 재현 가능한 workflow로 연결합니다.

현재 배포 단계는 `0.1.0` alpha입니다.

**Author:** Daehong Kim

**Affiliation:** School of Chemical and Biological Engineering, Seoul National
University, 1 Gwanak-ro, Gwanak-gu, Seoul 08826, Republic of Korea

**ORCID:** [0009-0007-1647-9270](https://orcid.org/0009-0007-1647-9270)

- `param_opt`: QM, OPLS, xTB 입력과 Martini 파라미터 준비
- `hydrogel_builder`: hydrogel network 생성과 post-build relaxation
- `property_extract`: state, structure, transport, finite-rate mechanics 분석
- `tools`: trajectory 변환과 bonded-topology audit

이 저장소는 workflow와 Python 코드를 제공합니다. GROMACS, Packmol,
xTB, ORCA, Bartender 및 Martini force-field 파일은 라이선스와 설치
환경이 서로 다르므로 `pip` 패키지에 포함하거나 자동 설치하지 않습니다.

## 주요 디렉터리 구조

- `hygel_martini/`
  - `core/`: 프로젝트 전역에서 사용하는 물리 상수, 유틸리티, 설정 로더 (`physics.py`, `utils.py`, `config.py`)
  - `tools/`: 범용 실행 도구 모음 (`xtb_traj_to_pdb.py` 등)
  - `bash_settings/`: 모든 워크플로의 Bash 실행 스크립트 통합 관리
    - `common/`: 공통 환경 설정 및 Slurm 제출 템플릿
    - `param_opt/`: Stage 02/03 관련 실행 스크립트
    - `hydrogel_builder/`: Stage 04 관련 실행 스크립트
    - `relaxation/`: Stage 05 관련 실행 스크립트
  - `param_opt/`: 파라미터 최적화 Python 패키지
    - `opls_to_martini/`: 02 단계 워크플로
    - `qm_to_martini/`: 03 단계 워크플로 (내부 `workflow_logic`, `analysis` 분리)
  - `hydrogel_builder/`: graph-first builder 및 relaxation
  - `property_extract/`: manifest 기반 물성 분석과 비교 gate

## 빠른 시작

Python 3.9 이상이 필요합니다.

```bash
git clone https://github.com/adgjl7777777/hygel_martini.git
cd /path/to/hygel_martini
python -m pip install -e .
```

개발 및 packaging test 도구까지 설치하려면 다음을 사용합니다.

```bash
python -m pip install -e ".[dev]"
pytest
```

설치 후 제공되는 주요 명령은 다음과 같습니다.

| 명령 | 역할 | 동일한 module 실행 |
|---|---|---|
| `hygel-builder` | hydrogel construction | `python -m hygel_martini.hydrogel_builder` |
| `hygel-relax` | post-build relaxation | `python -m hygel_martini.hydrogel_builder.relax` |
| `hygel-property` | property extraction | `python -m hygel_martini.property_extract` |
| `hygel-qm-to-opls` | Stage 01 | `python -m hygel_martini.param_opt.qm_to_opls` |
| `hygel-opls-to-martini` | Stage 02 | `python -m hygel_martini.param_opt.opls_to_martini` |
| `hygel-qm-to-martini` | Stage 03 | `python -m hygel_martini.param_opt.qm_to_martini` |
| `hygel-parameter-protocol` | E0--E6 bonded-parameter decision protocol | `python -m hygel_martini.param_opt.qm_to_martini.protocol` |
| `hygel-qm-reference-audit` | xTB/고수준 참조 적합성 gate | `python -m hygel_martini.param_opt.qm_to_martini.analysis.reference_qualification` |
| `hygel-audit-topology` | bonded graph audit | `python -m hygel_martini.tools.audit_hydrogel_topology` |

각 명령의 현재 옵션은 `COMMAND --help`로 확인할 수 있습니다. Example
launcher들은 이 패키지가 현재 Python 환경에 설치되어 있다고
가정하며, 공용 Bash launcher는 `hygel_martini/bash_settings/`에
모여 있습니다.

### 02. Existing OPLS/GROMACS -> Martini

02는 이미 존재하는 OPLS/GROMACS production trajectory를 Bartender fitting에 재사용합니다. 저장소에는 실제 trajectory가 포함되지 않으므로 `example/02_opls_to_martini/project/config/opls_existing_data.yaml`의 `data/...` 경로를 사용자 데이터로 바꿔서 씁니다.

```bash
cd /path/to/hygel_martini/example/02_opls_to_martini/project
MODE=setup bash run_existing_opls.sh
MODE=md bash run_existing_opls.sh
MODE=md_notrim bash run_existing_opls.sh
```

`MODE` 하나가 trim 여부, Bartender job 생성 여부, 실제 실행 여부를 함께 정합니다. 자세한 mode 표는 `example/02_opls_to_martini/project/README.md`를 봅니다.

### 03. QM -> Martini

```bash
cd /path/to/hygel_martini/example/03_qm_to_martini/project
bash run_qm_to_martini.sh config_common/common.yaml
```

**신규 기능:**
- **Auto-Trimming**: Trajectory 기반 모드(`xtb`, `existing`)에서 통계적 수렴 지점을 자동으로 감지하여 불안정 구간을 잘라냅니다.
- **xTB Restart**: 시뮬레이션 중단 시 `xtbrestart`를 감지하여 자동으로 이어서 실행하고 Trajectory를 병합합니다.
- **Existing xTB -> Bartender**: 이미 있는 `xtb_traj.pdb`/trimmed trajectory를 `run_compare.sh`로 Bartender에 다시 넣고, `postprocess.sh`로 screened force field를 만듭니다.

이미 있는 xTB trajectory를 Bartender에 적용하는 최소 예시는 아래입니다.

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

C/D/S 반복 실행은 `example/03_qm_to_martini/project/run_cds_iteration.sh`를 봅니다.

#### 후보에서 tested-domain parameter까지

`hygel-parameter-protocol`은 후보 생성을 최종 파라미터 확정과 분리합니다.
mapping, bead/nonbonded parent, 후보 함수, 데이터 역할, 독립 grouping,
목적함수와 threshold를 먼저 checksum 동결한 뒤 다음 누적 증거단계를
순서대로 판정합니다.

```text
E0 provenance -> E1 analytic eligibility -> E2 grouped selection
-> E3 numerical realization -> E4 target + upstream non-regression
-> E5 unopened one-shot confirmation -> E6 transfer qualification
```

한 criterion이라도 통과하지 못하면 그 iteration의 뒤 단계는 열리지
않습니다. E5까지 통과한 값만 sealed tested-domain release가 되며, E6의
length/single-chain/dilute-solution/hydrogel 결과를 같은 version의 계수
튜닝으로 되돌려 보내지 않습니다.

완전한 합성 예제는 다음 한 줄로 재생할 수 있습니다.

```bash
bash example/03_qm_to_martini/protocol_project/run_demo.sh
```

실제 연구에 적용할 때는 예제의 숫자가 아니라 schema와 순서를
재사용해야 합니다. 상세 설명은
[`docs/PARAMETERIZATION_PROTOCOL.md`](docs/PARAMETERIZATION_PROTOCOL.md)에
있습니다.

#### xTB ensemble의 고수준 참조 적합성 검사

`hygel-qm-reference-audit`는 계산 실행기가 아니라 후처리 gate입니다.
서로 다른 증거를 한 점수로 섞지 않고 다음 네 항목을 독립적으로
판정합니다.

- `energy`: xTB와 DFT 등 고수준 참조의 상대에너지 최소점, 쌍별 순서,
  MAE/RMSE/최대 오차
- `gradient`: 고수준 단일점 gradient의 RMS/최대 성분과 stationarity
- `endpoint`: 독립 최적화 endpoint의 구조 무결성, RMSD, 상대에너지에
  따른 `SINGLE_DFT_ENDPOINT_FAMILY` 또는 `MULTIPLE_DFT_ENDPOINTS`
- `overlap`: `E_reference - E_xTB` 중요도 가중치의 effective sample size와
  최대 단일 가중치

예시는 다음과 같습니다.

```bash
hygel-qm-reference-audit energy reference_energies.csv \
  --group-column chemistry \
  --output reference_energy_audit.json

hygel-qm-reference-audit overlap delta_energies.csv \
  --temperature-k 310 \
  --output overlap_audit.json
```

입력 CSV schema와 판정의 의미는
`hygel_martini/param_opt/qm_to_martini/analysis/README.md`에 있습니다.
이 gate의 통과는 sparse refinement를 시작할 수 있다는 뜻이지, xTB가
DFT ground truth가 되었다거나 최종 Martini 파라미터가 확정됐다는
뜻은 아닙니다.

### 04. Full Builder

```bash
cd /path/to/hygel_martini/example/04_full_builder/project
bash run_full_builder.sh maker.yaml
```

anisotropy smoke:

```bash
bash run_full_builder.sh maker_anisotropy_x.yaml
```

### 04_1. Example System

```bash
cd /path/to/hygel_martini/example/04_1_example_system/project
bash run_example_system.sh maker.yaml
```

### 05. Hydrogel Relaxation

```bash
cd /path/to/hygel_martini/example/05_hydrogel_relaxation/project
bash run_hydrogel_relaxation.sh maker_soft_em.yaml
bash run_hydrogel_relaxation.sh maker_soft_md.yaml
```

### 06. Physical Property

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

이 단계는 계산 성공, 분석 가능성, 직접 실험 비교 가능성을 서로 다른
gate로 취급합니다. 자세한 설정 형식은
`hygel_martini/property_extract/README.md`를 봅니다.

패키지에는 논문 workflow에서 사용한 reduced-network topology audit,
PBC-safe geometry/diffusion, periodic clearance, static structure factor,
voxel/field correlation, contact aggregation, paired finite-rate mechanics와
realization-level 통계 primitive가 포함되어 있습니다.

이 기능들이 어떤 validation 실패와 교정을 거쳐 채택되었는지는
`docs/VALIDATION_HISTORY_AND_DESIGN_RATIONALE.md`에 정리되어 있습니다.
특히 fixed-water/free-swelling, pore/clearance, finite-rate/equilibrium
mechanics의 경계를 먼저 확인하는 것을 권장합니다.

## 인용, 연구비, 라이선스

소프트웨어 인용 정보는 [`CITATION.cff`](CITATION.cff)에 있습니다. 관련
방법론 논문의 최종 서지정보가 확정되면 software release와 논문을 함께
인용하도록 갱신합니다.

This work supported by the National Research Foundation of Korea (NRF) grant
funded by the Korea government (MSIT) (RS-2025-25424498).

현재 저장소에는 임의의 오픈소스 라이선스를 붙이지 않았습니다. 연구과제
계약 및 서울대학교 산학협력단 적용 여부를 확인한 뒤 승인된 라이선스를
선택해야 합니다. 그 전의 정확한 사용 경계는
[`LICENSING.md`](LICENSING.md)를 따릅니다.

## 공용 서버에서의 주의사항

### GPU 제어

GROMACS 2026의 `gmx_mpi mdrun`는 `-nb cpu` 없으면 GPU를 자동으로 잡습니다.
다른 사용자의 GPU 작업과 충돌을 방지하려면 `config/simulation.yaml`에서 명시적으로 지정하세요.

```yaml
simulation_parameters:
  # null → CPU 전용 (기본값, 공용 서버 권장)
  # 0    → GPU 0번 전용 (CUDA_VISIBLE_DEVICES=0)
  # 1    → GPU 1번 전용 (CUDA_VISIBLE_DEVICES=1)
  gpu_id: null
```

`05` soft_em / soft_md의 GPU 제어는 `config/common.yaml`의 `runtime.gpu_id`를 사용합니다.

### CPU 스레드 수 제어

빌드 EM 단계(`backbone_stage`, `initial_hydrogel` 등)의 mdrun 스레드 수:

```yaml
simulation_parameters:
  # null → GROMACS 자동 감지 (전체 코어를 잡을 수 있어 공용 서버에서 주의)
  # N    → mdrun -ntomp N 으로 실행
  omp_threads: 4
```

`05` soft_em / soft_md의 스레드 제어는 `config/common.yaml`의 `runtime.omp_threads` 또는 각 섹션의 `ntomp`를 사용합니다.

SLURM 환경이라면 `SLURM_CPUS_PER_TASK`가 `omp_threads: null`일 때 자동으로 사용됩니다.

## 어떤 폴더를 봐야 하나

- `example/`
  fresh clone에서도 바로 쓸 수 있는 tracked example project (파라미터 주석 포함한 레퍼런스 yaml)
- `example_myrun/`
  `.gitignore`에 포함된 local-only 실행 공간이며 fresh clone에는 없음
- `hygel_martini/param_opt/`
  00, 01, 02, 03 쪽 Python 본체
- `hygel_martini/hydrogel_builder/`
  04, 04_1, 05 쪽 Python 본체
- `hygel_martini/property_extract/`
  06 물성 분석 Python 본체
- `martini_v300/`
  사용자가 별도로 준비하는 local force-field 리소스이며 Git 배포 대상이 아님

헷갈리면 `example/` 아래 각 `project/` 디렉터리만 보면 됩니다.

## 패키지 구조

### `hygel_martini.param_opt`

`hygel_martini.param_opt`는 workflow별 패키지로 나뉘어 있습니다.

- `hygel_martini.param_opt.bead_generator`
  00 단계 예정 위치
- `hygel_martini.param_opt.qm_to_opls`
  01 단계
- `hygel_martini.param_opt.opls_to_martini`
  02 단계
- `hygel_martini.param_opt.qm_to_martini`
  03 단계
- `hygel_martini.param_opt.polymer_maker`
  공용 polymer xyz 생성기

루트 `hygel_martini.param_opt`를 직접 실행하지 말고, workflow 모듈을
직접 실행합니다.

```bash
python -m hygel_martini.param_opt.bead_generator --help
python -m hygel_martini.param_opt.qm_to_opls --config ...
python -m hygel_martini.param_opt.opls_to_martini --config ...
python -m hygel_martini.param_opt.qm_to_martini --config ...
python -m hygel_martini.param_opt.qm_to_martini.analysis.reference_qualification --help
```

### `hydrogel_builder`

`hydrogel_builder`는 builder와 post-build relaxation을 나눠서 제공합니다.

```bash
hygel-builder path/to/maker.yaml
hygel-builder --config path/to/maker.yaml
hygel-relax path/to/maker_soft_em.yaml
```

실행 흐름은 아래처럼 이어집니다.

1. `04_*` 예제의 `run_example_system.sh` / `run_full_builder.sh`
2. `hygel-builder` 또는 `python -m hygel_martini.hydrogel_builder`
3. `hygel_martini.hydrogel_builder.generator.run_hydrogel_builder`
4. `hygel_martini.hydrogel_builder.config_params.generator.run_hydrogel_example`
5. `hygel_martini.hydrogel_builder.config_params.read_json.execute_mode`

후처리 relaxation은 아래 흐름입니다.

1. `05_hydrogel_relaxation/project/run_hydrogel_relaxation.sh`
2. `hygel-relax` 또는 `python -m hygel_martini.hydrogel_builder.relax`
3. `hygel_martini.hydrogel_builder.relax.generator.run_relax_workflow`
4. `hygel_martini.hydrogel_builder.relax.soft_em` 또는
   `hygel_martini.hydrogel_builder.relax.soft_md`

### BCK dynamic crosslink 의미

`dynamic_crosslink_targets_per_stub: 2`는 한 BCK stub이 가까운 backbone end 두 개에 결합한다는 뜻입니다. 따라서 BCK 두 개를 가진 linker 하나는 총 네 개의 BCK-backbone bond를 만들고, polymer junction 두 개를 만듭니다.

```text
BCK1: polymer A -- polymer B
BCK2: polymer C -- polymer D
```

내부 BCK1-BCK2 bond는 linker template/topology가 명시할 때만 존재합니다. `connectivity_aware`
모드에서 layout planner가 만든 exact endpoint-edge plan은 linker stub metadata로
이어지고, runtime dynamic-crosslink는 그 edge를 그대로 materialize합니다.
두 planned edge 중 어느 것을 어느 physical BCK stub에 연결할지만 거리로
결정하며, nearest search로 planner endpoint를 다른 chain으로 재배선하지
않습니다. Planned metadata가 없는 layout에서만 geometry-only
endpoint assignment와 global endpoint uniqueness validation을 사용합니다.

Connectivity-aware BCK 배치는 runtime endpoint repair가 아니라 layout local-matching 문제입니다. 한 local vertex의 네 위치 `(000), (011), (101), (110)`에서는 x/y/z 선택이 아래 세 perfect matching 중 하나를 고릅니다.

```text
x: (000-011), (101-110)
y: (000-101), (011-110)
z: (000-110), (011-101)
```

목표는 각 vertex의 x/y/z 선택을 골고루 배치하면서, backbone segment edge와 BCK junction edge를 합친 graph가 가능한 한 하나의 self-returning cycle이 되게 하는 것입니다. 이 기준 알고리즘은 `hygel_martini/hydrogel_builder/core_utils/layout/local_matching.py`에 순수 graph planner로 분리되어 있습니다. 작은 lattice는 x/y/z count 차이가 1 이하인 transition assignment를 exact enumeration으로 평가하고, 큰 lattice는 balanced Kotzig-style two-vertex swap, Metropolis annealing, sampled greedy refinement를 사용합니다.

설정상 역할은 아래처럼 분리합니다.

```yaml
simulation_parameters:
  # Layout graph planner를 켭니다. x/y/z transition 선택은 여기서 결정됩니다.
  linker_orientation_strategy: connectivity_aware

  # 이미 배치된 각 BCK stub이 몇 개의 backbone end에 결합할지 지정합니다.
  # 2이면 one two-BCK linker = two polymer junctions = four BCK-backbone bonds.
  dynamic_crosslink_targets_per_stub: 2

  # Explicit planner metadata가 없는 runtime endpoint search의 후보 폭입니다.
  # Loop/component 제한값이 아닙니다.
  dynamic_crosslink_candidate_limit: 64

# 빌드 후 initial_hydrogel.gro/itp bonded graph를 검사하는 안전장치입니다.
# x/y/z transition 최적화나 loop 수를 직접 제어하지 않습니다.
hydrogel_topology_connectivity_audit:
  enabled: true
  min_largest_component_fraction: 0.95
  max_components: 1
  fail_on_violation: true
```

`connectivity_guard`는 과거 이름입니다. 새 YAML에서는 더 명확한 `hydrogel_topology_connectivity_audit`를 사용하고, 기존 파일 호환을 위해서만 legacy alias로 읽습니다.

## 설정 파일 규칙

builder 예제는 보통 `maker.yaml`에서 여러 YAML을 include합니다.

```yaml
includes:
  - config/simulation.yaml   # 실행 환경, GPU, 스레드, GROMACS 경로
  - config/hydrogel.yaml     # backbone/linker 종류 및 비율
  - config/mdp.yaml          # EM MDP 파라미터 override
  - config/add_series.yaml   # 물/이온 추가 설정
  - config/backbone.yaml     # backbone 상세 결합 파라미터
```

자주 쓰는 placeholder는 두 개입니다.

- `${CONFIG_DIR}`
  현재 maker 파일이 있는 디렉터리
- `${REPO_ROOT}`
  저장소 루트

예시:

```yaml
simulation_parameters:
  output_dir: ${CONFIG_DIR}/output
  gromacs_include_path: ${REPO_ROOT}/martini_v300
```

### 주요 simulation.yaml 파라미터 빠른 참조

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `gromacs_executable_path` | str | `gmx_mpi` | GROMACS 실행 파일 |
| `gpu_id` | int\|null | `null` | null=CPU 전용, N=GPU N번 고정 |
| `omp_threads` | int\|null | `null` | EM mdrun 스레드 수. null=자동감지 |
| `segment_length` | int | — | backbone당 CG bead 수 |
| `number_of_cells` | int | 1 | 큰 셀 반복 횟수 |
| `mean_sep` | float | — | bead 간 평균 간격 (nm), water packing 기준 |
| `anisotropy` | false\|str | `false` | false=isotropic, x/y/z=해당 축 anisotropy |
| `gel_weight_fraction_mode` | str | — | `exclude_ions` 또는 `include_ions` |
| `grompp_maxwarn` | int | — | grompp 허용 경고 수 |
| `random_seed` | int | — | 재현성을 위한 난수 시드 |
| `debug_mode` | bool | false | true=output/debug.txt에 상세 로그 |
| `neutral_em_coulombtype` | str\|null | `Cut-off` | 중성 EM 단계 Coulomb 타입 |

## 자주 쓰는 환경 변수

- `ADDITIONAL_BASH_PROFILE`
  필요할 때만 추가로 source할 bash profile 경로
- `ENV_NAME`
  필요할 때만 activate할 conda 환경 이름
- `ENVIRONMENT_FILE`
  각 example launcher가 먼저 source하는 project-local shell 설정 파일
- `GMXRC_PATH`
  GROMACS 환경을 source할 GMXRC 경로
- `GMX_CMD`
  GROMACS 실행 명령 (shell 레벨 override; yaml의 `gromacs_executable_path`가 우선)
- `SLURM_CPUS_PER_TASK`, `SLURM_NTASKS`
  SLURM 환경에서 자동으로 mdrun `-ntomp` 값으로 사용됨 (yaml `omp_threads: null`일 때만)
- `OPENMPI_HOME`, `ORCA_HOME`
  ORCA 쪽 실행 경로

> **참고**: `OMP_NUM_THREADS` / `GMX_OPENMP_MAX_THREADS`는 직접 설정하지 않아도 됩니다.
> 빌드 단계 EM은 yaml의 `omp_threads`로, relaxation 단계는 `runtime.omp_threads`로 제어됩니다.

`02_opls_to_martini`의 GROMACS/Bartender 경로는 `config/opls_existing_data.yaml` 또는 `project/environment.sh`에서 맞춥니다.
`03_qm_to_martini`의 xTB / ORCA / Bartender 경로는 shell 환경 변수보다 `config_common/common.yaml` 쪽을 수정하는 것을 기준으로 합니다.

가능하면 스크립트를 직접 수정하지 말고 먼저 환경 변수로 맞추는 편이 낫습니다.

## 출력물과 디버깅

### `hydrogel_builder`

자주 보는 출력은 아래와 같습니다.

- `output/debug.txt`
- `output/dynamic_bonding_debug.log`
- `output/system.top`
- `output/final_system.gro`
- `output/final_optimized_system.gro`
- `output/*_geo_opt/grompp.log`
- `output/*_geo_opt/mdrun.log`
- `output/*_geo_opt/mdout.mdp`  ← grompp가 기록하는 최종 파라미터 요약

builder 쪽에서 막히면 보통 이 순서로 확인하면 됩니다.

1. `debug.txt`
2. 해당 단계의 `grompp.log`
3. 해당 단계의 `mdrun.log`
4. `dynamic_bonding_debug.log`
5. `system.top`

### `param_opt`

- `03`
  `runs/.../summary.json`
- `02`
  `opls_bartender_runs/.../summary.json`

### `hydrogel_builder.relax`

- `05`
  `relax_output/soft_em/final.gro`
  `relax_output/soft_md/soft_md.gro`

## 추가 문서

- `docs/VALIDATION_HISTORY_AND_DESIGN_RATIONALE.md`
  A/B validation의 실패, 교정, 폐기 기준과 현재 코드의 설계 추적성
- `example/README_ko.md`
  tracked example 구성 설명
- `hygel_martini/param_opt/README.md`
  00, 01, 02, 03 구조 설명
- `hygel_martini/hydrogel_builder/README.md`
  04, 04_1, 05 구조 설명
- `hygel_martini/property_extract/README.md`
  06 분석 job, validation manifest, requirement gate 설명
