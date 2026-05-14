# hygel_martini

`hygel_martini`는 두 개의 큰 축으로 나뉩니다.

- `param_opt`
  QM, OPLS, xTB 쪽 입력과 Martini 파라미터 준비
- `hydrogel_builder`
  실제 hydrogel system 생성과 post-build relaxation

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

## 빠른 시작

```bash
cd /path/to/hygel_martini
python -m pip install -e .
```

example launcher들은 이 설치가 현재 Python 환경에 이미 되어 있다고 가정합니다. 모든 실행 스크립트는 `hygel_martini/bash_settings/` 아래에 모여 있습니다.

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

## 어떤 폴더를 봐야 하나

- `example/`
  fresh clone에서도 바로 쓸 수 있는 tracked example project
- `param_opt/`
  00, 01, 02, 03 쪽 Python 본체
- `hydrogel_builder/`
  04, 04_1, 05 쪽 Python 본체
- `martini_v300/`
  Martini force-field 리소스

헷갈리면 `example/` 아래 각 `project/` 디렉터리만 보면 됩니다.

## 패키지 구조

### `param_opt`

`param_opt`는 workflow별 패키지로 나뉘어 있습니다.

- `param_opt.bead_generator`
  00 단계 예정 위치
- `param_opt.qm_to_opls`
  01 단계
- `param_opt.opls_to_martini`
  02 단계
- `param_opt.qm_to_martini`
  03 단계
- `param_opt.polymer_maker`
  공용 polymer xyz 생성기

루트 `param_opt`를 직접 실행하지 말고, workflow 모듈을 직접 실행합니다.

```bash
python -m param_opt.bead_generator --help
python -m param_opt.qm_to_opls --config ...
python -m param_opt.opls_to_martini --config ...
python -m param_opt.qm_to_martini --config ...
```

### `hydrogel_builder`

`hydrogel_builder`는 builder와 post-build relaxation을 나눠서 제공합니다.

```bash
python -m hydrogel_builder path/to/maker.yaml
python -m hydrogel_builder --config path/to/maker.yaml
python -m hydrogel_builder.relax path/to/maker_soft_em.yaml
```

실행 흐름은 아래처럼 이어집니다.

1. `04_*` 예제의 `run_example_system.sh` / `run_full_builder.sh`
2. `python -m hydrogel_builder`
3. `hydrogel_builder.generator.run_hydrogel_builder`
4. `hydrogel_builder.config_params.generator.run_hydrogel_example`
5. `hydrogel_builder.config_params.read_json.execute_mode`

후처리 relaxation은 아래 흐름입니다.

1. `05_hydrogel_relaxation/project/run_hydrogel_relaxation.sh`
2. `python -m hydrogel_builder.relax`
3. `hydrogel_builder.relax.generator.run_relax_workflow`
4. `hydrogel_builder.relax.soft_em` 또는 `hydrogel_builder.relax.soft_md`

## 설정 파일 규칙

builder 예제는 보통 `maker.yaml`에서 여러 YAML을 include합니다.

```yaml
includes:
  - config/simulation.yaml
  - config/hydrogel.yaml
  - config/mdp.yaml
  - config/add_series.yaml
  - config/backbone.yaml
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
  GROMACS 실행 명령
- `OMP_NUM_THREADS`, `GMX_OPENMP_MAX_THREADS`
  GROMACS/OpenMP thread 수
- `OPENMPI_HOME`, `ORCA_HOME`
  ORCA 쪽 실행 경로

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

- `example/README_ko.md`
  tracked example 구성 설명
- `param_opt/README.md`
  00, 01, 02, 03 구조 설명
- `hydrogel_builder/README.md`
  04, 04_1, 05 구조 설명
