# hygel_martini

`hygel_martini`는 hydrogel coarse-grained system 구축과
관련 파라미터 준비를 위한 Python 기반 워크플로 모음이다.

저장소는 두 개의 축으로 구성된다.

- `hydrogel_builder`
  hydrogel 구조 생성, topology 조합, water/ion 추가, GROMACS 기반
  geometry optimization
- `param_opt`
  ORCA, OPLS, xTB, Martini 입력 준비와 constructor 생성

이 문서는 저장소 구성, 예제 실행 방법, 설정 파일 규칙, 주요 출력물을 설명한다.

## 개요

이 저장소로 수행할 수 있는 대표 작업은 다음과 같다.

- monomer/linker template를 이용한 hydrogel 구조 생성
- anisotropy 또는 isotropy 기반 layout 구성
- polymer, small molecule, water, ion 단계적 추가
- Packmol과 GROMACS를 이용한 구조 정리
- ORCA/OPLS/xTB 기반 입력 준비와 CG 파라미터화 보조

## 디렉터리 구조

```text
hygel_martini/
├── hydrogel_builder/
│   ├── add_series/
│   ├── config_params/
│   ├── core_utils/
│   └── main_components/
├── param_opt/
├── martini_v300/
├── example/
├── example_myrun/
├── setup.py
└── README.md
```

각 디렉터리의 역할은 아래와 같다.

### `hydrogel_builder/`

hydrogel system 구축을 담당하는 본체 패키지다.

- `config_params`
  maker 파일 로드, YAML include 병합, 전체 실행 진입점
- `core_utils`
  IO, template loader, layout planner, runtime helper
- `main_components`
  `World`, `Hydrogel`, `Polymer`, `Attributes` 등 핵심 데이터 구조
- `add_series`
  polymer, molecule, water, ion 추가 단계

### `param_opt/`

builder에 들어가기 전 단계의 파라미터 준비 모듈이다.

- ORCA 입력 준비
- OPLS 관련 입력/결과 정리
- xTB 기반 구조 안정화
- constructor output 생성

### `martini_v300/`

Martini force-field ITP와 관련 기본 리소스를 보관한다.
builder example에서는 이 디렉터리를 include path로 사용한다.

### `example/`

배포용 예제를 보관하는 디렉터리다.

- 입력 파일
- 실행 스크립트
- 예제 설명 문서

를 기준으로 유지하는 것이 좋다.

### `example_myrun/`

로컬 실험용 작업 공간이다.
배포용 예제를 복사해서 테스트하거나 결과를 누적할 때 사용한다.

## 설치

### Python

- Python 3.9 이상
- `pip install -e .`

예시:

```bash
cd /path/to/hygel_martini
pip install -e .
```

### 외부 도구

예제에 따라 다음 도구가 필요하다.

- GROMACS
- Packmol
- ORCA
- xTB
- Conda

실행 스크립트는 기본 경로를 포함하고 있지만,
환경 변수를 이용해 다른 머신 환경에 맞출 수 있다.

## 실행 환경 변수

대표 launcher에서 자주 사용하는 override 변수는 다음과 같다.

- `CONDA_PROFILE`
  conda 초기화 스크립트 경로
- `CONDA_ENV_NAME`
  사용할 conda 환경 이름
- `GMX_CMD`
  GROMACS 실행 명령
- `GMX_BIN_DIR`, `GMX_LIB_DIR`
  GROMACS 바이너리와 라이브러리 경로
- `TOOLCHAIN_BIN_DIR`, `TOOLCHAIN_LIB_DIR`
  OpenMPI/CUDA 등 보조 toolchain 경로
- `GMXRC_PATH`
  `param_opt` 예제에서 source할 GMXRC 경로
- `OPENMPI_HOME`, `ORCA_HOME`
  ORCA 예제에서 사용할 OpenMPI, ORCA 위치
- `XTB_CMD`
  xTB 실행 파일 이름
- `PYTHON_BIN`
  일부 helper script에서 사용할 Python 실행 파일

환경이 다를 경우 스크립트를 직접 수정하기보다
먼저 환경 변수로 맞추는 편이 관리하기 쉽다.

## 예제 구성

`example/` 아래에는 번호 기반 예제가 정리되어 있다.

### `01_opls_from_orca`

ORCA 기반 OPLS 입력 준비 예제다.

- 실행 스크립트: `project/gemini.sh`
- 주요 설정: `project/maker.yaml`

### `02_martini_from_opls`

OPLS 기반 입력에서 Martini constructor를 생성하는 예제다.

- 실행 스크립트: `project/codex.sh`
- 주요 설정: `project/maker.yaml`

### `03_polymers_xtb_to_martini`

polymer 전처리, xTB 구조 안정화, 전기적 계산 보조 스크립트를 포함하는 예제다.

### `04_real_test_full_workflow`

builder 전체 워크플로를 확인하기 위한 대표 예제다.

- 실행 스크립트: `project/codex.sh`
- 축소 케이스: `project/maker_anisotropy_x.yaml`

### `05_sbma_dmaaps_system`

SBMA/DMAAPS 시스템 기반 예제다.

- 실행 스크립트: `project/codex.sh`
- 실제 시스템 구성을 확인할 때 사용한다.

보조 설명은 `example/README_ko.md`에 정리되어 있다.

## 빠른 시작

### hydrogel_builder 예제 실행

전체 builder 예제:

```bash
cd /path/to/hygel_martini
bash example/04_real_test_full_workflow/project/codex.sh
```

축소 anisotropy 예제:

```bash
cd /path/to/hygel_martini/example/04_real_test_full_workflow/project
bash codex.sh maker_anisotropy_x.yaml
```

SBMA/DMAAPS 예제:

```bash
cd /path/to/hygel_martini
bash example/05_sbma_dmaaps_system/project/codex.sh
```

### param_opt 예제 실행

OPLS -> Martini constructor 예제:

```bash
cd /path/to/hygel_martini
bash example/02_martini_from_opls/project/codex.sh
```

ORCA -> OPLS 예제:

```bash
cd /path/to/hygel_martini
bash example/01_opls_from_orca/project/gemini.sh
```

## example와 example_myrun 사용법

예제는 다음처럼 나누어 사용하는 편이 좋다.

- `example/`
  기준 예제 보관용
- `example_myrun/`
  로컬 재실행과 테스트용

예시:

```bash
cd /path/to/hygel_martini
mkdir -p example_myrun
cp -r example/04_real_test_full_workflow example_myrun/real_test_trial
cd example_myrun/real_test_trial/project
bash codex.sh maker_anisotropy_x.yaml
```

이 방식은 배포용 예제와 로컬 실험 결과를 분리하기 쉽다.

## hydrogel_builder 실행 흐름

대표 진입점은 다음 순서로 연결된다.

1. example의 `codex.sh`
2. example의 `run.py`
3. `hydrogel_builder.config_params.generator.run_hydrogel_example`
4. `hydrogel_builder.config_params.read_json.execute_mode`

`all` 모드에서 수행하는 주요 단계는 다음과 같다.

1. maker 파일 로드
2. YAML include 병합
3. 입력 파일 정리
4. atom type 및 ITP 정보 로드
5. backbone layout 생성
6. `World` materialization
7. dynamic crosslink 생성
8. backbone EM
9. chemical detail 확장
10. hydrogel EM
11. polymer 추가
12. molecule 추가
13. water 추가
14. ion 추가
15. 최종 topology 생성과 최종 EM

## param_opt 실행 흐름

`param_opt`는 builder와 별도로 동작한다.

예를 들어 `02_martini_from_opls` 예제는 저장소 루트에서 다음처럼 실행된다.

```bash
python3 -m param_opt --config example/02_martini_from_opls/project/maker.yaml
```

주요 하위 모듈은 다음과 같다.

- `param_opt/core`
  config, defaults, CLI 처리
- `param_opt/parameterize`
  ORCA, LigParGen 등 파라미터화 단계
- `param_opt/structure`
  monomer/polymer 구조 helper
- `param_opt/simulation`
  constructor output과 실행 스크립트 생성
- `param_opt/cg_fitting`
  CG fitting 관련 기능

## 설정 파일 구조

### include 기반 maker 파일

builder 예제의 maker 파일은 보통 include 기반 구조를 사용한다.

```yaml
includes:
  - config/simulation.yaml
  - config/hydrogel.yaml
  - config/mdp.yaml
  - config/add_series.yaml
  - config/backbone.yaml
```

이 방식은 설정을 역할별 파일로 나누기 쉽고,
필요한 부분만 수정하기도 편하다.

### 경로 placeholder

예제 설정 파일에서는 다음 placeholder를 사용할 수 있다.

- `${CONFIG_DIR}`
  maker 파일이 있는 디렉터리
- `${REPO_ROOT}`
  저장소 루트 디렉터리

예시:

```yaml
simulation_parameters:
  gromacs_include_path: ${REPO_ROOT}/martini_v300
  base_itp_file: ${REPO_ROOT}/martini_v300/martini_v3.0.0.itp
  output_dir: ${CONFIG_DIR}/output
```

구조 파일도 같은 방식으로 적을 수 있다.

```yaml
monomer:
  gro: ${CONFIG_DIR}/structure/pe.gro
  itp: ${CONFIG_DIR}/structure/pe.itp
```

`param_opt` 예제의 `paths.yaml`도 maker 파일 위치 기준으로 해석된다.

예시:

```yaml
paths:
  base_dir: ${CONFIG_DIR}
  out_root: ${CONFIG_DIR}/constructor_output
```

## 주요 출력물

builder 예제에서 자주 확인하는 파일은 다음과 같다.

- `debug.txt`
  전체 디버그 로그
- `dynamic_bonding_debug.log`
  dynamic crosslink 상세 로그
- `initial_backbone.gro`, `initial_backbone.itp`
  backbone-only 초기 결과
- `initial_hydrogel.gro`, `initial_hydrogel.itp`
  chemical detail 반영 직후 결과
- `packed_after_*.gro`
  packmol 단계별 중간 구조
- `system.top`
  최종 topology
- `final_system.gro`
  ion 추가 후 최종 구조
- `final_optimized_system.gro`
  마지막 geometry optimization 결과
- `*_geo_opt/grompp.log`
  GROMACS preprocessing 로그
- `*_geo_opt/mdrun.log`
  GROMACS 실행 로그

문제가 생겼을 때는 보통 아래 순서로 보면 된다.

1. `debug.txt`
2. 해당 단계의 `grompp.log`
3. 해당 단계의 `mdrun.log`
4. `dynamic_bonding_debug.log`
5. `system.top`

## 추가 문서

추가 문서는 `example/` 아래에 있다.

- `example/README_ko.md`
  example 설명
- `example/summary.md`
  구조와 파이프라인 요약
- `example/summary_ko.md`
  한글 요약

루트 README는 저장소 사용 안내서로 두고,
예제별 보조 설명이 필요할 때 example 문서를 참고하면 된다.
