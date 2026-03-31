# hygel_martini

`hygel_martini`는 두 개의 큰 축으로 나뉩니다.

- `param_opt`
  QM, OPLS, xTB 쪽 입력과 Martini 파라미터 준비
- `hydrogel_builder`
  실제 hydrogel system 생성과 post-build relaxation

처음 쓰는 경우에는 `example/`보다 `example_myrun/`을 먼저 보면 됩니다.
`example/`은 배포용 참고본이고, 실제 실행과 수정은 `example_myrun/` 기준으로
정리되어 있습니다.

정말 처음이면 `START_HERE_ko.md`부터 보는 편이 더 빠릅니다.

## 빠른 시작

```bash
cd /path/to/hygel_martini
pip install -e .
```

지금 바로 돌려볼 수 있는 예시는 `03`, `04`, `04_1`, `05`입니다.
`00`, `01`, `02` example 슬롯은 자리만 잡아둔 placeholder입니다.

### 03. QM -> Martini

```bash
cd /path/to/hygel_martini/example_myrun/03_qm_to_martini/project
bash hygel_run.sh
```

`md: off`로 두면 geometry optimization까지만 진행하고 Bartender/MD 단계는 생략합니다.

### 04. Full Builder

```bash
cd /path/to/hygel_martini/example_myrun/04_full_builder/project
bash hygel_run.sh
```

anisotropy smoke:

```bash
bash hygel_run.sh maker_anisotropy_x.yaml
```

### 04_1. Example System

```bash
cd /path/to/hygel_martini/example_myrun/04_1_example_system/project
bash hygel_run.sh
```

### 05. Hydrogel Relaxation

```bash
cd /path/to/hygel_martini/example_myrun/05_hydrogel_relaxation/project
bash hygel_run.sh maker_soft_em.yaml
bash hygel_run.sh maker_soft_md.yaml
```

## 어떤 폴더를 봐야 하나

- `example/`
  배포용 참고본
- `example_myrun/`
  실제 실행용 작업 공간
- `param_opt/`
  00, 01, 02, 03 쪽 Python 본체
- `hydrogel_builder/`
  04, 04_1, 05 쪽 Python 본체
- `martini_v300/`
  Martini force-field 리소스

헷갈리면 `example_myrun/`의 `project/` 디렉터리만 보면 됩니다.

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

1. `04_*` 예제의 `hygel_run.sh`
2. `python -m hydrogel_builder`
3. `hydrogel_builder.generator.run_hydrogel_builder`
4. `hydrogel_builder.config_params.generator.run_hydrogel_example`
5. `hydrogel_builder.config_params.read_json.execute_mode`

후처리 relaxation은 아래 흐름입니다.

1. `05_hydrogel_relaxation/project/hygel_run.sh`
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
  각 `hygel_run.sh`가 먼저 source하는 project-local shell 설정 파일
- `HYGEL_REPO_ROOT`
  예제 폴더를 다른 곳으로 옮겼을 때 직접 지정할 저장소 루트
- `GMXRC_PATH`
  GROMACS 환경을 source할 GMXRC 경로
- `GMX_CMD`
  GROMACS 실행 명령
- `OMP_NUM_THREADS`, `GMX_OPENMP_MAX_THREADS`
  GROMACS/OpenMP thread 수
- `OPENMPI_HOME`, `ORCA_HOME`
  ORCA 쪽 실행 경로

`03_qm_to_martini`의 xTB / ORCA / Bartender 경로는 shell 환경 변수보다
`config_common/common.yaml` 쪽을 수정하는 것을 기준으로 합니다.

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

### `hydrogel_builder.relax`

- `05`
  `relax_output/soft_em/final.gro`
  `relax_output/soft_md/soft_md.gro`

## 추가 문서

- `example_myrun/README_ko.md`
  실제 작업용 예제 설명
- `param_opt/README.md`
  00, 01, 02, 03 구조 설명
- `hydrogel_builder/README.md`
  04, 04_1, 05 구조 설명
