# 시작 가이드

처음 보는 사람 기준으로는 아래 순서가 가장 쉽습니다.

## 0. 설치

```bash
git clone <repo_url>
cd hygel_martini
conda create -n hygel python=3.11 -y
conda activate hygel
python -m pip install -e .
```

설치 후에는 각 예제의 `project/` 디렉터리에서 launcher를 실행합니다.
launcher는 이 설치가 현재 Python 환경에 이미 돼 있다고 가정합니다.

## 1. 가장 먼저 돌릴 것

`04_full_builder`부터 확인합니다.

```bash
cd /path/to/hygel_martini/example/04_full_builder/project
bash run_full_builder.sh maker.yaml
```

이 단계는 `hydrogel_builder`가 전체적으로 살아 있는지 가장 빨리 확인하는 용도입니다.

## 2. 실제 예시 시스템을 만들 때

`04_1_example_system`을 봅니다.

```bash
cd /path/to/hygel_martini/example/04_1_example_system/project
bash run_example_system.sh maker.yaml
```

## 3. build 뒤 추가 완화가 필요할 때

`05_hydrogel_relaxation`을 봅니다.

```bash
cd /path/to/hygel_martini/example/05_hydrogel_relaxation/project
bash run_hydrogel_relaxation.sh maker_soft_em.yaml
bash run_hydrogel_relaxation.sh maker_soft_md.yaml
```

## 4. xTB/ORCA/Bartender가 목적일 때

`03` 예시를 봅니다.

```bash
cd /path/to/hygel_martini/example/03_qm_to_martini/project
bash run_qm_to_martini.sh config_common/common.yaml
```

환경만 먼저 확인하고 싶으면:

```bash
bash run_qm_to_martini.sh --check-xtb --check-bartender config_common/common.yaml
```

`md: off`면 geometry optimization까지만 진행하고 Bartender/MD는 생략됩니다.

## 5. Slurm에서 실행할 때 (03만 해당)

```bash
cd /path/to/hygel_martini/example/03_qm_to_martini/project
sbatch run_slurm.sh
```

`run_slurm.sh` 맨 위의 `#SBATCH` 줄과 `DEFAULT_CONFIG_REL`을 먼저 수정합니다.
기본값은 `-N 1 -n 1 -c 32`입니다. 스레드 수는 `-c`로만 조정하세요.

## 6. 단계별 의미

- `00`  bead selector 예정 위치
- `01`  ORCA/QM → OPLS
- `02`  OPLS → Martini
- `03`  QM/xTB → Martini
- `04`  Full hydrogel builder
- `04_1`  예시 시스템 builder
- `05`  Post-build relaxation

지금 `example/00`, `01`, `02`는 placeholder이고, 실제 ready-to-run example은 `03`, `04`, `04_1`, `05`입니다.

## 7. 환경 설정 방법

각 `project/environment.sh`를 편집합니다 (launcher를 직접 수정하지 않습니다).

```bash
# environment.sh 예시
ADDITIONAL_BASH_PROFILE=""   # conda profile 경로 (필요할 때만)
ENV_NAME=""                  # activate할 conda env (필요할 때만)
PYTHON_BIN="${PYTHON_BIN:-python3}"
```

이미 올바른 conda env가 활성화된 상태라면 두 값 모두 비워두면 됩니다.

## 8. 공통 규칙

- 실제 실행은 각 예제의 `project/` 디렉터리에서 시작합니다.
- launcher 스크립트 이름은 각 디렉터리마다 다릅니다 (위 참고).
- `example/`이 tracked 기준입니다. `example_myrun/`은 gitignore된 로컬 작업 공간입니다.
