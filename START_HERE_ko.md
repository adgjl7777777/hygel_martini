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

설치 후에는 각 예제의 `project/` 디렉터리에서 `bash_settings`에 있는 launcher를 실행합니다.
launcher는 이 설치가 현재 Python 환경에 이미 돼 있다고 가정합니다.

## 1. 가장 먼저 돌릴 것

`04_full_builder`부터 확인합니다.

```bash
cd /path/to/hygel_martini/example/04_full_builder/project
bash ../../../hygel_martini/bash_settings/hydrogel_builder/run_full_builder.sh maker.yaml
```

이 단계는 `hydrogel_builder`가 전체적으로 살아 있는지 가장 빨리 확인하는 용도입니다.

## 2. 실제 예시 시스템을 만들 때

`04_1_example_system`을 봅니다.

```bash
cd /path/to/hygel_martini/example/04_1_example_system/project
bash ../../../hygel_martini/bash_settings/hydrogel_builder/run_example_system.sh maker.yaml
```

## 2-1. f=6 (hexafunctional, 정육면체 net) 시스템을 볼 때

`07_hexafunctional`을 봅니다. GROMACS 없이 layout/plan만 검사하려면:

```bash
cd /path/to/hygel_martini
PYTHONPATH=$PWD python3 - <<'SNIP'
from hygel_martini.hydrogel_builder.core_utils.layout.net_layout import generate_net_layout_plan
class P: pass
r = generate_net_layout_plan(P(), [{"id": "BB1"}], [{"id": "HEX"}],
                             net="pcu", repeats=4, cell_parameter=3.0,
                             max_span=6.0, rewire_seed=0)
print(r.summary())
SNIP
```

설정(`network_layout` 블록), net별 repeat 조건, rewiring 파라미터는
`docs/GENERAL_FUNCTIONALITY_NETWORKS.md`에 있습니다. f=6 경로는 GROMACS
end-to-end 빌드·EM 수렴·감사까지 통과했습니다. GROMACS/Packmol이 있으면
`hygel-builder maker.yaml`로 전체 빌드가 실행됩니다.

## 3. build 뒤 추가 완화가 필요할 때

`05_hydrogel_relaxation`을 봅니다.

```bash
cd /path/to/hygel_martini/example/05_hydrogel_relaxation/project
bash ../../../hygel_martini/bash_settings/relaxation/run_hydrogel_relaxation.sh maker_soft_em.yaml
```

## 4. 이미 있는 OPLS/GROMACS MD를 Martini fitting에 쓸 때

`02` 예시를 봅니다. 이 단계는 OPLS input 생성이나 GROMACS production run을 하지 않고, 이미 있는 `.xtc/.trr + .tpr (+ .edr)` 또는 `.pdb` trajectory를 Bartender refit에 재사용합니다.

```bash
cd /path/to/hygel_martini/example/02_opls_to_martini/project
MODE=setup bash run_existing_opls.sh
```

실제로 trim 후 Bartender까지 실행하려면:

```bash
MODE=md bash run_existing_opls.sh
```

trim 없이 기존 trajectory를 쓰려면:

```bash
MODE=md_notrim bash run_existing_opls.sh
```

결과 postprocess는:

```bash
INPUT_ROOT=opls_bartender_runs/S/topology_n0 \
MIRROR_ROOT=opls_bartender_runs \
OUTPUT_ROOT=postprocessing_result \
bash postprocess.sh
```

`02`는 실제 production trajectory를 저장소에 포함하지 않습니다. 먼저 `config/opls_existing_data.yaml`의 `data/...` 경로를 사용자 데이터로 바꿉니다.

## 5. xTB/ORCA/Bartender가 목적일 때

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

이미 만들어 둔 xTB trajectory에 Bartender만 다시 적용하려면 `run_compare.sh`를 씁니다.

```bash
LABEL=S \
BASE_CONFIG=config_common/common.yaml \
OUT_ROOT=compare_existing_terms/S/topology_n0 \
MD_TRAJ=md_S/S/relax_xtb_geoopt/xtb_traj.pdb \
TERM_MODE=topology_n \
TERM_N=0 \
MODE_TAG=topology_n0 \
bash run_compare.sh
```

Bartender 결과를 screened ITP로 정리하려면 이어서 `postprocess.sh`를 실행합니다.

```bash
LABEL=S \
MODE_TAG=topology_n0 \
INPUT_ROOT=compare_existing_terms/S/topology_n0 \
MIRROR_ROOT=compare_existing_terms \
OUTPUT_ROOT=postprocessing_result \
bash postprocess.sh
```

C/D/S 전체 반복은 `STAGE=compare|postprocess|both bash run_cds_iteration.sh`로 확인합니다. 자세한 설명은 `example/03_qm_to_martini/project/README.md`에 있습니다.

## 6. Slurm에서 실행할 때

```bash
cd /path/to/hygel_martini/example/03_qm_to_martini/project
sbatch run_slurm.sh config_common/common.yaml
```

## 7. 단계별 의미

- `00`  bead selector 예정 위치
- `01`  ORCA/QM → OPLS
- `02`  OPLS → Martini
- `03`  QM/xTB → Martini
- `04`  Full hydrogel builder
- `04_1`  예시 시스템 builder
- `05`  Post-build relaxation
- `07`  f=6 hexafunctional (pcu net) — 2-1 참조

지금 `example/00`, `01`은 placeholder입니다. `02`는 기존 OPLS/GROMACS data를 연결해야 하는 template-ready workflow이고, 실제 ready-to-run example은 `03`, `04`, `04_1`, `05`입니다.

## 8. 환경 설정 방법

모든 워크플로는 `hygel_martini/bash_settings/common/environment.sh`를 공통으로 참조할 수 있습니다. 
개별 `project/environment.sh`를 편집하여 설정을 덮어쓸 수 있습니다.

```bash
# environment.sh 예시
ADDITIONAL_BASH_PROFILE=""   # conda profile 경로 (필요할 때만)
ENV_NAME=""                  # activate할 conda env (필요할 때만)
PYTHON_BIN="${PYTHON_BIN:-python3}"
```

## 9. 공통 규칙

- 실제 실행은 각 예제의 `project/` 디렉터리에서 시작합니다.
- launcher 스크립트들은 `hygel_martini/bash_settings/` 아래에 워크플로별로 모여 있습니다.
- `example/`이 tracked 기준입니다. `example_myrun/`은 gitignore된 로컬 작업 공간입니다.
