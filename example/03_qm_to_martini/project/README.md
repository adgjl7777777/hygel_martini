# 03 QM to Martini

이 예제는 `hygel_martini.param_opt.qm_to_martini` workflow를 실행하는 tracked example입니다. Monomer xyz/init template에서 polymer geometry를 만들고, xTB/ORCA relaxation, xTB trajectory 생성, Bartender refit, postprocess screening까지 이어서 확인할 수 있습니다.

실행 산출물은 git에 올리지 않는 것을 전제로 합니다. 새 실험 결과를 오래 보관해야 하면 `example_myrun/03_qm_to_martini/project` 같은 로컬 작업 공간에 따로 둡니다.

## 빠른 시작

저장소 루트에서 editable install을 먼저 합니다.

```bash
cd /path/to/hygel_martini
python -m pip install -e .
```

그 다음 project 디렉터리에서 launcher를 실행합니다.

```bash
cd /path/to/hygel_martini/example/03_qm_to_martini/project
bash run_qm_to_martini.sh config_common/common.yaml
```

환경만 확인하려면:

```bash
bash run_qm_to_martini.sh --check-xtb --check-bartender config_common/common.yaml
```

## 주요 파일

- `run_qm_to_martini.sh`: package 안의 통합 launcher를 찾아 호출합니다.
- `run_slurm.sh`: Slurm batch에서 한 config를 실행하는 wrapper입니다.
- `run_compare.sh`: 이미 있는 xTB trajectory를 Bartender에 다시 물리는 단일 compare/refit launcher입니다.
- `postprocess.sh`: Bartender `gmx_out.itp`를 screening하는 단일 postprocess launcher입니다.
- `run_cds_iteration.sh`: C/D/S와 mode 조합을 반복 실행하는 단순한 batch helper입니다.
- `config_common/common.yaml`: S 기본 generation config입니다.
- `config_common/common_c.yaml`: C용 generation config입니다.
- `config_common/common_d.yaml`: D용 generation config입니다.
- `config_common/postprocess.yaml`: Bartender 후처리 screening config입니다.
- `config_compare/*.yaml`: term-generation mode별 override 예시입니다.

## Workflow 축

`config_common/*.yaml`에서 가장 중요한 축은 두 개입니다.

`bartender_pipeline.relaxation`:

- `xtb`: xTB geometry optimization을 수행합니다.
- `orca`: ORCA geometry optimization을 수행합니다.
- `off`: 생성된 geometry를 그대로 사용합니다.

`bartender_pipeline.md`:

- `xtb`: pipeline이 xTB MD trajectory를 만든 뒤 Bartender에 `-owntraj/-refit`로 넘깁니다.
- `existing`: 이미 존재하는 trajectory를 `bartender_pipeline.md_traj`에서 읽어 Bartender refit에 사용합니다.
- `bartender`: Bartender 내부 sampling 경로를 사용합니다.
- `xtb_nobartender`: xTB MD까지만 만들고 Bartender job은 만들지 않습니다.
- `off`: MD/Bartender 단계 없이 앞 단계까지만 수행합니다.

이미 만들어진 xTB trajectory에 Bartender만 다시 적용하려면 `run_compare.sh`를 씁니다. 이 wrapper는 내부에서 `relaxation=off`, `md=existing`, `execution.run_bartender=true`를 override합니다.

## 기본 generation

대표 진입점은 `config_common/common.yaml`입니다.

```bash
bash run_qm_to_martini.sh config_common/common.yaml
```

자주 쓰는 override:

```bash
bash run_qm_to_martini.sh config_common/common.yaml \
  --set bartender_pipeline.relaxation=orca

bash run_qm_to_martini.sh config_common/common.yaml \
  --set bartender_pipeline.md=off

bash run_qm_to_martini.sh config_common/common.yaml \
  --set 'system.sequences=[S,D,D,S]'

bash run_qm_to_martini.sh config_common/common.yaml \
  --set paths.out_root=/tmp/qm_to_martini_test
```

## 이미 있는 xTB trajectory로 Bartender refit

단일 케이스는 `run_compare.sh`로 실행합니다. 예를 들어 S의 기존 trajectory를 `topology_n0` mode로 Bartender에 다시 넣으려면:

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

raw xTB trajectory를 바로 넘길 수도 있습니다. 이 경우 pipeline이 auto-trim 후 Bartender refit에 쓸 PDB를 만듭니다.

```bash
MD_TRAJ=md_S/S/relax_xtb_geoopt/xtb.trj ...
```

로그를 남기려면:

```bash
LABEL=S \
BASE_CONFIG=config_common/common.yaml \
OUT_ROOT=compare_existing_terms/S/topology_n0 \
MD_TRAJ=md_S/S/relax_xtb_geoopt/xtb_traj.pdb \
TERM_MODE=topology_n \
TERM_N=0 \
MODE_TAG=topology_n0 \
LOG_PATH=compare_existing_terms/logs/S_topology_n0.log \
bash run_compare.sh
```

명령만 확인하려면 `DRY_RUN=1`을 붙입니다. 결과가 이미 있을 때 건너뛰려면 `SKIP_EXISTING=1`을 붙입니다.

## Bartender postprocess

Bartender가 만든 `gmx_out.itp`를 screening하려면 `postprocess.sh`를 씁니다.

```bash
LABEL=S \
MODE_TAG=topology_n0 \
INPUT_ROOT=compare_existing_terms/S/topology_n0 \
MIRROR_ROOT=compare_existing_terms \
OUTPUT_ROOT=postprocessing_result \
bash postprocess.sh
```

출력은 보통 아래에 생깁니다.

```text
postprocessing_result/S/topology_n0/
```

주요 결과 파일:

- `all_terms.json`: screening 전 전체 후보 term
- `all_terms.itp`: screening 전 전체 후보를 ITP 형태로 쓴 파일
- `screened_summary.json`: screening 결과 요약
- `screened_forcefield.itp`: 선택된 term만 모은 forcefield ITP
- `screening_report.json`: 입력, 출력, count, config summary
- `plots/*.csv`, `plots/*.pdf`: force metric과 RMSD plot

기본 postprocess config는 `config_common/postprocess.yaml`입니다. `postprocess.sh`는 `paths.out_root`, `paths.postprocess_mirror_root`, `paths.postprocess_output_root`를 command line override로 덮어씁니다. 따라서 config 안의 `out_root`는 단일 실행 예시값입니다.

## C/D/S 반복 실행

현재 반복은 `run_cds_iteration.sh` 하나에서 명시적으로 관리합니다.

```bash
labels=(C D S)
modes=(init_only all_unique topology_n0 topology_n1 topology_n2 topology_swap_n0 topology_swap_n1 topology_swap_n2)
```

모든 compare/refit:

```bash
bash run_cds_iteration.sh
```

모든 postprocess:

```bash
STAGE=postprocess bash run_cds_iteration.sh
```

compare 뒤 postprocess까지 이어서:

```bash
STAGE=both bash run_cds_iteration.sh
```

명령만 확인:

```bash
DRY_RUN=1 STAGE=compare bash run_cds_iteration.sh
DRY_RUN=1 STAGE=postprocess bash run_cds_iteration.sh
```

이 helper는 모든 job을 background로 던지고 마지막에 `wait`합니다. 기본값은 `MAX_JOBS=24`, `CORE_COUNT=24`이며, `CORE_START + (job index % CORE_COUNT)` 방식으로 core를 순환합니다. 작은 machine에서는 예를 들어 `MAX_JOBS=4 CORE_COUNT=4 bash run_cds_iteration.sh`처럼 줄여서 실행합니다.

반복 실행의 기본 trajectory 상대경로는 raw xTB 파일을 가정한 `relax_xtb_geoopt/xtb.trj`입니다. 이미 변환된 PDB를 쓰고 싶으면 아래처럼 바꿉니다.

```bash
TRAJ_REL=relax_xtb_geoopt/xtb_traj.pdb bash run_cds_iteration.sh
```

## Term-generation mode

| mode tag | TERM_MODE | TERM_N |
| --- | --- | --- |
| `init_only` | `init_only` | `0` |
| `all_unique` | `all_unique` | `0` |
| `topology_n0` | `topology_n` | `0` |
| `topology_n1` | `topology_n` | `1` |
| `topology_n2` | `topology_n` | `2` |
| `topology_swap_n0` | `topology_swap_n` | `0` |
| `topology_swap_n1` | `topology_swap_n` | `1` |
| `topology_swap_n2` | `topology_swap_n` | `2` |

`topology_n`은 graph topology 기준으로 가까운 조합만 허용합니다. `topology_swap_n`은 여기에 permutation/swap 허용을 더한 mode입니다.

## Postprocess 기준

`config_common/postprocess.yaml`의 screening은 다음 기준을 봅니다.

- `potentials`: angle/dihedral/improper의 funct 번호를 Bartender active line 그대로 쓸지, 숫자로 강제할지 정합니다.
- `bond_constraint_mode`: bond와 constraint가 같이 있을 때 어떻게 처리할지 정합니다.
- `candidate_source`: active line만 후보로 볼지, commented alternative까지 후보로 볼지 정합니다.
- `show_all_info`: commented alternative를 `all_terms`와 plot에 보존할지 정합니다.
- `multi_constant_metric`: RB dihedral처럼 constant가 여러 개인 potential의 force metric 계산법입니다.
- `thresholds.force_metric_min_mode`: force metric threshold가 absolute인지 relative인지 정합니다.
- `thresholds.force_metric_min`: section별 force metric 최소값입니다.
- `thresholds.rmsd_max`: RMSD 최대 허용값입니다.
- `write_plots`: PDF/CSV plot을 쓸지 정합니다.

기본값은 `potentials.<section>: bartender`, `candidate_source: active`, `bond_constraint_mode: bartender`입니다. 즉 Bartender가 active로 남긴 line을 그대로 screening 대상으로 삼습니다.

function만 비교하고 싶을 때는 output root를 나누고 env override를 씁니다.

```bash
LABEL=S \
MODE_TAG=topology_n1 \
INPUT_ROOT=compare_existing_terms/S/topology_n1 \
MIRROR_ROOT=compare_existing_terms \
OUTPUT_ROOT=postprocessing_result_angle10 \
POTENTIAL_ANGLES=10 \
bash postprocess.sh
```

## Troubleshooting

`BASE_CONFIG not found`:
`run_compare.sh`에 넘긴 `BASE_CONFIG`가 project directory 기준으로 존재하지 않는 경우입니다. 절대경로를 쓰거나 project 기준 상대경로를 확인합니다.

`missing=<...>/gmx_out.itp`:
postprocess 입력 compare 결과가 없습니다. 먼저 compare/refit 로그와 `compare_existing_terms/<LABEL>/<MODE>/<LABEL>/bartender_job/gmx_out.itp`를 확인합니다.

`taskset not found`:
local 실행에서 core pinning을 할 수 없습니다. Slurm 안에서는 `USE_SRUN=1`로 `srun`을 쓰거나 system에 `taskset`을 설치해야 합니다.

postprocess plot이 안 생김:
`write_plots: true`인지, 현재 Python 환경에 `matplotlib`이 있는지 확인합니다.

결과가 전부 reject됨:
`force_metric_min`, `rmsd_max`, `bond_constraint_mode`, `potentials`, input `gmx_out.itp`의 active/commented line을 같이 확인합니다. 먼저 `all_terms.json`과 `screening_report.json`을 보는 것이 좋습니다.
