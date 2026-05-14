# 02 OPLS to Martini

이 예제는 이미 존재하는 OPLS/GROMACS production data를 Martini/Bartender fitting에 재사용하기 위한 02 workflow입니다.

여기서는 OPLS input 생성이나 GROMACS MD 실행을 하지 않습니다. 필요한 것은 이미 만들어진 trajectory와 Bartender input입니다.

## 입력 개념

한 case는 보통 아래 파일을 가집니다.

- `geometry`: Bartender `-refit`에 넘길 reference geometry (`.gro`, `.pdb`, `.xyz` 등)
- `bartender_inp`: fitting할 Martini/Bartender `.inp`
- `trajectory`: 이미 있는 OPLS/GROMACS MD trajectory (`.xtc`, `.trr`, `.pdb`)
- `tpr`: `.xtc`/`.trr`를 PDB로 바꿀 때 쓰는 GROMACS `.tpr`
- `edr`: energy-based trim에 쓸 GROMACS `.edr`

`.pdb` trajectory를 이미 가지고 있으면 `tpr`는 없어도 됩니다. `edr`가 없으면 `auto_trim`은 energy 기준 t0를 잡지 못하고 `skip_frames`만 적용합니다.

## 기본 실행

```bash
cd /path/to/hygel_martini/example/02_opls_to_martini/project
bash run_opls_to_martini.sh config/opls_existing_data.yaml
```

기본값은 setup-only입니다. 즉 실제 GROMACS/Bartender를 바로 실행하지 않고 아래를 생성합니다.

```text
opls_bartender_runs/<LABEL>/<MODE>/<LABEL>/
  case.json
  trim/run_prepare_md.sh
  bartender_job/run_bartender.sh
opls_bartender_runs/run_all.sh
```

전체를 나중에 실행하려면:

```bash
bash opls_bartender_runs/run_all.sh
```

평소에는 아래 wrapper를 쓰면 됩니다. `MODE` 하나가 trim 여부, Bartender 여부, 실제 실행 여부를 같이 정합니다.

```bash
MODE=setup bash run_existing_opls.sh
MODE=md bash run_existing_opls.sh
MODE=md_notrim bash run_existing_opls.sh
MODE=trim bash run_existing_opls.sh
MODE=bartender bash run_existing_opls.sh
```

`config/opls_existing_data.yaml` 안의 `opls_data.execution.mode`로 같은 값을 고정해도 됩니다.

## execution mode

02에서는 기존 OPLS/GROMACS MD를 쓰기 때문에 사용자가 직접 낮은 레벨의 flag를 여러 개 맞추지 않도록 `opls_data.execution.mode` preset을 둡니다.

| mode | 하는 일 |
| --- | --- |
| `setup` | trim/Bartender shell job만 만들고 실행하지 않습니다. |
| `md` | 기존 MD trajectory를 PDB로 변환하고 trim한 뒤 Bartender를 실행합니다. |
| `md_notrim` | 기존 MD trajectory를 PDB로 변환하지만 energy auto-trim은 하지 않고 Bartender를 실행합니다. |
| `trim` | trajectory prepare/trim까지만 실행하고 Bartender job은 만들지 않습니다. |
| `bartender` | Bartender를 실행합니다. 준비된 `md_traj.pdb`가 없으면 trim/prepare를 먼저 실행합니다. |
| `bartender_notrim` | `bartender`와 같지만 auto-trim 없이 trajectory를 준비합니다. |
| `notrim_nobartender` | auto-trim 없이 trajectory만 준비하고 Bartender job은 만들지 않습니다. |
| `off` | metadata scaffold만 만듭니다. |

내부적으로는 이 mode가 `bartender_pipeline.md`, `bartender_pipeline.bartender.enabled`, `opls_data.execution.run_trim`, `opls_data.execution.run_bartender`를 같이 맞춥니다. 직접 낮은 레벨 값을 만질 필요는 거의 없습니다.

하위 호환을 위해 `bartender_pipeline.md`에 쓰는 `existing`, `gromacs`, `bartender-noxtb`는 `md` alias로 처리됩니다. `opls_data.execution.mode`에서도 `bartender_noxtb`, `md_nobartender`, `md_notrim_nobartender` 같은 alias를 받습니다.

## postprocess

Bartender가 끝난 뒤 screening만 따로 돌리려면:

```bash
bash run_opls_to_martini.sh config/postprocess.yaml --postprocess-only
```

또는 02용 wrapper를 씁니다.

```bash
INPUT_ROOT=opls_bartender_runs/S/topology_n0 \
MIRROR_ROOT=opls_bartender_runs \
OUTPUT_ROOT=postprocessing_result \
bash postprocess.sh
```

다른 output root에 threshold 비교를 하고 싶으면:

```bash
bash run_opls_to_martini.sh config/postprocess.yaml --postprocess-only \
  --set paths.out_root=opls_bartender_runs/S/topology_n0 \
  --set paths.postprocess_output_root=postprocessing_result_rmsd5 \
  --set bartender_pipeline.postprocess.screening.thresholds.rmsd_max=5.0
```

## tool check

```bash
bash run_opls_to_martini.sh config/opls_existing_data.yaml --check-gmx
bash run_opls_to_martini.sh config/opls_existing_data.yaml --check-bartender
```

## C/D/S batch helper

`config/opls_existing_data.yaml` 안에 C/D/S와 mode별 `cases` 또는 `variants`를 넣어둔 경우에는 아래 helper로 한 번에 실행할 수 있습니다.

```bash
MODE=setup STAGE=fit bash run_cds_iteration.sh
MODE=md STAGE=both bash run_cds_iteration.sh
STAGE=postprocess bash run_cds_iteration.sh
```

03과 달리 02에서는 term-generation을 새로 만들지 않습니다. 이미 준비된 `bartender_inp`를 `cases[].variants[]`에 나열하고, helper는 그 config를 실행하거나 결과 postprocess를 반복합니다.

## GROMACS 변환 세부값

`config/opls_existing_data.yaml`의 `opls_data.trim.trjconv_selections`는 `gmx trjconv` interactive selection에 들어갑니다.

기본값은:

```yaml
trjconv_selections:
  - System
trjconv_extra: []
```

`-center`를 쓰는 경우에는 보통 selection이 두 번 필요합니다.

```yaml
trjconv_selections:
  - Polymer
  - System
trjconv_extra:
  - -pbc
  - mol
  - -center
```
