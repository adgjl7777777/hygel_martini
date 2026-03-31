# 03 QM to Martini

이 디렉터리는 `param_opt.qm_to_martini`를 실제로 돌리는 active 예시입니다.
monomer xyz/init template에서 polymer를 만들고, 선택적으로 QM/xTB geometry optimization,
xTB trajectory 생성, Bartender 후처리를 이어서 수행합니다.

## 실행

지금은 `config_common/common.yaml` 하나를 기본 entrypoint로 씁니다.

```bash
bash run_qm_to_martini.sh
```

도움말과 환경 확인:

```bash
bash run_qm_to_martini.sh --help
bash run_qm_to_martini.sh --check-xtb --check-bartender
bash run_qm_to_martini.sh --postprocess-only
```

오버라이드 예시:

```bash
bash run_qm_to_martini.sh --set bartender_pipeline.relaxation=orca
bash run_qm_to_martini.sh --set bartender_pipeline.md=off
bash run_qm_to_martini.sh --set 'system.sequences=[S,D,D,S]'
bash run_qm_to_martini.sh --set paths.out_root=/tmp/qm_to_martini_test
bash run_qm_to_martini.sh config_common/postprocess.yaml --postprocess-only
```

직접 Python으로 실행:

```bash
python -m param_opt.qm_to_martini --config config_common/common.yaml
```

`run_qm_to_martini.sh`가 기본 launcher입니다.
기본 launcher는 `environment.sh`를 source한 뒤 현재 shell 환경을 씁니다.
conda/profile/python 쪽은 `environment.sh`에서 관리하고,
xTB/ORCA/Bartender 경로는 `config_common/common.yaml`만 수정하면 됩니다.
예제 디렉터리를 다른 위치로 옮겨 쓸 때는 `HYGEL_REPO_ROOT=/path/to/hygel_martini`를
주거나, 패키지를 환경에 설치한 상태로 실행하면 됩니다.

## 현재 active 파일

- `config_common/common.yaml`: generation용 active config
- `config_common/postprocess.yaml`: collect/merge 전용 config
- `run_qm_to_martini.sh`: primary launcher
- `environment.sh`: launcher용 shell environment 설정

## 경로 정리 원칙

- 지금은 `config_common/common.yaml`을 top-level config로 실행합니다.
- source config는 상대경로와 `${CONFIG_DIR}` 기준으로 유지합니다.
- 공통 monomer/init/spin/tool/output 설정을 `config_common/common.yaml`에 모읍니다.
- workflow 선택은 `bartender_pipeline.relaxation`과 `bartender_pipeline.md`
  두 축으로 맞춥니다.
- `relaxation: xtb` 또는 `relaxation: orca`는 해당 backend로 geometry optimization을 뜻합니다.
- `md: xtb`는 새로 생성한 xTB trajectory를 Bartender에 `-owntraj/-refit`로 넘깁니다.
- `md: existing`는 `bartender_pipeline.md_traj`에 적은 기존 trajectory를 Bartender에 넘깁니다.
- `md: xtb_nobartender`는 xTB MD까지만 만들고 Bartender job은 만들지 않습니다.
- `execution.run_relaxation`, `execution.run_bartender`로 launcher 실행 시 어디까지 자동 실행할지 정합니다.
- validation/runtime 로그는 `logs/` 아래로 모이고, `logs.enabled`, `logs.write_validation`, `logs.capture_runtime`로 끌 수 있습니다.
- `md: bartender`는 Bartender 내부 xTB sampling, `md: xtb`는 pipeline 쪽 xTB trajectory reuse,
  `md: off`는 geometry optimization까지만 수행합니다.
- 빠른 실험용 변경은 `--set key.path=value`로 바로 덮어쓸 수 있습니다.
- xTB reuse 경로는 `xtb.trj`를 바로 넘기지 않고 `xtb_traj.pdb`로 변환한 뒤
  Bartender `-owntraj/-refit`에 전달합니다.
- 실제 실행하면 결과는 `runs/` 아래에 생성됩니다.
- postprocess는 기본 generation에서 꺼져 있고, `--postprocess-only`로 따로 돌립니다.

## 디렉터리 의미

- `config_common/`: generation/postprocess config
- `reference/`: 비교용 reference bank
