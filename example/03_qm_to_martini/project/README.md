# 03 QM to Martini

이 디렉터리는 `param_opt.qm_to_martini`를 실제로 돌리는 active 예시입니다.
monomer xyz/init template에서 polymer를 만들고, 선택적으로 QM/xTB geometry optimization,
xTB trajectory 생성, Bartender 후처리를 이어서 수행합니다.

## 실행

가장 먼저 돌릴 프로파일:

```bash
bash hygel_run.sh config_1/maker.yaml
```

도움말과 환경 확인:

```bash
bash hygel_run.sh --help
bash hygel_run.sh --check-xtb --check-bartender
```

다른 기본 프로파일:

```bash
bash hygel_run.sh config_2/maker.yaml
bash hygel_run.sh config_3/maker.yaml
bash hygel_run.sh config_4/maker.yaml
```

직접 Python으로 실행:

```bash
python -m param_opt.qm_to_martini --config config_1/maker.yaml
```

`hygel_run.sh`는 기본적으로
`/nas_3/active/transcendence/anaconda3/etc/profile.d/conda.sh`를 사용합니다.
다른 conda를 써야 하면 `CONDA_PROFILE=/path/to/conda.sh`로 override하면 됩니다.

## 현재 active 파일

- `config_common/common.yaml`: 공통 monomer/tool/default runtime
- `config_1/`: `relaxation=xtb`, `md=bartender`
- `config_2/`: `relaxation=orca`, `md=bartender`
- `config_3/`: `relaxation=orca`, `md=xtb`
- `config_4/`: `relaxation=off`, `md=xtb`

## 경로 정리 원칙

- 새 프로파일은 `config_N/maker.yaml`을 top-level config로 실행합니다.
- source config는 상대경로와 `${CONFIG_DIR}` 기준으로 유지합니다.
- 공통 monomer/init/spin/tool 설정은 `config_common/common.yaml`에 모읍니다.
- workflow 선택은 `bartender_pipeline.relaxation`과 `bartender_pipeline.md`
  두 축으로 맞춥니다.
- `relaxation: xtb` 또는 `relaxation: orca`는 해당 backend로 geometry optimization을 뜻합니다.
- `md: bartender`는 Bartender 내부 xTB sampling, `md: xtb`는 pipeline 쪽 xTB trajectory reuse,
  `md: off`는 geometry optimization까지만 수행합니다.
- xTB reuse 경로는 `xtb.trj`를 바로 넘기지 않고 `xtb_traj.pdb`로 변환한 뒤
  Bartender `-owntraj/-refit`에 전달합니다.
- 실제 실행하면 결과는 `runs/` 아래에 생성됩니다.

## 디렉터리 의미

- `runs/`: 생성된 case, relaxation job, Bartender job, summary
- `reference/`: 비교용 reference bank
