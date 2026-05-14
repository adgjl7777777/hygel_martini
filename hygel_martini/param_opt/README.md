# param_opt

`param_opt`는 builder 전에 필요한 파라미터 준비 workflow 모음입니다.

루트 `param_opt`를 직접 실행하지 말고, 아래 workflow 모듈을 직접 실행합니다.

## workflow

- `qm_to_opls`
  01 단계. ORCA/QM 쪽 입력과 OPLS 준비물 생성
- `opls_to_martini`
  02 단계. 기존 OPLS/GROMACS trajectory를 Martini/Bartender fitting에 재사용하거나, legacy constructor 입력 생성
- `qm_to_martini`
  03 단계. QM/xTB relaxation 후 Bartender/Martini 입력 생성
- `bead_generator`
  00 단계 예정 위치
- `polymer_maker`
  monomer sequence로 polymer xyz를 만드는 공용 빌더
- `core`
  config, path, validation 같은 공통 유틸리티

각 workflow는 `defaults.py + generator.py + cli.py` 구조를 따릅니다.

## 직접 실행

```bash
python -m param_opt.qm_to_opls --config ...
python -m param_opt.opls_to_martini --config ...
python -m param_opt.qm_to_martini --config ...
```

`example/02_opls_to_martini/project`는 기존 OPLS/GROMACS data를 연결하는 tracked template입니다. 실제 production trajectory는 저장소에 없으므로 `config/opls_existing_data.yaml`의 `data/...` 경로를 사용자 데이터로 채웁니다.
`example_myrun/02_opls_to_martini`는 로컬 실험 사본을 둘 자리입니다.

### 02. Existing OPLS/GROMACS -> Martini

```bash
cd /nas_0/software_backup/hygel_martini/example/02_opls_to_martini/project
MODE=setup bash run_existing_opls.sh
MODE=md bash run_existing_opls.sh
MODE=md_notrim bash run_existing_opls.sh
```

`MODE`는 `opls_data.execution.mode`로 전달되며 trim 여부, Bartender 여부, 즉시 실행 여부를 한 번에 정합니다.

### 03. QM/xTB -> Martini

```bash
cd /nas_0/software_backup/hygel_martini/example_myrun/03_qm_to_martini/project
bash hygel_run.sh
```

## 기본 원칙

- workflow별 defaults는 각 패키지 안에만 둡니다.
- 기본 코어 수는 과한 parallel을 피하기 위해 1로 맞춰져 있습니다.
- launcher는 workflow별 진입점만 호출하고, 내부 분기 추측은 하지 않습니다.
