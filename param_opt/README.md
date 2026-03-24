# param_opt

`param_opt`는 builder 전에 필요한 파라미터 준비 workflow 모음입니다.

루트 `param_opt`를 직접 실행하지 말고, 아래 workflow 모듈을 직접 실행합니다.

## workflow

- `qm_to_opls`
  01 단계. ORCA/QM 쪽 입력과 OPLS 준비물 생성
- `opls_to_martini`
  02 단계. OPLS 기반에서 Martini constructor 쪽 입력 생성
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

`example_myrun/01_qm_to_opls`와 `example_myrun/02_opls_to_martini`는 현재 placeholder만 남겨둔 상태입니다.
따라서 지금은 Python 모듈 직접 실행 또는 별도 config 준비가 기준입니다.

### 03. QM/xTB -> Martini

```bash
cd /nas_0/software_backup/hygel_martini/example_myrun/03_qm_to_martini/project
bash hygel_run.sh config_1/maker.yaml
```

## 기본 원칙

- workflow별 defaults는 각 패키지 안에만 둡니다.
- 기본 코어 수는 과한 parallel을 피하기 위해 1로 맞춰져 있습니다.
- launcher는 workflow별 진입점만 호출하고, 내부 분기 추측은 하지 않습니다.
