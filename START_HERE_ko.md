# 시작 가이드

처음 보는 사람 기준으로는 아래 순서가 가장 쉽습니다.

## 1. 가장 먼저 돌릴 것

`04_full_builder`부터 확인합니다.

```bash
cd /nas_0/software_backup/hygel_martini/example_myrun/04_full_builder/project
bash hygel_run.sh
```

이 단계는 `hydrogel_builder`가 전체적으로 살아 있는지 가장 빨리 확인하는 용도입니다.

## 2. 실제 예시 시스템을 만들 때

`04_1_example_system`을 봅니다.

```bash
cd /nas_0/software_backup/hygel_martini/example_myrun/04_1_example_system/project
bash hygel_run.sh
```

## 3. build 뒤 추가 완화가 필요할 때

`05_hydrogel_relaxation`을 봅니다.

```bash
cd /nas_0/software_backup/hygel_martini/example_myrun/05_hydrogel_relaxation/project
bash hygel_run.sh maker_soft_em.yaml
bash hygel_run.sh maker_soft_md.yaml
```

## 4. xTB/ORCA/Bartender가 목적일 때

`03` 예시를 봅니다.

```bash
cd /nas_0/software_backup/hygel_martini/example_myrun/03_qm_to_martini/project
bash hygel_run.sh
```

환경만 먼저 확인하고 싶으면:

```bash
bash hygel_run.sh --check-xtb --check-bartender
```

`md: off`면 geometry optimization까지만 진행하고 Bartender/MD는 생략됩니다.

## 5. 단계별 의미

- `00`
  bead selector 예정 위치
- `01`
  ORCA/QM -> OPLS
- `02`
  OPLS -> Martini
- `03`
  QM/xTB -> Martini

지금 `example_myrun/00`, `01`, `02`는 placeholder이고, 실제 ready-to-run example은 `03`, `04`, `04_1`, `05`입니다.

## 6. 공통 규칙

- 실제 실행은 각 예제의 `project/` 디렉터리에서 시작합니다.
- 가능하면 Python 모듈을 직접 치기보다 `gemini.sh` 또는 `hygel_run.sh`를 먼저 사용합니다.
- `example/`보다 `example_myrun/`이 현재 작업 기준입니다.
