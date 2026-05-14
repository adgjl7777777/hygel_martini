# example 안내

이 폴더는 fresh clone에서도 바로 사용할 수 있게 정리한 tracked example 입력 세트입니다.

## 포함한 것

- `run_*.sh` launcher 스크립트
- `maker.yaml`, `maker_soft_*.yaml`, `config/`, `config_*`
- monomer xyz / init input
- `structure/`, `config/`, `soft_shrinker/` 같은 실행 helper와 입력 파일

## 제외한 것

- `output/`
- `runs/`
- `constructor_output/`
- 기타 생성 JSON과 임시 산출물

## 예제 구성

- `00_bead_selector`
- `01_qm_to_opls`
- `02_opls_to_martini`
- `03_qm_to_martini`
- `04_full_builder`
- `04_1_example_system`
- `05_hydrogel_relaxation`

현재 바로 실행 가능한 example은 `03`, `04`, `04_1`, `05`입니다.
`00`, `01`, `02`는 placeholder로만 남겨뒀습니다.

## `03_qm_to_martini`에서 볼 것

`03_qm_to_martini/project`는 xTB/ORCA/Bartender 쪽 tracked example입니다.

- 새 geometry/trajectory 생성: `bash run_qm_to_martini.sh config_common/common.yaml`
- 이미 있는 xTB trajectory를 Bartender에 적용: `run_compare.sh`
- Bartender output screening: `postprocess.sh`
- C/D/S 반복 실행: `run_cds_iteration.sh`

자세한 사용법은 `03_qm_to_martini/project/README.md`에 정리되어 있습니다.
