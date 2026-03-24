# example 안내

이 폴더는 `example_myrun/`에서 생성물과 작업 메모를 제거한 배포용 기준 입력 세트입니다.

## 포함한 것

- launcher 스크립트
- `maker.yaml`, `maker_soft_*.yaml`, `config/`, `config_*`
- monomer xyz / init input
- `structure/`, `soft_em/`, `soft_md/`, `soft_shrinker/` 같은 실행 필수 helper

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
