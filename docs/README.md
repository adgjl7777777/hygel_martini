# docs

| 문서 | 내용 | 언제 읽나 |
|---|---|---|
| [`GENERAL_FUNCTIONALITY_NETWORKS.md`](GENERAL_FUNCTIONALITY_NETWORKS.md) | f-일반 네트워크 구성: `network_layout` 설정, net별 repeat 조건, span 제약 rewiring, cyclic-topology 감사, 한계 | f=6 등 비-다이아몬드 시스템을 만들 때 가장 먼저 |
| [`DEFECTS_FOUND_AND_FIXED.md`](DEFECTS_FOUND_AND_FIXED.md) | `omni/general-ff-and-f6` 브랜치에서 발견·수정한 결함 32건과 측정으로 바뀐 주장들 | 어떤 동작이 왜 지금처럼 되어 있는지 추적할 때 |
| [`PARAMETERIZATION_PROTOCOL.md`](PARAMETERIZATION_PROTOCOL.md) | E0–E6 bonded-parameter 결정 protocol | Martini 파라미터를 확정할 때 |
| [`VALIDATION_HISTORY_AND_DESIGN_RATIONALE.md`](VALIDATION_HISTORY_AND_DESIGN_RATIONALE.md) | Series-01 A/B validation의 실패·교정·폐기 기준 | builder/분석 설계 배경이 필요할 때 |
| [`JCC_RELEASE_CHECKLIST.md`](JCC_RELEASE_CHECKLIST.md) | Series-01 원고 release 체크리스트 | Series-01 제출 작업 시 |
| [`FUNCTION_REFERENCE.md`](FUNCTION_REFERENCE.md) | 패키지 전체 함수 reference (164 모듈·77 클래스·950 함수/메서드). **source에서 생성**되며 낡으면 테스트가 실패 | 특정 함수가 무엇을 하는지 찾을 때 |
| `archive/` | 구버전 스냅샷. 현재 브랜치 상태를 반영하지 않음. `README_detailed_for_llm_series01.md`는 위 두 문서로 대체됨 | 이력 추적 시에만 |

패키지 밖이지만 함께 읽을 것:

| 문서 | 내용 |
|---|---|
| [`../README_FOR_LLM.md`](../README_FOR_LLM.md) | LLM/에이전트용 단일 진입 문서 (구조·설정·시리즈·불변식·workflow 호출 사슬·문제 위치 색인·claim boundary) |
| [`../example/08_des_thiourethane_aa/README.md`](../example/08_des_thiourethane_aa/README.md) | all-atom 예제의 정본 (원자 수·박스·밀도·근사) |
| [`../example/08_des_thiourethane_aa/validation/README.md`](../example/08_des_thiourethane_aa/validation/README.md) | 현재 force field가 DFT donor 순서를 뒤집는다는 측정과 그 범위 |
| [`../example/08_des_thiourethane_aa/sizing/README.md`](../example/08_des_thiourethane_aa/sizing/README.md) | 셀 크기 파라미터화, 조성 감사, fail-closed 실행 |
