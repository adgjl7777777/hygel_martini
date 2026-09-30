# HyGel Builder 개선 내역과 검증 근거

작성일: 2026-09-30
대상: Series-01 Builder의 개발 작업본 `0.1.1.dev2`

## 1. 무엇을 개선했는가

이번 개선은 **설계한 연결을 실제 출력까지 확인하고, 같은 입력의 재현성을 높이며, 불필요한 구조 생성 시간을 줄이는 작업**이다. 새로운 힘장을 개발하거나 기존 물성값을 맞추기 위해 매개변수를 변경한 작업은 아니다.

앞선 `0.1.1.dev0`의 신뢰성 보강과 최근 `dev1`·`dev2`의 성능·난수 수정을 함께 정리했다. 분석 코드에서 발견한 배향 계산 문제는 별도 항목으로 설명한다.

| 구분 | 개선 전 문제 | 적용한 개선 | 직접 얻은 효과 |
|---|---|---|---|
| 연결 계획 요구 | 계획 정보가 완전히 없어지면 기존 거리 기반 경로로 넘어갈 수 있음 | 명시적 계획을 요구하는 설정 추가 | 계획 누락을 초기 단계에서 발견 |
| 저장된 결합 검사 | 메모리에서 계획을 처리했다는 사실만으로 출력 파일의 정확성을 보장하기 어려움 | 출력 ITP를 다시 읽어 계획된 atom 쌍과 비교 | 연결 누락·중복·잘못된 끝점 검출 |
| 구성 단계 설정 | `enabled: false`여도 설정 블록이 있으면 물 등을 추가할 수 있음 | 명시적 boolean 스위치를 실행 단계 선택에 반영 | 요청한 조성대로 실행 |
| 설치·배포 검증 | 소스 폴더의 파일이 누락된 배포 리소스를 대신할 수 있음 | wheel 설치 후 소스 폴더 밖에서 시험 | 실제 설치본의 동작 확인 |
| 구조 생성 성능 | 붙일 monomer가 없어도 원자별 전체 이웃 검색 수행 | monomer 선택을 먼저 확인하고 빈 경우 검색 생략 | 같은 출력을 더 짧은 시간에 생성 |
| 난수 재현성 | Python·NumPy seed만 설정하고 별도 Numba stream은 초기화하지 않음 | 직렬 Numba geometry stream도 run seed로 초기화 | 같은 seed의 독립 CLI 실행에서 출력 일치 |

## 2. 연결 계획이 없어지면 중단하도록 했다 — dev0

### 문제와 수정

Builder는 계획된 backbone 끝점과 linker의 연결을 실제 atom 결합으로 바꾼다. 계획 metadata가 일부만 존재하는 상황은 기존에도 검사했지만, 계획이 **전부 사라진 상황**은 기존 기하학적 연결 방식으로 처리될 수 있었다. 이 경우 실행은 끝나더라도 사용자가 요청한 연결을 유지했는지 판단하기 어렵다.

이를 명확히 선택할 수 있도록 다음 설정을 추가했다.

```yaml
simulation_parameters:
  require_explicit_crosslink_plan: true
```

이 설정을 켜면 계획 metadata가 없거나 필요한 linker/backbone 끝점이 없는 경우 오류로 중단한다. 값은 YAML boolean이어야 한다. 문자열 `"true"` 등은 허용하지 않는다. 설정을 생략한 기존 입력은 이전 동작을 유지한다.

### 수정 위치

- [`dynamic_crosslink.py`](../hygel_martini/hydrogel_builder/core_utils/runtime/dynamic_crosslink.py): `plan_dynamic_crosslinks()`의 `require_explicit_plan` 인자와 누락 검사.
- [`read_json.py`](../hygel_martini/hydrogel_builder/config_params/read_json.py): `_perform_dynamic_crosslinking()`에서 설정을 확인하고 해당 경로로 전달.

이는 계획 기반 연결 알고리즘을 새로 만든 것이 아니라, **기존 계획 기반 경로를 사용해야 한다는 요구를 실행 중에도 강제하는 보강**이다.

## 3. 저장된 ITP의 연결을 독립적으로 검사하도록 했다 — dev0

### 왜 전체 그래프 연결 검사만으로 충분하지 않은가

계획이 `a–b`, `c–d`인데 실제 파일에 `a–c`, `b–d`가 쓰이면, 전체 구조가 여전히 연결돼 있을 수 있다. 고리의 결합 하나가 누락돼도 다른 경로가 남아 연결 성분 수는 그대로일 수 있다. 따라서 “연결돼 있다”는 사실만으로 정확한 끝점 배정을 확인할 수 없다.

### 추가한 검사

명시적 계획 경로에서 결합을 만들기 전에 기대하는 linker stub–backbone endpoint 쌍을 **1-based ITP atom 번호**로 기록한다. 그 후 백본 ITP와 화학 확장 후 ITP를 각각 저장한 직후 다시 읽는다.

```text
기대하는 연결 쌍 확정
  → planned_crosslinks.json 저장
  → 백본 ITP 작성 → 파일을 다시 읽어 계획과 비교
  → 화학 확장 및 ITP 작성 → 파일을 다시 읽어 계획과 비교
  → 다음 준비 단계
```

검사 기준은 writer의 현재 메모리 bond 목록이 아니라 앞서 정한 연결 쌍이다. 각 계획 결합이 정확히 한 번 존재해야 하고, 등록한 stub와 endpoint 사이에 예상 밖의 연결이 없어야 한다.

검사에서는 계획 누락, 누락·중복 atom 번호, 잘못된 계획 인덱스, endpoint 재사용, 결합 누락·중복·예상 밖의 끝점 연결, 읽을 수 없는 ITP를 처리한다. 실패하면 오류 보고서를 기록하고 다음 준비 단계 전에 중단한다. 빌드 시작 시 이전 실행의 runtime 계획도 초기화해 재사용 상태가 섞이지 않게 했다.

### 수정 위치와 출력

- [`persisted_plan.py`](../hygel_martini/hydrogel_builder/core_utils/runtime/persisted_plan.py): 새 `guard_persisted_plan()` 파일 파서와 비교 검사.
- [`read_json.py`](../hygel_martini/hydrogel_builder/config_params/read_json.py): `_guard_written_crosslinks()`와 두 ITP 작성 직후 호출.
- `planned_crosslinks.json`: 기대하는 atom 쌍과 routing 기록.
- `*.plan_audit.json`: 파일 hash, 기대/관측 연결 수, 누락·예상 밖의 연결, PASS/FAIL.
- [`test_explicit_plan_guard.py`](../tests/test_explicit_plan_guard.py): 오류 주입과 후속 단계 차단 시험.

### 실제 검증

작은 PEGDA 예제에서 입자 176개, 결합 184개, 계획 연결 32개를 확인했고 백본·최종 ITP 검사를 통과했다. 연결 성분 1개, 축약 그래프의 junction 8개·strand 16개·winding rank 3도 확인했다.

후속 실제 ITP 오류 주입 검증에서는 정상 대조 3개와 오류 사례 9개 모두 GROMACS 전처리를 통과했지만, 계획 연결 검사는 오류 9개를 모두 거부했다. 비교한 축약 그래프 검사는 그중 6개를 거부했다. 이는 해당 사례에서 **전처리·그래프 검사에 더해 끝점의 정체성을 확인하는 검사가 유용하다**는 근거다. 오류 ITP로 MD를 수행한 것은 아니다.

이 검사는 등록한 연결점과 끝점 사이의 결합을 대상으로 한다. 내부 linker 결합, side-chain의 모든 화학 상호작용, 힘장 적합성, 물리적 안정성은 별도 검사 대상이다. Builder가 반환한 뒤 사람이 파일을 고치는 상황까지 자동 감시하지 않는다.

## 4. `enabled: false`가 실제로 단계를 끄도록 했다 — dev0

기존에는 `add_water` 설정 블록이 존재하면 그 안의 `enabled: false`와 무관하게 물 추가 단계가 선택될 수 있었다. 건조한 구조를 요청해도 물이 추가될 수 있는 설정 해석 문제였다.

[`read_json.py`](../hygel_martini/hydrogel_builder/config_params/read_json.py)의 `_enabled_formulation_stages()`를 추가해 다음 네 단계의 명시적 스위치를 반영했다.

- `add_water`
- `add_small_ion`
- `add_molecule`
- `add_polymer`

```yaml
add_series_parameters:
  add_water:
    enabled: false
```

이 경우 물 추가 단계가 제외된다. `enabled`를 생략하면 기존의 설정 블록·수량 해석을 유지한다. 문자열 `"false"`처럼 잘못된 값은 오류로 알리고, 그 오류가 넓은 예외 처리에 묻히지 않도록 관련 예외 처리도 좁혔다.

이 수정은 명시적으로 꺼 둔 설정과 잘못된 boolean 입력의 동작을 바로잡는다. 기존 논문 입력의 조성을 변경하거나 과거 계산을 다시 실행한 것은 아니다.

## 5. 설치한 wheel 자체를 검증하도록 했다 — dev0 이후

소스 체크아웃에서 실행하면 설치 파일에 리소스가 빠져 있어도 주변 소스 파일을 읽어 시험이 성공할 수 있다. 실제 사용자의 설치 환경과 차이가 생기는 이유다.

CI 설정을 wheel 빌드 → 별도 가상환경 설치 → 소스 폴더 밖에서 시험 실행 순서로 바꿨다. 배포 리소스를 포함하는 설정과 휴대 가능한 작은 PEGDA 구성·전처리 예제도 보강했다.

관련 파일은 [CI 설정](../.github/workflows/ci.yml), [MANIFEST.in](../MANIFEST.in), [설치 smoke test](../tests/test_packaging_smoke.py), [작은 PEGDA 예제](../example/07_portable_pegda/README.md)다.

로컬 독립 설치 환경에서 기존 63개 시험을 통과했고, 실제 로드 경로가 설치된 `site-packages/hygel_martini`임을 확인했다. 작은 예제의 GROMACS 2026.0 전처리도 `-maxwarn 0`으로 통과했다. Python 3.11/3.12 원격 CI 구성을 작성한 것과 실제 원격 Actions 실행을 마친 것은 구분한다. 여기서 확인한 실행 근거는 로컬 설치 시험이다.

## 6. monomer가 없는 경우의 불필요한 전체 원자 검색을 제거했다 — dev1

### 발견한 병목

[`Hydrogel.py`](../hygel_martini/hydrogel_builder/main_components/Hydrogel.py)의 화학 확장 단계는 side-chain 방향을 선택하기 위해 주변 원자를 검색한다. 기존 실행 순서는 다음과 같았다.

```python
# 개념적 실행 순서
nearby_atoms = scan_all_atoms(...)
chosen_template = iterator.next()
if chosen_template is None:
    continue
```

PEG처럼 해당 단계에서 추가할 monomer가 없는 경우에도 원자별로 전체 원자 목록을 훑고 나서야 아무것도 붙이지 않는다는 사실을 확인했다. 많은 원자를 대상으로 전체 목록 검색이 반복되므로 큰 구조에서 비용이 크게 늘어나는 경로였다.

### 적용한 수정

monomer를 선택하는 `iterator.next()`와 `None` 판정을 전체 원자 검색 앞으로 옮겼다. 수정 위치는 현재 `Hydrogel.py`의 약 850행이다.

```python
# 개념적 실행 순서
chosen_template = iterator.next()
if chosen_template is None:
    continue
nearby_atoms = scan_all_atoms(...)
```

이 경로에서 monomer 선택은 World를 변경하지 않고, 생략한 주변 검색은 난수를 사용하지 않는다. 실제 monomer가 있는 경우에는 이후 배치 계산을 그대로 수행한다. 출력 의미를 바꾸지 않고 하지 않아도 되는 연산을 제거한 것이다.

### 성능과 출력 동일성

| 사례 | 기존 전체 실행 | 수정 후 전체 실행 | 검증 |
|---|---:|---:|---|
| PEG N1/L12 | 5.259초 | 3.907초 | 비교한 출력 7개 byte 동일 |
| PEG N2/L56 | 65.151초 | 17.061초 | GRO·ITP·계획·감사 출력 byte 동일 |

N2/L56의 관측 속도비는 약 **3.819배**다. 공유 호스트에서 한 번 측정한 전체 프로세스 시간이며, 여러 반복의 평균이나 모든 크기에 적용되는 배수가 아니다. 큰 N4의 기존 완결 실행 시간이 없어 N4의 속도 배수는 계산하지 않았다.

Random copolymer와 Pluronic도 난수 조건을 맞춘 비교에서 출력 동일성을 확인했다. Random의 최초 비교에서 side-chain 좌표가 달랐던 기록도 보존했고, 그 원인을 조사한 결과 아래의 Numba 난수 문제가 드러났다.

## 7. 별도 Numba 난수 stream을 run seed로 초기화했다 — dev2

### 왜 같은 seed인데 좌표가 달랐는가

기존 CLI는 `random.seed(seed)`와 `np.random.seed(seed)`를 호출했다. 그러나 side-chain geometry의 `random_normal_vector()`는 Numba로 컴파일된 함수이고, 그 안의 난수 상태는 Python에서 설정한 NumPy 난수 상태와 별개였다.

따라서 Python·NumPy의 seed가 같고 backbone이 같아도, 별도 CLI 프로세스에서 side-chain 좌표가 달라질 수 있었다. 처음에는 성능 패치 검증용 wrapper에서 양쪽 Numba stream을 명시적으로 맞춰 의미 보존을 확인했다. 이후 실제 CLI에도 초기화가 적용되도록 별도의 수정을 검증했다.

### 코드 변경

[`utility.py`](../hygel_martini/hydrogel_builder/core_utils/common/utility.py)에 다음 함수를 추가했다.

```python
@numba.njit(cache=True)
def seed_numba_random(seed):
    np.random.seed(seed)
```

[`build_hydrogel.py`](../hygel_martini/hydrogel_builder/config_params/build_hydrogel.py)의 `_seed_random_generators()`에서 기존 Python·NumPy 초기화 뒤 이 함수를 호출한다.

```python
random.seed(seed_val)
np.random.seed(seed_val)
seed_numba_random(seed_val)
```

Compiled stream 초기화가 Python·NumPy의 난수 draw를 소비하지 않도록 했다. 기존의 첫 다섯 draw가 유지되는지 별도로 확인했다.

### 실제 검증

- 설치 wheel의 기존 시험 **63개 통과**.
- 소스 폴더 밖에서 실제 CLI 프로세스 **17회** 실행.
- 동일 seed의 **8쌍**에서 저장된 GRO·ITP 및 제공되는 계획·감사 출력이 byte 단위로 일치.
- PEG, Pluronic, random·alternating·block 구성을 포함.
- 다른 seed의 random 사례에서는 GRO가 달라짐을 확인.
- 직렬 compiled geometry vector가 동일 seed에서 반복되고, 다른 seed에서 달라짐을 확인.
- 두 alias가 같은 DMAPS GRO/ITP를 사용하는 사례로 sequence 선택도 검사. Random 81/65, alternating 73/73, block 59/87의 관측 선택 수가 독립 기대값과 일치.

Alias 시험은 같은 화학 입력으로 sequence 선택 경로를 구분한 시험이다. 두 서로 다른 화학종의 물리적 거동을 검증한 결과로 해석하지 않는다.

재현성의 범위는 **확인한 직렬 호출 thread**다. 향후 병렬 worker별 난수 stream까지 보장하는 구현은 아니다. Sequence iterator의 `SEQUENCE_STRATEGY.seed`는 별도로 명시해야 하며, CLI run seed 하나가 모든 sequence 설정을 대신하지 않는다.

## 8. 배향 분석도 수정했지만 Builder 변경과는 구분한다

후처리 `structure_water.py`의 기존 배향 계산은 PEO residue 안에서 EO atom 번호가 연속한 쌍을 사용했다. 이 방식에는 실제 EO–EO 결합 13,696개 외에 서로 다른 strand의 **비결합 쌍 96개**가 포함돼 있었다.

실제 ITP `[bonds]`에서 양쪽 atom 이름이 EO인 결합을 선택하도록 보정하고, 채택한 12개 시스템의 최종 50 ns를 100 ps 간격으로 다시 계산했다. 총 **6,012개 프레임**이다. 새 분석 스크립트와 결과는 별도 근거 폴더에 보존했고, 원래 채택 분석 파일을 덮어쓰지 않았다.

수정된 네 상태의 세-realization 평균 배향값 범위는 **0.00671747–0.00682924**다. 국소 segment orientation이 거의 등방적이라는 기존 해석은 유지됐다. 기존 방식의 재계산 오차는 0이고, 별도 tensor 계산의 최대 차이는 약 3.17×10⁻¹⁵ 이하다.

이는 Builder가 잘못된 결합을 만들었다는 결과가 아니라, **분석이 결합 쌍을 정의하던 방식을 바로잡은 것**이다. 본문 숫자·배향 그림 패널·SI 정의와 비교표를 업데이트했다. 이번 보정에서 RDF·MSD·기계적 응답·Fourier 분석을 재계산한 것은 아니다.

## 9. 현재 어떤 코드에 적용돼 있는가

| 대상 | 상태 |
|---|---|
| 기존 Series-01 frozen 저장소 | `/nas_0/software_backup/hygel_martini`, 기존 `0.1.0` 보존 |
| 개발 작업본 | `/nas_0/software_backup/hygel_builder_reliability_20260929`, 현재 `0.1.1.dev2` |
| dev0 연결·설정·패키징 보강 | commit `50192afbd656dfce72b84ad66c3bfedec7796230`에 기록 |
| 최근 성능·난수·버전 수정 | 위 commit 이후 작업 트리의 tracked 파일 5개, 아직 미커밋 |
| 코드 채택 검증 | 변경 파일 5개의 hash가 검증된 dev2 후보와 일치 |
| 기존·실행 중인 설치 환경 | 이번 개발 소스 채택으로 교체하지 않음 |
| 공개 release·push·tag·제출 | 이번 작업에서 수행하지 않음 |

최근 tracked 변경 파일은 `Hydrogel.py`, `build_hydrogel.py`, `utility.py`, `hygel_martini/__init__.py`, `tests/test_packaging_smoke.py`다. 마지막 두 파일은 개발 버전 표기와 그 표기를 확인하는 기존 시험의 기대값 변경이다. 최근 diff는 5개 파일, 22행 추가·6행 삭제다. 이것은 dev0부터의 전체 누적 변경량이 아니라 **dev0 commit 이후의 diff**다.

최근 변경 전 파일과 hash를 백업한 뒤 후보와 동일한 코드를 적용했다. 실행 중인 N4의 별도 dev1 설치 환경과 Q8/Q9의 고정 MD 입력을 자동으로 바꾸지 않았다. 과거 논문 trajectory를 dev2로 생성한 것처럼 소급해서 기술하지 않는다.

## 10. 연구와 논문에서 강조할 수 있는 성과

이번 보강으로 다음 내용을 구체적인 검증 근거와 함께 설명할 수 있다.

1. **설계한 연결을 출력까지 확인한다.** 연결 성분 수뿐 아니라 실제 atom 끝점의 정체성을 검사하며, 실패하면 후속 준비 전에 멈춘다.
2. **동일 입력의 구조 재현성을 실제 CLI로 확인했다.** 설치 후보의 독립 프로세스 비교로 좌표·topology 일치를 입증했다.
3. **물리 모델을 바꾸지 않고 생성 비용을 줄였다.** 한 PEG 사례에서 출력 동일성과 실행 시간 감소를 함께 확인했다.
4. **설치본과 실행 근거를 함께 제공한다.** 작은 재현 예제, 설치 시험, 계획·감사 JSON, 입력·출력 hash를 연결한다.

이 성과는 구조 생성 소프트웨어의 신뢰성과 사용성을 직접 높인다. 추가 MD의 평형·통계 오차·실험 비교는 그 구조에서 얻은 물성을 평가하는 다음 단계이며, 여기의 소프트웨어 시험으로 대신하지 않는다.

## 11. 원본 근거와 상세 기록

- [dev0의 정확한 동작 범위](EXPLICIT_PLAN_RELIABILITY.md)
- [dev0 설치·예제 검증 기록](../validation_20260929/VERIFICATION.json)
- [실제 ITP 오류 주입 예제](/nas_0/software_backup/hygel_builder_reliability_20260929/validation_20260930_orientation_repair/paper_revision/si_files/all_review_20260930/)
- [성능 개선 후보 기록](/nas_0/software_backup/hygel_builder_reliability_runs_20260930_all/builder_performance_followup/HANDOFF_KO.md)
- [성능 수정 patch](/nas_0/software_backup/hygel_builder_reliability_runs_20260930_all/builder_performance_followup/algorithm.patch)
- [난수 재현성 결과](/nas_0/software_backup/hygel_builder_reliability_runs_20260930_all/builder_performance_followup/rng_seed_followup/si_ready/RESULT.json)
- [Numba·Python·NumPy stream 확인](/nas_0/software_backup/hygel_builder_reliability_runs_20260930_all/builder_performance_followup/rng_seed_followup/si_ready/RNG_STREAM_CHECK.json)
- [개발 작업본 채택 hash 검증](/nas_0/software_backup/hygel_builder_reliability_runs_20260930_all/builder_performance_followup/adoption_20260930/ADOPTION_VERIFICATION.json)
- [배향 보정 근거](/nas_0/software_backup/hygel_builder_reliability_20260929/validation_20260930_orientation_repair/paper_revision/si_files/orientation_bond_correction/README_KO.md)
- [최신 원고와 계산 인계](/nas_0/software_backup/hygel_builder_reliability_20260929/validation_20260930_orientation_repair/HANDOFF_KO.md)

과거 인계 문서의 “후보를 별도 복사본에서 검증했고 개발 작업본을 변경하지 않았다”는 설명은 그 검증 시점의 기록이다. 이후 개발 작업본에 채택한 단계는 `ADOPTION_VERIFICATION.json`에 따로 기록했다. 시점이 다른 두 기록을 혼동하지 않아야 한다.
