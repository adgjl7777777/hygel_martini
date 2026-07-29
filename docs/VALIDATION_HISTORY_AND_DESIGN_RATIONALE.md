# Validation history and design rationale

이 문서는 `hygel_martini`의 현재 코드와 분석 경계가 왜 그렇게
설계되었는지를 설명합니다. 성공한 최종 수치만 모은 결과표가 아니라,
과거 validation에서 발견된 실패 원인, 폐기된 해석, 후속 교정, 현재
패키지에 남긴 안전장치를 연결한 설계 기록입니다.

## 문서 감사 범위

2026-07-29에 local run workspace의 Markdown을 중심으로 다음 범위를
검토했습니다.

| 범위 | Markdown 수 | 역할 |
|---|---:|---|
| `example_myrun/A_validation_old` | 220 | 초기 PEG tuner, prototype property analysis, builder 실패와 판단 변경의 역사 |
| `example_myrun/B_validation` | 228 | topology 교정 후 재분석, provenance audit, corrected campaign |
| `example_myrun/04_2_validation` | 1 | 초기 통합 builder 사용법과 설정 설명 |
| `example_myrun/paper` | 7 | 최종 Series-01 claim, SI, 마감 판정 |

위 네 범위의 456개 파일은 SHA-256 기준 365개의 고유 내용으로
압축되었습니다. 91개는 archive 복사본이나 A/B 사이의 동일 문서였습니다.
`example_myrun` 전체에는 721개 Markdown이 있었지만, 발표·poster·별도
QM-to-Martini 작업과 같은 다른 기록은 이 builder/property 설계 근거에
섞지 않았습니다.

개별 threshold run의 짧은 README를 독립적인 scientific replicate로
세지 않았습니다. 예를 들어 corrected pore 폴더의 수십 개 README는 한
parameter sweep의 실행 메타데이터이며, 결론은 setup summary, result
summary, definition audit를 함께 읽어 판정했습니다.

`example_myrun`은 `.gitignore`된 local provenance workspace입니다. 아래
경로들은 개발 이력을 추적하기 위한 source-family 이름이지, public
package가 NAS 경로나 해당 trajectory에 의존한다는 뜻이 아닙니다.

## 증거 우선순위

디렉터리 이름에 `validation`이나 `paper_ready`가 들어 있어도 결과가
자동으로 유효해지지 않습니다. 서로 충돌하는 기록은 다음 순서로
판정했습니다.

1. 최종 `paper/prompt.md`, `paper/remaining.md`, active manuscript/SI와
   frozen analysis record
2. exact topology, 완료 marker, 분석 정의와 statistical unit가 확인된
   corrected B-validation 결과
3. A-validation의 진행 기록과 원인 분석
4. `_archive`, `wrong`, 오래된 `paper_ready` 자료

각 결과에는 최소한 다음 여섯 항목이 필요합니다.

1. 사용한 topology와 include/parameter layer
2. pre-correction 또는 corrected 여부
3. trajectory 정상 완료와 분석 window
4. observable의 정확한 수학적 정의
5. uncertainty를 계산한 statistical unit
6. 그 증거로 허용되는 claim ceiling

이 중 하나라도 복구되지 않으면 그 결과는 역사 또는 diagnostic일 수는
있지만 현재 정량 validation으로 승격하지 않습니다.

## 진행 과정과 판단이 바뀐 이유

### 1. Builder smoke test에서 materialized network로

초기 예제는 작은 PEO chain이나 희박한 one-cell smoke system을 통해
workflow가 끝까지 실행되는지를 확인했습니다. 이 단계의 성공은 CLI,
파일 생성, GROMACS 호출을 확인했지만, 고밀도 hydrogel의 topology와
packing을 검증하지는 못했습니다.

L110 PEGDA-like system으로 확장하면서 다음 문제가 드러났습니다.

- linker/backbone 배치 뒤 stretched bond와 PBC shift warning
- 큰 box에서 Packmol의 기본 `sidemax` 때문에 물이 box 일부에만 채워짐
- base structure를 centroid로 다시 이동해 이미 정한 좌표가 어긋남
- box header만 줄이고 atom coordinate를 같이 scale하지 않아 PBC
  반대편에서 충돌
- 큰 압력에 10%씩 반응한 box가 진동하거나 비등방으로 변형
- GPU 자동 선택과 OpenMP/MPI 자원 사용이 설정과 다르게 작동

이 기록 때문에 현재 package는 box와 coordinate를 함께 affine scale하고,
PBC wrap을 safety net으로 적용하며, Packmol에 명시적 fixed origin과
box-dependent `sidemax`를 전달합니다. Relaxation은 GPU, OpenMP thread,
MPI rank를 YAML에서 분리하고, 실패한 shrink 후보를 그대로 다음 단계로
넘기지 않습니다.

주요 역사 문서:

- `A_validation_old/project_peg/docs/md_notes/old/코드_수정_이력.md`
- `A_validation_old/project_peg/docs/md_notes/old/l110_build_step.md`
- `A_validation_old/wrong/_diagnostics/hygel_builder_overlap_20260625/`

### 2. Nearest-end repair에서 graph-first local matching으로

초기 dynamic-crosslink 시도는 이미 배치된 BCK가 가까운 backbone end를
runtime에서 탐색하도록 했습니다. 희박한 one-cell smoke에서는 충돌 없이
통과했지만, multi-cell 진단에서는 많은 초기 crosslink가 길게 늘어나고
closed network에서 `inconsistent shifts` warning이 남았습니다.

원인 분석의 핵심은 전역 x/y/z 개수보다 local assignment와 PBC image
일관성이었습니다. 한 local diamond vertex의 네 endpoint에는 정확히 세
개의 perfect matching이 있습니다.

```text
x: (000-011), (101-110)
y: (000-101), (011-110)
z: (000-110), (011-101)
```

따라서 현재 설계는 runtime nearest-end search로 graph를 뒤늦게
수리하는 대신, layout 단계에서 x/y/z local transition을 먼저 정합니다.
`local_matching.py`는 endpoint graph의 component와 degree를 평가하고,
가능하면 하나의 self-returning periodic graph와 균형 잡힌 axis 사용을
선택합니다. Dynamic-crosslink 단계는 그 materialized geometry에서
endpoint uniqueness와 BCK attachment를 수행합니다.

여기서 BCK 의미도 명시적으로 고정했습니다.

```text
BCK1: polymer A -- polymer B
BCK2: polymer C -- polymer D
```

한 two-stub linker는 네 개의 BCK-backbone bond와 두 개의 local polymer
junction을 만듭니다. BCK1-BCK2 내부 bond는 template/topology에
명시되었을 때만 존재합니다. Linker 한 개를 junction 한 개로 세거나,
chain reuse를 허용해 연결 수만 맞추는 구현은 이 불변식과 다릅니다.

주요 역사 문서:

- `A_validation_old/total.md`의 2026-06-24 local matching 기록
- `paper/graph_transition_matching_algorithm_note_20260624.md`
- `A_validation_old/wrong/_diagnostics/hygel_builder_dense_reproducer_20260625_codex/`

### 3. Connected graph만으로 충분하지 않았던 이유

초기 topology 검사는 atom graph가 한 component인지 주로 확인했습니다.
그러나 후속 감사에서 다음이 서로 다른 문제임이 확인되었습니다.

- atom-level connectivity
- junction 수와 degree
- junction 사이의 valid strand 수
- dangling/bridge/loop defect
- cycle rank
- periodic winding rank
- coordinate와 bond의 PBC image 일관성
- materialized angle/dihedral parameter completeness

또한 100,000개가 넘는 atom을 가진 GRO의 5자리 serial field는 wrap될 수
있습니다. Serial을 dictionary key로 사용한 옛 분석은 2x2x2 bond를
30 nm 이상으로 잘못 계산했습니다. 현재 topology analyzer는 topology
순서와 GRO line sequence로 좌표를 대응하고, fixed-width serial을 atom
identity로 신뢰하지 않습니다.

단순히 theoretical count와 actual count가 같다는 사실도 parameter가
맞다는 충분조건은 아닙니다. Corrected campaign에서는 atom index를
제거한 뒤 atom/bond/angle/dihedral parameter distribution을 reference와
비교하고, BCK mass, junction bond, water count, graph, geometry,
`grompp`를 각각 gate로 확인했습니다. 이전 checkpoint를 다른 bonded
topology에 이어 붙이는 것도 금지했습니다.

현재 public analyzer가 제공하는 cycle rank와 winding rank는 construction
audit입니다. 이것만으로 모든 strand가 elastically active하다거나
force field가 실험을 재현한다고 말하지 않습니다.

주요 역사 문서:

- `B_validation/peg/network_topology_analysis/network_topology_audit_fix.md`
- `B_validation/peg/network_topology_analysis/network_topology_summary.md`
- `B_validation/peg/independent_seed101_corrected_topology_v2_20260715/`
- `paper/validation_5fs_topology_report.md`

### 4. Topology parameter precedence 교정

Legacy PEG 결과 일부에는 BCK mass 54와 오래된 junction bond/parameter
경로가 남아 있었습니다. 다른 문단에서 corrected topology를 설명하면서
그 legacy trajectory의 mechanics나 heterogeneity를 함께 쓰는 provenance
모순도 발견되었습니다.

현재 builder는 `bonded_topology_patch_file`을 통해 intended bonded
definition을 명시적으로 materialize할 수 있습니다. Corrected campaign은
BCK mass 45, BCK-PEO/BCK-BCK bonded parameters, 전체 angle/dihedral
distribution과 GROMACS preflight를 함께 확인했습니다. Parameter patch가
누락되면 기존 값으로 조용히 fallback한 것으로 간주하지 않습니다.

이 교정 전에 생성된 trajectory는 안정적으로 보이더라도 corrected
topology 결과로 재명명하지 않습니다. 특히 old TPR/CPT에서 corrected
campaign을 resume하지 않습니다.

### 5. Fixed-water loading은 free swelling이 아님

초기 A-validation은 water-loading sweep에서 Qm9p0의 bead-volume
fraction이 실험에서 변환한 값과 가까운 것을 `swelling match`로
표현했습니다. 후속 state-point audit에서 이 해석은 수정되었습니다.

각 simulation의 물 분자 수는 고정되어 있습니다. NPT는 box volume을
바꾸지만 solvent reservoir와 물을 교환하지 않으므로 equilibrium water
uptake를 예측하지 않습니다. 또한 다음 세 정의는 같은 값이 아닙니다.

- bead nominal volume을 사용한 polymer fraction
- dry polymer mass/density와 simulated box를 사용한 fraction
- input mass-swelling ratio에서 변환한 composition fraction

Qm8p2는 실험 mass loading을 구성상 그대로 입력한 state이고, Qm9p0은
bead-volume 또는 dry-mass/box scale을 target 근처로 옮기기 위해 더 많은
물을 넣은 calibration state입니다. 정의 간 차이가 trajectory block
uncertainty보다 컸기 때문에, 현재 패키지는 다음 용어를 사용합니다.

- `imposed composition`
- `fixed-water state point`
- `hydration bracket`

`free swelling`, `equilibrium uptake prediction`, 또는 phi에서 다시
환산한 값을 독립적인 Qm prediction으로 부르지 않습니다.

주요 역사 문서:

- `A_validation_old/validation_atlas.md`
- `B_validation/paper_ready/manuscript_review_and_execution_plan_20260710.md`
- `B_validation/paper_ready/reanalysis_20260710/statepoint_definitions/`

### 6. Target-fitted pore에서 definition-bounded clearance로

Pore analysis는 가장 큰 해석 변경을 겪었습니다.

1. polymer-only CG PoreBlazer peak는 atomistic target보다 훨씬 컸습니다.
2. 모든 water를 occupied volume으로 넣으면 반대로 accessible region이
   거의 닫혔습니다.
3. residence-ranked water의 선택 fraction을 조절하면 target 부근을
   통과할 수 있었습니다.
4. 그러나 그 fraction은 target을 보고 선택했으므로 독립 validation이
   아니라 operational-definition sensitivity였습니다.
5. 여섯 frame은 시간적으로 상관된 sensitivity series이지 bootstrap
   replicate가 아니었습니다.

따라서 public package에는 target을 향해 selected-water fraction을 맞추는
도구를 중심 API로 넣지 않았습니다. 대신 obstacle selection, bead radius,
probe radius, grid spacing을 모두 명시하는 mixed-radius periodic
clearance를 제공합니다.

다음은 서로 다른 observable입니다.

- polymer-only clearance
- polymer + predefined interfacial-water clearance
- polymer + all-water clearance
- PoreBlazer PLD/LCD/PSD
- experimental mesh length

`2π/q`, local clearance median, PoreBlazer peak를 자동으로 하나의
`pore size`로 동일시하지 않습니다. 여러 사전 정의 표현에서 hydration
ordering이 유지되는지는 robust bracket으로 논의할 수 있지만, unique
mesh length를 주장할 수는 없습니다.

주요 역사 문서:

- `B_validation/peg/pore_analysis/pore_definition_guardrails.md`
- `B_validation/peg/pore_analysis_corrected_qm9p0/`
- `B_validation/paper_ready/reanalysis_20260710/pore_definition_audit/`
- `paper/prompt.md`

### 7. Water displacement의 PBC 처리

옛 radial mobility 계산은 긴 lag의 endpoint displacement에 한 번만
minimum-image convention을 적용했습니다. 입자가 box를 여러 번 건너면
이 방식은 이동을 잃으므로 diffusion 계산으로 유효하지 않습니다.

교정 분석은 저장된 매 frame에서 displacement를 누적해 trajectory를
unwrap하고, 여러 time origin과 명시적 fit window를 사용했습니다. 이후
GROMACS cross-check와 같은 water model/temperature의 pure-water control을
추가해 normalized mobility를 만들었습니다.

현재 `diffusion.py`는 이 이유로 trajectory unwrapping, multi-origin MSD,
fit을 분리합니다. 호출자는 다음을 provenance에 남겨야 합니다.

- water model과 temperature
- coordinate wrapping 상태
- time-origin 간격
- lag와 fit window
- selection이 origin마다 다시 정해지는지 여부
- bulk/matched-water normalization

한 network trajectory의 time origin 여러 개는 independent network
replicate가 아닙니다.

### 8. Mechanics 구현과 claim boundary

초기 mechanics 기록에는 direct compression, five-point deformation,
dense strain sweep, tensile rate sweep, log-rate extrapolation 등 여러
prototype가 있습니다. 이 중 일부는 topology 교정 전 trajectory였고,
일부는 nonlinear point, overlap, axis dependence, protocol mismatch를
포함했습니다. `proxy`라는 이름만 붙여 provenance 모순을 해결할 수
없으므로 최종 정량 결과에서는 제외했습니다.

최종 PEG workflow는 같은 origin에서 baseline, positive, negative branch를
짝지어 GROMACS pressure의 odd component를 계산합니다.

```text
P_odd = (P_plus - P_minus) / 2
G_apparent = -P_odd / gamma
```

이 설계는 common-mode baseline과 even nonlinear contamination을
분리합니다. Time grid, legend, component가 다르면 비교하지 않습니다.
여러 origin과 plane을 먼저 realization 내부에서 평균한 뒤, seed101,
seed202, seed303 realization에 같은 가중치를 줍니다. Origin 372개를
independent network 372개로 풀링하지 않습니다.

최종 paper에서 승격된 PEG 값은 corrected 세 realization의 1--5 ps
paired-step high-rate apparent shear response입니다.

| imposed state | equal-realization mean ± realization SD |
|---|---:|
| Qm8p2 | 120.533 ± 1.998 MPa |
| Qm9p0 | 119.482 ± 2.171 MPa |

차이의 방향이 realization/plane별로 일관되지 않아 일반적인 hydration
stiffening/softening은 주장하지 않습니다.

Frozen implemented Pluronic A1은 한 coordinate state의 세 stochastic
100 ps-ramp repeat에서 134.353 ± 28.656 MPa를 보였습니다. 이는 independent
network uncertainty가 아닙니다. Common 1 ns ramp에서는 PEG-water
separation과 A1 SNR promotion gate가 통과되지 않아 rate/resolution
boundary로 남았습니다.

따라서 패키지의 mechanics 결과는 다음 중 어느 것도 자동으로 뜻하지
않습니다.

- equilibrium 또는 plateau modulus
- storage/loss modulus
- zero-frequency response
- experimental rheometer/UTM modulus
- 서로 다른 Martini generation 사이의 material ranking

`mechanics_analysis.py`가 registered window, paired ramp, cycle block,
realization weighting, Holm correction을 따로 제공하는 이유가 여기에
있습니다.

### 9. Pluronic은 second chemistry implementation test

A-validation 중간에는 standalone Pluronic generator, polyply-only box,
fixed-box M2/M3 비교가 `hygel_builder` validation처럼 취급된 시기가
있었습니다. 이 경로들은 builder를 우회하므로 그 성공이나 실패가
builder evidence가 될 수 없어 `wrong`으로 격리되었습니다.

최종 Series-01은 새 parameter fitting이나 새 crosslinked model을
승격하지 않고, 기존 frozen M2-A1 topology와 700 ns trajectory를
implemented-model application benchmark로 사용합니다. A1은 다른
Martini generation과 water model을 사용하고 matched pure-water control과
independent network replicate가 없으므로, PEG와 절대 transport/stiffness
순위를 만들지 않습니다.

실패한 10 fs/Berendsen trajectory, exact-Nawaz rescue chronology,
standalone free-polymer 결과는 현재 A1 hydrogel 정량 claim의 대체물이
아닙니다.

## 폐기·보류 결과 ledger

| 과거 자료 또는 표현 | 현재 상태 | 이유 |
|---|---|---|
| 작은 PEO/one-cell smoke 성공 | workflow smoke only | 고밀도 packing, PBC network, property validity를 검증하지 않음 |
| compact L110 baseline | historical stable baseline | experimental target보다 under-swollen/too compact; 현재 corrected cohort가 아님 |
| standalone Pluronic/polyply/Packmol model | rejected as builder validation | `hygel_builder` construction path를 우회 |
| pre-correction PEG mechanics/heterogeneity | excluded from final quantitative claim | BCK mass/bonded topology provenance 불일치 |
| old seed101 checkpoint continuation | diagnostic only | corrected reference와 full materialized topology가 다름 |
| direct 50% compression | failed diagnostic | overlap/geometry failure, linear response가 아님 |
| selected-water target crossing | calibrated sensitivity only | target을 보고 fraction을 선택 |
| six-frame pore “bootstrap” | temporal sensitivity | correlated frames이며 independent resample이 아님 |
| endpoint-only minimum-image MSD | invalid algorithm | multiple PBC crossing displacement를 잃음 |
| 2x2x2 replicated cell | remap/dynamic stability check | 한 network의 periodic copy이며 independent seed가 아님 |
| origin/plane/cycle pooling | within-realization sampling | independent network uncertainty가 아님 |
| common 1 ns ramp | unresolved rate boundary | predefined separation/SNR gate 미통과 |
| old Pluronic 10 fs/Berendsen runs | excluded | pressure-scaling failure와 trajectory integrity 문제 |

이 ledger는 실패를 숨기기 위한 목록이 아닙니다. Package 사용자가 같은
분석 오류를 반복하거나 오래된 숫자를 더 강한 물성명으로 재사용하지
않도록 하는 promotion rule입니다.

## 현재 코드에 남은 설계 추적성

| 역사에서 확인된 문제 | 현재 구현 | 확인 지점 |
|---|---|---|
| x/y/z local junction 선택과 disconnected graph | graph-first local perfect matching | `hydrogel_builder/core_utils/layout/local_matching.py` |
| runtime crosslink 중 endpoint 중복/chain reuse | global endpoint uniqueness와 explicit `targets_per_stub` | `hydrogel_builder/core_utils/runtime/dynamic_crosslink.py` |
| corrected bonded parameter가 output에 반영되지 않을 위험 | explicit topology patch materialization | `hydrogel_builder/config_params/read_json.py` |
| Packmol 큰 box의 water half-fill | origin-preserving fixed structure와 dynamic `sidemax` | `hydrogel_builder/core_utils/runtime/packer.py` |
| box header만 변경해 생긴 PBC clash | coordinate+box affine scaling, PBC wrap | `hydrogel_builder/relax/soft_em.py`, `hard_em_shrink.py` |
| 실패한 shrink가 다음 단계로 전파 | finite-coordinate/energy/Fmax gate, rollback, step halving | `hydrogel_builder/relax/hard_em_shrink.py` |
| GPU 자동 점유와 thread ambiguity | YAML-controlled GPU/OpenMP/MPI command building | `hydrogel_builder/relax/soft_em.py`, `soft_md.py` |
| GRO 5-digit serial overflow | sequential GRO coordinate parsing | `property_extract/network_topology.py` |
| connected 여부만 본 topology 판정 | junction–strand graph, defect, cycle, winding, hash | `property_extract/network_topology.py` |
| fixed-water를 free swelling으로 오해 | imposed-composition naming과 explicit result role | `property_extract/swelling.py`, `result.py` |
| target-directed pore fitting | definition-explicit periodic clearance | `property_extract/pore_size.py`, `extractors/clearance.py` |
| wrapped endpoint MSD | cumulative unwrap와 multi-origin MSD | `property_extract/diffusion.py` |
| rate-dependent stress를 equilibrium modulus로 오해 | paired finite-rate response와 registered-window API | `property_extract/mechanics.py`, `mechanics_analysis.py` |
| origin을 replicate로 pooling | equal-realization summary와 multiplicity-aware tests | `property_extract/mechanics_analysis.py` |
| missing input에도 분석이 진행 | requirement gate와 explicit failure status | `property_extract/requirements.py`, `analysis_jobs.py` |

## 권장 validation 순서

분석을 많이 실행하는 것보다 promotion 순서를 지키는 것이 중요합니다.

1. **Configuration provenance**
   - resolved maker/include path와 parameter patch를 보존합니다.
2. **Materialized topology**
   - atom/bond/angle/dihedral, mass, water count, graph fingerprint를
     확인합니다.
3. **Coordinate and PBC health**
   - finite coordinate/box, MIC bond, overlap, winding을 확인합니다.
4. **GROMACS preflight and completion**
   - `grompp`, normal termination, expected final time와 output integrity를
     확인합니다.
5. **Registered state window**
   - temperature, pressure, density, volume drift와 imposed composition을
     명시합니다.
6. **Definition-bounded structure/transport**
   - selection, radii, grid, PBC unwrap, fit window와 control을 기록합니다.
7. **Finite-rate mechanics**
   - paired sign, time-zero, registered window, direction/anisotropy gate를
     확인합니다.
8. **Statistical promotion**
   - block/origin/repeat/network realization을 구분하고 claim ceiling을
     결정합니다.

`hygel-property requirements`와 manifest-driven analysis는 이 순서를
자동으로 모두 증명하지는 않지만, 입력 누락과 정의 없는 결과 생성을
초기에 차단하는 공통 interface를 제공합니다.

## 최종 Series-01 결론과 패키지 범위

최종 논문에서 닫힌 결론은 다음과 같습니다.

> `hygel_builder`는 PEGDA와 frozen implemented Pluronic A1이라는 서로
> 다른 coarse-grained hydrogel을 graph-first 방식으로 materialize하고,
> topology/composition/PBC gate로 감사하며, 검증된 구조를
> state--structure--transport--finite-rate mechanics의 공통 property
> interface까지 연결한다.

따라서 이 package의 성과는 구조 생성 코드만 있거나 deformation input만
실행되는 상태가 아닙니다. Construction부터 provenance-aware property
analysis까지 이어지는 구현과 rejection gate가 갖춰졌다는 것입니다.

동시에 다음은 현재 package나 Series-01의 완결 주장이 아닙니다.

- 보편적으로 calibration된 PEGDA/Pluronic force field
- solvent reservoir를 사용한 free-swelling prediction
- 정의와 무관한 unique pore/mesh size
- equilibrium 또는 experimental rheology reproduction
- PEG와 A1 사이의 force-field-independent material ranking

이 경계는 기능 부족을 숨기는 caveat가 아니라, 계산 가능한 observable과
검증 가능한 physical claim을 분리하는 패키지 설계의 일부입니다.
