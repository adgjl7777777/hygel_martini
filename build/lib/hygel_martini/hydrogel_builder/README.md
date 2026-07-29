# `hygel_martini.hydrogel_builder`

`hydrogel_builder`는 coarse-grained hydrogel의 graph-first construction과
post-build relaxation을 제공합니다. 단순히 GRO 파일을 쓰는 것이 아니라,
layout에서 의도한 network가 bonded topology와 coordinate에 동일하게
materialize되었는지를 검사하는 workflow입니다.

검증 실패와 설계 변경의 전체 근거는
`../../docs/VALIDATION_HISTORY_AND_DESIGN_RATIONALE.md`에 있습니다.

## 실행

패키지를 설치한 뒤 tracked example에서 실행합니다.

```bash
cd example/04_full_builder/project
bash run_full_builder.sh maker.yaml
```

작은 example system은 다음과 같습니다.

```bash
cd example/04_1_example_system/project
bash run_example_system.sh maker.yaml
```

동일한 builder entry point를 직접 호출할 수도 있습니다.

```bash
hygel-builder maker.yaml
python -m hygel_martini.hydrogel_builder maker.yaml
python -m hygel_martini.hydrogel_builder --config maker.yaml
```

`example_myrun`은 local-only 실행·검증 공간이며 fresh clone에 포함되지
않습니다. Public 사용법이나 package import를 그 절대 경로에 맞추지
마십시오.

## Graph-first BCK 의미

`dynamic_crosslink_targets_per_stub: 2`이면 BCK stub 하나가 서로 다른
backbone end 두 개를 연결해 local polymer junction 하나를 만듭니다.
Two-stub linker 하나의 의미는 다음과 같습니다.

```text
BCK1: polymer A -- polymer B
BCK2: polymer C -- polymer D
```

따라서 linker 하나는 네 개의 BCK-backbone bond와 두 개의 polymer
junction을 만듭니다. BCK1-BCK2 bond는 linker template/topology가
명시했을 때만 존재합니다.

한 diamond vertex의 네 endpoint에 가능한 perfect matching은 x/y/z 세
가지입니다.

```text
x: (000-011), (101-110)
y: (000-101), (011-110)
z: (000-110), (011-101)
```

`linker_orientation_strategy: connectivity_aware`는 layout 단계에서
이 local transition들을 선택합니다. 목표는 axis 사용을 균형 있게
유지하면서 backbone edge와 junction edge를 합친 graph의 component와
degree defect를 줄이는 것입니다. Runtime dynamic-crosslink는 이 graph를
임의 chain reuse로 다시 설계하지 않고, 배치된 BCK와 endpoint 사이의
materialization 및 uniqueness 검사를 담당합니다.

관련 구현:

- `core_utils/layout/local_matching.py`
- `core_utils/layout/isotropic_builder.py`
- `core_utils/runtime/dynamic_crosslink.py`

## 설정의 역할 분리

```yaml
simulation_parameters:
  # Layout graph에서 local x/y/z transition을 선택합니다.
  linker_orientation_strategy: connectivity_aware

  # Materialized BCK stub 하나가 받을 backbone end 수입니다.
  dynamic_crosslink_targets_per_stub: 2

  # Runtime endpoint search 후보 폭입니다.
  # component 수나 loop 수를 직접 지정하는 값이 아닙니다.
  dynamic_crosslink_candidate_limit: 64

  # Intended bonded definition을 최종 ITP에 명시적으로 적용할 때 사용합니다.
  bonded_topology_patch_file: /path/to/backbone.yaml

# 생성된 atom-level bonded graph의 사후 안전장치입니다.
# Layout transition optimizer 자체가 아닙니다.
hydrogel_topology_connectivity_audit:
  enabled: true
  min_largest_component_fraction: 0.95
  max_components: 1
  fail_on_violation: true
```

`connectivity_guard`는 오래된 alias입니다. 새 YAML은
`hydrogel_topology_connectivity_audit`를 사용하십시오.

`bonded_topology_patch_file`을 사용할 때는 patch file이 존재했다는 사실만
보지 말고, 최종 materialized ITP의 mass, bond, angle, dihedral
distribution을 intended reference와 비교해야 합니다. 다른 bonded
topology에서 만든 TPR/CPT를 resume하면 안 됩니다.

## Construction gate

긴 MD 전에 최소한 다음을 확인합니다.

1. expected atom, polymer, water, ion 수
2. BCK 수, BCK mass와 attachment 수
3. connected component와 largest-component fraction
4. junction degree와 backbone-end reuse
5. theoretical/actual bonded-term completeness
6. PBC minimum-image bond와 winding
7. finite coordinate와 overlap
8. independent `grompp` preflight

한 component라는 사실만으로 correct topology, elastically active
network, 또는 force-field validation이 증명되지는 않습니다.

Bonded-term completeness를 별도로 검사하는 CLI:

```bash
hygel-audit-topology \
  --itp output/initial_hydrogel.itp \
  --backbone-residues PEO \
  --linker-residues BCK \
  --require-junction-angles \
  --require-junction-dihedrals \
  --fail-on-issue
```

Reduced junction–strand graph와 PBC winding은 property CLI에서 검사할 수
있습니다.

```bash
hygel-property topology \
  --itp output/initial_hydrogel.itp \
  --gro output/final_optimized_system.gro \
  --junction-residue BCK \
  --output topology_audit.json
```

## Post-build relaxation

Tracked relaxation example:

```bash
cd example/05_hydrogel_relaxation/project
bash run_hydrogel_relaxation.sh maker_soft_em.yaml
bash run_hydrogel_relaxation.sh maker_soft_md.yaml
bash run_hydrogel_relaxation.sh maker_hard_em_shrink.yaml
```

직접 실행:

```bash
hygel-relax maker_soft_em.yaml
python -m hygel_martini.hydrogel_builder.relax maker_soft_md.yaml
```

지원 mode:

| `workflow.mode` | 역할 |
|---|---|
| `soft_em` | bonded interaction과 box response를 단계적으로 적용하는 EM |
| `soft_md` | post-build MD relaxation |
| `hard_em_shrink` | 큰 희박 box를 guarded affine shrink와 hard EM으로 목표 box에 접근 |

### Guarded hard shrink

`hard_em_shrink`는 각 후보 단계에서 coordinate와 box를 함께 등방
scale하고 PBC wrap 뒤 EM을 수행합니다. 유한 coordinate/box/energy와
`fmax_max` gate를 모두 통과한 구조만 새 `last_valid.gro`로 commit합니다.

EM gate가 실패하면 짧은 NVT recovery 뒤 다시 검사합니다. 재검사도
실패하면 직전 유효 구조로 rollback하고 shrink step을 절반으로 줄입니다.
최소 step에서도 통과하지 못하면 실패 상태로 중단하며 다음 파일을
남깁니다.

- `last_valid.gro`
- `state.json`
- `history.jsonl`

큰 희박 box에서 바로 aggressive NPT를 시작하거나, 실패한 candidate를
다음 단계 input으로 승격하지 않기 위한 workflow입니다.

### Soft-EM box mode

`soft_em.box_mode`는 다음 세 가지입니다.

| mode | 동작 | 주 용도 |
|---|---|---|
| `anisotropic` | x/y/z pressure에 독립적으로 반응 | 이미 안정한 비등방 system의 fine adjustment |
| `isotropic` | 평균 pressure로 세 축을 같은 비율로 scale | 초기 압축에서 box shape ratio 유지 |
| `cubic` | 평균 pressure 조절과 cubic-shape correction 결합 | 과거 비대칭 box의 복구 |

Box header만 줄이고 atom coordinate를 그대로 두면 새 PBC 경계에서
collision이 생길 수 있습니다. 현재 relaxation code는 coordinate와 box를
같이 scale하고 wrap합니다.

## GPU, OpenMP, MPI

Builder EM은 `simulation_parameters`, relaxation은 `runtime` 아래에서
resource를 지정합니다.

```yaml
runtime:
  # null이면 CPU-only mdrun flags를 명시적으로 사용합니다.
  # 단일 GPU라면 "0", MPI GPU mapping이라면 환경에 맞는 문자열을 씁니다.
  gpu_id: null
  omp_threads: 4
  mpi_np: null
  mpi_args: []
```

`gpu_id: null`을 단순히 환경 기본값에 맡기지 않고 CPU-only flag로
해석하는 이유는 공용 서버에서 GROMACS가 사용 가능한 GPU를 자동으로
점유하는 일을 막기 위해서입니다.

## 내부 구조

- `generator.py`, `cli.py`
  top-level builder entry
- `config_params`
  maker include 병합, path normalization, workflow orchestration
- `core_utils/layout`
  graph-first layout와 local matching
- `core_utils/runtime`
  topology materialization, dynamic crosslink, GROMACS/Packmol 실행
- `core_utils/io`
  Martini/GROMACS parser와 writer
- `main_components`
  `World`, `Hydrogel`, `Polymer` data model
- `add_series`
  polymer, molecule, water, ion 추가
- `relax`
  `soft_em`, `soft_md`, `hard_em_shrink`

## 실행 흐름

Builder:

1. `run_full_builder.sh` 또는 `run_example_system.sh`
2. `hygel-builder`
3. `generator.run_hydrogel_builder`
4. `config_params.generator.run_hydrogel_example`
5. `config_params.read_json.execute_mode`

Relaxation:

1. `run_hydrogel_relaxation.sh`
2. `hygel-relax`
3. `relax.generator.run_relax_workflow`
4. selected relaxation mode

## Claim boundary

Builder gate가 통과하면 intended graph가 coordinate와 topology로
materialize되었다는 construction claim을 지지합니다. 다음을 자동으로
증명하지는 않습니다.

- force-field parameter의 quantitative material accuracy
- equilibrium swelling
- unique pore/mesh length
- equilibrium 또는 experimental rheology

이후 property claim은 `hygel_martini.property_extract`의 requirement,
observable-definition, numerical, promotion gate를 별도로 통과해야
합니다.
