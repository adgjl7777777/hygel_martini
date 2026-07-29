# `hygel_martini.property_extract`

이 패키지는 hydrogel build와 trajectory에서 구조·수송·유한속도 mechanics
observable을 계산하고, 각 결과의 계산 가능성 및 claim boundary를 함께
기록합니다.

핵심 원칙은 다음 네 단계를 구분하는 것입니다.

1. 입력 파일과 trajectory가 완전한가
2. 분석기가 해당 observable을 계산할 수 있는가
3. 수치가 유한하고 등록된 gate를 통과하는가
4. 그 결과를 실험값이나 equilibrium 물성으로 해석할 수 있는가

계산 성공만으로 4번을 자동 승인하지 않습니다.

이 원칙이 생긴 구체적인 validation 이력과 폐기된 해석은
`../../docs/VALIDATION_HISTORY_AND_DESIGN_RATIONALE.md`를 봅니다.

## 제공 모듈

| 모듈 | 주요 기능 | 해석 경계 |
|---|---|---|
| `network_topology` | junction–strand 축약 graph, defect, cycle rank, PBC winding, connectivity hash | construction audit이며 force-field 검증이 아님 |
| `mechanics` | affine shear/uniaxial 변형, paired-step pressure decomposition, classical network scale | finite-rate/reference scale이며 equilibrium modulus가 아님 |
| `mechanics_analysis` | registered-window step response, paired ramp, cycle-block bootstrap, realization weighting, Holm correction | origin/cycle을 독립 network로 세지 않음 |
| `pore_size` | mixed-radius periodic clearance, probe-admissible fraction, periodic component size | unique pore/mesh size 또는 PoreBlazer PLD가 아님 |
| `geometry` | minimum image, chain unwrap, `Rg`, end-to-end, orientation tensor | 선택과 PBC 정의를 함께 기록해야 함 |
| `diffusion` | trajectory unwrap, multi-origin MSD, explicit-window diffusion fit | fit window와 water model에 종속 |
| `aggregation` | periodic contact graph와 component | cutoff와 site selection에 종속 |
| `spatial` | voxel heterogeneity, periodic field correlation, phase-randomized null | resolution-dependent diagnostic |
| `structure_factor` | CIC 기반 static `S(q)`와 radial/axis binning | `2π/q`를 자동으로 pore/mesh로 변환하지 않음 |
| `timeseries` | XVG parsing, time window, block statistics, drift | block은 network replicate가 아님 |
| `swelling` | imposed composition 및 box-volume 기반 polymer fraction | fixed-water state는 free swelling이 아님 |

## CLI

설치 후 `hygel-property` 또는 다음 module command를 사용합니다.

```bash
python -m hygel_martini.property_extract --help
```

### Reduced-network topology audit

```bash
hygel-property topology \
  --itp initial_hydrogel.itp \
  --gro production.gro \
  --junction-residue BCK \
  --expected-junctions 64 \
  --expected-strands 128 \
  --expected-winding-rank 3 \
  --output topology_audit.json
```

출력에는 atom-bond, junction-attachment, reduced-edge SHA-256 fingerprint가
포함됩니다. 서로 다른 좌표와 서로 다른 local-crosslink realization,
서로 다른 contracted macro-scaffold를 구분할 때 사용합니다.

### Paired-step finite-rate mechanics

먼저 동일한 origin에서 baseline, positive, negative branch의 같은 pressure
component를 XVG로 추출합니다. 세 XVG의 time grid와 legend가 일치해야
합니다.

```bash
hygel-property mechanics-step \
  --baseline-xvg base.xvg \
  --positive-xvg plus.xvg \
  --negative-xvg minus.xvg \
  --component Pres-XY \
  --gamma 0.02 \
  --window-start-ps 1 \
  --window-end-ps 5 \
  --output paired_step.json
```

계산식은 GROMACS pressure sign을 고려한 다음 odd response입니다.

```text
P_odd = (P_plus - P_minus) / 2
G_apparent = -P_odd / gamma
```

기본 pressure 단위가 bar이면 결과는 MPa입니다. 출력의 정확한 observable
이름은 `paired_step_finite_rate_apparent_shear_response`이며 equilibrium,
plateau, storage, zero-frequency 또는 experimental modulus가 아닙니다.

### Periodic local clearance

```bash
hygel-property clearance-frame \
  --gro production.gro \
  --selection-residue PEO \
  --selection-residue BCK \
  --bead-radius-nm 0.24 \
  --probe-radius-nm 0.1657 \
  --grid-spacing-nm 0.20 \
  --output clearance.json
```

이 계산은 orthorhombic periodic box의 cell-centred grid에서 가장 가까운
obstacle surface까지의 거리를 구합니다. Grid spacing, obstacle selection,
bead radius, probe radius를 바꾸면 observable 정의가 바뀝니다.

## Python API

### Topology

```python
from hygel_martini.property_extract import audit_reduced_network

audit = audit_reduced_network(
    "initial_hydrogel.itp",
    "production.gro",
    junction_residue="BCK",
)
print(audit["junction_count"])
print(audit["valid_strand_count"])
print(audit["periodic"]["winding_rank"])
```

### Mechanics와 realization weighting

```python
from hygel_martini.property_extract import (
    paired_step_xvg_summary,
    summarize_equal_realizations,
)

response = paired_step_xvg_summary(
    "base.xvg",
    "plus.xvg",
    "minus.xvg",
    component="Pres-XY",
    gamma=0.02,
    window_start_ps=1.0,
    window_end_ps=5.0,
)

ensemble = summarize_equal_realizations(
    {
        "seed101": response_seed101_origin_values,
        "seed202": response_seed202_origin_values,
        "seed303": response_seed303_origin_values,
    }
)
```

각 realization 내부 origin/plane을 먼저 평균한 뒤 realization mean에 같은
가중치를 줍니다. 서로 다른 origin 또는 stochastic repeat를 독립
network로 풀링하면 안 됩니다.

### Periodic clearance

```python
import numpy as np
from hygel_martini.property_extract import (
    calculate_periodic_clearance_distribution,
)

centres, density, summary = calculate_periodic_clearance_distribution(
    [
        (polymer_positions_nm, 0.24),
        (selected_water_positions_nm, 0.14),
    ],
    np.array([box_x_nm, box_y_nm, box_z_nm]),
    grid_spacing=0.20,
    probe_radius=0.1657,
    bins=80,
)
```

`polymer_only`, `polymer + interfacial water`, `polymer + all water`는 서로
다른 obstacle definition입니다. 정의를 섞어서 하나의 pore size로
보고하지 않습니다.

### Structure, transport, and spatial diagnostics

```python
from hygel_martini.property_extract import (
    bond_orientation_metrics,
    chain_contact_graph,
    fit_diffusion_coefficient,
    gyration_metrics,
    multi_origin_msd,
    periodic_field_correlation,
    voxel_counts,
)
```

각 함수는 순수 NumPy 배열을 받으므로 MDAnalysis selection/trajectory
iteration과 분리해서 테스트할 수 있습니다. PBC unwrap, selection,
sampling stride, fit window는 호출자가 명시하고 결과 provenance에
보존해야 합니다.

## YAML analysis jobs

여러 observable을 한 번에 실행할 때는 extractor registry를 사용합니다.

```yaml
analysis_jobs:
  topology:
    property: reduced_network_topology_audit
    extractor: topology.reduced_network
    inputs:
      itp: initial_hydrogel.itp
      gro: production.gro
    parameters:
      junction_residue: BCK
      expected_junction_count: 64
      expected_strand_count: 128
      expected_winding_rank: 3

  mechanics_xy:
    property: paired_step_finite_rate_apparent_shear_response
    extractor: mechanics.paired_step_xvg
    inputs:
      baseline_xvg: base_xy.xvg
      positive_xvg: plus_xy.xvg
      negative_xvg: minus_xy.xvg
    parameters:
      component: Pres-XY
      gamma: 0.02
      window_start_ps: 1.0
      window_end_ps: 5.0

  clearance:
    property: local_clearance_diameter_p50_nm
    extractor: clearance.periodic_grid
    inputs:
      gro: production.gro
    parameters:
      selection_residues: [PEO, BCK]
      bead_radius_nm: 0.24
      probe_radius_nm: 0.1657
      grid_spacing_nm: 0.20
```

실행:

```bash
hygel-property requirements --analysis analysis_jobs.yaml --strict
hygel-property analyze --analysis analysis_jobs.yaml
```

지원 extractor가 없거나 입력이 누락된 경우 결과를 만들었다고 간주하지
않고 `not_implemented`, `missing_required_md`, `invalid_input`,
`analysis_failed` 중 하나로 명시합니다.

## 검증

패키지의 수치 primitive와 topology fixture는 다음으로 확인합니다.

```bash
pytest tests/test_analysis_primitives.py \
       tests/test_network_topology.py \
       tests/test_mechanics_analysis.py
```

회귀 테스트는 PBC wrapping, mixed-radius clearance, structure factor,
diffusion fit, affine deformation, pressure sign, topology winding,
realization weighting을 포함합니다.
