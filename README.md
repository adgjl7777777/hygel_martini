# hygel_martini

Coarse-grained 고분자 네트워크(hydrogel, elastomer)를 **그래프로 먼저 계획하고,
좌표로 실체화한 뒤, 쓰인 topology를 감사**하는 연구용 Python 패키지입니다.
QM/OPLS 기반 파라미터 준비(`param_opt`), 네트워크 구성과 완화
(`hydrogel_builder`), 물성 추출(`property_extract`)을 하나의 재현 가능한
workflow로 연결합니다.

**Author:** Daehong Kim — School of Chemical and Biological Engineering,
Seoul National University ·
ORCID [0009-0007-1647-9270](https://orcid.org/0009-0007-1647-9270)

## 브랜치 안내

| 브랜치 | 상태 |
|---|---|
| `omni/general-ff-and-f6` (이 브랜치) | 활성 개발. **임의의 짝수 junction functionality**(f=6 hexafunctional 포함), net 기반 layout, span 제약 rewiring, cyclic-topology 감사 |
| `master` | Series-01 (PEGDA/Pluronic, tetrafunctional diamond) 논문 동결 상태. 원고가 커밋 hash를 인용하므로 재작성하지 않습니다 |

## 무엇이 되는가

- **Graph-first 구성.** 원하는 periodic network를 좌표 생성 전에 그래프로
  계획하고, planner가 고른 endpoint edge가 그대로 실체화되었는지 검사합니다.
- **격자(net) 선택.** 다이아몬드(`dia`, f=4)와 primitive cubic(`pcu`, f=6)을
  RCSR 정의 그대로 제공하며, transition-system planner·crosslink router·감사가
  전부 functionality-일반입니다.
- **현실적 topology.** 완전 격자는 bipartite라 홀수 loop가 없습니다. span 제약
  rewiring이 loop-order 분포를 물리적 파라미터 하나(`max_span`)로 조절합니다.
- **Topology 감사.** vertex symbol, loop-order 히스토그램, primary/secondary
  loop 분율, bipartite 여부, periodic winding, bonded-term completeness.
- **파라미터 준비.** xTB/ORCA → Bartender → screened Martini ITP, 그리고
  E0–E6 bonded-parameter 결정 protocol.
- **물성 추출.** state, structure, transport, clearance, finite-rate mechanics
  를 requirement/manifest gate와 함께.

조용히 틀리는 대신 시끄럽게 실패하도록 설계되어 있습니다. 이 브랜치에서
발견·수정된 결함의 전체 기록은
[`docs/DEFECTS_FOUND_AND_FIXED.md`](docs/DEFECTS_FOUND_AND_FIXED.md)에 있습니다.

## 설치

Python 3.9 이상.

```bash
git clone --recurse-submodules https://github.com/adgjl7777777/hygel_martini.git
cd hygel_martini
python -m pip install -e .          # 개발용은: pip install -e ".[dev]" && pytest
```

기존 clone에는 `git submodule update --init --recursive`로 PoreBlazer를
등록합니다(독립 프로그램이며 wheel에 포함되지 않습니다). GROMACS, Packmol,
xTB, ORCA, Bartender, Martini force-field 파일은 라이선스가 달라 자동 설치하지
않습니다.

> 이미 다른 사본이 editable로 설치된 환경에서는 `import hygel_martini`가 그
> 사본을 가리킵니다. 작업 사본으로 실행하려면 저장소 루트에서
> `PYTHONPATH=$PWD python3 ...` 형태로 실행하고, 결과가 이상하면
> `hygel_martini.__file__`부터 확인하십시오.

## 5분 시작

**다이아몬드(f=4) full build** — GROMACS/Packmol 필요:

```bash
cd example/04_full_builder/project
bash ../../../hygel_martini/bash_settings/hydrogel_builder/run_full_builder.sh maker.yaml
```

**Hexafunctional(f=6, pcu) layout 검사** — 외부 프로그램 불필요:

```bash
PYTHONPATH=$PWD python3 - <<'PY'
from hygel_martini.hydrogel_builder.core_utils.layout.net_layout import generate_net_layout_plan
class P: pass
r = generate_net_layout_plan(P(), [{"id": "BB1"}], [{"id": "HEX"}],
                             net="pcu", repeats=4, cell_parameter=3.0,
                             max_span=6.0, rewire_seed=0)
print(r.summary())
PY
```

단계별 실행 순서는 [`START_HERE_ko.md`](START_HERE_ko.md)를 따르십시오.

## 예제

| 예제 | 내용 | 상태 |
|---|---|---|
| `00_bead_selector` | bead selector 예정 위치 | placeholder |
| `01_qm_to_opls` | ORCA/QM → OPLS | placeholder |
| `02_opls_to_martini` | 기존 OPLS/GROMACS trajectory → Bartender refit | 사용자 데이터 연결 후 실행 |
| `03_qm_to_martini` | xTB/ORCA → Bartender → screened ITP | 바로 실행 가능 |
| `04_full_builder` | 다이아몬드 full hydrogel builder | 바로 실행 가능 |
| `04_1_example_system` | 작은 예시 시스템 | 바로 실행 가능 |
| `05_hydrogel_relaxation` | staged minimization / settling MD / guarded shrink | 바로 실행 가능 |
| `06_physical_property` | manifest 기반 물성 추출 | 바로 실행 가능 |
| `07_hexafunctional` | **f=6 crosslinker + `pcu` net + rewiring** | layout/plan 검증 완료, GROMACS end-to-end 미실행 |

자세한 구성은 [`example/README_ko.md`](example/README_ko.md).

## 명령

| 명령 | 역할 |
|---|---|
| `hygel-builder` | hydrogel construction (`python -m hygel_martini.hydrogel_builder`) |
| `hygel-relax` | post-build relaxation |
| `hygel-property` | 물성 추출과 topology 감사 |
| `hygel-qm-to-opls` / `hygel-opls-to-martini` / `hygel-qm-to-martini` | 파라미터 준비 stage 01/02/03 |
| `hygel-parameter-protocol` | E0–E6 bonded-parameter 결정 protocol |
| `hygel-qm-reference-audit` | xTB/고수준 참조 적합성 gate |
| `hygel-audit-topology` | bonded graph audit |
| `python -m hygel_martini.property_extract.cyclic_topology` | vertex symbol / loop-order / bipartite 감사 |

각 명령의 옵션은 `COMMAND --help`로 확인합니다. 공용 Bash launcher는
`hygel_martini/bash_settings/`에 있습니다.

## 저장소 구조

```
hygel_martini/
  core/                 공용 프리미티브: pbc(최소이미지, triclinic 지원),
                        gro(GRO reader), itp(GROMACS topology parser), physics
  param_opt/            qm_to_opls · opls_to_martini · qm_to_martini(+protocol)
  hydrogel_builder/
    config_params/      maker.yaml 병합·검증·workflow orchestration
    core_utils/layout/  nets(dia/pcu) · net_layout · rewire ·
                        local_matching(일반 f transition system) · diamond layout
    core_utils/runtime/ dynamic_crosslink(일반 f router) · packer · geo_opt
    core_utils/templates/ monomer/linker(N-stub) template loader
    relax/              soft_em · soft_md · hard_em_shrink
  property_extract/     분석·감사 (network_topology, cyclic_topology, ...)
example/                tracked 예제 (위 표)
docs/                   아래 문서
tests/                  pytest suite
```

## 문서

| 문서 | 내용 |
|---|---|
| [`START_HERE_ko.md`](START_HERE_ko.md) | 처음 실행하는 순서 |
| [`docs/GENERAL_FUNCTIONALITY_NETWORKS.md`](docs/GENERAL_FUNCTIONALITY_NETWORKS.md) | f-일반 네트워크: `network_layout` 설정, net별 제약, rewiring, 감사 |
| [`docs/DEFECTS_FOUND_AND_FIXED.md`](docs/DEFECTS_FOUND_AND_FIXED.md) | 이 브랜치에서 발견·수정한 결함 기록 |
| [`docs/PARAMETERIZATION_PROTOCOL.md`](docs/PARAMETERIZATION_PROTOCOL.md) | E0–E6 파라미터 결정 protocol |
| [`docs/VALIDATION_HISTORY_AND_DESIGN_RATIONALE.md`](docs/VALIDATION_HISTORY_AND_DESIGN_RATIONALE.md) | Series-01 validation 실패·교정 이력 |
| 모듈 README | `hydrogel_builder/`, `hydrogel_builder/relax/`, `param_opt/`, `property_extract/`, `param_opt/qm_to_martini/analysis/` |
| `docs/archive/` | 구버전 스냅샷 (Series-01 기준 상세 기술서 등) |

## Claim boundary

Builder gate 통과는 **의도한 그래프가 좌표와 topology로 실체화되었다**는
construction claim을 지지할 뿐, force-field 정확도·equilibrium swelling·
유일한 pore/mesh 길이·experimental rheology를 자동으로 증명하지 않습니다.
물성 claim은 `property_extract`의 requirement/observable/numerical/promotion
gate를 별도로 통과해야 합니다. loop-order 분포를 맞춘 것 역시 topology
statement이며 역학적 물성의 재현이 아닙니다. 이 브랜치의 f=6 경로는 layout과
plan 수준에서 검증되었고 **GROMACS end-to-end 빌드는 아직 수행되지
않았습니다.**

## 인용, 연구비, 라이선스

소프트웨어 인용 정보는 [`CITATION.cff`](CITATION.cff)에 있습니다. 관련
방법론 논문의 최종 서지정보가 확정되면 software release와 논문을 함께
인용하도록 갱신합니다.

This work was supported by the National Research Foundation of Korea (NRF)
grant funded by the Korea government (MSIT) (RS-2025-25424498).

라이선스와 외부 프로그램(PoreBlazer, Martini force field 등)의 조건은
[`LICENSING.md`](LICENSING.md)를 따릅니다.
