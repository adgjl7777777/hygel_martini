# 04_1 Example System

이 디렉터리는 YAML로 random DMAPS--SBMA copolymer와 10--12-bead LDD/LSS/LDS
linker template을 조합하는 construction 예시입니다. 현재 설정은
`connectivity_aware` endpoint-edge plan을 runtime까지 전달하고, 생성된 topology가
단일 bonded component가 아니면 build를 실패시킵니다. 실행 후에는
`dynamic_bonding_debug.log`의 planned/materialized edge hash와
`connectivity_audit.log`를 함께 확인합니다.

## 가장 쉬운 실행

먼저 저장소 루트에서 패키지를 현재 Python 환경에 설치합니다.

```bash
cd /path/to/hygel_martini
python -m pip install -e .
```

```bash
bash run_example_system.sh maker.yaml
```

도움말과 GROMACS 확인:

```bash
bash run_example_system.sh --help
bash run_example_system.sh --check-gmx
```

직접 Python으로 실행할 때:

```bash
python3 -m hygel_martini.hydrogel_builder maker.yaml
python3 -m hygel_martini.hydrogel_builder --config maker.yaml
```

## active 파일

- `run_example_system.sh`
  primary launcher입니다.
  conda + GMXRC 환경을 잡고 설치된 `python -m hydrogel_builder`를 호출합니다.
- `maker.yaml`
  이 프로젝트의 기본 진입점입니다.
- `config/`
  hydrogel layout, simulation, add-series 설정입니다.
- `structure/`
  monomer와 linker template입니다.

## 출력 위치

- 기본 출력은 `config/simulation.yaml`의 `simulation_parameters.output_dir`를 따릅니다.
- 현재 기본값은 `project/output`입니다.

## 판정 경계

- random sequence, heterogeneous long-linker materialization, exact endpoint-edge
  handoff, bonded-graph connectivity가 이 예제가 검증하는 범위입니다.
- process completion이나 EM completion만으로 판정하지 않고 written topology
  audit까지 통과해야 합니다.
- 이 dry construction 예제 자체는 sampled-state나 물성 검증을 주장하지 않습니다.
