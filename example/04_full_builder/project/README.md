# 04 Full Builder

이 디렉터리는 `hydrogel_builder` 전체 흐름을 가장 작게 확인하는 active smoke 예시입니다.

## 가장 쉬운 실행

```bash
bash hygel_run.sh
```

축소 anisotropy 테스트:

```bash
bash hygel_run.sh maker_anisotropy_x.yaml
```

도움말과 GROMACS 확인:

```bash
bash hygel_run.sh --help
bash hygel_run.sh --check-gmx
```

직접 Python으로 실행할 때:

```bash
python3 -m hydrogel_builder maker.yaml
python3 -m hydrogel_builder --config maker.yaml
```

## active 파일

- `hygel_run.sh`
  conda + GMXRC 환경을 잡고 `python -m hydrogel_builder`를 호출합니다.
- `maker.yaml`
  기본 smoke 진입점입니다.
- `maker_anisotropy_x.yaml`
  anisotropy 축소 테스트입니다.
- `config/`
  simulation, hydrogel, mdp, backbone, add-series 설정입니다.
- `structure/`
  backbone, linker, molecule template입니다.
- `soft_shrinker/`
  선택적 box relaxation helper입니다.

## 출력 위치

- 기본 출력은 `config/simulation.yaml`의 `simulation_parameters.output_dir`를 따릅니다.
- 현재 기본값은 `project/output`입니다.

## 다음 단계

- builder 뒤에 별도 완화가 필요하면 `../05_hydrogel_relaxation/project`를 사용합니다.
