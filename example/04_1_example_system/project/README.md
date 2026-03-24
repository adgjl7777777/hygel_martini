# 04_1 Example System

이 디렉터리는 실제 예시 시스템을 만드는 active builder 예시입니다.

## 가장 쉬운 실행

```bash
bash hygel_run.sh
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
  이 프로젝트의 기본 진입점입니다.
- `config/`
  hydrogel layout, simulation, add-series 설정입니다.
- `structure/`
  monomer와 linker template입니다.

## 출력 위치

- 기본 출력은 `config/simulation.yaml`의 `simulation_parameters.output_dir`를 따릅니다.
- 현재 기본값은 `project/output`입니다.

## 다음 단계

- 이 출력 뒤에 `soft_em`과 `soft_md`를 돌리려면 `../05_hydrogel_relaxation/project`를 사용합니다.
