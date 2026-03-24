# hydrogel_builder

`hydrogel_builder`는 builder와 post-build relaxation을 같이 제공합니다.

## 가장 쉬운 실행

직접 Python으로 실행할 때는 아래 둘 다 됩니다.

```bash
python -m hydrogel_builder maker.yaml
python -m hydrogel_builder --config maker.yaml
```

보통은 예제 `project/` 안의 `hygel_run.sh`를 먼저 쓰는 편이 더 쉽습니다.

```bash
cd /nas_0/software_backup/hygel_martini/example_myrun/04_full_builder/project
bash hygel_run.sh
```

```bash
cd /nas_0/software_backup/hygel_martini/example_myrun/04_1_example_system/project
bash hygel_run.sh
```

후처리 relaxation은 별도 모듈입니다.

```bash
cd /nas_0/software_backup/hygel_martini/example_myrun/05_hydrogel_relaxation/project
bash hygel_run.sh maker_soft_em.yaml
bash hygel_run.sh maker_soft_md.yaml
```

## 내부 구조

- `generator.py`
  가장 얇은 top-level 실행 진입점
- `cli.py`
  `python -m hydrogel_builder`용 CLI
- `relax`
  build 이후 `soft_em` / `soft_md`를 수행하는 후처리 모듈
- `config_params`
  maker 파일 로드, include 병합, 전체 workflow orchestration
- `core_utils`
  IO, layout, template loader, runtime helper
- `main_components`
  `World`, `Hydrogel`, `Polymer` 같은 핵심 데이터 구조
- `add_series`
  polymer, molecule, water, ion 추가 단계

## 실행 흐름

1. `hygel_run.sh`
2. `python -m hydrogel_builder`
3. `hydrogel_builder.generator.run_hydrogel_builder`
4. `hydrogel_builder.config_params.generator.run_hydrogel_example`
5. `hydrogel_builder.config_params.read_json.execute_mode`

후처리 relaxation은 아래 흐름입니다.

1. `05_hydrogel_relaxation/project/hygel_run.sh`
2. `python -m hydrogel_builder.relax`
3. `hydrogel_builder.relax.generator.run_relax_workflow`
4. `hydrogel_builder.relax.soft_em` 또는 `hydrogel_builder.relax.soft_md`

## 설정 규칙

- maker 파일은 보통 `includes:`로 `config/*.yaml`을 묶습니다.
- `${CONFIG_DIR}`는 현재 maker 파일 위치를 뜻합니다.
- `${REPO_ROOT}`는 저장소 루트를 뜻합니다.
- 기본 출력은 보통 `config/simulation.yaml`의 `output_dir` 아래에 생성됩니다.
