# 05 Hydrogel Relaxation

이 디렉터리는 `04_full_builder` 또는 `04_1_example_system`이 만든 구조를
후처리하는 active 예시입니다.

순서는 아래처럼 잡으면 됩니다.

```bash
bash run_hydrogel_relaxation.sh soft_em
bash run_hydrogel_relaxation.sh soft_md
```

도움말과 GROMACS 확인:

```bash
bash run_hydrogel_relaxation.sh --help
bash run_hydrogel_relaxation.sh --check-gmx
bash run_hydrogel_relaxation.sh --workflow-help
```

직접 Python으로 실행할 때:

```bash
python3 -m hydrogel_builder.relax maker_soft_em.yaml
python3 -m hydrogel_builder.relax maker_soft_md.yaml
```

## 기본 입력 경로

- 현재 기본값은 `../04_1_example_system/project/output`을 입력으로 봅니다.
- `04_full_builder` 결과를 쓰고 싶으면 `config/common.yaml`의 path만 바꾸면 됩니다.

## active 파일

- `run_hydrogel_relaxation.sh`
  primary launcher입니다.
  `soft_em` / `soft_md` stage alias 또는 maker yaml을 받아
  `python -m hydrogel_builder.relax`를 호출합니다.
- `maker_soft_em.yaml`
  gradual EM / box relaxation 진입점
- `maker_soft_md.yaml`
  `grompp + mdrun` 추가 완화 진입점
- `config/`
  공통 path/runtime와 mode별 설정
- `config/minim.mdp`
  soft EM에서 쓰는 minimization mdp
- `config/npt_1ns.mdp`
  soft MD에서 쓰는 mdp
