# 05 Hydrogel Relaxation

이 디렉터리는 `04_full_builder` 또는 `04_1_example_system`이 만든 구조를
후처리하는 active 예시입니다.

순서는 아래처럼 잡으면 됩니다.

먼저 저장소 루트에서 패키지를 현재 Python 환경에 설치합니다.

```bash
cd /path/to/hygel_martini
python -m pip install -e .
```

```bash
bash run_hydrogel_relaxation.sh maker_soft_em.yaml
bash run_hydrogel_relaxation.sh maker_soft_md.yaml
bash run_hydrogel_relaxation.sh maker_hard_em_shrink.yaml
```

도움말과 GROMACS 확인:

```bash
bash run_hydrogel_relaxation.sh --help
bash run_hydrogel_relaxation.sh --check-gmx
bash run_hydrogel_relaxation.sh --workflow-help
```

직접 Python으로 실행할 때:

```bash
python3 -m hygel_martini.hydrogel_builder.relax maker_soft_em.yaml
python3 -m hygel_martini.hydrogel_builder.relax maker_soft_md.yaml
python3 -m hygel_martini.hydrogel_builder.relax maker_hard_em_shrink.yaml
```

## 기본 입력 경로

- 현재 기본값은 `../04_1_example_system/project/output`을 입력으로 봅니다.
- `04_full_builder` 결과를 쓰고 싶으면 `config/common.yaml`의 path만 바꾸면 됩니다.

## active 파일

- `run_hydrogel_relaxation.sh`
  primary launcher입니다.
  maker yaml을 받아 설치된 `python -m hydrogel_builder.relax`를 호출합니다.
- `maker_soft_em.yaml`
  gradual EM / box relaxation 진입점
- `maker_soft_md.yaml`
  `grompp + mdrun` 추가 완화 진입점
- `maker_hard_em_shrink.yaml`
  1% 고정 비율 hard-EM shrink와 NVT recovery 진입점
- `config/`
  공통 path/runtime와 mode별 설정
- `config/minim.mdp`
  staged minimization(`soft_em`)에서 쓰는 minimization mdp
- `config/npt_1ns.mdp`
  soft MD에서 쓰는 mdp
- `config/hard_em_shrink.yaml`
  목표 box까지 guarded hard shrink 설정. `target_box_nm`는 formulation별 dense target으로 반드시 확인한다.
- `config/nvt_hard_shrink_recovery_100ps.mdp`
  shrink guard 실패 시 쓰는 짧은 2 fs NVT recovery mdp
