# hydrogel_builder.relax

`hydrogel_builder.relax`는 builder가 만든 구조를 후처리하는 단계입니다.

- `soft_em`
  box를 천천히 바꾸면서 EM-only preconditioner를 수행
- `soft_md`
  `grompp + mdrun` 기반 추가 relaxation / equilibration

권장 실행:

```bash
python -m hydrogel_builder.relax maker_soft_em.yaml
python -m hydrogel_builder.relax maker_soft_md.yaml
```

보통은 예제 디렉터리의 `hygel_run.sh`를 먼저 쓰는 편이 더 쉽습니다.

```bash
cd /nas_0/software_backup/hygel_martini/example_myrun/05_hydrogel_relaxation/project
bash hygel_run.sh maker_soft_em.yaml
bash hygel_run.sh maker_soft_md.yaml
```
