# 02 OPLS to Martini

이 example은 이미 존재하는 OPLS/GROMACS production data를 Bartender fitting에 재사용하는 흐름을 보여줍니다.

실제 진입점은 `project/`입니다.

```bash
cd /path/to/hygel_martini/example/02_opls_to_martini/project
bash run_opls_to_martini.sh config/opls_existing_data.yaml
```

실제 사용에서는 mode wrapper가 더 편합니다.

```bash
MODE=setup bash run_existing_opls.sh
MODE=md bash run_existing_opls.sh
MODE=md_notrim bash run_existing_opls.sh
```

이 workflow는 OPLS input 생성이나 GROMACS MD run을 하지 않습니다. `.xtc/.trr + .tpr (+ .edr)` 또는 이미 변환된 `.pdb` trajectory를 받아 `trim/run_prepare_md.sh`와 `bartender_job/run_bartender.sh`를 생성합니다.

자세한 사용법은 `project/README.md`를 보세요.
