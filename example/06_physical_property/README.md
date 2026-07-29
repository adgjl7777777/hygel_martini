# Stage 06: physical-property analysis

이 예제는 기존 MD output에서 topology, composition, periodic clearance,
finite-rate mechanics, polymer volume fraction을 계산하는 template입니다.
새 simulation을 자동으로 시작하지 않습니다.

1. `analysis_jobs.yaml`의 `/path/to/...` 입력을 실제 파일로 바꿉니다.
2. `template: true`를 `false`로 바꿉니다.
3. requirement gate를 먼저 확인합니다.

```bash
hygel-property requirements \
  --analysis analysis_jobs.yaml \
  --requirements md_requirements.yaml \
  --strict
```

모든 필수 입력이 준비된 뒤 분석합니다.

```bash
hygel-property analyze \
  --analysis analysis_jobs.yaml \
  --requirements md_requirements.yaml \
  --manifest validation_manifest.yaml
```

각 job의 `output.report`에 JSON 결과가 기록됩니다. 계산 실패나 입력
누락은 빈 결과로 넘어가지 않고 status로 구분됩니다.

`pore_size`라는 일반 이름 대신 `periodic clearance`를 사용합니다.
`mechanics-step` 결과도 finite-rate apparent response이며 equilibrium
또는 experimental modulus가 아닙니다.

왜 이 용어와 gate를 사용하는지는 저장소의
`docs/VALIDATION_HISTORY_AND_DESIGN_RATIONALE.md`에 있는 A/B validation
이력과 exclusion ledger를 참고하십시오.
