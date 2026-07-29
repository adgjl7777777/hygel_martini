# QM/xTB to Martini workflow

이 패키지는 xTB 기반 conformational ensemble을 준비하고 bonded term
후보를 비교하는 Stage 03 workflow입니다. 계산이 끝났다는 사실,
분석 가능한 trajectory라는 사실, 고수준 참조와 일치한다는 사실,
최종 Martini 파라미터가 검증됐다는 사실을 서로 다른 evidence
class로 취급합니다.

## 구성

- `cli.py`, `generator.py`, `defaults.py`
  Stage 03 입력 생성과 workflow 진입점
- `workflow_logic/`
  실행 분기와 bonded-term screening 규칙
- `analysis/`
  trajectory 요약, sweep 비교, trim sensitivity, 고수준 참조 적합성
  gate
- `tools/`
  workflow 보조 도구

## 권장 판단 순서

1. xTB trajectory와 산출물의 계산·구조 무결성을 확인합니다.
2. frozen rule로 단일-chain 및 multi-chain bonded 후보를 screening합니다.
3. 대표 구조를 사전에 고정하고 DFT 등 고수준 단일점/gradient/독립
   최적화를 계산합니다.
4. `reference_qualification`으로 상대에너지, stationarity, endpoint
   family, overlap을 각각 판정합니다.
5. 통과한 범위에서만 작은 수의 보정 계수를 fitting합니다.
6. fitting에 열지 않은 test 구조와 solution-state observable로
   validation합니다.

4번 gate는 다음처럼 실행합니다.

```bash
python -m \
  hygel_martini.param_opt.qm_to_martini.analysis.reference_qualification \
  --help
```

CSV 형식과 각 decision의 해석은
[`analysis/README.md`](analysis/README.md)를 봅니다.

## 주장 경계

- xTB ensemble은 screening 기준이 될 수 있지만 궁극적 ground truth로
  자동 승격되지 않습니다.
- isolated-fragment DFT는 model-chemistry discrepancy를 정량화할 수
  있지만 solution-state charged effect를 단독으로 검증하지 못합니다.
- 일반 OPLS 사용의 어려움이나 zwitterion의 electrostatics 중요성은
  bespoke parameter의 필요성을 설명하지만, 그 파라미터의 정확성을
  대신 입증하지 않습니다.
- 최종 parameter claim에는 미사용 test set과 조건 일치 solution
  validation이 필요합니다.
