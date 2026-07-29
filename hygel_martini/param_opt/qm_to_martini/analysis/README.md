# Reference qualification analysis

`reference_qualification.py`는 xTB ensemble을 DFT 등 고수준 참조로
보정하기 전에 적용하는 네 개의 독립 gate를 제공합니다. 모든 결과는
JSON으로 출력되며, 기본 threshold는 명령행 옵션으로 바꿀 수 있습니다.

## 1. 상대에너지 gate

입력 CSV:

```csv
chemistry,structure_id,xtb_energy_kj_mol,reference_energy_kj_mol
C,c0,-100.0,-200.0
C,c1,-97.5,-197.0
D,d0,-150.0,-250.0
D,d1,-148.0,-247.5
```

실행:

```bash
hygel-qm-reference-audit energy reference_energies.csv \
  --group-column chemistry \
  --max-error-kj 8.4 \
  --output energy_audit.json
```

각 방법의 에너지를 자체 최소점에서 0으로 옮긴 뒤 최소점 index,
모든 쌍의 ordering, MAE, RMSE, 최대 절대오차를 계산합니다. 기본
`PASS`는 같은 최소점, 완전한 쌍별 ordering 일치, 최대 오차
8.4 kJ/mol 이하를 모두 요구합니다. 이 수치는 자동 보편 기준이 아니라
프로젝트가 명시적으로 채택한 gate이므로 논문 protocol과 함께
고정해야 합니다.

## 2. Gradient stationarity gate

입력 CSV:

```csv
gx,gy,gz
0.000010,-0.000004,0.000002
-0.000008,0.000003,-0.000001
```

실행:

```bash
hygel-qm-reference-audit gradient gradient.csv \
  --units Eh/bohr \
  --rms-threshold 3e-5 \
  --max-threshold 1e-4 \
  --output gradient_audit.json
```

RMS와 최대 절대 gradient 성분이 모두 threshold 이하일 때만
`STATIONARY`입니다. 단일점 에너지가 계산됐다는 사실과 geometry가
그 수준에서 stationary하다는 사실을 분리하기 위한 검사입니다.

## 3. 독립 최적화 endpoint family

입력 CSV:

```csv
endpoint_id,rmsd_nm,delta_energy_kj_mol,integrity
seed1,0.000,0.0,true
seed2,0.031,1.2,true
seed3,0.084,3.5,true
```

여기서 RMSD와 상대에너지는 미리 선언한 representative endpoint에
대한 값이어야 합니다.

```bash
hygel-qm-reference-audit endpoint endpoints.csv \
  --rmsd-threshold-nm 0.05 \
  --energy-threshold-kj 2.0 \
  --output endpoint_audit.json
```

판정은 다음 세 상태 중 하나입니다.

- `STRUCTURAL_INTEGRITY_FAILURE`: 하나 이상의 endpoint가 구조 무결성
  검사를 통과하지 못함
- `SINGLE_DFT_ENDPOINT_FAMILY`: 모든 endpoint가 RMSD와 에너지 gate
  안에 있음
- `MULTIPLE_DFT_ENDPOINTS`: 구조는 유효하지만 하나 이상의 endpoint가
  RMSD 또는 에너지 gate 밖에 있음

다중 endpoint는 실패한 계산이라는 뜻이 아니라, 단일 minimum으로
축약해서 fitting하면 안 된다는 뜻입니다.

## 4. Importance-reweighting overlap

입력은 동일 xTB ensemble 구조에서 계산한
`delta_energy_kj_mol = E_reference - E_xTB`입니다.

```csv
structure_id,delta_energy_kj_mol
s000,0.0
s001,1.2
s002,-0.7
```

```bash
hygel-qm-reference-audit overlap delta_energies.csv \
  --temperature-k 310 \
  --min-ess-fraction 0.20 \
  --max-normalized-weight 0.20 \
  --output overlap_audit.json
```

가중치는 `exp[-(E_reference-E_xTB)/(RT)]`로 계산합니다. 기본
`SUFFICIENT_OVERLAP`은 effective sample size가 전체 구조의 20% 이상이고
어느 한 구조의 정규화 가중치도 0.20을 넘지 않을 때만 부여합니다.
`INSUFFICIENT_OVERLAP`이면 sparse reweighting으로 보정하지 말고 구조
선정 또는 sampling protocol을 다시 설계해야 합니다.

## 재현성 규칙

- CSV의 structure ID와 원본 geometry checksum을 함께 보존합니다.
- threshold, software version, model chemistry, charge/multiplicity,
  solvent model을 결과와 같은 record에 기록합니다.
- training 구조와 미사용 test 구조를 결과를 보기 전에 나눕니다.
- 이 도구가 만든 JSON은 진단 결과이며 fitting 결과가 아닙니다.
- isolated-fragment 통과 결과만으로 condensed/solution 상태의
  electrostatics를 검증했다고 주장하지 않습니다.
