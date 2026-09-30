# 개발 코드 업데이트: 0.1.1.dev2

별도 후보에서 검증한 두 수정을 개발 working copy에 적용했다. 기존 frozen 저장소와 실행 중인 MD/설치 환경은 유지했다. 공개 release나 commit/tag는 생성하지 않았다.

1. 빈 monomer 선택이면 불필요한 전체 atom 검색을 생략한다. PEG N2/L56의 GRO·ITP·연결 계획·감사가 바이트 단위로 동일했고, 한 번의 전체 실행은 65.151→17.061초였다. 큰 N4 구성의 속도 배수로 일반화하지 않는다.
2. 실제 CLI의 run seed가 직렬 Numba geometry 난수 stream도 초기화하도록 했다. 기존 Python/NumPy 난수 호출 순서는 유지한다. Sequence iterator의 별도 `SEQUENCE_STRATEGY.seed`는 계속 명시적으로 관리한다.

후보 wheel은 63개 기존 테스트와 17개 실제 CLI process를 통과했다. 같은 seed의 random/alternating/block 구성과 PEG·Pluronic 출력이 일치했고, random 구성의 다른 seed는 다른 GRO를 생성했다. 현재 코드의 변경 파일 5개는 검증된 후보와 hash가 동일하다. 물리 매개변수 파일은 변경하지 않았다.

코드 백업 및 적용 검증:
`/nas_0/software_backup/hygel_builder_reliability_runs_20260930_all/builder_performance_followup/adoption_20260930/`

설치 후보 wheel·패치·실행 검증:
`/nas_0/software_backup/hygel_builder_reliability_runs_20260930_all/builder_performance_followup/rng_seed_followup/si_ready/`

기존 논문 결과는 원래 frozen 버전과 입력을 사용한 기록이다. 이번 변경을 과거 계산에 소급 적용하지 않는다. 현재 N4는 의미 보존을 확인한 별도 dev1 설치 환경에서 진행하고, Q8/Q9는 각각 고정한 기존 MD 입력으로 진행한다. 라이선스·권리자·공개 release 상태는 기존 draft 조건을 유지한다.
