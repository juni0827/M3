# M3 피드백/샘플링 정합성 개선 PR 실행 계획

## 목표
다음 8개 이슈를 실제 병합 가능한 PR 단위로 분해해, 안전하게 순차 반영한다.

- Φ→Qualia 피드백 경로 일관성
- Energy↔Φ/qualia 대사 결합
- qualia.entropy 의미 분리
- affect_kernel vs qualia 인터페이스 정합
- 테스트 더미 반환 타입 일치
- 디코딩 마이크로 업데이트 하드코딩 완화
- IITPhiCalculator vs CES 역할 명확화
- Φ 스케일/정규화 정합

---

## PR #1 — 안전성 패치(타입/크래시/회귀 방지)

### 범위
- `llm_adapter/llm_core.py`
  - affect 입력 정규화 유틸 추가 (dict/list/np 모두 처리)
  - `_compute_temperature`, `_compute_top_k`에서 안전 인덱싱 적용
- `tests/test_m3_plan_features.py`
  - `_DummyCore.affect_kernel.get_state()`를 5차원 벡터 반환으로 정정
  - 타입 방어 회귀 테스트 추가

### 커밋 분할
1. `fix(adapter): normalize affect state adapter input shape`
2. `test(adapter): align DummyCore affect_kernel return type and add regression`

### 테스트 케이스 이름
- `test_adaptive_sampler_accepts_affect_dict_without_crash`
- `test_adaptive_sampler_topk_affect_vector_indexing_is_stable`
- `test_dummy_core_affect_kernel_returns_5d_vector`

### 리스크
- 기존에 dict 순서에 의존한 비명시적 동작이 변경될 수 있음.

### 롤백 플랜
- adapter 입력 정규화 유틸만 revert (PR #1 단독 revert 가능)
- 테스트는 유지 가능(행동 명세로 재사용)

---

## PR #2 — 의미 분리(entropy 이원화)

### 범위
- `llm_adapter/llm_core.py`
  - 디코딩 토큰 다양도용 상태를 `token_entropy`(또는 `decode_entropy`)로 분리
  - `qualia.entropy` 직접 overwrite 금지
- 필요 시 상태 스냅샷/로그 경로에 신규 필드 추가

### 커밋 분할
1. `refactor(adapter): split qualia entropy from token entropy`
2. `test(adapter): add entropy semantics separation tests`

### 테스트 케이스 이름
- `test_micro_update_updates_decode_entropy_not_qualia_entropy`
- `test_sampler_prefers_decode_entropy_when_present`

### 리스크
- 샘플링 동작(temperature/top-k)이 이전과 달라질 수 있음.

### 롤백 플랜
- feature flag로 기존 entropy 경로 fallback (`M3_HF_USE_LEGACY_ENTROPY=1`)
- 이상 시 env toggle만으로 즉시 복귀

---

## PR #3 — 마이크로 업데이트 동역학 정합(energy coupling)

### 범위
- `llm_adapter/llm_core.py`
  - `_micro_update_step_state`의 고정 에너지 차감 제거
  - `core.energy_ctrl` API(가능하면 `compute_cognitive_cost`/`update_energy`)를 호출하는 경량 경로로 교체
  - interval/warmup을 config 기반으로 노출

### 커밋 분할
1. `refactor(adapter): replace fixed energy drain with energy_ctrl-coupled update`
2. `feat(config): expose micro update coefficients and warmup knobs`
3. `test(adapter): verify energy dynamics coupling during decode`

### 테스트 케이스 이름
- `test_micro_update_uses_energy_controller_when_available`
- `test_micro_update_interval_respected`
- `test_micro_update_noop_when_core_missing`

### 리스크
- 디코딩 중 latency 증가 가능(컨트롤러 호출 비용)

### 롤백 플랜
- `M3_HF_MICRO_UPDATE_MODE=legacy` 토글 유지
- 성능 회귀 시 legacy 모드로 즉시 전환

---

## PR #4 — Φ 경로 일관성 및 스케일 정합

### 범위
- `m3/m3_core.py`
  - `_single_consciousness_step`에서 `compute_phi_simple` 호출 제거/수정 (`compute_phi`로 통일)
  - phi history 스케일 정책 명시(0~1 정규화 혹은 adapter-side norm)
- `llm_adapter/llm_core.py`
  - `phi_influence` 적용 전 정규화 함수 적용(동적 quantile 가능)

### 커밋 분할
1. `fix(core): unify phi compute call path to compute_phi`
2. `feat(adapter): apply phi normalization before sampler influence`
3. `test(core,adapter): phi path consistency and bounded influence`

### 테스트 케이스 이름
- `test_single_step_uses_compute_phi_api`
- `test_phi_influence_bounded_after_normalization`
- `test_phi_feedback_reaches_qualia_message_bus`

### 리스크
- phi 기반 행동(온도 조절)이 약해지거나 과해질 수 있음.

### 롤백 플랜
- 정규화 정책을 config로 전환(`phi_norm_mode=off|static|dynamic`)
- 문제 시 `off`로 즉시 revert 없이 운영 복귀

---

## PR #5 — 책임 분리(IITPhiCalculator vs CES)

### 범위
- `m3/m3_core.py` 또는 분리 파일
  - IITPhiCalculator: orchestration/API/메시지브로드캐스트
  - CauseEffectStructure: 저장/추정/근사 알고리즘
  - `method` 인자 실제 분기 반영 또는 제거(명세 일치)

### 커밋 분할
1. `refactor(core): clarify IITPhiCalculator orchestration responsibilities`
2. `refactor(core): isolate CES approximation engine contracts`
3. `test(core): method contract and delegation boundaries`

### 테스트 케이스 이름
- `test_phi_calculator_delegates_storage_to_ces`
- `test_compute_phi_method_argument_has_effect`
- `test_phi_broadcast_contains_trend_and_history_len`

### 리스크
- 내부 API 변경으로 주변 호출부 깨질 수 있음.

### 롤백 플랜
- 이전 클래스 시그니처 호환 shim 유지(1~2 릴리스)
- 문제 시 shim-only 모드로 회귀

---

## 병합 순서(권장)
1. PR #1 (안전성)  
2. PR #2 (entropy 의미 분리)  
3. PR #3 (에너지 결합)  
4. PR #4 (phi 경로/스케일)  
5. PR #5 (구조 리팩터)

> 앞 PR이 뒤 PR의 테스트 기반을 제공하도록 구성. 특히 #1~#3을 먼저 머지하면 #4~#5 리스크가 크게 줄어듦.

---

## 공통 테스트/검증 게이트
- 단위 테스트: 신규 테스트 + 기존 핵심 테스트 스위트
- 스모크 테스트: 짧은 디코딩 루프에서 크래시/NaN/Inf 없음 확인
- 관측성: `phi`, `energy`, `qualia.entropy`, `decode_entropy` 동시 로그 확인

권장 CI 잡 이름:
- `ci-unit-adapter`
- `ci-unit-core-phi`
- `ci-smoke-decode-dynamics`

---

## 운영 리스크 대응
- 모든 행동 변경은 env/config 토글로 감쌀 것
- 롤백 우선순위: 기능 OFF → PR revert
- 관찰 지표:
  - decode latency p95
  - token repetition ratio
  - phi influence effective range
  - energy depletion slope
