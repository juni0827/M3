# M3 LLM Adapter 업그레이드 요약

## 구현 완료 항목

### A. kNN-LM 키 안정화 + φ-aware 혼합 ✅

1. **고정 프로젝션 행렬 캐시**
   - `TorchConversationalPolicy.__init__`에 `self._R` 추가
   - `_build_cond_key()` 수정: 차원 변경 시에만 재생성, 이후 캐시된 행렬 재사용
   - 성능: 매 생성마다 랜덤 행렬 재생성 오버헤드 제거

2. **φ-aware 혼합 규칙**
   - `_alpha_scheduler_uncertainty()` 메소드 구현
   - 불확실도(엔트로피), 안정도 역, φ-delta 음수(퇴행)을 sigmoid 조합
   - 환경변수 설정:
     - `KNN_ALPHA_BASE`: 0.25 (기본 α)
     - `KNN_ALPHA_UNCERTAINTY_COEF`: 0.3
     - `KNN_ALPHA_INSTABILITY_COEF`: 0.25
     - `KNN_ALPHA_PHI_COEF`: 0.4
   - 분포 혼합: `P_mix = (1-α)·P + α·P_knn` (확률 공간)

### B. 정확한 스팬-토큰 정렬 + 가중 크레딧 분배 ✅

1. **정확한 토큰 카운트**
   - `tok.encode(response, add_special=True)` 로 실제 토큰 수 계산
   - `_global_token_idx` 단조 증가 관리 (생성 루프 내 1스텝마다 증가)
   - 스팬에 정확한 `token_range=(start, end)` 기록
   - MessageBus `close_span(span_id, token_end)` 호출

2. **가중 크레딧 분배**
   - `_token_importance_weights(response)` 메소드 구현
     - 툴 호출 토큰: 2.0 가중치
     - 키워드/근거 토큰: 1.5 가중치
     - 일반 토큰: 1.0 가중치
   - `_process_credit_messages()`에서 가중치 정규화 후 분배: `credit * w[i] / Σw`
   - 토큰별 중요도 기반 차등 크레딧 할당

### C. 메모리(패널)-어텐션 게이팅 + 정규화 ✅

1. **게이팅 스칼라 g**
   - Model에 `gate_proj: Linear(hidden, 1)` 추가
   - `LayerNorm(hidden)` 추가 (over-attention 방지)
   - 초기 바이어스 -1.0 → g ≈ 0.27 (sigmoid(-1))

2. **compute_mem_context() 메소드**
   - 어텐션: `att = softmax(Q·K^T / √H)`
   - LayerNorm 정규화: `dec_t_norm`, `ctx_norm`
   - 게이팅 g 계산: `g = sigmoid(gate_proj(dec_t_norm))`
   - 동적 bias 추가:
     - 불안정도: `+0.3·(1-stability)`
     - drift: `+0.2·drift`
     - φ 퇴행: `+0.3·(-phi_delta)`
   - 혼합: `(1-g)·h + g·ctx`

3. **generate() 적용**
   - core_state 수집: stability, drift, phi_delta
   - 게이팅 메모리 컨텍스트 계산 후 logits 생성

### D. 토큰-밸류 로짓 보정 + β 스케줄링 ✅

1. **β 스케줄러**
   - `_beta_schedule()` 메소드 구현
   - φ 증가 → β 감소 (모델 신뢰 증가)
   - 안정도 증가 → β 감소
   - 툴 성공 → β 감소
   - EMA 업데이트: `β_ema = 0.9·β_ema + 0.1·β_target`
   - 환경변수:
     - `LLM_ADAPTER_BETA_INIT`: 0.1
     - `LLM_ADAPTER_BETA_MIN`: 0.01
     - `LLM_ADAPTER_BETA_MAX`: 0.5

2. **로짓 보정 적용**
   - `logits' = logits + β·Q_token(t) + α_adv·adv`
   - β가 동적으로 조정되어 상황에 따라 토큰-밸류 신뢰도 변경

### E. 멀티태스크 동적 가중 및 캘리브레이션 ✅

1. **GradNorm 가중치 조정**
   - `_update_task_weights_gradnorm()` 메소드 구현
   - 상대 학습률(r_i = L_i(t) / L_i(0)) 기반 가중치 조정
   - α=1.5 비대칭 파라미터로 느린 수렴 태스크에 높은 가중치
   - 가중치 범위: [0.1, 5.0]
   - 10 샘플마다 업데이트
   - 환경변수: `GRADNORM_ALPHA=1.5`

2. **타겟 정규화**
   - `_normalize_targets()` 메소드 구현
   - φ_delta: log 변환 (부호 보존) → z-score [-3, 3]
   - stability_delta: sqrt 변환 → z-score [-3, 3]
   - 이상치 억제 및 학습 안정성 향상

3. **train_value_head() 통합**
   - 개별 손실 계산: loss_phi, loss_stab, loss_tool
   - 동적 가중치 적용: w_phi·loss_phi + w_stab·loss_stab + w_tool·loss_tool
   - GradNorm 업데이트 통합
   - 태스크 손실 히스토리 추적 (최근 50개)

### F. DPO 배치 학습 및 하드음성 필터 ✅

1. **샘플 믹싱 전략**
   - `dpo_batch_step()` 메소드 구현
   - Recent 50% + Past 30% + Random 20% 샘플링
   - 시간적 다양성 확보 (최근 편향 완화)

2. **하드 네거티브 필터링**
   - `tool_failure=True` 샘플 2x 복제 → rejection 가중치 증가
   - 도구 실패 케이스에 강한 선호도 학습

3. **미니배치 학습**
   - 배치 크기: 4 (기본값, 조정 가능)
   - 배치 내 손실 평균 후 역전파
   - Gradient clipping: max_norm=1.0

4. **φ-margin 통합**
   - 기존 dpo_step()의 φ-margin 로직 유지
   - v_phi 예측 차이를 preference signal로 활용

### G. 로깅/가시화 보강 ✅

1. **스팬-크레딧 메트릭**
   - `_process_credit_messages()`에 추가:
     - `credit_var`: 토큰별 크레딧 분산 (불균형 측정)
     - `credit_max`: 최대 크레딧 (중요 토큰 식별)
     - `credit_entropy`: 크레딧 엔트로피 (분포 복잡도)
   - 디버그 로그에 span_id, credit, tokens, var, max, entropy 기록

2. **α/g/β 시계열 로깅**
   - generate() 스텝 로깅에 추가:
     - `knn_alpha`: kNN 혼합 계수 α
     - `gate_g`: 메모리 게이팅 계수 g (mean)
     - `token_value_beta`: 토큰-밸류 보정 계수 β
   - `llm_adapter.jsonl` (환경변수 `LLM_ADAPTER_LOG`)에 기록

3. **φ-delta 상관도 분석**
   - train_value_head()에 추가:
     - 에폭 종료 후 Pearson correlation(phi_targets, phi_preds) 계산
     - logger.info로 출력: n_samples, avg_loss, phi_correlation
     - JSONL 로그에 value_train 이벤트 기록 (phi_correlation, task_weights 포함)

### 추가 개선 구현 요약

- **_last_value_estimates** 딕셔너리: φ/안정도/툴 성공 예측치 캐시 (α, β 스케줄러용)
- value-head 추정치를 generate() 직후 업데이트
- 모든 스케줄러가 실시간 core 상태 반영
- **_task_weights**: 멀티태스크 동적 가중치 딕셔너리 {'phi': 1.0, 'stab': 0.5, 'tool': 0.5}
- **_task_losses_history**: GradNorm용 손실 히스토리 (최근 50개)
- 정확한 토큰 카운팅: 생성 루프 내 `_global_token_idx += 1`

## 환경 변수 설정 가이드

```bash
# kNN-LM α 스케줄러
export KNN_ALPHA_BASE=0.25
export KNN_ALPHA_UNCERTAINTY_COEF=0.3
export KNN_ALPHA_INSTABILITY_COEF=0.25
export KNN_ALPHA_PHI_COEF=0.4

# 토큰-밸류 β 스케줄러
export LLM_ADAPTER_BETA_INIT=0.1
export LLM_ADAPTER_BETA_MIN=0.01
export LLM_ADAPTER_BETA_MAX=0.5

# 멀티태스크 GradNorm
export GRADNORM_ALPHA=1.5

# DPO 설정
export DPO_BETA=0.1

# 로깅
export LLM_ADAPTER_LOG=llm_adapter.jsonl

# 기존 설정 유지
export LLM_ADAPTER_TOKEN_ADV_ALPHA=0.5
export LLM_ADAPTER_KNN_TAU=0.10
export LLM_ADAPTER_KNN_K=8
```

## 핵심 개선 효과

1. **안정성**: 고정 프로젝션 행렬로 kNN 키 일관성 확보
2. **적응성**: φ-aware α, β 스케줄링으로 상황 기반 동적 조정
3. **정확성**: 정확한 토큰 정렬로 크레딧 누출/오할당 방지
4. **효율성**: 가중 크레딧 분배로 중요 토큰에 학습 집중
5. **견고성**: 게이팅 + LayerNorm으로 메모리 과의존 방지
6. **균형성**: GradNorm으로 멀티태스크 학습 균형 유지
7. **선호도**: DPO 배치 학습으로 하드 네거티브 강화
8. **가시성**: α/β/g/φ 시계열 로깅으로 디버깅 및 분석 용이

## 다음 단계 권장 사항

1. **시각화 도구**: α/β/g 값 추이를 플롯하는 스크립트 개발 (matplotlib/plotly)
2. **A/B 테스트**: 업그레이드 전후 비교 실험 (φ 증가율, 툴 성공률, 안정도)
3. **Huber 손실**: 토큰-밸류 학습에 도입하여 이상치 견고성 확보
4. **자동 하이퍼파라미터 튜닝**: GradNorm α, β 범위 등을 Optuna로 최적화
5. **온라인 DPO**: 생성 중 실시간 선호도 업데이트 (mini-batch 축적)
6. **메트릭 대시보드**: φ_correlation, credit_entropy 등을 Tensorboard/Weights&Biases에 통합

## 코드 변경 파일

- `llm_adapter.py`: 주요 업그레이드 모두 적용됨 (A-G)
- 추가 파일 없음 (모놀리식 통합)

## 테스트 방법

```python
# M3 core 실행 후
from llm_adapter import attach_llm_to_core

core = M3ConsciousnessCore()
attach_llm_to_core(core)

# 대화 테스트
response = core.llm.generate("Explain quantum entanglement")
print(response)

# 로그 확인
# llm_adapter.jsonl 에서:
# - gen_step 이벤트: knn_alpha, gate_g, token_value_beta 추이
# - value_train 이벤트: phi_correlation, task_weights 확인

# DPO 배치 학습 테스트
samples = [
    {'prompt': 'What is 2+2?', 'chosen': 'The answer is 4.', 'rejected': 'I don\'t know.', 'tool_failure': False},
    {'prompt': 'Calculate sqrt(16)', 'chosen': '4', 'rejected': 'Error', 'tool_failure': True},
]
core.llm.conv_policy.dpo_batch_step(samples, beta=0.1, batch_size=2)

# 로그 확인 - 스팬 크레딧 메트릭
# DEBUG 로그에서 "Span ... var=... max=... entropy=..." 확인
```

## 구현 세부 사항

### 메소드 추가/수정 목록

**llm_adapter.py**:

- `_beta_schedule()`: β 동적 스케줄링
- `_update_task_weights_gradnorm()`: GradNorm 가중치 조정
- `_normalize_targets()`: φ/stability 타겟 정규화
- `_alpha_scheduler_uncertainty()`: φ-aware α 스케줄링 (기존 수정)
- `dpo_batch_step()`: DPO 미니배치 학습 (신규)
- `train_value_head()`: 멀티태스크 동적 가중 + φ 상관도 로깅 (수정)
- `_process_credit_messages()`: 스팬-크레딧 메트릭 추가 (수정)
- `generate()`: 토큰 카운팅, α/β/g 로깅 (수정)

### 주요 알고리즘

**GradNorm**:

r_i(t) = L_i(t) / L_i(0)
r̃_i = r_i^α / mean(r_j^α)
w_i ← w_i · (1 + 0.1·(r̃_i - 1))
w_i ∈ [0.1, 5.0]

**타겟 정규화**:

φ_norm = clip(sign(φ_δ)·log(1 + |φ_δ|) / 2.0, -3, 3)
stab_norm = clip(sign(stab_δ)·sqrt(|stab_δ|) / 1.0, -3, 3)

**DPO 샘플 믹싱**:

Recent: samples[-50%:]
Past: samples[:30%]
Random: sample(중간 범위, 20%)
Hard Negative: if tool_failure → 2x 복제

---

**구현 날짜**: 2025-11-05  
**최종 업데이트**: 2025-11-05  
**버전**: M3 LLM Adapter v2.1 (φ-aware + Token-Critic + Gated Memory + Multi-task + DPO Batch + Enhanced Logging)
