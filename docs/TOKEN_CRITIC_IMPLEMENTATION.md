# 토큰-크리틱 헤드 구현 완료

## 구현 내용

토큰-크리틱 헤드가 성공적으로 구현되었습니다. 이 시스템은 문장 레벨 value와 토큰별 Q값을 결합하여 샘플링 직전에 로짓을 미세 보정합니다.

## 주요 변경사항

### 1. llm_adapter.py

#### Model 클래스 확장

```python
class TorchConversationalPolicy.Model:
    def __init__(self, ...):
        # 기존: 문장 레벨 value
        self.value = nn.Linear(hidden, 1)
        
        # 신규: 토큰 레벨 Q-value
        self.token_value = nn.Linear(hidden, vocab_size)
```

#### 초기화 파라미터 추가

- `_token_adv_buffer`: 토큰별 누적 advantage 딕셔너리
- `_token_adv_decay`: Advantage 감쇠율 (기본: 0.95)
- `_token_q_alpha`: 토큰 Q값 계수 (기본: 0.1)
- `_token_adv_alpha`: Advantage 계수 (기본: 0.5)
- `token_value_opt`: 토큰 value 헤드 전용 옵티마이저

#### 새 메서드

### _adv_headroom(vocab_size)

- 토큰별 advantage 버퍼에서 벡터 생성
- 매 호출마다 버퍼 감쇠 및 정리
- 반환: (1, V) 텐서 또는 None

#### train_token_value_head(records, epochs, mem)

- Teacher-forcing 방식으로 토큰 value 학습
- 각 토큰 위치에서 예측한 Q값을 크레딧 타겟과 MSE 최소화
- records: [{prompt, response, token_credits}]

#### update_token_advantages_from_credit(response, credit)

- 버스로부터 받은 크레딧을 토큰별로 균등 분배
- advantage 버퍼에 누적

### _maybe_train_token_value_from_spans(max_samples, epochs)

- 주기적으로 span 버퍼에서 학습 데이터 수집
- train_token_value_head 호출
- 학습 완료된 span 정리

#### generate() 메서드 수정

```python
for _ in range(max_len):
    logits = self.model.head(dec_t)                 # (1, V)
    qtok = self.model.token_value(dec_t).detach()   # (1, V)
    adv = self._adv_headroom(logits.shape[-1])      # (1, V) or None
    
    # 로짓 보정
    if adv is not None:
        logits = logits + 0.1*qtok + 0.5*adv
    else:
        logits = logits + 0.1*qtok
    
    tok = self._sample(logits, ...)
```

#### _process_credit_messages() 확장

- 크레딧 수신 시 `update_token_advantages_from_credit` 호출
- 주기적으로 `_maybe_train_token_value_from_spans` 호출

#### Span 관리 개선

- `_active_spans`에 'response' 필드 추가
- 생성 완료 후 response 저장하여 크레딧 처리 시 사용

## 환경 변수

새로 추가된 환경 변수:

```bash
# 토큰 Q값 관련
export LLM_ADAPTER_TOKEN_Q_ALPHA=0.1           # Q값 보정 계수
export LLM_ADAPTER_TOKEN_ADV_ALPHA=0.5         # Advantage 보정 계수
export LLM_ADAPTER_TOKEN_ADV_DECAY=0.95        # Advantage 감쇠율

# 학습 관련
export LLM_ADAPTER_TOKEN_VALUE_TRAIN_EVERY=50  # 학습 주기
export LLM_ADAPTER_TOKEN_VALUE_MAX_SAMPLES=32  # 최대 샘플 수
export LLM_ADAPTER_TOKEN_VALUE_EPOCHS=1        # 에폭 수
```

## 작동 흐름

### 샘플링 단계

1. 디코더 hidden state 계산
2. 기본 로짓 계산 (head)
3. 토큰 Q값 계산 (token_value, detached)
4. Advantage 버퍼에서 벡터 가져오기
5. 로짓 보정: logits + α*Q + β*adv
6. Top-k/Top-p 샘플링

### 크레딧 수신 단계

1. MessageBus로부터 credit 메시지 수신
2. Span의 크레딧 버퍼에 누적
3. 토큰별로 균등 분배하여 advantage 버퍼 업데이트

### 학습 단계

1. 일정 주기마다 실행
2. Span 버퍼에서 (prompt, response, token_credits) 수집
3. Teacher-forcing으로 각 토큰 위치의 Q값 예측
4. MSE loss로 token_value 헤드만 업데이트

## 장점

1. **동적 탐색/수렴**
   - Q값: 장기적 가치 학습
   - Advantage: 최근 경험 빠른 반영

2. **미분 가능 경로**
   - 토큰별 크레딧을 직접 역전파
   - λ-return과 자연스럽게 통합

3. **효율성**
   - 샘플링 시 detach로 메모리 절약
   - 주기적 학습으로 overhead 최소화

4. **점진적 개선**
   - Advantage 버퍼로 즉각 반영
   - Teacher-forcing으로 안정적 학습

## 테스트 방법

```python
from llm_adapter import attach_llm_to_core

# Core 생성 및 LLM 연결
core = M3Core(...)
adapter = attach_llm_to_core(core)

# 생성 테스트
response = adapter.generate("What is the capital of France?")

# 크레딧 시뮬레이션 (실제로는 MessageBus를 통해)
# adapter.conv_policy.update_token_advantages_from_credit(response, 1.5)

# Advantage 버퍼 확인
print(adapter.conv_policy._token_adv_buffer)

# 학습 데이터 수동 생성
records = [{
    'prompt': 'Test prompt',
    'response': 'Test response',
    'token_credits': [0.5, 0.8, 1.2, 0.9]  # 토큰별 크레딧
}]
adapter.conv_policy.train_token_value_head(records, epochs=1)
```

## 향후 개선 방향

1. **가중 크레딧 분배**: Attention weights 활용
2. **Multi-step Q-learning**: n-step return
3. **Priority Buffer**: TD-error 기반 샘플링
4. **Dueling Architecture**: V(s) + A(s,a) 분리

## 참고 문서

- [TOKEN_CRITIC_DESIGN.md](./TOKEN_CRITIC_DESIGN.md) - 상세 설계 문서
- [RULES_README.md](../TEXT/RULES_README.md) - M3 아키텍처 개요
