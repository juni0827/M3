# 토큰-크리틱 헤드 설계 문서

## 개요

토큰-크리틱 헤드는 LLM의 각 토큰별로 Q값을 예측하여 샘플링 시 로짓을 미세 보정하는 메커니즘입니다. 이를 통해 탐색/수렴의 동적 균형을 맞추고 λ-크레딧 할당을 통한 미분 가능한 학습 경로를 제공합니다.

## 구성 요소

### 1. 모델 아키텍처

#### TorchConversationalPolicy.Model

```python
class Model(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden: int):
        # ... 기존 구성 ...
        
        # 문장 레벨 value 헤드 (기존)
        self.value = nn.Linear(hidden, 1)
        
        # 토큰 레벨 Q-value 헤드 (신규)
        self.token_value = nn.Linear(hidden, vocab_size)
```plaintext

- **value**: 문장 전체의 스칼라 가치 예측 (V(s))
- **token_value**: 각 토큰별 Q값 예측 (Q(s, a_i) for each token i)

### 2. 샘플링 시 로짓 보정

#### generate() 메서드 수정

```python
for _ in range(max_len):
    # 1. 기본 로짓 계산
    logits = self.model.head(dec_t)  # (1, V)
    
    # 2. 토큰별 Q값 계산 (detach - 샘플링 시에는 그래디언트 전파 안 함)
    qtok = self.model.token_value(dec_t).detach()  # (1, V)
    
    # 3. 누적된 토큰별 advantage 가져오기
    adv = self._adv_headroom(logits.shape[-1])  # (1, V) or None
    
    # 4. 로짓 보정
    if adv is not None:
        logits = logits + α*qtok + β*adv
    else:
        logits = logits + α*qtok
    
    # 5. 샘플링
    tok = self._sample(logits, temperature, top_k, top_p)
```plaintext

**계수 튜닝**:

- `α` (token_q_alpha): 토큰 Q값의 영향력 (기본값: 0.1)
- `β` (token_adv_alpha): Advantage의 영향력 (기본값: 0.5)

### 3. 토큰별 Advantage 버퍼

#### 구조

```python
self._token_adv_buffer: Dict[int, float] = {}  # token_id -> accumulated advantage
```plaintext

#### 업데이트 메커니즘

```python
def update_token_advantages_from_credit(self, response: str, credit: float):
    """버스에서 받은 credit을 토큰별로 균등 분배"""
    token_ids = self.tok.encode(response, add_special=False)
    per_token_credit = credit / max(1, len(token_ids))
    
    for tok_id in token_ids:
        self._token_adv_buffer[tok_id] += per_token_credit
```plaintext

#### Decay 메커니즘

```python
def _adv_headroom(self, vocab_size: int):
    """버퍼에서 advantage 벡터 생성 및 감쇠"""
    adv_vec = torch.zeros(vocab_size)
    for tok_id, adv_val in self._token_adv_buffer.items():
        adv_vec[tok_id] = adv_val
    
    # 감쇠 (기본 0.95)
    self._token_adv_buffer = {
        k: v * self._token_adv_decay 
        for k, v in self._token_adv_buffer.items()
    }
    
    # 작은 값 제거
    self._token_adv_buffer = {
        k: v for k, v in self._token_adv_buffer.items() 
        if abs(v) > 1e-4
    }
    
    return adv_vec.unsqueeze(0)
```

### 4. 학습

#### 오프라인 학습 (Teacher-Forcing)

```python
def train_token_value_head(self, records: List[Dict], epochs: int = 1):
    """
    records: [{'prompt': str, 'response': str, 'token_credits': List[float]}]
    """
    for rec in records:
        # 1. 인코딩
        src_ids = encode(prompt)
        tgt_in = encode(response)[:-1]
        tgt_out = encode(response)[1:]
        
        # 2. Forward pass
        o, _ = self.decoder(tgt_in_emb, encoder_hidden)  # (1, T, H)
        
        # 3. 토큰별 Q값 예측
        token_q_pred = self.token_value(o)  # (1, T, V)
        
        # 4. 실제 생성된 토큰에 대한 Q값 추출
        pred_q = gather(token_q_pred, tgt_out)  # (1, T)
        
        # 5. Target Q값 (크레딧)
        target_q = tensor(token_credits)  # (1, T)
        
        # 6. MSE Loss
        loss = mse(pred_q, target_q)
        loss.backward()
        optimizer.step()
```

#### 온라인 학습 (Credit Assignment)

```python
# 1. 생성 시 span 등록
span_id = register_generation_span(response)

# 2. 버스로부터 크레딧 수신
def _process_credit_messages():
    msgs = bus.receive_all('llm_adapter')
    for msg in msgs:
        if msg.type == 'credit':
            span_id = msg.payload['span_id']
            credit = msg.payload['credit']
            
            # 토큰별로 분배
            buffer = self._span_credit_buffer[span_id]
            per_token = credit / len(buffer)
            for i in range(len(buffer)):
                buffer[i] += per_token
            
            # Advantage 버퍼 업데이트
            self.conv_policy.update_token_advantages_from_credit(
                response, credit
            )

# 3. 주기적 학습
if step % TRAIN_EVERY == 0:
    records = build_records_from_span_buffers()
    self.conv_policy.train_token_value_head(records)
```

## 환경 변수

### 토큰 Q값 관련

- `LLM_ADAPTER_TOKEN_Q_ALPHA` (기본: 0.1)
  - 토큰 Q값의 로짓 보정 계수
  - 높을수록 Q값의 영향력 증가

- `LLM_ADAPTER_TOKEN_ADV_ALPHA` (기본: 0.5)
  - Advantage의 로짓 보정 계수
  - 높을수록 최근 크레딧 이력의 영향력 증가

- `LLM_ADAPTER_TOKEN_ADV_DECAY` (기본: 0.95)
  - Advantage 버퍼의 감쇠율
  - 매 샘플링마다 버퍼 값에 곱해짐

### 학습 관련

- `LLM_ADAPTER_TOKEN_VALUE_TRAIN_EVERY` (기본: 50)
  - 토큰 value 헤드 학습 주기 (메시지 처리 횟수 기준)

- `LLM_ADAPTER_TOKEN_VALUE_MAX_SAMPLES` (기본: 32)
  - 한 번에 학습할 최대 샘플 수

- `LLM_ADAPTER_TOKEN_VALUE_EPOCHS` (기본: 1)
  - 토큰 value 헤드 학습 에폭 수

## 작동 흐름

### 1. 생성 단계

``
User Input
    ↓
LLMAdapter.generate()
    ↓
[For each token position]
    ↓

1. Compute base logits: head(hidden)
2. Compute token Q: token_value(hidden).detach()
3. Get advantage: _adv_headroom(vocab_size)
4. Combine: logits + α*Q + β*adv
5. Sample token
    ↓
Response
    ↓
Register span with MessageBus

### 2. 크레딧 수신 단계

MessageBus
    ↓
Credit Message {span_id, credit}
    ↓
_process_credit_messages()
    ↓

1. Accumulate in span_credit_buffer
2. Update token_adv_buffer (uniform distribution)
3. (Optional) Trigger periodic training

### 3. 학습 단계

Span Buffers (prompt, response, token_credits)
    ↓
_maybe_train_token_value_from_spans()
    ↓

1. Collect records from active spans
2. Teacher-forcing forward pass
3. Predict Q for each token
4. MSE loss vs. credit targets
5. Backprop (token_value head only)
    ↓
Updated token_value weights

## 장점

1. **동적 탐색/수렴 균형**
   - Q값이 높은 토큰: 더 자주 샘플링 → 수렴
   - Q값이 낮지만 advantage가 높은 토큰: 탐색 기회

2. **미분 가능한 크레딧 경로**
   - 토큰별 크레딧을 직접 역전파
   - λ-return과 자연스럽게 통합

3. **점진적 개선**
   - Advantage 버퍼로 최근 경험 빠르게 반영
   - Teacher-forcing으로 안정적 장기 학습

4. **계산 효율성**
   - 샘플링 시 detach() → 추가 메모리 부담 없음
   - 주기적 학습으로 overhead 최소화

## 튜닝 가이드

### 탐색 강화

```bash
export LLM_ADAPTER_TOKEN_Q_ALPHA=0.05      # Q값 영향 감소
export LLM_ADAPTER_TOKEN_ADV_ALPHA=0.8     # Advantage 영향 증가
export LLM_ADAPTER_TOKEN_ADV_DECAY=0.9     # 빠른 감쇠로 최신 경험 선호
```

### 수렴 강화

```bash
export LLM_ADAPTER_TOKEN_Q_ALPHA=0.2       # Q값 영향 증가
export LLM_ADAPTER_TOKEN_ADV_ALPHA=0.3     # Advantage 영향 감소
export LLM_ADAPTER_TOKEN_ADV_DECAY=0.98    # 느린 감쇠로 안정화
```

### 빠른 적응

```bash
export LLM_ADAPTER_TOKEN_VALUE_TRAIN_EVERY=20  # 자주 학습
export LLM_ADAPTER_TOKEN_VALUE_MAX_SAMPLES=64  # 더 많은 샘플
export LLM_ADAPTER_TOKEN_VALUE_EPOCHS=2        # 더 많은 에폭
```

## 향후 개선 방향

1. **가중 크레딧 분배**
   - 현재: 균등 분배
   - 개선: 토큰 중요도에 따른 가중 분배 (attention weights 활용)

2. **Multi-step Q-learning**
   - 현재: 1-step TD
   - 개선: n-step return or λ-return

3. **Priority Buffer**
   - 현재: FIFO
   - 개선: TD-error 기반 우선순위 샘플링

4. **Separate Value/Advantage Heads**
   - 현재: Q(s,a) 직접 예측
   - 개선: V(s) + A(s,a) Dueling 구조
