# 패널 메모리 시스템 설계 문서

## 개요

패널 메모리 시스템은 FeatureBank의 `panels()` 메서드를 통해 Φ_total을 의미층별로 분리된 벡터 리스트로 제공하며, 각 패널은 LLM의 Cross-Attention 메모리 토큰으로 사용됩니다.

## 핵심 개념

### 기존 방식의 한계

- `build()`: 모든 피처를 하나의 평탄한 벡터로 압축 → 의미 구조 손실
- 복제 반복: 단일 벡터를 M번 복제 → 토큰 간 차별성 없음

### 패널 방식의 장점

- **의미층 분리**: 각 패널이 독립적인 의미 영역 표현
  - stability_panel: 안정성/에너지/주체/정서
  - topo_panel: 시각/위상/φ
  - time_panel: 시계열 (RPE, entropy 히스토리)
  - tool_panel: 툴콜 통계
  - bus_panel: MessageBus 활동
  - topic_panel: 언어/주제 임베딩

- **Cross-Attention 활용**: 각 패널이 독립 토큰으로 작용
  - 디코더가 상황에 따라 관련 패널에 선택적 attend
  - 예: 시각 추론 → topo_panel 집중, 감정 생성 → stability_panel 집중

- **확장성**: 새 패널 추가 시 기존 구조 유지

## 구현

### 1. M3.py - FeatureBank.panels()

```python
def panels(self, core: "M3ConsciousnessCore") -> list[np.ndarray]:
    """
    Φ_total을 의미층별 '패널'로 쪼개어 D차 벡터 리스트로 반환한다.
    각 패널 = 메모리 토큰 1개로 사용.
    반환: [v1, v2, ..., vM], 각 v_i.shape == (D,)
    """
    D = int(getattr(self, "embed_dim", 32))  # 패널 임베딩 차원
    
    def _to_panel(x: np.ndarray | list[float] | float) -> np.ndarray:
        """입력을 길이 D의 정규화된 패널로 변환"""
        arr = np.atleast_1d(np.array(x, dtype=np.float32)).ravel()
        # Robust 정규화: (x - μ) / σ → tanh
        mu = float(np.mean(arr))
        sd = float(np.std(arr) + 1e-6)
        z = np.tanh((arr - mu) / sd)
        # 길이 맞추기
        if z.size < D:
            out = np.zeros(D, dtype=np.float32)
            out[:z.size] = z
            return out
        else:
            return z[:D].astype(np.float32)
    
    # 각 의미층 구성
    stability_panel = _to_panel([stability, delta_hat, energy, unity, arousal, valence, ...])
    topo_panel = _to_panel([contrast, entropy, edge_density, phi, ...])
    time_panel = _to_panel([rpe_stats, ent_stats, recent_history, ...])
    tool_panel = _to_panel([tool_success, tool_fail, tool_latency, ...])
    bus_panel = _to_panel([bus_keys, bus_latency, bus_depth, ...])
    topic_panel = _to_panel(language_embedding)
    
    return [stability_panel, topo_panel, time_panel, tool_panel, bus_panel, topic_panel]
```

### 2. llm_adapter.py - 메모리 준비 개선

#### generate() 메서드

```python
# memory from feature_bank.panels(core) - 패널 토큰 사용
mem = None
try:
    if hasattr(self._core, 'feature_bank') and hasattr(self._core.feature_bank, 'panels'):
        # panels() 반환: list[np.ndarray], 각 패널은 (D,) 형태
        panels_list = self._core.feature_bank.panels(self._core)
        if panels_list:
            # Stack panels to (M, D) where M = number of panels
            mem = np.stack(panels_list, axis=0).astype(np.float32)  # (M, D)
    elif hasattr(self._core, 'feature_bank') and hasattr(self._core.feature_bank, 'build'):
        # Fallback: build()로 단일 벡터를 복제
        z = self._core.feature_bank.build(self._core)
        z = np.array(z).astype(np.float32).reshape(1, -1)
        M = int(os.environ.get('LLM_ADAPTER_MEM_TOK', '4'))
        mem = np.repeat(z, M, axis=0)  # (M, D)
except Exception:
    mem = None
```

#### TorchConversationalPolicy.generate() - 메모리 입력 처리

```python
# Prepare memory (if provided)
mem_k = mem_v = None
if mem is not None and len(np.array(mem).shape) >= 1:
    m = np.array(mem)
    # NEW: 허용 형태 (D,), (M,D), (1,M,D)
    if m.ndim == 1:
        m = m[None, :]          # (1, D)  -> single token
    if m.ndim == 2:
        m = m[None, :, :]       # (M, D)  -> (1, M, D)
    # now m.ndim == 3: (1, M, D)
    M, D = int(m.shape[1]), int(m.shape[2])
    m_t = torch.tensor(m, dtype=torch.float32, device=self.device)
    self.model._ensure_mem_layers(D)
    mem_h = self.model.mem_proj(m_t)  # (1, M, H)
    mem_k = self.model.Wk(mem_h)
    mem_v = self.model.Wv(mem_h)
```

## 패널 구조 상세

### Stability Panel (안정성/내적 상태)

- **내용**: `stability, delta_hat, energy_activation, energy_ratio, unity, arousal, valence, entropy`
- **용도**: 감정 생성, 주체 일관성, 에너지 인식 응답
- **예시**: "피곤해요" → energy_ratio 낮음 감지

### Topo Panel (위상/시각/φ)

- **내용**: `vision_contrast, vision_entropy, edge_density, depth_cue, phi_last, phi_delta, phi_mean10, r_mean`
- **용도**: 시각 추론, 공간 이해, 통합정보 참조
- **예시**: "무엇이 보이나요?" → vision 피처 집중

### Time Panel (시계열)

- **내용**: RPE 통계 (평균/분산/최댓값), Entropy 통계, 최근 8스텝 원본
- **용도**: 최근 경험 요약, 트렌드 파악
- **예시**: "최근 어땠나요?" → RPE 히스토리 참조

### Tool Panel (툴콜 활동)

- **내용**: `success_count, fail_count, latency_mean, latency_std, top4_tool_frequencies`
- **용도**: 툴 사용 패턴 인식, 실패 대응
- **예시**: 툴 실패 많으면 → "지금 툴이 불안정합니다"

### Bus Panel (MessageBus 상태)

- **내용**: `top_keys_presence[4], latency_ms, depth`
- **용도**: 시스템 부하 인식, 통신 상태
- **예시**: bus_latency 높으면 → 간결한 응답

### Topic Panel (언어/주제)

- **내용**: `language_embed` (최근 대화 임베딩)
- **용도**: 주제 연속성, 문맥 유지
- **예시**: "그것에 대해 더 말해줘" → 이전 주제 참조

## Cross-Attention 메커니즘

```python
for _ in range(max_len):
    dec_t = decoder_hidden  # (1, H)
    
    # Query from decoder
    q = Wq(dec_t).unsqueeze(1)  # (1, 1, H)
    
    # Keys/Values from memory panels
    mem_k = Wk(mem_proj(panels))  # (1, M, H)
    mem_v = Wv(mem_proj(panels))  # (1, M, H)
    
    # Attention scores
    scores = (q @ mem_k.transpose(1,2)) / sqrt(H)  # (1, 1, M)
    att = softmax(scores)  # (1, 1, M)
    
    # Weighted sum of values
    ctx = att @ mem_v  # (1, 1, H)
    
    # Combine with decoder state
    dec_t = (dec_t + ctx.squeeze(1)) * 0.5
    
    # Continue decoding...


### Attention 분석 예시

**시각 질문**: "무엇이 보이나요?"


Attention weights (softmax):
stability_panel: 0.05
topo_panel:      0.65  ← 높은 가중치
time_panel:      0.10
tool_panel:      0.05
bus_panel:       0.05
topic_panel:     0.10


**감정 표현**: "어떤 기분인가요?"


Attention weights:
stability_panel: 0.70  ← 높은 가중치
topo_panel:      0.05
time_panel:      0.15
tool_panel:      0.02
bus_panel:       0.03
topic_panel:     0.05
```

## 환경 변수

### 기존 변수 (여전히 fallback에 사용)

- `LLM_ADAPTER_MEM_TOK` (기본: 4)
  - panels() 없을 때 build() 벡터 복제 횟수

### 새 변수 (향후 확장)

- `LLM_ADAPTER_PANEL_DIM` (기본: 32)
  - 각 패널의 임베딩 차원 (현재는 embed_dim 사용)

- `LLM_ADAPTER_ENABLE_PANELS` (기본: 1)
  - 패널 시스템 활성화 (0: build() fallback)

## 장점 요약

### 1. 의미 분리

- 각 패널이 독립적 의미 공간
- Cross-Attention으로 동적 선택

### 2. 해석 가능성

- Attention weights로 어떤 패널이 영향을 주는지 추적
- 디버깅/분석 용이

### 3. 확장성

- 새 패널 추가 시 기존 코드 변경 최소화
- 패널별 전처리/정규화 커스터마이징

### 4. 효율성

- 필요한 정보만 선택적으로 사용
- 평탄화된 단일 벡터보다 표현력 높음

## 향후 개선 방향

### 1. 학습 가능한 패널 프로젝션

```python
class LearnablePanelProjector(nn.Module):
    def __init__(self, panel_configs: Dict[str, int]):
        super().__init__()
        self.projectors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.GELU(),
                nn.Linear(64, embed_dim)
            )
            for name, in_dim in panel_configs.items()
        })
```

### 2. 패널별 Salience

```python
def panels_with_salience(self, core) -> list[tuple[np.ndarray, float]]:
    """각 패널에 중요도 스코어 추가"""
    panels = self.panels(core)
    saliences = [
        compute_salience(p, context=core.current_state)
        for p in panels
    ]
    return list(zip(panels, saliences))
```

### 3. 동적 패널 선택

```python
# Top-k 패널만 사용
panels_ranked = sorted(
    zip(panels, saliences),
    key=lambda x: x[1],
    reverse=True
)
top_panels = [p for p, s in panels_ranked[:k]]
```

### 4. 계층적 패널

```python
# 고수준 패널 = 저수준 패널들의 조합
meta_panel = combine([stability_panel, topo_panel])
action_panel = combine([tool_panel, bus_panel])
```

## 테스트 방법

```python
from M3 import M3ConsciousnessCore
from llm_adapter import attach_llm_to_core

# Core 생성
core = M3ConsciousnessCore(...)

# 패널 생성 테스트
panels = core.feature_bank.panels(core)
print(f"Number of panels: {len(panels)}")
for i, panel in enumerate(panels):
    print(f"Panel {i}: shape={panel.shape}, mean={panel.mean():.3f}, std={panel.std():.3f}")

# LLM 생성 테스트
adapter = attach_llm_to_core(core)
response = adapter.generate("What do you see?")

# Attention 분석 (향후 구현)
# att_weights = adapter.conv_policy.get_last_attention_weights()
# print(f"Panel attention: {att_weights}")
```

## 참고 문서

- [TOKEN_CRITIC_DESIGN.md](./TOKEN_CRITIC_DESIGN.md) - 토큰 크리틱 헤드
- [RULES_README.md](../TEXT/RULES_README.md) - M3 아키텍처
