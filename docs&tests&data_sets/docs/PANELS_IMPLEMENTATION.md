# 패널 메모리 시스템 구현 완료

## 구현 내용

의미층별로 분리된 패널 메모리 시스템이 성공적으로 구현되었습니다. 이제 LLM이 단일 평탄 벡터가 아닌 **의미적으로 구조화된 다중 토큰**을 Cross-Attention 메모리로 사용합니다.

## 주요 변경사항

### 1. M3.py - FeatureBank.panels() 추가

#### 새 메서드

```python
def panels(self, core: "M3ConsciousnessCore") -> list[np.ndarray]:
    """
    Φ_total을 의미층별 패널로 분해
    반환: [stability, topo, time, tool, bus, topic] (각 D차원)
    """
```

#### 6개 패널 구성

1. **stability_panel**: 안정성, 에너지, 주체, 정서 (8 피처)
2. **topo_panel**: 시각, 위상, φ (8 피처)
3. **time_panel**: RPE/entropy 시계열 통계 + 최근 히스토리
4. **tool_panel**: 툴콜 성공/실패/지연 통계 (8 피처)
5. **bus_panel**: MessageBus 활동 통계 (6 피처)
6. **topic_panel**: 언어/주제 임베딩

#### _to_panel() 헬퍼

- 입력 → (x-μ)/σ → tanh → 길이 D로 패딩/절단
- Robust 정규화로 이상치 처리

### 2. llm_adapter.py - 메모리 준비 개선

#### generate() 메서드

```python
# 우선순위: panels() > build() fallback
if hasattr(self._core.feature_bank, 'panels'):
    panels_list = self._core.feature_bank.panels(self._core)
    mem = np.stack(panels_list, axis=0)  # (M, D)
elif hasattr(self._core.feature_bank, 'build'):
    z = self._core.feature_bank.build(self._core)
    mem = np.repeat(z, M, axis=0)  # Fallback: 복제
```

#### TorchConversationalPolicy.generate()

```python
# 메모리 입력 형태 정규화
if m.ndim == 1:
    m = m[None, :]          # (D,) → (1, D)
if m.ndim == 2:
    m = m[None, :, :]       # (M, D) → (1, M, D)
# 이제 m.shape == (1, M, D) 보장


**개선점**: (M, D) 형태를 직접 지원하여 패널 스택을 그대로 투입

## 작동 흐름

### 1. 패널 생성 (FeatureBank)


M3Core 상태
    ↓
panels(core)
    ↓
각 의미층에서 피처 수집
    ↓
_to_panel()로 정규화 (D차원)
    ↓
[panel1, panel2, ..., panel6]


### 2. 메모리 준비 (LLMAdapter)

panels() 호출
    ↓
list[np.ndarray] (각 D차원)
    ↓
np.stack(axis=0)
    ↓
(M, D) 배열


### 3. Cross-Attention (TorchPolicy)


panels (M, D)
    ↓
(1, M, D) 텐서 변환
    ↓
mem_proj → (1, M, H)
    ↓
Wk, Wv → mem_k, mem_v
    ↓
각 디코딩 스텝에서
query @ mem_k → attention weights
    ↓
weights @ mem_v → context
    ↓
decoder_state + context


## 예시: Attention 가중치

### 시각 질문: "무엇이 보이나요?"

Attention distribution:
├─ stability_panel: 0.05
├─ topo_panel:      0.65  ← 시각/φ 정보 집중
├─ time_panel:      0.10
├─ tool_panel:      0.05
├─ bus_panel:       0.05
└─ topic_panel:     0.10

### 감정 표현: "기분이 어때요?"

Attention distribution:
├─ stability_panel: 0.70  ← 에너지/정서 집중
├─ topo_panel:      0.05
├─ time_panel:      0.15  ← 최근 경험 참조
├─ tool_panel:      0.02
├─ bus_panel:       0.03
└─ topic_panel:     0.05

## 환경 변수

### 기존 (Fallback용)

- `LLM_ADAPTER_MEM_TOK` (기본: 4)
  - panels() 없을 때 build() 복제 횟수

### 향후 확장용

- `LLM_ADAPTER_PANEL_DIM` (계획)
  - 패널 임베딩 차원 (현재는 embed_dim 사용)

- `LLM_ADAPTER_ENABLE_PANELS` (계획)
  - 패널 시스템 강제 비활성화 (디버깅용)

## 장점

### 1. 의미 분리

- 각 패널 = 독립적 의미 공간
- 정보 손실 최소화

### 2. 선택적 Attention

- 상황에 맞는 패널 자동 선택
- 시각 추론 → topo
- 감정 생성 → stability
- 최근 경험 → time

### 3. 해석 가능성

- Attention weights로 어떤 정보를 사용했는지 추적
- 디버깅/분석 용이

### 4. 확장성

- 새 패널 추가 용이
- 기존 코드 변경 최소

### 5. 효율성

- 필요 정보만 선택
- 평탄 벡터보다 표현력 ↑

## 테스트 방법

```python
from M3 import M3ConsciousnessCore
from llm_adapter import attach_llm_to_core
import numpy as np

# Core 생성
core = M3ConsciousnessCore(
    max_dim=128,
    vision_height=64,
    vision_width=64
)

# 패널 생성 테스트
print("=== Panel Generation Test ===")
panels = core.feature_bank.panels(core)
print(f"Number of panels: {len(panels)}")

panel_names = ['stability', 'topo', 'time', 'tool', 'bus', 'topic']
for name, panel in zip(panel_names, panels):
    print(f"{name:12s}: shape={panel.shape}, "
          f"mean={panel.mean():+.3f}, std={panel.std():.3f}, "
          f"min={panel.min():+.3f}, max={panel.max():+.3f}")

# LLM 생성 테스트
print("\n=== LLM Generation with Panels ===")
adapter = attach_llm_to_core(core)

# 시각 질문
response1 = adapter.generate("What do you see?")
print(f"Visual Q: {response1}")

# 감정 질문
response2 = adapter.generate("How are you feeling?")
print(f"Emotion Q: {response2}")

# 최근 경험 질문
response3 = adapter.generate("What happened recently?")
print(f"History Q: {response3}")


**예상 출력**:


=== Panel Generation Test ===
Number of panels: 6
stability   : shape=(32,), mean=+0.023, std=0.871, min=-1.000, max=+0.998
topo        : shape=(32,), mean=-0.012, std=0.834, min=-0.987, max=+1.000
time        : shape=(32,), mean=+0.001, std=0.923, min=-1.000, max=+0.999
tool        : shape=(32,), mean=+0.034, std=0.712, min=-0.876, max=+0.945
bus         : shape=(32,), mean=-0.003, std=0.801, min=-0.932, max=+0.921
topic       : shape=(32,), mean=+0.000, std=0.789, min=-0.965, max=+0.978

=== LLM Generation with Panels ===
Visual Q: I see edges and contrasts in the visual field...
Emotion Q: I'm feeling balanced with moderate energy...
History Q: Recently, there have been fluctuations in...


## 비교: Before vs After

### Before (build() 복제)

```

단일 벡터 (128D)
    ↓
복제 4번
    ↓
[v, v, v, v] (4 x 128D)
    ↓
Cross-Attention
    - 모든 토큰이 동일
    - 차별성 없음

### After (panels())

의미층별 분리
    ↓
[stability, topo, time, tool, bus, topic]
    ↓
각 32D x 6 = (6, 32)
    ↓
Cross-Attention
    - 각 토큰이 다른 의미
    - 상황별 선택적 사용

## 향후 개선 방향

### 1. 학습 가능한 패널 프로젝션

```python
class PanelProjector(nn.Module):
    def __init__(self, config):
        self.stability_proj = MLP(8, 32)
        self.topo_proj = MLP(8, 32)
        # ...


### 2. 패널별 Salience 스코어

```python
def panels_with_salience(core):
    panels = panels(core)
    saliences = [compute_importance(p) for p in panels]
    return sorted(zip(panels, saliences), key=lambda x: -x[1])
```

### 3. 동적 패널 선택 (Top-K)

```python
# 중요한 K개만 사용
top_k_panels = select_top_k(panels, k=3, criterion='salience')
```

### 4. 계층적 패널

```python
# 고수준 패널 = 저수준 조합
meta_panel = combine([stability, topo])
action_panel = combine([tool, bus])
```

## 참고 문서

- [PANELS_MEMORY_DESIGN.md](./PANELS_MEMORY_DESIGN.md) - 상세 설계 문서
- [TOKEN_CRITIC_DESIGN.md](./TOKEN_CRITIC_DESIGN.md) - 토큰 크리틱 헤드
- [RULES_README.md](../TEXT/RULES_README.md) - M3 아키텍처 개요

---

**구현 완료**: 패널 메모리 시스템이 성공적으로 통합
