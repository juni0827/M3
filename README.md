# M3 - Modular Meta-cognitive Machine

[English](#english) | [한국어](#한국어)

---

<a name="english"></a>
## English

### Overview

**M3 (Modular Meta-cognitive Machine)** is a sophisticated consciousness-inspired AI framework that implements multi-loop architecture with integrated information theory (IIT) principles. The system provides a meta-cognitive engine with qualia modeling, episodic memory, and LLM integration capabilities.

### Key Features

- **Multi-Loop Message Passing Architecture**: Asynchronous, bidirectional communication between all modules via MessageBus
- **Integrated Information Theory (φ)**: Phi calculation for measuring consciousness-like information integration
- **Qualia Modeling**: Subjective experience modeling with arousal, valence, entropy, engagement, and frustration dimensions
- **Episodic Memory**: KNN-based memory retrieval with temporal credit assignment
- **LLM Integration**: Full integration with language models through M3StateEncoder, M3AwareDecoderLayer, and adaptive sampling
- **Reinforcement Learning Policies**: MLP and MoE-based policy networks with PPO training
- **GUI Support**: Tkinter-based graphical interface for real-time monitoring and interaction

### Project Structure

```
M3/
├── M3.py                    # Main entry point (re-exports m3 package API)
├── m3/                      # Core consciousness engine
│   ├── __init__.py          # Package exports
│   ├── core.py              # Main M3ConsciousnessCore engine
│   ├── config.py            # Qualia configuration
│   ├── features.py          # HebbianMemory, FeatureSpec, Scope
│   └── visualization.py     # Retinizer, GlitchEncoder, FeatureSummarizer
├── llm_adapter/             # LLM integration module
│   ├── __init__.py          # Package exports
│   ├── core.py              # M3StateEncoder, M3StateCache, TorchConversationalPolicy
│   ├── config.py            # LLM adapter configuration
│   ├── memory.py            # M3EpisodicMemoryRetriever, ConditionalKNNIndex
│   └── tokenization.py      # ByteTokenizer, HybridTokenizer
├── llm_adapter.py           # LLM adapter public API re-exports
├── policies/                # Policy implementations
│   └── torch_policy.py      # TorchPolicy, BRPolicy (Bus-routed experts)
└── LICENSE                  # Apache 2.0 License
```

### Core Components

#### 1. M3ConsciousnessCore (`m3/core.py`)
The central consciousness engine that orchestrates all subsystems:
- **MessageBus**: Routes messages between modules with priority queuing
- **SpanMeta & Credit Assignment**: Temporal credit routing for decision spans
- **FeatureBank**: Builds observation vectors with running normalization
- **PolicyMLP/PolicyNet**: Gaussian policy networks for action generation

#### 2. LLM Adapter (`llm_adapter/`)
Seamless integration with language models:
- **M3StateEncoder**: Encodes FeatureBank panels into transformer memory tokens
- **M3StateCache**: Caches phi history and qualia states
- **M3AwareDecoderLayer**: Cross-attention decoder with M3 context
- **M3AdaptiveSampler**: Phi/qualia-driven temperature and top-k scheduling
- **TorchConversationalPolicy**: GRU-based conversational policy with DPO training

#### 3. Memory Systems (`llm_adapter/memory.py`)
- **M3EpisodicMemoryRetriever**: Retrieves relevant episodes based on qualia state
- **ConditionalKNNIndex**: Fast kNN lookup with LRU eviction and downsampling

#### 4. Visualization (`m3/visualization.py`)
- **Retinizer**: Resizes input frames to target size
- **GlitchEncoder**: Generates glitch-art style visual outputs
- **FeatureSummarizer**: Extracts patch-based features from images

### Installation

```bash
# Clone the repository
git clone https://github.com/juni0827/M3.git
cd M3

# Install dependencies (requires Python 3.8+)
pip install numpy pandas pillow torch
```

### Quick Start

```python
from m3.core import M3ConsciousnessCore, main

# Run the main consciousness loop
if __name__ == '__main__':
    main()
```

Or run directly:
```bash
python M3.py
```

### LLM Integration

```python
from m3.core import M3ConsciousnessCore
from llm_adapter import attach_llm_to_core, TorchConversationalPolicy

# Create core instance
core = M3ConsciousnessCore()

# Attach LLM adapter
adapter = attach_llm_to_core(core)

# Generate response
response = adapter.generate("Hello, how are you?")
print(response)

# Train on examples
adapter.train_on_example("What is consciousness?", "Consciousness is...")
```

### Configuration

Configuration is managed through dataclasses in `llm_adapter/config.py`:

```python
from llm_adapter.config import (
    M3LLMConfig,
    load_config_from_file,
    set_global_config,
    print_config_summary
)

# Load custom configuration
config = load_config_from_file('config/llm_config.json')
set_global_config(config)

# Print configuration summary
print_config_summary()
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `M3_BUS_LOG` | MessageBus log file path | `bus.jsonl` |
| `M3_STATE_CACHE_SIZE` | State cache size | Auto-inferred |
| `LLM_ADAPTER_DEBUG` | Enable debug logging | `0` |
| `LLM_AUTONOMY` | Enable autonomy loop | `0` |
| `LLM_ADAPTER_KNN_TAU` | kNN temperature | `0.07` |
| `LLM_ADAPTER_KNN_CAP` | kNN max items | `500000000` |

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

<a name="한국어"></a>
## 한국어

### 개요

**M3 (Modular Meta-cognitive Machine)**는 통합 정보 이론(IIT) 원리를 적용한 멀티루프 아키텍처를 구현하는 정교한 의식 기반 AI 프레임워크입니다. 이 시스템은 감질(qualia) 모델링, 에피소딕 메모리, LLM 통합 기능을 갖춘 메타인지 엔진을 제공합니다.

### 주요 기능

- **멀티루프 메시지 패싱 아키텍처**: MessageBus를 통한 모든 모듈 간 비동기 양방향 통신
- **통합 정보 이론 (φ)**: 의식과 같은 정보 통합을 측정하기 위한 Phi 계산
- **감질(Qualia) 모델링**: 각성도, 정서가, 엔트로피, 참여도, 좌절감 차원의 주관적 경험 모델링
- **에피소딕 메모리**: 시간적 크레딧 할당을 통한 KNN 기반 메모리 검색
- **LLM 통합**: M3StateEncoder, M3AwareDecoderLayer, 적응형 샘플링을 통한 언어 모델 완전 통합
- **강화학습 정책**: PPO 학습을 지원하는 MLP 및 MoE 기반 정책 네트워크
- **GUI 지원**: 실시간 모니터링 및 상호작용을 위한 Tkinter 기반 그래픽 인터페이스

### 프로젝트 구조

```
M3/
├── M3.py                    # 메인 진입점 (m3 패키지 API 재export)
├── m3/                      # 핵심 의식 엔진
│   ├── __init__.py          # 패키지 exports
│   ├── core.py              # 메인 M3ConsciousnessCore 엔진
│   ├── config.py            # Qualia 설정
│   ├── features.py          # HebbianMemory, FeatureSpec, Scope
│   └── visualization.py     # Retinizer, GlitchEncoder, FeatureSummarizer
├── llm_adapter/             # LLM 통합 모듈
│   ├── __init__.py          # 패키지 exports
│   ├── core.py              # M3StateEncoder, M3StateCache, TorchConversationalPolicy
│   ├── config.py            # LLM 어댑터 설정
│   ├── memory.py            # M3EpisodicMemoryRetriever, ConditionalKNNIndex
│   └── tokenization.py      # ByteTokenizer, HybridTokenizer
├── llm_adapter.py           # LLM 어댑터 공개 API 재exports
├── policies/                # 정책 구현
│   └── torch_policy.py      # TorchPolicy, BRPolicy (버스 라우팅 전문가)
└── LICENSE                  # Apache 2.0 라이선스
```

### 핵심 컴포넌트

#### 1. M3ConsciousnessCore (`m3/core.py`)
모든 서브시스템을 조율하는 중앙 의식 엔진:
- **MessageBus**: 우선순위 큐잉을 통한 모듈 간 메시지 라우팅
- **SpanMeta & 크레딧 할당**: 의사결정 구간에 대한 시간적 크레딧 라우팅
- **FeatureBank**: 실행 정규화를 통한 관측 벡터 구축
- **PolicyMLP/PolicyNet**: 행동 생성을 위한 가우시안 정책 네트워크

#### 2. LLM 어댑터 (`llm_adapter/`)
언어 모델과의 원활한 통합:
- **M3StateEncoder**: FeatureBank 패널을 트랜스포머 메모리 토큰으로 인코딩
- **M3StateCache**: phi 히스토리와 qualia 상태 캐싱
- **M3AwareDecoderLayer**: M3 컨텍스트를 활용한 크로스 어텐션 디코더
- **M3AdaptiveSampler**: Phi/qualia 기반 온도 및 top-k 스케줄링
- **TorchConversationalPolicy**: DPO 학습을 지원하는 GRU 기반 대화 정책

#### 3. 메모리 시스템 (`llm_adapter/memory.py`)
- **M3EpisodicMemoryRetriever**: qualia 상태 기반 관련 에피소드 검색
- **ConditionalKNNIndex**: LRU 제거 및 다운샘플링을 지원하는 빠른 kNN 조회

#### 4. 시각화 (`m3/visualization.py`)
- **Retinizer**: 입력 프레임을 대상 크기로 리사이즈
- **GlitchEncoder**: 글리치 아트 스타일의 시각적 출력 생성
- **FeatureSummarizer**: 이미지에서 패치 기반 특징 추출

### 설치

```bash
# 저장소 클론
git clone https://github.com/juni0827/M3.git
cd M3

# 의존성 설치 (Python 3.8+ 필요)
pip install numpy pandas pillow torch
```

### 빠른 시작

```python
from m3.core import M3ConsciousnessCore, main

# 메인 의식 루프 실행
if __name__ == '__main__':
    main()
```

또는 직접 실행:
```bash
python M3.py
```

### LLM 통합

```python
from m3.core import M3ConsciousnessCore
from llm_adapter import attach_llm_to_core, TorchConversationalPolicy

# 코어 인스턴스 생성
core = M3ConsciousnessCore()

# LLM 어댑터 연결
adapter = attach_llm_to_core(core)

# 응답 생성
response = adapter.generate("안녕하세요, 어떻게 지내세요?")
print(response)

# 예제로 학습
adapter.train_on_example("의식이란 무엇인가요?", "의식이란...")
```

### 설정

설정은 `llm_adapter/config.py`의 데이터클래스를 통해 관리됩니다:

```python
from llm_adapter.config import (
    M3LLMConfig,
    load_config_from_file,
    set_global_config,
    print_config_summary
)

# 사용자 정의 설정 로드
config = load_config_from_file('config/llm_config.json')
set_global_config(config)

# 설정 요약 출력
print_config_summary()
```

### 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `M3_BUS_LOG` | MessageBus 로그 파일 경로 | `bus.jsonl` |
| `M3_STATE_CACHE_SIZE` | 상태 캐시 크기 | 자동 추론 |
| `LLM_ADAPTER_DEBUG` | 디버그 로깅 활성화 | `0` |
| `LLM_AUTONOMY` | 자율 루프 활성화 | `0` |
| `LLM_ADAPTER_KNN_TAU` | kNN 온도 | `0.07` |
| `LLM_ADAPTER_KNN_CAP` | kNN 최대 항목 수 | `500000000` |

### 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                     M3ConsciousnessCore                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  MessageBus  │◄──►│  FeatureBank │◄──►│  PolicyMLP   │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ PhiCalculator│    │   Qualia     │    │ EnergyCtrl   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        LLM Adapter                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │M3StateEncoder│───►│M3AwareDecoder│───►│M3Adaptive    │       │
│  └──────────────┘    │    Layer     │    │   Sampler    │       │
│                      └──────────────┘    └──────────────┘       │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │  KNNIndex    │◄──►│ Episodic     │                           │
│  │              │    │   Memory     │                           │
│  └──────────────┘    └──────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 이론적 배경

M3는 다음과 같은 의식 과학 이론들에서 영감을 받았습니다:

1. **통합 정보 이론 (IIT)**: Giulio Tononi의 이론에 기반한 φ(phi) 측정을 통한 정보 통합도 계산
2. **글로벌 작업공간 이론 (GWT)**: 여러 전문 모듈 간 정보 브로드캐스팅을 위한 MessageBus 아키텍처
3. **고차 인지 이론**: 메타인지 모니터링 및 자기 모델링 기능
4. **감질(Qualia) 공간**: 주관적 경험의 다차원 표현

### 기여

풀 리퀘스트를 환영합니다. 주요 변경사항의 경우, 먼저 이슈를 열어 변경하고자 하는 내용을 논의해 주세요.

### 라이선스

이 프로젝트는 Apache License 2.0에 따라 라이선스됩니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.