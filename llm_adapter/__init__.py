from __future__ import annotations

from llm_adapter.config import (
    AutonomyRLConfig,
    BridgeAdaptConfig,
    DPOAutoCollectConfig,
    EarlyStopConfig,
    EpisodicANNConfig,
    KNNIndexConfig,
    M3AdaptiveSamplerConfig,
    M3AwareDecoderLayerConfig,
    M3EpisodicMemoryConfig,
    M3LLMConfig,
    StabilityConfig,
    TokenizerAutoVocabConfig,
    M3StateCacheConfig,
    M3StateEncoderConfig,
    TokenizerConfig,
    TorchPolicyConfig,
    create_default_config_file,
    get_global_config,
    load_config_from_file,
    print_config_summary,
    set_global_config,
    validate_config,
)
from llm_adapter.tokenization import M3Tokenizer, AutoTokenizer
from llm_adapter.memory import ConditionalKNNIndex, KNNItem, M3EpisodicMemoryRetriever
from llm_adapter.llm_core import (
    M3AdaptiveSampler,
    M3AwareDecoderLayer,
    M3StateCache,
    M3StateEncoder,
    HFBackend,
    TorchConversationalPolicy,
    attach_llm_to_core,
)
from llm_adapter.layers import PlasticBitLinear
from llm_adapter.m3_control_bridge import (
    M3ControlBridge,
    LayerGateRuntime,
    GenerationQualityGate,
    QualityGateResult,
    find_decoder_layers,
    NeuroModulator,
    NeuroModulatorRuntime,
    NeuroModControls,
)
from llm_adapter.remote import get_local_thinking

__all__ = [
    'attach_llm_to_core',
    'M3StateEncoder', 'M3StateCache', 'M3AwareDecoderLayer', 'M3AdaptiveSampler', 'HFBackend', 'TorchConversationalPolicy',
    'M3Tokenizer', 'AutoTokenizer',
    'ConditionalKNNIndex', 'M3EpisodicMemoryRetriever', 'KNNItem',
    'PlasticBitLinear',
    'M3ControlBridge', 'LayerGateRuntime', 'GenerationQualityGate', 'QualityGateResult', 'find_decoder_layers',
    'NeuroModulator', 'NeuroModulatorRuntime', 'NeuroModControls',
    'get_local_thinking',
    'M3LLMConfig', 'M3StateEncoderConfig', 'M3StateCacheConfig', 'M3AwareDecoderLayerConfig', 'M3AdaptiveSamplerConfig',
    'M3EpisodicMemoryConfig', 'KNNIndexConfig', 'TokenizerConfig', 'TorchPolicyConfig',
    'AutonomyRLConfig', 'EpisodicANNConfig', 'DPOAutoCollectConfig', 'EarlyStopConfig',
    'BridgeAdaptConfig', 'TokenizerAutoVocabConfig', 'StabilityConfig',
    'set_global_config', 'get_global_config', 'create_default_config_file', 'load_config_from_file', 'print_config_summary', 'validate_config',
]
