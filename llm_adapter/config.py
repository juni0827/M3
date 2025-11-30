from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger('llm_adapter')


# ===========================
# Configuration Management
# ===========================


@dataclass
class M3StateEncoderConfig:
    """Configuration for M3StateEncoder"""
    # Attention head candidates (in descending order of preference)
    nhead_candidates: List[int] = field(default_factory=lambda: [8, 4, 2, 1])

    # Fusion layers by model size
    fusion_layers_small: int = 1      # hidden_dim <= 256
    fusion_layers_medium: int = 2     # hidden_dim <= 512
    fusion_layers_large: int = 3      # hidden_dim > 512
    hidden_dim_small_threshold: int = 256
    hidden_dim_medium_threshold: int = 512

    # Dropout by model size
    dropout_small: float = 0.05       # hidden_dim <= 256
    dropout_medium: float = 0.1       # hidden_dim <= 512
    dropout_large: float = 0.15       # hidden_dim > 512


@dataclass
class M3StateCacheConfig:
    """Configuration for M3StateCache."""
    # Cache size calculation
    phi_history_multiplier: int = 2   # cache_size = phi_history_len * multiplier
    cache_size_min: int = 3
    cache_size_max: int = 100
    cache_size_default: int = 10      # fallback when no phi_calculator

    # Phi trend window
    trend_window_divisor: int = 3     # window = cache_size / divisor
    trend_window_min: int = 2
    trend_window_max: int = 10
    trend_threshold_base: float = 0.01  # 0.01 / window


@dataclass
class M3AwareDecoderLayerConfig:
    """Configuration for M3AwareDecoderLayer."""
    # Attention heads (same as encoder)
    nhead_candidates: List[int] = field(default_factory=lambda: [8, 4, 2, 1])

    # Feedforward dimension multiplier (Transformer standard)
    dim_feedforward_multiplier: int = 4

    # Dropout by model size
    dropout_small: float = 0.05
    dropout_medium: float = 0.1
    dropout_large: float = 0.15
    d_model_small_threshold: int = 256
    d_model_medium_threshold: int = 512


@dataclass
class M3AdaptiveSamplerConfig:
    """Configuration for M3AdaptiveSampler."""
    # Temperature bounds
    temp_min: float = 0.3
    temp_max: float = 2.0

    # Temperature computation weights
    phi_influence: float = 0.3
    energy_influence: float = 0.4
    meta_influence: float = 0.2

    # Top-k by exploration level
    top_k_high_exploration: int = 50        # exploration > 0.7
    top_k_medium_exploration: int = 30      # exploration > 0.4
    top_k_low_exploration: int = 10         # exploration <= 0.4
    exploration_high_threshold: float = 0.7
    exploration_medium_threshold: float = 0.4


@dataclass
class M3EpisodicMemoryConfig:
    """Configuration for M3EpisodicMemoryRetriever."""
    # Top-k calculation
    memory_size_divisor: int = 100    # top_k = mem_size // divisor
    top_k_min: int = 1
    top_k_max: int = 10
    top_k_default: int = 3            # fallback


@dataclass
class KNNIndexConfig:
    """Configuration for ConditionalKNNIndex."""
    tau: float = 0.07
    max_items: int = 500_000_000
    key_dim: int = 52400


@dataclass
class TokenizerConfig:
    """Configuration for ByteTokenizer and HybridTokenizer."""
    # Special tokens
    pad_id: int = 256
    bos_id: int = 257
    eos_id: int = 258

    # HybridTokenizer BPE settings
    extra_vocab: int = 16000
    num_merges: int = 16000

    # Training corpus repetitions for merge learning
    english_korean_repeat: int = 300
    programming_repeat: int = 200
    learning_repeat: int = 5000


@dataclass
class TorchPolicyConfig:
    """Configuration for TorchConversationalPolicy."""
    # Model architecture
    embed_dim: int = 512
    hidden_dim: int = 1024
    learning_rate: float = 1e-3

    # Gating initialization
    init_gate_value: float = 0.27  # sigmoid^-1(0.567)

    # Memory context weights
    stability_weight: float = 0.3
    drift_weight: float = 0.2
    phi_delta_weight: float = 0.3

    # Alpha scheduler coefficients
    alpha_base: float = 0.25
    alpha_phi_coef: float = 0.3
    alpha_entropy_coef: float = 0.25
    alpha_engagement_coef: float = 0.4
    alpha_min: float = 0.0
    alpha_max: float = 0.9

    # Beta scheduler range
    beta_min: float = 0.01
    beta_max: float = 0.5

    # Sampling defaults
    default_temperature: float = 0.8
    default_top_k: int = 50
    default_top_p: float = 0.9


@dataclass
class M3LLMConfig:
    """Master configuration combining all sub-configs."""
    state_encoder: M3StateEncoderConfig = field(default_factory=M3StateEncoderConfig)
    state_cache: M3StateCacheConfig = field(default_factory=M3StateCacheConfig)
    decoder_layer: M3AwareDecoderLayerConfig = field(default_factory=M3AwareDecoderLayerConfig)
    adaptive_sampler: M3AdaptiveSamplerConfig = field(default_factory=M3AdaptiveSamplerConfig)
    episodic_memory: M3EpisodicMemoryConfig = field(default_factory=M3EpisodicMemoryConfig)
    knn_index: KNNIndexConfig = field(default_factory=KNNIndexConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    torch_policy: TorchPolicyConfig = field(default_factory=TorchPolicyConfig)

    @classmethod
    def from_json(cls, path: str) -> 'M3LLMConfig':
        """Load configuration from JSON file."""
        if not os.path.exists(path):
            return cls()  # Return defaults

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls(
            state_encoder=M3StateEncoderConfig(**data.get('state_encoder', {})),
            state_cache=M3StateCacheConfig(**data.get('state_cache', {})),
            decoder_layer=M3AwareDecoderLayerConfig(**data.get('decoder_layer', {})),
            adaptive_sampler=M3AdaptiveSamplerConfig(**data.get('adaptive_sampler', {})),
            episodic_memory=M3EpisodicMemoryConfig(**data.get('episodic_memory', {})),
            knn_index=KNNIndexConfig(**data.get('knn_index', {})),
            tokenizer=TokenizerConfig(**data.get('tokenizer', {})),
            torch_policy=TorchPolicyConfig(**data.get('torch_policy', {}))
        )

    def to_json(self, path: str):
        """Save configuration to JSON file."""
        data: Dict[str, Dict[str, Any]] = {
            'state_encoder': {
                'nhead_candidates': self.state_encoder.nhead_candidates,
                'fusion_layers_small': self.state_encoder.fusion_layers_small,
                'fusion_layers_medium': self.state_encoder.fusion_layers_medium,
                'fusion_layers_large': self.state_encoder.fusion_layers_large,
                'hidden_dim_small_threshold': self.state_encoder.hidden_dim_small_threshold,
                'hidden_dim_medium_threshold': self.state_encoder.hidden_dim_medium_threshold,
                'dropout_small': self.state_encoder.dropout_small,
                'dropout_medium': self.state_encoder.dropout_medium,
                'dropout_large': self.state_encoder.dropout_large
            },
            'state_cache': {
                'phi_history_multiplier': self.state_cache.phi_history_multiplier,
                'cache_size_min': self.state_cache.cache_size_min,
                'cache_size_max': self.state_cache.cache_size_max,
                'cache_size_default': self.state_cache.cache_size_default,
                'trend_window_divisor': self.state_cache.trend_window_divisor,
                'trend_window_min': self.state_cache.trend_window_min,
                'trend_window_max': self.state_cache.trend_window_max,
                'trend_threshold_base': self.state_cache.trend_threshold_base
            },
            'decoder_layer': {
                'nhead_candidates': self.decoder_layer.nhead_candidates,
                'dim_feedforward_multiplier': self.decoder_layer.dim_feedforward_multiplier,
                'dropout_small': self.decoder_layer.dropout_small,
                'dropout_medium': self.decoder_layer.dropout_medium,
                'dropout_large': self.decoder_layer.dropout_large,
                'd_model_small_threshold': self.decoder_layer.d_model_small_threshold,
                'd_model_medium_threshold': self.decoder_layer.d_model_medium_threshold
            },
            'adaptive_sampler': {
                'temp_min': self.adaptive_sampler.temp_min,
                'temp_max': self.adaptive_sampler.temp_max,
                'phi_influence': self.adaptive_sampler.phi_influence,
                'energy_influence': self.adaptive_sampler.energy_influence,
                'meta_influence': self.adaptive_sampler.meta_influence,
                'top_k_high_exploration': self.adaptive_sampler.top_k_high_exploration,
                'top_k_medium_exploration': self.adaptive_sampler.top_k_medium_exploration,
                'top_k_low_exploration': self.adaptive_sampler.top_k_low_exploration,
                'exploration_high_threshold': self.adaptive_sampler.exploration_high_threshold,
                'exploration_medium_threshold': self.adaptive_sampler.exploration_medium_threshold
            },
            'episodic_memory': {
                'memory_size_divisor': self.episodic_memory.memory_size_divisor,
                'top_k_min': self.episodic_memory.top_k_min,
                'top_k_max': self.episodic_memory.top_k_max,
                'top_k_default': self.episodic_memory.top_k_default
            },
            'knn_index': {
                'tau': self.knn_index.tau,
                'max_items': self.knn_index.max_items,
                'key_dim': self.knn_index.key_dim
            },
            'tokenizer': {
                'pad_id': self.tokenizer.pad_id,
                'bos_id': self.tokenizer.bos_id,
                'eos_id': self.tokenizer.eos_id,
                'extra_vocab': self.tokenizer.extra_vocab,
                'num_merges': self.tokenizer.num_merges,
                'english_korean_repeat': self.tokenizer.english_korean_repeat,
                'programming_repeat': self.tokenizer.programming_repeat,
                'learning_repeat': self.tokenizer.learning_repeat
            },
            'torch_policy': {
                'embed_dim': self.torch_policy.embed_dim,
                'hidden_dim': self.torch_policy.hidden_dim,
                'learning_rate': self.torch_policy.learning_rate,
                'init_gate_value': self.torch_policy.init_gate_value,
                'stability_weight': self.torch_policy.stability_weight,
                'drift_weight': self.torch_policy.drift_weight,
                'phi_delta_weight': self.torch_policy.phi_delta_weight,
                'alpha_base': self.torch_policy.alpha_base,
                'alpha_phi_coef': self.torch_policy.alpha_phi_coef,
                'alpha_entropy_coef': self.torch_policy.alpha_entropy_coef,
                'alpha_engagement_coef': self.torch_policy.alpha_engagement_coef,
                'alpha_min': self.torch_policy.alpha_min,
                'alpha_max': self.torch_policy.alpha_max,
                'beta_min': self.torch_policy.beta_min,
                'beta_max': self.torch_policy.beta_max,
                'default_temperature': self.torch_policy.default_temperature,
                'default_top_k': self.torch_policy.default_top_k,
                'default_top_p': self.torch_policy.default_top_p
            }
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Global config instance (can be overridden)
_global_config = M3LLMConfig()


def set_global_config(config: M3LLMConfig):
    """Set global configuration."""
    global _global_config
    _global_config = config


def get_global_config() -> M3LLMConfig:
    """Get global configuration."""
    return _global_config


# ============================================================================
# Configuration Utilities
# ============================================================================


def create_default_config_file(output_path: str = 'config/llm_config.json') -> str:
    """
    Create default config file in JSON format.

    Args:
        output_path: Path to the output config file.

    Returns:
        Absolute path to the created config file.
    """
    config = M3LLMConfig()
    config.to_json(output_path)
    abs_path = os.path.abspath(output_path)
    logger.info(f"Default config file created at: {abs_path}")
    return abs_path


def load_config_from_file(config_path: str) -> M3LLMConfig:
    """
    JSON configuration file loader.

    Args:
        config_path: Path to the config file.

    Returns:
        M3LLMConfig instance.
    """
    config = M3LLMConfig.from_json(config_path)
    logger.info(f"Config loaded from: {config_path}")
    return config


def print_config_summary(config: Optional[M3LLMConfig] = None):
    """
    Print a summary of the M3LLMConfig.

    Args:
        config: M3LLMConfig instance (None means global config)
    """
    cfg = config or get_global_config()

    print("\n" + "=" * 60)
    print(" M3 LLM Adapter Configuration Summary")
    print("=" * 60)

    sections = [
        ('State Encoder', cfg.state_encoder),
        ('State Cache', cfg.state_cache),
        ('Decoder Layer', cfg.decoder_layer),
        ('Adaptive Sampler', cfg.adaptive_sampler),
        ('Episodic Memory', cfg.episodic_memory),
        ('KNN Index', cfg.knn_index),
        ('Tokenizer', cfg.tokenizer),
        ('Torch Policy', cfg.torch_policy)
    ]

    for section_name, section_config in sections:
        print(f"\n[{section_name}]")
        for key, value in section_config.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, float):
                    print(f"  {key:.<40} {value:.4f}")
                elif isinstance(value, list):
                    print(f"  {key:.<40} {value}")
                else:
                    print(f"  {key:.<40} {value}")

    print("\n" + "=" * 60 + "\n")


def validate_config(config: Optional[M3LLMConfig] = None) -> bool:
    """
    Validate the M3LLMConfig instance.

    Args:
        config: M3LLMConfig instance (None means global config)

    Returns:
        bool: True if valid, False otherwise
    """
    cfg = config or get_global_config()
    errors = []

    # State Encoder
    if not cfg.state_encoder.nhead_candidates:
        errors.append("state_encoder.nhead_candidates cannot be empty")

    if cfg.state_encoder.dropout_small < 0 or cfg.state_encoder.dropout_small > 1:
        errors.append("state_encoder.dropout_small must be in [0, 1]")

    # State Cache
    if cfg.state_cache.cache_size_min > cfg.state_cache.cache_size_max:
        errors.append("state_cache.cache_size_min must be <= cache_size_max")

    # Adaptive Sampler
    if cfg.adaptive_sampler.temp_min >= cfg.adaptive_sampler.temp_max:
        errors.append("adaptive_sampler.temp_min must be < temp_max")

    if cfg.adaptive_sampler.exploration_medium_threshold >= cfg.adaptive_sampler.exploration_high_threshold:
        errors.append("adaptive_sampler thresholds: medium must be < high")

    # Torch Policy
    if cfg.torch_policy.alpha_min >= cfg.torch_policy.alpha_max:
        errors.append("torch_policy.alpha_min must be < alpha_max")

    if cfg.torch_policy.beta_min >= cfg.torch_policy.beta_max:
        errors.append("torch_policy.beta_min must be < beta_max")

    if cfg.torch_policy.learning_rate <= 0:
        errors.append("torch_policy.learning_rate must be > 0")

    # State Encoder hidden dim thresholds
    if getattr(cfg.state_encoder, 'hidden_dim', None) is not None:
        if cfg.state_encoder.hidden_dim < 128:
            errors.append("state_encoder.hidden_dim must be >= 128")
        if cfg.state_encoder.hidden_dim > 1024:
            errors.append("state_encoder.hidden_dim must be <= 1024")

    if errors:
        logger.error("Config validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False

    logger.info("Config validation passed.")
    return True
