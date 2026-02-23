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
    """Configuration for M3StateEncoder."""
    nhead_candidates: List[int] = field(default_factory=lambda: [8, 4, 2, 1])
    fusion_layers_small: int = 1
    fusion_layers_medium: int = 2
    fusion_layers_large: int = 3
    hidden_dim_small_threshold: int = 256
    hidden_dim_medium_threshold: int = 512
    dropout_small: float = 0.05
    dropout_medium: float = 0.1
    dropout_large: float = 0.15


@dataclass
class M3StateCacheConfig:
    """Configuration for M3StateCache."""
    phi_history_multiplier: int = 2
    cache_size_min: int = 3
    cache_size_max: int = 100
    cache_size_default: int = 10
    trend_window_divisor: int = 3
    trend_window_min: int = 2
    trend_window_max: int = 10
    trend_threshold_base: float = 0.01


@dataclass
class M3AwareDecoderLayerConfig:
    """Configuration for M3AwareDecoderLayer."""
    nhead_candidates: List[int] = field(default_factory=lambda: [8, 4, 2, 1])
    dim_feedforward_multiplier: int = 4
    dropout_small: float = 0.05
    dropout_medium: float = 0.1
    dropout_large: float = 0.15
    d_model_small_threshold: int = 256
    d_model_medium_threshold: int = 512


@dataclass
class M3AdaptiveSamplerConfig:
    """Configuration for M3AdaptiveSampler."""
    temp_min: float = 0.3
    temp_max: float = 2.0
    phi_influence: float = 0.3
    energy_influence: float = 0.4
    meta_influence: float = 0.2
    top_k_high_exploration: int = 50
    top_k_medium_exploration: int = 30
    top_k_low_exploration: int = 10
    exploration_high_threshold: float = 0.7
    exploration_medium_threshold: float = 0.4


@dataclass
class M3EpisodicMemoryConfig:
    """Configuration for M3EpisodicMemoryRetriever."""
    memory_size_divisor: int = 100
    top_k_min: int = 1
    top_k_max: int = 10
    top_k_default: int = 3


@dataclass
class KNNIndexConfig:
    """Configuration for ConditionalKNNIndex."""
    tau: float = 0.07
    max_items: int = 500_000_000
    key_dim: int = 52400


@dataclass
class TokenizerConfig:
    """Configuration for M3 Tokenizers (Byte, BPE, Tiktoken, HF)."""
    pad_id: int = 256
    bos_id: int = 257
    eos_id: int = 258
    extra_vocab: int = 16000
    num_merges: int = 16000
    english_korean_repeat: int = 300
    programming_repeat: int = 200
    learning_repeat: int = 5000


@dataclass
class TorchPolicyConfig:
    """Configuration for TorchConversationalPolicy."""
    embed_dim: int = 1024
    hidden_dim: int = 1024
    num_layers: int = 6
    nhead: int = 1
    learning_rate: float = 1e-3
    dropout: float = 0.1
    init_gate_value: float = 0.27
    stability_weight: float = 0.3
    drift_weight: float = 0.2
    phi_delta_weight: float = 0.3
    alpha_base: float = 0.25
    alpha_phi_coef: float = 0.3
    alpha_entropy_coef: float = 0.25
    alpha_engagement_coef: float = 0.4
    alpha_min: float = 0.0
    alpha_max: float = 0.9
    beta_min: float = 0.01
    beta_max: float = 0.5
    default_temperature: float = 0.8
    default_top_k: int = 50
    default_top_p: float = 0.9


@dataclass
class PlasticBitLinearConfig:
    """Configuration for PlasticBitLinear layers."""
    trace_decay: float = 0.95
    learning_rate: float = 1e-4
    neuromodulation_scale: float = 0.1


@dataclass
class M3PlasticPolicyConfig:
    """Configuration for M3PlasticPolicy."""
    embed_dim: int = 1024
    hidden_dim: int = 1024
    num_layers: int = 6
    affect_dim: int = 5
    linear_config: PlasticBitLinearConfig = field(default_factory=PlasticBitLinearConfig)


@dataclass
class AutonomyRLConfig:
    """Autonomy Q-head RL update configuration."""
    enabled: bool = True
    use_linear_heads: bool = True
    state_source: str = "core"
    gamma: float = 0.95
    learning_rate: float = 3e-4
    replay_size: int = 4096
    batch_size: int = 16
    td_clip: float = 5.0
    target_speak_rate: float = 0.35
    lambda_min: float = 0.2
    lambda_max: float = 3.0
    reward_norm_ema: float = 0.95
    novelty_window: int = 64
    reward_w_dialog: float = 0.45
    reward_w_quality: float = 0.25
    reward_w_bus_credit: float = 0.20
    reward_w_novelty: float = 0.10
    safety_penalty: float = 0.75
    repetition_penalty: float = 0.35
    # Semantic dedup: suppress autonomy utterances too similar to recent ones
    semantic_dedup_enabled: bool = True
    semantic_dedup_threshold: float = 0.25  # novelty below this -> suppress
    semantic_dedup_max_retries: int = 2  # re-generate attempts before giving up


@dataclass
class EpisodicANNConfig:
    """ANN index acceleration configuration for episodic retrieval."""
    backend: str = "auto"  # auto|faiss|annoy|numpy
    candidate_k: int = 64
    min_items_for_ann: int = 128
    rebuild_interval: int = 256
    annoy_trees: int = 10
    faiss_metric: str = "ip"  # cosine via normalized IP
    log_select_once: bool = True
    cache_backend_probe: bool = True


@dataclass
class DPOAutoCollectConfig:
    """Automatic preference mining configuration for DPO."""
    enabled: bool = True
    max_pairs: int = 50000
    time_window_sec: int = 3600
    min_response_chars: int = 8
    output_file: str = "llm_preference.auto.jsonl"
    hard_negative_repeat_threshold: float = 0.35


@dataclass
class EarlyStopConfig:
    """Validation split and early stopping configuration."""
    enabled: bool = True
    val_fraction: float = 0.1
    patience: int = 3
    min_delta: float = 1e-4
    max_epochs: int = 20
    restore_best_weights: bool = True
    seed: int = 1337


@dataclass
class BridgeAdaptConfig:
    """Online adaptation config for ControlBridge."""
    enabled: bool = True
    learning_rate: float = 1e-4
    reward_scale: float = 1.0
    gate_reg: float = 1e-3
    bias_reg: float = 1e-4
    min_quality_score: float = 0.15
    cooldown_steps: int = 4


@dataclass
class TokenizerAutoVocabConfig:
    """Automatic tokenizer vocabulary rebuild configuration."""
    enabled: bool = True
    unknown_rate_threshold: float = 0.08
    min_observations: int = 128
    cooldown_steps: int = 500
    rebuild_vocab_size: int = 32000
    corpus_max_files: int = 64
    corpus_max_chars: int = 1_500_000
    min_corpus_chars: int = 5_000
    min_unique_terms: int = 128
    min_keep_vocab_ratio: float = 0.60
    rebuild_min_interval_sec: int = 1800
    state_file: str = "tokenizer_rebuild_state.json"


@dataclass
class StabilityConfig:
    """Training stability guard and regularization configuration."""
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    spectral_norm: bool = False
    max_weight_norm: float = 10.0
    grad_clip_norm: float = 1.0
    skip_non_finite: bool = True
    lr_backoff_on_nonfinite: bool = True
    backoff_factor: float = 0.5
    max_backoff_steps: int = 5
    clip_mode: str = "norm"
    # LR recovery: restore learning rate after consecutive successful steps
    lr_recovery_enabled: bool = True
    lr_recovery_streak: int = 10  # successful steps before recovery attempt
    lr_recovery_factor: float = 1.2  # multiply LR by this on recovery (capped at initial_lr)
    lr_initial: float = 0.0  # 0 = auto-detect from first optimizer step


@dataclass
class AdaptiveThresholdConfig:
    """Adaptive phi-threshold calibration for visualization and events."""
    enabled: bool = True
    warmup_steps: int = 256
    quantile_low: float = 0.50
    quantile_mid: float = 0.75
    quantile_high: float = 0.90
    announce_cooldown: int = 200
    announce_hysteresis: float = 0.02
    phi_floor_min: float = 0.005
    phi_floor_max: float = 0.05


@dataclass
class ObservationAdapterConfig:
    """Fixed-dimension observation adapter for policy compatibility."""
    enabled: bool = True
    target_policy_dim: int = 0
    projection_eps: float = 1e-6
    allow_policy_recreate: bool = False


@dataclass
class ConsciousnessBusConfig:
    """Configurable priority/filter/async behavior for ConsciousnessBus."""
    enabled: bool = True
    async_dispatch: bool = True
    max_queue: int = 4096
    drop_policy: str = "drop_low_priority"
    min_dispatch_interval_ms: int = 0
    default_topic: str = "bus"


@dataclass
class NeuroModulatorConfig:
    """Configuration for NeuroModulator weight-level consciousness control.

    Attributes:
        enabled: Whether the NeuroModulator is active.
        state_dim: Dimension of the M3 consciousness state vector input.
        trunk_dim: Hidden dimension of the shared trunk network.
        hidden_rank: Low-rank dimension for per-layer hidden bias generation.
        logit_rank: Low-rank dimension for output logit bias generation.
        strength: External modulation strength multiplier [0, inf).
        learning_rate: Optimizer learning rate for online adaptation.
        weight_decay: Optimizer L2 regularization coefficient.
        warmup_steps: Steps for exponential warmup from identity to full modulation.
        max_gain_delta: Maximum per-layer gain deviation from 1.0.
        max_logit_bias: Maximum absolute logit bias magnitude.
        grad_clip_norm: Maximum gradient norm for online learning updates.
        checkpoint_file: Path for persisting learned NeuroModulator weights.
    """
    enabled: bool = True
    state_dim: int = 256
    trunk_dim: int = 256
    hidden_rank: int = 16
    logit_rank: int = 32
    strength: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 100
    max_gain_delta: float = 0.3
    max_logit_bias: float = 2.0
    grad_clip_norm: float = 1.0
    checkpoint_file: str = "neuro_modulator.pt"


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
    plastic_policy: M3PlasticPolicyConfig = field(default_factory=M3PlasticPolicyConfig)
    autonomy_rl: AutonomyRLConfig = field(default_factory=AutonomyRLConfig)
    episodic_ann: EpisodicANNConfig = field(default_factory=EpisodicANNConfig)
    dpo_auto_collect: DPOAutoCollectConfig = field(default_factory=DPOAutoCollectConfig)
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    bridge_adapt: BridgeAdaptConfig = field(default_factory=BridgeAdaptConfig)
    tokenizer_auto_vocab: TokenizerAutoVocabConfig = field(default_factory=TokenizerAutoVocabConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    adaptive_threshold: AdaptiveThresholdConfig = field(default_factory=AdaptiveThresholdConfig)
    observation_adapter: ObservationAdapterConfig = field(default_factory=ObservationAdapterConfig)
    consciousness_bus: ConsciousnessBusConfig = field(default_factory=ConsciousnessBusConfig)
    neuro_modulator: NeuroModulatorConfig = field(default_factory=NeuroModulatorConfig)

    @classmethod
    def from_json(cls, path: str) -> 'M3LLMConfig':
        """Load configuration from JSON file."""
        if not os.path.exists(path):
            return cls()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        plastic_raw = dict(data.get('plastic_policy', {}) or {})
        linear_raw = dict(plastic_raw.get('linear_config', {}) or {})
        plastic_raw['linear_config'] = PlasticBitLinearConfig(**linear_raw)
        return cls(
            state_encoder=M3StateEncoderConfig(**data.get('state_encoder', {})),
            state_cache=M3StateCacheConfig(**data.get('state_cache', {})),
            decoder_layer=M3AwareDecoderLayerConfig(**data.get('decoder_layer', {})),
            adaptive_sampler=M3AdaptiveSamplerConfig(**data.get('adaptive_sampler', {})),
            episodic_memory=M3EpisodicMemoryConfig(**data.get('episodic_memory', {})),
            knn_index=KNNIndexConfig(**data.get('knn_index', {})),
            tokenizer=TokenizerConfig(**data.get('tokenizer', {})),
            torch_policy=TorchPolicyConfig(**data.get('torch_policy', {})),
            plastic_policy=M3PlasticPolicyConfig(**plastic_raw),
            autonomy_rl=AutonomyRLConfig(**data.get('autonomy_rl', {})),
            episodic_ann=EpisodicANNConfig(**data.get('episodic_ann', {})),
            dpo_auto_collect=DPOAutoCollectConfig(**data.get('dpo_auto_collect', {})),
            early_stop=EarlyStopConfig(**data.get('early_stop', {})),
            bridge_adapt=BridgeAdaptConfig(**data.get('bridge_adapt', {})),
            tokenizer_auto_vocab=TokenizerAutoVocabConfig(**data.get('tokenizer_auto_vocab', {})),
            stability=StabilityConfig(**data.get('stability', {})),
            adaptive_threshold=AdaptiveThresholdConfig(**data.get('adaptive_threshold', {})),
            observation_adapter=ObservationAdapterConfig(**data.get('observation_adapter', {})),
            consciousness_bus=ConsciousnessBusConfig(**data.get('consciousness_bus', {})),
            neuro_modulator=NeuroModulatorConfig(**data.get('neuro_modulator', {})),
        )

    def to_json(self, path: str):
        """Save configuration to JSON file."""
        data: Dict[str, Dict[str, Any]] = {
            'state_encoder': dict(self.state_encoder.__dict__),
            'state_cache': dict(self.state_cache.__dict__),
            'decoder_layer': dict(self.decoder_layer.__dict__),
            'adaptive_sampler': dict(self.adaptive_sampler.__dict__),
            'episodic_memory': dict(self.episodic_memory.__dict__),
            'knn_index': dict(self.knn_index.__dict__),
            'tokenizer': dict(self.tokenizer.__dict__),
            'torch_policy': dict(self.torch_policy.__dict__),
            'plastic_policy': {
                'embed_dim': self.plastic_policy.embed_dim,
                'hidden_dim': self.plastic_policy.hidden_dim,
                'num_layers': self.plastic_policy.num_layers,
                'affect_dim': self.plastic_policy.affect_dim,
                'linear_config': dict(self.plastic_policy.linear_config.__dict__),
            },
            'autonomy_rl': dict(self.autonomy_rl.__dict__),
            'episodic_ann': dict(self.episodic_ann.__dict__),
            'dpo_auto_collect': dict(self.dpo_auto_collect.__dict__),
            'early_stop': dict(self.early_stop.__dict__),
            'bridge_adapt': dict(self.bridge_adapt.__dict__),
            'tokenizer_auto_vocab': dict(self.tokenizer_auto_vocab.__dict__),
            'stability': dict(self.stability.__dict__),
            'adaptive_threshold': dict(self.adaptive_threshold.__dict__),
            'observation_adapter': dict(self.observation_adapter.__dict__),
            'consciousness_bus': dict(self.consciousness_bus.__dict__),
            'neuro_modulator': dict(self.neuro_modulator.__dict__),
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
    """Create default config file in JSON format."""
    config = M3LLMConfig()
    config.to_json(output_path)
    abs_path = os.path.abspath(output_path)
    logger.info(f"Default config file created at: {abs_path}")
    return abs_path


def load_config_from_file(config_path: str) -> M3LLMConfig:
    """Load JSON configuration from file."""
    config = M3LLMConfig.from_json(config_path)
    logger.info(f"Config loaded from: {config_path}")
    return config


def print_config_summary(config: Optional[M3LLMConfig] = None):
    """Print a summary of the M3LLMConfig."""
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
        ('Torch Policy', cfg.torch_policy),
        ('Autonomy RL', cfg.autonomy_rl),
        ('Episodic ANN', cfg.episodic_ann),
        ('DPO Auto Collect', cfg.dpo_auto_collect),
        ('Early Stop', cfg.early_stop),
        ('Bridge Adapt', cfg.bridge_adapt),
        ('Tokenizer Auto Vocab', cfg.tokenizer_auto_vocab),
        ('Stability', cfg.stability),
        ('Adaptive Threshold', cfg.adaptive_threshold),
        ('Observation Adapter', cfg.observation_adapter),
        ('Consciousness Bus', cfg.consciousness_bus),
        ('Neuro Modulator', cfg.neuro_modulator),
    ]
    for section_name, section_config in sections:
        print(f"\n[{section_name}]")
        for key, value in section_config.__dict__.items():
            if key.startswith('_'):
                continue
            if isinstance(value, float):
                print(f"  {key:.<40} {value:.4f}")
            else:
                print(f"  {key:.<40} {value}")
    print("\n" + "=" * 60 + "\n")


def validate_config(config: Optional[M3LLMConfig] = None) -> bool:
    """Validate the M3LLMConfig instance."""
    cfg = config or get_global_config()
    errors: List[str] = []

    if not cfg.state_encoder.nhead_candidates:
        errors.append("state_encoder.nhead_candidates cannot be empty")
    if cfg.state_encoder.dropout_small < 0 or cfg.state_encoder.dropout_small > 1:
        errors.append("state_encoder.dropout_small must be in [0, 1]")
    if cfg.state_cache.cache_size_min > cfg.state_cache.cache_size_max:
        errors.append("state_cache.cache_size_min must be <= cache_size_max")
    if cfg.adaptive_sampler.temp_min >= cfg.adaptive_sampler.temp_max:
        errors.append("adaptive_sampler.temp_min must be < temp_max")
    if cfg.adaptive_sampler.exploration_medium_threshold >= cfg.adaptive_sampler.exploration_high_threshold:
        errors.append("adaptive_sampler thresholds: medium must be < high")
    if cfg.torch_policy.alpha_min >= cfg.torch_policy.alpha_max:
        errors.append("torch_policy.alpha_min must be < alpha_max")
    if cfg.torch_policy.beta_min >= cfg.torch_policy.beta_max:
        errors.append("torch_policy.beta_min must be < beta_max")
    if cfg.torch_policy.learning_rate <= 0:
        errors.append("torch_policy.learning_rate must be > 0")
    if cfg.autonomy_rl.batch_size <= 0:
        errors.append("autonomy_rl.batch_size must be > 0")
    if cfg.episodic_ann.candidate_k <= 0:
        errors.append("episodic_ann.candidate_k must be > 0")
    if cfg.early_stop.patience < 0:
        errors.append("early_stop.patience must be >= 0")
    if not (0.0 <= cfg.early_stop.val_fraction < 1.0):
        errors.append("early_stop.val_fraction must be in [0, 1)")
    if cfg.tokenizer_auto_vocab.unknown_rate_threshold <= 0:
        errors.append("tokenizer_auto_vocab.unknown_rate_threshold must be > 0")
    if cfg.stability.grad_clip_norm <= 0:
        errors.append("stability.grad_clip_norm must be > 0")
    if cfg.autonomy_rl.lambda_min <= 0 or cfg.autonomy_rl.lambda_max <= cfg.autonomy_rl.lambda_min:
        errors.append("autonomy_rl.lambda bounds invalid")
    if cfg.tokenizer_auto_vocab.min_keep_vocab_ratio <= 0 or cfg.tokenizer_auto_vocab.min_keep_vocab_ratio > 1:
        errors.append("tokenizer_auto_vocab.min_keep_vocab_ratio must be in (0, 1]")
    if cfg.adaptive_threshold.warmup_steps < 0:
        errors.append("adaptive_threshold.warmup_steps must be >= 0")
    if cfg.consciousness_bus.max_queue <= 0:
        errors.append("consciousness_bus.max_queue must be > 0")
    if cfg.neuro_modulator.state_dim <= 0:
        errors.append("neuro_modulator.state_dim must be > 0")
    if cfg.neuro_modulator.trunk_dim <= 0:
        errors.append("neuro_modulator.trunk_dim must be > 0")
    if cfg.neuro_modulator.hidden_rank <= 0:
        errors.append("neuro_modulator.hidden_rank must be > 0")
    if cfg.neuro_modulator.logit_rank <= 0:
        errors.append("neuro_modulator.logit_rank must be > 0")
    if cfg.neuro_modulator.learning_rate <= 0:
        errors.append("neuro_modulator.learning_rate must be > 0")
    if cfg.neuro_modulator.strength < 0:
        errors.append("neuro_modulator.strength must be >= 0")
    if cfg.neuro_modulator.warmup_steps < 0:
        errors.append("neuro_modulator.warmup_steps must be >= 0")
    if cfg.neuro_modulator.max_gain_delta < 0:
        errors.append("neuro_modulator.max_gain_delta must be >= 0")
    if cfg.neuro_modulator.grad_clip_norm <= 0:
        errors.append("neuro_modulator.grad_clip_norm must be > 0")

    if errors:
        logger.error("Config validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    logger.info("Config validation passed.")
    return True
