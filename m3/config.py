"""
M3 Core Configuration Module

This module provides a unified configuration system for the M3 core engine.
It supports loading from JSON files, environment variables, and programmatic
configuration.

Usage:
    # Get the global config (auto-loads from environment)
    from m3.config import get_m3_config
    config = get_m3_config()

    # Load from a JSON file
    from m3.config import load_m3_config
    config = load_m3_config('path/to/config.json')

    # Programmatic configuration
    from m3.config import M3Config, QualiaConfig, set_m3_config
    config = M3Config(qualia=QualiaConfig(enabled=False))
    set_m3_config(config)

Environment Variables:
    M3_QUALIA_ENABLED - Enable/disable qualia closed-loop (true/false)
    M3_QUALIA_K_TEMP_DRIFT - Temperature gain from drift prediction
    M3_CES_SHARD_PREFIX_BITS - CES shard prefix bits
    M3_BUS_LOG - Path to the bus log file
    ... and more (see M3Config.from_env() for full list)
"""
from __future__ import annotations

import json
import logging
import os as _os
import warnings
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Iterator, Optional, Tuple

logger = logging.getLogger('m3.config')

# Path and environment-based defaults (backward compatibility)
QUALIA_LOG_PATH = _os.environ.get('M3_BUS_LOG', 'bus.jsonl')


# =============================================================================
# Qualia Closed-Loop Configuration
# =============================================================================

@dataclass
class QualiaConfig:
    """
    Configuration for the Qualia Closed-Loop system.

    The Qualia system monitors internal state and adjusts policy parameters
    (temperature, learning rate, replay bias) based on stability, meta-confidence,
    and intensity signals.

    Attributes:
        enabled: Enable/disable the qualia closed-loop system.

        k_temp_drift: Temperature gain from drift.prediction signal.
            Higher values make temperature more sensitive to prediction drift.
        k_temp_stab: Temperature gain from (1 - belief.stability).
            Higher values increase exploration when beliefs are unstable.
        k_lr_meta: Learning rate gain from meta_confidence.
            Negative values reduce LR when confident (recommended).
        k_lr_intensity: Learning rate gain from qualia intensity.
            Higher values increase LR during intense experiences.

        sigma_min: Minimum exploration noise (policy sigma).
        sigma_max: Maximum exploration noise (policy sigma).
        lr_min: Minimum learning rate.
        lr_max: Maximum learning rate.

        k_replay_valence: Replay bias gain from valence.
            Positive valence promotes consolidation.
        k_replay_intensity: Replay bias gain from intensity.
            Strong intensity promotes consolidation.
        replay_bias_min: Minimum replay consolidation bias.
        replay_bias_max: Maximum replay consolidation bias.

        replay_kl_budget: Base KL budget at |bias|=1.0 against uniform.
        replay_kl_max: Hard cap on KL to avoid distribution collapse.
    """
    enabled: bool = True

    # Gains (temperature/learning rate adjustments)
    k_temp_drift: float = 0.35
    k_temp_stab: float = 0.25
    k_lr_meta: float = -0.30
    k_lr_intensity: float = 0.20

    # Clamps
    sigma_min: float = 1e-3
    sigma_max: float = 10.0
    lr_min: float = 1e-5
    lr_max: float = 5e-2

    # Replay bias
    k_replay_valence: float = 0.5
    k_replay_intensity: float = 0.5
    replay_bias_min: float = -1.0
    replay_bias_max: float = 1.0
    replay_kl_budget: float = 0.20
    replay_kl_max: float = 0.80


# =============================================================================
# CES (Count-min Sketch / Episodic Storage) Configuration
# =============================================================================

@dataclass
class CESConfig:
    """
    Configuration for Count-min Sketch / Episodic Storage system.

    CES provides approximate frequency counting and key-value storage
    for episodic memory with efficient space usage.

    Attributes:
        shard_prefix_bits: Number of prefix bits for sharding (2^N shards).
        wal_segment_bytes: Size of each WAL segment file in bytes.
        wal_use_memmap: Use memory-mapped files for WAL (faster, more memory).
        wal_memmap_dir: Directory for memory-mapped files.

        cms_depth: Count-min sketch depth (number of hash functions).
        cms_width: Count-min sketch width (number of buckets per row).
        alpha: Exponential decay factor for frequency counts.

        topk_c: Constant multiplier for adaptive top-k calculation.
        topk_min: Minimum number of top-k results.
        topk_max: Maximum number of top-k results.

        promote_margin_min: Minimum margin for promotion.
        promote_margin_frac: Fractional margin for promotion.
        cooldown_updates: Number of updates before cooldown period.

        decay_half_life_updates: Half-life for decay in number of updates.
        cms_rehash_window_updates: Window for CMS rehashing.

        seed: Random seed for reproducibility.
        row_cache_capacity: Maximum number of rows in the cache.
    """
    shard_prefix_bits: int = 10
    wal_segment_bytes: int = 16 << 20  # 16MB
    wal_use_memmap: bool = False
    wal_memmap_dir: str = "/tmp"
    cms_depth: int = 4
    cms_width: int = 1 << 20  # 1M
    alpha: float = 0.3
    topk_c: float = 6.0
    topk_min: int = 8
    topk_max: int = 64
    promote_margin_min: int = 3
    promote_margin_frac: float = 0.01
    cooldown_updates: int = 8192
    decay_half_life_updates: int = 10_000_000
    cms_rehash_window_updates: int = 200_000_000
    seed: int = 1337
    row_cache_capacity: int = 200_000


# Backward compatibility alias with deprecation warning
class _CESConfig(CESConfig):
    """
    Deprecated: Use CESConfig instead.

    This class is kept for backward compatibility and will be removed
    in a future version.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "_CESConfig is deprecated; use CESConfig instead",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


# =============================================================================
# Master Configuration
# =============================================================================

@dataclass
class M3Config:
    """
    Master configuration for M3 core engine.

    Combines QualiaConfig and CESConfig into a single configuration object
    that can be loaded from JSON files or environment variables.

    Attributes:
        qualia: Qualia closed-loop configuration.
        ces: CES (Count-min Sketch / Episodic Storage) configuration.
        log_path: Path to the bus log file.

    Example:
        # Load from JSON file
        config = M3Config.from_json('config.json')

        # Load from environment variables
        config = M3Config.from_env()

        # Programmatic configuration
        config = M3Config(
            qualia=QualiaConfig(enabled=False),
            ces=CESConfig(seed=42)
        )
    """
    qualia: QualiaConfig = field(default_factory=QualiaConfig)
    ces: CESConfig = field(default_factory=CESConfig)
    log_path: str = field(
        default_factory=lambda: _os.environ.get('M3_BUS_LOG', 'bus.jsonl')
    )

    @classmethod
    def from_json(cls, path: str) -> 'M3Config':
        """
        Load configuration from a JSON file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            M3Config instance with loaded values.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        if not _os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls(
            qualia=QualiaConfig(**data.get('qualia', {})),
            ces=CESConfig(**data.get('ces', {})),
            log_path=data.get('log_path', _os.environ.get('M3_BUS_LOG', 'bus.jsonl'))
        )

    @classmethod
    def from_env(cls) -> 'M3Config':
        """
        Load configuration from environment variables.

        Environment variables use the M3_ prefix followed by the section
        and field name in uppercase, separated by underscores.

        Examples:
            M3_QUALIA_ENABLED=false
            M3_QUALIA_K_TEMP_DRIFT=0.5
            M3_CES_SHARD_PREFIX_BITS=12
            M3_LOG_PATH=/var/log/m3/bus.jsonl

        Returns:
            M3Config instance with values from environment.
        """
        qualia_data = _load_from_env('M3_QUALIA', QualiaConfig)
        ces_data = _load_from_env('M3_CES', CESConfig)
        log_path = _os.environ.get('M3_LOG_PATH', _os.environ.get('M3_BUS_LOG', 'bus.jsonl'))

        return cls(
            qualia=QualiaConfig(**qualia_data),
            ces=CESConfig(**ces_data),
            log_path=log_path
        )

    def to_json(self, path: str) -> None:
        """
        Save configuration to a JSON file.

        Args:
            path: Path to save the JSON configuration file.
        """
        # Create parent directories if needed
        parent_dir = _os.path.dirname(path)
        if parent_dir:
            _os.makedirs(parent_dir, exist_ok=True)

        data = self.to_dict()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            'qualia': asdict(self.qualia),
            'ces': asdict(self.ces),
            'log_path': self.log_path
        }


# =============================================================================
# Environment Variable Loading
# =============================================================================

def _load_from_env(prefix: str, config_class: type) -> Dict[str, Any]:
    """
    Load configuration values from environment variables.

    Args:
        prefix: Environment variable prefix (e.g., 'M3_QUALIA').
        config_class: Dataclass type to get field information.

    Returns:
        Dictionary of field name to value for fields found in environment.
    """
    result: Dict[str, Any] = {}

    for field_info in fields(config_class):
        env_key = f"{prefix}_{field_info.name}".upper()
        env_val = _os.environ.get(env_key)

        if env_val is not None:
            try:
                # Get the actual type (handle string annotations from __future__.annotations)
                field_type = field_info.type
                if isinstance(field_type, str):
                    field_type = field_type.lower()

                # Type conversion based on field type
                if field_type in (bool, 'bool'):
                    result[field_info.name] = env_val.lower() in ('true', '1', 'yes', 'on')
                elif field_type in (int, 'int'):
                    result[field_info.name] = int(env_val)
                elif field_type in (float, 'float'):
                    result[field_info.name] = float(env_val)
                else:
                    result[field_info.name] = env_val
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse {env_key}={env_val}: {e}")

    return result


# =============================================================================
# Global Configuration Management
# =============================================================================

_global_m3_config: Optional[M3Config] = None


def get_m3_config() -> M3Config:
    """
    Get the global M3 configuration.

    If no configuration has been set, automatically loads from environment
    variables using M3Config.from_env().

    Returns:
        The global M3Config instance.
    """
    global _global_m3_config
    if _global_m3_config is None:
        _global_m3_config = M3Config.from_env()
    return _global_m3_config


def set_m3_config(config: M3Config) -> None:
    """
    Set the global M3 configuration.

    Args:
        config: M3Config instance to use as global configuration.
    """
    global _global_m3_config
    _global_m3_config = config


def load_m3_config(path: str) -> M3Config:
    """
    Load configuration from a file and set it as the global configuration.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        The loaded M3Config instance.
    """
    config = M3Config.from_json(path)
    set_m3_config(config)
    return config


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_m3_config(config: Optional[M3Config] = None) -> bool:
    """
    Validate M3 configuration values.

    Checks that configuration values are within valid ranges and
    that constraint relationships are satisfied.

    Args:
        config: M3Config instance to validate. If None, validates global config.

    Returns:
        True if configuration is valid, False otherwise.
    """
    cfg = config or get_m3_config()
    errors = []

    # Qualia validation
    if cfg.qualia.sigma_min >= cfg.qualia.sigma_max:
        errors.append("qualia.sigma_min must be < sigma_max")
    if cfg.qualia.lr_min >= cfg.qualia.lr_max:
        errors.append("qualia.lr_min must be < lr_max")
    if not (0 <= cfg.qualia.replay_kl_budget <= 1):
        errors.append("qualia.replay_kl_budget must be in [0, 1]")
    if not (0 <= cfg.qualia.replay_kl_max <= 1):
        errors.append("qualia.replay_kl_max must be in [0, 1]")
    if cfg.qualia.replay_kl_budget > cfg.qualia.replay_kl_max:
        errors.append("qualia.replay_kl_budget must be <= replay_kl_max")
    if cfg.qualia.replay_bias_min >= cfg.qualia.replay_bias_max:
        errors.append("qualia.replay_bias_min must be < replay_bias_max")

    # CES validation
    if cfg.ces.topk_min > cfg.ces.topk_max:
        errors.append("ces.topk_min must be <= topk_max")
    if cfg.ces.shard_prefix_bits < 0 or cfg.ces.shard_prefix_bits > 20:
        errors.append("ces.shard_prefix_bits must be in [0, 20]")
    if cfg.ces.cms_depth < 1:
        errors.append("ces.cms_depth must be >= 1")
    if cfg.ces.cms_width < 1:
        errors.append("ces.cms_width must be >= 1")
    if cfg.ces.alpha < 0 or cfg.ces.alpha > 1:
        errors.append("ces.alpha must be in [0, 1]")

    if errors:
        for e in errors:
            logger.error(f"Config validation error: {e}")
        return False

    return True


# =============================================================================
# Backward Compatibility Layer
# =============================================================================

class _QualiaConfigProxy:
    """
    Proxy class for backward compatibility with dict-style access to QUALIA_CFG.

    This class allows existing code using QUALIA_CFG['key'] syntax to continue
    working while using the new QualiaConfig dataclass internally.
    """

    def __getitem__(self, key: str) -> Any:
        """Get a configuration value by key."""
        return getattr(get_m3_config().qualia, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a configuration value by key."""
        setattr(get_m3_config().qualia, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with a default."""
        return getattr(get_m3_config().qualia, key, default)

    def items(self) -> Iterator[Tuple[str, Any]]:
        """Iterate over (key, value) pairs."""
        return iter(asdict(get_m3_config().qualia).items())

    def keys(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(asdict(get_m3_config().qualia).keys())

    def values(self) -> Iterator[Any]:
        """Iterate over values."""
        return iter(asdict(get_m3_config().qualia).values())

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return hasattr(get_m3_config().qualia, key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(asdict(get_m3_config().qualia).keys())

    def __repr__(self) -> str:
        """String representation."""
        return repr(asdict(get_m3_config().qualia))


# Backward compatibility: QUALIA_CFG as proxy that reads from global config
QUALIA_CFG = _QualiaConfigProxy()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # New API
    'M3Config',
    'QualiaConfig',
    'CESConfig',
    'get_m3_config',
    'set_m3_config',
    'load_m3_config',
    'validate_m3_config',
    # Backward compatibility
    'QUALIA_CFG',
    'QUALIA_LOG_PATH',
    '_CESConfig',
    '_QualiaConfigProxy',
]
