from __future__ import annotations

import os as _os
from dataclasses import dataclass

# Path and environment-based defaults
_ROOT = _os.path.dirname(_os.path.dirname(__file__))
_LOG_DIR = _os.path.join(_ROOT, 'logs')
_os.makedirs(_LOG_DIR, exist_ok=True)
QUALIA_LOG_PATH = _os.environ.get('M3_BUS_LOG', _os.path.join(_LOG_DIR, 'bus.jsonl'))

# === Qualia Closed-Loop Configuration ========================================
QUALIA_CFG = {
    'enabled': True,
    # Gains (small, safe-by-default; user can tune),
    'k_temp_drift': 0.35,        # temperature gain from drift.prediction
    'k_temp_stab': 0.25,         # temperature gain from (1 - belief.stability)
    'k_lr_meta': -0.30,          # lr gain from meta_confidence (negative reduces lr when confident)
    'k_lr_intensity': 0.20,      # lr gain from qualia['I']
    # Clamps,
    'sigma_min': 1e-3,
    'sigma_max': 10.0,
    'lr_min': 1e-5,
    'lr_max': 5e-2,
    # Replay bias (write-only; sampler may read if available),
    'k_replay_valence': 0.5,     # positive valence promotes consolidation
    'k_replay_intensity': 0.5,   # strong intensity promotes consolidation
    'replay_bias_min': -1.0,
    'replay_bias_max': 1.0,
    # Replay sampling calibration (non-magic): KL budget against uniform,
    'replay_kl_budget': 0.20,   # base KL@|bias|=1.0 (interpretable budget, not a gain)
    'replay_kl_max': 0.80       # hard cap to avoid collapse
}


@dataclass
class _CESConfig:
    shard_prefix_bits: int = 10
    wal_segment_bytes: int = 16 << 20
    wal_use_memmap: bool = False
    wal_memmap_dir: str = "/tmp"
    cms_depth: int = 4
    cms_width: int = 1 << 20
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


__all__ = ['QUALIA_CFG', 'QUALIA_LOG_PATH', '_CESConfig']
