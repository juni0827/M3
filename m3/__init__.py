from m3.config import (
    # New API
    M3Config,
    QualiaConfig,
    CESConfig,
    get_m3_config,
    set_m3_config,
    load_m3_config,
    validate_m3_config,
    # Backward compatibility
    QUALIA_CFG,
    QUALIA_LOG_PATH,
    _CESConfig,
)
from m3.features import HebbianMemory, FeatureSpec, pack_learned_proj, pack_scalar, pack_spatial_pool, pack_stats_sample, Scope
from m3.visualization import GlitchEncoder, Retinizer, FeatureSummarizer, hilbert_index_to_xy, vector_to_grid
from m3.core import *  # re-export core engine and entrypoints for compatibility

__all__ = [name for name in globals().keys() if not name.startswith('_')]
