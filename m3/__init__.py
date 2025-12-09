from m3.config import QUALIA_CFG, QUALIA_LOG_PATH, _CESConfig
from m3.features import HebbianMemory, FeatureSpec, pack_learned_proj, pack_scalar, pack_spatial_pool, pack_stats_sample, Scope
from m3.visualization import GlitchEncoder, Retinizer, FeatureSummarizer, hilbert_index_to_xy, vector_to_grid
from m3.core import *  # re-export core engine and entrypoints for compatibility

__all__ = [name for name in globals().keys() if not name.startswith('_')]
