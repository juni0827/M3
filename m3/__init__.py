from typing import Optional

from m3.config import QUALIA_CFG, QUALIA_LOG_PATH, _CESConfig
from m3.features import HebbianMemory, FeatureSpec, pack_learned_proj, pack_scalar, pack_spatial_pool, pack_stats_sample, Scope
from m3.visualization import GlitchEncoder, Retinizer, FeatureSummarizer, hilbert_index_to_xy, vector_to_grid
from .device import (
    resolve_torch_device_obj,
    resolve_torch_device_string,
)
from .m3_core import M3ConsciousnessCore

try:
    # Optional alias preserved for older call sites that used M3Core.
    M3Core = M3ConsciousnessCore
except Exception:  # pragma: no cover
    M3Core = None


_DEVICE_CACHE = None


def get_device(preferred: Optional[str] = None, require_cuda: bool = True):
    """Resolve a torch device from preferred device string / environment."""
    return resolve_torch_device_obj(
        explicit=preferred,
        require_cuda=require_cuda,
        allow_cpu_fallback=not require_cuda,
    )


def require_cuda_device():
    """Backward-compatible strict CUDA resolver for CUDA-dependent paths."""
    return get_device(require_cuda=True)


def _resolve_cached_device():
    global _DEVICE_CACHE
    if _DEVICE_CACHE is None:
        try:
            _DEVICE_CACHE = get_device(require_cuda=False)
        except Exception:
            import torch
            _DEVICE_CACHE = torch.device('cpu')
    return _DEVICE_CACHE


def __getattr__(name):
    if name == 'DEVICE':
        return _resolve_cached_device()
    raise AttributeError(name)


__all__ = [
    'DEVICE',
    'QUALIA_CFG',
    'QUALIA_LOG_PATH',
    '_CESConfig',
    'HebbianMemory',
    'FeatureSpec',
    'pack_learned_proj',
    'pack_scalar',
    'pack_spatial_pool',
    'pack_stats_sample',
    'Scope',
    'GlitchEncoder',
    'Retinizer',
    'FeatureSummarizer',
    'hilbert_index_to_xy',
    'vector_to_grid',
    'M3ConsciousnessCore',
    'M3Core',
    'get_device',
    'require_cuda_device',
    'resolve_torch_device_obj',
    'resolve_torch_device_string',
]
