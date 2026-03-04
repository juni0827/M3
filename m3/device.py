from __future__ import annotations

import os
from typing import Optional


def _as_torch_module(torch_module=None):
    if torch_module is not None:
        return torch_module
    try:
        import torch as _torch
        return _torch
    except Exception as exc:
        raise RuntimeError('PyTorch is required for device resolution') from exc


def _normalize_device_string(preferred: Optional[str]) -> str:
    return (preferred or os.getenv('M3_TORCH_DEVICE', '')).strip()


def resolve_torch_device(
    explicit: Optional[str] = None,
    *,
    torch_module=None,
    require_cuda: bool = False,
    allow_cpu_fallback: bool = True,
) -> str:
    """Resolve a torch device with bounded validation.

    Args:
        explicit: explicit device string, overrides M3_TORCH_DEVICE.
        torch_module: optional torch module override.
        require_cuda: hard-fail when CUDA requested but unavailable.
        allow_cpu_fallback: return CPU when GPU is not available.

    Returns:
        Canonical torch.device string.
    """
    torch = _as_torch_module(torch_module)
    raw = _normalize_device_string(explicit)

    if raw:
        try:
            target = torch.device(raw)
        except Exception as exc:
            raise RuntimeError(f'Invalid device string: {raw}') from exc

        if target.type == 'cuda':
            if target.index is None:
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    return str(torch.device('cuda'))
                if require_cuda:
                    raise RuntimeError(
                        'CUDA is required but unavailable. '
                        f"Set CUDA_VISIBLE_DEVICES/M3_TORCH_DEVICE correctly."
                    )
                return 'cpu'

            if 0 <= target.index < torch.cuda.device_count():
                return str(target)

            if require_cuda:
                raise RuntimeError(
                    f'CUDA device index out of range: {raw}. '
                    f'Available CUDA count={torch.cuda.device_count()}.'
                )
            return 'cpu'

        if target.type == 'cpu':
            if require_cuda:
                raise RuntimeError(
                    "CUDA acceleration is required, but M3_TORCH_DEVICE is CPU."
                )
            return 'cpu'

        if require_cuda:
            raise RuntimeError(
                f'Unsupported non-cuda device requested: {target.type}. '
                'CUDA acceleration is required.'
            )
        return str(target)

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return 'cuda'
    if require_cuda and not allow_cpu_fallback:
        raise RuntimeError('CUDA is required, but no usable CUDA GPU is available.')
    if require_cuda:
        raise RuntimeError(
            'CUDA is required, but no usable CUDA GPU is available. '
            'Set M3_TORCH_DEVICE/CUDA_VISIBLE_DEVICES correctly.'
        )
    return 'cpu'


def resolve_torch_device_string(
    explicit: Optional[str] = None,
    torch_module=None,
    require_cuda: bool = False,
    allow_cpu_fallback: bool = True,
) -> str:
    """Backward-compatible string resolver alias."""
    return resolve_torch_device(
        explicit=explicit,
        torch_module=torch_module,
        require_cuda=require_cuda,
        allow_cpu_fallback=allow_cpu_fallback,
    )


def resolve_torch_device_obj(
    explicit: Optional[str] = None,
    *,
    torch_module=None,
    require_cuda: bool = False,
    allow_cpu_fallback: bool = True,
):
    """Return torch.device object directly."""
    torch = _as_torch_module(torch_module)
    return torch.device(
        resolve_torch_device(
            explicit=explicit,
            torch_module=torch,
            require_cuda=require_cuda,
            allow_cpu_fallback=allow_cpu_fallback,
        )
    )
