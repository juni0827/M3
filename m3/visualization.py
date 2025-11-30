from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from PIL import Image, ImageOps


def hilbert_index_to_xy(n: int, d: int):
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def vector_to_grid(vec: np.ndarray, side: int) -> np.ndarray:
    grid = np.zeros((side, side), dtype=np.float32)
    L = min(vec.size, side * side)
    for i in range(L):
        x, y = hilbert_index_to_xy(side, i)
        grid[y, x] = vec[i]
    if L < side * side:
        # Fill remaining cells with the mean of the remaining vector values if available,
        # otherwise use 0.0. Guard against empty slices.
        remaining = vec[L:]
        fill_value = float(remaining.mean()) if remaining.size > 0 else 0.0
        grid.flat[L:] = fill_value
    return grid


@dataclass
class Retinizer:
    target_size: tuple[int, int] = (256, 256)

    def __call__(self, arr: np.ndarray | None) -> np.ndarray:
        try:
            import numpy as _np
            if arr is None:
                return _np.zeros(self.target_size, dtype=_np.float32)
            a = _np.asarray(arr)
            if a.ndim == 0:
                return _np.full(self.target_size, float(a), dtype=_np.float32)
            if a.ndim == 3 and a.shape[2] >= 3:
                a = _np.dot(a[..., :3].astype(_np.float32), _np.array([0.2989, 0.5870, 0.1140], dtype=_np.float32))
            a = _np.clip(a, 0.0, 1.0)
            a_u8 = (a * 255).astype(_np.uint8)
            im = Image.fromarray(a_u8)
            try:
                im = ImageOps.fit(im, self.target_size, method=getattr(Image, 'Resampling', Image).LANCZOS)
            except Exception:
                im = ImageOps.fit(im, self.target_size)
            out = _np.asarray(im).astype(_np.float32) / 255.0
            if out.ndim == 3:
                out = out.mean(axis=-1)
            return out
        except Exception:
            try:
                import numpy as _np
                return _np.zeros(self.target_size, dtype=_np.float32)
            except Exception:
                return arr  # type: ignore


@dataclass
class FeatureSummarizer:
    patch: int = 8

    def __call__(self, retina: np.ndarray) -> np.ndarray:
        H, W = retina.shape
        ph, pw = self.patch, self.patch
        gy, gx = np.gradient(retina)
        mag = np.sqrt(gx * gx + gy * gy)
        feats = []
        for y in range(0, H, ph):
            for x in range(0, W, pw):
                block = retina[y:y + ph, x:x + pw]
                block_mag = mag[y:y + ph, x:x + pw]
                if block.size == 0:
                    continue
                feats.append([
                    float(block.mean()),
                    float(block.std()),
                    float(block_mag.mean()),
                    float(block_mag.std())
                ])
        return np.asarray(feats, dtype=np.float32).flatten()


@dataclass
class GlitchEncoder:
    out_size: tuple[int, int] = (768, 1024)  # (H, W)
    one_bit: bool = True

    def __call__(self, retina: np.ndarray, context_vec: np.ndarray, arousal: float = 0.5) -> np.ndarray:
        H, W = self.out_size
        _Res = getattr(Image, 'Resampling', Image)
        im = Image.fromarray((retina * 255).astype(np.uint8)).resize((W, H), resample=_Res.BICUBIC)
        arr = np.asarray(im).astype(np.float32) / 255.0
        ctx = np.asarray(context_vec, dtype=np.float32).copy()
        if ctx.size < H:
            tiles = int(np.ceil(H / max(1, ctx.size)))
            ctx = np.tile(ctx, tiles)[:H]
        ctx = (ctx - ctx.mean()) / (ctx.std() + 1e-6)
        A = float(np.clip(arousal, 0.0, 1.0))
        max_shift = int(W * (0.06 + 0.14 * A))
        shifts = (0.5 * (ctx + 1.0)) * max_shift
        shifts = shifts.astype(np.int32)
        shifted = np.zeros_like(arr)
        for y in range(H):
            s = int(shifts[y])
            shifted[y] = np.roll(arr[y], s)
        yy = np.arange(H, dtype=np.float32).reshape(-1, 1)
        base_freq = 1.5 + 7.5 * A
        freq = base_freq * (np.abs(ctx[:H]) / (np.max(np.abs(ctx[:H])) + 1e-6))
        phase = (yy * (freq.reshape(-1, 1) / H)) * 2 * np.pi
        stripe = (0.45 + 0.45 * A) * (0.5 + 0.5 * np.sin(phase))
        mod = np.clip(shifted * (0.75 + 0.2 * A) + stripe, 0, 1)
        if self.one_bit:
            thr = 0.5 + 0.18 * (A - 0.5)
            out = (mod > thr).astype(np.uint8) * 255
        else:
            out = (mod * 255).astype(np.uint8)
        return out


__all__ = [
    'hilbert_index_to_xy',
    'vector_to_grid',
    'Retinizer',
    'FeatureSummarizer',
    'GlitchEncoder',
]
