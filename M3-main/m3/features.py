from __future__ import annotations

from m3.attr_contract import attr_del, attr_get_optional, attr_get_required, attr_has, attr_set, guard_context, guard_eval, guard_step
import logging
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from numpy.random import default_rng

@dataclass
class HebbianMemory:
    """Simple Hebbian-style vector memory extended with episodic storage
    and similarity / replay interfaces used by higher-level modules.

    Fields:
        size: int - length of the internal memory vector
        lr: float - learning rate for update to the internal memory trace
        rng: Optional[np.random.Generator] - generator used for sampling during replay
        noise_mean: float - mean of noise for replay_episode
        noise_std: float - std of noise for replay_episode
    """
    size: int = 256
    lr: float = 0.05
    rng: Optional[Any] = None
    noise_mean: float = 0.0
    noise_std: float = 0.01
    memory: np.ndarray = field(init=False)
    items: List[np.ndarray] = field(init=False)
    episodes: Dict[int, List[np.ndarray]] = field(init=False)
    _next_episode_id: int = field(init=False, default=1)

    def __post_init__(self):
        self.memory = np.zeros(self.size, dtype=np.float32)
        self.items = []
        self.episodes = {}
        if self.rng is None:
            with guard_context(ctx='m3/features.py:42', catch_base=False) as __m3_guard_40_12:
                self.rng = default_rng()

            if __m3_guard_40_12.error is not None:
                self.rng = np.random.default_rng()

    def update(self, x: np.ndarray):
        """Update the global memory trace with vector x and also record as an item."""
        with guard_context(ctx='m3/features.py:49', catch_base=False) as __m3_guard_47_8:
            x = np.asarray(x, dtype=np.float32)

        if __m3_guard_47_8.error is not None:
            return
        if x.size == 0:
            return
        x = x - x.mean()
        x = np.clip(x, 0, 1)
        L = min(self.size, x.size)
        with guard_context(ctx='m3/features.py:58', catch_base=False) as __m3_guard_56_8:
            self.memory[:L] += float(self.lr) * x[:L]

        if __m3_guard_56_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        # store a compact copy as an item for later similarity search
        with guard_context(ctx='m3/features.py:64', catch_base=False) as __m3_guard_61_8:
            compact = x.flatten()[:self.size].astype(np.float32).copy()
            self.items.append(compact)

        if __m3_guard_61_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")

    def read(self) -> np.ndarray:
        m = self.memory.copy()
        n = np.linalg.norm(m)
        if n > 1e-6:
            m /= n
        return m

    def store_episode(self, frames: List[np.ndarray]) -> int:
        """Store a sequence (episode) and return its id."""
        eid = int(self._next_episode_id)
        self._next_episode_id += 1
        # store shallow copies trimmed to vector form
        seq = [np.asarray(f, dtype=np.float32).copy() for f in frames]
        self.episodes[eid] = seq
        return eid

    def replay_episode(self, eid: int, speed: float = 1.0, stochastic: bool = False) -> List[np.ndarray]:
        """Return the stored episode frames. If stochastic=True, add small noise sampled from RNG."""
        if eid not in self.episodes:
            return []
        seq = [f.copy() for f in self.episodes[eid]]
        if stochastic:
            for i in range(len(seq)):
                noise = self.rng.normal(self.noise_mean, self.noise_std, size=seq[i].shape).astype(np.float32)
                seq[i] = np.clip(seq[i] + noise, 0.0, 1.0)
        return seq

    # Similarity & search utilities
    def similarity_score(self, a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()
        la = np.linalg.norm(a)
        lb = np.linalg.norm(b)
        if la < 1e-8 or lb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (la * lb))

    def similarity_search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Return top_k (index, score) items most similar to query from stored items."""
        if not self.items:
            return []
        scores = []
        q = np.asarray(query, dtype=np.float32).flatten()
        for idx, it in enumerate(self.items):
            s = self.similarity_score(q, it)
            scores.append((idx, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def retrieve_similar_episodes(self, query: np.ndarray, top_k: int = 3, threshold: float = 0.1) -> List[Tuple[int, float]]:
        """Find episodes whose mean frame is similar to query. Returns list of (episode_id, score)."""
        if not self.episodes:
            return []
        q = np.asarray(query, dtype=np.float32).flatten()
        results: List[Tuple[int, float]] = []
        for eid, seq in self.episodes.items():
            if not seq:
                continue
            # If all frames in episode share the same shape, compute mean directly
            with guard_context(ctx='m3/features.py:133', catch_base=False) as __m3_guard_126_12:
                if all(s.shape == seq[0].shape for s in seq):
                    # stack along a new axis; if already 2D arrays this yields (N, H, W)
                    arr = np.stack([s for s in seq], axis=0)
                    mean_frame = np.mean(arr, axis=0)
                else:
                    mean_frame = np.mean(np.vstack([s.flatten() for s in seq]), axis=0)

            if __m3_guard_126_12.error is not None:
                mean_frame = np.mean(np.vstack([np.asarray(s).flatten() for s in seq]), axis=0)
            score = self.similarity_score(q, mean_frame)
            if score >= threshold:
                results.append((eid, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


@dataclass
class FeatureSpec:
    """Declarative spec for a feature to be included in FeatureBank.

    Fields:
        name: human-readable name
        source: Union[str, callable]  # context key string or callable(core) -> Any extractor
        packer: str  # 'scalar'|'stats_sample'|'spatial_pool'|'learned_proj'
        params: Dict[str, Any] = field(default_factory=dict)
        produced_dim: int = 1
        active: bool = True

    """
    name: str
    source: Union[str, Callable[[Any], Any]]
    packer: str = 'scalar'
    params: Dict[str, Any] = field(default_factory=dict)
    produced_dim: int = 1
    active: bool = True


def pack_scalar(val: Any, params: Dict[str, Any]) -> np.ndarray:
    try:
        return np.array([float(val)], dtype=np.float32)
    except Exception:
        with guard_context(ctx='m3/features.py:172', catch_base=False) as __m3_guard_168_8:
            a = np.asarray(val).ravel()
            if a.size:
                return np.array([float(a.ravel()[0])], dtype=np.float32)

        if __m3_guard_168_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
    return np.array([0.0], dtype=np.float32)


def pack_stats_sample(val: Any, params: Dict[str, Any]) -> np.ndarray:
    max_elems = int(params.get('samples', 8))
    with guard_context(ctx='m3/features.py:181', catch_base=False) as __m3_guard_179_4:
        a = np.asarray(val, dtype=np.float32).ravel()

    if __m3_guard_179_4.error is not None:
        a = np.zeros((0,), dtype=np.float32)
    if a.size == 0:
        return np.zeros(4 + max_elems, dtype=np.float32)
    idx = np.linspace(0, a.size - 1, num=min(max_elems, a.size)).astype(int)
    samp = a[idx].astype(np.float32)
    out = np.zeros(4 + max_elems, dtype=np.float32)
    out[0] = float(a.mean())
    out[1] = float(a.std())
    out[2] = float(a.min())
    out[3] = float(a.max())
    out[4:4 + samp.size] = samp
    return out


def pack_spatial_pool(val: Any, params: Dict[str, Any]) -> np.ndarray:
    """Pool a 2D image/array into a small descriptor vector.

    Params:
        grid: int or (h,w) - how many cells to pool into (default 4 -> 2x2)
        pool_type: 'mean'|'max' (default 'mean')
    """
    grid = params.get('grid', 4)
    pool_type = params.get('pool_type', 'mean')
    with guard_context(ctx='m3/features.py:207', catch_base=False) as __m3_guard_205_4:
        a = np.asarray(val, dtype=np.float32)

    if __m3_guard_205_4.error is not None:
        return np.zeros((4 + int(grid),), dtype=np.float32)
    # ensure 2D by averaging channels if needed
    if a.ndim == 3:
        a2 = a.mean(axis=-1)
    elif a.ndim == 2:
        a2 = a
    else:
        a2 = np.asarray(a).ravel()
        a2 = a2.reshape(1, -1)
    # resize to small grid
    with guard_context(ctx='m3/features.py:221', catch_base=False) as __m3_guard_218_4:
        h = int(np.clip(a2.shape[0], 1, 256))
        w = int(np.clip(a2.shape[1], 1, 256)) if a2.ndim == 2 else 1

    if __m3_guard_218_4.error is not None:
        h, w = a2.shape[0], (a2.shape[1] if a2.ndim == 2 else 1)
    # target grid
    if isinstance(grid, (list, tuple)):
        gh, gw = int(grid[0]), int(grid[1])
    else:
        # approximate square grid
        g = int(grid)
        gh = max(1, int(np.sqrt(g)))
        gw = max(1, g // gh)
    out = []
    for iy in range(gh):
        y0 = int(iy * h / gh)
        y1 = int((iy + 1) * h / gh)
        for ix in range(gw):
            x0 = int(ix * w / gw)
            x1 = int((ix + 1) * w / gw)
            block = a2[y0:y1, x0:x1] if a2.ndim == 2 else a2[:, x0:x1]
            if block.size == 0:
                out.append(0.0)
            else:
                if pool_type == 'max':
                    out.append(float(block.max()))
                else:
                    out.append(float(block.mean()))
    # prefix stats
    vec = np.array(out, dtype=np.float32)
    stats = np.array([
        float(vec.mean()) if vec.size else 0.0,
        float(vec.std()) if vec.size else 0.0,
        float(vec.min()) if vec.size else 0.0,
        float(vec.max()) if vec.size else 0.0
    ], dtype=np.float32)
    return np.concatenate([stats, vec], axis=0)


def pack_learned_proj(val: Any, params: Dict[str, Any]) -> np.ndarray:
    """Default learned projection: a fixed random projection unless a projector is supplied.

    Params:
        dim: output dim
        proj: optional callable(arr) -> arr
    """
    dim = int(params.get('dim', 16))
    proj = params.get('proj', None)
    with guard_context(ctx='m3/features.py:268', catch_base=False) as __m3_guard_266_4:
        a = np.asarray(val, dtype=np.float32).ravel()

    if __m3_guard_266_4.error is not None:
        a = np.zeros((1,), dtype=np.float32)
    if proj is not None and callable(proj):
        with guard_context(ctx='m3/features.py:280', catch_base=False) as __m3_guard_271_8:
            out = proj(a)
            out = np.asarray(out, dtype=np.float32).ravel()
            if out.size == dim:
                return out
            if out.size > dim:
                return out[:dim]
            pad = np.zeros((dim - out.size,), dtype=np.float32)
            return np.concatenate([out, pad], axis=0)

        if __m3_guard_271_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
    # fallback random projection using hash-based seed for determinism
    seed = int(hashlib.blake2b(a.tobytes() if a.size else b'0', digest_size=4).digest()[0]) if a.size else 0
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0, size=(dim, max(1, a.size))).astype(np.float32)
    if a.size == 0:
        return np.zeros((dim,), dtype=np.float32)
    out = W @ a
    # normalize
    if np.linalg.norm(out) > 1e-8:
        out = out / (np.linalg.norm(out) + 1e-6)
    return out.astype(np.float32)


__all__ = [
    'HebbianMemory',
    'FeatureSpec',
    'pack_scalar',
    'pack_stats_sample',
    'pack_spatial_pool',
    'pack_learned_proj',
]
